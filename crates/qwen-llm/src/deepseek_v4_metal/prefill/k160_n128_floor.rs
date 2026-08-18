use super::*;

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;
use std::time::Instant;

const FIXTURE: &str = "/Users/tito/models/deepseek-v4-flash-0731-reap-k160/DeepSeek-V4-Flash-0731-REAP-K160-Q3_K_Q4_K-00001-of-00004.gguf";
const N_TOKENS: usize = 128;
const EXPERTS: usize = 160;
const ROUTES: usize = N_TOKENS * MOE_TOP_K;
const REPRESENTATIVE_ACTIVE_EXPERTS: usize = 96;
const REPRESENTATIVE_HOT_ROUTES: usize = 116;
const LAYERS: f64 = 43.0;
const MIN_SCALED_GPU_SAVING_MS: f64 = 100.0;
const GUARD_BYTES: usize = 64;

fn guarded_f32(ctx: &MetalContext, shape: Vec<u64>, poison: f32) -> MetalTensor {
    let elements = shape.iter().product::<u64>() as usize;
    let mut bytes = vec![0xa5u8; GUARD_BYTES];
    bytes.extend_from_slice(bytemuck::cast_slice(&vec![poison; elements]));
    bytes.extend_from_slice(&[0x5au8; GUARD_BYTES]);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).expect("guarded N128 output"),
        offset: GUARD_BYTES as u64,
        shape,
        dtype: GgmlType::F32,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn assert_guards(label: &str, tensor: &MetalTensor) {
    let base = tensor.buffer.contents().as_ptr().cast::<u8>();
    let prefix = unsafe {
        std::slice::from_raw_parts(base.add(tensor.offset as usize - GUARD_BYTES), GUARD_BYTES)
    };
    let suffix = unsafe {
        std::slice::from_raw_parts(
            base.add(tensor.offset as usize + tensor.n_bytes() as usize),
            GUARD_BYTES,
        )
    };
    assert!(prefix.iter().all(|&byte| byte == 0xa5), "{label} prefix");
    assert!(suffix.iter().all(|&byte| byte == 0x5a), "{label} suffix");
}

fn representative_route_counts() -> Vec<usize> {
    let mut counts = vec![0usize; EXPERTS];
    counts[0] = REPRESENTATIVE_HOT_ROUTES;
    let remaining = ROUTES - REPRESENTATIVE_HOT_ROUTES;
    let peers = REPRESENTATIVE_ACTIVE_EXPERTS - 1;
    for (expert, slot) in counts
        .iter_mut()
        .enumerate()
        .take(REPRESENTATIVE_ACTIVE_EXPERTS)
        .skip(1)
    {
        *slot = remaining / peers + usize::from(expert <= remaining % peers);
    }
    assert_eq!(counts.iter().sum::<usize>(), ROUTES);
    assert!(counts.iter().all(|&count| count <= N_TOKENS));
    counts
}

fn representative_schedule() -> (Vec<i32>, Vec<i32>, Vec<ExpertBucket>) {
    let counts = representative_route_counts();
    let mut remaining = BinaryHeap::new();
    for (expert, &count) in counts.iter().enumerate() {
        if count > 0 {
            remaining.push((count, Reverse(expert)));
        }
    }

    let mut expert_ids = vec![0i32; ROUTES];
    for token in 0..N_TOKENS {
        let mut selected = Vec::with_capacity(MOE_TOP_K);
        for slot in 0..MOE_TOP_K {
            let (count, Reverse(expert)) = remaining
                .pop()
                .expect("representative route degree sequence is realizable");
            expert_ids[token * MOE_TOP_K + slot] = expert as i32;
            selected.push((count - 1, Reverse(expert)));
        }
        for item in selected {
            if item.0 > 0 {
                remaining.push(item);
            }
        }
    }
    assert!(remaining.is_empty());

    let mut rows = Vec::with_capacity(ROUTES);
    let mut slots = Vec::with_capacity(ROUTES);
    let mut schedule = Vec::with_capacity(REPRESENTATIVE_ACTIVE_EXPERTS);
    for expert in 0..EXPERTS {
        let start = rows.len();
        for (slot, &routed_expert) in expert_ids.iter().enumerate() {
            if routed_expert == expert as i32 {
                rows.push((slot / MOE_TOP_K) as i32);
                slots.push(slot as i32);
            }
        }
        if rows.len() > start {
            schedule.push(ExpertBucket {
                expert,
                start,
                len: rows.len() - start,
            });
        }
    }
    validate_packed_expert_schedule(N_TOKENS, EXPERTS, &expert_ids, &rows, &slots, &schedule)
        .expect("representative K160 schedule");
    (rows, slots, schedule)
}

fn load_bank(
    ctx: &MetalContext,
    gguf: &crate::gguf::GgufFile,
    name: &str,
    dtype: GgmlType,
    shape: [u64; 3],
) -> MetalTensor {
    let descriptor = gguf.find(name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(descriptor.dtype, dtype, "{name} dtype");
    assert_eq!(descriptor.shape, shape, "{name} shape");
    MetalTensor::from_bytes(
        ctx,
        gguf.slice(descriptor),
        descriptor.shape.clone(),
        descriptor.dtype,
    )
    .unwrap_or_else(|error| panic!("copy {name}: {error}"))
}

#[allow(clippy::too_many_arguments)]
fn encode_bucket_control(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    down_bank: &MetalTensor,
    input: &MetalTensor,
    rows: &MetalTensor,
    slots: &MetalTensor,
    schedule: &[ExpertBucket],
    expert_input: &MetalTensor,
    gate: &MetalTensor,
    up: &MetalTensor,
    inner: &MetalTensor,
    bucket_output: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), DeepSeekV4MetalError> {
    for bucket in schedule {
        let row_view = i32_slice(rows, bucket.start, bucket.len, "N128 control rows")?;
        let slot_view = i32_slice(slots, bucket.start, bucket.len, "N128 control slots")?;
        let input_view = f32_prefix(
            expert_input,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, bucket.len as u64],
            "N128 control input",
        )?;
        encode_get_rows_f32(
            ctx,
            enc,
            input,
            &row_view,
            &input_view,
            bucket.len,
            DEEPSEEK_V4_HIDDEN_SIZE,
        )?;
        let gate_view = f32_prefix(
            gate,
            vec![MOE_FFN_SIZE as u64, bucket.len as u64],
            "N128 control gate",
        )?;
        let up_view = f32_prefix(
            up,
            vec![MOE_FFN_SIZE as u64, bucket.len as u64],
            "N128 control up",
        )?;
        let inner_view = f32_prefix(
            inner,
            vec![MOE_FFN_SIZE as u64, bucket.len as u64],
            "N128 control inner",
        )?;
        for (bank, projection, label) in [
            (gate_bank, &gate_view, "N128 control gate projection"),
            (up_bank, &up_view, "N128 control up projection"),
        ] {
            let weight = expert_weight_view(
                bank,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                bucket.expert,
                label,
            )?;
            encode_batch_projection(
                ctx,
                enc,
                &weight,
                &input_view,
                projection,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                bucket.len,
                label,
            )?;
        }
        let elements = MOE_FFN_SIZE * bucket.len;
        encode_ds4_clamped_swiglu(
            ctx,
            enc,
            &gate_view.view_subrange(0, vec![elements as u64]),
            &up_view.view_subrange(0, vec![elements as u64]),
            &inner_view.view_subrange(0, vec![elements as u64]),
            10.0,
        )?;
        let output_view = f32_prefix(
            bucket_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, bucket.len as u64],
            "N128 control output",
        )?;
        let down_weight = expert_weight_view(
            down_bank,
            MOE_FFN_SIZE,
            DEEPSEEK_V4_HIDDEN_SIZE,
            bucket.expert,
            "N128 control down projection",
        )?;
        encode_batch_projection(
            ctx,
            enc,
            &down_weight,
            &inner_view,
            &output_view,
            MOE_FFN_SIZE,
            DEEPSEEK_V4_HIDDEN_SIZE,
            bucket.len,
            "N128 control down projection",
        )?;
        crate::metal::encode_scatter_rows_f32_unique(
            ctx,
            enc,
            &output_view,
            &slot_view,
            output,
            DEEPSEEK_V4_HIDDEN_SIZE,
            bucket.len,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_grouped_candidate(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    down_bank: &MetalTensor,
    input: &MetalTensor,
    rows: &MetalTensor,
    slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    inner: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), DeepSeekV4MetalError> {
    let (gate, up) =
        packed_grouped_gate_up_views(output, DEEPSEEK_V4_HIDDEN_SIZE, MOE_FFN_SIZE, ROUTES)?;
    for (bank, projection) in [(gate_bank, &gate), (up_bank, &up)] {
        encode_packed_grouped_mapped_k_block_f32_plan(
            ctx,
            enc,
            bank,
            input,
            rows,
            slots,
            plan,
            projection,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            EXPERTS,
            MOE_TOP_K,
            N_TOKENS,
            N_TOKENS,
            ROUTES,
        )?;
    }
    let elements = MOE_FFN_SIZE * ROUTES;
    encode_ds4_clamped_swiglu(
        ctx,
        enc,
        &gate.view_subrange(0, vec![elements as u64]),
        &up.view_subrange(0, vec![elements as u64]),
        &inner.view_subrange(0, vec![elements as u64]),
        10.0,
    )?;
    encode_packed_grouped_mapped_k_block_f32_plan(
        ctx,
        enc,
        down_bank,
        inner,
        slots,
        slots,
        plan,
        output,
        MOE_FFN_SIZE,
        DEEPSEEK_V4_HIDDEN_SIZE,
        EXPERTS,
        MOE_TOP_K,
        N_TOKENS,
        ROUTES,
        ROUTES,
    )
}

#[test]
#[ignore = "requires the local K160 fixture and about 4 GiB transient residency"]
fn k160_n128_grouped_q3q4_representative_layer_floor() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    if !Path::new(FIXTURE).exists() {
        eprintln!("[k160-n128-floor] skipped: K160 fixture missing");
        return;
    }
    let gguf = crate::gguf::GgufFile::open(FIXTURE).expect("open K160 GGUF");
    assert_eq!(ctx.device.name().to_string(), "Apple M4 Max");
    assert_eq!(gguf.tensors.len(), 1_328);
    assert_eq!(gguf.shard_count(), 4);
    assert_eq!(
        gguf.tensors
            .iter()
            .map(|descriptor| descriptor.n_bytes)
            .sum::<u64>(),
        89_920_886_108,
    );
    let gate_bank = load_bank(
        &ctx,
        &gguf,
        "blk.0.ffn_gate_exps.weight",
        GgmlType::Q3_K,
        [
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            MOE_FFN_SIZE as u64,
            EXPERTS as u64,
        ],
    );
    let up_bank = load_bank(
        &ctx,
        &gguf,
        "blk.0.ffn_up_exps.weight",
        GgmlType::Q3_K,
        [
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            MOE_FFN_SIZE as u64,
            EXPERTS as u64,
        ],
    );
    let down_bank = load_bank(
        &ctx,
        &gguf,
        "blk.0.ffn_down_exps.weight",
        GgmlType::Q4_K,
        [
            MOE_FFN_SIZE as u64,
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            EXPERTS as u64,
        ],
    );

    let (row_values, slot_values, schedule) = representative_schedule();
    let max_bucket = schedule.iter().map(|bucket| bucket.len).max().unwrap();
    let plan = PackedGroupedExpertPlan::new(N_TOKENS, &schedule, None)
        .expect("representative grouped plan");
    eprintln!(
        "[k160-n128-floor] active={} hot={} tiles={} occupancy={:.4}",
        schedule.len(),
        max_bucket,
        plan.dispatch_tiles,
        ROUTES as f64 / (plan.dispatch_tiles * PACKED_GROUPED_EXPERT_TILE_ROWS) as f64,
    );
    let rows = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&row_values),
        vec![ROUTES as u64],
        GgmlType::I32,
    )
    .expect("N128 source rows");
    let slots = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&slot_values),
        vec![ROUTES as u64],
        GgmlType::I32,
    )
    .expect("N128 destination slots");
    let input_values = (0..DEEPSEEK_V4_HIDDEN_SIZE * N_TOKENS)
        .map(|index| ((index * 17 + 3) % 257) as f32 * 0.000_05 - 0.006_4)
        .collect::<Vec<_>>();
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&input_values),
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, N_TOKENS as u64],
        GgmlType::F32,
    )
    .expect("N128 input");
    let expert_input = MetalTensor::zeros_f32(
        &ctx,
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, max_bucket as u64],
    )
    .expect("N128 control input");
    let gate = MetalTensor::zeros_f32(&ctx, vec![MOE_FFN_SIZE as u64, max_bucket as u64])
        .expect("N128 control gate");
    let up = MetalTensor::zeros_f32(&ctx, vec![MOE_FFN_SIZE as u64, max_bucket as u64])
        .expect("N128 control up");
    let inner = MetalTensor::zeros_f32(&ctx, vec![MOE_FFN_SIZE as u64, max_bucket as u64])
        .expect("N128 control inner");
    let bucket_output = MetalTensor::zeros_f32(
        &ctx,
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, max_bucket as u64],
    )
    .expect("N128 control bucket output");
    let control = guarded_f32(
        &ctx,
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, ROUTES as u64],
        7.0,
    );
    let candidate_inner = MetalTensor::zeros_f32(&ctx, vec![MOE_FFN_SIZE as u64, ROUTES as u64])
        .expect("N128 candidate inner");
    let candidate = guarded_f32(
        &ctx,
        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, ROUTES as u64],
        11.0,
    );

    let run = |grouped: bool| {
        let started = Instant::now();
        let command = ctx.queue.commandBuffer().expect("N128 command buffer");
        let encoder = KernelEncoder::begin(&command);
        let result = if grouped {
            encode_grouped_candidate(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &down_bank,
                &input,
                &rows,
                &slots,
                &plan,
                &candidate_inner,
                &candidate,
            )
        } else {
            encode_bucket_control(
                &ctx,
                &encoder,
                &gate_bank,
                &up_bank,
                &down_bank,
                &input,
                &rows,
                &slots,
                &schedule,
                &expert_input,
                &gate,
                &up,
                &inner,
                &bucket_output,
                &control,
            )
        };
        encoder.end();
        result.expect("encode N128 floor");
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());
        (
            started.elapsed().as_secs_f64() * 1e3,
            (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
        )
    };

    run(false);
    run(true);
    let a1 = run(false);
    let b1 = run(true);
    let b2 = run(true);
    let a2 = run(false);
    for (wall, gpu) in [a1, b1, b2, a2] {
        assert!(wall.is_finite() && wall > 0.0);
        assert!(gpu.is_finite() && gpu > 0.0 && gpu <= wall);
    }
    let control_values = host_read_f32(&control, "N128 control output").unwrap();
    let candidate_values = host_read_f32(&candidate, "N128 candidate output").unwrap();
    assert_eq!(
        candidate_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        control_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "grouped N128 layer changed bucket lineage",
    );
    assert_guards("N128 control output", &control);
    assert_guards("N128 candidate output", &candidate);
    let conservative_wall = a1.0.min(a2.0) - b1.0.max(b2.0);
    let conservative_gpu = a1.1.min(a2.1) - b1.1.max(b2.1);
    eprintln!(
        "[k160-n128-floor] wall_ms A/B/B/A={:.3}/{:.3}/{:.3}/{:.3}",
        a1.0, b1.0, b2.0, a2.0,
    );
    eprintln!(
        "[k160-n128-floor] gpu_ms A/B/B/A={:.3}/{:.3}/{:.3}/{:.3}",
        a1.1, b1.1, b2.1, a2.1,
    );
    eprintln!(
        "[k160-n128-floor] conservative_layer wall={conservative_wall:.3} gpu={conservative_gpu:.3} scaled_43 wall={:.3} gpu={:.3}",
        conservative_wall * LAYERS,
        conservative_gpu * LAYERS,
    );
    assert!(b1.1 < a1.1.min(a2.1) && b2.1 < a1.1.min(a2.1));
    assert!(
        conservative_gpu * LAYERS >= MIN_SCALED_GPU_SAVING_MS,
        "scaled conservative GPU saving {:.3} ms missed {:.3} ms gate",
        conservative_gpu * LAYERS,
        MIN_SCALED_GPU_SAVING_MS,
    );
}

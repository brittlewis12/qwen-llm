use super::*;
use objc2::rc::{Weak, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

pub(super) const GROUPS: [usize; 5] = [1, 8, 16, 17, 32];
pub(super) const DTYPES: [GgmlType; 3] = [GgmlType::F32, GgmlType::BF16, GgmlType::Q8_0];
pub(super) const WIDTH: usize = 256;
pub(super) const RANK: usize = 7;
pub(super) const GUARD: usize = 64;
pub(super) const BUDGET: u64 = 16 << 20;
pub(super) const RESERVE: u64 = 256 << 10;
pub(super) const LIFETIME_CPU_BYTES: u64 = 64 << 10;

pub(super) type BufferReleaseProbes = [Weak<ProtocolObject<dyn MTLBuffer>>; 2];

pub(super) fn case_config(groups: usize) -> Config {
    Config {
        // Keep rank-7 output-A tails independent of output-B row alignment.
        hidden: 1,
        intermediate: 32,
        heads: 2 * groups,
        head_dim: WIDTH / 2,
        query_rank: 32,
        output_rank: RANK,
        groups,
        index_heads: 1,
        index_dim: WIDTH / 2,
        index_topk: 2,
        window: 2,
        rope_dim: 0,
        candidate_block: 2,
        candidate_topk: 1,
        norm_eps: 1e-6,
        ..Config::default()
    }
}

fn schedule() -> Schedule {
    Schedule {
        layers: 1,
        kv_source_layers: vec![],
        index_source_layers: vec![],
        compression_ratios: vec![0],
        candidate_source_layer: None,
    }
}

pub(super) fn stored_row_bytes(dtype: GgmlType) -> usize {
    match dtype {
        GgmlType::F32 => WIDTH * 4,
        GgmlType::BF16 => WIDTH * 2,
        GgmlType::Q8_0 => WIDTH / 32 * 34,
        _ => panic!("fixture dtype"),
    }
}

pub(super) struct StoredBank {
    pub bytes: Vec<u8>,
    pub decoded: Vec<f32>,
    pub groups: usize,
    pub dtype: GgmlType,
}

impl StoredBank {
    pub fn new(groups: usize, dtype: GgmlType) -> Self {
        let mut bytes = vec![0xa7; GUARD];
        for group in 0..groups {
            for row in 0..RANK {
                match dtype {
                    GgmlType::F32 | GgmlType::BF16 => {
                        for col in 0..WIDTH {
                            let value = dense_coefficient(group, row, col);
                            if dtype == GgmlType::F32 {
                                bytes.extend(value.to_le_bytes());
                            } else {
                                bytes.extend(half::bf16::from_f32(value).to_le_bytes());
                            }
                        }
                    }
                    GgmlType::Q8_0 => {
                        for block in 0..WIDTH / 32 {
                            let scale = 0x1801 + (group * 32 + row * 3 + block) as u16;
                            bytes.extend(scale.to_le_bytes());
                            for j in 0..32 {
                                let q = ((row * 3 + block * 7 + j * 11) % 17) as i8 - 8
                                    + 2 * (group as i8 + 1);
                                bytes.push(q as u8);
                            }
                        }
                    }
                    _ => panic!("fixture dtype"),
                }
            }
        }
        bytes.extend([0x5c; GUARD]);
        let decoded = decode_stored(&bytes[GUARD..bytes.len() - GUARD], dtype);
        assert_eq!(decoded.len(), groups * RANK * WIDTH);
        Self {
            bytes,
            decoded,
            groups,
            dtype,
        }
    }

    pub fn group_bytes(&self) -> usize {
        RANK * stored_row_bytes(self.dtype)
    }

    pub fn unchanged(&self, actual: &[u8]) -> bool {
        actual == self.bytes
    }

    pub fn oracle(&self, attended: &[f32], bank_ids: &[usize]) -> Vec<Dot> {
        assert_eq!(attended.len(), self.groups * WIDTH);
        assert_eq!(bank_ids.len(), self.groups);
        (0..self.groups)
            .flat_map(|group| {
                let x = &attended[group * WIDTH..(group + 1) * WIDTH];
                let start = bank_ids[group] * RANK * WIDTH;
                self.decoded[start..start + RANK * WIDTH]
                    .chunks_exact(WIDTH)
                    .map(|row| dot(row, x))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

pub(super) fn dense_coefficient(group: usize, row: usize, col: usize) -> f32 {
    ((group + 1) * 29 + (row + 1) * 11 + (col * 13 % 37)) as f32 / 8191.0
}

fn decode_stored(bytes: &[u8], dtype: GgmlType) -> Vec<f32> {
    match dtype {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|v| half::bf16::from_le_bytes(v.try_into().unwrap()).to_f32())
            .collect(),
        GgmlType::Q8_0 => bytes
            .chunks_exact(34)
            .flat_map(|block| {
                let scale = half::f16::from_le_bytes(block[..2].try_into().unwrap()).to_f32();
                block[2..].iter().map(move |&q| (q as i8) as f32 * scale)
            })
            .collect(),
        _ => panic!("fixture dtype"),
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dot {
    pub value: f64,
    pub absolute_sum: f64,
    pub width: usize,
}

pub(super) fn dot(weights: &[f32], x: &[f32]) -> Dot {
    assert_eq!(weights.len(), x.len());
    assert!(weights.iter().chain(x).all(|v| v.is_finite()));
    let products = weights
        .iter()
        .zip(x)
        .map(|(&w, &x)| f64::from(w) * f64::from(x));
    Dot {
        value: products.clone().sum(),
        absolute_sum: products.map(f64::abs).sum(),
        width: x.len(),
    }
}

pub(super) fn check_dots(
    actual: &[f32],
    expected: &[Dot],
    group_rank: usize,
) -> std::result::Result<(), String> {
    if actual.len() != expected.len() || !actual.len().is_multiple_of(group_rank) {
        return Err("oracle shape".into());
    }
    for (group, (actual, expected)) in actual
        .chunks_exact(group_rank)
        .zip(expected.chunks_exact(group_rank))
        .enumerate()
    {
        if !expected.iter().any(|d| {
            let bound =
                1e-7 + 2.0 * (d.width + 2) as f64 * f64::from(f32::EPSILON) * d.absolute_sum;
            d.value.abs() > 1e-4f64.max(8.0 * bound)
        }) {
            return Err(format!("zero oracle group {group}"));
        }
        for (row, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            let bound = 1e-7
                + 2.0
                    * (expected.width + 2) as f64
                    * f64::from(f32::EPSILON)
                    * expected.absolute_sum;
            if !actual.is_finite()
                || !expected.value.is_finite()
                || !expected.absolute_sum.is_finite()
                || (f64::from(actual) - expected.value).abs() > bound
            {
                return Err(format!(
                    "group {group} row {row}: {actual} != {} bound={bound}",
                    expected.value
                ));
            }
        }
    }
    Ok(())
}

pub(super) struct Price {
    pub requests: Vec<u64>,
    pub logical: u64,
    pub metal_upper: u64,
    pub cpu_upper: u64,
}

pub(super) fn price(groups: usize, dtype: GgmlType) -> Price {
    let c = case_config(groups);
    let mut requests: Vec<u64> = shapes(&c, 2, &schedule())
        .unwrap()
        .into_iter()
        .map(|n| n as u64 * 4)
        .collect();
    requests.extend([
        (c.hidden * c.query_rank * 4) as u64,
        (c.query_rank * c.heads * c.head_dim * 4) as u64,
        (c.hidden * c.head_dim * 4) as u64,
        (c.groups * RANK * c.hidden * 4) as u64,
        (GUARD * 2 + groups * RANK * stored_row_bytes(dtype)) as u64,
        (GUARD * 2 + groups * RANK * 4) as u64,
        (c.hidden * 4) as u64,
    ]);
    if dtype == GgmlType::Q8_0 {
        requests.push(8);
    }
    let logical = requests.iter().sum::<u64>();
    let metal_upper = requests.iter().map(|n| n + 16384).sum();
    // Four full logical copies cover staging, decoded coefficients, readbacks,
    // and f64 oracles; 1 MiB covers small vectors, descriptors, and census rows.
    // Separate allowance for the two boxed weak handles, runtime weak-table
    // entries, and nested case/position autorelease-pool bookkeeping.
    let cpu_upper = 4 * logical + (1 << 20) + LIFETIME_CPU_BYTES;
    assert!(metal_upper + cpu_upper + RESERVE <= BUDGET);
    Price {
        requests,
        logical,
        metal_upper,
        cpu_upper,
    }
}

fn admit(ctx: &MetalContext, groups: usize, dtype: GgmlType) -> Price {
    let price = price(groups, dtype);
    let realized_upper: u64 = price
        .requests
        .iter()
        .map(|&n| {
            let upper = ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes;
            assert!(
                upper <= n + 16384,
                "device pricing exceeds fixture page bound"
            );
            upper
        })
        .sum();
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        realized_upper,
        price.cpu_upper,
        RESERVE,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.is_admitted(),
        "output-A preallocation admission: {admission:?}"
    );
    eprintln!(
        "output_a_fixture groups={groups} dtype={dtype:?} requests={} logical={} metal_upper={} device_priced={} cpu_upper={} reserve={RESERVE}",
        price.requests.len(),
        price.logical,
        price.metal_upper,
        realized_upper,
        price.cpu_upper
    );
    price
}

pub(super) fn a_launches(
    groups: usize,
    dtype: GgmlType,
    r2: bool,
) -> Vec<(&'static str, [u64; 3])> {
    if groups <= 16 {
        let name = match dtype {
            GgmlType::F32 => "kernel_expert_native_f32_f32",
            GgmlType::BF16 => "kernel_expert_native_bf16_f32",
            GgmlType::Q8_0 => "kernel_expert_native_q8_0_f32",
            _ => panic!("fixture dtype"),
        };
        vec![(name, [RANK.div_ceil(4) as u64, groups as u64, 1])]
    } else {
        let (name, rows) = match dtype {
            GgmlType::F32 if r2 => ("kernel_mat_vec_f32_f32_lcpp_r2", 2),
            GgmlType::F32 => ("kernel_mat_vec_f32_f32", 4),
            GgmlType::BF16 => ("kernel_mat_vec_bf16_f32", 4),
            GgmlType::Q8_0 => ("kernel_expert_native_q8_0_f32", 4),
            _ => panic!("fixture dtype"),
        };
        vec![(name, [RANK.div_ceil(rows) as u64, 1, 1]); groups]
    }
}

pub(super) fn check_census(
    census: &[metal::DispatchCensusRow],
    groups: usize,
    dtype: GgmlType,
) -> std::result::Result<(), String> {
    // The attended output's last producer: the chain's output kernel or the
    // fused decode attention's combine (with the inverse RoPE folded in).
    let anchor = census
        .iter()
        .rposition(|r| {
            matches!(
                r.kernel.as_str(),
                "kernel_deepseek_v41_attention_output"
                    | "kernel_deepseek_v41_attention_flash_combine"
                    | "kernel_deepseek_v41_attention_flash_combine_rope"
            )
        })
        .ok_or("missing attended-output anchor")?;
    let tail = &census[anchor + 1..];
    let b = tail
        .get(tail.len().saturating_sub(2))
        .ok_or("missing output B")?;
    let r2 = b.kernel == "kernel_mat_vec_f32_f32_lcpp_r2";
    let launches = a_launches(groups, dtype, r2);
    if tail.len() != launches.len() + 2 {
        return Err("output-A launch count".into());
    }
    for (row, (name, grid)) in tail.iter().zip(&launches) {
        if row.kernel != *name
            || [row.grid_width, row.grid_height, row.grid_depth] != *grid
            || [row.threads_width, row.threads_height, row.threads_depth] != [128, 1, 1]
            || row.encoder_concurrent
            || row.encoder_ordinal != census[anchor].encoder_ordinal
        {
            return Err(format!("output-A launch geometry: {row:?}"));
        }
    }
    let b_name = if r2 {
        "kernel_mat_vec_f32_f32_lcpp_r2"
    } else {
        "kernel_mat_vec_f32_f32"
    };
    if b.kernel != b_name
        || [b.grid_width, b.grid_height, b.grid_depth] != [1, 1, 1]
        || [b.threads_width, b.threads_height, b.threads_depth] != [128, 1, 1]
        || tail.last().unwrap().kernel != "kernel_deepseek_v41_attention_finite"
    {
        return Err("output-B/finiteness boundary".into());
    }
    Ok(())
}

fn dense(ctx: &MetalContext, input: usize, output: usize, role: usize) -> Linear<'_> {
    let values: Vec<_> = (0..input * output)
        .map(|i| (1 + (i * 7 + role * 11) % 23) as f32 / 1024.0)
        .collect();
    Linear {
        ctx,
        tensor: upload(ctx, &values, vec![input as u64, output as u64]).unwrap(),
        native_control: None,
        input,
        output,

        half_weight: None,
    }
}

fn weights<'a>(ctx: &'a MetalContext, c: &Config, stored: &StoredBank) -> Weights<'a> {
    let raw = MetalTensor::from_bytes(
        ctx,
        &stored.bytes,
        vec![(stored.bytes.len() / 4) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let bank = MetalTensor {
        buffer: raw.buffer.clone(),
        offset: GUARD as u64,
        shape: vec![WIDTH as u64, RANK as u64, c.groups as u64],
        dtype: stored.dtype,
        provenance: raw.provenance(),
    };
    let control = native_control(ctx, stored.dtype).unwrap();
    op::AttentionWeights {
        query_a: dense(ctx, c.hidden, c.query_rank, 1),
        query_norm: vec![1.0; c.query_rank],
        query_b: dense(ctx, c.query_rank, c.heads * c.head_dim, 2),
        kv: dense(ctx, c.hidden, c.head_dim, 3),
        kv_norm: vec![1.0; c.head_dim],
        sink: vec![-0.25; c.heads],
        output_a: (0..c.groups)
            .map(|group| Linear {
                ctx,
                tensor: bank.view_bytes(
                    (group * stored.group_bytes()) as u64,
                    vec![WIDTH as u64, RANK as u64],
                ),
                native_control: control.clone(),
                input: WIDTH,
                output: RANK,

                half_weight: None,
            })
            .collect(),
        output_b: dense(ctx, c.groups * RANK, c.hidden, 4),
        compressor: None,
        index: None,
        frequencies: vec![],
        head_dim: c.head_dim,
        window: c.window,
        eps: c.norm_eps,
        candidate_block_size: c.candidate_block,
        candidate_topk_blocks: c.candidate_topk,
    }
}

fn shared_bytes(t: &MetalTensor, len: usize) -> Vec<u8> {
    assert!(len <= t.buffer.length());
    assert_eq!(t.buffer.storageMode(), objc2_metal::MTLStorageMode::Shared);
    // Callers use immutable weights or wait for command completion first.
    unsafe { std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), len).to_vec() }
}

pub(super) fn low_guards_intact(values: &[f32], count: usize) -> bool {
    values.len() == count + 2 * GUARD / 4
        && values[..GUARD / 4]
            .iter()
            .chain(&values[GUARD / 4 + count..])
            .all(|v| v.to_bits() == 12345.25f32.to_bits())
}

pub(super) fn run_case(ctx: &MetalContext, groups: usize, dtype: GgmlType) {
    let transaction = ctx.begin_allocation_transaction();
    let probes = autoreleasepool(|_| run_pooled_case(ctx, groups, dtype));
    for (label, weak) in ["output_a_bank", "guarded_low"].into_iter().zip(&probes) {
        assert!(
            weak.load().is_none(),
            "{label} survived case pool drain: groups={groups} dtype={dtype:?}"
        );
    }
    eprintln!("output_a_fixture RELEASED groups={groups} dtype={dtype:?} bank=true low=true");
    drop(probes);
    drop(transaction);
}

fn run_pooled_case(ctx: &MetalContext, groups: usize, dtype: GgmlType) -> BufferReleaseProbes {
    let price = admit(ctx, groups, dtype);
    metal::allocation_census_begin();
    let stored = StoredBank::new(groups, dtype);
    let c = case_config(groups);
    let weights = weights(ctx, &c, &stored);
    assert_eq!(
        Workspace::output_a_bank(&weights, &c).is_some(),
        groups <= 16
    );
    let mut actual = Workspace::new_with_schedule(ctx, &c, 2, &[&weights], &schedule()).unwrap();
    let guarded = upload(
        ctx,
        &vec![12345.25; groups * RANK + 2 * GUARD / 4],
        vec![(groups * RANK + 2 * GUARD / 4) as u64],
    )
    .unwrap();
    actual.low = guarded.view_subrange((GUARD / 4) as u64, vec![(groups * RANK) as u64]);
    let input = upload(ctx, &vec![0.0; c.hidden], vec![c.hidden as u64]).unwrap();
    let allocations = metal::allocation_census_take();
    assert!(allocations.len() <= price.requests.len());
    assert!(allocations.iter().map(|r| r.buffer_length).sum::<u64>() <= price.metal_upper);
    let probes = [
        Weak::from_retained(&weights.output_a[0].tensor.buffer),
        Weak::from_retained(&guarded.buffer),
    ];
    assert!(
        probes.iter().all(|weak| weak.load().is_some()),
        "live Metal buffers must support weak probes"
    );
    let b_values = f32_slice(&weights.output_b.tensor).unwrap();
    for position in 0..2 {
        autoreleasepool(|_| {
            // No command is in flight while the CPU poisons/repopulates shared views.
            unsafe {
                let low = actual
                    .low
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(actual.low.offset as usize)
                    .cast::<f32>();
                for i in 0..groups * RANK {
                    low.add(i).write(f32::NAN);
                }
                let x = input.buffer.contents().as_ptr().cast::<f32>();
                for i in 0..c.hidden {
                    x.add(i).write((1 + i + position) as f32 / 32.0);
                }
            }
            metal::dispatch_census_begin();
            execute(ctx, |enc| {
                actual
                    .encode(ctx, enc, 0, position, &weights, &input)
                    .unwrap();
            });
            check_census(&metal::dispatch_census_take(), groups, dtype).unwrap();
            let attended = f32_slice(&actual.attended).unwrap();
            let expected = stored.oracle(attended, &(0..groups).collect::<Vec<_>>());
            check_dots(f32_slice(&actual.low).unwrap(), &expected, RANK).unwrap();
            assert!(check_dots(&vec![0.0; groups * RANK], &expected, RANK).is_err());
            if groups > 1 {
                for ids in [
                    vec![0; groups],
                    (0..groups).map(|g| (g + 1) % groups).collect(),
                ] {
                    let wrong: Vec<_> = stored
                        .oracle(attended, &ids)
                        .iter()
                        .map(|d| d.value as f32)
                        .collect();
                    assert!(
                        check_dots(&wrong, &expected, RANK).is_err(),
                        "actual attended vector cannot distinguish wrong bank IDs"
                    );
                }
            }
            let low = f32_slice(&actual.low).unwrap();
            let output: Vec<_> = b_values
                .chunks_exact(groups * RANK)
                .map(|row| dot(row, low))
                .collect();
            check_dots(f32_slice(&actual.output).unwrap(), &output, c.hidden).unwrap();
            assert!(low_guards_intact(
                f32_slice(&guarded).unwrap(),
                groups * RANK
            ));
            assert!(stored.unchanged(&shared_bytes(
                &weights.output_a[0].tensor,
                stored.bytes.len()
            )));
            assert_eq!(shared_bytes(&actual.status, 4), [0, 0, 0, 0]);
            eprintln!("output_a_fixture PASS groups={groups} dtype={dtype:?} position={position}");
        });
    }
    probes
}

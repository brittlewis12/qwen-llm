//! Primitive witnesses only. Ignored GPU tests require actual named hardware,
//! MTL_DEBUG_LAYER=1, QWEN_METAL_LEASE_WAIT=1 and an explicit production lease.

use super::*;
use crate::glm5_next_metal::packed;
use crate::metal::{self, expert_tensor_down as adapter, *};
use crate::tensor::TensorDesc;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLGPUFamily,
};

const HALF: Arith = Arith {
    lineage: PackedLineage::Fast,
    precision: FastPrecision::Half,
};
const GENERIC: &str = "kernel_moe_down_iq3_s_f32_grouped_slots_generic";
const TENSOR: &str = "kernel_moe_down_iq3_s_f32_grouped_slots_tensor";
const GUARD: usize = 256;
const WR: usize = 7;
const XR: usize = 13;
const TOP: usize = 8;

#[test]
fn cpu_down_policy_protects_all_arithmetic_modes_types_and_actual_rows() {
    let _scope = Scope::new(true);
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            for mode in [StageMode::Fast, StageMode::FastF32, StageMode::Exact] {
                for dtype in [
                    GgmlType::IQ3_S,
                    GgmlType::IQ4_XS,
                    GgmlType::IQ3_XXS,
                    GgmlType::IQ2_S,
                    GgmlType::Q6_K,
                    GgmlType::Q8_0,
                    GgmlType::F32,
                    GgmlType::F16,
                    GgmlType::Unknown,
                ] {
                    for rows in [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        16,
                        31,
                        32,
                        128,
                        512,
                        513,
                        1023,
                        1024,
                        1025,
                        2047,
                        2048,
                        2049,
                        4096,
                        usize::MAX,
                    ] {
                        let arith = Arith { lineage, precision };
                        let expected = arith == HALF
                            && mode == StageMode::Fast
                            && dtype == GgmlType::IQ3_S
                            && matches!(rows, 1024 | 2048);
                        assert_eq!(
                            selected(arith, mode, dtype, rows),
                            expected,
                            "{arith:?}/{mode:?}/{dtype:?}/{rows}"
                        );
                        let policy =
                            packed::grouped_down_policy(arith, mode, dtype, 2048, 4096, 288, rows);
                        assert_eq!(policy == GroupedDownPolicy::Iq3STensor, expected);
                        if arith.lineage == PackedLineage::Fast
                            && dtype == GgmlType::IQ3_S
                            && (32..=512).contains(&rows)
                        {
                            assert_eq!(policy, GroupedDownPolicy::Iq3SSmallCounts);
                        }
                    }
                }
            }
        }
    }
    // Capability is geometry-based, not tied to the stock artifact's dimensions.
    for (k, m, e) in [
        (2048, 4096, 288),
        (2048, 6144, 256),
        (4096, 8192, 16),
        (256, 65, 16),
    ] {
        assert!(adapter::geometry(k, m, e, TOP, 1024).is_ok());
        assert_eq!(
            packed::grouped_down_policy(HALF, StageMode::Fast, GgmlType::IQ3_S, k, m, e, 1024),
            GroupedDownPolicy::Iq3STensor
        );
    }
}

#[test]
fn cpu_down_scope_is_default_off_nested_thread_local_and_unwind_safe() {
    let requested = || selected(HALF, StageMode::Fast, GgmlType::IQ3_S, 1024);
    assert!(!requested());
    {
        let _on = Scope::new(true);
        assert!(requested());
        {
            let _off = Scope::new(false);
            assert!(!requested());
        }
        assert!(requested());
        assert!(
            std::panic::catch_unwind(|| {
                let _off = Scope::new(false);
                panic!("restore down policy");
            })
            .is_err()
        );
        assert!(requested());
        std::thread::spawn(|| {
            assert!(!enabled());
            let _on = Scope::new(true);
            assert!(enabled());
        })
        .join()
        .unwrap();
        assert!(requested());
    }
    assert!(!requested());
}

#[test]
fn cpu_down_geometry_rejects_partial_blocks_signed_limits_and_products() {
    for (k, m, e, top, rows) in [
        (0, 4096, 288, TOP, 1024),
        (2047, 4096, 288, TOP, 1024),
        (2048, 0, 288, TOP, 1024),
        (2048, 4096, 0, TOP, 1024),
        (2048, 4096, 288, 0, 1024),
        (2048, 4096, 288, 17, 1024),
        (2048, 4096, 288, TOP, 0),
        (usize::MAX, 4096, 288, TOP, 1024),
        (2048, usize::MAX, 288, TOP, 1024),
        (2048, 4096, usize::MAX, TOP, 1024),
        (2048, 4096, 288, TOP, usize::MAX),
        (2048, 4096, 288, TOP, i32::MAX as usize / TOP + 1),
        (2048, i32::MAX as usize + 1, 288, TOP, 1024),
        (
            i32::MAX as usize / 256 * 256,
            i32::MAX as usize,
            i32::MAX as usize,
            1,
            1024,
        ),
    ] {
        assert!(
            adapter::geometry(k, m, e, top, rows).is_err(),
            "{k}/{m}/{e}/{top}/{rows}"
        );
    }
    assert!(adapter::geometry(256, 65, 16, TOP, 1025).is_ok());
    assert!(adapter::capacity_supported(32, 128, 0, 8192));
    assert!(adapter::capacity_supported(32, 256, 512, 8704));
    for (width, threads, stat, limit) in [
        (16, 128, 0, 8192),
        (32, 127, 0, 8192),
        (32, 128, 0, 8191),
        (32, 128, 1, 8192),
        (32, 128, usize::MAX, usize::MAX),
    ] {
        assert!(!adapter::capacity_supported(width, threads, stat, limit));
    }
}

fn validation_lease() -> impl Sized {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    metal::acquire_metal_benchmark_lease().expect("explicit production GPU lease")
}

fn require_m5(ctx: &MetalContext) {
    assert!(
        ctx.device.name().to_string().contains("M5"),
        "requires actual M5 hardware"
    );
    assert!(ctx.device.supportsFamily(MTLGPUFamily::Apple10));
    assert!(
        ctx.tensor_ops_status().is_available(),
        "{:?}",
        ctx.tensor_ops_status()
    );
    assert!(
        metal::moe_grouped_tensor_selected(ctx),
        "global tensor rollback must be ON"
    );
}

fn backing(t: &MetalTensor) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    }
}

fn fingerprint(t: &MetalTensor) -> (u64, u64) {
    backing(t)
        .chunks(8)
        .enumerate()
        .fold((0u64, 0u64), |(a, b), (i, bytes)| {
            let mut word = [0u8; 8];
            word[..bytes.len()].copy_from_slice(bytes);
            let word = u64::from_le_bytes(word);
            (
                a.wrapping_add(word),
                b.wrapping_add(word.rotate_left((i % 64) as u32)),
            )
        })
}

fn guarded(ctx: &MetalContext, payload: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let mut bytes = vec![0xa5; GUARD];
    bytes.extend_from_slice(payload);
    bytes.resize(bytes.len() + GUARD, 0x5a);
    MetalTensor {
        buffer: ctx.buffer_from(&bytes).unwrap(),
        offset: GUARD as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn guards(t: &MetalTensor) {
    let bytes = backing(t);
    assert!(bytes[..GUARD].iter().all(|&b| b == 0xa5));
    assert!(bytes[bytes.len() - GUARD..].iter().all(|&b| b == 0x5a));
}

fn read(t: &MetalTensor) -> Vec<f32> {
    bytemuck::cast_slice(&backing(t)[t.offset as usize..t.offset as usize + t.n_bytes() as usize])
        .to_vec()
}

fn poison(t: &MetalTensor) {
    assert!(t.is_writable());
    unsafe {
        t.buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(t.offset as usize)
            .write_bytes(0xff, t.n_bytes() as usize);
    }
}

fn run(ctx: &MetalContext, body: impl FnOnce(&KernelEncoder)) -> Vec<DispatchCensusRow> {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    dispatch_census_begin();
    body(&enc);
    let census = dispatch_census_take();
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    census
}

struct Fixture {
    k: usize,
    m: usize,
    e: usize,
    rows: usize,
    w: MetalTensor,
    x: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    y: MetalTensor,
    weights: MetalTensor,
    output: MetalTensor,
    owners: Vec<usize>,
    full: Vec<f64>,
    half: Vec<f64>,
}

impl Fixture {
    fn new(ctx: &MetalContext, k: usize, m: usize, e: usize, rows: usize) -> Self {
        let _allocation = ctx.begin_allocation_transaction();
        adapter::geometry(k, m, e, TOP, rows).unwrap();
        let slots = rows * TOP;
        let row_bytes = k / 256 * 110;
        let logical = [
            row_bytes * m * e,
            k * slots * 4,
            e * 4,
            e * rows * 4,
            m * slots * 4,
            slots * 4,
            m * rows * 4,
            // Extra normalized input and route IDs used by composition tests.
            m * rows * 4,
            slots * 4,
        ];
        let gpu_upper: u64 = logical
            .iter()
            .map(|&n| {
                ctx.price_shared_buffer_upper((n + 2 * GUARD) as u64)
                    .unwrap()
                    .priced_upper_bytes
            })
            .sum();
        // Includes upload plus guarded upload copies, both readbacks, compact
        // f64 oracles, immutable fingerprints and composition scratch. Charge
        // CPU to working-set headroom even when macOS omits process budget.
        let cpu_upper = (logical.iter().sum::<usize>() as u64)
            .checked_mul(6)
            .unwrap()
            + (32 << 20);
        let admission = evaluate_metal_memory_admission(
            gpu_upper.checked_add(cpu_upper).unwrap(),
            crate::glm5_next::memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.is_admitted(), "{admission:?}");

        let mut compact = vec![0u8; WR * row_bytes];
        let mut rng = 0xdead_beefu32;
        for (i, block) in compact.chunks_exact_mut(110).enumerate() {
            for byte in &mut block[2..] {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (rng >> 24) as u8;
            }
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            block[..2].copy_from_slice(
                &half::f16::from_f32(sign * (1 + i % 7) as f32 * 0.000113).to_le_bytes(),
            );
        }
        let desc = TensorDesc {
            name: "glm_down_compact_iq3_s".into(),
            shape: vec![k as u64, WR as u64],
            dtype: GgmlType::IQ3_S,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: compact.len() as u64,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &compact).unwrap();
        let mut payload = Vec::with_capacity(logical[0]);
        for expert in 0..e {
            for row in 0..m {
                let r = (expert * 3 + row) % WR;
                payload.extend_from_slice(&compact[r * row_bytes..(r + 1) * row_bytes]);
            }
        }
        let mut w = guarded(
            ctx,
            &payload,
            vec![k as u64, m as u64, e as u64],
            GgmlType::IQ3_S,
        );
        w.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        drop(payload);
        let compact_x: Vec<f32> = (0..XR * k)
            .map(|i| (((i % k * 37 + i / k * 17) % 211) as f32 - 105.0) * 0.003713)
            .collect();
        let mut payload = Vec::with_capacity(logical[1]);
        for slot in 0..slots {
            payload.extend_from_slice(bytemuck::cast_slice(
                &compact_x[(slot % XR) * k..(slot % XR + 1) * k],
            ));
        }
        let x = guarded(ctx, &payload, vec![k as u64, slots as u64], GgmlType::F32);
        drop(payload);
        let mut full = vec![0f64; XR * WR];
        let mut half = full.clone();
        let h = |v: f32| half::f16::from_f32(v).to_f64();
        for row in 0..XR {
            for out in 0..WR {
                for i in 0..k {
                    let w = decoded[out * k + i];
                    let x = compact_x[row * k + i];
                    full[row * WR + out] += f64::from(w) * f64::from(x);
                    half[row * WR + out] += h(w) * h(x);
                }
            }
        }
        assert!(full.iter().map(|x| x.abs()).fold(0.0f64, f64::max) > 1e-5);
        let mut counts = vec![0i32; e];
        let mut ids = vec![i32::MAX; e * rows];
        let mut owners = Vec::with_capacity(slots);
        let tails = [0, 1, 15, 16, 17, 31, 32, 33];
        for slot in 0..slots {
            let lane = slot % TOP;
            let expert = if slot / TOP < tails[lane] {
                e - 1 - lane
            } else {
                lane
            };
            ids[expert * rows + counts[expert] as usize] = slot as i32;
            counts[expert] += 1;
            owners.push(expert);
        }
        for expert in 0..e {
            assert!(counts[expert] as usize <= rows);
            ids[expert * rows..expert * rows + counts[expert] as usize].reverse();
        }
        assert_eq!(counts[0] as usize, rows);
        for count in [0, 1, 15, 16, 17, 31, 32, 33] {
            assert!(counts.contains(&count), "missing bucket boundary {count}");
        }
        let counts = guarded(
            ctx,
            bytemuck::cast_slice(&counts),
            vec![e as u64],
            GgmlType::I32,
        );
        let ids = guarded(
            ctx,
            bytemuck::cast_slice(&ids),
            vec![(e * rows) as u64],
            GgmlType::I32,
        );
        let y = guarded(
            ctx,
            &vec![0xff; logical[4]],
            vec![m as u64, slots as u64],
            GgmlType::F32,
        );
        let weights: Vec<f32> = (0..slots).map(Self::route_weight).collect();
        let weights = guarded(
            ctx,
            bytemuck::cast_slice(&weights),
            vec![TOP as u64, rows as u64],
            GgmlType::F32,
        );
        let output = guarded(
            ctx,
            &vec![0xff; logical[6]],
            vec![m as u64, rows as u64],
            GgmlType::F32,
        );
        Self {
            k,
            m,
            e,
            rows,
            w,
            x,
            counts,
            ids,
            y,
            weights,
            output,
            owners,
            full,
            half,
        }
    }

    fn route_weight(slot: usize) -> f32 {
        (1 + (slot * 17 + slot / TOP * 3) % 19) as f32 * 0.0137
    }

    fn reference(&self, slot: usize, row: usize, half: bool) -> f64 {
        let oracle = if half { &self.half } else { &self.full };
        oracle[(slot % XR) * WR + (self.owners[slot] * 3 + row) % WR]
    }

    fn prepare<'a>(
        &self,
        ctx: &'a MetalContext,
        enc: &'a KernelEncoder,
    ) -> Result<Option<adapter::PreparedDown<'a>>, MetalError> {
        adapter::prepare(
            ctx,
            enc,
            &self.w,
            &self.x,
            &self.counts,
            &self.ids,
            &self.y,
            self.k,
            self.m,
            self.e,
            TOP,
            self.rows,
        )
    }

    fn encode(&self, ctx: &MetalContext, enc: &KernelEncoder, candidate: bool, reduce: bool) {
        if candidate {
            self.prepare(ctx, enc)
                .unwrap()
                .expect("candidate must be admitted on M5")
                .encode()
                .unwrap();
        } else {
            encode_moe_down_f32_grouped_slots_generic(
                ctx,
                enc,
                &self.w,
                &self.x,
                &self.counts,
                &self.ids,
                &self.y,
                self.k,
                self.m,
                self.e,
                self.rows,
            )
            .unwrap();
        }
        if reduce {
            encode_moe_weighted_sum_packed_f32(
                ctx,
                enc,
                &self.y,
                &self.weights,
                &self.output,
                self.m,
                TOP,
                self.rows,
            )
            .unwrap();
        }
    }

    fn validate(&self, actual: &[f32]) {
        assert_eq!(actual.len(), self.m * self.rows * TOP);
        let scale = self.full.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        for (i, &a) in actual.iter().enumerate() {
            let slot = i / self.m;
            if self.owners[slot] == usize::MAX {
                assert_eq!(a.to_bits(), u32::MAX, "unrouted slot was written");
            } else {
                assert!(a.is_finite());
                for (half, tolerance) in [(false, 1e-2), (true, 2e-3)] {
                    let b = self.reference(slot, i % self.m, half);
                    assert!(
                        (f64::from(a) - b).abs() <= tolerance * scale + 1e-5,
                        "K{} M{} E{} N{} at {i}: {a} vs {b} half={half}",
                        self.k,
                        self.m,
                        self.e,
                        self.rows
                    );
                }
            }
        }
        for t in [
            &self.w,
            &self.x,
            &self.counts,
            &self.ids,
            &self.y,
            &self.weights,
            &self.output,
        ] {
            guards(t);
        }
    }

    fn validate_reduction(&self, actual: &[f32]) {
        assert_eq!(actual.len(), self.m * self.rows);
        let scale = self.full.iter().map(|x| x.abs()).fold(0.0f64, f64::max) * TOP as f64 * 0.27;
        for (i, &a) in actual.iter().enumerate() {
            assert!(a.is_finite());
            for (half, tolerance) in [(false, 1e-2), (true, 3e-3)] {
                let expected: f64 = (0..TOP)
                    .map(|lane| {
                        let slot = i / self.m * TOP + lane;
                        self.reference(slot, i % self.m, half) * f64::from(Self::route_weight(slot))
                    })
                    .sum();
                assert!(
                    (f64::from(a) - expected).abs() <= tolerance * scale + 1e-5,
                    "weighted output at {i}: {a} vs {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease; numeric primitive, no timing"]
fn m5_down_real_and_older_shapes_generic_tensor_f64_and_weighted_output() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().expect("Metal required, never silently skipped");
    require_m5(&ctx);
    for (k, m, e, rows) in [
        (2048, 4096, 288, 1024),
        (2048, 4096, 288, 2048),
        (2048, 6144, 256, 1024),
        (4096, 8192, 16, 1024),
        (256, 65, 32, 1024),
        // Private primitive stride-tail diagnostic; NOT an admitted family row width.
        (256, 65, 32, 1025),
    ] {
        let f = Fixture::new(&ctx, k, m, e, rows);
        let inputs = [&f.w, &f.x, &f.counts, &f.ids, &f.weights];
        let before: Vec<_> = inputs.iter().map(|t| fingerprint(t)).collect();
        let baseline = run(&ctx, |enc| f.encode(&ctx, enc, false, true));
        assert_eq!(baseline.len(), 2);
        assert_eq!(baseline[0].kernel, GENERIC);
        let a = read(&f.y);
        f.validate(&a);
        f.validate_reduction(&read(&f.output));
        let reduced_a = read(&f.output);
        poison(&f.y);
        poison(&f.output);
        let candidate = run(&ctx, |enc| f.encode(&ctx, enc, true, true));
        assert_eq!(candidate.len(), 2);
        assert_eq!(candidate[0].kernel, TENSOR);
        assert_eq!(
            (
                candidate[0].grid_width,
                candidate[0].grid_height,
                candidate[0].grid_depth
            ),
            (rows.div_ceil(32) as u64, m.div_ceil(64) as u64, e as u64)
        );
        assert_eq!(candidate[0].threads_width, 128);
        assert_eq!(
            (
                candidate[0].grid_width,
                candidate[0].grid_height,
                candidate[0].grid_depth
            ),
            (
                baseline[0].grid_width,
                baseline[0].grid_height,
                baseline[0].grid_depth
            )
        );
        assert_eq!(candidate[1].kernel, baseline[1].kernel);
        let b = read(&f.y);
        f.validate(&b);
        let scale = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            a.iter()
                .zip(&b)
                .all(|(&a, &b)| (a - b).abs() <= 2e-3 * scale + 1e-5)
        );
        let reduced_b = read(&f.output);
        f.validate_reduction(&reduced_b);
        let scale = reduced_a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            reduced_a
                .iter()
                .zip(&reduced_b)
                .all(|(&a, &b)| (a - b).abs() <= 3e-3 * scale + 1e-5)
        );
        for (t, before) in inputs.iter().zip(before) {
            assert_eq!(fingerprint(t), before);
        }
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease"]
fn m5_down_sparse_empty_buckets_invalid_ids_and_slot_limit_tails() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let mut f = Fixture::new(&ctx, 256, 65, 32, 1024);
    let mut ids: Vec<i32> =
        bytemuck::cast_slice(&backing(&f.ids)[GUARD..GUARD + f.ids.n_bytes() as usize]).to_vec();
    let mut invalid = [-1, (f.rows * TOP) as i32, i32::MAX].into_iter();
    for id in &mut ids {
        if *id >= 0 && (*id as usize) < f.rows * TOP && *id != (f.rows * TOP - 1) as i32 {
            if let Some(replacement) = invalid.next() {
                f.owners[*id as usize] = usize::MAX;
                *id = replacement;
            }
        }
    }
    f.ids = guarded(
        &ctx,
        bytemuck::cast_slice(&ids),
        f.ids.shape.clone(),
        GgmlType::I32,
    );
    for candidate in [false, true] {
        poison(&f.y);
        run(&ctx, |enc| f.encode(&ctx, enc, candidate, false));
        f.validate(&read(&f.y));
    }
    f.counts = guarded(
        &ctx,
        bytemuck::cast_slice(&vec![0i32; f.e]),
        f.counts.shape.clone(),
        GgmlType::I32,
    );
    f.owners.fill(usize::MAX);
    for candidate in [false, true] {
        poison(&f.y);
        run(&ctx, |enc| f.encode(&ctx, enc, candidate, false));
        f.validate(&read(&f.y));
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease"]
fn m5_down_active_views_preserve_older_larger_backing_capacity() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let mut f = Fixture::new(&ctx, 256, 65, 32, 2048);
    f.rows = 1024;
    let slots = f.rows * TOP;
    f.owners.truncate(slots);
    f.x.shape = vec![f.k as u64, slots as u64];
    f.y.shape = vec![f.m as u64, slots as u64];
    f.output.shape = vec![f.m as u64, f.rows as u64];
    f.weights.shape = vec![TOP as u64, f.rows as u64];
    f.ids.shape = vec![(f.e * f.rows) as u64];
    let mut counts = vec![0i32; f.e];
    let mut ids = vec![i32::MAX; f.e * f.rows];
    for (slot, &expert) in f.owners.iter().enumerate() {
        ids[expert * f.rows + counts[expert] as usize] = slot as i32;
        counts[expert] += 1;
    }
    for (t, payload) in [
        (&f.counts, bytemuck::cast_slice(&counts)),
        (&f.ids, bytemuck::cast_slice(&ids)),
    ] {
        assert_eq!(t.n_bytes() as usize, payload.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize),
                payload.len(),
            );
        }
    }
    let views = [&f.x, &f.y, &f.ids, &f.weights, &f.output];
    let inactive: Vec<Vec<u8>> = views
        .iter()
        .map(|t| backing(t)[GUARD + t.n_bytes() as usize..].to_vec())
        .collect();
    for candidate in [false, true] {
        poison(&f.y);
        poison(&f.output);
        run(&ctx, |enc| f.encode(&ctx, enc, candidate, true));
        f.validate(&read(&f.y));
        f.validate_reduction(&read(&f.output));
        for (t, before) in views.iter().zip(&inactive) {
            assert_eq!(&backing(t)[GUARD + t.n_bytes() as usize..], before);
        }
    }
}

struct Composition {
    f: Fixture,
    input: MetalTensor,
    routes: MetalTensor,
}

impl Composition {
    fn new(ctx: &MetalContext) -> Self {
        let _allocation = ctx.begin_allocation_transaction();
        let f = Fixture::new(ctx, 256, 256, 32, 1024);
        let input: Vec<f32> = (0..f.m * f.rows)
            .map(|i| (((i * 37 + i / f.m * 19) % 211) as f32 - 105.0) * 0.003713)
            .collect();
        let input = guarded(
            ctx,
            bytemuck::cast_slice(&input),
            vec![f.m as u64, f.rows as u64],
            GgmlType::F32,
        );
        let routes: Vec<i32> = f.owners.iter().map(|&e| e as i32).collect();
        let routes = guarded(
            ctx,
            bytemuck::cast_slice(&routes),
            vec![TOP as u64, f.rows as u64],
            GgmlType::I32,
        );
        Self { f, input, routes }
    }

    fn tensors(&self) -> [MetalTensor; 11] {
        [
            &self.f.w,
            &self.f.w,
            &self.f.w,
            &self.input,
            &self.routes,
            &self.f.weights,
            &self.f.counts,
            &self.f.ids,
            &self.f.x,
            &self.f.y,
            &self.f.output,
        ]
        .map(Clone::clone)
    }

    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        candidate: bool,
    ) -> Result<(), MetalError> {
        let _scope = Scope::new(candidate);
        let policy = packed::grouped_down_policy(
            HALF,
            StageMode::Fast,
            GgmlType::IQ3_S,
            self.f.k,
            self.f.m,
            self.f.e,
            self.f.rows,
        );
        self.encode_bindings(ctx, enc, &self.tensors(), 7.0, policy)
    }

    fn encode_bindings(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        tensors: &[MetalTensor; 11],
        clamp: f32,
        policy: GroupedDownPolicy,
    ) -> Result<(), MetalError> {
        let [
            gate_bank,
            up_bank,
            down_bank,
            input,
            ids,
            weights,
            counts,
            slots,
            inner,
            slot_out,
            output,
        ] = tensors;
        let b = GroupedExperts {
            gate_bank,
            up_bank,
            down_bank,
            input,
            ids,
            weights,
            counts,
            slots,
            inner,
            slot_out,
            output,
        };
        metal::encode_grouped_routed_experts_with_down_policy(
            ctx,
            enc,
            &b,
            self.f.m,
            self.f.k,
            self.f.e,
            TOP,
            self.f.rows,
            clamp,
            policy,
        )
    }
}

fn assert_no_mutation(ctx: &MetalContext, body: impl FnOnce(&KernelEncoder)) {
    // Drain unsubmitted autoreleased buffers before the refusal sweep exhausts
    // the command queue's bounded pool. Failed preparations stay unsubmitted.
    objc2::rc::autoreleasepool(|_| {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        dispatch_census_begin();
        body(&enc);
        assert!(dispatch_census_take().is_empty());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
    });
}

#[test]
#[ignore = "actual M5, validation ON, production lease"]
fn m5_down_composition_preserves_fused_gate_up_then_unweighted_down_then_sum() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let c = Composition::new(&ctx);
    let inputs = [&c.f.w, &c.input, &c.routes, &c.f.weights];
    let before: Vec<_> = inputs.iter().map(|t| fingerprint(t)).collect();
    let baseline = run(&ctx, |enc| c.encode(&ctx, enc, false).unwrap());
    let inner_a = read(&c.f.x);
    let slots_a = read(&c.f.y);
    let out_a = read(&c.f.output);
    poison(&c.f.x);
    poison(&c.f.y);
    poison(&c.f.output);
    let candidate = run(&ctx, |enc| c.encode(&ctx, enc, true).unwrap());
    assert_eq!(candidate.len(), baseline.len());
    let down = candidate.len() - 2;
    assert_eq!(baseline[down].kernel, GENERIC);
    assert_eq!(candidate[down].kernel, TENSOR);
    assert!(
        down >= 3,
        "bucketing and sized fused gate/up split must remain"
    );
    for (i, (a, b)) in baseline.iter().zip(&candidate).enumerate() {
        if i != down {
            assert_eq!(a.kernel, b.kernel);
            assert!(!b.kernel.contains("tensor"));
            assert_eq!(
                (a.grid_width, a.grid_height, a.grid_depth),
                (b.grid_width, b.grid_height, b.grid_depth)
            );
        }
    }
    assert_eq!(
        inner_a,
        read(&c.f.x),
        "routing weights must NOT be applied before down"
    );
    for (a, b) in [(&slots_a, read(&c.f.y)), (&out_a, read(&c.f.output))] {
        let scale = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(scale > 1e-7);
        assert!(a.iter().zip(&b).all(|(&a, &b)| a.is_finite()
            && b.is_finite()
            && (a - b).abs() <= 3e-3 * scale + 1e-5));
    }
    let slot_out = read(&c.f.y);
    let reduced = read(&c.f.output);
    for (i, &a) in reduced.iter().enumerate() {
        let expected: f64 = (0..TOP)
            .map(|lane| {
                let slot = i / c.f.m * TOP + lane;
                f64::from(slot_out[slot * c.f.m + i % c.f.m])
                    * f64::from(Fixture::route_weight(slot))
            })
            .sum();
        assert!((f64::from(a) - expected).abs() <= 2e-6 * (1.0 + expected.abs()));
    }
    for (t, before) in inputs.iter().zip(before) {
        assert_eq!(fingerprint(t), before);
    }
    for t in c.tensors() {
        guards(&t);
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease; every refusal precedes bucket/gate-up"]
fn m5_down_invalid_bindings_and_preflight_failures_do_not_mutate() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let c = Composition::new(&ctx);
    let before: Vec<_> = c.tensors().iter().map(fingerprint).collect();
    for index in 0..5 {
        let mut bindings = [&c.f.w, &c.f.x, &c.f.counts, &c.f.ids, &c.f.y].map(Clone::clone);
        bindings[index].shape = vec![u64::MAX, u64::MAX];
        let [w, x, counts, ids, out] = &bindings;
        assert_no_mutation(&ctx, |enc| {
            assert!(
                adapter::prepare(
                    &ctx, enc, w, x, counts, ids, out, c.f.k, c.f.m, c.f.e, TOP, c.f.rows
                )
                .is_err()
            );
        });
    }
    let mut readonly = c.f.y.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let mut alias = c.f.y.clone();
    alias.buffer = c.f.x.buffer.clone();
    for out in [&readonly, &alias] {
        assert_no_mutation(&ctx, |enc| {
            assert!(
                adapter::prepare(
                    &ctx,
                    enc,
                    &c.f.w,
                    &c.f.x,
                    &c.f.counts,
                    &c.f.ids,
                    out,
                    c.f.k,
                    c.f.m,
                    c.f.e,
                    TOP,
                    c.f.rows
                )
                .is_err()
            );
        });
    }
    for index in 0..11 {
        for fault in 0..6 {
            let mut tensors = c.tensors();
            let t = &mut tensors[index];
            match fault {
                0 => t.shape = vec![u64::MAX, u64::MAX],
                1 => t.offset = u64::MAX - 3,
                2 => t.offset = t.buffer.length() as u64,
                3 => t.offset += 1,
                4 => t.shape[0] -= 1,
                _ => t.dtype = GgmlType::Unknown,
            }
            assert_no_mutation(&ctx, |enc| {
                assert!(
                    c.encode_bindings(&ctx, enc, &tensors, 7.0, GroupedDownPolicy::Iq3STensor)
                        .is_err(),
                    "binding={index}, fault={fault}"
                )
            });
        }
    }
    for index in 6..11 {
        let mut tensors = c.tensors();
        tensors[index].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        let _capacity = adapter::FaultScope::new(adapter::PreflightFault::Capacity);
        assert_no_mutation(&ctx, |enc| {
            assert!(
                c.encode_bindings(&ctx, enc, &tensors, 7.0, GroupedDownPolicy::Iq3STensor)
                    .is_err()
            )
        });
        tensors = c.tensors();
        // Keep both views physically valid, so this exercises alias checks
        // rather than merely rejecting an undersized bank allocation.
        tensors[0].buffer = c.f.x.buffer.clone();
        tensors[index].buffer = c.f.x.buffer.clone();
        assert_no_mutation(&ctx, |enc| {
            assert!(
                c.encode_bindings(&ctx, enc, &tensors, 7.0, GroupedDownPolicy::Iq3STensor)
                    .is_err(),
                "scratch aliases bank {index}"
            )
        });
    }
    let mut tensors = c.tensors();
    tensors[9].buffer = tensors[8].buffer.clone();
    assert_no_mutation(&ctx, |enc| {
        assert!(
            c.encode_bindings(&ctx, enc, &tensors, 7.0, GroupedDownPolicy::Iq3STensor)
                .is_err()
        )
    });
    for index in [0, 1, 2, 3, 8, 9] {
        let mut tensors = c.tensors();
        tensors[index].offset += 8;
        assert_no_mutation(&ctx, |enc| {
            assert!(
                c.encode_bindings(&ctx, enc, &tensors, 7.0, GroupedDownPolicy::Iq3STensor)
                    .is_err()
            )
        });
    }
    for clamp in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_no_mutation(&ctx, |enc| {
            assert!(
                c.encode_bindings(
                    &ctx,
                    enc,
                    &c.tensors(),
                    clamp,
                    GroupedDownPolicy::Iq3STensor
                )
                .is_err()
            )
        });
    }
    {
        let _missing = adapter::FaultScope::new(adapter::PreflightFault::MissingPipeline);
        assert_no_mutation(&ctx, |enc| {
            assert!(matches!(
                c.encode(&ctx, enc, true),
                Err(MetalError::NoFunction(_))
            ));
        });
    }
    {
        let _capacity = adapter::FaultScope::new(adapter::PreflightFault::Capacity);
        assert_no_mutation(&ctx, |enc| {
            assert!(c.f.prepare(&ctx, enc).unwrap().is_none())
        });
    }
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&command);
    dispatch_census_begin();
    assert!(c.encode(&ctx, &enc, true).is_err());
    assert!(c.f.prepare(&ctx, &enc).is_err());
    assert!(enc.bound_pipeline_name.borrow().is_none());
    assert!(dispatch_census_take().is_empty());
    enc.end();
    let command = ctx.queue.commandBuffer().unwrap();
    let mut enc = KernelEncoder::begin(&command);
    enc.raw.endEncoding();
    enc.ended = true;
    assert!(c.encode(&ctx, &enc, true).is_err());
    assert!(c.f.prepare(&ctx, &enc).is_err());
    assert!(enc.bound_pipeline_name.borrow().is_none());
    for (t, before) in c.tensors().iter().zip(before) {
        assert_eq!(fingerprint(t), before);
    }
    // Capacity fallback remains generic, not a partial tensor substitution.
    let _capacity = adapter::FaultScope::new(adapter::PreflightFault::Capacity);
    let census = run(&ctx, |enc| c.encode(&ctx, enc, true).unwrap());
    assert_eq!(census[census.len() - 2].kernel, GENERIC);
    assert!(read(&c.f.output).iter().all(|v| v.is_finite()));
}

fn fallback_witness(ctx: &MetalContext) {
    let c = Composition::new(ctx);
    {
        // If the tier/global decline accidentally prepares any tensor PSO,
        // this turns that mistake into a visible error instead of a fallback.
        let _missing = adapter::FaultScope::new(adapter::PreflightFault::MissingPipeline);
        assert_no_mutation(ctx, |enc| assert!(c.f.prepare(ctx, enc).unwrap().is_none()));
        assert!(read(&c.f.y).iter().all(|v| v.to_bits() == u32::MAX));
        let census = run(ctx, |enc| c.encode(ctx, enc, true).unwrap());
        assert_eq!(census[census.len() - 2].kernel, GENERIC);
        assert!(census.iter().all(|r| !r.kernel.contains("tensor")));
    }
    let a = read(&c.f.output);
    let census = run(ctx, |enc| c.encode(ctx, enc, false).unwrap());
    assert_eq!(census[census.len() - 2].kernel, GENERIC);
    assert_eq!(a, read(&c.f.output));
    assert!(a.iter().all(|v| v.is_finite()));
    for t in c.tensors() {
        guards(&t);
    }
}

#[test]
#[ignore = "actual M4, validation ON, production lease"]
fn m4_down_declines_before_preparation_and_keeps_generic_composition() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.device.name().to_string().contains("M4"),
        "requires actual M4 hardware"
    );
    assert!(matches!(
        ctx.tensor_ops_status(),
        TensorOpsStatus::Unavailable {
            reason: TensorOpsUnavailableReason::GpuFamilyAbsent { .. }
        }
    ));
    fallback_witness(&ctx);
}

#[test]
#[ignore = "actual M5 fresh process QWEN_EXPERT_TENSOR_TILE=0, validation ON, production lease"]
fn m5_down_fresh_process_global_zero_beats_family_scope() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    assert!(ctx.device.name().to_string().contains("M5"));
    assert!(ctx.tensor_ops_status().is_available());
    assert_eq!(std::env::var("QWEN_EXPERT_TENSOR_TILE").as_deref(), Ok("0"));
    assert!(!moe_grouped_tensor_selected(&ctx));
    fallback_witness(&ctx);
}

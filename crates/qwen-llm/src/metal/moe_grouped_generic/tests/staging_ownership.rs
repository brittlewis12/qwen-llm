use super::*;

const EDGES: [usize; 11] = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65];

#[derive(Clone, Copy, Debug)]
enum Body {
    Narrow,
    ExpertWide,
    DenseWide,
}

impl Body {
    const ALL: [Self; 3] = [Self::Narrow, Self::ExpertWide, Self::DenseWide];

    fn slots(self) -> usize {
        if matches!(self, Self::Narrow) { 32 } else { 64 }
    }

    fn source(self) -> &'static str {
        let (start, end) = match self {
            Self::Narrow => (
                "inline void moe_grouped_slots_mm_tensor_body(",
                "// 64-slot gate/up tile.",
            ),
            Self::ExpertWide => (
                "kernel void kernel_moe_wide_probe_f16act(",
                "kernel void kernel_moe_grouped_slots_mm_tensor(",
            ),
            Self::DenseWide => (
                "inline void moe_mat_mat_tensor_wide_body(",
                "kernel void kernel_mat_mat_tensor_wide(",
            ),
        };
        TENSOR_METAL
            .split_once(start)
            .unwrap()
            .1
            .split_once(end)
            .unwrap()
            .0
    }
}

// The checker enumerates the consumer's row-major cells independently of the
// producer's thread/fragment mapping. Counts, not identical stored values, prove
// ownership: even identical concurrent stores are a race.
#[derive(Clone, Copy, Default)]
struct Cell {
    writers: u16,
    value: u32,
}

impl Cell {
    fn write(&mut self, value: u32) {
        self.writers += 1;
        self.value = value;
    }
}

fn stage_model(body: Body, nr0: usize, nr1: usize, old: bool) -> [Vec<Cell>; 2] {
    let mut slabs = [
        vec![Cell::default(); 64 * 32],
        vec![Cell::default(); body.slots() * 32],
    ];
    for thread in 0..128 {
        let physical_weight = thread / 2;
        let source_weight = physical_weight.min(nr0 - 1);
        let weight_dst = if old { source_weight } else { physical_weight };
        for lane in 0..16 {
            let k = (thread % 2) * 16 + lane;
            let value = if old || physical_weight < nr0 {
                1 + (source_weight * 32 + k) as u32
            } else {
                0
            };
            slabs[0][weight_dst * 32 + k].write(value);
        }
        for panel in 0..body.slots() / 32 {
            let physical_slot = thread / 4 + panel * 32;
            let source_slot = physical_slot.min(nr1 - 1);
            let slot_dst = if old && matches!(body, Body::Narrow) {
                source_slot
            } else {
                physical_slot
            };
            for lane in 0..8 {
                let k = (thread % 4) * 8 + lane;
                let value = if physical_slot < nr1 || (old && matches!(body, Body::Narrow)) {
                    1 + (source_slot * 32 + k) as u32
                } else {
                    0
                };
                slabs[1][slot_dst * 32 + k].write(value);
            }
        }
    }
    slabs
}

fn defects(cells: &[Cell]) -> (usize, usize) {
    (
        cells.iter().filter(|cell| cell.writers > 1).count(),
        cells.iter().filter(|cell| cell.writers == 0).count(),
    )
}

#[test]
fn staging_physical_fragments_exhaustive_ownership_and_zero_coverage() {
    let mut tiles = 0;
    let mut cells = 0;
    for body in Body::ALL {
        for nr0 in 1..=64 {
            for nr1 in 1..=body.slots() {
                let slabs = stage_model(body, nr0, nr1, false);
                for (slab, valid_rows) in slabs.iter().zip([nr0, nr1]) {
                    assert_eq!(defects(slab), (0, 0), "{body:?} {nr0}/{nr1}");
                    for (index, cell) in slab.iter().enumerate() {
                        let expected = if index / 32 < valid_rows {
                            1 + index as u32
                        } else {
                            0
                        };
                        assert_eq!(cell.value, expected, "{body:?} {nr0}/{nr1} cell={index}");
                        cells += 1;
                    }
                }
                tiles += 1;
            }
        }
    }
    assert_eq!(tiles, 10_240);
    assert_eq!(cells, 39_845_888);
    eprintln!("staging ownership: {tiles} physical tiles, {cells} half cells, one writer/cell");
}

#[test]
fn staging_old_mapping_red_control_detects_collisions_and_holes() {
    let mut rejected = 0;
    for body in Body::ALL {
        for nr0 in 1..=64 {
            for nr1 in 1..=body.slots() {
                let slabs = stage_model(body, nr0, nr1, true);
                let weight_bad = nr0 < 64;
                let activation_bad = matches!(body, Body::Narrow) && nr1 < 32;
                for (slab, should_fail) in slabs.iter().zip([weight_bad, activation_bad]) {
                    let (collisions, holes) = defects(slab);
                    assert_eq!(collisions > 0, should_fail, "{body:?} {nr0}/{nr1}");
                    assert_eq!(holes > 0, should_fail, "{body:?} {nr0}/{nr1}");
                }
                rejected += usize::from(weight_bad || activation_bad);
            }
        }
    }
    assert_eq!(rejected, 10_111);
    // A concrete historical counterexample, not just a model that accepts itself.
    let [weights, activations] = stage_model(Body::Narrow, 1, 1, true);
    assert_eq!(defects(&weights), (32, 63 * 32));
    assert_eq!(defects(&activations), (32, 31 * 32));
    eprintln!("staging red control: {rejected} old-map tiles rejected with collisions AND holes");
}

#[test]
fn staging_source_contract_matches_physical_model_and_retains_k_order() {
    for geometry in [
        "#define TT_NR0 64",
        "#define TT_NR1 32",
        "#define TT_NR1_WIDE 64",
        "#define TT_NK  32",
        "#define TT_NL0 (TT_NK / 16)",
        "#define TT_NL1 (TT_NK / 8)",
    ] {
        assert!(TENSOR_METAL.contains(geometry));
    }
    for body in Body::ALL {
        let source = body.source();
        for needle in [
            "threadgroup half * stage_a = sa + pr0 * TT_NK + 16 * il0;",
            "half4x4 temp_a(0.0h);",
            "if (pr0 < nr0) DEQ(x_ptr, il, temp_a);",
            "il = (il + 2 < NL) ? il + 2 : il % 2;",
            "BYTES * ((2 + NL - 1) / NL)",
        ] {
            assert!(source.contains(needle), "{body:?}: missing {needle}");
        }
        assert!(
            source.contains("pr0 = (short)tiitg / TT_NL0;")
                || source.contains("pr0 = tiitg / TT_NL0;")
        );
        assert!(!source.contains("stage_a = sa + lr0"));
        assert!(
            source.contains("lr0 = pr0 < nr0 ? pr0 : nr0 - 1;")
                || source.contains("lr0 = min(int(pr0), int(nr0) - 1);")
        );
        for (offset, column) in [(0, 0), (4, 1), (8, 2), (12, 3)] {
            let store = format!("*(threadgroup half4 *)(stage_a + {offset})");
            let line = source.lines().find(|line| line.contains(&store)).unwrap();
            assert!(line.contains(&format!("= temp_a[{column}];")));
        }
        match body {
            Body::Narrow => {
                assert!(source.contains("pr1 = (short)tiitg / TT_NL1;"));
                assert!(source.contains("lr1 = pr1 < nr1 ? pr1 : nr1 - 1;"));
                assert!(source.contains("il0 = (tiitg % TT_NL0);"));
                assert!(source.contains("iy = 8 * (tiitg % TT_NL1);"));
                assert!(source.contains("stage_b = sb + pr1 * TT_NK + iy;"));
                assert!(source.contains("*(threadgroup half2x4 *)(stage_b) = yv;"));
                assert!(
                    source.contains(
                        "slot_valid = pr1 < nr1 && slot_id >= 0 && slot_id < slot_limit;"
                    )
                );
                assert!(source.contains("yv = slot_valid"));
                assert!(source.contains(": half2x4(0.0h);"));
            }
            Body::ExpertWide | Body::DenseWide => {
                for needle in [
                    "valid0 = lr1 < nr1;",
                    "stage_b0 = sb +",
                    "stage_b1 = sb +",
                    "else *(threadgroup half2x4 *)(stage_b0) = half2x4(0.0h);",
                    "else *(threadgroup half2x4 *)(stage_b1) = half2x4(0.0h);",
                ] {
                    assert!(source.contains(needle), "{body:?}: {needle}");
                }
                let (p0, p1) = if matches!(body, Body::ExpertWide) {
                    ("lr1", "(lr1 + 32)")
                } else {
                    ("phys0", "phys1")
                };
                assert!(source.contains(&format!("stage_b0 = sb + {p0} * TT_NK + iy;")));
                assert!(source.contains(&format!("stage_b1 = sb + {p1} * TT_NK + iy;")));
                if matches!(body, Body::DenseWide) {
                    assert!(source.contains("phys0 = lr1;"));
                    assert!(source.contains("phys1 = lr1 + 32;"));
                }
            }
        }
        let first_store = source.find("*(threadgroup half4 *)(stage_a").unwrap();
        let run = source.find("op.run(left, right, acc);").unwrap();
        assert!(source[..first_store].contains("threadgroup_barrier(mem_flags::mem_threadgroup);"));
        assert!(
            source[first_store..run].contains("threadgroup_barrier(mem_flags::mem_threadgroup);")
        );
    }
    let mut chunks = 0;
    for layout in MOE_GROUPED_GENERIC_LAYOUTS {
        let nl = layout.block_elems / 16;
        for k in [256, 512, 2304, 5120] {
            for half in 0..2 {
                let mut il = half;
                let mut block = half / nl;
                for step in 0..k / 32 {
                    assert_eq!(block * nl + il, step * 2 + half, "{} K={k}", layout.suffix);
                    il = if il + 2 < nl { il + 2 } else { il % 2 };
                    if il < 2 {
                        block += 2_usize.div_ceil(nl);
                    }
                    chunks += 1;
                }
            }
        }
    }
    assert_eq!(chunks, 11_264);
    eprintln!("staging K order: {chunks} canonical 16-value chunks across all 22 layouts");
}

#[test]
fn staging_counts_and_output_tails_exhaustive_tile_coverage() {
    let mut shapes = 0;
    for slots in [32usize, 64] {
        for count in 0..=129usize {
            for m in 1..=129usize {
                let mut output = vec![0u8; count * m];
                for r1 in (0..129).step_by(slots) {
                    if r1 >= count {
                        continue;
                    }
                    let nr1 = (count - r1).min(slots);
                    for r0 in (0..m).step_by(64) {
                        let nr0 = (m - r0).min(64);
                        for thread in 0..128 {
                            for panel in 0..slots / 32 {
                                let j = thread / 4 + panel * 32;
                                if j >= nr1 {
                                    continue;
                                }
                                let c0 = thread % 4 * 16;
                                for c in c0..(c0 + 16).min(nr0) {
                                    output[(r1 + j) * m + r0 + c] += 1;
                                }
                            }
                        }
                    }
                }
                assert!(
                    output.iter().all(|&writers| writers == 1),
                    "slots={slots} count={count} M={m}"
                );
                shapes += 1;
            }
        }
    }
    assert_eq!(shapes, 33_540);
    for body in [Body::Narrow, Body::ExpertWide] {
        let source = body.source();
        let guard = "if (count < 0 || uint(count) > args.N\n        || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;";
        assert!(source.contains(guard));
        let end = source.find(guard).unwrap() + guard.len();
        assert!(end < source.find("ids[").unwrap());
        assert!(end < source.find("threadgroup_barrier(").unwrap());
    }
    for count in [
        i32::MIN,
        -7,
        -1,
        0,
        1,
        2,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        66,
        i32::MAX,
    ] {
        for (min, max) in [(0, i32::MAX), (2, 64), (33, 65)] {
            for r1 in [0, 32, 64, 96, 128] {
                let reject =
                    count < 0 || count as u32 > 65 || count < min || count > max || r1 >= count;
                let admitted = (min.max(1)..=max.min(65)).contains(&count) && r1 < count;
                assert_eq!(!reject, admitted);
            }
        }
    }
    eprintln!(
        "staging output coverage: {shapes} shapes (all counts 0..129, M 1..129), 255 count guards"
    );
}

const K: usize = 512;
const CAPACITY: usize = 65;
const EXPERTS: usize = 5;
const SLOT_LIMIT: usize = 2 * CAPACITY * EXPERTS;
const GUARD: usize = 32;
const CANARY: f32 = -777.0;
const CLAMP: f32 = 0.1875;

fn half_values(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect()
}

fn boundary_shapes() -> Vec<(usize, usize)> {
    let mut shapes: Vec<_> = EDGES.iter().map(|&n| (n, 65)).collect();
    shapes.extend(EDGES[..10].iter().map(|&m| (65, m)));
    shapes.extend([(1, 1), (2, 2), (17, 17), (33, 33)]);
    assert_eq!(shapes.len(), 25);
    shapes
}

fn require_tensor_context() -> MetalContext {
    assert_eq!(
        std::env::var("MTL_DEBUG_LAYER").as_deref(),
        Ok("1"),
        "run witnesses with Metal validation"
    );
    let ctx = MetalContext::new().expect("staging witness requires Metal, not a skip/pass");
    assert!(
        moe_grouped_tensor_available(&ctx),
        "staging witness requires executable M5 tensor tier: {:?}",
        ctx.tensor_ops_status()
    );
    ctx
}

// Admission precedes even host fixture construction. One fixture is live at a
// time; the inventory includes both activation types and one guarded output.
fn fixture_inventory(dtype: GgmlType, m: usize, rows: usize, experts: usize) -> [usize; 6] {
    let (block, bytes) = dtype.storage_layout().unwrap();
    [
        experts * m * K / block as usize * bytes as usize + 2 * GUARD,
        rows * K * 4 + 2 * GUARD,
        rows * K * 2 + 2 * GUARD,
        experts * 4 + 2 * GUARD,
        experts * CAPACITY * 4 + 2 * GUARD,
        rows * m * 4 + 2 * GUARD,
    ]
}

fn admit(
    ctx: &MetalContext,
    dtype: GgmlType,
    m: usize,
    rows: usize,
    experts: usize,
) -> crate::metal::census::MetalAllocationTransactionGuard {
    let inventory = fixture_inventory(dtype, m, rows, experts);
    let priced: u64 = inventory
        .iter()
        .map(|&n| {
            ctx.price_shared_buffer_upper(n as u64)
                .unwrap()
                .priced_upper_bytes
        })
        .sum();
    // Decoded/half banks, reference sums, staging/reallocation, immutable-input
    // snapshots and readbacks are bounded by sixteen copies of this inventory.
    let cpu = 16 * inventory.iter().sum::<usize>() as u64;
    assert!(priced + cpu <= 64 << 20);
    let transaction = ctx.begin_allocation_transaction();
    let decision =
        evaluate_metal_memory_admission_with_cpu_bytes(priced, cpu, 0, ctx.memory_signals(), true);
    assert!(
        decision.is_admitted(),
        "staging fixture admission: {decision:?}"
    );
    eprintln!(
        "staging fixture dtype={dtype:?} K={K} M={m} rows={rows} experts={experts} priced_metal_bytes={priced} cpu_bytes={cpu} limit_bytes={}",
        64 << 20
    );
    transaction
}

fn view(ctx: &MetalContext, bytes: &[u8], dtype: GgmlType, shape: Vec<u64>) -> MetalTensor {
    offset_tensor(ctx, GUARD, bytes, GUARD, shape, dtype)
}

fn check_result(actual: &[f32], expected: &[f32], label: &str) -> usize {
    assert_eq!(actual.len(), expected.len());
    let mut active = Vec::new();
    let mut reference = Vec::new();
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(a.is_finite() && e.is_finite(), "{label}: nonfinite result");
        if e == CANARY {
            assert_eq!(
                a.to_bits(),
                CANARY.to_bits(),
                "{label}: inactive output changed"
            );
        } else {
            active.push(a);
            reference.push(e);
        }
    }
    if !active.is_empty() {
        let max_ref = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let error = max_abs_diff_finite(label, &active, &reference);
        assert!(
            error <= 2e-3 * max_ref + 2e-5,
            "{label}: active-only error={error} range={max_ref}"
        );
    }
    active.len()
}

#[test]
fn staging_active_only_comparison_red_control() {
    let mut expected = vec![CANARY; 1024];
    expected[13] = 0.125;
    assert_eq!(check_result(&expected, &expected, "positive control"), 1);
    let mut bad = expected.clone();
    bad[13] += 0.01;
    assert!(
        std::panic::catch_unwind(|| check_result(&bad, &expected, "active red control")).is_err()
    );
    bad = expected.clone();
    bad[1] = 0.0;
    assert!(
        std::panic::catch_unwind(|| check_result(&bad, &expected, "guard red control")).is_err()
    );
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        bad = expected.clone();
        bad[13] = nonfinite;
        assert!(
            std::panic::catch_unwind(|| check_result(&bad, &expected, "nonfinite red control"))
                .is_err()
        );
    }
}

#[test]
fn staging_witness_plan_and_cpu_half_fixtures_are_bounded_nontrivial() {
    let shapes = boundary_shapes();
    for edge in EDGES {
        assert!(shapes.iter().any(|&(count, _)| count == edge));
        assert!(shapes.iter().any(|&(_, m)| m == edge));
    }
    for dtype in [
        GgmlType::F32,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::IQ1_S,
        GgmlType::IQ1_M,
        GgmlType::IQ3_S,
        GgmlType::IQ3_XXS,
    ] {
        let payload: usize = fixture_inventory(dtype, 65, SLOT_LIMIT, EXPERTS)
            .iter()
            .sum();
        assert!(17 * payload < 64 << 20);
        let bank = synthetic_bank(dtype, K, 65, EXPERTS, 0x1234 ^ dtype as u64);
        for e in 0..EXPERTS {
            let w = half_values(&cpu_expert(dtype, &bank, K, 65, e));
            assert!(w.iter().all(|v| v.is_finite()));
            let x: Vec<_> = (0..K)
                .map(|i| half::f16::from_f32(((i * 17 % 131) as f32 - 65.0) / 137.0).to_f32())
                .collect();
            let result = matvec(&w, K, &x);
            assert!(result.iter().all(|v| v.is_finite()));
            assert!(result.iter().any(|v| v.abs() > 1e-5));
        }
        eprintln!(
            "staging CPU fixture dtype={dtype:?}: max_buffer_payload_bytes={payload} cpu_envelope_bytes={} (device pricing/admission deferred to M5)",
            16 * payload
        );
    }
}

struct GroupedFixture {
    m: usize,
    weight: MetalTensor,
    x32: MetalTensor,
    x16: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    projection: Vec<f32>,
    initial: Vec<f32>,
    active: Vec<bool>,
    b_div: usize,
    min: u32,
    max: u32,
    _transaction: crate::metal::census::MetalAllocationTransactionGuard,
}

impl GroupedFixture {
    fn new(
        ctx: &MetalContext,
        dtype: GgmlType,
        m: usize,
        count: i32,
        b_div: usize,
        min: u32,
        max: u32,
        bad_ids: bool,
    ) -> Self {
        let transaction = admit(ctx, dtype, m, SLOT_LIMIT, EXPERTS);
        let bank = synthetic_bank(dtype, K, m, EXPERTS, 0x1234 ^ dtype as u64);
        let x: Vec<f32> = (0..SLOT_LIMIT / b_div * K)
            .map(|i| (((i * 17 + i / K * 13) % 131) as f32 - 65.0) / 137.0)
            .collect();
        let hx = half_values(&x);
        let counts = [count, 2, 0, -7, 66];
        let mut ids = vec![0i32; EXPERTS * CAPACITY];
        for e in 0..EXPERTS {
            for i in 0..CAPACITY {
                ids[e * CAPACITY + i] = (2 * (e * CAPACITY + (i * 17) % CAPACITY) + 1) as i32;
            }
        }
        // Retain a valid first slot even at count=1; place bad IDs in both
        // 32-slot halves and on the 64-slot panel boundary when present.
        if bad_ids {
            for (index, bad) in [
                (1, -1),
                (15, SLOT_LIMIT as i32),
                (31, i32::MAX),
                (32, i32::MIN),
                (63, -9),
            ] {
                ids[index] = bad;
            }
        }
        let mut projection = vec![CANARY; SLOT_LIMIT * m];
        let initial: Vec<f32> = (0..SLOT_LIMIT * m)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 23.0)
            .collect();
        let mut active = vec![false; SLOT_LIMIT * m];
        for (e, &count) in counts.iter().enumerate() {
            if count <= 0 || count > CAPACITY as i32 || count < min as i32 || count > max as i32 {
                continue;
            }
            // Independent CPU codec, explicit Half operands, f64 dot rounded
            // once to the declared F32 accumulator result. No generic GPU oracle.
            let w = half_values(&cpu_expert(dtype, &bank, K, m, e));
            for &id in &ids[e * CAPACITY..][..count as usize] {
                if id < 0 || id as usize >= SLOT_LIMIT {
                    continue;
                }
                let slot = id as usize;
                let values = matvec(&w, K, &hx[slot / b_div * K..][..K]);
                projection[slot * m..][..m].copy_from_slice(&values);
                active[slot * m..][..m].fill(true);
            }
        }
        let admitted = |c| c > 0 && c <= CAPACITY as i32 && c >= min as i32 && c <= max as i32;
        let invalid_slots = if bad_ids {
            [1, 15, 31, 32, 63]
                .iter()
                .filter(|&&i| i < count as usize)
                .count()
        } else {
            0
        };
        let first_slots = if admitted(count) {
            count as usize - invalid_slots
        } else {
            0
        };
        let expected_slots = first_slots + if admitted(2) { 2 } else { 0 };
        assert_eq!(active.iter().filter(|&&v| v).count(), expected_slots * m);
        if expected_slots != 0 {
            assert!(
                projection
                    .iter()
                    .zip(&active)
                    .any(|(&v, &on)| on && v.abs() > 1e-7)
            );
        }
        let half_x: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
        Self {
            m,
            b_div,
            min,
            max,
            projection,
            initial,
            active,
            _transaction: transaction,
            weight: view(ctx, &bank, dtype, vec![K as u64, m as u64, EXPERTS as u64]),
            x32: view(
                ctx,
                bytemuck::cast_slice(&x),
                GgmlType::F32,
                vec![x.len() as u64],
            ),
            x16: view(
                ctx,
                bytemuck::cast_slice(&half_x),
                GgmlType::F16,
                vec![x.len() as u64],
            ),
            counts: view(
                ctx,
                bytemuck::cast_slice(&counts),
                GgmlType::I32,
                vec![EXPERTS as u64],
            ),
            ids: view(
                ctx,
                bytemuck::cast_slice(&ids),
                GgmlType::I32,
                vec![ids.len() as u64],
            ),
        }
    }

    fn run(&self, ctx: &MetalContext, wide: bool, activation: TensorActivation, epi: usize) {
        let x = if activation == TensorActivation::F16 {
            &self.x16
        } else {
            &self.x32
        };
        assert!(!wide || activation == TensorActivation::F16);
        let role = MoeGroupedTensorRole::ALL[epi];
        let name = if wide {
            wide_probe_pipeline_name(self.weight.dtype)
        } else {
            moe_grouped_tensor_pipeline_name(role, self.weight.dtype, activation)
        }
        .unwrap();
        let layout = moe_grouped_generic_layout(self.weight.dtype).unwrap();
        let args = GenericMmArgs {
            m: self.m as u32,
            n: CAPACITY as u32,
            k: K as u32,
            nb01: (K / layout.block_elems * layout.block_bytes) as u32,
            stride_b: K as u32,
            min_count: self.min,
            max_count: self.max,
            b_div: self.b_div as u32,
            slot_limit: SLOT_LIMIT as u32,
        };
        let mut expected = vec![CANARY; self.initial.len()];
        let mut initial = expected.clone();
        for i in 0..expected.len() {
            if !self.active[i] {
                continue;
            }
            initial[i] = self.initial[i];
            let d = f64::from(if epi == 2 {
                self.initial[i].min(CLAMP)
            } else {
                self.initial[i]
            });
            let c = if epi == 2 {
                self.projection[i].clamp(-CLAMP, CLAMP)
            } else {
                self.projection[i]
            };
            expected[i] = if epi == 0 {
                c
            } else {
                (d / (1.0 + (-d).exp()) * f64::from(c)) as f32
            };
        }
        let inputs = [&self.weight, x, &self.counts, &self.ids];
        let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
        // Repeat from the same initialized output, not recursively through the
        // epilogue. Numerical repetition is a witness, not an ownership proof.
        for _ in 0..2 {
            let out = view(
                ctx,
                bytemuck::cast_slice(&initial),
                GgmlType::F32,
                vec![initial.len() as u64],
            );
            dispatch_census_begin();
            one_shot(ctx, |enc| {
                enc.set_pipeline(&ctx.tensor_pipeline(&name)?);
                enc.set_bytes(0, &args);
                enc.set_tensor(1, &self.weight);
                enc.set_tensor(2, x);
                enc.set_tensor(3, &self.counts);
                enc.set_tensor(4, &self.ids);
                enc.set_tensor(5, &out);
                enc.set_bytes(6, &CLAMP);
                if wide {
                    enc.set_bytes(7, &(epi as i32));
                }
                enc.set_threadgroup_memory(0, if wide { 16384 } else { 8192 });
                enc.dispatch(
                    MTLSize {
                        width: CAPACITY.div_ceil(if wide { 64 } else { 32 }),
                        height: self.m.div_ceil(64),
                        depth: EXPERTS,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                Ok(())
            })
            .expect("staging grouped tensor execution");
            let census = dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].kernel, name);
            assert_eq!(census[0].tg_threads, 128);
            check_result(&tensor_f32_at_offset(&out), &expected, &name);
            assert_offset_guards(&out, GUARD, GUARD);
        }
        for (input, bytes) in inputs.iter().zip(before) {
            assert_eq!(
                tensor_backing_bytes(input),
                bytes,
                "{name}: immutable input changed"
            );
        }
    }
}

fn run_grouped(dtype: GgmlType) {
    let _lease = acquire_metal_benchmark_lease().expect("staging witness production lease");
    let ctx = require_tensor_context();
    let mut fixtures = 0;
    for (count, m, min, max, bad_ids) in boundary_shapes()
        .into_iter()
        .map(|(c, m)| (c as i32, m, 0, i32::MAX as u32, false))
        .chain([
            (0, 17, 0, i32::MAX as u32, false),
            (-1, 17, 0, i32::MAX as u32, false),
            (66, 17, 0, i32::MAX as u32, false),
            (i32::MIN, 17, 0, i32::MAX as u32, false),
            (i32::MAX, 17, 0, i32::MAX as u32, false),
            (17, 17, 18, 65, false),
            (33, 17, 0, 32, false),
            (65, 17, 0, i32::MAX as u32, true),
        ])
    {
        for b_div in [1, 2] {
            let f = GroupedFixture::new(&ctx, dtype, m, count, b_div, min, max, bad_ids);
            for epi in 0..3 {
                f.run(&ctx, false, TensorActivation::F32, epi);
                f.run(&ctx, false, TensorActivation::F16, epi);
                f.run(&ctx, true, TensorActivation::F16, epi);
            }
            fixtures += 1;
        }
    }
    assert_eq!(fixtures, 66);
    eprintln!(
        "staging grouped dtype={dtype:?}: {fixtures} fixtures, 1188 single-kernel command buffers (2 repeats), K={K}, <=64 MiB admitted/fixture"
    );
}

fn run_dense(dtype: GgmlType) {
    let _lease = acquire_metal_benchmark_lease().expect("staging dense witness production lease");
    let ctx = require_tensor_context();
    let mut dispatches = 0;
    for (rows, m) in boundary_shapes() {
        let _transaction = admit(&ctx, dtype, m, rows, 1);
        let bank = synthetic_bank(dtype, K, m, 1, 0xd00d ^ dtype as u64);
        let w = half_values(&cpu_expert(dtype, &bank, K, m, 0));
        let x: Vec<f32> = (0..rows * K)
            .map(|i| (((i * 19 + i / K * 11) % 127) as f32 - 63.0) / 139.0)
            .collect();
        let hx = half_values(&x);
        let expected: Vec<_> = hx
            .chunks_exact(K)
            .flat_map(|row| matvec(&w, K, row))
            .collect();
        let weight = view(&ctx, &bank, dtype, vec![K as u64, m as u64]);
        for activation in [TensorActivation::F32, TensorActivation::F16] {
            let x_bytes: Vec<u8> = if activation == TensorActivation::F32 {
                bytemuck::cast_slice(&x).to_vec()
            } else {
                x.iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                    .collect()
            };
            let input = view(
                &ctx,
                &x_bytes,
                if activation == TensorActivation::F32 {
                    GgmlType::F32
                } else {
                    GgmlType::F16
                },
                vec![K as u64, rows as u64],
            );
            for wide in [false, true] {
                if wide && dtype != GgmlType::Q8_0 {
                    continue;
                }
                let name = if wide {
                    mat_mat_tensor_wide_pipeline_name(dtype, activation)
                } else {
                    mat_mat_tensor_pipeline_name(dtype, activation)
                }
                .unwrap();
                let before = [tensor_backing_bytes(&weight), tensor_backing_bytes(&input)];
                for _ in 0..2 {
                    let out = view(
                        &ctx,
                        bytemuck::cast_slice(&vec![CANARY; rows * m]),
                        GgmlType::F32,
                        vec![m as u64, rows as u64],
                    );
                    dispatch_census_begin();
                    one_shot(&ctx, |enc| {
                        if wide {
                            encode_mat_mat_tensor_f32_wide(
                                &ctx, enc, &weight, &input, &out, K, m, rows,
                            )
                        } else {
                            encode_mat_mat_tensor_f32_narrow(
                                &ctx, enc, &weight, &input, &out, K, m, rows,
                            )
                        }
                    })
                    .expect("staging dense tensor execution");
                    let census = dispatch_census_take();
                    assert_eq!(census.len(), 1);
                    assert_eq!(census[0].kernel, name);
                    assert_eq!(census[0].tg_threads, 128);
                    assert_eq!(
                        check_result(&tensor_f32_at_offset(&out), &expected, &name),
                        rows * m
                    );
                    assert_offset_guards(&out, GUARD, GUARD);
                    dispatches += 1;
                }
                assert_eq!(tensor_backing_bytes(&weight), before[0]);
                assert_eq!(tensor_backing_bytes(&input), before[1]);
            }
        }
    }
    assert_eq!(dispatches, if dtype == GgmlType::Q8_0 { 200 } else { 100 });
    eprintln!(
        "staging dense dtype={dtype:?}: 25 shapes, {dispatches} single-kernel command buffers (2 repeats), K={K}, <=64 MiB admitted/fixture"
    );
}

macro_rules! witnesses {
    ($($grouped:ident, $dense:ident => $dtype:expr;)*) => {$(
        #[test]
        #[ignore = "M5 GPU witness; MTL_DEBUG_LAYER=1; self-leased; <=64 MiB/fixture; 1188 dispatches"]
        fn $grouped() { run_grouped($dtype); }
        #[test]
        #[ignore = "M5 GPU witness; MTL_DEBUG_LAYER=1; self-leased; <=64 MiB/fixture; 100 dispatches (Q8_0: 200)"]
        fn $dense() { run_dense($dtype); }
    )*};
}

witnesses! {
    staging_m5_grouped_f32, staging_m5_dense_f32 => GgmlType::F32;
    staging_m5_grouped_q8_0, staging_m5_dense_q8_0 => GgmlType::Q8_0;
    staging_m5_grouped_q4_k, staging_m5_dense_q4_k => GgmlType::Q4_K;
    staging_m5_grouped_iq1_s, staging_m5_dense_iq1_s => GgmlType::IQ1_S;
    staging_m5_grouped_iq1_m, staging_m5_dense_iq1_m => GgmlType::IQ1_M;
    staging_m5_grouped_iq3_s, staging_m5_dense_iq3_s => GgmlType::IQ3_S;
    staging_m5_grouped_iq3_xxs, staging_m5_dense_iq3_xxs => GgmlType::IQ3_XXS;
}

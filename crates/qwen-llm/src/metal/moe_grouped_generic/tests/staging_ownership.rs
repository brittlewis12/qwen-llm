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
const FIXTURE_LIMIT: usize = 64 << 20;
const WITNESS_DTYPES: [GgmlType; 7] = [
    GgmlType::F32,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::IQ3_S,
    GgmlType::IQ3_XXS,
];

fn half_values(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect()
}

fn boundary_shapes() -> Vec<(usize, usize)> {
    let mut shapes = Vec::with_capacity(25);
    shapes.extend(EDGES.iter().map(|&n| (n, 65)));
    shapes.extend(EDGES[..10].iter().map(|&m| (65, m)));
    shapes.extend([(1, 1), (2, 2), (17, 17), (33, 33)]);
    assert_eq!(shapes.len(), 25);
    shapes
}

#[derive(Clone, Copy, Debug)]
struct GroupedCase {
    count: i32,
    m: usize,
    min: u32,
    max: u32,
    bad_ids: bool,
}

fn grouped_cases() -> Vec<GroupedCase> {
    let mut cases = Vec::with_capacity(33);
    cases.extend(boundary_shapes().into_iter().map(|(count, m)| GroupedCase {
        count: count as i32,
        m,
        min: 0,
        max: i32::MAX as u32,
        bad_ids: false,
    }));
    cases.extend(
        [
            (0, 0, i32::MAX as u32, false),
            (-1, 0, i32::MAX as u32, false),
            (66, 0, i32::MAX as u32, false),
            (i32::MIN, 0, i32::MAX as u32, false),
            (i32::MAX, 0, i32::MAX as u32, false),
            (17, 18, 65, false),
            (33, 0, 32, false),
            (65, 0, i32::MAX as u32, true),
        ]
        .into_iter()
        .map(|(count, min, max, bad_ids)| GroupedCase {
            count,
            m: 17,
            min,
            max,
            bad_ids,
        }),
    );
    assert_eq!(cases.len(), 33);
    cases
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

#[derive(Clone, Copy, Debug)]
struct FixturePlan {
    dtype: GgmlType,
    m: usize,
    rows: usize,
    b_div: usize,
    grouped: bool,
}

#[derive(Clone, Copy, Debug)]
struct CpuBudget {
    source_bank: usize,
    decoded_pair: usize,
    activations: usize,
    references: usize,
    matvec_row: usize,
    guarded_staging: usize,
    snapshots: usize,
    readbacks: usize,
    bookkeeping: usize,
}

impl CpuBudget {
    fn total(self) -> usize {
        self.source_bank
            + self.decoded_pair
            + self.activations
            + self.references
            + self.matvec_row
            + self.guarded_staging
            + self.snapshots
            + self.readbacks
            + self.bookkeeping
    }
}

impl FixturePlan {
    fn grouped(dtype: GgmlType, m: usize, b_div: usize) -> Self {
        Self {
            dtype,
            m,
            rows: SLOT_LIMIT,
            b_div,
            grouped: true,
        }
    }

    fn dense(dtype: GgmlType, m: usize, rows: usize) -> Self {
        Self {
            dtype,
            m,
            rows,
            b_div: 1,
            grouped: false,
        }
    }

    // Weight / F32 input / F16 input / counts / IDs / one output. Both dense
    // activation buffers are explicit; absent routing buffers cost zero.
    fn inventory(self) -> [usize; 6] {
        let (block, bytes) = self.dtype.storage_layout().unwrap();
        let experts = if self.grouped { EXPERTS } else { 1 };
        let x = self.rows / self.b_div * K;
        [
            experts * self.m * K / block as usize * bytes as usize + 2 * GUARD,
            x * 4 + 2 * GUARD,
            x * 2 + 2 * GUARD,
            if self.grouped {
                experts * 4 + 2 * GUARD
            } else {
                0
            },
            if self.grouped {
                experts * CAPACITY * 4 + 2 * GUARD
            } else {
                0
            },
            self.rows * self.m * 4 + 2 * GUARD,
        ]
    }

    fn cpu_budget(self, upper: [usize; 6]) -> CpuBudget {
        let logical = self.inventory();
        let bank = logical[0] - 2 * GUARD;
        let x = self.rows / self.b_div * K;
        let out = self.rows * self.m;
        let routing = if self.grouped {
            EXPERTS * CAPACITY * 4
        } else {
            0
        };
        let input_upper = upper[1].max(upper[2]);
        let input_snapshot = upper[0] + input_upper + upper[3] + upper[4];
        // Sum named category peaks, even when their phases cannot overlap.
        // synthetic_bank's F32 flat_map can grow: old <= bank, new <= 2*bank.
        // The quantized branch allocates exactly bank bytes. Codec and Half
        // rounding deliberately retain two full F32 row-major banks together.
        CpuBudget {
            source_bank: bank * if self.dtype == GgmlType::F32 { 3 } else { 1 },
            decoded_pair: 2 * self.m * K * 4,
            activations: x * (4 + 4 + 2) + routing,
            references: if self.grouped {
                out * (4 + 4 + 1 + 4 + 4)
            } else {
                out * (4 + 4)
            },
            matvec_row: self.m * 4,
            guarded_staging: *logical.iter().max().unwrap(),
            snapshots: input_snapshot,
            // An F32 output readback can coexist with guard/backing comparison.
            readbacks: out * 4 + *upper.iter().max().unwrap(),
            // Census reserves 512 rows. Bound the encoder's two binding hazard
            // vectors, fixture/scheduler shapes, names and weak/snapshot slots.
            bookkeeping: 512 * std::mem::size_of::<DispatchCensusRow>()
                + 2 * 8 * std::mem::size_of::<(usize, u64, u64)>()
                + 33 * std::mem::size_of::<GroupedCase>()
                + 25 * std::mem::size_of::<(usize, usize)>()
                + 8 * (128 + 3 * std::mem::size_of::<u64>())
                + 6 * (std::mem::size_of::<Vec<u8>>() + std::mem::size_of::<BufferWeak>()),
        }
    }
}

struct Admission {
    upper: [usize; 6],
    _transaction: crate::metal::census::MetalAllocationTransactionGuard,
}

fn admit(ctx: &MetalContext, plan: FixturePlan) -> Admission {
    let transaction = ctx.begin_allocation_transaction();
    let upper = plan.inventory().map(|n| {
        if n == 0 {
            0
        } else {
            usize::try_from(
                ctx.price_shared_buffer_upper(n as u64)
                    .unwrap()
                    .priced_upper_bytes,
            )
            .unwrap()
        }
    });
    let priced = upper.iter().sum::<usize>();
    let components = plan.cpu_budget(upper);
    let cpu = components.total();
    assert!(priced + cpu <= FIXTURE_LIMIT);
    let decision = evaluate_metal_memory_admission_with_cpu_bytes(
        priced as u64,
        cpu as u64,
        0,
        ctx.memory_signals(),
        true,
    );
    assert!(
        decision.is_admitted(),
        "staging fixture admission: {decision:?}"
    );
    eprintln!(
        "staging fixture plan={plan:?} K={K} priced_metal_bytes={priced} cpu_bytes={cpu} components={components:?} limit_bytes={FIXTURE_LIMIT}"
    );
    Admission {
        upper,
        _transaction: transaction,
    }
}

fn guarded_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut guarded = vec![0xA5; bytes.len() + 2 * GUARD];
    guarded[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
    guarded[GUARD + bytes.len()..].fill(0x5A);
    guarded
}

fn view(ctx: &MetalContext, bytes: &[u8], dtype: GgmlType, shape: Vec<u64>) -> MetalTensor {
    // One exact staging allocation, rather than offset_tensor's growable Vec.
    let guarded = guarded_bytes(bytes);
    MetalTensor {
        buffer: ctx.buffer_from(&guarded).expect("guarded staging tensor"),
        offset: GUARD as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

type BufferWeak = objc2::rc::Weak<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

fn buffer_probe(tensor: &MetalTensor, upper: usize) -> BufferWeak {
    assert!(
        tensor.buffer.length() <= upper,
        "buffer exceeds admitted readback bound"
    );
    let weak = BufferWeak::from_retained(&tensor.buffer);
    assert!(weak.load().is_some(), "weak resource probe must start live");
    weak
}

fn assert_buffer_released(weak: &BufferWeak) {
    objc2::rc::autoreleasepool(|_| {
        assert!(
            weak.load().is_none(),
            "MTLBuffer survives completed work, Rust teardown and autorelease drain; stop before admitting more buffers"
        );
    });
}

fn check_result(actual: &[f32], expected: &[f32], label: &str) -> usize {
    assert_eq!(actual.len(), expected.len());
    let mut active = 0;
    let mut max_ref = 0.0f32;
    let mut error = 0.0f32;
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(a.is_finite() && e.is_finite(), "{label}: nonfinite result");
        if e == CANARY {
            assert_eq!(
                a.to_bits(),
                CANARY.to_bits(),
                "{label}: inactive output changed"
            );
        } else {
            active += 1;
            max_ref = max_ref.max(e.abs());
            let delta = (a - e).abs();
            assert!(delta.is_finite(), "{label}: nonfinite difference");
            error = error.max(delta);
        }
    }
    if active != 0 {
        assert!(
            error <= 2e-3 * max_ref + 2e-5,
            "{label}: active-only error={error} range={max_ref}"
        );
    }
    active
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
    for dtype in WITNESS_DTYPES {
        let plan = FixturePlan::grouped(dtype, 65, 1);
        let payload: usize = plan.inventory().iter().sum();
        let cpu = plan.cpu_budget(plan.inventory()).total();
        assert!(payload + cpu < FIXTURE_LIMIT);
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
            "staging CPU fixture dtype={dtype:?}: max_buffer_payload_bytes={payload} cpu_bytes={cpu} (device pricing/admission deferred to M5)"
        );
    }
}

fn modeled_upper(plan: FixturePlan, page: usize) -> [usize; 6] {
    plan.inventory().map(|bytes| {
        if bytes == 0 {
            0
        } else {
            price_shared_buffer_upper(
                bytes as u64,
                MetalBufferSizeAndAlign {
                    size: bytes as u64,
                    alignment: page as u64,
                },
                page as u64,
                u64::MAX,
            )
            .unwrap()
            .priced_upper_bytes as usize
        }
    })
}

fn check_budget_components(plan: FixturePlan, upper: [usize; 6]) {
    let budget = plan.cpu_budget(upper);
    let input_elements = plan.rows / plan.b_div * K;
    let output_elements = plan.rows * plan.m;
    // Independent necessary live allocations, expressed in logical elements,
    // not compressed bytes or the budget's category formulas.
    assert!(budget.decoded_pair >= 2 * plan.m * K * std::mem::size_of::<f32>());
    assert!(
        budget.activations
            >= input_elements * (2 * std::mem::size_of::<f32>() + std::mem::size_of::<half::f16>())
    );
    assert!(
        budget.references
            >= output_elements * std::mem::size_of::<f32>() * if plan.grouped { 4 } else { 2 }
    );
    assert!(budget.guarded_staging >= *plan.inventory().iter().max().unwrap());
    assert!(budget.snapshots >= upper[0] + upper[1].max(upper[2]) + upper[3] + upper[4]);
    assert!(budget.readbacks >= output_elements * 4 + *upper.iter().max().unwrap());
    assert!(
        budget.total() + upper.iter().sum::<usize>() <= FIXTURE_LIMIT,
        "{plan:?}: {budget:?}"
    );
}

fn check_observed_host_capacities(plan: FixturePlan, upper: [usize; 6]) {
    let experts = if plan.grouped { EXPERTS } else { 1 };
    let bank = synthetic_bank(plan.dtype, K, plan.m, experts, 0x1234 ^ plan.dtype as u64);
    let decoded = cpu_expert(plan.dtype, &bank, K, plan.m, 0);
    let rounded = half_values(&decoded);
    let x: Vec<_> = (0..plan.rows / plan.b_div * K)
        .map(|i| (i % 127) as f32 / 139.0)
        .collect();
    let hx = half_values(&x);
    let half_x: Vec<_> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
    let ids = vec![0i32; if plan.grouped { EXPERTS * CAPACITY } else { 0 }];
    let references: Vec<Vec<f32>> = (0..if plan.grouped { 4 } else { 2 })
        .map(|_| vec![CANARY; plan.rows * plan.m])
        .collect();
    let active = vec![false; if plan.grouped { plan.rows * plan.m } else { 0 }];
    let row = matvec(&rounded, K, &hx[..K]);
    let payload = if bank.len() >= x.len() * 4 {
        &bank[..]
    } else {
        bytemuck::cast_slice(&x)
    };
    let staging = guarded_bytes(payload);
    assert_eq!(staging.len(), *plan.inventory().iter().max().unwrap());
    let snapshots =
        [upper[0], upper[1].max(upper[2]), upper[3], upper[4]].map(|bytes| vec![0u8; bytes]);
    let output_readback = vec![0f32; plan.rows * plan.m];
    let backing_readback = vec![0u8; *upper.iter().max().unwrap()];
    let observed = [
        bank.capacity()
            + if plan.dtype == GgmlType::F32 {
                bank.len()
            } else {
                0
            },
        (decoded.capacity() + rounded.capacity()) * 4,
        (x.capacity() + hx.capacity()) * 4 + half_x.capacity() * 2 + ids.capacity() * 4,
        references.iter().map(|v| v.capacity() * 4).sum::<usize>() + active.capacity(),
        row.capacity() * 4,
        staging.capacity(),
        snapshots.iter().map(Vec::capacity).sum::<usize>(),
        output_readback.capacity() * 4 + backing_readback.capacity(),
    ];
    let budget = plan.cpu_budget(upper);
    let allowed = [
        budget.source_bank,
        budget.decoded_pair,
        budget.activations,
        budget.references,
        budget.matvec_row,
        budget.guarded_staging,
        budget.snapshots,
        budget.readbacks,
    ];
    for (component, (&actual, &limit)) in observed.iter().zip(&allowed).enumerate() {
        assert!(
            actual <= limit,
            "{plan:?} component={component} observed={actual} allowed={limit}"
        );
    }
    assert!(
        observed.iter().sum::<usize>() + budget.bookkeeping + upper.iter().sum::<usize>()
            <= FIXTURE_LIMIT
    );
}

#[test]
fn staging_cpu_budget_covers_every_scheduled_fixture() {
    let mut grouped = 0;
    let mut dense = 0;
    for dtype in WITNESS_DTYPES {
        for case in grouped_cases() {
            for b_div in [1, 2] {
                let plan = FixturePlan::grouped(dtype, case.m, b_div);
                for page in [1, 4096, 16384, 65536] {
                    let upper = modeled_upper(plan, page);
                    check_budget_components(plan, upper);
                    if page == 16384 {
                        check_observed_host_capacities(plan, upper);
                        eprintln!(
                            "staging budget case={case:?} plan={plan:?} upper={upper:?} cpu={:?} cpu_bytes={} metal_bytes={} complete_bytes={}",
                            plan.cpu_budget(upper),
                            plan.cpu_budget(upper).total(),
                            upper.iter().sum::<usize>(),
                            plan.cpu_budget(upper).total() + upper.iter().sum::<usize>()
                        );
                    }
                }
                grouped += 1;
            }
        }
        for (rows, m) in boundary_shapes() {
            let plan = FixturePlan::dense(dtype, m, rows);
            for page in [1, 4096, 16384, 65536] {
                let upper = modeled_upper(plan, page);
                check_budget_components(plan, upper);
                if page == 16384 {
                    check_observed_host_capacities(plan, upper);
                    eprintln!(
                        "staging budget plan={plan:?} upper={upper:?} cpu={:?} cpu_bytes={} metal_bytes={} complete_bytes={}",
                        plan.cpu_budget(upper),
                        plan.cpu_budget(upper).total(),
                        upper.iter().sum::<usize>(),
                        plan.cpu_budget(upper).total() + upper.iter().sum::<usize>()
                    );
                }
            }
            dense += 1;
        }
    }
    assert_eq!((grouped, dense), (462, 175));
    eprintln!(
        "staging CPU budget: 637 scheduled fixtures, 2548 complete price/component checks; no availability reserve"
    );
}

#[test]
fn staging_cpu_budget_red_control_rejects_old_iq1_dense_underprice() {
    let dtype = GgmlType::IQ1_S;
    let bank = synthetic_bank(dtype, K, 65, 1, 0xd00d ^ dtype as u64);
    let decoded = cpu_expert(dtype, &bank, K, 65, 0);
    let rounded = half_values(&decoded);
    let live_bytes = bank.capacity() + (decoded.capacity() + rounded.capacity()) * 4;
    let old_inventory = [
        6500 + 64,
        512 * 4 + 64,
        512 * 2 + 64,
        4 + 64,
        65 * 4 + 64,
        65 * 4 + 64,
    ];
    let old_cpu = 16 * old_inventory.iter().sum::<usize>();
    assert_eq!(old_cpu, 167_680);
    assert!(live_bytes >= 272_740);
    assert!(
        old_cpu < live_bytes,
        "old compressed-multiple budget must be rejected"
    );
    let plan = FixturePlan::dense(dtype, 65, 1);
    let corrected = plan.cpu_budget(modeled_upper(plan, 16384));
    assert_eq!(corrected.decoded_pair, 266_240);
    assert!(corrected.total() >= live_bytes);
    eprintln!(
        "staging CPU budget red: old_cpu={old_cpu}, observed_live_bytes={live_bytes}, corrected_cpu={} components={corrected:?}",
        corrected.total()
    );
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

    fn run(
        &self,
        ctx: &MetalContext,
        admission: &Admission,
        wide: bool,
        activation: TensorActivation,
        epi: usize,
    ) -> usize {
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
        assert!(name.len() <= 128);
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
            let weak = objc2::rc::autoreleasepool(|_| {
                let out = view(
                    ctx,
                    bytemuck::cast_slice(&initial),
                    GgmlType::F32,
                    vec![initial.len() as u64],
                );
                let weak = buffer_probe(&out, admission.upper[5]);
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
                drop(out);
                weak
            });
            assert_buffer_released(&weak);
        }
        for (input, bytes) in inputs.iter().zip(before) {
            assert_eq!(
                tensor_backing_bytes(input),
                bytes,
                "{name}: immutable input changed"
            );
        }
        2
    }
}

fn run_grouped(dtype: GgmlType) {
    let _lease = acquire_metal_benchmark_lease().expect("staging witness production lease");
    let ctx = require_tensor_context();
    let mut fixtures = 0;
    let mut output_releases = 0;
    for case in grouped_cases() {
        for b_div in [1, 2] {
            let admission = admit(&ctx, FixturePlan::grouped(dtype, case.m, b_div));
            let probes = objc2::rc::autoreleasepool(|_| {
                let f = GroupedFixture::new(
                    &ctx,
                    dtype,
                    case.m,
                    case.count,
                    b_div,
                    case.min,
                    case.max,
                    case.bad_ids,
                );
                let probes = [
                    buffer_probe(&f.weight, admission.upper[0]),
                    buffer_probe(&f.x32, admission.upper[1]),
                    buffer_probe(&f.x16, admission.upper[2]),
                    buffer_probe(&f.counts, admission.upper[3]),
                    buffer_probe(&f.ids, admission.upper[4]),
                ];
                for epi in 0..3 {
                    output_releases += f.run(&ctx, &admission, false, TensorActivation::F32, epi);
                    output_releases += f.run(&ctx, &admission, false, TensorActivation::F16, epi);
                    output_releases += f.run(&ctx, &admission, true, TensorActivation::F16, epi);
                }
                drop(f);
                probes
            });
            for weak in &probes {
                assert_buffer_released(weak);
            }
            drop(admission);
            fixtures += 1;
        }
    }
    assert_eq!(fixtures, 66);
    assert_eq!(output_releases, 1188);
    eprintln!(
        "staging grouped dtype={dtype:?}: {fixtures} fixtures, {output_releases} completed single-kernel command buffers/output weak releases, 330 fixture input weak releases, K={K}, <=64 MiB admitted/fixture"
    );
}

fn run_dense(dtype: GgmlType) {
    let _lease = acquire_metal_benchmark_lease().expect("staging dense witness production lease");
    let ctx = require_tensor_context();
    let mut dispatches = 0;
    for (rows, m) in boundary_shapes() {
        let admission = admit(&ctx, FixturePlan::dense(dtype, m, rows));
        let probes = objc2::rc::autoreleasepool(|_| {
            let bank = synthetic_bank(dtype, K, m, 1, 0xd00d ^ dtype as u64);
            let w = half_values(&cpu_expert(dtype, &bank, K, m, 0));
            let x: Vec<f32> = (0..rows * K)
                .map(|i| (((i * 19 + i / K * 11) % 127) as f32 - 63.0) / 139.0)
                .collect();
            let hx = half_values(&x);
            let mut expected = Vec::with_capacity(rows * m);
            for row in hx.chunks_exact(K) {
                expected.extend_from_slice(&matvec(&w, K, row));
            }
            let weight = view(&ctx, &bank, dtype, vec![K as u64, m as u64]);
            let half_x: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
            let x32 = view(
                &ctx,
                bytemuck::cast_slice(&x),
                GgmlType::F32,
                vec![K as u64, rows as u64],
            );
            let x16 = view(
                &ctx,
                bytemuck::cast_slice(&half_x),
                GgmlType::F16,
                vec![K as u64, rows as u64],
            );
            let probes = [
                buffer_probe(&weight, admission.upper[0]),
                buffer_probe(&x32, admission.upper[1]),
                buffer_probe(&x16, admission.upper[2]),
            ];
            for activation in [TensorActivation::F32, TensorActivation::F16] {
                let input = if activation == TensorActivation::F32 {
                    &x32
                } else {
                    &x16
                };
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
                    assert!(name.len() <= 128);
                    let before = [tensor_backing_bytes(&weight), tensor_backing_bytes(input)];
                    for _ in 0..2 {
                        let weak = objc2::rc::autoreleasepool(|_| {
                            let out = view(
                                &ctx,
                                bytemuck::cast_slice(&vec![CANARY; rows * m]),
                                GgmlType::F32,
                                vec![m as u64, rows as u64],
                            );
                            let weak = buffer_probe(&out, admission.upper[5]);
                            dispatch_census_begin();
                            one_shot(&ctx, |enc| {
                                if wide {
                                    encode_mat_mat_tensor_f32_wide(
                                        &ctx, enc, &weight, input, &out, K, m, rows,
                                    )
                                } else {
                                    encode_mat_mat_tensor_f32_narrow(
                                        &ctx, enc, &weight, input, &out, K, m, rows,
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
                            drop(out);
                            weak
                        });
                        assert_buffer_released(&weak);
                        dispatches += 1;
                    }
                    assert_eq!(tensor_backing_bytes(&weight), before[0]);
                    assert_eq!(tensor_backing_bytes(input), before[1]);
                }
            }
            drop((weight, x32, x16));
            probes
        });
        for weak in &probes {
            assert_buffer_released(weak);
        }
        drop(admission);
    }
    assert_eq!(dispatches, if dtype == GgmlType::Q8_0 { 200 } else { 100 });
    eprintln!(
        "staging dense dtype={dtype:?}: 25 shapes, {dispatches} completed single-kernel command buffers/output weak releases, 75 fixture input weak releases, K={K}, <=64 MiB admitted/fixture"
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

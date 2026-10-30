use super::*;
use crate::metal::test_support::*;

mod iq1;
mod staging_ownership;
mod wide_activation_type;

const MOE_METAL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/moe.metal"
));
const QUANT_TILES_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/quant_tiles.h"
));
const TENSOR_METAL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/tensor/moe_grouped_slots_tensor.metal"
));

/// Every `GgmlType` variant (the enum has no iterator).
const ALL_GGML_TYPES: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q8_1,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_K,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_XS,
    GgmlType::IQ3_XXS,
    GgmlType::IQ1_S,
    GgmlType::IQ4_NL,
    GgmlType::IQ3_S,
    GgmlType::IQ2_S,
    GgmlType::IQ4_XS,
    GgmlType::I8,
    GgmlType::I16,
    GgmlType::I32,
    GgmlType::I64,
    GgmlType::F64,
    GgmlType::IQ1_M,
    GgmlType::BF16,
    GgmlType::MXFP4,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
    GgmlType::Unknown,
];

/// Expert-bank dtypes the Metal loader can leave resident but that the
/// grouped MoE prefill path deliberately does not cover, with the reason.
/// Currently empty: every resident dtype has a generic instantiation.
const EXCLUDED: &[(GgmlType, &str)] = &[];

// ---------------------------------------------------------------------------
// Pure-CPU coverage tests (no Metal device).

/// Enumerate the dtypes a Qwen MoE expert bank can have once resident on
/// the GPU (`MetalModel::load` uses the ExpertBank role capability and
/// dequantizes every other dtype to F32) and require a grouped down and
/// gate/up instantiation for each, unless explicitly EXCLUDED.
#[test]
fn moe_grouped_generic_covers_every_resident_expert_dtype() {
    let mut resident: Vec<GgmlType> = ALL_GGML_TYPES
        .iter()
        .copied()
        .filter(|&d| {
            crate::metal_forward::weight_role_dtype_supported(
                crate::metal_forward::WeightRole::ExpertBank,
                d,
            )
        })
        .collect();
    // Non-native expert banks are converted to F32 at load.
    if !resident.contains(&GgmlType::F32) {
        resident.push(GgmlType::F32);
    }
    let mut missing = Vec::new();
    for &dtype in &resident {
        let excluded = EXCLUDED.iter().any(|(d, _)| *d == dtype);
        let supported = moe_grouped_generic_supported(dtype);
        assert!(
            !(excluded && supported),
            "{dtype:?} is EXCLUDED but has a generic instantiation; drop the stale exclusion"
        );
        if excluded {
            continue;
        }
        for role in MoeGroupedGenericRole::ALL {
            if moe_grouped_generic_pipeline_name(role, dtype).is_none() {
                missing.push(format!("{role:?}/{dtype:?}"));
            }
        }
        // The prefill dispatcher's eligibility must agree (same-dtype and
        // with the Q4_K_M-style combination around it).
        assert_eq!(
            crate::metal_dflash::prefill_moe_grouped_ineligibility(dtype, dtype, dtype, 2048, 512),
            None,
            "dispatcher rejects resident dtype {dtype:?}"
        );
        assert_eq!(
            crate::metal_dflash::prefill_moe_grouped_ineligibility(
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                dtype,
                2048,
                512
            ),
            None,
            "dispatcher rejects down dtype {dtype:?}"
        );
    }
    assert!(
        missing.is_empty(),
        "resident expert dtypes without a grouped MoE kernel (add a generic \
         instantiation or an EXCLUDED entry with a reason): {missing:?}"
    );
}

#[test]
fn moe_grouped_generic_layout_matches_ggml_storage() {
    for l in MOE_GROUPED_GENERIC_LAYOUTS {
        let (elems, bytes) = l.dtype.storage_layout().expect("storage layout");
        if elems == 1 {
            // Dense types are tiled as 32-element blocks.
            assert_eq!(l.block_elems, 32, "{:?}", l.dtype);
            assert_eq!(l.block_bytes as u64, 32 * bytes, "{:?}", l.dtype);
        } else {
            assert_eq!(l.block_elems as u64, elems, "{:?}", l.dtype);
            assert_eq!(l.block_bytes as u64, bytes, "{:?}", l.dtype);
        }
        assert_eq!(
            MOE_GROUPED_GENERIC_LAYOUTS
                .iter()
                .filter(|o| o.dtype == l.dtype || o.suffix == l.suffix)
                .count(),
            1,
            "duplicate mapping for {:?}",
            l.dtype
        );
    }
}

fn header_define(name: &str) -> usize {
    QUANT_TILES_H
        .lines()
        .find_map(|line| {
            let mut it = line.split_whitespace();
            (it.next() == Some("#define") && it.next() == Some(name))
                .then(|| it.next().and_then(|v| v.parse().ok()))
                .flatten()
        })
        .unwrap_or_else(|| panic!("quant_tiles.h has no numeric #define {name}"))
}

/// Every mapping entry must have a `host_name` instantiation in moe.metal
/// whose BYTES / NL / dequant function agree with the Rust layout, and every
/// generic `host_name` in moe.metal must be mapped.
#[test]
fn moe_grouped_generic_mapping_matches_metal_source() {
    for l in MOE_GROUPED_GENERIC_LAYOUTS {
        for role in MoeGroupedGenericRole::ALL {
            let name = moe_grouped_generic_pipeline_name(role, l.dtype).unwrap();
            let needle = format!("[[host_name(\"{name}\")]]");
            let lines: Vec<&str> = MOE_METAL.lines().filter(|s| s.contains(&needle)).collect();
            assert_eq!(
                lines.len(),
                1,
                "expected exactly one instantiation of {name}"
            );
            let line = lines[0];
            let args = line
                .rsplit_once('<')
                .and_then(|(_, rest)| rest.split_once('>'))
                .map(|(args, _)| args)
                .unwrap_or_else(|| panic!("cannot parse template args: {line}"));
            let args: Vec<&str> = args.split(',').map(str::trim).collect();
            let bytes_macro = args[0];
            let nl_macro = args[1];
            let deq = args[2];
            assert_eq!(
                header_define(bytes_macro),
                l.block_bytes,
                "{name}: {bytes_macro}"
            );
            assert_eq!(
                header_define(nl_macro),
                l.block_elems / 16,
                "{name}: {nl_macro}"
            );
            assert_eq!(deq, format!("qt_dequantize_{}", l.suffix), "{name}");
            assert!(
                QUANT_TILES_H.contains(&format!("inline void {deq}(")),
                "{deq} missing from quant_tiles.h"
            );
            let expected_epi = match role {
                MoeGroupedGenericRole::Down => Some("0"),
                MoeGroupedGenericRole::UpSiluMul => Some("1"),
                MoeGroupedGenericRole::Swiglu => None,
            };
            assert_eq!(args.get(3).copied(), expected_epi, "{name}: epilogue");
        }
    }
    for &dtype in MOE_GROUPED_CLAMPED_SWIGLU_DTYPES {
        let name = clamped_swiglu_pipeline_name(dtype).unwrap();
        let l = moe_grouped_generic_layout(dtype).unwrap();
        let needle = format!("[[host_name(\"{name}\")]]");
        let lines: Vec<&str> = MOE_METAL.lines().filter(|s| s.contains(&needle)).collect();
        assert_eq!(
            lines.len(),
            1,
            "expected exactly one instantiation of {name}"
        );
        assert!(
            lines[0].contains("kernel_moe_swiglu_clamped_grouped_slots_n16_generic<")
                && lines[0].contains(&format!("qt_dequantize_{}>", l.suffix)),
            "{name}: {}",
            lines[0]
        );
    }
    let instantiated = MOE_METAL
        .lines()
        .filter(|s| s.contains("[[host_name(\"") && s.contains("_grouped_slots_generic\")]]"))
        .count();
    assert_eq!(
        instantiated,
        MOE_GROUPED_GENERIC_LAYOUTS.len() * MoeGroupedGenericRole::ALL.len()
            + MOE_GROUPED_CLAMPED_SWIGLU_DTYPES.len(),
        "moe.metal has generic instantiations without a Rust mapping entry"
    );
}

/// The tensor tile instantiates the same dtype list through
/// `TT_INSTANTIATE(suffix, BYTES, NL, DEQ)`, with the same block geometry
/// and dequant function as the Rust layout, and shares the generic kernel's
/// argument struct field for field.
#[test]
fn moe_grouped_tensor_mapping_matches_metal_source() {
    let instantiations: Vec<Vec<&str>> = TENSOR_METAL
        .lines()
        .filter_map(|line| line.strip_prefix("TT_INSTANTIATE("))
        .map(|rest| {
            rest.trim_end_matches(')')
                .split(',')
                .map(str::trim)
                .collect()
        })
        .collect();
    assert_eq!(
        instantiations.len(),
        MOE_GROUPED_GENERIC_LAYOUTS.len(),
        "tensor instantiations {instantiations:?}"
    );
    for l in MOE_GROUPED_GENERIC_LAYOUTS {
        let args = instantiations
            .iter()
            .find(|a| a[0] == l.suffix)
            .unwrap_or_else(|| panic!("no TT_INSTANTIATE for {}", l.suffix));
        assert_eq!(header_define(args[1]), l.block_bytes, "{}: bytes", l.suffix);
        assert_eq!(
            header_define(args[2]),
            l.block_elems / 16,
            "{}: NL",
            l.suffix
        );
        assert_eq!(args[3], format!("qt_dequantize_{}", l.suffix));
        for activation in [TensorActivation::F32, TensorActivation::F16] {
            for role in MoeGroupedTensorRole::ALL {
                let name = moe_grouped_tensor_pipeline_name(role, l.dtype, activation).unwrap();
                let (prefix, rest) = name.split_once(l.suffix).unwrap();
                assert!(
                    TENSOR_METAL.contains(&format!("host_name(\"{prefix}\" #suffix \"{rest}\")")),
                    "{name}: no host_name pattern in the tensor instantiation macro"
                );
            }
            let dense = mat_mat_tensor_pipeline_name(l.dtype, activation).unwrap();
            let (prefix, rest) = dense.split_once(l.suffix).unwrap();
            assert!(
                TENSOR_METAL.contains(&format!("host_name(\"{prefix}\" #suffix \"{rest}\")")),
                "{dense}: no host_name pattern in the tensor instantiation macro"
            );
        }
    }
    let struct_fields = |source: &str| -> Vec<String> {
        let start = source
            .find("struct moe_group_generic_mm_args {")
            .expect("args struct");
        source[start..]
            .lines()
            .skip(1)
            .take_while(|l| !l.starts_with('}'))
            .map(|l| l.split("//").next().unwrap().trim().to_string())
            .collect()
    };
    assert_eq!(struct_fields(TENSOR_METAL), struct_fields(MOE_METAL));
}

// ---------------------------------------------------------------------------
// GPU equivalence tests on synthetic, valid quant blocks (no model fixtures).
// Filter: `generic_grouped_moe_`.

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    fn signed(&mut self, amp: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * amp
    }
}

/// Byte offsets of the f16 scale fields in one storage block. Everything
/// else is random bytes, which is a valid encoding for every listed format.
fn f16_scale_offsets(dtype: GgmlType) -> &'static [usize] {
    match dtype {
        GgmlType::Q2_0 | GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::IQ4_NL => &[0],
        GgmlType::Q4_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        GgmlType::IQ2_S
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_XXS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_XS => &[0],
        GgmlType::MXFP4 => &[],
        GgmlType::IQ1_S | GgmlType::IQ1_M => &[],
        other => panic!("no synthetic block recipe for {other:?}"),
    }
}

/// `n_expert` banks of `n_rows` rows x `n_in` columns, GGUF layout.
fn synthetic_bank(
    dtype: GgmlType,
    n_in: usize,
    n_rows: usize,
    n_expert: usize,
    seed: u64,
) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let n = n_in * n_rows * n_expert;
    match dtype {
        GgmlType::F32 => (0..n).flat_map(|_| rng.signed(0.5).to_le_bytes()).collect(),
        GgmlType::F16 => (0..n)
            .flat_map(|_| half::f16::from_f32(rng.signed(0.5)).to_bits().to_le_bytes())
            .collect(),
        GgmlType::BF16 => (0..n)
            .flat_map(|_| {
                half::bf16::from_f32(rng.signed(0.5))
                    .to_bits()
                    .to_le_bytes()
            })
            .collect(),
        _ => {
            let (elems, bytes) = dtype.storage_layout().unwrap();
            let (elems, bytes) = (elems as usize, bytes as usize);
            assert!(n_in.is_multiple_of(elems));
            let n_blocks = n / elems;
            let offsets = f16_scale_offsets(dtype);
            let mut out = vec![0u8; n_blocks * bytes];
            for block in out.chunks_exact_mut(bytes) {
                for b in block.iter_mut() {
                    *b = rng.next_u32() as u8;
                }
                if dtype == GgmlType::MXFP4 {
                    // E8M0 shared exponent: 2^(e-127) over a weight-like
                    // range (2^-10 .. 2^-4); codes stay random.
                    block[0] = 117 + (rng.next_u32() % 7) as u8;
                    continue;
                }
                if matches!(dtype, GgmlType::IQ1_S | GgmlType::IQ1_M) {
                    iq1_test_scale(block, dtype, rng.signed(1.0) / 256.0);
                    continue;
                }
                for (i, &off) in offsets.iter().enumerate() {
                    // d: signed ~[0.5, 1.5] / 256; dmin / m: positive ~ / 512.
                    let v = if i == 0 {
                        let mag = (0.5 + rng.unit()) / 256.0;
                        if rng.next_u32() & 1 == 0 { mag } else { -mag }
                    } else {
                        (0.5 + rng.unit()) / 512.0
                    };
                    block[off..off + 2]
                        .copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
                }
            }
            out
        }
    }
}

/// CPU dequant of expert `e` as row-major `[n_rows][n_in]` f32.
fn cpu_expert(dtype: GgmlType, bank: &[u8], n_in: usize, n_rows: usize, e: usize) -> Vec<f32> {
    let (elems, bytes) = dtype.storage_layout().unwrap();
    let expert_bytes = (n_in * n_rows) / elems as usize * bytes as usize;
    let desc = crate::tensor::TensorDesc {
        name: format!("generic_grouped_moe.{dtype:?}.expert{e}"),
        shape: vec![n_in as u64, n_rows as u64],
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: expert_bytes as u64,
    };
    crate::codec::dequant_to_f32(&desc, &bank[e * expert_bytes..(e + 1) * expert_bytes])
        .expect("cpu dequant")
}

fn matvec(w: &[f32], n_in: usize, x: &[f32]) -> Vec<f32> {
    w.chunks_exact(n_in)
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32
        })
        .collect()
}

/// Routing fixture with uneven expert loads: expert 0 gets > 32 slots (two
/// down N-tiles, three SwiGLU N-tiles), expert 3 a small partial tile,
/// expert 4 nothing.
struct Routing {
    n_tokens: usize,
    topk: usize,
    n_expert: usize,
    /// expert of each slot (slot = token * topk + k)
    slot_expert: Vec<usize>,
    counts: Vec<i32>,
    ids: Vec<i32>,
}

fn routing() -> Routing {
    let (n_tokens, topk, n_expert) = (45usize, 2usize, 5usize);
    let mut slot_expert = vec![0; n_tokens * topk];
    for t in 0..n_tokens {
        slot_expert[t * topk] = if t % 5 == 0 { 3 } else { 0 };
        slot_expert[t * topk + 1] = 1 + t % 2;
    }
    let mut counts = vec![0i32; n_expert];
    let mut ids = vec![-1i32; n_expert * n_tokens];
    for (slot, &e) in slot_expert.iter().enumerate() {
        ids[e * n_tokens + counts[e] as usize] = slot as i32;
        counts[e] += 1;
    }
    assert!(counts[0] > 32 && counts[3] < 16 && counts[4] == 0);
    Routing {
        n_tokens,
        topk,
        n_expert,
        slot_expert,
        counts,
        ids,
    }
}

fn i32_tensor(ctx: &MetalContext, v: &[i32]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(v),
        vec![v.len() as u64],
        GgmlType::F32,
    )
    .expect("i32 tensor")
}

/// I32-typed index tensor (the clamped generic recipe checks the dtype).
fn i32_typed_tensor(ctx: &MetalContext, v: &[i32]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(v),
        vec![v.len() as u64],
        GgmlType::I32,
    )
    .expect("i32 tensor")
}

/// F16 tensor holding `v` rounded to half (the tile's own activation
/// rounding, done once up front).
fn f16_tensor(ctx: &MetalContext, v: &[f32]) -> MetalTensor {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    MetalTensor::from_bytes(ctx, &bytes, vec![v.len() as u64], GgmlType::F16).expect("f16 tensor")
}

fn f32_tensor(ctx: &MetalContext, v: &[f32]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(v),
        vec![v.len() as u64],
        GgmlType::F32,
    )
    .expect("f32 tensor")
}

fn bank_tensor(
    ctx: &MetalContext,
    dtype: GgmlType,
    bank: &[u8],
    n_in: usize,
    n_rows: usize,
    n_expert: usize,
) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bank,
        vec![n_in as u64, n_rows as u64, n_expert as u64],
        dtype,
    )
    .expect("expert bank tensor")
}

fn assert_close(label: &str, gpu: &[f32], cpu: &[f32]) {
    assert_eq!(gpu.len(), cpu.len());
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "{label}: non-finite GPU output"
    );
    let max_abs =
        crate::compare::assert_max_abs_diff_f32(concat!(file!(), ":", line!()), gpu.iter(), cpu);
    let ref_max = cpu.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let dot: f64 = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let ng: f64 = gpu.iter().map(|v| f64::from(*v).powi(2)).sum();
    let nc: f64 = cpu.iter().map(|v| f64::from(*v).powi(2)).sum();
    let cos = dot / (ng.sqrt() * nc.sqrt()).max(1e-30);
    eprintln!("[{label}] cos={cos:.6} max|delta|={max_abs:.3e} ref_max={ref_max:.3e}");
    assert!(nc > 0.0, "{label}: degenerate all-zero reference");
    assert!(cos > 0.999, "{label}: cos={cos}");
    assert!(
        max_abs <= 1e-2 * ref_max + 1e-4,
        "{label}: max|delta|={max_abs} ref_max={ref_max}"
    );
}

const DOWN_N_IN: usize = 512;
const DOWN_N_OUT: usize = 320;
const GU_N_HIDDEN: usize = 512;
const GU_N_FFN: usize = 192;

fn cpu_down(dtype: GgmlType, bank: &[u8], r: &Routing, inner: &[f32]) -> Vec<f32> {
    let experts: Vec<Vec<f32>> = (0..r.n_expert)
        .map(|e| cpu_expert(dtype, bank, DOWN_N_IN, DOWN_N_OUT, e))
        .collect();
    let mut out = Vec::with_capacity(r.slot_expert.len() * DOWN_N_OUT);
    for (slot, &e) in r.slot_expert.iter().enumerate() {
        out.extend(matvec(
            &experts[e],
            DOWN_N_IN,
            &inner[slot * DOWN_N_IN..(slot + 1) * DOWN_N_IN],
        ));
    }
    out
}

fn cpu_swiglu(
    gate_dtype: GgmlType,
    gate_bank: &[u8],
    up_dtype: GgmlType,
    up_bank: &[u8],
    r: &Routing,
    x: &[f32],
) -> Vec<f32> {
    cpu_swiglu_with_clamp(gate_dtype, gate_bank, up_dtype, up_bank, r, x, None)
}

/// Per-slot gate and up projections (concatenated over slots) on the CPU
/// dequantized banks.
fn cpu_gate_up(
    gate_dtype: GgmlType,
    gate_bank: &[u8],
    up_dtype: GgmlType,
    up_bank: &[u8],
    r: &Routing,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let gates: Vec<Vec<f32>> = (0..r.n_expert)
        .map(|e| cpu_expert(gate_dtype, gate_bank, GU_N_HIDDEN, GU_N_FFN, e))
        .collect();
    let ups: Vec<Vec<f32>> = (0..r.n_expert)
        .map(|e| cpu_expert(up_dtype, up_bank, GU_N_HIDDEN, GU_N_FFN, e))
        .collect();
    let mut g_all = Vec::with_capacity(r.slot_expert.len() * GU_N_FFN);
    let mut u_all = Vec::with_capacity(r.slot_expert.len() * GU_N_FFN);
    for (slot, &e) in r.slot_expert.iter().enumerate() {
        let t = slot / r.topk;
        let xt = &x[t * GU_N_HIDDEN..(t + 1) * GU_N_HIDDEN];
        g_all.extend(matvec(&gates[e], GU_N_HIDDEN, xt));
        u_all.extend(matvec(&ups[e], GU_N_HIDDEN, xt));
    }
    (g_all, u_all)
}

/// `silu(g) * u`, or `silu(min(g, c)) * clamp(u, -c, c)` with a clamp.
fn swiglu_epilogue(g: &[f32], u: &[f32], clamp: Option<f32>) -> Vec<f32> {
    let silu = |g: f32| g / (1.0 + (-g).exp());
    g.iter()
        .zip(u)
        .map(|(&g, &u)| match clamp {
            None => silu(g) * u,
            Some(c) => silu(g.min(c)) * u.clamp(-c, c),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_swiglu_with_clamp(
    gate_dtype: GgmlType,
    gate_bank: &[u8],
    up_dtype: GgmlType,
    up_bank: &[u8],
    r: &Routing,
    x: &[f32],
    clamp: Option<f32>,
) -> Vec<f32> {
    let (g, u) = cpu_gate_up(gate_dtype, gate_bank, up_dtype, up_bank, r, x);
    swiglu_epilogue(&g, &u, clamp)
}

fn run_generic_down_case(dtype: GgmlType) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    admit_iq1_expert_fixture(&ctx);
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let bank = synthetic_bank(
        dtype,
        DOWN_N_IN,
        DOWN_N_OUT,
        r.n_expert,
        0x5eed ^ dtype as u64,
    );
    let mut rng = Lcg(0xd0 ^ dtype as u64);
    let inner: Vec<f32> = (0..n_slots * DOWN_N_IN).map(|_| rng.signed(0.5)).collect();
    let cpu = cpu_down(dtype, &bank, &r, &inner);

    let w = bank_tensor(&ctx, dtype, &bank, DOWN_N_IN, DOWN_N_OUT, r.n_expert);
    let inner_t = f32_tensor(&ctx, &inner);
    let counts_t = i32_tensor(&ctx, &r.counts);
    let ids_t = i32_tensor(&ctx, &r.ids);
    let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_moe_down_f32_grouped_slots_generic(
            &ctx, enc, &w, &inner_t, &counts_t, &ids_t, &out_t, DOWN_N_IN, DOWN_N_OUT, r.n_expert,
            r.n_tokens,
        )
    })
    .expect("generic grouped down");
    let gpu = read_back_f32(&out_t.buffer, n_slots * DOWN_N_OUT);
    assert_close(&format!("generic-down-{dtype:?}"), &gpu, &cpu);

    // Range split (the trace-bin dispatch shape) must reproduce the full run.
    let out_r = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        for (lo, hi) in [(0u32, 15u32), (16, 31), (32, i32::MAX as u32)] {
            encode_moe_down_f32_grouped_slots_generic_range(
                &ctx, enc, &w, &inner_t, &counts_t, &ids_t, &out_r, DOWN_N_IN, DOWN_N_OUT,
                r.n_expert, r.n_tokens, lo, hi,
            )?;
        }
        Ok(())
    })
    .expect("generic grouped down range");
    let ranged = read_back_f32(&out_r.buffer, n_slots * DOWN_N_OUT);
    assert_eq!(
        ranged, gpu,
        "{dtype:?}: range-split down differs from full dispatch"
    );
}

fn run_generic_gate_up_case(gate_dtype: GgmlType, up_dtype: GgmlType) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    admit_iq1_expert_fixture(&ctx);
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let gate_bank = synthetic_bank(
        gate_dtype,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
        0x9a7e ^ gate_dtype as u64,
    );
    let up_bank = synthetic_bank(
        up_dtype,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
        0x0b ^ ((up_dtype as u64) << 8),
    );
    let mut rng = Lcg(0x11 ^ gate_dtype as u64);
    let x: Vec<f32> = (0..r.n_tokens * GU_N_HIDDEN)
        .map(|_| rng.signed(0.5))
        .collect();
    let cpu = cpu_swiglu(gate_dtype, &gate_bank, up_dtype, &up_bank, &r, &x);

    let g = bank_tensor(
        &ctx,
        gate_dtype,
        &gate_bank,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
    );
    let u = bank_tensor(&ctx, up_dtype, &up_bank, GU_N_HIDDEN, GU_N_FFN, r.n_expert);
    let x_t = f32_tensor(&ctx, &x);
    let counts_t = i32_tensor(&ctx, &r.counts);
    let ids_t = i32_tensor(&ctx, &r.ids);
    let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * GU_N_FFN) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_moe_swiglu_f32_grouped_slots_generic(
            &ctx,
            enc,
            &g,
            &u,
            &x_t,
            &counts_t,
            &ids_t,
            &out_t,
            GU_N_HIDDEN,
            GU_N_FFN,
            r.n_expert,
            r.topk,
            r.n_tokens,
        )
    })
    .expect("generic grouped swiglu");
    let gpu = read_back_f32(&out_t.buffer, n_slots * GU_N_FFN);
    assert_close(
        &format!("generic-gate-up-{gate_dtype:?}-{up_dtype:?}"),
        &gpu,
        &cpu,
    );

    // The two-pass (mixed-dtype) path must also be right for same-dtype
    // banks: exercise it explicitly via the gate-store + up-silu-mul kernels.
    if gate_dtype == up_dtype {
        let out_2p = MetalTensor::zeros_f32(&ctx, vec![(n_slots * GU_N_FFN) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_generic_mm(
                &ctx,
                enc,
                MoeGroupedGenericRole::Down,
                "test_gate_pass",
                &g,
                &x_t,
                &counts_t,
                &ids_t,
                &out_2p,
                GU_N_HIDDEN,
                GU_N_FFN,
                r.n_expert,
                r.n_tokens,
                r.topk,
                0,
                i32::MAX as u32,
            )?;
            encode_generic_mm(
                &ctx,
                enc,
                MoeGroupedGenericRole::UpSiluMul,
                "test_up_pass",
                &u,
                &x_t,
                &counts_t,
                &ids_t,
                &out_2p,
                GU_N_HIDDEN,
                GU_N_FFN,
                r.n_expert,
                r.n_tokens,
                r.topk,
                0,
                i32::MAX as u32,
            )
        })
        .expect("generic two-pass gate/up");
        let two_pass = read_back_f32(&out_2p.buffer, n_slots * GU_N_FFN);
        assert_close(
            &format!("generic-gate-up-2pass-{gate_dtype:?}"),
            &two_pass,
            &cpu,
        );
    }
}

macro_rules! generic_grouped_moe_cases {
    ($($down:ident, $gate_up:ident => $dtype:expr;)*) => {
        $(
            #[test]
            fn $down() {
                run_generic_down_case($dtype);
            }

            #[test]
            fn $gate_up() {
                run_generic_gate_up_case($dtype, $dtype);
            }
        )*

        #[test]
        fn generic_grouped_moe_case_list_covers_mapping() {
            let listed = [$($dtype),*];
            for l in MOE_GROUPED_GENERIC_LAYOUTS {
                assert!(listed.contains(&l.dtype), "no GPU equivalence test for {:?}", l.dtype);
            }
        }
    };
}

generic_grouped_moe_cases! {
    generic_grouped_moe_down_q2_0_matches_cpu, generic_grouped_moe_gate_up_q2_0_matches_cpu => GgmlType::Q2_0;
    generic_grouped_moe_down_q2_k_matches_cpu, generic_grouped_moe_gate_up_q2_k_matches_cpu => GgmlType::Q2_K;
    generic_grouped_moe_down_q3_k_matches_cpu, generic_grouped_moe_gate_up_q3_k_matches_cpu => GgmlType::Q3_K;
    generic_grouped_moe_down_q4_k_matches_cpu, generic_grouped_moe_gate_up_q4_k_matches_cpu => GgmlType::Q4_K;
    generic_grouped_moe_down_q5_k_matches_cpu, generic_grouped_moe_gate_up_q5_k_matches_cpu => GgmlType::Q5_K;
    generic_grouped_moe_down_q6_k_matches_cpu, generic_grouped_moe_gate_up_q6_k_matches_cpu => GgmlType::Q6_K;
    generic_grouped_moe_down_q8_0_matches_cpu, generic_grouped_moe_gate_up_q8_0_matches_cpu => GgmlType::Q8_0;
    generic_grouped_moe_down_q4_0_matches_cpu, generic_grouped_moe_gate_up_q4_0_matches_cpu => GgmlType::Q4_0;
    generic_grouped_moe_down_q4_1_matches_cpu, generic_grouped_moe_gate_up_q4_1_matches_cpu => GgmlType::Q4_1;
    generic_grouped_moe_down_iq2_s_matches_cpu, generic_grouped_moe_gate_up_iq2_s_matches_cpu => GgmlType::IQ2_S;
    generic_grouped_moe_down_iq3_xxs_matches_cpu, generic_grouped_moe_gate_up_iq3_xxs_matches_cpu => GgmlType::IQ3_XXS;
    generic_grouped_moe_down_iq3_s_matches_cpu, generic_grouped_moe_gate_up_iq3_s_matches_cpu => GgmlType::IQ3_S;
    generic_grouped_moe_down_mxfp4_matches_cpu, generic_grouped_moe_gate_up_mxfp4_matches_cpu => GgmlType::MXFP4;
    generic_grouped_moe_down_iq2_xs_matches_cpu, generic_grouped_moe_gate_up_iq2_xs_matches_cpu => GgmlType::IQ2_XS;
    generic_grouped_moe_down_iq2_xxs_matches_cpu, generic_grouped_moe_gate_up_iq2_xxs_matches_cpu => GgmlType::IQ2_XXS;
    generic_grouped_moe_down_iq4_nl_matches_cpu, generic_grouped_moe_gate_up_iq4_nl_matches_cpu => GgmlType::IQ4_NL;
    generic_grouped_moe_down_iq4_xs_matches_cpu, generic_grouped_moe_gate_up_iq4_xs_matches_cpu => GgmlType::IQ4_XS;
    generic_grouped_moe_down_f32_matches_cpu, generic_grouped_moe_gate_up_f32_matches_cpu => GgmlType::F32;
    generic_grouped_moe_down_f16_matches_cpu, generic_grouped_moe_gate_up_f16_matches_cpu => GgmlType::F16;
    generic_grouped_moe_down_bf16_matches_cpu, generic_grouped_moe_gate_up_bf16_matches_cpu => GgmlType::BF16;
    generic_grouped_moe_down_iq1_s_matches_cpu, generic_grouped_moe_gate_up_iq1_s_matches_cpu => GgmlType::IQ1_S;
    generic_grouped_moe_down_iq1_m_matches_cpu, generic_grouped_moe_gate_up_iq1_m_matches_cpu => GgmlType::IQ1_M;
}

// ---------------------------------------------------------------------------
// Metal 4 tensor tile: same contracts, checked against the CPU reference and
// against the generic tile it replaces. Filter: `tensor_grouped_moe_`.
// Skipped (silently passing) where the tensor tier is absent.

/// Tensor vs generic: both stage identical half operands and sum in F32, so
/// only the reduction order differs.
fn assert_tile_agreement(label: &str, tensor: &[f32], generic: &[f32]) {
    assert_eq!(tensor.len(), generic.len());
    assert!(
        tensor.iter().chain(generic).all(|v| v.is_finite()),
        "{label}: nonfinite operand"
    );
    let ref_max = generic.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let max_abs = max_abs_diff_finite(label, tensor, generic);
    eprintln!("[{label}] tensor-vs-generic max|delta|={max_abs:.3e} ref_max={ref_max:.3e}");
    assert!(
        max_abs <= 2e-3 * ref_max + 1e-5,
        "{label}: tensor tile departs from the generic tile: max|delta|={max_abs} ref_max={ref_max}"
    );
}

fn run_tensor_down_case(dtype: GgmlType) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if matches!(dtype, GgmlType::IQ1_S | GgmlType::IQ1_M) {
        admit_iq1_expert_fixture(&ctx);
        assert!(
            moe_grouped_tensor_available(&ctx),
            "IQ1 test requires actual M5 execution: {:?}",
            ctx.tensor_ops_status()
        );
    }
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let bank = synthetic_bank(
        dtype,
        DOWN_N_IN,
        DOWN_N_OUT,
        r.n_expert,
        0x5eed ^ dtype as u64,
    );
    let mut rng = Lcg(0xd0 ^ dtype as u64);
    let inner: Vec<f32> = (0..n_slots * DOWN_N_IN).map(|_| rng.signed(0.5)).collect();
    let cpu = cpu_down(dtype, &bank, &r, &inner);

    let w = bank_tensor(&ctx, dtype, &bank, DOWN_N_IN, DOWN_N_OUT, r.n_expert);
    let inner_t = f32_tensor(&ctx, &inner);
    let counts_t = i32_typed_tensor(&ctx, &r.counts);
    let ids_t = i32_typed_tensor(&ctx, &r.ids);
    let out_tensor = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    let out_generic = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_moe_down_f32_grouped_slots_tensor(
            &ctx,
            enc,
            &w,
            &inner_t,
            &counts_t,
            &ids_t,
            &out_tensor,
            DOWN_N_IN,
            DOWN_N_OUT,
            r.n_expert,
            r.n_tokens,
        )?;
        encode_moe_down_f32_grouped_slots_generic(
            &ctx,
            enc,
            &w,
            &inner_t,
            &counts_t,
            &ids_t,
            &out_generic,
            DOWN_N_IN,
            DOWN_N_OUT,
            r.n_expert,
            r.n_tokens,
        )
    })
    .expect("tensor grouped down");
    let gpu = read_back_f32(&out_tensor.buffer, n_slots * DOWN_N_OUT);
    let generic = read_back_f32(&out_generic.buffer, n_slots * DOWN_N_OUT);
    assert_close(&format!("tensor-down-{dtype:?}"), &gpu, &cpu);
    assert_tile_agreement(&format!("tensor-down-{dtype:?}"), &gpu, &generic);
}

fn run_tensor_gate_up_case(dtype: GgmlType) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if matches!(dtype, GgmlType::IQ1_S | GgmlType::IQ1_M) {
        admit_iq1_expert_fixture(&ctx);
        assert!(
            moe_grouped_tensor_available(&ctx),
            "IQ1 test requires actual M5 execution: {:?}",
            ctx.tensor_ops_status()
        );
    }
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let gate_bank = synthetic_bank(
        dtype,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
        0x9a7e ^ dtype as u64,
    );
    let up_bank = synthetic_bank(
        dtype,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
        0x0b ^ ((dtype as u64) << 8),
    );
    let mut rng = Lcg(0x11 ^ dtype as u64);
    let x: Vec<f32> = (0..r.n_tokens * GU_N_HIDDEN)
        .map(|_| rng.signed(0.5))
        .collect();
    let g = bank_tensor(&ctx, dtype, &gate_bank, GU_N_HIDDEN, GU_N_FFN, r.n_expert);
    let u = bank_tensor(&ctx, dtype, &up_bank, GU_N_HIDDEN, GU_N_FFN, r.n_expert);
    let x_t = f32_tensor(&ctx, &x);
    // The generic clamped recipe checks the index dtypes, so type them.
    let counts_t = i32_typed_tensor(&ctx, &r.counts);
    let ids_t = i32_typed_tensor(&ctx, &r.ids);
    let (cpu_g, cpu_u) = cpu_gate_up(dtype, &gate_bank, dtype, &up_bank, &r, &x);
    // A clamp at a third of the projection range clips a good share of both
    // operands while keeping the reference comparable to the half-operand
    // rounding of the pre-clamp sums (a tiny clamp would not).
    let range = cpu_g
        .iter()
        .chain(&cpu_u)
        .map(|v| v.abs())
        .fold(0f32, f32::max);
    for clamp in [None, Some(0.3 * range)] {
        let cpu = swiglu_epilogue(&cpu_g, &cpu_u, clamp);
        let out_tensor = MetalTensor::zeros_f32(&ctx, vec![(n_slots * GU_N_FFN) as u64]).unwrap();
        let out_generic = MetalTensor::zeros_f32(&ctx, vec![(n_slots * GU_N_FFN) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_f32_grouped_slots_tensor(
                &ctx,
                enc,
                &g,
                &u,
                &x_t,
                &counts_t,
                &ids_t,
                &out_tensor,
                GU_N_HIDDEN,
                GU_N_FFN,
                r.n_expert,
                r.topk,
                r.n_tokens,
                clamp,
            )?;
            // The generic two-pass recipe with the matching epilogue.
            match clamp {
                None => {
                    encode_generic_mm(
                        &ctx,
                        enc,
                        MoeGroupedGenericRole::Down,
                        "test_gate_pass",
                        &g,
                        &x_t,
                        &counts_t,
                        &ids_t,
                        &out_generic,
                        GU_N_HIDDEN,
                        GU_N_FFN,
                        r.n_expert,
                        r.n_tokens,
                        r.topk,
                        0,
                        i32::MAX as u32,
                    )?;
                    encode_generic_mm(
                        &ctx,
                        enc,
                        MoeGroupedGenericRole::UpSiluMul,
                        "test_up_pass",
                        &u,
                        &x_t,
                        &counts_t,
                        &ids_t,
                        &out_generic,
                        GU_N_HIDDEN,
                        GU_N_FFN,
                        r.n_expert,
                        r.n_tokens,
                        r.topk,
                        0,
                        i32::MAX as u32,
                    )
                }
                Some(c) => encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                    &ctx,
                    enc,
                    &g,
                    &u,
                    &x_t,
                    &counts_t,
                    &ids_t,
                    &out_generic,
                    GU_N_HIDDEN,
                    GU_N_FFN,
                    r.n_expert,
                    r.topk,
                    r.n_tokens,
                    c,
                ),
            }
        })
        .expect("tensor grouped gate/up");
        let gpu = read_back_f32(&out_tensor.buffer, n_slots * GU_N_FFN);
        let generic = read_back_f32(&out_generic.buffer, n_slots * GU_N_FFN);
        let label = format!("tensor-gate-up-{dtype:?}-clamp{clamp:?}");
        assert_close(&label, &gpu, &cpu);
        assert_tile_agreement(&label, &gpu, &generic);
    }
}

macro_rules! tensor_grouped_moe_cases {
    ($($(#[$attr:meta])* $down:ident, $gate_up:ident => $dtype:expr;)*) => {
        $(
            $(#[$attr])*
            #[test]
            fn $down() {
                run_tensor_down_case($dtype);
            }

            $(#[$attr])*
            #[test]
            fn $gate_up() {
                run_tensor_gate_up_case($dtype);
            }
        )*

        #[test]
        fn tensor_grouped_moe_case_list_covers_mapping() {
            let listed = [$($dtype),*];
            for l in MOE_GROUPED_GENERIC_LAYOUTS {
                assert!(listed.contains(&l.dtype), "no tensor GPU equivalence test for {:?}", l.dtype);
            }
        }
    };
}

tensor_grouped_moe_cases! {
    tensor_grouped_moe_down_q2_0_matches, tensor_grouped_moe_gate_up_q2_0_matches => GgmlType::Q2_0;
    tensor_grouped_moe_down_q2_k_matches, tensor_grouped_moe_gate_up_q2_k_matches => GgmlType::Q2_K;
    tensor_grouped_moe_down_q3_k_matches, tensor_grouped_moe_gate_up_q3_k_matches => GgmlType::Q3_K;
    tensor_grouped_moe_down_q4_k_matches, tensor_grouped_moe_gate_up_q4_k_matches => GgmlType::Q4_K;
    tensor_grouped_moe_down_q5_k_matches, tensor_grouped_moe_gate_up_q5_k_matches => GgmlType::Q5_K;
    tensor_grouped_moe_down_q6_k_matches, tensor_grouped_moe_gate_up_q6_k_matches => GgmlType::Q6_K;
    tensor_grouped_moe_down_q8_0_matches, tensor_grouped_moe_gate_up_q8_0_matches => GgmlType::Q8_0;
    tensor_grouped_moe_down_q4_0_matches, tensor_grouped_moe_gate_up_q4_0_matches => GgmlType::Q4_0;
    tensor_grouped_moe_down_q4_1_matches, tensor_grouped_moe_gate_up_q4_1_matches => GgmlType::Q4_1;
    tensor_grouped_moe_down_iq2_s_matches, tensor_grouped_moe_gate_up_iq2_s_matches => GgmlType::IQ2_S;
    tensor_grouped_moe_down_iq3_xxs_matches, tensor_grouped_moe_gate_up_iq3_xxs_matches => GgmlType::IQ3_XXS;
    tensor_grouped_moe_down_iq3_s_matches, tensor_grouped_moe_gate_up_iq3_s_matches => GgmlType::IQ3_S;
    tensor_grouped_moe_down_mxfp4_matches, tensor_grouped_moe_gate_up_mxfp4_matches => GgmlType::MXFP4;
    tensor_grouped_moe_down_iq2_xs_matches, tensor_grouped_moe_gate_up_iq2_xs_matches => GgmlType::IQ2_XS;
    tensor_grouped_moe_down_iq2_xxs_matches, tensor_grouped_moe_gate_up_iq2_xxs_matches => GgmlType::IQ2_XXS;
    tensor_grouped_moe_down_iq4_nl_matches, tensor_grouped_moe_gate_up_iq4_nl_matches => GgmlType::IQ4_NL;
    tensor_grouped_moe_down_iq4_xs_matches, tensor_grouped_moe_gate_up_iq4_xs_matches => GgmlType::IQ4_XS;
    tensor_grouped_moe_down_f32_matches, tensor_grouped_moe_gate_up_f32_matches => GgmlType::F32;
    tensor_grouped_moe_down_f16_matches, tensor_grouped_moe_gate_up_f16_matches => GgmlType::F16;
    tensor_grouped_moe_down_bf16_matches, tensor_grouped_moe_gate_up_bf16_matches => GgmlType::BF16;
    #[ignore = "requires M5; run explicitly with --ignored"]
    tensor_grouped_moe_down_iq1_s_matches, tensor_grouped_moe_gate_up_iq1_s_matches => GgmlType::IQ1_S;
    #[ignore = "requires M5; run explicitly with --ignored"]
    tensor_grouped_moe_down_iq1_m_matches, tensor_grouped_moe_gate_up_iq1_m_matches => GgmlType::IQ1_M;
}

/// Dense tensor mat-mat against the CPU reference and the simdgroup
/// `mat_mat` dispatch at a ragged shape (rows not a multiple of 32, M not
/// a multiple of 64).
fn run_tensor_dense_case(dtype: GgmlType) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if matches!(dtype, GgmlType::IQ1_S | GgmlType::IQ1_M) {
        admit_iq1_expert_fixture(&ctx);
        assert!(
            moe_grouped_tensor_available(&ctx),
            "IQ1 test requires actual M5 execution: {:?}",
            ctx.tensor_ops_status()
        );
    }
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    const N_IN: usize = 512;
    const N_OUT: usize = 320;
    const ROWS: usize = 45;
    let bank = synthetic_bank(dtype, N_IN, N_OUT, 1, 0xde ^ dtype as u64);
    let w_cpu = cpu_expert(dtype, &bank, N_IN, N_OUT, 0);
    let mut rng = Lcg(0xa1 ^ dtype as u64);
    let x: Vec<f32> = (0..ROWS * N_IN).map(|_| rng.signed(0.5)).collect();
    let cpu: Vec<f32> = (0..ROWS)
        .flat_map(|r| matvec(&w_cpu, N_IN, &x[r * N_IN..(r + 1) * N_IN]))
        .collect();
    let w = MetalTensor::from_bytes(&ctx, &bank, vec![N_IN as u64, N_OUT as u64], dtype)
        .expect("weight tensor");
    let x_t = f32_tensor(&ctx, &x);
    let x_v = x_t.view_bytes(0, vec![N_IN as u64, ROWS as u64]);
    let y_tensor = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, ROWS as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_tensor_f32(&ctx, enc, &w, &x_v, &y_tensor, N_IN, N_OUT, ROWS)
    })
    .expect("tensor dense mat-mat");
    let gpu = read_back_f32(&y_tensor.buffer, ROWS * N_OUT);
    let label = format!("tensor-dense-{dtype:?}");
    assert_close(&label, &gpu, &cpu);
    // Half activations converted up front stage the same operand values,
    // so the result is bitwise identical.
    let xh = f16_tensor(&ctx, &x);
    let xh_v = xh.view_bytes(0, vec![N_IN as u64, ROWS as u64]);
    let y_half = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, ROWS as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_tensor_f32(&ctx, enc, &w, &xh_v, &y_half, N_IN, N_OUT, ROWS)
    })
    .expect("tensor dense mat-mat, half activations");
    assert_eq!(
        read_back_f32(&y_half.buffer, ROWS * N_OUT),
        gpu,
        "{label}: f16act differs from f32act"
    );
    if crate::metal_forward::mat_mat_dispatch_supported(dtype) {
        let y_mma = MetalTensor::zeros_f32(&ctx, vec![N_OUT as u64, ROWS as u64]).unwrap();
        one_shot(&ctx, |enc| {
            crate::metal_forward::encode_mat_mat_dispatch_with_policy(
                &ctx, enc, &w, &x_v, &y_mma, N_IN, N_OUT, ROWS, false,
            )
            .map_err(|e| MetalError::BadShape {
                kernel: "test_mat_mat",
                detail: e.to_string(),
            })
        })
        .expect("mma mat-mat");
        let mma = read_back_f32(&y_mma.buffer, ROWS * N_OUT);
        assert_tile_agreement(&label, &gpu, &mma);
    }
}

/// 64-slot dense tile against the CPU reference and the explicit 32-slot
/// tile at ragged shapes (partial first and second slot groups, output-row
/// tails).
fn run_tensor_dense_wide_case(n_out: usize, rows: usize, seed: u64) {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    const N_IN: usize = 512;
    let bank = synthetic_bank(GgmlType::Q8_0, N_IN, n_out, 1, seed);
    let w_cpu = cpu_expert(GgmlType::Q8_0, &bank, N_IN, n_out, 0);
    let mut rng = Lcg(seed ^ 0x5a);
    let x: Vec<f32> = (0..rows * N_IN).map(|_| rng.signed(0.5)).collect();
    let cpu: Vec<f32> = (0..rows)
        .flat_map(|r| matvec(&w_cpu, N_IN, &x[r * N_IN..(r + 1) * N_IN]))
        .collect();
    let w = MetalTensor::from_bytes(&ctx, &bank, vec![N_IN as u64, n_out as u64], GgmlType::Q8_0)
        .expect("weight tensor");
    let x_t = f32_tensor(&ctx, &x);
    let x_v = x_t.view_bytes(0, vec![N_IN as u64, rows as u64]);
    let narrow = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, rows as u64]).unwrap();
    let wide = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, rows as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_mat_mat_tensor_f32_narrow(&ctx, enc, &w, &x_v, &narrow, N_IN, n_out, rows)
    })
    .expect("narrow dense mat-mat");
    one_shot(&ctx, |enc| {
        encode_mat_mat_tensor_f32_wide(&ctx, enc, &w, &x_v, &wide, N_IN, n_out, rows)
    })
    .expect("wide dense mat-mat");
    let narrow = read_back_f32(&narrow.buffer, rows * n_out);
    let wide = read_back_f32(&wide.buffer, rows * n_out);
    let label = format!("tensor-dense-wide-{n_out}x{rows}");
    assert_close(&label, &wide, &cpu);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&wide),
        bits(&narrow),
        "wide and narrow dense tiles differ at {n_out}x{rows}"
    );
}

#[test]
fn tensor_dense_wide_q8_0_matches_narrow() {
    run_tensor_dense_wide_case(320, 130, 0x9d);
    run_tensor_dense_wide_case(321, 97, 0x9e);
}

/// Opt-in narrow/wide dense timing at short batch lengths (the selector
/// turns wide on at 64 rows). `QWEN_V41_TILE_TIMING=1`.
#[test]
fn tensor_dense_wide_batch_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if !moe_grouped_tensor_available(&ctx) {
        return;
    }
    let (n_in, n_out) = (8192usize, 5120usize);
    let bank = synthetic_bank(GgmlType::Q8_0, n_in, n_out, 1, 0x44);
    let w = MetalTensor::from_bytes(&ctx, &bank, vec![n_in as u64, n_out as u64], GgmlType::Q8_0)
        .expect("weight tensor");
    let mut rng = Lcg(0x51);
    let bytes = bank.len() as f64;
    for rows in [64usize, 96, 128, 192, 256] {
        let x: Vec<f32> = (0..rows * n_in).map(|_| rng.signed(0.5)).collect();
        let x_t = f32_tensor(&ctx, &x);
        let x_v = x_t.view_bytes(0, vec![n_in as u64, rows as u64]);
        let y_n = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, rows as u64]).unwrap();
        let y_w = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, rows as u64]).unwrap();
        let narrow = |enc: &KernelEncoder| {
            encode_mat_mat_tensor_f32_narrow(&ctx, enc, &w, &x_v, &y_n, n_in, n_out, rows).unwrap();
        };
        let wide = |enc: &KernelEncoder| {
            encode_mat_mat_tensor_f32_wide(&ctx, enc, &w, &x_v, &y_w, n_in, n_out, rows).unwrap();
        };
        heat(&ctx, 2, &[&narrow, &wide]);
        let flops = 2.0 * (rows * n_in * n_out) as f64;
        timed_tile(
            &ctx,
            &format!("output_b rows={rows} narrow"),
            4,
            flops,
            bytes,
            &narrow,
        );
        timed_tile(
            &ctx,
            &format!("output_b rows={rows} wide"),
            4,
            flops,
            bytes,
            &wide,
        );
    }
}

macro_rules! tensor_dense_cases {
    ($($(#[$attr:meta])* $name:ident => $dtype:expr;)*) => {
        $(
            $(#[$attr])*
            #[test]
            fn $name() {
                run_tensor_dense_case($dtype);
            }
        )*

        #[test]
        fn tensor_dense_case_list_covers_mapping() {
            let listed = [$($dtype),*];
            for l in MOE_GROUPED_GENERIC_LAYOUTS {
                assert!(listed.contains(&l.dtype), "no dense tensor test for {:?}", l.dtype);
            }
        }
    };
}

tensor_dense_cases! {
    tensor_dense_q2_0_matches => GgmlType::Q2_0;
    tensor_dense_q2_k_matches => GgmlType::Q2_K;
    tensor_dense_q3_k_matches => GgmlType::Q3_K;
    tensor_dense_q4_k_matches => GgmlType::Q4_K;
    tensor_dense_q5_k_matches => GgmlType::Q5_K;
    tensor_dense_q6_k_matches => GgmlType::Q6_K;
    tensor_dense_q8_0_matches => GgmlType::Q8_0;
    tensor_dense_q4_0_matches => GgmlType::Q4_0;
    tensor_dense_q4_1_matches => GgmlType::Q4_1;
    tensor_dense_iq2_s_matches => GgmlType::IQ2_S;
    tensor_dense_iq3_xxs_matches => GgmlType::IQ3_XXS;
    tensor_dense_iq3_s_matches => GgmlType::IQ3_S;
    tensor_dense_mxfp4_matches => GgmlType::MXFP4;
    tensor_dense_iq2_xs_matches => GgmlType::IQ2_XS;
    tensor_dense_iq2_xxs_matches => GgmlType::IQ2_XXS;
    tensor_dense_iq4_nl_matches => GgmlType::IQ4_NL;
    tensor_dense_iq4_xs_matches => GgmlType::IQ4_XS;
    tensor_dense_f32_matches => GgmlType::F32;
    tensor_dense_f16_matches => GgmlType::F16;
    tensor_dense_bf16_matches => GgmlType::BF16;
    #[ignore = "requires M5; run explicitly with --ignored"]
    tensor_dense_iq1_s_matches => GgmlType::IQ1_S;
    #[ignore = "requires M5; run explicitly with --ignored"]
    tensor_dense_iq1_m_matches => GgmlType::IQ1_M;
}

/// Opt-in A/B of the dense tensor tile against the simdgroup `mat_mat`
/// dispatch at the V4.1 dense projection shapes (Q8_0, 2048 rows),
/// interleaved on a heated GPU. `QWEN_V41_TILE_TIMING=1`.
#[test]
fn tensor_dense_production_shape_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    const ROWS: usize = 2048;
    let mut rng = Lcg(0x5e1f);
    for (label, n_in, n_out) in [
        ("q_b 1280->32768", 1280usize, 32768usize),
        ("output_a 4096->8192", 4096, 8192),
        ("output_b 8192->5120", 8192, 5120),
        ("shexp gate 5120->2304", 5120, 2304),
        ("shexp down 2304->5120", 2304, 5120),
        ("q_a 5120->1280", 5120, 1280),
        ("kv 5120->512", 5120, 512),
    ] {
        let dtype = GgmlType::Q8_0;
        let bank = synthetic_bank(dtype, n_in, n_out, 1, 0x77 ^ n_out as u64);
        let w = MetalTensor::from_bytes(&ctx, &bank, vec![n_in as u64, n_out as u64], dtype)
            .expect("weight tensor");
        let x: Vec<f32> = (0..ROWS * n_in).map(|_| rng.signed(0.5)).collect();
        let x_t = f32_tensor(&ctx, &x);
        let x_v = x_t.view_bytes(0, vec![n_in as u64, ROWS as u64]);
        let xh_t = f16_tensor(&ctx, &x);
        let xh_v = xh_t.view_bytes(0, vec![n_in as u64, ROWS as u64]);
        let y = MetalTensor::zeros_f32(&ctx, vec![n_out as u64, ROWS as u64]).unwrap();
        let mma = |enc: &KernelEncoder| {
            crate::metal_forward::encode_mat_mat_dispatch_with_policy(
                &ctx, enc, &w, &x_v, &y, n_in, n_out, ROWS, false,
            )
            .unwrap();
        };
        let tensor = |enc: &KernelEncoder| {
            encode_mat_mat_tensor_f32(&ctx, enc, &w, &x_v, &y, n_in, n_out, ROWS).unwrap();
        };
        let tensor_f16 = |enc: &KernelEncoder| {
            encode_mat_mat_tensor_f32(&ctx, enc, &w, &xh_v, &y, n_in, n_out, ROWS).unwrap();
        };
        heat(&ctx, 4, &[&mma, &tensor, &tensor_f16]);
        let flops = 2.0 * (ROWS * n_in * n_out) as f64;
        let bytes = bank.len() as f64;
        for round in 0..2 {
            timed_tile(
                &ctx,
                &format!("{label} mma r{round}"),
                6,
                flops,
                bytes,
                &mma,
            );
            timed_tile(
                &ctx,
                &format!("{label} tensor r{round}"),
                6,
                flops,
                bytes,
                &tensor,
            );
            timed_tile(
                &ctx,
                &format!("{label} tensor f16act r{round}"),
                6,
                flops,
                bytes,
                &tensor_f16,
            );
        }
    }
}

/// GPU Q5_K -> F16 dequant must write the natural weight order the dense
/// F16 tensor tile reads. Compare its half bits against the CPU reference.
#[test]
fn v41_q5_k_f16_dequant_matches_cpu() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    const N_IN: usize = 5120;
    const N_OUT: usize = 2304;
    let dtype = GgmlType::Q5_K;
    let bank = synthetic_bank(dtype, N_IN, N_OUT, 1, 0x5a);
    let src = MetalTensor::from_bytes(&ctx, &bank, vec![N_IN as u64, N_OUT as u64], dtype)
        .expect("source tensor");
    let dst_bytes = vec![0u8; N_IN * N_OUT * 2];
    let dst = MetalTensor::from_bytes(
        &ctx,
        &dst_bytes,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::F16,
    )
    .expect("dest tensor");
    let command = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&command);
    let pso = ctx
        .tensor_pipeline("v41_dequant_q5_k_f16")
        .expect("dequant pipeline");
    enc.set_pipeline(&pso);
    enc.set_tensor(0, &src);
    enc.set_tensor(1, &dst);
    enc.set_bytes(2, &[N_IN as u32, N_OUT as u32]);
    enc.dispatch(
        MTLSize {
            width: N_IN / 16,
            height: N_OUT,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    commit_and_wait(&command).unwrap();
    let gpu = read_back_u16(&dst);
    let desc = crate::tensor::TensorDesc {
        name: "q5k_gpu_dequant".into(),
        shape: vec![N_IN as u64, N_OUT as u64],
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bank.len() as u64,
    };
    let cpu = crate::codec::dequant_to_f32(&desc, &bank).expect("cpu dequant");
    let cpu: Vec<u16> = cpu
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let mismatches = gpu
        .iter()
        .zip(&cpu)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .count();
    eprintln!(
        "q5k_dequant gpu_vs_cpu mismatches={mismatches} of {}",
        cpu.len()
    );
    assert_eq!(
        mismatches, 0,
        "Q5_K GPU dequant differs from the CPU reference"
    );
}

#[test]
fn generic_grouped_moe_gate_up_mixed_q4_k_q5_k_matches_cpu() {
    run_generic_gate_up_case(GgmlType::Q4_K, GgmlType::Q5_K);
}

#[test]
fn generic_grouped_moe_gate_up_mixed_iq4_xs_q8_0_matches_cpu() {
    run_generic_gate_up_case(GgmlType::IQ4_XS, GgmlType::Q8_0);
}

#[test]
fn generic_grouped_moe_gate_up_mixed_rejects_concurrent_encoder() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    admit_iq1_expert_fixture(&ctx);
    let r = routing();
    let g_bank = synthetic_bank(GgmlType::Q4_K, GU_N_HIDDEN, GU_N_FFN, r.n_expert, 1);
    let u_bank = synthetic_bank(GgmlType::Q8_0, GU_N_HIDDEN, GU_N_FFN, r.n_expert, 2);
    let g = bank_tensor(
        &ctx,
        GgmlType::Q4_K,
        &g_bank,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
    );
    let u = bank_tensor(
        &ctx,
        GgmlType::Q8_0,
        &u_bank,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
    );
    let x_t = MetalTensor::zeros_f32(&ctx, vec![(r.n_tokens * GU_N_HIDDEN) as u64]).unwrap();
    let counts_t = i32_tensor(&ctx, &r.counts);
    let ids_t = i32_tensor(&ctx, &r.ids);
    let out_t =
        MetalTensor::zeros_f32(&ctx, vec![(r.n_tokens * r.topk * GU_N_FFN) as u64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    let result = encode_moe_swiglu_f32_grouped_slots_generic(
        &ctx,
        &enc,
        &g,
        &u,
        &x_t,
        &counts_t,
        &ids_t,
        &out_t,
        GU_N_HIDDEN,
        GU_N_FFN,
        r.n_expert,
        r.topk,
        r.n_tokens,
    );
    enc.end();
    assert!(
        result.is_err(),
        "mixed-dtype gate/up must refuse a concurrent encoder"
    );
}

/// Generic Q5_K down vs the hand-tuned Q5_K grouped down (same tile math).
#[test]
fn generic_grouped_moe_down_q5_k_matches_specialized() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let dtype = GgmlType::Q5_K;
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let bank = synthetic_bank(dtype, DOWN_N_IN, DOWN_N_OUT, r.n_expert, 0x55);
    let mut rng = Lcg(0x56);
    let inner: Vec<f32> = (0..n_slots * DOWN_N_IN).map(|_| rng.signed(0.5)).collect();
    let w = bank_tensor(&ctx, dtype, &bank, DOWN_N_IN, DOWN_N_OUT, r.n_expert);
    let inner_t = f32_tensor(&ctx, &inner);
    let counts_t = i32_tensor(&ctx, &r.counts);
    let ids_t = i32_tensor(&ctx, &r.ids);
    let generic_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    let special_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    one_shot(&ctx, |enc| {
        encode_moe_down_f32_grouped_slots_generic(
            &ctx, enc, &w, &inner_t, &counts_t, &ids_t, &generic_t, DOWN_N_IN, DOWN_N_OUT,
            r.n_expert, r.n_tokens,
        )?;
        encode_moe_down_q5_K_f32_grouped_slots(
            &ctx, enc, &w, &inner_t, &counts_t, &ids_t, &special_t, DOWN_N_IN, DOWN_N_OUT,
            r.n_expert, r.n_tokens,
        )
    })
    .expect("q5 generic vs specialized down");
    let generic = read_back_f32(&generic_t.buffer, n_slots * DOWN_N_OUT);
    let special = read_back_f32(&special_t.buffer, n_slots * DOWN_N_OUT);
    let exact = generic
        .iter()
        .zip(&special)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    eprintln!(
        "[generic-vs-specialized-q5-down] bit-identical {exact}/{}",
        generic.len()
    );
    assert_close("generic-vs-specialized-q5-down", &generic, &special);
    let max_abs = crate::compare::assert_max_abs_diff_f32(
        concat!(file!(), ":", line!()),
        generic.iter(),
        &special,
    );
    let ref_max = special.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(
        max_abs <= 1e-5 * ref_max,
        "generic vs specialized Q5_K down drift {max_abs}"
    );
}

/// Generic Q4_K / Q5_K SwiGLU vs the hand-tuned grouped n16 kernels.
#[test]
fn generic_grouped_moe_gate_up_q4_k_q5_k_match_specialized() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    admit_iq1_expert_fixture(&ctx);
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let mut rng = Lcg(0x77);
    let x: Vec<f32> = (0..r.n_tokens * GU_N_HIDDEN)
        .map(|_| rng.signed(0.5))
        .collect();
    let x_t = f32_tensor(&ctx, &x);
    let counts_t = i32_tensor(&ctx, &r.counts);
    let ids_t = i32_tensor(&ctx, &r.ids);
    for dtype in [GgmlType::Q4_K, GgmlType::Q5_K] {
        let gb = synthetic_bank(
            dtype,
            GU_N_HIDDEN,
            GU_N_FFN,
            r.n_expert,
            0x100 ^ dtype as u64,
        );
        let ub = synthetic_bank(
            dtype,
            GU_N_HIDDEN,
            GU_N_FFN,
            r.n_expert,
            0x200 ^ dtype as u64,
        );
        let g = bank_tensor(&ctx, dtype, &gb, GU_N_HIDDEN, GU_N_FFN, r.n_expert);
        let u = bank_tensor(&ctx, dtype, &ub, GU_N_HIDDEN, GU_N_FFN, r.n_expert);
        let generic_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * GU_N_FFN) as u64]).unwrap();
        let special_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * GU_N_FFN) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_moe_swiglu_f32_grouped_slots_generic(
                &ctx,
                enc,
                &g,
                &u,
                &x_t,
                &counts_t,
                &ids_t,
                &generic_t,
                GU_N_HIDDEN,
                GU_N_FFN,
                r.n_expert,
                r.topk,
                r.n_tokens,
            )?;
            if dtype == GgmlType::Q4_K {
                encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &g,
                    &u,
                    &x_t,
                    &counts_t,
                    &ids_t,
                    &special_t,
                    GU_N_HIDDEN,
                    GU_N_FFN,
                    r.n_expert,
                    r.topk,
                    r.n_tokens,
                )
            } else {
                encode_moe_swiglu_q5_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &g,
                    &u,
                    &x_t,
                    &counts_t,
                    &ids_t,
                    &special_t,
                    GU_N_HIDDEN,
                    GU_N_FFN,
                    r.n_expert,
                    r.topk,
                    r.n_tokens,
                )
            }
        })
        .expect("generic vs specialized swiglu");
        let generic = read_back_f32(&generic_t.buffer, n_slots * GU_N_FFN);
        let special = read_back_f32(&special_t.buffer, n_slots * GU_N_FFN);
        let exact = generic
            .iter()
            .zip(&special)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        eprintln!(
            "[generic-vs-specialized-{dtype:?}-swiglu] bit-identical {exact}/{}",
            generic.len()
        );
        assert_close(
            &format!("generic-vs-specialized-{dtype:?}-swiglu"),
            &generic,
            &special,
        );
        let max_abs = crate::compare::assert_max_abs_diff_f32(
            concat!(file!(), ":", line!()),
            generic.iter(),
            &special,
        );
        let ref_max = special.iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5 * ref_max,
            "generic vs specialized {dtype:?} swiglu drift {max_abs}"
        );
    }
}

#[test]
fn generic_grouped_moe_rejects_bad_contracts() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let Some(ctx) = metal_test_context() else {
        return;
    };
    admit_iq1_expert_fixture(&ctx);
    let r = routing();
    let n_slots = r.n_tokens * r.topk;
    let bank = synthetic_bank(GgmlType::Q4_K, DOWN_N_IN, DOWN_N_OUT, r.n_expert, 3);
    let w = bank_tensor(
        &ctx,
        GgmlType::Q4_K,
        &bank,
        DOWN_N_IN,
        DOWN_N_OUT,
        r.n_expert,
    );
    let inner_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_IN) as u64]).unwrap();
    let counts_t = i32_tensor(&ctx, &r.counts);
    let ids_t = i32_tensor(&ctx, &r.ids);
    let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_slots * DOWN_N_OUT) as u64]).unwrap();
    let unsupported = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let unsupported = MetalTensor {
        dtype: GgmlType::I32,
        ..unsupported
    };
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd);
    // n_in not a whole number of Q4_K super-blocks.
    assert!(
        encode_moe_down_f32_grouped_slots_generic(
            &ctx,
            &enc,
            &w,
            &inner_t,
            &counts_t,
            &ids_t,
            &out_t,
            DOWN_N_IN - 32,
            DOWN_N_OUT,
            r.n_expert,
            r.n_tokens,
        )
        .is_err()
    );
    // Bank too small for the claimed expert count.
    assert!(
        encode_moe_down_f32_grouped_slots_generic(
            &ctx,
            &enc,
            &w,
            &inner_t,
            &counts_t,
            &ids_t,
            &out_t,
            DOWN_N_IN,
            DOWN_N_OUT,
            r.n_expert + 1,
            r.n_tokens,
        )
        .is_err()
    );
    // No instantiation for this dtype.
    assert!(
        encode_moe_down_f32_grouped_slots_generic(
            &ctx,
            &enc,
            &unsupported,
            &inner_t,
            &counts_t,
            &ids_t,
            &out_t,
            DOWN_N_IN,
            DOWN_N_OUT,
            r.n_expert,
            r.n_tokens,
        )
        .is_err()
    );
    enc.end();
}

/// DeepSeek V4.1 prefill shape (384 experts, 2048 tokens x top-6, ~32 slots
/// per expert; gate/up K=5120 M=2304, down K=2304 M=5120) with round-robin
/// routing (token t, k -> expert (t*7+k*53)%E), shared by the tile timing
/// probes.
struct ProductionShape {
    counts_t: MetalTensor,
    ids_t: MetalTensor,
    x_t: MetalTensor,
    /// `x_t` rounded to half once (the `_f16act` operand).
    x_h: MetalTensor,
    inner_t: MetalTensor,
    inner_h: MetalTensor,
    out_t: MetalTensor,
}

impl ProductionShape {
    const EXPERTS: usize = 384;
    const TOKENS: usize = 2048;
    const TOPK: usize = 6;
    const HIDDEN: usize = 5120;
    const FFN: usize = 2304;
    const SLOTS: usize = Self::TOKENS * Self::TOPK;

    fn new(ctx: &MetalContext) -> Self {
        let mut counts = vec![0i32; Self::EXPERTS];
        let mut ids = vec![-1i32; Self::EXPERTS * Self::TOKENS];
        for t in 0..Self::TOKENS {
            for k in 0..Self::TOPK {
                let e = (t * 7 + k * 53) % Self::EXPERTS;
                ids[e * Self::TOKENS + counts[e] as usize] = (t * Self::TOPK + k) as i32;
                counts[e] += 1;
            }
        }
        let mut rng = Lcg(0x7ab5);
        let x: Vec<f32> = (0..Self::TOKENS * Self::HIDDEN)
            .map(|_| rng.signed(0.5))
            .collect();
        let inner_h = MetalTensor::from_bytes(
            ctx,
            &vec![0u8; Self::SLOTS * Self::FFN * 2],
            vec![(Self::SLOTS * Self::FFN) as u64],
            GgmlType::F16,
        )
        .unwrap();
        Self {
            counts_t: i32_typed_tensor(ctx, &counts),
            ids_t: i32_typed_tensor(ctx, &ids),
            x_h: f16_tensor(ctx, &x),
            x_t: f32_tensor(ctx, &x),
            inner_h,
            inner_t: MetalTensor::zeros_f32(ctx, vec![(Self::SLOTS * Self::FFN) as u64]).unwrap(),
            out_t: MetalTensor::zeros_f32(ctx, vec![(Self::SLOTS * Self::HIDDEN) as u64]).unwrap(),
        }
    }

    fn banks(&self, ctx: &MetalContext, dtype: GgmlType) -> ProductionBanks {
        let gate = synthetic_bank(
            dtype,
            Self::HIDDEN,
            Self::FFN,
            Self::EXPERTS,
            0x11 ^ dtype as u64,
        );
        let up = synthetic_bank(
            dtype,
            Self::HIDDEN,
            Self::FFN,
            Self::EXPERTS,
            0x22 ^ dtype as u64,
        );
        let down = synthetic_bank(
            dtype,
            Self::FFN,
            Self::HIDDEN,
            Self::EXPERTS,
            0x33 ^ dtype as u64,
        );
        ProductionBanks {
            gate_bytes: 2.0 * gate.len() as f64,
            down_bytes: down.len() as f64,
            gate_t: bank_tensor(ctx, dtype, &gate, Self::HIDDEN, Self::FFN, Self::EXPERTS),
            up_t: bank_tensor(ctx, dtype, &up, Self::HIDDEN, Self::FFN, Self::EXPERTS),
            down_t: bank_tensor(ctx, dtype, &down, Self::FFN, Self::HIDDEN, Self::EXPERTS),
        }
    }

    /// FLOPs of one gate+up (two GEMMs) and of one down.
    fn flops() -> (f64, f64) {
        let mac = (Self::SLOTS * Self::HIDDEN * Self::FFN) as f64;
        (2.0 * 2.0 * mac, 2.0 * mac)
    }
}

struct ProductionBanks {
    gate_bytes: f64,
    down_bytes: f64,
    gate_t: MetalTensor,
    up_t: MetalTensor,
    down_t: MetalTensor,
}

/// Median GPU time of `samples` one-encoder command buffers, printed as
/// TFLOPS and effective weight bandwidth.
fn timed_tile(
    ctx: &MetalContext,
    label: &str,
    samples: usize,
    flops: f64,
    bytes: f64,
    encode: &dyn Fn(&KernelEncoder),
) -> f64 {
    let mut us = Vec::new();
    for _ in 0..samples {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode(&enc);
        enc.end();
        commit_and_wait(&command).unwrap();
        us.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6);
    }
    us.sort_by(f64::total_cmp);
    let med = us[samples / 2];
    eprintln!(
        "tile_timing {label:<40} median {med:9.1} us  {:6.1} TFLOPS  weights {:6.0} GB/s",
        flops / (med * 1e-6) / 1e12,
        bytes / (med * 1e-6) / 1e9
    );
    med
}

fn heat(ctx: &MetalContext, rounds: usize, encoders: &[&dyn Fn(&KernelEncoder)]) {
    for _ in 0..rounds {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        for encode in encoders {
            encode(&enc);
        }
        enc.end();
        commit_and_wait(&command).unwrap();
    }
}

/// Opt-in timing of the generic half-staged expert tiles at the production
/// shape. Prints TFLOPS and effective weight bandwidth; asserts nothing.
/// `QWEN_V41_TILE_TIMING=1`.
#[test]
fn generic_tile_production_shape_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    let shape = ProductionShape::new(&ctx);
    let (gate_up_flops, down_flops) = ProductionShape::flops();
    for dtype in [
        GgmlType::Q3_K,
        GgmlType::IQ3_XXS,
        GgmlType::IQ2_S,
        GgmlType::MXFP4,
    ] {
        let b = shape.banks(&ctx, dtype);
        let gate_up = |enc: &KernelEncoder| {
            encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                &ctx,
                enc,
                &b.gate_t,
                &b.up_t,
                &shape.x_t,
                &shape.counts_t,
                &shape.ids_t,
                &shape.inner_t,
                ProductionShape::HIDDEN,
                ProductionShape::FFN,
                ProductionShape::EXPERTS,
                ProductionShape::TOPK,
                ProductionShape::TOKENS,
                10.0,
            )
            .unwrap();
        };
        let down_enc = |enc: &KernelEncoder| {
            encode_moe_down_f32_grouped_slots_generic(
                &ctx,
                enc,
                &b.down_t,
                &shape.inner_t,
                &shape.counts_t,
                &shape.ids_t,
                &shape.out_t,
                ProductionShape::FFN,
                ProductionShape::HIDDEN,
                ProductionShape::EXPERTS,
                ProductionShape::TOKENS,
            )
            .unwrap();
        };
        heat(&ctx, 6, &[&gate_up, &down_enc]);
        timed_tile(
            &ctx,
            &format!("{dtype:?} gate+up (2 GEMMs)"),
            8,
            gate_up_flops,
            b.gate_bytes,
            &gate_up,
        );
        timed_tile(
            &ctx,
            &format!("{dtype:?} down"),
            8,
            down_flops,
            b.down_bytes,
            &down_enc,
        );
    }
}

/// Opt-in A/B of the Metal 4 tensor tile against the generic tile at the
/// production shape, interleaved per role (generic, tensor, generic,
/// tensor) on a heated GPU. `QWEN_V41_TILE_TIMING=1`; skipped where the
/// tensor tier is absent.
#[test]
fn tensor_tile_production_shape_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    let shape = ProductionShape::new(&ctx);
    let (gate_up_flops, down_flops) = ProductionShape::flops();
    for dtype in [
        GgmlType::Q3_K,
        GgmlType::IQ3_XXS,
        GgmlType::IQ2_XS,
        GgmlType::MXFP4,
        GgmlType::Q8_0,
    ] {
        let b = shape.banks(&ctx, dtype);
        let gate_up_generic = |enc: &KernelEncoder| {
            encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                &ctx,
                enc,
                &b.gate_t,
                &b.up_t,
                &shape.x_t,
                &shape.counts_t,
                &shape.ids_t,
                &shape.inner_t,
                ProductionShape::HIDDEN,
                ProductionShape::FFN,
                ProductionShape::EXPERTS,
                ProductionShape::TOPK,
                ProductionShape::TOKENS,
                10.0,
            )
            .unwrap();
        };
        let gate_up_tensor = |enc: &KernelEncoder| {
            encode_moe_swiglu_f32_grouped_slots_tensor(
                &ctx,
                enc,
                &b.gate_t,
                &b.up_t,
                &shape.x_t,
                &shape.counts_t,
                &shape.ids_t,
                &shape.inner_t,
                ProductionShape::HIDDEN,
                ProductionShape::FFN,
                ProductionShape::EXPERTS,
                ProductionShape::TOPK,
                ProductionShape::TOKENS,
                Some(10.0),
            )
            .unwrap();
        };
        let down_generic = |enc: &KernelEncoder| {
            encode_moe_down_f32_grouped_slots_generic(
                &ctx,
                enc,
                &b.down_t,
                &shape.inner_t,
                &shape.counts_t,
                &shape.ids_t,
                &shape.out_t,
                ProductionShape::FFN,
                ProductionShape::HIDDEN,
                ProductionShape::EXPERTS,
                ProductionShape::TOKENS,
            )
            .unwrap();
        };
        let down_tensor = |enc: &KernelEncoder| {
            encode_moe_down_f32_grouped_slots_tensor(
                &ctx,
                enc,
                &b.down_t,
                &shape.inner_t,
                &shape.counts_t,
                &shape.ids_t,
                &shape.out_t,
                ProductionShape::FFN,
                ProductionShape::HIDDEN,
                ProductionShape::EXPERTS,
                ProductionShape::TOKENS,
            )
            .unwrap();
        };
        let gate_up_tensor_f16 = |enc: &KernelEncoder| {
            encode_moe_swiglu_f32_grouped_slots_tensor(
                &ctx,
                enc,
                &b.gate_t,
                &b.up_t,
                &shape.x_h,
                &shape.counts_t,
                &shape.ids_t,
                &shape.inner_t,
                ProductionShape::HIDDEN,
                ProductionShape::FFN,
                ProductionShape::EXPERTS,
                ProductionShape::TOPK,
                ProductionShape::TOKENS,
                Some(10.0),
            )
            .unwrap();
        };
        let down_tensor_f16 = |enc: &KernelEncoder| {
            encode_moe_down_f32_grouped_slots_tensor(
                &ctx,
                enc,
                &b.down_t,
                &shape.inner_h,
                &shape.counts_t,
                &shape.ids_t,
                &shape.out_t,
                ProductionShape::FFN,
                ProductionShape::HIDDEN,
                ProductionShape::EXPERTS,
                ProductionShape::TOKENS,
            )
            .unwrap();
        };
        heat(
            &ctx,
            4,
            &[
                &gate_up_generic,
                &gate_up_tensor,
                &gate_up_tensor_f16,
                &down_generic,
                &down_tensor,
                &down_tensor_f16,
            ],
        );
        for round in 0..2 {
            let tag = |role: &str, path: &str| format!("{dtype:?} {role} {path} r{round}");
            timed_tile(
                &ctx,
                &tag("gate+up", "generic"),
                6,
                gate_up_flops,
                b.gate_bytes,
                &gate_up_generic,
            );
            timed_tile(
                &ctx,
                &tag("gate+up", "tensor"),
                6,
                gate_up_flops,
                b.gate_bytes,
                &gate_up_tensor,
            );
            timed_tile(
                &ctx,
                &tag("gate+up", "tensor f16act"),
                6,
                gate_up_flops,
                b.gate_bytes,
                &gate_up_tensor_f16,
            );
            timed_tile(
                &ctx,
                &tag("down", "generic"),
                6,
                down_flops,
                b.down_bytes,
                &down_generic,
            );
            timed_tile(
                &ctx,
                &tag("down", "tensor"),
                6,
                down_flops,
                b.down_bytes,
                &down_tensor,
            );
            timed_tile(
                &ctx,
                &tag("down", "tensor f16act"),
                6,
                down_flops,
                b.down_bytes,
                &down_tensor_f16,
            );
        }
    }
}

/// Nonzero equivalence for the 64-slot down probe at a small production-like
/// shape: same IQ3_XXS bank and F16 activations, narrow 32-slot panels versus
/// wide 64-slot panels.
#[test]
fn expert_wide_panel_matches_narrow() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    const TOKENS: usize = 4096;
    const EXPERTS: usize = 384;
    const TOPK: usize = 6;
    const N_IN: usize = 512;
    const N_OUT: usize = 256;
    let slots = TOKENS * TOPK;
    let bank = synthetic_bank(GgmlType::IQ3_XXS, N_IN, N_OUT, EXPERTS, 0x5eed);
    let weight = bank_tensor(&ctx, GgmlType::IQ3_XXS, &bank, N_IN, N_OUT, EXPERTS);
    let mut counts = vec![0i32; EXPERTS];
    let mut ids = vec![-1i32; EXPERTS * TOKENS];
    for token in 0..TOKENS {
        for k in 0..TOPK {
            let expert = (token * 7 + k * 53) % EXPERTS;
            ids[expert * TOKENS + counts[expert] as usize] = (token * TOPK + k) as i32;
            counts[expert] += 1;
        }
    }
    let counts_t = i32_typed_tensor(&ctx, &counts);
    let ids_t = i32_typed_tensor(&ctx, &ids);
    let mut rng = Lcg(0x51ee);
    let x: Vec<f32> = (0..slots * N_IN).map(|_| rng.signed(0.5)).collect();
    let x_h = f16_tensor(&ctx, &x);
    let narrow_out = MetalTensor::zeros_f32(&ctx, vec![(slots * N_OUT) as u64]).unwrap();
    let wide_out = MetalTensor::zeros_f32(&ctx, vec![(slots * N_OUT) as u64]).unwrap();
    let narrow = |enc: &KernelEncoder| {
        encode_moe_down_f32_grouped_slots_tensor(
            &ctx,
            enc,
            &weight,
            &x_h,
            &counts_t,
            &ids_t,
            &narrow_out,
            N_IN,
            N_OUT,
            EXPERTS,
            TOKENS,
        )
        .unwrap();
    };
    let wide = |enc: &KernelEncoder| {
        encode_moe_down_wide_probe(
            &ctx, enc, &weight, &x_h, &counts_t, &ids_t, &wide_out, N_IN, N_OUT, EXPERTS, TOKENS,
        )
        .unwrap();
    };
    for encode in [&narrow as &dyn Fn(&KernelEncoder), &wide] {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode(&enc);
        enc.end();
        commit_and_wait(&command).unwrap();
    }
    assert_close(
        "wide64 down",
        &tensor_f32_at_offset(&wide_out),
        &tensor_f32_at_offset(&narrow_out),
    );

    let mut rng_gate = Lcg(0x6eed);
    let xg: Vec<f32> = (0..TOKENS * N_IN).map(|_| rng_gate.signed(0.5)).collect();
    let xg_h = f16_tensor(&ctx, &xg);
    let inner_narrow = MetalTensor::zeros_f32(&ctx, vec![(slots * N_OUT) as u64]).unwrap();
    let inner_wide = MetalTensor::zeros_f32(&ctx, vec![(slots * N_OUT) as u64]).unwrap();
    let narrow_swiglu = |enc: &KernelEncoder| {
        encode_moe_swiglu_f32_grouped_slots_tensor(
            &ctx,
            enc,
            &weight,
            &weight,
            &xg_h,
            &counts_t,
            &ids_t,
            &inner_narrow,
            N_IN,
            N_OUT,
            EXPERTS,
            TOPK,
            TOKENS,
            Some(10.0),
        )
        .unwrap();
    };
    let wide_swiglu = |enc: &KernelEncoder| {
        encode_moe_swiglu_wide_probe(
            &ctx,
            enc,
            &weight,
            &weight,
            &xg_h,
            &counts_t,
            &ids_t,
            &inner_wide,
            N_IN,
            N_OUT,
            EXPERTS,
            TOPK,
            TOKENS,
            10.0,
        )
        .unwrap();
    };
    // This test compares the explicit narrow and wide dispatches; keep the
    // production 4096 selector out of the narrow arm.
    unsafe {
        std::env::set_var("QWEN_V41_EXPERT_WIDE", "0");
    }
    for encode in [&narrow_swiglu as &dyn Fn(&KernelEncoder), &wide_swiglu] {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        encode(&enc);
        enc.end();
        commit_and_wait(&command).unwrap();
    }
    unsafe {
        std::env::remove_var("QWEN_V41_EXPERT_WIDE");
    }
    assert_close(
        "wide64 gate/up",
        &tensor_f32_at_offset(&inner_wide),
        &tensor_f32_at_offset(&inner_narrow),
    );
}

fn wide_compat_routing(original_rows: &[usize]) -> (Vec<i32>, Vec<i32>) {
    let n = original_rows.len();
    let mut counts = vec![0i32; 10];
    let mut ids = vec![-1; 10 * n];
    for (row, &original) in original_rows.iter().enumerate() {
        assert!(original < 4096);
        let mut end = 3807;
        let mut second = 1;
        for (i, tail) in [1, 31, 32, 33, 63, 64, 65].into_iter().enumerate() {
            if (end..end + tail).contains(&original) {
                second = i + 2;
            }
            end += tail;
        }
        assert_eq!(end, 4096);
        for (k, expert) in [0, second].into_iter().enumerate() {
            ids[expert * n + counts[expert] as usize] = (row * 2 + k) as i32;
            counts[expert] += 1;
        }
    }
    (counts, ids)
}

fn wide_compat_metal_bound() -> usize {
    // Three full outputs (including the F32 control) dominate the two full
    // plus two compact outputs. All weights are conservatively priced as F32.
    3 * (4096 * 2 * 288 + 128) * 4
        + (4096 + 2048) * 512 * 4
        + 2 * 10 * 512 * 288 * 4
        + 2 * (10 + 10 * 4096) * 4
        + 64 * 65536
}

#[test]
fn wide_compat_fixture_covers_slot_tails_and_bounded_compaction() {
    assert!(wide_compat_metal_bound() < 64 * 1024 * 1024);
    let (counts, ids) = wide_compat_routing(&(0..4096).collect::<Vec<_>>());
    assert_eq!(counts, [4096, 3807, 1, 31, 32, 33, 63, 64, 65, 0]);
    let mut seen = vec![false; 8192];
    for (expert, &count) in counts.iter().enumerate() {
        for &id in &ids[expert * 4096..expert * 4096 + count as usize] {
            assert!(!std::mem::replace(&mut seen[id as usize], true));
        }
        assert!(
            ids[expert * 4096 + count as usize..(expert + 1) * 4096]
                .iter()
                .all(|&id| id == -1)
        );
    }
    assert!(seen.into_iter().all(|v| v));
    for n in [1, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048] {
        let rows: Vec<_> = (0..n).map(|i| (4050 + i * 4051) % 4096).collect();
        let (counts, _) = wide_compat_routing(&rows);
        assert_eq!(counts.iter().sum::<i32>(), (n * 2) as i32);
        assert_eq!(counts[0], n as i32);
        assert_eq!(counts[9], 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn wide_compat_narrow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    x: &MetalTensor,
    counts: &MetalTensor,
    ids: &MetalTensor,
    out: &MetalTensor,
    n: usize,
) {
    for (weight, role, clamp) in [
        (gate, MoeGroupedTensorRole::Down, None),
        (up, MoeGroupedTensorRole::UpClamped, Some(10.0)),
    ] {
        let args = generic_mm_args(
            "wide_compat",
            weight,
            x,
            counts,
            ids,
            out,
            512,
            288,
            10,
            n,
            2,
            0,
            i32::MAX as u32,
        )
        .unwrap();
        let activation = if x.dtype == GgmlType::F16 {
            TensorActivation::F16
        } else {
            TensorActivation::F32
        };
        let name = moe_grouped_tensor_pipeline_name(role, weight.dtype, activation).unwrap();
        tensor::dispatch_tensor_mm(
            ctx, enc, &name, &args, weight, x, counts, ids, out, 288, 10, n, clamp,
        )
        .unwrap();
    }
}

#[test]
#[ignore = "M5 tensor tier required; self-leased, strict bits, <=64 MiB planned Metal buffers"]
fn wide_narrow_compacted_maya_gate_up_bitwise() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("wide compatibility lease");
    assert!(wide_compat_metal_bound() < 64 * 1024 * 1024);
    let ctx = MetalContext::new().unwrap();
    assert!(
        moe_grouped_tensor_selected(&ctx),
        "M5 tensor tier must be active, no fallback proof"
    );
    assert!(std::env::var_os("QWEN_V41_EXPERT_WIDE").is_none_or(|v| v != "0"));
    let types = [
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
    ];
    let guarded = |n: usize| {
        let root = f32_tensor(&ctx, &vec![4096.25; n * 2 * 288 + 128]);
        let view = root.view_bytes(64 * 4, vec![(n * 2 * 288) as u64]);
        (root, view)
    };
    let checked = |root: &MetalTensor| {
        let values = tensor_f32_at_offset(root);
        assert!(
            values[..64]
                .iter()
                .chain(&values[values.len() - 64..])
                .all(|&v| v == 4096.25)
        );
        let values = values[64..values.len() - 64].to_vec();
        assert!(values.iter().all(|v| v.is_finite() && *v != 4096.25));
        assert!(values.iter().any(|&v| v != 0.0));
        values
    };
    let submit = |encode: &dyn Fn(&KernelEncoder), wide: bool| {
        crate::metal::dispatch_census_begin();
        objc2::rc::autoreleasepool(|_| {
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            encode(&enc);
            enc.end();
            commit_and_wait(&command).unwrap();
        });
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 2);
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel.starts_with("kernel_moe_wide_probe_"))
                .count(),
            if wide { 2 } else { 0 }
        );
    };
    for (case, gate_type) in types.into_iter().enumerate() {
        for up_type in [gate_type, types[(case + 1) % types.len()]] {
            let gate = bank_tensor(
                &ctx,
                gate_type,
                &synthetic_bank(gate_type, 512, 288, 10, 0x1234),
                512,
                288,
                10,
            );
            let up = bank_tensor(
                &ctx,
                up_type,
                &synthetic_bank(up_type, 512, 288, 10, 0x9876),
                512,
                288,
                10,
            );
            let mut rng = Lcg(0x57ee);
            let x: Vec<_> = (0..4096 * 512).map(|_| rng.signed(0.5)).collect();
            let half = f16_tensor(&ctx, &x);
            let (counts, ids) = wide_compat_routing(&(0..4096).collect::<Vec<_>>());
            let counts = i32_typed_tensor(&ctx, &counts);
            let ids = i32_typed_tensor(&ctx, &ids);
            let (narrow_root, narrow) = guarded(4096);
            let (wide_root, wide) = guarded(4096);
            submit(
                &|enc| {
                    wide_compat_narrow(&ctx, enc, &gate, &up, &half, &counts, &ids, &narrow, 4096)
                },
                false,
            );
            submit(
                &|enc| {
                    encode_moe_swiglu_f32_grouped_slots_tensor(
                        &ctx,
                        enc,
                        &gate,
                        &up,
                        &half,
                        &counts,
                        &ids,
                        &wide,
                        512,
                        288,
                        10,
                        2,
                        4096,
                        Some(10.0),
                    )
                    .unwrap()
                },
                true,
            );
            let reference = checked(&wide_root);
            assert_eq!(
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                checked(&narrow_root)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{gate_type:?}/{up_type:?} full panels"
            );
            for n in [1, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048] {
                let rows: Vec<_> = (0..n).map(|i| (4050 + i * 4051) % 4096).collect();
                let compact: Vec<_> = rows
                    .iter()
                    .flat_map(|&r| x[r * 512..(r + 1) * 512].iter().copied())
                    .collect();
                let compact = f16_tensor(&ctx, &compact);
                let (cc, ci) = wide_compat_routing(&rows);
                let cc = i32_typed_tensor(&ctx, &cc);
                let ci = i32_typed_tensor(&ctx, &ci);
                for force_wide in [false, true] {
                    let (root, out) = guarded(n);
                    submit(
                        &|enc| {
                            if force_wide {
                                encode_moe_swiglu_wide_probe(
                                    &ctx, enc, &gate, &up, &compact, &cc, &ci, &out, 512, 288, 10,
                                    2, n, 10.0,
                                )
                                .unwrap();
                            } else {
                                encode_moe_swiglu_f32_grouped_slots_tensor(
                                    &ctx,
                                    enc,
                                    &gate,
                                    &up,
                                    &compact,
                                    &cc,
                                    &ci,
                                    &out,
                                    512,
                                    288,
                                    10,
                                    2,
                                    n,
                                    Some(10.0),
                                )
                                .unwrap();
                            }
                        },
                        force_wide,
                    );
                    let actual = checked(&root);
                    for (i, &original) in rows.iter().enumerate() {
                        for j in 0..2 * 288 {
                            assert_eq!(
                                actual[i * 2 * 288 + j].to_bits(),
                                reference[original * 2 * 288 + j].to_bits(),
                                "{gate_type:?}/{up_type:?} compact n={n} row={original} j={j} wide={force_wide}"
                            );
                        }
                    }
                }
            }
            let f32_input = f32_tensor(&ctx, &x);
            let (f32_root, f32_out) = guarded(4096);
            submit(
                &|enc| {
                    encode_moe_swiglu_f32_grouped_slots_tensor(
                        &ctx,
                        enc,
                        &gate,
                        &up,
                        &f32_input,
                        &counts,
                        &ids,
                        &f32_out,
                        512,
                        288,
                        10,
                        2,
                        4096,
                        Some(10.0),
                    )
                    .unwrap()
                },
                false,
            );
            assert_eq!(
                checked(&f32_root)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "F32 narrow control"
            );
            eprintln!("strict wide/compact case {gate_type:?}/{up_type:?} passed");
        }
    }
}

/// Opt-in 4096-token expert down timing: narrow 32-slot panels versus one
/// 64-slot probe panel. Synthetic IQ3_XXS bank, same useful slots; the 4096
/// count distribution makes the wide panel mostly full. `QWEN_V41_TILE_TIMING=1`.
#[test]
fn expert_wide_panel_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    let shape = ProductionShape::new(&ctx);
    let b = shape.banks(&ctx, GgmlType::IQ3_XXS);
    const TOKENS: usize = 4096;
    const EXPERTS: usize = ProductionShape::EXPERTS;
    const TOPK: usize = ProductionShape::TOPK;
    let slots = TOKENS * TOPK;
    let mut counts = vec![0i32; EXPERTS];
    let mut ids = vec![-1i32; EXPERTS * TOKENS];
    for token in 0..TOKENS {
        for k in 0..TOPK {
            let expert = (token * 7 + k * 53) % EXPERTS;
            ids[expert * TOKENS + counts[expert] as usize] = (token * TOPK + k) as i32;
            counts[expert] += 1;
        }
    }
    let counts_t = i32_typed_tensor(&ctx, &counts);
    let ids_t = i32_typed_tensor(&ctx, &ids);
    let inner_h =
        MetalTensor::zeros_f16(&ctx, vec![(slots * ProductionShape::FFN) as u64]).unwrap();
    let out_t =
        MetalTensor::zeros_f32(&ctx, vec![(slots * ProductionShape::HIDDEN) as u64]).unwrap();
    let narrow = |enc: &KernelEncoder| {
        encode_moe_down_f32_grouped_slots_tensor(
            &ctx,
            enc,
            &b.down_t,
            &inner_h,
            &counts_t,
            &ids_t,
            &out_t,
            ProductionShape::FFN,
            ProductionShape::HIDDEN,
            EXPERTS,
            TOKENS,
        )
        .unwrap();
    };
    let wide = |enc: &KernelEncoder| {
        encode_moe_down_wide_probe(
            &ctx,
            enc,
            &b.down_t,
            &inner_h,
            &counts_t,
            &ids_t,
            &out_t,
            ProductionShape::FFN,
            ProductionShape::HIDDEN,
            EXPERTS,
            TOKENS,
        )
        .unwrap();
    };
    let x_h =
        MetalTensor::zeros_f16(&ctx, vec![(TOKENS * ProductionShape::HIDDEN) as u64]).unwrap();
    let inner_t =
        MetalTensor::zeros_f32(&ctx, vec![(slots * ProductionShape::FFN) as u64]).unwrap();
    let narrow_gate_up = |enc: &KernelEncoder| {
        encode_moe_swiglu_f32_grouped_slots_tensor(
            &ctx,
            enc,
            &b.gate_t,
            &b.up_t,
            &x_h,
            &counts_t,
            &ids_t,
            &inner_t,
            ProductionShape::HIDDEN,
            ProductionShape::FFN,
            EXPERTS,
            TOPK,
            TOKENS,
            Some(10.0),
        )
        .unwrap();
    };
    let wide_gate_up = |enc: &KernelEncoder| {
        encode_moe_swiglu_wide_probe(
            &ctx,
            enc,
            &b.gate_t,
            &b.up_t,
            &x_h,
            &counts_t,
            &ids_t,
            &inner_t,
            ProductionShape::HIDDEN,
            ProductionShape::FFN,
            EXPERTS,
            TOPK,
            TOKENS,
            10.0,
        )
        .unwrap();
    };
    // The narrow arm must bypass the production 4096 selector; the wide arm
    // calls the explicit probe pipeline directly.
    unsafe {
        std::env::set_var("QWEN_V41_EXPERT_WIDE", "0");
    }
    heat(&ctx, 3, &[&narrow, &wide, &narrow_gate_up, &wide_gate_up]);
    let down_flops = 2.0 * (slots * ProductionShape::HIDDEN * ProductionShape::FFN) as f64;
    let gate_up_flops = 2.0 * down_flops;
    timed_tile(
        &ctx,
        "down 4096 narrow32",
        8,
        down_flops,
        b.down_bytes,
        &narrow,
    );
    timed_tile(&ctx, "down 4096 wide64", 8, down_flops, b.down_bytes, &wide);
    timed_tile(
        &ctx,
        "gate+up 4096 narrow32",
        8,
        gate_up_flops,
        b.gate_bytes,
        &narrow_gate_up,
    );
    timed_tile(
        &ctx,
        "gate+up 4096 wide64",
        8,
        gate_up_flops,
        b.gate_bytes,
        &wide_gate_up,
    );
    unsafe {
        std::env::remove_var("QWEN_V41_EXPERT_WIDE");
    }
}

/// Raw Metal 4 matmul2d ceiling with threadgroup-resident half operands at
/// the tile shapes the tensor expert tile could use (research kernel, no
/// device traffic in the loop). `QWEN_V41_TILE_TIMING=1`; prints TFLOPS.
#[test]
fn matmul2d_peak_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    if !moe_grouped_tensor_available(&ctx) {
        eprintln!("tensor tier unavailable: {:?}", ctx.tensor_ops_status());
        return;
    }
    let out = MetalTensor::zeros_f32(&ctx, vec![4096]).unwrap();
    let shapes: [(usize, usize, usize, usize); 8] = [
        (32, 64, 32, 4),
        (64, 32, 32, 4),
        (64, 64, 32, 4),
        (32, 64, 64, 4),
        (64, 64, 64, 4),
        (128, 64, 32, 4),
        (32, 32, 32, 1),
        (16, 64, 32, 4),
    ];
    for pass in 0..2 {
        for (m, n, k, sg) in shapes {
            let name = format!("kernel_tensor_peak_m{m}_n{n}_k{k}_sg{sg}");
            let pso = ctx.tensor_pipeline(&name).unwrap();
            for tgs in [320usize, 2560] {
                let iters: u32 = 1000;
                let mut us = Vec::new();
                for _ in 0..6 {
                    let command = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&command);
                    enc.set_pipeline(&pso);
                    enc.set_bytes(0, &iters);
                    enc.set_tensor(1, &out);
                    enc.dispatch(
                        MTLSize {
                            width: tgs,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32 * sg,
                            height: 1,
                            depth: 1,
                        },
                    );
                    enc.end();
                    commit_and_wait(&command).unwrap();
                    us.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6);
                }
                us.sort_by(f64::total_cmp);
                let med = us[3];
                let flops = tgs as f64 * iters as f64 * 2.0 * (m * n * k) as f64;
                eprintln!(
                    "matmul2d_peak pass{pass} {name:<36} tgs={tgs:<5} median {med:9.1} us  {:6.1} TFLOPS",
                    flops / (med * 1e-6) / 1e12
                );
            }
        }
    }
}

/// Raw simdgroup-MMA ceiling of the device (research kernel, no memory
/// traffic in the loop). `QWEN_V41_TILE_TIMING=1`; prints TFLOPS.
#[test]
fn simdgroup_mma_peak_timing() {
    if std::env::var_os("QWEN_V41_TILE_TIMING").is_none() {
        return;
    }
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = MetalContext::new().expect("Metal");
    let out = MetalTensor::zeros_f32(&ctx, vec![4096]).unwrap();
    for (name, acc) in [
        ("kernel_mma_peak_acc4", 4u32),
        ("kernel_mma_peak_acc8", 8),
        ("kernel_mma_peak_acc16", 16),
    ] {
        for tgs in [320usize, 1280, 2560] {
            let iters: u32 = 2000;
            let mut us = Vec::new();
            for _ in 0..6 {
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                let pso = ctx.pipeline(name).unwrap();
                enc.set_pipeline(&pso);
                enc.set_bytes(0, &iters);
                enc.set_tensor(1, &out);
                enc.dispatch(
                    MTLSize {
                        width: tgs,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.end();
                commit_and_wait(&command).unwrap();
                us.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6);
            }
            us.sort_by(f64::total_cmp);
            let med = us[3];
            let flops = tgs as f64 * 4.0 * iters as f64 * acc as f64 * 1024.0;
            eprintln!(
                "mma_peak {name:<22} tgs={tgs:<5} median {med:9.1} us  {:6.1} TFLOPS",
                flops / (med * 1e-6) / 1e12
            );
        }
    }
}

use super::*;
use crate::metal::test_support::*;

const MOE_METAL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/moe.metal"
));
const QUANT_TILES_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../kernels/quant_tiles.h"
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
    GgmlType::Unknown,
];

/// Expert-bank dtypes the Metal loader can leave resident but that the
/// grouped MoE prefill path deliberately does not cover, with the reason.
/// Currently empty: every resident dtype has a generic instantiation.
const EXCLUDED: &[(GgmlType, &str)] = &[];

// ---------------------------------------------------------------------------
// Pure-CPU coverage tests (no Metal device).

/// Enumerate the dtypes a Qwen MoE expert bank can have once resident on
/// the GPU (`MetalModel::load` keeps `weight_dtype_kept_native` dtypes and
/// dequantizes every other dtype to F32) and require a grouped down and
/// gate/up instantiation for each, unless explicitly EXCLUDED.
#[test]
fn moe_grouped_generic_covers_every_resident_expert_dtype() {
    let mut resident: Vec<GgmlType> = ALL_GGML_TYPES
        .iter()
        .copied()
        .filter(|&d| crate::metal_forward::weight_dtype_kept_native(d))
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
    let instantiated = MOE_METAL
        .lines()
        .filter(|s| s.contains("[[host_name(\"") && s.contains("_grouped_slots_generic\")]]"))
        .count();
    assert_eq!(
        instantiated,
        MOE_GROUPED_GENERIC_LAYOUTS.len() * MoeGroupedGenericRole::ALL.len(),
        "moe.metal has generic instantiations without a Rust mapping entry"
    );
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
        GgmlType::Q4_0 | GgmlType::Q8_0 | GgmlType::IQ4_NL => &[0],
        GgmlType::Q4_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        GgmlType::IQ2_S | GgmlType::IQ3_XXS | GgmlType::IQ3_S | GgmlType::IQ4_XS => &[0],
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
    let max_abs = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
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
    let gates: Vec<Vec<f32>> = (0..r.n_expert)
        .map(|e| cpu_expert(gate_dtype, gate_bank, GU_N_HIDDEN, GU_N_FFN, e))
        .collect();
    let ups: Vec<Vec<f32>> = (0..r.n_expert)
        .map(|e| cpu_expert(up_dtype, up_bank, GU_N_HIDDEN, GU_N_FFN, e))
        .collect();
    let mut out = Vec::with_capacity(r.slot_expert.len() * GU_N_FFN);
    for (slot, &e) in r.slot_expert.iter().enumerate() {
        let t = slot / r.topk;
        let xt = &x[t * GU_N_HIDDEN..(t + 1) * GU_N_HIDDEN];
        let g = matvec(&gates[e], GU_N_HIDDEN, xt);
        let u = matvec(&ups[e], GU_N_HIDDEN, xt);
        out.extend(g.iter().zip(&u).map(|(g, u)| (g / (1.0 + (-g).exp())) * u));
    }
    out
}

fn run_generic_down_case(dtype: GgmlType) {
    let Some(ctx) = metal_test_context() else {
        return;
    };
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
    let Some(ctx) = metal_test_context() else {
        return;
    };
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
    generic_grouped_moe_down_iq4_nl_matches_cpu, generic_grouped_moe_gate_up_iq4_nl_matches_cpu => GgmlType::IQ4_NL;
    generic_grouped_moe_down_iq4_xs_matches_cpu, generic_grouped_moe_gate_up_iq4_xs_matches_cpu => GgmlType::IQ4_XS;
    generic_grouped_moe_down_f32_matches_cpu, generic_grouped_moe_gate_up_f32_matches_cpu => GgmlType::F32;
    generic_grouped_moe_down_f16_matches_cpu, generic_grouped_moe_gate_up_f16_matches_cpu => GgmlType::F16;
    generic_grouped_moe_down_bf16_matches_cpu, generic_grouped_moe_gate_up_bf16_matches_cpu => GgmlType::BF16;
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
    let Some(ctx) = metal_test_context() else {
        return;
    };
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
    let max_abs = generic
        .iter()
        .zip(&special)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let ref_max = special.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(
        max_abs <= 1e-5 * ref_max,
        "generic vs specialized Q5_K down drift {max_abs}"
    );
}

/// Generic Q4_K / Q5_K SwiGLU vs the hand-tuned grouped n16 kernels.
#[test]
fn generic_grouped_moe_gate_up_q4_k_q5_k_match_specialized() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
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
        let max_abs = generic
            .iter()
            .zip(&special)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let ref_max = special.iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(
            max_abs <= 1e-5 * ref_max,
            "generic vs specialized {dtype:?} swiglu drift {max_abs}"
        );
    }
}

#[test]
fn generic_grouped_moe_rejects_bad_contracts() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
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
        dtype: GgmlType::IQ1_S,
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

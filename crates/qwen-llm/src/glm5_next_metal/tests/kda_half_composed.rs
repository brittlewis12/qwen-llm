//! Actual UD-IQ3_XXS composition witness with every other shipped default on.
//! Unlike the historical pinned-Q4 isolated experiments, this exercises both
//! owners of normalized_half, including its refresh after FFN normalization.
//! Run release/ignored with GLM53_GGUF, MTL_DEBUG_LAYER=1 and
//! QWEN_METAL_LEASE_WAIT=1; the test takes the production GPU lease itself.

use super::*;
use crate::glm5_next::memory::KdaHalfInputsScope;
use crate::metal::{dispatch_census_begin, dispatch_census_take};

fn floats(t: &MetalTensor) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast(),
            t.n_bytes() as usize / 4,
        )
    }
}

fn same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "non-finite at {i}");
        assert_eq!(a.to_bits(), b.to_bits(), "difference at {i}");
    }
}

fn shared(s: &Glm5NextSession, poison: bool) -> (usize, u64, Vec<u64>) {
    let half = s.packed.as_ref().unwrap().normalized_half.as_ref().unwrap();
    if poison {
        // Only between synchronous calls. Every consumer must overwrite the
        // relevant rows; stale KDA operands must never reach the FFN experts.
        unsafe {
            std::ptr::write_bytes(
                half.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(half.offset as usize),
                0xff,
                half.n_bytes() as usize,
            );
        }
    }
    (
        half.buffer.contents().as_ptr() as usize,
        half.offset,
        half.shape.clone(),
    )
}

#[test]
#[ignore = "M5 actual UD-IQ3_XXS, release, GLM53_GGUF, validation ON; internal production lease"]
fn kda_half_composes_with_shipped_routed_defaults() {
    assert!(!cfg!(debug_assertions));
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    for key in [
        "QWEN_GLM53_KDA_HALF_INPUTS",
        "QWEN_EXPERT_TENSOR_TILE",
        "QWEN_GLM53_TENSOR_GATE_UP",
        "QWEN_GLM53_IQ_TENSOR_GATE_UP",
        "QWEN_GLM53_DENSE_Q6_WIDE",
        "QWEN_GLM53_DENSE_Q8_WIDE",
        "QWEN_GLM53_PREFILL_ANY_ROWS",
        "QWEN_GLM53_PREFILL_IQ_ANY_ROWS",
        "QWEN_GLM53_SHORT_PREFILL",
        "QWEN_GLM53_PREFILL_ATTN",
    ] {
        assert!(
            crate::env_flag::read_default_on(key),
            "rollback active: {key}"
        );
    }
    assert!(memory::tensor_gate_up_enabled());
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(ctx.device.name().to_string().contains("M5"));
    assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
    let gguf = GgufFile::open(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF")).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    assert_eq!(weights.config.blocks.len(), 45);
    assert_eq!(weights.config.hidden_size, 4096);
    let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../scripts/reference/glm53/quality-v1.json"
    ))
    .unwrap();
    let doc = fixture["documents"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["stratum"] == "long")
        .unwrap();
    let tokens: Vec<u32> = doc["tokens"]
        .as_str()
        .unwrap()
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();

    // Construct under rollback: routed defaults alone must already own the
    // single buffer, and enabling KDA must neither allocate nor replace it.
    let mut session = {
        let _off = KdaHalfInputsScope::new(false);
        Glm5NextSession::with_prefill_rows(&ctx, &weights, 8192, 2048).unwrap()
    };
    assert_eq!(
        session.ledger().terms(),
        Glm5NextSession::plan_memory(&ctx, &weights, 8192, 2048)
            .unwrap()
            .terms()
    );
    let identity = shared(&session, false);
    assert_eq!(identity.2, [4096, 2048]);
    let base = session.capture_snapshot().unwrap();
    assert!(
        crate::metal::evaluate_metal_memory_admission(
            8u64 << 30,
            memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true
        )
        .is_admitted()
    );
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &(0..45).collect::<Vec<_>>(), 2048, 1).unwrap(),
    )
    .unwrap();

    for rows in [512, 1025, 2048, 4096] {
        let mut features = Vec::new();
        let mut reference = Vec::<Vec<f32>>::new();
        for enabled in [false, true] {
            let _arm = KdaHalfInputsScope::new(enabled);
            session.restore_snapshot(&base).unwrap();
            assert_eq!(shared(&session, true), identity);
            dispatch_census_begin();
            let output = session
                .prefill_with_captures(&ctx, &tokens[..rows], &mut bank)
                .unwrap();
            let census = dispatch_census_take();
            let staged = census
                .iter()
                .filter(|r| {
                    r.kernel.starts_with("kernel_mat_mat_") && r.kernel.contains("f16act_tensor")
                })
                .count();
            assert_eq!(staged > 0, enabled, "KDA did not switch: rows={rows}");
            if rows >= 1024 {
                assert!(census.iter().any(|r| r.kernel.starts_with("kernel_moe_")
                    && r.kernel.contains("f16act")), "routed default bypassed");
            }
            let actual = floats(output.features().unwrap());
            if enabled {
                same(&features, actual);
            } else {
                features = actual.to_vec();
            }
            let mut logits = floats(output.logits().unwrap()).to_vec();
            drop(output);
            for pos in 0..16 {
                if enabled {
                    same(&reference[pos], &logits);
                } else {
                    reference.push(logits.clone());
                }
                if pos < 15 {
                    logits = session.forward(&ctx, tokens[rows + pos]).unwrap();
                }
            }
            assert_eq!(shared(&session, false), identity);
        }
        // Ordinary execution too: capture must not mask a lifetime failure.
        let mut ordinary = Vec::new();
        for enabled in [false, true] {
            let _arm = KdaHalfInputsScope::new(enabled);
            session.restore_snapshot(&base).unwrap();
            assert_eq!(shared(&session, true), identity);
            let logits = session.prefill_packed(&ctx, &tokens[..rows]).unwrap();
            if enabled {
                same(&ordinary, &logits);
            } else {
                ordinary = logits;
            }
        }
    }
}

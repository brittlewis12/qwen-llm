//! Bounded same-binary Maya F32 down diagnostic. No production toggle.
//!
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   GLM53_GGUF=/absolute/path/to/Maya.gguf \
//!   GLM53_F32_DOWN_AB_OUT=/absolute/path/to/new-report.json \
//!   /path/to/release/qwen_llm-test-binary \
//!   glm5_next_metal::packed::f32_down_ab::maya_f32_down_ab \
//!   --ignored --exact --nocapture --test-threads=1
//!
//! Output is optional, create-new, flushed and synced before numerical bounds
//! are asserted. Timing includes the ordinary prefill wait/logits readback;
//! allocation, continuation and diagnostics are outside the timed interval.

use super::*;
use crate::glm5_next_metal::tests::{kl_divergence, long_qualification_text, perf_lease};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

#[test]
fn original_f32_down_scope_restores_nested_and_unwind_and_preserves_half() {
    let mode = |precision, down| {
        expert_mode(
            Arith {
                lineage: PackedLineage::Fast,
                precision,
            },
            GgmlType::IQ2_S,
            GgmlType::IQ2_S,
            down,
        )
    };
    assert_eq!(
        mode(FastPrecision::F32, GgmlType::IQ3_XXS),
        StageMode::FastF32
    );
    let half_modes: Vec<_> = [
        GgmlType::IQ3_XXS,
        GgmlType::Q4_K,
        GgmlType::IQ3_S,
        GgmlType::IQ4_XS,
    ]
    .into_iter()
    .map(|d| (d, mode(FastPrecision::Half, d)))
    .collect();
    with_original_f32_down_support(true, || {
        for &(down, half) in &half_modes {
            assert_eq!(mode(FastPrecision::Half, down), half);
            assert_eq!(mode(FastPrecision::DenseF32, down), half);
            assert_eq!(
                mode(FastPrecision::F32, down),
                if matches!(down, GgmlType::IQ3_S | GgmlType::IQ4_XS) {
                    StageMode::FastF32
                } else {
                    StageMode::Exact
                }
            );
        }
        // The primitive remains fully capable even while GLM uses the control.
        assert!(crate::metal::grouped_routed_experts_f32x_supported(
            GgmlType::IQ2_S,
            GgmlType::IQ2_S,
            GgmlType::IQ3_XXS
        ));
        with_original_f32_down_support(false, || {
            assert_eq!(
                mode(FastPrecision::F32, GgmlType::IQ3_XXS),
                StageMode::FastF32
            );
        });
        assert_eq!(
            mode(FastPrecision::F32, GgmlType::IQ3_XXS),
            StageMode::Exact
        );
        let panic = std::panic::catch_unwind(|| {
            with_original_f32_down_support(false, || panic!("scope unwind"))
        });
        assert!(panic.is_err());
        assert_eq!(
            mode(FastPrecision::F32, GgmlType::IQ3_XXS),
            StageMode::Exact
        );
    });
    assert_eq!(
        mode(FastPrecision::F32, GgmlType::IQ3_XXS),
        StageMode::FastF32
    );
    let panic = std::panic::catch_unwind(|| {
        with_original_f32_down_support(true, || panic!("outer unwind"))
    });
    assert!(panic.is_err());
    assert_eq!(
        mode(FastPrecision::F32, GgmlType::IQ3_XXS),
        StageMode::FastF32
    );
}

const ROWS: usize = 512;
const STEPS: usize = 3;
const KL_LIMIT: f64 = 1e-3;

#[test]
#[ignore = "Maya model timing: explicit GLM53_GGUF, release binary, no MTL_DEBUG_LAYER, production lease; optional new GLM53_F32_DOWN_AB_OUT JSON"]
fn maya_f32_down_ab() {
    assert!(
        !cfg!(debug_assertions),
        "timing requires a release test binary"
    );
    let _lease = perf_lease();
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name Maya"));
    let mut output = std::env::var_os("GLM53_F32_DOWN_AB_OUT").map(|path| {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("GLM53_F32_DOWN_AB_OUT must be a new writable JSON path")
    });
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).expect("Maya GGUF");
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(&ctx, &gguf, artifact.model(), ROWS + STEPS, ROWS)
        .expect("production memory preflight");
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{}", long_qualification_text()), false)
        .unwrap()
        .into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .take(ROWS + STEPS)
        .collect();
    assert_eq!(
        tokens.len(),
        ROWS + STEPS,
        "natural prompt and continuation"
    );
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("one Maya weight load");
    let mut affected_blocks = Vec::new();
    let mut maya_blocks = Vec::new();
    for (block, weights) in weights.blocks.iter().enumerate() {
        let FfnTensors::Moe(moe) = &weights.ffn else {
            continue;
        };
        let arith = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::F32,
        };
        let select = || {
            expert_mode(
                arith,
                moe.gate_experts.dtype,
                moe.up_experts.dtype,
                moe.down_experts.dtype,
            )
        };
        let original = with_original_f32_down_support(true, select);
        let candidate = with_original_f32_down_support(false, select);
        if original != candidate {
            assert_eq!(original, StageMode::Exact);
            assert_eq!(candidate, StageMode::FastF32);
            affected_blocks.push(block);
        }
        if (
            moe.gate_experts.dtype,
            moe.up_experts.dtype,
            moe.down_experts.dtype,
        ) == (GgmlType::IQ2_S, GgmlType::IQ2_S, GgmlType::IQ3_XXS)
        {
            assert_ne!(original, candidate);
            maya_blocks.push(block);
        }
    }
    assert!(
        !maya_blocks.is_empty(),
        "expected Maya IQ2_S gate/up + IQ3_XXS down; control must exercise row fallback"
    );
    // A1 and one candidate's four host logits rows coexist during comparison.
    // Reserve those rows, one readback row, two F64 KL working vectors and
    // 1 MiB of diagnostic overhead (the session itself is dropped before KL).
    let vocab = u64::from(weights.config.vocab_size);
    let cpu_reserve = (2 * (STEPS + 1) + 1) as u64 * vocab * 4 + 2 * vocab * 8 + (1 << 20);
    let run = |original| {
        with_original_f32_down_support(original, || {
            let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                &ctx,
                &weights,
                ROWS + STEPS,
                ROWS,
                cpu_reserve,
            )
            .unwrap();
            session.set_packed_lineage(PackedLineage::Fast).unwrap();
            session.set_fast_precision(FastPrecision::F32).unwrap();
            let started = std::time::Instant::now();
            let last = session.prefill_packed(&ctx, &tokens[..ROWS]).unwrap();
            let wall_ms = started.elapsed().as_secs_f64() * 1e3;
            let mut logits = vec![last];
            for &token in &tokens[ROWS..] {
                logits.push(session.forward(&ctx, token).unwrap());
            }
            assert_eq!(session.position(), ROWS + STEPS);
            for row in &logits {
                assert_eq!(row.len(), weights.config.vocab_size as usize);
                assert!(row.iter().all(|v| v.is_finite()), "non-finite model logits");
            }
            (wall_ms, logits)
        })
    };
    let mut warmups = Vec::new();
    for (arm, original) in [("A", true), ("B", false)] {
        let (wall_ms, _) = run(original);
        eprintln!("Maya F32 down warmup {arm}: {wall_ms:.2} ms / {ROWS} tokens");
        warmups.push(json!({"arm":arm,"prefill_wall_ms":wall_ms}));
    }
    let mut reference = None;
    let mut runs = Vec::new();
    let mut times = [Vec::new(), Vec::new()];
    let mut failures = Vec::new();
    for (label, original) in [("A1", true), ("B1", false), ("B2", false), ("A2", true)] {
        let (wall_ms, logits) = run(original);
        let baseline = reference.as_ref().unwrap_or(&logits);
        let kls: Vec<_> = baseline
            .iter()
            .zip(&logits)
            .map(|(a, b)| kl_divergence(a, b))
            .collect();
        assert_eq!(kls.len(), STEPS + 1);
        for (step, &kl) in kls.iter().enumerate() {
            if kl > KL_LIMIT {
                failures.push(format!(
                    "{label} continuation step {step}: KL(A1||run)={kl:.6e} > {KL_LIMIT}"
                ));
            }
        }
        let worst = kls.iter().copied().fold(0.0f64, f64::max);
        eprintln!(
            "Maya F32 down {label}: {wall_ms:.2} ms / {ROWS} tokens; KL(A1||run) last={:.3e}, worst through {STEPS} decode steps={worst:.3e}",
            kls[0]
        );
        times[usize::from(!original)].push(wall_ms);
        runs.push(json!({"label":label,"arm":if original {"A"} else {"B"},
            "prefill_wall_ms":wall_ms,"tokens_per_second":ROWS as f64 * 1000.0 / wall_ms,
            "kl_a1_to_run_by_step":kls,"finite_logits":true}));
        if reference.is_none() {
            reference = Some(logits);
        }
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (a_ms, b_ms) = (mean(&times[0]), mean(&times[1]));
    eprintln!(
        "Maya F32 down ABBA mean: A={a_ms:.2} ms, B={b_ms:.2} ms, A/B={:.3}x (diagnostic; no performance bound)",
        a_ms / b_ms
    );
    let document = json!({
        "schema":"glm53.maya_f32_down_ab.v1", "model":path,
        "test_binary":std::env::current_exe().unwrap(),
        "precision":"F32", "lineage":"Fast", "prefill_rows":ROWS, "capacity":ROWS+STEPS,
        "order":"warmup A B; measured A1 B1 B2 A2; one weight load; fresh session each run",
        "arms":{"A":"test-only original down eligibility IQ3_S/IQ4_XS; other F32 down uses row fallback",
            "B":"production full F32 down capability; all other policies unchanged"},
        "timing":"unprofiled prefill wall including GPU wait and final logits readback; allocation, continuation and diagnostics excluded; no MTL_DEBUG_LAYER",
        "prompt":"[gMASK]<sop> + long_qualification_text, first 512 tokens",
        "prompt_tokens":&tokens[..ROWS], "teacher_forced_tokens":&tokens[ROWS..],
        "affected_blocks":affected_blocks, "maya_iq3_xxs_down_blocks":maya_blocks,
        "cpu_reserve_bytes":cpu_reserve, "warmups":warmups, "runs":runs,
        "a_mean_ms":a_ms, "b_mean_ms":b_ms, "a_over_b_speed_ratio":a_ms/b_ms,
        "numerics":{"reference":"A1", "kl_limit":KL_LIMIT, "failures":failures},
        "performance_assertion":false
    });
    if let Some(out) = &mut output {
        serde_json::to_writer_pretty(&mut *out, &document).unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
        out.sync_all().unwrap();
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

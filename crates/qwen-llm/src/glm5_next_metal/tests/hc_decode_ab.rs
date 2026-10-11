//! Same-binary Maya decode HC diagnostic; production precision is unchanged.
//!
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   GLM53_GGUF=/absolute/path/to/Maya.gguf \
//!   GLM53_HC_DECODE_AB_OUT=/absolute/path/to/new-report.json \
//!   /path/to/release/test-binary \
//!   glm5_next_metal::packed::hc_decode_ab::maya_hc_decode_ab \
//!   --ignored --exact --nocapture --test-threads=1
//!
//! Output is optional and create-new. KL bound failures are saved and synced
//! before assertion. Allocation, common Half prefill, and CPU comparisons are
//! outside decode timing. Only warmups collect a first-step dispatch census;
//! measured decode uses ordinary unprofiled forward calls.

use super::*;
use crate::glm5_next_metal::tests::{kl_divergence, long_qualification_text, perf_lease};
use crate::metal::{
    dispatch_census_begin, dispatch_census_take, with_mhc4_pre_generic_baseline_override,
};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

const PROMPT: usize = 512;
const STEPS: usize = 16;
const KL_LIMIT: f64 = 1e-3;

#[test]
#[ignore = "Maya decode HC timing: explicit GLM53_GGUF, release binary, no MTL_DEBUG_LAYER, production lease; optional new GLM53_HC_DECODE_AB_OUT JSON"]
fn maya_hc_decode_ab() {
    assert!(
        !cfg!(debug_assertions),
        "timing requires a release test binary"
    );
    let _lease = perf_lease();
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name Maya"));
    let mut output = std::env::var_os("GLM53_HC_DECODE_AB_OUT").map(|path| {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("GLM53_HC_DECODE_AB_OUT must be a new writable JSON path")
    });
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).expect("Maya GGUF");
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(&ctx, &gguf, artifact.model(), PROMPT + STEPS, PROMPT)
        .expect("production memory preflight");
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{}", long_qualification_text()), false)
        .unwrap()
        .into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .take(PROMPT + STEPS)
        .collect();
    assert_eq!(
        tokens.len(),
        PROMPT + STEPS,
        "natural prompt plus common continuation"
    );
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("one Maya weight load");
    let mut hc_types = Vec::new();
    for (block, b) in weights.blocks.iter().enumerate() {
        for (stage, dtype) in [
            ("attention", b.attention_hc.mix.dtype),
            ("ffn", b.ffn_hc.mix.dtype),
        ] {
            hc_types.push(json!({"block":block,"stage":stage,"dtype":format!("{dtype:?}")}));
        }
    }
    assert!(
        weights
            .blocks
            .iter()
            .any(|b| b.attention_hc.mix.dtype != GgmlType::Q8_0
                || b.ffn_hc.mix.dtype != GgmlType::Q8_0),
        "Maya must exercise generic non-Q8 HC"
    );
    // Baseline and candidate full-vocabulary trajectories, one transient
    // readback row, two F64 KL vectors and bounded report/census overhead.
    let vocab = u64::from(weights.config.vocab_size);
    let cpu_reserve = (2 * STEPS + 1) as u64 * vocab * 4 + 2 * vocab * 8 + (4 << 20);
    let run = |baseline: bool, warmup: bool| {
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            PROMPT + STEPS,
            PROMPT,
            cpu_reserve,
        )
        .unwrap();
        // No precision setter: both arms keep the same default Half prefill
        // and normal decode operand precision.
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        let prefix = with_mhc4_pre_generic_baseline_override(false, || {
            session.prefill_packed(&ctx, &tokens[..PROMPT]).unwrap()
        });
        assert_eq!(prefix.len(), weights.config.vocab_size as usize);
        assert!(
            prefix.iter().all(|v| v.is_finite()),
            "non-finite prefix logits"
        );
        drop(prefix);
        assert_eq!(session.position(), PROMPT);
        // Primitive's closure scope restores its thread-local control on both
        // normal return and unwind. The override is applied only to decode.
        with_mhc4_pre_generic_baseline_override(baseline, || {
            let mut wall_ms = Vec::with_capacity(STEPS);
            let mut logits = Vec::with_capacity(STEPS);
            let mut warmup_dispatch = None;
            for (step, &token) in tokens[PROMPT..].iter().enumerate() {
                let census = warmup && step == 0;
                if census {
                    dispatch_census_begin();
                }
                let started = std::time::Instant::now();
                let row = session.forward(&ctx, token).unwrap();
                let ms = started.elapsed().as_secs_f64() * 1e3;
                if census {
                    let kernels = dispatch_census_take();
                    let partial = kernels
                        .iter()
                        .filter(|r| r.kernel.starts_with("kernel_mhc4_pre_native_partial_"))
                        .count();
                    let finish = kernels
                        .iter()
                        .filter(|r| r.kernel == "kernel_mhc4_pre_native_finish")
                        .count();
                    if baseline {
                        assert_eq!(
                            (partial, finish),
                            (0, 0),
                            "A must force the old generic HC composition"
                        );
                    } else {
                        assert!(finish > 0, "B must exercise the native HC candidate");
                        assert_eq!(partial, finish);
                    }
                    warmup_dispatch = Some(
                        json!({"native_partial_dispatches":partial,"native_finish_dispatches":finish}),
                    );
                }
                assert_eq!(row.len(), weights.config.vocab_size as usize);
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "non-finite decode logits at position {}",
                    PROMPT + step
                );
                wall_ms.push(ms);
                logits.push(row);
            }
            assert_eq!(session.position(), PROMPT + STEPS);
            (wall_ms, logits, warmup_dispatch)
        })
    };
    let mut warmups = Vec::new();
    for (arm, baseline) in [("A", true), ("B", false)] {
        let (_, _, dispatch) = run(baseline, true);
        eprintln!(
            "Maya HC decode warmup {arm}: {STEPS} common tokens; first-step dispatch {dispatch:?}"
        );
        warmups.push(json!({"arm":arm,"first_step_dispatch":dispatch}));
    }
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut runs = Vec::new();
    let mut arm_ms = [Vec::new(), Vec::new()];
    let mut failures = Vec::new();
    for (label, baseline) in [("A1", true), ("B1", false), ("B2", false), ("A2", true)] {
        let (wall_ms, logits, dispatch) = run(baseline, false);
        assert!(dispatch.is_none(), "measured decode must have no census");
        let a1 = reference.as_ref().unwrap_or(&logits);
        let kls: Vec<_> = a1
            .iter()
            .zip(&logits)
            .map(|(a, b)| kl_divergence(a, b))
            .collect();
        assert_eq!(kls.len(), STEPS);
        for (step, &kl) in kls.iter().enumerate() {
            if kl > KL_LIMIT {
                failures.push(format!(
                    "{label} position {}: KL(A1||run)={kl:.6e} > {KL_LIMIT}",
                    PROMPT + step
                ));
            }
        }
        let total_ms = wall_ms.iter().sum::<f64>();
        let mean_ms = total_ms / STEPS as f64;
        let tokens_per_second = STEPS as f64 * 1000.0 / total_ms;
        let worst = kls.iter().copied().fold(0.0f64, f64::max);
        eprintln!(
            "Maya HC decode {label}: mean {mean_ms:.2} ms/token, {tokens_per_second:.3} tokens/s, worst KL(A1||run)={worst:.3e}"
        );
        eprintln!("  per-step wall ms: {wall_ms:?}\n  per-step KL: {kls:?}");
        arm_ms[usize::from(!baseline)].extend_from_slice(&wall_ms);
        runs.push(json!({"label":label,"arm":if baseline {"A"} else {"B"},
            "decode_wall_ms_by_step":wall_ms,"mean_wall_ms_per_token":mean_ms,
            "tokens_per_second":tokens_per_second,"kl_a1_to_run_by_step":kls,"finite_logits":true}));
        if reference.is_none() {
            reference = Some(logits);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (a_ms, b_ms) = (mean(&arm_ms[0]), mean(&arm_ms[1]));
    eprintln!(
        "Maya HC decode ABBA: A {:.3} tokens/s, B {:.3} tokens/s, A/B wall ratio {:.3}x (no performance threshold)",
        1000.0 / a_ms,
        1000.0 / b_ms,
        a_ms / b_ms
    );
    let document = json!({
        "schema":"glm53.maya_hc_decode_ab.v1", "model":path, "test_binary":std::env::current_exe().unwrap(),
        "prompt_rows":PROMPT,"decode_steps_per_run":STEPS,"decode_positions":(PROMPT..PROMPT+STEPS).collect::<Vec<_>>(),
        "capacity":PROMPT+STEPS,"prefill_rows":PROMPT,"prefill_precision":"default Half","decode_precision":"unchanged normal decode",
        "order":"warmup A B; measured A1 B1 B2 A2; one weight load; fresh session and identical prefill each run",
        "arms":{"A":"primitive baseline override true: old 6-launch generic HC decode",
            "B":"primitive baseline override false: eligible native 4-launch normalized split-K and fused finish"},
        "timing":"unprofiled per-step forward wall, including GPU wait and logits readback; prefix prefill, allocation, CPU comparison excluded; no measured census or MTL_DEBUG_LAYER",
        "prompt":"[gMASK]<sop> + long_qualification_text, first 512 tokens",
        "prompt_tokens":&tokens[..PROMPT],"teacher_forced_tokens":&tokens[PROMPT..],"hc_types":hc_types,
        "cpu_reserve_bytes":cpu_reserve,"warmups":warmups,"runs":runs,
        "a_mean_wall_ms_per_token":a_ms,"b_mean_wall_ms_per_token":b_ms,
        "a_tokens_per_second":1000.0/a_ms,"b_tokens_per_second":1000.0/b_ms,"a_over_b_wall_ratio":a_ms/b_ms,
        "numerics":{"reference":"A1","kl_limit":KL_LIMIT,"failures":failures},"performance_assertion":false
    });
    if let Some(out) = &mut output {
        serde_json::to_writer_pretty(&mut *out, &document).unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
        out.sync_all().unwrap();
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

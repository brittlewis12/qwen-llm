//! Same-binary grouped latent diagnostics for Maya or the UD Q8 control.
//!
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   GLM53_GGUF=/absolute/path/to/model.gguf \
//!   GLM53_LATENT_MMA_AB_OUT=/absolute/path/to/new-report.json \
//!   /path/to/release/test-binary \
//!   glm5_next_metal::packed::latent_mma_ab::grouped_latent_mma_ab \
//!   --ignored --exact --nocapture --test-threads=1
//!
//! Optional GLM53_LATENT_MMA_AB_DEPTH=4096 adds a 512-row suffix after that
//! prefix. No grid search. Half remains the default. Each cell warms A/B,
//! verifies warmup-only dispatch census, then times A1/B1/B2/A2 without tracing.
//! Common prefixes use scalar baseline outside timing to isolate the suffix.
//! The historical MMA diagnostic explicitly forces MMA in Half and retains its
//! original KL bound; it does not represent Half's current production policy.
//! For scalar4 versus test-only Half scalar8, select `grouped_latent_scalar8_ab`
//! and use GLM53_LATENT_SCALAR8_AB_OUT for a separate new JSON file. That test
//! requires bit-identical final logits and three common continuation steps;
//! scalar8 performance is measured without a pass threshold.

use super::*;
use crate::glm5_next_metal::tests::{kl_divergence, long_qualification_text, perf_lease};
use crate::metal::{
    DispatchCensusRow, dispatch_census_begin, dispatch_census_take,
    with_grouped_projection_scalar_baseline_override, with_grouped_projection_scalar8_override,
};
use serde_json::{Value, json};
use std::io::{Seek, Write};
use std::path::PathBuf;

const STEPS: usize = 3;
const KL_LIMIT: f64 = 1e-3;
const Q6_MMA: &str = "kernel_latent_grouped_q6_K_f32_r2c16k64";
const Q6_SCALAR: &str = "kernel_latent_grouped_q6_K_f32";
const Q6_SCALAR8: &str = "kernel_latent_grouped_q6_K_f32_scalar8";

#[derive(Clone, Copy)]
struct Cell {
    name: &'static str,
    prefix: usize,
    rows: usize,
    workspace: usize,
}

impl Cell {
    fn end(self) -> usize {
        self.prefix + self.rows
    }
    fn capacity(self) -> usize {
        self.end() + STEPS
    }
}

fn save(out: &mut Option<std::fs::File>, document: &Value) {
    if let Some(out) = out {
        out.rewind().unwrap();
        out.set_len(0).unwrap();
        serde_json::to_writer_pretty(&mut *out, document).unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
        out.sync_all().unwrap();
    }
}

// Compare Q8 launch names and geometry, not encoder ordinals shifted by Q6
// substitution. Group depth separates latent heads from HC's groups=1.
fn q8_signature(rows: &[DispatchCensusRow], groups: u64) -> Vec<(String, [u64; 3], [u64; 3])> {
    let mut signature: Vec<_> = rows
        .iter()
        .filter(|r| {
            r.grid_depth == groups && r.kernel.contains("q8_0") && r.kernel.contains("grouped")
        })
        .map(|r| {
            (
                r.kernel.clone(),
                [r.grid_width, r.grid_height, r.grid_depth],
                [r.threads_width, r.threads_height, r.threads_depth],
            )
        })
        .collect();
    signature.sort();
    signature
}

fn verify_q6_census(
    rows: &[DispatchCensusRow],
    cell: Cell,
    baseline: bool,
    groups: u64,
    q6_calls: usize,
) -> Value {
    assert!(
        !rows
            .iter()
            .any(|r| r.grid_depth == groups && r.kernel == Q6_SCALAR8),
        "historical scalar4/MMA arms must not substitute scalar8"
    );
    let mma: Vec<_> = rows
        .iter()
        .filter(|r| r.grid_depth == groups && r.kernel == Q6_MMA)
        .collect();
    let scalar: Vec<_> = rows
        .iter()
        .filter(|r| r.grid_depth == groups && r.kernel == Q6_SCALAR)
        .collect();
    if baseline {
        assert!(mma.is_empty(), "A must retain scalar Q6 latent projection");
        assert_eq!(scalar.len(), q6_calls);
        assert!(
            scalar
                .iter()
                .all(|r| r.grid_height == cell.rows.div_ceil(4) as u64)
        );
    } else {
        assert_eq!(
            mma.len(),
            q6_calls,
            "B must actually dispatch eligible Q6 MMA"
        );
        assert!(mma.iter().all(|r| r.grid_width == (cell.rows / 128) as u64));
        let tail = cell.rows % 128;
        assert_eq!(scalar.len(), if tail > 0 { q6_calls } else { 0 });
        assert!(
            scalar
                .iter()
                .all(|r| r.grid_height == tail.div_ceil(4) as u64),
            "ragged candidate must encode only its scalar remainder"
        );
    }
    let encode = |r: &&DispatchCensusRow| {
        json!({"kernel":r.kernel,
        "grid":[r.grid_width,r.grid_height,r.grid_depth],
        "threads":[r.threads_width,r.threads_height,r.threads_depth]})
    };
    json!({"q6_mma":mma.iter().map(encode).collect::<Vec<_>>(),
        "q6_scalar":scalar.iter().map(encode).collect::<Vec<_>>()})
}

// Verify the actual scalar8 pipeline and its row geometry, including ragged cells.
fn verify_q6_scalar8_census(
    rows: &[DispatchCensusRow],
    cell: Cell,
    baseline: bool,
    groups: u64,
    q6_calls: usize,
) -> Value {
    if baseline {
        return verify_q6_census(rows, cell, true, groups, q6_calls);
    }
    let q6: Vec<_> = rows
        .iter()
        .filter(|r| {
            r.grid_depth == groups && r.kernel.starts_with("kernel_latent_grouped_q6_K_f32")
        })
        .collect();
    assert_eq!(q6.len(), q6_calls);
    assert!(
        q6.iter()
            .all(|r| r.kernel == Q6_SCALAR8 && r.grid_height == cell.rows.div_ceil(8) as u64),
        "explicit scalar8 scope must use the scalar8 pipeline, without MMA or scalar4 remainder"
    );
    json!({"q6_mma":[],"q6_scalar":q6.iter().map(|r| json!({"kernel":r.kernel,
        "grid":[r.grid_width,r.grid_height,r.grid_depth],
        "threads":[r.threads_width,r.threads_height,r.threads_depth]})).collect::<Vec<_>>(),
        "scalar_rows_per_tile":8})
}

#[test]
#[ignore = "model timing: explicit GLM53_GGUF (Maya or UD), release binary, no MTL_DEBUG_LAYER, production lease; optional new GLM53_LATENT_MMA_AB_OUT and DEPTH=4096"]
fn grouped_latent_mma_ab() {
    grouped_latent_ab_packet(false);
}

/// Half's production policy disallows MMA. A forces historical scalar4;
/// B explicitly enables test-only scalar8. Production Half retains scalar4.
/// Same cells/depth control as grouped_latent_mma_ab, separate evidence file.
#[test]
#[ignore = "Half scalar4/test-only scalar8 timing: GLM53_GGUF, release binary, no MTL_DEBUG_LAYER; optional new GLM53_LATENT_SCALAR8_AB_OUT and GLM53_LATENT_MMA_AB_DEPTH=4096"]
fn grouped_latent_scalar8_ab() {
    grouped_latent_ab_packet(true);
}

fn grouped_latent_ab_packet(scalar8: bool) {
    assert!(
        !cfg!(debug_assertions),
        "timing requires a release test binary"
    );
    let _lease = perf_lease();
    let path =
        PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name the artifact"));
    let output_env = if scalar8 {
        "GLM53_LATENT_SCALAR8_AB_OUT"
    } else {
        "GLM53_LATENT_MMA_AB_OUT"
    };
    let experiment = if scalar8 {
        "Half test-only scalar8"
    } else {
        "forced MMA"
    };
    let mut output = std::env::var_os(output_env).map(|path| {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap_or_else(|e| panic!("{output_env} must be a new writable JSON path: {e}"))
    });
    let mut cells = vec![
        Cell {
            name: "full512",
            prefix: 0,
            rows: 512,
            workspace: 512,
        },
        Cell {
            name: "full2048",
            prefix: 0,
            rows: 2048,
            workspace: 2048,
        },
        Cell {
            name: "suffix129_after512",
            prefix: 512,
            rows: 129,
            workspace: 512,
        },
    ];
    if let Some(depth) = std::env::var_os("GLM53_LATENT_MMA_AB_DEPTH") {
        assert_eq!(
            depth.to_str(),
            Some("4096"),
            "only the optional depth 4096 cell is supported"
        );
        cells.push(Cell {
            name: "suffix512_after4096",
            prefix: 4096,
            rows: 512,
            workspace: 512,
        });
    }
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).expect("model GGUF");
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let needed = cells.iter().map(|c| c.capacity()).max().unwrap();
    assert!(
        needed <= artifact.model().config.context_length as usize,
        "cells must fit model context"
    );
    for cell in &cells {
        preflight_session(
            &ctx,
            &gguf,
            artifact.model(),
            cell.capacity(),
            cell.workspace,
        )
        .expect("production memory preflight for cell");
    }
    // Repeat the existing natural corpus once to also supply the optional
    // 4096-prefix + 512-suffix + 3 continuation tokens without a synthetic ID stream.
    let text = long_qualification_text();
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{text}\n\n{text}"), false)
        .unwrap()
        .into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .take(needed)
        .collect();
    assert_eq!(
        tokens.len(),
        needed,
        "natural text must cover every requested cell"
    );
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("one resident model load");
    let groups = u64::from(weights.config.head_count);
    assert!(
        groups > 1,
        "census distinguishes latent head groups from HC"
    );
    let (mut q6_calls, mut q8_calls) = (0, 0);
    let mut banks = Vec::new();
    for (block, b) in weights.blocks.iter().enumerate() {
        let MixerTensors::Mla(mla) = &b.mixer else {
            continue;
        };
        for (role, bank) in [
            ("key_absorb", &mla.key_absorb),
            ("value_expand", &mla.value_expand),
        ] {
            match bank.dtype {
                GgmlType::Q6_K => {
                    assert!(bank.shape[0].is_multiple_of(256) && bank.shape[1].is_multiple_of(16));
                    q6_calls += 1;
                }
                GgmlType::Q8_0 => {
                    q8_calls += 1;
                }
                _ => {}
            }
            banks.push(json!({"block":block,"role":role,"dtype":format!("{:?}",bank.dtype),"shape":bank.shape}));
        }
    }
    assert!(
        q6_calls + q8_calls > 0,
        "artifact must exercise Q6 candidate or Q8 control"
    );
    let vocab = u64::from(weights.config.vocab_size);
    let cpu_reserve = (2 * (STEPS + 1) + 1) as u64 * vocab * 4 + 2 * vocab * 8 + (4 << 20);
    let mut report = json!({"schema":if scalar8 { "glm53.grouped_latent_scalar8_ab.v1" } else { "glm53.grouped_latent_mma_ab.v1" },"model":path,
        "test_binary":std::env::current_exe().unwrap(),"precision":"default Half","lineage":"default Fast",
        "order":"each cell: warmup A B with census; then untraced A1 B1 B2 A2; fresh sessions, one weight load",
        "arms":{"A":"Some(true): forced historical Q6 scalar4",
            "B":if scalar8 { "explicit test-only scalar8 override; Half caller allow_mma=false; baseline override None" }
                else { "Some(false): forced Q6 MMA prefix plus scalar4 remainder, including in Half" }},
        "prefix_policy":"identical scalar-baseline prefix, chunks of workspace rows, outside timing",
        "timing":"ordinary full prefill or suffix wall, including GPU wait and final logits readback; load, allocation, prefix, continuation, warmup and diagnostics excluded",
        "prompt":"[gMASK]<sop> + existing long_qualification_text repeated twice","tokens":tokens,
        "latent_banks":banks,"q6_calls_per_chunk":q6_calls,"q8_calls_per_chunk":q8_calls,
        "cpu_reserve_bytes":cpu_reserve,"kl_limit":if scalar8 { None } else { Some(KL_LIMIT) },
        "bit_identity_required":scalar8,"cells":[],"performance_assertion":false});
    save(&mut output, &report);
    let mut failures = Vec::new();
    for cell in cells {
        let ledger =
            Glm5NextSession::plan_memory(&ctx, &weights, cell.capacity(), cell.workspace).unwrap();
        let run = |baseline: bool, warmup: bool| {
            let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
                &ctx,
                &weights,
                cell.capacity(),
                cell.workspace,
                cpu_reserve,
            )
            .unwrap();
            assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
            assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
            if cell.prefix > 0 {
                let prefix = with_grouped_projection_scalar_baseline_override(true, || {
                    session
                        .prefill_packed(&ctx, &tokens[..cell.prefix])
                        .unwrap()
                });
                assert_eq!(prefix.len(), vocab as usize);
                assert!(prefix.iter().all(|v| v.is_finite()));
            }
            assert_eq!(session.position(), cell.prefix);
            assert!(!latent_mma_allowed(Arith {
                lineage: PackedLineage::Fast,
                precision: FastPrecision::Half
            }));
            let encode = || {
                if warmup {
                    dispatch_census_begin();
                }
                let started = (!warmup).then(std::time::Instant::now);
                let result = session.prefill_packed(&ctx, &tokens[cell.prefix..cell.end()]);
                let wall_ms = started.map(|t| t.elapsed().as_secs_f64() * 1e3);
                let census = warmup.then(dispatch_census_take);
                let mut logits = vec![result.unwrap()];
                for &token in &tokens[cell.end()..cell.capacity()] {
                    logits.push(session.forward(&ctx, token).unwrap());
                }
                assert_eq!(session.position(), cell.capacity());
                for row in &logits {
                    assert_eq!(row.len(), vocab as usize);
                    assert!(
                        row.iter().all(|v| v.is_finite()),
                        "{}: non-finite full-vocabulary logits",
                        cell.name
                    );
                }
                (wall_ms, logits, census)
            };
            if scalar8 && !baseline {
                // Baseline override is restored to None. Half disallows MMA;
                // only this test scope enables the scalar8 candidate.
                with_grouped_projection_scalar8_override(true, encode)
            } else {
                // Preserve the historical Half scalar4/forced-MMA experiment.
                with_grouped_projection_scalar_baseline_override(baseline, encode)
            }
        };
        let mut warmups = Vec::new();
        let mut q8_control = None;
        for (arm, baseline) in [("A", true), ("B", false)] {
            let (wall, _, census) = run(baseline, true);
            assert!(wall.is_none());
            let census = census.unwrap();
            let q6 = if scalar8 {
                verify_q6_scalar8_census(&census, cell, baseline, groups, q6_calls)
            } else {
                verify_q6_census(&census, cell, baseline, groups, q6_calls)
            };
            let q8 = q8_signature(&census, groups);
            assert_eq!(
                q8.len(),
                q8_calls * (1 + usize::from(cell.rows % 128 > 0)),
                "Q8 prefix/tail path count"
            );
            if let Some(a) = &q8_control {
                assert_eq!(&q8, a, "Q8 names and launch geometry must remain unchanged");
            } else {
                q8_control = Some(q8.clone());
            }
            eprintln!(
                "latent {experiment} {} warmup {arm}: Q6 MMA/scalar={}/{}, Q8 launches={}",
                cell.name,
                q6["q6_mma"].as_array().unwrap().len(),
                q6["q6_scalar"].as_array().unwrap().len(),
                q8.len()
            );
            warmups.push(json!({"arm":arm,"q6":q6,"q8_signature":q8}));
        }
        let mut reference: Option<Vec<Vec<f32>>> = None;
        let mut runs = Vec::new();
        let mut times = [Vec::new(), Vec::new()];
        let mut cell_failures = Vec::new();
        for (label, baseline) in [("A1", true), ("B1", false), ("B2", false), ("A2", true)] {
            let (wall, logits, census) = run(baseline, false);
            assert!(census.is_none(), "timed runs must not trace");
            let wall_ms = wall.unwrap();
            let a1 = reference.as_ref().unwrap_or(&logits);
            let kls: Vec<_> = a1
                .iter()
                .zip(&logits)
                .map(|(a, b)| kl_divergence(a, b))
                .collect();
            assert_eq!(kls.len(), STEPS + 1);
            let bit_identical = a1
                .iter()
                .zip(&logits)
                .all(|(a, b)| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            if scalar8 && !bit_identical {
                cell_failures.push(format!(
                    "{} {label}: scalar8 logits/continuation differ in bits from A1 scalar4",
                    cell.name
                ));
            }
            for (step, &kl) in kls.iter().enumerate() {
                if !scalar8 && kl > KL_LIMIT {
                    cell_failures.push(format!(
                        "{} {label} position {}: KL(A1||run)={kl:.6e} > {KL_LIMIT}",
                        cell.name,
                        cell.end() - 1 + step
                    ));
                }
            }
            eprintln!(
                "latent {experiment} {} {label}: {wall_ms:.2} ms, {:.3} tokens/s, bit-identical={bit_identical}, KL(A1||run) at join + 3 decode steps {kls:?}",
                cell.name,
                cell.rows as f64 * 1000.0 / wall_ms
            );
            times[usize::from(!baseline)].push(wall_ms);
            runs.push(json!({"label":label,"prefill_wall_ms":wall_ms,
                "tokens_per_second":cell.rows as f64 * 1000.0 / wall_ms,"finite_logits":true,
                "bit_identical_to_a1":bit_identical,"kl_a1_to_run_by_step":kls}));
            if reference.is_none() {
                reference = Some(logits);
            }
        }
        let mean = |t: &[f64]| t.iter().sum::<f64>() / t.len() as f64;
        let (a_ms, b_ms) = (mean(&times[0]), mean(&times[1]));
        eprintln!(
            "latent {experiment} {} ABBA mean: A={a_ms:.2} ms B={b_ms:.2} ms A/B={:.3}x; no performance threshold",
            cell.name,
            a_ms / b_ms
        );
        report["cells"].as_array_mut().unwrap().push(json!({"name":cell.name,"prefix_rows":cell.prefix,
            "timed_rows":cell.rows,"workspace_rows":cell.workspace,"capacity":cell.capacity(),
            "ledger_peak_bytes":ledger.peak_bytes(),"session_buffer_bytes":ledger.session_buffer_bytes(),
            "prefix_chunk_rows":cell.workspace,"timed_chunk_rows":cell.rows,
            "teacher_forced_tokens":&tokens[cell.end()..cell.capacity()],"warmups":warmups,"runs":runs,
            "a_mean_ms":a_ms,"b_mean_ms":b_ms,"a_over_b_wall_ratio":a_ms/b_ms,"kl_failures":cell_failures}));
        if scalar8 {
            let last = report["cells"].as_array_mut().unwrap().last_mut().unwrap();
            last["bit_identity_failures"] = json!(cell_failures);
            last.as_object_mut().unwrap().remove("kl_failures");
        }
        failures.extend(cell_failures);
        save(&mut output, &report);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

// Numerical observations only: neither scalar Fast nor Exact is a quality
// oracle. Exact supplies a common direction for choice regret on this prompt.
fn precision_probe_comparison(
    prefill_rows: usize,
    exact: &[Vec<f32>],
    a: &[Vec<f32>],
    b: &[Vec<f32>],
) -> Value {
    use crate::glm5_next_metal::tests::{argmax, choice_regret};
    assert!(prefill_rows > 0);
    assert_eq!(a.len(), STEPS + 1);
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), exact.len());
    let mut kls = Vec::new();
    let mut changes = 0;
    let mut rows = Vec::new();
    for (step, ((a, b), e)) in a.iter().zip(b).zip(exact).enumerate() {
        let kl = kl_divergence(a, b);
        let (a_top, b_top, e_top) = (argmax(a), argmax(b), argmax(e));
        let a_regret = choice_regret(e, a).0;
        let b_regret = choice_regret(e, b).0;
        changes += usize::from(a_top != b_top);
        kls.push(kl);
        rows.push(json!({"position":prefill_rows-1+step,"a_top1":a_top,"b_top1":b_top,"exact_top1":e_top,
            "top1_changed":a_top!=b_top,"exact_side_regret_a":a_regret,"exact_side_regret_b":b_regret}));
    }
    json!({"kl_a_to_b_by_step":kls,"top1_changes":changes,"rows":rows})
}

fn precision_probe_token_logprobs(
    prefill_rows: usize,
    logits: &[Vec<f32>],
    continuation: &[u32],
) -> Value {
    assert!(prefill_rows > 0);
    assert_eq!(logits.len(), STEPS + 1);
    assert_eq!(continuation.len(), STEPS);
    // Row 0 predicts continuation[0]. The fourth row has no scored target:
    // the probe teacher-forces exactly three common tokens, never greedy ones.
    let rows: Vec<_> = continuation.iter().enumerate().map(|(step, &token)| {
        let row = &logits[step];
        assert!((token as usize) < row.len());
        assert!(!row.is_empty() && row.iter().all(|v| v.is_finite()));
        let max = row.iter().map(|&v| f64::from(v)).fold(f64::NEG_INFINITY, f64::max);
        let log_z = row.iter().map(|&v| (f64::from(v)-max).exp()).sum::<f64>().ln();
        let log_prob = f64::from(row[token as usize])-max-log_z;
        assert!(log_prob.is_finite());
        json!({"token_position":prefill_rows+step,"conditioning_position":prefill_rows-1+step,"token":token,"log_probability":log_prob})
    }).collect();
    json!(rows)
}

#[test]
fn precision_probe_metrics_follow_prefill_width() {
    let logits = vec![vec![0.25, -0.5, 1.0]; STEPS + 1];
    let continuation = [2, 0, 1];
    for rows in [512, 2048] {
        let comparison = precision_probe_comparison(rows, &logits, &logits, &logits);
        let scores = precision_probe_token_logprobs(rows, &logits, &continuation);
        for step in 0..=STEPS {
            assert_eq!(comparison["rows"][step]["position"], json!(rows - 1 + step));
        }
        assert_eq!(scores.as_array().unwrap().len(), STEPS);
        for (step, token) in continuation.iter().enumerate() {
            assert_eq!(scores[step]["token_position"], json!(rows + step));
            assert_eq!(
                scores[step]["conditioning_position"],
                json!(rows - 1 + step)
            );
            assert_eq!(scores[step]["token"], json!(token));
        }
    }
}

/// One-prompt precision interaction probe, not a quality cohort or routing
/// causality proof. It contrasts the Q6 scalar->MMA effect under surrounding
/// Half versus F32 precision/dispatch, using forced-scalar native Exact as a
/// third numerical reference. No numerical pass threshold or performance gate.
/// GLM53_LATENT_PRECISION_ROWS selects 512 (default) or 2048 rows.
/// Optional GLM53_LATENT_PRECISION_OUT is a new durable JSON path.
#[test]
#[ignore = "bounded Maya precision probe: GLM53_GGUF, release binary, no MTL_DEBUG_LAYER, production lease; GLM53_LATENT_PRECISION_ROWS=512 (default) or 2048; optional new GLM53_LATENT_PRECISION_OUT"]
fn grouped_latent_precision_probe() {
    assert!(
        !cfg!(debug_assertions),
        "timing requires a release test binary"
    );
    let rows = std::env::var_os("GLM53_LATENT_PRECISION_ROWS")
        .map(|v| {
            v.to_str()
                .and_then(|s| s.parse::<usize>().ok())
                .expect("GLM53_LATENT_PRECISION_ROWS must be 512 or 2048")
        })
        .unwrap_or(512);
    let cell = Cell {
        name: match rows {
            512 => "precision_full512",
            2048 => "precision_full2048",
            _ => panic!("GLM53_LATENT_PRECISION_ROWS must be 512 or 2048"),
        },
        prefix: 0,
        rows,
        workspace: rows,
    };
    let _lease = perf_lease();
    let path = PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name Maya"));
    let mut output = std::env::var_os("GLM53_LATENT_PRECISION_OUT").map(|path| {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("GLM53_LATENT_PRECISION_OUT must be a new writable JSON path")
    });
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).expect("Maya GGUF");
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    assert!(cell.capacity() <= artifact.model().config.context_length as usize);
    preflight_session(
        &ctx,
        &gguf,
        artifact.model(),
        cell.capacity(),
        cell.workspace,
    )
    .unwrap();
    // Identical prefix/tokenization to the matching full cell in grouped_latent_mma_ab.
    let text = long_qualification_text();
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{text}\n\n{text}"), false)
        .unwrap()
        .into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .take(cell.capacity())
        .collect();
    assert_eq!(tokens.len(), cell.capacity());
    let continuation = &tokens[cell.end()..cell.capacity()];
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("one Maya weight load");
    let groups = u64::from(weights.config.head_count);
    assert!(groups > 1);
    let mut q6_calls = 0;
    for b in &weights.blocks {
        let MixerTensors::Mla(mla) = &b.mixer else {
            continue;
        };
        for bank in [&mla.key_absorb, &mla.value_expand] {
            if bank.dtype == GgmlType::Q6_K {
                assert!(bank.shape[0].is_multiple_of(256) && bank.shape[1].is_multiple_of(16));
                q6_calls += 1;
            }
        }
    }
    assert!(
        q6_calls > 0,
        "probe must exercise Maya Q6 grouped latent substitution"
    );
    let vocab = u64::from(weights.config.vocab_size);
    // Five retained trajectories plus the current run, one readback row,
    // two F64 KL vectors, and census/report overhead. Exact is not traced.
    let cpu_reserve = (6 * (STEPS + 1) + 1) as u64 * vocab * 4 + 2 * vocab * 8 + (4 << 20);
    let run = |lineage, precision, baseline, capture: bool, timed: bool| {
        assert!(!(capture && timed), "timed runs must not trace");
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            cell.capacity(),
            cell.workspace,
            cpu_reserve,
        )
        .unwrap();
        session.set_packed_lineage(lineage).unwrap();
        session.set_fast_precision(precision).unwrap();
        assert_eq!(session.packed_lineage(), Some(lineage));
        assert_eq!(session.fast_precision(), Some(precision));
        with_grouped_projection_scalar_baseline_override(baseline, || {
            if capture {
                dispatch_census_begin();
            }
            let started = timed.then(std::time::Instant::now);
            let result = session.prefill_packed(&ctx, &tokens[..cell.rows]);
            let wall_ms = started.map(|t| t.elapsed().as_secs_f64() * 1e3);
            let census = capture.then(dispatch_census_take);
            let mut logits = vec![result.unwrap()];
            for &token in continuation {
                logits.push(session.forward(&ctx, token).unwrap());
            }
            assert_eq!(session.position(), cell.capacity());
            for row in &logits {
                assert_eq!(row.len(), vocab as usize);
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "non-finite full-vocabulary logits"
                );
            }
            let census =
                census.map(|rows| verify_q6_census(&rows, cell, baseline, groups, q6_calls));
            (wall_ms, logits, census)
        })
    };
    let mut report = json!({"schema":"glm53.grouped_latent_precision_probe.v1","model":path,
        "test_binary":std::env::current_exe().unwrap(),"prompt_tokens":&tokens[..cell.rows],
        "cell":cell.name,"teacher_forced_tokens":continuation,"prefill_rows":cell.rows,
        "workspace_rows":cell.workspace,"capacity":cell.capacity(),
        "cpu_reserve_bytes":cpu_reserve,"q6_calls_per_chunk":q6_calls,
        "purpose":"test precision-dependent downstream amplification of Q6 scalar->MMA; surrounding precision also changes eligible dispatches; not routing-causality proof",
        "scope":"one natural prompt, not a quality cohort; neither old Fast nor Exact is assumed quality truth",
        "arms":{"E":"native Exact, forced scalar Q6","HalfA":"Fast Half, scalar Q6",
            "HalfB":"Fast Half, Q6 MMA","F32A":"Fast F32, scalar Q6","F32B":"Fast F32, Q6 MMA"},
        "regret_definition":"Exact best logit minus Exact logit of the arm's top-1 choice",
        "token_logprob_alignment":"first three logits rows predict the three teacher-forced tokens; final row has no scored target",
        "timing":format!("F32 A/B warmups, then untraced A1 B1 B2 A2; ordinary full{} prefill wall including wait/readback; allocation, continuation, diagnostics excluded", cell.rows),
        "observations":{},"within_precision":{},"f32_warmups":[],"f32_abba":[],"numeric_pass_bound":null,"performance_assertion":false});
    save(&mut output, &report);
    let mut trajectories: std::collections::BTreeMap<&str, Vec<Vec<f32>>> = Default::default();
    for (label, lineage, precision, baseline) in [
        ("E", PackedLineage::Exact, FastPrecision::Half, true),
        ("HalfA", PackedLineage::Fast, FastPrecision::Half, true),
        ("HalfB", PackedLineage::Fast, FastPrecision::Half, false),
    ] {
        let (_, logits, census) = run(lineage, precision, baseline, label != "E", false);
        let exact = trajectories.get("E").unwrap_or(&logits);
        let comparison = precision_probe_comparison(cell.rows, exact, exact, &logits);
        let logprobs = precision_probe_token_logprobs(cell.rows, &logits, continuation);
        eprintln!(
            "latent precision full{rows} {label}: E||arm {comparison}; continuation logprobs {logprobs}"
        );
        report["observations"][label] = json!({"exact_to_arm":comparison,
            "continuation_logprobs":logprobs,"q6_census":census});
        trajectories.insert(label, logits);
        save(&mut output, &report);
    }
    for (label, baseline) in [("F32A", true), ("F32B", false)] {
        let (wall, _, census) = run(
            PackedLineage::Fast,
            FastPrecision::F32,
            baseline,
            true,
            false,
        );
        assert!(wall.is_none());
        eprintln!("latent precision full{rows} warmup {label}: Q6 dispatch verified");
        report["f32_warmups"]
            .as_array_mut()
            .unwrap()
            .push(json!({"arm":label,"q6_census":census}));
        save(&mut output, &report);
    }
    let mut times = [Vec::new(), Vec::new()];
    for (label, arm, baseline) in [
        ("A1", "F32A", true),
        ("B1", "F32B", false),
        ("B2", "F32B", false),
        ("A2", "F32A", true),
    ] {
        let (wall, logits, census) = run(
            PackedLineage::Fast,
            FastPrecision::F32,
            baseline,
            false,
            true,
        );
        assert!(census.is_none());
        let ms = wall.unwrap();
        let exact = &trajectories["E"];
        let a1 = trajectories.get("F32A").unwrap_or(&logits);
        let vs_a1 = precision_probe_comparison(cell.rows, exact, a1, &logits);
        let vs_exact = precision_probe_comparison(cell.rows, exact, exact, &logits);
        let logprobs = precision_probe_token_logprobs(cell.rows, &logits, continuation);
        eprintln!(
            "latent precision full{rows} F32 {label}: {ms:.2} ms, {:.3} tokens/s, A1||run {vs_a1}",
            cell.rows as f64 * 1000.0 / ms
        );
        times[usize::from(!baseline)].push(ms);
        report["f32_abba"].as_array_mut().unwrap().push(json!({"label":label,"arm":arm,"prefill_wall_ms":ms,
            "tokens_per_second":cell.rows as f64*1000.0/ms,"a1_to_run":vs_a1,"exact_to_run":vs_exact,
            "continuation_logprobs":logprobs}));
        if !trajectories.contains_key(arm) {
            eprintln!(
                "latent precision full{rows} {arm}: E||arm {vs_exact}; continuation logprobs {logprobs}"
            );
            report["observations"][arm] = json!({"source_run":label,"exact_to_arm":vs_exact,"continuation_logprobs":logprobs});
            trajectories.insert(arm, logits);
        }
        save(&mut output, &report);
    }
    let exact = &trajectories["E"];
    for (precision, a, b) in [("Half", "HalfA", "HalfB"), ("F32", "F32A", "F32B")] {
        let comparison =
            precision_probe_comparison(cell.rows, exact, &trajectories[a], &trajectories[b]);
        eprintln!("latent precision full{rows} {precision} scalar||MMA: {comparison}");
        report["within_precision"][precision] = comparison;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (a_ms, b_ms) = (mean(&times[0]), mean(&times[1]));
    eprintln!(
        "latent precision full{rows} warmed F32 ABBA: scalar {a_ms:.2} ms, MMA {b_ms:.2} ms, A/B {:.3}x; no pass bound",
        a_ms / b_ms
    );
    report["f32_timing_summary"] = json!({"a_mean_ms":a_ms,"b_mean_ms":b_ms,"a_over_b_wall_ratio":a_ms/b_ms,
        "a_tokens_per_second":cell.rows as f64*1000.0/a_ms,"b_tokens_per_second":cell.rows as f64*1000.0/b_ms});
    save(&mut output, &report);
}

use super::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

const ORACLE_DEFAULT: &str =
    "/Volumes/wdblack/weights-archive/.fetch/analysis/runs/glm53-oracle/ckpt-v1";

fn oracle_dir() -> PathBuf {
    std::env::var_os("GLM53_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(ORACLE_DEFAULT))
}

fn checkpoint_tokens() -> Vec<u32> {
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../scripts/reference/glm53/ckpt-v1.json"
    ))
    .unwrap();
    manifest["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect()
}

/// GLMREF01: per position, (token, logits).
fn read_reference_logits(path: &std::path::Path) -> Vec<(u32, Vec<f32>)> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"GLMREF01");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (vocab, count) = (u32_at(8) as usize, u32_at(12) as usize);
    let mut offset = 16;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let token = u32_at(offset + 4);
        offset += 8;
        let logits = bytes[offset..offset + vocab * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        offset += vocab * 4;
        out.push((token, logits));
    }
    out
}

/// GLMCAP01 F32 (and I32, as exact F32) records keyed by (name, step,
/// layer, occurrence).
fn read_captures(
    path: &std::path::Path,
    wanted: &[&str],
) -> BTreeMap<(String, u32, u32, u32), Vec<f32>> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"GLMCAP01");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let i64_at = |o: usize| i64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let mut offset = 8;
    let mut out = BTreeMap::new();
    while offset < bytes.len() {
        let (step, layer, occurrence, name_len) = (
            u32_at(offset),
            u32_at(offset + 4),
            u32_at(offset + 8),
            u32_at(offset + 12) as usize,
        );
        offset += 16;
        let name = String::from_utf8(bytes[offset..offset + name_len].to_vec()).unwrap();
        offset += name_len;
        let dtype = u32_at(offset);
        offset += 4 + 32;
        let n = i64_at(offset) as usize;
        offset += 8;
        if dtype == 26 && wanted.contains(&name.as_str()) {
            // I32 records (selected ids) are small integers, exact in F32.
            let values = bytes[offset..offset + n]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b) as f32)
                .collect();
            out.insert((name, step, layer, occurrence), values);
        } else if dtype == 0 && wanted.contains(&name.as_str()) {
            let values = bytes[offset..offset + n]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            out.insert((name, step, layer, occurrence), values);
        }
        offset += n;
    }
    out
}

/// Panics unless both sides are nonempty, equally long and finite, so a
/// comparison can never pass vacuously or on NaN.
fn assert_comparable(what: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    assert!(!expected.is_empty(), "{what}: empty comparison");
    assert_finite(&format!("{what} (actual)"), actual);
    assert_finite(&format!("{what} (expected)"), expected);
}

/// `|actual - expected| / |expected|` (L2); finite by construction.
fn relative_error(actual: &[f32], expected: &[f32]) -> f64 {
    assert_comparable("relative error", actual, expected);
    let diff: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| ((a - e) as f64).powi(2))
        .sum();
    let norm: f64 = expected.iter().map(|e| (*e as f64).powi(2)).sum();
    (diff / norm.max(1e-30)).sqrt()
}

/// KL(reference || native) over softmaxed logits; finite by construction.
fn kl_divergence(reference: &[f32], native: &[f32]) -> f64 {
    assert_comparable("KL", native, reference);
    let log_softmax = |x: &[f32]| {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = x.iter().map(|v| (*v as f64 - max).exp()).sum();
        x.iter()
            .map(move |v| *v as f64 - max - sum.ln())
            .collect::<Vec<_>>()
    };
    let (p, q) = (log_softmax(reference), log_softmax(native));
    let kl: f64 = p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum();
    assert!(kl.is_finite(), "KL is not finite");
    kl
}

/// Largest elementwise |actual - expected|; finite by construction.
fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_comparable("max |diff|", actual, expected);
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max)
}

/// Largest value and its index; NaN is the largest of all, so a reduction
/// can never hide one.
fn worst(values: impl IntoIterator<Item = f64>) -> (f64, usize) {
    values
        .into_iter()
        .enumerate()
        .fold((0.0f64, 0usize), |best, (i, v)| {
            if v.is_nan() || best.0.is_nan() {
                if best.0.is_nan() { best } else { (v, i) }
            } else if v > best.0 {
                (v, i)
            } else {
                best
            }
        })
}

/// Whether `value` is within `bound`; NaN never is.
fn within(value: f64, bound: f64) -> bool {
    value <= bound
}

/// Logit regret of each side's top-1 choice under the other side's logits:
/// (reference best - reference logit of native's choice, native best -
/// native logit of reference's choice). Both are zero when top-1 agrees; a
/// flip is a near-tie only if both are small.
fn choice_regret(reference: &[f32], native: &[f32]) -> (f32, f32) {
    assert_comparable("choice regret", native, reference);
    let (r, n) = (argmax(reference), argmax(native));
    (reference[r] - reference[n], native[n] - native[r])
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

fn assert_finite(label: &str, values: &[f32]) {
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{label}: non-finite values"
    );
}

/// Compares native recurrent and indexer state after `position` with the
/// llama.cpp state capture (KDA `new_state`; the first complete pools of
/// `indexer_pool_k`; the pending key of the current position vs `indexer_k`).
fn compare_state(
    session: &Glm5NextSession<'_>,
    position: usize,
    state_captures: &BTreeMap<(String, u32, u32, u32), Vec<f32>>,
    default_captures: &BTreeMap<(String, u32, u32, u32), Vec<f32>>,
) {
    let step = position as u32;
    let complete_pools = (position + 1) / 4;
    for (layer, state) in session.layers.iter().enumerate() {
        let key = |name: &str| (name.to_string(), step, layer as u32, 0);
        match state {
            LayerState::Kda { state, .. } => {
                let native = read_f32(state).unwrap();
                let reference = state_captures
                    .get(&key("new_state"))
                    .unwrap_or_else(|| panic!("missing new_state step {step} layer {layer}"));
                assert_finite("native KDA state", &native);
                let error = relative_error(&native, reference);
                assert!(
                    error < 1e-3,
                    "step {step} layer {layer} KDA state rel {error:.3e}"
                );
            }
            LayerState::Mla {
                pending, pooled, ..
            } => {
                let reference = state_captures
                    .get(&key("indexer_pool_k"))
                    .unwrap_or_else(|| panic!("missing indexer_pool_k step {step} layer {layer}"));
                let native = read_f16(pooled).unwrap();
                let rows = complete_pools * 128;
                let error = relative_error(&native[..rows], &reference[..rows]);
                assert!(
                    error < 2e-3,
                    "step {step} layer {layer} pooled rel {error:.3e}"
                );
                let slot = position % 4;
                let pending = read_f16(pending).unwrap();
                let key_row = &pending[slot * 256..slot * 256 + 128];
                let reference_key = default_captures
                    .get(&key("indexer_k"))
                    .unwrap_or_else(|| panic!("missing indexer_k step {step} layer {layer}"));
                let error = relative_error(key_row, reference_key);
                assert!(
                    error < 2e-3,
                    "step {step} layer {layer} pending key rel {error:.3e}"
                );
            }
        }
    }
}

/// Model-scale GPU tests hold the cross-process production lease and pass
/// the wired-memory gate (unit-test `MetalContext`s only take a per-process
/// lock), and run with Metal API validation. Bind the result first so it
/// drops after every Metal resource of the test.
fn production_lease() -> impl Sized {
    assert_eq!(
        std::env::var("MTL_DEBUG_LAYER").as_deref(),
        Ok("1"),
        "model-scale GLM GPU tests require MTL_DEBUG_LAYER=1"
    );
    crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease")
}

/// Bit patterns of every layer's persistent state (KDA conv and S; MLA
/// latents, pending ring and pools).
fn state_bits(session: &Glm5NextSession<'_>) -> Vec<Vec<u32>> {
    let f32_bits = |t| read_f32(t).unwrap().iter().map(|v| v.to_bits()).collect();
    let f16_bits = |t| read_f16(t).unwrap().iter().map(|v| v.to_bits()).collect();
    session
        .layers
        .iter()
        .flat_map(|layer| match layer {
            LayerState::Kda { conv, state } => vec![f32_bits(conv), f32_bits(state)],
            LayerState::Mla {
                latent,
                pending,
                pooled,
            } => vec![f16_bits(latent), f16_bits(pending), f16_bits(pooled)],
        })
        .collect()
}

/// The P2 end-to-end checkpoint: the ckpt-v1 token IDs through all 45 blocks,
/// compared with the same-artifact llama.cpp oracle: logits per position, every
/// block's residual streams, KDA state and indexer pools at steps 3/7/14, and
/// unobserved-path parity; reports peak allocation and token latency.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the ckpt-v1 oracle and an idle GPU"]
fn checkpoint_v1_matches_llama_cpp_oracle() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let load = std::time::Instant::now();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    eprintln!(
        "weights: {} bytes in {:.1}s",
        weights.retained_bytes,
        load.elapsed().as_secs_f64()
    );
    let tokens = checkpoint_tokens();
    let reference = read_reference_logits(&oracle_dir().join("logits.bin"));
    let captures = read_captures(
        &oracle_dir().join("default.bin.captures"),
        &["l_out", "hc_attn_post", "indexer_k"],
    );
    let state_captures = read_captures(
        &oracle_dir().join("state.bin.captures"),
        &["new_state", "indexer_pool_k"],
    );
    assert_eq!(reference.len(), tokens.len());
    let blocks = weights.blocks.len();
    let mut observed_logits = Vec::with_capacity(tokens.len());
    {
        let mut session = Glm5NextSession::new(&ctx, &weights, 256).expect("session");
        eprintln!(
            "allocated with one session: {} bytes (ledger session peak {})",
            ctx.current_allocated_size(),
            session.ledger().phase_peaks().session,
        );
        assert_allocation_within_ledger(&session);
        let mut worst_kl = 0.0f64;
        for (position, &token) in tokens.iter().enumerate() {
            assert_eq!(reference[position].0, token);
            let mut residuals = Vec::with_capacity(2 * blocks);
            let logits = session
                .forward_observed(
                    &ctx,
                    token,
                    &[
                        Glm5NextProbe::AttentionResidual,
                        Glm5NextProbe::BlockResidual,
                    ],
                    &mut |probe, block, values| {
                        let name = match probe {
                            Glm5NextProbe::AttentionResidual => "hc_attn_post",
                            Glm5NextProbe::BlockResidual => "l_out",
                        };
                        let expected = captures
                            .get(&(name.to_string(), position as u32, block as u32, 0))
                            .unwrap_or_else(|| {
                                panic!("missing {name} capture pos {position} block {block}")
                            });
                        assert_finite(name, values);
                        residuals.push((name, block, relative_error(values, expected)));
                    },
                )
                .unwrap_or_else(|e| panic!("position {position}: {e}"));
            assert_eq!(
                residuals.len(),
                2 * blocks,
                "position {position}: residual probes"
            );
            let expected = &reference[position].1;
            assert_finite("native logits", &logits);
            assert_eq!(logits.len(), expected.len());
            let kl = kl_divergence(expected, &logits);
            assert!(kl.is_finite(), "position {position}: KL {kl}");
            let max_abs = max_abs_diff(&logits, expected);
            let worst = residuals
                .iter()
                .cloned()
                .fold(("", 0, 0.0f64), |w, e| if e.2 > w.2 { e } else { w });
            eprintln!(
                "pos {position:2} token {token:6}: kl {kl:.3e} max|dlogit| {max_abs:.3e} worst residual {}@{} {:.3e}",
                worst.0, worst.1, worst.2
            );
            assert_eq!(
                argmax(&logits),
                argmax(expected),
                "position {position}: top-1"
            );
            assert!(kl < 1e-6, "position {position}: KL {kl}");
            assert!(
                max_abs < 2e-3,
                "position {position}: max |dlogit| {max_abs}"
            );
            assert!(
                worst.2 < 1e-4,
                "position {position}: residual {}@{} {}",
                worst.0,
                worst.1,
                worst.2
            );
            worst_kl = worst_kl.max(kl);
            if matches!(position, 3 | 7 | 14) {
                compare_state(&session, position, &state_captures, &captures);
                eprintln!("pos {position:2}: KDA state, pools and pending key match");
            }
            observed_logits.push(logits);
        }
        eprintln!(
            "observed path: top-1 {}/{} worst KL {worst_kl:.3e}",
            tokens.len(),
            tokens.len()
        );
    }
    // Unobserved path: one command per token, same logits; latency.
    let mut session = Glm5NextSession::new(&ctx, &weights, 256).expect("timing session");
    let start = std::time::Instant::now();
    for (position, &token) in tokens.iter().enumerate() {
        let logits = session.forward(&ctx, token).unwrap();
        let drift = max_abs_diff(&logits, &observed_logits[position]);
        assert!(
            drift <= 1e-6,
            "position {position}: unobserved drift {drift}"
        );
    }
    eprintln!(
        "unobserved: {:.1} ms/token over {} tokens; allocated {} bytes",
        start.elapsed().as_secs_f64() * 1e3 / tokens.len() as f64,
        tokens.len(),
        ctx.current_allocated_size()
    );
}

/// Packed prefill against the same oracle. Exact lineage (decode kernels per
/// row) must reproduce the serial session: one 15-row chunk, 4-row chunks
/// (pools and recurrent state continue across chunks), and packed prefill of
/// 11 tokens followed by serial decode. Fast lineage (half-staged batched
/// projections and grouped experts, as in llama.cpp's batched prefill) must
/// keep top-1 and stay within a frozen KL and state envelope.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the ckpt-v1 oracle and an idle GPU"]
fn packed_prefill_matches_serial_and_oracle_on_checkpoint_v1() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokens = checkpoint_tokens();
    let reference = read_reference_logits(&oracle_dir().join("logits.bin"));
    let mut serial = Glm5NextSession::new(&ctx, &weights, 256).expect("serial session");
    let mut serial_logits = Vec::new();
    for &token in &tokens {
        serial_logits.push(serial.forward(&ctx, token).unwrap());
    }
    let check = |label: &str, position: usize, logits: &[f32], lineage: PackedLineage| {
        assert_finite(label, logits);
        let expected = &reference[position].1;
        let kl = kl_divergence(expected, logits);
        let drift = max_abs_diff(logits, &serial_logits[position]);
        eprintln!(
            "{label} pos {position:2}: kl vs llama.cpp {kl:.3e} max|dlogit| vs serial {drift:.3e}"
        );
        assert_eq!(
            argmax(logits),
            argmax(expected),
            "{label} pos {position}: top-1"
        );
        match lineage {
            PackedLineage::Exact => assert!(drift <= 1e-6, "{label} pos {position}: drift {drift}"),
            PackedLineage::Fast => assert!(
                kl.is_finite() && kl < 1e-3,
                "{label} pos {position}: KL {kl}"
            ),
        }
    };
    // Fast envelope: llama.cpp's own batched prefill on this artifact differs
    // from its serial decode by KL up to 1.7e-2 and max |dlogit| up to 1.56
    // (ckpt-v1, glm53_oracle --batch); Fast must stay tighter on logits. KDA
    // state amplifies input rounding (block 18 turns 5e-4 input error into
    // 2e-2 state error while Exact, same kernel, is bitwise), so the Fast
    // state bound only catches gross faults.
    let state_bound = |lineage| match lineage {
        PackedLineage::Exact => 1e-6,
        PackedLineage::Fast => 2.5e-1,
    };
    for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
        for rows in [16usize, 4] {
            let mut packed = Glm5NextSession::with_prefill_rows(&ctx, &weights, 256, rows).unwrap();
            packed.set_packed_lineage(lineage);
            let start = std::time::Instant::now();
            let logits = packed.prefill_packed(&ctx, &tokens).unwrap();
            eprintln!(
                "{lineage:?} rows {rows}: {:.1} ms for {} tokens",
                start.elapsed().as_secs_f64() * 1e3,
                tokens.len()
            );
            check(&format!("{lineage:?} rows {rows}"), 14, &logits, lineage);
            assert_eq!(packed.position(), tokens.len());
            let bound = state_bound(lineage);
            for (layer, (a, b)) in packed.layers.iter().zip(&serial.layers).enumerate() {
                let error = match (a, b) {
                    (
                        LayerState::Kda {
                            state: sa,
                            conv: ca,
                        },
                        LayerState::Kda {
                            state: sb,
                            conv: cb,
                        },
                    ) => {
                        let es = relative_error(&read_f32(sa).unwrap(), &read_f32(sb).unwrap());
                        let ec = relative_error(&read_f32(ca).unwrap(), &read_f32(cb).unwrap());
                        if lineage == PackedLineage::Fast && rows == 16 {
                            eprintln!("  layer {layer:2}: kda state {es:.3e} conv {ec:.3e}");
                        }
                        es.max(ec)
                    }
                    (
                        LayerState::Mla {
                            pooled: pa,
                            latent: la,
                            ..
                        },
                        LayerState::Mla {
                            pooled: pb,
                            latent: lb,
                            ..
                        },
                    ) => relative_error(
                        &read_f16(pa).unwrap()[..3 * 128],
                        &read_f16(pb).unwrap()[..3 * 128],
                    )
                    .max(relative_error(
                        &read_f16(la).unwrap()[..15 * 512],
                        &read_f16(lb).unwrap()[..15 * 512],
                    )),
                    _ => unreachable!("layer kinds match"),
                };
                assert!(
                    error <= bound,
                    "{lineage:?} rows {rows} layer {layer}: state {error:.3e}"
                );
            }
        }
        let mut mixed = Glm5NextSession::with_prefill_rows(&ctx, &weights, 256, 8).unwrap();
        mixed.set_packed_lineage(lineage);
        let logits = mixed.prefill_packed(&ctx, &tokens[..11]).unwrap();
        check(&format!("{lineage:?} mixed prefill"), 10, &logits, lineage);
        // Whole-request refusals execute nothing, and decode continues as if
        // the requests never happened (Exact: bitwise against serial below).
        // With 8-row chunks, "bad token after a full chunk" is the case a
        // per-chunk check would have half-executed.
        let vocab = weights.config.vocab_size;
        let overrun = vec![tokens[11]; 256 - 11 + 1];
        let mut after_chunk = vec![tokens[11]; 8];
        after_chunk.push(vocab);
        for (label, request) in [
            ("empty", &[][..]),
            ("bad tail token", &[tokens[11], tokens[12], vocab][..]),
            ("bad head token", &[vocab, tokens[11]][..]),
            ("bad token after a full chunk", &after_chunk[..]),
            ("capacity overrun", &overrun[..]),
        ] {
            assert_refused(&mut mixed, &format!("packed {label}"), |s| {
                s.prefill_packed(&ctx, request)
            });
            assert_refused(&mut mixed, &format!("serial {label}"), |s| {
                s.prefill(&ctx, request)
            });
        }
        assert_refused(&mut mixed, "forward bad token", |s| s.forward(&ctx, vocab));
        assert_refused(&mut mixed, "advance bad token", |s| s.advance(&ctx, vocab));
        for (position, &token) in tokens.iter().enumerate().skip(11) {
            let logits = mixed.forward(&ctx, token).unwrap();
            check(
                &format!("{lineage:?} mixed decode"),
                position,
                &logits,
                lineage,
            );
        }
        // Exactly full: a 12-position session takes 12 tokens, then refuses
        // every further token on every entry point.
        let mut full = Glm5NextSession::with_prefill_rows(&ctx, &weights, 12, 8).unwrap();
        full.set_packed_lineage(lineage);
        let logits = full.prefill_packed(&ctx, &tokens[..12]).unwrap();
        check(&format!("{lineage:?} full prefill"), 11, &logits, lineage);
        assert_eq!(full.position(), 12);
        let next = &tokens[12..13];
        assert_refused(&mut full, "full forward", |s| s.forward(&ctx, next[0]));
        assert_refused(&mut full, "full advance", |s| s.advance(&ctx, next[0]));
        assert_refused(&mut full, "full prefill", |s| s.prefill(&ctx, next));
        assert_refused(&mut full, "full packed", |s| s.prefill_packed(&ctx, next));
    }
    // Without packed scratch, prefill_packed falls back to the serial path,
    // which refuses the whole request too.
    let vocab = weights.config.vocab_size;
    assert_refused(&mut serial, "fallback bad token", |s| {
        s.prefill_packed(&ctx, &[tokens[0], tokens[1], vocab])
    });
}

/// In this isolated test process nothing else allocates or frees Metal memory
/// while a session is built, so the device-counter delta is the session's own
/// allocation and must fit the device-priced ledger.
fn assert_allocation_within_ledger(session: &Glm5NextSession<'_>) {
    let (observed, priced) = (
        session.observed_allocation_delta(),
        session.ledger().session_buffer_bytes(),
    );
    eprintln!(
        "session buffers: observed {observed} of {priced} priced ({:.1}%)",
        100.0 * observed as f64 / priced as f64
    );
    assert!(observed <= priced, "observed {observed} > priced {priced}");
}

/// `request` is refused and leaves the position, the poison flag and every
/// state bit unchanged.
fn assert_refused<'w, T>(
    session: &mut Glm5NextSession<'w>,
    label: &str,
    request: impl FnOnce(&mut Glm5NextSession<'w>) -> Result<T>,
) {
    let (position, before) = (session.position(), state_bits(session));
    assert!(request(session).is_err(), "{label} was accepted");
    assert_eq!(session.position(), position, "{label}: position moved");
    assert!(!session.poisoned, "{label}: refusal poisoned the session");
    assert!(state_bits(session) == before, "{label}: state changed");
}

/// Fast packed prefill over 200 tokens (one 128-row grouped block plus a
/// 72-row tail) against exact lineage, which equals serial decode: last-row
/// logits within the fast envelope, then three decode steps on each.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn packed_fast_matches_exact_over_a_grouped_block_and_tail() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let text = "[gMASK]<sop>The history of maritime trade begins with coastal exchange between \
        neighbouring settlements, long before ocean crossings were possible. Early sailors read the \
        winds, the stars and the colour of the water; they carried grain, timber, metal and stories. \
        Over centuries, harbours grew into cities, and the routes between them shaped languages, \
        laws and the spread of ideas. Merchants learned to keep accounts, insure cargo and trust \
        partners they had never met, and the ships themselves changed from rafts and dugouts to \
        planked hulls with keels, sails and rudders capable of crossing open seas in every season.";
    let text = format!("{text} {}", text.trim_start_matches("[gMASK]<sop>"));
    let tokens: Vec<u32> = tokenizer
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    assert!(
        tokens.len() >= 129,
        "need a full 128-row block, got {}",
        tokens.len()
    );
    let tokens = &tokens[..tokens.len().min(200)];
    let mut sessions = Vec::new();
    for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
        let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 512, 512).unwrap();
        assert_allocation_within_ledger(&session);
        session.set_packed_lineage(lineage);
        let start = std::time::Instant::now();
        let logits = session.prefill_packed(&ctx, tokens).unwrap();
        eprintln!(
            "{lineage:?}: {} tokens in {:.1} ms",
            tokens.len(),
            start.elapsed().as_secs_f64() * 1e3
        );
        sessions.push((session, logits));
    }
    let (exact, fast) = (&sessions[0].1, &sessions[1].1);
    assert_finite("fast logits", fast);
    let kl = kl_divergence(exact, fast);
    eprintln!("fast vs exact after {} tokens: kl {kl:.3e}", tokens.len());
    assert_eq!(argmax(fast), argmax(exact), "prefill top-1");
    assert!(kl < 1e-3, "prefill KL {kl}");
    let mut next = argmax(exact) as u32;
    for step in 0..3 {
        let a = sessions[0].0.forward(&ctx, next).unwrap();
        let b = sessions[1].0.forward(&ctx, next).unwrap();
        let kl = kl_divergence(&a, &b);
        eprintln!("decode step {step}: kl {kl:.3e}");
        assert!(kl < 1e-3, "decode step {step}: KL {kl}");
        next = argmax(&a) as u32;
    }
}

/// Eight unrelated passages (over 591 tokens) for long-prompt qualification.
const QUALIFICATION_TEXT: &str = "[gMASK]<sop>The history of maritime trade begins with \
    coastal exchange between neighbouring settlements, long before ocean crossings were possible. \
    Early sailors read the winds, the stars and the colour of the water; they carried grain, timber, \
    metal and stories. Over centuries, harbours grew into cities, and the routes between them shaped \
    languages, laws and the spread of ideas. Merchants learned to keep accounts, insure cargo and \
    trust partners they had never met.\n\nGlaciers form where more snow falls in winter than melts in \
    summer. Over decades the snow compacts into firn and then into dense blue ice, which begins to \
    flow under its own weight. A valley glacier moves only a few metres a year near its edges but \
    faster at its centre, carving the bedrock into a broad U shape and carrying boulders far from \
    their source. When the climate warms, the ice front retreats and leaves behind moraines, lakes \
    and polished rock that record its former reach. Scientists drill cores through ancient ice \
    sheets to read bubbles of trapped air.\n\nBaking bread begins with flour, water, salt and yeast. \
    Mixing hydrates the flour and lets gluten proteins link into an elastic network that traps the \
    gas produced by fermentation. During a slow rise the yeast and bacteria develop flavour, and \
    the dough doubles in volume. The baker shapes the loaf to build surface tension, lets it proof \
    once more and scores the top so it can expand evenly. In a hot oven the crust browns and the \
    crumb sets as starches gelatinise.\n\nLong before telescopes, observers tracked the planets \
    against the fixed stars and noticed that some of them occasionally moved backwards. Kepler \
    showed that planets travel on ellipses with the Sun at one focus, sweeping out equal areas in \
    equal times, and Newton later derived these laws from a single theory of gravitation. Today \
    spacecraft use the same mathematics to swing around planets, gaining speed for journeys to the \
    outer solar system.\n\nA honeybee colony is a society of tens of thousands of workers, a few \
    hundred drones and a single queen. Foragers visit flowers to collect nectar and pollen, and when \
    they return they perform a waggle dance whose angle and duration tell their sisters the \
    direction and distance of the food. Inside the hive, younger bees build wax comb, feed larvae \
    and fan their wings until nectar thickens into honey.\n\nPrime numbers have fascinated \
    mathematicians since antiquity. Euclid proved that there are infinitely many of them with an \
    argument so short that it still appears in introductory courses. Yet their distribution remains \
    mysterious: primes thin out as numbers grow, roughly in proportion to the logarithm, but the \
    gaps between them vary irregularly. Modern cryptography relies on the fact that multiplying two \
    large primes is easy while recovering them from their product appears to be extremely hard.\n\n\
    Rivers shape the land they cross. In mountains a young stream cuts a steep valley and carries \
    gravel downhill during spring floods. Lower down, the gradient eases, the water slows and the \
    river begins to meander, eroding the outside of each bend and depositing sand on the inside. \
    Over time a loop may be cut off entirely, leaving a crescent-shaped lake beside the new channel. \
    Near the sea the river spreads its load into a delta, where fertile silt has supported farming \
    communities for thousands of years.\n\nClocks have grown steadily more precise. Sundials divided \
    daylight, water clocks measured the night, and mechanical escapements in medieval towers struck \
    the hours for whole cities. The pendulum, studied by Galileo and built into clocks by Huygens, \
    reduced errors from minutes to seconds per day. Marine chronometers later allowed navigators to \
    determine longitude at sea, and quartz crystals brought accurate time to every wrist. Atomic \
    clocks now count the oscillations of caesium atoms so steadily that they would drift by less \
    than a second over millions of years.";

/// Twelve more unrelated passages: with [`QUALIFICATION_TEXT`] over 2,100
/// tokens, for gates across the sparse frontier.
const SPARSE_EXTRA_PASSAGES: &[&str] = &[
    "Volcanoes form where molten rock reaches the surface. At mid-ocean ridges the plates \
        pull apart and basalt wells up quietly, building new sea floor at roughly the rate \
        fingernails grow. Where one plate dives beneath another, water carried down with it \
        lowers the melting point of the mantle, and sticky, gas-rich magma rises to feed \
        explosive cones. Hot spots such as the one beneath Hawaii stay roughly fixed while the \
        plate slides over them, leaving a chain of islands that grow older and lower with \
        distance. Ash from large eruptions can circle the globe and cool the climate for a \
        year or two.",
    "A violin is a carefully balanced box of spruce and maple. The arched top vibrates when \
        the strings are bowed, and a small post of wood wedged inside carries vibrations to \
        the back. Makers in northern Italy refined the shape centuries ago, choosing timber by \
        its stiffness and weight and varnishing it with recipes that are still debated. \
        Players change the sound by the speed, pressure and position of the bow, and by \
        pressing the strings against the fingerboard to shorten their vibrating length. A \
        well-made instrument responds evenly across its range and projects to the back of a \
        large hall.",
    "Coral reefs are built by tiny animals that live in partnership with algae. The algae \
        live inside the coral tissue and supply sugars from photosynthesis, while the coral \
        provides shelter and nutrients. Over thousands of years the animals deposit limestone \
        skeletons that accumulate into reefs large enough to see from space. Reefs shelter a \
        quarter of all marine species despite covering a tiny fraction of the ocean floor. \
        When water becomes too warm the coral expels its algae and turns white, and if the \
        heat persists the colony starves.",
    "The printing press transformed how ideas travelled. Before movable type, books were \
        copied by hand and a single volume could take months to finish. Casting individual \
        letters in metal allowed pages to be composed, printed, broken up and reused, so that \
        hundreds of identical copies could be made in the time a scribe needed for one. \
        Pamphlets, newspapers and scientific journals followed, and literacy spread as reading \
        material became cheaper. Standardised spelling and grammar emerged partly because \
        printers needed consistent rules for their compositors.",
    "Bridges carry loads in a few basic ways. A beam bridge bends, with its top in \
        compression and its bottom in tension. An arch pushes outward against its abutments \
        and keeps its stones squeezed together, which is why Roman arches still stand without \
        mortar. Suspension bridges hang the deck from cables draped over tall towers and \
        anchored in heavy blocks at each end, while cable-stayed bridges run straight cables \
        from the towers to the deck. Engineers must also account for wind, temperature changes \
        that make steel expand and contract, and the rhythm of traffic and footsteps.",
    "Migratory birds cover astonishing distances. Arctic terns fly from the far north to \
        the Antarctic and back each year, seeing more daylight than any other animal. Many \
        songbirds travel at night, navigating by the stars, by the earth's magnetic field and \
        by landmarks such as coastlines and mountain ranges. Before departure they eat heavily \
        and may nearly double their weight, then burn the fat as fuel during long flights over \
        open water or desert. Wetlands and forests along the way serve as refuelling stops, \
        and their loss can break a route that has been used for millennia.",
    "Tea began as a medicinal drink in China and became a daily habit across much of the \
        world. The leaves of a single species of camellia produce green, black, white and \
        oolong teas depending on how long they are allowed to oxidise before being heated and \
        dried. Trade in tea shaped empires: caravans carried compressed bricks across \
        mountains, and clipper ships raced to bring the first harvest of the season to \
        European ports. In many cultures preparing and serving tea is a ritual of hospitality, \
        with its own utensils, gestures and etiquette.",
    "Mapmakers have always faced the problem of flattening a round world. Every projection \
        distorts something: areas, angles, distances or directions. The familiar Mercator map \
        keeps compass bearings true, which made it invaluable to sailors, but it inflates \
        regions near the poles so that Greenland appears as large as Africa. Equal-area \
        projections correct the sizes at the cost of shapes. Modern satellite surveys and \
        global positioning have made measurements far more precise, yet choosing a projection \
        still depends on what the map is meant to show.",
    "Cheese is a way of preserving milk. Bacteria convert the milk sugar into acid, and an \
        enzyme called rennet makes the proteins clump into curds that separate from the watery \
        whey. Cutting, heating and pressing the curds controls how much moisture remains, \
        which determines whether the cheese will be soft and fresh or hard and suitable for \
        long ageing. During ripening, moulds and bacteria break down fats and proteins into \
        hundreds of flavour compounds. Caves with steady temperature and humidity have been \
        used for centuries to age the finest wheels.",
    "Lighthouses guided ships long before radio and satellites. Early towers burned wood or \
        coal fires at the top, later replaced by oil lamps with polished reflectors. The \
        invention of the Fresnel lens, with its rings of glass prisms, allowed a modest flame \
        to be seen more than twenty miles out to sea. Each lighthouse flashed its own pattern \
        so that sailors could identify it on a chart. Keepers lived in isolation, trimming \
        wicks, winding clockwork and recording the weather, until automation made the job \
        unnecessary in most places.",
    "Sleep is not simply a pause in activity. During the night the brain cycles through \
        stages of light and deep sleep and periods of rapid eye movement, when most vivid \
        dreaming occurs. Deep sleep appears to help consolidate memories and clear waste \
        products from brain tissue, while dreaming sleep may help process emotions. Hormones \
        that regulate appetite, growth and stress follow rhythms tied to the sleep cycle. Even \
        a few nights of short sleep can slow reaction times and impair judgement as much as \
        moderate alcohol consumption.",
    "Glassmaking turns sand into something transparent. Silica melts only at very high \
        temperatures, so ancient glassmakers added soda or potash to lower the melting point \
        and lime to make the result durable. Glassblowers gather a blob of molten glass on the \
        end of a hollow pipe and inflate it like a balloon, shaping it with tools and gravity \
        before it cools. Window glass was once made by spinning a disc or blowing a long \
        cylinder and flattening it; today most flat glass is made by floating molten glass on \
        a bath of liquid tin, which leaves both surfaces perfectly smooth.",
];

fn sparse_qualification_text() -> String {
    let mut text = QUALIFICATION_TEXT.to_string();
    for passage in SPARSE_EXTRA_PASSAGES {
        text.push_str("\n\n");
        text.push_str(passage);
    }
    text
}

/// Prints the token ids of [`sparse_qualification_text`] for oracle runs
/// (CPU only; requires GLM53_GGUF for the tokenizer).
#[test]
#[ignore = "CPU-only helper; requires GLM53_GGUF"]
fn print_sparse_qualification_tokens() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens = tokenizer
        .encode(&sparse_qualification_text(), false)
        .unwrap();
    eprintln!("sparse qualification tokens ({}): {tokens:?}", tokens.len());
}

/// Valid persistent state after `visible` positions, by kind and layer: KDA
/// S and conv tails; MLA latent rows `[0, visible)`, completed pools
/// `[0, visible / 4)`, and the pending ring's live slots (the incomplete
/// pool's positions, in logical order).
struct ValidState {
    kda_state: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    latents: Vec<Vec<f32>>,
    pools: Vec<Vec<f32>>,
    pending: Vec<Vec<f32>>,
}

impl ValidState {
    fn read(session: &Glm5NextSession<'_>) -> Self {
        let visible = session.position();
        let (pool, kv, id) = (4, 512, 128);
        let mut s = Self {
            kda_state: Vec::new(),
            conv: Vec::new(),
            latents: Vec::new(),
            pools: Vec::new(),
            pending: Vec::new(),
        };
        for layer in &session.layers {
            match layer {
                LayerState::Kda { conv, state } => {
                    s.kda_state.push(read_f32(state).unwrap());
                    s.conv.push(read_f32(conv).unwrap());
                }
                LayerState::Mla {
                    latent,
                    pending,
                    pooled,
                } => {
                    s.latents
                        .push(read_f16(latent).unwrap()[..visible * kv].to_vec());
                    s.pools
                        .push(read_f16(pooled).unwrap()[..visible / pool * id].to_vec());
                    // Slot = position % 4, each [key 128 | gate 128].
                    s.pending
                        .push(read_f16(pending).unwrap()[..visible % pool * 2 * id].to_vec());
                }
            }
        }
        s
    }

    /// Per kind: (name, worst relative error over layers, worst layer index
    /// within the kind).
    fn errors(&self, reference: &Self) -> [(&'static str, f64, usize); 5] {
        // Only the pending ring may be legitimately empty (visible % 4 == 0),
        // and then on both sides.
        let kind = |name: &'static str, a: &[Vec<f32>], b: &[Vec<f32>], may_be_empty: bool| {
            assert_eq!(a.len(), b.len(), "{name}: layer count mismatch");
            assert!(!b.is_empty(), "{name}: no layers");
            let (e, i) = worst(a.iter().zip(b).map(|(a, b)| {
                if may_be_empty && b.is_empty() {
                    assert!(a.is_empty(), "{name}: one side empty");
                    0.0
                } else {
                    relative_error(a, b)
                }
            }));
            (name, e, i)
        };
        let r = reference;
        [
            kind("kda_state", &self.kda_state, &r.kda_state, false),
            kind("conv", &self.conv, &r.conv, false),
            kind("latents", &self.latents, &r.latents, false),
            kind("pools", &self.pools, &r.pools, false),
            kind("pending", &self.pending, &r.pending, true),
        ]
    }
}

/// Sequential reader of a GLMREF01 logits file (one record in memory).
struct ReferenceStream {
    reader: std::io::BufReader<std::fs::File>,
    vocab: usize,
    remaining: usize,
}

impl ReferenceStream {
    fn open(path: &std::path::Path) -> Self {
        use std::io::Read;
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut reader = std::io::BufReader::with_capacity(1 << 22, file);
        let mut header = [0u8; 16];
        reader.read_exact(&mut header).unwrap();
        assert_eq!(&header[..8], b"GLMREF01");
        let u32_at = |o: usize| u32::from_le_bytes(header[o..o + 4].try_into().unwrap());
        Self {
            reader,
            vocab: u32_at(8) as usize,
            remaining: u32_at(12) as usize,
        }
    }

    /// (position, token, logits) of the next record.
    fn next(&mut self) -> (u32, u32, Vec<f32>) {
        use std::io::Read;
        assert!(self.remaining > 0, "reference exhausted");
        self.remaining -= 1;
        let mut head = [0u8; 8];
        self.reader.read_exact(&mut head).unwrap();
        let mut bytes = vec![0u8; self.vocab * 4];
        self.reader.read_exact(&mut bytes).unwrap();
        (
            u32::from_le_bytes(head[..4].try_into().unwrap()),
            u32::from_le_bytes(head[4..].try_into().unwrap()),
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
    }
}

const QUAL_ORACLE_DEFAULT: &str =
    "/Volumes/wdblack/weights-archive/.fetch/analysis/runs/glm53-oracle/qual-v1";

fn qual_oracle_dir() -> PathBuf {
    std::env::var_os("GLM53_QUAL_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(QUAL_ORACLE_DEFAULT))
}

/// Fast-lineage qualification over a long prompt (qual-v1): a 559-token
/// prompt of unrelated passages (one 512-row chunk plus a 47-row tail; three
/// live pending slots) and 32 teacher-forced continuation tokens.
///
/// 1. Native serial decode matches the llama.cpp serial oracle (long-context
///    dense attention, pools and recurrence): KL <= 1e-8 up to an exact
///    router-score tie at position 156 that the two resolve with different
///    tie policies, top-1 at all 591 positions; and the Exact packed
///    reference equals serial decode bitwise.
/// 2. Fast prefill with chunks 512/128/64/97 and a 3-token packed prefix is
///    compared with that Exact reference on logits at every step and on each
///    state kind over its valid region, at the prompt end and after the
///    continuation, within bounds frozen from measurement (about 2x).
/// 3. Chunking is arithmetic-neutral: 512 and 128 (both whole 128-row
///    absorption blocks) and 64 and 97 (per-row absorption) give bitwise
///    identical logits and state.
///
/// Context (same tokens): llama.cpp's batched prefill differs from its own
/// serial decode by KL 8.6e-3 at the prompt end, 4.2e-2 worst, with 16 top-1
/// flips at reference margins up to 0.34. Fast error grows smoothly with
/// depth (near-tied expert routing flipping under small perturbations,
/// amplified by KDA), identically for every chunking.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the qual-v1 oracle and an idle GPU"]
fn packed_fast_qualifies_across_chunkings_with_teacher_forcing() {
    const PROMPT: usize = 559;
    const CONTINUATION: usize = 32;
    const CAPACITY: usize = PROMPT + CONTINUATION;
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(QUALIFICATION_TEXT, false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    assert!(tokens.len() >= CAPACITY, "only {} tokens", tokens.len());
    let tokens = &tokens[..CAPACITY];
    let (prompt, continuation) = (&tokens[..PROMPT], &tokens[PROMPT..]);

    // llama.cpp's own batched-vs-serial envelope on these tokens (context).
    let (mut serial_ref, mut batch_ref) = (
        ReferenceStream::open(&qual_oracle_dir().join("serial.bin")),
        ReferenceStream::open(&qual_oracle_dir().join("batch.bin")),
    );
    assert_eq!(serial_ref.remaining, CAPACITY);
    let envelope: Vec<f64> = tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| {
            let ((_, ts, s), (_, tb, b)) = (serial_ref.next(), batch_ref.next());
            assert_eq!((ts, tb), (token, token), "token {position}");
            kl_divergence(&s, &b)
        })
        .collect();
    eprintln!(
        "llama.cpp batched vs serial: prompt-end KL {:.3e}, worst {:.3e}",
        envelope[PROMPT - 1],
        worst(envelope.iter().copied()).0
    );

    // Exact packed reference.
    let mut exact = Glm5NextSession::with_prefill_rows(&ctx, &weights, CAPACITY, 512).unwrap();
    exact.set_packed_lineage(PackedLineage::Exact);
    let reference_prefill = exact.prefill_packed(&ctx, prompt).unwrap();
    let reference_state = ValidState::read(&exact);
    let reference_bits = state_bits(&exact);
    let reference_steps: Vec<Vec<f32>> = continuation
        .iter()
        .map(|&t| exact.forward(&ctx, t).unwrap())
        .collect();
    let reference_end = ValidState::read(&exact);
    drop(exact);

    // Native serial decode against the llama.cpp serial oracle at every
    // position; it must also reproduce the Exact reference bitwise. Native
    // and llama.cpp agree to KL <= 1.1e-9 through position 155. At position
    // 156, block 27's 8th and 9th biased router scores tie exactly: native
    // keeps the lower expert id (9), llama.cpp's bitonic sort keeps 115
    // (qual-v1 route156 capture). That one token spikes to KL 4.9e-3, and
    // the recurrent state carries a small offset afterwards.
    const ORACLE_TIE_POSITION: usize = 156;
    const ORACLE_PRE_TIE_KL: f64 = 1e-8;
    const ORACLE_KL: f64 = 1e-2;
    let mut serial_ref = ReferenceStream::open(&qual_oracle_dir().join("serial.bin"));
    let mut oracle_kls = Vec::with_capacity(CAPACITY);
    let mut oracle_top1 = 0;
    {
        let mut serial = Glm5NextSession::new(&ctx, &weights, CAPACITY).unwrap();
        let mut last = Vec::new();
        for (position, &token) in prompt.iter().enumerate() {
            last = serial.forward(&ctx, token).unwrap();
            let (p, t, expected) = serial_ref.next();
            assert_eq!((p as usize, t), (position, token));
            oracle_kls.push(kl_divergence(&expected, &last));
            oracle_top1 += usize::from(argmax(&expected) == argmax(&last));
        }
        assert!(last == reference_prefill, "exact packed != serial logits");
        assert!(
            state_bits(&serial) == reference_bits,
            "exact packed != serial state"
        );
    }
    for (step, logits) in reference_steps.iter().enumerate() {
        let (p, t, expected) = serial_ref.next();
        assert_eq!((p as usize, t), (PROMPT + step, continuation[step]));
        oracle_kls.push(kl_divergence(&expected, logits));
        oracle_top1 += usize::from(argmax(&expected) == argmax(logits));
    }
    assert_eq!(oracle_kls.len(), CAPACITY);
    let oracle_worst = worst(oracle_kls.iter().copied());
    let buckets: Vec<String> = [0, 16, 32, 64, 128, 256, 384, PROMPT, CAPACITY]
        .windows(2)
        .map(|w| {
            let span = &oracle_kls[w[0]..w[1]];
            let mut sorted = span.to_vec();
            sorted.sort_by(f64::total_cmp);
            format!(
                "[{},{}) median {:.1e} max {:.1e}",
                w[0],
                w[1],
                sorted[sorted.len() / 2],
                sorted[sorted.len() - 1]
            )
        })
        .collect();
    eprintln!(
        "native serial vs llama.cpp serial over {CAPACITY} positions: top-1 {oracle_top1}/{CAPACITY}, worst KL {:.3e} at {}",
        oracle_worst.0, oracle_worst.1
    );
    eprintln!("  KL by position: {}", buckets.join("; "));
    let pre_tie = worst(oracle_kls[..ORACLE_TIE_POSITION].iter().copied()).0;
    let mut failures = Vec::new();
    if !within(pre_tie, ORACLE_PRE_TIE_KL) {
        failures.push(format!("native vs oracle before the tie: KL {pre_tie:.3e}"));
    }
    if !within(oracle_worst.0, ORACLE_KL) || oracle_top1 != CAPACITY {
        failures.push(format!(
            "native vs oracle: worst KL {:.3e} at {}, top-1 {oracle_top1}/{CAPACITY}",
            oracle_worst.0, oracle_worst.1
        ));
    }

    // Calibrated regression bounds, not an independent holdout: about 2x the
    // worst measured over all variants when this test was introduced.
    const LOGIT_KL: f64 = 2e-2; // measured 9.1e-3
    const KDA_STATE: f64 = 2e-1; // 1.06e-1
    const CONV: f64 = 2.5e-1; // 1.33e-1
    const LATENTS: f64 = 1.25e-1; // 6.6e-2
    const POOLS: f64 = 1.5e-1; // 7.4e-2
    const PENDING: f64 = 1.75e-1; // 8.8e-2
    // Top-1 may flip only to a near-tied alternative: the logit regret of
    // each side's choice under the other side's logits stays small. This
    // bound predates the regret metric; measured flips: reference-side regret
    // <= 0.06, Fast-side regret <= 0.176 (deterministic across runs).
    const TOP1_REGRET: f32 = 0.2;
    let bound = |kind: &str| match kind {
        "kda_state" => KDA_STATE,
        "conv" => CONV,
        "latents" => LATENTS,
        "pools" => POOLS,
        "pending" => PENDING,
        other => unreachable!("{other}"),
    };
    // (rows, prefix, the variant it must equal bitwise)
    let variants: [(usize, usize, Option<usize>); 5] = [
        (512, 0, None),
        (128, 0, Some(0)),
        (64, 0, None),
        (97, 0, Some(2)),
        (512, 3, None),
    ];
    let mut fingerprints: Vec<Fingerprint> = Vec::new();
    for &(rows, prefix, equal_to) in &variants {
        let label = format!("fast rows {rows} prefix {prefix}");
        let mut fast = Glm5NextSession::with_prefill_rows(&ctx, &weights, CAPACITY, rows).unwrap();
        fast.set_packed_lineage(PackedLineage::Fast);
        let start = std::time::Instant::now();
        if prefix > 0 {
            fast.prefill_packed(&ctx, &prompt[..prefix]).unwrap();
        }
        let prefill = fast.prefill_packed(&ctx, &prompt[prefix..]).unwrap();
        let prefill_ms = start.elapsed().as_secs_f64() * 1e3;
        let prompt_state = ValidState::read(&fast);
        let prompt_bits = state_bits(&fast);
        let mut logits = vec![prefill];
        for &token in continuation {
            logits.push(fast.forward(&ctx, token).unwrap());
        }
        let end_errors = ValidState::read(&fast).errors(&reference_end);
        let prompt_errors = prompt_state.errors(&reference_state);
        drop(fast);
        let references: Vec<&Vec<f32>> = std::iter::once(&reference_prefill)
            .chain(reference_steps.iter())
            .collect();
        assert_eq!(references.len(), logits.len());
        let mut kls = Vec::with_capacity(logits.len());
        let mut flips = Vec::new();
        for (step, (reference, fast)) in references.iter().zip(&logits).enumerate() {
            kls.push(kl_divergence(reference, fast));
            let regret = choice_regret(reference, fast);
            if regret != (0.0, 0.0) {
                flips.push((step, regret));
            }
        }
        let worst_kl = worst(kls.iter().copied()).0;
        let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
        eprintln!(
            "{label}: prefill {prefill_ms:.0} ms; KL prefill {:.3e} mean {mean_kl:.3e} worst {worst_kl:.3e}; top-1 flips (step, regret ref/fast) {flips:?}",
            kls[0]
        );
        for (when, errors) in [("prompt", &prompt_errors), ("end", &end_errors)] {
            let line: Vec<String> = errors
                .iter()
                .map(|(kind, e, layer)| format!("{kind} {e:.3e}@{layer}"))
                .collect();
            eprintln!("  {when}: {}", line.join("  "));
            for (kind, error, layer) in errors {
                if !within(*error, bound(kind)) {
                    failures.push(format!(
                        "{label} {when}: {kind} {error:.3e} at layer {layer}"
                    ));
                }
            }
        }
        if !within(worst_kl, LOGIT_KL) {
            failures.push(format!("{label}: worst KL {worst_kl:.3e}"));
        }
        let near_tie = |r: f32| within(f64::from(r), f64::from(TOP1_REGRET));
        if let Some((step, regret)) = flips
            .iter()
            .find(|(_, (a, b))| !near_tie(*a) || !near_tie(*b))
        {
            failures.push(format!(
                "{label}: top-1 flip at step {step}, regret {regret:?}"
            ));
        }
        if let Some(index) = equal_to {
            let (bits, steps) = &fingerprints[index];
            let (r, p) = (variants[index].0, variants[index].1);
            if *bits != prompt_bits || *steps != logit_bits(&logits) {
                failures.push(format!("{label}: not bitwise equal to rows {r} prefix {p}"));
            }
        }
        fingerprints.push((prompt_bits, logit_bits(&logits)));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A variant's prompt-end state bits and its logit bits at every step.
type Fingerprint = (Vec<Vec<u32>>, Vec<Vec<u32>>);

fn logit_bits(steps: &[Vec<f32>]) -> Vec<Vec<u32>> {
    steps
        .iter()
        .map(|step| step.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// CPU negative controls for the comparison helpers: nonfinite, empty and
/// mismatched inputs fail loudly, reductions never hide NaN, and a top-1
/// flip is measured by the regret of the choice actually made.
#[test]
fn comparison_helpers_refuse_nonfinite_and_mismatched_inputs() {
    fn refused<T>(f: impl FnOnce() -> T) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
    }
    let ok = [1.0f32, 2.0, 3.0];
    for bad in [
        [1.0f32, f32::NAN, 3.0],
        [1.0, f32::INFINITY, 3.0],
        [f32::NEG_INFINITY, 2.0, 3.0],
    ] {
        assert!(refused(|| relative_error(&bad, &ok)));
        assert!(refused(|| relative_error(&ok, &bad)));
        assert!(refused(|| kl_divergence(&ok, &bad)));
        assert!(refused(|| kl_divergence(&bad, &ok)));
        assert!(refused(|| max_abs_diff(&bad, &ok)));
        assert!(refused(|| choice_regret(&ok, &bad)));
    }
    assert!(refused(|| relative_error(&ok[..2], &ok)));
    assert!(refused(|| relative_error(&[], &[])));
    assert!(refused(|| kl_divergence(&ok, &ok[..2])));

    assert!(worst([0.1, f64::NAN, 0.3]).0.is_nan());
    assert!(worst([f64::NAN, 0.3]).0.is_nan());
    assert_eq!(worst([0.1, 0.3, 0.2]), (0.3, 1));
    assert!(!within(f64::NAN, 1.0));
    assert!(within(1.0, 1.0) && !within(1.5, 1.0));

    assert_eq!(
        choice_regret(&[0.0, 1.0, 0.5], &[0.0, 2.0, 1.0]),
        (0.0, 0.0)
    );
    // The reference's runner-up is close (margin 0.1), but native picked a
    // far worse token under the reference: the regret says so.
    let (forward, reverse) = choice_regret(&[0.0, 1.0, 0.9], &[2.0, 1.0, 1.5]);
    assert!((forward - 1.0).abs() < 1e-6 && (reverse - 1.0).abs() < 1e-6);

    let state = |kda: f32, pending: Vec<f32>| ValidState {
        kda_state: vec![vec![1.0, kda]],
        conv: vec![vec![1.0]],
        latents: vec![vec![1.0]],
        pools: vec![vec![1.0]],
        pending: vec![pending],
    };
    assert!(refused(
        || state(f32::NAN, vec![]).errors(&state(2.0, vec![]))
    ));
    assert!(refused(
        || state(2.0, vec![]).errors(&state(f32::NAN, vec![]))
    ));
    assert!(refused(|| state(2.0, vec![1.0]).errors(&state(2.0, vec![]))));
    let mut short = state(2.0, vec![]);
    short.kda_state.push(vec![1.0, 2.0]);
    assert!(refused(|| short.errors(&state(2.0, vec![]))));
    let errors = state(2.0, vec![]).errors(&state(2.0, vec![]));
    assert!(errors.iter().all(|(_, e, _)| *e == 0.0));
}

const SPARSE_ORACLE_DEFAULT: &str =
    "/Volumes/wdblack/weights-archive/.fetch/analysis/runs/glm53-oracle/sparse-v1";

fn sparse_oracle_dir() -> PathBuf {
    std::env::var_os("GLM53_SPARSE_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(SPARSE_ORACLE_DEFAULT))
}

/// Executed MLA block indices of the release (every fourth block from 3).
const MLA_BLOCKS: [u32; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];

/// Component replay of sparse selection on llama.cpp's own inputs (sparse-v1
/// captures at positions 2050-2060, all 11 MLA blocks): the native 32-head
/// scorer runs on the captured `indexer_q` (rounded to F16 in the kernel's
/// contract), `indexer_weights` and `indexer_pool_k`, and the native selector
/// on both llama.cpp's and the native scores. Bounds were frozen before the
/// first observation:
/// - scores within 1e-5 of the row's largest magnitude;
/// - selection on llama.cpp's scores contains every strict winner and no
///   strict loser (llama.cpp fills threshold ties in atomic order);
/// - selection on native scores equals llama.cpp's set when the 512th/513th
///   gap exceeds twice the measured score error, else agrees on every pool
///   beyond that error from the threshold;
/// - at position 2050 (visible length 2051) llama.cpp selects every visible
///   pool (dense equivalence).
#[test]
#[ignore = "requires the sparse-v1 llama.cpp captures and Metal"]
fn sparse_selection_replays_llama_cpp_indexer_captures() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let names = [
        "indexer_q",
        "indexer_weights",
        "indexer_pool_k",
        "indexer_score",
        "indexer_top_k",
    ];
    let captures = read_captures(&sparse_oracle_dir().join("capture.bin.captures"), &names);
    const TOP: usize = 512;
    let (h, d) = (32usize, 128usize);
    let mut worst_score_error = 0.0f64;
    let mut exclusions = 0usize;
    for step in 2050u32..=2060 {
        let visible_pools = (step as usize + 1) / 4;
        for &layer in &MLA_BLOCKS {
            let get = |name: &str| {
                captures
                    .get(&(name.to_string(), step, layer, 0))
                    .unwrap_or_else(|| panic!("missing {name} step {step} layer {layer}"))
            };
            let (q, w, pool_k, llama_scores, llama_top) = (
                get("indexer_q"),
                get("indexer_weights"),
                get("indexer_pool_k"),
                get("indexer_score"),
                get("indexer_top_k"),
            );
            let label = format!("step {step} layer {layer}");
            let pools = pool_k.len() / d;
            assert!(
                pools >= visible_pools && llama_scores.len() == pools,
                "{label}"
            );
            assert_eq!(
                (q.len(), w.len(), llama_top.len()),
                (h * d, h, TOP),
                "{label}"
            );
            assert_finite(&label, &llama_scores[..visible_pools]);
            assert!(
                llama_scores[visible_pools..]
                    .iter()
                    .all(|s| *s == f32::NEG_INFINITY),
                "{label}: invisible pools must be masked"
            );
            // Pooled keys are F16 cache values: exact through F16.
            let keys: Vec<half::f16> = pool_k.iter().map(|&v| half::f16::from_f32(v)).collect();
            assert!(
                keys[..visible_pools * d]
                    .iter()
                    .zip(pool_k)
                    .all(|(k, v)| k.to_f32() == *v),
                "{label}: pooled keys are not F16 values"
            );
            let queries: Vec<half::f16> = q.iter().map(|&v| half::f16::from_f32(v)).collect();
            let tensor = |bytes: &[u8], shape: Vec<u64>, dtype| {
                crate::metal::MetalTensor::from_bytes(&ctx, bytes, shape, dtype).unwrap()
            };
            let q_t = tensor(
                bytemuck::cast_slice(&queries),
                vec![d as u64, h as u64, 1],
                GgmlType::F16,
            );
            let w_t = tensor(bytemuck::cast_slice(w), vec![h as u64, 1], GgmlType::F32);
            let k_t = tensor(
                bytemuck::cast_slice(&keys),
                vec![d as u64, pools as u64],
                GgmlType::F16,
            );
            let v_t = tensor(
                bytemuck::cast_slice(&[visible_pools as i32]),
                vec![1],
                GgmlType::I32,
            );
            let s_t = MetalTensor::zeros_f32(&ctx, vec![pools as u64, 1]).unwrap();
            let run = |encode: &dyn Fn(&KernelEncoder)| {
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                encode(&enc);
                enc.end();
                command.commit();
                crate::metal::wait_completed(&command).unwrap();
            };
            run(&|enc| {
                crate::metal::encode_lightning_scores_f16_matrix(
                    &ctx,
                    enc,
                    &crate::metal::LightningScores {
                        queries: &q_t,
                        head_weights: &w_t,
                        keys: &k_t,
                        visible_counts: &v_t,
                        scores: &s_t,
                    },
                    h,
                    d,
                    pools,
                    visible_pools,
                    1,
                )
                .unwrap()
            });
            let native = read_f32(&s_t).unwrap();
            let native = &native[..visible_pools];
            let reference = &llama_scores[..visible_pools];
            let scale = reference
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()))
                .max(1e-30);
            let error = max_abs_diff(native, reference);
            worst_score_error = worst_score_error.max(f64::from(error / scale));
            assert!(
                within(f64::from(error / scale), 1e-5),
                "{label}: score error {error} (scale {scale})"
            );
            let llama_set: std::collections::BTreeSet<i32> =
                llama_top.iter().map(|&v| v as i32).collect();
            if visible_pools <= TOP {
                let all: std::collections::BTreeSet<i32> = (0..visible_pools as i32).collect();
                assert_eq!(llama_set, all, "{label}: dense range selects every pool");
                continue;
            }
            exclusions += 1;
            // Threshold from llama.cpp's scores.
            let mut sorted: Vec<f32> = reference.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let (threshold, next) = (sorted[TOP - 1], sorted[TOP]);
            let select = |scores: &[f32]| -> std::collections::BTreeSet<i32> {
                let mut padded = scores.to_vec();
                padded.resize(pools, f32::NEG_INFINITY);
                let s_t = tensor(
                    bytemuck::cast_slice(&padded),
                    vec![pools as u64, 1],
                    GgmlType::F32,
                );
                let ids = MetalTensor::zeros_i32(&ctx, vec![TOP as u64, 1]).unwrap();
                let counts = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
                let status = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
                run(&|enc| {
                    crate::metal::encode_select_top_k_ids(
                        &ctx,
                        enc,
                        &crate::metal::TopKSelection {
                            scores: &s_t,
                            visible_counts: &v_t,
                            ids: &ids,
                            counts: &counts,
                            status: &status,
                        },
                        pools,
                        visible_pools,
                        TOP,
                        1,
                    )
                    .unwrap()
                });
                assert_eq!(
                    read_i32(&status).unwrap()[0],
                    crate::metal::SELECT_STATUS_OK,
                    "{label}"
                );
                assert_eq!(read_i32(&counts).unwrap()[0], TOP as i32, "{label}");
                read_i32(&ids).unwrap().into_iter().collect()
            };
            let respects = |set: &std::collections::BTreeSet<i32>, band: f32| {
                reference.iter().enumerate().all(|(pool, &s)| {
                    let inside = set.contains(&(pool as i32));
                    !(s > threshold + band && !inside) && !(s < threshold - band && inside)
                })
            };
            assert!(respects(&llama_set, 0.0), "{label}: llama.cpp set");
            let on_llama = select(reference);
            assert!(
                respects(&on_llama, 0.0),
                "{label}: native selection of llama.cpp scores"
            );
            if threshold > next {
                assert_eq!(on_llama, llama_set, "{label}: untied threshold");
            }
            let on_native = select(native);
            if threshold - next > 2.0 * error {
                assert_eq!(
                    on_native,
                    llama_set,
                    "{label}: native scores, gap {}",
                    threshold - next
                );
            } else {
                assert!(
                    respects(&on_native, error),
                    "{label}: native scores near the threshold"
                );
            }
        }
    }
    eprintln!(
        "replayed 11 positions x 11 MLA blocks: worst relative score error {worst_score_error:.3e}, {exclusions} sparse selections"
    );
}

/// Decode across the sparse frontier against llama.cpp (sparse-v1): Exact
/// packed prefill of 2048 tokens (dense), then teacher-forced decode of
/// positions 2048-2092 (dense through 2050, sparse selection from 2051: the
/// first exclusion, every tail length and ten new pools), each position's
/// logits against llama.cpp serial decode. Bounds frozen before the first
/// observation, at qual-v1's overall ceilings: KL <= 1e-2 and choice regret
/// <= 0.2 in both directions at every position. Also reports llama.cpp's
/// own observer effect (captured vs uncaptured logits at 2050-2060).
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the sparse-v1 oracle and an idle GPU"]
fn sparse_decode_crosses_the_frontier_against_llama_cpp() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&sparse_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    const PREFIX: usize = 2048;
    let total = tokens.len();
    assert!(total >= PREFIX + 40, "only {total} tokens");
    const KL: f64 = 1e-2;
    const REGRET: f64 = 0.2;

    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, total, 512).unwrap();
    session.set_packed_lineage(PackedLineage::Exact);
    assert_allocation_within_ledger(&session);
    let start = std::time::Instant::now();
    let prefix_logits = session.prefill_packed(&ctx, &tokens[..PREFIX]).unwrap();
    eprintln!(
        "exact packed prefill: {PREFIX} tokens in {:.1} s",
        start.elapsed().as_secs_f64()
    );
    let mut native = vec![prefix_logits];
    let start = std::time::Instant::now();
    for &token in &tokens[PREFIX..] {
        native.push(session.forward(&ctx, token).unwrap());
    }
    eprintln!(
        "decode {} -> {}: {:.1} ms/token",
        PREFIX,
        total,
        start.elapsed().as_secs_f64() * 1e3 / (total - PREFIX) as f64
    );
    // native[i] is the logits at position PREFIX - 1 + i.
    let mut reference = ReferenceStream::open(&sparse_oracle_dir().join("serial.bin"));
    assert_eq!(reference.remaining, total);
    let mut captured = ReferenceStream::open(&sparse_oracle_dir().join("capture.bin"));
    let mut failures = Vec::new();
    let mut top1 = 0;
    let mut rows = Vec::new();
    for position in 0..total {
        let (p, t, expected) = reference.next();
        let (_, _, with_captures) = captured.next();
        assert_eq!(
            (p as usize, t),
            (position, tokens[position]),
            "token {position}"
        );
        if (2050..=2060).contains(&position) {
            rows.push(format!(
                "{position}:{:.1e}",
                kl_divergence(&expected, &with_captures)
            ));
        }
        if position + 1 < PREFIX {
            continue;
        }
        let logits = &native[position + 1 - PREFIX];
        let kl = kl_divergence(&expected, logits);
        let (forward, reverse) = choice_regret(&expected, logits);
        top1 += usize::from(argmax(&expected) == argmax(logits));
        let sparse = position + 1 >= weights.config.sparse_frontier() as usize;
        eprintln!(
            "pos {position} ({}): kl {kl:.3e} regret {forward:.3}/{reverse:.3}",
            if sparse { "sparse" } else { "dense" }
        );
        if !within(kl, KL)
            || !within(f64::from(forward), REGRET)
            || !within(f64::from(reverse), REGRET)
        {
            failures.push(format!(
                "pos {position}: kl {kl:.3e} regret {forward:.3}/{reverse:.3}"
            ));
        }
    }
    eprintln!(
        "top-1 {top1}/{}; llama.cpp captured vs uncaptured KL: {}",
        native.len(),
        rows.join(" ")
    );
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Timing runs hold the production lease but must not run under Metal API
/// validation, which distorts timing.
fn perf_lease() -> impl Sized {
    assert!(
        std::env::var_os("MTL_DEBUG_LAYER").is_none(),
        "timing runs must not enable MTL_DEBUG_LAYER"
    );
    crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease")
}

/// Decode cost on either side of the sparse frontier: Fast packed prefill to
/// position 2040, then per-token wall time for teacher-forced decode of
/// positions 2040-2050 (dense, visible length up to 2051) and 2051-2092
/// (sparse). Reports medians; no bound (timing, not qualification).
#[test]
#[ignore = "timing; loads the 109.5 GiB GLM-5.3 trunk; requires GLM53_GGUF, no MTL_DEBUG_LAYER, an idle GPU"]
fn sparse_decode_cost_across_the_frontier() {
    let _lease = perf_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&sparse_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    const START: usize = 2040;
    let mut session =
        Glm5NextSession::with_prefill_rows(&ctx, &weights, tokens.len(), 510).unwrap();
    session.prefill_packed(&ctx, &tokens[..START]).unwrap();
    let frontier = weights.config.sparse_frontier() as usize;
    let (mut dense, mut sparse) = (Vec::new(), Vec::new());
    for (offset, &token) in tokens[START..].iter().enumerate() {
        let position = START + offset;
        let t = std::time::Instant::now();
        session.forward(&ctx, token).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if position + 1 >= frontier {
            sparse.push(ms);
        } else {
            dense.push(ms);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (d, s) = (median(&mut dense), median(&mut sparse));
    eprintln!(
        "decode median: dense {d:.2} ms ({} tokens), sparse {s:.2} ms ({} tokens), +{:.2} ms ({:+.1}%)",
        dense.len(),
        sparse.len(),
        s - d,
        100.0 * (s - d) / d
    );
}

/// Packed prefill across the sparse frontier. Bounds frozen before the first
/// observation:
/// 1. Exact packed == serial decode bitwise (last logits and every state
///    bit) over 2400 tokens for chunkings 512 (frontier 3 rows into a chunk,
///    then 349 sparse rows in six microbatches reusing scratch), a 37-token
///    prefix then 100-row chunks (the crossing chunk starts mid-pool at
///    2037), and a 3-token prefix then 64-row chunks.
/// 2. Fast packed with 512- and 128-row chunks is bitwise identical, and
///    within qual-v1's ceilings of Exact (KL <= 2e-2, choice regret <= 0.2)
///    at the last prompt position and over 8 teacher-forced continuation
///    steps.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the sparse-v1 oracle and an idle GPU"]
fn packed_sparse_prefill_matches_serial_across_the_frontier() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let mut text = sparse_qualification_text();
    for passage in SPARSE_EXTRA_PASSAGES {
        text.push_str("\n\n");
        text.push_str(passage);
    }
    let all: Vec<u32> = tokenizer
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    const PROMPT: usize = 2400;
    const CONTINUATION: usize = 8;
    assert!(
        all.len() >= PROMPT + CONTINUATION,
        "only {} tokens",
        all.len()
    );
    let (prompt, continuation) = (&all[..PROMPT], &all[PROMPT..PROMPT + CONTINUATION]);
    let capacity = PROMPT + CONTINUATION;
    let mut failures = Vec::new();

    // Serial reference.
    let start = std::time::Instant::now();
    let (serial_logits, serial_bits) = {
        let mut serial = Glm5NextSession::new(&ctx, &weights, capacity).unwrap();
        let logits = serial.prefill(&ctx, prompt).unwrap();
        (logits, state_bits(&serial))
    };
    eprintln!(
        "serial: {PROMPT} tokens in {:.1} s",
        start.elapsed().as_secs_f64()
    );

    // 1. Exact packed == serial.
    let packed_run = |lineage: PackedLineage, rows: usize, prefix: usize| {
        let mut session =
            Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, rows).unwrap();
        session.set_packed_lineage(lineage);
        let start = std::time::Instant::now();
        if prefix > 0 {
            session.prefill_packed(&ctx, &prompt[..prefix]).unwrap();
        }
        let logits = session.prefill_packed(&ctx, &prompt[prefix..]).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1e3;
        (session, logits, ms)
    };
    for (rows, prefix) in [(512usize, 0usize), (100, 37), (64, 3)] {
        let (session, logits, ms) = packed_run(PackedLineage::Exact, rows, prefix);
        let equal = logits == serial_logits;
        let state = state_bits(&session) == serial_bits;
        eprintln!(
            "exact rows {rows} prefix {prefix}: {ms:.0} ms; logits equal {equal}, state equal {state}"
        );
        if !equal || !state {
            failures.push(format!("exact rows {rows} prefix {prefix} != serial"));
        }
    }

    // 2. Fast: chunk invariance and the envelope against Exact.
    const KL: f64 = 2e-2;
    const REGRET: f64 = 0.2;
    let continue_run = |mut session: Glm5NextSession<'_>, first: Vec<f32>| {
        let mut steps = vec![first];
        for &token in continuation {
            steps.push(session.forward(&ctx, token).unwrap());
        }
        steps
    };
    let (exact_session, exact_logits, _) = packed_run(PackedLineage::Exact, 512, 0);
    let exact = continue_run(exact_session, exact_logits);
    let mut fast_bits = Vec::new();
    for rows in [512usize, 128] {
        let (session, logits, ms) = packed_run(PackedLineage::Fast, rows, 0);
        let fast = continue_run(session, logits);
        let label = format!("fast rows {rows}");
        let mut line = Vec::new();
        for (step, (e, f)) in exact.iter().zip(&fast).enumerate() {
            let kl = kl_divergence(e, f);
            let (a, b) = choice_regret(e, f);
            line.push(format!("{kl:.1e}"));
            if !within(kl, KL) || !within(f64::from(a), REGRET) || !within(f64::from(b), REGRET) {
                failures.push(format!(
                    "{label} step {step}: kl {kl:.3e} regret {a:.3}/{b:.3}"
                ));
            }
        }
        eprintln!(
            "{label}: prefill {ms:.0} ms; KL vs exact by step: {}",
            line.join(" ")
        );
        fast_bits.push(logit_bits(&fast));
    }
    if fast_bits[0] != fast_bits[1] {
        failures.push("fast 512 and 128 chunkings differ".into());
    }

    assert!(failures.is_empty(), "{failures:#?}");
}

/// Packed prefill of sparse-v1's 2093 tokens (crossing the frontier inside
/// the fifth 512-row chunk) against llama.cpp serial at position 2092.
/// Bounds frozen before the first observation: Exact KL <= 1e-2, Fast KL
/// <= 2e-2, choice regret <= 0.2 both ways.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the sparse-v1 oracle and an idle GPU"]
fn packed_sparse_prefill_matches_llama_cpp_at_the_prompt_end() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens: Vec<u32> = tokenizer
        .encode(&sparse_qualification_text(), false)
        .unwrap()
        .into_iter()
        .map(|t| t as u32)
        .collect();
    const REGRET: f64 = 0.2;
    let mut reference = ReferenceStream::open(&sparse_oracle_dir().join("serial.bin"));
    assert_eq!(reference.remaining, tokens.len());
    let mut expected = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        let (p, t, logits) = reference.next();
        assert_eq!((p as usize, t), (position, token), "token {position}");
        expected = logits;
    }
    let mut failures = Vec::new();
    for (lineage, bound) in [(PackedLineage::Exact, 1e-2), (PackedLineage::Fast, 2e-2)] {
        let mut session =
            Glm5NextSession::with_prefill_rows(&ctx, &weights, tokens.len(), 512).unwrap();
        session.set_packed_lineage(lineage);
        let logits = session.prefill_packed(&ctx, &tokens).unwrap();
        const ORACLE_POSITION: usize = 2092;
        let kl = kl_divergence(&expected, &logits);
        let (a, b) = choice_regret(&expected, &logits);
        eprintln!(
            "{lineage:?} packed vs llama.cpp at {ORACLE_POSITION}: kl {kl:.3e} regret {a:.3}/{b:.3}"
        );
        if !within(kl, bound) || !within(f64::from(a), REGRET) || !within(f64::from(b), REGRET) {
            failures.push(format!(
                "{lineage:?} vs llama.cpp: kl {kl:.3e} regret {a:.3}/{b:.3}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

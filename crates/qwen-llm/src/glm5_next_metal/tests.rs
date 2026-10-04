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
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        offset += vocab * 4;
        out.push((token, logits));
    }
    out
}

/// GLMCAP01 F32 records keyed by (name, step, layer, occurrence).
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
        if dtype == 0 && wanted.contains(&name.as_str()) {
            let values = bytes[offset..offset + n]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            out.insert((name, step, layer, occurrence), values);
        }
        offset += n;
    }
    out
}

fn relative_error(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let diff: f64 = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| ((a - e) as f64).powi(2))
        .sum();
    let norm: f64 = expected.iter().map(|e| (*e as f64).powi(2)).sum();
    (diff / norm.max(1e-30)).sqrt()
}

fn kl_divergence(reference: &[f32], native: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = x.iter().map(|v| (*v as f64 - max).exp()).sum();
        x.iter()
            .map(move |v| *v as f64 - max - sum.ln())
            .collect::<Vec<_>>()
    };
    let (p, q) = (log_softmax(reference), log_softmax(native));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
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
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires GLM53_GGUF, the ckpt-v1 oracle and an idle GPU"]
fn checkpoint_v1_matches_llama_cpp_oracle() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context (production lease)");
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
            "allocated with one session: {} bytes (ledger decode peak {})",
            ctx.current_allocated_size(),
            session.ledger().phase_peaks().decode
        );
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
            let max_abs = logits
                .iter()
                .zip(expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
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
        let drift = logits
            .iter()
            .zip(&observed_logits[position])
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
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
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires GLM53_GGUF, the ckpt-v1 oracle and an idle GPU"]
fn packed_prefill_matches_serial_and_oracle_on_checkpoint_v1() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context (production lease)");
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
        let drift = logits
            .iter()
            .zip(&serial_logits[position])
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
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
        // Whole-request refusals execute nothing: position, poison flag and
        // every state byte are unchanged, and decode continues as if the
        // requests never happened (Exact: bitwise against serial below).
        let before = state_bits(&mixed);
        let vocab = weights.config.vocab_size;
        let overrun = vec![tokens[11]; 256 - 11 + 1];
        for (label, request) in [
            ("empty", &[][..]),
            ("bad tail token", &[tokens[11], tokens[12], vocab][..]),
            ("bad head token", &[vocab, tokens[11]][..]),
            ("capacity overrun", &overrun[..]),
        ] {
            for packed in [true, false] {
                let result = if packed {
                    mixed.prefill_packed(&ctx, request)
                } else {
                    mixed.prefill(&ctx, request)
                };
                assert!(result.is_err(), "{label} (packed {packed}) was accepted");
                assert_eq!(mixed.position(), 11, "{label}: position moved");
                assert!(!mixed.poisoned, "{label}: refusal poisoned the session");
            }
        }
        assert!(state_bits(&mixed) == before, "refusals changed state");
        for (position, &token) in tokens.iter().enumerate().skip(11) {
            let logits = mixed.forward(&ctx, token).unwrap();
            check(
                &format!("{lineage:?} mixed decode"),
                position,
                &logits,
                lineage,
            );
        }
    }
}

/// Fast packed prefill over 200 tokens (one 128-row grouped block plus a
/// 72-row tail) against exact lineage, which equals serial decode: last-row
/// logits within the fast envelope, then three decode steps on each.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires GLM53_GGUF and an idle GPU"]
fn packed_fast_matches_exact_over_a_grouped_block_and_tail() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context (production lease)");
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

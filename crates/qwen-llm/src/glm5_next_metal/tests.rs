use super::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

mod natural;

const ORACLE_DEFAULT: &str =
    "/Volumes/wdblack/weights-archive/.fetch/analysis/runs/glm53-oracle/ckpt-v1";

fn oracle_dir() -> PathBuf {
    std::env::var_os("GLM53_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(ORACLE_DEFAULT))
}

const CKPT_V1_MANIFEST: &str = include_str!("../../../../scripts/reference/glm53/ckpt-v1.json");
const QUAL_V1_MANIFEST: &str = include_str!("../../../../scripts/reference/glm53/qual-v1.json");
const SPARSE_V1_MANIFEST: &str = include_str!("../../../../scripts/reference/glm53/sparse-v1.json");
const SPARSE_V2_MANIFEST: &str = include_str!("../../../../scripts/reference/glm53/sparse-v2.json");

/// Oracle evidence named by a reference manifest. Before a test reads `name`
/// from `dir`, its byte length and SHA-256 must equal the manifest's, and a
/// capture file's record count must equal the manifest's `records`, so a
/// replay is bound to the recorded producer run. Hashed once per process per
/// (path, length, SHA-256, records): the same expectation skips rehashing,
/// and a different manifest for an already verified path is checked afresh.
/// Oracle files are immutable evidence: a later read reopens the path and is
/// not re-bound to the verified bytes, so replacing a file mid-process is
/// outside this check.
fn verified(manifest: &str, dir: &std::path::Path, name: &str) -> PathBuf {
    use sha2::Digest;
    use std::io::Read;
    type Key = (PathBuf, u64, String, Option<u64>);
    static VERIFIED: std::sync::Mutex<std::collections::BTreeSet<Key>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let path = dir.join(name);
    let manifest: serde_json::Value = serde_json::from_str(manifest).unwrap();
    let entry = &manifest["files"][name];
    let bytes = entry["bytes"]
        .as_u64()
        .unwrap_or_else(|| panic!("{name}: no byte length in the manifest"));
    let sha256 = entry["sha256"]
        .as_str()
        .unwrap_or_else(|| panic!("{name}: no SHA-256 in the manifest"));
    let key: Key = (
        path.clone(),
        bytes,
        sha256.to_owned(),
        entry["records"].as_u64(),
    );
    if VERIFIED.lock().unwrap().contains(&key) {
        return path;
    }
    let length = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .len();
    assert_eq!(length, bytes, "{name}: size differs from the manifest");
    let mut file = std::fs::File::open(&path).unwrap();
    let mut hasher = sha2::Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, sha256, "{name}: SHA-256 differs from the manifest");
    if let Some(records) = entry["records"].as_u64() {
        assert_eq!(
            capture_record_count(&path),
            records as usize,
            "{name}: record count differs from the manifest"
        );
    }
    VERIFIED.lock().unwrap().insert(key);
    path
}

/// Records in a GLMCAP01 file (headers only; payloads skipped).
fn capture_record_count(path: &std::path::Path) -> usize {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"GLMCAP01");
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let mut offset = 8;
    let mut count = 0;
    while offset < bytes.len() {
        let name_len = u32_at(offset + 12) as usize;
        offset += 16 + name_len + 4 + 32;
        let n = i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8 + n;
        count += 1;
    }
    assert_eq!(offset, bytes.len(), "truncated capture record");
    count
}

fn checkpoint_tokens() -> Vec<u32> {
    let manifest: serde_json::Value = serde_json::from_str(CKPT_V1_MANIFEST).unwrap();
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
pub(super) fn kl_divergence(reference: &[f32], native: &[f32]) -> f64 {
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
pub(super) fn choice_regret(reference: &[f32], native: &[f32]) -> (f32, f32) {
    assert_comparable("choice regret", native, reference);
    let (r, n) = (argmax(reference), argmax(native));
    (reference[r] - reference[n], native[n] - native[r])
}

pub(super) fn argmax(x: &[f32]) -> usize {
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
    let reference = read_reference_logits(&verified(CKPT_V1_MANIFEST, &oracle_dir(), "logits.bin"));
    let captures = read_captures(
        &verified(CKPT_V1_MANIFEST, &oracle_dir(), "default.bin.captures"),
        &["l_out", "hc_attn_post", "indexer_k"],
    );
    let state_captures = read_captures(
        &verified(CKPT_V1_MANIFEST, &oracle_dir(), "state.bin.captures"),
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
    let reference = read_reference_logits(&verified(CKPT_V1_MANIFEST, &oracle_dir(), "logits.bin"));
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
        // Cancellation between packed chunks: a checkpoint that allows two
        // 4-row chunks stops the third; the session stays at position 8,
        // unpoisoned, and finishing the prompt reproduces serial decode
        // (Exact: bitwise).
        let mut cancelled = Glm5NextSession::with_prefill_rows(&ctx, &weights, 256, 4).unwrap();
        cancelled.set_packed_lineage(lineage);
        let mut allowed = 2;
        let error = cancelled
            .prefill_packed_with_checkpoint(&ctx, &tokens, &mut || {
                if allowed == 0 {
                    return Err("test cancel".into());
                }
                allowed -= 1;
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(error, Glm5NextMetalError::Cancelled(_)), "{error}");
        assert_eq!(cancelled.position(), 8);
        assert!(!cancelled.poisoned, "cancellation must not poison");
        let resumed = cancelled.prefill_packed(&ctx, &tokens[8..]).unwrap();
        check(
            &format!("{lineage:?} resumed after cancel"),
            14,
            &resumed,
            lineage,
        );
        drop(cancelled);
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

/// Over 4,300 tokens: the qualification passages and three rounds of the
/// extra passages, for long-context checks where selection excludes about
/// half of the visible pools (sparse-v2).
pub(super) fn long_qualification_text() -> String {
    let mut text = QUALIFICATION_TEXT.to_string();
    for _ in 0..3 {
        for passage in SPARSE_EXTRA_PASSAGES {
            text.push_str("\n\n");
            text.push_str(passage);
        }
    }
    text
}

/// Prints the token ids of [`long_qualification_text`] for oracle runs (CPU
/// only; requires GLM53_GGUF for the tokenizer).
#[test]
#[ignore = "CPU-only helper; requires GLM53_GGUF"]
fn print_long_qualification_tokens() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let tokens = tokenizer.encode(&long_qualification_text(), false).unwrap();
    eprintln!("long qualification tokens ({}): {tokens:?}", tokens.len());
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
        ReferenceStream::open(&verified(
            QUAL_V1_MANIFEST,
            &qual_oracle_dir(),
            "serial.bin",
        )),
        ReferenceStream::open(&verified(QUAL_V1_MANIFEST, &qual_oracle_dir(), "batch.bin")),
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
    let mut serial_ref = ReferenceStream::open(&verified(
        QUAL_V1_MANIFEST,
        &qual_oracle_dir(),
        "serial.bin",
    ));
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
        assert!(
            logit_bits(&[last]) == logit_bits(std::slice::from_ref(&reference_prefill)),
            "exact packed != serial logits"
        );
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

/// Component replay of sparse selection on llama.cpp's own inputs: the
/// native 32-head scorer runs on the captured `indexer_q` (rounded to F16 in
/// the kernel's contract), `indexer_weights` and `indexer_pool_k`, and the
/// native selector on both llama.cpp's and the native scores, at every
/// captured step in all 11 MLA blocks. Bounds were frozen before the first
/// observation:
/// - scores within 1e-5 of the row's largest magnitude;
/// - selection on llama.cpp's scores contains every strict winner and no
///   strict loser (llama.cpp fills threshold ties in atomic order);
/// - selection on native scores equals llama.cpp's set when the 512th/513th
///   gap exceeds twice the measured score error, else agrees on every pool
///   beyond twice that error from the threshold (two scores can move in
///   opposite directions); every set has 512 distinct visible pools;
/// - at visible lengths up to 2051 llama.cpp selects every visible pool
///   (dense equivalence).
///
/// Returns (worst relative score error, sparse selections checked).
fn replay_indexer_captures(manifest: &str, dir: &std::path::Path, steps: &[u32]) -> (f64, usize) {
    let ctx = MetalContext::new().expect("Metal context");
    let names = [
        "indexer_q",
        "indexer_weights",
        "indexer_pool_k",
        "indexer_score",
        "indexer_top_k",
    ];
    let captures = read_captures(&verified(manifest, dir, "capture.bin.captures"), &names);
    const TOP: usize = 512;
    let (h, d) = (32usize, 128usize);
    let mut worst_score_error = 0.0f64;
    let mut exclusions = 0usize;
    for &step in steps {
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
            // A selection: exactly 512 distinct visible pools, containing
            // every pool above the threshold band and none below it.
            let respects = |set: &std::collections::BTreeSet<i32>, band: f32| {
                set.len() == TOP
                    && set
                        .iter()
                        .all(|&id| id >= 0 && (id as usize) < visible_pools)
                    && reference.iter().enumerate().all(|(pool, &s)| {
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
                // Two scores may each move by the measured error in opposite
                // directions, so membership is ambiguous within 2 errors.
                assert!(
                    respects(&on_native, 2.0 * error),
                    "{label}: native scores near the threshold"
                );
            }
        }
    }
    eprintln!(
        "replayed {} positions x 11 MLA blocks: worst relative score error {worst_score_error:.3e}, {exclusions} sparse selections",
        steps.len()
    );
    (worst_score_error, exclusions)
}

/// [`replay_indexer_captures`] at the frontier (sparse-v1, positions
/// 2050-2060: the dense-equivalent control and ten sparse positions).
#[test]
#[ignore = "requires the sparse-v1 llama.cpp captures and Metal"]
fn sparse_selection_replays_llama_cpp_indexer_captures() {
    let steps: Vec<u32> = (2050..=2060).collect();
    let (_, sparse) = replay_indexer_captures(SPARSE_V1_MANIFEST, &sparse_oracle_dir(), &steps);
    assert_eq!(sparse, 10 * MLA_BLOCKS.len());
}

const SPARSE_V2_DEFAULT: &str =
    "/Volumes/wdblack/weights-archive/.fetch/analysis/runs/glm53-oracle/sparse-v2";

fn sparse_v2_dir() -> PathBuf {
    std::env::var_os("GLM53_SPARSE_V2_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(SPARSE_V2_DEFAULT))
}

/// [`replay_indexer_captures`] deep in the sparse range (sparse-v2,
/// positions 4063, 4064, 4079, 4095: about 1016-1024 visible pools, half
/// excluded).
#[test]
#[ignore = "requires the sparse-v2 llama.cpp captures and Metal"]
fn sparse_selection_replays_llama_cpp_captures_near_4096() {
    let steps = [4063, 4064, 4079, 4095];
    let (_, sparse) = replay_indexer_captures(SPARSE_V2_MANIFEST, &sparse_v2_dir(), &steps);
    assert_eq!(sparse, steps.len() * MLA_BLOCKS.len());
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
    let mut reference = ReferenceStream::open(&verified(
        SPARSE_V1_MANIFEST,
        &sparse_oracle_dir(),
        "serial.bin",
    ));
    assert_eq!(reference.remaining, total);
    let mut captured = ReferenceStream::open(&verified(
        SPARSE_V1_MANIFEST,
        &sparse_oracle_dir(),
        "capture.bin",
    ));
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
pub(super) fn perf_lease() -> impl Sized {
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
        let equal = logit_bits(&[logits]) == logit_bits(std::slice::from_ref(&serial_logits));
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
/// <= 2e-2, choice regret <= 0.2 both ways. Also: a session of capacity
/// exactly 2052 (KL <= 1e-2 at 2051, then every entry point refuses), and an
/// injected selection failure that must fail and poison the chunk with
/// per-(block, row) statuses intact.
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
    let mut reference = ReferenceStream::open(&verified(
        SPARSE_V1_MANIFEST,
        &sparse_oracle_dir(),
        "serial.bin",
    ));
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

    // Capacity exactly 2052: the last prompt row is the first sparse row (a
    // one-row microbatch at chunk offset 3); then every entry point refuses
    // another token without moving or poisoning the session.
    const FRONTIER: usize = 2052;
    let mut reference = ReferenceStream::open(&verified(
        SPARSE_V1_MANIFEST,
        &sparse_oracle_dir(),
        "serial.bin",
    ));
    let mut at_frontier = Vec::new();
    for _ in 0..FRONTIER {
        at_frontier = reference.next().2;
    }
    let mut full = Glm5NextSession::with_prefill_rows(&ctx, &weights, FRONTIER, 512).unwrap();
    full.set_packed_lineage(PackedLineage::Exact);
    let logits = full.prefill_packed(&ctx, &tokens[..FRONTIER]).unwrap();
    let kl = kl_divergence(&at_frontier, &logits);
    eprintln!(
        "capacity {FRONTIER}: exact packed vs llama.cpp at {}: kl {kl:.3e}",
        FRONTIER - 1
    );
    if !within(kl, 1e-2) {
        failures.push(format!("capacity {FRONTIER}: kl {kl:.3e}"));
    }
    let next = &tokens[FRONTIER..FRONTIER + 1];
    assert_refused(&mut full, "full forward", |s| s.forward(&ctx, next[0]));
    assert_refused(&mut full, "full advance", |s| s.advance(&ctx, next[0]));
    assert_refused(&mut full, "full prefill", |s| s.prefill(&ctx, next));
    assert_refused(&mut full, "full packed", |s| s.prefill_packed(&ctx, next));
    drop(full);

    // An injected selection failure (zero visible pools for chunk row 10 of
    // the chunk crossing the frontier) fails the chunk in MLA block 0 and
    // poisons the session; the statuses show exactly that row failing in
    // every block while every other sparse row succeeds.
    let mut broken = Glm5NextSession::with_prefill_rows(&ctx, &weights, tokens.len(), 512).unwrap();
    broken.prefill_packed(&ctx, &tokens[..2048]).unwrap();
    broken.corrupt_sparse_row = Some(10);
    let error = broken
        .prefill_packed(&ctx, &tokens[2048..])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("MLA block 0 row 10 sparse selection failed with status 1"),
        "{error}"
    );
    let (statuses, rows) = broken.packed_sparse_statuses().unwrap();
    let sparse_rows = 3..tokens.len() - 2048;
    for block in 0..MLA_BLOCKS.len() {
        for row in sparse_rows.clone() {
            let expected = if row == 10 {
                1
            } else {
                crate::metal::SELECT_STATUS_OK
            };
            assert_eq!(
                statuses[block * rows + row],
                expected,
                "block {block} row {row}"
            );
        }
    }
    assert!(
        broken.poisoned,
        "a failed selection must poison the session"
    );
    assert!(broken.forward(&ctx, tokens[0]).is_err());
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Negative controls for manifest-bound oracle evidence (CPU only): a size,
/// SHA-256 or record-count mismatch is refused; matching metadata passes.
#[test]
fn oracle_evidence_refuses_manifest_mismatches() {
    use sha2::Digest;
    let dir = std::env::temp_dir().join(format!("glm53-manifest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // One GLMCAP01 record: step, layer, occurrence, name, dtype, ne, payload.
    let mut capture = b"GLMCAP01".to_vec();
    for v in [7u32, 3, 0, 4] {
        capture.extend_from_slice(&v.to_le_bytes());
    }
    capture.extend_from_slice(b"name");
    capture.extend_from_slice(&0u32.to_le_bytes());
    capture.extend_from_slice(&[0u8; 32]);
    capture.extend_from_slice(&4i64.to_le_bytes());
    capture.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(dir.join("a.captures"), &capture).unwrap();
    let digest: String = sha2::Sha256::digest(&capture)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let manifest = |bytes: usize, sha: &str, records: usize| {
        format!(
            r#"{{"files": {{"a.captures": {{"bytes": {bytes}, "sha256": "{sha}", "records": {records}}}}}}}"#
        )
    };
    let refused = |m: String| {
        let dir = dir.clone();
        std::panic::catch_unwind(move || drop(verified(&m, &dir, "a.captures"))).is_err()
    };
    assert!(refused(manifest(capture.len() + 1, &digest, 1)), "size");
    assert!(refused(manifest(capture.len(), &"0".repeat(64), 1)), "hash");
    assert!(refused(manifest(capture.len(), &digest, 2)), "records");
    assert_eq!(capture_record_count(&dir.join("a.captures")), 1);
    verified(&manifest(capture.len(), &digest, 1), &dir, "a.captures");
    // Valid first, conflicting second: a cached verification of this path
    // must not answer for a different manifest.
    assert!(
        refused(manifest(capture.len() + 1, &digest, 1)),
        "size after a pass"
    );
    assert!(
        refused(manifest(capture.len(), &"0".repeat(64), 1)),
        "hash after a pass"
    );
    assert!(
        refused(manifest(capture.len(), &digest, 2)),
        "records after a pass"
    );
    verified(&manifest(capture.len(), &digest, 1), &dir, "a.captures");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Long-context sparse execution against llama.cpp (sparse-v2): packed
/// prefill of 4064 tokens (Exact and Fast, 512-row chunks), then 32
/// teacher-forced decode steps (positions 4064-4095), each position's logits
/// against llama.cpp serial decode. About 1016 pools are visible and
/// selection keeps 512. Bounds frozen before the first observation: Exact KL
/// <= 1e-2, Fast KL <= 2e-2, choice regret <= 0.2 both ways, at every
/// position from the prompt end on.
#[test]
#[ignore = "loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, the sparse-v2 oracle and an idle GPU"]
fn sparse_long_context_matches_llama_cpp_near_4096() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    const TOTAL: usize = 4096;
    const PROMPT: usize = 4064;
    const REGRET: f64 = 0.2;
    let tokens: Vec<u32> = tokenizer
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .take(TOTAL)
        .map(|t| t as u32)
        .collect();
    assert_eq!(tokens.len(), TOTAL);
    // Reference logits from the prompt end on.
    let mut reference = ReferenceStream::open(&verified(
        SPARSE_V2_MANIFEST,
        &sparse_v2_dir(),
        "serial.bin",
    ));
    assert_eq!(reference.remaining, TOTAL);
    let mut expected = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        let (p, t, logits) = reference.next();
        assert_eq!((p as usize, t), (position, token), "token {position}");
        if position + 1 >= PROMPT {
            expected.push(logits);
        }
    }
    let mut failures = Vec::new();
    for (lineage, bound) in [(PackedLineage::Exact, 1e-2), (PackedLineage::Fast, 2e-2)] {
        let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, TOTAL, 512).unwrap();
        session.set_packed_lineage(lineage);
        let start = std::time::Instant::now();
        let mut native = vec![session.prefill_packed(&ctx, &tokens[..PROMPT]).unwrap()];
        let prefill_s = start.elapsed().as_secs_f64();
        for &token in &tokens[PROMPT..] {
            native.push(session.forward(&ctx, token).unwrap());
        }
        // native[i] and expected[i] are the logits at position PROMPT - 1 + i.
        assert_eq!(native.len(), expected.len());
        let mut kls = Vec::with_capacity(native.len());
        let mut top1 = 0;
        for (i, (e, n)) in expected.iter().zip(&native).enumerate() {
            let kl = kl_divergence(e, n);
            let (a, b) = choice_regret(e, n);
            top1 += usize::from(argmax(e) == argmax(n));
            kls.push(kl);
            if !within(kl, bound) || !within(f64::from(a), REGRET) || !within(f64::from(b), REGRET)
            {
                failures.push(format!(
                    "{lineage:?} position {}: kl {kl:.3e} regret {a:.3}/{b:.3}",
                    PROMPT - 1 + i
                ));
            }
        }
        let (worst, at) = worst(kls.iter().copied());
        eprintln!(
            "{lineage:?}: prefill {PROMPT} in {prefill_s:.1} s; prompt-end KL {:.3e}; worst KL {worst:.3e} at {}; top-1 {top1}/{}",
            kls[0],
            PROMPT - 1 + at,
            native.len()
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Session preflight against the real artifact before any weight is mapped:
/// a 32k session is admitted; the full checkpoint context is refused with a
/// fitting capacity, which is itself admitted.
#[test]
#[ignore = "requires GLM53_GGUF (header and retained-window plan only) and Metal"]
fn preflight_admits_fitting_sessions_and_reports_the_largest() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let prepared = crate::glm5_next::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let model = prepared.model();
    let admitted = preflight_session(&ctx, &gguf, model, 32_768, 512).unwrap();
    assert!(admitted.ledger.capacity() == 32_768);
    let context = model.config.context_length as usize;
    match preflight_session(&ctx, &gguf, model, context, 512) {
        Err(Glm5NextMetalError::MemoryAdmission {
            required_bytes,
            budget_bytes,
            fitting_capacity: Some(fitting),
            ..
        }) => {
            eprintln!(
                "1M refused: required {required_bytes}, budget {budget_bytes}, fitting {fitting}"
            );
            assert!(required_bytes > budget_bytes);
            assert!((32_768..context as u64).contains(&fitting), "{fitting}");
            preflight_session(&ctx, &gguf, model, fitting as usize, 512).unwrap();
            assert!(preflight_session(&ctx, &gguf, model, fitting as usize + 1, 512).is_err());
        }
        other => panic!("expected a refusal with a fitting capacity, got {other:?}"),
    }
}

/// Lens surface: post-block captures of one decoded token equal ordinary
/// decode, readouts leave every persistent state bit and the position alone,
/// the last block reads out to the token's logits bit for bit. Malformed
/// readout inputs are refused before any work; a finite residual outside the
/// output norm's F32 domain is refused after the tail runs in scratch
/// (a result-domain refusal), and neither moves persistent state.
#[test]
#[ignore = "loads the GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn lens_captures_read_out_without_moving_state() {
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let c = &weights.config;
    let (blocks, width) = (c.executed_block_count(), c.hc_width() as usize);
    let tokens = checkpoint_tokens();
    let mut reference = Glm5NextSession::new(&ctx, &weights, 64).unwrap();
    let mut lensed = Glm5NextSession::new(&ctx, &weights, 64).unwrap();
    for &token in &tokens[..4] {
        reference.forward(&ctx, token).unwrap();
        lensed.forward(&ctx, token).unwrap();
    }
    for bad in [vec![], vec![3, 3], vec![5, 2], vec![blocks]] {
        assert!(
            lensed
                .forward_with_post_block_captures(&ctx, tokens[4], &bad)
                .is_err(),
            "{bad:?}"
        );
        assert_eq!(lensed.position(), 4);
    }
    let all: Vec<u32> = (0..blocks).collect();
    let capture = lensed
        .forward_with_post_block_captures(&ctx, tokens[4], &all)
        .unwrap();
    let expected = reference.forward(&ctx, tokens[4]).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&capture.logits), bits(&expected));
    assert_eq!((capture.position, lensed.position()), (4, 5));
    assert_eq!(capture.residuals.len(), blocks as usize * width);
    assert_finite("captured residuals", &capture.residuals);
    let before = state_bits(&lensed);
    let last = lensed
        .readout(&ctx, capture.site(blocks as usize - 1).unwrap())
        .unwrap();
    assert_eq!(
        bits(&last),
        bits(&expected),
        "last block reads out to logits"
    );
    for site in [0, blocks as usize / 2] {
        assert_finite(
            "early readout",
            &lensed.readout(&ctx, capture.site(site).unwrap()).unwrap(),
        );
    }
    let zeros = lensed.readout(&ctx, &vec![0.0; width]).unwrap();
    assert!(zeros.iter().all(|&v| v == 0.0));
    assert!(lensed.readout(&ctx, &vec![0.0; width - 1]).is_err());
    let mut nan = vec![0.0; width];
    nan[7] = f32::NAN;
    assert!(lensed.readout(&ctx, &nan).is_err());
    // Finite residuals whose output-norm sum of squares overflows F32 (four
    // identical 1e20 streams) are refused after the tail runs (a result-domain
    // refusal, not preflight); a large accepted residual still reads out.
    let refused = lensed.readout(&ctx, &vec![1.0e20; width]).unwrap_err();
    assert!(refused.to_string().contains("sum of squares"), "{refused}");
    assert_finite(
        "large accepted readout",
        &lensed.readout(&ctx, &vec![1.0e15; width]).unwrap(),
    );
    assert_eq!(state_bits(&lensed), before, "readouts moved session state");
    assert_eq!(lensed.position(), 5);
    for &token in &tokens[5..8] {
        assert_eq!(
            bits(&lensed.forward(&ctx, token).unwrap()),
            bits(&reference.forward(&ctx, token).unwrap()),
            "continuation after lens readouts"
        );
    }
    // A stage-profiled step is the same step: bitwise logits and state.
    let (profiled, report) = lensed.forward_stage_profiled(&ctx, tokens[8]).unwrap();
    assert_eq!(
        bits(&profiled),
        bits(&reference.forward(&ctx, tokens[8]).unwrap())
    );
    assert_eq!(state_bits(&lensed), state_bits(&reference));
    assert!(report.spans.len() > 45 * 6, "{}", report.spans.len());
    assert!(report.spans.iter().all(|span| span.gpu_ms >= 0.0));
    assert!(report.span_sum_ms > 0.0 && report.span_sum_ms <= report.command_gpu_ms * 1.001);
    drop((reference, lensed));

    // Readouts between packed chunks and past the sparse frontier leave the
    // continuation bit-identical too. Arbitrary in-vocabulary ids: this is
    // state equivalence, not a quality check.
    let frontier = c.sparse_frontier() as usize;
    let ids: Vec<u32> = (0..frontier + 8)
        .map(|i| 1000 + (i as u32 * 7919) % 50_000)
        .collect();
    let capacity = frontier + 16;
    let mut reference = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, 512).unwrap();
    let mut lensed = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, 512).unwrap();
    for (at, next) in [(600, 900), (frontier - 1, frontier + 4)] {
        let start = reference.position();
        reference.prefill_packed(&ctx, &ids[start..at]).unwrap();
        lensed.prefill_packed(&ctx, &ids[start..at]).unwrap();
        // The capture position is dense at 600 and sparse (visible length
        // 2052) at the frontier.
        let capture = lensed
            .forward_with_post_block_captures(&ctx, ids[at], &[0, blocks / 2, blocks - 1])
            .unwrap();
        let expected = reference.forward(&ctx, ids[at]).unwrap();
        assert_eq!(bits(&capture.logits), bits(&expected), "capture at {at}");
        for site in 0..3 {
            lensed.readout(&ctx, capture.site(site).unwrap()).unwrap();
        }
        assert_eq!(
            state_bits(&lensed),
            state_bits(&reference),
            "state after readouts at {at}"
        );
        assert_eq!(
            bits(&lensed.prefill_packed(&ctx, &ids[at + 1..next]).unwrap()),
            bits(&reference.prefill_packed(&ctx, &ids[at + 1..next]).unwrap()),
            "packed continuation after readouts at {at}"
        );
        assert_eq!(
            bits(&lensed.forward(&ctx, ids[next]).unwrap()),
            bits(&reference.forward(&ctx, ids[next]).unwrap()),
            "decode continuation after readouts at {at}"
        );
    }
}

/// Acquisition for the sampler replay (`qwen-bench sampler-replay`): real
/// full-vocabulary logits rows from release-preset chat decoding, written as
/// little-endian F32 rows to `GLM53_LOGITS_OUT` with a JSON sidecar. Not a
/// gate; it only records inputs for CPU timing.
#[test]
#[ignore = "acquisition: loads the GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF, GLM53_LOGITS_OUT and an idle GPU"]
fn capture_release_sampling_logits() {
    use crate::glm5_next_chat::{Effort, Message, RenderOptions, render};
    use crate::sampling::{Sampler, SamplingConfig};
    use std::io::Write;
    let _lease = production_lease();
    let out = PathBuf::from(std::env::var("GLM53_LOGITS_OUT").expect("GLM53_LOGITS_OUT"));
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let stops = artifact.generation_stops().unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let prompts = [
        ("Write a short poem about the sea.", Effort::High),
        (
            "Explain how a hash map handles collisions, in two paragraphs.",
            Effort::Low,
        ),
        ("What is 17 * 23? Show your work briefly.", Effort::Max),
    ];
    let steps = 48;
    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut records = Vec::new();
    for (index, (question, effort)) in prompts.iter().enumerate() {
        let text = render(
            &[Message::User((*question).into())],
            RenderOptions::generate(*effort, false),
        )
        .unwrap();
        let tokens: Vec<u32> = artifact
            .tokenizer()
            .encode(&text, false)
            .unwrap()
            .into_iter()
            .map(|id| id as u32)
            .collect();
        let mut session =
            Glm5NextSession::with_prefill_rows(&ctx, &weights, tokens.len() + steps, tokens.len())
                .unwrap();
        let mut logits = session.prefill_packed(&ctx, &tokens).unwrap();
        let config = SamplingConfig::glm5_next(1000 + index as u64);
        let mut sampler = Sampler::new(config).unwrap();
        let mut sampled = Vec::new();
        for _ in 0..steps {
            for value in &logits {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            let token = sampler.sample(&logits).unwrap().token;
            sampled.push(token);
            if stops.contains(&token) {
                break;
            }
            logits = session.forward(&ctx, token as u32).unwrap();
        }
        records.push(serde_json::json!({
            "prompt": question, "effort": effort, "prompt_tokens": tokens.len(),
            "seed": config.seed, "rows": sampled.len(), "sampled": sampled,
        }));
    }
    file.flush().unwrap();
    let rows: usize = records
        .iter()
        .map(|r| r["rows"].as_u64().unwrap() as usize)
        .sum();
    let sidecar = serde_json::json!({
        "schema": "qwen.sampler_replay_logits.v1", "family": "glm5-next",
        "vocab": weights.config.vocab_size, "rows": rows, "dtype": "f32_le",
        "sampler": "SamplingConfig::glm5_next", "requests": records,
    });
    std::fs::write(
        out.with_extension("json"),
        serde_json::to_vec_pretty(&sidecar).unwrap(),
    )
    .unwrap();
    eprintln!("wrote {rows} rows to {}", out.display());
}

/// Per-stage bytes of executed weights read by one decode step, from GGUF
/// tensor names: routed experts count top-k of the expert stack, the
/// embedding one row; the sparse query weights only past the frontier.
fn stage_weight_bytes(gguf: &GgufFile, c: &Glm5NextConfig) -> BTreeMap<&'static str, f64> {
    let mut bytes = BTreeMap::new();
    let executed = c.executed_block_count() as usize;
    for tensor in &gguf.tensors {
        let name = tensor.name.as_str();
        let n = tensor.n_bytes as f64;
        let (stage, scale) = if name.starts_with("token_embd") {
            ("embed", 1.0 / c.vocab_size as f64)
        } else if name.starts_with("output") {
            ("head", 1.0)
        } else if let Some(rest) = name.strip_prefix("blk.") {
            let (block, rest) = rest.split_once('.').unwrap();
            let block: usize = block.parse().unwrap();
            if block >= executed {
                continue;
            }
            let mla = c.blocks[block].mixer == MixerKind::Mla;
            let stage = if rest.starts_with("hc_attn_") || rest.starts_with("attn_norm") {
                "attention_pre"
            } else if rest.starts_with("hc_ffn_") || rest.starts_with("ffn_norm") {
                "ffn_pre"
            } else if rest.starts_with("indexer.attn_q_b") || rest.starts_with("indexer.proj") {
                "sparse_query"
            } else if rest.starts_with("indexer") {
                "mla_indexer"
            } else if mla && (rest.starts_with("attn_v_b") || rest.starts_with("attn_output")) {
                "mla_output"
            } else if mla && rest.starts_with("attn_") {
                "mla_projection"
            } else if rest.starts_with("attn_") || rest.starts_with("ssm_") {
                "kda"
            } else if rest.starts_with("ffn_gate_inp") || rest.starts_with("exp_probs_b") {
                "router"
            } else if rest.contains("_exps") {
                *bytes.entry("routed_experts").or_insert(0.0) +=
                    n * c.expert_used_count as f64 / c.expert_count as f64;
                continue;
            } else if rest.contains("_shexp") {
                "shared_expert"
            } else if rest.starts_with("ffn_") {
                "dense_ffn"
            } else {
                "unmapped"
            };
            (stage, 1.0)
        } else {
            ("unmapped", 1.0)
        };
        *bytes.entry(stage).or_insert(0.0) += n * scale;
    }
    bytes
}

/// Leverage map 2026-10-05 #2: decode attribution. At depths 64 and 4096,
/// unprofiled per-token wall time, then profiled steps (one sampled encoder
/// per stage) aggregated per stage kind, beside a per-stage weight-byte
/// model and the effective bandwidth it implies. Writes JSON to
/// `GLM53_STAGE_OUT`. Timing, not qualification: no bounds.
#[test]
#[ignore = "timing: loads the GLM-5.3 trunk; requires GLM53_GGUF, GLM53_STAGE_OUT, no MTL_DEBUG_LAYER and an idle GPU"]
fn decode_stage_attribution() {
    let _lease = perf_lease();
    let out = PathBuf::from(std::env::var("GLM53_STAGE_OUT").expect("GLM53_STAGE_OUT"));
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{}", long_qualification_text()), false)
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let c = &weights.config;
    let bytes = stage_weight_bytes(&gguf, c);
    let (steps, warm) = (12usize, 4usize);
    let capacity = 4096 + 2 * (steps + warm) + 8;
    assert!(tokens.len() >= capacity, "{} tokens", tokens.len());
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, 512).unwrap();
    let mut depths = Vec::new();
    for depth in [64usize, 4096] {
        let start = session.position();
        session.prefill_packed(&ctx, &tokens[start..depth]).unwrap();
        let mut next = depth;
        for _ in 0..warm {
            session.forward(&ctx, tokens[next]).unwrap();
            next += 1;
        }
        let mut wall = Vec::new();
        for _ in 0..steps {
            let started = std::time::Instant::now();
            session.forward(&ctx, tokens[next]).unwrap();
            wall.push(started.elapsed().as_secs_f64() * 1e3);
            next += 1;
        }
        let mut stage_ms: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut stage_spans: BTreeMap<&'static str, usize> = BTreeMap::new();
        let (mut command, mut span_sum) = (0.0, 0.0);
        for _ in 0..steps {
            let (_, report) = session.forward_stage_profiled(&ctx, tokens[next]).unwrap();
            next += 1;
            command += report.command_gpu_ms / steps as f64;
            span_sum += report.span_sum_ms / steps as f64;
            for span in &report.spans {
                *stage_ms.entry(span.stage.as_str()).or_default() += span.gpu_ms / steps as f64;
                *stage_spans.entry(span.stage.as_str()).or_default() += 1;
            }
        }
        let wall_ms = wall.iter().sum::<f64>() / wall.len() as f64;
        let stages: serde_json::Map<String, serde_json::Value> = stage_ms
            .iter()
            .map(|(&stage, &ms)| {
                let b = bytes.get(stage).copied().unwrap_or(0.0);
                let gb_s = if ms > 0.0 { b / (ms * 1e-3) / 1e9 } else { 0.0 };
                (
                    stage.to_string(),
                    serde_json::json!({"ms": ms, "share_of_spans": ms / span_sum,
                        "spans_per_step": stage_spans[stage] / steps, "weight_bytes": b,
                        "effective_gb_s": gb_s}),
                )
            })
            .collect();
        eprintln!(
            "depth {depth}: unprofiled {wall_ms:.2} ms/token; profiled command {command:.2} ms, span sum {span_sum:.2} ms"
        );
        let mut ranked: Vec<_> = stage_ms.iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(a.1));
        for (stage, ms) in ranked {
            let b = bytes.get(stage).copied().unwrap_or(0.0);
            eprintln!(
                "  {stage:18} {ms:7.3} ms  {:5.1}%  {:8.1} MB  {:6.1} GB/s",
                100.0 * ms / span_sum,
                b / 1e6,
                if *ms > 0.0 {
                    b / (ms * 1e-3) / 1e9
                } else {
                    0.0
                }
            );
        }
        depths.push(serde_json::json!({
            "depth": depth, "steps": steps, "unprofiled_wall_ms_per_token": wall_ms,
            "unprofiled_wall_ms": wall, "profiled_command_gpu_ms": command,
            "profiled_span_sum_ms": span_sum, "stages": stages,
        }));
    }
    let document = serde_json::json!({
        "schema": "glm53.decode_stage_attribution.v1",
        "method": "one command per step, one timestamp-sampled encoder per stage; spans scaled to the command's GPU time",
        "prompt": "[gMASK]<sop> + long_qualification_text (teacher-forced)",
        "weight_bytes_per_step": bytes, "depths": depths,
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
}

/// Per-position reuse comparison of `warm` against `cold` logits at the join
/// and teacher-forced continuation positions: KL both ways and both choice
/// regrets, with bounds frozen before observation (map #12, cx session
/// 01a1046). `positions` is the expected count (the join plus the
/// continuation). Returns (worst KL, worst regret, top-1 agreements, bitwise).
fn reuse_gate(
    label: &str,
    positions: usize,
    cold: &[Vec<f32>],
    warm: &[Vec<f32>],
    failures: &mut Vec<String>,
) -> (f64, f32, usize, bool) {
    const KL_BOUND: f64 = 2e-2;
    const REGRET_BOUND: f32 = 0.2;
    assert_eq!(cold.len(), positions, "{label}: cold position count");
    assert_eq!(warm.len(), positions, "{label}: warm position count");
    let (mut kl_worst, mut regret_worst, mut agree) = (0.0f64, 0.0f32, 0);
    let mut table = Vec::new();
    for (position, (c, w)) in cold.iter().zip(warm).enumerate() {
        assert_finite(&format!("{label} cold {position}"), c);
        assert_finite(&format!("{label} warm {position}"), w);
        let (forward, reverse) = (kl_divergence(c, w), kl_divergence(w, c));
        let kl = forward.max(reverse);
        let (r0, r1) = choice_regret(c, w);
        table.push(format!("{position}:{forward:.1e}/{reverse:.1e}"));
        // Every violation is recorded; the test fails once, after all cases.
        if !(within(kl, KL_BOUND) && r0 <= REGRET_BOUND && r1 <= REGRET_BOUND) {
            failures.push(format!(
                "{label} position {position}: KL {forward:.3e} cold||warm, {reverse:.3e} warm||cold (bound {KL_BOUND:e}), regret {r0:.3}/{r1:.3} (bound {REGRET_BOUND})"
            ));
        }
        kl_worst = kl_worst.max(kl);
        regret_worst = regret_worst.max(r0).max(r1);
        agree += usize::from(argmax(c) == argmax(w));
    }
    let bitwise = logit_bits(cold) == logit_bits(warm);
    eprintln!(
        "{label}: {} positions, worst KL {kl_worst:.3e}, worst regret {regret_worst:.3}, top-1 {agree}/{}, bitwise {bitwise}",
        cold.len(),
        cold.len()
    );
    eprintln!(
        "  KL cold||warm / warm||cold by position: {}",
        table.join(" ")
    );
    (kl_worst, regret_worst, agree, bitwise)
}

/// Diagnostic only (asserts nothing): how far `other` is from the Exact
/// `reference` per position, in the Fast policy's own direction
/// (reference||other), with top-1 flips and their regrets.
fn reference_report(label: &str, reference: &[Vec<f32>], other: &[Vec<f32>]) {
    let kls: Vec<f64> = reference
        .iter()
        .zip(other)
        .map(|(r, o)| kl_divergence(r, o))
        .collect();
    let flips: Vec<(usize, (f32, f32))> = reference
        .iter()
        .zip(other)
        .enumerate()
        .map(|(position, (r, o))| (position, choice_regret(r, o)))
        .filter(|(_, regret)| *regret != (0.0, 0.0))
        .collect();
    let (worst, at) = worst(kls.iter().copied());
    eprintln!(
        "  {label} vs Exact: worst KL {worst:.3e} at {at}, mean {:.3e}, top-1 flips (position, regret exact/other) {flips:?}",
        kls.iter().sum::<f64>() / kls.len() as f64
    );
}

/// KNOWN FAILING QUALIFICATION (map #12, 2026-10-06). Kept as the
/// reproducer, with its frozen bounds and assertions unchanged. Fast reuse
/// exceeds the warm/cold tolerance on these held-out prompts, and both cold
/// and warm Fast also exceed the existing Exact-reference policy (see the
/// diagnostic Exact report and `docs/bench/2026-10-06-glm53-review-
/// qualification/`). Cancellation and resume on an unchanged schedule are
/// bitwise. The general Fast-lineage qualification remains open.
///
/// Map #12: serve's default Fast packed lineage across live-session reuse.
/// Three teacher-forced cases, each compared at the join and 32 continuation
/// positions against one cold Fast prefill of the whole prompt:
///
/// 1. Ordinary continuation: Fast prefill of a 700-token prompt, 40 serially
///    decoded "generated" tokens, then a Fast prefill of the next-turn suffix
///    (to 1300, off the 512-row grid).
/// 2. Tool continuation across the sparse frontier: a rendered tools
///    conversation padded so the prompt ends 60-90 tokens below it; the
///    frozen generated call is decoded serially and still ends below it
///    (asserted); the next prompt (call results under `<|observation|>`)
///    must extend those tokens exactly, and its suffix alone crosses into
///    sparse attention (asserted).
/// 3. Cancellation: a Fast prefill cancelled at its second chunk boundary
///    and resumed from the committed position replays the uninterrupted
///    chunk schedule, so it must be bitwise equal to the cold run.
///
/// Every session selects the Fast lineage explicitly (asserted), and each
/// case compares exactly 33 positions. Bounds (frozen before observation):
/// per position, KL both ways <= 2e-2 and both choice regrets <= 0.2. Not
/// required for cases 1-2: bitwise equality, identical sampled text, or
/// identical recurrent state (Fast is numerical by design; the Exact
/// lineage keeps its own bitwise warm/cold check in serve).
#[test]
#[ignore = "known failing qualification (map #12); loads the 109.5 GiB GLM-5.3 trunk; requires MTL_DEBUG_LAYER=1, GLM53_GGUF and an idle GPU"]
fn fast_reuse_stays_within_the_fast_policy_across_reuse_boundaries() {
    use crate::glm5_next_chat::{
        self as chat, Effort, Message, RenderOptions, ToolCall, ToolDefinition,
    };
    const CONTINUATION: usize = 32;
    let _lease = production_lease();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(&path).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load weights");
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .unwrap()
            .into_iter()
            .map(|t| t as u32)
            .collect()
    };
    let frontier = weights.config.sparse_frontier() as usize;
    let long = encode(&long_qualification_text());
    let continuation: Vec<u32> = long[3000..3000 + CONTINUATION].to_vec();
    let positions = CONTINUATION + 1;
    let mut failures = Vec::new();
    // Serve's default lineage, selected explicitly rather than inherited.
    let fast_session = |capacity: usize| {
        let mut session =
            Glm5NextSession::with_prefill_rows(&ctx, &weights, capacity, 512).unwrap();
        session.set_packed_lineage(PackedLineage::Fast);
        assert_eq!(
            session.packed.as_ref().map(|packed| packed.lineage),
            Some(PackedLineage::Fast)
        );
        session
    };

    // Cold: one Fast prefill of the whole prompt, then the continuation.
    let cold = |prompt: &[u32]| -> Vec<Vec<f32>> {
        let mut session = fast_session(prompt.len() + CONTINUATION + 1);
        let mut logits = vec![session.prefill_packed(&ctx, prompt).unwrap()];
        for &token in &continuation {
            logits.push(session.forward(&ctx, token).unwrap());
        }
        logits
    };
    // Diagnostic reference: the same cold run in the Exact lineage (packed
    // Exact matches serial), to tell which Fast run moved, not only that the
    // two differ. Nothing is asserted on it.
    let exact = |prompt: &[u32]| -> Vec<Vec<f32>> {
        let mut session = Glm5NextSession::with_prefill_rows(
            &ctx,
            &weights,
            prompt.len() + CONTINUATION + 1,
            512,
        )
        .unwrap();
        session.set_packed_lineage(PackedLineage::Exact);
        let mut logits = vec![session.prefill_packed(&ctx, prompt).unwrap()];
        for &token in &continuation {
            logits.push(session.forward(&ctx, token).unwrap());
        }
        logits
    };
    // Warm: Fast prefill of the first turn, serial decode of the generated
    // tokens, Fast prefill of the suffix, then the continuation.
    let warm = |first: &[u32], generated: &[u32], suffix: &[u32]| -> Vec<Vec<f32>> {
        let capacity = first.len() + generated.len() + suffix.len() + CONTINUATION + 1;
        let mut session = fast_session(capacity);
        session.prefill_packed(&ctx, first).unwrap();
        for &token in generated {
            session.forward(&ctx, token).unwrap();
        }
        assert_eq!(session.position(), first.len() + generated.len());
        let mut logits = vec![session.prefill_packed(&ctx, suffix).unwrap()];
        for &token in &continuation {
            logits.push(session.forward(&ctx, token).unwrap());
        }
        logits
    };

    // 1. Ordinary continuation.
    let (first, generated, suffix) = (&long[..700], &long[700..740], &long[740..1300]);
    let joined: Vec<u32> = [first, generated, suffix].concat();
    let (cold_logits, warm_logits) = (cold(&joined), warm(first, generated, suffix));
    reuse_gate(
        "ordinary",
        positions,
        &cold_logits,
        &warm_logits,
        &mut failures,
    );
    let reference = exact(&joined);
    reference_report("ordinary cold", &reference, &cold_logits);
    reference_report("ordinary warm", &reference, &warm_logits);

    // 2. Tool continuation through <|observation|>, crossing the frontier.
    let tool = ToolDefinition::from_value(&serde_json::json!({"name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}))
    .unwrap();
    let options = RenderOptions::generate(Effort::Low, false);
    let text = long_qualification_text();
    let user = |chars: usize| {
        let mut end = chars.min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        format!(
            "{}\n\nWhat's the weather in Paris? Use the tool.",
            &text["[gMASK]<sop>".len()..end]
        )
    };
    // Pad the user turn until the first prompt ends 60-90 tokens below the
    // frontier, so the generated call still ends below it and the results
    // alone cross it.
    let mut chars = text.len();
    let mut attempts = 0;
    let first = loop {
        attempts += 1;
        assert!(attempts <= 32, "padding did not converge near the frontier");
        let rendered = chat::render_with_tools(
            &[Message::User(user(chars))],
            std::slice::from_ref(&tool),
            options,
        )
        .unwrap();
        let tokens = encode(&rendered);
        if tokens.len() < frontier - 60 && tokens.len() > frontier - 90 {
            break (user(chars), tokens);
        }
        let excess = tokens.len() as isize - (frontier as isize - 75);
        chars = (chars as isize - excess * 3).max(1) as usize;
    };
    let (user_text, first_tokens) = first;
    let generated_text = "need the weather</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    let generated_tokens = encode(generated_text);
    let messages = vec![
        Message::User(user_text),
        Message::Assistant {
            content: String::new(),
            reasoning: Some("need the weather".into()),
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"})
                    .as_object()
                    .unwrap()
                    .clone(),
            }],
        },
        // Long enough that the results alone cross the frontier.
        Message::Tool {
            call_id: "c1".into(),
            content: "Paris: 18C, clear skies, light wind from the west. ".repeat(24),
        },
    ];
    let next =
        encode(&chat::render_with_tools(&messages, std::slice::from_ref(&tool), options).unwrap());
    let consumed = first_tokens.len() + generated_tokens.len();
    assert_eq!(
        &next[..consumed],
        [first_tokens.as_slice(), generated_tokens.as_slice()].concat(),
        "the tool-result prompt must extend the consumed history"
    );
    // The consumed history (prompt plus generated call) ends below the
    // frontier, so the suffix prefill itself crosses it.
    assert!(
        consumed < frontier && next.len() > frontier + 32,
        "first {} generated {} consumed {consumed} next {} frontier {frontier}",
        first_tokens.len(),
        generated_tokens.len(),
        next.len()
    );
    eprintln!(
        "tool geometry: first {} + generated {} = consumed {consumed} < frontier {frontier} < next {}",
        first_tokens.len(),
        generated_tokens.len(),
        next.len()
    );
    let (cold_logits, warm_logits) = (
        cold(&next),
        warm(&first_tokens, &generated_tokens, &next[consumed..]),
    );
    reuse_gate(
        "tool loop across the frontier",
        positions,
        &cold_logits,
        &warm_logits,
        &mut failures,
    );
    let reference = exact(&next);
    reference_report("tool loop cold", &reference, &cold_logits);
    reference_report("tool loop warm", &reference, &warm_logits);

    // 3. Cancellation at the second chunk boundary, then resume.
    let prompt = &long[..1300];
    let mut session = fast_session(prompt.len() + CONTINUATION + 1);
    let mut calls = 0;
    let cancelled = session.prefill_packed_with_checkpoint(&ctx, prompt, &mut || {
        calls += 1;
        if calls == 2 {
            Err("cancel".into())
        } else {
            Ok(())
        }
    });
    assert!(matches!(cancelled, Err(Glm5NextMetalError::Cancelled(_))));
    let committed = session.position();
    assert_eq!(committed, 512, "cancelled after exactly one 512-row chunk");
    let mut resumed = vec![session.prefill_packed(&ctx, &prompt[committed..]).unwrap()];
    for &token in &continuation {
        resumed.push(session.forward(&ctx, token).unwrap());
    }
    // Resuming replays the uninterrupted chunk schedule (512 | 512 | 276),
    // so this case is held to bitwise equality, not only the numerical gate.
    let (.., bitwise) = reuse_gate(
        "cancel and resume",
        positions,
        &cold(prompt),
        &resumed,
        &mut failures,
    );
    if !bitwise {
        failures.push("cancel and resume: not bitwise on the same chunk schedule".into());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

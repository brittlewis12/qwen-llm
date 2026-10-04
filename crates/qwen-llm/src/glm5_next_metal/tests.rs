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

/// The P2 end-to-end checkpoint: the ckpt-v1 token IDs through all 45 blocks,
/// compared with the same-artifact llama.cpp oracle (logits per position and
/// per-block residual streams), plus peak allocation and token latency.
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
    let mut session = Glm5NextSession::new(&ctx, &weights, 256).expect("session");
    let reference = read_reference_logits(&oracle_dir().join("logits.bin"));
    let captures = read_captures(
        &oracle_dir().join("default.bin.captures"),
        &["l_out", "hc_attn_post"],
    );
    assert_eq!(reference.len(), tokens.len());
    let mut worst_kl = 0.0f64;
    let mut top1 = 0;
    for (position, &token) in tokens.iter().enumerate() {
        assert_eq!(reference[position].0, token);
        let mut block_errors = Vec::new();
        let start = std::time::Instant::now();
        let logits = session
            .forward_observed(
                &ctx,
                token,
                &[
                    Glm5NextProbe::AttentionResidual,
                    Glm5NextProbe::BlockResidual,
                ],
                &mut |probe, block, values| {
                    let (name, occurrence) = match probe {
                        Glm5NextProbe::AttentionResidual => ("hc_attn_post", 0),
                        Glm5NextProbe::BlockResidual => ("l_out", 0),
                    };
                    if let Some(expected) =
                        captures.get(&(name.to_string(), position as u32, block as u32, occurrence))
                    {
                        block_errors.push((name, block, relative_error(values, expected)));
                    }
                },
            )
            .unwrap_or_else(|e| panic!("position {position}: {e}"));
        let elapsed = start.elapsed();
        let expected = &reference[position].1;
        let kl = kl_divergence(expected, &logits);
        let max_abs = logits
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let agree = argmax(&logits) == argmax(expected);
        top1 += usize::from(agree);
        worst_kl = worst_kl.max(kl);
        let worst_block = block_errors
            .iter()
            .cloned()
            .fold(("", 0, 0.0f64), |w, e| if e.2 > w.2 { e } else { w });
        eprintln!(
            "pos {position:2} token {token:6}: top1 {} kl {kl:.3e} max|dlogit| {max_abs:.3e} worst residual {}@{} {:.3e} ({:.1} ms observed)",
            if agree { "ok" } else { "MISS" },
            worst_block.0,
            worst_block.1,
            worst_block.2,
            elapsed.as_secs_f64() * 1e3
        );
        if position == 0 {
            for (name, block, error) in &block_errors {
                eprintln!("  pos0 {name} block {block:2}: rel {error:.3e}");
            }
        }
    }
    // Unobserved token latency and peak allocation.
    let mut timing_session = Glm5NextSession::new(&ctx, &weights, 256).expect("timing session");
    let start = std::time::Instant::now();
    for &token in &tokens {
        timing_session.forward(&ctx, token).unwrap();
    }
    eprintln!(
        "unobserved: {:.1} ms/token; allocated {} bytes; ledger decode peak {}",
        start.elapsed().as_secs_f64() * 1e3 / tokens.len() as f64,
        ctx.current_allocated_size(),
        session.ledger().phase_peaks().decode
    );
    eprintln!("top1 {top1}/{} worst KL {worst_kl:.3e}", tokens.len());
    assert_eq!(top1, tokens.len(), "greedy top-1 disagreement");
    assert!(worst_kl < 5e-3, "KL {worst_kl}");
}

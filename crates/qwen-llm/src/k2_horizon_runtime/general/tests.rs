//! GPU equivalence of the general batched prefill against token-by-token
//! prefill on the same loaded checkpoint. Run with an explicit checkpoint:
//! `K2_GGUF=/path/K2-Horizon-7B-Q4_K_M.gguf cargo test -p qwen-llm --lib \
//!  k2_general_batched_prefill_matches_serial -- --ignored --test-threads=1`.
//!
//! Tiled mat-mat projections reorder accumulation relative to mat-vec, so
//! these gates are tolerances, not bitwise: final logits cosine/max-abs and
//! argmax, and every stored K/V row (all layers) by cosine and max-abs.

use super::*;

const LOGIT_COSINE: f64 = 0.9999;
const LOGIT_MAX_ABS: f64 = 0.25;
const KV_COSINE: f64 = 0.999;
const KV_MAX_ABS: f64 = 0.25;

/// Deterministic, non-periodic IDs inside the ordinary vocabulary range.
fn prompt(len: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1000 + ((state >> 33) % 60000) as u32
        })
        .collect()
}

#[derive(Debug)]
struct Metrics {
    cosine: f64,
    max_abs: f64,
    argmax: (usize, usize),
}

fn metrics(actual: &[f32], expected: &[f32]) -> Metrics {
    assert_eq!(actual.len(), expected.len());
    let (mut dot, mut aa, mut bb, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        aa += a * a;
        bb += b * b;
        max_abs = max_abs.max((a - b).abs());
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    Metrics {
        cosine: if aa == 0.0 && bb == 0.0 {
            1.0
        } else {
            dot / (aa.sqrt() * bb.sqrt())
        },
        max_abs,
        argmax: (argmax(actual), argmax(expected)),
    }
}

/// F16 K and V planes for rows `0..rows` of every layer, as F32.
fn kv_rows(session: &K2Session<'_, '_>, rows: u32) -> KvRows {
    let cache = &session.buffers.cache;
    assert_eq!(cache.offset, 0);
    // Checked shared F16 arena; no command is in flight here.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            cache.buffer.contents().as_ptr().cast::<u8>(),
            cache.n_bytes() as usize,
        )
    };
    let decode = |range: std::ops::Range<u64>| {
        let end = range.start + u64::from(rows) * session.request.row_bytes();
        assert!(end <= range.end);
        bytes[range.start as usize..end as usize]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| half::f16::from_le_bytes(b).to_f32())
            .collect::<Vec<_>>()
    };
    (0..36)
        .map(|layer| {
            let planes = session.request.layer_planes(layer).unwrap();
            (decode(planes.key), decode(planes.value))
        })
        .collect()
}

fn assert_logits(label: &str, actual: &[f32], expected: &[f32]) {
    let m = metrics(actual, expected);
    eprintln!("{label} logits {m:?}");
    assert!(m.cosine >= LOGIT_COSINE, "{label} logits {m:?}");
    assert!(m.max_abs <= LOGIT_MAX_ABS, "{label} logits {m:?}");
    assert_eq!(m.argmax.0, m.argmax.1, "{label} logits {m:?}");
}

type KvRows = Vec<(Vec<f32>, Vec<f32>)>;

fn assert_kv(label: &str, actual: &KvRows, expected: &KvRows, rows: u32) {
    let mut worst = (1.0f64, 0.0f64);
    for (layer, ((ak, av), (ek, ev))) in actual.iter().zip(expected).enumerate() {
        for (plane, a, e) in [("K", ak, ek), ("V", av, ev)] {
            let m = metrics(a, e);
            worst = (worst.0.min(m.cosine), worst.1.max(m.max_abs));
            assert!(
                m.cosine >= KV_COSINE && m.max_abs <= KV_MAX_ABS,
                "{label} layer {layer} {plane} {m:?}"
            );
        }
    }
    eprintln!(
        "{label} kv rows={rows} worst_cosine={:.8} worst_max_abs={:.6}",
        worst.0, worst.1
    );
}

fn serial_session<'m, 'c>(
    model: &'m mut K2LoadedModel<'c>,
    base: u32,
    tokens: &[u32],
) -> (K2Session<'m, 'c>, Vec<f32>) {
    model.prefill = PrefillMode::Serial;
    let model: &'m K2LoadedModel<'c> = model;
    let mut session = model.create_session(base).unwrap();
    let mut logits = Vec::new();
    for &token in tokens {
        logits = session.append(&[token]).unwrap();
    }
    (session, logits)
}

#[test]
#[ignore = "K2_GGUF GPU equivalence: general batched prefill vs token-by-token, production lease"]
fn k2_general_batched_prefill_matches_serial_default_selection() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load(&ctx, &source, 512).unwrap();
    assert_eq!(model.prefill, PrefillMode::General { chunk: 256 });
    let info = model.prefill_info(4408);
    assert_eq!(info.mode, "general_matmat_batch");
    assert_eq!(info.chunk_tokens, 256);
    assert_eq!(info.commands, 18);
    assert_eq!(model.prefill_info(1).mode, "serial_single_token");
}

#[test]
#[ignore = "K2_GGUF GPU equivalence: general batched prefill vs token-by-token, production lease"]
fn k2_general_batched_prefill_matches_serial_long_prompt() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    // 600 rows: one full 256 chunk, one crossing the 256/512 online-block
    // boundaries, and a ragged tail; attention spans multiple merged blocks.
    let tokens = prompt(600, 0x6b32);
    let mut model = K2LoadedModel::load(&ctx, &source, 601).unwrap();
    let (mut serial, serial_logits) = serial_session(&mut model, 0, &tokens);
    let serial_next = serial.append(&[tokens[7]]).unwrap();
    // Sessions are one-per-model; keep the serial cache as host copies.
    let expected_kv = kv_rows(&serial, 601);
    drop(serial);

    model.prefill = PrefillMode::General { chunk: 256 };
    let mut general = model.create_session(0).unwrap();
    let logits = general.append(&tokens).unwrap();
    assert_eq!(general.committed_len(), 600);
    assert_logits("long/general-256", &logits, &serial_logits);
    // Decode after a batched prefill stays on the serial single-token graph.
    let next = general.append(&[tokens[7]]).unwrap();
    assert_logits("long/next-token", &next, &serial_next);
    assert_kv("long", &kv_rows(&general, 601), &expected_kv, 601);
}

#[test]
#[ignore = "K2_GGUF GPU equivalence: general batched prefill vs token-by-token, production lease"]
fn k2_general_batched_prefill_matches_serial_offsets_partitions_and_chunks() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    let tokens = prompt(300, 0x1d);
    let base = 37;
    let mut model = K2LoadedModel::load(&ctx, &source, 300).unwrap();
    // Control: token-by-token, keeping logits at each partition boundary.
    model.prefill = PrefillMode::Serial;
    let boundaries = [2usize, 3, 35, 36, 100, 257, 299, 300];
    let mut controls = Vec::new();
    {
        let mut serial = model.create_session(base).unwrap();
        for (index, &token) in tokens.iter().enumerate() {
            let logits = serial.append(&[token]).unwrap();
            if boundaries.contains(&(index + 1)) {
                controls.push((index + 1, logits));
            }
        }
    }
    let control = |end: usize| &controls.iter().find(|(n, _)| *n == end).unwrap().1;
    for chunk in [2usize, 17, 256] {
        model.prefill = PrefillMode::General { chunk };
        let mut session = model.create_session(base).unwrap();
        let mut start = 0;
        // Includes 1-token appends (serial graph) between batched ones, so
        // batched chunks start at nonzero cache offsets and odd positions.
        for end in boundaries {
            let logits = session.append(&tokens[start..end]).unwrap();
            assert_logits(
                &format!("chunk={chunk} base={base} end={end}"),
                &logits,
                control(end),
            );
            start = end;
        }
        assert_eq!(session.committed_len(), 300);
    }
    // KV equivalence for one mixed partition against a fresh serial cache.
    model.prefill = PrefillMode::General { chunk: 256 };
    let mut general = model.create_session(base).unwrap();
    general.advance(&tokens[..1]).unwrap();
    general.advance(&tokens[1..290]).unwrap();
    general.append(&tokens[290..]).unwrap();
    let general_kv = kv_rows(&general, 300);
    drop(general);
    let (serial, _) = serial_session(&mut model, base, &tokens);
    let serial_kv = kv_rows(&serial, 300);
    drop(serial);
    assert_kv("offsets", &general_kv, &serial_kv, 300);
}

#[test]
#[ignore = "K2_GGUF GPU equivalence: general batched prefill vs token-by-token, production lease"]
fn k2_general_batched_prefill_matches_serial_captures_and_readout() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    let tokens = prompt(64, 0x77);
    let layers = [0u32, 17, 35];
    let mut model = K2LoadedModel::load(&ctx, &source, 64).unwrap();
    model.prefill = PrefillMode::Serial;
    let expected = {
        let mut serial = model.create_session(0).unwrap();
        serial.advance(&tokens[..63]).unwrap();
        serial.append_with_captures(&tokens[63..], &layers).unwrap()
    };
    model.prefill = PrefillMode::General { chunk: 256 };
    let mut general = model.create_session(0).unwrap();
    let actual = general.append_with_captures(&tokens, &layers).unwrap();
    assert_eq!(actual.absolute_position, expected.absolute_position);
    assert_logits("captures/logits", &actual.logits, &expected.logits);
    for (row, layer) in layers.iter().enumerate() {
        let range = row * 4096..(row + 1) * 4096;
        let m = metrics(&actual.residuals[range.clone()], &expected.residuals[range]);
        eprintln!("captures layer {layer} {m:?}");
        assert!(m.cosine >= 0.999, "captures layer {layer} {m:?}");
    }
    // The final residual survives the packed scratch for exact readout.
    let readout = general.readout(&actual.residuals[2 * 4096..]).unwrap();
    assert_eq!(
        readout.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        actual
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    // Same partition twice is deterministic (no atomics in the chunk graph).
    drop(general);
    let mut again = model.create_session(0).unwrap();
    let repeat = again.append_with_captures(&tokens, &layers).unwrap();
    assert_eq!(
        repeat
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        actual
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

//! Read-only GLM speculative-decoding cost probe.
//!
//! Measures, at one depth on the frozen GLM quality code corpus:
//!   * `serial`: one ordinary single-token forward (with full logits readback)
//!   * `verify`: `begin_append` + packed all-row verification for 2..=MAX rows
//!   * `batch`: independent `Glm5NextBatch` rounds at width N (optional)
//!
//! No engine/kernel choice is changed. This is a measurement only.
//!
//! ```text
//! QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task spec-econ --for 15m -- \
//!   target/release/examples/spec_verify_cost MODEL DEPTH [REPS] [BATCH]
//! ```
//!
//! Requires the canonical Metal lease (checked via `QWEN_METAL_LEASE_FD`).
use qwen_llm::{gguf::GgufFile, glm5_next_metal::*, metal::*};
use std::{io::Write, time::Instant};

const VERIFY_ROWS: [usize; 9] = [2, 3, 4, 5, 6, 7, 8, 12, 16];

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}

fn thermal() -> i32 {
    let s = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    String::from_utf8(s.stdout)
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn main() -> anyhow::Result<()> {
    assert!(
        std::env::var_os("QWEN_METAL_LEASE_FD").is_some(),
        "canonical Metal lease required"
    );
    let args: Vec<_> = std::env::args().collect();
    let model = args.get(1).expect("usage: MODEL DEPTH [REPS] [BATCH]");
    let depth: usize = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48 * 1024);
    let reps: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let batch: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0);

    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(model)?;
    let w = Glm5NextWeights::load(&ctx, &gg)?;

    // Frozen code documents (quality-v1 indices 8..=15 are the code cohort).
    let frozen = include_bytes!("../../../scripts/reference/glm53/quality-v1.json");
    let f: serde_json::Value = serde_json::from_slice(frozen)?;
    let docs: Vec<Vec<u32>> = f["documents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            d["tokens"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect()
        })
        .collect();
    let code = [8usize, 9, 11, 13];
    let base_doc = &docs[code[0]];
    let tokens: Vec<u32> = base_doc.iter().copied().cycle().take(depth + 64).collect();
    let capacity = depth + 64;
    let rows = 2048usize;

    emit(serde_json::json!({
        "event": "header",
        "depth": depth,
        "reps": reps,
        "batch": batch,
        "device": ctx.device_facts(),
        "vocab": w.config.vocab_size,
        "blocks": w.config.blocks.len(),
        "fixture_blake3": blake3::hash(frozen).to_hex().as_str(),
        "thermal_before": thermal(),
    }));

    // ---- single-sequence verify/serial curve ----
    let mut s = Glm5NextSession::with_prefill_rows(&ctx, &w, capacity, rows)?;
    let t = Instant::now();
    s.prefill_packed(&ctx, &tokens[..depth])?;
    emit(serde_json::json!({"event":"prefill","ms": t.elapsed().as_secs_f64()*1e3}));
    let baseline = s.capture_snapshot()?;
    let max_rows = *VERIFY_ROWS.last().unwrap();
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &w,
        Glm5NextRowBankPlan::new(&w.config, &[], 0, max_rows)?,
    )?;
    let mut cp = Glm5NextAppendCheckpoint::new(&ctx, &s)?;

    for _ in 0..1 {
        // Warm.
        s.restore_snapshot(&baseline)?;
        let _ = s.forward(&ctx, tokens[depth])?;
    }
    let mut serial = Vec::new();
    for _ in 0..reps {
        s.restore_snapshot(&baseline)?;
        let t = Instant::now();
        let _ = s.forward(&ctx, tokens[depth])?;
        serial.push(t.elapsed().as_secs_f64() * 1e3);
    }
    emit(serde_json::json!({
        "event":"serial","rows":1,"ms":serial,"median":median(serial.clone())
    }));

    for &m in &VERIFY_ROWS {
        let inputs = &tokens[depth..depth + m];
        // Warm one full transaction.
        s.restore_snapshot(&baseline)?;
        {
            let mut tx = s.begin_append(&ctx, &mut cp, &mut bank, inputs)?;
            let _ = tx.verify_packed()?;
            tx.rollback()?;
        }
        let mut begin = Vec::new();
        let mut verify = Vec::new();
        for _ in 0..reps {
            s.restore_snapshot(&baseline)?;
            let t0 = Instant::now();
            let mut tx = s.begin_append(&ctx, &mut cp, &mut bank, inputs)?;
            begin.push(t0.elapsed().as_secs_f64() * 1e3);
            let t1 = Instant::now();
            let n = {
                let out = tx.verify_packed()?;
                // Production acceptance reads every verified row to CPU inside
                // the verify timing (`Glm5NextDFlashRunner::step`), so include it.
                out.read_logits()?.len()
            };
            verify.push(t1.elapsed().as_secs_f64() * 1e3);
            assert_eq!(n, m * w.config.vocab_size as usize);
            tx.rollback()?;
        }
        emit(serde_json::json!({
            "event":"verify","rows":m,
            "begin_ms":begin,"verify_ms":verify,
            "begin_median":median(begin.clone()),
            "verify_median":median(verify.clone()),
            "total_median":median(begin.clone())+median(verify.clone()),
        }));
    }
    drop(bank);
    drop(cp);
    drop(s);

    // ---- optional independent batching baseline ----
    if batch > 1 {
        let mut sessions: Vec<_> = (0..batch)
            // A wide chunk makes the 48K prefill tractable; the batch step only
            // needs the first member's packed scratch to cover the width.
            .map(|_| Glm5NextSession::with_prefill_rows(&ctx, &w, capacity, 512))
            .collect::<std::result::Result<Vec<_>, Glm5NextMetalError>>()?;
        for (i, s) in sessions.iter_mut().enumerate() {
            let doc = &docs[code[i % code.len()]];
            let fill: Vec<u32> = doc.iter().copied().cycle().take(depth).collect();
            s.prefill_packed(&ctx, &fill)?;
        }
        let mut b = Glm5NextBatch::new(&ctx, &w, batch)?;
        let inputs: Vec<u32> = (0..batch).map(|i| tokens[depth + i]).collect();
        for _ in 0..2 {
            let _ = b.step(&ctx, &mut sessions.iter_mut().collect::<Vec<_>>(), &inputs)?;
        }
        let mut rounds = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let vs = b.step(&ctx, &mut sessions.iter_mut().collect::<Vec<_>>(), &inputs)?;
            rounds.push(t.elapsed().as_secs_f64() * 1e3);
            assert!(vs.iter().flatten().all(|v| v.is_finite()));
        }
        emit(serde_json::json!({
            "event":"batch","width":batch,"ms":rounds,
            "median":median(rounds.clone()),
            "per_member_median":median(rounds.clone())/batch as f64,
        }));
    }

    emit(serde_json::json!({"event":"complete","thermal_after":thermal()}));
    Ok(())
}

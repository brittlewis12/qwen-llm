//! Measurement-only GLM batch-path row-cost proxy.
//!
//! One 48K sequence is prefilled and snapshotted, then restored into N
//! sessions. An independent `Glm5NextBatch` round over K+1 of those sessions
//! prices the shared projection/expert/kernel path for K+1 rows. This is a
//! conservative upper bound on a batch-kernel verifier: the members are
//! independent (separate KDA/MLA state and KV writes) rather than a causal
//! chain, so it performs N separate state updates and reads the prefix N times.
//! No engine or kernel change; the logits are not a verify result.
//!
//! ```text
//! QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task spec-econ --for 12m -- \
//!   target/release/examples/spec_batch_rows MODEL DEPTH [REPS] [MAX_WIDTH]
//! ```
use qwen_llm::{gguf::GgufFile, glm5_next_metal::*, metal::*};
use std::{io::Write, time::Instant};

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
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
    let model = args.get(1).expect("MODEL DEPTH [REPS] [MAX_WIDTH]");
    let depth: usize = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(48 * 1024);
    let reps: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(6);
    let max_width: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4);

    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(model)?;
    let w = Glm5NextWeights::load(&ctx, &gg)?;

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
    let tokens: Vec<u32> = docs[8].iter().copied().cycle().take(depth + 64).collect();
    let capacity = depth + 64;

    emit(serde_json::json!({
        "event":"header","depth":depth,"reps":reps,"max_width":max_width,
        "device":ctx.device_facts(),"vocab":w.config.vocab_size,
        "fixture_blake3":blake3::hash(frozen).to_hex().as_str(),
    }));

    // One wide prefill, then a portable snapshot for the row members.
    let mut base_session = Glm5NextSession::with_prefill_rows(&ctx, &w, capacity, 2048)?;
    let t = Instant::now();
    base_session.prefill_packed(&ctx, &tokens[..depth])?;
    emit(serde_json::json!({"event":"prefill","ms":t.elapsed().as_secs_f64()*1e3}));
    let base = base_session.capture_snapshot()?;

    // Compact same-session state copied by one append checkpoint. This is the
    // state a per-row replay checkpoint would have to copy.
    {
        let mut bank =
            Glm5NextRowBank::new(&ctx, &w, Glm5NextRowBankPlan::new(&w.config, &[], 0, 2)?)?;
        let mut cp = Glm5NextAppendCheckpoint::new(&ctx, &base_session)?;
        let bytes = cp.priced_bytes();
        let inputs = &tokens[depth..depth + 2];
        let mut copies = Vec::new();
        for _ in 0..reps {
            base_session.restore_snapshot(&base)?;
            let t = Instant::now();
            let mut tx = base_session.begin_append(&ctx, &mut cp, &mut bank, inputs)?;
            copies.push(t.elapsed().as_secs_f64() * 1e3);
            tx.rollback()?;
        }
        emit(serde_json::json!({
            "event":"checkpoint","priced_bytes":bytes,
            "copy_ms":copies,"copy_median":median(copies.clone()),
            "session_buffer_bytes":base_session.ledger().session_buffer_bytes(),
            "capacity":capacity,
        }));
    }
    drop(base_session);

    // Restore the same 48K state into small-scratch members.
    let mut sessions: Vec<_> = (0..max_width)
        .map(|_| Glm5NextSession::with_prefill_rows(&ctx, &w, capacity, max_width))
        .collect::<std::result::Result<Vec<_>, Glm5NextMetalError>>()?;
    let t = Instant::now();
    for s in sessions.iter_mut() {
        s.restore_snapshot(&base)?;
    }
    emit(serde_json::json!({"event":"restore","ms":t.elapsed().as_secs_f64()*1e3}));

    let mut batch = Glm5NextBatch::new(&ctx, &w, max_width)?;
    for width in 2..=max_width {
        let inputs: Vec<u32> = tokens[depth..depth + width].to_vec();
        // Warm one round, then time independent rounds from the same state.
        let _ = batch.step(
            &ctx,
            &mut sessions[..width].iter_mut().collect::<Vec<_>>(),
            &inputs,
        )?;
        let mut rounds = Vec::new();
        for _ in 0..reps {
            for s in sessions[..width].iter_mut() {
                s.restore_snapshot(&base)?;
            }
            let t = Instant::now();
            let vs = batch.step(
                &ctx,
                &mut sessions[..width].iter_mut().collect::<Vec<_>>(),
                &inputs,
            )?;
            rounds.push(t.elapsed().as_secs_f64() * 1e3);
            assert!(vs.iter().flatten().all(|v| v.is_finite()));
        }
        emit(serde_json::json!({
            "event":"batch","width":width,"ms":rounds,
            "median":median(rounds.clone()),
            "per_member_median":median(rounds.clone())/width as f64,
        }));
    }

    emit(serde_json::json!({"event":"complete"}));
    Ok(())
}

//! Production row-preserving callbacks: parent and nested peer state exactness.
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, Snapshot},
        gguf::{ResidentWeights, resident_storage_plan_with_target},
    },
    gguf::GgufFile,
    metal::{self, MetalContext},
    tokenizer::NativeTokenizer,
};
use serde_json::json;
use std::{io::Write, sync::Arc};

fn emit(v: serde_json::Value) {
    println!("{v}");
    std::io::stdout().flush().unwrap();
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.is_finite() && x.to_bits() == y.to_bits())
}
fn root(
    f: &Forward<'_>,
    ids: &[u32],
    sliced: bool,
) -> Result<(Snapshot, Vec<f32>, Vec<f32>, usize)> {
    let mut state = f.new_state(98_304)?;
    let mut peer = f.new_state(1024)?;
    ensure!(
        f.begin_cold_prefill_episode(&mut state, ids.len())
            .map_err(anyhow::Error::msg)?
    );
    let mut callbacks = 0;
    let mut peer_logits = Vec::new();
    let logits = loop {
        let slice = if sliced {
            f.cold_prefill_episode_layer_interleaved(
                &mut state,
                ids,
                None,
                4096,
                400,
                500,
                &mut || {
                    if callbacks == 0 {
                        peer_logits = f
                            .prefill(&mut peer, &ids[..150], None)
                            .map_err(|e| e.to_string())?;
                    }
                    callbacks += 1;
                    Ok(())
                },
            )
        } else {
            f.cold_prefill_episode_slice(&mut state, ids, None, 4096)
        }
        .map_err(anyhow::Error::msg)?;
        if slice.complete {
            break slice.logits;
        }
    };
    let snapshot = f.capture_snapshot(&state)?;
    let next = f
        .step(&mut state, ids[0], None)
        .map_err(anyhow::Error::msg)?;
    drop(state);
    if sliced {
        ensure!(callbacks > 0, "root must actually yield");
        let mut reference = f.new_state(1024)?;
        let expected = f.prefill(&mut reference, &ids[..150], None)?;
        ensure!(same(&expected, &peer_logits), "root peer logits");
        ensure!(
            f.capture_snapshot(&reference)?.diagnostic_digest()
                == f.capture_snapshot(&peer)?.diagnostic_digest(),
            "root peer state"
        );
    }
    Ok((snapshot, logits, next, callbacks))
}
fn append(
    f: &Forward<'_>,
    prefix: &Snapshot,
    ids: &[u32],
    sliced: bool,
) -> Result<(Vec<f32>, String, Vec<f32>, usize)> {
    let mut state = f.new_state(98_304)?;
    f.restore_snapshot(&mut state, prefix)?;
    let mut peer = f.new_state(98_304)?;
    f.restore_snapshot(&mut peer, prefix)?;
    let mut callbacks = 0;
    let mut peer_logits = Vec::new();
    let logits = if sliced {
        f.prefill_layer_interleaved(&mut state, &ids[..1024], None, true, 400, 500, &mut || {
            if callbacks == 0 {
                peer_logits = f
                    .prefill(&mut peer, &ids[..150], None)
                    .map_err(|e| e.to_string())?;
            }
            callbacks += 1;
            Ok(())
        })?
    } else {
        f.prefill(&mut state, &ids[..1024], None)?
    };
    let digest = f.capture_snapshot(&state)?.diagnostic_digest();
    let next = f
        .step(&mut state, ids[1024], None)
        .map_err(anyhow::Error::msg)?;
    drop(state);
    if sliced {
        ensure!(callbacks > 0, "append must actually yield");
        let mut reference = f.new_state(98_304)?;
        f.restore_snapshot(&mut reference, prefix)?;
        let expected = f.prefill(&mut reference, &ids[..150], None)?;
        ensure!(same(&expected, &peer_logits), "append peer logits");
        ensure!(
            f.capture_snapshot(&reference)?.diagnostic_digest()
                == f.capture_snapshot(&peer)?.diagnostic_digest(),
            "append peer state"
        );
    }
    Ok((logits, digest, next, callbacks))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 3, "model corpus");
    let ctx = MetalContext::new()?;
    let gguf = Arc::new(GgufFile::open(&a[1])?);
    let plan = resident_storage_plan_with_target(
        &gguf,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
        Some(4_294_967_296),
    )?;
    let weights = ResidentWeights::load_with_storage_plan(&ctx, gguf.clone(), &plan)?;
    let f = Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
        .map_err(anyhow::Error::msg)?;
    let tok = NativeTokenizer::from_gguf(&gguf)?;
    let mut text = std::fs::read_to_string(&a[2])?;
    while tok.encode(&text, false)?.len() < 83 * 1024 {
        text.push_str(&text.clone());
    }
    let ids: Vec<_> = tok
        .encode(&text, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    emit(json!({"event":"loaded","root_rows":49152,"append_rows":1024,"append_depth":81920}));
    let (base, logits, next, _) = root(&f, &ids[..49152], false)?;
    let (candidate, other, other_next, callbacks) = root(&f, &ids[..49152], true)?;
    ensure!(
        same(&logits, &other) && same(&next, &other_next),
        "root and next-token logits"
    );
    ensure!(
        base.diagnostic_digest() == candidate.diagnostic_digest(),
        "root state"
    );
    emit(
        json!({"event":"root","bitwise":true,"callbacks":callbacks,"state_digest":base.diagnostic_digest()}),
    );
    drop(candidate);
    let mut extended = f.new_state(98_304)?;
    f.restore_snapshot(&mut extended, &base)?;
    drop(base);
    f.prefill(&mut extended, &ids[49152..81920], None)?;
    let prefix = f.capture_snapshot(&extended)?;
    drop(extended);
    let (a, da, na, _) = append(&f, &prefix, &ids[81920..], false)?;
    let (b, db, nb, callbacks) = append(&f, &prefix, &ids[81920..], true)?;
    ensure!(
        same(&a, &b) && da == db && same(&na, &nb),
        "append logits/state/next-token logits"
    );
    emit(json!({"event":"append","bitwise":true,"callbacks":callbacks,"state_digest":da}));
    Ok(())
}

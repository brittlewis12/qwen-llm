//! Production row-preserving callbacks: parent and nested peer state exactness.
use anyhow::{Result, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward, Snapshot, profile},
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
type Commands = Vec<(String, [usize; 4], usize)>;
fn signature(report: profile::Report) -> Commands {
    let mut v = report
        .operations
        .into_iter()
        .filter(|o| o.completed_submits > 0)
        .map(|o| (o.operation.to_owned(), o.dimensions, o.completed_submits))
        .collect::<Vec<_>>();
    v.sort();
    v
}
fn root(
    f: &Forward<'_>,
    ids: &[u32],
    sliced: bool,
    peer_work: bool,
) -> Result<(Snapshot, Vec<f32>, Vec<f32>, usize, Commands)> {
    let mut state = f.new_state(98_304)?;
    let mut peer = f.new_state(1024)?;
    ensure!(
        f.begin_cold_prefill_episode(&mut state, ids.len())
            .map_err(anyhow::Error::msg)?
    );
    let mut callbacks = 0;
    let mut peer_logits = Vec::new();
    let scope = profile::Scope::begin().map_err(anyhow::Error::msg)?;
    let logits = loop {
        let slice = if sliced {
            f.cold_prefill_episode_waiting_interleaved(
                &mut state,
                ids,
                None,
                4096,
                700,
                &mut || {
                    if peer_work && callbacks == 0 {
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
    let commands = signature(scope.finish());
    let snapshot = f.capture_snapshot(&state)?;
    let next = f
        .step(&mut state, ids[0], None)
        .map_err(anyhow::Error::msg)?;
    drop(state);
    if sliced {
        ensure!(callbacks > 0, "root must poll actual drains");
    }
    if sliced && peer_work {
        let mut reference = f.new_state(1024)?;
        let expected = f.prefill(&mut reference, &ids[..150], None)?;
        ensure!(same(&expected, &peer_logits), "root peer logits");
        ensure!(
            f.capture_snapshot(&reference)?.diagnostic_digest()
                == f.capture_snapshot(&peer)?.diagnostic_digest(),
            "root peer state"
        );
    }
    Ok((snapshot, logits, next, callbacks, commands))
}
fn append(
    f: &Forward<'_>,
    prefix: &Snapshot,
    ids: &[u32],
    sliced: bool,
    peer_work: bool,
) -> Result<(Vec<f32>, String, Vec<f32>, usize, Commands)> {
    let mut state = f.new_state(98_304)?;
    f.restore_snapshot(&mut state, prefix)?;
    let mut peer = f.new_state(98_304)?;
    f.restore_snapshot(&mut peer, prefix)?;
    let mut callbacks = 0;
    let mut peer_logits = Vec::new();
    let scope = profile::Scope::begin().map_err(anyhow::Error::msg)?;
    let logits = if sliced {
        f.prefill_waiting_interleaved(&mut state, &ids[..1024], None, true, 700, &mut || {
            if peer_work && callbacks == 0 {
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
    let commands = signature(scope.finish());
    let digest = f.capture_snapshot(&state)?.diagnostic_digest();
    let next = f
        .step(&mut state, ids[1024], None)
        .map_err(anyhow::Error::msg)?;
    drop(state);
    if sliced {
        ensure!(callbacks > 0, "append must poll actual drains");
    }
    if sliced && peer_work {
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
    Ok((logits, digest, next, callbacks, commands))
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
    let (base, logits, next, _, commands) = root(&f, &ids[..49152], false, false)?;
    let (empty, empty_logits, empty_next, polls, empty_commands) =
        root(&f, &ids[..49152], true, false)?;
    ensure!(
        commands == empty_commands,
        "empty root must keep command topology"
    );
    ensure!(
        same(&logits, &empty_logits)
            && same(&next, &empty_next)
            && base.diagnostic_digest() == empty.diagnostic_digest(),
        "empty root bits"
    );
    emit(json!({"event":"root_empty","bitwise":true,"polls":polls,"same_commands":true}));
    drop(empty);
    let (candidate, other, other_next, callbacks, _) = root(&f, &ids[..49152], true, true)?;
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
    let (a, da, na, _, commands) = append(&f, &prefix, &ids[81920..], false, false)?;
    let (empty, de, ne, polls, empty_commands) = append(&f, &prefix, &ids[81920..], true, false)?;
    ensure!(
        commands == empty_commands,
        "empty append must keep command topology"
    );
    ensure!(
        same(&a, &empty) && da == de && same(&na, &ne),
        "empty append bits"
    );
    emit(json!({"event":"append_empty","bitwise":true,"polls":polls,"same_commands":true}));
    let (b, db, nb, callbacks, _) = append(&f, &prefix, &ids[81920..], true, true)?;
    ensure!(
        same(&a, &b) && da == db && same(&na, &nb),
        "append logits/state/next-token logits"
    );
    emit(json!({"event":"append","bitwise":true,"callbacks":callbacks,"state_digest":da}));
    Ok(())
}

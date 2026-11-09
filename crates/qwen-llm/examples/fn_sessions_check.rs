//! Model witness: whole-call prefill/serial decode versus preserved native
//! command quanta with independent owners interleaved. No files of logits/state.
//! Run under with-metal-lease: MODEL CODE PROSE DEPTH [STEPS=128].
use qwen_llm::{
    gguf::GgufFile,
    metal::MetalContext,
    qwen4exp::Qwen4ExpConfig,
    qwen4exp_runtime::{Qwen4ExpDecodeOptions, Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{collections::VecDeque, fs};
fn stream(ids: &[u32], offset: usize, n: usize) -> Vec<u32> {
    (0..n).map(|i| ids[(i + offset) % ids.len()]).collect()
}
fn lz(v: &[f32]) -> f64 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + v.iter().map(|x| (*x as f64 - m).exp()).sum::<f64>().ln()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let depth: usize = a[4].parse()?;
    let steps: usize = a.get(5).map_or(Ok(128), |s| s.parse())?;
    let ctx = MetalContext::new()?;
    let gg = GgufFile::open(&a[1])?;
    let c = Qwen4ExpConfig::from_gguf(&gg)?;
    let tok = Tokenizer::from_gguf(&gg)?;
    let code: Vec<u32> = tok
        .encode(&fs::read_to_string(&a[2])?, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let prose: Vec<u32> = tok
        .encode(&fs::read_to_string(&a[3])?, false)?
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert!(!code.is_empty() && !prose.is_empty());
    let extent = depth + 8192 + steps;
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gg,
        Qwen4ExpSessionCapacity::for_forward_limit(&c, extent)?,
        extent,
        Qwen4ExpDecodeOptions::default(),
    )?;
    let (factory, first) = loaded.session_factory(&ctx)?;
    let mut runners = vec![
        first,
        factory.create_runner(&ctx)?,
        factory.create_runner(&ctx)?,
    ];
    let mut unrelated = factory.create_runner(&ctx)?;
    unrelated.prefill(&code[..64.min(code.len())])?;
    let prefixes = [depth, depth + 17, depth + 34];
    let appends = [1123usize, 1084, 1596];
    let docs = [
        stream(&code, 0, extent),
        stream(&prose, 0, extent),
        stream(&code, code.len() / 3, extent),
    ];
    let mut baseline = Vec::new();
    for (i, r) in runners.iter_mut().enumerate() {
        let p = prefixes[i];
        let end = p + appends[i];
        r.prefill(&docs[i][..p])?;
        r.prefill_continuation_with_command_checkpoint(&docs[i][p..end - 3], || Ok(()))?;
        r.prefill_continuation_with_command_checkpoint(&docs[i][end - 3..end], || Ok(()))?;
        let mut vectors = Vec::new();
        for step in 0..steps {
            let v = r.forward_token(docs[i][end + step])?.to_vec();
            assert!(v.iter().all(|x| x.is_finite()));
            vectors.push(v);
        }
        baseline.push(vectors);
    }
    let mut schedules = Vec::new();
    for (i, r) in runners.iter_mut().enumerate() {
        r.reset()?;
        let p = prefixes[i];
        let end = p + appends[i];
        let mut commands = VecDeque::new();
        for segment in [0..p, p..end - 3, end - 3..end] {
            commands.extend(r.prefill_command_ranges(segment)?);
        }
        schedules.push(commands);
    }
    let mut interposed = 0;
    while schedules.iter().any(|s| !s.is_empty()) {
        for i in 0..runners.len() {
            if let Some(range) = schedules[i].pop_front() {
                assert_eq!(runners[i].next_position(), range.start);
                let end = range.end;
                runners[i]
                    .prefill_continuation_with_command_checkpoint(&docs[i][range], || Ok(()))?;
                assert_eq!(runners[i].next_position(), end);
                unrelated.forward_token(code[interposed % code.len()])?;
                interposed += 1;
            }
        }
    }
    let (mut exact, mut dnll, mut kl) = (0usize, 0f64, 0f64);
    let mut hash = blake3::Hasher::new();
    for step in 0..steps {
        for i in 0..runners.len() {
            let end = prefixes[i] + appends[i];
            let actual = runners[i].forward_token(docs[i][end + step])?.to_vec();
            let reference = &baseline[i][step];
            assert!(actual.iter().all(|x| x.is_finite()));
            let equal = bytemuck::cast_slice::<f32, u8>(&actual)
                == bytemuck::cast_slice::<f32, u8>(reference);
            assert!(equal, "interleaving changed sequence={i} step={step}");
            exact += usize::from(equal);
            hash.update(bytemuck::cast_slice(&actual));
            let (z, bz) = (lz(&actual), lz(reference));
            let target = docs[i][end + step + 1] as usize;
            dnll += (z - actual[target] as f64) - (bz - reference[target] as f64);
            if !equal {
                kl += reference
                    .iter()
                    .zip(&actual)
                    .map(|(&p, &q)| (p as f64 - bz).exp() * ((p as f64 - bz) - (q as f64 - z)))
                    .sum::<f64>();
            }
        }
    }
    let entry = runners[0].capture_snapshot()?;
    let mut fork = factory.create_runner(&ctx)?;
    fork.restore_snapshot(&entry)?;
    let mut fork_exact = 0;
    for i in 0..32 {
        let t = code[i % code.len()];
        let a = runners[0].forward_token(t)?.to_vec();
        unrelated.forward_token(t)?;
        let b = fork.forward_token(t)?.to_vec();
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&a),
            bytemuck::cast_slice::<f32, u8>(&b)
        );
        fork_exact += 1;
    }
    assert_eq!(exact, steps * 3);
    assert_eq!(dnll, 0.);
    assert_eq!(kl, 0.);
    println!(
        "{}",
        json!({"depths":prefixes,"appends":appends,"teacher_vectors":exact,"bitwise":true,"signed_dNLL":dnll/(steps*3)as f64,"mean_KL":kl/(steps*3)as f64,"fork_exact":fork_exact,"interposed_steps":interposed,"stream_hash":hash.finalize().to_hex().to_string(),"packed_rows":runners[0].packed_prefill_capacity()})
    );
    Ok(())
}

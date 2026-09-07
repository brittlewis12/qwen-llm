use super::*;
use crate::gguf::GgufFile;
use crate::loader::Model;
use crate::metal::with_tiled_vt_pilot;
use crate::metal_forward::{MetalModel, SessionSnapshot};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn state_hashes(s: &SessionSnapshot, logits: &[f32]) -> Vec<String> {
    assert!(logits.iter().all(|x| x.is_finite()));
    vec![
        hash(bytemuck::cast_slice(logits)),
        hash(&s.kv_k_arena),
        hash(&s.kv_v_arena),
        hash(&s.gdn_state_arena),
        hash(&s.gdn_conv_arena),
    ]
}

#[test]
#[ignore = "serial Metal, exact full-model restored-tail transpose phase"]
fn tiled_vt_restored_prefill_phase() {
    let path = std::env::var("QWEN_FULL_VERIFY_MODEL").unwrap();
    let fixture = std::env::var("QWEN_FULL_VERIFY_FIXTURE").unwrap();
    let text = std::fs::read_to_string(fixture).unwrap();
    assert_eq!(
        hash(text.as_bytes()),
        "7f982cd896c7809430bdcdf489723e6eed6bfdc2a04882ee15bcf2a65c676352"
    );
    let tokenizer = crate::tokenizer::Tokenizer::open(&path).unwrap();
    let tokens = tokenizer.encode(&text, false).unwrap();
    assert!(tokens.len() >= 32784);
    assert_eq!(
        hash(bytemuck::cast_slice(&tokens[..32752])),
        "2f9c6aedee579283f7a74353e562d9222b15fa8fa1ba9eece73ca339b033398f"
    );
    for (key, _) in std::env::vars_os() {
        assert!(!key.to_string_lossy().starts_with("QWEN_ATTN_"));
    }
    let gguf = GgufFile::open(&path).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    assert_eq!(model.arch.kind, crate::model::ArchKind::Dense);
    assert_eq!(model.arch.n_layer, 64);
    assert_eq!(model.token_embd.dtype, GgmlType::Q8_0);
    let ctx = MetalContext::new().unwrap();
    let metal = MetalModel::load(&ctx, &gguf, &model).unwrap();
    let forward = MetalForward::new(&ctx, &metal);
    let mut seed: Option<SessionSnapshot> = None;
    for base in [8840usize, 32752] {
        let end = base + 32;
        let mut prime = MetalSession::fresh(&ctx, &metal, end).unwrap();
        let start_pos = seed.as_ref().map_or(0, |s| s.prefix_len());
        if let Some(previous) = seed.take() {
            prime.restore_from(&previous, &previous.identity).unwrap();
        }
        let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
            &ctx, &metal, 1024, end,
        )
        .unwrap();
        let start = Instant::now();
        prefill_tokens_with_multi_hidden(
            &forward,
            &tokens[start_pos..base],
            start_pos as u32,
            &mut prime,
            &mut scratch,
            &[],
            None,
        )
        .unwrap();
        println!(
            "VT_PREFILL_JSON {}",
            json!({"kind":"prime", "base":base, "from":start_pos,
            "wall_ms":start.elapsed().as_secs_f64()*1e3})
        );
        drop(scratch);
        let identity = prime.snapshot_identity(0x20260907, 0x20260907);
        let snapshot = prime
            .snapshot(identity.clone(), tokens[..base].to_vec(), None)
            .unwrap();
        drop(prime);
        let mut expected = None;
        for (index, enabled) in [false, true, true, false, false, true, true, false]
            .into_iter()
            .enumerate()
        {
            let mut session = MetalSession::fresh(&ctx, &metal, end).unwrap();
            let start = Instant::now();
            session.restore_from(&snapshot, &identity).unwrap();
            let restore_ms = start.elapsed().as_secs_f64() * 1e3;
            let allocation_start = Instant::now();
            let before = ctx.current_allocated_size();
            let plan =
                plan_single_chunk_prefill_scratch(&metal, 32, end, PrefillScratchConfig::default())
                    .unwrap();
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal, plan).unwrap();
            let scratch_bytes = ctx.current_allocated_size() - before;
            let allocation_ms = allocation_start.elapsed().as_secs_f64() * 1e3;
            let prefill_start = Instant::now();
            let (logits, tiled_calls) = with_tiled_vt_pilot(enabled, || {
                prefill_tokens_with_multi_hidden(
                    &forward,
                    &tokens[base..end],
                    base as u32,
                    &mut session,
                    &mut scratch,
                    &[],
                    None,
                )
                .unwrap()
            });
            let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1e3;
            let total_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(tiled_calls, if enabled { 16 } else { 0 });
            println!(
                "VT_PREFILL_JSON {}",
                json!({"kind":if index<4 {"warmup"} else {"sample"},
                "base":base, "suffix":32, "index":index, "arm":if enabled {"B"} else {"A"},
                "restore_ms":restore_ms, "allocation_ms":allocation_ms, "prefill_ms":prefill_ms,
                "total_ms":total_ms, "scratch_bytes":scratch_bytes, "tiled_calls":tiled_calls})
            );
            drop(scratch);
            if (4..7).contains(&index) {
                continue;
            }
            let final_state = session
                .snapshot(identity.clone(), tokens[..end].to_vec(), None)
                .unwrap();
            let hashes = state_hashes(&final_state, &logits);
            assert_eq!(final_state.kv_n_pos, vec![end; 16]);
            if let Some(expected) = &expected {
                assert_eq!(&hashes, expected);
            } else {
                expected = Some(hashes.clone());
            }
            println!(
                "VT_PREFILL_JSON {}",
                json!({"kind":"bitwise", "base":base, "index":index, "hashes":hashes})
            );
        }
        seed = Some(snapshot);
    }
}

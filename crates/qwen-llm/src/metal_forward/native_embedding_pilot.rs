use super::*;
use crate::gguf::GgufFile;
use crate::loader::Model;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn metadata(model: &Model<'_>) -> Value {
    let source = model.token_embd;
    assert_eq!(model.arch.kind, crate::model::ArchKind::Dense);
    assert_eq!(model.arch.n_layer, 64);
    assert_eq!(source.dtype, GgmlType::Q8_0);
    assert_eq!(source.shape, [5120, 248320]);
    assert!(!model.tied_embeddings);
    assert_ne!(source.name, model.lm_head.name);
    assert!(native_quant_embedding_supported(
        source.dtype,
        &source.shape
    ));
    let f32_bytes = source.n_elements().checked_mul(4).unwrap();
    json!({
        "arch": format!("{:?}", model.arch),
        "source_dtype": format!("{:?}", source.dtype), "shape": source.shape,
        "tied": model.tied_embeddings, "mtp_bound": model.mtp.is_some(), "source_bytes": source.n_bytes,
        "f32_bytes": f32_bytes, "logical_resident_saving": f32_bytes - source.n_bytes,
        "default_promoted": native_quant_embedding_default_promoted(
            &model.arch, model.tied_embeddings, source.dtype, &source.shape),
        "lm_head_dtype": format!("{:?}", model.lm_head.dtype),
        "lm_head_bytes": model.lm_head.n_bytes,
    })
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn float_hash(values: &[f32]) -> String {
    assert!(values.iter().all(|x| x.is_finite()));
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    hash(&bytes)
}

fn state_hashes(snapshot: &SessionSnapshot) -> Value {
    json!({"identity": format!("{:?}", snapshot.identity),
        "tokens": snapshot.prefix_tokens, "positions": snapshot.kv_n_pos,
        "k": hash(&snapshot.kv_k_arena), "v": hash(&snapshot.kv_v_arena),
        "gdn_state": hash(&snapshot.gdn_state_arena), "gdn_conv": hash(&snapshot.gdn_conv_arena),
        "lengths": [snapshot.kv_k_arena.len(), snapshot.kv_v_arena.len(),
            snapshot.gdn_state_arena.len(), snapshot.gdn_conv_arena.len()]})
}

#[test]
#[ignore = "CPU-only; requires QWEN_EMBED_RESIDENCY_MODEL"]
fn dense_q8_embedding_metadata_witness() {
    let path = std::env::var("QWEN_EMBED_RESIDENCY_MODEL").unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    eprintln!("embedding-metadata {}", metadata(&model));
}

#[test]
#[ignore = "serial Metal; requires QWEN_EMBED_RESIDENCY_MODEL, QWEN_EMBED_ORACLE_OUT and native-embed override"]
fn dense_q8_embedding_residency_and_state_witness() {
    use crate::metal::{encode_get_rows_f32, one_shot, read_back_f32};
    use crate::metal_dflash::{
        MetalDFlashLayerMajorScratch, PrefillScratchConfig,
        plan_prefill_scratch_with_matrix_max_pos_configured, prefill_tokens_with_multi_hidden,
    };
    use crate::sampling::{Sampler, SamplingConfig};
    use std::io::Write;

    let path = std::env::var("QWEN_EMBED_RESIDENCY_MODEL").unwrap();
    let output = std::env::var("QWEN_EMBED_ORACLE_OUT").unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let model = Model::from_gguf(&gguf).unwrap();
    let metadata = metadata(&model);
    let expected_native = match native_quant_embedding_mode() {
        NativeQuantEmbeddingMode::Forced => true,
        NativeQuantEmbeddingMode::Disabled => false,
        _ => panic!("explicit QWEN_NATIVE_QUANT_EMBED=0 or 1 required"),
    };
    let ctx = MetalContext::new().unwrap();
    let before = ctx.current_allocated_size();
    let start = std::time::Instant::now();
    let metal = MetalModel::load(&ctx, &gguf, &model).unwrap();
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    let model_bytes = ctx.current_allocated_size() - before;
    let embedding_bytes = metal.token_embd.n_bytes();
    assert_eq!(
        metal.token_embd.dtype,
        if expected_native {
            GgmlType::Q8_0
        } else {
            GgmlType::F32
        }
    );
    assert_eq!(
        embedding_bytes,
        if expected_native {
            model.token_embd.n_bytes
        } else {
            model.token_embd.n_elements() * 4
        }
    );
    assert_eq!(metal.lm_head.dtype, model.lm_head.dtype);
    assert_eq!(metal.lm_head.n_bytes(), model.lm_head.n_bytes);
    assert_ne!(
        Retained::as_ptr(&metal.token_embd.buffer),
        Retained::as_ptr(&metal.lm_head.buffer)
    );

    let row_ids = vec![
        0i32, 1, 31, 32, 255, 256, 12345, 65535, 131071, 248044, 248046, 248319, 0,
    ];
    let gather = |ids: &[i32]| {
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(ids),
            vec![ids.len() as u64],
            GgmlType::I32,
        )
        .unwrap();
        let out = MetalTensor::zeros_f32(&ctx, vec![ids.len() as u64 * 5120]).unwrap();
        one_shot(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &metal.token_embd, &input, &out, ids.len(), 5120)
        })
        .unwrap();
        read_back_f32(&out.buffer, ids.len() * 5120)
    };
    let packed_rows = gather(&row_ids);
    assert!(packed_rows.iter().any(|&x| x != 0.0));
    let scalar_rows: Vec<_> = row_ids.iter().flat_map(|&id| gather(&[id])).collect();
    assert_eq!(float_hash(&packed_rows), float_hash(&scalar_rows));

    let tokenizer = crate::tokenizer::Tokenizer::open(&path).unwrap();
    let prompt = "<|im_start|>user\nWrite Python that merges two sorted lists without duplicates. Include tests.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = tokenizer.encode(prompt, false).unwrap();
    let capacity = tokens.len() + 64;
    let forward = MetalForward::new(&ctx, &metal);
    let mut session = MetalSession::fresh(&ctx, &metal, capacity).unwrap();
    let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        &metal,
        tokens.len() as u32,
        capacity,
        PrefillScratchConfig::default(),
    )
    .unwrap();
    let mut scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &metal, plan).unwrap();
    let mut logits = prefill_tokens_with_multi_hidden(
        &forward,
        &tokens,
        0,
        &mut session,
        &mut scratch,
        &[],
        None,
    )
    .unwrap();
    drop(scratch);
    let identity = session.snapshot_identity(0x20260907, 0x20260907);
    let prefill_logits = float_hash(&logits);
    let prefill_state = state_hashes(
        &session
            .snapshot(identity.clone(), tokens.clone(), None)
            .unwrap(),
    );
    let stops = gguf.stop_token_ids().unwrap();
    let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
    let mut generated = Vec::new();
    let mut logits_hashes = Vec::new();
    let mut consumed = tokens.clone();
    for step in 0..64 {
        logits_hashes.push(float_hash(&logits));
        let token = sampler.sample(&logits).unwrap().token;
        generated.push(token);
        if stops.contains(&token) || step == 63 {
            break;
        }
        logits = forward
            .single_token(token, consumed.len() as u32, &mut session)
            .unwrap();
        consumed.push(token);
    }
    let final_state = state_hashes(&session.snapshot(identity, consumed, None).unwrap());
    let record = json!({"metadata": metadata, "native": expected_native,
        "model_allocated_bytes": model_bytes, "embedding_bytes": embedding_bytes,
        "embedding_backing_bytes": metal.token_embd.buffer.length(), "load_ms": load_ms,
        "row_ids": row_ids, "row_hash": float_hash(&packed_rows), "prompt_ids": tokens,
        "prefill_logits": prefill_logits, "prefill_state": prefill_state,
        "generated": generated, "logits_hashes": logits_hashes, "final_state": final_state});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&record).unwrap())
        .unwrap();
    eprintln!(
        "embedding-witness native={expected_native} embedding_bytes={embedding_bytes} model_bytes={model_bytes} constructor_ms={load_ms:.3} greedy_tokens={}",
        record["generated"].as_array().unwrap().len()
    );
}

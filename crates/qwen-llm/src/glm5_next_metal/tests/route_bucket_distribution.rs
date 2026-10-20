use super::*;
use crate::glm5_next_metal::tests::{long_qualification_text, perf_lease};
use serde_json::json;
use std::io::Write;

#[test]
#[ignore = "release, leased GPU; GLM53_GGUF=Maya and GLM53_STAGE_OUT=new JSON path"]
fn maya_iq2_s_pp2048_bucket_distribution() {
    assert!(!cfg!(debug_assertions), "requires release test binary");
    let _lease = perf_lease();
    let model_path = std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF must name Maya");
    let output_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(std::env::var_os("GLM53_STAGE_OUT").expect("GLM53_STAGE_OUT"));
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .unwrap();
    let ctx = MetalContext::new().expect("Metal context");
    let gguf = GgufFile::open(std::path::PathBuf::from(model_path)).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).expect("load Maya IQ2_S");
    const PROMPT: usize = 2048;
    const CHUNK: usize = 512;
    let tokens: Vec<u32> = crate::tokenizer::Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode(&long_qualification_text(), false)
        .unwrap()
        .into_iter()
        .take(PROMPT)
        .map(|id| u32::try_from(id).unwrap())
        .collect();
    assert_eq!(tokens.len(), PROMPT);
    let experts = weights.config.expert_count as usize;
    let moe_layers = weights
        .blocks
        .iter()
        .filter(|block| matches!(&block.ffn, FfnTensors::Moe(_)))
        .count();
    route_bucket_capture::begin(&ctx, experts, moe_layers * PROMPT.div_ceil(CHUNK)).unwrap();
    let mut session =
        Glm5NextSession::with_prefill_rows(&ctx, &weights, PROMPT + 8, CHUNK).unwrap();
    let logits = session.prefill_packed(&ctx, &tokens).unwrap();
    assert_finite("bucket capture logits", &logits);
    assert_eq!(session.position(), PROMPT);
    let (capture, samples, n_experts) = route_bucket_capture::take().expect("capture");
    assert_eq!(n_experts, experts);
    assert_eq!(samples.len(), moe_layers * PROMPT.div_ceil(CHUNK));

    let base = unsafe {
        capture
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(capture.offset as usize)
            .cast::<i32>()
    };
    let mut records = Vec::with_capacity(samples.len());
    for (sample_index, sample) in samples.iter().enumerate() {
        let bucket_counts = unsafe {
            std::slice::from_raw_parts(base.add(sample_index * experts), experts).to_vec()
        };
        assert!(bucket_counts.iter().all(|&count| count >= 0));
        let active: Vec<usize> = bucket_counts
            .iter()
            .filter_map(|&count| usize::try_from(count).ok().filter(|&count| count > 0))
            .collect();
        assert!(!active.is_empty());
        let expected_routes = sample.rows * weights.config.expert_used_count as usize;
        assert_eq!(active.iter().sum::<usize>(), expected_routes);
        records.push(json!({
            "chunk": sample.chunk,
            "layer": sample.layer,
            "rows": sample.rows,
            "active_experts": active.len(),
            "rows_per_active_expert": active,
        }));
    }

    let mut by_layer = serde_json::Map::new();
    let mut all_counts = Vec::new();
    let mut n16_full = 0usize;
    let mut n16_tail = 0usize;
    let mut n32_full = 0usize;
    let mut n32_tail = 0usize;
    for sample in &samples {
        let active: Vec<usize> = {
            let sample_index = samples
                .iter()
                .position(|candidate| {
                    candidate.chunk == sample.chunk && candidate.layer == sample.layer
                })
                .unwrap();
            unsafe {
                std::slice::from_raw_parts(base.add(sample_index * experts), experts)
                    .iter()
                    .filter_map(|&count| usize::try_from(count).ok().filter(|&count| count > 0))
                    .collect()
            }
        };
        all_counts.extend(active.iter().copied());
        for count in active.iter().copied().filter(|&count| count < 17) {
            n16_full += count / 16;
            n16_tail += usize::from(count % 16 != 0);
        }
        for count in active.iter().copied().filter(|&count| count >= 17) {
            n32_full += count / 32;
            n32_tail += usize::from(count % 32 != 0);
        }
        let layer = by_layer
            .entry(sample.layer.to_string())
            .or_insert_with(|| json!({"active_experts_per_chunk": []}));
        layer["active_experts_per_chunk"]
            .as_array_mut()
            .unwrap()
            .push(json!({"chunk":sample.chunk,"count":active.len()}));
    }
    all_counts.sort_unstable();
    let percentile = |numerator: usize, denominator: usize| -> usize {
        let index = all_counts
            .len()
            .saturating_mul(numerator)
            .div_ceil(denominator)
            .saturating_sub(1);
        all_counts[index]
    };
    let document = json!({
        "schema": "glm53.maya_iq2_s_pp2048_route_bucket_distribution.v1",
        "model": "GLM-5.3-Flash-Maya-M-IQ2_S",
        "prompt_tokens": PROMPT,
        "chunk_rows": CHUNK,
        "chunks": PROMPT.div_ceil(CHUNK),
        "expert_count": experts,
        "top_k": weights.config.expert_used_count,
        "bucket_rows_percentiles": {
            "sample_population": "positive expert bucket counts, one sample per active expert per MoE layer per chunk",
            "p10": percentile(10,100), "p50": percentile(50,100),
            "p90": percentile(90,100), "max": *all_counts.last().unwrap(),
            "active_bucket_samples": all_counts.len(),
        },
        "tile_split": {
            "dispatch_split_threshold": 17,
            "n16": {"full_16_row_tiles": n16_full, "tail_tiles": n16_tail},
            "n32": {"full_32_row_tiles": n32_full, "tail_tiles": n32_tail},
        },
        "layers": by_layer,
        "records": records,
    });
    serde_json::to_writer_pretty(&mut output, &document).unwrap();
    writeln!(output).unwrap();
}

use anyhow::{Context, Result, ensure};
use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{MetalContext, MetalTensor};
use qwen_llm::metal_forward::{
    MetalForward, MetalModel, MetalSession, t9_ffn_capture_install, t9_ffn_capture_reset_token,
    t9_ffn_capture_uninstall,
};
use qwen_llm::model::ArchKind;
use qwen_llm::tokenizer::Tokenizer;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const DEFAULT_PROMPT: &str =
    "Give a concise explanation of why the sky appears blue during the day.";

#[derive(Serialize)]
struct CaptureManifest {
    schema_version: u32,
    model: String,
    model_bytes: u64,
    prompt: String,
    prompt_token_ids: Vec<i32>,
    captured_token_ids: Vec<i32>,
    layers: Vec<usize>,
    hidden_size: usize,
    intermediate_size: usize,
    tensor: TensorManifest,
}

#[derive(Serialize)]
struct TensorManifest {
    name: &'static str,
    dtype: &'static str,
    shape: [usize; 3],
    byte_length: u64,
    sha256: String,
}

fn parse_layers(raw: Option<String>, n_layers: usize) -> Result<Vec<usize>> {
    let layers = match raw {
        Some(raw) => raw
            .split(',')
            .map(|value| {
                value
                    .trim()
                    .parse::<usize>()
                    .with_context(|| format!("parse layer index {value:?}"))
            })
            .collect::<Result<Vec<_>>>()?,
        None => (0..n_layers).collect(),
    };
    ensure!(!layers.is_empty(), "FFN_CENSUS_LAYERS is empty");
    ensure!(
        layers.windows(2).all(|pair| pair[0] < pair[1]),
        "FFN_CENSUS_LAYERS must be strictly ascending and unique"
    );
    ensure!(
        layers.last().copied().unwrap_or_default() < n_layers,
        "FFN_CENSUS_LAYERS exceeds model layer count {n_layers}"
    );
    Ok(layers)
}

fn read_f32_tensor(tensor: &MetalTensor) -> Vec<f32> {
    let mut values = vec![0.0f32; tensor.n_elements() as usize];
    unsafe {
        let source = (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize)
            as *const f32;
        std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), values.len());
    }
    values
}

fn argmax(values: &[f32]) -> Result<i32> {
    ensure!(!values.is_empty(), "cannot select argmax from empty logits");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "logits contain non-finite values"
    );
    let index = values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("non-empty logits");
    i32::try_from(index).context("argmax token ID exceeds i32")
}

fn file_sha256(path: &Path) -> Result<(u64, String)> {
    let mut source = std::fs::File::open(path)
        .with_context(|| format!("open capture for hashing {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut bytes = 0u64;
    loop {
        let count = source
            .read(&mut buffer)
            .with_context(|| format!("read capture for hashing {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        bytes += count as u64;
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
}

fn main() -> Result<()> {
    let model_path = PathBuf::from(
        std::env::var_os("FFN_CENSUS_MODEL").context("FFN_CENSUS_MODEL is required")?,
    );
    let out_dir = PathBuf::from(
        std::env::var_os("FFN_CENSUS_OUT")
            .unwrap_or_else(|| "target/profiles/ffn-inner-census".into()),
    );
    let prompt = std::env::var("FFN_CENSUS_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let capture_tokens = std::env::var("FFN_CENSUS_TOKENS")
        .unwrap_or_else(|_| "8".into())
        .parse::<usize>()
        .context("parse FFN_CENSUS_TOKENS")?;
    ensure!(capture_tokens > 0, "FFN_CENSUS_TOKENS must be positive");

    let gguf = GgufFile::open(&model_path)
        .with_context(|| format!("open GGUF {}", model_path.display()))?;
    let model = Model::from_gguf(&gguf).context("load model metadata")?;
    ensure!(
        model.arch.kind == ArchKind::Dense,
        "FFN inner census currently requires a dense Qwen model"
    );
    let n_layers = model.arch.n_layer as usize;
    let hidden_size = model.arch.hidden_size as usize;
    let intermediate_size = model.arch.intermediate_size as usize;
    ensure!(
        intermediate_size.is_multiple_of(256),
        "intermediate size {intermediate_size} is not block-256 aligned"
    );
    let layers = parse_layers(std::env::var("FFN_CENSUS_LAYERS").ok(), n_layers)?;
    let captured_layer_count = layers.len();
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load tokenizer")?;
    let prompt_token_ids = tokenizer
        .encode(&prompt, false)
        .context("tokenize census prompt")?;
    ensure!(
        !prompt_token_ids.is_empty(),
        "census prompt tokenized empty"
    );

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output directory {}", out_dir.display()))?;
    let tensor_path = out_dir.join("inner.f32le");
    let context = MetalContext::new().context("create Metal context")?;
    let metal_model = MetalModel::load(&context, &gguf, &model).context("load Metal model")?;
    let forward = MetalForward::new(&context, &metal_model);
    let mut session = MetalSession::fresh(
        &context,
        &metal_model,
        prompt_token_ids.len() + capture_tokens + 1,
    )
    .context("create Metal session")?;

    let mut logits = Vec::new();
    for (position, &token) in prompt_token_ids.iter().enumerate() {
        logits = forward
            .single_token(token, position as u32, &mut session)
            .with_context(|| format!("prefill token at position {position}"))?;
    }

    let mut slots = Vec::with_capacity(layers.len());
    let mut buffers = Vec::with_capacity(layers.len());
    for &layer in &layers {
        let hidden = MetalTensor::zeros_f32(&context, vec![hidden_size as u64])
            .with_context(|| format!("allocate hidden capture for layer {layer}"))?;
        let inner = MetalTensor::zeros_f32(&context, vec![intermediate_size as u64])
            .with_context(|| format!("allocate inner capture for layer {layer}"))?;
        slots.push((layer, hidden.clone(), inner.clone()));
        buffers.push((hidden, inner));
    }
    t9_ffn_capture_install(slots);

    let capture_result = (|| -> Result<Vec<i32>> {
        let mut writer = BufWriter::new(
            std::fs::File::create(&tensor_path)
                .with_context(|| format!("create {}", tensor_path.display()))?,
        );
        let mut captured_token_ids = Vec::with_capacity(capture_tokens);
        for sample in 0..capture_tokens {
            let token = argmax(&logits)?;
            captured_token_ids.push(token);
            let position = prompt_token_ids.len() + sample;
            t9_ffn_capture_reset_token();
            logits = forward
                .single_token(token, position as u32, &mut session)
                .with_context(|| format!("capture token at position {position}"))?;
            for (_, inner) in &buffers {
                let values = read_f32_tensor(inner);
                ensure!(
                    values.iter().all(|value| value.is_finite()),
                    "captured non-finite FFN inner value at position {position}"
                );
                writer
                    .write_all(bytemuck::cast_slice(&values))
                    .with_context(|| format!("write {}", tensor_path.display()))?;
            }
        }
        writer
            .flush()
            .with_context(|| format!("flush {}", tensor_path.display()))?;
        Ok(captured_token_ids)
    })();
    t9_ffn_capture_uninstall();
    let captured_token_ids = capture_result?;

    let (byte_length, sha256) = file_sha256(&tensor_path)?;
    let expected_bytes = capture_tokens * layers.len() * intermediate_size * size_of::<f32>();
    ensure!(
        byte_length == expected_bytes as u64,
        "capture byte length {byte_length} != expected {expected_bytes}"
    );
    let manifest = CaptureManifest {
        schema_version: 1,
        model: model_path.display().to_string(),
        model_bytes: std::fs::metadata(&model_path)?.len(),
        prompt,
        prompt_token_ids,
        captured_token_ids,
        layers,
        hidden_size,
        intermediate_size,
        tensor: TensorManifest {
            name: "inner.f32le",
            dtype: "f32le",
            shape: [capture_tokens, captured_layer_count, intermediate_size],
            byte_length,
            sha256,
        },
    };
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;
    eprintln!(
        "ffn_inner_census: samples={} layers={} values={} output={}",
        capture_tokens,
        manifest.layers.len(),
        byte_length / size_of::<f32>() as u64,
        manifest_path.display()
    );
    Ok(())
}

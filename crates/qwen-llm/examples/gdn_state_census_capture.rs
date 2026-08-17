use anyhow::{Context, Result, ensure};
use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{MetalContext, MetalTensor};
use qwen_llm::metal_forward::{
    MetalForward, MetalModel, MetalSession, gdn_alpha_census_capture_install,
    gdn_alpha_census_capture_uninstall,
};
use qwen_llm::model::ArchKind;
use qwen_llm::tokenizer::Tokenizer;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct CaptureManifest {
    schema_version: u32,
    model: String,
    model_bytes: u64,
    source_text: String,
    consumed_token_ids: Vec<i32>,
    state_positions: Vec<usize>,
    gdn_indices: Vec<usize>,
    gdn_absolute_layers: Vec<usize>,
    value_heads: usize,
    head_dim: usize,
    alpha: TensorManifest,
    states: Vec<StateManifest>,
}

#[derive(Serialize)]
struct TensorManifest {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    byte_length: u64,
    sha256: String,
}

#[derive(Serialize)]
struct StateManifest {
    position: usize,
    gdn_index: usize,
    absolute_layer: usize,
    tensor: TensorManifest,
}

fn parse_indices(raw: &str, label: &str) -> Result<Vec<usize>> {
    let values = raw
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .with_context(|| format!("parse {label} value {value:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!values.is_empty(), "{label} is empty");
    ensure!(
        values.windows(2).all(|pair| pair[0] < pair[1]),
        "{label} must be strictly ascending and unique"
    );
    Ok(values)
}

fn absolute_layer(gdn_index: usize) -> usize {
    4 * (gdn_index / 3) + gdn_index % 3
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

fn tensor_manifest(path: &Path, shape: Vec<usize>) -> Result<TensorManifest> {
    let (byte_length, sha256) = file_sha256(path)?;
    let expected = shape.iter().product::<usize>() * size_of::<f32>();
    ensure!(
        byte_length == expected as u64,
        "{} has {byte_length} bytes, expected {expected}",
        path.display()
    );
    Ok(TensorManifest {
        name: path
            .file_name()
            .context("capture path has no filename")?
            .to_string_lossy()
            .into_owned(),
        dtype: "f32le",
        shape,
        byte_length,
        sha256,
    })
}

fn main() -> Result<()> {
    ensure!(
        std::env::var("QWEN_DECODE_DENSE_CONCURRENT_GDN").as_deref() == Ok("0"),
        "GDN state census requires QWEN_DECODE_DENSE_CONCURRENT_GDN=0 so the serial observer owns every GDN layer"
    );
    let model_path = PathBuf::from(
        std::env::var_os("GDN_CENSUS_MODEL").context("GDN_CENSUS_MODEL is required")?,
    );
    let text_path = PathBuf::from(
        std::env::var_os("GDN_CENSUS_TEXT_FILE").context("GDN_CENSUS_TEXT_FILE is required")?,
    );
    let out_dir = PathBuf::from(
        std::env::var_os("GDN_CENSUS_OUT")
            .unwrap_or_else(|| "target/profiles/gdn-state-census".into()),
    );
    let positions = parse_indices(
        &std::env::var("GDN_CENSUS_POSITIONS").unwrap_or_else(|_| "64,512".into()),
        "GDN_CENSUS_POSITIONS",
    )?;
    let gdn_indices = parse_indices(
        &std::env::var("GDN_CENSUS_INDICES").unwrap_or_else(|_| "0,23,47".into()),
        "GDN_CENSUS_INDICES",
    )?;
    let token_count = *positions.last().expect("non-empty positions");

    let gguf = GgufFile::open(&model_path)
        .with_context(|| format!("open GGUF {}", model_path.display()))?;
    let model = Model::from_gguf(&gguf).context("load model metadata")?;
    ensure!(
        model.arch.kind == ArchKind::Dense,
        "GDN census requires a dense model"
    );
    let n_gdn = (0..model.arch.n_layer)
        .filter(|&layer| model.arch.layer_kind(layer) == qwen_llm::model::LayerKind::GatedDeltaNet)
        .count();
    ensure!(
        gdn_indices.last().copied().unwrap_or_default() < n_gdn,
        "GDN_CENSUS_INDICES exceeds {n_gdn} GDN layers"
    );
    let value_heads = model.arch.gdn_n_v_heads as usize;
    let head_dim = model.arch.gdn_head_dim as usize;
    let state_elements = value_heads * head_dim * head_dim;

    let source_text = std::fs::read_to_string(&text_path)
        .with_context(|| format!("read source text {}", text_path.display()))?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load tokenizer")?;
    let token_ids = tokenizer
        .encode(&source_text, false)
        .context("tokenize source text")?;
    ensure!(
        token_ids.len() >= token_count,
        "source text has {} tokens, needs {token_count}",
        token_ids.len()
    );
    let consumed_token_ids = token_ids[..token_count].to_vec();

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output directory {}", out_dir.display()))?;
    let alpha_path = out_dir.join("alpha.f32le");
    let context = MetalContext::new().context("create Metal context")?;
    let metal_model = MetalModel::load(&context, &gguf, &model).context("load Metal model")?;
    let forward = MetalForward::new(&context, &metal_model);
    let mut session = MetalSession::fresh(&context, &metal_model, token_count + 1)
        .context("create Metal session")?;

    let mut alpha_slots = Vec::with_capacity(n_gdn);
    let mut alpha_buffers = Vec::with_capacity(n_gdn);
    for gdn_index in 0..n_gdn {
        let buffer = MetalTensor::zeros_f32(&context, vec![value_heads as u64])
            .with_context(|| format!("allocate alpha capture for GDN {gdn_index}"))?;
        alpha_slots.push((gdn_index, buffer.clone()));
        alpha_buffers.push(buffer);
    }
    gdn_alpha_census_capture_install(alpha_slots);

    let capture_result = (|| -> Result<Vec<StateManifest>> {
        let mut alpha_writer = BufWriter::new(
            std::fs::File::create(&alpha_path)
                .with_context(|| format!("create {}", alpha_path.display()))?,
        );
        let mut states = Vec::new();
        for (position, &token) in consumed_token_ids.iter().enumerate() {
            forward
                .single_token(token, position as u32, &mut session)
                .with_context(|| format!("forward source token at position {position}"))?;
            for alpha in &alpha_buffers {
                let values = read_f32_tensor(alpha);
                ensure!(
                    values
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0),
                    "captured invalid alpha at position {position}"
                );
                ensure!(
                    values.iter().any(|value| *value > 0.0),
                    "GDN alpha observer was not populated at position {position}"
                );
                alpha_writer
                    .write_all(bytemuck::cast_slice(&values))
                    .with_context(|| format!("write {}", alpha_path.display()))?;
            }

            let following_position = position + 1;
            if positions.binary_search(&following_position).is_ok() {
                let identity = session.snapshot_identity(0, 0);
                let snapshot = session
                    .snapshot(
                        identity,
                        consumed_token_ids[..following_position].to_vec(),
                        None,
                    )
                    .with_context(|| format!("snapshot position {following_position}"))?;
                let state_bytes = state_elements * size_of::<f32>();
                ensure!(
                    snapshot.gdn_state_arena.len() == n_gdn * state_bytes,
                    "snapshot state arena has unexpected size"
                );
                for &gdn_index in &gdn_indices {
                    let start = gdn_index * state_bytes;
                    let path =
                        out_dir.join(format!("state-p{following_position}-gdn{gdn_index}.f32le"));
                    std::fs::write(&path, &snapshot.gdn_state_arena[start..start + state_bytes])
                        .with_context(|| format!("write {}", path.display()))?;
                    states.push(StateManifest {
                        position: following_position,
                        gdn_index,
                        absolute_layer: absolute_layer(gdn_index),
                        tensor: tensor_manifest(&path, vec![value_heads, head_dim, head_dim])?,
                    });
                }
            }
        }
        alpha_writer
            .flush()
            .with_context(|| format!("flush {}", alpha_path.display()))?;
        Ok(states)
    })();
    gdn_alpha_census_capture_uninstall();
    let states = capture_result?;

    let alpha = tensor_manifest(&alpha_path, vec![token_count, n_gdn, value_heads])?;
    let manifest = CaptureManifest {
        schema_version: 1,
        model: model_path.display().to_string(),
        model_bytes: std::fs::metadata(&model_path)?.len(),
        source_text: text_path.display().to_string(),
        consumed_token_ids,
        state_positions: positions,
        gdn_absolute_layers: gdn_indices
            .iter()
            .map(|&index| absolute_layer(index))
            .collect(),
        gdn_indices,
        value_heads,
        head_dim,
        alpha,
        states,
    };
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;
    eprintln!(
        "gdn_state_census: tokens={} state_files={} output={}",
        token_count,
        manifest.states.len(),
        manifest_path.display()
    );
    Ok(())
}

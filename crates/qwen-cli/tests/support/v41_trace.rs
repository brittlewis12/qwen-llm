//! Optional synchronous oracle dumps. Binary files are provisional until the
//! final run.json says success; a failed step/run deliberately leaves evidence.
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::forward::{Config, LayerTrace};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct Shapes {
    hidden: usize,
    vocab: usize,
    layers: usize,
}

impl Shapes {
    fn hidden_elements(self) -> Result<usize> {
        self.hidden
            .checked_mul(4)
            .context("trace hidden shape overflow")
    }
}

struct Step {
    position: usize,
    token: u32,
    next_layer: usize,
}

pub(super) struct Trace {
    root: PathBuf,
    shapes: Shapes,
    manifest: Value,
    positions: Vec<Value>,
    pending: Option<Step>,
}

impl Trace {
    pub(super) fn create(
        path: &Path,
        config: &Config,
        layers: usize,
        prompt: &[u32],
    ) -> Result<Self> {
        Self::create_inner(
            path,
            Shapes {
                hidden: config.hidden,
                vocab: config.vocab,
                layers,
            },
            json!({"prompt_token_ids":prompt, "context":prompt.len(), "max_output_tokens":1,
                "execution":"gpu_step", "comparison_scope":"hf_conversion_and_execution_characterization"}),
        )
    }

    fn create_inner(path: &Path, shapes: Shapes, mut manifest: Value) -> Result<Self> {
        ensure!(
            !path.as_os_str().is_empty(),
            "trace directory must not be empty"
        );
        match fs::symlink_metadata(path) {
            Ok(metadata) => {
                ensure!(
                    metadata.file_type().is_dir(),
                    "trace path must be a non-symlink directory: {}",
                    path.display()
                );
                ensure!(
                    fs::read_dir(path)?.next().transpose()?.is_none(),
                    "trace directory must be empty: {}",
                    path.display()
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(path).with_context(|| {
                    format!(
                        "create trace directory {} (parent must exist)",
                        path.display()
                    )
                })?;
            }
            Err(error) => return Err(error.into()),
        }
        manifest["schema_version"] = json!(1);
        manifest["status"] = json!("incomplete");
        manifest["mode"] = json!("NativeF32");
        manifest["router_tie_policy"] = json!("lower_expert_id_then_id_sorted");
        manifest["dtype"] = json!("float32");
        manifest["byte_order"] = json!("little");
        manifest["artifact_format"] = json!("gguf");
        manifest["provenance_verified"] = json!(false);
        manifest["layer_count"] = json!(shapes.layers);
        manifest["shapes"] =
            json!({"hidden":[4,shapes.hidden],"next_pre_mix":[4],"logits":[shapes.vocab]});
        manifest["bytes_per_file"] = json!({"hidden":byte_count(shapes.hidden_elements()?)?,"pre":byte_count(4)?,"logits":byte_count(shapes.vocab)?});
        manifest["positions"] = json!([]);
        write_json_new(&path.join("run.json"), &manifest)?;
        Ok(Self {
            root: path.to_owned(),
            shapes,
            manifest,
            positions: vec![],
            pending: None,
        })
    }

    pub(super) fn begin_step(&mut self, position: usize, token: u32) -> Result<()> {
        ensure!(self.pending.is_none(), "previous trace step is incomplete");
        ensure!(
            position == self.positions.len(),
            "trace positions must be contiguous from zero"
        );
        ensure!(
            (token as usize) < self.shapes.vocab,
            "trace input token outside vocabulary"
        );
        self.pending = Some(Step {
            position,
            token,
            next_layer: 0,
        });
        Ok(())
    }

    pub(super) fn layer(&mut self, trace: LayerTrace<'_>) -> Result<()> {
        let step = self
            .pending
            .as_mut()
            .context("trace callback without an active step")?;
        ensure!(
            trace.position == step.position
                && trace.layer == step.next_layer
                && trace.layer < self.shapes.layers,
            "trace callback position/layer mismatch"
        );
        ensure!(
            trace.hidden.len() == self.shapes.hidden_elements()? && trace.next_pre_mix.len() == 4,
            "trace callback tensor shape mismatch"
        );
        let prefix = format!("pos-{:05}-layer-{:02}", trace.position, trace.layer);
        write_f32(
            &self.root.join(format!("{prefix}.hidden.f32")),
            trace.hidden,
        )?;
        write_f32(
            &self.root.join(format!("{prefix}.pre.f32")),
            trace.next_pre_mix,
        )?;
        step.next_layer += 1;
        Ok(())
    }

    pub(super) fn finish_step(&mut self, logits: &[f32]) -> Result<()> {
        let step = self.pending.as_ref().context("no active trace step")?;
        ensure!(
            step.next_layer == self.shapes.layers,
            "trace step did not record all layers"
        );
        ensure!(
            logits.len() == self.shapes.vocab,
            "trace logits shape mismatch"
        );
        write_f32(
            &self
                .root
                .join(format!("pos-{:05}-logits.f32", step.position)),
            logits,
        )?;
        self.positions
            .push(json!({"position":step.position,"token_id":step.token}));
        self.pending = None;
        Ok(())
    }

    pub(super) fn finish(mut self, generated: &[u32], stop: Option<u32>) -> Result<()> {
        ensure!(
            self.pending.is_none() && !self.positions.is_empty(),
            "cannot complete an empty/partial trace"
        );
        ensure!(
            !generated.is_empty()
                && generated
                    .iter()
                    .all(|&id| (id as usize) < self.shapes.vocab),
            "invalid generated trace token IDs"
        );
        ensure!(
            stop.is_none() || generated.last().copied() == stop,
            "trace stop token does not match final sample"
        );
        let prompt = self.manifest["prompt_token_ids"]
            .as_array()
            .context("missing trace prompt IDs")?;
        let required = prompt
            .len()
            .checked_add(generated.len() - 1)
            .context("trace position count overflow")?;
        ensure!(
            self.positions.len() == required,
            "trace positions do not match prompt and generated token counts"
        );
        for (index, position) in self.positions.iter().enumerate() {
            let token = if index < prompt.len() {
                prompt[index].as_u64().context("invalid trace prompt ID")?
            } else {
                generated[index - prompt.len()] as u64
            };
            ensure!(
                position["token_id"].as_u64() == Some(token),
                "trace consumed token mismatch at position {index}"
            );
        }
        self.manifest["positions"] = json!(self.positions);
        self.manifest["generated_token_ids"] = json!(generated);
        self.manifest["stop_token_id"] = json!(stop);
        self.manifest["status"] = json!("success");
        // One final manifest replacement, not a per-layer transaction system.
        let temporary = self.root.join("run.json.tmp");
        write_json_new(&temporary, &self.manifest)?;
        fs::rename(temporary, self.root.join("run.json"))?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "v41_trace/tests.rs"]
mod tests;

fn byte_count(elements: usize) -> Result<u64> {
    u64::try_from(
        elements
            .checked_mul(4)
            .context("trace byte count overflow")?,
    )
    .context("trace byte count exceeds u64")
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let expected = byte_count(values.len())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create trace {}", path.display()))?;
    let mut bytes = [0u8; 4096];
    for chunk in values.chunks(bytes.len() / 4) {
        for (value, output) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            output.copy_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes[..chunk.len() * 4])?;
    }
    file.flush()?;
    ensure!(
        file.metadata()?.len() == expected,
        "trace file byte count mismatch: {}",
        path.display()
    );
    Ok(())
}

fn write_json_new(path: &Path, value: &Value) -> Result<()> {
    let mut file = File::create_new(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

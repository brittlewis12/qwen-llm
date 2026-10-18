//! Eight append-only NPY histories and sparse attention checkpoint outputs.
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::forward::capture::{AttentionCapture, Capture, LAYERS};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(super) const DEFAULT_CHECKPOINTS: [usize; 4] = [2046, 2047, 9998, 9999];
const HEADER: usize = 256;

#[cfg(test)]
#[path = "v41_capture/tests.rs"]
mod tests;

pub(super) struct Writer {
    root: PathBuf,
    width: usize,
    checkpoints: Vec<usize>,
    files: Vec<File>,
    rows: [usize; 8],
    next_layer: usize,
    tokens: Vec<u32>,
    manifest: Value,
    finished: bool,
}

impl Writer {
    pub(super) fn create(path: &Path, checkpoints: &[usize], provenance: Value) -> Result<Self> {
        Self::create_inner(path, checkpoints, 5120, provenance)
    }

    fn create_inner(
        path: &Path,
        checkpoints: &[usize],
        width: usize,
        provenance: Value,
    ) -> Result<Self> {
        ensure!(
            !checkpoints.is_empty() && checkpoints.iter().all(|&p| p >= 1),
            "capture checkpoints must be nonempty positions >= 1"
        );
        ensure!(
            checkpoints.windows(2).all(|p| p[0] < p[1]),
            "capture checkpoints must be strictly increasing"
        );
        ensure!(!path.as_os_str().is_empty(), "empty capture directory");
        match fs::symlink_metadata(path) {
            Ok(meta) => {
                ensure!(
                    meta.file_type().is_dir(),
                    "capture path must be a non-symlink directory"
                );
                ensure!(
                    fs::read_dir(path)?.next().transpose()?.is_none(),
                    "capture directory must be empty"
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(path)?,
            Err(e) => return Err(e.into()),
        }
        let manifest = json!({
            "format":"dsv41-long-context-v1", "mode":"NativeF32",
            "status":"incomplete", "provenance":provenance,
            "requested_checkpoints":checkpoints, "inputs":{}, "checkpoints":[],
            "token_ids":[], "hidden_size":width,
        });
        write_json_new(&path.join("manifest.json"), &manifest)?;
        let files = LAYERS
            .iter()
            .map(|id| {
                let mut f = File::create_new(path.join(format!("input-{id:02}.npy")))?;
                header(&mut f, &[0, width])?;
                Ok(f)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            root: path.to_owned(),
            width,
            checkpoints: checkpoints.to_vec(),
            files,
            rows: [0; 8],
            next_layer: 0,
            tokens: vec![],
            manifest,
            finished: false,
        })
    }

    pub(super) fn begin_step(&mut self, position: usize, token: u32) -> Result<()> {
        if !self.wants_position(position) {
            return Ok(());
        }
        ensure!(
            self.next_layer == 0
                && self.rows.iter().all(|&n| n == position)
                && self.tokens.len() == position,
            "capture history must be contiguous from zero"
        );
        self.tokens.push(token);
        Ok(())
    }

    fn record(&mut self, c: AttentionCapture<'_>) -> Result<()> {
        let slot = self.next_layer;
        ensure!(
            c.layer == LAYERS[slot]
                && c.position == self.rows[slot]
                && self.tokens.len() == c.position + 1,
            "capture callback order mismatch"
        );
        ensure!(c.input.len() == self.width, "capture input shape mismatch");
        ensure!(
            c.checkpoint.is_some() == self.is_checkpoint(c.position),
            "capture checkpoint mismatch"
        );
        write_values(&mut self.files[slot], c.input)?;
        self.rows[slot] += 1;
        if let Some(checkpoint) = c.checkpoint {
            let compressed_count = (c.position + 1) / if c.layer < 20 { 2 } else { 1 };
            ensure!(
                checkpoint.compressed_ids.len() == compressed_count.min(512),
                "capture selected ID count mismatch"
            );
            ensure!(
                checkpoint.output.len() == self.width,
                "capture output shape mismatch"
            );
            ensure!(
                checkpoint.kv_owner == if c.layer >= 20 { 20 } else { c.layer }
                    && checkpoint.index_source == c.layer,
                "capture owner mismatch"
            );
            ensure!(
                checkpoint.compressed_ids.iter().all(|&id| id >= 0
                    && (id as usize) < (c.position + 1) / if c.layer < 20 { 2 } else { 1 })
                    && checkpoint.compressed_ids.windows(2).all(|p| p[0] < p[1]),
                "invalid compressed IDs"
            );
            ensure!(
                checkpoint.candidate_mask.is_some() == (c.layer >= 20),
                "missing candidate mask"
            );
            let output = format!("output-{:05}-{:02}.npy", c.position, c.layer);
            let mut f = File::create_new(self.root.join(&output))?;
            header(&mut f, &[self.width])?;
            write_values(&mut f, checkpoint.output)?;
            f.flush()?;
            let entries = self.manifest["checkpoints"].as_array_mut().unwrap();
            if slot == 0 {
                entries.push(json!({"position":c.position,"layers":{}}));
            }
            let mut entry = json!({"output":output,"kv_owner":checkpoint.kv_owner,"index_source":checkpoint.index_source,"compressed_ids":checkpoint.compressed_ids});
            if let Some(mask) = checkpoint.candidate_mask {
                ensure!(
                    mask.iter().all(|&n| n == 0 || n == 1),
                    "invalid candidate mask"
                );
                ensure!(
                    mask.len() == (c.position + 1).div_ceil(8)
                        && mask.iter().filter(|&&n| n == 1).count() == mask.len().min(2048)
                        && mask.last() == Some(&1),
                    "capture candidate block count/newest mismatch"
                );
                entry["candidate_blocks"] = json!(
                    mask.iter()
                        .enumerate()
                        .filter_map(|(id, &keep)| (keep != 0).then_some(id))
                        .collect::<Vec<_>>()
                );
            }
            entries.last_mut().context("missing checkpoint record")?["layers"]
                [c.layer.to_string()] = entry;
        }
        self.next_layer = (slot + 1) % LAYERS.len();
        Ok(())
    }

    fn save(&mut self, status: &str) -> Result<()> {
        for (slot, &layer) in LAYERS.iter().enumerate() {
            header(&mut self.files[slot], &[self.rows[slot], self.width])?;
            self.files[slot].flush()?;
            ensure!(
                self.files[slot].metadata()?.len()
                    == (HEADER + self.rows[slot] * self.width * 4) as u64,
                "capture input byte count mismatch"
            );
            self.manifest["inputs"][layer.to_string()] =
                json!({"file":format!("input-{layer:02}.npy"),"start":0,"rows":self.rows[slot]});
        }
        self.manifest["status"] = json!(status);
        self.manifest["token_ids"] = json!(self.tokens);
        let temp = self.root.join("manifest.json.tmp");
        write_json_new(&temp, &self.manifest)?;
        fs::rename(temp, self.root.join("manifest.json"))?;
        Ok(())
    }

    pub(super) fn failure(&mut self, error: &str, cancelled: bool) {
        self.manifest["reason"] = json!(error);
        self.manifest["cancelled"] = json!(cancelled);
    }

    pub(super) fn finish(mut self) -> Result<()> {
        let count = self.checkpoints.last().unwrap() + 1;
        ensure!(
            self.next_layer == 0 && self.rows.iter().all(|&n| n == count),
            "capture ended before all requested checkpoints; manifest remains incomplete"
        );
        self.save("success")?;
        self.finished = true;
        Ok(())
    }
}

impl Capture for Writer {
    fn wants_position(&self, position: usize) -> bool {
        position <= *self.checkpoints.last().unwrap()
    }
    fn is_checkpoint(&self, position: usize) -> bool {
        self.checkpoints.binary_search(&position).is_ok()
    }
    fn layer(&mut self, capture: AttentionCapture<'_>) -> std::result::Result<(), String> {
        self.record(capture).map_err(|e| format!("{e:#}"))
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.finished {
            if self.manifest["reason"].is_null() {
                self.manifest["reason"] = json!(
                    "run failed, cancelled, or ended before capture completion; partial files are not a successful comparison input"
                );
            }
            if let Err(error) = self.save("incomplete") {
                eprintln!("capture incomplete manifest: {error:#}");
            }
        }
    }
}

fn header(file: &mut File, shape: &[usize]) -> Result<()> {
    let shape = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape},), }}");
    ensure!(text.len() < HEADER - 10, "NPY header overflow");
    let mut bytes = [b' '; HEADER];
    bytes[..8].copy_from_slice(b"\x93NUMPY\x01\x00");
    bytes[8..10].copy_from_slice(&((HEADER - 10) as u16).to_le_bytes());
    bytes[10..10 + text.len()].copy_from_slice(text.as_bytes());
    bytes[HEADER - 1] = b'\n';
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&bytes)?;
    file.seek(SeekFrom::End(0))?;
    Ok(())
}

fn write_values(file: &mut File, values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite capture values"
    );
    let mut bytes = [0; 4096];
    for chunk in values.chunks(1024) {
        for (v, dst) in chunk.iter().zip(bytes.chunks_exact_mut(4)) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        file.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

fn write_json_new(path: &Path, value: &Value) -> Result<()> {
    let mut file = File::create_new(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

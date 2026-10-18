//! Targeted attention capture. Callbacks run only after GPU completion.
use super::*;

#[cfg(test)]
mod tests;

pub const LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

pub struct AttentionCapture<'a> {
    pub position: usize,
    pub layer: usize,
    pub input: &'a [f32],
    pub checkpoint: Option<Checkpoint<'a>>,
}

pub struct Checkpoint<'a> {
    pub output: &'a [f32],
    pub kv_owner: usize,
    pub index_source: usize,
    /// Ascending compressed positions, without a window offset.
    pub compressed_ids: &'a [i32],
    /// Candidate-source block mask (one i32 per block), not token positions.
    pub candidate_mask: Option<&'a [i32]>,
}

pub trait Capture {
    fn wants_position(&self, position: usize) -> bool;
    fn is_checkpoint(&self, position: usize) -> bool;
    fn layer(&mut self, capture: AttentionCapture<'_>) -> Result<()>;
}

pub(super) struct Snapshot {
    selected_count: usize,
    kv_owner: usize,
    index_source: usize,
    input: MetalTensor,
    output: Option<MetalTensor>,
    selected: Option<MetalTensor>,
    candidates: Option<MetalTensor>,
}

impl Snapshot {
    pub(super) fn new(
        ctx: &MetalContext,
        c: &Config,
        position: usize,
        layer: usize,
        checkpoint: bool,
        schedule: &super::super::gguf::Schedule,
    ) -> Result<Self> {
        let tensor = |n: usize, dtype| {
            MetalTensor::zeros_dtype_unstaged(ctx, vec![n.max(1) as u64], dtype)
                .map_err(|e| e.to_string())
        };
        let owner = schedule
            .owner(layer)
            .ok_or("capture layer has no KV owner")?;
        let source = schedule
            .index_source_layers
            .iter()
            .copied()
            .rfind(|&id| id <= layer)
            .ok_or("capture layer has no index source")?;
        let count = ((position + 1) / schedule.ratio(layer)).min(c.index_topk);
        Ok(Self {
            selected_count: count,
            kv_owner: owner,
            index_source: source,
            input: tensor(c.hidden, GgmlType::F32)?,
            output: checkpoint
                .then(|| tensor(c.hidden, GgmlType::F32))
                .transpose()?,
            selected: checkpoint
                .then(|| tensor(count, GgmlType::I32))
                .transpose()?,
            candidates: (checkpoint
                && schedule
                    .candidate_source_layer
                    .is_some_and(|source| layer >= source))
            .then(|| tensor((position + 1).div_ceil(c.candidate_block), GgmlType::I32))
            .transpose()?,
        })
    }

    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        output: &MetalTensor,
        selected: &MetalTensor,
        candidates: &MetalTensor,
        count: usize,
    ) -> Result<()> {
        let copy = |src, dst: &MetalTensor, n| {
            if n == 0 {
                return Ok(());
            }
            let result = if dst.dtype == GgmlType::I32 {
                metal::encode_copy_offset_i32(ctx, enc, src, 0, dst, n)
            } else {
                metal::encode_copy_offset_f32(ctx, enc, src, 0, dst, n)
            };
            result.map_err(|e| e.to_string())
        };
        copy(input, &self.input, self.input.shape[0] as usize)?;
        if let Some(dst) = &self.output {
            copy(output, dst, dst.shape[0] as usize)?;
        }
        if let Some(dst) = &self.selected {
            copy(selected, dst, count)?;
        }
        if let Some(dst) = &self.candidates {
            copy(candidates, dst, dst.shape[0] as usize)?;
        }
        Ok(())
    }

    pub(super) fn deliver(
        &self,
        _c: &Config,
        position: usize,
        layer: usize,
        sink: &mut dyn Capture,
    ) -> Result<()> {
        let ints = |t: &MetalTensor, n| unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add((t.offset / 4) as usize),
                n,
            )
        };
        let checkpoint = if let Some(output) = &self.output {
            Some(Checkpoint {
                output: f32_slice(output)?,
                kv_owner: self.kv_owner,
                index_source: self.index_source,
                compressed_ids: ints(self.selected.as_ref().unwrap(), self.selected_count),
                candidate_mask: self
                    .candidates
                    .as_ref()
                    .map(|t| ints(t, t.shape[0] as usize)),
            })
        } else {
            None
        };
        sink.layer(AttentionCapture {
            position,
            layer,
            input: f32_slice(&self.input)?,
            checkpoint,
        })
    }
}

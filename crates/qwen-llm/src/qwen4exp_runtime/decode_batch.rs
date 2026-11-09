//! Native independent decode seam. No consecutive-token or prefill vehicle.
use super::*;
use crate::qwen4exp_text_session::{
    QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES, decode_batch as encode,
};

pub struct Qwen4ExpDecodeBatch {
    head: MetalTensor,
    geometry: crate::qwen4exp_text_session::Qwen4ExpTextSessionMetalGeometry,
    inputs: MetalTensor,
    logits: MetalTensor,
    maximum: usize,
    projection: crate::qwen4exp_metal::decode_projection::Scratch,
}
struct Round<'a, 's> {
    members: &'a mut [&'s mut Qwen4ExpTextSessionMetalWorkspace],
    started: bool,
    submitted: bool,
    completed: bool,
}
impl Drop for Round<'_, '_> {
    fn drop(&mut self) {
        if !self.started || self.completed {
            return;
        }
        for s in self.members.iter_mut() {
            if self.submitted {
                let _ = s.release_after();
            } else {
                unsafe {
                    let _ = s.abandon_uncommitted();
                }
            }
        }
        encode::poison(self.members);
    }
}
impl Qwen4ExpDecodeBatch {
    pub fn new(
        runner: &Qwen4ExpTextRunner<'_, '_, '_>,
        maximum: usize,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        if !(1..=4).contains(&maximum)
            || !runner.ctx.tensor_ops_status().is_available()
            || runner.weights.output.dtype != GgmlType::Q6_K
        {
            return invalid("native batch requires M5, Q6 head and 1..=4 members");
        }
        let columns = maximum.next_power_of_two();
        let ctx = runner.ctx;
        let h = runner.weights.geometry.hidden_size();
        let v = runner.weights.geometry.vocab_size();
        let (mut max_in, mut max_out) = (h, h);
        for block in &runner.weights.post_ple {
            use crate::qwen4exp_post_ple_block::Qwen4ExpPostPleMixerMetalWeights as M;
            let weights: Vec<_> = match block.mixer {
                M::GatedDeltaNet(w) => vec![w.qkv, w.gate, w.output],
                M::QwenSparseAttention(w) => vec![
                    w.index_query,
                    w.index_key,
                    w.query,
                    w.key,
                    w.value,
                    w.output,
                ],
            };
            for w in weights {
                max_in = max_in.max(w.shape[0] as usize);
                max_out = max_out.max(w.shape[1] as usize);
            }
        }
        let bytes = |n: usize| {
            n.checked_mul(columns)
                .and_then(|n| n.checked_mul(4))
                .map(|n| n as u64)
                .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("native batch extent overflow".into()))
        };
        let (input_bytes, output_bytes) = (bytes(h)?, bytes(v)?);
        let _allocation = ctx.begin_allocation_transaction();
        let price = |n| {
            ctx.price_shared_buffer_upper(n)
                .map(|p| p.priced_upper_bytes)
                .map_err(|e| Qwen4ExpRuntimeError::Invalid(format!("native batch price: {e}")))
        };
        let priced = [input_bytes, output_bytes, bytes(max_in)?, bytes(max_out)?]
            .into_iter()
            .try_fold(0u64, |n, b| {
                n.checked_add(price(b)?).ok_or_else(|| {
                    Qwen4ExpRuntimeError::Invalid("native batch price overflow".into())
                })
            })?;
        let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            priced,
            output_bytes * 2,
            QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(denied.into());
        }
        Ok(Self {
            head: runner.weights.output.clone(),
            geometry: runner.weights.geometry.clone(),
            inputs: MetalTensor::zeros_f32(ctx, vec![h as u64, columns as u64])?,
            logits: MetalTensor::zeros_f32(ctx, vec![v as u64, columns as u64])?,
            maximum,
            projection: crate::qwen4exp_metal::decode_projection::Scratch {
                inputs: MetalTensor::zeros_f32(ctx, vec![max_in as u64, columns as u64])?,
                outputs: MetalTensor::zeros_f32(ctx, vec![max_out as u64, columns as u64])?,
            },
        })
    }
    pub fn cpu_reserve_bytes(&self) -> u64 {
        self.logits.n_bytes() * 2
    }
    pub fn step(
        &mut self,
        members: &mut [&mut Qwen4ExpTextRunner<'_, '_, '_>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, Qwen4ExpRuntimeError> {
        if members.is_empty() || members.len() > self.maximum || members.len() != tokens.len() {
            return invalid("invalid native batch width");
        }
        let ctx = members[0].ctx;
        let weights = members[0].weights.clone();
        let table = members[0].ple_table;
        for (s, &t) in members.iter().zip(tokens) {
            if !std::ptr::eq(ctx, s.ctx)
                || !std::ptr::eq(
                    Retained::as_ptr(&s.weights.output.buffer),
                    Retained::as_ptr(&self.head.buffer),
                )
                || s.weights.output.offset != self.head.offset
                || !std::ptr::eq(s.weights.output, weights.output)
                || s.weights.geometry != self.geometry
                || s.weights.geometry != weights.geometry
            {
                return invalid("native batch weights/device/geometry differ");
            }
            s.validate_prefill_request(&[t])?;
        }
        if members.len() == 1 {
            return Ok(vec![members[0].forward_token(tokens[0])?.to_vec()]);
        }
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("no native batch command".into()))?;
        let mut workspaces: Vec<_> = members.iter_mut().map(|s| &mut s.workspace).collect();
        let mut round = Round {
            members: &mut workspaces,
            started: false,
            submitted: false,
            completed: false,
        };
        let enc = KernelEncoder::begin(&command);
        let columns = tokens.len().next_power_of_two();
        let inputs = self.inputs.view_subrange(
            0,
            vec![weights.geometry.hidden_size() as u64, columns as u64],
        );
        let logits = self.logits.view_subrange(
            0,
            vec![weights.geometry.vocab_size() as u64, columns as u64],
        );
        let result = encode::encode(
            ctx,
            &enc,
            tokens,
            table,
            &weights,
            round.members,
            &inputs,
            &logits,
            &mut round.started,
            &self.projection,
        );
        enc.end();
        result?;
        command.commit();
        round.submitted = true;
        for s in round.members.iter_mut() {
            s.release_after()?;
        }
        let values: Vec<_> = round
            .members
            .iter()
            .map(|s| s.logits().map(|v| v.to_vec()))
            .collect::<Result<_, _>>()?;
        if values.iter().flatten().any(|x| !x.is_finite()) {
            return invalid("nonfinite native batch logits");
        }
        round.completed = true;
        drop(round);
        drop(workspaces);
        for s in members {
            s.last_token_timing = None;
            s.last_prefill_timing = None;
        }
        Ok(values)
    }
}

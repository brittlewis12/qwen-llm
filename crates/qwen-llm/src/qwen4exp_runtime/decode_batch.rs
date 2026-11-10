//! Native independent decode seam. No consecutive-token or prefill vehicle.
use super::*;
use crate::qwen4exp_text_session::{
    QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES, decode_batch as encode,
};
crate::env_flag!(
    /// Read each member's token from a device buffer and stage PLE/embedding/controls on the GPU. 0 restores the CPU PLE row gather.
    default_on gpu_inputs_enabled, "QWEN_FN_BATCH_GPU_INPUTS"
);

pub struct Qwen4ExpDecodeBatch {
    head: MetalTensor,
    geometry: crate::qwen4exp_text_session::Qwen4ExpTextSessionMetalGeometry,
    inputs: MetalTensor,
    logits: MetalTensor,
    tokens: Vec<MetalTensor>,
    gpu_samplers: Vec<
        Option<(
            crate::sampling::SamplingConfig,
            crate::metal::GpuTokenSampler,
        )>,
    >,
    maximum: usize,
    #[cfg(feature = "perf-ablations")]
    pub diagnostic_ms: [f64; 3],
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
            tokens: (0..maximum)
                .map(|_| MetalTensor::zeros_i32(ctx, vec![1]))
                .collect::<Result<_, _>>()?,
            gpu_samplers: (0..maximum).map(|_| None).collect(),
            maximum,
            #[cfg(feature = "perf-ablations")]
            diagnostic_ms: [0.; 3],
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
        if gpu_inputs_enabled() && members.len() > 1 {
            return self.step_gpu_inputs(members, tokens);
        }
        self.step_cpu_inputs(members, tokens)
    }
    /// Device-input batch forward. The host samples the token as usual, writes
    /// it into the member's device token buffer, and the GPU input prefix gathers
    /// PLE rows, selects the embedding and writes the control scalars. This
    /// removes the CPU PLE gather from every batched round.
    pub fn step_gpu_inputs(
        &mut self,
        members: &mut [&mut Qwen4ExpTextRunner<'_, '_, '_>],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, Qwen4ExpRuntimeError> {
        let b = members.len();
        if !(2..=self.maximum).contains(&b) || tokens.len() != b {
            return invalid("invalid native GPU-input batch width");
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
        let columns = b.next_power_of_two();
        let h = self.geometry.hidden_size();
        let v = self.geometry.vocab_size();
        let inputs = self.inputs.view_subrange(0, vec![h as u64, columns as u64]);
        let logits = self.logits.view_subrange(0, vec![v as u64, columns as u64]);
        let mut workspaces: Vec<_> = members.iter_mut().map(|r| &mut r.workspace).collect();
        let mut diagnostic = [0f64; 3];
        let values = crate::qwen4exp_text_session::step_batch_gpu_inputs(
            ctx,
            table,
            &weights,
            &inputs,
            &logits,
            &self.projection,
            &mut workspaces,
            tokens,
            &self.tokens[..b],
            &mut diagnostic,
        )?;
        #[cfg(feature = "perf-ablations")]
        {
            self.diagnostic_ms = diagnostic;
        }
        #[cfg(not(feature = "perf-ablations"))]
        let _ = diagnostic;
        for r in members.iter_mut() {
            r.last_token_timing = None;
            r.last_prefill_timing = None;
        }
        Ok(values)
    }
    /// Device-input batch forward that samples every member on the GPU at the
    /// end of the command. The host writes the pending token and reads back only
    /// the next tokens; the next round is still encoded after this wait.
    pub fn step_tokens(
        &mut self,
        members: &mut [&mut Qwen4ExpTextRunner<'_, '_, '_>],
        samplers: &mut [&mut crate::sampling::Sampler],
        tokens: &[u32],
    ) -> Result<Vec<u32>, Qwen4ExpRuntimeError> {
        let b = members.len();
        if !(2..=self.maximum).contains(&b) || tokens.len() != b || samplers.len() != b {
            return invalid("invalid native GPU-sampling batch width");
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
        let vocab = self.geometry.vocab_size();
        for (i, sampler) in samplers.iter().enumerate() {
            let config = sampler.config();
            let rebuild = match &self.gpu_samplers[i] {
                Some((cached, _)) => *cached != config,
                None => true,
            };
            if rebuild {
                self.gpu_samplers[i] = Some((
                    config,
                    crate::metal::GpuTokenSampler::new(ctx, vocab, config)?,
                ));
            }
        }
        let uniforms: Vec<f32> = samplers.iter().map(|s| s.gpu_uniform(0)).collect();
        let gpu_samplers: Vec<&crate::metal::GpuTokenSampler> = self.gpu_samplers[..b]
            .iter()
            .map(|s| &s.as_ref().expect("GPU sampler").1)
            .collect();
        let columns = b.next_power_of_two();
        let h = self.geometry.hidden_size();
        let v = self.geometry.vocab_size();
        let inputs = self.inputs.view_subrange(0, vec![h as u64, columns as u64]);
        let logits = self.logits.view_subrange(0, vec![v as u64, columns as u64]);
        let mut workspaces: Vec<_> = members.iter_mut().map(|r| &mut r.workspace).collect();
        let mut diagnostic = [0f64; 3];
        let values = crate::qwen4exp_text_session::step_batch_gpu_tokens(
            ctx,
            table,
            &weights,
            &inputs,
            &logits,
            &self.projection,
            &mut workspaces,
            tokens,
            &self.tokens[..b],
            &gpu_samplers,
            &uniforms,
            &mut diagnostic,
        )?;
        for sampler in samplers.iter_mut() {
            sampler.commit_gpu_draw();
        }
        #[cfg(feature = "perf-ablations")]
        {
            self.diagnostic_ms = diagnostic;
        }
        #[cfg(not(feature = "perf-ablations"))]
        let _ = diagnostic;
        for r in members.iter_mut() {
            r.last_token_timing = None;
            r.last_prefill_timing = None;
        }
        Ok(values)
    }
    /// CPU-staged batch forward: the parent gathers PLE rows and writes the
    /// token IDs, then the GPU runs the shared batch kernels.
    pub fn step_cpu_inputs(
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
        #[cfg(feature = "perf-ablations")]
        let start = std::time::Instant::now();
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
        #[cfg(feature = "perf-ablations")]
        let causal = crate::metal::fn_batch_probe::causal();
        #[cfg(not(feature = "perf-ablations"))]
        let causal = false;
        let enc = if causal {
            KernelEncoder::begin_causal(&command)
        } else {
            KernelEncoder::begin(&command)
        };
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
        if causal {
            if result.is_ok() {
                enc.finish_causal_scope(ctx, false)?;
            } else {
                enc.cancel_causal_scope();
            }
        }
        enc.end();
        result?;
        #[cfg(feature = "perf-ablations")]
        let encode_ms = start.elapsed().as_secs_f64() * 1000.;
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
        #[cfg(feature = "perf-ablations")]
        {
            self.diagnostic_ms = [
                encode_ms,
                (command.GPUEndTime() - command.GPUStartTime()) * 1000.,
                start.elapsed().as_secs_f64() * 1000.,
            ];
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

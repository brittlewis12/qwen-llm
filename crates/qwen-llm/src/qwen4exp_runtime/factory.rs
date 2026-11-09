//! Borrow resident weights once while independent requests own their runners.
use super::*;

pub struct Qwen4ExpSessionFactory<'model, 'gguf> {
    resident: &'model Qwen4ExpMetalWeights,
    weights: Qwen4ExpTextSessionMetalWeights<'model>,
    table: PleIq4NlTable<'gguf>,
    plan: Qwen4ExpTextSessionPlan,
    capacity: Qwen4ExpSessionCapacity,
    decode: Qwen4ExpDecodeOptions,
    gpu_ple_banks: crate::qwen4exp_text_session::GpuPleBankCache,
    device: u64,
}

impl<'gguf> Qwen4ExpLoadedModel<'gguf> {
    /// Lend the resident weights and move the existing workspace into the first
    /// independent runner. No new allocation or state reset. The model stays
    /// borrowed until all runners and the factory have been dropped.
    pub fn session_factory<'ctx, 'model>(
        &'model mut self,
        ctx: &'ctx MetalContext,
    ) -> Result<
        (
            Qwen4ExpSessionFactory<'model, 'gguf>,
            Qwen4ExpTextRunner<'ctx, 'model, 'gguf>,
        ),
        Qwen4ExpRuntimeError,
    > {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid("session factory belongs to another Metal device");
        }
        let weights = Qwen4ExpTextSessionMetalWeights::bind(
            &self.weights,
            self.capacity.qsa_physical_capacity,
        )?;
        let workspace = self.workspace.take().ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("loaded model session was consumed".into())
        })?;
        let decode = Qwen4ExpDecodeOptions {
            causal_executor: workspace.causal_executor,
            gpu_token_loop: workspace.gpu_token_loop,
            guarded_topk: workspace.guarded_topk_enabled(),
            hc_up_mix: workspace.hc_up_mix_enabled(),
        };
        let factory = Qwen4ExpSessionFactory {
            resident: &self.weights,
            weights: weights.clone(),
            table: self.ple_table,
            plan: self.session_plan.clone(),
            capacity: self.capacity,
            decode,
            gpu_ple_banks: workspace.gpu_ple_banks.clone(),
            device: self.device_registry_id,
        };
        let runner = Qwen4ExpTextRunner {
            ctx,
            weights,
            ple_table: self.ple_table,
            workspace,
            capacity: self.capacity,
            last_token_timing: None,
            last_prefill_timing: None,
        };
        Ok((factory, runner))
    }
}

impl<'model, 'gguf> Qwen4ExpSessionFactory<'model, 'gguf> {
    /// Keep the normal admitted prefill geometry for every live state. Admission
    /// is refreshed under the allocator transaction; active peers remain priced.
    pub fn create_runner<'ctx>(
        &self,
        ctx: &'ctx MetalContext,
    ) -> Result<Qwen4ExpTextRunner<'ctx, 'model, 'gguf>, Qwen4ExpRuntimeError> {
        self.create_runner_with_reserve(ctx, 0)
    }

    pub fn create_runner_with_reserve<'ctx>(
        &self,
        ctx: &'ctx MetalContext,
        cpu_reserve: u64,
    ) -> Result<Qwen4ExpTextRunner<'ctx, 'model, 'gguf>, Qwen4ExpRuntimeError> {
        if ctx.device.registryID() != self.device {
            return invalid("session factory device differs");
        }
        let _transaction = ctx.begin_allocation_transaction();
        let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            self.plan.memory_plan().session_priced_upper_bytes(),
            cpu_reserve,
            crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(denied.into());
        }
        let admitted = self
            .plan
            .clone()
            .admit_after_residency(self.resident, ctx.memory_signals())?;
        let mut workspace = Qwen4ExpTextSessionMetalWorkspace::from_admitted(ctx, admitted)?;
        workspace.gpu_ple_banks = self.gpu_ple_banks.clone();
        workspace.causal_executor = self.decode.causal_executor;
        workspace.gpu_token_loop = self.decode.gpu_token_loop;
        workspace.configure_guarded_topk(ctx, self.decode.guarded_topk)?;
        workspace.configure_hc_up_mix(ctx, self.decode.hc_up_mix)?;
        Ok(Qwen4ExpTextRunner {
            ctx,
            weights: self.weights.clone(),
            ple_table: self.table,
            workspace,
            capacity: self.capacity,
            last_token_timing: None,
            last_prefill_timing: None,
        })
    }

    pub fn retained_buffers(&self) -> Vec<&crate::metal::Buffer> {
        crate::metal::retained_gguf_buffers(self.resident.tensors().map(|(_, tensor)| tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn command_quanta_preserve_frontier_geometry_and_scalar_tails() {
        let ranges = prefill_command_ranges_from(0, 16384, Some(2048), true, 2051).unwrap();
        assert_eq!(&ranges[..3], &[0..2048, 2048..2051, 2051..4099]);
        assert_eq!(ranges.last().unwrap().end, 16384);
        for range in ranges {
            assert_eq!(
                prefill_command_ranges_from(range.start, range.len(), Some(2048), true, 2051)
                    .unwrap(),
                vec![range]
            );
        }
        assert_eq!(
            prefill_command_ranges_from(0, 4096, Some(2048), true, 2051).unwrap(),
            vec![0..2048, 2048..4096]
        );
        assert_eq!(
            prefill_command_ranges_from(8192, 2049, Some(2048), true, 2051).unwrap(),
            vec![8192..10240, 10240..10241]
        );
    }
    #[test]
    fn command_quanta_keep_unadmitted_selected_tokens_scalar() {
        let expected: Vec<_> = (2051..2070).map(|p| p..p + 1).collect();
        assert_eq!(
            prefill_command_ranges_from(2051, 19, Some(2048), false, 2051).unwrap(),
            expected
        );
        assert_eq!(
            prefill_command_ranges_from(2051, 19, None, false, 2051).unwrap(),
            expected
        );
        assert!(
            prefill_command_ranges_from(8192, 0, Some(2048), true, 2051)
                .unwrap()
                .is_empty()
        );
    }
}

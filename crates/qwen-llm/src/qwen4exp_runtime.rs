//! Synchronous single-session runtime for Qwen3.8-Flash-Next text generation.

use crate::gguf::GgufFile;
use crate::metal::{KernelEncoder, MetalContext, MetalMemoryAdmission};
use crate::qwen4exp::{Qwen4ExpConfig, Qwen4ExpError};
use crate::qwen4exp_ple::PleIq4NlTable;
use crate::qwen4exp_residency::{
    Qwen4ExpMetalWeightPlan, Qwen4ExpMetalWeights, Qwen4ExpResidencyError,
};
use crate::qwen4exp_text_session::{
    Qwen4ExpCompletedLogits, Qwen4ExpTextSessionError, Qwen4ExpTextSessionMetalWeights,
    Qwen4ExpTextSessionMetalWorkspace, Qwen4ExpTextSessionPlan, encode_qwen4exp_text_token,
};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDevice};

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpRuntimeError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error(transparent)]
    Session(#[from] Qwen4ExpTextSessionError),
    #[error("invalid Qwen3.8-Flash-Next runtime contract: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpSessionCapacity {
    forward_limit: usize,
    qsa_physical_capacity: usize,
}

impl Qwen4ExpSessionCapacity {
    pub fn for_forward_limit(
        config: &Qwen4ExpConfig,
        forward_limit: usize,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        config.validate()?;
        if forward_limit == 0 {
            return invalid("forward limit must be nonzero");
        }
        let context_length = config.context_length as usize;
        if forward_limit > context_length {
            return invalid(format!(
                "forward limit {forward_limit} exceeds model context {context_length}"
            ));
        }
        let alignment = config
            .compress_ratios
            .iter()
            .copied()
            .filter(|ratio| *ratio != 0)
            .try_fold(1_usize, |alignment, ratio| {
                checked_lcm(alignment, ratio as usize)
            })?;
        if alignment == 1 {
            return invalid("text runtime requires at least one compressed QSA layer");
        }
        let rounded = forward_limit
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or_else(|| {
                Qwen4ExpRuntimeError::Invalid("QSA physical capacity overflow".into())
            })?;
        let qsa_physical_capacity = if rounded <= context_length {
            rounded
        } else if context_length.is_multiple_of(alignment) {
            context_length
        } else {
            return invalid(format!(
                "model context {context_length} cannot hold aligned QSA capacity for {forward_limit} forwards"
            ));
        };
        Ok(Self {
            forward_limit,
            qsa_physical_capacity,
        })
    }

    pub fn forward_limit(self) -> usize {
        self.forward_limit
    }

    pub fn qsa_physical_capacity(self) -> usize {
        self.qsa_physical_capacity
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qwen4ExpRuntimeAdmission {
    pub aggregate: MetalMemoryAdmission,
    pub weights: MetalMemoryAdmission,
    pub session: MetalMemoryAdmission,
}

pub struct Qwen4ExpLoadedModel<'gguf> {
    weights: Qwen4ExpMetalWeights,
    ple_table: PleIq4NlTable<'gguf>,
    workspace: Option<Qwen4ExpTextSessionMetalWorkspace>,
    capacity: Qwen4ExpSessionCapacity,
    admission: Qwen4ExpRuntimeAdmission,
    observed_weight_bytes: u64,
    device_registry_id: u64,
}

impl<'gguf> Qwen4ExpLoadedModel<'gguf> {
    pub fn load(
        ctx: &MetalContext,
        gguf: &'gguf GgufFile,
        capacity: Qwen4ExpSessionCapacity,
    ) -> Result<Self, Qwen4ExpRuntimeError> {
        let weight_plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(ctx, gguf)?;
        let expected_capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            weight_plan.config(),
            capacity.forward_limit,
        )?;
        if expected_capacity != capacity {
            return invalid("session capacity differs from the released model geometry");
        }
        let session_plan = Qwen4ExpTextSessionPlan::for_config(
            ctx,
            weight_plan.config(),
            capacity.qsa_physical_capacity,
            weight_plan.memory_plan(),
        )?;

        let _allocation_transaction = ctx.begin_allocation_transaction();
        let aggregate = session_plan
            .memory_plan()
            .admission_before_residency(ctx.memory_signals(), 1)?;
        if !aggregate.admitted {
            return invalid(format!(
                "combined weight and session admission denied: reason={} required={:?}",
                aggregate.reason.as_str(),
                aggregate.required_bytes
            ));
        }
        let admitted_weights = weight_plan.admit(ctx.memory_signals())?;
        let realized = Qwen4ExpMetalWeights::realize(ctx, gguf, admitted_weights)?;
        let weight_admission = realized.admission();
        let observed_weight_bytes = realized.observed_allocation_delta();
        let weights = realized.into_weights();
        let ple_table = weights.ple_source().bind(gguf)?;
        let admitted_session =
            session_plan.admit_after_residency(&weights, ctx.memory_signals())?;
        let workspace = Qwen4ExpTextSessionMetalWorkspace::from_admitted(ctx, admitted_session)?;
        let session_admission = workspace.admission();

        Ok(Self {
            weights,
            ple_table,
            workspace: Some(workspace),
            capacity,
            admission: Qwen4ExpRuntimeAdmission {
                aggregate,
                weights: weight_admission,
                session: session_admission,
            },
            observed_weight_bytes,
            device_registry_id: ctx.device.registryID(),
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        self.weights.config()
    }

    pub fn capacity(&self) -> Qwen4ExpSessionCapacity {
        self.capacity
    }

    pub fn admission(&self) -> Qwen4ExpRuntimeAdmission {
        self.admission
    }

    pub fn observed_weight_bytes(&self) -> u64 {
        self.observed_weight_bytes
    }

    pub fn observed_session_bytes(&self) -> u64 {
        self.workspace.as_ref().map_or(
            0,
            Qwen4ExpTextSessionMetalWorkspace::observed_allocation_delta,
        )
    }

    pub fn create_runner<'ctx, 'model>(
        &'model mut self,
        ctx: &'ctx MetalContext,
    ) -> Result<Qwen4ExpTextRunner<'ctx, 'model, 'gguf>, Qwen4ExpRuntimeError> {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "loaded model belongs to Metal device registry {}, runner context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let weights = Qwen4ExpTextSessionMetalWeights::bind(
            &self.weights,
            self.capacity.qsa_physical_capacity,
        )?;
        let workspace = self.workspace.take().ok_or_else(|| {
            Qwen4ExpRuntimeError::Invalid("loaded model session was consumed".into())
        })?;
        Ok(Qwen4ExpTextRunner {
            ctx,
            weights,
            ple_table: self.ple_table,
            workspace,
            capacity: self.capacity,
        })
    }
}

pub struct Qwen4ExpTextRunner<'ctx, 'model, 'gguf> {
    ctx: &'ctx MetalContext,
    weights: Qwen4ExpTextSessionMetalWeights<'model>,
    ple_table: PleIq4NlTable<'gguf>,
    workspace: Qwen4ExpTextSessionMetalWorkspace,
    capacity: Qwen4ExpSessionCapacity,
}

impl Qwen4ExpTextRunner<'_, '_, '_> {
    pub fn capacity(&self) -> Qwen4ExpSessionCapacity {
        self.capacity
    }

    pub fn next_position(&self) -> usize {
        self.workspace.committed_length()
    }

    pub fn remaining_forwards(&self) -> usize {
        self.capacity
            .forward_limit
            .saturating_sub(self.next_position())
    }

    pub fn logits(&self) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError> {
        Ok(self.workspace.logits()?)
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
    ) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError> {
        if self.next_position() >= self.capacity.forward_limit {
            return invalid(format!(
                "logical forward limit {} is exhausted",
                self.capacity.forward_limit
            ));
        }
        forward_qwen4exp_text_token_sync(
            self.ctx,
            token_id,
            self.ple_table,
            &self.weights,
            &mut self.workspace,
        )
    }

    pub fn prefill(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Qwen4ExpCompletedLogits<'_>, Qwen4ExpRuntimeError> {
        if token_ids.is_empty() {
            return invalid("prompt token sequence must be nonempty");
        }
        if token_ids.len() > self.remaining_forwards() {
            return invalid(format!(
                "prompt requires {} forwards but only {} remain",
                token_ids.len(),
                self.remaining_forwards()
            ));
        }
        let (last, prefix) = token_ids
            .split_last()
            .expect("nonempty prompt has a final token");
        for &token_id in prefix {
            let _ = self.forward_token(token_id)?;
        }
        self.forward_token(*last)
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpRuntimeError> {
        self.workspace.reset()?;
        Ok(())
    }
}

pub fn forward_qwen4exp_text_token_sync<'a>(
    ctx: &MetalContext,
    token_id: u32,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpTextSessionMetalWorkspace,
) -> Result<Qwen4ExpCompletedLogits<'a>, Qwen4ExpRuntimeError> {
    let position = workspace.committed_length();
    let command = ctx.queue.commandBuffer().ok_or_else(|| {
        Qwen4ExpRuntimeError::Invalid("Metal command queue returned no command buffer".into())
    })?;
    let encoder = KernelEncoder::begin(&command);
    let pending = match encode_qwen4exp_text_token(
        ctx, &encoder, token_id, position, table, weights, workspace,
    ) {
        Ok(pending) => pending,
        Err(error) => {
            encoder.end();
            let abandon = unsafe { workspace.abandon_uncommitted() };
            drop(command);
            if let Err(abandon_error) = abandon {
                return invalid(format!(
                    "token encode failed ({error}); abandoning its command also failed ({abandon_error})"
                ));
            }
            return Err(error.into());
        }
    };
    drop(pending);
    encoder.end();
    command.commit();
    workspace.release_after()?;
    Ok(workspace.logits()?)
}

fn checked_lcm(left: usize, right: usize) -> Result<usize, Qwen4ExpRuntimeError> {
    let divisor = gcd(left, right);
    left.checked_div(divisor)
        .and_then(|value| value.checked_mul(right))
        .ok_or_else(|| Qwen4ExpRuntimeError::Invalid("QSA alignment overflow".into()))
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpRuntimeError> {
    Err(Qwen4ExpRuntimeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{Sampler, SamplingConfig};
    use crate::tokenizer::Tokenizer;

    #[test]
    fn request_forward_limit_rounds_only_physical_qsa_rows() {
        let config = Qwen4ExpConfig::flash_next_reference();
        for (forward_limit, physical) in [
            (1, 4),
            (4, 4),
            (5, 8),
            (262_143, 262_144),
            (262_144, 262_144),
        ] {
            let capacity =
                Qwen4ExpSessionCapacity::for_forward_limit(&config, forward_limit).unwrap();
            assert_eq!(capacity.forward_limit(), forward_limit);
            assert_eq!(capacity.qsa_physical_capacity(), physical);
        }
        assert!(Qwen4ExpSessionCapacity::for_forward_limit(&config, 0).is_err());
        assert!(Qwen4ExpSessionCapacity::for_forward_limit(&config, 262_145).is_err());
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_RUNTIME_GGUF to the pinned full release"]
    fn released_runner_generates_expected_text_and_replays_prompt_logits() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_RUNTIME_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let tokenizer = Tokenizer::from_gguf(&gguf).expect("load released tokenizer");
        let prompt = "<|im_start|>user\nReply with exactly: HELLO<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let prompt_tokens = tokenizer.encode(prompt, false).unwrap();
        assert_eq!(prompt_tokens.len(), 18);
        let prompt_tokens = prompt_tokens
            .into_iter()
            .map(|token| u32::try_from(token).unwrap())
            .collect::<Vec<_>>();
        let ctx = MetalContext::new().expect("initialize Metal");
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(
            &Qwen4ExpConfig::flash_next_reference(),
            prompt_tokens.len() + 7,
        )
        .unwrap();
        let mut loaded = Qwen4ExpLoadedModel::load(&ctx, &gguf, capacity).unwrap();
        let mut runner = loaded.create_runner(&ctx).unwrap();
        let first = runner.prefill(&prompt_tokens).unwrap().to_vec();
        let stop_tokens = gguf.stop_token_ids().unwrap();
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let mut logits = first.clone();
        let mut generated = Vec::new();
        let mut output = Vec::new();
        for _ in 0..8 {
            let token = sampler.sample(&logits).unwrap().token;
            generated.push(token);
            if stop_tokens.contains(&token) {
                break;
            }
            output.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(token).unwrap());
            logits = runner
                .forward_token(u32::try_from(token).unwrap())
                .unwrap()
                .to_vec();
        }
        assert_eq!(generated, [49_006, 1_537, 248_046]);
        assert_eq!(output, b"HELLO");
        assert_eq!(runner.next_position(), prompt_tokens.len() + 2);
        runner.reset().unwrap();
        assert_eq!(runner.next_position(), 0);
        assert!(runner.logits().is_err());
        let replay = runner.prefill(&prompt_tokens).unwrap().to_vec();
        assert_eq!(first, replay);
    }
}

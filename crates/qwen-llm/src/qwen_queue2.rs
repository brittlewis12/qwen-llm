use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalAdditionalQueueResidencySetGuard, MetalBlock, MetalForward, MetalModel,
    MetalSession, MfError, kv_cache_dtype_for_arch,
};
use crate::model::ArchKind;
use crate::sampling::{GreedySelection, SamplingError};
use crate::tensor::{GgmlType, ggml_type_layout_raw};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

pub(crate) const QWEN_QUEUE2_WIDTH: usize = 2;
pub(crate) const QWEN_QUEUE2_DYNAMIC_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum QwenQueue2Error {
    #[error("Qwen queue2 validation: {0}")]
    Validation(String),
    #[error("Qwen queue2 cancelled before command commit")]
    CancelledBeforeCommit,
    #[error("Qwen queue2 executor is poisoned after a committed failure")]
    Poisoned,
    #[error("Qwen queue2 command {slot} failed: status={status} error={error}")]
    CommandBuffer {
        slot: usize,
        status: String,
        error: String,
    },
    #[error("Qwen queue2 greedy selection failed in slot {slot}: {source}")]
    GreedySelection {
        slot: usize,
        #[source]
        source: SamplingError,
    },
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QwenQueue2Step {
    pub argmax_ids: [i32; QWEN_QUEUE2_WIDTH],
    pub gpu_ms: [Option<f64>; QWEN_QUEUE2_WIDTH],
}

pub(crate) struct QwenQueue2Executor<'a> {
    forward: MetalForward<'a>,
    _queue_residency: [Option<MetalAdditionalQueueResidencySetGuard>; QWEN_QUEUE2_WIDTH],
    contexts: [MetalContext; QWEN_QUEUE2_WIDTH],
    ids: [MetalTensor; QWEN_QUEUE2_WIDTH],
    selections: [MetalTensor; QWEN_QUEUE2_WIDTH],
    poisoned: bool,
}

impl<'a> QwenQueue2Executor<'a> {
    pub fn new(ctx: &'a MetalContext, model: &'a MetalModel) -> Result<Self, QwenQueue2Error> {
        let contexts = [ctx.with_new_command_queue()?, ctx.with_new_command_queue()?];
        let queue_residency = [
            model.attach_residency_to_queue(&contexts[0].queue),
            model.attach_residency_to_queue(&contexts[1].queue),
        ];
        let ids = [
            MetalTensor::zeros_i32(ctx, vec![1])?,
            MetalTensor::zeros_i32(ctx, vec![1])?,
        ];
        let selections = [
            MetalTensor::zeros_i32(ctx, vec![1])?,
            MetalTensor::zeros_i32(ctx, vec![1])?,
        ];
        Ok(Self {
            forward: MetalForward::new(ctx, model),
            _queue_residency: queue_residency,
            contexts,
            ids,
            selections,
            poisoned: false,
        })
    }

    pub fn step_greedy(
        &mut self,
        token_ids: [i32; QWEN_QUEUE2_WIDTH],
        positions: [u32; QWEN_QUEUE2_WIDTH],
        sessions: [&mut MetalSession; QWEN_QUEUE2_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<QwenQueue2Step, QwenQueue2Error> {
        if self.poisoned {
            return Err(QwenQueue2Error::Poisoned);
        }
        if cancelled() {
            return Err(QwenQueue2Error::CancelledBeforeCommit);
        }
        let [session0, session1] = sessions;
        let mut sessions = [session0, session1];
        self.validate(token_ids, positions, &sessions)?;
        self.write_ids(token_ids)?;

        let command0 = self.contexts[0].queue.commandBuffer().ok_or_else(|| {
            QwenQueue2Error::Validation("slot 0 command buffer unavailable".into())
        })?;
        let command1 = self.contexts[1].queue.commandBuffer().ok_or_else(|| {
            QwenQueue2Error::Validation("slot 1 command buffer unavailable".into())
        })?;
        let commands = [command0, command1];
        // Encoding advances host KV frontiers before the command is committed.
        // Restore them before propagating either an error or a caught unwind.
        let encoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for slot in 0..QWEN_QUEUE2_WIDTH {
                let encoder = KernelEncoder::begin(&commands[slot]);
                let result = self.forward.encode_single_token_greedy(
                    &encoder,
                    positions[slot],
                    sessions[slot],
                    &self.ids[slot],
                    &self.selections[slot],
                );
                encoder.end();
                result?;
            }
            if cancelled() {
                return Err(QwenQueue2Error::CancelledBeforeCommit);
            }
            Ok::<(), QwenQueue2Error>(())
        }));
        match encoded {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                Self::restore_frontiers(&mut sessions, positions);
                return Err(error);
            }
            Err(payload) => {
                Self::restore_frontiers(&mut sessions, positions);
                std::panic::resume_unwind(payload);
            }
        }
        for command in &commands {
            command.commit();
        }
        for command in &commands {
            command.waitUntilCompleted();
        }
        for (slot, command) in commands.iter().enumerate() {
            let status = command.status();
            let error = command.error();
            if status != MTLCommandBufferStatus::Completed || error.is_some() {
                self.poisoned = true;
                Self::poison_sessions(
                    &mut sessions,
                    "a committed independent-queue command failed",
                );
                return Err(QwenQueue2Error::CommandBuffer {
                    slot,
                    status: format!("{status:?}"),
                    error: format!("{error:?}"),
                });
            }
        }

        let mut argmax_ids = [0i32; QWEN_QUEUE2_WIDTH];
        for (slot, argmax_id) in argmax_ids.iter_mut().enumerate() {
            let raw = match self.read_selection(slot) {
                Ok(raw) => raw,
                Err(error) => {
                    self.poisoned = true;
                    Self::poison_sessions(
                        &mut sessions,
                        "independent-queue greedy readback failed",
                    );
                    return Err(error);
                }
            };
            *argmax_id = match GreedySelection::from_encoded(raw).into_token() {
                Ok(token) => token,
                Err(source) => {
                    self.poisoned = true;
                    Self::poison_sessions(
                        &mut sessions,
                        "independent-queue greedy selection failed",
                    );
                    return Err(QwenQueue2Error::GreedySelection { slot, source });
                }
            };
        }
        let gpu_ms = std::array::from_fn(|slot| {
            let start = commands[slot].GPUStartTime();
            let end = commands[slot].GPUEndTime();
            (start.is_finite() && end.is_finite() && start > 0.0 && end > start)
                .then_some((end - start) * 1e3)
        });
        Ok(QwenQueue2Step { argmax_ids, gpu_ms })
    }

    fn validate(
        &self,
        token_ids: [i32; QWEN_QUEUE2_WIDTH],
        positions: [u32; QWEN_QUEUE2_WIDTH],
        sessions: &[&mut MetalSession; QWEN_QUEUE2_WIDTH],
    ) -> Result<(), QwenQueue2Error> {
        let vocab = self.forward.model.arch.vocab_size;
        for slot in 0..QWEN_QUEUE2_WIDTH {
            let token = token_ids[slot];
            if token < 0 || token as u32 >= vocab {
                return Err(QwenQueue2Error::Validation(format!(
                    "slot {slot} token {token} is outside vocab {vocab}"
                )));
            }
            let position = positions[slot] as usize;
            let session = &*sessions[slot];
            session.ensure_usable()?;
            if session.has_internal_mutable_alias() {
                return Err(QwenQueue2Error::Validation(format!(
                    "slot {slot} aliases mutable storage internally"
                )));
            }
            if position >= session.kv_capacity {
                return Err(QwenQueue2Error::Validation(format!(
                    "slot {slot} position {position} exceeds capacity {}",
                    session.kv_capacity
                )));
            }
            for (layer, actual) in session.kv_n_pos.iter().copied().enumerate() {
                if actual != position {
                    return Err(QwenQueue2Error::Validation(format!(
                        "slot {slot} attention layer {layer} frontier {actual} != {position}"
                    )));
                }
            }
        }
        if sessions[0].aliases_mutable_session(sessions[1]) {
            return Err(QwenQueue2Error::Validation(
                "the two sessions alias mutable storage".into(),
            ));
        }
        Ok(())
    }

    fn write_ids(&self, token_ids: [i32; QWEN_QUEUE2_WIDTH]) -> Result<(), QwenQueue2Error> {
        for (slot, (tensor, token_id)) in self.ids.iter().zip(token_ids).enumerate() {
            if tensor.dtype != GgmlType::I32
                || tensor.n_elements() != 1
                || !tensor.offset.is_multiple_of(4)
            {
                return Err(QwenQueue2Error::Validation(format!(
                    "slot {slot} input scratch contract mismatch"
                )));
            }
            let offset = usize::try_from(tensor.offset / 4)
                .map_err(|_| QwenQueue2Error::Validation("input offset exceeds usize".into()))?;
            unsafe {
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add(offset)
                    .write(token_id);
            }
        }
        Ok(())
    }

    fn read_selection(&self, slot: usize) -> Result<i32, QwenQueue2Error> {
        let tensor = &self.selections[slot];
        if tensor.dtype != GgmlType::I32
            || tensor.n_elements() != 1
            || !tensor.offset.is_multiple_of(4)
        {
            return Err(QwenQueue2Error::Validation(format!(
                "slot {slot} selection scratch contract mismatch"
            )));
        }
        let offset = usize::try_from(tensor.offset / 4)
            .map_err(|_| QwenQueue2Error::Validation("selection offset exceeds usize".into()))?;
        Ok(unsafe {
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add(offset)
                .read()
        })
    }

    fn restore_frontiers(
        sessions: &mut [&mut MetalSession; QWEN_QUEUE2_WIDTH],
        positions: [u32; QWEN_QUEUE2_WIDTH],
    ) {
        for slot in 0..QWEN_QUEUE2_WIDTH {
            sessions[slot].kv_n_pos.fill(positions[slot] as usize);
        }
    }

    fn poison_sessions(
        sessions: &mut [&mut MetalSession; QWEN_QUEUE2_WIDTH],
        reason: &'static str,
    ) {
        for session in sessions {
            session.poison(reason);
        }
    }
}

pub(crate) fn qwen_queue2_session_upper_bytes(
    ctx: &MetalContext,
    model: &MetalModel,
    capacity: usize,
) -> Result<u64, QwenQueue2Error> {
    if capacity == 0 {
        return Err(QwenQueue2Error::Validation(
            "session capacity must be positive".into(),
        ));
    }
    let arch = &model.arch;
    let checked_mul = |left: u64, right: u64, label: &'static str| {
        left.checked_mul(right)
            .ok_or_else(|| QwenQueue2Error::Validation(format!("{label} overflow")))
    };
    let checked_add = |left: u64, right: u64, label: &'static str| {
        left.checked_add(right)
            .ok_or_else(|| QwenQueue2Error::Validation(format!("{label} overflow")))
    };
    let h = u64::from(arch.hidden_size);
    let f = if arch.kind == ArchKind::Moe {
        u64::from(
            arch.expert_shared_feed_forward_length
                .max(arch.expert_count)
                .max(arch.expert_feed_forward_length),
        )
    } else {
        u64::from(arch.intermediate_size)
    };
    let head_dim = u64::from(arch.attn_head_dim);
    let n_q = u64::from(arch.n_q_heads);
    let n_kv = u64::from(arch.n_kv_heads);
    let q_dim = checked_mul(n_q, head_dim, "Q dimension")?;
    let kv_dim = checked_mul(n_kv, head_dim, "KV dimension")?;
    let q_group = n_q
        .checked_div(n_kv)
        .filter(|_| n_q % n_kv == 0)
        .ok_or_else(|| {
            QwenQueue2Error::Validation("Q heads are not divisible by KV heads".into())
        })?;
    let moe_router_n = if arch.kind == ArchKind::Moe {
        u64::from(arch.expert_count.max(1))
    } else {
        1
    };
    let moe_topk_n = if arch.kind == ArchKind::Moe {
        u64::from(arch.expert_used_count.max(1).min(arch.expert_count.max(1)))
    } else {
        1
    };
    let moe_inner_n = if arch.kind == ArchKind::Moe {
        checked_mul(
            moe_topk_n,
            u64::from(arch.expert_feed_forward_length.max(1)),
            "MoE inner elements",
        )?
    } else {
        1
    };
    let moe_expert_out_n = if arch.kind == ArchKind::Moe {
        checked_mul(moe_topk_n, h, "MoE expert output elements")?
    } else {
        1
    };
    let vh = u64::from(arch.gdn_head_dim);
    let n_v = u64::from(arch.gdn_n_v_heads);
    let n_k = u64::from(arch.gdn_n_k_heads);
    let conv_heads = checked_add(checked_mul(2, n_k, "GDN K heads")?, n_v, "GDN heads")?;
    let conv_dim = checked_mul(conv_heads, vh, "GDN convolution dimension")?;
    let v_dim = checked_mul(n_v, vh, "GDN V dimension")?;
    let k_dim = checked_mul(n_k, vh, "GDN K dimension")?;
    let gdn_conv_elems = checked_mul(
        u64::from(arch.gdn_conv_kernel).saturating_sub(1),
        conv_dim,
        "GDN convolution state elements",
    )?;
    let gdn_state_elems = checked_mul(
        checked_mul(n_v, vh, "GDN state rows")?,
        vh,
        "GDN state elements",
    )?;
    let attn_q_full_elems = checked_mul(2, q_dim, "attention Q/gate elements")?;
    let attn_v4_o_partial_elems = checked_mul(
        checked_mul(
            checked_mul(n_kv, ATTN_V4_MAX_NWG as u64, "attention partial groups")?,
            q_group,
            "attention partial Q groups",
        )?,
        head_dim,
        "attention output partial elements",
    )?;
    let attn_v4_ml_partial_elems = checked_mul(
        checked_mul(
            checked_mul(n_kv, ATTN_V4_MAX_NWG as u64, "attention ML groups")?,
            q_group,
            "attention ML Q groups",
        )?,
        2,
        "attention ML partial elements",
    )?;

    let mut total = 0u64;
    let mut add_allocation = |logical_bytes: u64, count: u64| -> Result<(), QwenQueue2Error> {
        if count == 0 {
            return Ok(());
        }
        let priced = ctx.shared_buffer_size_and_align(logical_bytes)?.size;
        let repeated = checked_mul(priced, count, "priced session allocations")?;
        total = checked_add(total, repeated, "session allocation total")?;
        Ok(())
    };
    let f32_bytes = |elements: u64| checked_mul(elements, 4, "F32 allocation bytes");

    let gdn_layers = model
        .blocks
        .iter()
        .filter(|block| matches!(block, MetalBlock::Gdn(_)))
        .count() as u64;
    let attention_layers = model
        .blocks
        .iter()
        .filter(|block| matches!(block, MetalBlock::Attn(_)))
        .count() as u64;
    add_allocation(f32_bytes(gdn_conv_elems)?, gdn_layers)?;
    add_allocation(f32_bytes(gdn_state_elems)?, gdn_layers)?;

    let kv_elements = checked_mul(
        u64::try_from(capacity)
            .map_err(|_| QwenQueue2Error::Validation("capacity exceeds u64".into()))?,
        kv_dim,
        "KV elements per layer",
    )?;
    let kv_bytes = match kv_cache_dtype_for_arch(arch) {
        GgmlType::F16 => checked_mul(kv_elements, 2, "F16 KV bytes")?,
        GgmlType::Q8_0 => {
            let (block, bytes) =
                ggml_type_layout_raw(GgmlType::Q8_0 as u32).expect("Q8_0 layout is defined");
            if !kv_elements.is_multiple_of(block) {
                return Err(QwenQueue2Error::Validation(format!(
                    "Q8_0 KV elements {kv_elements} are not block aligned"
                )));
            }
            checked_mul(kv_elements / block, bytes, "Q8_0 KV bytes")?
        }
        other => {
            return Err(QwenQueue2Error::Validation(format!(
                "unsupported KV dtype {other:?}"
            )));
        }
    };
    add_allocation(
        kv_bytes,
        checked_mul(attention_layers, 2, "KV allocation count")?,
    )?;

    let fixed_f32_allocations = [
        h,
        h,
        f,
        f,
        f,
        h,
        conv_dim,
        conv_dim,
        v_dim,
        n_v,
        n_v,
        n_v,
        n_v,
        k_dim,
        k_dim,
        v_dim,
        v_dim,
        h,
        attn_q_full_elems,
        q_dim,
        q_dim,
        q_dim,
        kv_dim,
        kv_dim,
        kv_dim,
        q_dim,
        attn_v4_o_partial_elems,
        attn_v4_ml_partial_elems,
        u64::from(arch.vocab_size),
        moe_router_n,
        moe_topk_n,
        moe_topk_n,
        1,
        moe_inner_n,
        moe_expert_out_n,
    ];
    for elements in fixed_f32_allocations {
        add_allocation(f32_bytes(elements)?, 1)?;
    }
    add_allocation(4, 2)?;
    Ok(total)
}

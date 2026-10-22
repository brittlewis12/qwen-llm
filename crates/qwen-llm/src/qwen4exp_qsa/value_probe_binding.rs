use super::*;
use std::rc::Rc;

enum Arm {
    Baseline,
    Candidate(Rc<value_mma_probe::Scratch>),
}

thread_local! {
    static BINDING: std::cell::RefCell<Option<Arm>> = const { std::cell::RefCell::new(None) };
}

pub(crate) struct Scope(Option<Arm>);

impl Scope {
    pub(crate) fn baseline() -> Self {
        Self::install(Arm::Baseline)
    }

    pub(crate) fn new(scratch: &Rc<value_mma_probe::Scratch>, enabled: bool) -> Self {
        Self::install(if enabled {
            Arm::Candidate(Rc::clone(scratch))
        } else {
            Arm::Baseline
        })
    }

    fn install(arm: Arm) -> Self {
        BINDING.with(|slot| {
            assert!(slot.borrow().is_none(), "selected value probes cannot nest");
            Self(slot.replace(Some(arm)))
        })
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        BINDING.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_gate_projection: &MetalTensor,
    value_cache: &MetalTensor,
    token_ids: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    logits: &MetalTensor,
    attention: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<bool, MetalError> {
    BINDING.with(|slot| {
        let slot = slot.borrow();
        let Some(arm) = slot.as_ref() else {
            return Ok(false);
        };
        let Arm::Candidate(scratch) = arm else {
            encode_attention_softmax_value_packed_gqa4(
                ctx,
                enc,
                query_gate_projection,
                value_cache,
                token_ids,
                selected_count,
                selector_status,
                logits,
                attention,
                geometry,
                start_position,
                query_count,
            )?;
            return Ok(true);
        };
        value_mma_probe::encode(
            ctx,
            enc,
            query_gate_projection,
            value_cache,
            token_ids,
            selected_count,
            selector_status,
            logits,
            attention,
            geometry,
            start_position,
            query_count,
            scratch,
        )?;
        Ok(true)
    })
}

//! Idle residency keep-alive for no-copy weights.
//!
//! Metal wires the pages of a no-copy buffer while commands use it and
//! unwires them about two seconds after the GPU goes idle (PERF-LOG
//! 2026-10-05, placement screen); the next command then re-wires every page
//! it touches (~1 s for GLM-5.3-Flash's 109.5 GiB). A pulse is one ordinary
//! command buffer that marks the buffers used for reading and runs a single
//! one-element dispatch, submitted inside that window to keep the pages
//! wired between requests.
//!
//! It is not a residency set: nothing is requested, committed or held beyond
//! the pulse's own command, so it adds no state that outlives the process's
//! in-flight commands. A pulse never waits; the next pulse surfaces a
//! failed one.

use super::{Buffer, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_fill_f32};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLResource, MTLResourceUsage,
};

pub struct ResidencyKeepAlive {
    marker: MetalTensor,
    in_flight: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pulses: u64,
}

/// What [`ResidencyKeepAlive::pulse`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepAlivePulse {
    Submitted,
    /// The previous pulse has not completed; nothing was submitted.
    StillInFlight,
}

impl ResidencyKeepAlive {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            marker: MetalTensor::zeros_f32(ctx, vec![1])?,
            in_flight: None,
            pulses: 0,
        })
    }

    pub fn pulses(&self) -> u64 {
        self.pulses
    }

    /// Submit one pulse over `buffers` unless the previous one is still in
    /// flight. A previous pulse that failed is reported here.
    pub fn pulse(
        &mut self,
        ctx: &MetalContext,
        buffers: &[&Buffer],
    ) -> Result<KeepAlivePulse, MetalError> {
        if let Some(previous) = &self.in_flight {
            match previous.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    let error = previous
                        .error()
                        .map(|e| e.localizedDescription().to_string())
                        .unwrap_or_default();
                    self.in_flight = None;
                    return Err(MetalError::CommandBufferFailed {
                        status: "keep-alive pulse error".into(),
                        error,
                    });
                }
                _ => return Ok(KeepAlivePulse::StillInFlight),
            }
        }
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::CommandBufferFailed {
                status: "keep-alive pulse".into(),
                error: "no command buffer".into(),
            })?;
        let enc = KernelEncoder::begin(&command);
        for buffer in buffers {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***buffer);
            enc.raw.useResource_usage(resource, MTLResourceUsage::Read);
        }
        encode_fill_f32(ctx, &enc, &self.marker, 0.0)?;
        enc.end();
        command.commit();
        self.in_flight = Some(command);
        self.pulses += 1;
        Ok(KeepAlivePulse::Submitted)
    }
}

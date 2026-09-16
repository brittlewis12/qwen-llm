//! One-token Metal MoE execution for Qwen3.8-Flash-Next.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    MetalTimestampSampleBuffer, encode_axpy_rowwise_f32, encode_axpy_scalar_f32,
    encode_copy_offset_f32, encode_dot_sigmoid_f32, encode_mat_mat_f32_router_e8p32_strict,
    encode_moe_down_iq4_nl_f32, encode_moe_down_iq4_nl_f32_fast,
    encode_moe_down_iq4_nl_f32_grouped_slots, encode_moe_down_iq4_nl_f32_grouped_slots_m128_n16,
    encode_moe_down_q8_0_f32_grouped_slots, encode_moe_down_weighted_sum_q8_0_f32,
    encode_moe_route_bucket_slots_f32, encode_moe_swiglu_iq3_xxs_f32,
    encode_moe_swiglu_iq3_xxs_f32_fast, encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16,
    encode_moe_swiglu_iq4_xs_f32, encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16,
    encode_moe_weighted_sum_f32, encode_moe_weighted_sum_packed_f32, encode_shared_swiglu_q8_0_f32,
    encode_silu_mul_f32, encode_topk_logits_softmax_dot_sigmoid_packed_f32,
    encode_topk_logits_softmax_f32,
};
#[cfg(test)]
use crate::metal::{
    dispatch_census_tag_scope, encode_copy_offset_i32,
    encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range,
};
use crate::metal_forward::{
    MfError, encode_mat_mat_dispatch, encode_mat_vec_dispatch, validate_f32_q8_mat_mat_addressing,
};
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig};
use crate::qwen4exp_profile::{
    QWEN4EXP_PACKED_PROFILE_MOE_SPANS, QWEN4EXP_PACKED_PROFILE_MOE_STAGES,
    QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES, Qwen4ExpPackedProfileLabel,
    Qwen4ExpPackedProfileRecorder, Qwen4ExpPackedProfileSpan, begin_optional, end_optional,
    stage_encoder,
};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource,
};

const MAX_TOP_K: usize = 16;

#[cfg(test)]
#[path = "qwen4exp_moe_observe.rs"]
pub(crate) mod singleton_observe;
const MAX_PACKED_TOKENS: usize = 2_048;
const PACKED_ROUTER_E8P32_STRICT_DEVICE: &str = "Apple M4 Max";
const PACKED_ROUTER_E8P32_STRICT_HIDDEN: usize = 2_560;
const PACKED_ROUTER_E8P32_STRICT_EXPERTS: usize = 512;
const PACKED_ROUTER_E8P32_STRICT_N512_TOKENS: usize = 512;
const PACKED_ROUTER_E8P32_STRICT_N527_TOKENS: usize = 527;
const PACKED_ROUTER_E8P32_STRICT_FULL_CHUNK_TOKENS: usize = 2_048;
pub(crate) const PACKED_ROUTER_E8P32_STRICT_TOKEN_COUNTS: [usize; 3] = [
    PACKED_ROUTER_E8P32_STRICT_N512_TOKENS,
    PACKED_ROUTER_E8P32_STRICT_N527_TOKENS,
    PACKED_ROUTER_E8P32_STRICT_FULL_CHUNK_TOKENS,
];
const PACKED_IQ4_DOWN_M128_N16_DEVICE: &str = "Apple M4 Max";
const PACKED_IQ4_DOWN_M128_N16_HIDDEN: usize = 2_560;
const PACKED_IQ4_DOWN_M128_N16_ROUTED: usize = 640;
const PACKED_IQ4_DOWN_M128_N16_EXPERTS: usize = 512;
const PACKED_IQ4_DOWN_M128_N16_TOP_K: usize = 10;
const PACKED_IQ4_DOWN_M128_N16_N512_TOKENS: usize = 512;
const PACKED_IQ4_DOWN_M128_N16_N527_TOKENS: usize = 527;
const PACKED_IQ4_DOWN_M128_N16_TOKEN_COUNTS: [usize; 2] = [
    PACKED_IQ4_DOWN_M128_N16_N512_TOKENS,
    PACKED_IQ4_DOWN_M128_N16_N527_TOKENS,
];

#[cfg(test)]
const QWEN4EXP_IQ3_GATE_UP_CAPTURE_TOKENS: usize = 2_048;

#[cfg(test)]
#[derive(Clone)]
struct Qwen4ExpMoeRouteCountCaptureBinding {
    output: MetalTensor,
    seen_layers: std::rc::Rc<std::cell::RefCell<[bool; 48]>>,
}

#[cfg(test)]
const QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS: usize = 43;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen4ExpIq3GateUpProbeArm {
    NoWork,
    Count1To8,
    Count9To16,
    Count17To32,
    Count33To64,
    Count65Plus,
    Full,
}

#[cfg(test)]
impl Qwen4ExpIq3GateUpProbeArm {
    pub(crate) const ALL: [Self; 7] = [
        Self::NoWork,
        Self::Count1To8,
        Self::Count9To16,
        Self::Count17To32,
        Self::Count33To64,
        Self::Count65Plus,
        Self::Full,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NoWork => "no_work",
            Self::Count1To8 => "count_1_8",
            Self::Count9To16 => "count_9_16",
            Self::Count17To32 => "count_17_32",
            Self::Count33To64 => "count_33_64",
            Self::Count65Plus => "count_65_plus",
            Self::Full => "full",
        }
    }

    const fn bounds(self) -> (u32, u32) {
        match self {
            Self::NoWork => (0, 0),
            Self::Count1To8 => (1, 8),
            Self::Count9To16 => (9, 16),
            Self::Count17To32 => (17, 32),
            Self::Count33To64 => (33, 64),
            Self::Count65Plus => (65, i32::MAX as u32),
            Self::Full => (0, i32::MAX as u32),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpIq3GateUpCaptureRecord {
    pub ordinal: usize,
    pub layer: u32,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct Qwen4ExpIq3GateUpCaptureBanks {
    pub inputs: MetalTensor,
    pub counts: MetalTensor,
    pub slots: MetalTensor,
    pub tokens: usize,
}

#[cfg(test)]
impl Qwen4ExpIq3GateUpCaptureBanks {
    pub(crate) fn input_view(&self, ordinal: usize) -> MetalTensor {
        self.inputs.view_subrange(
            (ordinal * PACKED_ROUTER_E8P32_STRICT_HIDDEN * self.tokens) as u64,
            vec![PACKED_ROUTER_E8P32_STRICT_HIDDEN as u64, self.tokens as u64],
        )
    }

    pub(crate) fn counts_view(&self, ordinal: usize) -> MetalTensor {
        self.counts.view_subrange(
            (ordinal * PACKED_ROUTER_E8P32_STRICT_EXPERTS) as u64,
            vec![PACKED_ROUTER_E8P32_STRICT_EXPERTS as u64],
        )
    }

    pub(crate) fn slots_view(&self, ordinal: usize) -> MetalTensor {
        self.slots.view_subrange(
            (ordinal * PACKED_ROUTER_E8P32_STRICT_EXPERTS * self.tokens) as u64,
            vec![
                self.tokens as u64,
                PACKED_ROUTER_E8P32_STRICT_EXPERTS as u64,
            ],
        )
    }
}

#[cfg(test)]
#[derive(Clone)]
struct Qwen4ExpIq3GateUpCaptureBinding {
    banks: Qwen4ExpIq3GateUpCaptureBanks,
    next_ordinal: std::rc::Rc<std::cell::Cell<usize>>,
    records: std::rc::Rc<std::cell::RefCell<Vec<Qwen4ExpIq3GateUpCaptureRecord>>>,
    seen_layers: std::rc::Rc<std::cell::RefCell<[bool; 48]>>,
}

crate::env_flag!(
    default_on configured_qwen4exp_moe_iq3_fast_enabled,
    "QWEN4EXP_MOE_IQ3_FAST"
);
crate::env_flag!(
    default_on qwen4exp_moe_iq4_down_fast_enabled,
    "QWEN4EXP_MOE_IQ4_DOWN_FAST"
);
crate::env_flag!(
    default_on configured_qwen4exp_packed_router_e8p32_strict_enabled,
    "QWEN4EXP_PACKED_ROUTER_E8P32_STRICT"
);
crate::env_flag!(
    default_on configured_qwen4exp_moe_iq4_down_m128_n16_enabled,
    "QWEN4EXP_MOE_IQ4_DOWN_M128_N16"
);

#[cfg(test)]
thread_local! {
    static QWEN4EXP_MOE_IQ3_FAST_OVERRIDE: std::cell::Cell<Option<bool>> = const {
        std::cell::Cell::new(None)
    };
    static QWEN4EXP_PACKED_ROUTER_E8P32_STRICT_OVERRIDE: std::cell::Cell<Option<bool>> = const {
        std::cell::Cell::new(None)
    };
    static QWEN4EXP_MOE_IQ4_DOWN_M128_N16_OVERRIDE: std::cell::Cell<Option<bool>> = const {
        std::cell::Cell::new(None)
    };
    static QWEN4EXP_MOE_ROUTE_COUNT_CAPTURE: std::cell::RefCell<Option<Qwen4ExpMoeRouteCountCaptureBinding>> = const {
        std::cell::RefCell::new(None)
    };
    static QWEN4EXP_IQ3_GATE_UP_CAPTURE: std::cell::RefCell<Option<Qwen4ExpIq3GateUpCaptureBinding>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn with_qwen4exp_moe_iq3_fast_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct RestoreOverride(Option<bool>);

    impl Drop for RestoreOverride {
        fn drop(&mut self) {
            QWEN4EXP_MOE_IQ3_FAST_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = QWEN4EXP_MOE_IQ3_FAST_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let _restore = RestoreOverride(previous);
    f()
}

fn qwen4exp_moe_iq3_fast_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = QWEN4EXP_MOE_IQ3_FAST_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    configured_qwen4exp_moe_iq3_fast_enabled()
}

#[cfg(test)]
fn with_qwen4exp_moe_iq4_down_m128_n16_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct RestoreOverride(Option<bool>);

    impl Drop for RestoreOverride {
        fn drop(&mut self) {
            QWEN4EXP_MOE_IQ4_DOWN_M128_N16_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = QWEN4EXP_MOE_IQ4_DOWN_M128_N16_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let _restore = RestoreOverride(previous);
    f()
}

fn qwen4exp_moe_iq4_down_m128_n16_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = QWEN4EXP_MOE_IQ4_DOWN_M128_N16_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    configured_qwen4exp_moe_iq4_down_m128_n16_enabled()
}

#[cfg(test)]
pub(crate) fn with_qwen4exp_packed_router_e8p32_strict_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> R {
    struct RestoreOverride(Option<bool>);

    impl Drop for RestoreOverride {
        fn drop(&mut self) {
            QWEN4EXP_PACKED_ROUTER_E8P32_STRICT_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = QWEN4EXP_PACKED_ROUTER_E8P32_STRICT_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let _restore = RestoreOverride(previous);
    f()
}

fn qwen4exp_packed_router_e8p32_strict_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = QWEN4EXP_PACKED_ROUTER_E8P32_STRICT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    configured_qwen4exp_packed_router_e8p32_strict_enabled()
}

#[cfg(test)]
pub(crate) fn with_qwen4exp_moe_route_count_capture<R>(
    output: &MetalTensor,
    f: impl FnOnce() -> R,
) -> (R, usize) {
    const LAYERS: usize = 48;
    const EXPERTS: usize = 512;

    assert_eq!(output.dtype, GgmlType::I32);
    assert_eq!(output.shape, [EXPERTS as u64, LAYERS as u64]);
    assert!(output.is_writable());
    assert!(!qwen4exp_iq3_gate_up_capture_active());

    struct RestoreCapture(Option<Qwen4ExpMoeRouteCountCaptureBinding>);

    impl Drop for RestoreCapture {
        fn drop(&mut self) {
            QWEN4EXP_MOE_ROUTE_COUNT_CAPTURE.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }

    let seen_layers = std::rc::Rc::new(std::cell::RefCell::new([false; LAYERS]));
    let previous = QWEN4EXP_MOE_ROUTE_COUNT_CAPTURE.with(|slot| {
        slot.borrow_mut()
            .replace(Qwen4ExpMoeRouteCountCaptureBinding {
                output: output.clone(),
                seen_layers: seen_layers.clone(),
            })
    });
    let _restore = RestoreCapture(previous);
    let result = f();
    let captured_layers = seen_layers
        .borrow()
        .iter()
        .filter(|&&captured| captured)
        .count();
    (result, captured_layers)
}

#[cfg(test)]
pub(crate) fn qwen4exp_moe_route_count_capture_active() -> bool {
    QWEN4EXP_MOE_ROUTE_COUNT_CAPTURE.with(|slot| slot.borrow().is_some())
}

#[cfg(test)]
pub(crate) fn with_qwen4exp_iq3_gate_up_capture<R>(
    banks: &Qwen4ExpIq3GateUpCaptureBanks,
    f: impl FnOnce() -> R,
) -> (R, Vec<Qwen4ExpIq3GateUpCaptureRecord>) {
    assert_eq!(banks.tokens, QWEN4EXP_IQ3_GATE_UP_CAPTURE_TOKENS);
    assert_eq!(banks.inputs.dtype, GgmlType::F32);
    assert_eq!(banks.counts.dtype, GgmlType::I32);
    assert_eq!(banks.slots.dtype, GgmlType::I32);
    assert_eq!(
        banks.inputs.shape,
        [
            PACKED_ROUTER_E8P32_STRICT_HIDDEN as u64,
            banks.tokens as u64,
            QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS as u64,
        ]
    );
    assert_eq!(
        banks.counts.shape,
        [
            PACKED_ROUTER_E8P32_STRICT_EXPERTS as u64,
            QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS as u64,
        ]
    );
    assert_eq!(
        banks.slots.shape,
        [
            banks.tokens as u64,
            PACKED_ROUTER_E8P32_STRICT_EXPERTS as u64,
            QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS as u64,
        ]
    );
    assert!(banks.inputs.is_writable());
    assert!(banks.counts.is_writable());
    assert!(banks.slots.is_writable());
    assert!(!qwen4exp_moe_route_count_capture_active());
    QWEN4EXP_IQ3_GATE_UP_CAPTURE.with(|slot| {
        assert!(slot.borrow().is_none(), "IQ3 gate/up capture cannot nest");
    });

    struct RestoreCapture;

    impl Drop for RestoreCapture {
        fn drop(&mut self) {
            QWEN4EXP_IQ3_GATE_UP_CAPTURE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }

    let next_ordinal = std::rc::Rc::new(std::cell::Cell::new(0));
    let records = std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(
        QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS,
    )));
    let seen_layers = std::rc::Rc::new(std::cell::RefCell::new([false; 48]));
    QWEN4EXP_IQ3_GATE_UP_CAPTURE.with(|slot| {
        *slot.borrow_mut() = Some(Qwen4ExpIq3GateUpCaptureBinding {
            banks: banks.clone(),
            next_ordinal,
            records: records.clone(),
            seen_layers,
        });
    });
    let _restore = RestoreCapture;
    let result = f();
    let records = records.borrow().clone();
    (result, records)
}

#[cfg(test)]
pub(crate) fn qwen4exp_iq3_gate_up_capture_active() -> bool {
    QWEN4EXP_IQ3_GATE_UP_CAPTURE.with(|slot| slot.borrow().is_some())
}

#[cfg(test)]
const fn qwen4exp_iq3_gate_up_probe_layer(layer: u32) -> bool {
    layer < 48 && !matches!(layer, 2 | 4 | 30 | 46 | 47)
}

#[cfg(test)]
fn encode_qwen4exp_moe_route_count_capture(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    route_counts: &MetalTensor,
    expert_count: usize,
    layer: u32,
) -> Result<(), Qwen4ExpMoeError> {
    const LAYERS: usize = 48;
    const EXPERTS: usize = 512;

    QWEN4EXP_MOE_ROUTE_COUNT_CAPTURE.with(|slot| {
        let capture = slot.borrow();
        let Some(capture) = capture.as_ref() else {
            return Ok(());
        };
        if expert_count != EXPERTS {
            return invalid(format!(
                "route-count capture requires {EXPERTS} experts, got {expert_count}"
            ));
        }
        let layer = usize::try_from(layer).map_err(|_| {
            Qwen4ExpMoeError::Invalid("route-count capture layer exceeds usize".into())
        })?;
        if layer >= LAYERS {
            return invalid(format!(
                "route-count capture layer {layer} exceeds {LAYERS}"
            ));
        }
        if capture.seen_layers.borrow()[layer] {
            return invalid(format!(
                "route-count capture received duplicate layer {layer}"
            ));
        }
        let destination = capture
            .output
            .view_subrange((layer * EXPERTS) as u64, vec![EXPERTS as u64]);
        encode_copy_offset_i32(ctx, enc, route_counts, 0, &destination, EXPERTS)?;
        capture.seen_layers.borrow_mut()[layer] = true;
        Ok(())
    })
}

fn packed_router_e8p32_strict_scope_qualified(
    device_name: &str,
    hidden_size: usize,
    expert_count: usize,
    router_dtype: GgmlType,
    tokens: usize,
) -> bool {
    device_name == PACKED_ROUTER_E8P32_STRICT_DEVICE
        && hidden_size == PACKED_ROUTER_E8P32_STRICT_HIDDEN
        && expert_count == PACKED_ROUTER_E8P32_STRICT_EXPERTS
        && router_dtype == GgmlType::F32
        && PACKED_ROUTER_E8P32_STRICT_TOKEN_COUNTS.contains(&tokens)
}

fn packed_router_e8p32_strict_qualified(
    ctx: &MetalContext,
    geometry: Qwen4ExpMoeMetalGeometry,
    router_dtype: GgmlType,
    tokens: usize,
) -> bool {
    qwen4exp_packed_router_e8p32_strict_enabled()
        && packed_router_e8p32_strict_scope_qualified(
            &ctx.device.name().to_string(),
            geometry.hidden_size,
            geometry.expert_count,
            router_dtype,
            tokens,
        )
}

fn packed_iq4_down_m128_n16_scope_qualified(
    device_name: &str,
    geometry: Qwen4ExpMoeMetalGeometry,
    dtype: GgmlType,
    tokens: usize,
) -> bool {
    device_name == PACKED_IQ4_DOWN_M128_N16_DEVICE
        && geometry.hidden_size == PACKED_IQ4_DOWN_M128_N16_HIDDEN
        && geometry.routed_intermediate_size == PACKED_IQ4_DOWN_M128_N16_ROUTED
        && geometry.expert_count == PACKED_IQ4_DOWN_M128_N16_EXPERTS
        && geometry.experts_per_token == PACKED_IQ4_DOWN_M128_N16_TOP_K
        && dtype == GgmlType::IQ4_NL
        && PACKED_IQ4_DOWN_M128_N16_TOKEN_COUNTS.contains(&tokens)
}

fn packed_iq4_down_m128_n16_qualified(
    ctx: &MetalContext,
    geometry: Qwen4ExpMoeMetalGeometry,
    dtype: GgmlType,
    tokens: usize,
) -> bool {
    qwen4exp_moe_iq4_down_m128_n16_enabled()
        && packed_iq4_down_m128_n16_scope_qualified(
            &ctx.device.name().to_string(),
            geometry,
            dtype,
            tokens,
        )
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpMoeError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next MoE contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next MoE command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpMoeMetalGeometry {
    hidden_size: usize,
    expert_count: usize,
    experts_per_token: usize,
    routed_intermediate_size: usize,
    shared_intermediate_size: usize,
}

impl Qwen4ExpMoeMetalGeometry {
    pub fn new(
        hidden_size: usize,
        expert_count: usize,
        experts_per_token: usize,
        routed_intermediate_size: usize,
        shared_intermediate_size: usize,
    ) -> Result<Self, Qwen4ExpMoeError> {
        let geometry = Self {
            hidden_size,
            expert_count,
            experts_per_token,
            routed_intermediate_size,
            shared_intermediate_size,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(config: &Qwen4ExpConfig) -> Result<Self, Qwen4ExpMoeError> {
        Self::new(
            config.hidden_size as usize,
            config.moe.expert_count as usize,
            config.moe.experts_per_token as usize,
            config.moe.expert_intermediate_size as usize,
            config.moe.shared_expert_intermediate_size as usize,
        )
    }

    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub fn expert_count(self) -> usize {
        self.expert_count
    }

    pub fn experts_per_token(self) -> usize {
        self.experts_per_token
    }

    pub fn routed_intermediate_size(self) -> usize {
        self.routed_intermediate_size
    }

    pub fn shared_intermediate_size(self) -> usize {
        self.shared_intermediate_size
    }

    fn validate(self) -> Result<(), Qwen4ExpMoeError> {
        if self.hidden_size == 0
            || self.expert_count == 0
            || self.experts_per_token == 0
            || self.routed_intermediate_size == 0
            || self.shared_intermediate_size == 0
        {
            return invalid("MoE dimensions and top-k must be nonzero");
        }
        if self.experts_per_token > self.expert_count || self.experts_per_token > MAX_TOP_K {
            return invalid(format!(
                "MoE top-k {} must be no larger than expert count {} or {MAX_TOP_K}",
                self.experts_per_token, self.expert_count
            ));
        }
        if self.expert_count > i32::MAX as usize {
            return invalid("MoE expert IDs must fit signed i32");
        }
        if !self.hidden_size.is_multiple_of(256) {
            return invalid(format!(
                "MoE hidden size {} must be divisible by 256 for released routed weights",
                self.hidden_size
            ));
        }
        if !self.routed_intermediate_size.is_multiple_of(32) {
            return invalid(format!(
                "MoE routed intermediate size {} must be divisible by 32",
                self.routed_intermediate_size
            ));
        }
        if !self.shared_intermediate_size.is_multiple_of(32) {
            return invalid(format!(
                "MoE shared intermediate size {} must be divisible by 32",
                self.shared_intermediate_size
            ));
        }

        for (name, value) in [
            ("hidden size", self.hidden_size),
            ("expert count", self.expert_count),
            ("top-k", self.experts_per_token),
            ("routed intermediate size", self.routed_intermediate_size),
            ("shared intermediate size", self.shared_intermediate_size),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("MoE {name} {value} exceeds u32"));
            }
        }

        let products = [
            ("router", &[self.hidden_size, self.expert_count][..]),
            (
                "routed expert bank",
                &[
                    self.hidden_size,
                    self.routed_intermediate_size,
                    self.expert_count,
                ][..],
            ),
            (
                "routed down bank",
                &[
                    self.routed_intermediate_size,
                    self.hidden_size,
                    self.expert_count,
                ][..],
            ),
            (
                "routed inner scratch",
                &[self.experts_per_token, self.routed_intermediate_size][..],
            ),
            (
                "routed output scratch",
                &[self.experts_per_token, self.hidden_size][..],
            ),
            (
                "shared gate/up",
                &[self.hidden_size, self.shared_intermediate_size][..],
            ),
            (
                "shared down",
                &[self.shared_intermediate_size, self.hidden_size][..],
            ),
        ];
        for (name, factors) in products {
            let elements = checked_product(factors, name)?;
            if u32::try_from(elements).is_err() {
                return invalid(format!("MoE {name} offsets exceed u32"));
            }
            elements.checked_mul(4).ok_or_else(|| {
                Qwen4ExpMoeError::Invalid(format!("MoE {name} byte count overflow"))
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpMoeMetalWeights<'a> {
    pub geometry: Qwen4ExpMoeMetalGeometry,
    pub router: &'a MetalTensor,
    pub routed_gate: &'a MetalTensor,
    pub routed_up: &'a MetalTensor,
    pub routed_down: &'a MetalTensor,
    pub shared_router: &'a MetalTensor,
    pub shared_gate: &'a MetalTensor,
    pub shared_up: &'a MetalTensor,
    pub shared_down: &'a MetalTensor,
}

impl<'a> Qwen4ExpMoeMetalWeights<'a> {
    pub fn bind(weights: &'a Qwen4ExpMetalWeights, layer: u32) -> Result<Self, Qwen4ExpMoeError> {
        if layer >= weights.config().layer_count {
            return invalid(format!(
                "MoE layer {layer} is outside {} layers",
                weights.config().layer_count
            ));
        }
        let prefix = format!("blk.{layer}");
        Ok(Self {
            geometry: Qwen4ExpMoeMetalGeometry::from_config(weights.config())?,
            router: weights.require_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?,
            routed_gate: weights.require_tensor(&format!("{prefix}.ffn_gate_exps.weight"))?,
            routed_up: weights.require_tensor(&format!("{prefix}.ffn_up_exps.weight"))?,
            routed_down: weights.require_tensor(&format!("{prefix}.ffn_down_exps.weight"))?,
            shared_router: weights
                .require_tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"))?,
            shared_gate: weights.require_tensor(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            shared_up: weights.require_tensor(&format!("{prefix}.ffn_up_shexp.weight"))?,
            shared_down: weights.require_tensor(&format!("{prefix}.ffn_down_shexp.weight"))?,
        })
    }
}

pub struct Qwen4ExpMoeMetalWorkspace {
    geometry: Qwen4ExpMoeMetalGeometry,
    router_logits: MetalTensor,
    topk_ids: MetalTensor,
    topk_weights: MetalTensor,
    shared_gate: MetalTensor,
    routed_inner: MetalTensor,
    routed_expert_output: MetalTensor,
    shared_inner: MetalTensor,
    shared_output: MetalTensor,
    output: MetalTensor,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
}

pub(crate) struct Qwen4ExpMoePackedMotorScratch {
    geometry: Qwen4ExpMoeMetalGeometry,
    capacity: usize,
    router_logits: MetalTensor,
    topk_ids: MetalTensor,
    topk_weights: MetalTensor,
    shared_scale: MetalTensor,
    route_counts: MetalTensor,
    route_slots: MetalTensor,
    routed_inner: MetalTensor,
    routed_expert_output: MetalTensor,
    shared_gate_projection: MetalTensor,
    shared_up_projection: MetalTensor,
    shared_inner: MetalTensor,
    shared_output: MetalTensor,
    output: MetalTensor,
}

struct Qwen4ExpMoePackedViews {
    router_logits: MetalTensor,
    topk_ids: MetalTensor,
    topk_weights: MetalTensor,
    shared_scale: MetalTensor,
    route_counts: MetalTensor,
    route_slots: MetalTensor,
    routed_inner: MetalTensor,
    routed_expert_output: MetalTensor,
    shared_gate_projection: MetalTensor,
    shared_up_projection: MetalTensor,
    shared_inner: MetalTensor,
    shared_output: MetalTensor,
    output: MetalTensor,
}

impl Qwen4ExpMoePackedMotorScratch {
    pub(crate) fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpMoeMetalGeometry,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpMoeError> {
        geometry.validate()?;
        if capacity == 0 || capacity > MAX_PACKED_TOKENS {
            return invalid(format!(
                "packed MoE capacity must be in 1..={MAX_PACKED_TOKENS}, got {capacity}"
            ));
        }
        let _required_bytes = Self::required_bytes(geometry, capacity)?;
        let max_buffer_length = ctx.device.maxBufferLength();
        let checked_elements = |name: &str, factors: &[usize]| {
            let elements = checked_product(factors, name)?;
            if u32::try_from(elements).is_err() {
                return invalid(format!(
                    "MoE packed {name} has {elements} elements, exceeding u32"
                ));
            }
            let bytes = elements.checked_mul(size_of::<u32>()).ok_or_else(|| {
                Qwen4ExpMoeError::Invalid(format!("MoE packed {name} byte count overflow"))
            })?;
            if bytes > max_buffer_length {
                return invalid(format!(
                    "MoE packed {name} requires {bytes} bytes, exceeding Metal maxBufferLength {max_buffer_length}"
                ));
            }
            Ok(elements)
        };
        for (name, factors) in [
            ("router logits", vec![geometry.expert_count, capacity]),
            ("top-k IDs", vec![geometry.experts_per_token, capacity]),
            ("top-k weights", vec![geometry.experts_per_token, capacity]),
            ("shared scale", vec![capacity]),
            ("route counts", vec![geometry.expert_count]),
            ("route slots", vec![geometry.expert_count, capacity]),
            (
                "routed inner",
                vec![
                    geometry.routed_intermediate_size,
                    geometry.experts_per_token,
                    capacity,
                ],
            ),
            (
                "routed expert output",
                vec![geometry.hidden_size, geometry.experts_per_token, capacity],
            ),
            (
                "shared gate projection",
                vec![geometry.shared_intermediate_size, capacity],
            ),
            (
                "shared up projection",
                vec![geometry.shared_intermediate_size, capacity],
            ),
            (
                "shared inner",
                vec![geometry.shared_intermediate_size, capacity],
            ),
            ("shared output", vec![geometry.hidden_size, capacity]),
            ("output", vec![geometry.hidden_size, capacity]),
        ] {
            checked_elements(name, &factors)?;
        }

        Ok(Self {
            geometry,
            capacity,
            router_logits: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.expert_count as u64, capacity as u64],
            )?,
            topk_ids: MetalTensor::zeros_i32(
                ctx,
                vec![geometry.experts_per_token as u64, capacity as u64],
            )?,
            topk_weights: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.experts_per_token as u64, capacity as u64],
            )?,
            shared_scale: MetalTensor::zeros_f32(ctx, vec![capacity as u64])?,
            route_counts: MetalTensor::zeros_i32(ctx, vec![geometry.expert_count as u64])?,
            route_slots: MetalTensor::zeros_i32(
                ctx,
                vec![(geometry.expert_count * capacity) as u64],
            )?,
            routed_inner: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.routed_intermediate_size as u64,
                    geometry.experts_per_token as u64,
                    capacity as u64,
                ],
            )?,
            routed_expert_output: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.hidden_size as u64,
                    geometry.experts_per_token as u64,
                    capacity as u64,
                ],
            )?,
            shared_gate_projection: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.shared_intermediate_size as u64, capacity as u64],
            )?,
            shared_up_projection: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.shared_intermediate_size as u64, capacity as u64],
            )?,
            shared_inner: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.shared_intermediate_size as u64, capacity as u64],
            )?,
            shared_output: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.hidden_size as u64, capacity as u64],
            )?,
            output: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.hidden_size as u64, capacity as u64],
            )?,
        })
    }

    pub(crate) fn required_bytes(
        geometry: Qwen4ExpMoeMetalGeometry,
        capacity: usize,
    ) -> Result<usize, Qwen4ExpMoeError> {
        geometry.validate()?;
        if capacity == 0 || capacity > MAX_PACKED_TOKENS {
            return invalid(format!(
                "packed MoE capacity must be in 1..={MAX_PACKED_TOKENS}, got {capacity}"
            ));
        }
        let element_counts = [
            checked_product(&[geometry.expert_count, capacity], "packed router logits")?,
            checked_product(&[geometry.experts_per_token, capacity], "packed top-k IDs")?,
            checked_product(
                &[geometry.experts_per_token, capacity],
                "packed top-k weights",
            )?,
            capacity,
            geometry.expert_count,
            checked_product(&[geometry.expert_count, capacity], "packed route slots")?,
            checked_product(
                &[
                    geometry.routed_intermediate_size,
                    geometry.experts_per_token,
                    capacity,
                ],
                "packed routed inner",
            )?,
            checked_product(
                &[geometry.hidden_size, geometry.experts_per_token, capacity],
                "packed routed expert output",
            )?,
            checked_product(
                &[geometry.shared_intermediate_size, capacity],
                "packed shared gate projection",
            )?,
            checked_product(
                &[geometry.shared_intermediate_size, capacity],
                "packed shared up projection",
            )?,
            checked_product(
                &[geometry.shared_intermediate_size, capacity],
                "packed shared inner",
            )?,
            checked_product(&[geometry.hidden_size, capacity], "packed shared output")?,
            checked_product(&[geometry.hidden_size, capacity], "packed output")?,
        ];
        let elements = element_counts.into_iter().try_fold(0_usize, |sum, count| {
            sum.checked_add(count).ok_or_else(|| {
                Qwen4ExpMoeError::Invalid("packed MoE total element count overflow".into())
            })
        })?;
        elements
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Qwen4ExpMoeError::Invalid("packed MoE total byte count overflow".into()))
    }

    fn prefix_view(
        &self,
        name: &str,
        tensor: &MetalTensor,
        elements_per_token: usize,
        tokens: usize,
        shape: Vec<u64>,
    ) -> Result<MetalTensor, Qwen4ExpMoeError> {
        if tokens == 0 || tokens > self.capacity {
            return invalid(format!(
                "{name} token count {tokens} is outside capacity {}",
                self.capacity
            ));
        }
        let elements = elements_per_token
            .checked_mul(tokens)
            .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("{name} element count overflow")))?;
        if shape.iter().product::<u64>() != elements as u64 {
            return invalid(format!("{name} prefix shape {shape:?} has the wrong size"));
        }
        let view = tensor.view_subrange(0, shape);
        if view.n_elements() as usize != elements {
            return invalid(format!("{name} prefix view has the wrong element count"));
        }
        Ok(view)
    }

    fn views(&self, tokens: usize) -> Result<Qwen4ExpMoePackedViews, Qwen4ExpMoeError> {
        let g = self.geometry;
        Ok(Qwen4ExpMoePackedViews {
            router_logits: self.prefix_view(
                "packed MoE router logits",
                &self.router_logits,
                g.expert_count,
                tokens,
                vec![g.expert_count as u64, tokens as u64],
            )?,
            topk_ids: self.prefix_view(
                "packed MoE top-k IDs",
                &self.topk_ids,
                g.experts_per_token,
                tokens,
                vec![g.experts_per_token as u64, tokens as u64],
            )?,
            topk_weights: self.prefix_view(
                "packed MoE top-k weights",
                &self.topk_weights,
                g.experts_per_token,
                tokens,
                vec![g.experts_per_token as u64, tokens as u64],
            )?,
            shared_scale: self.prefix_view(
                "packed MoE shared scale",
                &self.shared_scale,
                1,
                tokens,
                vec![tokens as u64],
            )?,
            route_counts: self
                .route_counts
                .view_subrange(0, vec![g.expert_count as u64]),
            route_slots: self.prefix_view(
                "packed MoE route slots",
                &self.route_slots,
                g.expert_count,
                tokens,
                vec![tokens as u64, g.expert_count as u64],
            )?,
            routed_inner: self.prefix_view(
                "packed MoE routed inner",
                &self.routed_inner,
                g.experts_per_token * g.routed_intermediate_size,
                tokens,
                vec![
                    g.routed_intermediate_size as u64,
                    g.experts_per_token as u64,
                    tokens as u64,
                ],
            )?,
            routed_expert_output: self.prefix_view(
                "packed MoE routed expert output",
                &self.routed_expert_output,
                g.experts_per_token * g.hidden_size,
                tokens,
                vec![
                    g.hidden_size as u64,
                    g.experts_per_token as u64,
                    tokens as u64,
                ],
            )?,
            shared_gate_projection: self.prefix_view(
                "packed MoE shared gate projection",
                &self.shared_gate_projection,
                g.shared_intermediate_size,
                tokens,
                vec![g.shared_intermediate_size as u64, tokens as u64],
            )?,
            shared_up_projection: self.prefix_view(
                "packed MoE shared up projection",
                &self.shared_up_projection,
                g.shared_intermediate_size,
                tokens,
                vec![g.shared_intermediate_size as u64, tokens as u64],
            )?,
            shared_inner: self.prefix_view(
                "packed MoE shared inner",
                &self.shared_inner,
                g.shared_intermediate_size,
                tokens,
                vec![g.shared_intermediate_size as u64, tokens as u64],
            )?,
            shared_output: self.prefix_view(
                "packed MoE shared output",
                &self.shared_output,
                g.hidden_size,
                tokens,
                vec![g.hidden_size as u64, tokens as u64],
            )?,
            output: self.prefix_view(
                "packed MoE output",
                &self.output,
                g.hidden_size,
                tokens,
                vec![g.hidden_size as u64, tokens as u64],
            )?,
        })
    }
}

impl Qwen4ExpMoeMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpMoeMetalGeometry,
    ) -> Result<Self, Qwen4ExpMoeError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            router_logits: MetalTensor::zeros_f32(ctx, vec![geometry.expert_count as u64])?,
            topk_ids: MetalTensor::zeros_i32(ctx, vec![geometry.experts_per_token as u64])?,
            topk_weights: MetalTensor::zeros_f32(ctx, vec![geometry.experts_per_token as u64])?,
            shared_gate: MetalTensor::zeros_f32(ctx, vec![1])?,
            routed_inner: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.routed_intermediate_size as u64,
                    geometry.experts_per_token as u64,
                ],
            )?,
            routed_expert_output: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.hidden_size as u64,
                    geometry.experts_per_token as u64,
                ],
            )?,
            shared_inner: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.shared_intermediate_size as u64],
            )?,
            shared_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            active_command: None,
            state_poisoned: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpMoeMetalGeometry {
        self.geometry
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpMoeError> {
        self.require_idle()?;
        self.state_poisoned = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpMoeError> {
        let Some(command) = self.active_command.clone() else {
            return Ok(());
        };
        let status = command.status();
        if matches!(
            status,
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return invalid(format!(
                "workspace owner is not committed (status {status:?}); commit it or abandon the uncommitted command"
            ));
        }
        command.waitUntilCompleted();
        let status = command.status();
        let error = command.error().map(|error| error.to_string());
        self.active_command = None;
        if status == MTLCommandBufferStatus::Completed && error.is_none() {
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpMoeError::CommandBuffer(format!(
                "status={status:?}, error={error:?}"
            )))
        }
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command buffer. Committing it later may race a subsequent owner.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpMoeError> {
        let Some(command) = self.active_command.as_ref() else {
            return Ok(());
        };
        let status = command.status();
        if status != MTLCommandBufferStatus::NotEnqueued {
            return invalid(format!(
                "only a NotEnqueued workspace owner can be abandoned, got {status:?}"
            ));
        }
        self.active_command = None;
        self.state_poisoned = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpMoeError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "consume or copy the MoE output in its owning command, then release the workspace"]
pub struct Qwen4ExpMoeMetalRead<'a> {
    workspace: &'a mut Qwen4ExpMoeMetalWorkspace,
}

pub struct Qwen4ExpMoeMetalOutput<'a> {
    workspace: &'a Qwen4ExpMoeMetalWorkspace,
}

impl Qwen4ExpMoeMetalRead<'_> {
    pub fn output(&self) -> Qwen4ExpMoeMetalOutput<'_> {
        Qwen4ExpMoeMetalOutput {
            workspace: self.workspace,
        }
    }
}

impl Qwen4ExpMoeMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hidden_size as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpMoeError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("MoE output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("MoE output must be copied by its owning command buffer");
        }
        require_tensor(
            "MoE copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hidden_size as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("MoE output", &self.workspace.output),
                ("MoE copied output destination", destination),
            ],
        )?;
        let mut tensors = workspace_tensors(self.workspace);
        tensors.push(("MoE copied output destination", destination));
        require_disjoint(&tensors)?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(
            ctx,
            enc,
            &self.workspace.output,
            0,
            destination,
            self.workspace.geometry.hidden_size,
        )?;
        Ok(())
    }
}

pub fn encode_qwen4exp_moe<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpMoeMetalWorkspace,
) -> Result<Qwen4ExpMoeMetalRead<'a>, Qwen4ExpMoeError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if weights.geometry != workspace.geometry {
        return invalid("MoE weight and workspace geometry differ");
    }
    validate_contract(ctx, input, weights, workspace)?;
    preflight(ctx, weights)?;
    reserve_command(workspace, enc)?;

    if let Err(error) = encode_step(ctx, enc, input, weights, workspace) {
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpMoeMetalRead { workspace })
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    workspace: &Qwen4ExpMoeMetalWorkspace,
) -> Result<(), Qwen4ExpMoeError> {
    encode_singleton_step(
        ctx,
        enc,
        input,
        weights,
        Qwen4ExpMoeSingletonBuffers {
            router_logits: &workspace.router_logits,
            topk_ids: &workspace.topk_ids,
            topk_weights: &workspace.topk_weights,
            shared_scale: &workspace.shared_gate,
            routed_inner: &workspace.routed_inner,
            routed_expert_output: &workspace.routed_expert_output,
            shared_inner: &workspace.shared_inner,
            shared_output: &workspace.shared_output,
            output: &workspace.output,
        },
    )
}

#[derive(Clone, Copy)]
struct Qwen4ExpMoeSingletonBuffers<'a> {
    router_logits: &'a MetalTensor,
    topk_ids: &'a MetalTensor,
    topk_weights: &'a MetalTensor,
    shared_scale: &'a MetalTensor,
    routed_inner: &'a MetalTensor,
    routed_expert_output: &'a MetalTensor,
    shared_inner: &'a MetalTensor,
    shared_output: &'a MetalTensor,
    output: &'a MetalTensor,
}

fn encode_singleton_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    #[cfg(test)]
    let capture = singleton_observe::before(ctx, enc, input, weights)?;
    #[cfg(test)]
    let tag = capture.and_then(|layer| dispatch_census_tag_scope(|| format!("moe.native.{layer}")));
    encode_singleton_router(ctx, enc, input, weights, buffers)?;
    encode_singleton_gate_up(ctx, enc, input, weights, buffers)?;
    encode_singleton_down(ctx, enc, weights, buffers)?;
    encode_singleton_shared_gate_up(ctx, enc, input, weights, buffers)?;
    encode_singleton_shared_down(ctx, enc, weights, buffers)?;
    encode_singleton_accumulate(ctx, enc, buffers)?;
    #[cfg(test)]
    {
        drop(tag);
        singleton_observe::after(ctx, enc, capture, buffers)?;
    }
    Ok(())
}

fn encode_singleton_router(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    let g = weights.geometry;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.router,
        input,
        buffers.router_logits,
        g.hidden_size,
        g.expert_count,
    )?;
    #[cfg(test)]
    let probed = singleton_observe::topk_native::encode_if_active(ctx, enc, buffers, g);
    #[cfg(not(test))]
    let probed = false;
    if !probed {
        encode_topk_logits_softmax_f32(
            ctx,
            enc,
            buffers.router_logits,
            buffers.topk_ids,
            buffers.topk_weights,
            g.expert_count,
            g.experts_per_token,
        )?;
    }
    encode_dot_sigmoid_f32(
        ctx,
        enc,
        weights.shared_router,
        input,
        buffers.shared_scale,
        g.hidden_size,
    )?;
    Ok(())
}

fn encode_singleton_gate_up(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    let g = weights.geometry;
    match weights.routed_gate.dtype {
        GgmlType::IQ3_XXS => {
            let encode = if qwen4exp_moe_iq3_fast_enabled() {
                encode_moe_swiglu_iq3_xxs_f32_fast
            } else {
                encode_moe_swiglu_iq3_xxs_f32
            };
            encode(
                ctx,
                enc,
                weights.routed_gate,
                weights.routed_up,
                input,
                buffers.topk_ids,
                buffers.routed_inner,
                g.hidden_size,
                g.routed_intermediate_size,
                g.expert_count,
                g.experts_per_token,
            )?
        }
        GgmlType::IQ4_XS => encode_moe_swiglu_iq4_xs_f32(
            ctx,
            enc,
            weights.routed_gate,
            weights.routed_up,
            input,
            buffers.topk_ids,
            buffers.routed_inner,
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            g.experts_per_token,
        )?,
        dtype => return invalid(format!("unsupported routed gate/up dtype {dtype:?}")),
    }
    Ok(())
}

fn encode_singleton_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    let g = weights.geometry;
    match weights.routed_down.dtype {
        GgmlType::IQ4_NL => {
            let encode = if qwen4exp_moe_iq4_down_fast_enabled() {
                encode_moe_down_iq4_nl_f32_fast
            } else {
                encode_moe_down_iq4_nl_f32
            };
            encode(
                ctx,
                enc,
                weights.routed_down,
                buffers.routed_inner,
                buffers.topk_ids,
                buffers.routed_expert_output,
                g.routed_intermediate_size,
                g.hidden_size,
                g.expert_count,
                g.experts_per_token,
            )?;
            encode_moe_weighted_sum_f32(
                ctx,
                enc,
                buffers.routed_expert_output,
                buffers.topk_weights,
                buffers.output,
                g.hidden_size,
                g.experts_per_token,
            )?;
        }
        GgmlType::Q8_0 => encode_moe_down_weighted_sum_q8_0_f32(
            ctx,
            enc,
            weights.routed_down,
            buffers.routed_inner,
            buffers.topk_ids,
            buffers.topk_weights,
            buffers.output,
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            g.experts_per_token,
        )?,
        dtype => return invalid(format!("unsupported routed down dtype {dtype:?}")),
    }
    Ok(())
}

fn encode_singleton_shared_gate_up(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    let g = weights.geometry;
    encode_shared_swiglu_q8_0_f32(
        ctx,
        enc,
        weights.shared_gate,
        weights.shared_up,
        input,
        buffers.shared_inner,
        g.hidden_size,
        g.shared_intermediate_size,
    )?;
    Ok(())
}

fn encode_singleton_shared_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    let g = weights.geometry;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.shared_down,
        buffers.shared_inner,
        buffers.shared_output,
        g.shared_intermediate_size,
        g.hidden_size,
    )?;
    Ok(())
}

fn encode_singleton_accumulate(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    buffers: Qwen4ExpMoeSingletonBuffers<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    encode_axpy_scalar_f32(
        ctx,
        enc,
        buffers.shared_output,
        buffers.shared_scale,
        buffers.output,
    )?;
    Ok(())
}

struct Qwen4ExpMoePackedExecution<'input, 'weights> {
    input: &'input MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'weights>,
    views: Qwen4ExpMoePackedViews,
    tokens: usize,
}

impl Qwen4ExpMoePackedExecution<'_, '_> {
    fn encode_router(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.router", layer, mixer),
        )?;
        encode_packed_router_projection(
            ctx,
            enc,
            self.weights.router,
            self.input,
            &self.views.router_logits,
            g,
            self.tokens,
        )?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_topk(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.topk", layer, mixer),
        )?;
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            ctx,
            enc,
            &self.views.router_logits,
            self.weights.shared_router,
            self.input,
            &self.views.topk_ids,
            &self.views.topk_weights,
            &self.views.shared_scale,
            g.expert_count,
            g.experts_per_token,
            g.hidden_size,
            self.tokens,
        )?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_bucket(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.bucket", layer, mixer),
        )?;
        encode_moe_route_bucket_slots_f32(
            ctx,
            enc,
            &self.views.topk_ids,
            &self.views.route_counts,
            &self.views.route_slots,
            g.expert_count,
            self.tokens,
            g.experts_per_token,
        )?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_route(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        self.encode_router(ctx, enc, layer, mixer, profile.as_deref_mut())?;
        self.encode_topk(ctx, enc, layer, mixer, profile.as_deref_mut())?;
        self.encode_bucket(ctx, enc, layer, mixer, profile.as_deref_mut())?;
        Ok(())
    }

    fn encode_routed_gate_up(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.routed_gate_up", layer, mixer),
        )?;
        match self.weights.routed_gate.dtype {
            GgmlType::IQ3_XXS => encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
                ctx,
                enc,
                self.weights.routed_gate,
                self.weights.routed_up,
                self.input,
                &self.views.route_counts,
                &self.views.route_slots,
                &self.views.routed_inner,
                g.hidden_size,
                g.routed_intermediate_size,
                g.expert_count,
                g.experts_per_token,
                self.tokens,
            )?,
            GgmlType::IQ4_XS => encode_moe_swiglu_iq4_xs_f32_grouped_slots_n16(
                ctx,
                enc,
                self.weights.routed_gate,
                self.weights.routed_up,
                self.input,
                &self.views.route_counts,
                &self.views.route_slots,
                &self.views.routed_inner,
                g.hidden_size,
                g.routed_intermediate_size,
                g.expert_count,
                g.experts_per_token,
                self.tokens,
            )?,
            dtype => {
                return invalid(format!("unsupported packed routed gate/up dtype {dtype:?}"));
            }
        }
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    #[cfg(test)]
    fn encode_iq3_gate_up_capture(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
    ) -> Result<(), Qwen4ExpMoeError> {
        if self.weights.routed_gate.dtype != GgmlType::IQ3_XXS
            || !qwen4exp_iq3_gate_up_probe_layer(layer)
        {
            return Ok(());
        }
        let Some(binding) = QWEN4EXP_IQ3_GATE_UP_CAPTURE.with(|slot| slot.borrow().clone()) else {
            return Ok(());
        };
        let layer_index = usize::try_from(layer).map_err(|_| {
            Qwen4ExpMoeError::Invalid("IQ3 gate/up capture layer exceeds usize".into())
        })?;
        if binding.seen_layers.borrow()[layer_index] {
            return invalid(format!(
                "IQ3 gate/up capture received duplicate layer {layer}"
            ));
        }
        let ordinal = binding.next_ordinal.get();
        let expected_ordinal = (0..layer)
            .filter(|&prior| qwen4exp_iq3_gate_up_probe_layer(prior))
            .count();
        if ordinal != expected_ordinal || ordinal >= QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS {
            return invalid(format!(
                "IQ3 gate/up capture layer {layer} has ordinal {ordinal}, expected {expected_ordinal}"
            ));
        }
        let g = self.weights.geometry;
        if self.tokens != binding.banks.tokens
            || g.hidden_size != PACKED_ROUTER_E8P32_STRICT_HIDDEN
            || g.expert_count != PACKED_ROUTER_E8P32_STRICT_EXPERTS
        {
            return invalid("IQ3 gate/up capture differs from released full-chunk geometry");
        }
        let input_destination = binding.banks.input_view(ordinal);
        let counts_destination = binding.banks.counts_view(ordinal);
        let slots_destination = binding.banks.slots_view(ordinal);
        require_same_device(
            ctx,
            &[
                ("IQ3 gate/up input source", self.input),
                ("IQ3 gate/up counts source", &self.views.route_counts),
                ("IQ3 gate/up slots source", &self.views.route_slots),
                ("IQ3 gate/up input capture", &input_destination),
                ("IQ3 gate/up counts capture", &counts_destination),
                ("IQ3 gate/up slots capture", &slots_destination),
            ],
        )?;
        require_disjoint(&[
            ("IQ3 gate/up input source", self.input),
            ("IQ3 gate/up counts source", &self.views.route_counts),
            ("IQ3 gate/up slots source", &self.views.route_slots),
            ("IQ3 gate/up input capture", &input_destination),
            ("IQ3 gate/up counts capture", &counts_destination),
            ("IQ3 gate/up slots capture", &slots_destination),
        ])?;
        let tag =
            |kind| format!("qwen4exp.iq3_gate_up_capture.ordinal{ordinal}.layer{layer}.{kind}");
        {
            let _tag = dispatch_census_tag_scope(|| tag("input"));
            encode_copy_offset_f32(
                ctx,
                enc,
                self.input,
                0,
                &input_destination,
                g.hidden_size * self.tokens,
            )?;
        }
        {
            let _tag = dispatch_census_tag_scope(|| tag("counts"));
            encode_copy_offset_i32(
                ctx,
                enc,
                &self.views.route_counts,
                0,
                &counts_destination,
                g.expert_count,
            )?;
        }
        {
            let _tag = dispatch_census_tag_scope(|| tag("slots"));
            encode_copy_offset_i32(
                ctx,
                enc,
                &self.views.route_slots,
                0,
                &slots_destination,
                g.expert_count * self.tokens,
            )?;
        }
        binding
            .records
            .borrow_mut()
            .push(Qwen4ExpIq3GateUpCaptureRecord { ordinal, layer });
        binding.next_ordinal.set(ordinal + 1);
        binding.seen_layers.borrow_mut()[layer_index] = true;
        Ok(())
    }

    fn encode_routed_down(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.routed_down", layer, mixer),
        )?;
        match self.weights.routed_down.dtype {
            GgmlType::IQ4_NL => {
                if packed_iq4_down_m128_n16_qualified(
                    ctx,
                    g,
                    self.weights.routed_down.dtype,
                    self.tokens,
                ) {
                    encode_moe_down_iq4_nl_f32_grouped_slots_m128_n16(
                        ctx,
                        enc,
                        self.weights.routed_down,
                        &self.views.routed_inner,
                        &self.views.route_counts,
                        &self.views.route_slots,
                        &self.views.routed_expert_output,
                        g.routed_intermediate_size,
                        g.hidden_size,
                        g.expert_count,
                        self.tokens,
                    )?;
                    static REPORTED: std::sync::Once = std::sync::Once::new();
                    REPORTED.call_once(|| {
                        eprintln!(
                            "qwen4exp: IQ4_NL packed-down M128xN16 active; rollback=QWEN4EXP_MOE_IQ4_DOWN_M128_N16=0"
                        );
                    });
                } else {
                    encode_moe_down_iq4_nl_f32_grouped_slots(
                        ctx,
                        enc,
                        self.weights.routed_down,
                        &self.views.routed_inner,
                        &self.views.route_counts,
                        &self.views.route_slots,
                        &self.views.routed_expert_output,
                        g.routed_intermediate_size,
                        g.hidden_size,
                        g.expert_count,
                        self.tokens,
                    )?;
                }
            }
            GgmlType::Q8_0 => encode_moe_down_q8_0_f32_grouped_slots(
                ctx,
                enc,
                self.weights.routed_down,
                &self.views.routed_inner,
                &self.views.route_counts,
                &self.views.route_slots,
                &self.views.routed_expert_output,
                g.routed_intermediate_size,
                g.hidden_size,
                g.expert_count,
                self.tokens,
            )?,
            dtype => return invalid(format!("unsupported packed routed down dtype {dtype:?}")),
        }
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_routed_reduce(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.routed_reduce", layer, mixer),
        )?;
        encode_moe_weighted_sum_packed_f32(
            ctx,
            enc,
            &self.views.routed_expert_output,
            &self.views.topk_weights,
            &self.views.output,
            g.hidden_size,
            g.experts_per_token,
            self.tokens,
        )?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn encode_shared(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        mixer: MixerKind,
        mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    ) -> Result<(), Qwen4ExpMoeError> {
        let g = self.weights.geometry;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.shared_gate_up", layer, mixer),
        )?;
        encode_mat_mat_dispatch(
            ctx,
            enc,
            self.weights.shared_gate,
            self.input,
            &self.views.shared_gate_projection,
            g.hidden_size,
            g.shared_intermediate_size,
            self.tokens,
        )?;
        encode_mat_mat_dispatch(
            ctx,
            enc,
            self.weights.shared_up,
            self.input,
            &self.views.shared_up_projection,
            g.hidden_size,
            g.shared_intermediate_size,
            self.tokens,
        )?;
        encode_silu_mul_f32(
            ctx,
            enc,
            &self.views.shared_gate_projection,
            &self.views.shared_up_projection,
            &self.views.shared_inner,
        )?;
        end_optional(&mut profile, enc, marker)?;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.shared_down", layer, mixer),
        )?;
        encode_mat_mat_dispatch(
            ctx,
            enc,
            self.weights.shared_down,
            &self.views.shared_inner,
            &self.views.shared_output,
            g.shared_intermediate_size,
            g.hidden_size,
            self.tokens,
        )?;
        end_optional(&mut profile, enc, marker)?;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("moe.shared_merge", layer, mixer),
        )?;
        encode_axpy_rowwise_f32(
            ctx,
            enc,
            &self.views.shared_output,
            &self.views.shared_scale,
            &self.views.output,
            g.hidden_size,
            self.tokens,
        )?;
        end_optional(&mut profile, enc, marker)?;
        Ok(())
    }

    fn into_output(self) -> MetalTensor {
        self.views.output
    }
}

fn encode_packed_router_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    geometry: Qwen4ExpMoeMetalGeometry,
    tokens: usize,
) -> Result<(), Qwen4ExpMoeError> {
    if packed_router_e8p32_strict_qualified(ctx, geometry, weight.dtype, tokens) {
        encode_mat_mat_f32_router_e8p32_strict(
            ctx,
            enc,
            weight,
            input,
            output,
            geometry.hidden_size,
            geometry.expert_count,
            tokens,
        )?;
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            eprintln!(
                "qwen4exp: strict-order E8P32 packed router active; rollback=QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0"
            );
        });
    } else {
        encode_mat_mat_dispatch(
            ctx,
            enc,
            weight,
            input,
            output,
            geometry.hidden_size,
            geometry.expert_count,
            tokens,
        )?;
    }
    Ok(())
}

/// Encode packed MoE rows into transaction-owned scratch.
///
/// # Safety
///
/// The caller must retain every tensor and exclusive logical ownership of
/// `scratch` until the command completes successfully or is permanently
/// abandoned. Any encoding or command failure makes scratch contents
/// indeterminate; the enclosing transaction must be poisoned.
pub(crate) unsafe fn encode_qwen4exp_moe_packed_motor(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    scratch: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
) -> Result<MetalTensor, Qwen4ExpMoeError> {
    #[cfg(test)]
    if qwen4exp_moe_route_count_capture_active() {
        return invalid("route-count capture requires a layer-aware packed MoE call");
    }
    #[cfg(test)]
    if qwen4exp_iq3_gate_up_capture_active() {
        return invalid("IQ3 gate/up capture requires a layer-aware packed MoE call");
    }
    unsafe {
        encode_qwen4exp_moe_packed_motor_for_layer(
            ctx,
            enc,
            input,
            weights,
            scratch,
            tokens,
            0,
            MixerKind::GatedDeltaNet,
        )
    }
}

/// Encode packed MoE rows while preserving the model-layer identity used by
/// profiling and test-only diagnostics.
///
/// # Safety
///
/// The caller must retain every tensor and exclusive logical ownership of
/// `scratch` until the command completes successfully or is permanently
/// abandoned. Any encoding or command failure makes scratch contents
/// indeterminate; the enclosing transaction must be poisoned.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_qwen4exp_moe_packed_motor_for_layer(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    scratch: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
    layer: u32,
    mixer: MixerKind,
) -> Result<MetalTensor, Qwen4ExpMoeError> {
    unsafe {
        encode_qwen4exp_moe_packed_motor_inner(
            ctx, enc, input, weights, scratch, tokens, layer, mixer, None,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_qwen4exp_moe_packed_motor_profiled(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    scratch: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
    layer: u32,
    mixer: MixerKind,
    recorder: &mut Qwen4ExpPackedProfileRecorder<'_>,
) -> Result<MetalTensor, Qwen4ExpMoeError> {
    unsafe {
        encode_qwen4exp_moe_packed_motor_inner(
            ctx,
            enc,
            input,
            weights,
            scratch,
            tokens,
            layer,
            mixer,
            Some(recorder),
        )
    }
}

/// Encode packed MoE rows across seven serial sampled encoders.
///
/// # Safety
///
/// The caller must retain the command, sample buffer, input, weights, and
/// exclusive scratch ownership until completion or permanent abandonment. Any
/// error requires the enclosing transaction to preserve its poisoned state
/// until successful abandonment.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_qwen4exp_moe_packed_motor_stage_sampled(
    ctx: &MetalContext,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    first_stage: usize,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    scratch: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
    layer: u32,
    mixer: MixerKind,
) -> Result<(MetalTensor, Vec<Qwen4ExpPackedProfileSpan>), Qwen4ExpMoeError> {
    #[cfg(test)]
    if qwen4exp_moe_route_count_capture_active() {
        return invalid("route-count capture is unavailable in stage-sampled packed MoE");
    }
    #[cfg(test)]
    if qwen4exp_iq3_gate_up_capture_active() {
        return invalid("IQ3 gate/up capture is unavailable in stage-sampled packed MoE");
    }
    if tokens <= 1 {
        return invalid("sampled packed MoE profiling requires at least two tokens");
    }
    let stage_index = |offset: usize| {
        first_stage
            .checked_add(offset)
            .ok_or_else(|| Qwen4ExpMoeError::Invalid("packed MoE stage index overflow".into()))
    };
    let router_encoder = stage_encoder(command, samples, stage_index(0)?)?;
    validate_encoder(ctx, &router_encoder)?;
    if weights.geometry != scratch.geometry {
        return invalid("packed MoE weight and scratch geometry differ");
    }
    validate_packed_contract(ctx, input, weights, scratch, tokens)?;
    let views = scratch.views(tokens)?;
    preflight_packed(ctx, weights, tokens)?;
    let execution = Qwen4ExpMoePackedExecution {
        input,
        weights,
        views,
        tokens,
    };

    execution.encode_router(ctx, &router_encoder, layer, mixer, None)?;
    router_encoder.end();
    let topk_encoder = stage_encoder(command, samples, stage_index(1)?)?;
    execution.encode_topk(ctx, &topk_encoder, layer, mixer, None)?;
    topk_encoder.end();
    let bucket_encoder = stage_encoder(command, samples, stage_index(2)?)?;
    execution.encode_bucket(ctx, &bucket_encoder, layer, mixer, None)?;
    bucket_encoder.end();
    let gate_up_encoder = stage_encoder(
        command,
        samples,
        stage_index(QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES)?,
    )?;
    execution.encode_routed_gate_up(ctx, &gate_up_encoder, layer, mixer, None)?;
    gate_up_encoder.end();
    let down_encoder = stage_encoder(
        command,
        samples,
        stage_index(QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES + 1)?,
    )?;
    execution.encode_routed_down(ctx, &down_encoder, layer, mixer, None)?;
    down_encoder.end();
    let reduce_encoder = stage_encoder(
        command,
        samples,
        stage_index(QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES + 2)?,
    )?;
    execution.encode_routed_reduce(ctx, &reduce_encoder, layer, mixer, None)?;
    reduce_encoder.end();
    let shared_encoder = stage_encoder(
        command,
        samples,
        stage_index(QWEN4EXP_PACKED_PROFILE_MOE_STAGES - 1)?,
    )?;
    execution.encode_shared(ctx, &shared_encoder, layer, mixer, None)?;
    shared_encoder.end();

    let output = execution.into_output();
    let routing_first_stage = stage_index(0)?;
    let routing_last_stage = stage_index(QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES - 1)?;
    let mut spans = Vec::with_capacity(QWEN4EXP_PACKED_PROFILE_MOE_SPANS);
    spans.push(Qwen4ExpPackedProfileSpan {
        label: Qwen4ExpPackedProfileLabel::detail("moe.routing", layer, mixer),
        depth: 2,
        start_sample: routing_first_stage * 2,
        end_sample: routing_last_stage * 2 + 1,
    });
    for (offset, name) in ["moe.router", "moe.topk", "moe.bucket"]
        .into_iter()
        .enumerate()
    {
        let stage = stage_index(offset)?;
        spans.push(Qwen4ExpPackedProfileSpan {
            label: Qwen4ExpPackedProfileLabel::detail(name, layer, mixer),
            depth: 3,
            start_sample: stage * 2,
            end_sample: stage * 2 + 1,
        });
    }
    for (offset, name) in [
        "moe.routed_gate_up",
        "moe.routed_down",
        "moe.routed_reduce",
        "moe.shared_tail",
    ]
    .into_iter()
    .enumerate()
    {
        let stage = stage_index(QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES + offset)?;
        spans.push(Qwen4ExpPackedProfileSpan {
            label: Qwen4ExpPackedProfileLabel::detail(name, layer, mixer),
            depth: 2,
            start_sample: stage * 2,
            end_sample: stage * 2 + 1,
        });
    }
    debug_assert_eq!(spans.len(), QWEN4EXP_PACKED_PROFILE_MOE_SPANS);
    Ok((output, spans))
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_qwen4exp_moe_packed_motor_inner(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    scratch: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
    layer: u32,
    mixer: MixerKind,
    mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
) -> Result<MetalTensor, Qwen4ExpMoeError> {
    validate_encoder(ctx, enc)?;
    if weights.geometry != scratch.geometry {
        return invalid("packed MoE weight and scratch geometry differ");
    }
    validate_packed_contract(ctx, input, weights, scratch, tokens)?;
    let views = scratch.views(tokens)?;

    if tokens == 1 {
        preflight(ctx, weights)?;
        encode_singleton_step(
            ctx,
            enc,
            input,
            weights,
            Qwen4ExpMoeSingletonBuffers {
                router_logits: &views.router_logits,
                topk_ids: &views.topk_ids,
                topk_weights: &views.topk_weights,
                shared_scale: &views.shared_scale,
                routed_inner: &views.routed_inner,
                routed_expert_output: &views.routed_expert_output,
                shared_inner: &views.shared_inner,
                shared_output: &views.shared_output,
                output: &views.output,
            },
        )?;
        return Ok(views.output);
    }

    preflight_packed(ctx, weights, tokens)?;
    let execution = Qwen4ExpMoePackedExecution {
        input,
        weights,
        views,
        tokens,
    };
    execution.encode_route(ctx, enc, layer, mixer, profile.as_deref_mut())?;
    #[cfg(test)]
    execution.encode_iq3_gate_up_capture(ctx, enc, layer)?;
    #[cfg(test)]
    encode_qwen4exp_moe_route_count_capture(
        ctx,
        enc,
        &execution.views.route_counts,
        execution.weights.geometry.expert_count,
        layer,
    )?;
    execution.encode_routed_gate_up(ctx, enc, layer, mixer, profile.as_deref_mut())?;
    execution.encode_routed_down(ctx, enc, layer, mixer, profile.as_deref_mut())?;
    execution.encode_routed_reduce(ctx, enc, layer, mixer, profile.as_deref_mut())?;
    execution.encode_shared(ctx, enc, layer, mixer, profile.as_deref_mut())?;
    Ok(execution.into_output())
}

#[cfg(test)]
pub(crate) fn encode_qwen4exp_iq3_gate_up_captured_arm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    banks: &Qwen4ExpIq3GateUpCaptureBanks,
    records: &[Qwen4ExpIq3GateUpCaptureRecord],
    weights: &[Qwen4ExpMoeMetalWeights<'_>],
    output: &MetalTensor,
    arm: Qwen4ExpIq3GateUpProbeArm,
) -> Result<(), Qwen4ExpMoeError> {
    if records.len() != QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS || weights.len() != 48 {
        return invalid(format!(
            "captured IQ3 gate/up replay requires {QWEN4EXP_IQ3_GATE_UP_PROBE_LAYERS} records and 48 weights, got {}/{}",
            records.len(),
            weights.len()
        ));
    }
    require_tensor(
        "captured IQ3 gate/up output",
        output,
        GgmlType::F32,
        &[640, 10, QWEN4EXP_IQ3_GATE_UP_CAPTURE_TOKENS as u64],
        true,
    )?;
    let (min_count, max_count) = arm.bounds();
    for (expected_ordinal, record) in records.iter().enumerate() {
        let expected_layer = (0_u32..48)
            .filter(|&layer| qwen4exp_iq3_gate_up_probe_layer(layer))
            .nth(expected_ordinal)
            .expect("43 credited ordinals map to model layers");
        if record.ordinal != expected_ordinal || record.layer != expected_layer {
            return invalid(format!(
                "captured IQ3 gate/up record {record:?} differs from ordinal {expected_ordinal} layer {expected_layer}"
            ));
        }
        let layer = usize::try_from(record.layer).map_err(|_| {
            Qwen4ExpMoeError::Invalid("captured IQ3 gate/up layer exceeds usize".into())
        })?;
        let layer_weights = weights[layer];
        let g = layer_weights.geometry;
        if g.hidden_size != PACKED_ROUTER_E8P32_STRICT_HIDDEN
            || g.expert_count != PACKED_ROUTER_E8P32_STRICT_EXPERTS
            || g.experts_per_token != 10
            || g.routed_intermediate_size != 640
            || layer_weights.routed_gate.dtype != GgmlType::IQ3_XXS
            || layer_weights.routed_up.dtype != GgmlType::IQ3_XXS
        {
            return invalid(format!(
                "captured IQ3 gate/up layer {} differs from standard released geometry",
                record.layer
            ));
        }
        let input = banks.input_view(record.ordinal);
        let counts = banks.counts_view(record.ordinal);
        let slots = banks.slots_view(record.ordinal);
        require_same_device(
            ctx,
            &[
                ("captured IQ3 gate/up input", &input),
                ("captured IQ3 gate/up counts", &counts),
                ("captured IQ3 gate/up slots", &slots),
                ("captured IQ3 gate weights", layer_weights.routed_gate),
                ("captured IQ3 up weights", layer_weights.routed_up),
                ("captured IQ3 gate/up output", output),
            ],
        )?;
        require_disjoint(&[
            ("captured IQ3 gate/up input", &input),
            ("captured IQ3 gate/up counts", &counts),
            ("captured IQ3 gate/up slots", &slots),
            ("captured IQ3 gate/up output", output),
        ])?;
        let _tag = dispatch_census_tag_scope(|| {
            format!(
                "qwen4exp.iq3_gate_up_range.{}.ordinal{}.layer{}",
                arm.as_str(),
                record.ordinal,
                record.layer
            )
        });
        encode_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_range(
            ctx,
            enc,
            layer_weights.routed_gate,
            layer_weights.routed_up,
            &input,
            &counts,
            &slots,
            output,
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            g.experts_per_token,
            banks.tokens,
            min_count,
            max_count,
        )?;
    }
    Ok(())
}

pub(crate) fn validate_contract(
    ctx: &MetalContext,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    workspace: &Qwen4ExpMoeMetalWorkspace,
) -> Result<(), Qwen4ExpMoeError> {
    let g = workspace.geometry;
    require_tensor(
        "MoE input",
        input,
        GgmlType::F32,
        &[g.hidden_size as u64],
        false,
    )?;
    require_offset_alignment("MoE input", input, 16)?;
    validate_weights(weights, g)?;

    for (name, tensor, dtype, shape) in [
        (
            "MoE router logits",
            &workspace.router_logits,
            GgmlType::F32,
            vec![g.expert_count as u64],
        ),
        (
            "MoE top-k IDs",
            &workspace.topk_ids,
            GgmlType::I32,
            vec![g.experts_per_token as u64],
        ),
        (
            "MoE top-k weights",
            &workspace.topk_weights,
            GgmlType::F32,
            vec![g.experts_per_token as u64],
        ),
        (
            "MoE shared gate scalar",
            &workspace.shared_gate,
            GgmlType::F32,
            vec![1],
        ),
        (
            "MoE routed inner",
            &workspace.routed_inner,
            GgmlType::F32,
            vec![
                g.routed_intermediate_size as u64,
                g.experts_per_token as u64,
            ],
        ),
        (
            "MoE routed expert output",
            &workspace.routed_expert_output,
            GgmlType::F32,
            vec![g.hidden_size as u64, g.experts_per_token as u64],
        ),
        (
            "MoE shared inner",
            &workspace.shared_inner,
            GgmlType::F32,
            vec![g.shared_intermediate_size as u64],
        ),
        (
            "MoE shared output",
            &workspace.shared_output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
        (
            "MoE output",
            &workspace.output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, true)?;
    }

    let named_weights = named_weights(weights);
    require_read_only_weights(&named_weights)?;

    let mut tensors = vec![("MoE input", input)];
    tensors.extend(named_weights);
    tensors.extend(workspace_tensors(workspace));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)
}

fn validate_weights(
    weights: Qwen4ExpMoeMetalWeights<'_>,
    g: Qwen4ExpMoeMetalGeometry,
) -> Result<(), Qwen4ExpMoeError> {
    require_projection(
        "MoE router",
        weights.router,
        g.hidden_size,
        g.expert_count,
        &[GgmlType::F32],
    )?;
    require_offset_alignment("MoE router", weights.router, 16)?;
    require_expert_bank(
        "MoE routed gate bank",
        weights.routed_gate,
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        &[GgmlType::IQ3_XXS, GgmlType::IQ4_XS],
    )?;
    require_expert_bank(
        "MoE routed up bank",
        weights.routed_up,
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        &[GgmlType::IQ3_XXS, GgmlType::IQ4_XS],
    )?;
    if weights.routed_gate.dtype != weights.routed_up.dtype {
        return invalid(format!(
            "MoE routed gate/up dtypes differ: {:?}/{:?}",
            weights.routed_gate.dtype, weights.routed_up.dtype
        ));
    }
    require_expert_bank(
        "MoE routed down bank",
        weights.routed_down,
        g.routed_intermediate_size,
        g.hidden_size,
        g.expert_count,
        &[GgmlType::IQ4_NL, GgmlType::Q8_0],
    )?;
    require_tensor(
        "MoE shared router",
        weights.shared_router,
        GgmlType::F32,
        &[g.hidden_size as u64],
        false,
    )?;
    require_projection(
        "MoE shared gate",
        weights.shared_gate,
        g.hidden_size,
        g.shared_intermediate_size,
        &[GgmlType::Q8_0],
    )?;
    require_projection(
        "MoE shared up",
        weights.shared_up,
        g.hidden_size,
        g.shared_intermediate_size,
        &[GgmlType::Q8_0],
    )?;
    require_projection(
        "MoE shared down",
        weights.shared_down,
        g.shared_intermediate_size,
        g.hidden_size,
        &[GgmlType::Q8_0],
    )?;
    Ok(())
}

fn named_weights(weights: Qwen4ExpMoeMetalWeights<'_>) -> [(&'static str, &MetalTensor); 8] {
    [
        ("MoE router", weights.router),
        ("MoE routed gate bank", weights.routed_gate),
        ("MoE routed up bank", weights.routed_up),
        ("MoE routed down bank", weights.routed_down),
        ("MoE shared router", weights.shared_router),
        ("MoE shared gate", weights.shared_gate),
        ("MoE shared up", weights.shared_up),
        ("MoE shared down", weights.shared_down),
    ]
}

pub(crate) fn validate_packed_contract(
    ctx: &MetalContext,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    scratch: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
) -> Result<(), Qwen4ExpMoeError> {
    if tokens == 0 || tokens > scratch.capacity {
        return invalid(format!(
            "packed MoE token count {tokens} is outside capacity {}",
            scratch.capacity
        ));
    }
    let g = scratch.geometry;
    if g.expert_count > 512 {
        return invalid(format!(
            "packed MoE supports at most 512 experts, got {}",
            g.expert_count
        ));
    }
    let slot_count = checked_product(&[tokens, g.experts_per_token], "packed route slots")?;
    if i32::try_from(slot_count).is_err() {
        return invalid(format!(
            "packed MoE route slot count {slot_count} exceeds i32"
        ));
    }
    require_tensor(
        "packed MoE input",
        input,
        GgmlType::F32,
        &[g.hidden_size as u64, tokens as u64],
        false,
    )?;
    require_offset_alignment("packed MoE input", input, 16)?;
    validate_weights(weights, g)?;
    for (dtype, n_in, n_out) in [
        (weights.router.dtype, g.hidden_size, g.expert_count),
        (
            weights.shared_gate.dtype,
            g.hidden_size,
            g.shared_intermediate_size,
        ),
        (
            weights.shared_up.dtype,
            g.hidden_size,
            g.shared_intermediate_size,
        ),
        (
            weights.shared_down.dtype,
            g.shared_intermediate_size,
            g.hidden_size,
        ),
    ] {
        validate_f32_q8_mat_mat_addressing(dtype, n_in, n_out, tokens)?;
    }

    for (name, tensor, dtype, shape) in [
        (
            "packed MoE router logits",
            &scratch.router_logits,
            GgmlType::F32,
            vec![g.expert_count as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE top-k IDs",
            &scratch.topk_ids,
            GgmlType::I32,
            vec![g.experts_per_token as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE top-k weights",
            &scratch.topk_weights,
            GgmlType::F32,
            vec![g.experts_per_token as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE shared scale",
            &scratch.shared_scale,
            GgmlType::F32,
            vec![scratch.capacity as u64],
        ),
        (
            "packed MoE route counts",
            &scratch.route_counts,
            GgmlType::I32,
            vec![g.expert_count as u64],
        ),
        (
            "packed MoE route slots",
            &scratch.route_slots,
            GgmlType::I32,
            vec![(g.expert_count * scratch.capacity) as u64],
        ),
        (
            "packed MoE routed inner",
            &scratch.routed_inner,
            GgmlType::F32,
            vec![
                g.routed_intermediate_size as u64,
                g.experts_per_token as u64,
                scratch.capacity as u64,
            ],
        ),
        (
            "packed MoE routed expert output",
            &scratch.routed_expert_output,
            GgmlType::F32,
            vec![
                g.hidden_size as u64,
                g.experts_per_token as u64,
                scratch.capacity as u64,
            ],
        ),
        (
            "packed MoE shared gate projection",
            &scratch.shared_gate_projection,
            GgmlType::F32,
            vec![g.shared_intermediate_size as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE shared up projection",
            &scratch.shared_up_projection,
            GgmlType::F32,
            vec![g.shared_intermediate_size as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE shared inner",
            &scratch.shared_inner,
            GgmlType::F32,
            vec![g.shared_intermediate_size as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE shared output",
            &scratch.shared_output,
            GgmlType::F32,
            vec![g.hidden_size as u64, scratch.capacity as u64],
        ),
        (
            "packed MoE output",
            &scratch.output,
            GgmlType::F32,
            vec![g.hidden_size as u64, scratch.capacity as u64],
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, true)?;
    }

    let named_weights = named_weights(weights);
    require_read_only_weights(&named_weights)?;
    let mut tensors = vec![("packed MoE input", input)];
    tensors.extend(named_weights);
    tensors.extend(packed_scratch_tensors(scratch));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)
}

fn packed_scratch_tensors(
    scratch: &Qwen4ExpMoePackedMotorScratch,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("packed MoE router logits", &scratch.router_logits),
        ("packed MoE top-k IDs", &scratch.topk_ids),
        ("packed MoE top-k weights", &scratch.topk_weights),
        ("packed MoE shared scale", &scratch.shared_scale),
        ("packed MoE route counts", &scratch.route_counts),
        ("packed MoE route slots", &scratch.route_slots),
        ("packed MoE routed inner", &scratch.routed_inner),
        (
            "packed MoE routed expert output",
            &scratch.routed_expert_output,
        ),
        (
            "packed MoE shared gate projection",
            &scratch.shared_gate_projection,
        ),
        (
            "packed MoE shared up projection",
            &scratch.shared_up_projection,
        ),
        ("packed MoE shared inner", &scratch.shared_inner),
        ("packed MoE shared output", &scratch.shared_output),
        ("packed MoE output", &scratch.output),
    ]
}

fn workspace_tensors(workspace: &Qwen4ExpMoeMetalWorkspace) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("MoE router logits", &workspace.router_logits),
        ("MoE top-k IDs", &workspace.topk_ids),
        ("MoE top-k weights", &workspace.topk_weights),
        ("MoE shared gate scalar", &workspace.shared_gate),
        ("MoE routed inner", &workspace.routed_inner),
        ("MoE routed expert output", &workspace.routed_expert_output),
        ("MoE shared inner", &workspace.shared_inner),
        ("MoE shared output", &workspace.shared_output),
        ("MoE output", &workspace.output),
    ]
}

pub(crate) fn preflight(
    ctx: &MetalContext,
    weights: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    preflight_projection(ctx, weights.router.dtype)?;
    preflight_projection(ctx, weights.shared_down.dtype)?;
    let (routed_gate_kernel, routed_gate_threads) = match weights.routed_gate.dtype {
        GgmlType::IQ3_XXS if qwen4exp_moe_iq3_fast_enabled() => {
            ("kernel_moe_swiglu_iq3_xxs_f32_fast", 64)
        }
        GgmlType::IQ3_XXS => ("kernel_moe_swiglu_iq3_xxs_f32", 64),
        GgmlType::IQ4_XS => ("kernel_moe_swiglu_iq4_xs_f32", 128),
        dtype => return invalid(format!("unsupported routed gate/up dtype {dtype:?}")),
    };
    require_pipeline_capacity(ctx, "kernel_topk_logits_softmax_f32", 1, 0)?;
    require_pipeline_capacity(ctx, "kernel_dot_sigmoid_f32", 32, 0)?;
    require_pipeline_capacity(ctx, routed_gate_kernel, routed_gate_threads, 0)?;
    match weights.routed_down.dtype {
        GgmlType::IQ4_NL => {
            if qwen4exp_moe_iq4_down_fast_enabled() {
                require_pipeline_capacity(
                    ctx,
                    "kernel_moe_down_iq4_nl_f32_fast",
                    64,
                    32 * size_of::<f32>(),
                )?;
            } else {
                require_pipeline_capacity(ctx, "kernel_moe_down_iq4_nl_f32", 128, 0)?;
            }
            ctx.pipeline("kernel_moe_weighted_sum_f32")?;
        }
        GgmlType::Q8_0 => require_pipeline_capacity(
            ctx,
            "kernel_moe_down_weighted_sum_q8_0_f32",
            128,
            32 * 2 * size_of::<f32>(),
        )?,
        dtype => return invalid(format!("unsupported routed down dtype {dtype:?}")),
    }
    require_pipeline_capacity(
        ctx,
        "kernel_shared_swiglu_q8_0_f32_lcpp",
        128,
        32 * 2 * 2 * size_of::<f32>(),
    )?;
    ctx.pipeline("kernel_axpy_scalar_f32")?;
    Ok(())
}

pub(crate) fn preflight_packed(
    ctx: &MetalContext,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    tokens: usize,
) -> Result<(), Qwen4ExpMoeError> {
    #[cfg(test)]
    if qwen4exp_moe_route_count_capture_active() {
        ctx.pipeline("kernel_copy_offset_i32")?;
    }
    #[cfg(test)]
    if qwen4exp_iq3_gate_up_capture_active() {
        ctx.pipeline("kernel_copy_offset_f32")?;
        ctx.pipeline("kernel_copy_offset_i32")?;
    }
    if packed_router_e8p32_strict_qualified(ctx, weights.geometry, weights.router.dtype, tokens) {
        require_pipeline_capacity(ctx, "kernel_mat_mat_f32_f32_router_e8p32_strict", 32, 0)?;
    } else {
        preflight_packed_projection(ctx, weights.router.dtype)?;
    }
    for dtype in [
        weights.shared_gate.dtype,
        weights.shared_up.dtype,
        weights.shared_down.dtype,
    ] {
        preflight_packed_projection(ctx, dtype)?;
    }
    let routed_gate_kernel = match weights.routed_gate.dtype {
        GgmlType::IQ3_XXS => "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16",
        GgmlType::IQ4_XS => "kernel_moe_swiglu_iq4_xs_f32_grouped_slots_n16",
        dtype => return invalid(format!("unsupported packed routed gate/up dtype {dtype:?}")),
    };
    require_pipeline_capacity(
        ctx,
        "kernel_topk_logits_softmax_dot_sigmoid_packed_f32",
        512,
        512 * 3 * size_of::<u32>(),
    )?;
    require_pipeline_capacity(ctx, "kernel_moe_route_bucket_slots_f32", 256, 0)?;
    require_pipeline_capacity(ctx, routed_gate_kernel, 128, 16_384)?;
    match weights.routed_down.dtype {
        GgmlType::IQ4_NL => {
            if packed_iq4_down_m128_n16_qualified(
                ctx,
                weights.geometry,
                weights.routed_down.dtype,
                tokens,
            ) {
                require_pipeline_capacity(
                    ctx,
                    "kernel_moe_down_iq4_nl_f32_grouped_slots_m128_n16",
                    128,
                    9_216,
                )?;
            } else {
                require_pipeline_capacity(
                    ctx,
                    "kernel_moe_down_iq4_nl_f32_grouped_slots",
                    128,
                    8_192,
                )?;
            }
        }
        GgmlType::Q8_0 => {
            require_pipeline_capacity(ctx, "kernel_moe_down_q8_0_f32_grouped_slots", 128, 8_192)?
        }
        dtype => return invalid(format!("unsupported packed routed down dtype {dtype:?}")),
    }
    require_pipeline_capacity(ctx, "kernel_moe_weighted_sum_packed_f32", 32, 0)?;
    ctx.pipeline("kernel_silu_mul_f32")?;
    ctx.pipeline("kernel_axpy_rowwise_f32")?;
    Ok(())
}

fn preflight_packed_projection(
    ctx: &MetalContext,
    dtype: GgmlType,
) -> Result<(), Qwen4ExpMoeError> {
    if !crate::qwen4exp_metal::preflight_projection_pipelines(ctx, dtype, true, false)? {
        return invalid(format!("unsupported packed MoE projection dtype {dtype:?}"));
    }
    Ok(())
}
fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpMoeError> {
    let kernels: &[(&str, usize, usize)] = match dtype {
        GgmlType::F32 => &[
            ("kernel_mat_vec_f32_f32", 128, 0),
            (
                "kernel_mat_vec_f32_f32_lcpp_r2",
                128,
                32 * 2 * size_of::<f32>(),
            ),
        ],
        GgmlType::Q8_0 => &[
            ("kernel_mat_vec_q8_0_f32", 64, 0),
            (
                "kernel_mat_vec_q8_0_f32_lcpp",
                128,
                32 * 2 * size_of::<f32>(),
            ),
        ],
        _ => return invalid(format!("unsupported MoE projection dtype {dtype:?}")),
    };
    for &(kernel, threads, dynamic_memory) in kernels {
        require_pipeline_capacity(ctx, kernel, threads, dynamic_memory)?;
    }
    Ok(())
}

fn require_pipeline_capacity(
    ctx: &MetalContext,
    name: &str,
    threads: usize,
    dynamic_memory: usize,
) -> Result<(), Qwen4ExpMoeError> {
    let pipeline = ctx.pipeline(name)?;
    let execution_width = pipeline.threadExecutionWidth();
    if execution_width != 32 {
        return invalid(format!(
            "MoE pipeline {name} requires SIMD width 32, got {execution_width}"
        ));
    }
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    if max_threads < threads {
        return invalid(format!(
            "MoE pipeline {name} requires {threads} threads per threadgroup, got {max_threads}"
        ));
    }
    let static_memory = pipeline.staticThreadgroupMemoryLength();
    let required_memory = static_memory
        .checked_add(dynamic_memory)
        .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("MoE pipeline {name} memory overflow")))?;
    let available_memory = ctx.device.maxThreadgroupMemoryLength();
    if required_memory > available_memory {
        return invalid(format!(
            "MoE pipeline {name} requires {required_memory} threadgroup bytes ({static_memory} static plus {dynamic_memory} dynamic), device exposes {available_memory}"
        ));
    }
    Ok(())
}

fn validate_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), Qwen4ExpMoeError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("dependent MoE dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "MoE encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpMoeMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpMoeError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    Ok(())
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    allowed_dtypes: &[GgmlType],
) -> Result<(), Qwen4ExpMoeError> {
    let shape = [n_in as u64, n_out as u64];
    if tensor.shape != shape || !allowed_dtypes.contains(&tensor.dtype) {
        return invalid(format!(
            "{name} must use {allowed_dtypes:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    require_row_alignment(name, tensor, n_in)?;
    require_range(name, tensor)
}

fn require_expert_bank(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    allowed_dtypes: &[GgmlType],
) -> Result<(), Qwen4ExpMoeError> {
    let shape = [n_in as u64, n_out as u64, expert_count as u64];
    if tensor.shape != shape || !allowed_dtypes.contains(&tensor.dtype) {
        return invalid(format!(
            "{name} must use {allowed_dtypes:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    require_row_alignment(name, tensor, n_in)?;
    require_range(name, tensor)
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpMoeError> {
    if tensor.dtype != dtype || tensor.shape != shape {
        return invalid(format!(
            "{name} must be {dtype:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    require_range(name, tensor)
}

fn require_row_alignment(
    name: &str,
    tensor: &MetalTensor,
    row_elements: usize,
) -> Result<(), Qwen4ExpMoeError> {
    let (block_elements, _) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpMoeError::Invalid(format!("{name} has unsupported dtype {:?}", tensor.dtype))
    })?;
    if !(row_elements as u64).is_multiple_of(block_elements) {
        return invalid(format!(
            "{name} row width {row_elements} is not aligned to {block_elements} elements for {:?}",
            tensor.dtype
        ));
    }
    Ok(())
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpMoeError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| Qwen4ExpMoeError::Invalid("tensor element count overflow".into()))?;
    let (block_elements, block_bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpMoeError::Invalid(format!("unsupported tensor dtype {:?}", tensor.dtype))
    })?;
    if block_elements == 0 || !elements.is_multiple_of(block_elements) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    (elements / block_elements)
        .checked_mul(block_bytes)
        .ok_or_else(|| Qwen4ExpMoeError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpMoeError> {
    let alignment = match tensor.dtype {
        GgmlType::F32 | GgmlType::I32 => 4,
        GgmlType::IQ3_XXS | GgmlType::IQ4_XS | GgmlType::IQ4_NL | GgmlType::Q8_0 => 2,
        dtype => return invalid(format!("{name} has unsupported dtype {dtype:?}")),
    };
    if !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    let bytes = storage_bytes(tensor)?;
    let end = tensor
        .offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_offset_alignment(
    name: &str,
    tensor: &MetalTensor,
    alignment: u64,
) -> Result<(), Qwen4ExpMoeError> {
    if alignment == 0 || !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    Ok(())
}

fn require_read_only_weights(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpMoeError> {
    for (name, tensor) in tensors {
        if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
            return invalid(format!("{name} must have read-only weight provenance"));
        }
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpMoeError> {
    let expected = ctx.device.registryID();
    for (name, tensor) in tensors {
        let actual = tensor.buffer.device().registryID();
        if actual != expected {
            return invalid(format!(
                "{name} belongs to Metal device registry {actual}, expected {expected}"
            ));
        }
    }
    Ok(())
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpMoeError> {
    for left in 0..tensors.len() {
        let left_bytes = storage_bytes(tensors[left].1)?;
        for right in left + 1..tensors.len() {
            if Retained::as_ptr(&tensors[left].1.buffer)
                != Retained::as_ptr(&tensors[right].1.buffer)
            {
                continue;
            }
            let right_bytes = storage_bytes(tensors[right].1)?;
            let left_end = tensors[left].1.offset.saturating_add(left_bytes);
            let right_end = tensors[right].1.offset.saturating_add(right_bytes);
            if tensors[left].1.offset < right_end && tensors[right].1.offset < left_end {
                return invalid(format!("{} overlaps {}", tensors[left].0, tensors[right].0));
            }
        }
    }
    Ok(())
}

fn checked_product(factors: &[usize], name: &str) -> Result<usize, Qwen4ExpMoeError> {
    factors.iter().try_fold(1_usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("MoE {name} element count overflow")))
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpMoeError> {
    Err(Qwen4ExpMoeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests;

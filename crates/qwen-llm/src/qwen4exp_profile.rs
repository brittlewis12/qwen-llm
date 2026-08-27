use crate::metal::{KernelEncoder, MetalError, MetalTimestampSampleBuffer};
use crate::qwen4exp::MixerKind;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;

pub(crate) const QWEN4EXP_PACKED_PROFILE_SAMPLE_CAPACITY: usize = 256;
pub(crate) const QWEN4EXP_PACKED_PROFILE_GDN_LAYER: u32 = 5;
pub(crate) const QWEN4EXP_PACKED_PROFILE_QSA_LAYER: u32 = 7;
pub(crate) const QWEN4EXP_PACKED_PROFILE_BLOCK_STAGES: usize = 6;
pub(crate) const QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES: usize = 3;
pub(crate) const QWEN4EXP_PACKED_PROFILE_MOE_STAGES: usize =
    QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES + 4;
pub(crate) const QWEN4EXP_PACKED_PROFILE_MOE_SPANS: usize =
    QWEN4EXP_PACKED_PROFILE_ROUTING_STAGES + 5;
pub(crate) const QWEN4EXP_PACKED_PROFILE_GDN_STAGES: usize =
    QWEN4EXP_PACKED_PROFILE_BLOCK_STAGES + QWEN4EXP_PACKED_PROFILE_MOE_STAGES - 1;
pub(crate) const QWEN4EXP_PACKED_PROFILE_QSA_STAGES: usize = QWEN4EXP_PACKED_PROFILE_BLOCK_STAGES;

pub(crate) fn is_stage_profiled_layer(layer: u32, mixer: MixerKind) -> bool {
    stage_profile_count(layer, mixer).is_some()
}

pub(crate) fn stage_profile_count(layer: u32, mixer: MixerKind) -> Option<usize> {
    match (layer, mixer) {
        (QWEN4EXP_PACKED_PROFILE_GDN_LAYER, MixerKind::GatedDeltaNet) => {
            Some(QWEN4EXP_PACKED_PROFILE_GDN_STAGES)
        }
        (QWEN4EXP_PACKED_PROFILE_QSA_LAYER, MixerKind::QwenSparseAttention) => {
            Some(QWEN4EXP_PACKED_PROFILE_QSA_STAGES)
        }
        _ => None,
    }
}

pub(crate) fn packed_stage_sample_count(post_ple_blocks: usize) -> Result<usize, MetalError> {
    post_ple_blocks
        .checked_add(2)
        .and_then(|base| base.checked_add(QWEN4EXP_PACKED_PROFILE_GDN_STAGES - 1))
        .and_then(|base| base.checked_add(QWEN4EXP_PACKED_PROFILE_QSA_STAGES - 1))
        .and_then(|stages| stages.checked_mul(2))
        .ok_or_else(|| MetalError::Counter("packed profile sample count overflow".into()))
}

pub(crate) fn packed_stage_span_count(post_ple_blocks: usize) -> Result<usize, MetalError> {
    post_ple_blocks
        .checked_add(2)
        .and_then(|base| base.checked_add(2 * QWEN4EXP_PACKED_PROFILE_BLOCK_STAGES))
        .and_then(|base| base.checked_add(QWEN4EXP_PACKED_PROFILE_MOE_SPANS))
        .ok_or_else(|| MetalError::Counter("packed profile span count overflow".into()))
}

pub(crate) fn stage_encoder(
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    samples: &MetalTimestampSampleBuffer,
    stage: usize,
) -> Result<KernelEncoder, MetalError> {
    let start_sample = stage
        .checked_mul(2)
        .ok_or_else(|| MetalError::Counter("packed stage sample index overflow".into()))?;
    let end_sample = start_sample + 1;
    if end_sample >= samples.sample_count() {
        return Err(MetalError::Counter(format!(
            "packed stage {stage} timestamp index {end_sample} exceeds {} samples",
            samples.sample_count()
        )));
    }
    KernelEncoder::try_begin_sampled(command, samples, start_sample, end_sample, false)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Qwen4ExpPackedProfileScope {
    Coarse,
    Detail,
}

impl Qwen4ExpPackedProfileScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coarse => "coarse",
            Self::Detail => "detail",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Qwen4ExpPackedProfileLabel {
    pub scope: Qwen4ExpPackedProfileScope,
    pub name: &'static str,
    pub layer: Option<u32>,
    pub mixer: Option<MixerKind>,
}

impl Qwen4ExpPackedProfileLabel {
    pub(crate) const fn coarse(
        name: &'static str,
        layer: Option<u32>,
        mixer: Option<MixerKind>,
    ) -> Self {
        Self {
            scope: Qwen4ExpPackedProfileScope::Coarse,
            name,
            layer,
            mixer,
        }
    }

    pub(crate) const fn detail(name: &'static str, layer: u32, mixer: MixerKind) -> Self {
        Self {
            scope: Qwen4ExpPackedProfileScope::Detail,
            name,
            layer: Some(layer),
            mixer: Some(mixer),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpPackedProfileSpan {
    pub label: Qwen4ExpPackedProfileLabel,
    pub depth: usize,
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpPackedProfileMarker {
    label: Qwen4ExpPackedProfileLabel,
    start_sample: usize,
    depth: usize,
}

pub(crate) struct Qwen4ExpPackedProfileRecorder<'a> {
    samples: &'a MetalTimestampSampleBuffer,
    detailed_gdn_layer: u32,
    detailed_qsa_layer: u32,
    next_sample: usize,
    open: Vec<Qwen4ExpPackedProfileLabel>,
    spans: Vec<Qwen4ExpPackedProfileSpan>,
}

impl<'a> Qwen4ExpPackedProfileRecorder<'a> {
    pub(crate) fn new(
        samples: &'a MetalTimestampSampleBuffer,
        detailed_gdn_layer: u32,
        detailed_qsa_layer: u32,
    ) -> Result<Self, MetalError> {
        if samples.sample_count() < QWEN4EXP_PACKED_PROFILE_SAMPLE_CAPACITY {
            return Err(MetalError::Counter(format!(
                "packed profile has {} timestamp samples, need at least {}",
                samples.sample_count(),
                QWEN4EXP_PACKED_PROFILE_SAMPLE_CAPACITY
            )));
        }
        if detailed_gdn_layer == detailed_qsa_layer {
            return Err(MetalError::Counter(
                "packed profile detail layers must be distinct".into(),
            ));
        }
        Ok(Self {
            samples,
            detailed_gdn_layer,
            detailed_qsa_layer,
            next_sample: 0,
            open: Vec::new(),
            spans: Vec::new(),
        })
    }

    pub(crate) fn is_detailed_layer(&self, layer: u32) -> bool {
        layer == self.detailed_gdn_layer || layer == self.detailed_qsa_layer
    }

    pub(crate) fn begin(
        &mut self,
        enc: &KernelEncoder,
        label: Qwen4ExpPackedProfileLabel,
    ) -> Result<Qwen4ExpPackedProfileMarker, MetalError> {
        let start_sample = self.sample(enc)?;
        let depth = self.open.len();
        self.open.push(label);
        Ok(Qwen4ExpPackedProfileMarker {
            label,
            start_sample,
            depth,
        })
    }

    pub(crate) fn end(
        &mut self,
        enc: &KernelEncoder,
        marker: Qwen4ExpPackedProfileMarker,
    ) -> Result<(), MetalError> {
        let Some(label) = self.open.pop() else {
            return Err(MetalError::Counter(
                "packed profile ended a stage with an empty stack".into(),
            ));
        };
        if label != marker.label || self.open.len() != marker.depth {
            return Err(MetalError::Counter(format!(
                "packed profile stage nesting mismatch: open={label:?} end={:?}",
                marker.label
            )));
        }
        let end_sample = self.sample(enc)?;
        self.spans.push(Qwen4ExpPackedProfileSpan {
            label,
            depth: marker.depth,
            start_sample: marker.start_sample,
            end_sample,
        });
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<(usize, Vec<Qwen4ExpPackedProfileSpan>), MetalError> {
        if !self.open.is_empty() {
            return Err(MetalError::Counter(format!(
                "packed profile finished with {} open stages",
                self.open.len()
            )));
        }
        if self.spans.is_empty() {
            return Err(MetalError::Counter(
                "packed profile recorded no stages".into(),
            ));
        }
        Ok((self.next_sample, self.spans))
    }

    fn sample(&mut self, enc: &KernelEncoder) -> Result<usize, MetalError> {
        if self.next_sample >= self.samples.sample_count() {
            return Err(MetalError::Counter(format!(
                "packed profile timestamp buffer exhausted at sample {}",
                self.next_sample
            )));
        }
        let sample = self.next_sample;
        self.next_sample += 1;
        enc.sample_counters(self.samples, sample, true);
        Ok(sample)
    }
}

#[inline(always)]
pub(crate) fn begin_optional(
    recorder: &mut Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    enc: &KernelEncoder,
    label: Qwen4ExpPackedProfileLabel,
) -> Result<Option<Qwen4ExpPackedProfileMarker>, MetalError> {
    recorder
        .as_deref_mut()
        .map(|recorder| recorder.begin(enc, label))
        .transpose()
}

#[inline(always)]
pub(crate) fn end_optional(
    recorder: &mut Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    enc: &KernelEncoder,
    marker: Option<Qwen4ExpPackedProfileMarker>,
) -> Result<(), MetalError> {
    if let (Some(recorder), Some(marker)) = (recorder.as_deref_mut(), marker) {
        recorder.end(enc, marker)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_stage_plan_expands_only_standard_layers_five_and_seven() {
        assert!(is_stage_profiled_layer(5, MixerKind::GatedDeltaNet));
        assert!(is_stage_profiled_layer(7, MixerKind::QwenSparseAttention));
        assert!(!is_stage_profiled_layer(5, MixerKind::QwenSparseAttention));
        assert!(!is_stage_profiled_layer(3, MixerKind::QwenSparseAttention));
        assert_eq!(stage_profile_count(5, MixerKind::GatedDeltaNet), Some(12));
        assert_eq!(
            stage_profile_count(7, MixerKind::QwenSparseAttention),
            Some(6)
        );
        assert_eq!(packed_stage_sample_count(46).unwrap(), 128);
        assert_eq!(packed_stage_span_count(46).unwrap(), 68);
        assert!(packed_stage_sample_count(usize::MAX).is_err());
        assert!(packed_stage_span_count(usize::MAX).is_err());
    }
}

//! Stage recording, profiles, and route/selection records.

use super::*;

/// SHA-256 of the exact metallib embedded in this diagnostics-enabled binary.
#[cfg(feature = "dsv4-diagnostics")]
pub fn deepseek_v4_diagnostics_metallib_sha256() -> [u8; 32] {
    use sha2::{Digest, Sha256};

    Sha256::digest(crate::KERNELS_METALLIB).into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MemorySamples {
    pub before_residency_bytes: u64,
    pub after_residency_bytes: u64,
    pub after_session_bytes: u64,
    pub after_first_forward_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4LayerCommandProfile {
    pub layer: usize,
    pub attention_kind: AttentionKind,
    pub routing_kind: DeepSeekV4RoutingKind,
    pub encode_cpu_ms: f64,
    pub command_gpu_ms: f64,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4CommandProfile {
    pub position: u32,
    pub forward_wall_ms: f64,
    pub layers: Vec<DeepSeekV4LayerCommandProfile>,
}

#[derive(Clone, Copy, Debug, Default)]
#[doc(hidden)]
pub struct DeepSeekV4WholeTokenProfile {
    pub position: u32,
    pub forward_wall_ms: f64,
    pub guards_phase_cpu_ms: f64,
    pub record_reset_cpu_ms: f64,
    pub command_encoder_create_cpu_ms: f64,
    pub encode_cpu_ms: f64,
    pub commit_cpu_ms: f64,
    pub commit_wait_wall_ms: f64,
    pub command_gpu_start_seconds: f64,
    pub command_gpu_end_seconds: f64,
    pub command_gpu_ms: f64,
    pub command_status_cpu_ms: f64,
    pub record_read_cpu_ms: f64,
    pub record_validate_callback_cpu_ms: f64,
    pub causal_commit_cpu_ms: f64,
}

impl DeepSeekV4WholeTokenProfile {
    pub fn wait_residual_ms(self) -> f64 {
        self.commit_wait_wall_ms - self.command_gpu_ms
    }

    pub fn outside_gpu_ms(self) -> f64 {
        self.forward_wall_ms - self.command_gpu_ms
    }

    pub fn accounted_outside_gpu_ms(self) -> f64 {
        self.guards_phase_cpu_ms
            + self.record_reset_cpu_ms
            + self.command_encoder_create_cpu_ms
            + self.encode_cpu_ms
            + self.wait_residual_ms()
            + self.command_status_cpu_ms
            + self.record_read_cpu_ms
            + self.record_validate_callback_cpu_ms
            + self.causal_commit_cpu_ms
    }

    pub fn reconstruction_residual_ms(self) -> f64 {
        self.outside_gpu_ms() - self.accounted_outside_gpu_ms()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum DeepSeekV4StageKind {
    AttentionHyperConnection,
    AttentionPrepare,
    AttentionCore,
    AttentionOutput,
    HyperConnectionBridge,
    MoeRouter,
    MoeRoutedExperts,
    MoeSharedExpert,
    MoeCombine,
    LayerTail,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4StageTiming {
    pub kind: DeepSeekV4StageKind,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
    pub fraction_of_layer_gpu: f64,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4SampledLayerProfile {
    pub layer: usize,
    pub command_gpu_ms: f64,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
    pub encoder_boundary_ms_scaled: f64,
    pub stages: Vec<DeepSeekV4StageTiming>,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4StageProfile {
    pub position: u32,
    pub forward_wall_ms: f64,
    pub layers: Vec<DeepSeekV4LayerCommandProfile>,
    pub sampled_layers: Vec<DeepSeekV4SampledLayerProfile>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DeepSeekV4PendingStageSample {
    pub(super) layer: usize,
    pub(super) kind: DeepSeekV4StageKind,
    pub(super) start_sample: usize,
    pub(super) end_sample: usize,
}

pub(super) const DEEPSEEK_V4_STAGE_KINDS: [DeepSeekV4StageKind; 10] = [
    DeepSeekV4StageKind::AttentionHyperConnection,
    DeepSeekV4StageKind::AttentionPrepare,
    DeepSeekV4StageKind::AttentionCore,
    DeepSeekV4StageKind::AttentionOutput,
    DeepSeekV4StageKind::HyperConnectionBridge,
    DeepSeekV4StageKind::MoeRouter,
    DeepSeekV4StageKind::MoeRoutedExperts,
    DeepSeekV4StageKind::MoeSharedExpert,
    DeepSeekV4StageKind::MoeCombine,
    DeepSeekV4StageKind::LayerTail,
];

pub(super) fn resolve_deepseek_v4_layer_stage_samples(
    layer: usize,
    records: &[DeepSeekV4PendingStageSample],
    timestamps: &[u64],
    command_gpu_ms: f64,
) -> Result<DeepSeekV4SampledLayerProfile, DeepSeekV4MetalError> {
    if records.len() != DEEPSEEK_V4_STAGE_KINDS.len() {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} produced {} stages, expected {}",
            records.len(),
            DEEPSEEK_V4_STAGE_KINDS.len()
        ));
    }
    if !command_gpu_ms.is_finite() || command_gpu_ms <= 0.0 {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} has invalid command GPU duration {command_gpu_ms}"
        ));
    }
    for (record, expected) in records.iter().zip(DEEPSEEK_V4_STAGE_KINDS) {
        if record.layer != layer {
            return invalid(format!(
                "DeepSeek V4 sampled layer {layer} contains a record for layer {}",
                record.layer
            ));
        }
        if record.kind != expected {
            return invalid(format!(
                "DeepSeek V4 sampled layer {layer} recorded {:?}, expected {expected:?}",
                record.kind
            ));
        }
        if record.start_sample >= timestamps.len() || record.end_sample >= timestamps.len() {
            return invalid(format!(
                "DeepSeek V4 sampled layer {layer} stage {:?} indexes samples {}..{} from {} timestamps",
                record.kind,
                record.start_sample,
                record.end_sample,
                timestamps.len()
            ));
        }
    }

    let first_timestamp = timestamps[records[0].start_sample];
    let last_timestamp = timestamps[records[records.len() - 1].end_sample];
    let sampled_span_ticks = last_timestamp.checked_sub(first_timestamp).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "DeepSeek V4 sampled layer {layer} returned non-monotonic span timestamps"
        ))
    })?;
    if sampled_span_ticks == 0 {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} returned a zero timestamp span"
        ));
    }

    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stage_ticks = 0u64;
    let mut boundary_ticks = 0u64;
    let mut stages = Vec::with_capacity(records.len());
    let mut previous_end = None;
    for record in records {
        let start_timestamp = timestamps[record.start_sample];
        let end_timestamp = timestamps[record.end_sample];
        let duration_ticks = end_timestamp.checked_sub(start_timestamp).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "DeepSeek V4 sampled layer {layer} stage {:?} returned inverted timestamps",
                record.kind
            ))
        })?;
        if let Some(previous_end) = previous_end {
            let gap = start_timestamp.checked_sub(previous_end).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 sampled layer {layer} stage {:?} overlaps its predecessor",
                    record.kind
                ))
            })?;
            boundary_ticks = boundary_ticks.checked_add(gap).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 sampled layer {layer} encoder-gap tick total overflow"
                ))
            })?;
        }
        previous_end = Some(end_timestamp);
        stage_ticks = stage_ticks.checked_add(duration_ticks).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "DeepSeek V4 sampled layer {layer} stage tick total overflow"
            ))
        })?;
        let duration_ms_scaled = duration_ticks as f64 * scale_ms_per_tick;
        stages.push(DeepSeekV4StageTiming {
            kind: record.kind,
            start_timestamp,
            end_timestamp,
            duration_ticks,
            duration_ms_scaled,
            fraction_of_layer_gpu: duration_ms_scaled / command_gpu_ms,
        });
    }
    let accounted_ticks = stage_ticks.checked_add(boundary_ticks).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "DeepSeek V4 sampled layer {layer} accounted tick total overflow"
        ))
    })?;
    if accounted_ticks != sampled_span_ticks {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} accounts for {accounted_ticks} ticks across a {sampled_span_ticks}-tick span"
        ));
    }
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(DeepSeekV4SampledLayerProfile {
        layer,
        command_gpu_ms,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
        encoder_boundary_ms_scaled: boundary_ticks as f64 * scale_ms_per_tick,
        stages,
    })
}

pub(super) struct DeepSeekV4StageRecorder {
    pub(super) samples: MetalTimestampSampleBuffer,
    pub(super) sampled_layer_mask: [bool; DEEPSEEK_V4_LAYER_COUNT],
    pub(super) next_sample: usize,
    pub(super) records: Vec<DeepSeekV4PendingStageSample>,
    pub(super) command_gpu_ms: [Option<f64>; DEEPSEEK_V4_LAYER_COUNT],
    pub(super) resolved: Option<Vec<DeepSeekV4SampledLayerProfile>>,
}

impl DeepSeekV4StageRecorder {
    const STAGES_PER_LAYER: usize = DEEPSEEK_V4_STAGE_KINDS.len();

    pub(super) fn new(
        ctx: &MetalContext,
        sampled_layers: &[usize],
    ) -> Result<Self, DeepSeekV4MetalError> {
        if sampled_layers.is_empty() {
            return invalid("DeepSeek V4 stage profile requires at least one sampled layer");
        }
        let mut sampled_layer_mask = [false; DEEPSEEK_V4_LAYER_COUNT];
        for &layer in sampled_layers {
            if layer >= DEEPSEEK_V4_LAYER_COUNT {
                return invalid(format!(
                    "DeepSeek V4 stage-profile layer {layer} is outside 0..{DEEPSEEK_V4_LAYER_COUNT}"
                ));
            }
            if std::mem::replace(&mut sampled_layer_mask[layer], true) {
                return invalid(format!(
                    "DeepSeek V4 stage-profile layer {layer} was requested twice"
                ));
            }
        }
        let sample_count = sampled_layers
            .len()
            .checked_mul(Self::STAGES_PER_LAYER)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("DeepSeek V4 stage sample count overflow".into())
            })?;
        Ok(Self {
            samples: ctx.timestamp_sample_buffer(sample_count)?,
            sampled_layer_mask,
            next_sample: 0,
            records: Vec::with_capacity(sample_count / 2),
            command_gpu_ms: [None; DEEPSEEK_V4_LAYER_COUNT],
            resolved: None,
        })
    }

    pub(super) fn samples_layer(&self, layer: usize) -> bool {
        self.sampled_layer_mask[layer]
    }

    pub(super) fn begin(
        &mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        kind: DeepSeekV4StageKind,
    ) -> Result<KernelEncoder, DeepSeekV4MetalError> {
        if !self.samples_layer(layer) {
            return invalid(format!(
                "DeepSeek V4 stage recorder was asked to sample unselected layer {layer}"
            ));
        }
        let start_sample = self.next_sample;
        let end_sample = start_sample.checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("DeepSeek V4 stage sample index overflow".into())
        })?;
        if end_sample >= self.samples.sample_count() {
            return invalid(format!(
                "DeepSeek V4 stage timestamp buffer exhausted at sample {end_sample}"
            ));
        }
        self.next_sample = end_sample + 1;
        self.records.push(DeepSeekV4PendingStageSample {
            layer,
            kind,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::begin_sampled(
            command,
            &self.samples,
            start_sample,
            end_sample,
            false,
        ))
    }

    pub(super) fn record_command_gpu_ms(&mut self, layer: usize, command_gpu_ms: f64) {
        if self.samples_layer(layer) {
            self.command_gpu_ms[layer] = Some(command_gpu_ms);
        }
    }

    pub(super) fn resolve(&mut self, ctx: &MetalContext) -> Result<(), DeepSeekV4MetalError> {
        let timestamps = ctx.resolve_timestamp_samples(&self.samples, self.next_sample)?;
        let mut resolved = Vec::with_capacity(
            self.sampled_layer_mask
                .iter()
                .filter(|sampled| **sampled)
                .count(),
        );
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            if !self.samples_layer(layer) {
                continue;
            }
            let records = self
                .records
                .iter()
                .filter(|record| record.layer == layer)
                .copied()
                .collect::<Vec<_>>();
            let command_gpu_ms = self.command_gpu_ms[layer].ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 sampled layer {layer} has no command GPU duration"
                ))
            })?;
            resolved.push(resolve_deepseek_v4_layer_stage_samples(
                layer,
                &records,
                &timestamps,
                command_gpu_ms,
            )?);
        }
        self.resolved = Some(resolved);
        Ok(())
    }

    pub(super) fn take_resolved(
        &mut self,
    ) -> Result<Vec<DeepSeekV4SampledLayerProfile>, DeepSeekV4MetalError> {
        self.resolved.take().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 stage samples were not resolved before forward completion".into(),
            )
        })
    }
}

impl DeepSeekV4CommandProfile {
    pub fn encode_cpu_ms(&self) -> f64 {
        self.layers.iter().map(|layer| layer.encode_cpu_ms).sum()
    }

    pub fn command_gpu_ms(&self) -> f64 {
        self.layers.iter().map(|layer| layer.command_gpu_ms).sum()
    }
}

#[derive(Clone)]
pub(super) struct DeepSeekV4SelectionRecord {
    pub(super) visible_count: MetalTensor,
    pub(super) selected_count: MetalTensor,
    pub(super) status: MetalTensor,
}

impl DeepSeekV4SelectionRecord {
    pub(super) fn validate(&self) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(&self.visible_count, &[1], true, "CSA visible-count record")?;
        validate_i32(
            &self.selected_count,
            &[1],
            true,
            "CSA selected-count record",
        )?;
        validate_i32(&self.status, &[1], true, "CSA status record")
    }
}

pub(super) struct DeepSeekV4LayerSelectionRecords {
    pub(super) integers: MetalTensor,
}

pub(super) struct DeepSeekV4CompletedLayerSelectionRecords {
    pub(super) integers: Vec<i32>,
}

impl DeepSeekV4LayerSelectionRecords {
    pub(super) fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
        Ok(Self {
            integers: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
        })
    }

    pub(super) fn layer(
        &self,
        layer: usize,
    ) -> Result<DeepSeekV4SelectionRecord, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "CSA selection-record layer {layer} is out of range"
            ));
        }
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            true,
            "CSA layer-selection records",
        )?;
        let base = checked_mul(
            layer,
            DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH,
            "CSA selection-record layer offset",
        )? as u64;
        let record = DeepSeekV4SelectionRecord {
            visible_count: self.integers.view_subrange(base, vec![1]),
            selected_count: self.integers.view_subrange(base + 1, vec![1]),
            status: self.integers.view_subrange(base + 2, vec![1]),
        };
        record.validate()?;
        Ok(record)
    }

    pub(super) fn reset_for_token(&self) -> Result<(), DeepSeekV4MetalError> {
        host_write_i32(
            &self.integers,
            &[-1; DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH * DEEPSEEK_V4_LAYER_COUNT],
            "reset CSA layer-selection records",
        )
    }

    pub(super) fn read_completed(
        &self,
    ) -> Result<DeepSeekV4CompletedLayerSelectionRecords, DeepSeekV4MetalError> {
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            false,
            "completed CSA layer-selection records",
        )?;
        Ok(DeepSeekV4CompletedLayerSelectionRecords {
            integers: host_read_i32(&self.integers, "completed CSA layer-selection records")?,
        })
    }
}

impl DeepSeekV4CompletedLayerSelectionRecords {
    pub(super) fn validate_layer(
        &self,
        layer: usize,
        expected_visible_count: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "completed CSA selection layer {layer} is out of range"
            ));
        }
        let base = checked_mul(
            layer,
            DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH,
            "completed CSA selection-record layer offset",
        )?;
        let visible_count = self.integers[base];
        let selected_count = self.integers[base + 1];
        let status = self.integers[base + 2];
        if visible_count != expected_visible_count as i32
            || selected_count != DEEPSEEK_V4_CSA_TOP_K as i32
            || status != 0
        {
            return invalid(format!(
                "layer {layer} sparse CSA selection failed with visible={visible_count} expected={expected_visible_count} selected={selected_count} status={status}"
            ));
        }
        Ok(())
    }
}

pub(super) const DEEPSEEK_V4_GROUPED_DENSE_STAGED_ROWS: usize = 16;

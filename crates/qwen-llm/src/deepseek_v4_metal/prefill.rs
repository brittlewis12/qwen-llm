use super::*;
#[cfg(feature = "dsv4-diagnostics")]
use std::rc::Rc;
use std::sync::OnceLock;

pub const DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS: usize = 4_096;
pub const DEEPSEEK_V4_PREFILL_MAX_TOKENS: usize = 4_096;
const PACKED_MATRIX_MIN_TOKENS: usize = 2_048;

const QUERY_WIDTH: usize = 64 * 512;
const GROUP_WIDTH: usize = QUERY_WIDTH / 8;
const LOW_RANK_WIDTH: usize = 8 * 1_024;
const COMPRESSOR_ATTENTION_WIDTH: usize = 2 * 512;
const COMPRESSOR_INDEXER_WIDTH: usize = 2 * 128;
const INDEXER_HEAD_COUNT: usize = 64;
const INDEXER_HEAD_DIM: usize = 128;
const INDEXER_QUERY_WIDTH: usize = INDEXER_HEAD_COUNT * INDEXER_HEAD_DIM;
const MOE_FFN_SIZE: usize = 2_048;
const MOE_EXPERT_COUNT: usize = 256;
const MOE_TOP_K: usize = 6;
#[cfg(feature = "dsv4-diagnostics")]
const MHC_DELETE_SITE_COUNT: usize = DEEPSEEK_V4_LAYER_COUNT * 2;
#[cfg(any(test, feature = "dsv4-diagnostics"))]
const PACKED_ROUTE_RECORD_WIDTH: usize = 4;
const PACKED_COMPACT_ROUTE_HEADER_WIDTH: usize = 8;
const PACKED_COMPACT_ROUTE_STATUS_READY: i32 = 1;
#[cfg(test)]
const PACKED_COMPACT_ROUTE_STATUS_INVALID_ROUTE: i32 = -301;

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4MhcDeleteArm {
    Current,
    Producer,
    Zero,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4MhcExecutionKind {
    Capture,
    Current,
    Producer,
    Zero,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4MhcSiteKind {
    Attention,
    Ffn,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4MhcSiteKind {
    fn index(self) -> usize {
        match self {
            Self::Attention => 0,
            Self::Ffn => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::Ffn => "ffn",
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4MhcBufferRole {
    None,
    OrdinaryMixes,
    Oracle,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MhcSiteRecord {
    pub ordinal: usize,
    pub layer: usize,
    pub site: DeepSeekV4MhcSiteKind,
    pub producer_runs: bool,
    pub producer_output_role: DeepSeekV4MhcBufferRole,
    pub producer_output_offset: Option<u64>,
    pub controls_input_role: DeepSeekV4MhcBufferRole,
    pub controls_input_offset: u64,
    pub site_bytes: u64,
    pub shape: [u64; 2],
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4MhcCommandKind {
    PreExpert,
    Expert,
    SharedOverlap,
    MergedPreExpert,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct DeepSeekV4MhcCommandInterval {
    pub submission_ordinal: usize,
    pub layer: usize,
    pub kind: DeepSeekV4MhcCommandKind,
    pub gpu_start_seconds: f64,
    pub gpu_end_seconds: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4MhcCommandInterval {
    pub fn duration_ms(&self) -> f64 {
        (self.gpu_end_seconds - self.gpu_start_seconds) * 1e3
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone)]
struct PackedMhcOracleBuffer {
    storage: Rc<PackedMhcOracleStorage>,
    n_tokens: usize,
    device_registry_id: u64,
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedMhcOracleStorage {
    tensor: MetalTensor,
    site_views: Box<[MetalTensor]>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedMhcOracleBuffer {
    fn new_capture(ctx: &MetalContext, n_tokens: usize) -> Result<Self, DeepSeekV4MetalError> {
        checked_token_count(n_tokens)?;
        let tensor = MetalTensor::zeros_f32(
            ctx,
            vec![
                DEEPSEEK_V4_HC_PARAMETER_COUNT as u64,
                n_tokens as u64,
                MHC_DELETE_SITE_COUNT as u64,
            ],
        )?;
        Self::from_tensor(tensor, n_tokens, ctx.device.registryID())
    }

    fn from_tensor(
        tensor: MetalTensor,
        n_tokens: usize,
        device_registry_id: u64,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let site_elements =
            checked_mul(n_tokens, DEEPSEEK_V4_HC_PARAMETER_COUNT, "mHC oracle site")?;
        let site_views = (0..MHC_DELETE_SITE_COUNT)
            .map(|ordinal| {
                let offset = checked_mul(ordinal, site_elements, "mHC oracle offset")?;
                Ok(tensor.view_subrange(
                    offset as u64,
                    vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, n_tokens as u64],
                ))
            })
            .collect::<Result<Vec<_>, DeepSeekV4MetalError>>()?
            .into_boxed_slice();
        Ok(Self {
            storage: Rc::new(PackedMhcOracleStorage { tensor, site_views }),
            n_tokens,
            device_registry_id,
        })
    }

    fn site_view(
        &self,
        layer: usize,
        site: DeepSeekV4MhcSiteKind,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("mHC oracle layer {layer} is out of range"));
        }
        let ordinal = layer * 2 + site.index();
        self.storage
            .site_views
            .get(ordinal)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("mHC oracle site is missing".into()))
    }

    fn validate_device_and_tokens(
        &self,
        ctx: &MetalContext,
        n_tokens: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "mHC oracle belongs to device {}, got {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        if self.n_tokens != n_tokens {
            return invalid(format!(
                "mHC oracle has {} tokens, requested {n_tokens}",
                self.n_tokens
            ));
        }
        Ok(())
    }

    fn current_payload_sha256(&self) -> Result<[u8; 32], DeepSeekV4MetalError> {
        use sha2::{Digest, Sha256};

        let values = host_read_f32(&self.storage.tensor, "sealed mHC oracle")?;
        Ok(Sha256::digest(bytemuck::cast_slice(&values)).into())
    }

    fn copy_payload_bits(&self) -> Result<Vec<u32>, DeepSeekV4MetalError> {
        Ok(host_read_f32(&self.storage.tensor, "sealed mHC oracle")?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MhcOracleIdentity {
    pub model_content_id: [u8; 32],
    pub compatibility_id: [u8; 32],
    pub token_sha256: [u8; 32],
    pub policy_sha256: [u8; 32],
    pub policy_manifest: String,
    pub start_position: u32,
    pub n_tokens: usize,
    pub rms_epsilon_bits: u32,
    pub hc_epsilon_bits: u32,
    pub device_registry_id: u64,
    pub residency_tensor_count: usize,
    pub residency_source_bytes: u64,
    pub expert_count: usize,
}

#[cfg(feature = "dsv4-diagnostics")]
struct DeepSeekV4MhcCapture {
    buffer: PackedMhcOracleBuffer,
    identity: DeepSeekV4MhcOracleIdentity,
    payload_sha256: [u8; 32],
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MhcTimedEndpoint {
    pub sha256: [u8; 32],
    pub position: u32,
    pub token_count: usize,
    pub logits_count: usize,
    pub hidden_count: usize,
}

#[cfg(feature = "dsv4-diagnostics")]
fn mhc_timed_endpoint_sha256(
    logits_bits: &[u32],
    hidden_bits: &[u32],
    tokens: &[u32],
    position: u32,
) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update(b"qwen.dsv4.mhc-delete-timed-endpoint.v1\0");
    for values in [logits_bits, hidden_bits, tokens] {
        digest.update((values.len() as u64).to_le_bytes());
        digest.update(bytemuck::cast_slice(values));
    }
    digest.update(position.to_le_bytes());
    digest.finalize().into()
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MhcEndpointEvidence {
    endpoint_logits_bits: Vec<u32>,
    endpoint_hidden_bits: Vec<u32>,
    endpoint_position: u32,
    endpoint_tokens: Vec<u32>,
    endpoint_prefix_digest: [u8; 32],
    endpoint_compatibility_id: [u8; 32],
    endpoint_causal_digest: [u8; 32],
    endpoint_observation: DeepSeekV4SnapshotObservation,
    continuation_logits_bits: Vec<u32>,
    continuation_hidden_bits: Vec<u32>,
    continuation_position: u32,
    continuation_tokens: Vec<u32>,
    continuation_prefix_digest: [u8; 32],
    continuation_compatibility_id: [u8; 32],
    continuation_causal_digest: [u8; 32],
    continuation_observation: DeepSeekV4SnapshotObservation,
    sha256: [u8; 32],
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4MhcEndpointEvidence {
    fn refresh_sha256(&mut self) {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        digest.update(b"qwen.dsv4.mhc-delete-endpoint.v1\0");
        for values in [&self.endpoint_logits_bits, &self.endpoint_hidden_bits] {
            digest.update((values.len() as u64).to_le_bytes());
            digest.update(bytemuck::cast_slice(values));
        }
        digest.update(self.endpoint_position.to_le_bytes());
        digest.update((self.endpoint_tokens.len() as u64).to_le_bytes());
        digest.update(bytemuck::cast_slice(&self.endpoint_tokens));
        digest.update(self.endpoint_prefix_digest);
        digest.update(self.endpoint_compatibility_id);
        digest.update(self.endpoint_causal_digest);
        digest.update([self.endpoint_observation as u8]);
        for values in [
            &self.continuation_logits_bits,
            &self.continuation_hidden_bits,
        ] {
            digest.update((values.len() as u64).to_le_bytes());
            digest.update(bytemuck::cast_slice(values));
        }
        digest.update(self.continuation_position.to_le_bytes());
        digest.update((self.continuation_tokens.len() as u64).to_le_bytes());
        digest.update(bytemuck::cast_slice(&self.continuation_tokens));
        digest.update(self.continuation_prefix_digest);
        digest.update(self.continuation_compatibility_id);
        digest.update(self.continuation_causal_digest);
        digest.update([self.continuation_observation as u8]);
        self.sha256 = digest.finalize().into();
    }

    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub fn endpoint_position(&self) -> u32 {
        self.endpoint_position
    }

    pub fn continuation_position(&self) -> u32 {
        self.continuation_position
    }

    pub fn endpoint_causal_digest(&self) -> [u8; 32] {
        self.endpoint_causal_digest
    }

    pub fn continuation_causal_digest(&self) -> [u8; 32] {
        self.continuation_causal_digest
    }

    pub fn timed_endpoint(&self) -> DeepSeekV4MhcTimedEndpoint {
        DeepSeekV4MhcTimedEndpoint {
            sha256: mhc_timed_endpoint_sha256(
                &self.endpoint_logits_bits,
                &self.endpoint_hidden_bits,
                &self.endpoint_tokens,
                self.endpoint_position,
            ),
            position: self.endpoint_position,
            token_count: self.endpoint_tokens.len(),
            logits_count: self.endpoint_logits_bits.len(),
            hidden_count: self.endpoint_hidden_bits.len(),
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
pub struct DeepSeekV4MhcVerifiedCapture {
    capture: DeepSeekV4MhcCapture,
    evidence: DeepSeekV4MhcEndpointEvidence,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4MhcVerifiedCapture {
    pub fn identity(&self) -> &DeepSeekV4MhcOracleIdentity {
        &self.capture.identity
    }

    pub fn payload_sha256(&self) -> [u8; 32] {
        self.capture.payload_sha256
    }

    pub fn payload_bytes(&self) -> u64 {
        self.capture.buffer.storage.tensor.n_bytes()
    }

    pub fn evidence(&self) -> &DeepSeekV4MhcEndpointEvidence {
        &self.evidence
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone)]
pub struct DeepSeekV4MhcOracle {
    buffer: PackedMhcOracleBuffer,
    identity: DeepSeekV4MhcOracleIdentity,
    payload_sha256: [u8; 32],
    endpoint_sha256: [u8; 32],
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4MhcOracle {
    fn validate_replay(
        &self,
        ctx: &MetalContext,
        n_tokens: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.buffer.validate_device_and_tokens(ctx, n_tokens)
    }

    pub fn identity(&self) -> &DeepSeekV4MhcOracleIdentity {
        &self.identity
    }

    pub fn n_tokens(&self) -> usize {
        self.buffer.n_tokens
    }

    pub fn payload_bytes(&self) -> u64 {
        self.buffer.storage.tensor.n_bytes()
    }

    pub fn sealed_sha256(&self) -> [u8; 32] {
        self.payload_sha256
    }

    pub fn endpoint_sha256(&self) -> [u8; 32] {
        self.endpoint_sha256
    }

    pub fn current_payload_sha256(&self) -> Result<[u8; 32], DeepSeekV4MetalError> {
        self.buffer.current_payload_sha256()
    }

    pub fn copy_payload_bits(&self) -> Result<Vec<u32>, DeepSeekV4MetalError> {
        self.buffer.copy_payload_bits()
    }
}

/// Compare two independent captures and their complete endpoint evidence
/// before making either payload replayable.
#[cfg(feature = "dsv4-diagnostics")]
#[doc(hidden)]
pub fn seal_mhc_delete_oracle_pair(
    left: DeepSeekV4MhcVerifiedCapture,
    right: DeepSeekV4MhcVerifiedCapture,
) -> Result<DeepSeekV4MhcOracle, DeepSeekV4MetalError> {
    if left.capture.identity != right.capture.identity {
        return invalid("independent mHC capture identities differ");
    }
    if left.evidence != right.evidence {
        return invalid("independent mHC capture endpoints differ");
    }
    if left.capture.payload_sha256 != right.capture.payload_sha256 {
        return invalid("independent mHC capture payload digests differ");
    }
    let left_bits = left.capture.buffer.copy_payload_bits()?;
    let right_bits = right.capture.buffer.copy_payload_bits()?;
    if left_bits != right_bits {
        return invalid("independent mHC capture payload bits differ");
    }
    if left.capture.buffer.current_payload_sha256()? != left.capture.payload_sha256
        || right.capture.buffer.current_payload_sha256()? != right.capture.payload_sha256
    {
        return invalid("mHC capture payload changed before sealing");
    }
    Ok(DeepSeekV4MhcOracle {
        buffer: left.capture.buffer,
        identity: left.capture.identity,
        payload_sha256: left.capture.payload_sha256,
        endpoint_sha256: left.evidence.sha256,
    })
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct DeepSeekV4MhcDeleteProfile {
    pub execution: DeepSeekV4MhcExecutionKind,
    pub queue_identity: u64,
    pub wall_ms: f64,
    pub sites: Vec<DeepSeekV4MhcSiteRecord>,
    pub command_intervals: Vec<DeepSeekV4MhcCommandInterval>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4MhcDeleteProfile {
    pub fn raw_gpu_ms(&self) -> f64 {
        self.command_intervals
            .iter()
            .map(DeepSeekV4MhcCommandInterval::duration_ms)
            .sum()
    }

    pub fn union_gpu_ms(&self) -> f64 {
        let mut intervals = self
            .command_intervals
            .iter()
            .map(|interval| (interval.gpu_start_seconds, interval.gpu_end_seconds))
            .collect::<Vec<_>>();
        intervals.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut total = 0.0;
        let mut active: Option<(f64, f64)> = None;
        for (start, end) in intervals {
            match active {
                Some((active_start, active_end)) if start <= active_end => {
                    active = Some((active_start, active_end.max(end)));
                }
                Some((active_start, active_end)) => {
                    total += active_end - active_start;
                    active = Some((start, end));
                }
                None => active = Some((start, end)),
            }
        }
        if let Some((start, end)) = active {
            total += end - start;
        }
        total * 1e3
    }
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedMhcSitePlan<'a> {
    producer_runs: bool,
    producer_output: Option<&'a MetalTensor>,
    controls_input: Option<&'a MetalTensor>,
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedMhcExecution {
    kind: DeepSeekV4MhcExecutionKind,
    oracle: Option<PackedMhcOracleBuffer>,
    identity: Option<DeepSeekV4MhcOracleIdentity>,
    next_site: usize,
    record_sites: bool,
    sites: Vec<DeepSeekV4MhcSiteRecord>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedMhcExecution {
    fn capture(oracle: PackedMhcOracleBuffer) -> Self {
        Self::new(
            DeepSeekV4MhcExecutionKind::Capture,
            Some(oracle),
            None,
            true,
        )
    }

    fn replay(
        arm: DeepSeekV4MhcDeleteArm,
        oracle: &DeepSeekV4MhcOracle,
        record_sites: bool,
    ) -> Self {
        let kind = match arm {
            DeepSeekV4MhcDeleteArm::Current => DeepSeekV4MhcExecutionKind::Current,
            DeepSeekV4MhcDeleteArm::Producer => DeepSeekV4MhcExecutionKind::Producer,
            DeepSeekV4MhcDeleteArm::Zero => DeepSeekV4MhcExecutionKind::Zero,
        };
        Self::new(
            kind,
            (arm != DeepSeekV4MhcDeleteArm::Current).then(|| oracle.buffer.clone()),
            Some(oracle.identity.clone()),
            record_sites,
        )
    }

    fn new(
        kind: DeepSeekV4MhcExecutionKind,
        oracle: Option<PackedMhcOracleBuffer>,
        identity: Option<DeepSeekV4MhcOracleIdentity>,
        record_sites: bool,
    ) -> Self {
        Self {
            kind,
            oracle,
            identity,
            next_site: 0,
            record_sites,
            sites: if record_sites {
                Vec::with_capacity(MHC_DELETE_SITE_COUNT)
            } else {
                Vec::new()
            },
        }
    }

    fn bind_identity(
        &mut self,
        identity: DeepSeekV4MhcOracleIdentity,
    ) -> Result<(), DeepSeekV4MetalError> {
        if let Some(expected) = &self.identity
            && expected != &identity
        {
            return invalid("mHC replay identity differs from the sealed capture");
        }
        self.identity = Some(identity);
        Ok(())
    }

    fn site_plan<'a>(
        &'a mut self,
        layer: usize,
        site: DeepSeekV4MhcSiteKind,
        ordinary_mixes: &'a MetalTensor,
    ) -> Result<PackedMhcSitePlan<'a>, DeepSeekV4MetalError> {
        let ordinal = layer
            .checked_mul(2)
            .and_then(|value| value.checked_add(site.index()))
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("mHC site ordinal overflow".into()))?;
        if ordinal != self.next_site {
            return invalid(format!(
                "mHC site order mismatch: expected {}, got {ordinal} ({layer}/{site:?})",
                self.next_site
            ));
        }
        let oracle_metadata = self
            .oracle
            .as_ref()
            .map(|buffer| {
                buffer.site_view(layer, site).map(|view| {
                    (
                        view.offset,
                        view.n_bytes(),
                        [DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, view.shape[1]],
                    )
                })
            })
            .transpose()?;
        let (producer_runs, producer_role, controls_role) = match self.kind {
            DeepSeekV4MhcExecutionKind::Capture => (
                true,
                DeepSeekV4MhcBufferRole::Oracle,
                DeepSeekV4MhcBufferRole::Oracle,
            ),
            DeepSeekV4MhcExecutionKind::Current => (
                true,
                DeepSeekV4MhcBufferRole::OrdinaryMixes,
                DeepSeekV4MhcBufferRole::OrdinaryMixes,
            ),
            DeepSeekV4MhcExecutionKind::Producer => (
                true,
                DeepSeekV4MhcBufferRole::OrdinaryMixes,
                DeepSeekV4MhcBufferRole::Oracle,
            ),
            DeepSeekV4MhcExecutionKind::Zero => (
                false,
                DeepSeekV4MhcBufferRole::None,
                DeepSeekV4MhcBufferRole::Oracle,
            ),
        };
        if self.kind != DeepSeekV4MhcExecutionKind::Current && oracle_metadata.is_none() {
            return invalid("mHC oracle-backed arm omitted its buffer");
        }
        if self.record_sites {
            let oracle_metadata = oracle_metadata.unwrap_or((
                ordinary_mixes.offset,
                ordinary_mixes.n_bytes(),
                [
                    DEEPSEEK_V4_HC_PARAMETER_COUNT as u64,
                    ordinary_mixes.shape[1],
                ],
            ));
            let producer_offset = if producer_runs {
                Some(match self.kind {
                    DeepSeekV4MhcExecutionKind::Capture => oracle_metadata.0,
                    DeepSeekV4MhcExecutionKind::Current | DeepSeekV4MhcExecutionKind::Producer => {
                        ordinary_mixes.offset
                    }
                    DeepSeekV4MhcExecutionKind::Zero => unreachable!(),
                })
            } else {
                None
            };
            let controls = match self.kind {
                DeepSeekV4MhcExecutionKind::Current => (
                    ordinary_mixes.offset,
                    ordinary_mixes.n_bytes(),
                    [
                        DEEPSEEK_V4_HC_PARAMETER_COUNT as u64,
                        ordinary_mixes.shape[1],
                    ],
                ),
                DeepSeekV4MhcExecutionKind::Capture
                | DeepSeekV4MhcExecutionKind::Producer
                | DeepSeekV4MhcExecutionKind::Zero => oracle_metadata,
            };
            self.sites.push(DeepSeekV4MhcSiteRecord {
                ordinal,
                layer,
                site,
                producer_runs,
                producer_output_role: producer_role,
                producer_output_offset: producer_offset,
                controls_input_role: controls_role,
                controls_input_offset: controls.0,
                site_bytes: controls.1,
                shape: controls.2,
            });
        }
        self.next_site += 1;
        let oracle = self
            .oracle
            .as_ref()
            .map(|buffer| buffer.site_view(layer, site))
            .transpose()?;
        let producer_output = match self.kind {
            DeepSeekV4MhcExecutionKind::Capture => oracle,
            DeepSeekV4MhcExecutionKind::Current | DeepSeekV4MhcExecutionKind::Producer => {
                Some(ordinary_mixes)
            }
            DeepSeekV4MhcExecutionKind::Zero => None,
        };
        let controls_input = match self.kind {
            DeepSeekV4MhcExecutionKind::Capture
            | DeepSeekV4MhcExecutionKind::Producer
            | DeepSeekV4MhcExecutionKind::Zero => oracle,
            DeepSeekV4MhcExecutionKind::Current => Some(ordinary_mixes),
        };
        Ok(PackedMhcSitePlan {
            producer_runs,
            producer_output,
            controls_input,
        })
    }

    fn finish(
        self,
    ) -> Result<(Vec<DeepSeekV4MhcSiteRecord>, DeepSeekV4MhcOracleIdentity), DeepSeekV4MetalError>
    {
        if self.next_site != MHC_DELETE_SITE_COUNT
            || (self.record_sites && self.sites.len() != MHC_DELETE_SITE_COUNT)
        {
            return invalid(format!(
                "mHC execution recorded {} of {MHC_DELETE_SITE_COUNT} sites",
                self.sites.len()
            ));
        }
        let identity = self.identity.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("mHC execution omitted its capture identity".into())
        })?;
        Ok((self.sites, identity))
    }
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedMhcCommandCollector {
    intervals: Vec<DeepSeekV4MhcCommandInterval>,
    wall_started: Option<std::time::Instant>,
    wall_ms: Option<f64>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl Default for PackedMhcCommandCollector {
    fn default() -> Self {
        Self {
            intervals: Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT * 3),
            wall_started: None,
            wall_ms: None,
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedMhcCommandCollector {
    fn begin_wall(&mut self) -> Result<(), DeepSeekV4MetalError> {
        if self.wall_started.is_some() || self.wall_ms.is_some() {
            return invalid("mHC packet wall timer was already started");
        }
        self.wall_started = Some(std::time::Instant::now());
        Ok(())
    }

    fn end_wall(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let started = self.wall_started.take().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("mHC packet wall timer was not started".into())
        })?;
        self.wall_ms = Some(started.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    fn record(
        &mut self,
        layer: usize,
        kind: DeepSeekV4MhcCommandKind,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> Result<(), DeepSeekV4MetalError> {
        let status = command.status();
        let error = command.error();
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        if status != MTLCommandBufferStatus::Completed
            || error.is_some()
            || !start.is_finite()
            || !end.is_finite()
            || start <= 0.0
            || end <= start
        {
            return invalid(format!(
                "mHC passive command is invalid at layer {layer} {kind:?}: status={status:?} error={error:?} interval={start}/{end}"
            ));
        }
        self.intervals.push(DeepSeekV4MhcCommandInterval {
            submission_ordinal: self.intervals.len(),
            layer,
            kind,
            gpu_start_seconds: start,
            gpu_end_seconds: end,
        });
        Ok(())
    }

    fn finish(self) -> Result<(f64, Vec<DeepSeekV4MhcCommandInterval>), DeepSeekV4MetalError> {
        let wall_ms = self.wall_ms.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("mHC packet wall timer did not complete".into())
        })?;
        if let Some((index, interval)) = self
            .intervals
            .iter()
            .enumerate()
            .find(|(index, interval)| interval.submission_ordinal != *index)
        {
            return invalid(format!(
                "mHC passive command ordinal mismatch at {index}: {}",
                interval.submission_ordinal
            ));
        }
        let mut index = 0;
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let first = self.intervals.get(index).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "mHC passive command ledger is missing layer {layer}"
                ))
            })?;
            if first.layer != layer {
                return invalid(format!(
                    "mHC passive command ledger expected layer {layer}, got {}",
                    first.layer
                ));
            }
            match first.kind {
                DeepSeekV4MhcCommandKind::MergedPreExpert => index += 1,
                DeepSeekV4MhcCommandKind::PreExpert => {
                    index += 1;
                    if self.intervals.get(index).is_some_and(|interval| {
                        interval.layer == layer
                            && interval.kind == DeepSeekV4MhcCommandKind::SharedOverlap
                    }) {
                        index += 1;
                    }
                    let expert = self.intervals.get(index).ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "mHC passive command ledger is missing layer {layer} expert"
                        ))
                    })?;
                    if expert.layer != layer || expert.kind != DeepSeekV4MhcCommandKind::Expert {
                        return invalid(format!(
                            "mHC passive command ledger expected layer {layer} expert, got {}/{:?}",
                            expert.layer, expert.kind
                        ));
                    }
                    index += 1;
                }
                kind => {
                    return invalid(format!(
                        "mHC passive command ledger starts layer {layer} with {kind:?}"
                    ));
                }
            }
        }
        if index != self.intervals.len() {
            return invalid(format!(
                "mHC passive command ledger has {} trailing intervals",
                self.intervals.len() - index
            ));
        }
        Ok((wall_ms, self.intervals))
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedRouteArgs {
    expert_count: u32,
    top_k: u32,
    n_tokens: u32,
    produced_tokens: u32,
    vocab_size: u32,
    generation: u32,
    routed_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedRouteScheduleArgs {
    expert_count: u32,
    top_k: u32,
    n_tokens: u32,
    generation: u32,
    max_tiles32: u32,
    max_tiles16: u32,
}

struct PackedGpuRouteBuffers<'a> {
    logits: &'a MetalTensor,
    token_ids: &'a MetalTensor,
    expert_ids: &'a MetalTensor,
    weights: &'a MetalTensor,
    route_generations: &'a MetalTensor,
    route_status: &'a MetalTensor,
    counts: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    slot_ids: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    schedule_generations: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    aggregate: &'a MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    signature: &'a MetalTensor,
    compact_header: &'a MetalTensor,
}

fn packed_route_args(
    n_tokens: usize,
    produced_tokens: usize,
    vocab_size: usize,
    generation: NonZeroU32,
    routed_scale: f32,
) -> Result<PackedRouteArgs, DeepSeekV4MetalError> {
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return invalid("packed GPU route scale must be finite and positive");
    }
    Ok(PackedRouteArgs {
        expert_count: MOE_EXPERT_COUNT as u32,
        top_k: MOE_TOP_K as u32,
        n_tokens: checked_token_count(n_tokens)?,
        produced_tokens: u32::try_from(produced_tokens).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed GPU route producer count exceeds u32".into())
        })?,
        vocab_size: u32::try_from(vocab_size).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed GPU route vocabulary exceeds u32".into())
        })?,
        generation: generation.get(),
        routed_scale,
    })
}

fn packed_route_schedule_args(
    n_tokens: usize,
    generation: NonZeroU32,
) -> Result<PackedRouteScheduleArgs, DeepSeekV4MetalError> {
    Ok(PackedRouteScheduleArgs {
        expert_count: MOE_EXPERT_COUNT as u32,
        top_k: MOE_TOP_K as u32,
        n_tokens: checked_token_count(n_tokens)?,
        generation: generation.get(),
        max_tiles32: PACKED_GROUPED_EXPERT_MAX_TILES as u32,
        max_tiles16: PACKED_GROUPED_IQ2_MMA16_MAX_TILES as u32,
    })
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_completion(generation: u32, n_tokens: usize) -> u32 {
    0xd551_0000 ^ generation ^ ((n_tokens as u32) << 8)
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_signature_completion(generation: u32, n_tokens: usize) -> u32 {
    0xd552_0000 ^ generation ^ ((n_tokens as u32) << 8)
}

fn packed_route_compact_completion(generation: u32, n_tokens: usize) -> u32 {
    0xd553_0000 ^ generation ^ ((n_tokens as u32) << 8)
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_signature_mix(hash: u32, value: u32) -> u32 {
    (hash ^ value).wrapping_mul(16_777_619)
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn packed_route_signature_hash(
    expert_ids: &[i32],
    weights: &[f32],
    counts: &[i32],
    slot_ids: &[i32],
    n_tokens: usize,
) -> Result<u32, DeepSeekV4MetalError> {
    if expert_ids.len() != n_tokens * MOE_TOP_K
        || weights.len() != expert_ids.len()
        || counts.len() != MOE_EXPERT_COUNT
        || slot_ids.len() != n_tokens * MOE_EXPERT_COUNT
    {
        return invalid("packed GPU route signature payload has invalid geometry");
    }
    let mut partials = [0u32; 256];
    for tid in 0..256 {
        let mut hash = 2_166_136_261 ^ tid as u32;
        for index in (tid..expert_ids.len()).step_by(256) {
            hash = packed_route_signature_mix(hash, expert_ids[index] as u32);
            hash = packed_route_signature_mix(hash, weights[index].to_bits());
        }
        hash = packed_route_signature_mix(hash, counts[tid] as u32);
        let base = tid * n_tokens;
        for &slot in &slot_ids[base..base + n_tokens] {
            hash = packed_route_signature_mix(hash, slot as u32);
        }
        partials[tid] = hash;
    }
    Ok(partials
        .into_iter()
        .fold(2_166_136_261, packed_route_signature_mix))
}

impl PackedGpuRouteBuffers<'_> {
    #[allow(clippy::too_many_arguments)]
    fn encode_learned(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        bias: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        generation: NonZeroU32,
        routed_scale: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed learned GPU route")?;
        if produced_tokens == 0 || produced_tokens > n_tokens {
            return invalid(format!(
                "packed learned route producer count {produced_tokens} is outside 1..={n_tokens}"
            ));
        }
        validate_f32(
            bias,
            &[MOE_EXPERT_COUNT as u64],
            false,
            "packed learned route bias",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_learned")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed learned route requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &packed_route_args(n_tokens, produced_tokens, 0, generation, routed_scale)?,
        );
        enc.set_tensor(1, self.logits);
        enc.set_tensor(2, bias);
        enc.set_tensor(3, self.expert_ids);
        enc.set_tensor(4, self.weights);
        enc.set_tensor(5, self.route_generations);
        enc.set_tensor(6, self.route_status);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: produced_tokens,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_hash(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_to_expert: &MetalTensor,
        n_tokens: usize,
        produced_tokens: usize,
        generation: NonZeroU32,
        routed_scale: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed hash GPU route")?;
        if produced_tokens == 0 || produced_tokens > n_tokens {
            return invalid(format!(
                "packed hash route producer count {produced_tokens} is outside 1..={n_tokens}"
            ));
        }
        validate_i32_bank(token_to_expert, MOE_TOP_K, "packed hash route map")?;
        let vocab_size = usize::try_from(token_to_expert.shape[1]).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed hash vocabulary exceeds usize".into())
        })?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_hash")?;
        if pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed hash route requires 256 threads per threadgroup");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &packed_route_args(
                n_tokens,
                produced_tokens,
                vocab_size,
                generation,
                routed_scale,
            )?,
        );
        enc.set_tensor(1, self.logits);
        enc.set_tensor(2, self.token_ids);
        enc.set_tensor(3, token_to_expert);
        enc.set_tensor(4, self.expert_ids);
        enc.set_tensor(5, self.weights);
        enc.set_tensor(6, self.route_generations);
        enc.set_tensor(7, self.route_status);
        enc.dispatch(
            MTLSize {
                width: produced_tokens.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    fn encode_schedule(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        produced_experts: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route schedule")?;
        if produced_experts == 0 || produced_experts > MOE_EXPERT_COUNT {
            return invalid(format!(
                "packed schedule producer count {produced_experts} is outside 1..={MOE_EXPERT_COUNT}"
            ));
        }
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_schedule")?;
        if pso.maxTotalThreadsPerThreadgroup() < produced_experts {
            return invalid("packed schedule exceeds pipeline threadgroup capacity");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.expert_ids);
        enc.set_tensor(2, self.counts);
        enc.set_tensor(3, self.slot_ids);
        enc.set_tensor(4, self.schedule_generations);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: produced_experts,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    fn encode_validate(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route validator")?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_validate")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed route validator requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.route_generations);
        enc.set_tensor(2, self.route_status);
        enc.set_tensor(3, self.expert_ids);
        enc.set_tensor(4, self.weights);
        enc.set_tensor(5, self.counts);
        enc.set_tensor(6, self.slot_ids);
        enc.set_tensor(7, self.schedule_generations);
        enc.set_tensor(8, self.aggregate);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    fn encode_signature(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route signature")?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_signature")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed route signature requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.aggregate);
        enc.set_tensor(2, self.expert_ids);
        enc.set_tensor(3, self.weights);
        enc.set_tensor(4, self.counts);
        enc.set_tensor(5, self.slot_ids);
        enc.set_tensor(6, self.signature);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_compact(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        bucket_rows: &MetalTensor,
        bucket_slots: &MetalTensor,
        tiles32: &MetalTensor,
        tiles16: &MetalTensor,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "packed GPU route compaction")?;
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed GPU compact routes")?;
        validate_i32(
            bucket_rows,
            &[route_count as u64],
            true,
            "packed GPU compact rows",
        )?;
        validate_i32(
            bucket_slots,
            &[route_count as u64],
            true,
            "packed GPU compact slots",
        )?;
        validate_i32(
            tiles32,
            &[PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64],
            true,
            "packed GPU compact 32-row tiles",
        )?;
        validate_i32(
            tiles16,
            &[PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64],
            true,
            "packed GPU compact 16-row tiles",
        )?;
        validate_i32(
            self.compact_header,
            &[PACKED_COMPACT_ROUTE_HEADER_WIDTH as u64],
            true,
            "packed GPU compact header",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_packed_route_compact")?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
            return invalid("packed route compaction requires SIMD width 32 and 256 threads");
        }
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &packed_route_schedule_args(n_tokens, generation)?);
        enc.set_tensor(1, self.route_generations);
        enc.set_tensor(2, self.route_status);
        enc.set_tensor(3, self.expert_ids);
        enc.set_tensor(4, self.weights);
        enc.set_tensor(5, self.counts);
        enc.set_tensor(6, bucket_rows);
        enc.set_tensor(7, bucket_slots);
        enc.set_tensor(8, tiles32);
        enc.set_tensor(9, tiles16);
        enc.set_tensor(10, self.compact_header);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

pub(super) struct DeepSeekV4PrefillScratch {
    token_ids: MetalTensor,
    embedding: MetalTensor,
    residual_primary: MetalTensor,
    residual_secondary: MetalTensor,
    hyper: PrefillHyperScratch,
    attention: PrefillAttentionScratch,
    compressor: PrefillCompressorScratch,
    moe: PrefillMoeScratch,
}

struct PrefillHyperScratch {
    ones: MetalTensor,
    normalized: MetalTensor,
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    combination: MetalTensor,
    collapsed: MetalTensor,
}

struct PrefillAttentionScratch {
    raw_cache_before_chunk: MetalTensor,
    raw_chunk: MetalTensor,
    normalized_input: MetalTensor,
    q_lora_raw: MetalTensor,
    q_lora: MetalTensor,
    queries_raw: MetalTensor,
    queries: MetalTensor,
    kv_raw: MetalTensor,
    kv: MetalTensor,
    attention: MetalTensor,
    low_rank: MetalTensor,
    output: MetalTensor,
    head_norm_ones: MetalTensor,
    group_input: MetalTensor,
    group_output: MetalTensor,
    sparse_csa: PrefillSparseCsaScratch,
}

struct PrefillSparseCsaScratch {
    capacity_rows: usize,
    index_queries: MetalTensor,
    head_weights: MetalTensor,
    visible_counts: MetalTensor,
    scores: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    selected_mask: MetalTensor,
    cache_order_ids: MetalTensor,
    selected_counts: MetalTensor,
    status: MetalTensor,
}

struct PrefillCompressorScratch {
    attention_kv: MetalTensor,
    attention_score: MetalTensor,
    indexer_kv: MetalTensor,
    indexer_score: MetalTensor,
    hca_kv: MetalTensor,
    hca_score: MetalTensor,
    pooled_rows: MetalTensor,
    normalized_rows: MetalTensor,
    #[cfg(feature = "dsv4-diagnostics")]
    q8_matrix_invocations: Cell<u32>,
}

struct PrefillMoeScratch {
    expert_count: usize,
    normalized_input: MetalTensor,
    logits: MetalTensor,
    hash_ids: MetalTensor,
    expert_ids: MetalTensor,
    weights: MetalTensor,
    bucket_rows: MetalTensor,
    bucket_slots: MetalTensor,
    expert_input: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    inner: MetalTensor,
    bucket_output: MetalTensor,
    expert_outputs: MetalTensor,
    routed_output: MetalTensor,
    shared_output: MetalTensor,
    final_output: MetalTensor,
    grouped_tiles: MetalTensor,
    grouped_iq2_mma16_tiles: MetalTensor,
    grouped_inner: MetalTensor,
    gpu_route: PrefillGpuRouteScratch,
    #[cfg(feature = "dsv4-diagnostics")]
    grouped_iq2_invocations: Cell<u32>,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    grouped_iq3_invocations: Cell<u32>,
}

struct PrefillGpuRouteScratch {
    route_generations: MetalTensor,
    route_status: MetalTensor,
    counts: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    slot_ids: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    schedule_generations: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    aggregate: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    signature: MetalTensor,
    compact_header: MetalTensor,
    next_generation: Cell<u32>,
}

impl DeepSeekV4PrefillScratch {
    pub(super) fn new(
        ctx: &MetalContext,
        csa_capacity_rows: usize,
        expert_count: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if csa_capacity_rows < DEEPSEEK_V4_CSA_TOP_K
            || !csa_capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "packed sparse CSA capacity {csa_capacity_rows} is not an aligned top-k superset"
            ));
        }
        validate_packed_expert_count(expert_count)?;
        let n = DEEPSEEK_V4_PREFILL_MAX_TOKENS as u64;
        let compressed_rows = DEEPSEEK_V4_PREFILL_MAX_TOKENS.div_ceil(4) as u64;
        let h = DEEPSEEK_V4_HIDDEN_SIZE as u64;
        let residual = residual_len(DEEPSEEK_V4_HIDDEN_SIZE)? as u64;
        let ones = vec![1.0f32; residual as usize];
        Ok(Self {
            token_ids: MetalTensor::zeros_dtype(ctx, vec![n], GgmlType::I32)?,
            embedding: MetalTensor::zeros_f32(ctx, vec![h, n])?,
            residual_primary: MetalTensor::zeros_f32(
                ctx,
                vec![h, DEEPSEEK_V4_CONNECTION_COUNT as u64, n],
            )?,
            residual_secondary: MetalTensor::zeros_f32(
                ctx,
                vec![h, DEEPSEEK_V4_CONNECTION_COUNT as u64, n],
            )?,
            hyper: PrefillHyperScratch {
                ones: MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&ones),
                    vec![residual],
                    GgmlType::F32,
                )?,
                normalized: MetalTensor::zeros_f32(ctx, vec![residual, n])?,
                mixes: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, n])?,
                pre: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n])?,
                post: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n])?,
                combination: MetalTensor::zeros_f32(
                    ctx,
                    vec![
                        DEEPSEEK_V4_CONNECTION_COUNT as u64,
                        DEEPSEEK_V4_CONNECTION_COUNT as u64,
                        n,
                    ],
                )?,
                collapsed: MetalTensor::zeros_f32(ctx, vec![h, n])?,
            },
            attention: PrefillAttentionScratch {
                raw_cache_before_chunk: MetalTensor::zeros_f16(
                    ctx,
                    vec![512, DEEPSEEK_V4_LOCAL_WINDOW as u64],
                )?,
                raw_chunk: MetalTensor::zeros_f16(ctx, vec![512, n])?,
                normalized_input: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                q_lora_raw: MetalTensor::zeros_f32(ctx, vec![1_024, n])?,
                q_lora: MetalTensor::zeros_f32(ctx, vec![1_024, n])?,
                queries_raw: MetalTensor::zeros_f32(ctx, vec![QUERY_WIDTH as u64, n])?,
                queries: MetalTensor::zeros_f32(ctx, vec![QUERY_WIDTH as u64, n])?,
                kv_raw: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                kv: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                attention: MetalTensor::zeros_f32(ctx, vec![QUERY_WIDTH as u64, n])?,
                low_rank: MetalTensor::zeros_f32(ctx, vec![LOW_RANK_WIDTH as u64, n])?,
                output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                head_norm_ones: MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&vec![1.0f32; 512]),
                    vec![512],
                    GgmlType::F32,
                )?,
                group_input: MetalTensor::zeros_f32(ctx, vec![GROUP_WIDTH as u64, n])?,
                group_output: MetalTensor::zeros_f32(ctx, vec![1_024, n])?,
                sparse_csa: PrefillSparseCsaScratch {
                    capacity_rows: csa_capacity_rows,
                    index_queries: MetalTensor::zeros_f32(
                        ctx,
                        vec![INDEXER_HEAD_DIM as u64, INDEXER_HEAD_COUNT as u64, n],
                    )?,
                    head_weights: MetalTensor::zeros_f32(ctx, vec![INDEXER_HEAD_COUNT as u64, n])?,
                    visible_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                    scores: MetalTensor::zeros_f32(ctx, vec![csa_capacity_rows as u64, n])?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    selected_mask: MetalTensor::zeros_i32(ctx, vec![csa_capacity_rows as u64, n])?,
                    cache_order_ids: MetalTensor::zeros_i32(
                        ctx,
                        vec![DEEPSEEK_V4_CSA_TOP_K as u64, n],
                    )?,
                    selected_counts: MetalTensor::zeros_i32(ctx, vec![n])?,
                    status: MetalTensor::zeros_i32(ctx, vec![n])?,
                },
            },
            compressor: PrefillCompressorScratch {
                attention_kv: MetalTensor::zeros_f32(
                    ctx,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n],
                )?,
                attention_score: MetalTensor::zeros_f32(
                    ctx,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n],
                )?,
                indexer_kv: MetalTensor::zeros_f32(ctx, vec![COMPRESSOR_INDEXER_WIDTH as u64, n])?,
                indexer_score: MetalTensor::zeros_f32(
                    ctx,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64, n],
                )?,
                hca_kv: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                hca_score: MetalTensor::zeros_f32(ctx, vec![512, n])?,
                pooled_rows: MetalTensor::zeros_f32(ctx, vec![512, compressed_rows])?,
                normalized_rows: MetalTensor::zeros_f32(ctx, vec![512, compressed_rows])?,
                #[cfg(feature = "dsv4-diagnostics")]
                q8_matrix_invocations: Cell::new(0),
            },
            moe: PrefillMoeScratch {
                expert_count,
                normalized_input: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                logits: MetalTensor::zeros_f32(ctx, vec![MOE_EXPERT_COUNT as u64, n])?,
                hash_ids: MetalTensor::zeros_dtype(ctx, vec![MOE_TOP_K as u64, n], GgmlType::I32)?,
                expert_ids: MetalTensor::zeros_dtype(
                    ctx,
                    vec![MOE_TOP_K as u64, n],
                    GgmlType::I32,
                )?,
                weights: MetalTensor::zeros_f32(ctx, vec![MOE_TOP_K as u64, n])?,
                bucket_rows: MetalTensor::zeros_dtype(
                    ctx,
                    vec![MOE_TOP_K as u64, n],
                    GgmlType::I32,
                )?,
                bucket_slots: MetalTensor::zeros_dtype(
                    ctx,
                    vec![MOE_TOP_K as u64, n],
                    GgmlType::I32,
                )?,
                expert_input: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                gate: MetalTensor::zeros_f32(ctx, vec![MOE_FFN_SIZE as u64, n])?,
                up: MetalTensor::zeros_f32(ctx, vec![MOE_FFN_SIZE as u64, n])?,
                inner: MetalTensor::zeros_f32(ctx, vec![MOE_FFN_SIZE as u64, n])?,
                bucket_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                expert_outputs: MetalTensor::zeros_f32(ctx, vec![h, MOE_TOP_K as u64, n])?,
                routed_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                shared_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                final_output: MetalTensor::zeros_f32(ctx, vec![h, n])?,
                grouped_tiles: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS as u64],
                )?,
                grouped_iq2_mma16_tiles: MetalTensor::zeros_i32(
                    ctx,
                    vec![PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS as u64],
                )?,
                grouped_inner: MetalTensor::zeros_f32(
                    ctx,
                    vec![
                        MOE_FFN_SIZE as u64,
                        MOE_TOP_K as u64,
                        PACKED_GROUPED_EXPERT_MAX_TOKENS as u64,
                    ],
                )?,
                gpu_route: PrefillGpuRouteScratch {
                    route_generations: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64],
                    )?,
                    route_status: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64],
                    )?,
                    counts: MetalTensor::zeros_i32(ctx, vec![MOE_EXPERT_COUNT as u64])?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    slot_ids: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_GPU_ROUTE_MAX_TOKENS as u64, MOE_EXPERT_COUNT as u64],
                    )?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    schedule_generations: MetalTensor::zeros_i32(
                        ctx,
                        vec![MOE_EXPERT_COUNT as u64],
                    )?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    aggregate: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_RECORD_WIDTH as u64])?,
                    #[cfg(any(test, feature = "dsv4-diagnostics"))]
                    signature: MetalTensor::zeros_i32(ctx, vec![PACKED_ROUTE_RECORD_WIDTH as u64])?,
                    compact_header: MetalTensor::zeros_i32(
                        ctx,
                        vec![PACKED_COMPACT_ROUTE_HEADER_WIDTH as u64],
                    )?,
                    next_generation: Cell::new(1),
                },
                #[cfg(feature = "dsv4-diagnostics")]
                grouped_iq2_invocations: Cell::new(0),
                #[cfg(all(test, feature = "dsv4-diagnostics"))]
                grouped_iq3_invocations: Cell::new(0),
            },
        })
    }
}

pub(super) fn append_session_allocation_requests(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
    csa_capacity_rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let n = DEEPSEEK_V4_PREFILL_MAX_TOKENS;
    let h = DEEPSEEK_V4_HIDDEN_SIZE;
    let residual = residual_len(h)?;
    let f32_bytes = std::mem::size_of::<f32>();
    let f16_bytes = std::mem::size_of::<u16>();
    let i32_bytes = std::mem::size_of::<i32>();
    let mut push = |name: &str, elements: usize, element_bytes: usize| {
        push_session_allocation(requests, format!("prefill.{name}"), elements, element_bytes)
    };

    push("token_ids", n, i32_bytes)?;
    push(
        "embedding",
        checked_mul(n, h, "prefill embedding")?,
        f32_bytes,
    )?;
    for name in ["residual_primary", "residual_secondary"] {
        push(
            name,
            checked_mul(n, residual, "prefill residual")?,
            f32_bytes,
        )?;
    }
    push("hyper.ones", residual, f32_bytes)?;
    push(
        "hyper.normalized",
        checked_mul(n, residual, "prefill hyper normalized")?,
        f32_bytes,
    )?;
    push(
        "hyper.mixes",
        checked_mul(n, DEEPSEEK_V4_HC_PARAMETER_COUNT, "prefill hyper mixes")?,
        f32_bytes,
    )?;
    for name in ["hyper.pre", "hyper.post"] {
        push(
            name,
            checked_mul(n, DEEPSEEK_V4_CONNECTION_COUNT, "prefill hyper gates")?,
            f32_bytes,
        )?;
    }
    push(
        "hyper.combination",
        checked_mul(
            n,
            DEEPSEEK_V4_CONNECTION_COUNT * DEEPSEEK_V4_CONNECTION_COUNT,
            "prefill hyper combinations",
        )?,
        f32_bytes,
    )?;
    push(
        "hyper.collapsed",
        checked_mul(n, h, "prefill hyper collapsed")?,
        f32_bytes,
    )?;

    for (name, width) in [
        ("attention.normalized_input", h),
        ("attention.q_lora_raw", 1_024),
        ("attention.q_lora", 1_024),
        ("attention.queries_raw", QUERY_WIDTH),
        ("attention.queries", QUERY_WIDTH),
        ("attention.kv_raw", 512),
        ("attention.kv", 512),
        ("attention.attention", QUERY_WIDTH),
        ("attention.low_rank", LOW_RANK_WIDTH),
        ("attention.output", h),
        ("attention.group_input", GROUP_WIDTH),
        ("attention.group_output", 1_024),
    ] {
        push(name, checked_mul(n, width, name)?, f32_bytes)?;
    }
    push(
        "attention.raw_cache_before_chunk",
        checked_mul(512, DEEPSEEK_V4_LOCAL_WINDOW, "prefill raw-ring snapshot")?,
        f16_bytes,
    )?;
    push(
        "attention.raw_chunk",
        checked_mul(512, n, "prefill raw chunk")?,
        f16_bytes,
    )?;
    push("attention.head_norm_ones", 512, f32_bytes)?;
    push(
        "attention.sparse_csa.index_queries",
        checked_mul(n, INDEXER_QUERY_WIDTH, "packed sparse index queries")?,
        f32_bytes,
    )?;
    push(
        "attention.sparse_csa.head_weights",
        checked_mul(n, INDEXER_HEAD_COUNT, "packed sparse head weights")?,
        f32_bytes,
    )?;
    push("attention.sparse_csa.visible_counts", n, i32_bytes)?;
    push(
        "attention.sparse_csa.scores",
        checked_mul(n, csa_capacity_rows, "packed sparse score scratch")?,
        f32_bytes,
    )?;
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    push(
        "attention.sparse_csa.selected_mask",
        checked_mul(n, csa_capacity_rows, "packed sparse selection mask")?,
        i32_bytes,
    )?;
    push(
        "attention.sparse_csa.cache_order_ids",
        checked_mul(n, DEEPSEEK_V4_CSA_TOP_K, "packed sparse selected IDs")?,
        i32_bytes,
    )?;
    for name in ["selected_counts", "status"] {
        push(&format!("attention.sparse_csa.{name}"), n, i32_bytes)?;
    }

    for name in ["compressor.attention_kv", "compressor.attention_score"] {
        push(
            name,
            checked_mul(n, COMPRESSOR_ATTENTION_WIDTH, name)?,
            f32_bytes,
        )?;
    }
    for name in ["compressor.indexer_kv", "compressor.indexer_score"] {
        push(
            name,
            checked_mul(n, COMPRESSOR_INDEXER_WIDTH, name)?,
            f32_bytes,
        )?;
    }
    for name in ["compressor.hca_kv", "compressor.hca_score"] {
        push(name, checked_mul(n, 512, name)?, f32_bytes)?;
    }
    let compressed_rows = n.div_ceil(4);
    for name in ["compressor.pooled_rows", "compressor.normalized_rows"] {
        push(name, checked_mul(compressed_rows, 512, name)?, f32_bytes)?;
    }

    for (name, width, element_bytes) in [
        ("moe.normalized_input", h, f32_bytes),
        ("moe.logits", MOE_EXPERT_COUNT, f32_bytes),
        ("moe.hash_ids", MOE_TOP_K, i32_bytes),
        ("moe.expert_ids", MOE_TOP_K, i32_bytes),
        ("moe.weights", MOE_TOP_K, f32_bytes),
        ("moe.bucket_rows", MOE_TOP_K, i32_bytes),
        ("moe.bucket_slots", MOE_TOP_K, i32_bytes),
        ("moe.expert_input", h, f32_bytes),
        ("moe.gate", MOE_FFN_SIZE, f32_bytes),
        ("moe.up", MOE_FFN_SIZE, f32_bytes),
        ("moe.inner", MOE_FFN_SIZE, f32_bytes),
        ("moe.bucket_output", h, f32_bytes),
        ("moe.expert_outputs", MOE_TOP_K * h, f32_bytes),
        ("moe.routed_output", h, f32_bytes),
        ("moe.shared_output", h, f32_bytes),
        ("moe.final_output", h, f32_bytes),
    ] {
        push(name, checked_mul(n, width, name)?, element_bytes)?;
    }
    push(
        "moe.grouped_tiles",
        PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS,
        i32_bytes,
    )?;
    push(
        "moe.grouped_iq2_mma16_tiles",
        PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS,
        i32_bytes,
    )?;
    push(
        "moe.grouped_inner",
        checked_mul(
            checked_mul(
                PACKED_GROUPED_EXPERT_MAX_TOKENS,
                MOE_TOP_K,
                "packed grouped slots",
            )?,
            MOE_FFN_SIZE,
            "packed grouped inner",
        )?,
        f32_bytes,
    )?;
    for (name, elements) in [
        (
            "moe.gpu_route.route_generations",
            PACKED_GPU_ROUTE_MAX_TOKENS,
        ),
        ("moe.gpu_route.route_status", PACKED_GPU_ROUTE_MAX_TOKENS),
        ("moe.gpu_route.counts", MOE_EXPERT_COUNT),
        (
            "moe.gpu_route.compact_header",
            PACKED_COMPACT_ROUTE_HEADER_WIDTH,
        ),
    ] {
        push(name, elements, i32_bytes)?;
    }
    #[cfg(feature = "dsv4-diagnostics")]
    for (name, elements) in [
        (
            "moe.gpu_route.slot_ids",
            checked_mul(
                PACKED_GPU_ROUTE_MAX_TOKENS,
                MOE_EXPERT_COUNT,
                "packed GPU route slots",
            )?,
        ),
        ("moe.gpu_route.schedule_generations", MOE_EXPERT_COUNT),
        ("moe.gpu_route.aggregate", PACKED_ROUTE_RECORD_WIDTH),
        ("moe.gpu_route.signature", PACKED_ROUTE_RECORD_WIDTH),
    ] {
        push(name, elements, i32_bytes)?;
    }
    Ok(())
}

fn checked_token_count(n_tokens: usize) -> Result<u32, DeepSeekV4MetalError> {
    if n_tokens == 0 || n_tokens > DEEPSEEK_V4_PREFILL_MAX_TOKENS {
        return invalid(format!(
            "DeepSeek V4 packed prefill requires 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS} tokens, got {n_tokens}"
        ));
    }
    u32::try_from(n_tokens)
        .map_err(|_| DeepSeekV4MetalError::Invalid("prefill token count exceeds u32".into()))
}

fn validate_packed_expert_count(expert_count: usize) -> Result<(), DeepSeekV4MetalError> {
    if expert_count == 0 || expert_count > MOE_EXPERT_COUNT {
        return invalid(format!(
            "packed MoE expert count {expert_count} is outside 1..={MOE_EXPERT_COUNT}"
        ));
    }
    Ok(())
}

fn f32_prefix(
    tensor: &MetalTensor,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let elements = shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} shape element count overflow"))
        })
    })?;
    if elements > tensor.n_elements() {
        return invalid(format!(
            "{name} prefix requires {elements} elements, backing has {}",
            tensor.n_elements()
        ));
    }
    let view = tensor.view_subrange(0, shape);
    validate_f32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn f16_prefix(
    tensor: &MetalTensor,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let elements = shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} shape element count overflow"))
        })
    })?;
    if elements > tensor.n_elements() {
        return invalid(format!(
            "{name} prefix requires {elements} elements, backing has {}",
            tensor.n_elements()
        ));
    }
    let view = tensor.view_subrange(0, shape);
    validate_f16(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn i32_prefix(
    tensor: &MetalTensor,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let elements = shape.iter().try_fold(1_u64, |total, &dimension| {
        total.checked_mul(dimension).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} shape element count overflow"))
        })
    })?;
    if elements > tensor.n_elements() {
        return invalid(format!(
            "{name} prefix requires {elements} elements, backing has {}",
            tensor.n_elements()
        ));
    }
    let view = tensor.view_subrange(0, shape);
    validate_i32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn f32_row(
    tensor: &MetalTensor,
    row: usize,
    width: usize,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let offset = checked_mul(row, width, &format!("{name} row offset"))?;
    let view = tensor.view_subrange(offset as u64, shape);
    validate_f32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn i32_slice(
    tensor: &MetalTensor,
    offset: usize,
    len: usize,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let view = tensor.view_subrange(offset as u64, vec![len as u64]);
    validate_i32(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn encode_batch_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    checked_token_count(n_tokens)?;
    validate_matvec_weight(weight, n_in, n_out, name)?;
    if weight.dtype == GgmlType::MXFP4 {
        return invalid(format!(
            "{name} requires MXFP4 matmat, which is not implemented"
        ));
    }
    if input.dtype != GgmlType::F32
        || output.dtype != GgmlType::F32
        || input.n_elements() != checked_mul(n_tokens, n_in, name)? as u64
        || output.n_elements() != checked_mul(n_tokens, n_out, name)? as u64
        || !output.is_writable()
    {
        return invalid(format!(
            "{name} packed projection requires F32 [{n_tokens},{n_in}] -> [{n_tokens},{n_out}]"
        ));
    }
    if weight.dtype == GgmlType::Q8_0 {
        return crate::metal::encode_mat_vec_q8_0_batch_f32(
            ctx, enc, weight, input, output, n_in, n_out, n_tokens,
        )
        .map_err(DeepSeekV4MetalError::Metal);
    }
    // DeepSeek's packed projections pin a bitwise matrix-tile lineage at
    // every N (the N=1 scalar-lineage gates); do not inherit the Qwen-tuned
    // N=1 mat-vec shortcut, which changed the accumulation order in
    // 940516a8 and turned those gates red.
    crate::metal_forward::encode_mat_mat_dispatch_with_policy(
        ctx, enc, weight, input, output, n_in, n_out, n_tokens, false,
    )
    .map_err(|error| match error {
        crate::metal_forward::MfError::Metal(error) => DeepSeekV4MetalError::Metal(error),
        other => DeepSeekV4MetalError::Invalid(format!("{name} packed projection failed: {other}")),
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_q8_f32_mma_r2c4k64(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed Q8 F32 R2C4K64 projection")?;
    checked_token_count(n_tokens)?;
    validate_matvec_weight(weight, n_in, n_out, "packed Q8 F32 R2C4K64 weight")?;
    validate_f32(
        input,
        &[n_in as u64, n_tokens as u64],
        false,
        "packed Q8 F32 R2C4K64 input",
    )?;
    validate_f32(
        output,
        &[n_out as u64, n_tokens as u64],
        true,
        "packed Q8 F32 R2C4K64 output",
    )?;
    if weight.dtype != GgmlType::Q8_0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !(1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&n_tokens)
    {
        return invalid("packed Q8 F32 R2C4K64 projection has invalid geometry or storage");
    }
    let padded_tokens = n_tokens.div_ceil(32) * 32;
    let padded_elements = checked_mul(padded_tokens, n_in, "packed Q8 F32 R2C4K64 padded input")?;
    let padded_bytes = checked_mul(
        padded_elements,
        std::mem::size_of::<f32>(),
        "packed Q8 F32 R2C4K64 padded input bytes",
    )?;
    let padded_end = input
        .offset
        .checked_add(u64::try_from(padded_bytes).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed Q8 F32 R2C4K64 padded input exceeds u64".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed Q8 F32 R2C4K64 padded input end overflow".into())
        })?;
    if padded_end > input.buffer.length() as u64 || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return invalid("packed Q8 F32 R2C4K64 requires padded input backing and 4 KiB TGM");
    }
    let output_end = output.offset.checked_add(output.n_bytes()).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("packed Q8 F32 R2C4K64 output end overflow".into())
    })?;
    let overlaps_padded_input = Retained::as_ptr(&input.buffer) == Retained::as_ptr(&output.buffer)
        && input.offset < output_end
        && output.offset < padded_end;
    if overlaps_padded_input || packed_grouped_tensor_ranges_overlap(weight, output) {
        return invalid("packed Q8 F32 R2C4K64 output overlaps an input");
    }
    let pso = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c4k64")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("packed Q8 F32 R2C4K64 requires one 32-thread SIMD group");
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let row_bytes = checked_mul(n_in / 32, 34, "packed Q8 F32 R2C4K64 row bytes")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 output exceeds u32".into())
            })?,
            n: u32::try_from(n_tokens).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 token count exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 stride exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: n_tokens.div_ceil(32),
            height: n_out / 16,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_q8_f32_mma_r2c16k64(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed Q8 F32 R2C16K64 projection")?;
    checked_token_count(n_tokens)?;
    validate_matvec_weight(weight, n_in, n_out, "packed Q8 F32 R2C16K64 weight")?;
    validate_f32(
        input,
        &[n_in as u64, n_tokens as u64],
        false,
        "packed Q8 F32 R2C16K64 input",
    )?;
    validate_f32(
        output,
        &[n_out as u64, n_tokens as u64],
        true,
        "packed Q8 F32 R2C16K64 output",
    )?;
    if weight.dtype != GgmlType::Q8_0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !n_tokens.is_multiple_of(128)
        || n_tokens > DEEPSEEK_V4_PREFILL_MAX_TOKENS
    {
        return invalid("packed Q8 F32 R2C16K64 projection has invalid geometry or storage");
    }
    if packed_grouped_tensor_ranges_overlap(input, output)
        || packed_grouped_tensor_ranges_overlap(weight, output)
    {
        return invalid("packed Q8 F32 R2C16K64 output overlaps an input");
    }
    let pso = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c16k64")?;
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < 128
        || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return invalid("packed Q8 F32 R2C16K64 requires four SIMDgroups and 4 KiB TGM");
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let row_bytes = checked_mul(n_in / 32, 34, "packed Q8 F32 R2C16K64 row bytes")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 output exceeds u32".into())
            })?,
            n: u32::try_from(n_tokens).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 token count exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed Q8 F32 stride exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: n_tokens / 128,
            height: n_out / 16,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_q8_f32_mma_r2c16k64_grouped(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    group_count: usize,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped Q8 F32 R2C16K64 projection")?;
    checked_token_count(n_tokens)?;
    let input_width = checked_mul(n_in, group_count, "grouped Q8 F32 input width")?;
    let output_width = checked_mul(n_out, group_count, "grouped Q8 F32 output width")?;
    validate_matvec_weight(
        weight,
        n_in,
        output_width,
        "packed grouped Q8 F32 R2C16K64 weight",
    )?;
    validate_f32(
        input,
        &[input_width as u64, n_tokens as u64],
        false,
        "packed grouped Q8 F32 R2C16K64 input",
    )?;
    validate_f32(
        output,
        &[output_width as u64, n_tokens as u64],
        true,
        "packed grouped Q8 F32 R2C16K64 output",
    )?;
    if weight.dtype != GgmlType::Q8_0
        || group_count == 0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !n_tokens.is_multiple_of(128)
        || n_tokens > DEEPSEEK_V4_PREFILL_MAX_TOKENS
    {
        return invalid(
            "packed grouped Q8 F32 R2C16K64 projection has invalid geometry or storage",
        );
    }
    if packed_grouped_tensor_ranges_overlap(input, output)
        || packed_grouped_tensor_ranges_overlap(weight, output)
    {
        return invalid("packed grouped Q8 F32 R2C16K64 output overlaps an input");
    }
    let pso = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c16k64_grouped")?;
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < 128
        || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return invalid("packed grouped Q8 F32 R2C16K64 requires four SIMDgroups and 4 KiB TGM");
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        groups: u32,
        nb01: u32,
        stride_b: u32,
        stride_c: u32,
    }
    let row_bytes = checked_mul(n_in / 32, 34, "grouped Q8 F32 R2C16K64 row bytes")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 output exceeds u32".into())
            })?,
            n: u32::try_from(n_tokens).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 token count exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 input exceeds u32".into())
            })?,
            groups: u32::try_from(group_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 group count exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(input_width).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 input stride exceeds u32".into())
            })?,
            stride_c: u32::try_from(output_width).map_err(|_| {
                DeepSeekV4MetalError::Invalid("grouped Q8 F32 output stride exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: n_tokens / 128,
            height: n_out / 16,
            depth: group_count,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_state_batch_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_tokens: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if weight.dtype != GgmlType::Q8_0 {
        return encode_batch_projection(
            ctx, enc, weight, input, output, n_in, n_out, n_tokens, name,
        );
    }
    validate_matvec_weight(weight, n_in, n_out, name)?;
    crate::metal::encode_mat_vec_q8_0_batch_f32(
        ctx, enc, weight, input, output, n_in, n_out, n_tokens,
    )
    .map_err(DeepSeekV4MetalError::Metal)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcBatchArgs {
    hidden_size: u32,
    n_tokens: u32,
}

impl PrefillHyperScratch {
    fn encode_initial_repeat(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        embeddings: &MetalTensor,
        residual: &MetalTensor,
        n_tokens: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_repeat_batch")?;
        let n_tokens_u32 = checked_token_count(n_tokens)?;
        validate_f32(
            embeddings,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed embeddings",
        )?;
        validate_f32(
            residual,
            &[
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            true,
            "packed initial residual",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_repeat_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcBatchArgs {
                hidden_size: u32_hidden(DEEPSEEK_V4_HIDDEN_SIZE)?,
                n_tokens: n_tokens_u32,
            },
        );
        enc.set_tensor(1, embeddings);
        enc.set_tensor(2, residual);
        let total = checked_mul(
            n_tokens,
            residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?,
            "packed repeated residual",
        )?;
        enc.dispatch(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        function: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
        n_tokens: usize,
        rms_eps: f32,
        hc_eps: f32,
        #[cfg(feature = "dsv4-diagnostics")] mhc_execution: Option<&mut PackedMhcExecution>,
        #[cfg(feature = "dsv4-diagnostics")] mhc_site: DeepSeekV4MhcSiteKind,
        #[cfg(feature = "dsv4-diagnostics")] layer: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_pre_batch")?;
        let n_tokens_u32 = checked_token_count(n_tokens)?;
        validate_eps(rms_eps, "packed mHC RMSNorm epsilon")?;
        validate_eps(hc_eps, "packed mHC epsilon")?;
        let residual_width = residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?;
        validate_f32(
            residual,
            &[
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            false,
            "packed mHC residual",
        )?;
        validate_matvec_weight(
            function,
            residual_width,
            DEEPSEEK_V4_HC_PARAMETER_COUNT,
            "packed mHC function",
        )?;
        validate_f32(scale, &[3], false, "packed mHC scale")?;
        validate_f32(
            base,
            &[DEEPSEEK_V4_HC_PARAMETER_COUNT as u64],
            false,
            "packed mHC base",
        )?;
        let normalized = f32_prefix(
            &self.normalized,
            vec![residual_width as u64, n_tokens as u64],
            "packed mHC normalized",
        )?;
        let mixes = f32_prefix(
            &self.mixes,
            vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64, n_tokens as u64],
            "packed mHC mixes",
        )?;
        let pre = f32_prefix(
            &self.pre,
            vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n_tokens as u64],
            "packed mHC pre gates",
        )?;
        let post = f32_prefix(
            &self.post,
            vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n_tokens as u64],
            "packed mHC post gates",
        )?;
        let combination = f32_prefix(
            &self.combination,
            vec![
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed mHC combinations",
        )?;
        let collapsed = f32_prefix(
            &self.collapsed,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed mHC collapsed",
        )?;
        #[cfg(feature = "dsv4-diagnostics")]
        let tag_mhc_dispatches = mhc_execution
            .as_ref()
            .map_or_else(crate::metal::dispatch_census_is_active, |execution| {
                execution.record_sites
            });
        #[cfg(feature = "dsv4-diagnostics")]
        let (producer_runs, producer_output, controls_input) =
            if let Some(execution) = mhc_execution {
                let plan = execution.site_plan(layer, mhc_site, &mixes)?;
                (
                    plan.producer_runs,
                    plan.producer_output,
                    plan.controls_input,
                )
            } else {
                (true, None, None)
            };
        #[cfg(not(feature = "dsv4-diagnostics"))]
        let producer_runs = true;
        if producer_runs {
            #[cfg(feature = "dsv4-diagnostics")]
            let rms_tag = tag_mhc_dispatches
                .then(|| {
                    crate::metal::dispatch_census_tag_scope(|| {
                        format!("dsv4_mhc:{layer}:{}:rms", mhc_site.label())
                    })
                })
                .flatten();
            encode_rms_norm_batched_f32(
                ctx,
                enc,
                residual,
                &self.ones,
                &normalized,
                n_tokens,
                residual_width,
                rms_eps,
            )?;
            #[cfg(feature = "dsv4-diagnostics")]
            drop(rms_tag);
            #[cfg(feature = "dsv4-diagnostics")]
            let function_tag = tag_mhc_dispatches
                .then(|| {
                    crate::metal::dispatch_census_tag_scope(|| {
                        format!("dsv4_mhc:{layer}:{}:function", mhc_site.label())
                    })
                })
                .flatten();
            encode_batch_projection(
                ctx,
                enc,
                function,
                &normalized,
                #[cfg(feature = "dsv4-diagnostics")]
                producer_output.unwrap_or(&mixes),
                #[cfg(not(feature = "dsv4-diagnostics"))]
                &mixes,
                residual_width,
                DEEPSEEK_V4_HC_PARAMETER_COUNT,
                n_tokens,
                "packed mHC function",
            )?;
            #[cfg(feature = "dsv4-diagnostics")]
            drop(function_tag);
        }
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ControlsArgs {
            n_tokens: u32,
            eps: f32,
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let controls_tag = tag_mhc_dispatches
            .then(|| {
                crate::metal::dispatch_census_tag_scope(|| {
                    format!("dsv4_mhc:{layer}:{}:controls", mhc_site.label())
                })
            })
            .flatten();
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_controls_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &ControlsArgs {
                n_tokens: n_tokens_u32,
                eps: hc_eps,
            },
        );
        #[cfg(feature = "dsv4-diagnostics")]
        enc.set_tensor(1, controls_input.unwrap_or(&mixes));
        #[cfg(not(feature = "dsv4-diagnostics"))]
        enc.set_tensor(1, &mixes);
        enc.set_tensor(2, scale);
        enc.set_tensor(3, base);
        enc.set_tensor(4, &pre);
        enc.set_tensor(5, &post);
        enc.set_tensor(6, &combination);
        enc.dispatch(
            MTLSize {
                width: n_tokens,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        #[cfg(feature = "dsv4-diagnostics")]
        drop(controls_tag);

        let pso = ctx.pipeline("kernel_deepseek_v4_hc_collapse_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcBatchArgs {
                hidden_size: u32_hidden(DEEPSEEK_V4_HIDDEN_SIZE)?,
                n_tokens: n_tokens_u32,
            },
        );
        enc.set_tensor(1, residual);
        enc.set_tensor(2, &pre);
        enc.set_tensor(3, &collapsed);
        let total = checked_mul(n_tokens, DEEPSEEK_V4_HIDDEN_SIZE, "packed mHC collapse")?;
        enc.dispatch(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(collapsed)
    }

    fn encode_post(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        block_output: &MetalTensor,
        residual: &MetalTensor,
        output: &MetalTensor,
        n_tokens: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_post_batch")?;
        let n_tokens_u32 = checked_token_count(n_tokens)?;
        validate_f32(
            block_output,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed mHC block output",
        )?;
        let residual_shape = [
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            DEEPSEEK_V4_CONNECTION_COUNT as u64,
            n_tokens as u64,
        ];
        validate_f32(
            residual,
            &residual_shape,
            false,
            "packed mHC source residual",
        )?;
        validate_f32(output, &residual_shape, true, "packed mHC output residual")?;
        let post = f32_prefix(
            &self.post,
            vec![DEEPSEEK_V4_CONNECTION_COUNT as u64, n_tokens as u64],
            "packed mHC post gates",
        )?;
        let combination = f32_prefix(
            &self.combination,
            vec![
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed mHC combinations",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_post_batch")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcBatchArgs {
                hidden_size: u32_hidden(DEEPSEEK_V4_HIDDEN_SIZE)?,
                n_tokens: n_tokens_u32,
            },
        );
        enc.set_tensor(1, block_output);
        enc.set_tensor(2, residual);
        enc.set_tensor(3, &post);
        enc.set_tensor(4, &combination);
        enc.set_tensor(5, output);
        let total = checked_mul(
            n_tokens,
            residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?,
            "packed mHC post",
        )?;
        enc.dispatch(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

struct PackedAttentionViews {
    normalized_input: MetalTensor,
    q_lora: MetalTensor,
    queries: MetalTensor,
    kv: MetalTensor,
    attention: MetalTensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Q8PrecisionProjection {
    Exact,
    #[cfg(test)]
    HalfMatrix,
    F32Matrix,
    WideF32Matrix,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedQ8MatrixPolicy {
    Auto,
    Exact,
    F32Matrix,
    WideF32Matrix,
}

crate::env_flag!(
    default_on packed_q8_compressor_matrix_enabled,
    "QWEN_DSV4_PACKED_Q8_COMPRESSOR_MATRIX"
);
crate::env_flag!(
    default_on packed_q8_shared_matrix_enabled,
    "QWEN_DSV4_PACKED_Q8_SHARED_MATRIX"
);
fn parse_packed_router_e8p32_strict_env(value: Option<&str>) -> Result<bool, ()> {
    let Some(value) = value else {
        return Ok(true);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(()),
    }
}

fn packed_router_e8p32_strict_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("QWEN_DSV4_PACKED_ROUTER_E8P32_STRICT") {
        None => true,
        Some(value) => {
            let parsed = value
                .to_str()
                .ok_or(())
                .and_then(|value| parse_packed_router_e8p32_strict_env(Some(value)));
            match parsed {
                Ok(enabled) => enabled,
                Err(()) => {
                    eprintln!(
                        "deepseek_v4: invalid QWEN_DSV4_PACKED_ROUTER_E8P32_STRICT value; disabling exact E8 router"
                    );
                    false
                }
            }
        }
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedQaKvMatrixMode {
    Off,
    Qa,
    Kv,
    Both,
}

impl PackedQaKvMatrixMode {
    #[cfg(feature = "dsv4-diagnostics")]
    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Qa => "qa",
            Self::Kv => "kv",
            Self::Both => "both",
        }
    }

    fn qa(self) -> bool {
        matches!(self, Self::Qa | Self::Both)
    }

    fn kv(self) -> bool {
        matches!(self, Self::Kv | Self::Both)
    }
}

fn parse_packed_q8_qa_kv_matrix_mode(value: Option<&str>) -> Result<PackedQaKvMatrixMode, ()> {
    let Some(value) = value else {
        return Ok(PackedQaKvMatrixMode::Both);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "false" | "no" | "off" => Ok(PackedQaKvMatrixMode::Off),
        "qa" => Ok(PackedQaKvMatrixMode::Qa),
        "kv" => Ok(PackedQaKvMatrixMode::Kv),
        "1" | "true" | "yes" | "on" | "both" => Ok(PackedQaKvMatrixMode::Both),
        _ => Err(()),
    }
}

fn packed_q8_qa_kv_matrix_mode() -> PackedQaKvMatrixMode {
    static MODE: std::sync::OnceLock<PackedQaKvMatrixMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var_os("QWEN_DSV4_PACKED_Q8_QA_KV_MATRIX");
        let parsed = match value.as_deref() {
            None => parse_packed_q8_qa_kv_matrix_mode(None),
            Some(value) => value
                .to_str()
                .ok_or(())
                .and_then(|value| parse_packed_q8_qa_kv_matrix_mode(Some(value))),
        };
        match parsed {
            Ok(mode) => mode,
            Err(()) => {
                eprintln!(
                    "deepseek_v4: invalid QWEN_DSV4_PACKED_Q8_QA_KV_MATRIX value; disabling Q-A/raw-KV matrix policy"
                );
                PackedQaKvMatrixMode::Off
            }
        }
    })
}

const PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES: u64 = 104_202_502_492;
const PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES: u64 = 89_920_886_108;
const PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES: u64 = 89_060_075_612;

fn packed_q8_matrix_asset_qualified(source_bytes: u64, expert_count: usize) -> bool {
    matches!(
        (source_bytes, expert_count),
        (PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES, 256)
            | (PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES, 160)
            | (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216)
    )
}

fn packed_q8_matrix_chunk_qualified(n_tokens: usize) -> bool {
    matches!(
        n_tokens,
        PACKED_MATRIX_MIN_TOKENS | DEEPSEEK_V4_PREFILL_MAX_TOKENS
    )
}

fn packed_q8_matrix_execution_chunk_qualified(n_tokens: usize) -> bool {
    (1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&n_tokens)
}

fn packed_q8_partial_matrix_chunk_qualified(n_tokens: usize) -> bool {
    (256..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&n_tokens)
}

fn packed_router_e8p32_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    device_name == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE
        && tensor_count == 1_328
        && source_bytes == PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES
        && expert_count == 160
        && packed_q8_partial_matrix_chunk_qualified(n_tokens)
}

fn packed_q8_qa_kv_matrix_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    device_name == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE
        && tensor_count == 1_328
        && source_bytes == PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES
        && expert_count == 160
        && packed_q8_partial_matrix_chunk_qualified(n_tokens)
}

fn packed_q8_compressor_matrix_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    let chunk_qualified = if matches!(
        (source_bytes, expert_count),
        (PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES, 256)
            | (PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES, 160)
            | (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216)
    ) {
        packed_q8_partial_matrix_chunk_qualified(n_tokens)
    } else {
        packed_q8_matrix_chunk_qualified(n_tokens)
    };
    chunk_qualified
        && device_name == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE
        && tensor_count == 1_328
        && packed_q8_matrix_asset_qualified(source_bytes, expert_count)
}

fn packed_gpu_route_compact_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    packed_q8_matrix_chunk_qualified(n_tokens)
        && packed_q8_compressor_matrix_scope_qualified(
            device_name,
            tensor_count,
            source_bytes,
            expert_count,
            n_tokens,
        )
}

fn packed_indexer_q_matrix_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    n_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS
        && packed_q8_compressor_matrix_scope_qualified(
            device_name,
            tensor_count,
            source_bytes,
            expert_count,
            n_tokens,
        )
}

fn packed_q8_compressor_matrix_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> bool {
    packed_q8_compressor_matrix_enabled()
        && packed_q8_compressor_matrix_scope_qualified(
            &ctx.device.name().to_string(),
            residency.report().tensor_count,
            residency.report().source_bytes,
            residency.config().expert_count as usize,
            n_tokens,
        )
}

fn packed_q8_shared_matrix_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> bool {
    packed_q8_shared_matrix_enabled()
        && packed_q8_compressor_matrix_scope_qualified(
            &ctx.device.name().to_string(),
            residency.report().tensor_count,
            residency.report().source_bytes,
            residency.config().expert_count as usize,
            n_tokens,
        )
}

fn parse_packed_q8_qb_policy(
    value: Option<&str>,
) -> Result<PackedQ8MatrixPolicy, DeepSeekV4MetalError> {
    match value {
        None | Some("auto") => Ok(PackedQ8MatrixPolicy::Auto),
        Some("exact") => Ok(PackedQ8MatrixPolicy::Exact),
        Some("f32_matrix") => Ok(PackedQ8MatrixPolicy::F32Matrix),
        Some("wide_f32_matrix") => Ok(PackedQ8MatrixPolicy::WideF32Matrix),
        Some(value) => invalid(format!(
            "QWEN_DSV4_PACKED_Q8_QB must be auto, exact, f32_matrix, or wide_f32_matrix, got {value:?}"
        )),
    }
}

fn parse_packed_q8_output_policy(
    value: Option<&str>,
) -> Result<PackedQ8MatrixPolicy, DeepSeekV4MetalError> {
    match value {
        None | Some("auto") => Ok(PackedQ8MatrixPolicy::Auto),
        Some("exact") => Ok(PackedQ8MatrixPolicy::Exact),
        Some("f32_matrix") => Ok(PackedQ8MatrixPolicy::F32Matrix),
        Some("wide_f32_matrix") => Ok(PackedQ8MatrixPolicy::WideF32Matrix),
        Some(value) => invalid(format!(
            "QWEN_DSV4_PACKED_Q8_OUTPUT must be auto, exact, f32_matrix, or wide_f32_matrix, got {value:?}"
        )),
    }
}

fn packed_q8_output_projection_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> Result<Q8PrecisionProjection, DeepSeekV4MetalError> {
    let value = std::env::var("QWEN_DSV4_PACKED_Q8_OUTPUT").ok();
    let policy = parse_packed_q8_output_policy(value.as_deref())?;
    let profile_qualified = packed_q8_compressor_matrix_scope_qualified(
        &ctx.device.name().to_string(),
        residency.report().tensor_count,
        residency.report().source_bytes,
        residency.config().expert_count as usize,
        n_tokens,
    );
    Ok(resolve_packed_q8_matrix_policy(
        policy,
        profile_qualified,
        n_tokens,
    ))
}

fn resolve_packed_q8_matrix_policy(
    policy: PackedQ8MatrixPolicy,
    profile_qualified: bool,
    n_tokens: usize,
) -> Q8PrecisionProjection {
    match policy {
        PackedQ8MatrixPolicy::Auto if profile_qualified && n_tokens.is_multiple_of(128) => {
            Q8PrecisionProjection::WideF32Matrix
        }
        PackedQ8MatrixPolicy::Auto if profile_qualified => Q8PrecisionProjection::F32Matrix,
        PackedQ8MatrixPolicy::Auto | PackedQ8MatrixPolicy::Exact => Q8PrecisionProjection::Exact,
        PackedQ8MatrixPolicy::F32Matrix => Q8PrecisionProjection::F32Matrix,
        PackedQ8MatrixPolicy::WideF32Matrix if n_tokens.is_multiple_of(128) => {
            Q8PrecisionProjection::WideF32Matrix
        }
        PackedQ8MatrixPolicy::WideF32Matrix => Q8PrecisionProjection::F32Matrix,
    }
}

fn packed_q8_qb_projection_for_chunk(
    ctx: &MetalContext,
    residency: &DeepSeekV4MetalResidency,
    n_tokens: usize,
) -> Result<Q8PrecisionProjection, DeepSeekV4MetalError> {
    let value = std::env::var("QWEN_DSV4_PACKED_Q8_QB").ok();
    let policy = parse_packed_q8_qb_policy(value.as_deref())?;
    let profile_qualified = packed_q8_compressor_matrix_scope_qualified(
        &ctx.device.name().to_string(),
        residency.report().tensor_count,
        residency.report().source_bytes,
        residency.config().expert_count as usize,
        n_tokens,
    );
    Ok(resolve_packed_q8_matrix_policy(
        policy,
        profile_qualified,
        n_tokens,
    ))
}

impl Q8PrecisionProjection {
    fn uses_full_chunk_f32(self, n_tokens: usize) -> bool {
        match self {
            Self::F32Matrix => packed_q8_matrix_execution_chunk_qualified(n_tokens),
            Self::WideF32Matrix => {
                packed_q8_matrix_execution_chunk_qualified(n_tokens) && n_tokens.is_multiple_of(128)
            }
            _ => false,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            #[cfg(test)]
            Self::HalfMatrix => "half_matrix",
            Self::F32Matrix => "f32_matrix",
            Self::WideF32Matrix => "wide_f32_matrix",
        }
    }
}

#[derive(Clone)]
struct PackedSparseCsaViews {
    query_offset: usize,
    query_count: usize,
    cache_order_ids: MetalTensor,
    selected_counts: MetalTensor,
    visible_counts: MetalTensor,
    index_queries: MetalTensor,
    head_weights: MetalTensor,
    scores: MetalTensor,
    #[cfg(any(test, feature = "dsv4-diagnostics"))]
    selected_mask: MetalTensor,
    status: MetalTensor,
}

#[derive(Clone, Copy)]
struct PackedCsaSelectionView<'a> {
    query_offset: usize,
    query_count: usize,
    cache_order_ids: &'a MetalTensor,
    selected_counts: &'a MetalTensor,
    visible_counts: &'a MetalTensor,
}

impl PackedSparseCsaViews {
    fn selection_view(&self) -> PackedCsaSelectionView<'_> {
        PackedCsaSelectionView {
            query_offset: self.query_offset,
            query_count: self.query_count,
            cache_order_ids: &self.cache_order_ids,
            selected_counts: &self.selected_counts,
            visible_counts: &self.visible_counts,
        }
    }
}

fn csa_visible_rows(position: u32) -> usize {
    (u64::from(position) + 1) as usize / 4
}

fn sparse_csa_query_offset(start_position: u32, n_tokens: usize) -> Option<usize> {
    (0..n_tokens).find(|&token| {
        let position = u64::from(start_position) + token as u64;
        (position + 1) / 4 > DEEPSEEK_V4_CSA_TOP_K as u64
    })
}

#[cfg(feature = "dsv4-diagnostics")]
fn validate_fp4_selection_counterfactual_packed(
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let sparse_query_count = sparse_csa_query_offset(start_position, n_tokens)
        .map(|offset| n_tokens - offset)
        .unwrap_or(0);
    if sparse_query_count > 1 {
        return invalid(format!(
            "FP4 selection counterfactual requires at most one packed sparse query, got {sparse_query_count}"
        ));
    }
    Ok(())
}

fn packed_sparse_visible_counts(
    start_position: u32,
    query_offset: usize,
    n_tokens: usize,
    final_rows: usize,
) -> Result<Vec<i32>, DeepSeekV4MetalError> {
    if query_offset >= n_tokens || final_rows <= DEEPSEEK_V4_CSA_TOP_K {
        return invalid(format!(
            "packed sparse visibility geometry is invalid: offset={query_offset} tokens={n_tokens} final_rows={final_rows}"
        ));
    }
    let final_rows_i32 = i32::try_from(final_rows).map_err(|_| {
        DeepSeekV4MetalError::Invalid("packed sparse final row count exceeds i32".into())
    })?;
    let visible = (query_offset..n_tokens)
        .map(|token| {
            let position = start_position
                .checked_add(u32::try_from(token).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(
                        "packed sparse token offset exceeds u32".into(),
                    )
                })?)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("packed sparse position overflow".into())
                })?;
            let count = csa_visible_rows(position);
            if count <= DEEPSEEK_V4_CSA_TOP_K || count > final_rows {
                return invalid(format!(
                    "packed sparse token {token} sees {count} rows outside 513..={final_rows} final rows"
                ));
            }
            i32::try_from(count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed sparse visible count exceeds i32".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if visible.last().copied() != Some(final_rows_i32) {
        return invalid(format!(
            "packed sparse final visibility {:?} differs from published row count {final_rows}",
            visible.last()
        ));
    }
    Ok(visible)
}

fn tiled_hca_query_offset(start_position: u32, n_tokens: usize) -> Option<usize> {
    (0..n_tokens).find(|&token| {
        let position = u64::from(start_position) + token as u64;
        (position + 1) / 128 > DEEPSEEK_V4_HCA_TILE_ROWS as u64
    })
}

impl PrefillSparseCsaScratch {
    #[cfg(not(feature = "dsv4-diagnostics"))]
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        start_position: u32,
        query_offset: usize,
        n_tokens: usize,
        indexer_q_matrix: bool,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<PackedSparseCsaViews, DeepSeekV4MetalError> {
        let prepared = self.encode_prepare(
            ctx,
            enc,
            q_lora,
            normalized_input,
            indexer_q_weight,
            indexer_projection,
            rows,
            start_position,
            query_offset,
            n_tokens,
            indexer_q_matrix,
            rope,
        )?;
        self.encode_f16_score_and_select(ctx, enc, rows, &prepared)?;
        Ok(prepared)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_prepare(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        start_position: u32,
        query_offset: usize,
        n_tokens: usize,
        indexer_q_matrix: bool,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<PackedSparseCsaViews, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_packed_sparse_csa_indexer")?;
        checked_token_count(n_tokens)?;
        if query_offset >= n_tokens
            || rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != self.capacity_rows
        {
            return invalid(format!(
                "packed sparse CSA geometry is invalid: offset={query_offset} tokens={n_tokens} rows={}/{}",
                rows.count, rows.capacity_rows
            ));
        }
        validate_f32(
            q_lora,
            &[1_024, n_tokens as u64],
            false,
            "packed sparse CSA Q-LoRA input",
        )?;
        validate_f32(
            normalized_input,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed sparse CSA normalized input",
        )?;
        validate_matvec_weight(
            indexer_q_weight,
            1_024,
            INDEXER_QUERY_WIDTH,
            "packed indexer Q weight",
        )?;
        validate_matvec_weight(
            indexer_projection,
            DEEPSEEK_V4_HIDDEN_SIZE,
            INDEXER_HEAD_COUNT,
            "packed indexer projection weight",
        )?;
        validate_f16(
            rows.indexer_cache,
            &[INDEXER_HEAD_DIM as u64, rows.capacity_rows as u64],
            false,
            "packed sparse CSA indexer cache",
        )?;

        let query_count = n_tokens - query_offset;
        let normalized_suffix = normalized_input.view_subrange(
            checked_mul(
                query_offset,
                DEEPSEEK_V4_HIDDEN_SIZE,
                "packed sparse normalized-input offset",
            )? as u64,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, query_count as u64],
        );
        let index_query_storage = f32_prefix(
            &self.index_queries,
            vec![
                INDEXER_HEAD_DIM as u64,
                INDEXER_HEAD_COUNT as u64,
                n_tokens as u64,
            ],
            "packed sparse index query storage",
        )?;
        let head_weights = f32_prefix(
            &self.head_weights,
            vec![INDEXER_HEAD_COUNT as u64, query_count as u64],
            "packed sparse head weights",
        )?;
        let visible_counts = i32_prefix(
            &self.visible_counts,
            vec![query_count as u64],
            "packed sparse visible counts",
        )?;
        let scores = f32_prefix(
            &self.scores,
            vec![rows.capacity_rows as u64, query_count as u64],
            "packed sparse scores",
        )?;
        #[cfg(any(test, feature = "dsv4-diagnostics"))]
        let selected_mask = i32_prefix(
            &self.selected_mask,
            vec![rows.capacity_rows as u64, query_count as u64],
            "packed sparse selection mask",
        )?;
        let cache_order_ids = i32_prefix(
            &self.cache_order_ids,
            vec![DEEPSEEK_V4_CSA_TOP_K as u64, query_count as u64],
            "packed sparse cache-order IDs",
        )?;
        let selected_counts = i32_prefix(
            &self.selected_counts,
            vec![query_count as u64],
            "packed sparse selected counts",
        )?;
        let status = i32_prefix(
            &self.status,
            vec![query_count as u64],
            "packed sparse selection status",
        )?;

        let visible =
            packed_sparse_visible_counts(start_position, query_offset, n_tokens, rows.count)?;
        host_write_i32(
            &visible_counts,
            &visible,
            "packed sparse CSA visible counts",
        )?;

        let index_queries = if indexer_q_matrix {
            let matrix_output = index_query_storage
                .view_subrange(0, vec![INDEXER_QUERY_WIDTH as u64, n_tokens as u64]);
            encode_q8_f32_mma_r2c16k64(
                ctx,
                enc,
                indexer_q_weight,
                q_lora,
                &matrix_output,
                1_024,
                INDEXER_QUERY_WIDTH,
                n_tokens,
            )?;
            index_query_storage.view_subrange(
                checked_mul(
                    query_offset,
                    INDEXER_QUERY_WIDTH,
                    "packed sparse matrix-query offset",
                )? as u64,
                vec![
                    INDEXER_HEAD_DIM as u64,
                    INDEXER_HEAD_COUNT as u64,
                    query_count as u64,
                ],
            )
        } else {
            let q_lora_suffix = q_lora.view_subrange(
                checked_mul(query_offset, 1_024, "packed sparse Q-LoRA offset")? as u64,
                vec![1_024, query_count as u64],
            );
            let index_queries = index_query_storage.view_subrange(
                0,
                vec![
                    INDEXER_HEAD_DIM as u64,
                    INDEXER_HEAD_COUNT as u64,
                    query_count as u64,
                ],
            );
            encode_batch_projection(
                ctx,
                enc,
                indexer_q_weight,
                &q_lora_suffix,
                &index_queries,
                1_024,
                INDEXER_QUERY_WIDTH,
                query_count,
                "packed indexer Q",
            )?;
            index_queries
        };
        let batched_indexer_rope = packed_indexer_batched_rope_enabled();
        static POLICY_LOGGED: std::sync::Once = std::sync::Once::new();
        POLICY_LOGGED.call_once(|| {
            eprintln!(
                "deepseek_v4: packed indexer query RoPE policy={}; rollback=QWEN_DSV4_PACKED_INDEXER_BATCHED_ROPE=0",
                if batched_indexer_rope {
                    "batched"
                } else {
                    "scalar"
                },
            );
        });
        if batched_indexer_rope {
            let first_position = start_position
                .checked_add(u32::try_from(query_offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed sparse query offset exceeds u32".into())
                })?)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed sparse query start position overflow".into(),
                    )
                })?;
            encode_ds4_rope_tail_adjacent_batch_in_place(
                ctx,
                enc,
                &index_queries,
                first_position,
                query_count,
                1,
                rope,
                false,
            )?;
        } else {
            for local in 0..query_count {
                let query = f32_row(
                    &index_queries,
                    local,
                    INDEXER_QUERY_WIDTH,
                    vec![INDEXER_HEAD_DIM as u64, INDEXER_HEAD_COUNT as u64],
                    "packed sparse index query row",
                )?;
                let token = query_offset + local;
                let position = start_position
                    .checked_add(u32::try_from(token).map_err(|_| {
                        DeepSeekV4MetalError::Invalid(
                            "packed sparse token offset exceeds u32".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid("packed sparse position overflow".into())
                    })?;
                encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &query, position, rope, false)?;
            }
        }
        encode_hadamard_128_rows_in_place(
            ctx,
            enc,
            &index_queries,
            checked_mul(
                query_count,
                INDEXER_HEAD_COUNT,
                "packed indexer Hadamard rows",
            )?,
        )?;
        encode_batch_projection(
            ctx,
            enc,
            indexer_projection,
            &normalized_suffix,
            &head_weights,
            DEEPSEEK_V4_HIDDEN_SIZE,
            INDEXER_HEAD_COUNT,
            query_count,
            "packed indexer head weights",
        )?;
        encode_scale_f32_in_place(
            ctx,
            enc,
            &head_weights,
            1.0 / (INDEXER_HEAD_COUNT as f32 * INDEXER_HEAD_DIM as f32).sqrt(),
            "packed indexer head weights",
        )?;
        Ok(PackedSparseCsaViews {
            query_offset,
            query_count,
            cache_order_ids,
            selected_counts,
            visible_counts,
            index_queries,
            head_weights,
            scores,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            selected_mask,
            status,
        })
    }

    #[cfg(not(feature = "dsv4-diagnostics"))]
    fn encode_f16_score_and_select(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        prepared: &PackedSparseCsaViews,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.encode_f16_scores(ctx, enc, rows, prepared)?;
        self.encode_f16_selection(ctx, enc, rows, prepared)
    }

    fn encode_f16_scores(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        prepared: &PackedSparseCsaViews,
    ) -> Result<(), DeepSeekV4MetalError> {
        let bounded = packed_indexer_visible_dispatch_enabled();
        let tiled_f32 = packed_indexer_tiled_f32_enabled();
        if bounded {
            static POLICY_LOGGED: std::sync::Once = std::sync::Once::new();
            POLICY_LOGGED.call_once(|| {
                eprintln!(
                    "deepseek_v4: packed indexer score dispatch is visibility-bounded; rollback=QWEN_DSV4_PACKED_INDEXER_VISIBLE_DISPATCH=0"
                );
            });
        }
        if tiled_f32 {
            static TILED_POLICY_LOGGED: std::sync::Once = std::sync::Once::new();
            TILED_POLICY_LOGGED.call_once(|| {
                eprintln!(
                    "deepseek_v4: packed indexer score policy=tiled_f32; rollback=QWEN_DSV4_PACKED_INDEXER_TILED_F32=0"
                );
            });
        }
        let encode = if tiled_f32 {
            encode_lightning_indexer_scores_f16_tiled_f32_with_limit
        } else {
            encode_lightning_indexer_scores_f16_with_limit
        };
        encode(
            ctx,
            enc,
            &prepared.index_queries,
            &prepared.head_weights,
            rows.indexer_cache,
            &prepared.visible_counts,
            &prepared.scores,
            INDEXER_HEAD_COUNT,
            INDEXER_HEAD_DIM,
            rows.capacity_rows,
            if bounded {
                rows.count
            } else {
                rows.capacity_rows
            },
            prepared.query_count,
        )
    }

    fn encode_f16_selection(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        prepared: &PackedSparseCsaViews,
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(any(test, feature = "dsv4-diagnostics"))]
        encode_select_top_k_f32(
            ctx,
            enc,
            &prepared.scores,
            &prepared.visible_counts,
            &prepared.selected_mask,
            None,
            &prepared.cache_order_ids,
            &prepared.selected_counts,
            &prepared.status,
            rows.capacity_rows,
            rows.count,
            DEEPSEEK_V4_CSA_TOP_K,
            prepared.query_count,
        )?;
        #[cfg(not(any(test, feature = "dsv4-diagnostics")))]
        encode_select_top_k_radix4_ids_f32(
            ctx,
            enc,
            &prepared.scores,
            &prepared.visible_counts,
            &prepared.cache_order_ids,
            &prepared.selected_counts,
            &prepared.status,
            rows.capacity_rows,
            rows.count,
            DEEPSEEK_V4_CSA_TOP_K,
            prepared.query_count,
        )?;
        Ok(())
    }

    fn validate_completed(&self, query_count: usize) -> Result<(), DeepSeekV4MetalError> {
        checked_token_count(query_count)?;
        let status = i32_prefix(
            &self.status,
            vec![query_count as u64],
            "packed sparse selection status",
        )?;
        let selected_counts = i32_prefix(
            &self.selected_counts,
            vec![query_count as u64],
            "packed sparse selected counts",
        )?;
        let statuses = host_read_i32(&status, "packed sparse selection status")?;
        let counts = host_read_i32(&selected_counts, "packed sparse selected counts")?;
        if let Some(query) = statuses
            .iter()
            .zip(&counts)
            .position(|(&status, &count)| status != 0 || count != DEEPSEEK_V4_CSA_TOP_K as i32)
        {
            return invalid(format!(
                "packed sparse CSA query {query} selection failed with status={} count={}",
                statuses[query], counts[query]
            ));
        }
        Ok(())
    }
}

impl PrefillAttentionScratch {
    #[allow(clippy::too_many_arguments)]
    fn encode_prepare(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        attention_norm: &MetalTensor,
        q_a: &MetalTensor,
        q_a_norm: &MetalTensor,
        q_b: &MetalTensor,
        kv_weight: &MetalTensor,
        kv_norm: &MetalTensor,
        n_tokens: usize,
        rms_eps: f32,
        q_b_projection: Q8PrecisionProjection,
        q_a_kv_matrix: PackedQaKvMatrixMode,
    ) -> Result<PackedAttentionViews, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_prepare_batch")?;
        checked_token_count(n_tokens)?;
        validate_eps(rms_eps, "packed attention RMSNorm epsilon")?;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked()?;
        validate_f32(
            input,
            &[config.hidden_size as u64, n_tokens as u64],
            false,
            "packed attention input",
        )?;
        validate_f32(
            attention_norm,
            &[config.hidden_size as u64],
            false,
            "packed attention norm weight",
        )?;
        validate_f32(
            q_a_norm,
            &[config.q_lora_rank as u64],
            false,
            "packed Q A norm weight",
        )?;
        validate_f32(
            kv_norm,
            &[config.head_dim as u64],
            false,
            "packed KV norm weight",
        )?;
        let normalized_input = f32_prefix(
            &self.normalized_input,
            vec![config.hidden_size as u64, n_tokens as u64],
            "packed attention normalized input",
        )?;
        let q_lora_raw = f32_prefix(
            &self.q_lora_raw,
            vec![config.q_lora_rank as u64, n_tokens as u64],
            "packed raw Q LoRA",
        )?;
        let q_lora = f32_prefix(
            &self.q_lora,
            vec![config.q_lora_rank as u64, n_tokens as u64],
            "packed Q LoRA",
        )?;
        let queries_raw = f32_prefix(
            &self.queries_raw,
            vec![dims.query_width as u64, n_tokens as u64],
            "packed raw queries",
        )?;
        let queries = f32_prefix(
            &self.queries,
            vec![dims.query_width as u64, n_tokens as u64],
            "packed queries",
        )?;
        let kv_raw = f32_prefix(
            &self.kv_raw,
            vec![config.head_dim as u64, n_tokens as u64],
            "packed raw KV",
        )?;
        let kv = f32_prefix(
            &self.kv,
            vec![config.head_dim as u64, n_tokens as u64],
            "packed KV",
        )?;
        let attention = f32_prefix(
            &self.attention,
            vec![dims.query_width as u64, n_tokens as u64],
            "packed attention heads",
        )?;

        encode_rms_norm_batched_f32(
            ctx,
            enc,
            input,
            attention_norm,
            &normalized_input,
            n_tokens,
            config.hidden_size,
            rms_eps,
        )?;
        if q_a_kv_matrix.qa() {
            if n_tokens.is_multiple_of(128) {
                encode_q8_f32_mma_r2c16k64(
                    ctx,
                    enc,
                    q_a,
                    &normalized_input,
                    &q_lora_raw,
                    config.hidden_size,
                    config.q_lora_rank,
                    n_tokens,
                )?;
            } else {
                encode_q8_f32_mma_r2c4k64(
                    ctx,
                    enc,
                    q_a,
                    &normalized_input,
                    &q_lora_raw,
                    config.hidden_size,
                    config.q_lora_rank,
                    n_tokens,
                )?;
            }
        } else {
            encode_batch_projection(
                ctx,
                enc,
                q_a,
                &normalized_input,
                &q_lora_raw,
                config.hidden_size,
                config.q_lora_rank,
                n_tokens,
                "packed Q A",
            )?;
        }
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &q_lora_raw,
            q_a_norm,
            &q_lora,
            n_tokens,
            config.q_lora_rank,
            rms_eps,
        )?;
        if q_b_projection.uses_full_chunk_f32(n_tokens) {
            match q_b_projection {
                Q8PrecisionProjection::F32Matrix => encode_q8_f32_mma_r2c4k64(
                    ctx,
                    enc,
                    q_b,
                    &q_lora,
                    &queries_raw,
                    config.q_lora_rank,
                    dims.query_width,
                    n_tokens,
                )?,
                Q8PrecisionProjection::WideF32Matrix => encode_q8_f32_mma_r2c16k64(
                    ctx,
                    enc,
                    q_b,
                    &q_lora,
                    &queries_raw,
                    config.q_lora_rank,
                    dims.query_width,
                    n_tokens,
                )?,
                _ => unreachable!("full-chunk F32 predicate admitted a non-F32 policy"),
            }
        } else {
            encode_batch_projection(
                ctx,
                enc,
                q_b,
                &q_lora,
                &queries_raw,
                config.q_lora_rank,
                dims.query_width,
                n_tokens,
                "packed Q B",
            )?;
        }
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &queries_raw,
            &self.head_norm_ones,
            &queries,
            checked_mul(n_tokens, config.head_count, "packed query rows")?,
            config.head_dim,
            rms_eps,
        )?;
        if q_a_kv_matrix.kv() {
            if n_tokens.is_multiple_of(128) {
                encode_q8_f32_mma_r2c16k64(
                    ctx,
                    enc,
                    kv_weight,
                    &normalized_input,
                    &kv_raw,
                    config.hidden_size,
                    config.head_dim,
                    n_tokens,
                )?;
            } else {
                encode_q8_f32_mma_r2c4k64(
                    ctx,
                    enc,
                    kv_weight,
                    &normalized_input,
                    &kv_raw,
                    config.hidden_size,
                    config.head_dim,
                    n_tokens,
                )?;
            }
        } else {
            encode_state_batch_projection(
                ctx,
                enc,
                kv_weight,
                &normalized_input,
                &kv_raw,
                config.hidden_size,
                config.head_dim,
                n_tokens,
                "packed KV",
            )?;
        }
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &kv_raw,
            kv_norm,
            &kv,
            n_tokens,
            config.head_dim,
            rms_eps,
        )?;
        Ok(PackedAttentionViews {
            normalized_input,
            q_lora,
            queries,
            kv,
            attention,
        })
    }

    fn encode_output(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        attention: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        n_tokens: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_output_batch")?;
        checked_token_count(n_tokens)?;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked()?;
        validate_f32(
            attention,
            &[dims.query_width as u64, n_tokens as u64],
            false,
            "packed attention heads",
        )?;
        validate_matvec_weight(
            output_a,
            dims.group_width,
            dims.low_rank_width,
            "packed output A",
        )?;
        validate_matvec_weight(
            output_b,
            dims.low_rank_width,
            config.hidden_size,
            "packed output B",
        )?;
        let low_rank = f32_prefix(
            &self.low_rank,
            vec![dims.low_rank_width as u64, n_tokens as u64],
            "packed low-rank attention",
        )?;
        let output = f32_prefix(
            &self.output,
            vec![config.hidden_size as u64, n_tokens as u64],
            "packed attention output",
        )?;
        let group_input = f32_prefix(
            &self.group_input,
            vec![dims.group_width as u64, n_tokens as u64],
            "packed attention group input",
        )?;
        let group_output = f32_prefix(
            &self.group_output,
            vec![config.output_rank as u64, n_tokens as u64],
            "packed attention group output",
        )?;
        for group in 0..config.group_count {
            encode_group_pack(
                ctx,
                enc,
                attention,
                &group_input,
                n_tokens,
                dims.query_width,
                dims.group_width,
                group,
                false,
            )?;
            let weight = group_weight_view(output_a, dims.group_width, config.output_rank, group)?;
            encode_batch_projection(
                ctx,
                enc,
                &weight,
                &group_input,
                &group_output,
                dims.group_width,
                config.output_rank,
                n_tokens,
                "packed grouped output A",
            )?;
            encode_group_pack(
                ctx,
                enc,
                &group_output,
                &low_rank,
                n_tokens,
                dims.low_rank_width,
                config.output_rank,
                group,
                true,
            )?;
        }
        encode_batch_projection(
            ctx,
            enc,
            output_b,
            &low_rank,
            &output,
            dims.low_rank_width,
            config.hidden_size,
            n_tokens,
            "packed output B",
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_output_q8_precision(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        attention: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        n_tokens: usize,
        output_a_projection: Q8PrecisionProjection,
        output_b_projection: Q8PrecisionProjection,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_Q8_precision_output_batch")?;
        checked_token_count(n_tokens)?;
        let config = deepseek_v4_session_attention_config();
        let dims = config.checked()?;
        validate_f32(
            attention,
            &[dims.query_width as u64, n_tokens as u64],
            false,
            "Q8 precision attention heads",
        )?;
        for (weight, n_in, n_out, name) in [
            (
                output_a,
                dims.group_width,
                dims.low_rank_width,
                "Q8 precision output A",
            ),
            (
                output_b,
                dims.low_rank_width,
                config.hidden_size,
                "Q8 precision output B",
            ),
        ] {
            validate_matvec_weight(weight, n_in, n_out, name)?;
            if weight.dtype != GgmlType::Q8_0 {
                return invalid(format!("{name} must be Q8_0, got {:?}", weight.dtype));
            }
        }
        let low_rank = f32_prefix(
            &self.low_rank,
            vec![dims.low_rank_width as u64, n_tokens as u64],
            "Q8 precision low rank",
        )?;
        let output = f32_prefix(
            &self.output,
            vec![config.hidden_size as u64, n_tokens as u64],
            "Q8 precision output",
        )?;
        let group_input = f32_prefix(
            &self.group_input,
            vec![dims.group_width as u64, n_tokens as u64],
            "Q8 precision group input",
        )?;
        let group_output = f32_prefix(
            &self.group_output,
            vec![config.output_rank as u64, n_tokens as u64],
            "Q8 precision group output",
        )?;
        let encode_projection = |weight: &MetalTensor,
                                 input: &MetalTensor,
                                 output: &MetalTensor,
                                 n_in: usize,
                                 n_out: usize,
                                 projection: Q8PrecisionProjection,
                                 name: &str| {
            match projection {
                Q8PrecisionProjection::Exact => encode_batch_projection(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens, name,
                ),
                #[cfg(test)]
                Q8PrecisionProjection::HalfMatrix => crate::metal::encode_mat_mat_q8_0_f32(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens,
                )
                .map_err(DeepSeekV4MetalError::Metal),
                Q8PrecisionProjection::F32Matrix => encode_q8_f32_mma_r2c4k64(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens,
                ),
                Q8PrecisionProjection::WideF32Matrix => encode_q8_f32_mma_r2c16k64(
                    ctx, enc, weight, input, output, n_in, n_out, n_tokens,
                ),
            }
        };
        if output_a_projection == Q8PrecisionProjection::WideF32Matrix
            && packed_q8_grouped_output_enabled()
        {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: strided grouped Q8 output A active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_OUTPUT_GROUPED=0"
                );
            }
            encode_q8_f32_mma_r2c16k64_grouped(
                ctx,
                enc,
                output_a,
                attention,
                &low_rank,
                dims.group_width,
                config.output_rank,
                config.group_count,
                n_tokens,
            )?;
        } else {
            for group in 0..config.group_count {
                encode_group_pack(
                    ctx,
                    enc,
                    attention,
                    &group_input,
                    n_tokens,
                    dims.query_width,
                    dims.group_width,
                    group,
                    false,
                )?;
                let weight =
                    group_weight_view(output_a, dims.group_width, config.output_rank, group)?;
                encode_projection(
                    &weight,
                    &group_input,
                    &group_output,
                    dims.group_width,
                    config.output_rank,
                    output_a_projection,
                    "Q8 precision grouped output A",
                )?;
                encode_group_pack(
                    ctx,
                    enc,
                    &group_output,
                    &low_rank,
                    n_tokens,
                    dims.low_rank_width,
                    config.output_rank,
                    group,
                    true,
                )?;
            }
        }
        encode_projection(
            output_b,
            &low_rank,
            &output,
            dims.low_rank_width,
            config.hidden_size,
            output_b_projection,
            "Q8 precision output B",
        )?;
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_group_pack(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    n_tokens: usize,
    row_width: usize,
    group_width: usize,
    group: usize,
    scatter: bool,
) -> Result<(), DeepSeekV4MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        row_width: u32,
        group_width: u32,
        group: u32,
    }
    let n_tokens_u32 = checked_token_count(n_tokens)?;
    let required_input = if scatter {
        checked_mul(n_tokens, group_width, "group scatter input")?
    } else {
        checked_mul(n_tokens, row_width, "group pack input")?
    };
    let required_output = if scatter {
        checked_mul(n_tokens, row_width, "group scatter output")?
    } else {
        checked_mul(n_tokens, group_width, "group pack output")?
    };
    if input.dtype != GgmlType::F32
        || output.dtype != GgmlType::F32
        || input.n_elements() != required_input as u64
        || output.n_elements() != required_output as u64
        || !output.is_writable()
        || group
            .checked_add(1)
            .and_then(|count| count.checked_mul(group_width))
            .is_none_or(|end| end > row_width)
    {
        return invalid("packed attention group copy has invalid geometry");
    }
    let kernel = if scatter {
        "kernel_deepseek_v4_scatter_low_rank_group"
    } else {
        "kernel_deepseek_v4_pack_attention_group"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens_u32,
            row_width: u32::try_from(row_width)
                .map_err(|_| DeepSeekV4MetalError::Invalid("group row width exceeds u32".into()))?,
            group_width: u32::try_from(group_width)
                .map_err(|_| DeepSeekV4MetalError::Invalid("group width exceeds u32".into()))?,
            group: u32::try_from(group)
                .map_err(|_| DeepSeekV4MetalError::Invalid("group index exceeds u32".into()))?,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    let total = checked_mul(n_tokens, group_width, "group copy elements")?;
    enc.dispatch(
        MTLSize {
            width: total.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

enum PackedCompressorViews {
    SlidingWindow,
    CompressedSparse {
        attention_kv: MetalTensor,
        attention_score: MetalTensor,
        indexer_kv: MetalTensor,
        indexer_score: MetalTensor,
    },
    HeavilyCompressed {
        attention_kv: MetalTensor,
        attention_score: MetalTensor,
    },
}

impl PrefillCompressorScratch {
    #[cfg(feature = "dsv4-diagnostics")]
    fn reset_q8_matrix_invocations(&self) {
        self.q8_matrix_invocations.set(0);
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn q8_matrix_invocations(&self) -> u32 {
        self.q8_matrix_invocations.get()
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_projection(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        weight: &MetalTensor,
        input: &MetalTensor,
        output: &MetalTensor,
        n_out: usize,
        n_tokens: usize,
        name: &str,
        use_matrix: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !use_matrix || weight.dtype != GgmlType::Q8_0 {
            return encode_state_batch_projection(
                ctx,
                enc,
                weight,
                input,
                output,
                DEEPSEEK_V4_HIDDEN_SIZE,
                n_out,
                n_tokens,
                name,
            );
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let next = self
            .q8_matrix_invocations
            .get()
            .checked_add(1)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed Q8 compressor matrix invocation count overflow".into(),
                )
            })?;
        crate::metal::encode_mat_mat_q8_0_f32(
            ctx,
            enc,
            weight,
            input,
            output,
            DEEPSEEK_V4_HIDDEN_SIZE,
            n_out,
            n_tokens,
        )
        .map_err(DeepSeekV4MetalError::Metal)?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.q8_matrix_invocations.set(next);
        Ok(())
    }

    fn encode_layer_projections(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        normalized_input: &MetalTensor,
        n_tokens: usize,
        use_matrix: bool,
    ) -> Result<PackedCompressorViews, DeepSeekV4MetalError> {
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        match residency
            .config()
            .attention_kinds
            .get(layer)
            .copied()
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed compressor layer {layer} is out of range"
                ))
            })? {
            AttentionKind::SlidingWindow => Ok(PackedCompressorViews::SlidingWindow),
            AttentionKind::CompressedSparse => {
                let attention_kv = f32_prefix(
                    &self.attention_kv,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n_tokens as u64],
                    "packed CSA compressor KV",
                )?;
                let attention_score = f32_prefix(
                    &self.attention_score,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64, n_tokens as u64],
                    "packed CSA compressor score",
                )?;
                let indexer_kv = f32_prefix(
                    &self.indexer_kv,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64, n_tokens as u64],
                    "packed indexer compressor KV",
                )?;
                let indexer_score = f32_prefix(
                    &self.indexer_score,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64, n_tokens as u64],
                    "packed indexer compressor score",
                )?;
                for (weight, output, width, name) in [
                    (
                        tensor("attn_compressor_kv.weight")?,
                        &attention_kv,
                        COMPRESSOR_ATTENTION_WIDTH,
                        "packed CSA compressor KV",
                    ),
                    (
                        tensor("attn_compressor_gate.weight")?,
                        &attention_score,
                        COMPRESSOR_ATTENTION_WIDTH,
                        "packed CSA compressor score",
                    ),
                    (
                        tensor("indexer_compressor_kv.weight")?,
                        &indexer_kv,
                        COMPRESSOR_INDEXER_WIDTH,
                        "packed indexer compressor KV",
                    ),
                    (
                        tensor("indexer_compressor_gate.weight")?,
                        &indexer_score,
                        COMPRESSOR_INDEXER_WIDTH,
                        "packed indexer compressor score",
                    ),
                ] {
                    self.encode_projection(
                        ctx,
                        enc,
                        weight,
                        normalized_input,
                        output,
                        width,
                        n_tokens,
                        name,
                        use_matrix,
                    )?;
                }
                Ok(PackedCompressorViews::CompressedSparse {
                    attention_kv,
                    attention_score,
                    indexer_kv,
                    indexer_score,
                })
            }
            AttentionKind::HeavilyCompressed => {
                let attention_kv = f32_prefix(
                    &self.hca_kv,
                    vec![512, n_tokens as u64],
                    "packed HCA compressor KV",
                )?;
                let attention_score = f32_prefix(
                    &self.hca_score,
                    vec![512, n_tokens as u64],
                    "packed HCA compressor score",
                )?;
                self.encode_projection(
                    ctx,
                    enc,
                    tensor("attn_compressor_kv.weight")?,
                    normalized_input,
                    &attention_kv,
                    512,
                    n_tokens,
                    "packed HCA compressor KV",
                    use_matrix,
                )?;
                self.encode_projection(
                    ctx,
                    enc,
                    tensor("attn_compressor_gate.weight")?,
                    normalized_input,
                    &attention_score,
                    512,
                    n_tokens,
                    "packed HCA compressor score",
                    use_matrix,
                )?;
                Ok(PackedCompressorViews::HeavilyCompressed {
                    attention_kv,
                    attention_score,
                })
            }
        }
    }
}

impl DeepSeekV4CompressorFrontiers {
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_projected_chunk(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        start_position: u32,
        row_count: usize,
        projected: &PackedCompressorViews,
        scratch: &PrefillCompressorScratch,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        let frontier = self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed compressor layer {layer} is out of range"
            ))
        })?;
        match (frontier, projected) {
            (
                DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                PackedCompressorViews::SlidingWindow,
            ) => Ok(()),
            (
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer },
                PackedCompressorViews::CompressedSparse {
                    attention_kv,
                    attention_score,
                    indexer_kv,
                    indexer_score,
                },
            ) => {
                attention.encode_projected_chunk(
                    ctx,
                    enc,
                    attention_kv,
                    attention_score,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    &scratch.pooled_rows,
                    &scratch.normalized_rows,
                    start_position,
                    row_count,
                    rope,
                    rms_eps,
                )?;
                indexer.encode_projected_chunk(
                    ctx,
                    enc,
                    indexer_kv,
                    indexer_score,
                    tensor("indexer_compressor_ape.weight")?,
                    tensor("indexer_compressor_norm.weight")?,
                    &scratch.pooled_rows,
                    &scratch.normalized_rows,
                    start_position,
                    row_count,
                    rope,
                    rms_eps,
                )
            }
            (
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention },
                PackedCompressorViews::HeavilyCompressed {
                    attention_kv,
                    attention_score,
                },
            ) => attention.encode_projected_chunk(
                ctx,
                enc,
                attention_kv,
                attention_score,
                tensor("attn_compressor_ape.weight")?,
                tensor("attn_compressor_norm.weight")?,
                &scratch.pooled_rows,
                &scratch.normalized_rows,
                start_position,
                row_count,
                rope,
                rms_eps,
            ),
            _ => invalid(format!(
                "packed compressor projection kind differs from layer {layer}"
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_projected_row(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        row: usize,
        position: u32,
        projected: &PackedCompressorViews,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        let frontier = self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed compressor layer {layer} is out of range"
            ))
        })?;
        match (frontier, projected) {
            (
                DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                PackedCompressorViews::SlidingWindow,
            ) => Ok(()),
            (
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer },
                PackedCompressorViews::CompressedSparse {
                    attention_kv,
                    attention_score,
                    indexer_kv,
                    indexer_score,
                },
            ) => {
                let attention_kv = f32_row(
                    attention_kv,
                    row,
                    COMPRESSOR_ATTENTION_WIDTH,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64],
                    "packed CSA compressor KV row",
                )?;
                let attention_score = f32_row(
                    attention_score,
                    row,
                    COMPRESSOR_ATTENTION_WIDTH,
                    vec![COMPRESSOR_ATTENTION_WIDTH as u64],
                    "packed CSA compressor score row",
                )?;
                attention.encode_projected(
                    ctx,
                    enc,
                    &attention_kv,
                    &attention_score,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    rope,
                    rms_eps,
                )?;
                let indexer_kv = f32_row(
                    indexer_kv,
                    row,
                    COMPRESSOR_INDEXER_WIDTH,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64],
                    "packed indexer compressor KV row",
                )?;
                let indexer_score = f32_row(
                    indexer_score,
                    row,
                    COMPRESSOR_INDEXER_WIDTH,
                    vec![COMPRESSOR_INDEXER_WIDTH as u64],
                    "packed indexer compressor score row",
                )?;
                indexer.encode_projected(
                    ctx,
                    enc,
                    &indexer_kv,
                    &indexer_score,
                    tensor("indexer_compressor_ape.weight")?,
                    tensor("indexer_compressor_norm.weight")?,
                    position,
                    rope,
                    rms_eps,
                )
            }
            (
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention },
                PackedCompressorViews::HeavilyCompressed {
                    attention_kv,
                    attention_score,
                },
            ) => {
                let attention_kv = f32_row(
                    attention_kv,
                    row,
                    512,
                    vec![512],
                    "packed HCA compressor KV row",
                )?;
                let attention_score = f32_row(
                    attention_score,
                    row,
                    512,
                    vec![512],
                    "packed HCA compressor score row",
                )?;
                attention.encode_projected(
                    ctx,
                    enc,
                    &attention_kv,
                    &attention_score,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    rope,
                    rms_eps,
                )
            }
            _ => invalid(format!(
                "packed compressor projection kind differs from layer {layer}"
            )),
        }
    }
}

struct PackedMoeViews {
    normalized_input: MetalTensor,
    logits: MetalTensor,
    hash_ids: Option<MetalTensor>,
}

struct ExpertBucket {
    expert: usize,
    start: usize,
    len: usize,
}

const PACKED_GROUPED_EXPERT_TILE_ROWS: usize = 32;
const PACKED_GROUPED_EXPERT_MAX_TILES: usize = MOE_EXPERT_COUNT
    + (DEEPSEEK_V4_PREFILL_MAX_TOKENS * MOE_TOP_K - MOE_EXPERT_COUNT)
        / PACKED_GROUPED_EXPERT_TILE_ROWS;
const PACKED_GROUPED_EXPERT_DESCRIPTOR_WORDS: usize = PACKED_GROUPED_EXPERT_MAX_TILES * 3;
const PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES: usize = 4_096;
const PACKED_GROUPED_EXPERT_INLINE_MAX_TILES: usize = PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES / 12;
const PACKED_GROUPED_IQ2_MMA16_TILE_ROWS: usize = 16;
const PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS: usize = 128;
const PACKED_GROUPED_Q3Q4_NARROW_TOKENS: usize = 128;
const PACKED_GROUPED_IQ2_MMA16_MEDIUM_TOKENS: usize = PACKED_MATRIX_MIN_TOKENS;
const PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS: usize = DEEPSEEK_V4_PREFILL_MAX_TOKENS;
const PACKED_GROUPED_IQ2_MMA16_MAX_TILES: usize = MOE_EXPERT_COUNT
    + (PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS * MOE_TOP_K - MOE_EXPERT_COUNT)
        / PACKED_GROUPED_IQ2_MMA16_TILE_ROWS;
const PACKED_GROUPED_IQ2_MMA16_DESCRIPTOR_WORDS: usize = PACKED_GROUPED_IQ2_MMA16_MAX_TILES * 3;
const _: () = assert!(PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS <= DEEPSEEK_V4_PREFILL_MAX_TOKENS);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedGroupedExpertTile {
    expert: u32,
    start: u32,
    count: u32,
}

const _: () = assert!(std::mem::size_of::<PackedGroupedExpertTile>() == 12);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * PACKED_GROUPED_EXPERT_INLINE_MAX_TILES
        <= PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * (PACKED_GROUPED_EXPERT_INLINE_MAX_TILES + 1)
        > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * PACKED_GROUPED_EXPERT_MAX_TILES
        > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);
const _: () = assert!(
    std::mem::size_of::<PackedGroupedExpertTile>() * PACKED_GROUPED_IQ2_MMA16_MAX_TILES
        > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES
);

fn validate_packed_expert_schedule(
    n_tokens: usize,
    expert_count: usize,
    expert_ids: &[i32],
    bucket_rows: &[i32],
    bucket_slots: &[i32],
    schedule: &[ExpertBucket],
) -> Result<(), DeepSeekV4MetalError> {
    validate_packed_expert_count(expert_count)?;
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed expert route count")?;
    if expert_ids.len() != route_count
        || bucket_rows.len() != route_count
        || bucket_slots.len() != route_count
    {
        return invalid("packed expert schedule payload has invalid length");
    }
    let mut cursor = 0usize;
    let mut previous_expert = None;
    let mut seen_slots = vec![false; route_count];
    for bucket in schedule {
        if bucket.expert >= expert_count
            || previous_expert.is_some_and(|expert| bucket.expert <= expert)
            || bucket.start != cursor
            || bucket.len == 0
        {
            return invalid("packed expert schedule has invalid bucket geometry");
        }
        let end = bucket.start.checked_add(bucket.len).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed expert bucket end overflow".into())
        })?;
        if end > route_count {
            return invalid("packed expert bucket exceeds route assignments");
        }
        let mut previous_slot = None;
        for index in bucket.start..end {
            let row = usize::try_from(bucket_rows[index]).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed expert schedule has negative row".into())
            })?;
            let slot = usize::try_from(bucket_slots[index]).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed expert schedule has negative slot".into())
            })?;
            if row >= n_tokens
                || slot >= route_count
                || row != slot / MOE_TOP_K
                || expert_ids[slot] != bucket.expert as i32
                || previous_slot.is_some_and(|previous| slot <= previous)
                || std::mem::replace(&mut seen_slots[slot], true)
            {
                return invalid("packed expert schedule violates expert/token/slot order");
            }
            previous_slot = Some(slot);
        }
        cursor = end;
        previous_expert = Some(bucket.expert);
    }
    if cursor != route_count || seen_slots.iter().any(|seen| !seen) {
        return invalid("packed expert schedule does not cover every route exactly once");
    }
    Ok(())
}

fn packed_grouped_expert_tiles(
    n_tokens: usize,
    schedule: &[ExpertBucket],
) -> Result<Vec<PackedGroupedExpertTile>, DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed grouped route count")?;
    let mut cursor = 0usize;
    let mut previous_expert = None;
    let mut tiles = Vec::new();
    for bucket in schedule {
        if bucket.expert >= MOE_EXPERT_COUNT
            || previous_expert.is_some_and(|expert| bucket.expert <= expert)
            || bucket.start != cursor
            || bucket.len == 0
        {
            return invalid("packed grouped expert schedule has invalid bucket geometry");
        }
        for offset in (0..bucket.len).step_by(PACKED_GROUPED_EXPERT_TILE_ROWS) {
            let count = (bucket.len - offset).min(PACKED_GROUPED_EXPERT_TILE_ROWS);
            tiles.push(PackedGroupedExpertTile {
                expert: u32::try_from(bucket.expert).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed grouped expert exceeds u32".into())
                })?,
                start: u32::try_from(bucket.start + offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed grouped start exceeds u32".into())
                })?,
                count: count as u32,
            });
        }
        cursor = bucket.start + bucket.len;
        previous_expert = Some(bucket.expert);
    }
    if cursor != route_count || tiles.is_empty() || tiles.len() > PACKED_GROUPED_EXPERT_MAX_TILES {
        return invalid(format!(
            "packed grouped expert plan has {} assignments and {} tiles",
            cursor,
            tiles.len()
        ));
    }
    Ok(tiles)
}

struct PackedGroupedExpertPlan {
    tiles: Vec<PackedGroupedExpertTile>,
    buffer: Option<MetalTensor>,
    dispatch_tiles: usize,
}

impl PackedGroupedExpertPlan {
    fn new(
        n_tokens: usize,
        schedule: &[ExpertBucket],
        tile_buffer: Option<&MetalTensor>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::from_tiles(
            packed_grouped_expert_tiles(n_tokens, schedule)?,
            tile_buffer,
        )
    }

    fn new_iq2_mma16(
        n_tokens: usize,
        schedule: &[ExpertBucket],
        tile_buffer: Option<&MetalTensor>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::from_tiles(
            packed_grouped_iq2_mma16_tiles(n_tokens, schedule)?,
            tile_buffer,
        )
    }

    fn from_tiles(
        tiles: Vec<PackedGroupedExpertTile>,
        tile_buffer: Option<&MetalTensor>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let bytes = std::mem::size_of_val(tiles.as_slice());
        let buffer = if bytes > PACKED_GROUPED_EXPERT_INLINE_MAX_BYTES {
            let tile_buffer = tile_buffer.ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed grouped expert plan requires {bytes} descriptor bytes"
                ))
            })?;
            let words = bytemuck::cast_slice::<PackedGroupedExpertTile, i32>(&tiles);
            let view = i32_prefix(
                tile_buffer,
                vec![words.len() as u64],
                "packed grouped expert descriptors",
            )?;
            host_write_i32(&view, words, "packed grouped expert descriptors")?;
            Some(view)
        } else {
            None
        };
        let dispatch_tiles = tiles.len();
        Ok(Self {
            tiles,
            buffer,
            dispatch_tiles,
        })
    }

    fn from_device(
        buffer: &MetalTensor,
        dispatch_tiles: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        validate_i32(
            buffer,
            &[(dispatch_tiles * 3) as u64],
            false,
            "packed grouped device descriptors",
        )?;
        Ok(Self {
            tiles: Vec::new(),
            buffer: Some(buffer.clone()),
            dispatch_tiles,
        })
    }

    fn bind(&self, enc: &KernelEncoder, index: usize) {
        if let Some(buffer) = self.buffer.as_ref() {
            enc.set_tensor(index, buffer);
        } else {
            enc.set_bytes_slice(index, &self.tiles);
        }
    }
}

fn packed_grouped_iq2_mma16_tiles(
    n_tokens: usize,
    schedule: &[ExpertBucket],
) -> Result<Vec<PackedGroupedExpertTile>, DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed IQ2 MMA route count")?;
    let mut cursor = 0usize;
    let mut previous_expert = None;
    let mut tiles = Vec::new();
    for bucket in schedule {
        if bucket.expert >= MOE_EXPERT_COUNT
            || previous_expert.is_some_and(|expert| bucket.expert <= expert)
            || bucket.start != cursor
            || bucket.len == 0
        {
            return invalid("packed IQ2 MMA schedule has invalid bucket geometry");
        }
        for offset in (0..bucket.len).step_by(PACKED_GROUPED_IQ2_MMA16_TILE_ROWS) {
            tiles.push(PackedGroupedExpertTile {
                expert: u32::try_from(bucket.expert).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed IQ2 MMA expert exceeds u32".into())
                })?,
                start: u32::try_from(bucket.start + offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed IQ2 MMA start exceeds u32".into())
                })?,
                count: (bucket.len - offset).min(PACKED_GROUPED_IQ2_MMA16_TILE_ROWS) as u32,
            });
        }
        cursor = bucket.start + bucket.len;
        previous_expert = Some(bucket.expert);
    }
    if cursor != route_count || tiles.is_empty() || tiles.len() > PACKED_GROUPED_IQ2_MMA16_MAX_TILES
    {
        return invalid(format!(
            "packed IQ2 MMA plan has {} assignments and {} tiles",
            cursor,
            tiles.len()
        ));
    }
    Ok(tiles)
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_swiglu_iq2_xs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    normalized_input: &MetalTensor,
    bucket_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    inner: &MetalTensor,
    hidden: usize,
    ffn: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped IQ2_XS gate/up")?;
    validate_packed_expert_count(expert_count)?;
    if !hidden.is_multiple_of(256)
        || top_k != MOE_TOP_K
        || gate_bank.dtype != GgmlType::IQ2_XS
        || up_bank.dtype != GgmlType::IQ2_XS
        || !clamp.is_finite()
        || clamp <= 0.0
    {
        return invalid("packed grouped gate/up requires aligned IQ2_XS banks and positive clamp");
    }
    validate_expert_bank(
        gate_bank,
        hidden,
        ffn,
        expert_count,
        "packed grouped gate bank",
    )?;
    validate_expert_bank(up_bank, hidden, ffn, expert_count, "packed grouped up bank")?;
    validate_f32(
        normalized_input,
        &[hidden as u64, n_tokens as u64],
        false,
        "packed grouped normalized input",
    )?;
    validate_i32(
        bucket_slots,
        &[(n_tokens * top_k) as u64],
        false,
        "packed grouped slots",
    )?;
    validate_f32(
        inner,
        &[ffn as u64, top_k as u64, n_tokens as u64],
        true,
        "packed grouped inner",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        hidden: u32,
        ffn: u32,
        n_expert: u32,
        top_k: u32,
        n_tokens: u32,
        clamp: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_swiglu_iq2_xs_f32")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("packed grouped IQ2_XS gate/up requires SIMD width 32");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            hidden: u32::try_from(hidden).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped hidden exceeds u32".into())
            })?,
            ffn: u32::try_from(ffn).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped FFN exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped experts exceed u32".into())
            })?,
            top_k: u32::try_from(top_k).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped top-k exceeds u32".into())
            })?,
            n_tokens: checked_token_count(n_tokens)?,
            clamp,
        },
    );
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, normalized_input);
    enc.set_tensor(4, bucket_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, inner);
    enc.set_threadgroup_memory(0, 64 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: ffn,
            height: tile_count,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_down_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    down_bank: &MetalTensor,
    inner: &MetalTensor,
    bucket_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped IQ3_XXS down")?;
    validate_packed_expert_count(expert_count)?;
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled() {
        return invalid("packed grouped IQ3_XXS down requires the SIMD-matrix policy");
    }
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || top_k != MOE_TOP_K
        || down_bank.dtype != GgmlType::IQ3_XXS
    {
        return invalid("packed grouped down requires aligned IQ3_XXS storage");
    }
    validate_expert_bank(
        down_bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped down bank",
    )?;
    validate_f32(
        inner,
        &[n_in as u64, top_k as u64, n_tokens as u64],
        false,
        "packed grouped inner",
    )?;
    validate_i32(
        bucket_slots,
        &[(n_tokens * top_k) as u64],
        false,
        "packed grouped down slots",
    )?;
    validate_f32(
        output,
        &[n_out as u64, top_k as u64, n_tokens as u64],
        true,
        "packed grouped output",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        top_k: u32,
        n_tokens: u32,
    }
    let row_bytes = checked_mul(n_in / 256, 98, "packed grouped IQ3_XXS row bytes")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_down_iq3_xxs_f32_mm")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid("packed grouped IQ3_XXS down requires four SIMD groups");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped output exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped stride exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped experts exceed u32".into())
            })?,
            top_k: u32::try_from(top_k).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped top-k exceeds u32".into())
            })?,
            n_tokens: checked_token_count(n_tokens)?,
        },
    );
    enc.set_tensor(1, down_bank);
    enc.set_tensor(2, inner);
    enc.set_tensor(3, bucket_slots);
    plan.bind(enc, 4);
    enc.set_tensor(5, output);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedIq2MatrixWorkUnit {
    Mma16,
    Mm64x32,
    Mm64x32F16,
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq2_xs_f32_matrix(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
    work_unit: PackedIq2MatrixWorkUnit,
) -> Result<(), DeepSeekV4MetalError> {
    let (tile_rows, threads, threadgroup_bytes, kernel, label) = match work_unit {
        PackedIq2MatrixWorkUnit::Mma16 => (
            16usize,
            32usize,
            4_096usize,
            "kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mma16",
            "packed grouped mapped IQ2_XS F32 MMA16 projection",
        ),
        PackedIq2MatrixWorkUnit::Mm64x32 => (
            64,
            128,
            12_288,
            "kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mm64x32",
            "packed grouped mapped IQ2_XS F32 MM64x32 projection",
        ),
        PackedIq2MatrixWorkUnit::Mm64x32F16 => (
            64,
            128,
            8_192,
            "kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f16_mm64x32",
            "packed grouped mapped IQ2_XS F16 MM64x32 projection",
        ),
    };
    require_serial(enc, label)?;
    validate_packed_expert_count(expert_count)?;
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(tile_rows)
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
        || bank.dtype != GgmlType::IQ2_XS
    {
        return invalid(format!("{label} has invalid geometry or storage"));
    }
    if ctx.device.maxThreadgroupMemoryLength() < threadgroup_bytes {
        return invalid(format!(
            "{label} requires {threadgroup_bytes} bytes of threadgroup memory"
        ));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, label)?;
    validate_f32(input, &[n_in as u64, source_count as u64], false, label)?;
    let map_count = checked_mul(n_tokens, top_k, label)?;
    validate_i32(source_rows, &[map_count as u64], false, label)?;
    validate_i32(destination_slots, &[map_count as u64], false, label)?;
    validate_f32(
        output,
        &[n_out as u64, destination_count as u64],
        true,
        label,
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
    }
    let row_bytes = checked_mul(n_in / 256, 74, label)?;
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads {
        return invalid(format!("{label} requires {} SIMD groups", threads / 32));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA output exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA stride exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA experts exceed u32".into())
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA map count exceeds u32".into())
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA source count exceeds u32".into())
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed IQ2 MMA destination count exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, source_rows);
    enc.set_tensor(4, destination_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, threadgroup_bytes);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / tile_rows,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq2_xs_swiglu_f32_matrix(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    gate: &MetalTensor,
    up: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
    clamp: f32,
    work_unit: PackedIq2MatrixWorkUnit,
) -> Result<(), DeepSeekV4MetalError> {
    for (bank, projection) in [(gate_bank, gate), (up_bank, up)] {
        encode_packed_grouped_mapped_iq2_xs_f32_matrix(
            ctx,
            enc,
            bank,
            input,
            source_rows,
            destination_slots,
            plan,
            projection,
            n_in,
            n_out,
            expert_count,
            top_k,
            n_tokens,
            source_count,
            destination_count,
            work_unit,
        )?;
    }
    let projected_elements = checked_mul(
        n_out,
        destination_count,
        "packed BM16 IQ2 projected elements",
    )?;
    encode_ds4_clamped_swiglu(
        ctx,
        enc,
        &gate.view_subrange(0, vec![projected_elements as u64]),
        &up.view_subrange(0, vec![projected_elements as u64]),
        &output.view_subrange(0, vec![projected_elements as u64]),
        clamp,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_k_block_f32_plan(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let (block_bytes, kernel, label) = match bank.dtype {
        GgmlType::Q3_K => (
            110usize,
            "kernel_deepseek_v4_packed_grouped_mapped_q3_K_f32_mm",
            "packed grouped mapped Q3_K projection",
        ),
        GgmlType::Q4_K => (
            144,
            "kernel_deepseek_v4_packed_grouped_mapped_q4_K_f32_mm",
            "packed grouped mapped Q4_K projection",
        ),
        dtype => {
            return invalid(format!(
                "packed grouped mapped K-block dtype {dtype:?} is unsupported"
            ));
        }
    };
    require_serial(enc, label)?;
    validate_packed_expert_count(expert_count)?;
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
    {
        return invalid(format!("{label} has invalid geometry"));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, label)?;
    validate_f32(input, &[n_in as u64, source_count as u64], false, label)?;
    let map_count = checked_mul(n_tokens, top_k, label)?;
    validate_i32(source_rows, &[map_count as u64], false, label)?;
    validate_i32(destination_slots, &[map_count as u64], false, label)?;
    validate_f32(
        output,
        &[n_out as u64, destination_count as u64],
        true,
        label,
    )?;
    if ctx.device.maxThreadgroupMemoryLength() < 8_192 {
        return invalid(format!("{label} requires 8192 bytes of threadgroup memory"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
    }
    let row_bytes = checked_mul(n_in / 256, block_bytes, label)?;
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid(format!("{label} requires four SIMD groups"));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} output exceeds u32"))
            })?,
            k: u32::try_from(n_in)
                .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{label} input exceeds u32")))?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} row bytes exceed u32"))
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} stride exceeds u32"))
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} expert count exceeds u32"))
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} map count exceeds u32"))
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} source count exceeds u32"))
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("{label} destination count exceeds u32"))
            })?,
        },
    );
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, source_rows);
    enc.set_tensor(4, destination_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: plan.dispatch_tiles,
            height: n_out / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq3_xxs_f32_plan(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped mapped IQ3_XXS projection")?;
    validate_packed_expert_count(expert_count)?;
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled() {
        return invalid("packed grouped mapped IQ3_XXS requires the SIMD-matrix policy");
    }
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
        || bank.dtype != GgmlType::IQ3_XXS
    {
        return invalid("packed grouped mapped IQ3_XXS has invalid geometry or storage");
    }
    validate_expert_bank(
        bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ3_XXS bank",
    )?;
    validate_f32(
        input,
        &[n_in as u64, source_count as u64],
        false,
        "packed grouped mapped IQ3_XXS input",
    )?;
    let map_count = checked_mul(n_tokens, top_k, "packed grouped mapped row count")?;
    validate_i32(
        source_rows,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS source rows",
    )?;
    validate_i32(
        destination_slots,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS destination slots",
    )?;
    validate_f32(
        output,
        &[n_out as u64, destination_count as u64],
        true,
        "packed grouped mapped IQ3_XXS output",
    )?;
    let tile_count = plan.dispatch_tiles;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
    }
    let row_bytes = checked_mul(n_in / 256, 98, "packed grouped mapped IQ3_XXS row")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq3_xxs_f32_mm")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid("packed grouped mapped IQ3_XXS requires four SIMD groups");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped output exceeds u32".into())
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped input exceeds u32".into())
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped row bytes exceed u32".into())
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped stride exceeds u32".into())
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped experts exceed u32".into())
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped mapped count exceeds u32".into())
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped source count exceeds u32".into())
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed grouped destination count exceeds u32".into())
            })?,
        },
    );
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, source_rows);
    enc.set_tensor(4, destination_slots);
    plan.bind(enc, 5);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    schedule: &[ExpertBucket],
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let plan = PackedGroupedExpertPlan::new(n_tokens, schedule, None)?;
    encode_packed_grouped_mapped_iq3_xxs_f32_plan(
        ctx,
        enc,
        bank,
        input,
        source_rows,
        destination_slots,
        &plan,
        output,
        n_in,
        n_out,
        expert_count,
        top_k,
        n_tokens,
        source_count,
        destination_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_all_iq3(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    down_bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    plan: &PackedGroupedExpertPlan,
    output: &MetalTensor,
    inner: &MetalTensor,
    hidden: usize,
    ffn: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    let route_count = checked_mul(n_tokens, top_k, "packed grouped all-IQ3 routes")?;
    let projected_elements = checked_mul(ffn, route_count, "packed grouped all-IQ3 projection")?;
    let (gate, up) = packed_grouped_gate_up_views(output, hidden, ffn, route_count)?;
    for (bank, projection) in [(gate_bank, &gate), (up_bank, &up)] {
        encode_packed_grouped_mapped_iq3_xxs_f32_plan(
            ctx,
            enc,
            bank,
            input,
            source_rows,
            destination_slots,
            plan,
            projection,
            hidden,
            ffn,
            expert_count,
            top_k,
            n_tokens,
            n_tokens,
            route_count,
        )?;
    }
    encode_ds4_clamped_swiglu(
        ctx,
        enc,
        &gate.view_subrange(0, vec![projected_elements as u64]),
        &up.view_subrange(0, vec![projected_elements as u64]),
        &inner.view_subrange(0, vec![projected_elements as u64]),
        clamp,
    )?;
    encode_packed_grouped_mapped_iq3_xxs_f32_plan(
        ctx,
        enc,
        down_bank,
        inner,
        destination_slots,
        destination_slots,
        plan,
        output,
        ffn,
        hidden,
        expert_count,
        top_k,
        n_tokens,
        route_count,
        route_count,
    )
}

fn packed_grouped_tensor_ranges_overlap(left: &MetalTensor, right: &MetalTensor) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let left_end = left.offset + left.n_bytes();
    let right_end = right.offset + right.n_bytes();
    left.offset < right_end && right.offset < left_end
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_packed_grouped_mapped_swiglu_iq3_xxs_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    source_rows: &MetalTensor,
    destination_slots: &MetalTensor,
    schedule: &[ExpertBucket],
    gate_up_arena: &MetalTensor,
    inner: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    top_k: usize,
    n_tokens: usize,
    source_count: usize,
    destination_count: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "packed grouped mapped IQ3_XXS gate/up/SwiGLU")?;
    validate_packed_expert_count(expert_count)?;
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled() {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU requires the SIMD-matrix policy");
    }
    if !n_in.is_multiple_of(256)
        || !n_out.is_multiple_of(64)
        || n_in != 2 * n_out
        || top_k != MOE_TOP_K
        || source_count == 0
        || destination_count == 0
        || gate_bank.dtype != GgmlType::IQ3_XXS
        || up_bank.dtype != GgmlType::IQ3_XXS
        || !clamp.is_finite()
        || clamp <= 0.0
    {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU has invalid geometry or storage");
    }
    validate_expert_bank(
        gate_bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ3_XXS gate bank",
    )?;
    validate_expert_bank(
        up_bank,
        n_in,
        n_out,
        expert_count,
        "packed grouped mapped IQ3_XXS up bank",
    )?;
    validate_f32(
        input,
        &[n_in as u64, source_count as u64],
        false,
        "packed grouped mapped IQ3_XXS SwiGLU input",
    )?;
    let map_count = checked_mul(n_tokens, top_k, "packed grouped mapped SwiGLU row count")?;
    validate_i32(
        source_rows,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS SwiGLU source rows",
    )?;
    validate_i32(
        destination_slots,
        &[map_count as u64],
        false,
        "packed grouped mapped IQ3_XXS SwiGLU destination slots",
    )?;
    let (gate, up) = packed_grouped_gate_up_views(gate_up_arena, n_in, n_out, destination_count)?;
    validate_f32(
        inner,
        &[n_out as u64, destination_count as u64],
        true,
        "packed grouped mapped IQ3_XXS SwiGLU inner",
    )?;
    if packed_grouped_tensor_ranges_overlap(&gate, inner)
        || packed_grouped_tensor_ranges_overlap(&up, inner)
    {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU inner overlaps gate/up arena");
    }
    let tiles = packed_grouped_expert_tiles(n_tokens, schedule)?;
    let tile_count = tiles.len();
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
        n_expert: u32,
        map_count: u32,
        source_count: u32,
        destination_count: u32,
        clamp: f32,
    }
    let row_bytes = checked_mul(n_in / 256, 98, "packed grouped mapped IQ3_XXS SwiGLU row")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_swiglu_iq3_xxs_f32_mm")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid("packed grouped mapped IQ3_XXS SwiGLU requires four SIMD groups");
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u32::try_from(n_out).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU output exceeds u32".into(),
                )
            })?,
            k: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU input exceeds u32".into(),
                )
            })?,
            nb01: u32::try_from(row_bytes).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU row bytes exceed u32".into(),
                )
            })?,
            stride_b: u32::try_from(n_in).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU stride exceeds u32".into(),
                )
            })?,
            n_expert: u32::try_from(expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU experts exceed u32".into(),
                )
            })?,
            map_count: u32::try_from(map_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU count exceeds u32".into(),
                )
            })?,
            source_count: u32::try_from(source_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU source count exceeds u32".into(),
                )
            })?,
            destination_count: u32::try_from(destination_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "packed grouped mapped SwiGLU destination count exceeds u32".into(),
                )
            })?,
            clamp,
        },
    );
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, source_rows);
    enc.set_tensor(5, destination_slots);
    enc.set_bytes_slice(6, &tiles);
    enc.set_tensor(7, &gate);
    enc.set_tensor(8, &up);
    enc.set_tensor(9, inner);
    enc.set_threadgroup_memory(0, 8_192);
    enc.dispatch(
        MTLSize {
            width: tile_count,
            height: n_out / 64,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn packed_grouped_gate_up_views(
    arena: &MetalTensor,
    hidden: usize,
    ffn: usize,
    destination_count: usize,
) -> Result<(MetalTensor, MetalTensor), DeepSeekV4MetalError> {
    let arena_elements = checked_mul(
        destination_count,
        hidden,
        "packed grouped IQ3 gate/up arena",
    )?;
    let half_elements = checked_mul(destination_count, ffn, "packed grouped IQ3 gate/up half")?;
    if hidden != 2 * ffn
        || checked_mul(half_elements, 2, "packed grouped IQ3 gate/up split")? != arena_elements
    {
        return invalid("packed grouped IQ3 arena does not split into gate/up halves");
    }
    validate_f32(
        arena,
        &[hidden as u64, destination_count as u64],
        true,
        "packed grouped IQ3 gate/up arena",
    )?;
    let gate = arena.view_subrange(0, vec![ffn as u64, destination_count as u64]);
    let up = arena.view_subrange(
        half_elements as u64,
        vec![ffn as u64, destination_count as u64],
    );
    if Retained::as_ptr(&gate.buffer) != Retained::as_ptr(&up.buffer)
        || Retained::as_ptr(&gate.buffer) != Retained::as_ptr(&arena.buffer)
        || gate.offset + gate.n_bytes() != up.offset
        || up.offset + up.n_bytes() != arena.offset + arena.n_bytes()
    {
        return invalid("packed grouped IQ3 gate/up arena views are not exact and disjoint");
    }
    Ok((gate, up))
}

fn packed_grouped_expert_kernels_supported(ctx: &MetalContext) -> bool {
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled()
        || ctx.device.maxThreadgroupMemoryLength() < 8_192
    {
        return false;
    }
    let Ok(gate_up) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_swiglu_iq2_xs_f32") else {
        return false;
    };
    let Ok(down) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_down_iq3_xxs_f32_mm") else {
        return false;
    };
    gate_up.threadExecutionWidth() == 32
        && gate_up.maxTotalThreadsPerThreadgroup() >= 32
        && down.threadExecutionWidth() == 32
        && down.maxTotalThreadsPerThreadgroup() >= 128
}

fn packed_grouped_iq2_mma16_candidate_supported(ctx: &MetalContext) -> bool {
    if ctx.device.maxThreadgroupMemoryLength() < 4_096 {
        return false;
    }
    let Ok(projection) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mma16")
    else {
        return false;
    };
    projection.threadExecutionWidth() == 32 && projection.maxTotalThreadsPerThreadgroup() >= 32
}

fn packed_iq2_mm64x32_candidate_supported(ctx: &MetalContext) -> bool {
    if ctx.device.maxThreadgroupMemoryLength() < 12_288 {
        return false;
    }
    let Ok(projection) =
        ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mm64x32")
    else {
        return false;
    };
    projection.threadExecutionWidth() == 32 && projection.maxTotalThreadsPerThreadgroup() >= 128
}

fn packed_iq2_f16_mm64x32_candidate_supported(ctx: &MetalContext) -> bool {
    if ctx.device.maxThreadgroupMemoryLength() < 8_192 {
        return false;
    }
    let Ok(projection) =
        ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f16_mm64x32")
    else {
        return false;
    };
    projection.threadExecutionWidth() == 32 && projection.maxTotalThreadsPerThreadgroup() >= 128
}

crate::env_flag!(
    default_on packed_grouped_iq2_mma16_enabled,
    "QWEN_DSV4_PACKED_BM16_IQ2"
);

crate::env_flag!(
    default_on packed_iq2_mm64x32_enabled,
    "QWEN_DSV4_PACKED_IQ2_MM64X32"
);

crate::env_flag!(
    default_on packed_iq2_f16_mm64x32_enabled,
    "QWEN_DSV4_PACKED_IQ2_F16_MATRIX"
);

crate::env_flag!(
    default_on packed_batched_rope_enabled,
    "QWEN_DSV4_BATCHED_ROPE"
);

crate::env_flag!(
    default_on packed_batched_compressor_enabled,
    "QWEN_DSV4_BATCHED_COMPRESSOR"
);

crate::env_flag!(
    default_on packed_selected_online_enabled,
    "QWEN_DSV4_PACKED_SELECTED_ONLINE"
);

crate::env_flag!(
    default_on packed_grouped_dense_attention_enabled,
    "QWEN_DSV4_PACKED_GROUP8_DENSE"
);

crate::env_flag!(
    default_on packed_indexer_batched_rope_enabled,
    "QWEN_DSV4_PACKED_INDEXER_BATCHED_ROPE"
);

crate::env_flag!(
    default_on packed_indexer_visible_dispatch_enabled,
    "QWEN_DSV4_PACKED_INDEXER_VISIBLE_DISPATCH"
);

crate::env_flag!(
    default_on packed_indexer_tiled_f32_enabled,
    "QWEN_DSV4_PACKED_INDEXER_TILED_F32"
);

crate::env_flag!(
    default_on packed_q8_grouped_output_enabled,
    "QWEN_DSV4_PACKED_Q8_OUTPUT_GROUPED"
);

crate::env_flag!(
    default_on packed_indexer_q_matrix_enabled,
    "QWEN_DSV4_PACKED_INDEXER_Q_MATRIX"
);

crate::env_flag!(
    default_off packed_gpu_route_compact_enabled,
    "QWEN_DSV4_PACKED_GPU_ROUTE_COMPACT"
);

crate::env_flag!(
    default_on packed_grouped_iq3_enabled,
    "QWEN_DSV4_PACKED_GROUPED_IQ3"
);

crate::env_flag!(
    default_on packed_grouped_q3q4_enabled,
    "QWEN_DSV4_PACKED_GROUPED_Q3Q4"
);

crate::env_flag!(
    default_on packed_shared_route_overlap_enabled,
    "QWEN_DSV4_PACKED_SHARED_ROUTE_OVERLAP"
);

crate::env_flag!(
    default_on packed_mxfp4_matrix_enabled,
    "QWEN_DSV4_PACKED_MXFP4_MATRIX"
);

crate::env_flag!(
    default_off packed_gpu_route_iq3_enabled,
    "QWEN_DSV4_PACKED_GPU_ROUTE_IQ3"
);

fn packed_grouped_iq3_candidate_supported(ctx: &MetalContext) -> bool {
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled()
        || ctx.device.maxThreadgroupMemoryLength() < 8_192
    {
        return false;
    }
    let Ok(projection) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_iq3_xxs_f32_mm")
    else {
        return false;
    };
    projection.threadExecutionWidth() == 32 && projection.maxTotalThreadsPerThreadgroup() >= 128
}

fn packed_grouped_q3q4_candidate_supported(ctx: &MetalContext) -> bool {
    if ctx.device.maxThreadgroupMemoryLength() < 8_192 {
        return false;
    }
    for kernel in [
        "kernel_deepseek_v4_packed_grouped_mapped_q3_K_f32_mm",
        "kernel_deepseek_v4_packed_grouped_mapped_q4_K_f32_mm",
    ] {
        let Ok(pipeline) = ctx.pipeline(kernel) else {
            return false;
        };
        if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 128 {
            return false;
        }
    }
    true
}

fn packed_mxfp4_matrix_candidate_supported(ctx: &MetalContext) -> bool {
    if ctx.device.maxThreadgroupMemoryLength() < 12 * 1024 {
        return false;
    }
    let Ok(pipeline) = ctx.pipeline("kernel_mat_mat_mxfp4_f32_mm64x32") else {
        return false;
    };
    pipeline.threadExecutionWidth() == 32 && pipeline.maxTotalThreadsPerThreadgroup() >= 128
}

#[allow(clippy::too_many_arguments)]
fn packed_grouped_q3q4_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    device_name == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE
        && tensor_count == 1_328
        && source_bytes == PACKED_Q8_MATRIX_REAP_K160_SOURCE_BYTES
        && expert_count == 160
        && (n_tokens == PACKED_GROUPED_Q3Q4_NARROW_TOKENS
            || packed_q8_partial_matrix_chunk_qualified(n_tokens))
        && gate_dtype == GgmlType::Q3_K
        && up_dtype == GgmlType::Q3_K
        && down_dtype == GgmlType::Q4_K
}

#[allow(clippy::too_many_arguments)]
fn packed_shared_route_overlap_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    packed_q8_partial_matrix_chunk_qualified(n_tokens)
        && packed_grouped_q3q4_scope_qualified(
            device_name,
            tensor_count,
            source_bytes,
            expert_count,
            n_tokens,
            gate_dtype,
            up_dtype,
            down_dtype,
        )
}

fn packed_mxfp4_matrix_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    device_name == PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_DEVICE
        && tensor_count == 1_328
        && (n_tokens == PACKED_MATRIX_MIN_TOKENS || n_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS)
        && matches!(
            (source_bytes, expert_count),
            (
                PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
                MOE_EXPERT_COUNT,
            ) | (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216,)
        )
}

#[cfg(test)]
fn packed_grouped_iq3_fused_candidate_supported(ctx: &MetalContext) -> bool {
    if !crate::metal::matmat_iq3_xxs_mm_is_enabled()
        || ctx.device.maxThreadgroupMemoryLength() < 8_192
    {
        return false;
    }
    let Ok(fused) = ctx.pipeline("kernel_deepseek_v4_packed_grouped_mapped_swiglu_iq3_xxs_f32_mm")
    else {
        return false;
    };
    fused.threadExecutionWidth() == 32 && fused.maxTotalThreadsPerThreadgroup() >= 128
}

const PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const PACKED_GROUPED_EXPERT_MAX_TOKENS: usize = DEEPSEEK_V4_PREFILL_MAX_TOKENS;
const PACKED_GPU_ROUTE_MAX_TOKENS: usize = 2_048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedGroupedExpertMode {
    Auto,
    ForceOn,
    ForceOff,
}

fn parse_packed_grouped_expert_mode(value: Option<&str>) -> PackedGroupedExpertMode {
    match value {
        None | Some("auto" | "AUTO") => PackedGroupedExpertMode::Auto,
        Some("1" | "true" | "TRUE" | "yes" | "YES") => PackedGroupedExpertMode::ForceOn,
        Some("0" | "false" | "FALSE" | "no" | "NO") => PackedGroupedExpertMode::ForceOff,
        Some(_) => PackedGroupedExpertMode::ForceOff,
    }
}

fn packed_grouped_expert_mode() -> PackedGroupedExpertMode {
    static MODE: OnceLock<PackedGroupedExpertMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var("QWEN_DSV4_PACKED_GROUPED_EXPERTS").ok();
        parse_packed_grouped_expert_mode(value.as_deref())
    })
}

fn packed_grouped_iq2_mma16_qualified(n_tokens: usize) -> bool {
    matches!(
        n_tokens,
        PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS
            | PACKED_GROUPED_IQ2_MMA16_MEDIUM_TOKENS
            | PACKED_GROUPED_IQ2_MMA16_WIDE_TOKENS
    )
}

fn packed_grouped_iq2_matrix_execution_chunk_qualified(n_tokens: usize) -> bool {
    n_tokens == PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS
        || packed_q8_partial_matrix_chunk_qualified(n_tokens)
}

fn packed_grouped_iq2_matrix_scope_qualified(
    device_name: &str,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    n_tokens: usize,
) -> bool {
    let partial_asset_qualified = tensor_count == 1_328
        && matches!(
            (source_bytes, expert_count),
            (
                PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
                MOE_EXPERT_COUNT
            ) | (PACKED_Q8_MATRIX_REAP_K216_SOURCE_BYTES, 216)
        );
    device_name == PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE
        && (packed_grouped_iq2_mma16_qualified(n_tokens)
            || (partial_asset_qualified && packed_q8_partial_matrix_chunk_qualified(n_tokens)))
}

fn packed_grouped_expert_scope(
    mode: PackedGroupedExpertMode,
    n_tokens: usize,
) -> Result<bool, DeepSeekV4MetalError> {
    if n_tokens <= PACKED_GROUPED_EXPERT_MAX_TOKENS {
        return Ok(true);
    }
    if mode == PackedGroupedExpertMode::ForceOn {
        return invalid(format!(
            "packed grouped experts are qualified through {PACKED_GROUPED_EXPERT_MAX_TOKENS} tokens, got {n_tokens}"
        ));
    }
    Ok(false)
}

fn packed_grouped_iq_expert_count_qualified(expert_count: usize) -> bool {
    matches!(expert_count, 216 | MOE_EXPERT_COUNT)
}

fn packed_grouped_expert_policy(
    ctx: &MetalContext,
    n_tokens: usize,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
) -> Result<PackedExpertPolicy, DeepSeekV4MetalError> {
    let device_name = ctx.device.name().to_string();
    let enabled = match packed_grouped_expert_mode() {
        PackedGroupedExpertMode::Auto => device_name == PACKED_GROUPED_EXPERT_QUALIFIED_DEVICE,
        PackedGroupedExpertMode::ForceOn => true,
        PackedGroupedExpertMode::ForceOff => false,
    } && packed_grouped_expert_kernels_supported(ctx);
    let f16_iq2 = packed_iq2_f16_mm64x32_enabled();
    let wide_iq2 = packed_iq2_mm64x32_enabled();
    let iq2_matrix_supported = if f16_iq2 {
        packed_iq2_f16_mm64x32_candidate_supported(ctx)
    } else if wide_iq2 {
        packed_iq2_mm64x32_candidate_supported(ctx)
    } else {
        packed_grouped_iq2_mma16_candidate_supported(ctx)
    };
    if packed_grouped_iq2_mma16_enabled()
        && enabled
        && packed_grouped_iq2_matrix_scope_qualified(
            &device_name,
            tensor_count,
            source_bytes,
            expert_count,
            n_tokens,
        )
        && iq2_matrix_supported
    {
        return Ok(PackedExpertPolicy::GroupedIq2XsIq3XxsMma16QualifiedChunk);
    }
    Ok(if enabled {
        PackedExpertPolicy::GroupedIq2XsIq3Xxs
    } else {
        PackedExpertPolicy::Current
    })
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
pub(super) fn packed_grouped_expert_enabled_for_test(ctx: &MetalContext) -> bool {
    packed_grouped_expert_policy(
        ctx,
        PACKED_GROUPED_IQ2_MMA16_NARROW_TOKENS,
        1_328,
        PACKED_Q8_COMPRESSOR_MATRIX_QUALIFIED_SOURCE_BYTES,
        MOE_EXPERT_COUNT,
    )
    .expect("valid packed grouped expert policy")
    .uses_iq2_target()
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
pub(super) fn packed_grouped_iq3_candidate_supported_for_test(ctx: &MetalContext) -> bool {
    packed_grouped_iq3_candidate_supported(ctx)
}

struct PackedLayerTrace {
    layer: usize,
    pre_expert_seconds: f64,
    pre_expert_gpu_seconds: f64,
    pre_expert_encode_seconds: f64,
    pre_expert_wait_seconds: f64,
    pre_expert_wait_residual_seconds: f64,
    pre_expert_post_seconds: f64,
    route_seconds: f64,
    post_route_seconds: f64,
    post_route_gpu_seconds: f64,
    post_route_encode_seconds: f64,
    post_route_wait_seconds: f64,
    post_route_wait_residual_seconds: f64,
    bucket_count: usize,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedPrefillStageKind {
    BeforeAttentionBody,
    SparseIndexerPrepare,
    SparseIndexerScore,
    SparseSelection,
    AttentionCore,
    InverseRope,
    AttentionOutputProjections,
    AfterAttentionOutput,
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) const PACKED_PREFILL_STAGE_KINDS: [PackedPrefillStageKind; 8] = [
    PackedPrefillStageKind::BeforeAttentionBody,
    PackedPrefillStageKind::SparseIndexerPrepare,
    PackedPrefillStageKind::SparseIndexerScore,
    PackedPrefillStageKind::SparseSelection,
    PackedPrefillStageKind::AttentionCore,
    PackedPrefillStageKind::InverseRope,
    PackedPrefillStageKind::AttentionOutputProjections,
    PackedPrefillStageKind::AfterAttentionOutput,
];

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillStageTiming {
    pub kind: PackedPrefillStageKind,
    pub start_timestamp: Option<u64>,
    pub end_timestamp: Option<u64>,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillStageTransition {
    pub from: PackedPrefillStageKind,
    pub to: PackedPrefillStageKind,
    pub delta_ticks: i128,
    pub gap_ticks: u64,
    pub overlap_ticks: u64,
    pub gap_ms_scaled: f64,
    pub overlap_ms_scaled: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillSampledLayerProfile {
    pub layer: usize,
    pub command_gpu_ms: f64,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
    pub encoder_gap_ms_scaled: f64,
    pub encoder_overlap_ms_scaled: f64,
    pub stages: Vec<PackedPrefillStageTiming>,
    pub transitions: Vec<PackedPrefillStageTransition>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPrefillStageProfile {
    pub sampled: bool,
    pub command_gpu_ms: Vec<f64>,
    pub sampled_layers: Vec<PackedPrefillSampledLayerProfile>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug)]
struct PackedPrefillPendingStageSample {
    layer: usize,
    kind: PackedPrefillStageKind,
    samples: Option<(usize, usize)>,
}

#[cfg(feature = "dsv4-diagnostics")]
fn resolve_packed_prefill_layer_stage_samples(
    layer: usize,
    records: &[PackedPrefillPendingStageSample],
    timestamps: &[u64],
    command_gpu_ms: f64,
    allowed_empty: &[PackedPrefillStageKind],
) -> Result<PackedPrefillSampledLayerProfile, DeepSeekV4MetalError> {
    if records.len() != PACKED_PREFILL_STAGE_KINDS.len() {
        return invalid(format!(
            "packed prefill sampled layer {layer} produced {} stages, expected {}",
            records.len(),
            PACKED_PREFILL_STAGE_KINDS.len()
        ));
    }
    if !command_gpu_ms.is_finite() || command_gpu_ms <= 0.0 {
        return invalid(format!(
            "packed prefill sampled layer {layer} has invalid command GPU duration {command_gpu_ms}"
        ));
    }
    for (record, expected) in records.iter().zip(PACKED_PREFILL_STAGE_KINDS) {
        if record.layer != layer || record.kind != expected {
            return invalid(format!(
                "packed prefill sampled layer {layer} recorded {:?} for layer {}, expected {expected:?}",
                record.kind, record.layer
            ));
        }
        match record.samples {
            Some((start_sample, end_sample)) => {
                if start_sample >= timestamps.len() || end_sample >= timestamps.len() {
                    return invalid(format!(
                        "packed prefill sampled layer {layer} stage {:?} indexes samples {start_sample}..{end_sample} from {} timestamps",
                        record.kind,
                        timestamps.len()
                    ));
                }
            }
            None => {
                if !allowed_empty.contains(&record.kind) {
                    return invalid(format!(
                        "packed prefill sampled layer {layer} has unexpected empty stage {:?}",
                        record.kind
                    ));
                }
            }
        }
    }

    let first_timestamp = records
        .iter()
        .find_map(|record| record.samples.map(|(start, _)| timestamps[start]))
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed prefill sampled layer {layer} has no physical stages"
            ))
        })?;
    let last_timestamp = records
        .iter()
        .rev()
        .find_map(|record| record.samples.map(|(_, end)| timestamps[end]))
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed prefill sampled layer {layer} has no physical stages"
            ))
        })?;
    let sampled_span_ticks = last_timestamp.checked_sub(first_timestamp).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "packed prefill sampled layer {layer} returned non-monotonic span timestamps"
        ))
    })?;
    if sampled_span_ticks == 0 {
        return invalid(format!(
            "packed prefill sampled layer {layer} returned a zero timestamp span"
        ));
    }

    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stage_ticks = 0u64;
    let mut gap_ticks = 0u64;
    let mut overlap_ticks = 0u64;
    let mut previous_start = None;
    let mut previous_end = None;
    let mut previous_kind = None;
    let mut stages = Vec::with_capacity(records.len());
    let mut transitions = Vec::with_capacity(records.len() - 1);
    for record in records {
        let Some((start_sample, end_sample)) = record.samples else {
            stages.push(PackedPrefillStageTiming {
                kind: record.kind,
                start_timestamp: None,
                end_timestamp: None,
                duration_ticks: 0,
                duration_ms_scaled: 0.0,
            });
            continue;
        };
        let start_timestamp = timestamps[start_sample];
        let end_timestamp = timestamps[end_sample];
        let duration_ticks = end_timestamp.checked_sub(start_timestamp).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed prefill sampled layer {layer} stage {:?} returned inverted timestamps",
                record.kind
            ))
        })?;
        if previous_start.is_some_and(|previous| start_timestamp < previous)
            || previous_end.is_some_and(|previous| end_timestamp < previous)
        {
            return invalid(format!(
                "packed prefill sampled layer {layer} stage {:?} reverses physical start/end order",
                record.kind
            ));
        }
        if let (Some(previous_end), Some(previous_kind)) = (previous_end, previous_kind) {
            let delta_ticks = start_timestamp as i128 - previous_end as i128;
            let (transition_gap, transition_overlap) = if delta_ticks >= 0 {
                (delta_ticks as u64, 0)
            } else {
                (0, (-delta_ticks) as u64)
            };
            gap_ticks = gap_ticks.checked_add(transition_gap).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed prefill encoder-gap tick total overflow".into(),
                )
            })?;
            overlap_ticks = overlap_ticks
                .checked_add(transition_overlap)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed prefill encoder-overlap tick total overflow".into(),
                    )
                })?;
            transitions.push(PackedPrefillStageTransition {
                from: previous_kind,
                to: record.kind,
                delta_ticks,
                gap_ticks: transition_gap,
                overlap_ticks: transition_overlap,
                gap_ms_scaled: transition_gap as f64 * scale_ms_per_tick,
                overlap_ms_scaled: transition_overlap as f64 * scale_ms_per_tick,
            });
        }
        previous_start = Some(start_timestamp);
        previous_end = Some(end_timestamp);
        previous_kind = Some(record.kind);
        stage_ticks = stage_ticks.checked_add(duration_ticks).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill sampled stage tick total overflow".into())
        })?;
        stages.push(PackedPrefillStageTiming {
            kind: record.kind,
            start_timestamp: Some(start_timestamp),
            end_timestamp: Some(end_timestamp),
            duration_ticks,
            duration_ms_scaled: duration_ticks as f64 * scale_ms_per_tick,
        });
    }
    let accounted_ticks = stage_ticks as i128 + gap_ticks as i128 - overlap_ticks as i128;
    if accounted_ticks != sampled_span_ticks as i128 {
        return invalid(format!(
            "packed prefill sampled layer {layer} stage/gap/overlap ticks {accounted_ticks} do not close span {sampled_span_ticks}"
        ));
    }
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(PackedPrefillSampledLayerProfile {
        layer,
        command_gpu_ms,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
        encoder_gap_ms_scaled: gap_ticks as f64 * scale_ms_per_tick,
        encoder_overlap_ms_scaled: overlap_ticks as f64 * scale_ms_per_tick,
        stages,
        transitions,
    })
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedPrefillStageRecorder {
    sampled: bool,
    samples: Option<MetalTimestampSampleBuffer>,
    next_sample: usize,
    records: Vec<PackedPrefillPendingStageSample>,
    command_gpu_ms: [Option<f64>; DEEPSEEK_V4_LAYER_COUNT],
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedPrefillStageRecorder {
    fn new(ctx: &MetalContext, sampled: bool) -> Result<Self, DeepSeekV4MetalError> {
        let record_count = DEEPSEEK_V4_LAYER_COUNT
            .checked_mul(PACKED_PREFILL_STAGE_KINDS.len())
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed prefill timestamp record count overflow".into(),
                )
            })?;
        let sample_count = record_count.checked_mul(2).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill timestamp sample count overflow".into())
        })?;
        Ok(Self {
            sampled,
            samples: if sampled {
                Some(ctx.timestamp_sample_buffer(sample_count)?)
            } else {
                None
            },
            next_sample: 0,
            records: Vec::with_capacity(if sampled { record_count } else { 0 }),
            command_gpu_ms: [None; DEEPSEEK_V4_LAYER_COUNT],
        })
    }

    fn begin_encoder(
        &mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        kind: PackedPrefillStageKind,
    ) -> Result<KernelEncoder, DeepSeekV4MetalError> {
        if !self.sampled || layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid("packed prefill stage recorder received an invalid sampled layer");
        }
        let start_sample = self.next_sample;
        let end_sample = start_sample.checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill sample index overflow".into())
        })?;
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill timestamp buffer is absent".into())
        })?;
        if end_sample >= samples.sample_count() {
            return invalid(format!(
                "packed prefill timestamp buffer exhausted at sample {end_sample}"
            ));
        }
        self.next_sample = end_sample + 1;
        self.records.push(PackedPrefillPendingStageSample {
            layer,
            kind,
            samples: Some((start_sample, end_sample)),
        });
        Ok(KernelEncoder::try_begin_sampled(
            command,
            samples,
            start_sample,
            end_sample,
            false,
        )?)
    }

    fn record_empty_stage(
        &mut self,
        layer: usize,
        kind: PackedPrefillStageKind,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled || layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid("packed prefill empty stage has invalid sampled layer");
        }
        let record_index = self.records.len();
        let expected_layer = record_index / PACKED_PREFILL_STAGE_KINDS.len();
        let expected_kind =
            PACKED_PREFILL_STAGE_KINDS[record_index % PACKED_PREFILL_STAGE_KINDS.len()];
        if layer != expected_layer || kind != expected_kind {
            return invalid(format!(
                "packed prefill empty stage {kind:?} for layer {layer} expected {expected_kind:?} for layer {expected_layer}"
            ));
        }
        self.records.push(PackedPrefillPendingStageSample {
            layer,
            kind,
            samples: None,
        });
        Ok(())
    }

    fn record_command_gpu_seconds(
        &mut self,
        layer: usize,
        command_gpu_seconds: f64,
    ) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT
            || !command_gpu_seconds.is_finite()
            || command_gpu_seconds <= 0.0
            || self.command_gpu_ms[layer].is_some()
        {
            return invalid(format!(
                "packed prefill layer {layer} has invalid or duplicate GPU duration {command_gpu_seconds}"
            ));
        }
        self.command_gpu_ms[layer] = Some(command_gpu_seconds * 1e3);
        Ok(())
    }

    fn resolve(
        self,
        ctx: &MetalContext,
    ) -> Result<PackedPrefillStageProfile, DeepSeekV4MetalError> {
        let command_gpu_ms = self
            .command_gpu_ms
            .into_iter()
            .enumerate()
            .map(|(layer, duration)| {
                duration.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed prefill layer {layer} has no GPU duration"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !self.sampled {
            if self.samples.is_some() || self.next_sample != 0 || !self.records.is_empty() {
                return invalid("ordinary packed prefill profile retained sampled state");
            }
            return Ok(PackedPrefillStageProfile {
                sampled: false,
                command_gpu_ms,
                sampled_layers: Vec::new(),
            });
        }
        let expected_records = DEEPSEEK_V4_LAYER_COUNT * PACKED_PREFILL_STAGE_KINDS.len();
        let expected_samples = self
            .records
            .iter()
            .filter(|record| record.samples.is_some())
            .count()
            * 2;
        if self.records.len() != expected_records || self.next_sample != expected_samples {
            return invalid(format!(
                "packed prefill stage recorder produced {} records/{} samples, expected {expected_records}/{expected_samples}",
                self.records.len(),
                self.next_sample
            ));
        }
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed prefill timestamp buffer is absent".into())
        })?;
        let timestamps = ctx.resolve_timestamp_samples(samples, self.next_sample)?;
        let mut sampled_layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        for (layer, &duration) in command_gpu_ms.iter().enumerate() {
            let records = &self.records[layer * PACKED_PREFILL_STAGE_KINDS.len()
                ..(layer + 1) * PACKED_PREFILL_STAGE_KINDS.len()];
            sampled_layers.push(resolve_packed_prefill_layer_stage_samples(
                layer,
                records,
                &timestamps,
                duration,
                &[
                    PackedPrefillStageKind::SparseIndexerPrepare,
                    PackedPrefillStageKind::SparseIndexerScore,
                    PackedPrefillStageKind::SparseSelection,
                ],
            )?);
        }
        Ok(PackedPrefillStageProfile {
            sampled: true,
            command_gpu_ms,
            sampled_layers,
        })
    }
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedPrefillLayerEncoder<'command, 'recorder> {
    command: &'command Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    layer: usize,
    sampled: bool,
    recorder: Option<&'recorder mut PackedPrefillStageRecorder>,
    encoder: Option<KernelEncoder>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl<'command, 'recorder> PackedPrefillLayerEncoder<'command, 'recorder> {
    fn begin(
        command: &'command Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        mut recorder: Option<&'recorder mut PackedPrefillStageRecorder>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let sampled = recorder.as_deref().is_some_and(|recorder| recorder.sampled);
        let encoder = if sampled {
            recorder.as_deref_mut().unwrap().begin_encoder(
                command,
                layer,
                PackedPrefillStageKind::BeforeAttentionBody,
            )?
        } else {
            KernelEncoder::begin(command)
        };
        Ok(Self {
            command,
            layer,
            sampled,
            recorder,
            encoder: Some(encoder),
        })
    }

    fn boundary(&mut self, next: PackedPrefillStageKind) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
        self.encoder = Some(
            self.recorder
                .as_deref_mut()
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "sampled packed prefill encoder lost its recorder".into(),
                    )
                })?
                .begin_encoder(self.command, self.layer, next)?,
        );
        Ok(())
    }

    fn skip_stages(
        &mut self,
        skipped: &[PackedPrefillStageKind],
        next: PackedPrefillStageKind,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
        let recorder = self.recorder.as_deref_mut().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("sampled packed prefill encoder lost its recorder".into())
        })?;
        for &kind in skipped {
            recorder.record_empty_stage(self.layer, kind)?;
        }
        self.encoder = Some(recorder.begin_encoder(self.command, self.layer, next)?);
        Ok(())
    }

    fn end(mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
impl std::ops::Deref for PackedPrefillLayerEncoder<'_, '_> {
    type Target = KernelEncoder;

    fn deref(&self) -> &Self::Target {
        self.encoder
            .as_ref()
            .expect("packed prefill layer encoder ended before stage completion")
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackedPostRouteStageKind {
    RoutedExperts,
    RoutedGateUp,
    RoutedSwiGlu,
    RoutedDown,
    SharedExpert,
    ExpertCombine,
    HyperPostAndHead,
}

#[cfg(feature = "dsv4-diagnostics")]
pub(super) const PACKED_POST_ROUTE_STAGE_KINDS: [PackedPostRouteStageKind; 4] = [
    PackedPostRouteStageKind::RoutedExperts,
    PackedPostRouteStageKind::SharedExpert,
    PackedPostRouteStageKind::ExpertCombine,
    PackedPostRouteStageKind::HyperPostAndHead,
];

#[cfg(feature = "dsv4-diagnostics")]
const PACKED_BM16_POST_ROUTE_STAGE_KINDS: [PackedPostRouteStageKind; 6] = [
    PackedPostRouteStageKind::RoutedGateUp,
    PackedPostRouteStageKind::RoutedSwiGlu,
    PackedPostRouteStageKind::RoutedDown,
    PackedPostRouteStageKind::SharedExpert,
    PackedPostRouteStageKind::ExpertCombine,
    PackedPostRouteStageKind::HyperPostAndHead,
];

#[cfg(feature = "dsv4-diagnostics")]
fn packed_post_route_stage_kinds(split_routed: bool) -> &'static [PackedPostRouteStageKind] {
    if split_routed {
        &PACKED_BM16_POST_ROUTE_STAGE_KINDS
    } else {
        &PACKED_POST_ROUTE_STAGE_KINDS
    }
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_post_route_expert_counts(
    n_tokens: usize,
    schedule: &[ExpertBucket],
    expert_count: usize,
) -> Result<[u16; MOE_EXPERT_COUNT], DeepSeekV4MetalError> {
    checked_token_count(n_tokens)?;
    validate_packed_expert_count(expert_count)?;
    let expected = checked_mul(n_tokens, MOE_TOP_K, "packed expert-count coverage")?;
    let mut counts = [0u16; MOE_EXPERT_COUNT];
    let mut covered = 0usize;
    let mut previous_expert = None;
    for bucket in schedule {
        if bucket.expert >= expert_count
            || bucket.len == 0
            || bucket.start != covered
            || previous_expert.is_some_and(|previous| bucket.expert <= previous)
        {
            return invalid("packed expert-count schedule is not canonical");
        }
        counts[bucket.expert] = u16::try_from(bucket.len)
            .map_err(|_| DeepSeekV4MetalError::Invalid("packed expert count exceeds u16".into()))?;
        covered = covered
            .checked_add(bucket.len)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed expert count overflow".into()))?;
        previous_expert = Some(bucket.expert);
    }
    if covered != expected {
        return invalid(format!(
            "packed expert-count coverage {covered} does not match {expected}"
        ));
    }
    Ok(counts)
}

#[cfg(feature = "dsv4-diagnostics")]
fn packed_post_route_expert_ids(
    n_tokens: usize,
    expert_count: usize,
    expert_ids: &[i32],
    bucket_rows: &[i32],
    bucket_slots: &[i32],
    schedule: &[ExpertBucket],
) -> Result<Vec<u16>, DeepSeekV4MetalError> {
    validate_packed_expert_schedule(
        n_tokens,
        expert_count,
        expert_ids,
        bucket_rows,
        bucket_slots,
        schedule,
    )?;
    let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed route-ID coverage")?;
    let mut reconstructed = vec![u16::MAX; route_count];
    for bucket in schedule {
        for &stored_slot in &bucket_slots[bucket.start..bucket.start + bucket.len] {
            let slot = usize::try_from(stored_slot).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed route-ID slot is negative".into())
            })?;
            if slot >= route_count || reconstructed[slot] != u16::MAX {
                return invalid("packed route-ID slot coverage is invalid");
            }
            reconstructed[slot] = u16::try_from(bucket.expert).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed route-ID expert exceeds u16".into())
            })?;
        }
    }
    if reconstructed.contains(&u16::MAX) {
        return invalid("packed route-ID capture omitted a slot");
    }
    for (&reconstructed, &stored) in reconstructed.iter().zip(expert_ids) {
        if usize::from(reconstructed) >= expert_count || i32::from(reconstructed) != stored {
            return invalid("packed route-ID capture differs from routed storage");
        }
    }

    let mut rebuilt_rows = Vec::with_capacity(route_count);
    let mut rebuilt_slots = Vec::with_capacity(route_count);
    let mut rebuilt_schedule = Vec::with_capacity(schedule.len());
    for expert in 0..expert_count {
        let start = rebuilt_slots.len();
        for (slot, &routed_expert) in reconstructed.iter().enumerate() {
            if usize::from(routed_expert) == expert {
                rebuilt_rows.push((slot / MOE_TOP_K) as i32);
                rebuilt_slots.push(slot as i32);
            }
        }
        if rebuilt_slots.len() > start {
            rebuilt_schedule.push(ExpertBucket {
                expert,
                start,
                len: rebuilt_slots.len() - start,
            });
        }
    }
    if rebuilt_rows != bucket_rows
        || rebuilt_slots != bucket_slots
        || rebuilt_schedule.len() != schedule.len()
    {
        return invalid("packed route-ID capture does not rebuild the routed schedule");
    }
    for (rebuilt, original) in rebuilt_schedule.iter().zip(schedule) {
        if rebuilt.expert != original.expert
            || rebuilt.start != original.start
            || rebuilt.len != original.len
        {
            return invalid("packed route-ID capture changes expert bucket geometry");
        }
    }
    Ok(reconstructed)
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedPostRouteLayerMetadata {
    pub layer: usize,
    pub gate_dtype: GgmlType,
    pub up_dtype: GgmlType,
    pub down_dtype: GgmlType,
    pub bucket_count: usize,
    pub expert_counts: [u16; MOE_EXPERT_COUNT],
    pub route_expert_ids: Vec<u16>,
    pub route_weight_bits: Vec<u32>,
    pub grouped_q3q4: bool,
    pub grouped_iq2: bool,
    pub grouped_iq3: bool,
    pub bm16: bool,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPostRouteStageTiming {
    pub kind: PackedPostRouteStageKind,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPostRouteSampledLayerProfile {
    pub layer: usize,
    pub command_gpu_ms: f64,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
    pub encoder_gap_ms_scaled: f64,
    pub encoder_overlap_ms_scaled: f64,
    pub stages: Vec<PackedPostRouteStageTiming>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedPostRouteStageProfile {
    pub sampled: bool,
    pub q8_compressor_matrix_invocations: u32,
    pub command_gpu_ms: Vec<f64>,
    pub metadata: Vec<PackedPostRouteLayerMetadata>,
    pub sampled_layers: Vec<PackedPostRouteSampledLayerProfile>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Debug)]
pub struct PackedChunkProfile {
    pub pre_expert: PackedPrefillStageProfile,
    pub post_route: PackedPostRouteStageProfile,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug)]
struct PackedPostRoutePendingStageSample {
    layer: usize,
    kind: PackedPostRouteStageKind,
    start_sample: usize,
    end_sample: usize,
}

#[cfg(feature = "dsv4-diagnostics")]
fn resolve_packed_post_route_layer_stage_samples(
    layer: usize,
    records: &[PackedPostRoutePendingStageSample],
    expected_kinds: &[PackedPostRouteStageKind],
    timestamps: &[u64],
    command_gpu_ms: f64,
) -> Result<PackedPostRouteSampledLayerProfile, DeepSeekV4MetalError> {
    if records.len() != expected_kinds.len() {
        return invalid(format!(
            "packed post-route sampled layer {layer} produced {} stages, expected {}",
            records.len(),
            expected_kinds.len()
        ));
    }
    if !command_gpu_ms.is_finite() || command_gpu_ms <= 0.0 {
        return invalid(format!(
            "packed post-route sampled layer {layer} has invalid command GPU duration {command_gpu_ms}"
        ));
    }
    for (record, &expected) in records.iter().zip(expected_kinds) {
        if record.layer != layer || record.kind != expected {
            return invalid(format!(
                "packed post-route sampled layer {layer} recorded {:?} for layer {}, expected {expected:?}",
                record.kind, record.layer
            ));
        }
        if record.start_sample >= timestamps.len() || record.end_sample >= timestamps.len() {
            return invalid(format!(
                "packed post-route sampled layer {layer} stage {:?} indexes samples {}..{} from {} timestamps",
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
            "packed post-route sampled layer {layer} returned non-monotonic span timestamps"
        ))
    })?;
    if sampled_span_ticks == 0 {
        return invalid(format!(
            "packed post-route sampled layer {layer} returned a zero timestamp span"
        ));
    }

    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stage_ticks = 0u64;
    let mut gap_ticks = 0u64;
    let mut overlap_ticks = 0u64;
    let mut previous_start = None;
    let mut previous_end = None;
    let mut stages = Vec::with_capacity(records.len());
    for record in records {
        let start_timestamp = timestamps[record.start_sample];
        let end_timestamp = timestamps[record.end_sample];
        let duration_ticks = end_timestamp.checked_sub(start_timestamp).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed post-route sampled layer {layer} stage {:?} returned inverted timestamps",
                record.kind
            ))
        })?;
        if previous_start.is_some_and(|previous| start_timestamp < previous)
            || previous_end.is_some_and(|previous| end_timestamp < previous)
        {
            return invalid(format!(
                "packed post-route sampled layer {layer} stage {:?} reverses physical start/end order",
                record.kind
            ));
        }
        if let Some(previous_end) = previous_end {
            if start_timestamp >= previous_end {
                gap_ticks = gap_ticks
                    .checked_add(start_timestamp - previous_end)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "packed post-route encoder-gap tick total overflow".into(),
                        )
                    })?;
            } else {
                overlap_ticks = overlap_ticks
                    .checked_add(previous_end - start_timestamp)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "packed post-route encoder-overlap tick total overflow".into(),
                        )
                    })?;
            }
        }
        previous_start = Some(start_timestamp);
        previous_end = Some(end_timestamp);
        stage_ticks = stage_ticks.checked_add(duration_ticks).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed post-route sampled stage tick total overflow".into(),
            )
        })?;
        stages.push(PackedPostRouteStageTiming {
            kind: record.kind,
            start_timestamp,
            end_timestamp,
            duration_ticks,
            duration_ms_scaled: duration_ticks as f64 * scale_ms_per_tick,
        });
    }
    let accounted_ticks = stage_ticks as i128 + gap_ticks as i128 - overlap_ticks as i128;
    if accounted_ticks != sampled_span_ticks as i128 {
        return invalid(format!(
            "packed post-route sampled layer {layer} stage/gap/overlap ticks {accounted_ticks} do not close span {sampled_span_ticks}"
        ));
    }
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(PackedPostRouteSampledLayerProfile {
        layer,
        command_gpu_ms,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
        encoder_gap_ms_scaled: gap_ticks as f64 * scale_ms_per_tick,
        encoder_overlap_ms_scaled: overlap_ticks as f64 * scale_ms_per_tick,
        stages,
    })
}

#[cfg(feature = "dsv4-diagnostics")]
struct PackedPostRouteStageRecorder {
    sampled: bool,
    samples: Option<MetalTimestampSampleBuffer>,
    next_sample: usize,
    records: Vec<PackedPostRoutePendingStageSample>,
    command_gpu_ms: [Option<f64>; DEEPSEEK_V4_LAYER_COUNT],
    metadata: [Option<PackedPostRouteLayerMetadata>; DEEPSEEK_V4_LAYER_COUNT],
}

#[cfg(feature = "dsv4-diagnostics")]
impl PackedPostRouteStageRecorder {
    fn new(ctx: &MetalContext, sampled: bool) -> Result<Self, DeepSeekV4MetalError> {
        let record_count = DEEPSEEK_V4_LAYER_COUNT
            .checked_mul(PACKED_BM16_POST_ROUTE_STAGE_KINDS.len())
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "packed post-route timestamp record count overflow".into(),
                )
            })?;
        let sample_count = record_count.checked_mul(2).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed post-route timestamp sample count overflow".into(),
            )
        })?;
        Ok(Self {
            sampled,
            samples: if sampled {
                Some(ctx.timestamp_sample_buffer(sample_count)?)
            } else {
                None
            },
            next_sample: 0,
            records: Vec::with_capacity(if sampled { record_count } else { 0 }),
            command_gpu_ms: [None; DEEPSEEK_V4_LAYER_COUNT],
            metadata: std::array::from_fn(|_| None),
        })
    }

    fn begin_encoder(
        &mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        kind: PackedPostRouteStageKind,
    ) -> Result<KernelEncoder, DeepSeekV4MetalError> {
        if !self.sampled || layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid("packed post-route stage recorder received an invalid sampled layer");
        }
        let start_sample = self.next_sample;
        let end_sample = start_sample.checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed post-route sample index overflow".into())
        })?;
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed post-route timestamp buffer is absent".into())
        })?;
        if end_sample >= samples.sample_count() {
            return invalid(format!(
                "packed post-route timestamp buffer exhausted at sample {end_sample}"
            ));
        }
        self.next_sample = end_sample + 1;
        self.records.push(PackedPostRoutePendingStageSample {
            layer,
            kind,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::try_begin_sampled(
            command,
            samples,
            start_sample,
            end_sample,
            false,
        )?)
    }

    fn record_layer(
        &mut self,
        metadata: PackedPostRouteLayerMetadata,
    ) -> Result<(), DeepSeekV4MetalError> {
        let layer = metadata.layer;
        if layer >= DEEPSEEK_V4_LAYER_COUNT || self.metadata[layer].replace(metadata).is_some() {
            return invalid(format!(
                "packed post-route layer {layer} has invalid or duplicate metadata"
            ));
        }
        Ok(())
    }

    fn record_command_gpu_seconds(
        &mut self,
        layer: usize,
        command_gpu_seconds: f64,
    ) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT
            || !command_gpu_seconds.is_finite()
            || command_gpu_seconds <= 0.0
            || self.command_gpu_ms[layer].is_some()
        {
            return invalid(format!(
                "packed post-route layer {layer} has invalid or duplicate GPU duration {command_gpu_seconds}"
            ));
        }
        self.command_gpu_ms[layer] = Some(command_gpu_seconds * 1e3);
        Ok(())
    }

    fn resolve(
        self,
        ctx: &MetalContext,
    ) -> Result<PackedPostRouteStageProfile, DeepSeekV4MetalError> {
        let command_gpu_ms = self
            .command_gpu_ms
            .into_iter()
            .enumerate()
            .map(|(layer, duration)| {
                duration.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed post-route layer {layer} has no GPU duration"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let metadata = self
            .metadata
            .into_iter()
            .enumerate()
            .map(|(layer, metadata)| {
                metadata.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed post-route layer {layer} has no metadata"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !self.sampled {
            if self.samples.is_some() || self.next_sample != 0 || !self.records.is_empty() {
                return invalid("ordinary packed post-route profile retained sampled state");
            }
            return Ok(PackedPostRouteStageProfile {
                sampled: false,
                q8_compressor_matrix_invocations: 0,
                command_gpu_ms,
                metadata,
                sampled_layers: Vec::new(),
            });
        }
        let expected_records = metadata
            .iter()
            .map(|metadata| {
                packed_post_route_stage_kinds(metadata.bm16 || metadata.grouped_q3q4).len()
            })
            .sum::<usize>();
        let expected_samples = expected_records * 2;
        if self.records.len() != expected_records || self.next_sample != expected_samples {
            return invalid(format!(
                "packed post-route stage recorder produced {} records/{} samples, expected {expected_records}/{expected_samples}",
                self.records.len(),
                self.next_sample
            ));
        }
        let samples = self.samples.as_ref().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed post-route timestamp buffer is absent".into())
        })?;
        let timestamps = ctx.resolve_timestamp_samples(samples, self.next_sample)?;
        let mut sampled_layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        let mut record_cursor = 0usize;
        for (layer, &duration) in command_gpu_ms.iter().enumerate() {
            let expected_kinds =
                packed_post_route_stage_kinds(metadata[layer].bm16 || metadata[layer].grouped_q3q4);
            let record_end = record_cursor + expected_kinds.len();
            let records = &self.records[record_cursor..record_end];
            sampled_layers.push(resolve_packed_post_route_layer_stage_samples(
                layer,
                records,
                expected_kinds,
                &timestamps,
                duration,
            )?);
            record_cursor = record_end;
        }
        Ok(PackedPostRouteStageProfile {
            sampled: true,
            q8_compressor_matrix_invocations: 0,
            command_gpu_ms,
            metadata,
            sampled_layers,
        })
    }
}

struct PackedPostRouteLayerEncoder<'a> {
    _command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    #[cfg(feature = "dsv4-diagnostics")]
    layer: usize,
    #[cfg(feature = "dsv4-diagnostics")]
    sampled: bool,
    #[cfg(feature = "dsv4-diagnostics")]
    split_routed: bool,
    #[cfg(feature = "dsv4-diagnostics")]
    recorder: Option<&'a mut PackedPostRouteStageRecorder>,
    encoder: Option<KernelEncoder>,
}

impl<'a> PackedPostRouteLayerEncoder<'a> {
    #[cfg(feature = "dsv4-diagnostics")]
    fn begin(
        command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        split_routed: bool,
        mut recorder: Option<&'a mut PackedPostRouteStageRecorder>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let sampled = recorder.as_deref().is_some_and(|recorder| recorder.sampled);
        let encoder = if sampled {
            recorder.as_deref_mut().unwrap().begin_encoder(
                command,
                layer,
                if split_routed {
                    PackedPostRouteStageKind::RoutedGateUp
                } else {
                    PackedPostRouteStageKind::RoutedExperts
                },
            )?
        } else {
            KernelEncoder::begin(command)
        };
        Ok(Self {
            _command: command,
            layer,
            sampled,
            split_routed,
            recorder,
            encoder: Some(encoder),
        })
    }

    #[cfg(not(feature = "dsv4-diagnostics"))]
    fn begin(
        command: &'a Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Ok(Self {
            _command: command,
            encoder: Some(KernelEncoder::begin(command)),
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn boundary(&mut self, next: PackedPostRouteStageKind) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
        self.encoder = Some(
            self.recorder
                .as_deref_mut()
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "sampled packed post-route encoder lost its recorder".into(),
                    )
                })?
                .begin_encoder(self._command, self.layer, next)?,
        );
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn splits_bm16_stages(&self) -> bool {
        self.sampled && self.split_routed
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn splits_routed_stages(&self) -> bool {
        self.sampled && self.split_routed
    }

    fn end(mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.end();
        }
    }
}

struct CommittedPackedCommand {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl Drop for CommittedPackedCommand {
    fn drop(&mut self) {
        // Lifetime guard only: keeps buffers alive until the GPU is done.
        // Completion status is checked on the normal path before readback.
        crate::metal::wait_unchecked(&self.command);
    }
}

impl std::ops::Deref for PackedPostRouteLayerEncoder<'_> {
    type Target = KernelEncoder;

    fn deref(&self) -> &Self::Target {
        self.encoder
            .as_ref()
            .expect("packed post-route layer encoder ended before stage completion")
    }
}

#[derive(Clone, Copy)]
enum PackedRouteSource<'a> {
    Hash,
    Learned(&'a MetalTensor),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedRoutePolicy {
    Cpu,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    CpuNoCompactPromotion,
    GpuCompact,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    GpuExperimental,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    GpuExperimentalCpuWeights,
}

impl PackedRoutePolicy {
    #[cfg(feature = "dsv4-diagnostics")]
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::CpuNoCompactPromotion => "cpu_no_compact_promotion",
            Self::GpuCompact => "gpu_compact",
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::GpuExperimental => "gpu_experimental",
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::GpuExperimentalCpuWeights => "gpu_experimental_cpu_weights",
        }
    }

    fn uses_gpu(self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::CpuNoCompactPromotion => false,
            Self::GpuCompact => true,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::GpuExperimental => true,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            Self::GpuExperimentalCpuWeights => true,
        }
    }
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
fn packed_hash_route_has_duplicate_slots(
    token_ids: &[u32],
    token_to_expert: &MetalTensor,
    expert_count: usize,
) -> Result<bool, DeepSeekV4MetalError> {
    validate_i32_bank(
        token_to_expert,
        MOE_TOP_K,
        "packed diagnostic hash route map",
    )?;
    validate_packed_expert_count(expert_count)?;
    let vocab_size = usize::try_from(token_to_expert.shape[1]).map_err(|_| {
        DeepSeekV4MetalError::Invalid("packed diagnostic hash vocabulary exceeds usize".into())
    })?;
    // Bound validation above and immutable Shared model storage make selected
    // row reads safe without copying the complete multi-megabyte hash map into
    // the timing-sensitive diagnostic packet.
    let map = unsafe {
        let pointer = token_to_expert
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(token_to_expert.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts(pointer, token_to_expert.n_elements() as usize)
    };
    for (token_index, &token_id) in token_ids.iter().enumerate() {
        let token_id = usize::try_from(token_id).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!(
                "packed diagnostic token {token_index} exceeds usize"
            ))
        })?;
        if token_id >= vocab_size {
            return invalid(format!(
                "packed diagnostic token {token_index} id {token_id} exceeds {vocab_size}"
            ));
        }
        let start = checked_mul(token_id, MOE_TOP_K, "packed diagnostic hash row")?;
        let mut seen = [false; MOE_EXPERT_COUNT];
        for slot in 0..MOE_TOP_K {
            let expert = usize::try_from(map[start + slot]).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed diagnostic token {token_index} slot {slot} has a negative expert"
                ))
            })?;
            if expert >= expert_count {
                return invalid(format!(
                    "packed diagnostic token {token_index} slot {slot} expert {expert} exceeds {expert_count}"
                ));
            }
            if std::mem::replace(&mut seen[expert], true) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
fn packed_diagnostic_route_policy(
    gpu_route: bool,
    preserve_cpu_weights: bool,
    duplicate_hash_routes: bool,
) -> Result<PackedRoutePolicy, DeepSeekV4MetalError> {
    if duplicate_hash_routes {
        return Ok(PackedRoutePolicy::CpuNoCompactPromotion);
    }
    match (gpu_route, preserve_cpu_weights) {
        (false, false) => Ok(PackedRoutePolicy::Cpu),
        (true, false) => Ok(PackedRoutePolicy::GpuExperimental),
        (true, true) => Ok(PackedRoutePolicy::GpuExperimentalCpuWeights),
        (false, true) => invalid("packed route cannot preserve CPU weights without GPU scheduling"),
    }
}

#[cfg(feature = "dsv4-diagnostics")]
fn validate_packed_route_policy_scope(
    policy: PackedRoutePolicy,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if policy.uses_gpu() && n_tokens > PACKED_GPU_ROUTE_MAX_TOKENS {
        return invalid(format!(
            "experimental packed GPU routing is qualified through {PACKED_GPU_ROUTE_MAX_TOKENS} tokens"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedExpertPolicy {
    Current,
    GroupedIq2XsIq3Xxs,
    GroupedIq2XsIq3XxsMma16QualifiedChunk,
    GroupedIq2XsIq3XxsAndIq3Xxs,
    GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk,
}

impl PackedExpertPolicy {
    #[cfg(feature = "dsv4-diagnostics")]
    fn label(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::GroupedIq2XsIq3Xxs => "grouped_iq2_iq3",
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk => "grouped_iq2_iq3_mma16",
            Self::GroupedIq2XsIq3XxsAndIq3Xxs => "grouped_iq2_iq3_and_all_iq3",
            Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk => {
                "grouped_iq2_iq3_mma16_and_all_iq3"
            }
        }
    }

    fn uses_iq2_target(self) -> bool {
        match self {
            Self::Current => false,
            Self::GroupedIq2XsIq3Xxs => true,
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk => true,
            Self::GroupedIq2XsIq3XxsAndIq3Xxs => true,
            Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk => true,
        }
    }

    fn uses_iq2_mma16(self, n_tokens: usize) -> bool {
        matches!(
            self,
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk
                | Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
        ) && packed_grouped_iq2_matrix_execution_chunk_qualified(n_tokens)
    }

    fn uses_iq3_target(self) -> bool {
        matches!(
            self,
            Self::GroupedIq2XsIq3XxsAndIq3Xxs
                | Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
        )
    }

    fn with_iq3_target(self) -> Self {
        match self {
            Self::GroupedIq2XsIq3Xxs => Self::GroupedIq2XsIq3XxsAndIq3Xxs,
            Self::GroupedIq2XsIq3XxsMma16QualifiedChunk => {
                Self::GroupedIq2XsIq3XxsMma16AndIq3XxsQualifiedChunk
            }
            other => other,
        }
    }
}

fn packed_gpu_compact_expert_layer_qualified(
    ctx: &MetalContext,
    policy: PackedExpertPolicy,
    n_tokens: usize,
    allow_iq3_route: bool,
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    (policy.uses_iq2_mma16(n_tokens)
        && gate_dtype == GgmlType::IQ2_XS
        && up_dtype == GgmlType::IQ2_XS
        && down_dtype == GgmlType::IQ3_XXS
        && packed_grouped_expert_kernels_supported(ctx))
        || (allow_iq3_route
            && policy.uses_iq3_target()
            && gate_dtype == GgmlType::IQ3_XXS
            && up_dtype == GgmlType::IQ3_XXS
            && down_dtype == GgmlType::IQ3_XXS
            && packed_grouped_iq3_candidate_supported(ctx))
}

impl PrefillMoeScratch {
    fn gpu_route_buffers<'a>(
        &'a self,
        logits: &'a MetalTensor,
        token_ids: &'a MetalTensor,
    ) -> PackedGpuRouteBuffers<'a> {
        PackedGpuRouteBuffers {
            logits,
            token_ids,
            expert_ids: &self.expert_ids,
            weights: &self.weights,
            route_generations: &self.gpu_route.route_generations,
            route_status: &self.gpu_route.route_status,
            counts: &self.gpu_route.counts,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            slot_ids: &self.gpu_route.slot_ids,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            schedule_generations: &self.gpu_route.schedule_generations,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            aggregate: &self.gpu_route.aggregate,
            #[cfg(any(test, feature = "dsv4-diagnostics"))]
            signature: &self.gpu_route.signature,
            compact_header: &self.gpu_route.compact_header,
        }
    }

    fn take_gpu_route_generation(&self) -> Result<NonZeroU32, DeepSeekV4MetalError> {
        let generation =
            NonZeroU32::new(self.gpu_route.next_generation.get()).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("packed GPU route generation reached zero".into())
            })?;
        let next = generation.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "packed GPU route generation exhausted before wrap".into(),
            )
        })?;
        self.gpu_route.next_generation.set(next);
        Ok(generation)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gpu_route_compact(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        token_ids: &MetalTensor,
        token_to_expert: Option<&MetalTensor>,
        n_tokens: usize,
        routed_scale: f32,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let buffers = self.gpu_route_buffers(&views.logits, token_ids);
        match source {
            PackedRouteSource::Hash => buffers.encode_hash(
                ctx,
                enc,
                token_to_expert.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed compact hash route has no token-to-expert map".into(),
                    )
                })?,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
            PackedRouteSource::Learned(bias) => buffers.encode_learned(
                ctx,
                enc,
                bias,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
        }
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed compact route count")?;
        let rows = i32_prefix(
            &self.bucket_rows,
            vec![route_count as u64],
            "packed compact route rows",
        )?;
        let slots = i32_prefix(
            &self.bucket_slots,
            vec![route_count as u64],
            "packed compact route slots",
        )?;
        buffers.encode_compact(
            ctx,
            enc,
            &rows,
            &slots,
            &self.grouped_tiles,
            &self.grouped_iq2_mma16_tiles,
            n_tokens,
            generation,
        )
    }

    fn capture_gpu_compact_schedule(
        &self,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<Vec<ExpertBucket>, DeepSeekV4MetalError> {
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed compact route count")?;
        let header = host_read_i32(
            &self.gpu_route.compact_header,
            "packed compact route header",
        )?;
        let expected_completion = packed_route_compact_completion(generation.get(), n_tokens);
        if header.len() != PACKED_COMPACT_ROUTE_HEADER_WIDTH
            || header[0] != generation.get() as i32
            || header[1] != PACKED_COMPACT_ROUTE_STATUS_READY
            || header[2] != route_count as i32
            || header[3] < 0
            || header[3] as usize > MOE_EXPERT_COUNT
            || header[4] < 0
            || header[4] as usize > PACKED_GROUPED_IQ2_MMA16_MAX_TILES
            || header[5] < 0
            || header[5] as usize > PACKED_GROUPED_EXPERT_MAX_TILES
            || header[6] as u32 != expected_completion
            || header[7] != n_tokens as i32
        {
            return invalid(format!(
                "packed compact route header {header:?} is invalid for generation {} and N={n_tokens}",
                generation.get(),
            ));
        }
        let counts = host_read_i32(&self.gpu_route.counts, "packed compact expert counts")?;
        if counts.len() != MOE_EXPERT_COUNT {
            return invalid("packed compact expert counts have invalid length");
        }
        let mut cursor = 0usize;
        let mut schedule = Vec::with_capacity(header[3].max(0) as usize);
        let mut tile16_count = 0usize;
        let mut tile32_count = 0usize;
        for (expert, count) in counts.into_iter().enumerate() {
            let count = usize::try_from(count).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed compact expert {expert} has negative count"
                ))
            })?;
            if count > route_count {
                return invalid(format!(
                    "packed compact expert {expert} count {count} exceeds {route_count} routes"
                ));
            }
            if count != 0 {
                schedule.push(ExpertBucket {
                    expert,
                    start: cursor,
                    len: count,
                });
                tile16_count += count.div_ceil(PACKED_GROUPED_IQ2_MMA16_TILE_ROWS);
                tile32_count += count.div_ceil(PACKED_GROUPED_EXPERT_TILE_ROWS);
            }
            cursor = cursor.checked_add(count).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("packed compact route count overflow".into())
            })?;
        }
        if cursor != route_count
            || schedule.len() != header[3] as usize
            || tile16_count != header[4] as usize
            || tile32_count != header[5] as usize
        {
            return invalid(format!(
                "packed compact schedule has {cursor} routes, {} experts, {tile16_count} 16-row tiles, and {tile32_count} 32-row tiles",
                schedule.len(),
            ));
        }
        Ok(schedule)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_shared_expert(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        normalized_input: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        shared_matrix: bool,
        shared_clamp: f32,
        n_tokens: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_shared_expert_batch")?;
        checked_token_count(n_tokens)?;
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("packed shared-expert clamp must be finite and positive");
        }
        let gate = f32_prefix(
            &self.gate,
            vec![MOE_FFN_SIZE as u64, n_tokens as u64],
            "packed shared gate",
        )?;
        let up = f32_prefix(
            &self.up,
            vec![MOE_FFN_SIZE as u64, n_tokens as u64],
            "packed shared up",
        )?;
        let inner = f32_prefix(
            &self.inner,
            vec![MOE_FFN_SIZE as u64, n_tokens as u64],
            "packed shared inner",
        )?;
        let shared_output = f32_prefix(
            &self.shared_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed shared output",
        )?;
        if shared_matrix && !packed_q8_matrix_execution_chunk_qualified(n_tokens) {
            return invalid("packed shared Q8 matrix policy reached an unsupported chunk");
        }
        if shared_matrix && shared_gate.dtype == GgmlType::Q8_0 {
            if n_tokens.is_multiple_of(128) {
                encode_q8_f32_mma_r2c16k64(
                    ctx,
                    enc,
                    shared_gate,
                    normalized_input,
                    &gate,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    n_tokens,
                )?;
            } else {
                encode_q8_f32_mma_r2c4k64(
                    ctx,
                    enc,
                    shared_gate,
                    normalized_input,
                    &gate,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    n_tokens,
                )?;
            }
        } else {
            encode_batch_projection(
                ctx,
                enc,
                shared_gate,
                normalized_input,
                &gate,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                n_tokens,
                "packed shared gate",
            )?;
        }
        if shared_matrix && shared_up.dtype == GgmlType::Q8_0 {
            if n_tokens.is_multiple_of(128) {
                encode_q8_f32_mma_r2c16k64(
                    ctx,
                    enc,
                    shared_up,
                    normalized_input,
                    &up,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    n_tokens,
                )?;
            } else {
                encode_q8_f32_mma_r2c4k64(
                    ctx,
                    enc,
                    shared_up,
                    normalized_input,
                    &up,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    n_tokens,
                )?;
            }
        } else {
            encode_batch_projection(
                ctx,
                enc,
                shared_up,
                normalized_input,
                &up,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                n_tokens,
                "packed shared up",
            )?;
        }
        let flat_len = checked_mul(n_tokens, MOE_FFN_SIZE, "packed shared SwiGLU")?;
        encode_ds4_clamped_swiglu(
            ctx,
            enc,
            &gate.view_subrange(0, vec![flat_len as u64]),
            &up.view_subrange(0, vec![flat_len as u64]),
            &inner.view_subrange(0, vec![flat_len as u64]),
            shared_clamp,
        )?;
        if shared_matrix && shared_down.dtype == GgmlType::Q8_0 {
            if n_tokens.is_multiple_of(128) {
                encode_q8_f32_mma_r2c16k64(
                    ctx,
                    enc,
                    shared_down,
                    &inner,
                    &shared_output,
                    MOE_FFN_SIZE,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    n_tokens,
                )?;
            } else {
                encode_q8_f32_mma_r2c4k64(
                    ctx,
                    enc,
                    shared_down,
                    &inner,
                    &shared_output,
                    MOE_FFN_SIZE,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    n_tokens,
                )?;
            }
        } else {
            encode_batch_projection(
                ctx,
                enc,
                shared_down,
                &inner,
                &shared_output,
                MOE_FFN_SIZE,
                DEEPSEEK_V4_HIDDEN_SIZE,
                n_tokens,
                "packed shared down",
            )?;
        }
        Ok(shared_output)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "dsv4-diagnostics")]
    fn encode_gpu_route_schedule(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        token_ids: &MetalTensor,
        token_to_expert: Option<&MetalTensor>,
        n_tokens: usize,
        routed_scale: f32,
        generation: NonZeroU32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let buffers = self.gpu_route_buffers(&views.logits, token_ids);
        match source {
            PackedRouteSource::Hash => buffers.encode_hash(
                ctx,
                enc,
                token_to_expert.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed GPU hash route has no token-to-expert map".into(),
                    )
                })?,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
            PackedRouteSource::Learned(bias) => buffers.encode_learned(
                ctx,
                enc,
                bias,
                n_tokens,
                n_tokens,
                generation,
                routed_scale,
            )?,
        }
        buffers.encode_schedule(ctx, enc, n_tokens, MOE_EXPERT_COUNT, generation)?;
        buffers.encode_validate(ctx, enc, n_tokens, generation)?;
        buffers.encode_signature(ctx, enc, n_tokens, generation)
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn capture_gpu_route_schedule(
        &self,
        n_tokens: usize,
        generation: NonZeroU32,
    ) -> Result<Vec<ExpertBucket>, DeepSeekV4MetalError> {
        checked_token_count(n_tokens)?;
        let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed GPU route count")?;
        let schedule_count = checked_mul(n_tokens, MOE_EXPERT_COUNT, "packed GPU schedule count")?;
        let mut expert_ids = host_read_i32(&self.expert_ids, "packed GPU route IDs")?;
        let mut weights = host_read_f32(&self.weights, "packed GPU route weights")?;
        let mut route_generations = host_read_i32(
            &self.gpu_route.route_generations,
            "packed GPU route generations",
        )?;
        let mut route_status =
            host_read_i32(&self.gpu_route.route_status, "packed GPU route statuses")?;
        let counts = host_read_i32(&self.gpu_route.counts, "packed GPU route counts")?;
        let mut slot_ids = host_read_i32(&self.gpu_route.slot_ids, "packed GPU route slot IDs")?;
        let schedule_generations = host_read_i32(
            &self.gpu_route.schedule_generations,
            "packed GPU schedule generations",
        )?;
        let aggregate = host_read_i32(&self.gpu_route.aggregate, "packed GPU route aggregate")?;
        let signature = host_read_i32(&self.gpu_route.signature, "packed GPU route signature")?;
        expert_ids.truncate(route_count);
        weights.truncate(route_count);
        route_generations.truncate(n_tokens);
        route_status.truncate(n_tokens);
        slot_ids.truncate(schedule_count);

        let generation_i32 = generation.get() as i32;
        if route_generations
            .iter()
            .any(|&value| value != generation_i32)
        {
            return invalid("packed GPU route contains a stale token producer");
        }
        if let Some((token, &status)) = route_status
            .iter()
            .enumerate()
            .find(|(_, status)| **status != DEEPSEEK_V4_ROUTE_STATUS_READY)
        {
            return invalid(format!(
                "packed GPU route token {token} failed with status {status}"
            ));
        }
        if schedule_generations
            .iter()
            .any(|&value| value != generation_i32)
        {
            return invalid("packed GPU route contains a stale schedule producer");
        }
        let expected_aggregate = [
            generation_i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            route_count as i32,
            packed_route_completion(generation.get(), n_tokens) as i32,
        ];
        if aggregate != expected_aggregate {
            return invalid(format!(
                "packed GPU route aggregate {aggregate:?} differs from {expected_aggregate:?}"
            ));
        }
        let expected_signature = [
            generation_i32,
            DEEPSEEK_V4_ROUTE_STATUS_READY,
            packed_route_signature_hash(&expert_ids, &weights, &counts, &slot_ids, n_tokens)?
                as i32,
            packed_route_signature_completion(generation.get(), n_tokens) as i32,
        ];
        if signature != expected_signature {
            return invalid(format!(
                "packed GPU route signature {signature:?} differs from {expected_signature:?}"
            ));
        }

        for token in 0..n_tokens {
            let mut seen = [false; MOE_EXPERT_COUNT];
            for slot in 0..MOE_TOP_K {
                let index = token * MOE_TOP_K + slot;
                let expert = usize::try_from(expert_ids[index]).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "packed GPU route token {token} slot {slot} has negative expert {}",
                        expert_ids[index]
                    ))
                })?;
                if expert >= MOE_EXPERT_COUNT || std::mem::replace(&mut seen[expert], true) {
                    return invalid(format!(
                        "packed GPU route token {token} slot {slot} has invalid expert {expert}"
                    ));
                }
                let weight = weights[index];
                if !weight.is_finite() || weight < 0.0 {
                    return invalid(format!(
                        "packed GPU route token {token} slot {slot} has invalid weight {weight}"
                    ));
                }
            }
        }

        let mut compact_rows = Vec::with_capacity(route_count);
        let mut compact_slots = Vec::with_capacity(route_count);
        let mut schedule = Vec::new();
        for (expert, &count_i32) in counts.iter().enumerate().take(MOE_EXPERT_COUNT) {
            let count = usize::try_from(count_i32).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed GPU route expert {expert} has negative count {}",
                    count_i32
                ))
            })?;
            if count > n_tokens {
                return invalid(format!(
                    "packed GPU route expert {expert} count {count} exceeds {n_tokens}"
                ));
            }
            let base = expert * n_tokens;
            let start = compact_rows.len();
            let mut expected = Vec::with_capacity(count);
            for token in 0..n_tokens {
                for slot in 0..MOE_TOP_K {
                    let global_slot = token * MOE_TOP_K + slot;
                    if expert_ids[global_slot] == expert as i32 {
                        expected.push(global_slot as i32);
                    }
                }
            }
            if expected.len() != count || slot_ids[base..base + count] != expected {
                return invalid(format!(
                    "packed GPU route expert {expert} schedule differs from token/slot order"
                ));
            }
            if slot_ids[base + count..base + n_tokens]
                .iter()
                .any(|&slot| slot != -1)
            {
                return invalid(format!(
                    "packed GPU route expert {expert} has non-sentinel padding"
                ));
            }
            for &global_slot in &expected {
                compact_rows.push(global_slot / MOE_TOP_K as i32);
                compact_slots.push(global_slot);
            }
            if count > 0 {
                schedule.push(ExpertBucket {
                    expert,
                    start,
                    len: count,
                });
            }
        }
        if compact_rows.len() != route_count {
            return invalid(format!(
                "packed GPU route schedule has {} assignments, expected {route_count}",
                compact_rows.len()
            ));
        }
        validate_packed_expert_schedule(
            n_tokens,
            self.expert_count,
            &expert_ids,
            &compact_rows,
            &compact_slots,
            &schedule,
        )?;
        let rows = i32_prefix(
            &self.bucket_rows,
            vec![route_count as u64],
            "packed GPU compact route rows",
        )?;
        let slots = i32_prefix(
            &self.bucket_slots,
            vec![route_count as u64],
            "packed GPU compact route slots",
        )?;
        host_write_i32(&rows, &compact_rows, "packed GPU compact route rows")?;
        host_write_i32(&slots, &compact_slots, "packed GPU compact route slots")?;
        Ok(schedule)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    fn audit_gpu_route_against_cpu(
        &self,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        n_tokens: usize,
        layer: usize,
        chunk_start: u32,
        generation: NonZeroU32,
    ) -> Result<(Vec<i32>, Vec<f32>), DeepSeekV4MetalError> {
        use sha2::{Digest, Sha256};

        let logits = host_read_f32(&views.logits, "packed route audit logits")?;
        let hash_ids = match source {
            PackedRouteSource::Hash => Some(host_read_i32(
                views.hash_ids.as_ref().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("packed route audit has no hash IDs".into())
                })?,
                "packed route audit hash IDs",
            )?),
            PackedRouteSource::Learned(_) => None,
        };
        let bias = match source {
            PackedRouteSource::Hash => None,
            PackedRouteSource::Learned(bias) => {
                Some(host_read_f32(bias, "packed route audit correction bias")?)
            }
        };
        let mut gpu_ids = host_read_i32(&self.expert_ids, "packed route audit GPU IDs")?;
        let mut gpu_weights = host_read_f32(&self.weights, "packed route audit GPU weights")?;
        gpu_ids.truncate(n_tokens * MOE_TOP_K);
        gpu_weights.truncate(n_tokens * MOE_TOP_K);
        let mut cpu_ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut cpu_weights = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut minimum_cutoff_margin = f32::INFINITY;
        let mut first_id_mismatch = None;
        let mut mismatch_tokens = 0usize;
        let mut symmetric_difference = 0usize;
        let mut weight_bit_mismatches = 0usize;
        let mut maximum_weight_delta = 0.0_f32;
        for token in 0..n_tokens {
            let start = token * MOE_EXPERT_COUNT;
            let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(
                &logits[start..start + MOE_EXPERT_COUNT],
            )
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed route audit scores for token {token}: {error}"
                ))
            })?;
            let decision = if let Some(hash_ids) = hash_ids.as_ref() {
                let start = token * MOE_TOP_K;
                let selected = hash_ids[start..start + MOE_TOP_K]
                    .iter()
                    .map(|&expert| usize::try_from(expert).expect("validated hash expert"))
                    .collect::<Vec<_>>();
                crate::deepseek_v4_oracle::hash_route(&scores, &selected, 1.5)
            } else {
                let bias = bias.as_ref().expect("learned route bias");
                let mut ranked = (0..MOE_EXPERT_COUNT).collect::<Vec<_>>();
                ranked.sort_unstable_by(|&left, &right| {
                    (scores[right] + bias[right])
                        .total_cmp(&(scores[left] + bias[left]))
                        .then_with(|| left.cmp(&right))
                });
                let margin = (scores[ranked[MOE_TOP_K - 1]] + bias[ranked[MOE_TOP_K - 1]])
                    - (scores[ranked[MOE_TOP_K]] + bias[ranked[MOE_TOP_K]]);
                minimum_cutoff_margin = minimum_cutoff_margin.min(margin);
                crate::deepseek_v4_oracle::learned_route(&scores, bias, MOE_TOP_K, 1.5)
            }
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed route audit decision for token {token}: {error}"
                ))
            })?;
            let expected_ids = decision
                .expert_ids
                .iter()
                .map(|&expert| expert as i32)
                .collect::<Vec<_>>();
            let gpu_start = token * MOE_TOP_K;
            let actual_ids = &gpu_ids[gpu_start..gpu_start + MOE_TOP_K];
            if actual_ids != expected_ids {
                mismatch_tokens += 1;
                symmetric_difference += actual_ids
                    .iter()
                    .filter(|expert| !expected_ids.contains(expert))
                    .count()
                    + expected_ids
                        .iter()
                        .filter(|expert| !actual_ids.contains(expert))
                        .count();
                if first_id_mismatch.is_none() {
                    first_id_mismatch = Some((token, expected_ids.clone(), actual_ids.to_vec()));
                }
            }
            for (slot, &expected) in decision.weights.iter().enumerate() {
                let actual = gpu_weights[gpu_start + slot];
                weight_bit_mismatches += usize::from(actual.to_bits() != expected.to_bits());
                maximum_weight_delta = maximum_weight_delta.max((actual - expected).abs());
            }
            cpu_ids.extend_from_slice(&expected_ids);
            cpu_weights.extend_from_slice(&decision.weights);
        }
        let source = if hash_ids.is_some() {
            "hash"
        } else {
            "learned"
        };
        let cutoff_margin = if minimum_cutoff_margin.is_finite() {
            format!("{minimum_cutoff_margin:.9}")
        } else {
            "n/a".into()
        };
        eprintln!(
            "deepseek_v4 packed_route_audit chunk_start={chunk_start} n={n_tokens} layer={layer} source={source} generation={} id_mismatch_tokens={mismatch_tokens} symmetric_difference={symmetric_difference} weight_bit_mismatches={weight_bit_mismatches} max_weight_delta={maximum_weight_delta:.9} min_rank6_rank7_margin={cutoff_margin} first_id_mismatch={first_id_mismatch:?} cpu_ids_sha256={:x} gpu_ids_sha256={:x} cpu_weights_sha256={:x} gpu_weights_sha256={:x}",
            generation.get(),
            Sha256::digest(bytemuck::cast_slice(&cpu_ids)),
            Sha256::digest(bytemuck::cast_slice(&gpu_ids)),
            Sha256::digest(bytemuck::cast_slice(&cpu_weights)),
            Sha256::digest(bytemuck::cast_slice(&gpu_weights)),
        );
        Ok((cpu_ids, cpu_weights))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_router(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        ffn_norm: &MetalTensor,
        gate_inp: &MetalTensor,
        token_ids: &MetalTensor,
        hash_map: Option<&MetalTensor>,
        n_tokens: usize,
        rms_eps: f32,
        router_e8p32_strict: bool,
    ) -> Result<PackedMoeViews, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_router_batch")?;
        checked_token_count(n_tokens)?;
        validate_eps(rms_eps, "packed MoE RMSNorm epsilon")?;
        validate_f32(
            input,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            false,
            "packed MoE input",
        )?;
        validate_f32(
            ffn_norm,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64],
            false,
            "packed MoE norm weight",
        )?;
        let normalized_input = f32_prefix(
            &self.normalized_input,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed MoE normalized input",
        )?;
        let logits = f32_prefix(
            &self.logits,
            vec![self.expert_count as u64, n_tokens as u64],
            "packed MoE logits",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            input,
            ffn_norm,
            &normalized_input,
            n_tokens,
            DEEPSEEK_V4_HIDDEN_SIZE,
            rms_eps,
        )?;
        if router_e8p32_strict && gate_inp.dtype == GgmlType::F32 {
            crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                ctx,
                enc,
                gate_inp,
                &normalized_input,
                &logits,
                DEEPSEEK_V4_HIDDEN_SIZE,
                self.expert_count,
                n_tokens,
            )
            .map_err(DeepSeekV4MetalError::Metal)?;
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: strict-order E8P32 packed router active for K160 N={n_tokens}; rollback=QWEN_DSV4_PACKED_ROUTER_E8P32_STRICT=0"
                );
            });
        } else {
            encode_batch_projection(
                ctx,
                enc,
                gate_inp,
                &normalized_input,
                &logits,
                DEEPSEEK_V4_HIDDEN_SIZE,
                self.expert_count,
                n_tokens,
                "packed MoE router",
            )?;
        }
        let hash_ids = if let Some(hash_map) = hash_map {
            let hash_ids = i32_prefix(
                &self.hash_ids,
                vec![MOE_TOP_K as u64, n_tokens as u64],
                "packed hash route IDs",
            )?;
            encode_hash_gather(ctx, enc, token_ids, hash_map, &hash_ids, n_tokens)?;
            Some(hash_ids)
        } else {
            None
        };
        Ok(PackedMoeViews {
            normalized_input,
            logits,
            hash_ids,
        })
    }

    fn route(
        &self,
        views: &PackedMoeViews,
        source: PackedRouteSource<'_>,
        n_tokens: usize,
        routed_scale: f32,
    ) -> Result<Vec<ExpertBucket>, DeepSeekV4MetalError> {
        checked_token_count(n_tokens)?;
        if !routed_scale.is_finite() || routed_scale <= 0.0 {
            return invalid("packed MoE routed scale must be finite and positive");
        }
        let logits = host_read_f32(&views.logits, "packed MoE logits")?;
        let hash_ids = match source {
            PackedRouteSource::Hash => Some(host_read_i32(
                views.hash_ids.as_ref().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed hash route did not gather token IDs".into(),
                    )
                })?,
                "packed hash route IDs",
            )?),
            PackedRouteSource::Learned(_) => None,
        };
        let bias = match source {
            PackedRouteSource::Hash => None,
            PackedRouteSource::Learned(bias) => {
                validate_f32(
                    bias,
                    &[self.expert_count as u64],
                    false,
                    "packed router correction bias",
                )?;
                Some(host_read_f32(bias, "packed router correction bias")?)
            }
        };
        let mut expert_ids = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut weights = Vec::with_capacity(n_tokens * MOE_TOP_K);
        for token in 0..n_tokens {
            let start = token * self.expert_count;
            let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(
                &logits[start..start + self.expert_count],
            )
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "packed router scores for token {token}: {error}"
                ))
            })?;
            let decision = if let Some(hash_ids) = hash_ids.as_ref() {
                let start = token * MOE_TOP_K;
                let selected = hash_ids[start..start + MOE_TOP_K]
                    .iter()
                    .map(|&expert| {
                        let expert = usize::try_from(expert).map_err(|_| {
                            DeepSeekV4MetalError::Invalid(format!(
                                "packed hash route contains negative ID {expert}"
                            ))
                        })?;
                        if expert >= self.expert_count {
                            return invalid(format!(
                                "packed hash route expert {expert} exceeds {}",
                                self.expert_count
                            ));
                        }
                        Ok(expert)
                    })
                    .collect::<Result<Vec<_>, DeepSeekV4MetalError>>()?;
                crate::deepseek_v4_oracle::hash_route(&scores, &selected, routed_scale)
            } else {
                crate::deepseek_v4_oracle::learned_route(
                    &scores,
                    bias.as_ref().expect("learned route bias"),
                    MOE_TOP_K,
                    routed_scale,
                )
            }
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!("packed route for token {token}: {error}"))
            })?;
            expert_ids.extend(decision.expert_ids.iter().map(|&expert| expert as i32));
            weights.extend_from_slice(&decision.weights);
        }
        let expert_ids_view = i32_prefix(
            &self.expert_ids,
            vec![MOE_TOP_K as u64, n_tokens as u64],
            "packed selected expert IDs",
        )?;
        let weights_view = f32_prefix(
            &self.weights,
            vec![MOE_TOP_K as u64, n_tokens as u64],
            "packed selected expert weights",
        )?;
        host_write_i32(&expert_ids_view, &expert_ids, "packed selected expert IDs")?;
        host_write_f32(&weights_view, &weights, "packed selected expert weights")?;

        let mut by_expert = (0..self.expert_count)
            .map(|_| Vec::<(usize, usize)>::new())
            .collect::<Vec<_>>();
        for token in 0..n_tokens {
            for slot in 0..MOE_TOP_K {
                let expert = expert_ids[token * MOE_TOP_K + slot] as usize;
                by_expert[expert].push((token, token * MOE_TOP_K + slot));
            }
        }
        let mut bucket_rows = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut bucket_slots = Vec::with_capacity(n_tokens * MOE_TOP_K);
        let mut schedule = Vec::new();
        for (expert, assignments) in by_expert.into_iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let start = bucket_rows.len();
            for (token, slot) in assignments {
                bucket_rows.push(token as i32);
                bucket_slots.push(slot as i32);
            }
            schedule.push(ExpertBucket {
                expert,
                start,
                len: bucket_rows.len() - start,
            });
        }
        if bucket_rows.len() != n_tokens * MOE_TOP_K {
            return invalid("packed expert bucket schedule lost route assignments");
        }
        validate_packed_expert_schedule(
            n_tokens,
            self.expert_count,
            &expert_ids,
            &bucket_rows,
            &bucket_slots,
            &schedule,
        )?;
        let bucket_rows_view = i32_prefix(
            &self.bucket_rows,
            vec![(n_tokens * MOE_TOP_K) as u64],
            "packed expert bucket rows",
        )?;
        let bucket_slots_view = i32_prefix(
            &self.bucket_slots,
            vec![(n_tokens * MOE_TOP_K) as u64],
            "packed expert bucket slots",
        )?;
        host_write_i32(&bucket_rows_view, &bucket_rows, "packed expert bucket rows")?;
        host_write_i32(
            &bucket_slots_view,
            &bucket_slots,
            "packed expert bucket slots",
        )?;
        Ok(schedule)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_experts(
        &self,
        ctx: &MetalContext,
        enc: &mut PackedPostRouteLayerEncoder<'_>,
        normalized_input: &MetalTensor,
        schedule: &[ExpertBucket],
        gpu_compacted: bool,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_policy: PackedExpertPolicy,
        grouped_q3q4_qualified: bool,
        mxfp4_matrix: bool,
        shared_matrix: bool,
        shared_precomputed: bool,
        expert_clamp: f32,
        shared_clamp: f32,
        n_tokens: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_experts_batch")?;
        checked_token_count(n_tokens)?;
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("packed expert clamp must be finite and positive");
        }
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("packed shared-expert clamp must be finite and positive");
        }
        validate_expert_bank(
            gate_bank,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            self.expert_count,
            "packed routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_FFN_SIZE,
            self.expert_count,
            "packed routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            MOE_FFN_SIZE,
            DEEPSEEK_V4_HIDDEN_SIZE,
            self.expert_count,
            "packed routed down bank",
        )?;
        if gpu_compacted
            && !packed_gpu_compact_expert_layer_qualified(
                ctx,
                expert_policy,
                n_tokens,
                packed_gpu_route_iq3_enabled(),
                gate_bank.dtype,
                up_bank.dtype,
                down_bank.dtype,
            )
        {
            return invalid("packed GPU compaction reached an unqualified expert layer");
        }
        let expert_outputs = f32_prefix(
            &self.expert_outputs,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                MOE_TOP_K as u64,
                n_tokens as u64,
            ],
            "packed expert outputs",
        )?;
        let used_grouped_q3q4 = if grouped_q3q4_qualified
            && packed_grouped_q3q4_candidate_supported(ctx)
        {
            let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed grouped Q3/Q4 routes")?;
            let rows = i32_prefix(
                &self.bucket_rows,
                vec![route_count as u64],
                "packed grouped Q3/Q4 source rows",
            )?;
            let slots = i32_prefix(
                &self.bucket_slots,
                vec![route_count as u64],
                "packed grouped Q3/Q4 destination slots",
            )?;
            let expert_output_flat = expert_outputs
                .view_subrange(0, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, route_count as u64]);
            let (gate, up) = packed_grouped_gate_up_views(
                &expert_output_flat,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                route_count,
            )?;
            let grouped_inner = f32_prefix(
                &self.grouped_inner,
                vec![MOE_FFN_SIZE as u64, route_count as u64],
                "packed grouped Q3/Q4 inner",
            )?;
            let plan = PackedGroupedExpertPlan::new(n_tokens, schedule, Some(&self.grouped_tiles))?;
            for (bank, projection) in [(gate_bank, &gate), (up_bank, &up)] {
                encode_packed_grouped_mapped_k_block_f32_plan(
                    ctx,
                    enc,
                    bank,
                    normalized_input,
                    &rows,
                    &slots,
                    &plan,
                    projection,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    self.expert_count,
                    MOE_TOP_K,
                    n_tokens,
                    n_tokens,
                    route_count,
                )?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if enc.splits_routed_stages() {
                enc.boundary(PackedPostRouteStageKind::RoutedSwiGlu)?;
            }
            let projected_elements = checked_mul(
                MOE_FFN_SIZE,
                route_count,
                "packed grouped Q3/Q4 projected elements",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate.view_subrange(0, vec![projected_elements as u64]),
                &up.view_subrange(0, vec![projected_elements as u64]),
                &grouped_inner.view_subrange(0, vec![projected_elements as u64]),
                expert_clamp,
            )?;
            #[cfg(feature = "dsv4-diagnostics")]
            if enc.splits_routed_stages() {
                enc.boundary(PackedPostRouteStageKind::RoutedDown)?;
            }
            encode_packed_grouped_mapped_k_block_f32_plan(
                ctx,
                enc,
                down_bank,
                &grouped_inner,
                &slots,
                &slots,
                &plan,
                &expert_output_flat,
                MOE_FFN_SIZE,
                DEEPSEEK_V4_HIDDEN_SIZE,
                self.expert_count,
                MOE_TOP_K,
                n_tokens,
                route_count,
                route_count,
            )?;
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: grouped Q3_K/Q4_K packed experts active for N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_GROUPED_Q3Q4=0"
                );
            }
            true
        } else {
            false
        };
        let used_grouped_iq2 = if expert_policy.uses_iq2_target()
            && gate_bank.dtype == GgmlType::IQ2_XS
            && up_bank.dtype == GgmlType::IQ2_XS
            && down_bank.dtype == GgmlType::IQ3_XXS
            && packed_grouped_expert_kernels_supported(ctx)
        {
            let slots = i32_prefix(
                &self.bucket_slots,
                vec![(n_tokens * MOE_TOP_K) as u64],
                "packed grouped expert slots",
            )?;
            let grouped_inner = f32_prefix(
                &self.grouped_inner,
                vec![MOE_FFN_SIZE as u64, MOE_TOP_K as u64, n_tokens as u64],
                "packed grouped expert inner",
            )?;
            let grouped_plan = if gpu_compacted {
                PackedGroupedExpertPlan::from_device(
                    &self.grouped_tiles,
                    PACKED_GROUPED_EXPERT_MAX_TILES,
                )?
            } else {
                PackedGroupedExpertPlan::new(n_tokens, schedule, Some(&self.grouped_tiles))?
            };
            let used_iq2_mma16 = if expert_policy.uses_iq2_mma16(n_tokens) {
                let work_unit = if packed_iq2_f16_mm64x32_enabled() {
                    PackedIq2MatrixWorkUnit::Mm64x32F16
                } else if packed_iq2_mm64x32_enabled() {
                    PackedIq2MatrixWorkUnit::Mm64x32
                } else {
                    PackedIq2MatrixWorkUnit::Mma16
                };
                let mma16_plan = if work_unit == PackedIq2MatrixWorkUnit::Mma16 {
                    Some(if gpu_compacted {
                        PackedGroupedExpertPlan::from_device(
                            &self.grouped_iq2_mma16_tiles,
                            PACKED_GROUPED_IQ2_MMA16_MAX_TILES,
                        )?
                    } else {
                        PackedGroupedExpertPlan::new_iq2_mma16(
                            n_tokens,
                            schedule,
                            Some(&self.grouped_iq2_mma16_tiles),
                        )?
                    })
                } else {
                    None
                };
                let matrix_plan = mma16_plan.as_ref().unwrap_or(&grouped_plan);
                let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed BM16 IQ2 routes")?;
                let rows = i32_prefix(
                    &self.bucket_rows,
                    vec![route_count as u64],
                    "packed BM16 IQ2 source rows",
                )?;
                let expert_output_flat = expert_outputs
                    .view_subrange(0, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, route_count as u64]);
                let (gate, up) = packed_grouped_gate_up_views(
                    &expert_output_flat,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    route_count,
                )?;
                #[cfg(feature = "dsv4-diagnostics")]
                if enc.splits_bm16_stages() {
                    for (bank, projection) in [(gate_bank, &gate), (up_bank, &up)] {
                        encode_packed_grouped_mapped_iq2_xs_f32_matrix(
                            ctx,
                            enc,
                            bank,
                            normalized_input,
                            &rows,
                            &slots,
                            matrix_plan,
                            projection,
                            DEEPSEEK_V4_HIDDEN_SIZE,
                            MOE_FFN_SIZE,
                            self.expert_count,
                            MOE_TOP_K,
                            n_tokens,
                            n_tokens,
                            route_count,
                            work_unit,
                        )?;
                    }
                    enc.boundary(PackedPostRouteStageKind::RoutedSwiGlu)?;
                    let projected_elements = checked_mul(
                        MOE_FFN_SIZE,
                        route_count,
                        "packed BM16 IQ2 projected elements",
                    )?;
                    encode_ds4_clamped_swiglu(
                        ctx,
                        enc,
                        &gate.view_subrange(0, vec![projected_elements as u64]),
                        &up.view_subrange(0, vec![projected_elements as u64]),
                        &grouped_inner.view_subrange(0, vec![projected_elements as u64]),
                        expert_clamp,
                    )?;
                    enc.boundary(PackedPostRouteStageKind::RoutedDown)?;
                } else {
                    encode_packed_grouped_mapped_iq2_xs_swiglu_f32_matrix(
                        ctx,
                        enc,
                        gate_bank,
                        up_bank,
                        normalized_input,
                        &rows,
                        &slots,
                        matrix_plan,
                        &gate,
                        &up,
                        &grouped_inner,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        MOE_FFN_SIZE,
                        self.expert_count,
                        MOE_TOP_K,
                        n_tokens,
                        n_tokens,
                        route_count,
                        expert_clamp,
                        work_unit,
                    )?;
                }
                #[cfg(not(feature = "dsv4-diagnostics"))]
                encode_packed_grouped_mapped_iq2_xs_swiglu_f32_matrix(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    normalized_input,
                    &rows,
                    &slots,
                    matrix_plan,
                    &gate,
                    &up,
                    &grouped_inner,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    self.expert_count,
                    MOE_TOP_K,
                    n_tokens,
                    n_tokens,
                    route_count,
                    expert_clamp,
                    work_unit,
                )?;
                true
            } else {
                false
            };
            if !used_iq2_mma16 {
                encode_packed_grouped_swiglu_iq2_xs_f32(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    normalized_input,
                    &slots,
                    &grouped_plan,
                    &grouped_inner,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    self.expert_count,
                    MOE_TOP_K,
                    n_tokens,
                    expert_clamp,
                )?;
            }
            encode_packed_grouped_down_iq3_xxs_f32(
                ctx,
                enc,
                down_bank,
                &grouped_inner,
                &slots,
                &grouped_plan,
                &expert_outputs,
                MOE_FFN_SIZE,
                DEEPSEEK_V4_HIDDEN_SIZE,
                self.expert_count,
                MOE_TOP_K,
                n_tokens,
            )?;
            #[cfg(feature = "dsv4-diagnostics")]
            {
                self.grouped_iq2_invocations.set(
                    self.grouped_iq2_invocations
                        .get()
                        .checked_add(1)
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid(
                                "packed grouped IQ2 invocation count overflow".into(),
                            )
                        })?,
                );
            }
            true
        } else {
            false
        };
        let used_grouped_iq3 = if expert_policy.uses_iq3_target()
            && gate_bank.dtype == GgmlType::IQ3_XXS
            && up_bank.dtype == GgmlType::IQ3_XXS
            && down_bank.dtype == GgmlType::IQ3_XXS
            && packed_grouped_iq3_candidate_supported(ctx)
        {
            let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed grouped IQ3 routes")?;
            let rows = i32_prefix(
                &self.bucket_rows,
                vec![route_count as u64],
                "packed grouped IQ3 source rows",
            )?;
            let slots = i32_prefix(
                &self.bucket_slots,
                vec![route_count as u64],
                "packed grouped IQ3 destination slots",
            )?;
            let expert_output_flat = expert_outputs
                .view_subrange(0, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, route_count as u64]);
            let grouped_inner = f32_prefix(
                &self.grouped_inner,
                vec![MOE_FFN_SIZE as u64, route_count as u64],
                "packed grouped IQ3 inner",
            )?;
            let grouped_plan = if gpu_compacted {
                PackedGroupedExpertPlan::from_device(
                    &self.grouped_tiles,
                    PACKED_GROUPED_EXPERT_MAX_TILES,
                )?
            } else {
                PackedGroupedExpertPlan::new(n_tokens, schedule, Some(&self.grouped_tiles))?
            };
            encode_packed_grouped_all_iq3(
                ctx,
                enc,
                gate_bank,
                up_bank,
                down_bank,
                normalized_input,
                &rows,
                &slots,
                &grouped_plan,
                &expert_output_flat,
                &grouped_inner,
                DEEPSEEK_V4_HIDDEN_SIZE,
                MOE_FFN_SIZE,
                self.expert_count,
                MOE_TOP_K,
                n_tokens,
                expert_clamp,
            )?;
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            {
                self.grouped_iq3_invocations.set(
                    self.grouped_iq3_invocations
                        .get()
                        .checked_add(1)
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid(
                                "packed grouped IQ3 invocation count overflow".into(),
                            )
                        })?,
                );
            }
            true
        } else {
            false
        };
        let used_grouped_target = used_grouped_q3q4 || used_grouped_iq2 || used_grouped_iq3;
        let mut mxfp4_matrix_dispatches = 0usize;
        let mut mxfp4_matrix_columns = 0usize;
        let mut mxfp4_scalar_columns = 0usize;
        if !used_grouped_target {
            for bucket in schedule {
                let gate_weight = expert_weight_view(
                    gate_bank,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    bucket.expert,
                    "packed routed gate slice",
                )?;
                let up_weight = expert_weight_view(
                    up_bank,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    MOE_FFN_SIZE,
                    bucket.expert,
                    "packed routed up slice",
                )?;
                let down_weight = expert_weight_view(
                    down_bank,
                    MOE_FFN_SIZE,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    bucket.expert,
                    "packed routed down slice",
                )?;
                for bucket_offset in (0..bucket.len).step_by(n_tokens) {
                    let chunk_len = (bucket.len - bucket_offset).min(n_tokens);
                    let chunk_start = bucket.start.checked_add(bucket_offset).ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid("packed expert chunk start overflow".into())
                    })?;
                    let rows = i32_slice(
                        &self.bucket_rows,
                        chunk_start,
                        chunk_len,
                        "packed expert input rows",
                    )?;
                    let slots = i32_slice(
                        &self.bucket_slots,
                        chunk_start,
                        chunk_len,
                        "packed expert output slots",
                    )?;
                    let expert_input = f32_prefix(
                        &self.expert_input,
                        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, chunk_len as u64],
                        "packed expert input",
                    )?;
                    encode_get_rows_f32(
                        ctx,
                        enc,
                        normalized_input,
                        &rows,
                        &expert_input,
                        chunk_len,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                    )?;
                    let gate = f32_prefix(
                        &self.gate,
                        vec![MOE_FFN_SIZE as u64, chunk_len as u64],
                        "packed routed gate",
                    )?;
                    let up = f32_prefix(
                        &self.up,
                        vec![MOE_FFN_SIZE as u64, chunk_len as u64],
                        "packed routed up",
                    )?;
                    let inner = f32_prefix(
                        &self.inner,
                        vec![MOE_FFN_SIZE as u64, chunk_len as u64],
                        "packed routed inner",
                    )?;
                    let bucket_output = f32_prefix(
                        &self.bucket_output,
                        vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, chunk_len as u64],
                        "packed routed bucket output",
                    )?;
                    encode_batch_projection(
                        ctx,
                        enc,
                        &gate_weight,
                        &expert_input,
                        &gate,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        MOE_FFN_SIZE,
                        chunk_len,
                        "packed routed gate",
                    )?;
                    encode_batch_projection(
                        ctx,
                        enc,
                        &up_weight,
                        &expert_input,
                        &up,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        MOE_FFN_SIZE,
                        chunk_len,
                        "packed routed up",
                    )?;
                    let flat_len = checked_mul(chunk_len, MOE_FFN_SIZE, "packed SwiGLU")?;
                    let gate_flat = gate.view_subrange(0, vec![flat_len as u64]);
                    let up_flat = up.view_subrange(0, vec![flat_len as u64]);
                    let inner_flat = inner.view_subrange(0, vec![flat_len as u64]);
                    encode_ds4_clamped_swiglu(
                        ctx,
                        enc,
                        &gate_flat,
                        &up_flat,
                        &inner_flat,
                        expert_clamp,
                    )?;
                    if down_weight.dtype == GgmlType::MXFP4 && mxfp4_matrix && chunk_len >= 16 {
                        crate::metal::encode_mat_mat_mxfp4_f32_mm64x32(
                            ctx,
                            enc,
                            &down_weight,
                            &inner,
                            &bucket_output,
                            MOE_FFN_SIZE,
                            DEEPSEEK_V4_HIDDEN_SIZE,
                            chunk_len,
                        )?;
                        mxfp4_matrix_dispatches += 1;
                        mxfp4_matrix_columns =
                            mxfp4_matrix_columns.checked_add(chunk_len).ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed MXFP4 matrix column count overflow".into(),
                                )
                            })?;
                    } else if down_weight.dtype == GgmlType::MXFP4 {
                        if mxfp4_matrix {
                            mxfp4_scalar_columns =
                                mxfp4_scalar_columns.checked_add(chunk_len).ok_or_else(|| {
                                    DeepSeekV4MetalError::Invalid(
                                        "packed MXFP4 scalar column count overflow".into(),
                                    )
                                })?;
                        }
                        for row in 0..chunk_len {
                            let inner_row = f32_row(
                                &inner,
                                row,
                                MOE_FFN_SIZE,
                                vec![MOE_FFN_SIZE as u64],
                                "packed MXFP4 routed inner row",
                            )?;
                            let output_row = f32_row(
                                &bucket_output,
                                row,
                                DEEPSEEK_V4_HIDDEN_SIZE,
                                vec![DEEPSEEK_V4_HIDDEN_SIZE as u64],
                                "packed MXFP4 routed output row",
                            )?;
                            encode_projection(
                                ctx,
                                enc,
                                &down_weight,
                                &inner_row,
                                &output_row,
                                MOE_FFN_SIZE,
                                DEEPSEEK_V4_HIDDEN_SIZE,
                                "packed MXFP4 routed down",
                            )?;
                        }
                    } else {
                        encode_batch_projection(
                            ctx,
                            enc,
                            &down_weight,
                            &inner,
                            &bucket_output,
                            MOE_FFN_SIZE,
                            DEEPSEEK_V4_HIDDEN_SIZE,
                            chunk_len,
                            "packed routed down",
                        )?;
                    }
                    crate::metal::encode_scatter_rows_f32_unique(
                        ctx,
                        enc,
                        &bucket_output,
                        &slots,
                        &expert_outputs,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        chunk_len,
                    )?;
                }
            }
        }
        if mxfp4_matrix_dispatches > 0 {
            eprintln!(
                "deepseek_v4: MXFP4 F32 matrix routed down active gate={:?} up={:?} matrix_dispatches={} matrix_columns={} scalar_columns={}; rollback=QWEN_DSV4_PACKED_MXFP4_MATRIX=0",
                gate_bank.dtype,
                up_bank.dtype,
                mxfp4_matrix_dispatches,
                mxfp4_matrix_columns,
                mxfp4_scalar_columns,
            );
        }

        let shared_output = if shared_precomputed {
            f32_prefix(
                &self.shared_output,
                vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
                "packed precomputed shared output",
            )?
        } else {
            #[cfg(feature = "dsv4-diagnostics")]
            enc.boundary(PackedPostRouteStageKind::SharedExpert)?;
            self.encode_shared_expert(
                ctx,
                enc,
                normalized_input,
                shared_gate,
                shared_up,
                shared_down,
                shared_matrix,
                shared_clamp,
                n_tokens,
            )?
        };

        #[cfg(feature = "dsv4-diagnostics")]
        enc.boundary(PackedPostRouteStageKind::ExpertCombine)?;
        let weights = f32_prefix(
            &self.weights,
            vec![MOE_TOP_K as u64, n_tokens as u64],
            "packed selected expert weights",
        )?;
        let routed_output = f32_prefix(
            &self.routed_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed routed output",
        )?;
        let final_output = f32_prefix(
            &self.final_output,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed MoE output",
        )?;
        crate::metal::encode_moe_weighted_sum_packed_f32(
            ctx,
            enc,
            &expert_outputs,
            &weights,
            &routed_output,
            DEEPSEEK_V4_HIDDEN_SIZE,
            MOE_TOP_K,
            n_tokens,
        )?;
        crate::metal::encode_add_f32(ctx, enc, &routed_output, &shared_output, &final_output)?;

        #[cfg(feature = "dsv4-diagnostics")]
        enc.boundary(PackedPostRouteStageKind::HyperPostAndHead)?;
        Ok(final_output)
    }
}

fn encode_hash_gather(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &MetalTensor,
    token_to_expert: &MetalTensor,
    output: &MetalTensor,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    validate_i32(token_ids, &[n_tokens as u64], false, "packed token IDs")?;
    validate_i32_bank(token_to_expert, MOE_TOP_K, "packed token-to-expert map")?;
    validate_i32(
        output,
        &[MOE_TOP_K as u64, n_tokens as u64],
        true,
        "packed hash route IDs",
    )?;
    let vocab_size = usize::try_from(token_to_expert.shape[1])
        .map_err(|_| DeepSeekV4MetalError::Invalid("hash vocabulary exceeds usize".into()))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        top_k: u32,
        vocab_size: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_hash_gather")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: checked_token_count(n_tokens)?,
            top_k: MOE_TOP_K as u32,
            vocab_size: u32::try_from(vocab_size)
                .map_err(|_| DeepSeekV4MetalError::Invalid("hash vocabulary exceeds u32".into()))?,
        },
    );
    enc.set_tensor(1, token_ids);
    enc.set_tensor(2, token_to_expert);
    enc.set_tensor(3, output);
    let total = checked_mul(n_tokens, MOE_TOP_K, "packed hash route IDs")?;
    enc.dispatch(
        MTLSize {
            width: total.div_ceil(64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_copy_raw_ring_f16_bits(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    source: &MetalTensor,
    destination: &MetalTensor,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_copy_raw_ring_f16_bits")?;
    let shape = [512, DEEPSEEK_V4_LOCAL_WINDOW as u64];
    validate_f16(source, &shape, false, "packed source raw ring")?;
    validate_f16(destination, &shape, true, "packed preserved raw ring")?;
    let elements = checked_mul(512, DEEPSEEK_V4_LOCAL_WINDOW, "packed raw-ring copy")?;
    let elements = u32::try_from(elements)
        .map_err(|_| DeepSeekV4MetalError::Invalid("packed raw-ring copy exceeds u32".into()))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_copy_u16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n: elements });
    enc.note_read(source);
    enc.set_tensor(1, source);
    enc.note_write(destination);
    enc.set_tensor(2, destination);
    enc.dispatch(
        MTLSize {
            width: (elements as usize).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_publish_raw_chunk_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    source: &MetalTensor,
    chunk: &MetalTensor,
    ring: &MetalTensor,
    start_position: u32,
    n_tokens: usize,
    head_dim: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_publish_raw_chunk_f16")?;
    checked_token_count(n_tokens)?;
    validate_f32(
        source,
        &[head_dim as u64, n_tokens as u64],
        false,
        "packed raw chunk source",
    )?;
    validate_f16(
        chunk,
        &[head_dim as u64, n_tokens as u64],
        true,
        "packed raw chunk",
    )?;
    validate_f16(
        ring,
        &[head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        true,
        "packed raw ring",
    )?;
    start_position
        .checked_add(u32::try_from(n_tokens - 1).map_err(|_| {
            DeepSeekV4MetalError::Invalid("packed raw chunk count exceeds u32".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed raw chunk position overflow".into())
        })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_dim: u32,
        row_count: u32,
        start_position: u32,
        window: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_publish_raw_chunk_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_dim: u32::try_from(head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed raw chunk width exceeds u32".into())
            })?,
            row_count: checked_token_count(n_tokens)?,
            start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
        },
    );
    enc.set_tensor(1, source);
    enc.set_tensor(2, chunk);
    enc.set_tensor(3, ring);
    let count = checked_mul(n_tokens, head_dim, "packed raw chunk elements")?;
    enc.dispatch(
        MTLSize {
            width: count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    checked_token_count(n_tokens)?;
    let config = deepseek_v4_session_attention_config();
    let raw_chunk = f16_prefix(
        raw_cache,
        vec![config.head_dim as u64, n_tokens as u64],
        "packed dense raw chunk",
    )?;
    let Some(query_offset) = (kind == AttentionKind::HeavilyCompressed)
        .then(|| tiled_hca_query_offset(start_position, n_tokens))
        .flatten()
    else {
        return encode_packed_cooperative_dense_sink_attention_f16(
            ctx,
            enc,
            queries,
            &raw_chunk,
            raw_cache_before_chunk,
            compressed,
            sinks,
            output,
            kind,
            start_position,
            n_tokens,
        );
    };
    let rows = compressed.ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(
            "packed tiled HCA requires a published compressed history".into(),
        )
    })?;
    let query_width = config.checked()?.query_width;
    if query_offset > 0 {
        let prefix_queries = f32_prefix(
            queries,
            vec![query_width as u64, query_offset as u64],
            "packed cooperative-HCA prefix queries",
        )?;
        let prefix_output = f32_prefix(
            output,
            vec![query_width as u64, query_offset as u64],
            "packed cooperative-HCA prefix output",
        )?;
        let prefix_raw_chunk = f16_prefix(
            &raw_chunk,
            vec![config.head_dim as u64, query_offset as u64],
            "packed cooperative-HCA prefix raw chunk",
        )?;
        let prefix_end = start_position
            .checked_add(u32::try_from(query_offset).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed HCA prefix exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed HCA prefix overflow".into()))?;
        encode_packed_cooperative_dense_sink_attention_f16(
            ctx,
            enc,
            &prefix_queries,
            &prefix_raw_chunk,
            raw_cache_before_chunk,
            Some(DeepSeekV4PublishedRows {
                cache: rows.cache,
                count: prefix_end as usize / 128,
                capacity_rows: rows.capacity_rows,
            }),
            sinks,
            &prefix_output,
            kind,
            start_position,
            query_offset,
        )?;
    }
    encode_tiled_dense_sink_attention_f16(
        ctx,
        enc,
        queries,
        &raw_chunk,
        raw_cache_before_chunk,
        DeepSeekV4RawCacheLayout::Chunk,
        rows,
        sinks,
        output,
        start_position,
        query_offset,
        n_tokens - query_offset,
        128,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_cooperative_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let encode = if packed_grouped_dense_attention_enabled() {
        encode_grouped_online_dense_sink_attention_f16
    } else {
        encode_cooperative_dense_sink_attention_f16
    };
    encode(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        DeepSeekV4RawCacheLayout::Chunk,
        compressed,
        sinks,
        output,
        kind,
        start_position,
        n_tokens,
        deepseek_v4_session_attention_config(),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_packed_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    rows: DeepSeekV4CsaRows<'_>,
    sparse: PackedCsaSelectionView<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    start_position: u32,
    n_tokens: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_packed_selected_attention")?;
    let config = deepseek_v4_session_attention_config();
    let dims = config.checked()?;
    checked_token_count(n_tokens)?;
    let sparse_end = sparse
        .query_offset
        .checked_add(sparse.query_count)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("packed sparse query range overflow".into())
        })?;
    if sparse.query_count == 0
        || sparse.query_offset >= n_tokens
        || sparse_end != n_tokens
        || rows.count <= DEEPSEEK_V4_CSA_TOP_K
        || rows.count > rows.capacity_rows
    {
        return invalid(format!(
            "packed selected attention geometry is invalid: offset={} count={} tokens={n_tokens} rows={}/{}",
            sparse.query_offset, sparse.query_count, rows.count, rows.capacity_rows
        ));
    }
    validate_f32(
        queries,
        &[dims.query_width as u64, n_tokens as u64],
        false,
        "packed selected attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, n_tokens as u64],
        false,
        "packed selected raw chunk",
    )?;
    validate_f16(
        raw_cache_before_chunk,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "packed selected preserved raw cache",
    )?;
    validate_f16(
        rows.attention_cache,
        &[config.head_dim as u64, rows.capacity_rows as u64],
        false,
        "packed selected compressed cache",
    )?;
    validate_i32(
        sparse.cache_order_ids,
        &[DEEPSEEK_V4_CSA_TOP_K as u64, sparse.query_count as u64],
        false,
        "packed selected cache-order IDs",
    )?;
    for (tensor, name) in [
        (sparse.selected_counts, "packed selected row counts"),
        (sparse.visible_counts, "packed selected visible counts"),
    ] {
        validate_i32(tensor, &[sparse.query_count as u64], false, name)?;
    }
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "packed selected attention sinks",
    )?;
    validate_f32(
        output,
        &[dims.query_width as u64, n_tokens as u64],
        true,
        "packed selected attention output",
    )?;

    let online = packed_selected_online_enabled();
    let direct_load = online && deepseek_v4_online_direct_load_enabled();
    static POLICY_LOGGED: std::sync::Once = std::sync::Once::new();
    POLICY_LOGGED.call_once(|| {
        eprintln!(
            "deepseek_v4: packed selected attention policy={}; staged=QWEN_DSV4_ONLINE_DIRECT_LOAD=0 legacy=QWEN_DSV4_PACKED_SELECTED_ONLINE=0",
            if direct_load {
                "online-direct"
            } else if online {
                "online-staged"
            } else {
                "legacy"
            },
        );
    });
    encode_cooperative_selected_sink_attention_f16(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        DeepSeekV4RawCacheLayout::Chunk,
        rows.attention_cache,
        rows.capacity_rows,
        sparse.cache_order_ids,
        sparse.selected_counts,
        sparse.visible_counts,
        sinks,
        output,
        start_position,
        sparse.query_offset,
        sparse.query_count,
        n_tokens,
        DEEPSEEK_V4_CSA_TOP_K,
        online,
        direct_load,
        config,
    )
}

impl DeepSeekV4Session {
    fn packed_grouped_expert_policy_for_chunk(
        &self,
        ctx: &MetalContext,
        n_tokens: usize,
    ) -> Result<PackedExpertPolicy, DeepSeekV4MetalError> {
        packed_grouped_expert_policy(
            ctx,
            n_tokens,
            self.residency.report().tensor_count,
            self.residency.report().source_bytes,
            self.prefill.moe.expert_count,
        )
    }

    /// Execute one layer-major chunk and expose logits for its final token.
    /// Weight projections are batched; causal cache/compressor transitions
    /// remain position-ordered and preserve any retained prefix.
    pub fn prefill_tokens(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.prefill_tokens_with_progress(ctx, token_ids, |_| {})
    }

    pub fn prefill_tokens_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        mut layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.execute_packed_tokens_with_progress(ctx, token_ids, true, &mut layer_completed)?;
        self.logits()
    }

    /// Advance one layer-major teacher-forced chunk without computing logits.
    /// Successful advancement revokes the session's current logits and final
    /// hidden observation; host values copied out earlier remain owned copies.
    pub fn advance_tokens(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<(), DeepSeekV4MetalError> {
        self.execute_packed_tokens_with_progress(ctx, token_ids, false, &mut |_| {})
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub fn profile_packed_chunk(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
    ) -> Result<PackedChunkProfile, DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        let expert_policy = if packed_grouped_expert_scope(grouped_mode, token_ids.len())? {
            self.packed_grouped_expert_policy_for_chunk(ctx, token_ids.len())?
        } else {
            PackedExpertPolicy::Current
        };
        let mut pre_expert = PackedPrefillStageRecorder::new(ctx, sampled)?;
        let mut post_route = PackedPostRouteStageRecorder::new(ctx, sampled)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            Some(&mut pre_expert),
            Some(&mut post_route),
            None,
            None,
            &mut |_| {},
        )?;
        let pre_expert = pre_expert.resolve(ctx)?;
        let mut post_route = post_route.resolve(ctx)?;
        post_route.q8_compressor_matrix_invocations =
            self.prefill.compressor.q8_matrix_invocations();
        Ok(PackedChunkProfile {
            pre_expert,
            post_route,
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub fn profile_packed_post_route(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
    ) -> Result<PackedPostRouteStageProfile, DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        let expert_policy = if packed_grouped_expert_scope(grouped_mode, token_ids.len())? {
            self.packed_grouped_expert_policy_for_chunk(ctx, token_ids.len())?
        } else {
            PackedExpertPolicy::Current
        };
        let mut recorder = PackedPostRouteStageRecorder::new(ctx, sampled)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            Some(&mut recorder),
            None,
            None,
            &mut |_| {},
        )?;
        let mut profile = recorder.resolve(ctx)?;
        profile.q8_compressor_matrix_invocations = self.prefill.compressor.q8_matrix_invocations();
        Ok(profile)
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn mhc_delete_expert_policy(
        &self,
        ctx: &MetalContext,
        n_tokens: usize,
    ) -> Result<PackedExpertPolicy, DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        if packed_grouped_iq_expert_count_qualified(self.prefill.moe.expert_count)
            && packed_grouped_expert_scope(grouped_mode, n_tokens)?
        {
            self.packed_grouped_expert_policy_for_chunk(ctx, n_tokens)
        } else {
            Ok(PackedExpertPolicy::Current)
        }
    }

    #[cfg(feature = "dsv4-diagnostics")]
    #[allow(clippy::too_many_arguments)]
    fn mhc_delete_oracle_identity(
        &self,
        ctx: &MetalContext,
        token_ids: &[u32],
        start_position: u32,
        route_policy: PackedRoutePolicy,
        expert_policy: PackedExpertPolicy,
        mxfp4_matrix: bool,
        q_b_projection: Q8PrecisionProjection,
        output_projection: Q8PrecisionProjection,
        compressor_matrix: bool,
        shared_matrix: bool,
        router_e8p32_strict: bool,
        q_a_kv_matrix: PackedQaKvMatrixMode,
        indexer_q_matrix: bool,
        stage_profile_active: bool,
        post_route_profile_active: bool,
    ) -> Result<DeepSeekV4MhcOracleIdentity, DeepSeekV4MetalError> {
        use sha2::{Digest, Sha256};

        let model_content_id = self.snapshot_model_content_id.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "mHC oracle capture requires a bound model-content identity".into(),
            )
        })?;
        if !packed_shared_route_overlap_enabled() {
            return invalid("mHC campaign requires the production shared-overlap policy");
        }
        let compatibility = self.snapshot_compatibility_digest()?;
        let config = self.residency.config();
        let policy_manifest = format!(
            "route={};expert={};mxfp4_matrix={mxfp4_matrix};qb={};out={};compressor={compressor_matrix};shared={shared_matrix};router={router_e8p32_strict};qakv={};indexer={indexer_q_matrix};group8={};rope={};batched_compressor={};indexer_rope={};indexer_visible={};indexer_tiled={};selected_online={};direct_load={};shared_overlap={};q3q4={};gpu_iq3={};grouped_iq3={};iq2_f16={};iq2_mm={};grouped_output={};stage={stage_profile_active};post={post_route_profile_active}",
            route_policy.label(),
            expert_policy.label(),
            q_b_projection.label(),
            output_projection.label(),
            q_a_kv_matrix.label(),
            packed_grouped_dense_attention_enabled(),
            packed_batched_rope_enabled(),
            packed_batched_compressor_enabled(),
            packed_indexer_batched_rope_enabled(),
            packed_indexer_visible_dispatch_enabled(),
            packed_indexer_tiled_f32_enabled(),
            packed_selected_online_enabled(),
            deepseek_v4_online_direct_load_enabled(),
            packed_shared_route_overlap_enabled(),
            packed_grouped_q3q4_enabled(),
            packed_gpu_route_iq3_enabled(),
            packed_grouped_iq3_enabled(),
            packed_iq2_f16_mm64x32_enabled(),
            packed_iq2_mm64x32_enabled(),
            packed_q8_grouped_output_enabled(),
        );
        let mut policy = Sha256::new();
        policy.update(b"qwen.dsv4.mhc-delete-policy.v1\0");
        policy.update(policy_manifest.as_bytes());
        let mut tokens = Sha256::new();
        tokens.update(b"qwen.dsv4.mhc-delete-tokens.v1\0");
        tokens.update(bytemuck::cast_slice(token_ids));
        Ok(DeepSeekV4MhcOracleIdentity {
            model_content_id: *model_content_id.as_bytes(),
            compatibility_id: *compatibility.as_bytes(),
            token_sha256: tokens.finalize().into(),
            policy_sha256: policy.finalize().into(),
            policy_manifest,
            start_position,
            n_tokens: token_ids.len(),
            rms_epsilon_bits: config.attention_rms_epsilon.to_bits(),
            hc_epsilon_bits: config.hyper_connection_epsilon.to_bits(),
            device_registry_id: ctx.device.registryID(),
            residency_tensor_count: self.residency.report().tensor_count,
            residency_source_bytes: self.residency.report().source_bytes,
            expert_count: self.prefill.moe.expert_count,
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn execute_mhc_delete_capture(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<(DeepSeekV4MhcCapture, DeepSeekV4MhcDeleteProfile), DeepSeekV4MetalError> {
        let buffer = PackedMhcOracleBuffer::new_capture(ctx, token_ids.len())?;
        let mut execution = PackedMhcExecution::capture(buffer.clone());
        let mut collector = PackedMhcCommandCollector::default();
        let expert_policy = self.mhc_delete_expert_policy(ctx, token_ids.len())?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            true,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            None,
            Some(&mut execution),
            Some(&mut collector),
            &mut |_| {},
        )?;
        let (sites, identity) = execution.finish()?;
        let payload_sha256 = buffer.current_payload_sha256()?;
        let (wall_ms, command_intervals) = collector.finish()?;
        let capture = DeepSeekV4MhcCapture {
            buffer,
            identity,
            payload_sha256,
        };
        Ok((
            capture,
            DeepSeekV4MhcDeleteProfile {
                execution: DeepSeekV4MhcExecutionKind::Capture,
                queue_identity: Retained::as_ptr(&ctx.queue).cast::<()>() as usize as u64,
                wall_ms,
                sites,
                command_intervals,
            },
        ))
    }

    /// Capture all 86 exact packed mHC function outputs and complete
    /// endpoint/restored-continuation evidence as one indivisible transaction.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn capture_mhc_delete_oracle(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        continuation_token: u32,
    ) -> Result<(DeepSeekV4MhcVerifiedCapture, DeepSeekV4MhcDeleteProfile), DeepSeekV4MetalError>
    {
        let (capture, profile) = self.execute_mhc_delete_capture(ctx, token_ids)?;
        let evidence = self.capture_mhc_delete_endpoint_evidence(ctx, continuation_token)?;
        Ok((DeepSeekV4MhcVerifiedCapture { capture, evidence }, profile))
    }

    /// Execute an untimed structural C arm and discard its unreplayable
    /// payload after returning the 86-site and command ledgers.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn preflight_mhc_delete_capture(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<DeepSeekV4MhcDeleteProfile, DeepSeekV4MetalError> {
        let (_, profile) = self.execute_mhc_delete_capture(ctx, token_ids)?;
        Ok(profile)
    }

    /// Copy only the fixed-size endpoint evidence permitted after a timed arm.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn capture_mhc_delete_timed_endpoint(
        &self,
    ) -> Result<DeepSeekV4MhcTimedEndpoint, DeepSeekV4MetalError> {
        let logits_bits = self
            .copy_logits_f32()?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let hidden_bits = self
            .copy_final_normalized_hidden_f32()?
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let tokens = self.committed_tokens();
        Ok(DeepSeekV4MhcTimedEndpoint {
            sha256: mhc_timed_endpoint_sha256(
                &logits_bits,
                &hidden_bits,
                tokens,
                self.next_position(),
            ),
            position: self.next_position(),
            token_count: tokens.len(),
            logits_count: logits_bits.len(),
            hidden_count: hidden_bits.len(),
        })
    }

    /// Capture complete bitwise endpoint and restored-continuation evidence.
    /// This is intentionally separate from timed packet execution.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn capture_mhc_delete_endpoint_evidence(
        &mut self,
        ctx: &MetalContext,
        continuation_token: u32,
    ) -> Result<DeepSeekV4MhcEndpointEvidence, DeepSeekV4MetalError> {
        let endpoint_logits_bits = self
            .copy_logits_f32()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let endpoint_hidden_bits = self
            .copy_final_normalized_hidden_f32()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let endpoint_position = self.next_position();
        let endpoint_tokens = self.committed_tokens().to_vec();
        let endpoint = self.capture_causal_snapshot()?;
        self.restore_causal_snapshot(&endpoint)?;
        self.forward_token(ctx, continuation_token)?;
        let continuation_logits_bits = self
            .copy_logits_f32()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let continuation_hidden_bits = self
            .copy_final_normalized_hidden_f32()?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let continuation_position = self.next_position();
        let continuation_tokens = self.committed_tokens().to_vec();
        let continuation = self.capture_causal_snapshot()?;
        if endpoint.source_observation() != DeepSeekV4SnapshotObservation::Available
            || continuation.source_observation() != DeepSeekV4SnapshotObservation::Available
        {
            return invalid("mHC endpoint evidence lost its observation state");
        }
        let mut evidence = DeepSeekV4MhcEndpointEvidence {
            endpoint_logits_bits,
            endpoint_hidden_bits,
            endpoint_position,
            endpoint_tokens,
            endpoint_prefix_digest: *endpoint.prefix_digest(),
            endpoint_compatibility_id: *endpoint.compatibility_digest().as_bytes(),
            endpoint_causal_digest: *endpoint.causal_digest(),
            endpoint_observation: endpoint.source_observation(),
            continuation_logits_bits,
            continuation_hidden_bits,
            continuation_position,
            continuation_tokens,
            continuation_prefix_digest: *continuation.prefix_digest(),
            continuation_compatibility_id: *continuation.compatibility_digest().as_bytes(),
            continuation_causal_digest: *continuation.causal_digest(),
            continuation_observation: continuation.source_observation(),
            sha256: [0; 32],
        };
        evidence.refresh_sha256();
        Ok(evidence)
    }

    /// Execute one frozen mHC deletion arm under the ordinary packed policy.
    /// Passive command intervals are read only after existing waits.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn execute_mhc_delete_arm(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        arm: DeepSeekV4MhcDeleteArm,
        oracle: &DeepSeekV4MhcOracle,
    ) -> Result<DeepSeekV4MhcDeleteProfile, DeepSeekV4MetalError> {
        self.execute_mhc_delete_arm_inner(ctx, token_ids, arm, oracle, false)
    }

    /// Execute one untimed structural-preflight arm and retain its 86-site
    /// producer/consumer ledger.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn preflight_mhc_delete_arm(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        arm: DeepSeekV4MhcDeleteArm,
        oracle: &DeepSeekV4MhcOracle,
    ) -> Result<DeepSeekV4MhcDeleteProfile, DeepSeekV4MetalError> {
        self.execute_mhc_delete_arm_inner(ctx, token_ids, arm, oracle, true)
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn execute_mhc_delete_arm_inner(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        arm: DeepSeekV4MhcDeleteArm,
        oracle: &DeepSeekV4MhcOracle,
        record_sites: bool,
    ) -> Result<DeepSeekV4MhcDeleteProfile, DeepSeekV4MetalError> {
        oracle.validate_replay(ctx, token_ids.len())?;
        let mut execution = PackedMhcExecution::replay(arm, oracle, record_sites);
        let execution_kind = execution.kind;
        let mut collector = PackedMhcCommandCollector::default();
        let expert_policy = self.mhc_delete_expert_policy(ctx, token_ids.len())?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            true,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            None,
            Some(&mut execution),
            Some(&mut collector),
            &mut |_| {},
        )?;
        let (sites, _) = execution.finish()?;
        let (wall_ms, command_intervals) = collector.finish()?;
        Ok(DeepSeekV4MhcDeleteProfile {
            execution: execution_kind,
            queue_identity: Retained::as_ptr(&ctx.queue).cast::<()>() as usize as u64,
            wall_ms,
            sites,
            command_intervals,
        })
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_route_policy_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        gpu_route: bool,
        preserve_cpu_weights: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        let mut duplicate_hash_routes = false;
        if gpu_route {
            for layer in 0..self.residency.config().hash_layer_count as usize {
                duplicate_hash_routes |= packed_hash_route_has_duplicate_slots(
                    token_ids,
                    self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?,
                    self.prefill.moe.expert_count,
                )?;
            }
        }
        let route_policy =
            packed_diagnostic_route_policy(gpu_route, preserve_cpu_weights, duplicate_hash_routes)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            route_policy,
            PackedExpertPolicy::Current,
            None,
            None,
            None,
            None,
            &mut |_| {},
        )
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_expert_policy_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        grouped_target: bool,
        grouped_iq3_target: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        let expert_policy = match (grouped_target, grouped_iq3_target) {
            (false, false) => PackedExpertPolicy::Current,
            (true, false) => PackedExpertPolicy::GroupedIq2XsIq3Xxs,
            (true, true) => PackedExpertPolicy::GroupedIq2XsIq3XxsAndIq3Xxs,
            (false, true) => {
                return invalid("packed grouped IQ3 test policy requires the IQ2 baseline");
            }
        };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            None,
            None,
            None,
            &mut |_| {},
        )
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_stage_profile_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
    ) -> Result<PackedPrefillStageProfile, DeepSeekV4MetalError> {
        let mut recorder = PackedPrefillStageRecorder::new(ctx, sampled)?;
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            PackedExpertPolicy::GroupedIq2XsIq3Xxs,
            Some(&mut recorder),
            None,
            None,
            None,
            &mut |_| {},
        )?;
        recorder.resolve(ctx)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn execute_packed_tokens_with_post_route_stage_profile_for_test(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        sampled: bool,
        grouped_iq3: bool,
    ) -> Result<PackedPostRouteStageProfile, DeepSeekV4MetalError> {
        let mut recorder = PackedPostRouteStageRecorder::new(ctx, sampled)?;
        let expert_policy = if grouped_iq3 {
            PackedExpertPolicy::GroupedIq2XsIq3XxsAndIq3Xxs
        } else {
            PackedExpertPolicy::GroupedIq2XsIq3Xxs
        };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            None,
            Some(&mut recorder),
            None,
            None,
            &mut |_| {},
        )?;
        recorder.resolve(ctx)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn packed_route_generation_for_test(&self) -> u32 {
        self.prefill.moe.gpu_route.next_generation.get()
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn packed_grouped_iq2_invocations_for_test(&self) -> u32 {
        self.prefill.moe.grouped_iq2_invocations.get()
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    pub(super) fn packed_grouped_iq3_invocations_for_test(&self) -> u32 {
        self.prefill.moe.grouped_iq3_invocations.get()
    }

    fn execute_packed_tokens_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let grouped_mode = packed_grouped_expert_mode();
        let expert_policy =
            if packed_grouped_iq_expert_count_qualified(self.prefill.moe.expert_count)
                && packed_grouped_expert_scope(grouped_mode, token_ids.len())?
            {
                self.packed_grouped_expert_policy_for_chunk(ctx, token_ids.len())?
            } else {
                PackedExpertPolicy::Current
            };
        self.execute_packed_tokens_with_progress_policy(
            ctx,
            token_ids,
            emit_logits,
            PackedRoutePolicy::Cpu,
            expert_policy,
            #[cfg(feature = "dsv4-diagnostics")]
            None,
            #[cfg(feature = "dsv4-diagnostics")]
            None,
            #[cfg(feature = "dsv4-diagnostics")]
            None,
            #[cfg(feature = "dsv4-diagnostics")]
            None,
            layer_completed,
        )
    }

    fn execute_packed_tokens_with_progress_policy(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        emit_logits: bool,
        route_policy: PackedRoutePolicy,
        expert_policy: PackedExpertPolicy,
        #[cfg(feature = "dsv4-diagnostics")] stage_recorder: Option<
            &mut PackedPrefillStageRecorder,
        >,
        #[cfg(feature = "dsv4-diagnostics")] post_route_stage_recorder: Option<
            &mut PackedPostRouteStageRecorder,
        >,
        #[cfg(feature = "dsv4-diagnostics")] mut mhc_execution: Option<&mut PackedMhcExecution>,
        #[cfg(feature = "dsv4-diagnostics")] mut mhc_command_collector: Option<
            &mut PackedMhcCommandCollector,
        >,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let trace_layers = std::env::var_os("QWEN_DSV4_PREFILL_TRACE").is_some();
        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics
            .ensure_no_active_capture("execute packed tokens")?;
        #[cfg(feature = "dsv4-diagnostics")]
        if mhc_execution.is_some()
            && (mhc_command_collector.is_none()
                || stage_recorder.is_some()
                || post_route_stage_recorder.is_some()
                || trace_layers
                || self.fp4_shadow_diagnostics.lineage_enabled()
                || self.fp4_shadow_diagnostics.is_capturing()
                || self.fp4_selection_mode.is_counterfactual())
        {
            return invalid(
                "mHC oracle execution requires ordinary untraced packed topology and passive command timing",
            );
        }
        if self.prefill.moe.expert_count != MOE_EXPERT_COUNT && route_policy.uses_gpu() {
            return invalid("packed GPU routing is not enabled for compact-expert DeepSeek V4");
        }
        self.residency.validate_context(ctx)?;
        let n_tokens = checked_token_count(token_ids.len())?;
        let device_name = ctx.device.name().to_string();
        let residency_tensor_count = self.residency.report().tensor_count;
        let residency_source_bytes = self.residency.report().source_bytes;
        let expert_count = self.prefill.moe.expert_count;
        let route_policy = if route_policy == PackedRoutePolicy::Cpu
            && expert_count == MOE_EXPERT_COUNT
            && packed_gpu_route_compact_enabled()
            && token_ids.len() <= PACKED_GPU_ROUTE_MAX_TOKENS
            && packed_gpu_route_compact_scope_qualified(
                &device_name,
                residency_tensor_count,
                residency_source_bytes,
                expert_count,
                token_ids.len(),
            ) {
            PackedRoutePolicy::GpuCompact
        } else {
            route_policy
        };
        let expert_policy =
            if packed_grouped_iq3_enabled() && packed_grouped_iq3_candidate_supported(ctx) {
                expert_policy.with_iq3_target()
            } else {
                expert_policy
            };
        let mxfp4_matrix = packed_mxfp4_matrix_enabled()
            && packed_mxfp4_matrix_scope_qualified(
                &device_name,
                residency_tensor_count,
                residency_source_bytes,
                expert_count,
                token_ids.len(),
            )
            && packed_mxfp4_matrix_candidate_supported(ctx);
        if self.prefill.moe.expert_count != MOE_EXPERT_COUNT {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: packed compact-expert prefill uses CPU routing with runtime expert geometry"
                );
            });
        }
        let q_b_projection =
            packed_q8_qb_projection_for_chunk(ctx, &self.residency, token_ids.len())?;
        let output_projection =
            packed_q8_output_projection_for_chunk(ctx, &self.residency, token_ids.len())?;
        let compressor_matrix =
            packed_q8_compressor_matrix_for_chunk(ctx, &self.residency, token_ids.len());
        let shared_matrix =
            packed_q8_shared_matrix_for_chunk(ctx, &self.residency, token_ids.len());
        let router_e8p32_strict = packed_router_e8p32_strict_enabled()
            && packed_router_e8p32_scope_qualified(
                &device_name,
                residency_tensor_count,
                residency_source_bytes,
                expert_count,
                token_ids.len(),
            );
        let q_a_kv_matrix = if packed_q8_qa_kv_matrix_scope_qualified(
            &device_name,
            residency_tensor_count,
            residency_source_bytes,
            expert_count,
            token_ids.len(),
        ) {
            packed_q8_qa_kv_matrix_mode()
        } else {
            PackedQaKvMatrixMode::Off
        };
        let indexer_q_matrix = packed_indexer_q_matrix_enabled()
            && packed_indexer_q_matrix_scope_qualified(
                &device_name,
                residency_tensor_count,
                residency_source_bytes,
                expert_count,
                token_ids.len(),
            );
        #[cfg(feature = "dsv4-diagnostics")]
        self.prefill.compressor.reset_q8_matrix_invocations();
        #[cfg(feature = "dsv4-diagnostics")]
        validate_packed_route_policy_scope(route_policy, token_ids.len())?;
        if expert_policy.uses_iq2_target() && token_ids.len() > PACKED_GROUPED_EXPERT_MAX_TOKENS {
            return invalid(format!(
                "packed grouped expert policy exceeds its {PACKED_GROUPED_EXPERT_MAX_TOKENS}-token qualification"
            ));
        }
        let start_position = self.phase.ready_position()?;
        let end_position = start_position
            .checked_add(n_tokens)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
        for (index, &token) in token_ids.iter().enumerate() {
            if token as usize >= DEEPSEEK_V4_VOCAB_SIZE {
                return invalid(format!(
                    "packed token {index} id {token} is outside vocabulary {DEEPSEEK_V4_VOCAB_SIZE}"
                ));
            }
            let position = start_position
                .checked_add(u32::try_from(index).map_err(|_| {
                    DeepSeekV4MetalError::Invalid("packed token index exceeds u32".into())
                })?)
                .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
            self.capacity.validate_position(position)?;
        }
        self.validate_committed_token_append(start_position, token_ids.len())?;
        #[cfg(feature = "dsv4-diagnostics")]
        if self.fp4_selection_mode.is_counterfactual() {
            // Reject unsupported geometry while the session is still ready;
            // token staging and causal mutation both occur below this gate.
            validate_fp4_selection_counterfactual_packed(start_position, token_ids.len())?;
        }
        #[cfg(feature = "dsv4-diagnostics")]
        if let Some(execution) = mhc_execution.as_deref_mut() {
            if start_position != 0 {
                return invalid(format!(
                    "mHC oracle execution requires position zero, got {start_position}"
                ));
            }
            let identity = self.mhc_delete_oracle_identity(
                ctx,
                token_ids,
                start_position,
                route_policy,
                expert_policy,
                mxfp4_matrix,
                q_b_projection,
                output_projection,
                compressor_matrix,
                shared_matrix,
                router_e8p32_strict,
                q_a_kv_matrix,
                indexer_q_matrix,
                stage_recorder.is_some(),
                post_route_stage_recorder.is_some(),
            )?;
            execution.bind_identity(identity)?;
        }
        #[cfg(feature = "dsv4-diagnostics")]
        if let Some(collector) = mhc_command_collector.as_deref_mut() {
            collector.begin_wall()?;
        }
        let token_values = token_ids
            .iter()
            .map(|&token| token as i32)
            .collect::<Vec<_>>();
        let token_view = i32_prefix(
            &self.prefill.token_ids,
            vec![token_ids.len() as u64],
            "packed token IDs",
        )?;
        host_write_i32(&token_view, &token_values, "packed token IDs")?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            let final_position = end_position - 1;
            let sparse_query_count = sparse_csa_query_offset(start_position, token_ids.len())
                .map(|offset| token_ids.len() - offset)
                .unwrap_or(0);
            self.fp4_shadow_diagnostics
                .begin_packed(final_position, sparse_query_count)?;
        }

        let begun_position = self.phase.begin_mutation()?;
        debug_assert_eq!(begun_position, start_position);
        let result = self.prefill_tokens_inner(
            ctx,
            token_ids,
            &token_view,
            start_position,
            emit_logits,
            route_policy,
            expert_policy,
            mxfp4_matrix,
            q_b_projection,
            output_projection,
            compressor_matrix,
            shared_matrix,
            router_e8p32_strict,
            q_a_kv_matrix,
            indexer_q_matrix,
            trace_layers,
            #[cfg(feature = "dsv4-diagnostics")]
            stage_recorder,
            #[cfg(feature = "dsv4-diagnostics")]
            post_route_stage_recorder,
            #[cfg(feature = "dsv4-diagnostics")]
            mhc_execution,
            #[cfg(feature = "dsv4-diagnostics")]
            mhc_command_collector.as_deref_mut(),
            layer_completed,
        );
        match result {
            Ok(()) => {
                self.commit_tokens(token_ids);
                self.phase
                    .complete_mutation(start_position, end_position, emit_logits)?;
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(collector) = mhc_command_collector {
                    collector.end_wall()?;
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn prefill_tokens_inner(
        &mut self,
        ctx: &MetalContext,
        token_ids: &[u32],
        token_view: &MetalTensor,
        start_position: u32,
        emit_logits: bool,
        route_policy: PackedRoutePolicy,
        expert_policy: PackedExpertPolicy,
        mxfp4_matrix: bool,
        q_b_projection: Q8PrecisionProjection,
        output_projection: Q8PrecisionProjection,
        compressor_matrix: bool,
        shared_matrix: bool,
        router_e8p32_strict: bool,
        q_a_kv_matrix: PackedQaKvMatrixMode,
        indexer_q_matrix: bool,
        trace_layers: bool,
        #[cfg(feature = "dsv4-diagnostics")] mut stage_recorder: Option<
            &mut PackedPrefillStageRecorder,
        >,
        #[cfg(feature = "dsv4-diagnostics")] mut post_route_stage_recorder: Option<
            &mut PackedPostRouteStageRecorder,
        >,
        #[cfg(feature = "dsv4-diagnostics")] mut mhc_execution: Option<&mut PackedMhcExecution>,
        #[cfg(feature = "dsv4-diagnostics")] mut mhc_command_collector: Option<
            &mut PackedMhcCommandCollector,
        >,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let n_tokens = token_ids.len();
        let device_name = ctx.device.name().to_string();
        let residency_tensor_count = self.residency.report().tensor_count;
        let residency_source_bytes = self.residency.report().source_bytes;
        let expert_count = self.prefill.moe.expert_count;
        if packed_grouped_dense_attention_enabled() {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: grouped-head online dense attention active; rollback=QWEN_DSV4_PACKED_GROUP8_DENSE=0"
                );
            }
        }
        if compressor_matrix {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: Q8 compressor matrices active for N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_COMPRESSOR_MATRIX=0"
                );
            }
        }
        if shared_matrix {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: shared-expert Q8 matrices active for N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_SHARED_MATRIX=0"
                );
            }
        }
        if indexer_q_matrix {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: wide F32 Q8 sparse-indexer Q matrix active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_INDEXER_Q_MATRIX=0"
                );
            }
        }
        if q_a_kv_matrix != PackedQaKvMatrixMode::Off {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: F32 Q8 Q-A/raw-KV matrix mode={q_a_kv_matrix:?} active for N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_QA_KV_MATRIX=0"
                );
            }
        }
        if q_b_projection.uses_full_chunk_f32(n_tokens) {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let rollback = match q_b_projection {
                    Q8PrecisionProjection::WideF32Matrix => "f32_matrix",
                    _ => "exact",
                };
                eprintln!(
                    "deepseek_v4: Q8 Q-B matrix policy={} active for N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_QB={rollback}; exact_rollback=QWEN_DSV4_PACKED_Q8_QB=exact",
                    q_b_projection.label(),
                );
            }
        }
        if output_projection.uses_full_chunk_f32(n_tokens) {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let rollback = match output_projection {
                    Q8PrecisionProjection::WideF32Matrix => "f32_matrix",
                    _ => "exact",
                };
                eprintln!(
                    "deepseek_v4: Q8 output A/B matrix policy={} active for N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_Q8_OUTPUT={rollback}; exact_rollback=QWEN_DSV4_PACKED_Q8_OUTPUT=exact",
                    output_projection.label(),
                );
            }
        }
        if expert_policy.uses_iq2_mma16(n_tokens) {
            let mut eligible_layers = 0usize;
            for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
                let gate = self.layer_tensor(layer, "ffn_gate_exps.weight")?;
                let up = self.layer_tensor(layer, "ffn_up_exps.weight")?;
                let down = self.layer_tensor(layer, "ffn_down_exps.weight")?;
                eligible_layers += usize::from(
                    gate.dtype == GgmlType::IQ2_XS
                        && up.dtype == GgmlType::IQ2_XS
                        && down.dtype == GgmlType::IQ3_XXS,
                );
            }
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if eligible_layers > 0 && !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                if packed_iq2_f16_mm64x32_enabled() {
                    eprintln!(
                        "deepseek_v4: half-staged 64x32 IQ2 packed prefill active for qualified N={n_tokens} chunks; eligible_layers={eligible_layers}; rollback=QWEN_DSV4_PACKED_IQ2_F16_MATRIX=0"
                    );
                } else if packed_iq2_mm64x32_enabled() {
                    eprintln!(
                        "deepseek_v4: 64x32 IQ2 packed prefill active for qualified N={n_tokens} chunks; eligible_layers={eligible_layers}; rollback=QWEN_DSV4_PACKED_IQ2_MM64X32=0"
                    );
                } else {
                    eprintln!(
                        "deepseek_v4: BM16 IQ2 packed prefill active for qualified N={n_tokens} chunks; eligible_layers={eligible_layers}; rollback=QWEN_DSV4_PACKED_BM16_IQ2=0"
                    );
                }
            }
        }
        if expert_policy.uses_iq3_target() {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: grouped all-IQ3 packed experts active for full N={n_tokens} chunks; rollback=QWEN_DSV4_PACKED_GROUPED_IQ3=0"
                );
            }
        }
        if route_policy == PackedRoutePolicy::GpuCompact {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let scope = if packed_gpu_route_iq3_enabled() {
                    "IQ2/all-IQ3"
                } else {
                    "IQ2"
                };
                eprintln!(
                    "deepseek_v4: compact GPU routing active for qualified N={n_tokens} {scope} layers; unprofiled execution merges router and experts; rollback=QWEN_DSV4_PACKED_GPU_ROUTE_COMPACT=0"
                );
            }
        }
        let last_position = start_position
            .checked_add(u32::try_from(n_tokens - 1).map_err(|_| {
                DeepSeekV4MetalError::Invalid("packed token count exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("packed position overflow".into()))?;
        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;
        let attention_config = deepseek_v4_session_attention_config();
        let attention_dims = attention_config.checked()?;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_score_plan = self
            .fp4_selection_mode
            .score_plan(self.fp4_shadow_diagnostics.is_capturing());
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_score_dispatch_ledger = DeepSeekV4Fp4ScoreDispatchLedger::new(
            DeepSeekV4Fp4ShadowExecution::Packed,
            last_position,
            fp4_score_plan.kind(),
            fp4_score_plan.consumed_source(),
        );
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = None;
        }
        let batched_rope = packed_batched_rope_enabled();
        let batched_compressor = packed_batched_compressor_enabled();
        if !batched_rope {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: packed batched RoPE disabled; rollback=QWEN_DSV4_BATCHED_ROPE=0"
                );
            }
        }
        if !batched_compressor {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "deepseek_v4: packed batched compressor disabled; rollback=QWEN_DSV4_BATCHED_COMPRESSOR=0"
                );
            }
        }
        let mut layer_traces = Vec::with_capacity(if trace_layers {
            DEEPSEEK_V4_LAYER_COUNT
        } else {
            0
        });
        let embedding = f32_prefix(
            &self.prefill.embedding,
            vec![DEEPSEEK_V4_HIDDEN_SIZE as u64, n_tokens as u64],
            "packed embeddings",
        )?;
        let residual_primary = f32_prefix(
            &self.prefill.residual_primary,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed primary residual",
        )?;
        let residual_secondary = f32_prefix(
            &self.prefill.residual_secondary,
            vec![
                DEEPSEEK_V4_HIDDEN_SIZE as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
                n_tokens as u64,
            ],
            "packed secondary residual",
        )?;

        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let pre_expert_started = trace_layers.then(std::time::Instant::now);
            let routed_gate_dtype = self.layer_tensor(layer, "ffn_gate_exps.weight")?.dtype;
            let routed_up_dtype = self.layer_tensor(layer, "ffn_up_exps.weight")?.dtype;
            let routed_down_dtype = self.layer_tensor(layer, "ffn_down_exps.weight")?.dtype;
            let compact_gpu_route = route_policy == PackedRoutePolicy::GpuCompact
                && packed_gpu_compact_expert_layer_qualified(
                    ctx,
                    expert_policy,
                    n_tokens,
                    packed_gpu_route_iq3_enabled(),
                    routed_gate_dtype,
                    routed_up_dtype,
                    routed_down_dtype,
                );
            #[cfg(feature = "dsv4-diagnostics")]
            let merge_gpu_route = compact_gpu_route
                && !trace_layers
                && stage_recorder.is_none()
                && post_route_stage_recorder.is_none()
                && !self.fp4_shadow_diagnostics.is_capturing()
                && !self.fp4_selection_mode.is_counterfactual();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let merge_gpu_route = compact_gpu_route;
            #[cfg(feature = "dsv4-diagnostics")]
            let overlap_shared_route = packed_shared_route_overlap_enabled()
                && route_policy == PackedRoutePolicy::Cpu
                && packed_shared_route_overlap_scope_qualified(
                    &device_name,
                    residency_tensor_count,
                    residency_source_bytes,
                    expert_count,
                    n_tokens,
                    routed_gate_dtype,
                    routed_up_dtype,
                    routed_down_dtype,
                )
                && !trace_layers
                && stage_recorder.is_none()
                && post_route_stage_recorder.is_none()
                && !self.fp4_shadow_diagnostics.is_capturing()
                && !self.fp4_selection_mode.is_counterfactual();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let overlap_shared_route = packed_shared_route_overlap_enabled()
                && route_policy == PackedRoutePolicy::Cpu
                && packed_shared_route_overlap_scope_qualified(
                    &device_name,
                    residency_tensor_count,
                    residency_source_bytes,
                    expert_count,
                    n_tokens,
                    routed_gate_dtype,
                    routed_up_dtype,
                    routed_down_dtype,
                );
            let gpu_route_generation = if compact_gpu_route
                || (route_policy.uses_gpu() && route_policy != PackedRoutePolicy::GpuCompact)
            {
                Some(self.prefill.moe.take_gpu_route_generation()?)
            } else {
                None
            };
            let raw_cache = self.raw_cache_layer(layer)?;
            let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;
            let attention_kind = self.residency.config().attention_kinds[layer];
            let sparse_query_offset = (attention_kind == AttentionKind::CompressedSparse)
                .then(|| sparse_csa_query_offset(start_position, n_tokens))
                .flatten();
            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate packed layer {layer} router command buffer"
                ))
            })?;
            #[cfg(feature = "dsv4-diagnostics")]
            let mut encoder =
                PackedPrefillLayerEncoder::begin(&command, layer, stage_recorder.as_deref_mut())?;
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let encoder = KernelEncoder::begin(&command);
            let router_result = (|| {
                #[cfg(feature = "dsv4-diagnostics")]
                let mut captured_sparse: Option<PackedSparseCsaViews> = None;
                #[cfg(not(feature = "dsv4-diagnostics"))]
                let captured_sparse: Option<PackedSparseCsaViews> = None;
                encode_copy_raw_ring_f16_bits(
                    ctx,
                    &encoder,
                    &raw_cache,
                    &self.prefill.attention.raw_cache_before_chunk,
                )?;
                if layer == 0 {
                    encode_get_rows_f32(
                        ctx,
                        &encoder,
                        self.residency.require_tensor("token_embd.weight")?,
                        token_view,
                        &embedding,
                        n_tokens,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                    )?;
                    self.prefill.hyper.encode_initial_repeat(
                        ctx,
                        &encoder,
                        &embedding,
                        &residual_primary,
                        n_tokens,
                    )?;
                }

                let attention_input = self.prefill.hyper.encode_pre(
                    ctx,
                    &encoder,
                    &residual_primary,
                    self.layer_tensor(layer, "hc_attn_fn.weight")?,
                    self.layer_tensor(layer, "hc_attn_scale.weight")?,
                    self.layer_tensor(layer, "hc_attn_base.weight")?,
                    n_tokens,
                    rms_eps,
                    hc_eps,
                    #[cfg(feature = "dsv4-diagnostics")]
                    mhc_execution.as_deref_mut(),
                    #[cfg(feature = "dsv4-diagnostics")]
                    DeepSeekV4MhcSiteKind::Attention,
                    #[cfg(feature = "dsv4-diagnostics")]
                    layer,
                )?;

                let attention = self.prefill.attention.encode_prepare(
                    ctx,
                    &encoder,
                    &attention_input,
                    self.layer_tensor(layer, "attn_norm.weight")?,
                    self.layer_tensor(layer, "attn_q_a.weight")?,
                    self.layer_tensor(layer, "attn_q_a_norm.weight")?,
                    self.layer_tensor(layer, "attn_q_b.weight")?,
                    self.layer_tensor(layer, "attn_kv.weight")?,
                    self.layer_tensor(layer, "attn_kv_a_norm.weight")?,
                    n_tokens,
                    rms_eps,
                    q_b_projection,
                    q_a_kv_matrix,
                )?;

                let compressor = self.prefill.compressor.encode_layer_projections(
                    ctx,
                    &encoder,
                    &self.residency,
                    layer,
                    &attention.normalized_input,
                    n_tokens,
                    compressor_matrix,
                )?;

                let query_heads = attention.queries.view_subrange(
                    0,
                    vec![
                        attention_config.head_dim as u64,
                        attention_config.head_count as u64,
                        n_tokens as u64,
                    ],
                );
                if batched_rope {
                    encode_ds4_rope_tail_adjacent_batch_in_place(
                        ctx,
                        &encoder,
                        &query_heads,
                        start_position,
                        n_tokens,
                        1,
                        rope,
                        false,
                    )?;
                    encode_ds4_rope_tail_adjacent_batch_in_place(
                        ctx,
                        &encoder,
                        &attention.kv,
                        start_position,
                        n_tokens,
                        1,
                        rope,
                        false,
                    )?;
                } else {
                    for row in 0..n_tokens {
                        let position = start_position
                            .checked_add(u32::try_from(row).map_err(|_| {
                                DeepSeekV4MetalError::Invalid("packed RoPE row exceeds u32".into())
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed RoPE position overflow".into(),
                                )
                            })?;
                        let queries = f32_row(
                            &attention.queries,
                            row,
                            attention_dims.query_width,
                            vec![
                                attention_config.head_dim as u64,
                                attention_config.head_count as u64,
                            ],
                            "packed query row",
                        )?;
                        let kv = f32_row(
                            &attention.kv,
                            row,
                            attention_config.head_dim,
                            vec![attention_config.head_dim as u64],
                            "packed KV row",
                        )?;
                        encode_ds4_rope_tail_adjacent_in_place(
                            ctx, &encoder, &queries, position, rope, false,
                        )?;
                        encode_ds4_rope_tail_adjacent_in_place(
                            ctx, &encoder, &kv, position, rope, false,
                        )?;
                    }
                }
                let raw_chunk = f16_prefix(
                    &self.prefill.attention.raw_chunk,
                    vec![attention_config.head_dim as u64, n_tokens as u64],
                    "packed raw chunk",
                )?;
                encode_publish_raw_chunk_f16(
                    ctx,
                    &encoder,
                    &attention.kv,
                    &raw_chunk,
                    &raw_cache,
                    start_position,
                    n_tokens,
                    attention_config.head_dim,
                )?;

                if batched_compressor {
                    self.compressor_frontiers.encode_layer_projected_chunk(
                        ctx,
                        &encoder,
                        &self.residency,
                        layer,
                        start_position,
                        n_tokens,
                        &compressor,
                        &self.prefill.compressor,
                        rope,
                        rms_eps,
                    )?;
                } else {
                    for row in 0..n_tokens {
                        let position = start_position
                            .checked_add(u32::try_from(row).map_err(|_| {
                                DeepSeekV4MetalError::Invalid("packed row exceeds u32".into())
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid("packed position overflow".into())
                            })?;
                        self.compressor_frontiers.encode_layer_projected_row(
                            ctx,
                            &encoder,
                            &self.residency,
                            layer,
                            row,
                            position,
                            &compressor,
                            rope,
                            rms_eps,
                        )?;
                    }
                }

                #[cfg(feature = "dsv4-diagnostics")]
                if sparse_query_offset.is_some() {
                    encoder.boundary(PackedPrefillStageKind::SparseIndexerPrepare)?;
                } else {
                    encoder.skip_stages(
                        &[
                            PackedPrefillStageKind::SparseIndexerPrepare,
                            PackedPrefillStageKind::SparseIndexerScore,
                            PackedPrefillStageKind::SparseSelection,
                        ],
                        PackedPrefillStageKind::AttentionCore,
                    )?;
                }

                let compressed = self
                    .compressor_frontiers
                    .attention_rows(layer, last_position)?;
                if let Some(query_offset) = sparse_query_offset {
                    let rows = self
                        .compressor_frontiers
                        .csa_rows(layer, last_position)?
                        .ok_or_else(|| {
                            DeepSeekV4MetalError::Invalid(format!(
                                "CSA layer {layer} has no rows at sparse position {last_position}"
                            ))
                        })?;
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    let sparse = self.prefill.attention.sparse_csa.encode(
                        ctx,
                        &encoder,
                        &attention.q_lora,
                        &attention.normalized_input,
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        start_position,
                        query_offset,
                        n_tokens,
                        indexer_q_matrix,
                        rope,
                    )?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    let sparse = self.prefill.attention.sparse_csa.encode_prepare(
                        ctx,
                        &encoder,
                        &attention.q_lora,
                        &attention.normalized_input,
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        start_position,
                        query_offset,
                        n_tokens,
                        indexer_q_matrix,
                        rope,
                    )?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    fp4_score_dispatch_ledger.record_common_prepare()?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    encoder.boundary(PackedPrefillStageKind::SparseIndexerScore)?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    if fp4_score_plan.runs_f16() {
                        self.prefill
                            .attention
                            .sparse_csa
                            .encode_f16_scores(ctx, &encoder, rows, &sparse)?;
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    if fp4_score_plan.runs_fp4() {
                        if sparse.query_count != 1 {
                            return invalid(format!(
                                "FP4 packed shadow expected one sparse query, got {}",
                                sparse.query_count
                            ));
                        }
                        self.fp4_shadow.encode(
                            ctx,
                            &encoder,
                            &sparse.index_queries,
                            &sparse.head_weights,
                            rows,
                            &sparse.visible_counts,
                        )?;
                        fp4_score_dispatch_ledger.record_fp4_pipeline()?;
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    if self.fp4_shadow_diagnostics.is_capturing() {
                        captured_sparse = Some(sparse.clone());
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    encoder.boundary(PackedPrefillStageKind::SparseSelection)?;
                    #[cfg(feature = "dsv4-diagnostics")]
                    if fp4_score_plan.runs_f16() {
                        self.prefill
                            .attention
                            .sparse_csa
                            .encode_f16_selection(ctx, &encoder, rows, &sparse)?;
                        fp4_score_dispatch_ledger.record_f16_score_and_selector()?;
                    }
                    #[cfg(feature = "dsv4-diagnostics")]
                    let selected = if fp4_score_plan.consumes_fp4() {
                        PackedCsaSelectionView {
                            query_offset: sparse.query_offset,
                            query_count: sparse.query_count,
                            cache_order_ids: &self.fp4_shadow.cache_order_ids,
                            selected_counts: &self.fp4_shadow.selected_count,
                            visible_counts: &self.fp4_shadow.eligible_visible,
                        }
                    } else {
                        sparse.selection_view()
                    };
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    let selected = sparse.selection_view();
                    #[cfg(feature = "dsv4-diagnostics")]
                    encoder.boundary(PackedPrefillStageKind::AttentionCore)?;
                    if query_offset > 0 {
                        let dense_queries = f32_prefix(
                            &attention.queries,
                            vec![attention_dims.query_width as u64, query_offset as u64],
                            "packed dense-prefix queries",
                        )?;
                        let dense_output = f32_prefix(
                            &attention.attention,
                            vec![attention_dims.query_width as u64, query_offset as u64],
                            "packed dense-prefix attention",
                        )?;
                        let dense_last = start_position
                            .checked_add(u32::try_from(query_offset - 1).map_err(|_| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed dense-prefix offset exceeds u32".into(),
                                )
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed dense-prefix position overflow".into(),
                                )
                            })?;
                        let dense_count = csa_visible_rows(dense_last);
                        encode_packed_dense_sink_attention_f16(
                            ctx,
                            &encoder,
                            &dense_queries,
                            &raw_chunk,
                            &self.prefill.attention.raw_cache_before_chunk,
                            Some(DeepSeekV4PublishedRows {
                                cache: rows.attention_cache,
                                count: dense_count,
                                capacity_rows: rows.capacity_rows,
                            }),
                            self.layer_tensor(layer, "attn_sinks.weight")?,
                            &dense_output,
                            attention_kind,
                            start_position,
                            query_offset,
                        )?;
                    }
                    encode_packed_selected_sink_attention_f16(
                        ctx,
                        &encoder,
                        &attention.queries,
                        &raw_chunk,
                        &self.prefill.attention.raw_cache_before_chunk,
                        rows,
                        selected,
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        &attention.attention,
                        start_position,
                        n_tokens,
                    )?;
                } else {
                    encode_packed_dense_sink_attention_f16(
                        ctx,
                        &encoder,
                        &attention.queries,
                        &raw_chunk,
                        &self.prefill.attention.raw_cache_before_chunk,
                        compressed,
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        &attention.attention,
                        attention_kind,
                        start_position,
                        n_tokens,
                    )?;
                }
                #[cfg(feature = "dsv4-diagnostics")]
                encoder.boundary(PackedPrefillStageKind::InverseRope)?;
                let attention_heads = attention.attention.view_subrange(
                    0,
                    vec![
                        attention_config.head_dim as u64,
                        attention_config.head_count as u64,
                        n_tokens as u64,
                    ],
                );
                if batched_rope {
                    encode_ds4_rope_tail_adjacent_batch_in_place(
                        ctx,
                        &encoder,
                        &attention_heads,
                        start_position,
                        n_tokens,
                        1,
                        rope,
                        true,
                    )?;
                } else {
                    for row in 0..n_tokens {
                        let position = start_position
                            .checked_add(u32::try_from(row).map_err(|_| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed inverse-RoPE row exceeds u32".into(),
                                )
                            })?)
                            .ok_or_else(|| {
                                DeepSeekV4MetalError::Invalid(
                                    "packed inverse-RoPE position overflow".into(),
                                )
                            })?;
                        let attention_row = f32_row(
                            &attention.attention,
                            row,
                            attention_dims.query_width,
                            vec![
                                attention_config.head_dim as u64,
                                attention_config.head_count as u64,
                            ],
                            "packed attention row",
                        )?;
                        encode_ds4_rope_tail_adjacent_in_place(
                            ctx,
                            &encoder,
                            &attention_row,
                            position,
                            rope,
                            true,
                        )?;
                    }
                }

                #[cfg(feature = "dsv4-diagnostics")]
                encoder.boundary(PackedPrefillStageKind::AttentionOutputProjections)?;

                let output_a = self.layer_tensor(layer, "attn_output_a.weight")?;
                let output_b = self.layer_tensor(layer, "attn_output_b.weight")?;
                let attention_output = if output_projection.uses_full_chunk_f32(n_tokens) {
                    self.prefill.attention.encode_output_q8_precision(
                        ctx,
                        &encoder,
                        &attention.attention,
                        output_a,
                        output_b,
                        n_tokens,
                        output_projection,
                        output_projection,
                    )?
                } else {
                    self.prefill.attention.encode_output(
                        ctx,
                        &encoder,
                        &attention.attention,
                        output_a,
                        output_b,
                        n_tokens,
                    )?
                };

                #[cfg(feature = "dsv4-diagnostics")]
                encoder.boundary(PackedPrefillStageKind::AfterAttentionOutput)?;

                self.prefill.hyper.encode_post(
                    ctx,
                    &encoder,
                    &attention_output,
                    &residual_primary,
                    &residual_secondary,
                    n_tokens,
                )?;
                let ffn_input = self.prefill.hyper.encode_pre(
                    ctx,
                    &encoder,
                    &residual_secondary,
                    self.layer_tensor(layer, "hc_ffn_fn.weight")?,
                    self.layer_tensor(layer, "hc_ffn_scale.weight")?,
                    self.layer_tensor(layer, "hc_ffn_base.weight")?,
                    n_tokens,
                    rms_eps,
                    hc_eps,
                    #[cfg(feature = "dsv4-diagnostics")]
                    mhc_execution.as_deref_mut(),
                    #[cfg(feature = "dsv4-diagnostics")]
                    DeepSeekV4MhcSiteKind::Ffn,
                    #[cfg(feature = "dsv4-diagnostics")]
                    layer,
                )?;

                let hash_map = if layer < self.residency.config().hash_layer_count as usize {
                    Some(self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?)
                } else {
                    None
                };
                let moe_views = self.prefill.moe.encode_router(
                    ctx,
                    &encoder,
                    &ffn_input,
                    self.layer_tensor(layer, "ffn_norm.weight")?,
                    self.layer_tensor(layer, "ffn_gate_inp.weight")?,
                    token_view,
                    hash_map,
                    n_tokens,
                    rms_eps,
                    router_e8p32_strict,
                )?;
                if let Some(generation) = gpu_route_generation {
                    let source = if hash_map.is_some() {
                        PackedRouteSource::Hash
                    } else {
                        PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
                    };
                    if compact_gpu_route {
                        self.prefill.moe.encode_gpu_route_compact(
                            ctx,
                            &encoder,
                            &moe_views,
                            source,
                            token_view,
                            hash_map,
                            n_tokens,
                            self.residency.config().expert_weights_scale,
                            generation,
                        )?;
                    } else {
                        #[cfg(feature = "dsv4-diagnostics")]
                        self.prefill.moe.encode_gpu_route_schedule(
                            ctx,
                            &encoder,
                            &moe_views,
                            source,
                            token_view,
                            hash_map,
                            n_tokens,
                            self.residency.config().expert_weights_scale,
                            generation,
                        )?;
                        #[cfg(not(feature = "dsv4-diagnostics"))]
                        return invalid("packed diagnostic GPU route is unavailable");
                    }
                }
                Ok::<_, DeepSeekV4MetalError>((moe_views, captured_sparse))
            })();
            encoder.end();
            let (moe_views, captured_sparse) = router_result?;
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let _ = captured_sparse;
            let pre_expert_encode_seconds = pre_expert_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            #[cfg(feature = "dsv4-diagnostics")]
            let stage_profile_active = stage_recorder.is_some();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let stage_profile_active = false;
            let (
                pre_expert_wait_seconds,
                pre_expert_command_seconds,
                pre_expert_gpu_seconds,
                pre_expert_wait_residual_seconds,
            ) = if merge_gpu_route {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                let wait_started = trace_layers.then(std::time::Instant::now);
                command.commit();
                crate::metal::wait_completed(&command)?;
                let wait_seconds = wait_started
                    .as_ref()
                    .map_or(0.0, |started| started.elapsed().as_secs_f64());
                let command_seconds = pre_expert_started
                    .as_ref()
                    .map_or(0.0, |started| started.elapsed().as_secs_f64());
                let gpu_seconds = if trace_layers || stage_profile_active {
                    command.GPUEndTime() - command.GPUStartTime()
                } else {
                    0.0
                };
                if let Some(error) = command.error() {
                    return invalid(format!(
                        "packed layer {layer} router command failed: {error:?}"
                    ));
                }
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(collector) = mhc_command_collector.as_deref_mut() {
                    collector.record(layer, DeepSeekV4MhcCommandKind::PreExpert, &command)?;
                }
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(recorder) = stage_recorder.as_deref_mut() {
                    recorder.record_command_gpu_seconds(layer, gpu_seconds)?;
                }
                (
                    wait_seconds,
                    command_seconds,
                    gpu_seconds,
                    wait_seconds - gpu_seconds,
                )
            };
            let pre_expert_seconds = if merge_gpu_route {
                0.0
            } else {
                pre_expert_started
                    .as_ref()
                    .map_or(0.0, |started| started.elapsed().as_secs_f64())
            };
            let pre_expert_post_seconds = pre_expert_seconds - pre_expert_command_seconds;

            let shared_overlap_command = if overlap_shared_route {
                let shared_command = ctx.queue.commandBuffer().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "failed to allocate packed layer {layer} shared-expert command buffer"
                    ))
                })?;
                let shared_encoder = KernelEncoder::begin(&shared_command);
                let shared_result = self.prefill.moe.encode_shared_expert(
                    ctx,
                    &shared_encoder,
                    &moe_views.normalized_input,
                    self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                    self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                    self.layer_tensor(layer, "ffn_down_shexp.weight")?,
                    shared_matrix,
                    self.residency.config().swiglu_clamp_shared[layer],
                    n_tokens,
                );
                shared_encoder.end();
                shared_result?;
                shared_command.commit();
                let shared_command = CommittedPackedCommand {
                    command: shared_command,
                };
                static REPORTED: std::sync::Once = std::sync::Once::new();
                REPORTED.call_once(|| {
                    eprintln!(
                        "deepseek_v4: shared expert overlaps CPU route planning for qualified K160 chunks; rollback=QWEN_DSV4_PACKED_SHARED_ROUTE_OVERLAP=0"
                    );
                });
                Some(shared_command)
            } else {
                None
            };

            let route_started = trace_layers.then(std::time::Instant::now);
            let mut schedule = if merge_gpu_route {
                Vec::new()
            } else if compact_gpu_route {
                self.prefill.moe.capture_gpu_compact_schedule(
                    n_tokens,
                    gpu_route_generation.expect("compact route generation"),
                )?
            } else {
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(generation) = gpu_route_generation {
                    #[cfg(test)]
                    let cpu_route = {
                        let source = if layer < self.residency.config().hash_layer_count as usize {
                            PackedRouteSource::Hash
                        } else {
                            PackedRouteSource::Learned(
                                self.layer_tensor(layer, "exp_probs_b.bias")?,
                            )
                        };
                        self.prefill.moe.audit_gpu_route_against_cpu(
                            &moe_views,
                            source,
                            n_tokens,
                            layer,
                            start_position,
                            generation,
                        )?
                    };
                    let schedule = self
                        .prefill
                        .moe
                        .capture_gpu_route_schedule(n_tokens, generation)?;
                    #[cfg(test)]
                    if route_policy == PackedRoutePolicy::GpuExperimentalCpuWeights {
                        let (cpu_ids, cpu_weights) = cpu_route;
                        let route_count = n_tokens * MOE_TOP_K;
                        let mut gpu_ids = host_read_i32(
                            &self.prefill.moe.expert_ids,
                            "packed GPU hybrid route IDs",
                        )?;
                        gpu_ids.truncate(route_count);
                        if gpu_ids != cpu_ids {
                            return invalid(format!(
                                "packed GPU hybrid layer {layer} route IDs differ from Rust"
                            ));
                        }
                        let weights = f32_prefix(
                            &self.prefill.moe.weights,
                            vec![route_count as u64],
                            "packed GPU hybrid route weights",
                        )?;
                        host_write_f32(&weights, &cpu_weights, "packed GPU hybrid route weights")?;
                    }
                    schedule
                } else {
                    let source = if layer < self.residency.config().hash_layer_count as usize {
                        PackedRouteSource::Hash
                    } else {
                        PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
                    };
                    self.prefill.moe.route(
                        &moe_views,
                        source,
                        n_tokens,
                        self.residency.config().expert_weights_scale,
                    )?
                }
                #[cfg(not(feature = "dsv4-diagnostics"))]
                {
                    let source = if layer < self.residency.config().hash_layer_count as usize {
                        PackedRouteSource::Hash
                    } else {
                        PackedRouteSource::Learned(self.layer_tensor(layer, "exp_probs_b.bias")?)
                    };
                    self.prefill.moe.route(
                        &moe_views,
                        source,
                        n_tokens,
                        self.residency.config().expert_weights_scale,
                    )?
                }
            };
            let route_seconds = route_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());

            let post_route_started = trace_layers.then(std::time::Instant::now);
            let routed_gate = self.layer_tensor(layer, "ffn_gate_exps.weight")?;
            let routed_up = self.layer_tensor(layer, "ffn_up_exps.weight")?;
            let routed_down = self.layer_tensor(layer, "ffn_down_exps.weight")?;
            let shared_gate = self.layer_tensor(layer, "ffn_gate_shexp.weight")?;
            let shared_up = self.layer_tensor(layer, "ffn_up_shexp.weight")?;
            let shared_down = self.layer_tensor(layer, "ffn_down_shexp.weight")?;
            let grouped_q3q4_qualified = packed_grouped_q3q4_enabled()
                && packed_grouped_q3q4_scope_qualified(
                    &ctx.device.name().to_string(),
                    self.residency.report().tensor_count,
                    self.residency.report().source_bytes,
                    self.prefill.moe.expert_count,
                    n_tokens,
                    routed_gate.dtype,
                    routed_up.dtype,
                    routed_down.dtype,
                )
                && packed_grouped_q3q4_candidate_supported(ctx);
            #[cfg(feature = "dsv4-diagnostics")]
            if let Some(recorder) = post_route_stage_recorder.as_deref_mut() {
                let route_count = checked_mul(n_tokens, MOE_TOP_K, "packed metadata routes")?;
                let expert_ids = host_read_i32(
                    &i32_prefix(
                        &self.prefill.moe.expert_ids,
                        vec![route_count as u64],
                        "packed metadata expert IDs",
                    )?,
                    "packed metadata expert IDs",
                )?;
                let route_weight_bits = host_read_f32(
                    &f32_prefix(
                        &self.prefill.moe.weights,
                        vec![route_count as u64],
                        "packed metadata route weights",
                    )?,
                    "packed metadata route weights",
                )?
                .into_iter()
                .map(f32::to_bits)
                .collect();
                let bucket_rows = host_read_i32(
                    &i32_prefix(
                        &self.prefill.moe.bucket_rows,
                        vec![route_count as u64],
                        "packed metadata bucket rows",
                    )?,
                    "packed metadata bucket rows",
                )?;
                let bucket_slots = host_read_i32(
                    &i32_prefix(
                        &self.prefill.moe.bucket_slots,
                        vec![route_count as u64],
                        "packed metadata bucket slots",
                    )?,
                    "packed metadata bucket slots",
                )?;
                let grouped_iq2 = expert_policy.uses_iq2_target()
                    && routed_gate.dtype == GgmlType::IQ2_XS
                    && routed_up.dtype == GgmlType::IQ2_XS
                    && routed_down.dtype == GgmlType::IQ3_XXS
                    && packed_grouped_expert_kernels_supported(ctx);
                let grouped_iq3 = expert_policy.uses_iq3_target()
                    && routed_gate.dtype == GgmlType::IQ3_XXS
                    && routed_up.dtype == GgmlType::IQ3_XXS
                    && routed_down.dtype == GgmlType::IQ3_XXS
                    && packed_grouped_iq3_candidate_supported(ctx);
                recorder.record_layer(PackedPostRouteLayerMetadata {
                    layer,
                    gate_dtype: routed_gate.dtype,
                    up_dtype: routed_up.dtype,
                    down_dtype: routed_down.dtype,
                    bucket_count: schedule.len(),
                    expert_counts: packed_post_route_expert_counts(
                        n_tokens,
                        &schedule,
                        self.prefill.moe.expert_count,
                    )?,
                    route_expert_ids: packed_post_route_expert_ids(
                        n_tokens,
                        self.prefill.moe.expert_count,
                        &expert_ids,
                        &bucket_rows,
                        &bucket_slots,
                        &schedule,
                    )?,
                    route_weight_bits,
                    grouped_q3q4: grouped_q3q4_qualified,
                    grouped_iq2,
                    grouped_iq3,
                    bm16: grouped_iq2 && expert_policy.uses_iq2_mma16(n_tokens),
                })?;
            }

            let separate_expert_command = if merge_gpu_route {
                None
            } else {
                Some(ctx.queue.commandBuffer().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "failed to allocate packed layer {layer} expert command buffer"
                    ))
                })?)
            };
            let expert_command = separate_expert_command.as_ref().unwrap_or(&command);
            #[cfg(feature = "dsv4-diagnostics")]
            let mut encoder = PackedPostRouteLayerEncoder::begin(
                expert_command,
                layer,
                grouped_q3q4_qualified
                    || (expert_policy.uses_iq2_mma16(n_tokens)
                        && routed_gate.dtype == GgmlType::IQ2_XS
                        && routed_up.dtype == GgmlType::IQ2_XS
                        && routed_down.dtype == GgmlType::IQ3_XXS
                        && packed_grouped_expert_kernels_supported(ctx)),
                post_route_stage_recorder.as_deref_mut(),
            )?;
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let mut encoder = PackedPostRouteLayerEncoder::begin(expert_command)?;
            let expert_result = (|| {
                let moe_output = self.prefill.moe.encode_experts(
                    ctx,
                    &mut encoder,
                    &moe_views.normalized_input,
                    &schedule,
                    compact_gpu_route,
                    routed_gate,
                    routed_up,
                    routed_down,
                    shared_gate,
                    shared_up,
                    shared_down,
                    expert_policy,
                    grouped_q3q4_qualified,
                    mxfp4_matrix,
                    shared_matrix,
                    shared_overlap_command.is_some(),
                    self.residency.config().swiglu_clamp_experts[layer],
                    self.residency.config().swiglu_clamp_shared[layer],
                    n_tokens,
                )?;
                self.prefill.hyper.encode_post(
                    ctx,
                    &encoder,
                    &moe_output,
                    &residual_secondary,
                    &residual_primary,
                    n_tokens,
                )?;
                if layer + 1 == DEEPSEEK_V4_LAYER_COUNT && emit_logits {
                    let final_residual = f32_row(
                        &residual_primary,
                        n_tokens - 1,
                        residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?,
                        vec![
                            DEEPSEEK_V4_HIDDEN_SIZE as u64,
                            DEEPSEEK_V4_CONNECTION_COUNT as u64,
                        ],
                        "packed final residual",
                    )?;
                    self.hyper_connection.encode_head(
                        ctx,
                        &encoder,
                        &final_residual,
                        self.residency.require_tensor("output_hc_fn.weight")?,
                        self.residency.require_tensor("output_hc_scale.weight")?,
                        self.residency.require_tensor("output_hc_base.weight")?,
                        &self.final_hidden,
                        rms_eps,
                        hc_eps,
                    )?;
                    encode_rms_norm_mul_f32(
                        ctx,
                        &encoder,
                        &self.final_hidden,
                        self.residency.require_tensor("output_norm.weight")?,
                        &self.final_normalized_hidden,
                        rms_eps,
                    )?;
                    encode_projection(
                        ctx,
                        &encoder,
                        self.residency.require_tensor("output.weight")?,
                        &self.final_normalized_hidden,
                        &self.logits,
                        DEEPSEEK_V4_HIDDEN_SIZE,
                        DEEPSEEK_V4_VOCAB_SIZE,
                        "packed output logits",
                    )?;
                }
                Ok::<(), DeepSeekV4MetalError>(())
            })();
            encoder.end();
            expert_result?;
            let post_route_encode_seconds = post_route_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            let post_route_wait_started = trace_layers.then(std::time::Instant::now);
            expert_command.commit();
            crate::metal::wait_unchecked(&expert_command);
            let post_route_wait_seconds = post_route_wait_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            if let Some(shared_command) = &shared_overlap_command
                && let Err(error) = crate::metal::wait_completed(&shared_command.command)
            {
                return invalid(format!(
                    "packed layer {layer} shared-expert command failed: {error}"
                ));
            }
            if let Err(error) = crate::metal::command_buffer_completed(&expert_command) {
                return invalid(format!("packed layer {layer} command failed: {error}"));
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if let Some(collector) = mhc_command_collector.as_deref_mut() {
                if let Some(shared_command) = &shared_overlap_command {
                    collector.record(
                        layer,
                        DeepSeekV4MhcCommandKind::SharedOverlap,
                        &shared_command.command,
                    )?;
                }
                collector.record(
                    layer,
                    if merge_gpu_route {
                        DeepSeekV4MhcCommandKind::MergedPreExpert
                    } else {
                        DeepSeekV4MhcCommandKind::Expert
                    },
                    expert_command,
                )?;
            }
            if merge_gpu_route {
                schedule = self.prefill.moe.capture_gpu_compact_schedule(
                    n_tokens,
                    gpu_route_generation.expect("compact route generation"),
                )?;
            }
            if let Some(query_offset) = sparse_query_offset
                && {
                    #[cfg(feature = "dsv4-diagnostics")]
                    {
                        fp4_score_plan.runs_f16()
                    }
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    {
                        true
                    }
                }
            {
                self.prefill
                    .attention
                    .sparse_csa
                    .validate_completed(n_tokens - query_offset)?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if self.fp4_shadow_diagnostics.is_capturing()
                && attention_kind == AttentionKind::CompressedSparse
            {
                let sparse = captured_sparse.as_ref().ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "FP4 packed shadow layer {layer} did not retain sparse views"
                    ))
                })?;
                let rows = self
                    .compressor_frontiers
                    .csa_rows(layer, last_position)?
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "FP4 packed shadow layer {layer} has no published rows"
                        ))
                    })?;
                let report = self.fp4_shadow.capture_layer(
                    layer,
                    last_position,
                    rows,
                    &sparse.scores,
                    &sparse.selected_mask,
                    &sparse.cache_order_ids,
                    &sparse.selected_counts,
                    &sparse.status,
                )?;
                self.fp4_shadow_diagnostics.capture_layer(report)?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if fp4_score_plan.consumes_fp4() && sparse_query_offset.is_some() {
                self.fp4_shadow.validate_completed()?;
                self.fp4_shadow.record_counterfactual_selection(
                    &mut self.fp4_counterfactual_trace,
                    DeepSeekV4Fp4ShadowExecution::Packed,
                    DeepSeekV4Fp4SelectionSource::Fp4,
                    last_position,
                    layer,
                )?;
            }
            let measure_post_route_gpu = trace_layers;
            #[cfg(feature = "dsv4-diagnostics")]
            let measure_post_route_gpu =
                measure_post_route_gpu || post_route_stage_recorder.is_some();
            let measured_post_route_gpu_seconds = measure_post_route_gpu
                .then(|| expert_command.GPUEndTime() - expert_command.GPUStartTime());
            let post_route_gpu_seconds = if trace_layers {
                measured_post_route_gpu_seconds.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "packed post-route trace omitted command GPU duration".into(),
                    )
                })?
            } else {
                0.0
            };
            #[cfg(feature = "dsv4-diagnostics")]
            if let Some(recorder) = post_route_stage_recorder.as_deref_mut() {
                recorder.record_command_gpu_seconds(
                    layer,
                    measured_post_route_gpu_seconds.ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "packed post-route profile omitted command GPU duration".into(),
                        )
                    })?,
                )?;
            }
            let post_route_wait_residual_seconds = post_route_wait_seconds - post_route_gpu_seconds;
            let post_route_seconds = post_route_started
                .as_ref()
                .map_or(0.0, |started| started.elapsed().as_secs_f64());
            if trace_layers {
                layer_traces.push(PackedLayerTrace {
                    layer,
                    pre_expert_seconds,
                    pre_expert_gpu_seconds,
                    pre_expert_encode_seconds,
                    pre_expert_wait_seconds,
                    pre_expert_wait_residual_seconds,
                    pre_expert_post_seconds,
                    route_seconds,
                    post_route_seconds,
                    post_route_gpu_seconds,
                    post_route_encode_seconds,
                    post_route_wait_seconds,
                    post_route_wait_residual_seconds,
                    bucket_count: schedule.len(),
                });
            }
            layer_completed(layer);
        }
        if trace_layers {
            for trace in &layer_traces {
                eprintln!(
                    "deepseek_v4 packed layer={} pre_expert={:.4}s pre_expert_gpu={:.4}s pre_expert_encode={:.4}s pre_expert_wait={:.4}s pre_expert_wait_residual={:.4}s pre_expert_post={:.4}s route={:.4}s post_route={:.4}s post_route_gpu={:.4}s post_route_encode={:.4}s post_route_wait={:.4}s post_route_wait_residual={:.4}s buckets={}",
                    trace.layer,
                    trace.pre_expert_seconds,
                    trace.pre_expert_gpu_seconds,
                    trace.pre_expert_encode_seconds,
                    trace.pre_expert_wait_seconds,
                    trace.pre_expert_wait_residual_seconds,
                    trace.pre_expert_post_seconds,
                    trace.route_seconds,
                    trace.post_route_seconds,
                    trace.post_route_gpu_seconds,
                    trace.post_route_encode_seconds,
                    trace.post_route_wait_seconds,
                    trace.post_route_wait_residual_seconds,
                    trace.bucket_count,
                );
            }
            let pre_expert_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_seconds)
                .sum::<f64>();
            let pre_expert_gpu_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_gpu_seconds)
                .sum::<f64>();
            let pre_expert_encode_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_encode_seconds)
                .sum::<f64>();
            let pre_expert_wait_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_wait_seconds)
                .sum::<f64>();
            let pre_expert_wait_residual_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_wait_residual_seconds)
                .sum::<f64>();
            let pre_expert_post_total = layer_traces
                .iter()
                .map(|trace| trace.pre_expert_post_seconds)
                .sum::<f64>();
            let route_total = layer_traces
                .iter()
                .map(|trace| trace.route_seconds)
                .sum::<f64>();
            let post_route_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_seconds)
                .sum::<f64>();
            let post_route_gpu_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_gpu_seconds)
                .sum::<f64>();
            let post_route_encode_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_encode_seconds)
                .sum::<f64>();
            let post_route_wait_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_wait_seconds)
                .sum::<f64>();
            let post_route_wait_residual_total = layer_traces
                .iter()
                .map(|trace| trace.post_route_wait_residual_seconds)
                .sum::<f64>();
            let pre_expert_negative_residuals = layer_traces
                .iter()
                .filter(|trace| trace.pre_expert_wait_residual_seconds < 0.0)
                .count();
            let post_route_negative_residuals = layer_traces
                .iter()
                .filter(|trace| trace.post_route_wait_residual_seconds < 0.0)
                .count();
            eprintln!(
                "deepseek_v4 packed totals route_policy={route_policy:?} expert_policy={expert_policy:?} pre_expert={pre_expert_total:.3}s pre_expert_gpu={pre_expert_gpu_total:.3}s pre_expert_encode={pre_expert_encode_total:.3}s pre_expert_wait={pre_expert_wait_total:.3}s pre_expert_wait_residual={pre_expert_wait_residual_total:.3}s pre_expert_post={pre_expert_post_total:.3}s pre_expert_negative_residuals={pre_expert_negative_residuals} route={route_total:.3}s post_route={post_route_total:.3}s post_route_gpu={post_route_gpu_total:.3}s post_route_encode={post_route_encode_total:.3}s post_route_wait={post_route_wait_total:.3}s post_route_wait_residual={post_route_wait_residual_total:.3}s post_route_negative_residuals={post_route_negative_residuals}"
            );
        }
        #[cfg(feature = "dsv4-diagnostics")]
        self.fp4_shadow_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            fp4_score_dispatch_ledger.validate_completed()?;
            self.fp4_score_dispatch_ledger = Some(fp4_score_dispatch_ledger);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "prefill/k160_n128_floor.rs"]
mod k160_n128_floor;

#[cfg(test)]
mod tests;

//! Resumable GLM request metadata. Causal state is lent to the backend for
//! one bounded owner quantum, then returned to its request by the scheduler.
use super::*;

pub(super) enum Begun<'w> {
    Run(Box<Run<'w>>),
    Complete(GenerationOutcome),
}

pub(super) struct Run<'w> {
    pub session: Option<Glm5NextSession<'w>>,
    pub tokens: Vec<u32>,
    pub history: Vec<u32>,
    pub sampler: Sampler,
    pub maximum: usize,
    pub vocab_size: u32,
    pub required: usize,
    pub reserve: u64,
    pub chunk_enabled: bool,
    pub conversation: Option<super::super::checkpoint_ladder::Conversation>,
    pub lineage: PackedLineage,
    pub canonical: bool,
    pub schedule: Option<Schedule>,
    pub cuts: Vec<usize>,
    pub selected_rows: usize,
    pub namespace: Option<CacheNamespace>,
    pub chunk_namespace: Option<CacheNamespace>,
    pub chunk_target: Option<usize>,
    pub matched: usize,
    pub restore_origin: &'static str,
    pub restore_ms: f64,
    pub reuse_source: &'static str,
    pub boundary_ms: f64,
    pub prefill_t0: Instant,
    pub capture_ms: f64,
    pub captures: Vec<String>,
    pub chunk_capture: Option<chunk_checkpoint::Capture>,
    pub protect: Option<EntryId>,
    pub position: usize,
    pub logits: Vec<f32>,
    pub ends: Vec<usize>,
    pub segment: usize,
    pub prefill_ms: f64,
    pub chunk_capture_ms: f64,
    pub ready: bool,
}

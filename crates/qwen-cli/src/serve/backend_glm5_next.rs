//! GLM-5.3-Flash serve: borrowed resident weights and one live session kept
//! across requests. KDA recurrent state cannot rewind, so a request reuses
//! the session only when its prompt strictly extends exactly what the
//! session consumed (a replayed conversation does: the model ends a turn by
//! sampling `<|user|>`, the next turn's opener); anything else drops the
//! session and prefills a fresh one. No snapshots.
use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::render_glm5_next::{self as render, FAMILY};
use anyhow::{Context, Result, bail, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::glm5_next::Glm5NextPreparedArtifact;
use qwen_llm::glm5_next_chat::{CHAT_STOPS, VerifiedChatProfile};
use qwen_llm::glm5_next_metal::{
    DEFAULT_PREFILL_ROWS, Glm5NextMetalError, Glm5NextSession, Glm5NextWeights, PackedLineage,
    prefetch_retained_with_cancel, preflight_session,
};
use qwen_llm::metal::MetalContext;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::sampling::Sampler;
use std::io;
use std::time::Instant;

/// Default-on rollback lever for live-session reuse.
const PREFIX_REUSE_ENV: &str = "QWEN_GLM_PREFIX_REUSE";

/// CPU startup facts: the admitted artifact, its verified chat profile and
/// the session geometry fixed for the server's lifetime.
pub(super) struct Prepared<'g> {
    artifact: Glm5NextPreparedArtifact<'g>,
    pub(super) capacity: usize,
    default_max: usize,
    prefill_rows: usize,
    profile: VerifiedChatProfile,
    /// Longest decoded token, for the tool block's byte budget.
    max_piece_bytes: usize,
}

impl<'g> Prepared<'g> {
    pub(super) fn new(
        gguf: &'g GgufFile,
        invocation: &crate::cli::ServeInvocation,
    ) -> Result<Self> {
        ensure!(
            invocation.drafter.is_none(),
            "{FAMILY} serve does not support a drafter"
        );
        let artifact = Glm5NextPreparedArtifact::inspect(gguf)
            .with_context(|| format!("admit {FAMILY} artifact"))?;
        let (capacity, default_max) = super::fixed_session_limits(
            ModelFamily::Glm5Next,
            artifact.config().context_length as usize,
            invocation.max_context_tokens,
            invocation.max_tokens,
        )?;
        ensure!(
            capacity > 0 && default_max > 0,
            "{FAMILY} serve needs positive --max-context-tokens and --max-tokens"
        );
        // There is no raw serve lane: without a verified profile nothing renders.
        let profile = artifact.chat_profile().map_err(|error| {
            anyhow::anyhow!("{FAMILY} serve renders verified text chat: {error}")
        })?;
        let prefill_rows = if artifact.packed_prefill() {
            DEFAULT_PREFILL_ROWS.min(capacity)
        } else {
            0
        };
        let max_piece_bytes = artifact.tokenizer().max_decoded_piece_bytes();
        crate::shutdown::checkpoint()?;
        Ok(Self {
            artifact,
            capacity,
            default_max,
            prefill_rows,
            profile,
            max_piece_bytes,
        })
    }

    pub(super) fn describe(&self) -> String {
        format!(
            "family=glm5_next input=verified_chat_and_tools renderer={} capacity={} default_max_tokens={} prefill_rows={} snapshot_cache_bytes=0",
            self.profile.renderer, self.capacity, self.default_max, self.prefill_rows
        )
    }
}

/// Device admission, prefetch and load, before the listener accepts: a
/// session that cannot fit is refused with the capacity that would.
pub(super) fn load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    prepared: &Prepared<'_>,
) -> Result<Glm5NextWeights> {
    match preflight_session(
        ctx,
        gguf,
        prepared.artifact.model(),
        prepared.capacity,
        prepared.prefill_rows,
    ) {
        Ok(_) => {}
        Err(
            error @ Glm5NextMetalError::MemoryAdmission {
                fitting_capacity, ..
            },
        ) => match fitting_capacity {
            Some(n) => bail!("{error}; pass --max-context-tokens {n} or less"),
            None => bail!("{error}; free device memory or use a smaller artifact"),
        },
        Err(error) => return Err(error).context("admit GLM-5.3 serve session"),
    }
    let prefetch =
        prefetch_retained_with_cancel(ctx, gguf, 0.98, &|| crate::shutdown::checkpoint().is_err())
            .context("prefetch GLM-5.3 retained windows")?;
    tracing::info!(
        target: "qwen_diag",
        "serve prefetch: family=glm5_next windows={} cold_windows={} bytes_read={} wall_ms={:.1}",
        prefetch.windows,
        prefetch.cold_windows,
        prefetch.bytes_read,
        prefetch.wall.as_secs_f64() * 1e3
    );
    crate::shutdown::checkpoint()?;
    Glm5NextWeights::load(ctx, gguf).context("load GLM-5.3 weights")
}

pub(super) struct Glm5NextBackend<'w, 'g> {
    ctx: &'w MetalContext,
    weights: &'w Glm5NextWeights,
    prepared: Prepared<'g>,
    model_id: String,
    session: Option<Glm5NextSession<'w>>,
    /// Exactly the tokens `session` has consumed, or empty when unknown.
    history: Vec<u32>,
    prefix_reuse: bool,
    idle_residency: super::idle_residency::IdleResidency,
    /// Packed-prefill arithmetic for new sessions (tests compare warm and
    /// cold paths under `Exact`, which matches serial decode bitwise).
    lineage: PackedLineage,
}

impl<'w, 'g> Glm5NextBackend<'w, 'g> {
    pub(super) fn new(
        ctx: &'w MetalContext,
        weights: &'w Glm5NextWeights,
        prepared: Prepared<'g>,
        model_id: String,
        idle_residency: std::time::Duration,
    ) -> Self {
        Self {
            ctx,
            weights,
            prepared,
            model_id,
            session: None,
            history: Vec::new(),
            prefix_reuse: qwen_llm::env_flag::read_default_on(PREFIX_REUSE_ENV),
            idle_residency: super::idle_residency::IdleResidency::new("glm5_next", idle_residency),
            lineage: PackedLineage::default(),
        }
    }

    fn fresh_session(&self, cpu_reserve: u64) -> Result<Glm5NextSession<'w>, ServeError> {
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            self.ctx,
            self.weights,
            self.prepared.capacity,
            self.prepared.prefill_rows,
            cpu_reserve,
        )
        .map_err(|error| ServeError::server_error(format!("{FAMILY} session: {error}")))?;
        session.set_packed_lineage(self.lineage);
        Ok(session)
    }

    /// One short prefill on a throwaway session before the listener accepts,
    /// so the first request is not charged the weights' first GPU use.
    pub(super) fn warm_up(&mut self) -> Result<f64> {
        let started = Instant::now();
        let tokens = self
            .prepared
            .artifact
            .tokenizer()
            .encode("[gMASK]<sop>", false)?
            .into_iter()
            .map(|id| u32::try_from(id).context("warm-up token id"))
            .collect::<Result<Vec<_>>>()?;
        let mut session = self
            .fresh_session(0)
            .map_err(|e| anyhow::anyhow!(e.message))?;
        session
            .prefill_packed_with_checkpoint(self.ctx, &tokens, &mut || {
                crate::shutdown::checkpoint().map_err(|e| e.to_string())
            })
            .context("GLM-5.3 serve warm-up")?;
        drop(session);
        // The server is about to accept: open the idle-residency window.
        self.idle_residency.note_activity();
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }
}

impl GenerationBackend for Glm5NextBackend<'_, '_> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn request_profile(&self) -> super::request_profile::RequestProfile {
        super::request_profile::RequestProfile::Glm5Next {
            default_max_tokens: self.prepared.default_max,
            capacity: self.prepared.capacity,
            max_piece_bytes: self.prepared.max_piece_bytes,
        }
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.idle_residency.before_request();
        // Everything a client can get wrong is refused before the live
        // session is touched, so a 400 keeps the cached conversation.
        let maximum = request
            .max_output_tokens
            .unwrap_or(self.prepared.default_max);
        let mut sampler = Sampler::new(render::sampling(request))
            .map_err(|e| ServeError::invalid_request(None, format!("{FAMILY} sampler: {e}")))?;
        let vocab_size = self.prepared.artifact.config().vocab_size;
        let tokenizer = self.prepared.artifact.tokenizer();
        // The renderer writes [gMASK]<sop>; the glm4 tokenizer adds nothing.
        let tokens = decode_loop::encode_checked(tokenizer, prompt, false, vocab_size, FAMILY)?;
        let required =
            decode_loop::required_forwards(FAMILY, tokens.len(), maximum, self.prepared.capacity)?;
        // A tool block's memory is admitted as it grows (the tools partition),
        // not reserved here at its rarely approached worst case.
        let reserve = sink.transport_reserve_bytes();
        super::transport_memory::admit_resident_transport(
            reserve,
            MetalContext::process_limit_bytes_remaining(),
        )?;
        sink.tick().map_err(BackendFailure::Aborted)?;

        let prefill_t0 = Instant::now();
        // Taken before the session is used and republished only for a state
        // the session is known to hold, so a failure leaves no stale history.
        let history = std::mem::take(&mut self.history);
        let committed = self.session.as_ref().map_or(0, Glm5NextSession::position);
        let reused = decode_loop::extending_prefix(&history, &tokens, committed, self.prefix_reuse);
        if reused == 0 {
            // Release the old session's memory before allocating its successor.
            self.session = None;
            // A new session's buffers and this request's transport allowance
            // are admitted as one requirement.
            self.session = Some(self.fresh_session(reserve)?);
        }
        let ctx = self.ctx;
        let session = self.session.as_mut().expect("session present");

        let mut checkpoint_abort: Option<io::Error> = None;
        let start = session.position();
        let logits = session.prefill_packed_with_checkpoint(ctx, &tokens[reused..], &mut || {
            sink.tick().map_err(|error| {
                checkpoint_abort = Some(error);
                "transport aborted during GLM-5.3 prefill".into()
            })
        });
        // Idle residency follows weight use: a prefill that ran (or committed
        // chunks before failing) used them; admission refusals and session
        // allocation above did not.
        if logits.is_ok() || session.position() > start {
            self.idle_residency.note_execution();
        }
        let logits = match logits {
            Ok(logits) => logits,
            Err(error) => {
                if let (Glm5NextMetalError::Cancelled(_), Some(abort)) = (&error, checkpoint_abort)
                {
                    // Cancelled at a chunk boundary: the committed prefix is
                    // consistent, so a retry of this prompt resumes from it.
                    let position = session.position();
                    self.history = tokens[..position].to_vec();
                    return Err(BackendFailure::Aborted(abort));
                }
                self.session = None;
                return Err(ServeError::server_error(format!("{FAMILY} prefill: {error}")).into());
            }
        };
        if let Err(abort) = sink.tick() {
            // The prompt is consumed but no retry can extend it (an equal
            // prompt has no fresh row): release the session now.
            self.session = None;
            return Err(BackendFailure::Aborted(abort));
        }
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
        let generation = decode_loop::decode_serial(
            decode_loop::DecodeRequest {
                family: FAMILY,
                logits,
                max_tokens: maximum,
                stop_tokens: &CHAT_STOPS,
                vocab_size,
            },
            &mut sampler,
            tokenizer,
            sink,
            |token| Ok(session.forward(ctx, token)?),
        );
        let generation = match generation {
            Ok(generation) => generation,
            Err(failure) => {
                // A decode abort or failure clears the cache: a retry's
                // prompt never extends a partially generated history.
                self.session = None;
                return Err(failure);
            }
        };
        if session.position() != tokens.len() + generation.transitions {
            self.session = None;
            return Err(ServeError::server_error(format!(
                "{FAMILY} consumed-prefix accounting mismatch"
            ))
            .into());
        }
        let mut history = history;
        history.clear();
        history.extend_from_slice(&tokens);
        history.extend(
            generation.tokens[..generation.transitions]
                .iter()
                .map(|&token| token as u32),
        );
        self.history = history;
        tracing::info!(
            target: "qwen_diag",
            "serve phases: family=glm5_next prefill_ms={prefill_ms:.1} reused_tokens={reused} prefill_tokens={} decode_ms={:.1} required_forwards={required} capacity={} transitions={}",
            tokens.len() - reused,
            generation.wall_ms,
            self.prepared.capacity,
            generation.transitions,
        );
        Ok(super::outcome::finish_generation(
            tokens.len(),
            &generation,
            reused,
            0.0,
        ))
    }

    fn idle(&mut self) {
        let weights = self.weights;
        self.idle_residency
            .on_idle(self.ctx, || weights.retained_buffers());
    }

    fn request_finished(&mut self) {
        self.idle_residency.request_finished();
    }

    fn shutdown(&mut self) {
        self.idle_residency.shutdown();
        self.session = None;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests;

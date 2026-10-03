//! Durable tier for the Qwen engine backend.
//!
//! Write policy: dense Qwen snapshots are GBs, so nothing is written per
//! request. Three paths write, all deduplicated against jobs in flight and
//! entries already on disk:
//!
//! - **Idle publication:** once no request has arrived for
//!   `plan.idle_publish_after` (default 30 s), the latest request's
//!   continuation boundary (its transcript boundary, else its completed
//!   boundary) is written, at most once per idle period and only when the
//!   writer is empty; if pressure evicted it, the best-ranked entry not yet
//!   on disk is written instead. Agent loops send requests seconds apart, so
//!   this usually coalesces to about one write per human turn, and a restart
//!   or crash after it keeps the session.
//! - **Spill:** an entry leaving RAM through budget eviction or idle/age
//!   expiry (pressure-relief evictions are not spilled: that memory is needed
//!   now).
//! - **Graceful shutdown:** within `plan.shutdown_budget` (default 30 s), the
//!   continuation target first, then the other ranked entries, one at a time,
//!   each reported as published, failed, timed out, no room or already on
//!   disk. Ranked writes never evict a record (the store refuses them under
//!   its writer lock instead), so they cannot displace the target.
//!
//! Every queued write reserves the memory its staged decode check will
//! allocate, from enqueue until acknowledgement; request, capture, promotion
//! and idle admission count it.
//!
//! A write is acknowledged back from the worker before its cache entry is
//! marked durable (never optimistically), and marking uses payload pointer
//! identity, so an equivalent replacement is never marked. Entries promoted
//! from disk are marked durable on insertion. A durable-marked target whose
//! record has since left the disk is rewritten at shutdown.
//!
//! Read policy: before the RAM lookup, a disk record strictly longer than
//! the best RAM match is promoted into RAM (subject to the strict cache
//! budget and the capture memory admission), so one restore path serves
//! both tiers.

use super::super::durable::{
    DurablePlan, DurableWorker, Resolved, queue_cap_bytes, resolve_content_identity,
};
use super::EngineBackend;
use anyhow::{Context as _, Result, ensure};
use qwen_llm::checkpoint_identity::same_identity_sources;
use qwen_llm::checkpoint_store::DurableCheckpointStore;
use qwen_llm::gguf::GgufFile;
use qwen_llm::runtime::{CheckpointTicket, DetachedCheckpointPublisher, PreparedCheckpoint};
use qwen_llm::snapshot_policy::EntryId;
use std::cell::RefCell;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

/// One queued write; the ticket names its payload for acknowledgement.
pub(super) struct QwenJob {
    prepared: PreparedCheckpoint,
    ticket: CheckpointTicket,
    /// Whether publication may evict other records to make room (false for
    /// lower-ranked shutdown writes, which must not displace the target).
    /// The store enforces it under its writer lock.
    may_evict: bool,
}

/// How the worker's write of one job ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WriteAck {
    Published,
    /// A write that may not evict found no room (store budget or the
    /// volume's free-space reserve); nothing was removed.
    NoRoom,
    Failed,
}

/// Idle-publication state for the latest request.
struct IdlePublish {
    /// The latest request's continuation boundary.
    target: Option<EntryId>,
    /// When the latest request finished (or the last failed idle write).
    since: Instant,
    /// This idle period has queued its target (or has nothing to write).
    done: bool,
    /// The queued idle write, to recognize its acknowledgement.
    queued: Option<CheckpointTicket>,
    /// Failed writes of the current target.
    failures: u32,
}

/// Failed writes of one target before idle publication gives up on it.
const IDLE_PUBLISH_MAX_FAILURES: u32 = 2;

/// How [`EngineBackend::enqueue_durable`] disposed of a checkpoint.
enum Enqueue {
    Queued(CheckpointTicket),
    OnDisk(CheckpointTicket),
    InFlight,
    Skipped,
    Dropped,
}

pub(super) struct QwenDurable {
    plan: DurablePlan,
    store: DurableCheckpointStore,
    worker: DurableWorker<QwenJob>,
    /// Existence checks (the worker owns its own publisher).
    publisher: DetachedCheckpointPublisher,
    /// `(ticket, outcome)` from the worker after each write.
    acks: mpsc::Receiver<(CheckpointTicket, WriteAck)>,
    /// Jobs queued or being written, by payload (for acknowledgement) and by
    /// record key (an evicted and recaptured boundary is a new payload with
    /// the same record), each with the memory its write reserves (see
    /// [`write_copy_bytes`]) from enqueue until acknowledgement. Interior
    /// mutability: the request path drains spills through `&self`.
    pending: RefCell<Vec<(CheckpointTicket, String, u64)>>,
    idle: RefCell<IdlePublish>,
}

/// How one shutdown write ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlushOutcome {
    Published,
    OnDisk,
    Failed,
    TimedOut,
    /// A ranked write that would have evicted a record (possibly the target).
    NoRoom,
}

/// Memory a write of `prepared` allocates beyond the payload it holds: a
/// second full copy, decoded by the staged integrity check or by validating
/// a record already at its key (in every integrity mode). Reserved from
/// enqueue (not from when the write starts, so no admission in between sees
/// it free) until acknowledgement; while that copy is resident, the memory
/// signals count it too, which errs toward refusing.
fn write_copy_bytes(prepared: &PreparedCheckpoint) -> u64 {
    prepared.snapshot_bytes()
}

impl FlushOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::OnDisk => "already_on_disk",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::NoRoom => "no_room",
        }
    }

    fn persisted(self) -> bool {
        matches!(self, Self::Published | Self::OnDisk)
    }
}

/// `disk` when the restored RAM hit is exactly the prefix just promoted.
pub(super) fn restore_source(ram_matched: Option<usize>, promoted: Option<usize>) -> &'static str {
    match (ram_matched, promoted) {
        (Some(matched), Some(promoted)) if matched == promoted => "disk",
        (Some(_), _) => "ram",
        (None, _) => "none",
    }
}

impl EngineBackend {
    /// Enable the durable tier. The strong model identity resolves on the
    /// worker thread over a second open of the loaded files; the read path
    /// stays inactive until it does.
    pub(crate) fn attach_durable(
        &mut self,
        plan: DurablePlan,
        model_path: &Path,
        ram_budget_bytes: u64,
    ) -> Result<()> {
        let gguf = GgufFile::open(model_path)
            .with_context(|| format!("reopen {} for identity", model_path.display()))?;
        ensure!(
            same_identity_sources(&gguf, self.loaded.gguf()),
            "model files changed since load; durable identity would not name the resident weights"
        );
        let store = DurableCheckpointStore::new(&plan.root, plan.max_bytes);
        let publisher = self.loaded.detached_checkpoint_publisher();
        let worker_store = store.clone();
        let max_record_bytes = plan.max_record_bytes;
        let (ack_sender, acks) = mpsc::channel();
        let worker = DurableWorker::spawn("qwen", queue_cap_bytes(ram_budget_bytes), move || {
            // Creates the namespace so the identity cache entry persists.
            worker_store
                .has_managed_blobs()
                .context("open durable snapshot store")?;
            let (content_id, detail) =
                resolve_content_identity(&gguf, &worker_store.identity_cache())?;
            drop(gguf);
            Ok(Resolved {
                content_id,
                detail,
                writer: move |job: QwenJob| {
                    let result = publisher.publish_with(
                        &worker_store,
                        &content_id,
                        &job.prepared,
                        max_record_bytes,
                        job.may_evict,
                    );
                    let ack = match &result {
                        Ok(_) => WriteAck::Published,
                        Err(error) if error.is_no_room_without_eviction() => WriteAck::NoRoom,
                        Err(_) => WriteAck::Failed,
                    };
                    // The receiver outlives the worker unless the backend
                    // dropped the tier, in which case nobody needs the ack.
                    let _ = ack_sender.send((job.ticket, ack));
                    let report = result?;
                    Ok(format!(
                        "tokens={} blob_bytes={} outcome={:?} may_evict={} managed_bytes={} evicted_entries={} staged_integrity_ms={:.1}",
                        job.prepared.matched_prefix_len(),
                        report.blob_bytes,
                        report.outcome,
                        job.may_evict,
                        report.managed_bytes_after,
                        report.evicted_entries,
                        report.staged_integrity.elapsed.as_secs_f64() * 1e3,
                    ))
                },
            })
        })?;
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=qwen {plan} queue_cap_bytes={} write_policy=spill_on_evict_or_expire+idle_publish+shutdown_flush identity=resolving",
            worker.cap_bytes(),
        );
        self.loaded.set_prefix_cache_spill(true);
        self.durable = Some(QwenDurable {
            plan,
            store,
            worker,
            publisher: self.loaded.detached_checkpoint_publisher(),
            acks,
            pending: RefCell::new(Vec::new()),
            idle: RefCell::new(IdlePublish {
                target: None,
                since: Instant::now(),
                done: true,
                queued: None,
                failures: 0,
            }),
        });
        Ok(())
    }

    /// Memory queued and running durable writes reserve beyond cached
    /// payloads ([`write_copy_bytes`] per job, from enqueue until
    /// acknowledgement). Request, capture, promotion and idle-publication
    /// admission add it.
    pub(super) fn durable_reserved_bytes(&self) -> u64 {
        self.durable.as_ref().map_or(0, |durable| {
            durable
                .pending
                .borrow()
                .iter()
                .map(|(_, _, bytes)| bytes)
                .sum()
        })
    }

    /// Record a completed request: its continuation boundary becomes the
    /// idle-publication target and the idle clock restarts.
    pub(super) fn durable_note_request_end(&self, target: Option<EntryId>) {
        if let Some(durable) = self.durable.as_ref() {
            *durable.idle.borrow_mut() = IdlePublish {
                target,
                since: Instant::now(),
                done: false,
                queued: None,
                failures: 0,
            };
        }
    }

    /// Any request just ended (completed, failed or aborted): restart the
    /// idle clock so publication needs a full idle period without requests.
    pub(super) fn durable_note_request_unwind(&self) {
        if let Some(durable) = self.durable.as_ref() {
            durable.idle.borrow_mut().since = Instant::now();
        }
    }

    /// Apply write acknowledgements: forget their pending jobs, mark
    /// published payloads durable in the cache (pointer identity), and count
    /// a failed idle write toward giving up on its target. Returns the
    /// acknowledgements applied.
    fn settle_acks(&self) -> Vec<(CheckpointTicket, WriteAck)> {
        let Some(durable) = self.durable.as_ref() else {
            return Vec::new();
        };
        let acks: Vec<_> = durable.acks.try_iter().collect();
        for (ticket, ack) in &acks {
            durable
                .pending
                .borrow_mut()
                .retain(|(pending, _, _)| !pending.same(ticket));
            if *ack == WriteAck::Published {
                self.loaded.mark_prefix_cache_durable(ticket);
                continue;
            }
            let mut idle = durable.idle.borrow_mut();
            if idle
                .queued
                .as_ref()
                .is_some_and(|queued| queued.same(ticket))
            {
                // Retry after another idle period, a bounded number of times.
                idle.queued = None;
                idle.failures += 1;
                idle.done = idle.failures >= IDLE_PUBLISH_MAX_FAILURES;
                idle.since = Instant::now();
            }
        }
        acks
    }

    /// Queue `prepared` unless no write is needed (see [`Self::write_key`]).
    /// `may_evict` lets the store evict other records to make room.
    /// `wait_until` blocks for queue room (shutdown only).
    fn enqueue_durable(
        durable: &QwenDurable,
        prepared: PreparedCheckpoint,
        trusted: bool,
        may_evict: bool,
        wait_until: Option<Instant>,
    ) -> Enqueue {
        match Self::write_key(durable, &prepared, trusted) {
            Ok(key) => Self::queue_write(durable, prepared, key, may_evict, wait_until)
                .map_or(Enqueue::Dropped, Enqueue::Queued),
            Err(needless) => needless,
        }
    }

    /// The record key of a write `prepared` needs, or why it needs none: too
    /// short, already in flight (same payload or same record), or `trusted`
    /// and still on disk. `trusted` means this process validated the record:
    /// its write was acknowledged or it was promoted from disk (the entry's
    /// durable marking). An untrusted record found on disk is published
    /// anyway; the store then validates or repairs it.
    fn write_key(
        durable: &QwenDurable,
        prepared: &PreparedCheckpoint,
        trusted: bool,
    ) -> Result<String, Enqueue> {
        if prepared.matched_prefix_len() < durable.plan.min_tokens {
            return Err(Enqueue::Skipped);
        }
        let key =
            durable
                .publisher
                .dedup_key(&durable.store, prepared, durable.plan.max_record_bytes);
        if durable
            .pending
            .borrow()
            .iter()
            .any(|(ticket, pending_key, _)| ticket.refers_to(prepared) || *pending_key == key)
        {
            return Err(Enqueue::InFlight);
        }
        if trusted
            && let Some(content_id) = durable.worker.content_id()
            && durable
                .publisher
                .contains(
                    &durable.store,
                    &content_id,
                    prepared,
                    durable.plan.max_record_bytes,
                )
                .unwrap_or(false)
        {
            return Err(Enqueue::OnDisk(prepared.ticket()));
        }
        Ok(key)
    }

    /// Queue a write [`Self::write_key`] found needed; `None` when the queue
    /// refused it.
    fn queue_write(
        durable: &QwenDurable,
        prepared: PreparedCheckpoint,
        key: String,
        may_evict: bool,
        wait_until: Option<Instant>,
    ) -> Option<CheckpointTicket> {
        let ticket = prepared.ticket();
        let bytes = prepared.snapshot_bytes();
        let reserved = write_copy_bytes(&prepared);
        let job = QwenJob {
            prepared,
            ticket: ticket.clone(),
            may_evict,
        };
        // Pending (and reserved) before the worker can see the job, so an
        // acknowledgement never precedes its pending entry and no admission
        // sees the write's memory as free.
        durable
            .pending
            .borrow_mut()
            .push((ticket.clone(), key, reserved));
        let queued = match wait_until {
            Some(deadline) => durable.worker.enqueue_until(job, bytes, deadline),
            None => durable.worker.try_enqueue(job, bytes),
        };
        if queued {
            Some(ticket)
        } else {
            durable
                .pending
                .borrow_mut()
                .retain(|(pending, _, _)| !pending.same(&ticket));
            None
        }
    }

    /// Queue checkpoints the RAM cache released since the last drain. The
    /// admission that released one (a capture or promotion evicting by
    /// budget) did not count its write, so each needs headroom for its
    /// decode copy now; without it the spill is dropped (logged) and its
    /// payload freed, as for pressure-relief evictions.
    pub(super) fn drain_spills(&self) {
        let spills = self.loaded.take_prefix_cache_spills();
        self.settle_acks();
        let Some(durable) = self.durable.as_ref() else {
            return;
        };
        for (prepared, trusted) in spills {
            let Ok(key) = Self::write_key(durable, &prepared, trusted) else {
                continue;
            };
            if let Err(reason) = super::super::snapshot_capture_admission(
                write_copy_bytes(&prepared)
                    .saturating_add(self.durable_reserved_bytes())
                    .saturating_add(self.control_cpu_reserve),
                self.loaded.context().memory_signals(),
            ) {
                tracing::info!(
                    target: "qwen_diag",
                    "serve durable: family=qwen spill dropped reason=memory_headroom detail={reason:?} tokens={} snapshot_bytes={}",
                    prepared.matched_prefix_len(),
                    prepared.snapshot_bytes(),
                );
                continue;
            }
            Self::queue_write(durable, prepared, key, true, None);
        }
    }

    pub(super) fn durable_idle(&mut self) {
        if self
            .durable
            .as_ref()
            .is_some_and(|durable| durable.worker.failed())
        {
            self.loaded.set_prefix_cache_spill(false);
            self.durable = None;
            return;
        }
        self.drain_spills();
        self.idle_publish();
    }

    /// Publish the latest request's continuation boundary once the server
    /// has been idle long enough and the writer has nothing else to do.
    fn idle_publish(&self) {
        let Some(durable) = self.durable.as_ref() else {
            return;
        };
        let Some(after) = durable.plan.idle_publish_after else {
            return;
        };
        let (target, idle_ms) = {
            let idle = durable.idle.borrow();
            if idle.done
                || idle.since.elapsed() < after
                || durable.worker.content_id().is_none()
                || !durable.worker.is_idle()
            {
                return;
            }
            (idle.target, idle.since.elapsed().as_secs_f64() * 1e3)
        };
        // The target, else (evicted by pressure, or never captured) the
        // ranked entries, best first: the first one that still needs a write.
        let (candidates, role) =
            match target.and_then(|entry| self.loaded.prefix_cache_entry_checkpoint(entry)) {
                Some(entry) => (vec![entry], "target"),
                None => (
                    self.loaded.prefix_cache_persist_candidates(),
                    "ranked_fallback",
                ),
            };
        let examined = candidates.len();
        for (prepared, trusted) in candidates {
            // On disk, in flight or too short: nothing to write for it.
            let Ok(key) = Self::write_key(durable, &prepared, trusted) else {
                continue;
            };
            let tokens = prepared.matched_prefix_len();
            let bytes = prepared.snapshot_bytes();
            // Do not start a write without headroom for the copy it decodes.
            if let Err(reason) = super::super::snapshot_capture_admission(
                write_copy_bytes(&prepared)
                    .saturating_add(self.durable_reserved_bytes())
                    .saturating_add(self.control_cpu_reserve),
                self.loaded.context().memory_signals(),
            ) {
                durable.idle.borrow_mut().since = Instant::now();
                tracing::info!(
                    target: "qwen_diag",
                    "serve durable: family=qwen idle publish deferred reason=memory_headroom detail={reason:?} role={role} tokens={tokens} snapshot_bytes={bytes}"
                );
                return;
            }
            // Refused by the queue: try again next tick.
            let Some(ticket) = Self::queue_write(durable, prepared, key, true, None) else {
                return;
            };
            let mut idle = durable.idle.borrow_mut();
            idle.done = true;
            idle.queued = Some(ticket);
            tracing::info!(
                target: "qwen_diag",
                "serve durable: family=qwen idle publish queued role={role} tokens={tokens} snapshot_bytes={bytes} idle_ms={idle_ms:.0}"
            );
            return;
        }
        durable.idle.borrow_mut().done = true;
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=qwen idle publish skipped reason={} role={role} candidates={examined} idle_ms={idle_ms:.0}",
            if examined == 0 { "nothing_cached" } else { "nothing_to_write" },
        );
    }

    /// Promote the longest durable prefix strictly longer than the best RAM
    /// match. Returns the promoted matched length. Every failure is logged
    /// and falls back to RAM/cold behavior.
    pub(super) fn promote_durable_prefix(&self, prompt_ids: &[i32]) -> Option<usize> {
        let durable = self.durable.as_ref()?;
        let content_id = durable.worker.content_id()?;
        let min_tokens = durable.plan.min_tokens;
        if prompt_ids.len() < min_tokens.max(1) {
            return None;
        }
        let floor = self
            .loaded
            .lookup_cached_prefix(prompt_ids)
            .map_or(0, |lookup| lookup.matched_prefix_len());
        // The lookup's expiry sweep may have released entries; queue them
        // before the promotion's memory admission below.
        self.drain_spills();
        if floor >= prompt_ids.len() {
            return None;
        }
        let t0 = Instant::now();
        let mut denied = None;
        let result = self.loaded.promote_durable_prefix(
            &durable.store,
            &content_id,
            prompt_ids,
            durable.plan.max_record_bytes,
            |matched, blob_bytes| {
                if matched <= floor || matched < min_tokens {
                    return false;
                }
                if !self.loaded.prefix_cache_strict_eligible(blob_bytes) {
                    denied = Some("cache_budget");
                    return false;
                }
                if super::super::admit_snapshot_capture(
                    blob_bytes
                        .saturating_add(self.durable_reserved_bytes())
                        .saturating_add(self.control_cpu_reserve),
                    || self.loaded.context().memory_signals(),
                    |bytes| self.loaded.evict_prefix_cache_for(bytes),
                )
                .is_err()
                {
                    denied = Some("memory_headroom");
                    return false;
                }
                true
            },
        );
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        match result {
            Ok(promotion) => {
                let promoted = promotion.promoted_prefix_len();
                if promoted.is_some() || promotion.lookup.candidates_examined > 0 {
                    tracing::info!(
                        target: "qwen_diag",
                        "serve durable: family=qwen lookup hit={} matched={} ram_matched={floor} candidates={} corrupt_removed={} unusable_skipped={} snapshot_bytes={} denied={} lookup_ms={ms:.1}",
                        promoted.is_some(),
                        promotion.lookup.matched_prefix_len,
                        promotion.lookup.candidates_examined,
                        promotion.lookup.corrupt_entries_removed,
                        promotion.lookup.unusable_skipped,
                        promotion.snapshot_bytes,
                        denied.unwrap_or("none"),
                    );
                }
                promoted
            }
            Err(error) => {
                tracing::warn!(
                    target: "qwen_diag",
                    "serve durable: family=qwen lookup failed after {ms:.1} ms; continuing without disk: {error}"
                );
                None
            }
        }
    }

    /// Graceful-shutdown flush within `plan.shutdown_budget`: jobs already
    /// queued finish first, then the latest continuation target (rewritten
    /// if its record left the disk), then the other ranked entries, one write
    /// at a time so each outcome is reported.
    pub(super) fn durable_shutdown(&mut self) {
        self.drain_spills();
        let Some(durable) = self.durable.as_ref() else {
            return;
        };
        let started = Instant::now();
        if durable.worker.content_id().is_none() {
            tracing::info!(
                target: "qwen_diag",
                "serve durable: family=qwen shutdown flush skipped: identity still resolving",
            );
            return;
        }
        let deadline = started + durable.plan.shutdown_budget;
        durable.worker.wait_idle(deadline);
        self.settle_acks();
        let target = durable.idle.borrow().target;
        let mut candidates = Vec::new();
        if let Some((prepared, trusted)) =
            target.and_then(|entry| self.loaded.prefix_cache_entry_checkpoint(entry))
        {
            candidates.push((prepared, trusted, "target"));
        }
        for (prepared, trusted) in self.loaded.prefix_cache_persist_candidates() {
            let is_target = candidates
                .first()
                .is_some_and(|(target, _, _)| target.ticket().refers_to(&prepared));
            if !is_target {
                candidates.push((prepared, trusted, "ranked"));
            }
        }
        let target_prepared = candidates
            .first()
            .filter(|(_, _, role)| *role == "target")
            .map(|(prepared, _, _)| prepared.ticket());
        let mut target_outcome = None;
        let (mut persisted, mut lost) = (0usize, 0usize);
        for (prepared, trusted, role) in candidates {
            let tokens = prepared.matched_prefix_len();
            let write_t0 = Instant::now();
            // A ranked write never evicts a record (the target's among them):
            // the store refuses it, under its writer lock, when the record
            // fits only by eviction (budget or volume reserve).
            let may_evict = role == "target";
            let outcome = if Instant::now() >= deadline {
                FlushOutcome::TimedOut
            } else {
                match Self::enqueue_durable(durable, prepared, trusted, may_evict, Some(deadline)) {
                    Enqueue::Skipped => continue,
                    Enqueue::OnDisk(ticket) => {
                        self.loaded.mark_prefix_cache_durable(&ticket);
                        FlushOutcome::OnDisk
                    }
                    Enqueue::Queued(ticket) => {
                        if durable.worker.wait_idle(deadline) {
                            let ack = self
                                .settle_acks()
                                .into_iter()
                                .find_map(|(acked, ack)| acked.same(&ticket).then_some(ack));
                            match ack {
                                Some(WriteAck::Published) => FlushOutcome::Published,
                                Some(WriteAck::NoRoom) => FlushOutcome::NoRoom,
                                Some(WriteAck::Failed) | None => FlushOutcome::Failed,
                            }
                        } else {
                            FlushOutcome::TimedOut
                        }
                    }
                    Enqueue::InFlight => FlushOutcome::TimedOut,
                    Enqueue::Dropped => FlushOutcome::Failed,
                }
            };
            if outcome.persisted() {
                persisted += 1;
            } else {
                lost += 1;
            }
            tracing::info!(
                target: "qwen_diag",
                "serve durable: family=qwen shutdown write role={role} tokens={tokens} outcome={} ms={:.1}",
                outcome.as_str(),
                write_t0.elapsed().as_secs_f64() * 1e3,
            );
            if role == "target" {
                target_outcome = Some(outcome);
            }
        }
        // Report where the target stands now, not only how its write went.
        let content_id = durable.worker.content_id();
        let target_on_disk = target
            .and_then(|entry| self.loaded.prefix_cache_entry_checkpoint(entry))
            .filter(|(prepared, _)| {
                target_prepared
                    .as_ref()
                    .is_some_and(|ticket| ticket.refers_to(prepared))
            })
            .zip(content_id)
            .map(|((prepared, _), content_id)| {
                durable
                    .publisher
                    .contains(
                        &durable.store,
                        &content_id,
                        &prepared,
                        durable.plan.max_record_bytes,
                    )
                    .unwrap_or(false)
            });
        let stats = durable.worker.stats();
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=qwen shutdown flush target={} target_on_disk={} persisted={persisted} not_persisted={lost} elapsed_ms={:.1} budget_ms={} written_total={} failed_total={} dropped_total={}",
            target_outcome.map_or("none", FlushOutcome::as_str),
            target_on_disk.map_or("none", |on_disk| if on_disk { "true" } else { "false" }),
            started.elapsed().as_secs_f64() * 1e3,
            durable.plan.shutdown_budget.as_millis(),
            stats.written,
            stats.failed,
            stats.dropped,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::durable::{DurableDir, DurableSnapshotConfig};
    use super::super::super::http::{GenerationBackend, GenerationSink};
    use super::super::{IM_START_MARKER, request_sampler, transcript_boundary};
    use super::*;
    use qwen_llm::runtime::SequenceConfig;
    use std::io;
    use std::time::Duration;

    #[derive(Default)]
    struct Sink(Vec<u8>);

    impl GenerationSink for Sink {
        fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.0.extend_from_slice(bytes);
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn backend(model: &Path, dir: &Path, idle_publish_after: Option<Duration>) -> EngineBackend {
        let runtime = qwen_llm::runtime::Runtime::metal().unwrap();
        let loaded = runtime.load_model(model).unwrap();
        let family = qwen_llm::model_family::ModelFamily::detect(loaded.gguf()).unwrap();
        let template = crate::prompt_template::serve_qwen_template(family, loaded.gguf()).unwrap();
        let no_thinking = crate::supports_qwen_no_thinking_prompt(family, loaded.gguf());
        let mut backend = EngineBackend::new(
            loaded,
            "durable-restart-test".into(),
            4,
            Some(128),
            128,
            None,
            template,
            no_thinking,
        )
        .unwrap();
        let mut plan = DurableSnapshotConfig {
            dir: DurableDir::Path(dir.to_path_buf()),
            max_mib: Some(4096),
            min_tokens: 1,
            // Idle publication off unless a test turns it on below.
            idle_publish_secs: 0,
            ..DurableSnapshotConfig::off()
        }
        .resolve("qwen")
        .unwrap()
        .unwrap();
        plan.idle_publish_after = idle_publish_after;
        backend.attach_durable(plan, model, 1 << 30).unwrap();
        let started = Instant::now();
        while backend
            .durable
            .as_ref()
            .unwrap()
            .worker
            .content_id()
            .is_none()
        {
            assert!(
                started.elapsed() < Duration::from_secs(300),
                "identity never resolved"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        backend
    }

    fn test_model_and_dir(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let model = std::path::PathBuf::from(
            std::env::var("QWEN_DURABLE_TEST_MODEL")
                .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf".into()),
        );
        let dir = std::env::temp_dir().join(format!(
            "qwen-serve-durable-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        (model, dir)
    }

    /// Idle publication survives a crash: after a request, idle ticks write
    /// the latest continuation (transcript) boundary once, acknowledge it and
    /// mark the entry durable; further ticks do not rewrite it. Dropping the
    /// backend without any shutdown flush stands in for a crash; a new
    /// backend over the same directory then promotes that exact boundary.
    #[test]
    #[ignore = "serial Metal; loads QWEN_DURABLE_TEST_MODEL (default 0.8B) twice"]
    fn idle_publication_writes_the_continuation_once_and_survives_a_crash() {
        let (model, dir) = test_model_and_dir("idle");
        let request = crate::open_responses::items::parse_request(&serde_json::json!({
            "model":"durable-restart-test", "input":"Hi",
            "temperature":0.0, "max_output_tokens":4,
        }))
        .unwrap();
        let mut first = backend(&model, &dir, Some(Duration::ZERO));
        let prompt = first.render_prompt(&request).unwrap();
        let prompt_ids = first.tokenizer.encode(&prompt, false).unwrap();
        let im_start = first.tokenizer.encode(IM_START_MARKER, false).unwrap();
        let boundary = transcript_boundary(&prompt_ids, im_start[0]).unwrap();
        first
            .generate(&request, &prompt, &mut Sink::default())
            .unwrap();
        first.request_finished();
        let target = first.durable.as_ref().unwrap().idle.borrow().target;
        let (prepared, marked) = first
            .loaded
            .prefix_cache_entry_checkpoint(target.expect("a continuation target"))
            .unwrap();
        assert!(!marked);
        let published = prepared.matched_tokens();
        assert_eq!(published, prompt_ids[..boundary], "the transcript boundary");
        drop(prepared);

        let started = Instant::now();
        loop {
            first.idle();
            let durable = first.durable.as_ref().unwrap();
            if durable.worker.is_idle() && durable.pending.borrow().is_empty() {
                let (_, marked) = first
                    .loaded
                    .prefix_cache_entry_checkpoint(target.unwrap())
                    .unwrap();
                if marked {
                    break;
                }
            }
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "never published"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        for _ in 0..5 {
            first.idle();
        }
        let stats = first.durable.as_ref().unwrap().worker.stats();
        assert_eq!((stats.written, stats.failed), (1, 0), "one idle write");
        let durable = first.durable.as_ref().unwrap();
        let (prepared, marked) = first
            .loaded
            .prefix_cache_entry_checkpoint(target.unwrap())
            .unwrap();
        assert!(marked);
        assert!(
            durable
                .publisher
                .contains(
                    &durable.store,
                    &durable.worker.content_id().unwrap(),
                    &prepared,
                    durable.plan.max_record_bytes,
                )
                .unwrap()
        );
        drop(prepared);
        // A crash: no shutdown flush, no further writes.
        drop(first);

        let second = backend(&model, &dir, None);
        assert_eq!(
            second.promote_durable_prefix(&prompt_ids),
            Some(published.len()),
            "the idle-published continuation boundary is restorable"
        );
        drop(second);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Restart continuity: capture in one backend, flush on shutdown, drop
    /// it, then a new backend over the same directory restores the prompt
    /// boundary from disk and continues bit-identically (serial prefill of a
    /// short prompt keeps the comparison exact, as in the RAM parity test).
    #[test]
    #[ignore = "serial Metal; loads QWEN_DURABLE_TEST_MODEL (default 0.8B) twice"]
    fn restart_restores_prompt_boundary_from_disk_with_identical_continuation() {
        let model = std::path::PathBuf::from(
            std::env::var("QWEN_DURABLE_TEST_MODEL")
                .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf".into()),
        );
        let dir = std::env::temp_dir().join(format!(
            "qwen-serve-durable-restart-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let request = crate::open_responses::items::parse_request(&serde_json::json!({
            "model":"durable-restart-test", "input":"Hi",
            "temperature":0.0, "max_output_tokens":4,
        }))
        .unwrap();

        let mut first = backend(&model, &dir, None);
        let prompt = first.render_prompt(&request).unwrap();
        let prompt_ids = first.tokenizer.encode(&prompt, false).unwrap();
        let im_start = first.tokenizer.encode(IM_START_MARKER, false).unwrap();
        let boundary = transcript_boundary(&prompt_ids, im_start[0]).unwrap();
        let mut cold = Sink::default();
        let outcome = first.generate(&request, &prompt, &mut cold).unwrap();
        assert_eq!(outcome.usage.cached_tokens, 0);
        assert_eq!(first.restore_source, "none");
        first.shutdown();
        let stats = first.durable.as_ref().unwrap().worker.stats();
        assert!(stats.written >= 1, "{stats:?}");
        assert_eq!(stats.failed, 0);
        drop(first);

        let mut second = backend(&model, &dir, None);
        let mut warm = Sink::default();
        let outcome = second.generate(&request, &prompt, &mut warm).unwrap();
        assert_eq!(second.restore_source, "disk");
        assert_eq!(outcome.usage.cached_tokens, boundary);
        assert_eq!(warm.0, cold.0, "continuation differs after restart");

        // The completed boundary captured after the disk restore equals a
        // cold single-token reference bit for bit.
        let mut reference = second
            .loaded
            .create_sequence(SequenceConfig::new(128))
            .unwrap();
        let mut logits = Vec::new();
        for (position, &token) in prompt_ids.iter().enumerate() {
            logits = second
                .loaded
                .forward()
                .single_token(token, position as u32, unsafe {
                    reference.metal_session_mut()
                })
                .unwrap();
            reference.advance_by(1).unwrap();
        }
        let generation = crate::generate_serial(
            logits,
            4,
            &second.loaded.gguf().stop_token_ids().unwrap(),
            &mut request_sampler(&request).unwrap(),
            |_| Ok(()),
            |token| {
                let position = reference.position();
                let logits =
                    second
                        .loaded
                        .forward()
                        .single_token(token, position as u32, unsafe {
                            reference.metal_session_mut()
                        })?;
                reference.advance_by(1)?;
                Ok(logits)
            },
        )
        .unwrap();
        let mut completed = prompt_ids.clone();
        completed.extend_from_slice(&generation.tokens);
        let identity = second.loaded.snapshot_identity(&reference).unwrap();
        let expected = reference
            .metal_session()
            .snapshot(
                identity.clone(),
                completed[..completed.len() - 1].to_vec(),
                None,
            )
            .unwrap();
        let lookup = second.loaded.lookup_cached_prefix(&completed).unwrap();
        let mut restored = second
            .loaded
            .create_sequence(SequenceConfig::new(128))
            .unwrap();
        let report = second
            .loaded
            .restore_prepared_cached_prefix(lookup, &mut restored, &completed)
            .unwrap();
        assert!(report.exact);
        let actual = restored
            .metal_session()
            .snapshot(identity, completed[..completed.len() - 1].to_vec(), None)
            .unwrap();
        assert_eq!(actual.kv_k_arena, expected.kv_k_arena);
        assert_eq!(actual.kv_v_arena, expected.kv_v_arena);
        assert_eq!(actual.gdn_conv_arena, expected.gdn_conv_arena);
        assert_eq!(actual.gdn_state_arena, expected.gdn_state_arena);
        drop(second);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restore_source_reports_disk_only_for_the_promoted_hit() {
        assert_eq!(restore_source(None, None), "none");
        assert_eq!(restore_source(Some(10), None), "ram");
        assert_eq!(restore_source(Some(10), Some(10)), "disk");
        // A promotion that lost to a longer (or exact-logits) RAM hit.
        assert_eq!(restore_source(Some(12), Some(10)), "ram");
        // Promoted but refused by the RAM lookup's completion rules.
        assert_eq!(restore_source(None, Some(10)), "none");
    }
}

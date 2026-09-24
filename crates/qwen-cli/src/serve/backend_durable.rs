//! Durable tier for the Qwen engine backend.
//!
//! Write policy: dense Qwen snapshots are GBs, so nothing is written per
//! turn. An entry is written when it *leaves* RAM through budget eviction or
//! idle/age expiry (the prefix cache retains those payloads until drained
//! here), and on graceful shutdown the top-ranked entries not already on disk
//! are flushed within [`SHUTDOWN_FLUSH_BUDGET`]. Pressure-relief evictions
//! are not spilled: that memory is needed immediately. Entries promoted from
//! disk are marked durable in the cache and never rewritten.
//!
//! Read policy: before the RAM lookup, a disk record strictly longer than
//! the best RAM match is promoted into RAM (subject to the strict cache
//! budget and the capture memory admission), so one restore path serves
//! both tiers.

use super::super::durable::{
    DurablePlan, DurableWorker, Resolved, SHUTDOWN_FLUSH_BUDGET, queue_cap_bytes,
    resolve_content_identity,
};
use super::EngineBackend;
use anyhow::{Context as _, Result, ensure};
use qwen_llm::checkpoint_identity::same_identity_sources;
use qwen_llm::checkpoint_store::DurableCheckpointStore;
use qwen_llm::gguf::GgufFile;
use qwen_llm::runtime::PreparedCheckpoint;
use std::path::Path;
use std::time::Instant;

pub(super) struct QwenDurable {
    plan: DurablePlan,
    store: DurableCheckpointStore,
    worker: DurableWorker<PreparedCheckpoint>,
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
                writer: move |prepared: PreparedCheckpoint| {
                    let report = publisher.publish(
                        &worker_store,
                        &content_id,
                        &prepared,
                        max_record_bytes,
                    )?;
                    Ok(format!(
                        "tokens={} blob_bytes={} outcome={:?} managed_bytes={} evicted_entries={}",
                        prepared.matched_prefix_len(),
                        report.blob_bytes,
                        report.outcome,
                        report.managed_bytes_after,
                        report.evicted_entries,
                    ))
                },
            })
        })?;
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=qwen {plan} queue_cap_bytes={} write_policy=spill_on_evict_or_expire+shutdown_flush identity=resolving",
            worker.cap_bytes(),
        );
        self.loaded.set_prefix_cache_spill(true);
        self.durable = Some(QwenDurable {
            plan,
            store,
            worker,
        });
        Ok(())
    }

    /// Queue checkpoints the RAM cache released since the last drain.
    pub(super) fn drain_spills(&self) {
        let spills = self.loaded.take_prefix_cache_spills();
        let Some(durable) = self.durable.as_ref() else {
            return;
        };
        for prepared in spills {
            if prepared.matched_prefix_len() >= durable.plan.min_tokens {
                let bytes = prepared.snapshot_bytes();
                durable.worker.try_enqueue(prepared, bytes);
            }
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
                    blob_bytes,
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

    /// Best-effort flush of the highest-ranked RAM entries not already on
    /// disk, bounded by [`SHUTDOWN_FLUSH_BUDGET`].
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
        let deadline = started + SHUTDOWN_FLUSH_BUDGET;
        let (mut queued, mut skipped) = (0usize, 0usize);
        for prepared in self.loaded.prefix_cache_persist_candidates() {
            if prepared.matched_prefix_len() < durable.plan.min_tokens {
                continue;
            }
            let bytes = prepared.snapshot_bytes();
            if Instant::now() < deadline && durable.worker.enqueue_until(prepared, bytes, deadline)
            {
                queued += 1;
            } else {
                skipped += 1;
            }
        }
        let drained = durable.worker.wait_idle(deadline);
        let stats = durable.worker.stats();
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=qwen shutdown flush queued={queued} skipped={skipped} drained={drained} elapsed_ms={:.1} written_total={} failed_total={} dropped_total={}",
            started.elapsed().as_secs_f64() * 1e3,
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

    fn backend(model: &Path, dir: &Path) -> EngineBackend {
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
        let plan = DurableSnapshotConfig {
            dir: DurableDir::Path(dir.to_path_buf()),
            max_mib: Some(4096),
            min_tokens: 1,
        }
        .resolve("qwen")
        .unwrap()
        .unwrap();
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

        let mut first = backend(&model, &dir);
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

        let mut second = backend(&model, &dir);
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

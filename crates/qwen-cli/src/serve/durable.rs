//! Durable (cross-restart) tier under the serve RAM snapshot caches.
//!
//! - [`DurableSnapshotConfig`] is the `--durable-snapshot-*` flag surface and
//!   resolves to a per-family [`DurablePlan`] (directory, disk budget,
//!   minimum persisted prefix).
//! - [`DurableWorker`] is the single background thread that owns every disk
//!   write. It first resolves the model's strong content identity (which may
//!   hash the whole GGUF on a cold identity cache), then drains a bounded
//!   queue of write jobs. The request path only enqueues; it never waits on
//!   encode or fsync. Until the identity resolves, the durable read path is
//!   inactive and queued jobs wait (within the byte bound).
//!
//! Families decide *what* to write: Qwen spills entries leaving RAM through
//! budget eviction or expiry and flushes its top-ranked entries on graceful
//! shutdown (dense snapshots are GBs, so never every turn); DeepSeek V4
//! writes behind every captured boundary (its snapshots are small).

use anyhow::{Context as _, Result, bail};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
pub(crate) const DEFAULT_MIN_TOKENS: usize = 1024;
/// `auto` disk budget: min(this, 10% of the volume's free space).
const AUTO_MAX_BYTES: u64 = 64 * GIB;
const AUTO_FREE_SPACE_DIVISOR: u64 = 10;
/// `auto` when free space cannot be read.
const AUTO_FALLBACK_BYTES: u64 = 8 * GIB;
/// One record may not exceed this even under a larger disk budget.
const MAX_RECORD_BYTES: u64 = 16 * GIB;
const MIN_QUEUE_BYTES: u64 = GIB;
const MAX_QUEUE_BYTES: u64 = 16 * GIB;
/// Graceful-shutdown persistence budget.
pub(crate) const SHUTDOWN_FLUSH_BUDGET: Duration = Duration::from_secs(10);
const DEFAULT_SUBDIR: &str = ".cache/qwen-llm/serve-checkpoints";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DurableDir {
    /// `~/.cache/qwen-llm/serve-checkpoints`.
    Default,
    Off,
    Path(PathBuf),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DurableSnapshotConfig {
    pub(crate) dir: DurableDir,
    /// `None` is `auto`.
    pub(crate) max_mib: Option<u64>,
    pub(crate) min_tokens: usize,
}

impl DurableSnapshotConfig {
    pub(crate) const fn off() -> Self {
        Self {
            dir: DurableDir::Off,
            max_mib: None,
            min_tokens: DEFAULT_MIN_TOKENS,
        }
    }

    /// Resolve for one family; `Ok(None)` is disabled. Each family gets its
    /// own subdirectory (and therefore its own byte budget).
    pub(crate) fn resolve(&self, family: &str) -> Result<Option<DurablePlan>> {
        let base = match &self.dir {
            DurableDir::Off => return Ok(None),
            DurableDir::Path(path) => path.clone(),
            DurableDir::Default => match std::env::var_os("HOME") {
                Some(home) if !home.is_empty() => PathBuf::from(home).join(DEFAULT_SUBDIR),
                _ => bail!("HOME is unset; pass --durable-snapshot-dir PATH or off"),
            },
        };
        let root = base.join(family);
        let (max_bytes, budget) = match self.max_mib {
            Some(mib) => (
                mib.checked_mul(MIB)
                    .context("--durable-snapshot-max-mib byte conversion overflow")?,
                BudgetSource::Explicit,
            ),
            None => {
                let free_bytes = qwen_llm::checkpoint_store::volume_free_bytes(&root);
                (
                    auto_budget_bytes(free_bytes),
                    BudgetSource::Auto { free_bytes },
                )
            }
        };
        if max_bytes == 0 {
            return Ok(None);
        }
        Ok(Some(DurablePlan {
            root,
            max_bytes,
            max_record_bytes: max_bytes.min(MAX_RECORD_BYTES),
            min_tokens: self.min_tokens,
            budget,
        }))
    }
}

pub(crate) fn parse_durable_dir(value: &str) -> std::result::Result<DurableDir, String> {
    match value {
        "" => Err("expected a directory or `off`".into()),
        "off" => Ok(DurableDir::Off),
        path => Ok(DurableDir::Path(PathBuf::from(path))),
    }
}

/// `None` is `auto`.
pub(crate) fn parse_durable_max_mib(value: &str) -> std::result::Result<Option<u64>, String> {
    if value == "auto" {
        return Ok(None);
    }
    value
        .parse::<u64>()
        .map(Some)
        .map_err(|error| format!("expected `auto` or a MiB count, got {value:?}: {error}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BudgetSource {
    Explicit,
    Auto { free_bytes: Option<u64> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DurablePlan {
    pub(crate) root: PathBuf,
    /// Managed blob bytes on disk (the stores evict by mtime LRU to fit).
    pub(crate) max_bytes: u64,
    /// Per-record allocation bound for encode and decode.
    pub(crate) max_record_bytes: u64,
    /// Prefixes shorter than this are neither persisted nor looked up.
    pub(crate) min_tokens: usize,
    budget: BudgetSource,
}

impl std::fmt::Display for DurablePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "durable_dir={} durable_max_bytes={} durable_max_record_bytes={} durable_min_tokens={}",
            self.root.display(),
            self.max_bytes,
            self.max_record_bytes,
            self.min_tokens,
        )?;
        match self.budget {
            BudgetSource::Explicit => write!(f, " durable_budget_source=explicit"),
            BudgetSource::Auto { free_bytes } => write!(
                f,
                " durable_budget_source=auto free_bytes={}",
                free_bytes.map_or_else(|| "unknown".to_owned(), |bytes| bytes.to_string())
            ),
        }
    }
}

/// min(64 GiB, 10% of free space); a fixed fallback when free space is
/// unreadable.
pub(crate) fn auto_budget_bytes(free_bytes: Option<u64>) -> u64 {
    free_bytes.map_or(AUTO_FALLBACK_BYTES, |free| {
        (free / AUTO_FREE_SPACE_DIVISOR).min(AUTO_MAX_BYTES)
    })
}

/// Bytes a spill/write-behind queue may hold beyond the RAM cache budget:
/// a quarter of that budget, clamped to [1 GiB, 16 GiB]. One snapshot larger
/// than this still goes through, alone, when the queue is idle.
pub(crate) fn queue_cap_bytes(ram_budget_bytes: u64) -> u64 {
    (ram_budget_bytes / 4).clamp(MIN_QUEUE_BYTES, MAX_QUEUE_BYTES)
}

/// Resolve the strong content identity over `gguf` (a second open of the
/// loaded files; see `same_identity_sources`) and describe how it resolved.
pub(crate) fn resolve_content_identity(
    gguf: &GgufFile,
    cache: &CheckpointIdentityCache,
) -> Result<([u8; 32], String)> {
    let report = checkpoint_content_identity(gguf, cache)
        .context("resolve strong model content identity")?;
    Ok((
        report.content_id,
        format!(
            "identity_cache={:?} hashed_bytes={}",
            report.outcome, report.bytes_hashed
        ),
    ))
}

/// What the worker's resolution step produces before it drains jobs.
pub(crate) struct Resolved<W> {
    pub(crate) content_id: [u8; 32],
    /// Logged when the tier becomes active.
    pub(crate) detail: String,
    pub(crate) writer: W,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct WorkerStats {
    pub(crate) enqueued: u64,
    pub(crate) dropped: u64,
    pub(crate) written: u64,
    pub(crate) failed: u64,
}

struct State<J> {
    jobs: VecDeque<(J, u64)>,
    queued_bytes: u64,
    writing: bool,
    closed: bool,
    failed: bool,
    drop_logged: bool,
    stats: WorkerStats,
}

struct Shared<J> {
    state: Mutex<State<J>>,
    changed: Condvar,
}

impl<J> Shared<J> {
    fn lock(&self) -> MutexGuard<'_, State<J>> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

/// One background thread owning identity resolution and every durable write
/// for a family. Jobs count their payload bytes against `cap_bytes` from
/// enqueue until written; a job that does not fit is dropped (logged once).
/// Dropping the worker closes the queue without joining: an in-flight write
/// may finish or be abandoned at process exit, and the stores clean up
/// abandoned staging files on their next publication.
pub(crate) struct DurableWorker<J> {
    shared: Arc<Shared<J>>,
    identity: Arc<OnceLock<[u8; 32]>>,
    cap_bytes: u64,
    label: &'static str,
}

impl<J: Send + 'static> DurableWorker<J> {
    pub(crate) fn spawn<W>(
        label: &'static str,
        cap_bytes: u64,
        resolve: impl FnOnce() -> Result<Resolved<W>> + Send + 'static,
    ) -> Result<Self>
    where
        W: FnMut(J) -> Result<String> + Send + 'static,
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                queued_bytes: 0,
                writing: false,
                closed: false,
                failed: false,
                drop_logged: false,
                stats: WorkerStats::default(),
            }),
            changed: Condvar::new(),
        });
        let identity = Arc::new(OnceLock::new());
        let worker_shared = Arc::clone(&shared);
        let worker_identity = Arc::clone(&identity);
        std::thread::Builder::new()
            .name(format!("qwen-durable-{label}"))
            .spawn(move || run_worker(label, worker_shared, worker_identity, resolve))
            .context("spawn durable snapshot worker")?;
        Ok(Self {
            shared,
            identity,
            cap_bytes,
            label,
        })
    }
}

impl<J> DurableWorker<J> {
    /// The strong content identity once resolved; the read path and any
    /// identity-bound capture stay inactive until then.
    pub(crate) fn content_id(&self) -> Option<[u8; 32]> {
        self.identity.get().copied()
    }

    /// Identity resolution failed; the tier is permanently inactive.
    pub(crate) fn failed(&self) -> bool {
        self.shared.lock().failed
    }

    pub(crate) fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    pub(crate) fn stats(&self) -> WorkerStats {
        self.shared.lock().stats
    }

    /// Queue without waiting. `false` means the job was dropped: the queue
    /// is full, closed, or the tier failed.
    pub(crate) fn try_enqueue(&self, job: J, bytes: u64) -> bool {
        let mut state = self.shared.lock();
        if state.failed || state.closed {
            return false;
        }
        if !fits(state.queued_bytes, bytes, self.cap_bytes) {
            state.stats.dropped += 1;
            let first_drop = !std::mem::replace(&mut state.drop_logged, true);
            let queued_bytes = state.queued_bytes;
            // Log and free the payload after unlocking: the request path
            // enqueues, and must not wait on a blocked log writer.
            drop(state);
            if first_drop {
                tracing::warn!(
                    target: "qwen_diag",
                    "serve durable: family={} write queue full; dropping snapshot (logged once) job_bytes={bytes} queued_bytes={queued_bytes} cap_bytes={}",
                    self.label,
                    self.cap_bytes,
                );
            }
            drop(job);
            return false;
        }
        push(&mut state, job, bytes);
        drop(state);
        self.shared.changed.notify_all();
        true
    }

    /// Queue, waiting up to `deadline` for room. Used only off the request
    /// path (graceful shutdown). A job larger than the cap waits for an idle
    /// queue.
    pub(crate) fn enqueue_until(&self, job: J, bytes: u64, deadline: Instant) -> bool {
        let mut state = self.shared.lock();
        loop {
            if state.failed || state.closed {
                return false;
            }
            if fits(state.queued_bytes, bytes, self.cap_bytes) {
                push(&mut state, job, bytes);
                drop(state);
                self.shared.changed.notify_all();
                return true;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            state = self
                .shared
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poison| poison.into_inner())
                .0;
        }
    }

    /// Wait until every queued job is written, the tier fails, or `deadline`.
    /// Returns whether the queue drained.
    pub(crate) fn wait_idle(&self, deadline: Instant) -> bool {
        let mut state = self.shared.lock();
        loop {
            if state.jobs.is_empty() && !state.writing {
                return true;
            }
            if state.failed {
                return false;
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            state = self
                .shared
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poison| poison.into_inner())
                .0;
        }
    }
}

impl<J> Drop for DurableWorker<J> {
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.changed.notify_all();
    }
}

/// Room for `bytes` within the cap, or, for a job larger than the whole cap,
/// an idle queue (queued bytes count until written, so zero also means
/// nothing in flight). Such a job came out of the RAM cache, so the RAM
/// budget bounds it, and only one is held beyond the cap at a time; without
/// this, long-context snapshots larger than the cap could never persist.
fn fits(queued: u64, bytes: u64, cap: u64) -> bool {
    queued.checked_add(bytes).is_some_and(|total| total <= cap) || (bytes > cap && queued == 0)
}

fn push<J>(state: &mut State<J>, job: J, bytes: u64) {
    state.jobs.push_back((job, bytes));
    state.queued_bytes += bytes;
    state.stats.enqueued += 1;
}

fn run_worker<J, W>(
    label: &'static str,
    shared: Arc<Shared<J>>,
    identity: Arc<OnceLock<[u8; 32]>>,
    resolve: impl FnOnce() -> Result<Resolved<W>>,
) where
    W: FnMut(J) -> Result<String>,
{
    let started = Instant::now();
    let mut writer = match resolve() {
        Ok(resolved) => {
            let _ = identity.set(resolved.content_id);
            tracing::info!(
                target: "qwen_diag",
                "serve durable: family={label} tier active after {:.1} ms {}",
                started.elapsed().as_secs_f64() * 1e3,
                resolved.detail,
            );
            resolved.writer
        }
        Err(error) => {
            tracing::warn!(
                target: "qwen_diag",
                "serve durable: family={label} tier disabled after {:.1} ms: {error:#}",
                started.elapsed().as_secs_f64() * 1e3,
            );
            let mut state = shared.lock();
            state.failed = true;
            let abandoned = std::mem::take(&mut state.jobs);
            state.queued_bytes = 0;
            drop(state);
            shared.changed.notify_all();
            drop(abandoned);
            return;
        }
    };
    loop {
        let (job, bytes) = {
            let mut state = shared.lock();
            loop {
                if state.closed {
                    return;
                }
                if let Some(job) = state.jobs.pop_front() {
                    state.writing = true;
                    break job;
                }
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(|poison| poison.into_inner());
            }
        };
        let write_t0 = Instant::now();
        let result = writer(job);
        let write_ms = write_t0.elapsed().as_secs_f64() * 1e3;
        {
            let mut state = shared.lock();
            state.writing = false;
            state.queued_bytes -= bytes;
            match &result {
                Ok(_) => state.stats.written += 1,
                Err(_) => state.stats.failed += 1,
            }
        }
        shared.changed.notify_all();
        // Logged after unlocking so a blocked log writer never holds the
        // queue the request path enqueues into.
        match result {
            Ok(detail) => tracing::info!(
                target: "qwen_diag",
                "serve durable: family={label} wrote snapshot write_ms={write_ms:.1} {detail}",
            ),
            Err(error) => tracing::warn!(
                target: "qwen_diag",
                "serve durable: family={label} snapshot write failed after {write_ms:.1} ms: {error:#}",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    const LONG: Duration = Duration::from_secs(5);

    #[test]
    fn flags_default_and_parse() {
        assert_eq!(parse_durable_dir("off"), Ok(DurableDir::Off));
        assert_eq!(
            parse_durable_dir("/tmp/x"),
            Ok(DurableDir::Path("/tmp/x".into()))
        );
        assert!(parse_durable_dir("").is_err());
        assert_eq!(parse_durable_max_mib("auto"), Ok(None));
        assert_eq!(parse_durable_max_mib("2048"), Ok(Some(2048)));
        assert!(parse_durable_max_mib("Auto").is_err());
        assert!(parse_durable_max_mib("-1").is_err());
    }

    #[test]
    fn plan_resolution_uses_family_subdirs_and_budgets() {
        assert_eq!(DurableSnapshotConfig::off().resolve("qwen").unwrap(), None);
        let explicit = DurableSnapshotConfig {
            dir: DurableDir::Path("/tmp/durable".into()),
            max_mib: Some(2048),
            min_tokens: 7,
        };
        let plan = explicit.resolve("deepseek_v4").unwrap().unwrap();
        assert_eq!(plan.root, std::path::Path::new("/tmp/durable/deepseek_v4"));
        assert_eq!(plan.max_bytes, 2 * GIB);
        assert_eq!(plan.max_record_bytes, 2 * GIB);
        assert_eq!(plan.min_tokens, 7);
        assert!(plan.to_string().contains("durable_budget_source=explicit"));
        let huge = DurableSnapshotConfig {
            max_mib: Some(100 * 1024),
            ..explicit.clone()
        };
        assert_eq!(
            huge.resolve("qwen").unwrap().unwrap().max_record_bytes,
            MAX_RECORD_BYTES
        );
        // Zero MiB disables rather than creating an unusable store.
        let zero = DurableSnapshotConfig {
            max_mib: Some(0),
            ..explicit.clone()
        };
        assert_eq!(zero.resolve("qwen").unwrap(), None);
        let overflow = DurableSnapshotConfig {
            max_mib: Some(u64::MAX),
            ..explicit
        };
        assert!(overflow.resolve("qwen").is_err());
    }

    #[test]
    fn auto_budget_is_a_tenth_of_free_space_capped_at_64_gib() {
        assert_eq!(auto_budget_bytes(Some(100 * GIB)), 10 * GIB);
        assert_eq!(auto_budget_bytes(Some(640 * GIB)), 64 * GIB);
        assert_eq!(auto_budget_bytes(Some(4 * 1024 * GIB)), 64 * GIB);
        assert_eq!(auto_budget_bytes(Some(0)), 0);
        assert_eq!(auto_budget_bytes(None), AUTO_FALLBACK_BYTES);
        // Reads a real volume through a not-yet-created leaf.
        let free = qwen_llm::checkpoint_store::volume_free_bytes(
            &std::env::temp_dir().join("qwen-durable-missing/a/b"),
        );
        assert!(free.is_some());
    }

    #[test]
    fn queue_cap_tracks_ram_budget_within_bounds() {
        assert_eq!(queue_cap_bytes(0), GIB);
        assert_eq!(queue_cap_bytes(32 * GIB), 8 * GIB);
        assert_eq!(queue_cap_bytes(u64::MAX), 16 * GIB);
    }

    /// A worker whose writer blocks on a gate so tests control draining.
    fn gated_worker(cap: u64) -> (DurableWorker<u32>, mpsc::Sender<()>, mpsc::Receiver<u32>) {
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<u32>();
        let worker = DurableWorker::spawn("test", cap, move || {
            Ok(Resolved {
                content_id: [7; 32],
                detail: String::new(),
                writer: move |job: u32| {
                    gate_rx.recv().expect("gate");
                    done_tx.send(job).unwrap();
                    Ok(String::new())
                },
            })
        })
        .unwrap();
        (worker, gate_tx, done_rx)
    }

    fn wait_for_identity<J>(worker: &DurableWorker<J>) {
        let started = Instant::now();
        while worker.content_id().is_none() {
            assert!(started.elapsed() < LONG, "identity never resolved");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn queue_is_bounded_by_bytes_and_drops_when_full() {
        let (worker, gate, done) = gated_worker(100);
        wait_for_identity(&worker);
        assert_eq!(worker.content_id(), Some([7; 32]));
        assert!(worker.try_enqueue(1, 60));
        assert!(worker.try_enqueue(2, 40));
        // Full (the in-flight job still counts until written).
        assert!(!worker.try_enqueue(3, 1));
        assert!(!worker.try_enqueue(4, 101));
        assert_eq!(worker.stats().dropped, 2);
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 1);
        // Job 1's bytes are released once written.
        let started = Instant::now();
        while !worker.try_enqueue(5, 60) {
            assert!(started.elapsed() < LONG);
            std::thread::sleep(Duration::from_millis(1));
        }
        gate.send(()).unwrap();
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 2);
        assert_eq!(done.recv_timeout(LONG).unwrap(), 5);
        assert!(worker.wait_idle(Instant::now() + LONG));
        let stats = worker.stats();
        assert_eq!((stats.enqueued, stats.written, stats.failed), (3, 3, 0));
    }

    #[test]
    fn jobs_queued_before_identity_wait_for_it() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<u32>();
        let worker = DurableWorker::spawn("test", 10, move || {
            release_rx.recv().unwrap();
            Ok(Resolved {
                content_id: [1; 32],
                detail: String::new(),
                writer: move |job: u32| {
                    done_tx.send(job).unwrap();
                    Ok(String::new())
                },
            })
        })
        .unwrap();
        assert!(worker.content_id().is_none());
        assert!(worker.try_enqueue(9, 10));
        assert!(!worker.wait_idle(Instant::now() + Duration::from_millis(20)));
        release_tx.send(()).unwrap();
        assert_eq!(done_rx.recv_timeout(LONG).unwrap(), 9);
        assert!(worker.wait_idle(Instant::now() + LONG));
        assert_eq!(worker.content_id(), Some([1; 32]));
    }

    #[test]
    fn failed_identity_disables_the_tier() {
        let worker: DurableWorker<u32> = DurableWorker::spawn("test", 10, || {
            Err::<Resolved<fn(u32) -> Result<String>>, _>(anyhow::anyhow!("no identity"))
        })
        .unwrap();
        let started = Instant::now();
        while !worker.failed() {
            assert!(started.elapsed() < LONG);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(worker.content_id().is_none());
        assert!(!worker.try_enqueue(1, 1));
        assert_eq!(worker.stats().enqueued, 0);
    }

    #[test]
    fn oversized_job_goes_alone_when_the_queue_is_idle() {
        let (worker, gate, done) = gated_worker(100);
        wait_for_identity(&worker);
        assert!(worker.try_enqueue(1, 10));
        assert!(
            !worker.try_enqueue(2, 150),
            "busy queue refuses an oversized job"
        );
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 1);
        assert!(worker.wait_idle(Instant::now() + LONG));
        assert!(
            worker.try_enqueue(3, 150),
            "idle queue admits one oversized job"
        );
        assert!(
            !worker.try_enqueue(4, 1),
            "nothing joins it while in flight"
        );
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 3);
        assert!(worker.wait_idle(Instant::now() + LONG));
        assert!(worker.try_enqueue(5, 60));
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 5);
        assert!(worker.wait_idle(Instant::now() + LONG));
    }

    #[test]
    fn enqueue_until_waits_for_room_then_gives_up_at_deadline() {
        let (worker, gate, done) = gated_worker(10);
        wait_for_identity(&worker);
        assert!(worker.try_enqueue(1, 10));
        assert!(!worker.enqueue_until(3, 5, Instant::now() + Duration::from_millis(20)));
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            gate.send(()).unwrap();
            gate
        });
        assert!(worker.enqueue_until(4, 5, Instant::now() + LONG));
        let gate = releaser.join().unwrap();
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 1);
        assert_eq!(done.recv_timeout(LONG).unwrap(), 4);
        assert!(worker.wait_idle(Instant::now() + LONG));
        // At shutdown an oversized job waits for the queue to drain, then goes.
        assert!(worker.try_enqueue(6, 10));
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            gate.send(()).unwrap();
            gate
        });
        assert!(worker.enqueue_until(7, 11, Instant::now() + LONG));
        let gate = releaser.join().unwrap();
        gate.send(()).unwrap();
        assert_eq!(done.recv_timeout(LONG).unwrap(), 6);
        assert_eq!(done.recv_timeout(LONG).unwrap(), 7);
        assert!(worker.wait_idle(Instant::now() + LONG));
    }

    #[test]
    fn write_failures_are_counted_and_release_bytes() {
        let worker = DurableWorker::spawn("test", 10, || {
            Ok(Resolved {
                content_id: [0; 32],
                detail: String::new(),
                writer: |job: u32| {
                    if job == 0 {
                        bail!("disk full")
                    }
                    Ok(String::new())
                },
            })
        })
        .unwrap();
        assert!(worker.try_enqueue(0, 10));
        assert!(worker.wait_idle(Instant::now() + LONG));
        assert!(worker.try_enqueue(1, 10));
        assert!(worker.wait_idle(Instant::now() + LONG));
        let stats = worker.stats();
        assert_eq!((stats.written, stats.failed), (1, 1));
    }
}

//! Opt-in shared HTTP-to-visible-token timings. No clocks or allocation when off.
//! A bounded quantum history also covers requests not yet dequeued by a backend.
use serde::Serialize;
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

qwen_llm::env_flag!(
    /// Shared per-request HTTP/render/tokenize/cache/admission/prefill/sample/SSE timing and bounded quantum IDs. Opt-in diagnostic; disabled requests allocate nothing and read no clocks. Emits metadata only, without prompt or token content.
    default_off enabled,
    "QWEN_SERVE_TTFT_TIMING"
);
const MISSING: u64 = u64::MAX;
const HISTORY: usize = 8192;
const MAX_SPANS: usize = 8192;
static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);
static NEXT_ROUND: AtomicU64 = AtomicU64::new(1);
static EPOCH: OnceLock<(Instant, u128)> = OnceLock::new();
fn epoch() -> &'static (Instant, u128) {
    EPOCH.get_or_init(|| {
        (
            Instant::now(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        )
    })
}
fn now() -> u64 {
    epoch().0.elapsed().as_nanos().min((u64::MAX - 1) as u128) as u64
}

#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Stage {
    Accepted,
    HttpReceive,
    BodyReceived,
    Parsed,
    HeadersFlushed,
    Enqueued,
    Dequeued,
    Submitted,
    AdmissionStart,
    Admitted,
    PrefillComplete,
    FirstSampled,
    FirstRawPiece,
    FirstSseFlushed,
    FirstVisibleFlushed,
    Finished,
}
const NAMES: [&str; 16] = [
    "accepted",
    "http_receive",
    "body_received",
    "parsed",
    "headers_flushed",
    "enqueued",
    "dequeued",
    "submitted",
    "admission_start",
    "admitted",
    "prefill_complete",
    "first_sampled",
    "first_raw_piece",
    "first_sse_flushed",
    "first_visible_flushed",
    "finished",
];

#[derive(Clone, Debug, Default)]
pub(crate) struct Trace(Option<Arc<Data>>);
#[derive(Debug)]
struct Data {
    id: u64,
    stages: [AtomicU64; 16],
    details: Mutex<Details>,
    visible_emitted: AtomicU64,
}
#[derive(Debug, Default)]
struct Details {
    response_id: Option<String>,
    family: Option<String>,
    prompt: usize,
    reused: usize,
    spans: Vec<SpanRecord>,
    dropped: usize,
}
#[derive(Debug, Serialize)]
struct SpanRecord {
    name: &'static str,
    round_id: u64,
    start_ns: u64,
    end_ns: u64,
    position: usize,
    rows: usize,
}
thread_local! {
    static CURRENT: RefCell<Trace> = const { RefCell::new(Trace(None)) };
    static ROUND: Cell<u64> = const { Cell::new(0) };
}
pub(crate) fn current() -> Trace {
    CURRENT.with(|x| x.borrow().clone())
}
pub(crate) fn mark(stage: Stage) {
    current().mark(stage);
}
pub(crate) struct Scope(Trace);
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|x| *x.borrow_mut() = self.0.clone());
    }
}

impl Trace {
    pub(crate) fn new() -> Self {
        if enabled() {
            Self::capture()
        } else {
            Self::default()
        }
    }
    #[cfg(test)]
    pub(crate) fn fixture() -> Self {
        Self::capture()
    }
    fn capture() -> Self {
        let trace = Self(Some(Arc::new(Data {
            id: NEXT_REQUEST.fetch_add(1, Ordering::Relaxed),
            stages: std::array::from_fn(|_| AtomicU64::new(MISSING)),
            details: Mutex::new(Details::default()),
            visible_emitted: AtomicU64::new(0),
        })));
        trace.mark(Stage::Accepted);
        trace
    }
    pub(crate) fn reserve_bytes(&self) -> u64 {
        if self.0.is_some() {
            16 * 1024 * 1024
        } else {
            0
        }
    }
    pub(crate) fn id(&self) -> u64 {
        self.0.as_ref().map_or(0, |x| x.id)
    }
    pub(crate) fn enter(&self) -> Scope {
        Scope(CURRENT.with(|x| x.replace(self.clone())))
    }
    pub(crate) fn mark(&self, stage: Stage) {
        if let Some(x) = &self.0 {
            let slot = &x.stages[stage as usize];
            if slot.load(Ordering::Relaxed) == MISSING {
                let _ = slot.compare_exchange(MISSING, now(), Ordering::Relaxed, Ordering::Relaxed);
            }
        }
    }
    pub(crate) fn response_id(&self, id: &str) {
        if let Some(x) = &self.0 {
            x.details
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .response_id = Some(id.to_owned());
        }
    }
    pub(crate) fn extent(&self, family: &str, prompt: usize, reused: usize) {
        if let Some(x) = &self.0 {
            let mut d = x.details.lock().unwrap_or_else(|e| e.into_inner());
            d.family = Some(family.to_owned());
            d.prompt = prompt;
            d.reused = reused;
        }
    }
    pub(crate) fn visible(&self) {
        self.mark(Stage::FirstSseFlushed);
        self.mark(Stage::FirstVisibleFlushed);
        if let Some(x) = &self.0
            && x.visible_emitted.swap(1, Ordering::Relaxed) == 0
        {
            self.emit("first_visible");
        }
    }
    pub(crate) fn finish(&self) {
        self.mark(Stage::Finished);
        self.emit("finished");
    }
    pub(crate) fn snapshot(&self, event: &str) -> Option<Value> {
        let x = self.0.as_ref()?;
        let d = x.details.lock().unwrap_or_else(|e| e.into_inner());
        let response = d.response_id.as_ref()?;
        let stages: serde_json::Map<String, Value> = NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let n = x.stages[i].load(Ordering::Relaxed);
                (
                    name.to_string(),
                    if n == MISSING { Value::Null } else { json!(n) },
                )
            })
            .collect();
        let start = x.stages[Stage::Accepted as usize].load(Ordering::Relaxed);
        let visible = x.stages[Stage::FirstVisibleFlushed as usize].load(Ordering::Relaxed);
        let cut = if visible == MISSING { now() } else { visible };
        let h = history().lock().unwrap_or_else(|e| e.into_inner());
        let rounds:Vec<_>=h.iter().filter(|r| r.start<=cut && r.end.unwrap_or(cut)>=start).map(|r| json!({"round_id":r.id,"kind":r.kind,"request_ids":r.requests,"position":r.position,"rows":r.rows,"start_ns":r.start,"end_ns":r.end.map(|e|e.min(cut)).unwrap_or(cut),"clipped":r.end.is_none_or(|e|e>cut)})).collect();
        Some(
            json!({"version":1,"event":event,"request_id":x.id,"response_id":response,"family":d.family,"prompt_tokens":d.prompt,"reused_tokens":d.reused,"new_rows":d.prompt.saturating_sub(d.reused),"epoch_unix_ns":epoch().1.to_string(),"stages_ns":stages,"spans":d.spans,"spans_dropped":d.dropped,"rounds":rounds,"history_first_round":h.front().map(|r|r.id)}),
        )
    }
    fn emit(&self, event: &str) {
        if let Some(v) = self.snapshot(event) {
            tracing::info!(target:"qwen_diag","serve ttft timing: {v}");
        }
    }
}

pub(crate) struct Span {
    trace: Trace,
    name: &'static str,
    start: u64,
    round: u64,
    position: usize,
    rows: usize,
}
impl Span {
    pub(crate) fn current(name: &'static str, position: usize, rows: usize) -> Self {
        let trace = current();
        let start = if trace.0.is_some() { now() } else { 0 };
        Self {
            trace,
            name,
            start,
            round: ROUND.get(),
            position,
            rows,
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(x) = &self.trace.0 {
            let end = now();
            let mut d = x.details.lock().unwrap_or_else(|e| e.into_inner());
            if d.spans.len() < MAX_SPANS {
                d.spans.push(SpanRecord {
                    name: self.name,
                    round_id: self.round,
                    start_ns: self.start,
                    end_ns: end,
                    position: self.position,
                    rows: self.rows,
                });
            } else {
                d.dropped += 1;
            }
        }
    }
}

#[derive(Clone)]
struct Quantum {
    id: u64,
    kind: &'static str,
    requests: Vec<u64>,
    start: u64,
    end: Option<u64>,
    position: usize,
    rows: usize,
}
static ROUNDS: OnceLock<Mutex<VecDeque<Quantum>>> = OnceLock::new();
fn history() -> &'static Mutex<VecDeque<Quantum>> {
    ROUNDS.get_or_init(|| Mutex::new(VecDeque::with_capacity(HISTORY)))
}
pub(crate) struct Round {
    id: u64,
    previous: u64,
}
impl Round {
    pub(crate) fn new(
        kind: &'static str,
        traces: impl Iterator<Item = Trace>,
        position: usize,
        rows: usize,
    ) -> Self {
        if !enabled() {
            return Self { id: 0, previous: 0 };
        }
        let requests = traces.map(|t| t.id()).filter(|&id| id != 0).collect();
        Self::record(kind, requests, position, rows)
    }
    fn record(kind: &'static str, requests: Vec<u64>, position: usize, rows: usize) -> Self {
        let id = NEXT_ROUND.fetch_add(1, Ordering::Relaxed);
        let start = now();
        let mut h = history().lock().unwrap_or_else(|e| e.into_inner());
        if h.len() == HISTORY {
            h.pop_front();
        }
        h.push_back(Quantum {
            id,
            kind,
            requests,
            start,
            end: None,
            position,
            rows,
        });
        drop(h);
        let previous = ROUND.replace(id);
        Self { id, previous }
    }
}
impl Drop for Round {
    fn drop(&mut self) {
        if self.id != 0 {
            let end = now();
            if let Some(r) = history()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter_mut()
                .rev()
                .find(|r| r.id == self.id)
            {
                r.end = Some(end);
            }
            ROUND.set(self.previous);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_trace_has_no_state_or_spans() {
        let t = Trace::default();
        let _s = t.enter();
        mark(Stage::Parsed);
        {
            let _p = Span::current("render", 0, 0);
        }
        assert!(t.0.is_none());
        assert!(t.snapshot("finished").is_none());
        assert_eq!(t.id(), 0);
    }
    #[test]
    fn cross_thread_stages_and_first_visible_survive_scope_restore() {
        let t = Trace::capture();
        t.response_id("fixture");
        let _scope = t.enter();
        {
            let _p = Span::current("render", 0, 0);
        }
        let c = t.clone();
        std::thread::spawn(move || {
            c.mark(Stage::FirstSampled);
            c.mark(Stage::FirstVisibleFlushed);
        })
        .join()
        .unwrap();
        t.mark(Stage::FirstVisibleFlushed);
        let v = t.snapshot("first_visible").unwrap();
        assert_eq!(v["response_id"], "fixture");
        assert_eq!(v["spans"].as_array().unwrap().len(), 1);
        let a = v["stages_ns"]["accepted"].as_u64().unwrap();
        let b = v["stages_ns"]["first_sampled"].as_u64().unwrap();
        let c = v["stages_ns"]["first_visible_flushed"].as_u64().unwrap();
        assert!(a <= b && b <= c);
    }
    #[test]
    fn round_ids_join_prefill_and_clip_at_first_visible() {
        let t = Trace::capture();
        t.response_id("quantum-fixture");
        let _s = t.enter();
        let r = Round::record("prefill", vec![t.id()], 80, 256);
        {
            let _p = Span::current("prefill_slice", 80, 256);
        }
        t.mark(Stage::FirstVisibleFlushed);
        let v = t.snapshot("first_visible").unwrap();
        let q = v["rounds"]
            .as_array()
            .unwrap()
            .iter()
            .find(|q| q["round_id"] == r.id)
            .unwrap();
        assert_eq!(q["request_ids"], json!([t.id()]));
        assert_eq!(q["clipped"], true);
        assert_eq!(v["spans"][0]["round_id"], r.id);
        drop(r);
    }
    #[test]
    fn enabled_capture_cost_and_snapshot_are_bounded() {
        let started = Instant::now();
        for _ in 0..100 {
            let t = Trace::capture();
            t.response_id("cost-fixture");
            t.extent("fixture", 1000, 744);
            let _scope = t.enter();
            for _ in 0..20 {
                let _s = Span::current("fixture", 744, 256);
            }
            t.mark(Stage::FirstSampled);
            t.mark(Stage::FirstVisibleFlushed);
            let v = t.snapshot("first_visible").unwrap();
            assert_eq!(v["spans"].as_array().unwrap().len(), 20);
            assert_eq!(v["spans_dropped"], 0);
        }
        eprintln!(
            "TTFT_CPU_CAPTURE us_per_request={:.3} phases=20 snapshot=true",
            started.elapsed().as_secs_f64() * 1e6 / 100.0
        );
    }
}

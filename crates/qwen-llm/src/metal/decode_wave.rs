//! Diagnostic adaptation of scripts/research/fn-decode-roofline/capture.patch.
//! Frozen dispatches remove causality and bound occupancy; never execute inference.
//! Original command-buffer cuts retain their own snapshots, including CPU uploads.
use super::*;
pub(super) fn active() -> bool {
    CAPTURE_COUNT.load(Ordering::Relaxed) != 0
}
pub(super) fn capture(enc: &KernelEncoder, grid: MTLSize, threads: MTLSize) -> bool {
    if !active() {
        return false;
    }
    CAPTURE.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut() {
            c.steps.push(CapturedStep {
                kernel: enc
                    .bound_pipeline_name
                    .borrow()
                    .as_deref()
                    .unwrap()
                    .to_string(),
                grid,
                threads,
                state: enc.replay.borrow().as_ref().unwrap().as_ref().clone(),
                concurrent: enc.concurrent,
                encoder: c.encoders,
            });
        }
    });
    true
}
fn restore_ranges(saved: &[SavedRange]) {
    for s in saved {
        unsafe {
            std::ptr::copy_nonoverlapping(
                s.bytes.as_ptr(),
                s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                s.bytes.len(),
            );
        }
    }
}
#[derive(Clone, serde::Serialize)]
pub(super) struct CaptureTensor {
    pub(super) dtype: String,
    pub(super) shape: Vec<u64>,
    pub(super) bytes: u64,
    pub(super) writable: bool,
}
struct CapturedStep {
    kernel: String,
    grid: MTLSize,
    threads: MTLSize,
    state: Replay,
    concurrent: bool,
    encoder: usize,
}
struct SavedRange {
    buffer: Buffer,
    offset: usize,
    bytes: Vec<u8>,
}
#[derive(Default)]
struct CaptureState {
    steps: Vec<CapturedStep>,
    saved: Vec<SavedRange>,
    error: Option<String>,
    encoders: usize,
    cuts: Vec<(usize, usize)>,
    commands: Vec<usize>,
    retained_commands: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>>,
}
static CAPTURE_COUNT: AtomicUsize = AtomicUsize::new(0);
thread_local! {static CAPTURE:RefCell<Option<CaptureState>>=const{RefCell::new(None)};}
pub struct DecodeCapture {
    _thread: PhantomData<Rc<()>>,
}
pub struct DecodeGraph {
    state: CaptureState,
}
pub fn decode_capture_begin() -> DecodeCapture {
    assert!(
        CAPTURE_COUNT
            .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok(),
        "nested capture"
    );
    CAPTURE.with(|c| {
        assert!(c.borrow().is_none());
        *c.borrow_mut() = Some(CaptureState::default());
    });
    DecodeCapture {
        _thread: PhantomData,
    }
}
impl Drop for DecodeCapture {
    fn drop(&mut self) {
        if CAPTURE.with(|c| c.borrow_mut().take()).is_some() {
            CAPTURE_COUNT.fetch_sub(1, Ordering::Relaxed);
        }
    }
}
pub fn decode_capture_finish(guard: DecodeCapture) -> DecodeGraph {
    let state = CAPTURE.with(|c| c.borrow_mut().take()).unwrap();
    CAPTURE_COUNT.fetch_sub(1, Ordering::Relaxed);
    drop(guard);
    DecodeGraph { state }
}
pub(crate) fn capture_encoder_end(
    command: &Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
) {
    let command_id = Retained::as_ptr(command) as *const () as usize;
    if CAPTURE_COUNT.load(Ordering::Relaxed) == 0 {
        return;
    }
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        let Some(c) = c.as_mut() else {
            return;
        };
        c.encoders += 1;
        let first = c.cuts.last().map(|v| v.0).unwrap_or(0);
        use std::collections::BTreeMap;
        let mut read_only = std::collections::BTreeSet::new();
        for step in &c.steps[first..] {
            for (i, m) in &step.state.metadata {
                if !m.writable {
                    if let Some((_, Slot::Buffer(b, _))) =
                        step.state.slots.iter().find(|(j, _)| j == i)
                    {
                        read_only.insert(Retained::as_ptr(b) as *const () as usize);
                    }
                }
            }
        }
        let mut ranges = BTreeMap::<usize, (Buffer, Vec<(usize, usize)>)>::new();
        for step in &c.steps[first..] {
            for (i, slot) in &step.state.slots {
                if let Slot::Buffer(b, off) = slot {
                    let id = Retained::as_ptr(b) as *const () as usize;
                    if read_only.contains(&id) {
                        continue;
                    }
                    let len = step
                        .state
                        .metadata
                        .get(i)
                        .map(|m| m.bytes as usize)
                        .unwrap_or_else(|| b.length() - off);
                    if len > 1024 * 1024 * 1024 {
                        c.error = Some(format!("unclassified large writable buffer {len}"));
                        return;
                    }
                    assert!(*off + len <= b.length());
                    ranges
                        .entry(id)
                        .or_insert_with(|| (b.clone(), Vec::new()))
                        .1
                        .push((*off, *off + len));
                }
            }
        }
        let mut merged = Vec::new();
        let mut total = 0usize;
        for (_, (b, mut v)) in ranges {
            v.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (lo, hi) in v {
                if let Some(last) = spans.last_mut() {
                    if lo <= last.1 {
                        last.1 = last.1.max(hi);
                        continue;
                    }
                }
                spans.push((lo, hi));
            }
            for (lo, hi) in spans {
                total += hi - lo;
                merged.push((b.clone(), lo, hi));
            }
        }
        if total > 2 * 1024 * 1024 * 1024 {
            c.error = Some(format!("snapshot budget {total}"));
            return;
        }
        for (b, lo, hi) in merged {
            let bytes = unsafe {
                std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(lo), hi - lo)
            }
            .to_vec();
            c.saved.push(SavedRange {
                buffer: b,
                offset: lo,
                bytes,
            });
        }
        let cut = (c.steps.len(), c.saved.len());
        if c.commands.last() == Some(&command_id) {
            *c.cuts.last_mut().unwrap() = cut;
        } else {
            c.cuts.push(cut);
            c.commands.push(command_id);
            c.retained_commands.push(command.clone());
        }
    });
}
impl DecodeGraph {
    pub fn metadata(&self) -> serde_json::Value {
        let rows:Vec<_>=self.state.steps.iter().map(|s|{
            let bindings:Vec<_>=s.state.slots.iter().map(|(i,v)| match v {
                Slot::Buffer(b,off)=>serde_json::json!({"slot":i,"buffer":Retained::as_ptr(b)as*const()as usize,"offset":off,"buffer_bytes":b.length(),"tensor":s.state.metadata.get(i)}),
                Slot::Bytes(b)=>serde_json::json!({"slot":i,"u32":b.chunks_exact(4).map(|q|u32::from_le_bytes(q.try_into().unwrap())).collect::<Vec<_>>()}),
            }).collect();
            serde_json::json!({"kernel":s.kernel,"grid":[s.grid.width,s.grid.height,s.grid.depth],"threads":[s.threads.width,s.threads.height,s.threads.depth],"concurrent":s.concurrent,"bindings":bindings})
        }).collect();
        serde_json::json!({"steps":rows,"encoders":self.state.encoders,"commands":self.state.cuts.len(),"snapshot_bytes":self.state.saved.iter().map(|s|s.bytes.len()).sum::<usize>(),"snapshot_error":self.state.error})
    }
    pub fn outputs(&self, batch: usize) -> Vec<Vec<f32>> {
        let s = self.state.steps.last().unwrap();
        assert_eq!(s.kernel, "kernel_deepseek_v41_scatter_columns");
        (2..2 + batch)
            .map(|slot| {
                let (_, Slot::Buffer(b, off)) =
                    s.state.slots.iter().find(|(i, _)| *i == slot).unwrap()
                else {
                    panic!("logit buffer")
                };
                let m = &s.state.metadata[&slot];
                assert_eq!(m.dtype, "F32");
                unsafe {
                    std::slice::from_raw_parts(
                        b.contents().as_ptr().cast::<u8>().add(*off).cast::<f32>(),
                        m.bytes as usize / 4,
                    )
                }
                .to_vec()
            })
            .collect()
    }
    pub fn segments(&self) -> usize {
        self.state.cuts.len()
    }
    fn restore_segment(&self, segment: usize) {
        assert!(self.state.error.is_none(), "{:?}", self.state.error);
        let lo = if segment == 0 {
            0
        } else {
            self.state.cuts[segment - 1].1
        };
        let hi = self.state.cuts[segment].1;
        restore_ranges(&self.state.saved[lo..hi]);
    }
    fn step_range(&self, segment: usize) -> std::ops::Range<usize> {
        let lo = if segment == 0 {
            0
        } else {
            self.state.cuts[segment - 1].0
        };
        lo..self.state.cuts[segment].0
    }
    pub fn replay_segment(
        &self,
        ctx: &MetalContext,
        segment: usize,
        access: Option<&AccessMap>,
    ) -> f64 {
        self.restore_segment(segment);
        let cmd = ctx.queue.commandBuffer().unwrap();
        if access.is_some() {
            let enc = KernelEncoder::begin_concurrent(&cmd);
            self.encode_segment(&enc, segment, access);
            enc.end();
        } else {
            let mut encoder = None;
            let mut previous = None;
            for step in &self.state.steps[self.step_range(segment)] {
                if previous != Some(step.encoder) {
                    if let Some(enc) = encoder.take() {
                        KernelEncoder::end(enc);
                    }
                    encoder = Some(KernelEncoder::begin(&cmd));
                    previous = Some(step.encoder);
                }
                let enc = encoder.as_ref().unwrap();
                step.state.apply(&enc.raw.borrow());
                enc.raw
                    .borrow()
                    .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            }
            if let Some(enc) = encoder {
                enc.end();
            }
        }
        cmd.commit();
        wait_completed(&cmd).unwrap();
        assert!(cmd.error().is_none(), "{:?}", cmd.error());
        (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.
    }
    pub fn kernels(&self) -> Vec<String> {
        self.state
            .steps
            .iter()
            .map(|s| s.kernel.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}
// Research-only dependency/occupancy predictors. Never used by a forward pass.
type AccessMap = std::collections::BTreeMap<String, Vec<(usize, bool)>>;
type Span = (usize, usize, usize, bool);
fn step_spans(step: &CapturedStep, access: &AccessMap) -> Vec<Span> {
    access[&step.kernel]
        .iter()
        .filter_map(|&(slot, write)| {
            let (_, Slot::Buffer(b, off)) = step.state.slots.iter().find(|(i, _)| *i == slot)?
            else {
                return None;
            };
            let len = step
                .state
                .metadata
                .get(&slot)
                .map(|m| m.bytes as usize)
                .unwrap_or(b.length() - off);
            Some((
                Retained::as_ptr(b) as *const () as usize,
                *off,
                off + len,
                write,
            ))
        })
        .collect()
}
fn conflict(a: &Span, b: &Span) -> bool {
    a.0 == b.0 && a.1 < b.2 && b.1 < a.2 && (a.3 || b.3)
}
impl DecodeGraph {
    /// Conservative real dependency schedule; all conflicting accesses drain.
    fn encode_segment(&self, enc: &KernelEncoder, segment: usize, access: Option<&AccessMap>) {
        let mut pending = Vec::<Span>::new();
        for step in &self.state.steps[self.step_range(segment)] {
            if let Some(access) = access {
                let spans = step_spans(step, access);
                if spans.iter().any(|s| pending.iter().any(|p| conflict(s, p))) {
                    enc.raw
                        .borrow()
                        .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                    pending.clear();
                }
                pending.extend(spans);
            }
            step.state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
        }
    }

    /// Give every dispatch its recorded inputs and private writable buffers.
    /// Removing dependencies is deliberately noncausal: this is a predictor,
    /// never a candidate forward implementation.
    pub fn freeze(&self, ctx: &MetalContext, access: &AccessMap) -> FrozenGraph {
        let mut steps = Vec::new();
        let mut saved = Vec::new();
        let mut expected = Vec::new();
        let mut segment = 0;
        for (index, step) in self.state.steps.iter().enumerate() {
            if let Some(original_segment) = std::iter::once(0)
                .chain(self.state.cuts.iter().map(|v| v.0))
                .take(self.state.cuts.len())
                .position(|lo| lo == index)
            {
                self.restore_segment(original_segment);
            }
            let mut ranges = std::collections::BTreeMap::<usize, (Buffer, usize, usize)>::new();
            for &(slot, _) in &access[&step.kernel] {
                let Some((_, Slot::Buffer(b, off))) =
                    step.state.slots.iter().find(|(i, _)| *i == slot)
                else {
                    continue;
                };
                if step.state.metadata.get(&slot).is_some_and(|m| !m.writable) {
                    continue;
                }
                let len = step
                    .state
                    .metadata
                    .get(&slot)
                    .map(|m| m.bytes as usize)
                    .unwrap_or(b.length() - off);
                let id = Retained::as_ptr(b) as *const () as usize;
                ranges
                    .entry(id)
                    .and_modify(|r| {
                        r.1 = r.1.min(*off);
                        r.2 = r.2.max(off + len);
                    })
                    .or_insert((b.clone(), *off, off + len));
            }
            let mut mapping = std::collections::BTreeMap::new();
            for (id, (b, lo, hi)) in ranges {
                assert!(hi - lo <= 1024 * 1024 * 1024);
                let bytes = unsafe {
                    std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(lo), hi - lo)
                }
                .to_vec();
                let clone = ctx.buffer_from(&bytes).unwrap();
                saved.push(SavedRange {
                    buffer: clone.clone(),
                    offset: 0,
                    bytes,
                });
                mapping.insert(id, (clone, lo));
            }
            let mut state = step.state.clone();
            state
                .slots
                .retain(|(slot, _)| access[&step.kernel].iter().any(|(used, _)| used == slot));
            for (_, slot) in &mut state.slots {
                if let Slot::Buffer(b, off) = slot {
                    if let Some((clone, lo)) =
                        mapping.get(&(Retained::as_ptr(b) as *const () as usize))
                    {
                        *b = clone.clone();
                        *off -= lo;
                    }
                }
            }
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            step.state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            enc.end();
            cmd.commit();
            wait_completed(&cmd).unwrap();
            for &(slot, write) in &access[&step.kernel] {
                if !write {
                    continue;
                }
                let Some((_, Slot::Buffer(b, off))) =
                    step.state.slots.iter().find(|(i, _)| *i == slot)
                else {
                    continue;
                };
                let len = step
                    .state
                    .metadata
                    .get(&slot)
                    .map(|m| m.bytes as usize)
                    .unwrap_or(b.length() - off);
                let (clone, lo) = &mapping[&(Retained::as_ptr(b) as *const () as usize)];
                let bytes = unsafe {
                    std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(*off), len)
                }
                .to_vec();
                expected.push(SavedRange {
                    buffer: clone.clone(),
                    offset: off - lo,
                    bytes,
                });
            }
            steps.push((state, step.grid, step.threads, segment));
            if step.kernel.contains("hc_post") {
                segment += 1;
            }
        }
        FrozenGraph {
            steps,
            saved,
            expected,
        }
    }
}
pub struct FrozenGraph {
    steps: Vec<(Replay, MTLSize, MTLSize, usize)>,
    saved: Vec<SavedRange>,
    expected: Vec<SavedRange>,
}
impl FrozenGraph {
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({"private_bytes":self.saved.iter().map(|r|r.bytes.len()).sum::<usize>(),
            "checked_output_bytes":self.expected.iter().map(|r|r.bytes.len()).sum::<usize>(),
            "segments":self.steps.last().unwrap().3+1})
    }
    pub fn restore(&self) {
        for s in &self.saved {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    s.bytes.as_ptr(),
                    s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                    s.bytes.len(),
                );
            }
        }
    }
    pub fn check(&self) -> bool {
        self.expected.iter().all(|s| unsafe {
            std::slice::from_raw_parts(
                s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                s.bytes.len(),
            ) == s.bytes
        })
    }
    pub fn replay(&self, enc: &KernelEncoder, per_segment_barrier: bool) {
        let mut segment = 0;
        for (state, grid, threads, next) in &self.steps {
            if per_segment_barrier && segment != *next {
                enc.raw
                    .borrow()
                    .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            segment = *next;
            state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(*grid, *threads);
        }
    }
}

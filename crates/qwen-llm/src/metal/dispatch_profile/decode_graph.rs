//! Research-only captured whole-round counterfactuals, adapted from the FN predictor.
use super::*;
// TEMPORARY DIAGNOSTIC, never enabled in ordinary execution.
pub(super) struct CapturedStep {
    pub(super) kernel: String,
    pub(super) grid: MTLSize,
    pub(super) threads: MTLSize,
    pub(super) state: Replay,
    pub(super) concurrent: bool,
}
struct SavedRange {
    buffer: Buffer,
    offset: usize,
    bytes: Vec<u8>,
}
#[derive(Default)]
pub(super) struct CaptureState {
    pub(super) steps: Vec<CapturedStep>,
    saved: Vec<SavedRange>,
    error: Option<String>,
    encoders: usize,
}
pub(super) static CAPTURE_COUNT: AtomicUsize = AtomicUsize::new(0);
thread_local! {pub(super) static CAPTURE:RefCell<Option<CaptureState>>=const{RefCell::new(None)};}
pub struct DecodeCapture;
pub struct DecodeGraph {
    state: CaptureState,
}
pub fn decode_capture_begin() -> DecodeCapture {
    assert!(
        !super::decode_wave::active(),
        "capture protocols cannot be nested"
    );
    assert_eq!(CAPTURE_COUNT.fetch_add(1, Ordering::Relaxed), 0);
    CAPTURE.with(|c| {
        assert!(c.borrow().is_none());
        *c.borrow_mut() = Some(CaptureState::default());
    });
    DecodeCapture
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
pub(crate) fn capture_encoder_end() {
    if CAPTURE_COUNT.load(Ordering::Relaxed) == 0 {
        return;
    }
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        let Some(c) = c.as_mut() else {
            return;
        };
        c.encoders += 1;
        if c.encoders != 1 {
            c.error = Some("capture requires one encoder".into());
            return;
        }
        use std::collections::BTreeMap;
        let mut read_only = std::collections::BTreeSet::new();
        for step in &c.steps {
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
        for step in &c.steps {
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
        if total > 8 * 1024 * 1024 * 1024 {
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
        serde_json::json!({"steps":rows,"encoders":self.state.encoders,"snapshot_bytes":self.state.saved.iter().map(|s|s.bytes.len()).sum::<usize>(),"snapshot_error":self.state.error})
    }
    pub fn restore(&self) {
        assert!(self.state.error.is_none(), "{:?}", self.state.error);
        for s in &self.state.saved {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    s.bytes.as_ptr(),
                    s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                    s.bytes.len(),
                );
            }
        }
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
    pub fn replay(&self, enc: &KernelEncoder, kernel: Option<&str>, repeats: usize) {
        for _ in 0..repeats {
            for step in &self.state.steps {
                if kernel.is_some_and(|k| k != step.kernel) {
                    continue;
                }
                assert!(!step.concurrent);
                step.state.apply(&enc.raw.borrow());
                enc.raw
                    .borrow()
                    .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            }
        }
    }
    pub fn replay_one(&self, enc: &KernelEncoder, kernel: &str, repeats: usize) {
        let s = self
            .state
            .steps
            .iter()
            .find(|s| s.kernel == kernel)
            .unwrap();
        for _ in 0..repeats {
            s.state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(s.grid, s.threads);
        }
    }
    pub fn output(&self) -> Vec<f32> {
        let s = self.state.steps.last().unwrap();
        let (_, Slot::Buffer(b, off)) = s.state.slots.iter().find(|(i, _)| *i == 3).unwrap() else {
            panic!("output slot")
        };
        let m = s.state.metadata.get(&3).unwrap();
        assert_eq!(m.dtype, "F32");
        unsafe {
            std::slice::from_raw_parts(
                b.contents().as_ptr().cast::<u8>().add(*off).cast::<f32>(),
                m.bytes as usize / 4,
            )
        }
        .to_vec()
    }
    pub fn noop(
        &self,
        enc: &KernelEncoder,
        p: &Pipeline,
        out: &MetalTensor,
        kernel: Option<&str>,
        matched_grid: bool,
    ) {
        enc.set_pipeline(p);
        enc.set_tensor(0, out);
        enc.set_threadgroup_memory(0, 0);
        for s in &self.state.steps {
            if kernel.is_some_and(|k| k != s.kernel) {
                continue;
            }
            let g = if matched_grid {
                s.grid
            } else {
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                }
            };
            enc.set_bytes(1, &[g.width as u32, g.height as u32]);
            enc.dispatch(
                g,
                if matched_grid {
                    s.threads
                } else {
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    }
                },
            );
        }
    }
}

struct SubstituteStep {
    source: Buffer,
    destinations: Vec<(Buffer, usize, usize)>,
    args: [u32; 33],
    words: usize,
}
pub struct DecodeSubstitutions<'a> {
    graph: &'a DecodeGraph,
    steps: Vec<SubstituteStep>,
}
impl DecodeGraph {
    pub fn prepare_substitutions(
        &self,
        ctx: &MetalContext,
        writes: &std::collections::BTreeMap<String, Vec<usize>>,
    ) -> DecodeSubstitutions<'_> {
        self.restore();
        let mut substitutions = Vec::new();
        for step in &self.state.steps {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            step.state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            enc.end();
            cmd.commit();
            wait_completed(&cmd).unwrap();
            let mut destinations = Vec::new();
            let mut bytes = Vec::new();
            let mut args = [0u32; 33];
            for &slot in &writes[&step.kernel] {
                let Some((_, Slot::Buffer(b, offset))) =
                    step.state.slots.iter().find(|(i, _)| *i == slot)
                else {
                    continue;
                };
                let meta = step.state.metadata.get(&slot);
                if meta.is_some_and(|m| !m.writable) {
                    continue;
                }
                let len = meta
                    .map(|m| m.bytes as usize)
                    .unwrap_or_else(|| b.length() - offset);
                assert_eq!(len % 4, 0, "{} slot{} bytes{}", step.kernel, slot, len);
                assert_eq!(offset % 4, 0);
                assert!(len <= 1024 * 1024 * 1024);
                assert!(destinations.len() < 16);
                let i = destinations.len();
                args[1 + i] = (len / 4) as u32;
                args[17 + i] = (bytes.len() / 4) as u32;
                bytes.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(*offset), len)
                });
                destinations.push((b.clone(), *offset, len));
            }
            args[0] = destinations.len() as u32;
            let words = bytes.len() / 4;
            if bytes.is_empty() {
                bytes.resize(4, 0);
            }
            let source = ctx.buffer_from(&bytes).unwrap();
            substitutions.push(SubstituteStep {
                source,
                destinations,
                args,
                words,
            });
        }
        DecodeSubstitutions {
            graph: self,
            steps: substitutions,
        }
    }
}
impl DecodeSubstitutions<'_> {
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({"bytes":self.steps.iter().map(|s|s.words*4).sum::<usize>(),"outputs":self.steps.iter().map(|s|s.destinations.len()).sum::<usize>()})
    }
    pub fn replay(
        &self,
        enc: &KernelEncoder,
        copy: &Pipeline,
        groups: &std::collections::BTreeMap<String, usize>,
        mask: u64,
    ) {
        for (step, sub) in self.graph.state.steps.iter().zip(&self.steps) {
            if mask & (1u64 << groups[&step.kernel]) != 0 {
                step.state.apply(&enc.raw.borrow());
                enc.raw
                    .borrow()
                    .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            } else {
                enc.set_pipeline(copy);
                enc.set_bytes_slice(0, &sub.args);
                enc.set_buffer(1, &sub.source, 0);
                enc.set_threadgroup_memory(0, 0);
                for i in 0..16 {
                    if let Some((b, offset, _)) = sub.destinations.get(i) {
                        enc.set_buffer(2 + i, b, *offset as u64);
                    } else {
                        enc.set_buffer(2 + i, &sub.source, 0);
                    }
                }
                enc.dispatch(
                    MTLSize {
                        width: sub.words.max(1).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
    }
}

impl SubstituteStep {
    fn encode_copy(&self, enc: &KernelEncoder, copy: &Pipeline) {
        enc.set_pipeline(copy);
        enc.set_bytes_slice(0, &self.args);
        enc.set_buffer(1, &self.source, 0);
        enc.set_threadgroup_memory(0, 0);
        for i in 0..16 {
            if let Some((b, offset, _)) = self.destinations.get(i) {
                enc.set_buffer(2 + i, b, *offset as u64);
            } else {
                enc.set_buffer(2 + i, &self.source, 0);
            }
        }
        enc.dispatch(
            MTLSize {
                width: self.words.max(1).div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }
}
impl DecodeSubstitutions<'_> {
    pub fn replay_copy_mask(
        &self,
        enc: &KernelEncoder,
        copy: &Pipeline,
        noop: &Pipeline,
        counter: &MetalTensor,
        groups: &std::collections::BTreeMap<String, usize>,
        mask: u64,
    ) {
        for (step, sub) in self.graph.state.steps.iter().zip(&self.steps) {
            if mask & (1u64 << groups[&step.kernel]) != 0 {
                sub.encode_copy(enc, copy);
            } else {
                enc.set_pipeline(noop);
                enc.set_tensor(0, counter);
                enc.set_bytes(1, &[1u32, 1u32]);
                enc.set_threadgroup_memory(0, 0);
                enc.dispatch(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
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
    pub fn replay_dag(&self, enc: &KernelEncoder, access: &AccessMap) -> usize {
        let mut pending = Vec::<Span>::new();
        let mut barriers = 0;
        for step in &self.state.steps {
            let spans = step_spans(step, access);
            if spans.iter().any(|s| pending.iter().any(|p| conflict(s, p))) {
                enc.raw
                    .borrow()
                    .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                pending.clear();
                barriers += 1;
            }
            step.state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            pending.extend(spans);
        }
        barriers
    }

    /// Give every dispatch its recorded inputs and private writable buffers.
    /// Removing dependencies is deliberately noncausal: this is a predictor,
    /// never a candidate forward implementation.
    pub fn freeze(&self, ctx: &MetalContext, access: &AccessMap) -> FrozenGraph {
        self.restore();
        let mut steps = Vec::new();
        let mut saved = Vec::new();
        let mut expected = Vec::new();
        let mut expected_steps = Vec::new();
        let mut segment = 0;
        for step in &self.state.steps {
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
                expected_steps.push(steps.len());
                expected.push(SavedRange {
                    buffer: clone.clone(),
                    offset: off - lo,
                    bytes,
                });
            }
            steps.push((state, step.grid, step.threads, segment));
            if step.kernel == "kernel_deepseek_v4_hc_post_batch" {
                segment += 1;
            }
        }
        FrozenGraph {
            steps,
            saved,
            expected,
            expected_steps,
        }
    }
}
pub struct FrozenGraph {
    steps: Vec<(Replay, MTLSize, MTLSize, usize)>,
    saved: Vec<SavedRange>,
    expected: Vec<SavedRange>,
    expected_steps: Vec<usize>,
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

pub(super) fn active() -> bool {
    CAPTURE_COUNT.load(Ordering::Relaxed) != 0
}
impl DecodeGraph {
    pub fn mutable_hash(&self) -> String {
        let mut hash = blake3::Hasher::new();
        for r in &self.state.saved {
            let b = unsafe {
                std::slice::from_raw_parts(
                    r.buffer.contents().as_ptr().cast::<u8>().add(r.offset),
                    r.bytes.len(),
                )
            };
            hash.update(b);
        }
        hash.finalize().to_hex().to_string()
    }
}

impl DecodeSubstitutions<'_> {
    pub fn replay_hold_groups(
        &self,
        enc: &KernelEncoder,
        copy: &Pipeline,
        groups: &[usize],
        hold: u64,
    ) {
        assert_eq!(groups.len(), self.steps.len());
        for ((step, sub), group) in self.graph.state.steps.iter().zip(&self.steps).zip(groups) {
            if hold & (1u64 << group) != 0 {
                sub.encode_copy(enc, copy);
            } else {
                step.state.apply(&enc.raw.borrow());
                enc.raw
                    .borrow()
                    .dispatchThreadgroups_threadsPerThreadgroup(step.grid, step.threads);
            }
        }
    }
    pub fn group_bytes(&self, groups: &[usize], group: usize) -> usize {
        self.steps
            .iter()
            .zip(groups)
            .filter(|(_, g)| **g == group)
            .map(|(s, _)| s.words * 4)
            .sum()
    }
}
impl FrozenGraph {
    pub fn replay_skip_groups(
        &self,
        enc: &KernelEncoder,
        groups: &[usize],
        skip: u64,
        concurrent: bool,
    ) {
        assert_eq!(groups.len(), self.steps.len());
        for ((state, grid, threads, _), group) in self.steps.iter().zip(groups) {
            if skip & (1u64 << group) != 0 {
                continue;
            }
            state.apply(&enc.raw.borrow());
            enc.raw
                .borrow()
                .dispatchThreadgroups_threadsPerThreadgroup(*grid, *threads);
        }
        let _ = concurrent;
    }
}

/// Reorder ready dispatches while retaining every captured RAW/WAR/WAW edge.
/// Unlike private-input replay, this uses the original buffers and real causes.
pub struct DecodeWavefront<'a> {
    graph: &'a DecodeGraph,
    waves: Vec<Vec<usize>>,
}
impl DecodeGraph {
    pub fn wavefront(&self, access: &AccessMap) -> DecodeWavefront<'_> {
        self.wavefront_groups(access, &vec![0; self.state.steps.len()], u64::MAX)
    }
    pub fn wavefront_groups(
        &self,
        access: &AccessMap,
        groups: &[usize],
        overlap: u64,
    ) -> DecodeWavefront<'_> {
        let spans: Vec<_> = self
            .state
            .steps
            .iter()
            .map(|s| step_spans(s, access))
            .collect();
        let mut levels = vec![0usize; spans.len()];
        let mut waves = Vec::<Vec<usize>>::new();
        let (mut floor, mut latest) = (0, 0);
        for i in 0..spans.len() {
            levels[i] = floor;
            for j in 0..i {
                if spans[i]
                    .iter()
                    .any(|a| spans[j].iter().any(|b| conflict(a, b)))
                {
                    levels[i] = levels[i].max(levels[j] + 1);
                }
            }
            if overlap & (1u64 << groups[i]) == 0 {
                levels[i] = levels[i].max(latest + 1);
                floor = levels[i] + 1;
            }
            latest = latest.max(levels[i]);
            if waves.len() <= levels[i] {
                waves.resize_with(levels[i] + 1, Vec::new);
            }
            waves[levels[i]].push(i);
        }
        DecodeWavefront { graph: self, waves }
    }
}
impl DecodeWavefront<'_> {
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({"waves":self.waves.len(),"max_width":self.waves.iter().map(Vec::len).max(),"widths":self.waves.iter().map(Vec::len).collect::<Vec<_>>()})
    }
    pub fn replay(&self, enc: &KernelEncoder) {
        for (level, wave) in self.waves.iter().enumerate() {
            if level > 0 {
                enc.raw
                    .borrow()
                    .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            for &i in wave {
                let s = &self.graph.state.steps[i];
                s.state.apply(&enc.raw.borrow());
                enc.raw
                    .borrow()
                    .dispatchThreadgroups_threadsPerThreadgroup(s.grid, s.threads);
            }
        }
    }
}

impl DecodeGraph {
    /// Real tensor storage and selected IDs; not hardware DRAM transactions.
    pub fn weight_byte_bounds(&self, groups: &[usize], access: &AccessMap) -> serde_json::Value {
        use std::collections::BTreeSet;
        let mut logical = vec![0u64; 16];
        let mut unique = vec![0u64; 16];
        let mut seen = BTreeSet::new();
        for (s, &g) in self.state.steps.iter().zip(groups) {
            if s.kernel.contains("get_rows") {
                continue;
            }
            for (slot, value) in &s.state.slots {
                if !access[&s.kernel].iter().any(|(i, _)| i == slot) {
                    continue;
                }
                let Slot::Buffer(b, off) = value else {
                    continue;
                };
                let Some(m) = s.state.metadata.get(slot) else {
                    continue;
                };
                if m.writable {
                    continue;
                }
                if (g == 5 || g == 6) && m.shape.len() == 3 && (*slot == 1 || *slot == 6) {
                    let experts = m.shape[2];
                    let (_, Slot::Buffer(ids, base)) =
                        s.state.slots.iter().find(|(i, _)| *i == 3).unwrap()
                    else {
                        panic!("ids")
                    };
                    let n = s.state.metadata[&3].bytes as usize / 4;
                    let values = unsafe {
                        std::slice::from_raw_parts(
                            ids.contents()
                                .as_ptr()
                                .cast::<u8>()
                                .add(*base)
                                .cast::<i32>(),
                            n,
                        )
                    };
                    assert!(values.iter().all(|id| *id >= 0 && (*id as u64) < experts));
                    logical[g] += m.bytes / experts * n as u64;
                    unique[g] +=
                        m.bytes / experts * values.iter().collect::<BTreeSet<_>>().len() as u64;
                } else {
                    logical[g] += m.bytes;
                    if seen.insert((g, Retained::as_ptr(b) as *const () as usize, *off, m.bytes)) {
                        unique[g] += m.bytes;
                    }
                }
            }
        }
        serde_json::json!({"logical":logical,"unique":unique,"note":"all readonly bindings; expert banks scaled by real selected IDs; excludes mutable caches and temporary traffic"})
    }
}

impl FrozenGraph {
    pub fn check_except_groups(&self, groups: &[usize], skip: u64) -> bool {
        self.expected
            .iter()
            .zip(&self.expected_steps)
            .filter(|(_, i)| skip & (1u64 << groups[**i]) == 0)
            .all(|(s, _)| unsafe {
                std::slice::from_raw_parts(
                    s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                    s.bytes.len(),
                ) == s.bytes
            })
    }
}

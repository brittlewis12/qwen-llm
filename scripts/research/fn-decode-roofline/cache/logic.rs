//! Cache read-touch and exact marginal attribution. Research only.
use super::*;
use objc2_metal::MTLArgumentEncoder;
impl KernelEncoder {
    pub fn buffer_barrier_probe(&self) {
        self.raw
            .borrow()
            .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
    }
}

#[derive(Clone)]
pub struct WeightSpan {
    buffer: Buffer,
    offset: usize,
    bytes: usize,
}
pub struct CachePacket {
    args: Vec<u32>,
    table: Buffer,
    sink: Buffer,
    banks: Vec<WeightSpan>,
    pipeline: Pipeline,
    groups: usize,
    expected: u32,
    bytes: usize,
}
impl CachePacket {
    pub fn new(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
        pipeline: &Pipeline,
        banks: Vec<WeightSpan>,
        minimum_groups: usize,
        ones: bool,
    ) -> Self {
        assert!(!banks.is_empty() && banks.len() <= 32);
        let table = ctx
            .buffer_from(&vec![0u8; encoder.encodedLength()])
            .unwrap();
        unsafe {
            encoder.setArgumentBuffer_offset(Some(&table), 0);
        }
        let mut args = vec![0u32; 67];
        args[0] = banks.len() as u32;
        let mut blocks = 0usize;
        let mut bytes = 0usize;
        let mut expected = 0u32;
        for (i, b) in banks.iter().enumerate() {
            assert_eq!(b.bytes % 16, 0);
            assert!(b.offset + b.bytes <= b.buffer.length());
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&b.buffer), b.offset, i);
            }
            args[2 + i] = (b.bytes / 16) as u32;
            args[34 + i] = blocks as u32;
            blocks += b.bytes.div_ceil(65536);
            bytes += b.bytes;
            if !ones {
                let words = unsafe {
                    std::slice::from_raw_parts(
                        b.buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(b.offset)
                            .cast::<u32>(),
                        b.bytes / 4,
                    )
                };
                expected = words.iter().fold(expected, |s, &x| s.wrapping_add(x));
            }
        }
        args[34 + banks.len()] = blocks as u32;
        args[1] = blocks as u32;
        let groups = blocks.max(minimum_groups);
        if ones {
            assert_eq!(banks.len(), 1);
            assert_eq!(banks[0].bytes % 65536, 0);
            expected = (groups * 16384) as u32;
        } else {
            assert_eq!(groups, blocks);
        }
        let sink = ctx.buffer_from(&vec![0u32; groups * 8]).unwrap();
        Self {
            args,
            table,
            sink,
            banks,
            pipeline: pipeline.clone(),
            groups,
            expected,
            bytes,
        }
    }
    pub fn sweep(
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
        pipeline: &Pipeline,
        buffer: &Buffer,
        bytes: usize,
        minimum_groups: usize,
    ) -> Self {
        Self::new(
            ctx,
            encoder,
            pipeline,
            vec![WeightSpan {
                buffer: buffer.clone(),
                offset: 0,
                bytes,
            }],
            minimum_groups,
            true,
        )
    }
    pub fn encode(&self, e: &KernelEncoder) {
        e.set_pipeline(&self.pipeline);
        e.set_buffer(0, &self.table, 0);
        e.set_bytes_slice(1, &self.args);
        e.set_buffer(2, &self.sink, 0);
        for i in 0..3 {
            e.set_threadgroup_memory(i, 0);
        }
        for b in &self.banks {
            e.use_resource_read(&b.buffer);
        }
        e.dispatch(size(self.groups, 1), size(256, 1));
    }
    pub fn check(&self) -> bool {
        let words = unsafe {
            std::slice::from_raw_parts(self.sink.contents().as_ptr().cast::<u32>(), self.groups * 8)
        };
        words.iter().fold(0u32, |s, &x| s.wrapping_add(x)) == self.expected
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn traffic(&self) -> usize {
        self.groups * 65536
    }
}
fn projection(k: &str) -> bool {
    k.starts_with("kernel_mat_vec_")
        || k.contains("hc_down_silu")
        || k.contains("hc_up_mix")
        || k.starts_with("kernel_qwen4exp_ffn_projections")
}
impl LayerReplay {
    pub fn cache_weights(&self, scope: &str) -> Vec<WeightSpan> {
        let first_up = self
            .steps
            .iter()
            .position(|s| s.kernel.contains("hc_up_mix"))
            .unwrap();
        let ff = self
            .steps
            .iter()
            .position(|s| s.kernel.starts_with("kernel_qwen4exp_ffn_projections"))
            .unwrap();
        let last_projection = (0..ff)
            .rfind(|&i| self.steps[i].kernel.starts_with("kernel_mat_vec_"))
            .unwrap();
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for (index, s) in self.steps.iter().enumerate() {
            let include = match scope {
                "hc" => index <= first_up,
                "early" => {
                    index <= first_up
                        || (index < last_projection && s.kernel.starts_with("kernel_mat_vec_"))
                }
                "proj_shared" => projection(&s.kernel) && !s.kernel.contains("hc_"),
                "all_known" => true,
                _ => panic!("scope"),
            };
            if !include {
                continue;
            }
            for &(slot, _) in &self.access[&s.kernel] {
                let Some(m) = s.state.metadata.get(&slot) else {
                    continue;
                };
                if m.writable || (m.shape.len() == 3 && m.shape[2] == 512) {
                    continue;
                }
                let (b, o) = binding(s, slot);
                let id = Retained::as_ptr(&b) as *const () as usize;
                if seen.insert((id, o, m.bytes)) {
                    out.push(WeightSpan {
                        buffer: b,
                        offset: o,
                        bytes: m.bytes as usize,
                    });
                }
            }
        }
        out
    }
    pub fn cache_packets(
        &self,
        ctx: &MetalContext,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
        pipeline: &Pipeline,
        scope: &str,
        chunk: usize,
    ) -> Vec<CachePacket> {
        let spans = self.cache_weights(scope);
        if chunk == 0 {
            return vec![CachePacket::new(ctx, encoder, pipeline, spans, 0, false)];
        }
        let mut chunks = Vec::new();
        let mut spans_now = Vec::new();
        let mut room = chunk;
        for span in spans {
            let mut off = 0;
            while off < span.bytes {
                let n = (span.bytes - off).min(room);
                spans_now.push(WeightSpan {
                    buffer: span.buffer.clone(),
                    offset: span.offset + off,
                    bytes: n,
                });
                off += n;
                room -= n;
                if room == 0 {
                    chunks.push(CachePacket::new(
                        ctx,
                        encoder,
                        pipeline,
                        std::mem::take(&mut spans_now),
                        0,
                        false,
                    ));
                    room = chunk;
                }
            }
        }
        if !spans_now.is_empty() {
            chunks.push(CachePacket::new(
                ctx, encoder, pipeline, spans_now, 0, false,
            ));
        }
        chunks
    }
    pub fn states_match(&self) -> bool {
        self.expected.iter().skip(1).all(|s| unsafe {
            std::slice::from_raw_parts(
                s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                s.bytes.len(),
            ) == s.bytes
        })
    }
    pub fn cache_matvecs(&self, ctx: &MetalContext, scope: &str) -> CacheMatvecs {
        let weights = self.cache_weights(scope);
        self.restore();
        let mut steps = Vec::new();
        let mut saved = Vec::new();
        let mut expected = Vec::new();
        for s in &self.steps {
            let selected = projection(&s.kernel)
                && self.access[&s.kernel].iter().any(|&(slot, _)| {
                    let Some(m) = s.state.metadata.get(&slot) else {
                        return false;
                    };
                    if m.writable {
                        return false;
                    }
                    let (b, o) = binding(s, slot);
                    weights.iter().any(|w| {
                        Retained::as_ptr(&w.buffer) == Retained::as_ptr(&b)
                            && w.offset == o
                            && w.bytes == m.bytes as usize
                    })
                });
            let mut state = s.state.clone();
            if selected {
                state
                    .slots
                    .retain(|(slot, _)| self.access[&s.kernel].iter().any(|(i, _)| i == slot));
                let mut ranges = BTreeMap::<usize, (Buffer, usize, usize)>::new();
                for &(slot, _) in &self.access[&s.kernel] {
                    let Some((_, Slot::Buffer(b, o))) =
                        s.state.slots.iter().find(|(i, _)| *i == slot)
                    else {
                        continue;
                    };
                    if s.state.metadata.get(&slot).is_some_and(|m| !m.writable) {
                        continue;
                    }
                    let n = s
                        .state
                        .metadata
                        .get(&slot)
                        .map(|m| m.bytes as usize)
                        .unwrap_or(b.length() - o);
                    let id = Retained::as_ptr(b) as *const () as usize;
                    ranges
                        .entry(id)
                        .and_modify(|r| {
                            r.1 = r.1.min(*o);
                            r.2 = r.2.max(o + n);
                        })
                        .or_insert((b.clone(), *o, o + n));
                }
                let mut map = BTreeMap::new();
                for (id, (b, lo, hi)) in ranges {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            b.contents().as_ptr().cast::<u8>().add(lo),
                            hi - lo,
                        )
                    }
                    .to_vec();
                    let clone = ctx.buffer_from(&bytes).unwrap();
                    saved.push(SavedRange {
                        buffer: clone.clone(),
                        offset: 0,
                        bytes,
                    });
                    map.insert(id, (clone, lo));
                }
                for (_, slot) in &mut state.slots {
                    if let Slot::Buffer(b, o) = slot {
                        if let Some((clone, lo)) =
                            map.get(&(Retained::as_ptr(b) as *const () as usize))
                        {
                            *b = clone.clone();
                            *o -= lo;
                        }
                    }
                }
            }
            run(ctx, |e| append(e, s));
            if selected {
                for &(slot, write) in &self.access[&s.kernel] {
                    if !write {
                        continue;
                    }
                    let Some((_, Slot::Buffer(b, o))) =
                        s.state.slots.iter().find(|(i, _)| *i == slot)
                    else {
                        continue;
                    };
                    let (_, Slot::Buffer(clone, co)) =
                        state.slots.iter().find(|(i, _)| *i == slot).unwrap()
                    else {
                        panic!()
                    };
                    let n = s
                        .state
                        .metadata
                        .get(&slot)
                        .map(|m| m.bytes as usize)
                        .unwrap_or(b.length() - o);
                    let bytes = unsafe {
                        std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(*o), n)
                    }
                    .to_vec();
                    expected.push(SavedRange {
                        buffer: clone.clone(),
                        offset: *co,
                        bytes,
                    });
                }
                steps.push(CapturedStep {
                    kernel: s.kernel.clone(),
                    state,
                    grid: s.grid,
                    threads: s.threads,
                    concurrent: false,
                });
            }
        }
        CacheMatvecs {
            steps,
            saved,
            expected,
        }
    }
}
pub struct CacheMatvecs {
    steps: Vec<CapturedStep>,
    saved: Vec<SavedRange>,
    expected: Vec<SavedRange>,
}
impl CacheMatvecs {
    pub fn restore(&self) {
        restore(&self.saved);
    }
    pub fn replay(&self, e: &KernelEncoder) {
        for s in &self.steps {
            append(e, s);
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
}
fn low_bandwidth(k: &str) -> bool {
    k.contains("norm")
        || k.contains("select")
        || k.contains("expand_ids")
        || k.contains("gdn_step")
        || k.contains("topk")
        || k.contains("softmax")
        || k.contains("merge")
        || k.contains("inject")
        || k == "layer_ffn_epilogue"
}
impl LayerVariant {
    pub fn prefetched_replay(&self, e: &KernelEncoder, packets: &[CachePacket], placement: &str) {
        let eligible: Vec<_> = self
            .waves
            .iter()
            .enumerate()
            .filter_map(|(n, w)| {
                w.iter()
                    .any(|&i| low_bandwidth(&self.steps[i].kernel))
                    .then_some(n)
            })
            .collect();
        let one = self
            .waves
            .iter()
            .position(|w| {
                w.iter().any(|&i| {
                    let k = &self.steps[i].kernel;
                    match placement {
                        "chain" => {
                            k.contains("qsa_select_scan") || k == "kernel_gdn_step_decay_f32"
                        }
                        "route" => k.contains("topk"),
                        "publish" => k.contains("pool_publish"),
                        _ => false,
                    }
                })
            })
            .unwrap_or(self.waves.len() - 1);
        for (n, w) in self.waves.iter().enumerate() {
            if n > 0 {
                e.raw
                    .borrow()
                    .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            for &i in w {
                append(e, &self.steps[i]);
            }
            if placement == "spread" {
                if let Some(pos) = eligible.iter().position(|&x| x == n) {
                    let lo = pos * packets.len() / eligible.len();
                    let hi = (pos + 1) * packets.len() / eligible.len();
                    for j in lo..hi {
                        packets[packets.len() - 1 - j].encode(e);
                    }
                }
            } else if (placement == "start" && n == 0)
                || (placement == "tail" && n + 1 == self.waves.len())
                || ((placement == "chain" || placement == "route" || placement == "publish")
                    && n == one)
            {
                for p in packets {
                    p.encode(e);
                }
            }
        }
        if placement == "serial" {
            e.raw
                .borrow()
                .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            for p in packets {
                p.encode(e);
            }
        }
    }
}

const GROUPS: [&str; 6] = [
    "hc_glue",
    "attention_projections",
    "selection",
    "attention_cache",
    "ffn_shared_router",
    "ffn_routed_epilogue",
];
fn group_ids(steps: &[CapturedStep]) -> Vec<usize> {
    let mut ff = false;
    let mut selected = false;
    steps
        .iter()
        .map(|s| {
            let k = &s.kernel;
            if k.starts_with("kernel_qwen4exp_ffn_projections") {
                ff = true;
            }
            if k.contains("hc_") || k == "kernel_copy_offset_f32" {
                return 0;
            }
            if ff {
                if k.contains("topk")
                    || k.contains("moe_down")
                    || k.contains("swiglu")
                    || k == "layer_ffn_epilogue"
                    || k.contains("weighted_sum")
                {
                    return 5;
                }
                return 4;
            }
            if k.starts_with("kernel_mat_vec_") {
                return 1;
            }
            if !selected {
                if k.contains("expand_ids") {
                    selected = true;
                }
                return 2;
            }
            3
        })
        .collect()
}
fn payload(s: &CapturedStep, access: &AccessMap) -> f64 {
    let mut bytes = 0f64;
    let mut seen = BTreeSet::new();
    for &(slot, _) in &access[&s.kernel] {
        let Some(m) = s.state.metadata.get(&slot) else {
            continue;
        };
        if m.writable {
            continue;
        }
        let (b, o) = binding(s, slot);
        if !seen.insert((Retained::as_ptr(&b) as *const () as usize, o, m.bytes)) {
            continue;
        }
        bytes += m.bytes as f64
            * if m.shape.len() == 3 && m.shape[2] == 512 {
                10. / 512.
            } else {
                1.
            };
    }
    if s.kernel == "kernel_qwen4exp_qsa_decode_logits_reuse_f32" {
        bytes += words(s, 0)[3] as f64 * 1024.;
    }
    if s.kernel == "kernel_qwen4exp_qsa_split_f16" {
        bytes += words(s, 0)[0] as f64 * 1024.;
    }
    if s.kernel == "kernel_qwen4exp_qsa_index_scores_4x128_f16" {
        bytes += words(s, 0)[0] as f64 * 256.;
    }
    if s.kernel == "kernel_qwen4exp_qsa_publish_kv_f16" {
        bytes += 2048.;
    }
    bytes
}
pub struct CacheAttribution<'a> {
    variant: &'a LayerVariant,
    subs: Vec<SubstituteStep>,
    groups: Vec<usize>,
    counter: Buffer,
    access: AccessMap,
}
impl LayerVariant {
    pub fn cache_attribution<'a>(
        &'a self,
        ctx: &MetalContext,
        layer: &LayerReplay,
        access: &AccessMap,
    ) -> CacheAttribution<'a> {
        layer.restore();
        let mut slots: Vec<Option<SubstituteStep>> = (0..self.steps.len()).map(|_| None).collect();
        for wave in &self.waves {
            for &i in wave {
                let s = &self.steps[i];
                run(ctx, |e| append(e, s));
                let mut destinations = Vec::new();
                let mut bytes = Vec::new();
                let mut args = [0u32; 17];
                for &(slot, write) in &access[&s.kernel] {
                    if !write {
                        continue;
                    }
                    let Some((_, Slot::Buffer(b, o))) =
                        s.state.slots.iter().find(|(j, _)| *j == slot)
                    else {
                        continue;
                    };
                    if s.state.metadata.get(&slot).is_some_and(|m| !m.writable) {
                        continue;
                    }
                    let n = s
                        .state
                        .metadata
                        .get(&slot)
                        .map(|m| m.bytes as usize)
                        .unwrap_or(b.length() - o);
                    assert_eq!(n % 4, 0);
                    let j = destinations.len();
                    assert!(j < 8);
                    args[1 + j] = (n / 4) as u32;
                    args[9 + j] = (bytes.len() / 4) as u32;
                    bytes.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(*o), n)
                    });
                    destinations.push((b.clone(), *o, n));
                }
                args[0] = destinations.len() as u32;
                let words = bytes.len() / 4;
                if bytes.is_empty() {
                    bytes.resize(4, 0);
                }
                let source = ctx.buffer_from(&bytes).unwrap();
                slots[i] = Some(SubstituteStep {
                    source,
                    destinations,
                    args,
                    words,
                });
            }
        }
        CacheAttribution {
            variant: self,
            subs: slots.into_iter().map(Option::unwrap).collect(),
            groups: group_ids(&self.steps),
            counter: ctx.buffer_from(&vec![0u32; self.steps.len()]).unwrap(),
            access: access.clone(),
        }
    }
}
impl CacheAttribution<'_> {
    pub fn len(&self) -> usize {
        self.variant.steps.len()
    }
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({"groups":GROUPS,"steps":self.variant.steps.iter().enumerate().map(|(i,s)|serde_json::json!({"index":i,"kernel":s.kernel,"group":self.groups[i],"logical_bytes":payload(s,&self.access)})).collect::<Vec<_>>()})
    }
    pub fn replay(
        &self,
        e: &KernelEncoder,
        copy: &Pipeline,
        noop: &Pipeline,
        mask: u64,
        copy_only: bool,
        omit: Option<usize>,
    ) {
        for (n, wave) in self.variant.waves.iter().enumerate() {
            if n > 0 {
                e.raw
                    .borrow()
                    .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
            }
            for &i in wave {
                let active = if let Some(j) = omit {
                    i != j
                } else {
                    mask & (1 << self.groups[i]) != 0
                };
                if !copy_only && active {
                    append(e, &self.variant.steps[i]);
                } else if !copy_only || active {
                    self.subs[i].encode_copy(e, copy);
                } else {
                    e.set_pipeline(noop);
                    e.set_buffer(0, &self.counter, (i * 4) as u64);
                    e.set_bytes(1, &[1u32, 1]);
                    e.set_threadgroup_memory(0, 0);
                    e.dispatch(size(1, 1), size(32, 1));
                }
            }
        }
    }
}

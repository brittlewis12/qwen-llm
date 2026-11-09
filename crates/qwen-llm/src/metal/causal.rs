//! Dependency waves for a caller-owned compute pass. Recording owns bindings,
//! never tensor contents; normal Metal buffer barriers order successive waves.
use super::*;
use objc2_metal::MTLBinding;

#[cfg(any(test, feature = "perf-ablations"))]
thread_local! {
    static REPLAY_MODE:Cell<u8>=const {Cell::new(3)};
    static TAIL_ENABLED:Cell<bool>=const {Cell::new(false)};
    static TAIL_TRACE:std::cell::RefCell<Vec<serde_json::Value>>=const {std::cell::RefCell::new(Vec::new())};
}
#[inline]
fn replay_mode() -> u8 {
    #[cfg(any(test, feature = "perf-ablations"))]
    {
        return REPLAY_MODE.with(Cell::get);
    }
    #[cfg(not(any(test, feature = "perf-ablations")))]
    {
        3
    }
}

#[derive(Clone)]
pub(crate) enum Binding {
    Buffer {
        buffer: Buffer,
        offset: usize,
        len: usize,
    },
    Bytes(Vec<u8>),
}

#[derive(Clone, Default)]
struct Bindings {
    pipeline: Option<Pipeline>,
    slots: Vec<(usize, Binding)>,
    threadgroup: Vec<(usize, usize)>,
    resources: Vec<Buffer>,
}

impl Bindings {
    fn bind(&mut self, index: usize, binding: Binding) {
        if let Some((_, value)) = self.slots.iter_mut().find(|(i, _)| *i == index) {
            *value = binding;
        } else {
            self.slots.push((index, binding));
        }
    }
    fn buffer(&self, index: usize) -> Option<&Binding> {
        self.slots.iter().find(|(i, _)| *i == index).map(|(_, v)| v)
    }
    fn words(&self, index: usize) -> Option<Vec<u32>> {
        match self.buffer(index)? {
            Binding::Bytes(b) => Some(
                b.chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect(),
            ),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct Node {
    state: Bindings,
    grid: MTLSize,
    threads: MTLSize,
}
impl Node {
    fn name(&self) -> &str {
        &self.state.pipeline.as_ref().unwrap().name
    }
    fn encode(&self, enc: &KernelEncoder) {
        enc.set_pipeline(self.state.pipeline.as_ref().unwrap());
        for (i, b) in &self.state.slots {
            match b {
                Binding::Buffer { buffer, offset, .. } => {
                    enc.set_buffer(*i, buffer, *offset as u64)
                }
                Binding::Bytes(bytes) => enc.set_bytes_slice(*i, bytes),
            }
        }
        for &(i, n) in &self.state.threadgroup {
            enc.set_threadgroup_memory(i, n);
        }
        for b in &self.state.resources {
            enc.use_resource_read(b);
        }
        enc.dispatch(self.grid, self.threads);
    }
}

#[derive(Default)]
pub(crate) struct Recording {
    bindings: Bindings,
    nodes: Vec<Node>,
}
impl Recording {
    pub(crate) fn pipeline(&mut self, p: &Pipeline) {
        self.bindings.pipeline = Some(p.clone());
    }
    pub(crate) fn buffer(&mut self, i: usize, b: &Buffer, offset: u64) {
        let offset = offset as usize;
        self.bindings.bind(
            i,
            Binding::Buffer {
                buffer: b.clone(),
                offset,
                len: b.length() - offset,
            },
        );
    }
    pub(crate) fn tensor(&mut self, i: usize, t: &MetalTensor) {
        self.bindings.bind(
            i,
            Binding::Buffer {
                buffer: t.buffer.clone(),
                offset: t.offset as usize,
                len: t.n_bytes() as usize,
            },
        );
    }
    pub(crate) fn bytes(&mut self, i: usize, b: &[u8]) {
        self.bindings.bind(i, Binding::Bytes(b.to_vec()));
    }
    pub(crate) fn threadgroup(&mut self, i: usize, n: usize) {
        if let Some((_, old)) = self.bindings.threadgroup.iter_mut().find(|(j, _)| *j == i) {
            *old = n;
        } else {
            self.bindings.threadgroup.push((i, n));
        }
    }
    pub(crate) fn resource(&mut self, b: &Buffer) {
        self.bindings.resources.push(b.clone());
    }
    pub(crate) fn dispatch(&mut self, grid: MTLSize, threads: MTLSize) {
        self.nodes.push(Node {
            state: self.bindings.clone(),
            grid,
            threads,
        });
    }
}

#[derive(Clone, Copy)]
struct Access {
    buffer: usize,
    lo: usize,
    hi: usize,
    write: bool,
}
fn conflict(a: Access, b: Access) -> bool {
    a.buffer == b.buffer && (a.write || b.write) && a.lo < b.hi && b.lo < a.hi
}

impl MetalContext {
    fn buffer_accesses(&self, p: &Pipeline) -> Result<Arc<Vec<(usize, bool)>>, MetalError> {
        if let Some(a) = p.accesses.get() {
            return Ok(a.clone());
        }
        let mut reflection = None;
        unsafe {
            self.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &p.function,
                    objc2_metal::MTLPipelineOption::BindingInfo,
                    Some(&mut reflection),
                )
        }
        .map_err(|e| MetalError::Pipeline(p.name.to_string(), e.to_string()))?;
        let reflection = reflection.ok_or_else(|| {
            MetalError::Pipeline(p.name.to_string(), "missing buffer reflection".into())
        })?;
        let accesses = Arc::new(
            reflection
                .bindings()
                .iter()
                .filter(|b| b.r#type().0 == 0 && b.isUsed())
                .map(|b| (b.index(), b.access().0 != 0))
                .collect(),
        );
        let _ = p.accesses.set(accesses);
        Ok(p.accesses.get().unwrap().clone())
    }
}

impl KernelEncoder {
    #[cfg(any(test, feature = "perf-ablations"))]
    pub fn take_diagnostic_causal_tail() -> Vec<serde_json::Value> {
        TAIL_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
    }
    /// Research-only control: bit 0 schedules waves, bit 1 fuses scalar leaves.
    /// It changes no normal-build policy and is scoped to the calling thread.
    #[cfg(any(test, feature = "perf-ablations"))]
    pub fn diagnostic_causal_replay(mode: u8) -> impl Drop {
        assert!(mode < 8);
        struct Guard(u8, bool);
        impl Drop for Guard {
            fn drop(&mut self) {
                REPLAY_MODE.with(|m| m.set(self.0));
                TAIL_ENABLED.with(|m| m.set(self.1));
            }
        }
        Guard(
            REPLAY_MODE.with(|m| m.replace(mode)),
            TAIL_ENABLED.with(|m| m.replace(true)),
        )
    }
    /// Begin a bounded causal scope within a concurrent pass. Scopes and
    /// independent sequence owners compose without sharing mutable session state.
    pub(crate) fn begin_causal_scope(&self) {
        assert!(self.concurrent);
        assert!(self.recording.borrow().is_none());
        *self.recording.borrow_mut() = Some(Recording::default());
    }
    pub(crate) fn cancel_causal_scope(&self) {
        self.recording.borrow_mut().take();
    }
    pub(crate) fn finish_causal_scope(
        &self,
        ctx: &MetalContext,
        fuse_ffn: bool,
    ) -> Result<(), MetalError> {
        let mut recording = self.recording.borrow_mut().take().expect("causal scope");
        if fuse_ffn && (replay_mode() & 2 != 0 || replay_mode() >= 4) {
            fuse_flash_next(ctx, &mut recording.nodes, replay_mode())?;
        }
        let mut spans = Vec::with_capacity(recording.nodes.len());
        for n in &recording.nodes {
            let mut accesses = Vec::new();
            for &(i, write) in ctx
                .buffer_accesses(n.state.pipeline.as_ref().unwrap())?
                .iter()
            {
                if let Some(Binding::Buffer {
                    buffer,
                    offset,
                    len,
                }) = n.state.buffer(i)
                {
                    accesses.push(Access {
                        buffer: Retained::as_ptr(buffer) as *const () as usize,
                        lo: *offset,
                        hi: offset + len,
                        write,
                    });
                }
            }
            spans.push(accesses);
        }
        let waves = if replay_mode() & 1 != 0 {
            dependency_waves(&spans)
        } else {
            (0..spans.len()).map(|i| vec![i]).collect()
        };
        #[cfg(any(test, feature = "perf-ablations"))]
        if TAIL_ENABLED.with(Cell::get)
            && (recording.nodes.len() < 10 || TAIL_TRACE.with(|t| t.borrow().len() < 3))
        {
            let rows:Vec<_>=recording.nodes.iter().zip(&spans).map(|(n,s)|serde_json::json!({"kernel":n.name(),"args":n.state.words(0),"spans":s.iter().map(|a|serde_json::json!({"buffer":a.buffer,"lo":a.lo,"hi":a.hi,"write":a.write})).collect::<Vec<_>>()})).collect();
            TAIL_TRACE.with(|t| {
                t.borrow_mut()
                    .push(serde_json::json!({"rows":rows,"waves":waves}))
            });
        }
        for wave in waves {
            self.buffer_barrier();
            for i in wave {
                recording.nodes[i].encode(self);
            }
        }
        Ok(())
    }

    /// Order buffer accesses and start a fresh debug hazard scope. Unlike a
    /// threadgroup spin barrier this has Metal's normal forward-progress rules.
    pub fn buffer_barrier(&self) {
        self.raw
            .borrow()
            .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        #[cfg(debug_assertions)]
        {
            self.hazard_reads.borrow_mut().clear();
            self.hazard_writes.borrow_mut().clear();
        }
    }
}

fn dependency_waves(spans: &[Vec<Access>]) -> Vec<Vec<usize>> {
    let mut edges = vec![Vec::new(); spans.len()];
    let mut degree = vec![0; spans.len()];
    for j in 0..spans.len() {
        for k in j + 1..spans.len() {
            if spans[j]
                .iter()
                .any(|&a| spans[k].iter().any(|&b| conflict(a, b)))
            {
                edges[j].push(k);
                degree[k] += 1;
            }
        }
    }
    let mut ready: Vec<_> = degree
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i))
        .collect();
    let mut waves = Vec::new();
    while !ready.is_empty() {
        let wave = std::mem::take(&mut ready);
        for &j in &wave {
            for &k in &edges[j] {
                degree[k] -= 1;
                if degree[k] == 0 {
                    ready.push(k);
                }
            }
        }
        waves.push(wave);
    }
    waves
}

fn same_buffer(a: Option<&Binding>, b: Option<&Binding>) -> bool {
    match (a, b) {
        (
            Some(Binding::Buffer {
                buffer: a,
                offset: ao,
                ..
            }),
            Some(Binding::Buffer {
                buffer: b,
                offset: bo,
                ..
            }),
        ) => Retained::as_ptr(a) == Retained::as_ptr(b) && ao == bo,
        _ => false,
    }
}

fn access(binding: Option<&Binding>, write: bool) -> Option<Access> {
    match binding {
        Some(Binding::Buffer {
            buffer,
            offset,
            len,
        }) => Some(Access {
            buffer: Retained::as_ptr(buffer) as *const () as usize,
            lo: *offset,
            hi: offset + len,
            write,
        }),
        _ => None,
    }
}
fn touches(
    ctx: &MetalContext,
    nodes: &[Node],
    range: Access,
    writes_only: bool,
) -> Result<bool, MetalError> {
    for n in nodes {
        for &(slot, write) in ctx
            .buffer_accesses(n.state.pipeline.as_ref().unwrap())?
            .iter()
        {
            if (!writes_only || write)
                && access(n.state.buffer(slot), true).is_some_and(|a| conflict(a, range))
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Fuse only the proven scalar epilogue pattern, with all producer/consumer
/// identities checked. Other model geometries and packed kernels stay intact.
fn fuse_flash_next(ctx: &MetalContext, nodes: &mut Vec<Node>, mode: u8) -> Result<(), MetalError> {
    let sum = nodes
        .iter()
        .position(|n| n.name() == "kernel_moe_weighted_sum_f32");
    let axpy = nodes
        .iter()
        .position(|n| n.name() == "kernel_axpy_scalar_f32");
    let inject = nodes
        .iter()
        .rposition(|n| n.name() == "kernel_qwen4exp_hc_inject_f32");
    if let (true, Some(sum), Some(axpy), Some(inject)) = (mode < 4 || mode >= 6, sum, axpy, inject)
    {
        if sum < axpy && axpy + 2 == inject && nodes[axpy + 1].name() == "kernel_copy_offset_f32" {
            let s = &nodes[sum];
            let a = &nodes[axpy];
            let c = &nodes[axpy + 1];
            let inj = &nodes[inject];
            let sw = s.state.words(0).unwrap_or_default();
            let iw = inj.state.words(0).unwrap_or_default();
            let cw = c.state.words(0).unwrap_or_default();
            if sw.len() == 2
                && iw.len() == 2
                && sw[0] == iw[1]
                && cw.get(1) == Some(&0)
                && cw.first() == Some(&sw[0])
                && same_buffer(s.state.buffer(3), a.state.buffer(3))
                && same_buffer(a.state.buffer(3), c.state.buffer(1))
                && same_buffer(c.state.buffer(2), inj.state.buffer(1))
                // Moving a reduction extends both input lifetimes. Native
                // MoE paths can reuse their expert outputs as shared scratch.
                && !touches(ctx, &nodes[sum+1..inject], access(s.state.buffer(1),false).unwrap(),true)?
                && !touches(ctx, &nodes[sum+1..inject], access(s.state.buffer(2),false).unwrap(),true)?
                && !touches(ctx, &nodes[sum+1..axpy], access(s.state.buffer(3),true).unwrap(),false)?
            {
                let mut state = Bindings {
                    pipeline: Some(ctx.pipeline("kernel_qwen4exp_ffn_epilogue_f32")?),
                    ..Default::default()
                };
                for (dst, n, src) in [
                    (1, s, 1),
                    (2, s, 2),
                    (3, a, 1),
                    (4, a, 2),
                    (5, inj, 2),
                    (6, inj, 3),
                ] {
                    state.bind(dst, n.state.buffer(src).unwrap().clone());
                }
                state.bind(
                    0,
                    Binding::Bytes(bytemuck::cast_slice(&[sw[0], iw[0], sw[1]]).to_vec()),
                );
                let fused = Node {
                    state,
                    grid: MTLSize {
                        width: (sw[0] as usize).div_ceil(256),
                        height: iw[0] as usize,
                        depth: 1,
                    },
                    threads: MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                };
                nodes[inject] = fused;
                for i in [axpy + 1, axpy, sum] {
                    nodes.remove(i);
                }
            }
        }
    }
    // A scalar mixer copy followed immediately by injection has no other reader.
    if mode >= 6 {
        return Ok(());
    }
    let mut i = 0;
    while i + 1 < nodes.len() {
        if nodes[i].name() == "kernel_copy_offset_f32"
            && nodes[i + 1].name() == "kernel_qwen4exp_hc_inject_f32"
            && nodes[i]
                .state
                .words(0)
                .is_some_and(|w| w.get(1) == Some(&0))
            && same_buffer(nodes[i].state.buffer(2), nodes[i + 1].state.buffer(1))
            && nodes[i]
                .state
                .words(0)
                .zip(nodes[i + 1].state.words(0))
                .is_some_and(|(c, inj)| c.first() == inj.get(1))
            && !touches(
                ctx,
                &nodes[i + 2..],
                access(nodes[i].state.buffer(2), false).unwrap(),
                false,
            )?
        {
            let b = nodes[i].state.buffer(1).unwrap().clone();
            nodes[i + 1].state.bind(1, b);
            nodes.remove(i);
        } else {
            i += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn waves_preserve_read_write_and_alias_dependencies() {
        let a = |buffer, lo, hi, write| Access {
            buffer,
            lo,
            hi,
            write,
        };
        let spans = vec![
            vec![a(1, 0, 16, true)],
            vec![a(2, 0, 16, true)],
            vec![a(1, 0, 16, false)],
            vec![a(1, 16, 32, true)],
            vec![a(1, 8, 24, true)],
        ];
        let waves = dependency_waves(&spans);
        assert_eq!(waves, vec![vec![0, 1, 3], vec![2], vec![4]]);
    }

    #[test]
    fn ffn_epilogue_f64_error_no_worse_than_unfused_chain() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        let (hidden, branches, topk) = (257usize, 3usize, 5usize);
        let values = |n: usize, salt: usize| {
            (0..n)
                .map(|i| ((i * 167 + salt * 31) % 521) as f32 / 521. - 0.5)
                .collect::<Vec<_>>()
        };
        let (experts, weights, shared, raw, residual) = (
            values(hidden * topk, 1),
            values(topk, 2),
            values(hidden, 3),
            values(branches, 4),
            values(hidden * branches, 5),
        );
        let scale = 0.37f32;
        let make = |v: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(v),
                vec![v.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let (e, w, s, r, old, gate) = (
            make(&experts),
            make(&weights),
            make(&shared),
            make(&raw),
            make(&residual),
            make(&[scale]),
        );
        let native = make(&residual);
        let fused = make(&residual);
        let rewritten = make(&residual);
        let sum = MetalTensor::zeros_f32(&ctx, vec![hidden as u64]).unwrap();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode_moe_weighted_sum_f32(&ctx, &enc, &e, &w, &sum, hidden, topk).unwrap();
        encode_axpy_scalar_f32(&ctx, &enc, &s, &gate, &sum).unwrap();
        enc.set_pipeline(&ctx.pipeline("kernel_qwen4exp_hc_inject_f32").unwrap());
        enc.set_bytes(0, &[branches as u32, hidden as u32]);
        enc.set_tensor(1, &sum);
        enc.set_tensor(2, &r);
        enc.set_tensor(3, &native);
        enc.dispatch(
            MTLSize {
                width: hidden.div_ceil(256),
                height: branches,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.set_pipeline(&ctx.pipeline("kernel_qwen4exp_ffn_epilogue_f32").unwrap());
        enc.set_bytes(0, &[hidden as u32, branches as u32, topk as u32]);
        for (i, t) in [&e, &w, &s, &gate, &r, &fused].into_iter().enumerate() {
            enc.set_tensor(i + 1, t);
        }
        enc.dispatch(
            MTLSize {
                width: hidden.div_ceil(256),
                height: branches,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
        cmd.commit();
        wait_completed(&cmd).unwrap();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin_causal(&cmd);
        let copied = MetalTensor::zeros_f32(&ctx, vec![hidden as u64]).unwrap();
        encode_moe_weighted_sum_f32(&ctx, &enc, &e, &w, &sum, hidden, topk).unwrap();
        encode_axpy_scalar_f32(&ctx, &enc, &s, &gate, &sum).unwrap();
        encode_copy_offset_f32(&ctx, &enc, &sum, 0, &copied, hidden).unwrap();
        enc.set_pipeline(&ctx.pipeline("kernel_qwen4exp_hc_inject_f32").unwrap());
        enc.set_bytes(0, &[branches as u32, hidden as u32]);
        enc.set_tensor(1, &copied);
        enc.set_tensor(2, &r);
        enc.set_tensor(3, &rewritten);
        enc.dispatch(
            MTLSize {
                width: hidden.div_ceil(256),
                height: branches,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.finish_causal_scope(&ctx, true).unwrap();
        enc.end();
        cmd.commit();
        wait_completed(&cmd).unwrap();
        let read = |t: &MetalTensor| unsafe {
            std::slice::from_raw_parts(
                t.buffer.contents().as_ptr().cast::<f32>(),
                hidden * branches,
            )
        };
        let (a, b) = (read(&native), read(&fused));
        assert_eq!(
            read(&rewritten),
            b,
            "recorded epilogue rewrite differs from direct leaf"
        );
        let mut error = [(0f64, 0f64); 2];
        for branch in 0..branches {
            for j in 0..hidden {
                let sum = (0..topk)
                    .map(|k| f64::from(weights[k]) * f64::from(experts[k * hidden + j]))
                    .sum::<f64>()
                    + f64::from(scale) * f64::from(shared[j]);
                let reference = f64::from(residual[branch * hidden + j])
                    + sum * 2. / (1. + (-f64::from(raw[branch]) / branches as f64).exp());
                for (slot, v) in [a, b].into_iter().enumerate() {
                    let d = (f64::from(v[branch * hidden + j]) - reference).abs();
                    error[slot].0 += d * d;
                    error[slot].1 = error[slot].1.max(d);
                }
            }
        }
        assert!(
            error[1].0 <= error[0].0 && error[1].1 <= error[0].1,
            "native={:?} fused={:?}",
            error[0],
            error[1]
        );
        drop(old);
    }
}

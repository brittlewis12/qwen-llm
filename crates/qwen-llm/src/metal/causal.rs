//! Dependency waves for a caller-owned compute pass. Recording owns bindings,
//! never tensor contents; normal Metal buffer barriers order successive waves.
use super::*;
use objc2_metal::MTLBinding;

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
        if fuse_ffn {
            fuse_flash_next(ctx, &mut recording.nodes)?;
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
        let waves = dependency_waves(&spans);
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

/// Fuse only the proven scalar epilogue pattern, with all producer/consumer
/// identities checked. Other model geometries and packed kernels stay intact.
fn fuse_flash_next(ctx: &MetalContext, nodes: &mut Vec<Node>) -> Result<(), MetalError> {
    let sum = nodes
        .iter()
        .position(|n| n.name() == "kernel_moe_weighted_sum_f32");
    let axpy = nodes
        .iter()
        .position(|n| n.name() == "kernel_axpy_scalar_f32");
    let inject = nodes
        .iter()
        .rposition(|n| n.name() == "kernel_qwen4exp_hc_inject_f32");
    if let (Some(sum), Some(axpy), Some(inject)) = (sum, axpy, inject) {
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
    let mut i = 0;
    while i + 1 < nodes.len() {
        if nodes[i].name() == "kernel_copy_offset_f32"
            && nodes[i + 1].name() == "kernel_qwen4exp_hc_inject_f32"
            && nodes[i]
                .state
                .words(0)
                .is_some_and(|w| w.get(1) == Some(&0))
            && same_buffer(nodes[i].state.buffer(2), nodes[i + 1].state.buffer(1))
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
}

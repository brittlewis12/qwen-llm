//! Fixed-input, output-preserving counterfactuals. Research only; no inference policy.
use super::*;
use std::cell::RefCell;
#[derive(Clone, Copy)]
pub enum TokenPriceMode {
    Capture,
    Hold(u32),
    Stream(u32),
}
struct Output {
    target: Buffer,
    offset: usize,
    bytes: usize,
    held: Buffer,
}
struct Record {
    name: String,
    bit: u32,
    inputs: Vec<(usize, usize)>,
    args: Option<Vec<u32>>,
    grid: [usize; 3],
    outputs: Vec<Output>,
    stream: Option<Stream>,
    generic: Option<Generic>,
}
#[derive(Default)]
pub struct TokenPriceBook {
    records: Rc<RefCell<Vec<Record>>>,
}
struct Active {
    records: Rc<RefCell<Vec<Record>>>,
    mode: TokenPriceMode,
    cursor: usize,
}
thread_local! {static ACTIVE: RefCell<Option<Active>>=const {RefCell::new(None)};}
pub struct TokenPriceGuard {
    _thread: std::marker::PhantomData<Rc<()>>,
}
impl Drop for TokenPriceGuard {
    fn drop(&mut self) {
        ACTIVE.with(|s| {
            s.borrow_mut().take();
        });
    }
}
impl TokenPriceBook {
    pub fn scope(&self, mode: TokenPriceMode) -> TokenPriceGuard {
        if matches!(mode, TokenPriceMode::Capture) {
            self.records.borrow_mut().clear();
        }
        ACTIVE.with(|s| {
            assert!(s.borrow().is_none());
            *s.borrow_mut() = Some(Active {
                records: self.records.clone(),
                mode,
                cursor: 0,
            });
        });
        TokenPriceGuard {
            _thread: std::marker::PhantomData,
        }
    }
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!(self.records.borrow().iter().map(|r|serde_json::json!({"kernel":r.name,"bit":r.bit,"held_bytes":r.outputs.iter().map(|o|o.bytes).sum::<usize>(),"inputs":r.inputs,"args":r.args,"grid":r.grid})).collect::<Vec<_>>())
    }
}
impl TokenPriceGuard {
    pub fn finish(self) {
        ACTIVE.with(|s| {
            let s = s.borrow();
            let a = s.as_ref().unwrap();
            if matches!(a.mode, TokenPriceMode::Hold(_) | TokenPriceMode::Stream(_)) {
                assert_eq!(
                    a.cursor,
                    a.records.borrow().len(),
                    "counterfactual dispatch count"
                );
            }
        });
        drop(self);
    }
}
// Exhaustive, disjoint dispatch cohorts. The catch-all remains visible in metadata.
fn bit(name: &str) -> u32 {
    let n = name.to_ascii_lowercase();
    if n.contains("q6_k") {
        1
    } else if (n.contains("moe_") || n.contains("expert_compat"))
        && (n.contains("iq")
            || n.contains("q4")
            || n.contains("q5")
            || n.contains("q2")
            || n.contains("q8"))
    {
        2 | if n.contains("iq2_s") {
            8192
        } else if n.contains("iq3_xxs") {
            32768
        } else if n.contains("iq3_s") {
            16384
        } else if n.contains("iq4_xs") {
            65536
        } else if n.contains("iq4_nl") {
            131072
        } else if n.contains("q2_0") {
            262144
        } else {
            524288
        }
    } else if n.contains("qsa_split")
        || n.contains("attention_logits")
        || n.contains("decode_logits_reuse")
    {
        4
    } else if n.contains("qsa_select_")
        || n.contains("qsa_expand_ids")
        || n.contains("qsa_index_scores")
    {
        8
    } else if n.contains("qsa_") {
        16
    } else if n.contains("gdn_") || n.contains("ssm_conv") || n.contains("l2_norm_pair") {
        32
    } else if n.contains("hc_") || n.contains("hyper_") {
        if n.contains("bf16") { 64 } else { 4096 }
    } else if n.contains("moe_")
        || n.contains("router")
        || n.contains("topk")
        || n.contains("ffn_epilogue")
        || n.contains("swiglu")
    {
        128
    } else if n.contains("ple_") || n.contains("embedding") {
        256
    } else if n.contains("mat_vec") {
        512
    } else if n.contains("copy") {
        1024
    } else {
        2048
    }
}

fn copy(
    ctx: &MetalContext,
    e: &KernelEncoder,
    src: &Buffer,
    so: usize,
    dst: &Buffer,
    doff: usize,
    bytes: usize,
) -> Result<(), MetalError> {
    assert_eq!(bytes % 4, 0);
    let p = ctx.pipeline("kernel_copy_offset_i32")?;
    e.set_pipeline(&p);
    e.set_bytes(0, &[(bytes / 4) as u32, 0]);
    e.set_buffer(1, src, so as u64);
    e.set_buffer(2, dst, doff as u64);
    e.set_threadgroup_memory(0, 0);
    e.dispatch(
        MTLSize {
            width: (bytes / 4).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
pub(super) fn replace(ctx: &MetalContext, e: &KernelEncoder, n: &Node) -> Result<bool, MetalError> {
    if bit(n.name()) == 0 {
        return Ok(false);
    }
    ACTIVE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(a) = s.as_mut() else {
            return Ok(false);
        };
        let (mask, read) = match a.mode {
            TokenPriceMode::Capture => return Ok(false),
            TokenPriceMode::Hold(mask) => (mask, false),
            TokenPriceMode::Stream(mask) => (mask, true),
        };
        let records = a.records.borrow();
        let r = &records[a.cursor];
        a.cursor += 1;
        assert_eq!(r.name, n.name(), "counterfactual dispatch order");
        if mask & r.bit == 0 {
            return Ok(false);
        }
        if read {
            if let Some(generic) = r.generic.as_ref() {
                let grid = 1024usize;
                let threads = 128usize;
                let p = ctx.pipeline("kernel_research_bytes_stream")?;
                e.set_pipeline(&p);
                e.set_bytes(
                    0,
                    &BytesStreamArgs {
                        bytes: generic.bytes as u64,
                        read: 1,
                        words: 0,
                    },
                );
                e.set_buffer(1, &generic.binding.0, generic.binding.1 as u64);
                e.set_buffer(2, &generic.checksum, 0);
                e.set_threadgroup_memory(0, 0);
                e.dispatch(
                    MTLSize {
                        width: grid,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads,
                        height: 1,
                        depth: 1,
                    },
                );
                for o in &r.outputs {
                    copy(ctx, e, &o.held, 0, &o.target, o.offset, o.bytes)?;
                }
                return Ok(true);
            }
            let stream = r.stream.as_ref().expect("expert stream cohort");
            e.set_pipeline(&ctx.pipeline("kernel_research_v41_expert_stream")?);
            e.set_bytes(0, &stream.args);
            for (i, (b, off)) in stream.bindings.iter().enumerate() {
                e.set_buffer(i + 1, b, *off as u64);
            }
            e.dispatch(stream.grid, stream.threads);
        } else {
            for o in &r.outputs {
                copy(ctx, e, &o.held, 0, &o.target, o.offset, o.bytes)?;
            }
        }
        Ok(true)
    })
}
pub(super) fn after_wave(
    ctx: &MetalContext,
    e: &KernelEncoder,
    nodes: &[Node],
    wave: &[usize],
) -> Result<(), MetalError> {
    ACTIVE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(a) = s.as_mut() else { return Ok(()) };
        if !matches!(a.mode, TokenPriceMode::Capture) {
            return Ok(());
        }
        if !wave.iter().any(|&i| bit(nodes[i].name()) != 0) {
            return Ok(());
        }
        e.buffer_barrier();
        for &i in wave {
            let n = &nodes[i];
            let bit = bit(n.name());
            if bit == 0 {
                continue;
            }
            let mut outputs = Vec::new();
            for &(slot, write) in ctx
                .buffer_accesses(n.state.pipeline.as_ref().unwrap())?
                .iter()
            {
                if !write {
                    continue;
                }
                if let Some(Binding::Buffer {
                    buffer,
                    offset,
                    len,
                }) = n.state.buffer(slot)
                {
                    assert!(
                        *len < 64 * 1024 * 1024,
                        "unexpected large held output {}",
                        n.name()
                    );
                    let held = ctx.buffer_from(&vec![0u8; *len])?;
                    copy(ctx, e, buffer, *offset, &held, 0, *len)?;
                    outputs.push(Output {
                        target: buffer.clone(),
                        offset: *offset,
                        bytes: *len,
                        held,
                    });
                }
            }
            assert!(!outputs.is_empty(), "missing held outputs {}", n.name());
            let (stream, generic) = if bit & 2 != 0 {
                let args = n.state.words(0).expect("expert args");
                let swiglu = n.name().contains("swiglu");
                let binding = |slot| match n.state.buffer(slot).unwrap() {
                    Binding::Buffer {
                        buffer,
                        offset,
                        len,
                    } => (buffer.clone(), *offset, *len),
                    _ => panic!("expert buffer"),
                };
                let bank = binding(1);
                let bank_b = binding(if swiglu { 2 } else { 1 });
                let ids = binding(if swiglu { 4 } else { 3 });
                let checksum = ctx.buffer_from(&vec![
                    0u8;
                    n.grid.width
                        * n.grid.height
                        * n.grid.depth
                        * (n.threads.width / 32)
                        * 4
                ])?;
                assert_eq!(outputs.len(), 1);
                let out = &outputs[0];
                (
                    Some(Stream {
                        args: StreamArgs {
                            row_bytes: (bank.2 / args[2] as usize / args[1] as usize) as u64,
                            n_out: args[1],
                            experts: args[2],
                            channels: (args[1] as usize).div_ceil(n.grid.width) as u32,
                            slots: n.grid.height as u32,
                            banks: if swiglu { 2 } else { 1 },
                            read: 1,
                        },
                        grid: n.grid,
                        threads: n.threads,
                        bindings: vec![
                            (bank.0, bank.1),
                            (bank_b.0, bank_b.1),
                            (ids.0, ids.1),
                            (out.held.clone(), 0),
                            (out.target.clone(), out.offset),
                            (checksum, 0),
                        ],
                    }),
                    None,
                )
            } else if bit == 64 {
                let mut best: Option<(usize, usize, usize)> = None;
                for &(slot, _) in ctx
                    .buffer_accesses(n.state.pipeline.as_ref().unwrap())?
                    .iter()
                    .filter(|(_, write)| !write)
                {
                    if let Some(Binding::Buffer { offset, len, .. }) = n.state.buffer(slot) {
                        if best.is_none_or(|(_, _, l)| *len > l) {
                            best = Some((slot, *offset, *len));
                        }
                    }
                }
                (
                    None,
                    best.map(|(slot, off, len)| Generic {
                        binding: match n.state.buffer(slot).unwrap() {
                            Binding::Buffer { buffer, .. } => (buffer.clone(), off),
                            _ => unreachable!(),
                        },
                        bytes: len,
                        checksum: ctx
                            .buffer_from(&vec![0u8; 1024 * 4 * 4])
                            .expect("generic stream checksum"),
                    }),
                )
            } else {
                (None, None)
            };
            a.records.borrow_mut().push(Record {
                name: n.name().into(),
                bit,
                inputs: ctx
                    .buffer_accesses(n.state.pipeline.as_ref().unwrap())?
                    .iter()
                    .filter_map(|&(slot, write)| {
                        if write {
                            None
                        } else {
                            match n.state.buffer(slot) {
                                Some(Binding::Buffer { len, .. }) => Some((slot, *len)),
                                _ => None,
                            }
                        }
                    })
                    .collect(),
                args: n.state.words(0),
                grid: [n.grid.width, n.grid.height, n.grid.depth],
                outputs,
                stream,
                generic,
            });
        }
        Ok(())
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StreamArgs {
    row_bytes: u64,
    n_out: u32,
    experts: u32,
    channels: u32,
    slots: u32,
    banks: u32,
    read: u32,
}
struct Stream {
    args: StreamArgs,
    grid: MTLSize,
    threads: MTLSize,
    bindings: Vec<(Buffer, usize)>,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BytesStreamArgs {
    bytes: u64,
    read: u32,
    words: u32,
}
struct Generic {
    binding: (Buffer, usize),
    bytes: usize,
    checksum: Buffer,
}
impl TokenPriceBook {
    /// Isolated held-output copy tax. No inference/session commit is claimed.
    pub fn copy_tax(&self, ctx: &MetalContext, mask: u32) -> Result<f64, MetalError> {
        let cmd = ctx.queue.commandBuffer().expect("copy-tax command");
        let enc = KernelEncoder::begin(&cmd);
        // Force a nonempty command for the baseline too.
        let records = self.records.borrow();
        let first = &records[0].outputs[0];
        let sink = ctx.buffer_from(&[0u8; 4])?;
        copy(ctx, &enc, &first.held, 0, &sink, 0, 4)?;
        for r in records.iter().filter(|r| r.bit & mask != 0) {
            for o in &r.outputs {
                copy(ctx, &enc, &o.held, 0, &o.target, o.offset, o.bytes)?;
            }
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != objc2_metal::MTLCommandBufferStatus::Completed {
            return Err(MetalError::Counter("copy-tax command failed".into()));
        }
        Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3)
    }
}

//! Output-preserving counterfactuals. Enabled only by perf-ablations.
use crate::{metal::*, tensor::GgmlType};
use objc2_metal::MTLSize;
use std::cell::RefCell;
const CAP: u64 = 24 << 30;
#[derive(Clone)]
struct Frame {
    role: u8,
    shape: Vec<u64>,
    output: MetalTensor,
}
#[derive(Default)]
struct State {
    mode: u8,
    mask: u8,
    stream: bool,
    cursor: usize,
    frames: Vec<Frame>,
    bytes: u64,
    sink: Option<MetalTensor>,
    reads: [u64; 2],
}
thread_local! {static STATE:RefCell<State>=RefCell::new(State::default());}
pub fn capture(ctx: &MetalContext) {
    STATE.with(|s| {
        *s.borrow_mut() = State {
            mode: 1,
            sink: Some(MetalTensor::zeros_i32(ctx, vec![262144]).unwrap()),
            ..State::default()
        }
    });
}
pub fn start(mask: u8, stream: bool) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.mode = 2;
        s.mask = mask;
        s.stream = stream;
        s.cursor = 0;
        s.reads = [0; 2];
    });
}
pub fn finish() -> (usize, u64, [u64; 2]) {
    STATE.with(|s| {
        let s = s.borrow();
        if s.mode == 2 {
            assert_eq!(s.cursor, s.frames.len());
        }
        (s.frames.len(), s.bytes, s.reads)
    })
}
pub fn clear() {
    STATE.with(|s| *s.borrow_mut() = State::default());
}
fn bits(t: &MetalTensor) -> MetalTensor {
    MetalTensor {
        dtype: GgmlType::I32,
        shape: vec![t.n_bytes() / 4],
        ..t.clone()
    }
}
fn copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
) -> Result<(), MetalError> {
    encode_copy_offset_i32(
        ctx,
        enc,
        &bits(src),
        0,
        &bits(dst),
        src.n_bytes() as usize / 4,
    )
}
fn stream(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    t: &MetalTensor,
    counts: Option<&MetalTensor>,
    experts: usize,
    sink: &MetalTensor,
) -> Result<u64, MetalError> {
    let words = t.n_bytes() / 4 / experts as u64;
    let chunks = words.div_ceil(16384);
    assert!(chunks * experts as u64 <= sink.n_elements());
    assert_eq!(t.offset % 4, 0);
    let p = ctx.pipeline("kernel_glm_append_price_stream")?;
    enc.set_pipeline(&p);
    enc.set_bytes(
        0,
        &[words as u32, chunks as u32, u32::from(counts.is_some()), 0],
    );
    enc.set_tensor(1, t);
    enc.set_tensor(2, counts.unwrap_or(sink));
    enc.set_tensor(3, sink);
    enc.dispatch(
        MTLSize {
            width: chunks as usize,
            height: experts,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(t.n_bytes())
}
pub fn before(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    role: u8,
    y: &MetalTensor,
    w: &MetalTensor,
    x: &MetalTensor,
    counts: Option<&MetalTensor>,
    experts: usize,
) -> Result<bool, MetalError> {
    STATE.with(|cell| {
        let mut s = cell.borrow_mut();
        if s.mode != 2 {
            return Ok(false);
        }
        let i = s.cursor;
        s.cursor += 1;
        let f = &s.frames[i];
        assert_eq!((f.role, &f.shape), (role, &y.shape));
        if s.mask & role == 0 {
            return Ok(false);
        }
        let old = f.output.clone();
        if s.stream {
            let sink = s.sink.as_ref().unwrap();
            let n =
                stream(ctx, enc, w, counts, experts, sink)? + stream(ctx, enc, x, None, 1, sink)?;
            s.reads[usize::from(role == 2)] += n;
        }
        copy(ctx, enc, &old, y)?;
        Ok(true)
    })
}
pub fn after(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    role: u8,
    y: &MetalTensor,
) -> Result<(), MetalError> {
    STATE.with(|cell| {
        let mut s = cell.borrow_mut();
        if s.mode != 1 {
            return Ok(());
        }
        assert!(s.bytes + y.n_bytes() <= CAP);
        let out = MetalTensor::zeros_i32(ctx, vec![y.n_bytes() / 4])?;
        copy(ctx, enc, y, &out)?;
        s.bytes += y.n_bytes();
        s.frames.push(Frame {
            role,
            shape: y.shape.clone(),
            output: out,
        });
        Ok(())
    })
}

// Explicit route override for the whole-append probe: `Some(0)` forces the
// incumbent tile, `Some(2)` forces the adaptive panels, and `None` leaves the
// production env default in charge.
thread_local! {static DOWN_MODE:std::cell::Cell<Option<u8>>=const{std::cell::Cell::new(None)};static DOWN_HITS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}
pub fn down_mode(mode: u8) {
    assert!(mode <= 2);
    DOWN_MODE.with(|v| v.set(Some(mode)));
}
pub fn down_mode_override() -> Option<u8> {
    DOWN_MODE.with(std::cell::Cell::get)
}
/// Clear the probe override so the production env default selects the route.
pub fn down_mode_default() {
    DOWN_MODE.with(|v| v.set(None));
}
pub fn note_down() {
    DOWN_HITS.with(|v| v.set(v.get() + 1));
}
pub fn reset_hits() {
    DOWN_HITS.with(|v| v.set(0));
}
pub fn hits() -> usize {
    DOWN_HITS.with(std::cell::Cell::get)
}

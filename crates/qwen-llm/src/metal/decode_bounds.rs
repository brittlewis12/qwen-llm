//! Counterfactual decode bounds, available only with `perf-ablations`.
//! Not a model execution policy: skipped stages are intentionally incorrect.
use super::*;
use std::cell::RefCell;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Mode {
    Reference,
    Capture,
    AttentionIo,
    PostControls,
    AllHc,
    RoutedRead,
    RoutedHold,
    /// Skip the batched dense Q5_K/Q8_0 projections (weights and compute).
    DenseHold,
    /// Skip the fused decode attention core (flash partial + combine/RoPE).
    AttentionCoreHold,
    /// Skip the flash split partial (QK + split softmax + PV + KV reads).
    FlashPartialHold,
    /// Skip the flash split combine (and its inverse RoPE).
    FlashCombineHold,
}
#[derive(Clone, Default, Debug, serde::Serialize)]
pub struct Stats {
    pub skipped: BTreeMap<String, usize>,
    pub expert_calls: usize,
    pub expert_bytes: u64,
}
#[derive(Default)]
struct Book {
    saved: Vec<Saved>,
}
struct Active {
    mode: Mode,
    book: Rc<RefCell<Book>>,
    stats: Stats,
    label: &'static str,
    cursor: usize,
}
thread_local! {static ACTIVE:RefCell<Option<Active>>=const {RefCell::new(None)};}
#[derive(Default)]
pub struct Experiment {
    book: Rc<RefCell<Book>>,
}
pub struct Guard {
    done: bool,
    _thread: std::marker::PhantomData<Rc<()>>,
}
impl Experiment {
    pub fn scope(&self, mode: Mode) -> Result<Guard, MetalError> {
        if ACTIVE.with(|a| a.borrow().is_some()) {
            return Err(MetalError::Counter("nested decode bound".into()));
        }
        if mode == Mode::Capture {
            self.book.borrow_mut().saved.clear();
        }
        ACTIVE.with(|a| {
            let mut a = a.borrow_mut();
            if a.is_some() {
                return Err(MetalError::Counter("nested decode bound".into()));
            }
            *a = Some(Active {
                mode,
                book: self.book.clone(),
                stats: Stats::default(),
                label: "",
                cursor: 0,
            });
            Ok(Guard {
                done: false,
                _thread: std::marker::PhantomData,
            })
        })
    }
    /// Pure streaming block over the captured expert calls, preserving their
    /// grids, IDs and row strides. The held output copy is ~21 MB/round.
    pub fn encode_streams(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        read: bool,
    ) -> Result<(), MetalError> {
        for s in &self.book.borrow().saved {
            encode_stream(ctx, enc, s, read, None)?;
        }
        Ok(())
    }
    pub fn captured_calls(&self) -> usize {
        self.book.borrow().saved.len()
    }
    pub fn captured_bytes(&self) -> u64 {
        self.book.borrow().saved.iter().map(|s| s.bytes()).sum()
    }
}
impl Guard {
    pub fn finish(mut self) -> Stats {
        self.done = true;
        ACTIVE.with(|a| a.borrow_mut().take().unwrap().stats)
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        if !self.done {
            ACTIVE.with(|a| {
                a.borrow_mut().take();
            });
        }
    }
}

pub(crate) fn is_active() -> bool {
    ACTIVE.with(|v| v.borrow().is_some())
}

pub(crate) fn label(label: &'static str) {
    ACTIVE.with(|a| {
        if let Some(a) = a.borrow_mut().as_mut() {
            a.label = label;
        }
    });
}
pub(crate) fn skip(name: &str) -> bool {
    ACTIVE.with(|a| {
        let mut a = a.borrow_mut();
        let Some(a) = a.as_mut() else { return false };
        let io = matches!(a.label, "attn_qkv" | "attn_output")
            && (name.contains("gather_columns")
                || name.contains("scatter_columns")
                || name.starts_with("kernel_rms_norm"));
        let post = matches!(
            name,
            "kernel_deepseek_v41_hc_post_controls" | "kernel_deepseek_v41_ffn_post_controls"
        );
        let pre = name.starts_with("kernel_deepseek_v41_hc_pre");
        // Batched dense projections: the skinny MMA Q5_K/Q8_0 pair, plus the
        // solo split equivalents so the same bound works at any width.
        let dense = name.starts_with("kernel_dense_skinny_mma_q5_K")
            || name.starts_with("kernel_dense_pair_skinny_mma_q8_0")
            || name.starts_with("kernel_expert_native_q5_K_f32_split")
            || name.starts_with("kernel_dense_pair_q8_0_f32_splitk");
        let core = name.starts_with("kernel_deepseek_v41_attention_flash");
        let combine = name.starts_with("kernel_deepseek_v41_attention_flash_combine");
        let partial = core && !combine;
        let yes = match a.mode {
            Mode::AttentionIo => io || name == "kernel_deepseek_v41_attention_rank_norms",
            Mode::PostControls => post,
            Mode::AllHc => pre || post,
            Mode::DenseHold => dense,
            Mode::AttentionCoreHold => core,
            Mode::FlashPartialHold => partial,
            Mode::FlashCombineHold => combine,
            _ => false,
        };
        if yes {
            *a.stats.skipped.entry(name.into()).or_default() += 1;
        }
        yes
    })
}
pub(crate) fn elide_norm_phase(n: usize) -> bool {
    ACTIVE.with(|a| {
        let mut a = a.borrow_mut();
        let Some(a) = a.as_mut() else { return false };
        if a.mode == Mode::AttentionIo {
            *a.stats.skipped.entry("rank_norm_phase".into()).or_default() += n;
            true
        } else {
            false
        }
    })
}
pub(crate) fn elide_post_boundary() -> bool {
    ACTIVE.with(|a| {
        a.borrow()
            .as_ref()
            .is_some_and(|a| matches!(a.mode, Mode::PostControls | Mode::AllHc))
    })
}

pub(crate) struct Expert<'a> {
    pub bank: &'a MetalTensor,
    pub bank_b: Option<&'a MetalTensor>,
    pub ids: &'a MetalTensor,
    pub output: &'a MetalTensor,
    pub row_bytes: u64,
    pub n_out: usize,
    pub experts: usize,
    pub channels: usize,
    pub grid: MTLSize,
    pub threads: MTLSize,
}
struct Saved {
    bank: MetalTensor,
    bank_b: Option<MetalTensor>,
    ids: MetalTensor,
    output: MetalTensor,
    held: MetalTensor,
    checksum: MetalTensor,
    row_bytes: u64,
    n_out: usize,
    experts: usize,
    channels: usize,
    grid: MTLSize,
    threads: MTLSize,
}
impl Saved {
    fn bytes(&self) -> u64 {
        self.n_out as u64
            * self.row_bytes
            * (self.grid.height * self.grid.depth) as u64
            * if self.bank_b.is_some() { 2 } else { 1 }
    }
}
/// Wrap one real routed projection. Capture copies run only in the excluded
/// trajectory; read/hold substitutes copy its exact saved output in the same grid.
pub(crate) fn expert(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    r: Expert<'_>,
    original: impl FnOnce(),
) -> Result<(), MetalError> {
    if r.experts <= 1 {
        original();
        return Ok(());
    }
    let active = ACTIVE.with(|a| {
        let mut a = a.borrow_mut();
        a.as_mut().map(|a| {
            let i = a.cursor;
            a.cursor += 1;
            a.stats.expert_calls += 1;
            a.stats.expert_bytes += r.n_out as u64
                * r.row_bytes
                * (r.grid.height * r.grid.depth) as u64
                * if r.bank_b.is_some() { 2 } else { 1 };
            (a.mode, a.book.clone(), i)
        })
    });
    let Some((mode, book, index)) = active else {
        original();
        return Ok(());
    };
    if matches!(mode, Mode::RoutedRead | Mode::RoutedHold) {
        let book = book.borrow();
        let s = book
            .saved
            .get(index)
            .ok_or_else(|| MetalError::Counter("missing expert capture".into()))?;
        if s.n_out != r.n_out
            || s.row_bytes != r.row_bytes
            || s.channels != r.channels
            || s.grid.width != r.grid.width
            || s.grid.height != r.grid.height
            || s.grid.depth != r.grid.depth
        {
            return Err(MetalError::Counter("expert shape changed".into()));
        }
        return encode_stream(ctx, enc, s, mode == Mode::RoutedRead, Some(r.output));
    }
    if mode != Mode::Capture {
        original();
        return Ok(());
    }
    let held = MetalTensor::zeros_dtype_unstaged(ctx, r.output.shape.clone(), GgmlType::F32)?;
    let ids = MetalTensor::zeros_dtype_unstaged(ctx, r.ids.shape.clone(), GgmlType::I32)?;
    let words = r.grid.width
        * r.grid.height
        * r.grid.depth
        * (r.threads.width * r.threads.height * r.threads.depth / 32);
    let checksum = MetalTensor::zeros_dtype_unstaged(ctx, vec![words as u64], GgmlType::I32)?;
    original();
    super::encode_copy_offset_f32(ctx, enc, r.output, 0, &held, r.output.n_elements() as usize)?;
    super::encode_copy_offset_i32(ctx, enc, r.ids, 0, &ids, r.ids.n_elements() as usize)?;
    book.borrow_mut().saved.push(Saved {
        bank: r.bank.clone(),
        bank_b: r.bank_b.cloned(),
        ids,
        output: r.output.clone(),
        held,
        checksum,
        row_bytes: r.row_bytes,
        n_out: r.n_out,
        experts: r.experts,
        channels: r.channels,
        grid: r.grid,
        threads: r.threads,
    });
    Ok(())
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
fn encode_stream(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    s: &Saved,
    read: bool,
    destination: Option<&MetalTensor>,
) -> Result<(), MetalError> {
    enc.set_pipeline(&ctx.pipeline("kernel_research_v41_expert_stream")?);
    enc.set_bytes(
        0,
        &StreamArgs {
            row_bytes: s.row_bytes,
            n_out: s.n_out as u32,
            experts: s.experts as u32,
            channels: s.channels as u32,
            slots: s.grid.height as u32,
            banks: if s.bank_b.is_some() { 2 } else { 1 },
            read: read as u32,
        },
    );
    for (i, t) in [
        &s.bank,
        s.bank_b.as_ref().unwrap_or(&s.bank),
        &s.ids,
        &s.held,
        destination.unwrap_or(&s.output),
        &s.checksum,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(i + 1, t);
    }
    enc.dispatch(s.grid, s.threads);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_bound_is_limited_to_attention_copies_and_norms() {
        let e = Experiment::default();
        let g = e.scope(Mode::AttentionIo).unwrap();
        label("ffn_router");
        assert!(!skip("kernel_deepseek_v41_gather_columns"));
        label("attn_qkv");
        assert!(skip("kernel_deepseek_v41_gather_columns"));
        assert!(skip("kernel_deepseek_v41_attention_rank_norms"));
        assert!(!skip("kernel_dense_skinny_mma_q5_K_f32"));
        let s = g.finish();
        assert_eq!(s.skipped.values().sum::<usize>(), 2);
        assert!(!skip("kernel_deepseek_v41_gather_columns"));
    }
    #[test]
    fn post_bound_preserves_pre_and_restores_on_drop() {
        let e = Experiment::default();
        {
            let _g = e.scope(Mode::PostControls).unwrap();
            assert!(elide_post_boundary());
            assert!(skip("kernel_deepseek_v41_hc_post_controls"));
            assert!(skip("kernel_deepseek_v41_ffn_post_controls"));
            assert!(!skip("kernel_deepseek_v41_hc_pre_batch"));
            assert!(e.scope(Mode::Reference).is_err());
        }
        assert!(!elide_post_boundary());
        assert!(!skip("kernel_deepseek_v41_hc_post_controls"));
    }
    #[test]
    fn reference_is_a_noop_and_all_hc_is_explicit() {
        let e = Experiment::default();
        let g = e.scope(Mode::Reference).unwrap();
        assert!(!skip("kernel_deepseek_v41_hc_pre_batch"));
        assert!(!elide_norm_phase(3));
        g.finish();
        let g = e.scope(Mode::AllHc).unwrap();
        assert!(skip("kernel_deepseek_v41_hc_pre_batch"));
        g.finish();
    }
}

thread_local! { static BATCH_CAUSAL: std::cell::Cell<Option<bool>> = const {std::cell::Cell::new(None)}; }
pub fn batch_causal_override() -> Option<bool> {
    BATCH_CAUSAL.with(std::cell::Cell::get)
}
pub fn batch_causal_scope(enabled: bool) -> impl Drop {
    struct Scope(Option<bool>, std::marker::PhantomData<Rc<()>>);
    impl Drop for Scope {
        fn drop(&mut self) {
            BATCH_CAUSAL.with(|v| v.set(self.0));
        }
    }
    Scope(
        BATCH_CAUSAL.with(|v| v.replace(Some(enabled))),
        std::marker::PhantomData,
    )
}

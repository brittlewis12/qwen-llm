//! Bounded, test-only raw router capture; no routing arithmetic changes.
use super::*;
use std::{cell::RefCell, rc::Rc};

pub(crate) const LAYERS: [u32; 2] = [20, 31];
pub(crate) const EXPERTS: usize = 512;
pub(crate) const TOP_K: usize = 10;

#[derive(Clone)]
pub(crate) struct Bank {
    pub scores: MetalTensor,
    pub ids: MetalTensor,
    pub weights: MetalTensor,
    rows: usize,
}

impl Bank {
    pub(crate) fn new(ctx: &MetalContext, rows: usize) -> Self {
        assert!(matches!(rows, 2 | 4 | 8));
        let shapes = [
            vec![EXPERTS as u64, rows as u64, 2],
            vec![TOP_K as u64, rows as u64, 2],
        ];
        let reserve = crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
        let mut priced = 0;
        for shape in [&shapes[0], &shapes[1], &shapes[1]] {
            let bytes = shape.iter().product::<u64>() * 4;
            priced += crate::metal::price_shared_buffer_upper(
                bytes,
                ctx.shared_buffer_size_and_align(bytes).unwrap(),
                crate::metal::host_page_size_bytes().unwrap() as u64,
                ctx.max_buffer_length() as u64,
            )
            .unwrap()
            .priced_upper_bytes;
        }
        ctx.pipeline("kernel_copy_offset_f32").unwrap();
        ctx.pipeline("kernel_copy_offset_i32").unwrap();
        let _allocation = ctx.begin_allocation_transaction();
        assert!(
            crate::metal::evaluate_metal_memory_admission(
                priced,
                reserve,
                ctx.memory_signals(),
                true
            )
            .is_admitted()
        );
        let bank = Self {
            scores: MetalTensor::zeros_f32(ctx, shapes[0].clone()).unwrap(),
            ids: MetalTensor::zeros_i32(ctx, shapes[1].clone()).unwrap(),
            weights: MetalTensor::zeros_f32(ctx, shapes[1].clone()).unwrap(),
            rows,
        };
        assert!(
            crate::metal::evaluate_metal_memory_admission(0, reserve, ctx.memory_signals(), true)
                .is_admitted()
        );
        bank
    }
}

struct Binding {
    bank: Bank,
    seen: Rc<RefCell<[bool; 2]>>,
}
thread_local! { static ACTIVE: RefCell<Option<Binding>> = const { RefCell::new(None) }; }

pub(crate) fn capture<R>(bank: &Bank, body: impl FnOnce() -> R) -> (R, [bool; 2]) {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
    let seen = Rc::new(RefCell::new([false; 2]));
    ACTIVE.with(|slot| {
        assert!(slot.borrow().is_none(), "raw router capture cannot nest");
        *slot.borrow_mut() = Some(Binding {
            bank: bank.clone(),
            seen: seen.clone(),
        });
    });
    let _restore = Restore;
    let result = body();
    let seen = *seen.borrow();
    (result, seen)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    layer: u32,
    rows: usize,
    profiled: bool,
    scores: &MetalTensor,
    ids: &MetalTensor,
    weights: &MetalTensor,
) -> Result<(), Qwen4ExpMoeError> {
    ACTIVE.with(|slot| {
        let active = slot.borrow();
        let Some(binding) = active.as_ref() else {
            return Ok(());
        };
        if profiled {
            return invalid("raw router capture requires plain packed execution");
        }
        let Some(index) = LAYERS.iter().position(|&l| l == layer) else {
            return Ok(());
        };
        if rows != binding.bank.rows || binding.seen.borrow()[index] {
            return invalid("raw router capture width mismatch or duplicate layer");
        }
        let bank = &binding.bank;
        let score_dst = bank.scores.view_subrange(
            (index * rows * EXPERTS) as u64,
            vec![(rows * EXPERTS) as u64],
        );
        let ids_dst = bank
            .ids
            .view_subrange((index * rows * TOP_K) as u64, vec![(rows * TOP_K) as u64]);
        let weights_dst = bank
            .weights
            .view_subrange((index * rows * TOP_K) as u64, vec![(rows * TOP_K) as u64]);
        if scores.dtype != GgmlType::F32
            || ids.dtype != GgmlType::I32
            || weights.dtype != GgmlType::F32
            || scores.n_elements() != (rows * EXPERTS) as u64
            || ids.n_elements() != (rows * TOP_K) as u64
            || weights.n_elements() != ids.n_elements()
        {
            return invalid("raw router capture source layout mismatch");
        }
        let tensors = [
            ("scores", scores),
            ("ids", ids),
            ("weights", weights),
            ("score capture", &score_dst),
            ("id capture", &ids_dst),
            ("weight capture", &weights_dst),
        ];
        require_same_device(ctx, &tensors)?;
        require_disjoint(&tensors)?;
        encode_copy_offset_f32(ctx, enc, scores, 0, &score_dst, rows * EXPERTS)?;
        encode_copy_offset_i32(ctx, enc, ids, 0, &ids_dst, rows * TOP_K)?;
        encode_copy_offset_f32(ctx, enc, weights, 0, &weights_dst, rows * TOP_K)?;
        binding.seen.borrow_mut()[index] = true;
        Ok(())
    })
}

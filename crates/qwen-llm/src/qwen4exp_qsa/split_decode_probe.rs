use super::*;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) const SCRATCH_FLOATS: usize = 24 * 64 * 258;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub layer: u32,
    pub position: usize,
    pub ids: usize,
    pub split: bool,
}

struct Binding {
    split: bool,
    scratch: MetalTensor,
    records: Rc<RefCell<Vec<Record>>>,
}

thread_local! {
    static BINDING: RefCell<Option<Binding>> = const { RefCell::new(None) };
}

pub(crate) fn with_probe<R>(
    split: bool,
    scratch: &MetalTensor,
    f: impl FnOnce() -> R,
) -> (R, Vec<Record>) {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            BINDING.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
    let records = Rc::new(RefCell::new(Vec::new()));
    BINDING.with(|slot| {
        assert!(slot.borrow().is_none(), "split probes cannot nest");
        *slot.borrow_mut() = Some(Binding {
            split,
            scratch: scratch.clone(),
            records: records.clone(),
        });
    });
    let _restore = Restore;
    let value = f();
    let records = records.borrow().clone();
    (value, records)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    ids: u32,
    capacity: u32,
    splits: u32,
    keys_per_split: u32,
}

pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    position: usize,
    ids: usize,
) -> Result<bool, Qwen4ExpQsaError> {
    BINDING.with(|slot| {
        let binding = slot.borrow();
        let Some(binding) = binding.as_ref() else {
            return Ok(false);
        };
        let g = workspace.geometry;
        assert_eq!((g.query_heads, g.kv_heads, g.head_dim), (24, 2, 256));
        assert!(ids > 0 && ids <= 2051 && position < g.capacity);
        binding.records.borrow_mut().push(Record {
            layer: QWEN4EXP_QSA_CAPTURE_LAYER
                .with(|layer| layer.get().expect("native layer scope")),
            position,
            ids,
            split: binding.split,
        });
        if !binding.split {
            return Ok(false);
        }
        require_tensor(
            "split scratch",
            &binding.scratch,
            GgmlType::F32,
            &[SCRATCH_FLOATS as u64],
            true,
        )?;
        require_same_device(ctx, &[("split scratch", &binding.scratch)])?;
        let mut tensors = workspace_tensors(workspace);
        tensors.push(("split scratch", &binding.scratch));
        require_disjoint(&tensors)?;
        let splits = ids.div_ceil(32).min(64);
        let args = Args {
            ids: ids as u32,
            capacity: g.capacity as u32,
            splits: splits as u32,
            keys_per_split: ids.div_ceil(splits) as u32,
        };
        let pso = ctx.pipeline("kernel_qwen4exp_qsa_split_f16")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, &workspace.query);
        enc.set_tensor(2, &workspace.key_cache);
        enc.set_tensor(3, &workspace.value_cache);
        enc.set_tensor(4, &workspace.token_ids);
        enc.set_tensor(5, &binding.scratch);
        enc.dispatch(
            MTLSize {
                width: splits,
                height: 2,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        let pso = ctx.pipeline("kernel_qwen4exp_qsa_split_merge_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, &binding.scratch);
        enc.set_tensor(2, &workspace.raw_gate);
        enc.set_tensor(3, &workspace.attention);
        enc.dispatch(
            MTLSize {
                width: 24,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok(true)
    })
}

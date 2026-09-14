use super::*;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) use super::split_decode::SCRATCH_FLOATS;

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

pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    position: usize,
    ids: usize,
) -> Result<Option<bool>, Qwen4ExpQsaError> {
    BINDING.with(|slot| {
        let binding = slot.borrow();
        let Some(binding) = binding.as_ref() else {
            return Ok(None);
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
            return Ok(Some(false));
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
        super::split_decode::encode(ctx, enc, workspace, &binding.scratch, ids)?;
        Ok(Some(true))
    })
}

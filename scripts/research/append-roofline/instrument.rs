//! Scratch-only instrumentation. Never installed in the committed library.
use crate::metal::{KernelEncoder, MetalContext, MetalTensor, encode_copy_offset_i32};
use std::cell::RefCell;
thread_local! {
    static ROUTES: RefCell<Option<Vec<(u32, MetalTensor, Vec<u64>)>>> = const { RefCell::new(None) };
    static GPU: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}
pub fn gpu_clear() {
    GPU.with(|x| x.borrow_mut().clear());
}
pub fn gpu_record(ms: f64) {
    GPU.with(|x| x.borrow_mut().push(ms));
}
pub fn gpu_times() -> Vec<f64> {
    GPU.with(|x| x.borrow().clone())
}
pub fn routes_begin() {
    ROUTES.with(|x| {
        assert!(x.borrow().is_none());
        *x.borrow_mut() = Some(Vec::new());
    });
}
pub(crate) fn routes_copy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    layer: u32,
    counts: &MetalTensor,
    banks: [&MetalTensor; 3],
) -> Result<(), crate::metal::MetalError> {
    ROUTES.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(rows) = slot.as_mut() {
            let n = counts.n_elements() as usize;
            let copy = MetalTensor::zeros_i32(ctx, vec![n as u64])?;
            encode_copy_offset_i32(ctx, enc, counts, 0, &copy, n)?;
            rows.push((layer, copy, banks.iter().map(|b| b.n_bytes()).collect()));
        }
        Ok(())
    })
}
pub fn routes_finish(tokens: usize, top: usize) -> serde_json::Value {
    let rows = ROUTES.with(|x| x.borrow_mut().take()).unwrap();
    serde_json::Value::Array(
        rows.into_iter()
            .map(|(layer, t, banks)| {
                let counts = unsafe {
                    std::slice::from_raw_parts(
                        t.buffer
                            .contents()
                            .as_ptr()
                            .cast::<i32>()
                            .add(t.offset as usize / 4),
                        t.n_elements() as usize,
                    )
                };
                assert!(counts.iter().all(|&c| c >= 0));
                assert_eq!(
                    counts.iter().map(|&c| c as usize).sum::<usize>(),
                    tokens * top
                );
                serde_json::json!({"layer":layer,"counts":counts,"bank_bytes":banks})
            })
            .collect(),
    )
}

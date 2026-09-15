use super::*;

thread_local! {
    static PROBE: std::cell::Cell<Option<(bool, usize)>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn with_probe<R>(enabled: bool, f: impl FnOnce() -> R) -> (R, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            PROBE.set(None);
        }
    }
    assert!(PROBE.get().is_none(), "HC decode probes cannot nest");
    PROBE.set(Some((enabled, 0)));
    let _reset = Reset;
    let result = f();
    (result, PROBE.get().unwrap().1)
}

pub(super) fn encode_if_requested(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    up: &MetalTensor,
    scratch: &GatedResidualMetalScratch,
) -> Result<bool, MetalError> {
    let Some((enabled, calls)) = PROBE.get() else {
        return Ok(false);
    };
    assert_eq!(
        (scratch.branch_count, scratch.hidden_size, scratch.low_rank),
        (4, 2560, 320)
    );
    assert_eq!(up.dtype, GgmlType::Q8_0);
    PROBE.set(Some((enabled, calls + 1)));
    if !enabled {
        return Ok(false);
    }
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_up_mix_q8_k320")?;
    assert_eq!(pipeline.threadExecutionWidth(), 32);
    assert!(pipeline.maxTotalThreadsPerThreadgroup() >= 128);
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [
        up,
        &scratch.low,
        &scratch.normalized,
        &scratch.raw_gate,
        &scratch.mixed,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(index, tensor);
    }
    enc.dispatch(
        MTLSize {
            width: 640,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

#[test]
fn scope_restores_after_panic_and_rejects_nesting() {
    let panic = std::panic::catch_unwind(|| with_probe(true, || with_probe(false, || ())));
    assert!(panic.is_err());
    assert_eq!(with_probe(false, || 7), (7, 0));
}

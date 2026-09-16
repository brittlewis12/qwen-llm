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

pub(super) fn route_if_requested(
    dtype: GgmlType,
    scratch: &GatedResidualMetalScratch,
) -> Option<bool> {
    let Some((enabled, calls)) = PROBE.get() else {
        return None;
    };
    assert_eq!(
        (scratch.branch_count, scratch.hidden_size, scratch.low_rank),
        (4, 2560, 320)
    );
    assert_eq!(dtype, GgmlType::Q8_0);
    PROBE.set(Some((enabled, calls + 1)));
    Some(enabled)
}

#[test]
fn scope_restores_after_panic_and_rejects_nesting() {
    let panic = std::panic::catch_unwind(|| with_probe(true, || with_probe(false, || ())));
    assert!(panic.is_err());
    assert_eq!(with_probe(false, || 7), (7, 0));
}

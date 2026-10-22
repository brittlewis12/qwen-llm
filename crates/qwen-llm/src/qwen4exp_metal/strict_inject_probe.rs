use super::*;

thread_local! {
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct Scope(bool, std::marker::PhantomData<std::rc::Rc<()>>);

impl Scope {
    pub(crate) fn new(enabled: bool) -> Self {
        assert_eq!(packet_hc::active_mode(), packet_hc::Mode::Baseline);
        assert!(!qwen4exp_hc_packed_projection_override_active());
        Self(ENABLED.replace(enabled), std::marker::PhantomData)
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        ENABLED.set(self.0);
    }
}

pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<bool, Qwen4ExpMetalError> {
    if !ENABLED.get() || weight.dtype != GgmlType::F32 || n_in != 10240 || n_out != 4 {
        return Ok(false);
    }
    crate::metal::hc4_strict::encode_mat_mat_f32_hc4p32_strict_test(
        ctx, enc, weight, input, output, n_in, n_out, rows,
    )?;
    Ok(true)
}

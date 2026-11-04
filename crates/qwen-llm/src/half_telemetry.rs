//! Debug-only per-site activation telemetry for half-staging sites.
//!
//! Half-staged activations are bounded for DS4, V4.1 and GLM by clamps and
//! norms. The unclamped SwiGLU down inputs in qwen35/qwen35moe/qwen4exp and
//! the V4.1 low-rank `output_b` input have no measured bound. This module
//! records `max|x|` at those sites with an exponent-bit device atomic when
//! `QWEN_HALF_STAGING_TELEMETRY` is set.
//!
//! The recording kernel is dispatched into the caller's encoder immediately
//! before the production half-staged dispatch, so the value is exactly the F32
//! activation the kernel is about to cast. Production dispatch, buffers and
//! kernel selection are unchanged when the flag is off.

use crate::metal::{KernelEncoder, MetalContext, MetalTensor};
use crate::tensor::GgmlType;
use objc2_metal::MTLBuffer;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

crate::env_flag!(
    /// Debug-only per-site max|x| telemetry at half-staging sites. Default off;
    /// production dispatch and bindings are unchanged when unset.
    default_off half_staging_telemetry_enabled,
    "QWEN_HALF_STAGING_TELEMETRY"
);

/// Optional output path for the final telemetry record (JSON lines).
const OUTPUT_ENV: &str = "QWEN_HALF_STAGING_TELEMETRY_OUT";

/// Stable site names. Keep the order; the index is the device buffer slot.
pub const SITES: &[&str] = &[
    "dense.down",
    "qwen35moe.shared.down",
    "qwen35moe.routed.down",
    "qwen4exp.routed.down",
    "qwen4exp.shared.down",
    "v41.output_b.input",
    "v41.routed.down",
];

pub const SITE_DENSE_DOWN: &str = SITES[0];
pub const SITE_QWEN35MOE_SHARED_DOWN: &str = SITES[1];
pub const SITE_QWEN35MOE_ROUTED_DOWN: &str = SITES[2];
pub const SITE_QWEN4EXP_ROUTED_DOWN: &str = SITES[3];
pub const SITE_QWEN4EXP_SHARED_DOWN: &str = SITES[4];
pub const SITE_V41_OUTPUT_B_INPUT: &str = SITES[5];
pub const SITE_V41_ROUTED_DOWN: &str = SITES[6];

struct TelemetryState {
    tensor: SendTensor,
    seen: [bool; SITES.len()],
}

/// The telemetry buffer is a shared Metal buffer; it is safe to read from the
/// dump thread after the run's command buffers have completed.
struct SendTensor(MetalTensor);
unsafe impl Send for SendTensor {}
unsafe impl Sync for SendTensor {}

static STATE: OnceLock<Mutex<Option<TelemetryState>>> = OnceLock::new();

fn state() -> &'static Mutex<Option<TelemetryState>> {
    STATE.get_or_init(|| Mutex::new(None))
}

pub fn enabled() -> bool {
    half_staging_telemetry_enabled()
}

/// Record `max|x|` for one half-staging site into the caller's command buffer.
///
/// No-op unless the debug flag is on and `x` is an F32 tensor.
pub fn record(ctx: &MetalContext, enc: &KernelEncoder, site: &'static str, x: &MetalTensor) {
    if !enabled() || x.dtype != GgmlType::F32 {
        return;
    }
    let Some(site_index) = SITES.iter().position(|&name| name == site) else {
        return;
    };
    let n = x.n_elements();
    let Ok(n) = u32::try_from(n) else {
        return;
    };
    if n == 0 {
        return;
    }
    let Ok(pso) = ctx.pipeline("kernel_half_staging_amax") else {
        return;
    };
    let mut guard = match state().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        let Ok(tensor) = MetalTensor::zeros_i32(ctx, vec![SITES.len() as u64]) else {
            return;
        };
        *guard = Some(TelemetryState {
            tensor: SendTensor(tensor),
            seen: [false; SITES.len()],
        });
    }
    let Some(state) = guard.as_mut() else {
        return;
    };
    state.seen[site_index] = true;
    // Cap the grid and let each thread walk a strided slice. The kernel
    // reduces per simdgroup/threadgroup before the single device atomic.
    const MAX_TELEMETRY_THREADS: u32 = 65_536;
    let threads = n.min(MAX_TELEMETRY_THREADS).div_ceil(256) * 256;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &n);
    enc.set_tensor(1, x);
    enc.set_tensor(2, &state.tensor.0);
    enc.set_bytes(3, &(site_index as u32));
    enc.set_bytes(4, &threads);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: (threads / 256) as usize,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Print the recorded peaks and optionally write them to `QWEN_HALF_STAGING_TELEMETRY_OUT`.
///
/// Call after the model run has committed and waited for its command buffers.
pub fn dump_if_enabled() {
    if !enabled() {
        return;
    }
    let guard = match state().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let Some(state) = guard.as_ref() else {
        return;
    };
    let values: &[u32] = unsafe {
        std::slice::from_raw_parts(
            state.tensor.0.buffer.contents().as_ptr().cast::<u32>(),
            SITES.len(),
        )
    };
    let mut lines = BTreeMap::new();
    for (index, &name) in SITES.iter().enumerate() {
        if !state.seen[index] {
            continue;
        }
        let bits = values[index];
        let value = f32::from_bits(bits);
        let line = format!(
            "half-staging-telemetry site={name} max_abs={value:.9e} bits=0x{bits:08x} limit={:.0}",
            crate::half_range::RMS_NORM_MAX
        );
        eprintln!("{line}");
        lines.insert(name, line);
    }
    if let Some(path) = std::env::var_os(OUTPUT_ENV) {
        let mut body = String::new();
        for line in lines.values() {
            body.push_str(line);
            body.push('\n');
        }
        let _ = std::fs::write(path, body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_names_are_unique_and_stable() {
        let mut seen = std::collections::BTreeSet::new();
        for &site in SITES {
            assert!(seen.insert(site), "duplicate site {site}");
        }
        assert_eq!(SITE_DENSE_DOWN, "dense.down");
        assert_eq!(SITE_V41_OUTPUT_B_INPUT, "v41.output_b.input");
    }
}

//! Gated DeltaNet and SSM convolution kernels.

use super::*;

/// GDN α-chain fusion: fused (a + dt_bias), softplus, then mul by a_log.
///
/// Replaces 3 dispatches per GDN layer:
///   encode_add_inplace_f32(a, dt_bias)
///   encode_softplus_f32(a → out)
///   encode_mul_f32(out, a_log → out)
///
/// Saves 2 dispatches × 32 layers = 64 dispatches/token. Per the v0.25
/// intra-profiler the chain measured ~0.04 ms/layer; this should shave
/// 1.5–2.5 ms/token (Codex predicted 2–3 ms upper bound).
///
/// CPU oracle: `softplus(a + dt_bias) * a_log` from `crate::forward`.
pub fn encode_gdn_alpha_chain_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    let n = a.n_elements() as usize;
    if dt_bias.n_elements() as usize != n
        || a_log.n_elements() as usize != n
        || out.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain",
            detail: format!(
                "lengths a={} dt={} alog={} out={n}",
                a.n_elements(),
                dt_bias.n_elements(),
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_alpha_chain_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// GDN decay-chain fusion: `exp(softplus(a + dt_bias) * a_log)`.
///
/// The standard alpha-chain writes the log-decay `g`; the GDN recurrence
/// then needs `exp(g)` for every state row. This variant writes the per-head
/// decay once so `kernel_gdn_step_decay_f32` can reuse it across all rows.
pub fn encode_gdn_decay_chain_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    out: &MetalTensor,
) -> Result<(), MetalError> {
    let n = a.n_elements() as usize;
    if dt_bias.n_elements() as usize != n
        || a_log.n_elements() as usize != n
        || out.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain",
            detail: format!(
                "lengths a={} dt={} alog={} out={n}",
                a.n_elements(),
                dt_bias.n_elements(),
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_decay_chain_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Activation VJP for `decay = exp(softplus(a + dt_bias) * a_log)`.
///
/// `dt_bias` and transformed `a_log` are frozen model parameters. The saved
/// direct `decay` is used to chain through the exact forward primal.
pub fn encode_gdn_decay_chain_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    decay: &MetalTensor,
    grad_decay: &MetalTensor,
    grad_a: &MetalTensor,
) -> Result<(), MetalError> {
    const KERNEL: &str = "gdn_decay_chain_vjp";
    let input_shape = a.shape.clone();
    let (n, _) = checked_shape_bytes(&input_shape, std::mem::size_of::<f32>())?;
    if n == 0 || u32::try_from(n).is_err() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("element count {n} must fit nonzero u32"),
        });
    }
    let shape = vec![n as u64];
    if input_shape != shape {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("a must be one compact row, got {input_shape:?}"),
        });
    }
    for (name, tensor) in [
        ("a", a),
        ("dt_bias", dt_bias),
        ("a_log", a_log),
        ("decay", decay),
        ("grad_decay", grad_decay),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, &shape, false)?;
    }
    validate_compact_f32_tensor(KERNEL, "grad_a", grad_a, &shape, true)?;
    if [a, dt_bias, a_log, decay, grad_decay]
        .iter()
        .any(|input| tensor_ranges_overlap(grad_a, input))
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "grad_a must not overlap any input".into(),
        });
    }
    let pipeline = ctx.pipeline("kernel_gdn_decay_chain_vjp_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(0, &NArgs { n: n as u32 });
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, decay);
    enc.set_tensor(5, grad_decay);
    enc.set_tensor(6, grad_a);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Batched GDN α-chain (v0.73a layer-major batching).
///
/// Computes `out[r, c] = softplus(a[r, c] + dt_bias[c]) * a_log[c]` over
/// `[N, n_v]` row-major `a` / `out` with `[n_v]` `dt_bias` / `a_log`
/// broadcast across the N rows.
///
/// Bit-identical to N successive calls of `encode_gdn_alpha_chain_f32`
/// over each row of `a` (validated by
/// `gdn_alpha_chain_batched_matches_per_row` test).
///
/// Used by the layer-major packed_verify GDN batching (v0.73a) to lift
/// the per-token alpha-chain dispatch out of the inner N loop alongside
/// in_proj_qkv / in_proj_z / beta_proj / alpha_proj. dt_bias and a_log
/// are layer-shared GGUF weights (`[n_v]` F32); a and out are
/// `MetalDFlashLayerMajorScratch::gdn_a_pack` / `gdn_alpha_pack`
/// (`[N, n_v]`).
pub fn encode_gdn_alpha_chain_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,       // [N, n_v] row-major
    dt_bias: &MetalTensor, // [n_v]
    a_log: &MetalTensor,   // [n_v]
    out: &MetalTensor,     // [N, n_v] row-major
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    let n = n_rows * n_cols;
    if a.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "a expected {n_rows}*{n_cols}={n} elements, got {}",
                a.n_elements()
            ),
        });
    }
    if out.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "out expected {n_rows}*{n_cols}={n} elements, got {}",
                out.n_elements()
            ),
        });
    }
    if dt_bias.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "dt_bias expected {n_cols} elements, got {}",
                dt_bias.n_elements()
            ),
        });
    }
    if a_log.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_alpha_chain_batched",
            detail: format!(
                "a_log expected {n_cols} elements, got {}",
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_alpha_chain_batched_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        n_cols: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            n_cols: n_cols as u32,
        },
    );
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_gdn_decay_chain_batched_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    a: &MetalTensor,
    dt_bias: &MetalTensor,
    a_log: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), MetalError> {
    let n = n_rows * n_cols;
    if a.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "a expected {n_rows}*{n_cols}={n} elements, got {}",
                a.n_elements()
            ),
        });
    }
    if out.n_elements() as usize != n {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "out expected {n_rows}*{n_cols}={n} elements, got {}",
                out.n_elements()
            ),
        });
    }
    if dt_bias.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "dt_bias expected {n_cols} elements, got {}",
                dt_bias.n_elements()
            ),
        });
    }
    if a_log.n_elements() as usize != n_cols {
        return Err(MetalError::BadShape {
            kernel: "gdn_decay_chain_batched",
            detail: format!(
                "a_log expected {n_cols} elements, got {}",
                a_log.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_gdn_decay_chain_batched_f32")?;
    enc.set_pipeline(&pso);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        n_cols: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n: n as u32,
            n_cols: n_cols as u32,
        },
    );
    enc.set_tensor(1, a);
    enc.set_tensor(2, dt_bias);
    enc.set_tensor(3, a_log);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// SSM conv1d step + SiLU. Per-channel depthwise convolution of width K
/// (=4 for Qwen3.5/3.6), then SiLU. Mutates `conv_buf` (slides time
/// window). See `kernels/ssm_conv.metal` for layout details.
///
/// CPU oracle: the conv block in `forward::Forward::gdn_step` (lines
/// ~358-400 of forward.rs).
pub fn encode_ssm_conv_silu_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_now: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    out: &MetalTensor,
    conv_dim: usize,
) -> Result<(), MetalError> {
    if qkv_now.n_elements() as usize != conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!("qkv_now.n={} != conv_dim={conv_dim}", qkv_now.n_elements()),
        });
    }
    if out.n_elements() as usize != conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!("out.n={} != conv_dim", out.n_elements()),
        });
    }
    // conv_buf must be (K-1) * conv_dim
    if conv_buf.n_elements() as usize != 3 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!(
                "conv_buf.n={} != (K-1)*conv_dim={}",
                conv_buf.n_elements(),
                3 * conv_dim
            ),
        });
    }
    // conv_w must be conv_dim * K
    if conv_w.n_elements() as usize != 4 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "ssm_conv",
            detail: format!(
                "conv_w.n={} != K*conv_dim={}",
                conv_w.n_elements(),
                4 * conv_dim
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        conv_dim: u32,
    }
    let pso = ctx.pipeline("kernel_ssm_conv_silu_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            conv_dim: conv_dim as u32,
        },
    );
    enc.set_tensor(1, qkv_now);
    enc.set_tensor(2, conv_buf);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = conv_dim.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Activation VJP for an immutable width-4 depthwise conv+SiLU step and its
/// shifted three-row causal state.
#[allow(clippy::too_many_arguments)]
pub fn encode_ssm_conv_silu_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_now: &MetalTensor,
    conv_state_in: &MetalTensor,
    conv_w: &MetalTensor,
    grad_out: &MetalTensor,
    grad_state_out: &MetalTensor,
    grad_qkv: &MetalTensor,
    grad_state_in: &MetalTensor,
    conv_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "ssm_conv_silu_vjp";
    if conv_dim == 0 || u32::try_from(conv_dim).is_err() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("conv_dim {conv_dim} must fit nonzero u32"),
        });
    }
    let state_elements = conv_dim
        .checked_mul(3)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "state element count overflow".into(),
        })?;
    let weight_elements = conv_dim
        .checked_mul(4)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "weight element count overflow".into(),
        })?;
    let vector_shape = vec![conv_dim as u64];
    let state_shape = vec![state_elements as u64];
    let weight_shape = vec![weight_elements as u64];
    for (name, tensor, shape) in [
        ("qkv_now", qkv_now, &vector_shape),
        ("conv_state_in", conv_state_in, &state_shape),
        ("conv_w", conv_w, &weight_shape),
        ("grad_out", grad_out, &vector_shape),
        ("grad_state_out", grad_state_out, &state_shape),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, shape, false)?;
    }
    validate_compact_f32_tensor(KERNEL, "grad_qkv", grad_qkv, &vector_shape, true)?;
    validate_compact_f32_tensor(KERNEL, "grad_state_in", grad_state_in, &state_shape, true)?;
    let inputs = [qkv_now, conv_state_in, conv_w, grad_out, grad_state_out];
    if inputs
        .iter()
        .any(|input| tensor_ranges_overlap(grad_qkv, input))
        || inputs
            .iter()
            .any(|input| tensor_ranges_overlap(grad_state_in, input))
        || tensor_ranges_overlap(grad_qkv, grad_state_in)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "gradient outputs must not overlap inputs or each other".into(),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        conv_dim: u32,
    }
    let pipeline = ctx.pipeline("kernel_ssm_conv_silu_vjp_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            conv_dim: conv_dim as u32,
        },
    );
    enc.set_tensor(1, qkv_now);
    enc.set_tensor(2, conv_state_in);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, grad_out);
    enc.set_tensor(5, grad_state_out);
    enc.set_tensor(6, grad_qkv);
    enc.set_tensor(7, grad_state_in);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: conv_dim.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Conv+SiLU/state VJP consuming separate raw-Q, raw-K, and V cotangents.
#[allow(clippy::too_many_arguments)]
pub fn encode_ssm_conv_silu_split_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_now: &MetalTensor,
    conv_state_in: &MetalTensor,
    conv_w: &MetalTensor,
    grad_q_raw: &MetalTensor,
    grad_k_raw: &MetalTensor,
    grad_v: &MetalTensor,
    grad_state_out: &MetalTensor,
    grad_qkv: &MetalTensor,
    grad_state_in: &MetalTensor,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "ssm_conv_silu_split_vjp";
    if n_k_heads == 0
        || n_v_heads == 0
        || head_dim == 0
        || u32::try_from(n_k_heads).is_err()
        || u32::try_from(n_v_heads).is_err()
        || u32::try_from(head_dim).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "head counts and dimension must fit nonzero u32, got n_k={n_k_heads} n_v={n_v_heads} dim={head_dim}"
            ),
        });
    }
    let qk_elements = n_k_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "Q/K element count overflow".into(),
        })?;
    let v_elements = n_v_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "V element count overflow".into(),
        })?;
    let conv_dim = qk_elements
        .checked_mul(2)
        .and_then(|value| value.checked_add(v_elements))
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "convolution dimension overflow".into(),
        })?;
    if u32::try_from(conv_dim).is_err() || u32::try_from(qk_elements).is_err() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "convolution geometry exceeds u32 shader addressing".into(),
        });
    }
    let state_elements = conv_dim
        .checked_mul(3)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "state element count overflow".into(),
        })?;
    let weight_elements = conv_dim
        .checked_mul(4)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "weight element count overflow".into(),
        })?;
    let qk_shape = vec![qk_elements as u64];
    let v_shape = vec![v_elements as u64];
    let conv_shape = vec![conv_dim as u64];
    let state_shape = vec![state_elements as u64];
    let weight_shape = vec![weight_elements as u64];
    for (name, tensor, shape) in [
        ("qkv_now", qkv_now, &conv_shape),
        ("conv_state_in", conv_state_in, &state_shape),
        ("conv_w", conv_w, &weight_shape),
        ("grad_q_raw", grad_q_raw, &qk_shape),
        ("grad_k_raw", grad_k_raw, &qk_shape),
        ("grad_v", grad_v, &v_shape),
        ("grad_state_out", grad_state_out, &state_shape),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, shape, false)?;
    }
    validate_compact_f32_tensor(KERNEL, "grad_qkv", grad_qkv, &conv_shape, true)?;
    validate_compact_f32_tensor(KERNEL, "grad_state_in", grad_state_in, &state_shape, true)?;
    let inputs = [
        qkv_now,
        conv_state_in,
        conv_w,
        grad_q_raw,
        grad_k_raw,
        grad_v,
        grad_state_out,
    ];
    if inputs
        .iter()
        .any(|input| tensor_ranges_overlap(grad_qkv, input))
        || inputs
            .iter()
            .any(|input| tensor_ranges_overlap(grad_state_in, input))
        || tensor_ranges_overlap(grad_qkv, grad_state_in)
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "gradient outputs must not overlap inputs or each other".into(),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        conv_dim: u32,
        qk_dim: u32,
    }
    let pipeline = ctx.pipeline("kernel_ssm_conv_silu_split_vjp_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            conv_dim: conv_dim as u32,
            qk_dim: qk_elements as u32,
        },
    );
    enc.set_tensor(1, qkv_now);
    enc.set_tensor(2, conv_state_in);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, grad_q_raw);
    enc.set_tensor(5, grad_k_raw);
    enc.set_tensor(6, grad_v);
    enc.set_tensor(7, grad_state_out);
    enc.set_tensor(8, grad_qkv);
    enc.set_tensor(9, grad_state_in);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: conv_dim.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Temporal VJP for a packed width-4 depthwise conv+SiLU sequence.
///
/// Checkpoint row `t` is the post-token state after token `t`. Reverse token
/// `t` therefore reads `initial_state` when `t == 0` and checkpoint `t - 1`
/// otherwise. Both the full `n_tokens` checkpoint tape and the production
/// `n_tokens - 1` tape that omits the unused final state are accepted.
#[allow(clippy::too_many_arguments)]
pub fn encode_ssm_conv_silu_split_packed_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_pack: &MetalTensor,
    initial_state: &MetalTensor,
    state_checkpoints: &MetalTensor,
    n_checkpoints: usize,
    conv_w: &MetalTensor,
    grad_q_raw_pack: &MetalTensor,
    grad_k_raw_pack: &MetalTensor,
    grad_v_pack: &MetalTensor,
    grad_final_state: &MetalTensor,
    grad_qkv_pack: &MetalTensor,
    grad_initial_state: &MetalTensor,
    grad_state_scratch_a: &MetalTensor,
    grad_state_scratch_b: &MetalTensor,
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "ssm_conv_silu_split_packed_vjp";
    if enc.is_concurrent() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "temporal state dependencies require a serial encoder".into(),
        });
    }
    if n_tokens == 0
        || u32::try_from(n_tokens).is_err()
        || n_k_heads == 0
        || n_v_heads == 0
        || head_dim == 0
        || u32::try_from(n_k_heads).is_err()
        || u32::try_from(n_v_heads).is_err()
        || u32::try_from(head_dim).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "token/head geometry must fit nonzero u32, got tokens={n_tokens} n_k={n_k_heads} n_v={n_v_heads} head_dim={head_dim}"
            ),
        });
    }
    if n_checkpoints < n_tokens - 1 || n_checkpoints > n_tokens {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected n_tokens-1 or n_tokens post-state checkpoints, got tokens={n_tokens} checkpoints={n_checkpoints}"
            ),
        });
    }
    let qk_elements = n_k_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "Q/K element count overflow".into(),
        })?;
    let v_elements = n_v_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "V element count overflow".into(),
        })?;
    let conv_dim = qk_elements
        .checked_mul(2)
        .and_then(|value| value.checked_add(v_elements))
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "convolution dimension overflow".into(),
        })?;
    if u32::try_from(conv_dim).is_err() || u32::try_from(qk_elements).is_err() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "convolution geometry exceeds u32 shader addressing".into(),
        });
    }
    let state_elements = conv_dim
        .checked_mul(3)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "convolution state element count overflow".into(),
        })?;
    let weight_elements = conv_dim
        .checked_mul(4)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "convolution weight element count overflow".into(),
        })?;
    let packed = |per_token: usize, name: &str| {
        n_tokens
            .checked_mul(per_token)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: format!("{name} packed element count overflow"),
            })
    };
    let qkv_pack_elements = packed(conv_dim, "QKV")?;
    let qk_pack_elements = packed(qk_elements, "Q/K gradient")?;
    let v_pack_elements = packed(v_elements, "V gradient")?;
    let checkpoint_elements =
        n_checkpoints
            .checked_mul(state_elements)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "checkpoint element count overflow".into(),
            })?;
    let shape = |elements: usize| -> Result<Vec<u64>, MetalError> {
        Ok(vec![u64::try_from(elements).map_err(|_| {
            MetalError::BadShape {
                kernel: KERNEL,
                detail: format!("element count {elements} does not fit u64"),
            }
        })?])
    };
    let qkv_pack_shape = shape(qkv_pack_elements)?;
    let qk_pack_shape = shape(qk_pack_elements)?;
    let v_pack_shape = shape(v_pack_elements)?;
    let state_shape = shape(state_elements)?;
    let checkpoint_shape = shape(checkpoint_elements)?;
    let weight_shape = shape(weight_elements)?;
    for (name, tensor, expected_shape) in [
        ("qkv_pack", qkv_pack, &qkv_pack_shape),
        ("initial_state", initial_state, &state_shape),
        ("state_checkpoints", state_checkpoints, &checkpoint_shape),
        ("conv_w", conv_w, &weight_shape),
        ("grad_q_raw_pack", grad_q_raw_pack, &qk_pack_shape),
        ("grad_k_raw_pack", grad_k_raw_pack, &qk_pack_shape),
        ("grad_v_pack", grad_v_pack, &v_pack_shape),
        ("grad_final_state", grad_final_state, &state_shape),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, expected_shape, false)?;
    }
    for (name, tensor, expected_shape) in [
        ("grad_qkv_pack", grad_qkv_pack, &qkv_pack_shape),
        ("grad_initial_state", grad_initial_state, &state_shape),
        ("grad_state_scratch_a", grad_state_scratch_a, &state_shape),
        ("grad_state_scratch_b", grad_state_scratch_b, &state_shape),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, expected_shape, true)?;
    }
    validate_vjp_storage_disjoint(
        KERNEL,
        &[
            qkv_pack,
            initial_state,
            state_checkpoints,
            conv_w,
            grad_q_raw_pack,
            grad_k_raw_pack,
            grad_v_pack,
            grad_final_state,
        ],
        &[
            grad_qkv_pack,
            grad_initial_state,
            grad_state_scratch_a,
            grad_state_scratch_b,
        ],
    )?;

    let mut current_grad_state = grad_final_state;
    for token in (0..n_tokens).rev() {
        let qkv = qkv_pack.view_subrange((token * conv_dim) as u64, vec![conv_dim as u64]);
        let grad_q_raw =
            grad_q_raw_pack.view_subrange((token * qk_elements) as u64, vec![qk_elements as u64]);
        let grad_k_raw =
            grad_k_raw_pack.view_subrange((token * qk_elements) as u64, vec![qk_elements as u64]);
        let grad_v =
            grad_v_pack.view_subrange((token * v_elements) as u64, vec![v_elements as u64]);
        let grad_qkv =
            grad_qkv_pack.view_subrange((token * conv_dim) as u64, vec![conv_dim as u64]);
        let checkpoint = (token > 0).then(|| {
            state_checkpoints.view_subrange(
                ((token - 1) * state_elements) as u64,
                vec![state_elements as u64],
            )
        });
        let state_in = checkpoint.as_ref().unwrap_or(initial_state);
        let reverse_index = n_tokens - 1 - token;
        let next_grad_state = if token == 0 {
            grad_initial_state
        } else if reverse_index.is_multiple_of(2) {
            grad_state_scratch_a
        } else {
            grad_state_scratch_b
        };
        encode_ssm_conv_silu_split_vjp_f32(
            ctx,
            enc,
            &qkv,
            state_in,
            conv_w,
            &grad_q_raw,
            &grad_k_raw,
            &grad_v,
            current_grad_state,
            &grad_qkv,
            next_grad_state,
            n_k_heads,
            n_v_heads,
            head_dim,
        )?;
        current_grad_state = next_grad_state;
    }
    Ok(())
}

crate::env_flag!(
    default_off configured_prefill_gdn_prep_parallel_enabled,
    "QWEN_PREFILL_GDN_PREP_PARALLEL"
);

#[cfg(test)]
thread_local! {
    static PREFILL_GDN_PREP_PARALLEL_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_prefill_gdn_prep_parallel_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> R {
    struct RestoreOverride(Option<bool>);

    impl Drop for RestoreOverride {
        fn drop(&mut self) {
            PREFILL_GDN_PREP_PARALLEL_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }

    let previous = PREFILL_GDN_PREP_PARALLEL_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let _restore = RestoreOverride(previous);
    f()
}

pub(crate) fn prefill_gdn_prep_parallel_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = PREFILL_GDN_PREP_PARALLEL_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    configured_prefill_gdn_prep_parallel_enabled()
}

#[cfg(test)]
pub(crate) fn prefill_gdn_prep_parallel_enabled_for_test() -> bool {
    prefill_gdn_prep_parallel_enabled()
}

#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_prep_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_pack: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    encode_gdn_prep_packed_f32_inner(
        ctx, enc, qkv_pack, conv_buf, conv_w, q_pack, k_pack, v_pack, n_tokens, n_k_heads,
        n_v_heads, head_dim, true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_prep_packed_serial_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_pack: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    encode_gdn_prep_packed_f32_inner(
        ctx, enc, qkv_pack, conv_buf, conv_w, q_pack, k_pack, v_pack, n_tokens, n_k_heads,
        n_v_heads, head_dim, false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_prep_packed_f32_inner(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_pack: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
    allow_parallel: bool,
) -> Result<(), MetalError> {
    let qk_dim = n_k_heads * head_dim;
    let v_dim = n_v_heads * head_dim;
    let conv_dim = (2 * n_k_heads + n_v_heads) * head_dim;
    if qkv_pack.n_elements() as usize != n_tokens * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("qkv_pack expected {} elements", n_tokens * conv_dim),
        });
    }
    if conv_buf.n_elements() as usize != 3 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("conv_buf expected {} elements", 3 * conv_dim),
        });
    }
    if conv_w.n_elements() as usize != 4 * conv_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("conv_w expected {} elements", 4 * conv_dim),
        });
    }
    if q_pack.n_elements() as usize != n_tokens * qk_dim
        || k_pack.n_elements() as usize != n_tokens * qk_dim
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("q/k pack expected {} elements", n_tokens * qk_dim),
        });
    }
    if v_pack.n_elements() as usize != n_tokens * v_dim {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed",
            detail: format!("v_pack expected {} elements", n_tokens * v_dim),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_dim: u32,
    }
    let use_parallel = allow_parallel && prefill_gdn_prep_parallel_enabled();
    if use_parallel && n_tokens >= 3 {
        let args = Args {
            n_tokens: n_tokens as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
            conv_dim: conv_dim as u32,
        };
        let pso = ctx.pipeline("kernel_gdn_prep_parallel_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, qkv_pack);
        enc.set_tensor(2, conv_buf);
        enc.set_tensor(3, conv_w);
        enc.set_tensor(4, q_pack);
        enc.set_tensor(5, k_pack);
        enc.set_tensor(6, v_pack);
        let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(256);
        let total = n_tokens * conv_dim;
        let n_tg = total.div_ceil(tg_threads);
        enc.dispatch(
            MTLSize {
                width: n_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_threads,
                height: 1,
                depth: 1,
            },
        );

        let pso = ctx.pipeline("kernel_gdn_prep_parallel_state_f32")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, qkv_pack);
        enc.set_tensor(2, conv_buf);
        let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(256);
        let total = 3 * conv_dim;
        let n_tg = total.div_ceil(tg_threads);
        enc.dispatch(
            MTLSize {
                width: n_tg,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_threads,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }
    let pso = ctx.pipeline("kernel_gdn_prep_packed_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
            conv_dim: conv_dim as u32,
        },
    );
    enc.set_tensor(1, qkv_pack);
    enc.set_tensor(2, conv_buf);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, q_pack);
    enc.set_tensor(5, k_pack);
    enc.set_tensor(6, v_pack);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = conv_dim.div_ceil(tg_threads);
    enc.dispatch(
        MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_prep_packed_ckpt_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    qkv_pack: &MetalTensor,
    conv_buf: &MetalTensor,
    conv_w: &MetalTensor,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    conv_ckpt: &MetalTensor,
    n_tokens: usize,
    n_checkpoints: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    let qk_dim = n_k_heads * head_dim;
    let v_dim = n_v_heads * head_dim;
    let conv_dim = (2 * n_k_heads + n_v_heads) * head_dim;
    if n_checkpoints > n_tokens
        || qkv_pack.n_elements() as usize != n_tokens * conv_dim
        || conv_buf.n_elements() as usize != 3 * conv_dim
        || conv_w.n_elements() as usize != 4 * conv_dim
        || q_pack.n_elements() as usize != n_tokens * qk_dim
        || k_pack.n_elements() as usize != n_tokens * qk_dim
        || v_pack.n_elements() as usize != n_tokens * v_dim
        || conv_ckpt.n_elements() as usize != n_checkpoints * 3 * conv_dim
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_prep_packed_ckpt",
            detail: format!(
                "invalid packed checkpoint geometry: tokens={n_tokens} checkpoints={n_checkpoints} conv_dim={conv_dim}"
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_checkpoints: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_dim: u32,
    }
    let pso = ctx.pipeline("kernel_gdn_prep_packed_ckpt_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens as u32,
            n_checkpoints: n_checkpoints as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
            conv_dim: conv_dim as u32,
        },
    );
    enc.set_tensor(1, qkv_pack);
    enc.set_tensor(2, conv_buf);
    enc.set_tensor(3, conv_w);
    enc.set_tensor(4, q_pack);
    enc.set_tensor(5, k_pack);
    enc.set_tensor(6, v_pack);
    enc.set_tensor(7, conv_ckpt);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: conv_dim.div_ceil(tg_threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Single-step GDN recurrence: per-V-head delta-rule update + output.
///
/// Performs (per V-head) all of:
///   * decay:  S ← exp(g) · S
///   * inner:  s_k = S · k
///   * delta:  Δ = (v − s_k) · β
///   * update: S += Δ ⊗ k
///   * output: o = S · q
///
/// The usual `1 / √head_dim` factor is folded exactly into the following
/// RMSNormGated epsilon, so this kernel leaves the output unscaled.
///
/// All in one kernel, with S held in registers across the (decay → inner
/// → update → output) sequence. State is read from / written back to
/// `state`; the rest are read-only (one timestep per call). For multi-
/// token prefill we'd loop the recurrence inside the kernel; v1 is
/// single-token decode, so T=1.
///
/// Hardcoded for `head_dim = 128` (Qwen3.5/3.6 GDN). When that changes,
/// the kernel needs templating on `dks_per_lane = head_dim / 32`.
///
/// CPU oracle: per-V-head loop in `crate::forward::Forward::gdn_step`.
pub fn encode_gdn_step_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    g: &MetalTensor,
    beta: &MetalTensor,
    state: &MetalTensor,
    out: &MetalTensor,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if head_dim != 128 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("head_dim={head_dim} but kernel hardcodes 128"),
        });
    }
    if !n_v_heads.is_multiple_of(n_k_heads) {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("n_v_heads={n_v_heads} not multiple of n_k_heads={n_k_heads}"),
        });
    }
    let want_qk = (n_k_heads * head_dim) as u64;
    let want_v = (n_v_heads * head_dim) as u64;
    if q.n_elements() != want_qk || k.n_elements() != want_qk {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("q/k expected {want_qk} elements (n_k_heads={n_k_heads})"),
        });
    }
    if v.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("v expected {want_v} elements (n_v_heads={n_v_heads})"),
        });
    }
    if g.n_elements() != n_v_heads as u64 || beta.n_elements() != n_v_heads as u64 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("g/beta expected {n_v_heads} elements"),
        });
    }
    let want_state = (n_v_heads * head_dim * head_dim) as u64;
    if state.n_elements() != want_state {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("state expected {want_state} elements"),
        });
    }
    if out.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step",
            detail: format!("out expected {want_v} elements"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let pso = ctx.pipeline("kernel_gdn_step_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_v_heads: n_v_heads as u32,
            n_k_heads: n_k_heads as u32,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, v);
    enc.set_tensor(4, g);
    enc.set_tensor(5, beta);
    enc.set_tensor(6, state);
    enc.set_tensor(7, out);

    // 2D grid: (head_dim, n_v_heads). One simdgroup (32 threads) per (dv, hi).
    enc.dispatch(
        MTLSize {
            width: head_dim,
            height: n_v_heads,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// GDN recurrence variant that takes precomputed `decay = exp(g)`.
/// Output is unscaled; `1 / sqrt(head_dim)` is folded into RMSNormGated.
pub fn encode_gdn_step_decay_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    decay: &MetalTensor,
    beta: &MetalTensor,
    state: &MetalTensor,
    out: &MetalTensor,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if head_dim != 128 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("head_dim={head_dim} but kernel hardcodes 128"),
        });
    }
    if !n_v_heads.is_multiple_of(n_k_heads) {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("n_v_heads={n_v_heads} not multiple of n_k_heads={n_k_heads}"),
        });
    }
    let want_qk = (n_k_heads * head_dim) as u64;
    let want_v = (n_v_heads * head_dim) as u64;
    if q.n_elements() != want_qk || k.n_elements() != want_qk {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("q/k expected {want_qk} elements (n_k_heads={n_k_heads})"),
        });
    }
    if v.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("v expected {want_v} elements (n_v_heads={n_v_heads})"),
        });
    }
    if decay.n_elements() != n_v_heads as u64 || beta.n_elements() != n_v_heads as u64 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("decay/beta expected {n_v_heads} elements"),
        });
    }
    let want_state = (n_v_heads * head_dim * head_dim) as u64;
    if state.n_elements() != want_state {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("state expected {want_state} elements"),
        });
    }
    if out.n_elements() != want_v {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay",
            detail: format!("out expected {want_v} elements"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let pso = ctx.pipeline("kernel_gdn_step_decay_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_v_heads: n_v_heads as u32,
            n_k_heads: n_k_heads as u32,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, v);
    enc.set_tensor(4, decay);
    enc.set_tensor(5, beta);
    enc.set_tensor(6, state);
    enc.set_tensor(7, out);

    enc.dispatch(
        MTLSize {
            width: head_dim,
            height: n_v_heads,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Activation VJP for one direct-decay GDN recurrence step.
///
/// The immutable `state_in` is the state before the step. Cotangents may enter
/// through both the recurrence output and the post-step state. Q/K gradients
/// are accumulated over every V head mapped by `hi % n_k_heads`; all output
/// tensors and the two row scratch tensors must be distinct writable F32
/// storage. This first research primitive is fixed to `head_dim = 128` and one
/// cotangent query.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_step_decay_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    decay: &MetalTensor,
    beta: &MetalTensor,
    state_in: &MetalTensor,
    grad_out: &MetalTensor,
    grad_state_out: &MetalTensor,
    grad_q: &MetalTensor,
    grad_k: &MetalTensor,
    grad_v: &MetalTensor,
    grad_decay: &MetalTensor,
    grad_beta: &MetalTensor,
    grad_state_in: &MetalTensor,
    grad_correction_scratch: &MetalTensor,
    residual_scratch: &MetalTensor,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "gdn_step_decay_vjp";
    if enc.is_concurrent() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "dependent VJP dispatches require a serial encoder".into(),
        });
    }
    if head_dim != 128
        || n_v_heads == 0
        || n_k_heads == 0
        || !n_v_heads.is_multiple_of(n_k_heads)
        || u32::try_from(n_v_heads).is_err()
        || u32::try_from(n_k_heads).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected head_dim=128 and nonzero u32 head counts with n_v divisible by n_k, got n_v={n_v_heads} n_k={n_k_heads} head_dim={head_dim}"
            ),
        });
    }
    let qk_elements = n_k_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "Q/K element count overflow".into(),
        })?;
    let vector_elements = n_v_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "V-head element count overflow".into(),
        })?;
    let state_elements =
        vector_elements
            .checked_mul(head_dim)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "state element count overflow".into(),
            })?;
    let validate = |name: &str,
                    tensor: &MetalTensor,
                    elements: usize,
                    writable: bool|
     -> Result<(), MetalError> {
        let elements_u64 = u64::try_from(elements).map_err(|_| MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("{name} element count does not fit u64"),
        })?;
        let shape = vec![elements_u64];
        let (_, bytes) = checked_shape_bytes(&shape, std::mem::size_of::<f32>())?;
        if tensor.dtype != GgmlType::F32
            || tensor.shape != shape
            || (writable && !tensor.is_writable())
            || !tensor_physical_range_valid(tensor, bytes, 4)
        {
            return Err(MetalError::BadShape {
                kernel: KERNEL,
                detail: format!(
                    "{name} expected {}F32 {shape:?}, got {:?} {:?} writable={} offset={}",
                    if writable { "writable " } else { "" },
                    tensor.dtype,
                    tensor.shape,
                    tensor.is_writable(),
                    tensor.offset
                ),
            });
        }
        Ok(())
    };
    for (name, tensor, elements) in [
        ("q", q, qk_elements),
        ("k", k, qk_elements),
        ("v", v, vector_elements),
        ("decay", decay, n_v_heads),
        ("beta", beta, n_v_heads),
        ("state_in", state_in, state_elements),
        ("grad_out", grad_out, vector_elements),
        ("grad_state_out", grad_state_out, state_elements),
    ] {
        validate(name, tensor, elements, false)?;
    }
    for (name, tensor, elements) in [
        ("grad_q", grad_q, qk_elements),
        ("grad_k", grad_k, qk_elements),
        ("grad_v", grad_v, vector_elements),
        ("grad_decay", grad_decay, n_v_heads),
        ("grad_beta", grad_beta, n_v_heads),
        ("grad_state_in", grad_state_in, state_elements),
        (
            "grad_correction_scratch",
            grad_correction_scratch,
            vector_elements,
        ),
        ("residual_scratch", residual_scratch, vector_elements),
    ] {
        validate(name, tensor, elements, true)?;
    }
    let inputs = [q, k, v, decay, beta, state_in, grad_out, grad_state_out];
    let outputs = [
        grad_q,
        grad_k,
        grad_v,
        grad_decay,
        grad_beta,
        grad_state_in,
        grad_correction_scratch,
        residual_scratch,
    ];
    for (index, output) in outputs.iter().enumerate() {
        if inputs
            .iter()
            .any(|input| tensor_ranges_overlap(output, input))
            || outputs[index + 1..]
                .iter()
                .any(|other| tensor_ranges_overlap(output, other))
        {
            return Err(MetalError::BadShape {
                kernel: KERNEL,
                detail: "gradient and scratch outputs must not overlap inputs or each other".into(),
            });
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let args = Args {
        n_v_heads: n_v_heads as u32,
        n_k_heads: n_k_heads as u32,
    };

    let rows = ctx.pipeline("kernel_gdn_step_decay_vjp_rows_f32")?;
    let qk = ctx.pipeline("kernel_gdn_step_decay_vjp_qk_f32")?;
    let scalars = ctx.pipeline("kernel_gdn_step_decay_vjp_scalars_f32")?;
    enc.set_pipeline(&rows);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, v);
    enc.set_tensor(4, decay);
    enc.set_tensor(5, beta);
    enc.set_tensor(6, state_in);
    enc.set_tensor(7, grad_out);
    enc.set_tensor(8, grad_state_out);
    enc.set_tensor(9, grad_state_in);
    enc.set_tensor(10, grad_v);
    enc.set_tensor(11, grad_correction_scratch);
    enc.set_tensor(12, residual_scratch);
    enc.dispatch(
        MTLSize {
            width: head_dim,
            height: n_v_heads,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    enc.set_pipeline(&qk);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, decay);
    enc.set_tensor(4, beta);
    enc.set_tensor(5, state_in);
    enc.set_tensor(6, grad_out);
    enc.set_tensor(7, grad_state_out);
    enc.set_tensor(8, grad_correction_scratch);
    enc.set_tensor(9, residual_scratch);
    enc.set_tensor(10, grad_q);
    enc.set_tensor(11, grad_k);
    enc.dispatch(
        MTLSize {
            width: n_k_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        },
    );

    enc.set_pipeline(&scalars);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, beta);
    enc.set_tensor(4, state_in);
    enc.set_tensor(5, grad_out);
    enc.set_tensor(6, grad_state_out);
    enc.set_tensor(7, grad_correction_scratch);
    enc.set_tensor(8, residual_scratch);
    enc.set_tensor(9, grad_decay);
    enc.set_tensor(10, grad_beta);
    enc.set_threadgroup_memory(0, 8 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_v_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Temporal VJP for a packed direct-decay GDN recurrence.
///
/// `state_checkpoints[t]` is the post-token state after token `t`; reverse
/// token `t` reads `initial_state` for token zero and checkpoint `t - 1`
/// otherwise. The final post-token checkpoint is never needed, so this accepts
/// either `n_tokens - 1` checkpoints or a full `n_tokens` forward tape. State
/// cotangents are carried backward through two reusable scratch tensors.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_step_decay_packed_vjp_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    decay_pack: &MetalTensor,
    beta_pack: &MetalTensor,
    initial_state: &MetalTensor,
    state_checkpoints: &MetalTensor,
    n_checkpoints: usize,
    grad_out_pack: &MetalTensor,
    grad_final_state: &MetalTensor,
    grad_q_pack: &MetalTensor,
    grad_k_pack: &MetalTensor,
    grad_v_pack: &MetalTensor,
    grad_decay_pack: &MetalTensor,
    grad_beta_pack: &MetalTensor,
    grad_initial_state: &MetalTensor,
    grad_state_scratch_a: &MetalTensor,
    grad_state_scratch_b: &MetalTensor,
    grad_correction_scratch: &MetalTensor,
    residual_scratch: &MetalTensor,
    n_tokens: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "gdn_step_decay_packed_vjp";
    if enc.is_concurrent() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "temporal state dependencies require a serial encoder".into(),
        });
    }
    if n_tokens == 0
        || u32::try_from(n_tokens).is_err()
        || head_dim != 128
        || n_v_heads == 0
        || n_k_heads == 0
        || !n_v_heads.is_multiple_of(n_k_heads)
        || u32::try_from(n_v_heads).is_err()
        || u32::try_from(n_k_heads).is_err()
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected nonzero u32 token/head counts, head_dim=128, and n_v divisible by n_k; got tokens={n_tokens} n_v={n_v_heads} n_k={n_k_heads} head_dim={head_dim}"
            ),
        });
    }
    if n_checkpoints < n_tokens - 1 || n_checkpoints > n_tokens {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected n_tokens-1 or n_tokens post-state checkpoints, got tokens={n_tokens} checkpoints={n_checkpoints}"
            ),
        });
    }
    let qk_elements = n_k_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "Q/K element count overflow".into(),
        })?;
    let vector_elements = n_v_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "V-head element count overflow".into(),
        })?;
    let state_elements =
        vector_elements
            .checked_mul(head_dim)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "state element count overflow".into(),
            })?;
    let packed = |per_token: usize, name: &str| {
        n_tokens
            .checked_mul(per_token)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: format!("{name} packed element count overflow"),
            })
    };
    let qk_pack_elements = packed(qk_elements, "Q/K")?;
    let vector_pack_elements = packed(vector_elements, "V/output")?;
    let scalar_pack_elements = packed(n_v_heads, "decay/beta")?;
    let checkpoint_elements =
        n_checkpoints
            .checked_mul(state_elements)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "checkpoint element count overflow".into(),
            })?;
    let shape = |elements: usize| -> Result<Vec<u64>, MetalError> {
        Ok(vec![u64::try_from(elements).map_err(|_| {
            MetalError::BadShape {
                kernel: KERNEL,
                detail: format!("element count {elements} does not fit u64"),
            }
        })?])
    };
    let qk_pack_shape = shape(qk_pack_elements)?;
    let vector_pack_shape = shape(vector_pack_elements)?;
    let scalar_pack_shape = shape(scalar_pack_elements)?;
    let state_shape = shape(state_elements)?;
    let checkpoint_shape = shape(checkpoint_elements)?;
    let vector_shape = shape(vector_elements)?;
    for (name, tensor, expected_shape) in [
        ("q_pack", q_pack, &qk_pack_shape),
        ("k_pack", k_pack, &qk_pack_shape),
        ("v_pack", v_pack, &vector_pack_shape),
        ("decay_pack", decay_pack, &scalar_pack_shape),
        ("beta_pack", beta_pack, &scalar_pack_shape),
        ("initial_state", initial_state, &state_shape),
        ("state_checkpoints", state_checkpoints, &checkpoint_shape),
        ("grad_out_pack", grad_out_pack, &vector_pack_shape),
        ("grad_final_state", grad_final_state, &state_shape),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, expected_shape, false)?;
    }
    for (name, tensor, expected_shape) in [
        ("grad_q_pack", grad_q_pack, &qk_pack_shape),
        ("grad_k_pack", grad_k_pack, &qk_pack_shape),
        ("grad_v_pack", grad_v_pack, &vector_pack_shape),
        ("grad_decay_pack", grad_decay_pack, &scalar_pack_shape),
        ("grad_beta_pack", grad_beta_pack, &scalar_pack_shape),
        ("grad_initial_state", grad_initial_state, &state_shape),
        ("grad_state_scratch_a", grad_state_scratch_a, &state_shape),
        ("grad_state_scratch_b", grad_state_scratch_b, &state_shape),
        (
            "grad_correction_scratch",
            grad_correction_scratch,
            &vector_shape,
        ),
        ("residual_scratch", residual_scratch, &vector_shape),
    ] {
        validate_compact_f32_tensor(KERNEL, name, tensor, expected_shape, true)?;
    }
    validate_vjp_storage_disjoint(
        KERNEL,
        &[
            q_pack,
            k_pack,
            v_pack,
            decay_pack,
            beta_pack,
            initial_state,
            state_checkpoints,
            grad_out_pack,
            grad_final_state,
        ],
        &[
            grad_q_pack,
            grad_k_pack,
            grad_v_pack,
            grad_decay_pack,
            grad_beta_pack,
            grad_initial_state,
            grad_state_scratch_a,
            grad_state_scratch_b,
            grad_correction_scratch,
            residual_scratch,
        ],
    )?;

    let mut current_grad_state = grad_final_state;
    for token in (0..n_tokens).rev() {
        let q = q_pack.view_subrange((token * qk_elements) as u64, vec![qk_elements as u64]);
        let k = k_pack.view_subrange((token * qk_elements) as u64, vec![qk_elements as u64]);
        let v = v_pack.view_subrange(
            (token * vector_elements) as u64,
            vec![vector_elements as u64],
        );
        let decay = decay_pack.view_subrange((token * n_v_heads) as u64, vec![n_v_heads as u64]);
        let beta = beta_pack.view_subrange((token * n_v_heads) as u64, vec![n_v_heads as u64]);
        let grad_out = grad_out_pack.view_subrange(
            (token * vector_elements) as u64,
            vec![vector_elements as u64],
        );
        let grad_q =
            grad_q_pack.view_subrange((token * qk_elements) as u64, vec![qk_elements as u64]);
        let grad_k =
            grad_k_pack.view_subrange((token * qk_elements) as u64, vec![qk_elements as u64]);
        let grad_v = grad_v_pack.view_subrange(
            (token * vector_elements) as u64,
            vec![vector_elements as u64],
        );
        let grad_decay =
            grad_decay_pack.view_subrange((token * n_v_heads) as u64, vec![n_v_heads as u64]);
        let grad_beta =
            grad_beta_pack.view_subrange((token * n_v_heads) as u64, vec![n_v_heads as u64]);
        let checkpoint = (token > 0).then(|| {
            state_checkpoints.view_subrange(
                ((token - 1) * state_elements) as u64,
                vec![state_elements as u64],
            )
        });
        let state_in = checkpoint.as_ref().unwrap_or(initial_state);
        let reverse_index = n_tokens - 1 - token;
        let next_grad_state = if token == 0 {
            grad_initial_state
        } else if reverse_index.is_multiple_of(2) {
            grad_state_scratch_a
        } else {
            grad_state_scratch_b
        };
        encode_gdn_step_decay_vjp_f32(
            ctx,
            enc,
            &q,
            &k,
            &v,
            &decay,
            &beta,
            state_in,
            &grad_out,
            current_grad_state,
            &grad_q,
            &grad_k,
            &grad_v,
            &grad_decay,
            &grad_beta,
            next_grad_state,
            grad_correction_scratch,
            residual_scratch,
            n_v_heads,
            n_k_heads,
            head_dim,
        )?;
        current_grad_state = next_grad_state;
    }
    Ok(())
}

pub fn encode_gdn_step_decay_packed_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    decay_pack: &MetalTensor,
    beta_pack: &MetalTensor,
    state: &MetalTensor,
    out_pack: &MetalTensor,
    n_tokens: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    encode_gdn_step_decay_packed_inner(
        ctx, enc, q_pack, k_pack, v_pack, decay_pack, beta_pack, state, out_pack, None, 0,
        n_tokens, n_v_heads, n_k_heads, head_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_step_decay_packed_ckpt_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    decay_pack: &MetalTensor,
    beta_pack: &MetalTensor,
    state: &MetalTensor,
    out_pack: &MetalTensor,
    state_ckpt: &MetalTensor,
    n_checkpoints: usize,
    n_tokens: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    encode_gdn_step_decay_packed_inner(
        ctx,
        enc,
        q_pack,
        k_pack,
        v_pack,
        decay_pack,
        beta_pack,
        state,
        out_pack,
        Some(state_ckpt),
        n_checkpoints,
        n_tokens,
        n_v_heads,
        n_k_heads,
        head_dim,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_step_decay_packed_inner(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_pack: &MetalTensor,
    k_pack: &MetalTensor,
    v_pack: &MetalTensor,
    decay_pack: &MetalTensor,
    beta_pack: &MetalTensor,
    state: &MetalTensor,
    out_pack: &MetalTensor,
    state_ckpt: Option<&MetalTensor>,
    n_checkpoints: usize,
    n_tokens: usize,
    n_v_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if head_dim != 128 {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("head_dim={head_dim} but kernel hardcodes 128"),
        });
    }
    if !n_v_heads.is_multiple_of(n_k_heads) {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("n_v_heads={n_v_heads} not multiple of n_k_heads={n_k_heads}"),
        });
    }
    let qk_per_token = n_k_heads * head_dim;
    let v_per_token = n_v_heads * head_dim;
    if q_pack.n_elements() as usize != n_tokens * qk_per_token
        || k_pack.n_elements() as usize != n_tokens * qk_per_token
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("q/k expected {} elements per pack", n_tokens * qk_per_token),
        });
    }
    if v_pack.n_elements() as usize != n_tokens * v_per_token {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("v expected {} elements", n_tokens * v_per_token),
        });
    }
    if decay_pack.n_elements() as usize != n_tokens * n_v_heads
        || beta_pack.n_elements() as usize != n_tokens * n_v_heads
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("decay/beta expected {} elements", n_tokens * n_v_heads),
        });
    }
    let want_state = n_v_heads * head_dim * head_dim;
    if state.n_elements() as usize != want_state {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("state expected {want_state} elements"),
        });
    }
    if out_pack.n_elements() as usize != n_tokens * v_per_token {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!("out expected {} elements", n_tokens * v_per_token),
        });
    }
    if n_checkpoints > n_tokens
        || state_ckpt.is_some_and(|checkpoint| {
            checkpoint.n_elements() as usize != n_checkpoints * want_state
        })
        || (n_checkpoints > 0 && state_ckpt.is_none())
    {
        return Err(MetalError::BadShape {
            kernel: "gdn_step_decay_packed",
            detail: format!(
                "checkpoint geometry mismatch: checkpoints={n_checkpoints} state_elems={want_state}"
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_checkpoints: u32,
        n_v_heads: u32,
        n_k_heads: u32,
    }
    let use_nsg4 = n_v_heads.is_multiple_of(4) && head_dim == 128;
    let kernel = if use_nsg4 {
        "kernel_gdn_step_decay_packed_nsg4_f32"
    } else {
        "kernel_gdn_step_decay_packed_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens as u32,
            n_checkpoints: n_checkpoints as u32,
            n_v_heads: n_v_heads as u32,
            n_k_heads: n_k_heads as u32,
        },
    );
    enc.set_tensor(1, q_pack);
    enc.set_tensor(2, k_pack);
    enc.set_tensor(3, v_pack);
    enc.set_tensor(4, decay_pack);
    enc.set_tensor(5, beta_pack);
    enc.set_tensor(6, state);
    enc.set_tensor(7, out_pack);
    enc.set_tensor(8, state_ckpt.unwrap_or(state));
    if use_nsg4 {
        enc.dispatch(
            MTLSize {
                width: head_dim / 4,
                height: n_v_heads,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
    } else {
        enc.dispatch(
            MTLSize {
                width: head_dim,
                height: n_v_heads,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    /// GDN α-chain fusion vs the 3-dispatch reference (add_inplace +
    /// softplus + mul). Must match within fp32 rounding noise.
    #[test]
    fn gdn_alpha_chain_matches_unfused() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n = 48usize; // n_v_heads for 27B
        let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
        let dt: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let alog: Vec<f32> = (0..n).map(|i| -1.0 - (i % 5) as f32 * 0.2).collect();

        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let dt_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&dt),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let alog_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&alog),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();

        // Fused path.
        let fused = one_shot_f32_out(&ctx, n, |enc, out| {
            encode_gdn_alpha_chain_f32(&ctx, enc, &a_t, &dt_t, &alog_t, out)
        });

        // Unfused reference: build via 3 sequential dispatches in one cmdbuf.
        let a_ref = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n as u64],
            GgmlType::F32,
        )
        .unwrap();
        let unfused_t = MetalTensor::zeros_f32(&ctx, vec![n as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_add_inplace_f32(&ctx, enc, &a_ref, &dt_t)?;
            encode_softplus_f32(&ctx, enc, &a_ref, &unfused_t)?;
            encode_mul_f32(&ctx, enc, &unfused_t, &alog_t, &unfused_t)
        })
        .unwrap();
        let unfused = read_back_f32(&unfused_t.buffer, n);

        let max_abs = fused
            .iter()
            .zip(unfused.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("[gdn_alpha_chain] max|Δ|={max_abs:.2e}");
        assert!(
            max_abs < 1e-5,
            "gdn_alpha_chain fused vs unfused mismatch: max|Δ|={max_abs}"
        );

        // Also validate vs explicit CPU formula.
        for i in 0..n {
            let v = a[i] + dt[i];
            let sp = if v > 20.0 {
                v
            } else if v < -20.0 {
                v.exp()
            } else {
                (1.0 + v.exp()).ln()
            };
            let expected = sp * alog[i];
            assert!(
                (fused[i] - expected).abs() < 1e-5,
                "i={i}: fused={} expected={expected}",
                fused[i]
            );
        }
    }

    /// v0.73a: batched α-chain over `[N, n_v]` must produce
    /// bit-identical output to N successive single-row α-chain calls
    /// (broadcasting `dt_bias` and `a_log` across rows).
    #[test]
    fn gdn_alpha_chain_batched_matches_per_row() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n_rows = 16usize; // N for DFlash block_size
        let n_cols = 48usize; // n_v_heads for 27B
        let n = n_rows * n_cols;
        let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.5).collect();
        let dt: Vec<f32> = (0..n_cols).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let alog: Vec<f32> = (0..n_cols).map(|i| -1.0 - (i % 5) as f32 * 0.2).collect();

        let a_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&a),
            vec![n_rows as u64, n_cols as u64],
            GgmlType::F32,
        )
        .unwrap();
        let dt_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&dt),
            vec![n_cols as u64],
            GgmlType::F32,
        )
        .unwrap();
        let alog_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&alog),
            vec![n_cols as u64],
            GgmlType::F32,
        )
        .unwrap();

        // Batched fused path.
        let batched = one_shot_f32_out(&ctx, n, |enc, out| {
            encode_gdn_alpha_chain_batched_f32(&ctx, enc, &a_t, &dt_t, &alog_t, out, n_rows, n_cols)
        });

        // Per-row reference: N invocations of the single-row kernel,
        // each on a row-view of a / out.
        let out_t = MetalTensor::zeros_f32(&ctx, vec![n_rows as u64, n_cols as u64]).unwrap();
        one_shot(&ctx, |enc| {
            for r in 0..n_rows {
                let a_row = a_t.view_subrange((r * n_cols) as u64, vec![n_cols as u64]);
                let out_row = out_t.view_subrange((r * n_cols) as u64, vec![n_cols as u64]);
                encode_gdn_alpha_chain_f32(&ctx, enc, &a_row, &dt_t, &alog_t, &out_row)?;
            }
            Ok(())
        })
        .unwrap();
        let per_row = read_back_f32(&out_t.buffer, n);

        // Bit-exact required (same kernel arithmetic, same broadcast, same order).
        for i in 0..n {
            assert_eq!(
                batched[i].to_bits(),
                per_row[i].to_bits(),
                "i={i} (r={}, c={}): batched={} per_row={}",
                i / n_cols,
                i % n_cols,
                batched[i],
                per_row[i]
            );
        }
    }

    #[test]
    fn ssm_conv_silu_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Real shapes: 0.8B has conv_dim = 2*16*128 + 16*128 = 6144;
        // 27B has conv_dim = 2*16*128 + 48*128 = 10240.
        for &conv_dim in &[6144usize, 10240] {
            const K: usize = 4;
            let qkv_now: Vec<f32> = (0..conv_dim)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let conv_buf: Vec<f32> = (0..(K - 1) * conv_dim)
                .map(|i| ((i % 13) as f32 - 6.0) * 5e-3)
                .collect();
            let conv_w: Vec<f32> = (0..conv_dim * K)
                .map(|i| ((i % 7) as f32 - 3.0) * 1e-2)
                .collect();

            // CPU oracle.
            let mut buf_cpu = conv_buf.clone();
            let out_cpu = ssm_conv_silu_cpu_ref(&qkv_now, &mut buf_cpu, &conv_w, conv_dim);

            // GPU.
            let qkv_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&qkv_now),
                vec![conv_dim as u64],
                GgmlType::F32,
            )
            .unwrap();
            let buf_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&conv_buf),
                vec![((K - 1) * conv_dim) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let w_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&conv_w),
                vec![(conv_dim * K) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let out_t = MetalTensor::zeros_f32(&ctx, vec![conv_dim as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_ssm_conv_silu_f32(&ctx, enc, &qkv_t, &buf_t, &w_t, &out_t, conv_dim)
            })
            .unwrap();

            let out_gpu = read_back_f32(&out_t.buffer, conv_dim);
            let buf_gpu = read_back_f32(&buf_t.buffer, (K - 1) * conv_dim);

            let max_out = out_gpu
                .iter()
                .zip(out_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let max_buf = buf_gpu
                .iter()
                .zip(buf_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "[ssm_conv conv_dim={conv_dim}] max|out_Δ|={max_out:.2e} max|buf_Δ|={max_buf:.2e}"
            );
            assert!(max_out < 1e-5, "out drift {max_out}");
            assert!(max_buf < 1e-7, "buf drift {max_buf}");
        }
    }

    #[test]
    fn gdn_prep_packed_checkpoints_match_token_steps() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N: usize = 4;
        const HEAD_DIM: usize = 128;
        const N_K_HEADS: usize = 1;
        const N_V_HEADS: usize = 4;
        let qk_dim = N_K_HEADS * HEAD_DIM;
        let v_dim = N_V_HEADS * HEAD_DIM;
        let conv_dim = 2 * qk_dim + v_dim;
        let conv_state_elems = 3 * conv_dim;

        let qkv: Vec<f32> = (0..N * conv_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.003)
            .collect();
        let conv_initial: Vec<f32> = (0..conv_state_elems)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.002)
            .collect();
        let conv_w: Vec<f32> = (0..4 * conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.004)
            .collect();

        let qkv_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&qkv),
            vec![(N * conv_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let conv_w_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&conv_w),
            vec![(4 * conv_dim) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let conv_packed = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&conv_initial),
            vec![conv_state_elems as u64],
            GgmlType::F32,
        )
        .unwrap();
        let q_packed = MetalTensor::zeros_f32(&ctx, vec![(N * qk_dim) as u64]).unwrap();
        let k_packed = MetalTensor::zeros_f32(&ctx, vec![(N * qk_dim) as u64]).unwrap();
        let v_packed = MetalTensor::zeros_f32(&ctx, vec![(N * v_dim) as u64]).unwrap();
        let ckpt_packed =
            MetalTensor::zeros_f32(&ctx, vec![(N * conv_state_elems) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_gdn_prep_packed_ckpt_f32(
                &ctx,
                enc,
                &qkv_t,
                &conv_packed,
                &conv_w_t,
                &q_packed,
                &k_packed,
                &v_packed,
                &ckpt_packed,
                N,
                N,
                N_K_HEADS,
                N_V_HEADS,
                HEAD_DIM,
            )
        })
        .unwrap();

        let conv_token = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&conv_initial),
            vec![conv_state_elems as u64],
            GgmlType::F32,
        )
        .unwrap();
        let out_token = MetalTensor::zeros_f32(&ctx, vec![(N * conv_dim) as u64]).unwrap();
        let ckpt_token = MetalTensor::zeros_f32(&ctx, vec![(N * conv_state_elems) as u64]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        for token in 0..N {
            let enc = KernelEncoder::begin(&cmd);
            let qkv_row = qkv_t.view_subrange((token * conv_dim) as u64, vec![conv_dim as u64]);
            let out_row = out_token.view_subrange((token * conv_dim) as u64, vec![conv_dim as u64]);
            encode_ssm_conv_silu_f32(
                &ctx,
                &enc,
                &qkv_row,
                &conv_token,
                &conv_w_t,
                &out_row,
                conv_dim,
            )
            .unwrap();
            enc.end();
            let blit = BlitEncoder::begin(&cmd);
            let ckpt_row = ckpt_token.view_subrange(
                (token * conv_state_elems) as u64,
                vec![conv_state_elems as u64],
            );
            blit.copy_tensor(&conv_token, &ckpt_row);
            blit.end();
        }
        cmd.commit();
        wait_completed(&cmd).expect("Metal command buffer failed");

        let packed_conv = read_back_f32(&conv_packed.buffer, conv_state_elems);
        let token_conv = read_back_f32(&conv_token.buffer, conv_state_elems);
        let packed_ckpt = read_back_f32(&ckpt_packed.buffer, N * conv_state_elems);
        let token_ckpt = read_back_f32(&ckpt_token.buffer, N * conv_state_elems);
        let packed_q = read_back_f32(&q_packed.buffer, N * qk_dim);
        let packed_k = read_back_f32(&k_packed.buffer, N * qk_dim);
        let packed_v = read_back_f32(&v_packed.buffer, N * v_dim);
        let token_out = read_back_f32(&out_token.buffer, N * conv_dim);

        assert!(
            packed_conv
                .iter()
                .zip(&token_conv)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed conv final state differs from token steps"
        );
        assert!(
            packed_ckpt
                .iter()
                .zip(&token_ckpt)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed conv checkpoints differ from token steps"
        );
        for token in 0..N {
            for channel in 0..conv_dim {
                let packed = if channel < qk_dim {
                    packed_q[token * qk_dim + channel]
                } else if channel < 2 * qk_dim {
                    packed_k[token * qk_dim + channel - qk_dim]
                } else {
                    packed_v[token * v_dim + channel - 2 * qk_dim]
                };
                assert_eq!(
                    packed.to_bits(),
                    token_out[token * conv_dim + channel].to_bits(),
                    "packed conv output mismatch at token={token} channel={channel}"
                );
            }
        }
    }

    #[test]
    fn gdn_recurrence_packed_checkpoints_match_token_steps() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N: usize = 4;
        const HEAD_DIM: usize = 128;
        const N_K_HEADS: usize = 1;
        const N_V_HEADS: usize = 4;
        let qk_per_token = N_K_HEADS * HEAD_DIM;
        let v_per_token = N_V_HEADS * HEAD_DIM;
        let state_elems = N_V_HEADS * HEAD_DIM * HEAD_DIM;

        let q: Vec<f32> = (0..N * qk_per_token)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.004)
            .collect();
        let k: Vec<f32> = (0..N * qk_per_token)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.003)
            .collect();
        let v: Vec<f32> = (0..N * v_per_token)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.002)
            .collect();
        let decay: Vec<f32> = (0..N * N_V_HEADS)
            .map(|i| 0.9 + (i % 7) as f32 * 0.01)
            .collect();
        let beta: Vec<f32> = (0..N * N_V_HEADS)
            .map(|i| 0.2 + (i % 5) as f32 * 0.1)
            .collect();
        let state_initial: Vec<f32> = (0..state_elems)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
            .collect();

        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let q_t = tensor(&q);
        let k_t = tensor(&k);
        let v_t = tensor(&v);
        let decay_t = tensor(&decay);
        let beta_t = tensor(&beta);
        let state_packed = tensor(&state_initial);
        let out_packed = MetalTensor::zeros_f32(&ctx, vec![(N * v_per_token) as u64]).unwrap();
        let ckpt_packed = MetalTensor::zeros_f32(&ctx, vec![(N * state_elems) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_gdn_step_decay_packed_ckpt_f32(
                &ctx,
                enc,
                &q_t,
                &k_t,
                &v_t,
                &decay_t,
                &beta_t,
                &state_packed,
                &out_packed,
                &ckpt_packed,
                N,
                N,
                N_V_HEADS,
                N_K_HEADS,
                HEAD_DIM,
            )
        })
        .unwrap();

        let state_token = tensor(&state_initial);
        let out_token = MetalTensor::zeros_f32(&ctx, vec![(N * v_per_token) as u64]).unwrap();
        let ckpt_token = MetalTensor::zeros_f32(&ctx, vec![(N * state_elems) as u64]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        for token in 0..N {
            let enc = KernelEncoder::begin(&cmd);
            encode_gdn_step_decay_f32(
                &ctx,
                &enc,
                &q_t.view_subrange((token * qk_per_token) as u64, vec![qk_per_token as u64]),
                &k_t.view_subrange((token * qk_per_token) as u64, vec![qk_per_token as u64]),
                &v_t.view_subrange((token * v_per_token) as u64, vec![v_per_token as u64]),
                &decay_t.view_subrange((token * N_V_HEADS) as u64, vec![N_V_HEADS as u64]),
                &beta_t.view_subrange((token * N_V_HEADS) as u64, vec![N_V_HEADS as u64]),
                &state_token,
                &out_token.view_subrange((token * v_per_token) as u64, vec![v_per_token as u64]),
                N_V_HEADS,
                N_K_HEADS,
                HEAD_DIM,
            )
            .unwrap();
            enc.end();
            let blit = BlitEncoder::begin(&cmd);
            let ckpt_row =
                ckpt_token.view_subrange((token * state_elems) as u64, vec![state_elems as u64]);
            blit.copy_tensor(&state_token, &ckpt_row);
            blit.end();
        }
        cmd.commit();
        wait_completed(&cmd).expect("Metal command buffer failed");

        let packed_state = read_back_f32(&state_packed.buffer, state_elems);
        let token_state = read_back_f32(&state_token.buffer, state_elems);
        let packed_out = read_back_f32(&out_packed.buffer, N * v_per_token);
        let token_out = read_back_f32(&out_token.buffer, N * v_per_token);
        let packed_ckpt = read_back_f32(&ckpt_packed.buffer, N * state_elems);
        let token_ckpt = read_back_f32(&ckpt_token.buffer, N * state_elems);
        assert!(
            packed_state
                .iter()
                .zip(&token_state)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed recurrence final state differs from token steps"
        );
        assert!(
            packed_out
                .iter()
                .zip(&token_out)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed recurrence outputs differ from token steps"
        );
        assert!(
            packed_ckpt
                .iter()
                .zip(&token_ckpt)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "packed recurrence checkpoints differ from token steps"
        );
    }

    #[test]
    fn gdn_step_matches_cpu() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        // Cover both Qwen3.5/3.6 sizes:
        //   0.8B: n_v_heads = 16, head_dim = 128
        //   27B:  n_v_heads = 48, head_dim = 128
        // 0.8B has n_v == n_k == 16; 27B has n_v=48, n_k=16 (3:1 repeat).
        for &(n_v, n_k) in &[(16usize, 16usize), (48, 16)] {
            let hd = 128usize;
            // Synthetic but realistic-magnitude inputs. Q/K are sized to
            // n_k heads; V is sized to n_v heads.
            let q: Vec<f32> = (0..n_k * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k: Vec<f32> = (0..n_k * hd)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v: Vec<f32> = (0..n_v * hd)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();
            let g: Vec<f32> = (0..n_v).map(|i| -((i % 7) as f32) * 1e-3).collect();
            let beta: Vec<f32> = (0..n_v).map(|i| 0.5 + ((i % 11) as f32) * 1e-2).collect();
            // Random-ish but deterministic state, including the
            // post-first-token regime (nonzero initial state) since
            // codex flagged "first-token only" coverage as inadequate.
            let state: Vec<f32> = (0..n_v * hd * hd)
                .map(|i| ((i % 13) as f32 - 6.0) * 1e-3)
                .collect();

            // CPU oracle.
            let mut state_cpu = state.clone();
            let out_cpu = gdn_step_cpu_ref(&q, &k, &v, &g, &beta, &mut state_cpu, n_v, n_k, hd);

            // GPU.
            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_k * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k),
                vec![(n_k * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let v_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&v),
                vec![(n_v * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let g_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&g),
                vec![n_v as u64],
                GgmlType::F32,
            )
            .unwrap();
            let beta_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&beta),
                vec![n_v as u64],
                GgmlType::F32,
            )
            .unwrap();
            let state_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&state),
                vec![(n_v * hd * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_v * hd) as u64]).unwrap();

            one_shot(&ctx, |enc| {
                encode_gdn_step_f32(
                    &ctx, enc, &q_t, &k_t, &v_t, &g_t, &beta_t, &state_t, &out_t, n_v, n_k, hd,
                )
            })
            .unwrap();

            let out_gpu = read_back_f32(&out_t.buffer, n_v * hd);
            let state_gpu = read_back_f32(&state_t.buffer, n_v * hd * hd);

            // Output comparison.
            let max_out = out_gpu
                .iter()
                .zip(out_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            // State comparison (this is the recurrent variable; correctness
            // here matters more than the output for multi-step decode).
            let max_state = state_gpu
                .iter()
                .zip(state_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);

            eprintln!(
                "[gdn_step n_v={n_v} n_k={n_k}] max|out_Δ|={max_out:.2e}  max|state_Δ|={max_state:.2e}"
            );
            // simd_sum reduction order can drift slightly from the
            // sequential CPU version; 1e-4 covers it for our magnitudes.
            assert!(max_out < 1e-4, "out drift {max_out}");
            assert!(max_state < 1e-4, "state drift {max_state}");
        }
    }

    #[test]
    fn gdn_step_decay_vjp_matches_adjoint_and_finite_differences() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_V: usize = 6;
        const N_K: usize = 2;
        const HEAD_DIM: usize = 128;
        let q: Vec<f32> = (0..N_K * HEAD_DIM)
            .map(|index| ((index * 11 + 3) % 43) as f32 * 0.002 - 0.041)
            .collect();
        let k: Vec<f32> = (0..N_K * HEAD_DIM)
            .map(|index| ((index * 13 + 5) % 47) as f32 * 0.0017 - 0.039)
            .collect();
        let v: Vec<f32> = (0..N_V * HEAD_DIM)
            .map(|index| ((index * 17 + 1) % 53) as f32 * 0.0023 - 0.057)
            .collect();
        let decay: Vec<f32> = (0..N_V).map(|head| 0.89 + head as f32 * 0.021).collect();
        let beta: Vec<f32> = (0..N_V).map(|head| 0.23 + head as f32 * 0.14).collect();
        let state: Vec<f32> = (0..N_V * HEAD_DIM * HEAD_DIM)
            .map(|index| ((index * 19 + 7) % 59) as f32 * 0.0007 - 0.019)
            .collect();
        let grad_out: Vec<f32> = (0..N_V * HEAD_DIM)
            .map(|index| ((index * 23 + 2) % 61) as f32 * 0.0011 - 0.031)
            .collect();
        let grad_state_out: Vec<f32> = (0..N_V * HEAD_DIM * HEAD_DIM)
            .map(|index| ((index * 29 + 11) % 67) as f32 * 0.00009 - 0.003)
            .collect();

        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let q_t = tensor(&q);
        let k_t = tensor(&k);
        let v_t = tensor(&v);
        let decay_t = tensor(&decay);
        let beta_t = tensor(&beta);
        let state_t = tensor(&state);
        let grad_out_t = tensor(&grad_out);
        let grad_state_out_t = tensor(&grad_state_out);
        let grad_q_t = MetalTensor::zeros_f32(&ctx, vec![(N_K * HEAD_DIM) as u64]).unwrap();
        let grad_k_t = MetalTensor::zeros_f32(&ctx, vec![(N_K * HEAD_DIM) as u64]).unwrap();
        let grad_v_t = MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM) as u64]).unwrap();
        let grad_decay_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_beta_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_state_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM * HEAD_DIM) as u64]).unwrap();
        let grad_c_t = MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM) as u64]).unwrap();
        let residual_t = MetalTensor::zeros_f32(&ctx, vec![(N_V * HEAD_DIM) as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_gdn_step_decay_vjp_f32(
                &ctx,
                encoder,
                &q_t,
                &k_t,
                &v_t,
                &decay_t,
                &beta_t,
                &state_t,
                &grad_out_t,
                &grad_state_out_t,
                &grad_q_t,
                &grad_k_t,
                &grad_v_t,
                &grad_decay_t,
                &grad_beta_t,
                &grad_state_t,
                &grad_c_t,
                &residual_t,
                N_V,
                N_K,
                HEAD_DIM,
            )
        })
        .unwrap();

        let actual = GdnStepVjpReference {
            grad_q: read_back_f32(&grad_q_t.buffer, N_K * HEAD_DIM)
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_k: read_back_f32(&grad_k_t.buffer, N_K * HEAD_DIM)
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_v: read_back_f32(&grad_v_t.buffer, N_V * HEAD_DIM)
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_decay: read_back_f32(&grad_decay_t.buffer, N_V)
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_beta: read_back_f32(&grad_beta_t.buffer, N_V)
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_state: read_back_f32(&grad_state_t.buffer, N_V * HEAD_DIM * HEAD_DIM)
                .into_iter()
                .map(f64::from)
                .collect(),
        };
        let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
        let q64 = as_f64(&q);
        let k64 = as_f64(&k);
        let v64 = as_f64(&v);
        let decay64 = as_f64(&decay);
        let beta64 = as_f64(&beta);
        let state64 = as_f64(&state);
        let grad_out64 = as_f64(&grad_out);
        let grad_state_out64 = as_f64(&grad_state_out);
        let expected = gdn_step_decay_vjp_f64(
            &q64,
            &k64,
            &v64,
            &decay64,
            &beta64,
            &state64,
            &grad_out64,
            &grad_state_out64,
            N_V,
            N_K,
            HEAD_DIM,
        );
        for (name, gpu, cpu, tolerance) in [
            ("q", &actual.grad_q, &expected.grad_q, 3e-5),
            ("k", &actual.grad_k, &expected.grad_k, 3e-5),
            ("v", &actual.grad_v, &expected.grad_v, 2e-6),
            ("decay", &actual.grad_decay, &expected.grad_decay, 5e-5),
            ("beta", &actual.grad_beta, &expected.grad_beta, 5e-5),
            ("state", &actual.grad_state, &expected.grad_state, 3e-6),
        ] {
            let max_abs = gpu
                .iter()
                .zip(cpu)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0f64, f64::max);
            assert!(max_abs < tolerance, "{name} VJP error {max_abs}");
        }

        let objective =
            |q: &[f64], k: &[f64], v: &[f64], decay: &[f64], beta: &[f64], state: &[f64]| {
                gdn_step_decay_objective_f64(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    state,
                    &grad_out64,
                    &grad_state_out64,
                    N_V,
                    N_K,
                    HEAD_DIM,
                )
            };
        let epsilon = 1e-5;
        let finite_difference = |values: &[f64], index: usize, evaluate: &dyn Fn(&[f64]) -> f64| {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
        };
        for &index in &[0usize, 127, N_K * HEAD_DIM - 1] {
            let fd = finite_difference(&q64, index, &|candidate| {
                objective(candidate, &k64, &v64, &decay64, &beta64, &state64)
            });
            assert!((fd - actual.grad_q[index]).abs() < 2e-5);
            let fd = finite_difference(&k64, index, &|candidate| {
                objective(&q64, candidate, &v64, &decay64, &beta64, &state64)
            });
            assert!((fd - actual.grad_k[index]).abs() < 2e-5);
        }
        for &index in &[0usize, 255, N_V * HEAD_DIM - 1] {
            let fd = finite_difference(&v64, index, &|candidate| {
                objective(&q64, &k64, candidate, &decay64, &beta64, &state64)
            });
            assert!((fd - actual.grad_v[index]).abs() < 2e-5);
        }
        for index in 0..N_V {
            let fd = finite_difference(&decay64, index, &|candidate| {
                objective(&q64, &k64, &v64, candidate, &beta64, &state64)
            });
            assert!((fd - actual.grad_decay[index]).abs() < 3e-5);
            let fd = finite_difference(&beta64, index, &|candidate| {
                objective(&q64, &k64, &v64, &decay64, candidate, &state64)
            });
            assert!((fd - actual.grad_beta[index]).abs() < 3e-5);
        }
        for &index in &[
            0usize,
            HEAD_DIM - 1,
            HEAD_DIM,
            2 * HEAD_DIM * HEAD_DIM + 31 * HEAD_DIM + 32,
            N_V * HEAD_DIM * HEAD_DIM - 1,
        ] {
            let fd = finite_difference(&state64, index, &|candidate| {
                objective(&q64, &k64, &v64, &decay64, &beta64, candidate)
            });
            assert!((fd - actual.grad_state[index]).abs() < 2e-5);
        }

        let direction = |len: usize, stride: usize| {
            (0..len)
                .map(|index| ((index * stride + 3) % 29) as f64 * 0.001 - 0.014)
                .collect::<Vec<_>>()
        };
        let dq = direction(q64.len(), 5);
        let dk = direction(k64.len(), 7);
        let dv = direction(v64.len(), 11);
        let ddecay = direction(decay64.len(), 13);
        let dbeta = direction(beta64.len(), 17);
        let dstate = direction(state64.len(), 19);
        let inner = |gradient: &[f64], tangent: &[f64]| {
            gradient
                .iter()
                .zip(tangent)
                .map(|(gradient, tangent)| gradient * tangent)
                .sum::<f64>()
        };
        let reverse_directional = inner(&actual.grad_q, &dq)
            + inner(&actual.grad_k, &dk)
            + inner(&actual.grad_v, &dv)
            + inner(&actual.grad_decay, &ddecay)
            + inner(&actual.grad_beta, &dbeta)
            + inner(&actual.grad_state, &dstate);
        let shift = |base: &[f64], tangent: &[f64], amount: f64| {
            base.iter()
                .zip(tangent)
                .map(|(base, tangent)| base + amount * tangent)
                .collect::<Vec<_>>()
        };
        let directional_epsilon = 1e-5;
        let plus = objective(
            &shift(&q64, &dq, directional_epsilon),
            &shift(&k64, &dk, directional_epsilon),
            &shift(&v64, &dv, directional_epsilon),
            &shift(&decay64, &ddecay, directional_epsilon),
            &shift(&beta64, &dbeta, directional_epsilon),
            &shift(&state64, &dstate, directional_epsilon),
        );
        let minus = objective(
            &shift(&q64, &dq, -directional_epsilon),
            &shift(&k64, &dk, -directional_epsilon),
            &shift(&v64, &dv, -directional_epsilon),
            &shift(&decay64, &ddecay, -directional_epsilon),
            &shift(&beta64, &dbeta, -directional_epsilon),
            &shift(&state64, &dstate, -directional_epsilon),
        );
        let forward_directional = (plus - minus) / (2.0 * directional_epsilon);
        assert!(
            (forward_directional - reverse_directional).abs() < 2e-5,
            "directional adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
        );

        for (tensor, original) in [
            (&q_t, &q),
            (&k_t, &k),
            (&v_t, &v),
            (&decay_t, &decay),
            (&beta_t, &beta),
            (&state_t, &state),
            (&grad_out_t, &grad_out),
            (&grad_state_out_t, &grad_state_out),
        ] {
            assert_eq!(
                read_back_f32(&tensor.buffer, original.len())
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn gdn_step_decay_packed_vjp_matches_temporal_oracle_and_adjoint() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_TOKENS: usize = 4;
        const N_CHECKPOINTS: usize = N_TOKENS - 1;
        const N_V: usize = 3;
        const N_K: usize = 1;
        const HEAD_DIM: usize = 128;
        let qk_elements = N_K * HEAD_DIM;
        let vector_elements = N_V * HEAD_DIM;
        let state_elements = vector_elements * HEAD_DIM;
        let q: Vec<f32> = (0..N_TOKENS * qk_elements)
            .map(|index| ((index * 11 + 3) % 43) as f32 * 0.0017 - 0.035)
            .collect();
        let k: Vec<f32> = (0..N_TOKENS * qk_elements)
            .map(|index| ((index * 13 + 5) % 47) as f32 * 0.0015 - 0.033)
            .collect();
        let v: Vec<f32> = (0..N_TOKENS * vector_elements)
            .map(|index| ((index * 17 + 1) % 53) as f32 * 0.0019 - 0.049)
            .collect();
        let decay: Vec<f32> = (0..N_TOKENS * N_V)
            .map(|index| 0.89 + (index % N_V) as f32 * 0.026 + (index / N_V) as f32 * 0.004)
            .collect();
        let beta: Vec<f32> = (0..N_TOKENS * N_V)
            .map(|index| 0.21 + (index % N_V) as f32 * 0.17 + (index / N_V) as f32 * 0.013)
            .collect();
        let initial_state: Vec<f32> = (0..state_elements)
            .map(|index| ((index * 19 + 7) % 59) as f32 * 0.00061 - 0.017)
            .collect();
        let grad_out: Vec<f32> = (0..N_TOKENS * vector_elements)
            .map(|index| ((index * 23 + 2) % 61) as f32 * 0.00093 - 0.027)
            .collect();
        let grad_final_state: Vec<f32> = (0..state_elements)
            .map(|index| ((index * 29 + 11) % 67) as f32 * 0.000071 - 0.0023)
            .collect();
        let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
        let q64 = as_f64(&q);
        let k64 = as_f64(&k);
        let v64 = as_f64(&v);
        let decay64 = as_f64(&decay);
        let beta64 = as_f64(&beta);
        let initial_state64 = as_f64(&initial_state);
        let grad_out64 = as_f64(&grad_out);
        let grad_final_state64 = as_f64(&grad_final_state);
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let q_t = tensor(&q);
        let k_t = tensor(&k);
        let v_t = tensor(&v);
        let decay_t = tensor(&decay);
        let beta_t = tensor(&beta);
        let initial_state_t = tensor(&initial_state);
        let forward_state_t = tensor(&initial_state);
        let forward_out_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
        let checkpoints_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_CHECKPOINTS * state_elements) as u64]).unwrap();
        let grad_out_t = tensor(&grad_out);
        let grad_final_state_t = tensor(&grad_final_state);
        let grad_q_t = MetalTensor::zeros_f32(&ctx, vec![q.len() as u64]).unwrap();
        let grad_k_t = MetalTensor::zeros_f32(&ctx, vec![k.len() as u64]).unwrap();
        let grad_v_t = MetalTensor::zeros_f32(&ctx, vec![v.len() as u64]).unwrap();
        let grad_decay_t = MetalTensor::zeros_f32(&ctx, vec![decay.len() as u64]).unwrap();
        let grad_beta_t = MetalTensor::zeros_f32(&ctx, vec![beta.len() as u64]).unwrap();
        let grad_initial_state_t =
            MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_state_a_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_state_b_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_correction_t = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let residual_t = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_gdn_step_decay_packed_ckpt_f32(
                &ctx,
                encoder,
                &q_t,
                &k_t,
                &v_t,
                &decay_t,
                &beta_t,
                &forward_state_t,
                &forward_out_t,
                &checkpoints_t,
                N_CHECKPOINTS,
                N_TOKENS,
                N_V,
                N_K,
                HEAD_DIM,
            )
        })
        .unwrap();
        let checkpoints = read_back_f32(&checkpoints_t.buffer, N_CHECKPOINTS * state_elements);
        one_shot(&ctx, |encoder| {
            encode_gdn_step_decay_packed_vjp_f32(
                &ctx,
                encoder,
                &q_t,
                &k_t,
                &v_t,
                &decay_t,
                &beta_t,
                &initial_state_t,
                &checkpoints_t,
                N_CHECKPOINTS,
                &grad_out_t,
                &grad_final_state_t,
                &grad_q_t,
                &grad_k_t,
                &grad_v_t,
                &grad_decay_t,
                &grad_beta_t,
                &grad_initial_state_t,
                &grad_state_a_t,
                &grad_state_b_t,
                &grad_correction_t,
                &residual_t,
                N_TOKENS,
                N_V,
                N_K,
                HEAD_DIM,
            )
        })
        .unwrap();
        let actual = GdnStepVjpReference {
            grad_q: read_back_f32(&grad_q_t.buffer, q.len())
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_k: read_back_f32(&grad_k_t.buffer, k.len())
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_v: read_back_f32(&grad_v_t.buffer, v.len())
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_decay: read_back_f32(&grad_decay_t.buffer, decay.len())
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_beta: read_back_f32(&grad_beta_t.buffer, beta.len())
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_state: read_back_f32(&grad_initial_state_t.buffer, state_elements)
                .into_iter()
                .map(f64::from)
                .collect(),
        };
        let expected = gdn_sequence_vjp_f64(
            &q64,
            &k64,
            &v64,
            &decay64,
            &beta64,
            &initial_state64,
            &grad_out64,
            &grad_final_state64,
            N_TOKENS,
            N_V,
            N_K,
            HEAD_DIM,
        );
        for (name, gpu, cpu, tolerance) in [
            ("q", &actual.grad_q, &expected.grad_q, 4e-5),
            ("k", &actual.grad_k, &expected.grad_k, 5e-5),
            ("v", &actual.grad_v, &expected.grad_v, 4e-6),
            ("decay", &actual.grad_decay, &expected.grad_decay, 7e-5),
            ("beta", &actual.grad_beta, &expected.grad_beta, 5e-5),
            (
                "initial_state",
                &actual.grad_state,
                &expected.grad_state,
                5e-6,
            ),
        ] {
            let max_abs = gpu
                .iter()
                .zip(cpu)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0f64, f64::max);
            assert!(max_abs < tolerance, "{name} temporal VJP error {max_abs}");
        }

        let objective =
            |q: &[f64], k: &[f64], v: &[f64], decay: &[f64], beta: &[f64], state: &[f64]| {
                gdn_sequence_objective_f64(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    state,
                    &grad_out64,
                    &grad_final_state64,
                    N_TOKENS,
                    N_V,
                    N_K,
                    HEAD_DIM,
                )
            };
        let epsilon = 1e-5;
        let finite_difference = |values: &[f64], index: usize, evaluate: &dyn Fn(&[f64]) -> f64| {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
        };
        for &index in &[0usize, qk_elements - 1, q64.len() - 1] {
            let fd = finite_difference(&q64, index, &|candidate| {
                objective(candidate, &k64, &v64, &decay64, &beta64, &initial_state64)
            });
            assert!((fd - actual.grad_q[index]).abs() < 4e-5);
            let fd = finite_difference(&k64, index, &|candidate| {
                objective(&q64, candidate, &v64, &decay64, &beta64, &initial_state64)
            });
            assert!((fd - actual.grad_k[index]).abs() < 4e-5);
        }
        for &index in &[0usize, vector_elements - 1, v64.len() - 1] {
            let fd = finite_difference(&v64, index, &|candidate| {
                objective(&q64, &k64, candidate, &decay64, &beta64, &initial_state64)
            });
            assert!((fd - actual.grad_v[index]).abs() < 3e-5);
        }
        for &index in &[0usize, N_V, decay64.len() - 1] {
            let fd = finite_difference(&decay64, index, &|candidate| {
                objective(&q64, &k64, &v64, candidate, &beta64, &initial_state64)
            });
            assert!((fd - actual.grad_decay[index]).abs() < 5e-5);
            let fd = finite_difference(&beta64, index, &|candidate| {
                objective(&q64, &k64, &v64, &decay64, candidate, &initial_state64)
            });
            assert!((fd - actual.grad_beta[index]).abs() < 5e-5);
        }
        for &index in &[0usize, HEAD_DIM, initial_state64.len() - 1] {
            let fd = finite_difference(&initial_state64, index, &|candidate| {
                objective(&q64, &k64, &v64, &decay64, &beta64, candidate)
            });
            assert!((fd - actual.grad_state[index]).abs() < 3e-5);
        }

        let direction = |len: usize, stride: usize| {
            (0..len)
                .map(|index| ((index * stride + 3) % 37) as f64 * 0.0007 - 0.012)
                .collect::<Vec<_>>()
        };
        let dq = direction(q64.len(), 5);
        let dk = direction(k64.len(), 7);
        let dv = direction(v64.len(), 11);
        let ddecay = direction(decay64.len(), 13);
        let dbeta = direction(beta64.len(), 17);
        let dstate = direction(initial_state64.len(), 19);
        let inner = |gradient: &[f64], tangent: &[f64]| {
            gradient
                .iter()
                .zip(tangent)
                .map(|(gradient, tangent)| gradient * tangent)
                .sum::<f64>()
        };
        let reverse_directional = inner(&actual.grad_q, &dq)
            + inner(&actual.grad_k, &dk)
            + inner(&actual.grad_v, &dv)
            + inner(&actual.grad_decay, &ddecay)
            + inner(&actual.grad_beta, &dbeta)
            + inner(&actual.grad_state, &dstate);
        let shift = |base: &[f64], tangent: &[f64], amount: f64| {
            base.iter()
                .zip(tangent)
                .map(|(base, tangent)| base + amount * tangent)
                .collect::<Vec<_>>()
        };
        let plus = objective(
            &shift(&q64, &dq, epsilon),
            &shift(&k64, &dk, epsilon),
            &shift(&v64, &dv, epsilon),
            &shift(&decay64, &ddecay, epsilon),
            &shift(&beta64, &dbeta, epsilon),
            &shift(&initial_state64, &dstate, epsilon),
        );
        let minus = objective(
            &shift(&q64, &dq, -epsilon),
            &shift(&k64, &dk, -epsilon),
            &shift(&v64, &dv, -epsilon),
            &shift(&decay64, &ddecay, -epsilon),
            &shift(&beta64, &dbeta, -epsilon),
            &shift(&initial_state64, &dstate, -epsilon),
        );
        let forward_directional = (plus - minus) / (2.0 * epsilon);
        assert!(
            (forward_directional - reverse_directional).abs() < 6e-5,
            "temporal adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
        );
        for (tensor, original) in [
            (&q_t, &q),
            (&k_t, &k),
            (&v_t, &v),
            (&decay_t, &decay),
            (&beta_t, &beta),
            (&initial_state_t, &initial_state),
            (&checkpoints_t, &checkpoints),
            (&grad_out_t, &grad_out),
            (&grad_final_state_t, &grad_final_state),
        ] {
            assert_eq!(
                read_back_f32(&tensor.buffer, original.len())
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn gdn_step_decay_vjp_rejects_unsafe_contracts() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_V: usize = 3;
        const N_K: usize = 1;
        const HEAD_DIM: usize = 128;
        let qk_elements = N_K * HEAD_DIM;
        let vector_elements = N_V * HEAD_DIM;
        let state_elements = N_V * HEAD_DIM * HEAD_DIM;
        let q = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let k = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let v = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let decay = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let beta = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_out = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let grad_state_out = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_q = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let grad_k = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let grad_v = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let grad_decay = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_beta = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_c = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let residual = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let invoke = |encoder: &KernelEncoder, grad_q_output: &MetalTensor| {
            encode_gdn_step_decay_vjp_f32(
                &ctx,
                encoder,
                &q,
                &k,
                &v,
                &decay,
                &beta,
                &state,
                &grad_out,
                &grad_state_out,
                grad_q_output,
                &grad_k,
                &grad_v,
                &grad_decay,
                &grad_beta,
                &grad_state,
                &grad_c,
                &residual,
                N_V,
                N_K,
                HEAD_DIM,
            )
        };

        let command = ctx.queue.commandBuffer().unwrap();
        let concurrent = KernelEncoder::begin_concurrent(&command);
        invoke(&concurrent, &grad_q).expect_err("concurrent encoder must fail");
        concurrent.end();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke(&encoder, &q).expect_err("output/input alias must fail");
        encoder.end();

        let mut read_only = grad_q.clone();
        read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke(&encoder, &read_only).expect_err("read-only output must fail");
        encoder.end();

        let short = MetalTensor::zeros_f32(&ctx, vec![(qk_elements - 1) as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke(&encoder, &short).expect_err("short output must fail");
        encoder.end();
    }

    #[test]
    fn gdn_packed_temporal_vjps_enforce_safe_tapes_and_storage() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_TOKENS: usize = 3;
        const N_V: usize = 3;
        const N_K: usize = 1;
        const HEAD_DIM: usize = 128;
        let qk_elements = N_K * HEAD_DIM;
        let vector_elements = N_V * HEAD_DIM;
        let state_elements = vector_elements * HEAD_DIM;
        let q = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let k = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let v = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
        let decay = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
        let beta = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
        let initial_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let full_checkpoints =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * state_elements) as u64]).unwrap();
        let short_checkpoints =
            MetalTensor::zeros_f32(&ctx, vec![((N_TOKENS - 2) * state_elements) as u64]).unwrap();
        let grad_out =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
        let grad_final_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_q = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let grad_k = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let grad_v =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * vector_elements) as u64]).unwrap();
        let grad_decay = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
        let grad_beta = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * N_V) as u64]).unwrap();
        let grad_initial_state = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let state_scratch_a = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let state_scratch_b = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let correction_scratch =
            MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let residual_scratch = MetalTensor::zeros_f32(&ctx, vec![vector_elements as u64]).unwrap();
        let invoke_recurrence = |encoder: &KernelEncoder,
                                 q_input: &MetalTensor,
                                 checkpoints: &MetalTensor,
                                 n_checkpoints: usize,
                                 grad_q_output: &MetalTensor| {
            encode_gdn_step_decay_packed_vjp_f32(
                &ctx,
                encoder,
                q_input,
                &k,
                &v,
                &decay,
                &beta,
                &initial_state,
                checkpoints,
                n_checkpoints,
                &grad_out,
                &grad_final_state,
                grad_q_output,
                &grad_k,
                &grad_v,
                &grad_decay,
                &grad_beta,
                &grad_initial_state,
                &state_scratch_a,
                &state_scratch_b,
                &correction_scratch,
                &residual_scratch,
                N_TOKENS,
                N_V,
                N_K,
                HEAD_DIM,
            )
        };
        one_shot(&ctx, |encoder| {
            invoke_recurrence(encoder, &q, &full_checkpoints, N_TOKENS, &grad_q)
        })
        .unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let concurrent = KernelEncoder::begin_concurrent(&command);
        invoke_recurrence(&concurrent, &q, &full_checkpoints, N_TOKENS, &grad_q)
            .expect_err("concurrent temporal recurrence must fail");
        concurrent.end();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke_recurrence(&encoder, &q, &short_checkpoints, N_TOKENS - 2, &grad_q)
            .expect_err("missing pre-state checkpoint must fail");
        encoder.end();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke_recurrence(&encoder, &q, &full_checkpoints, N_TOKENS, &q)
            .expect_err("packed gradient/input alias must fail");
        encoder.end();

        let mut malformed_q = q.clone();
        malformed_q.shape = vec![u64::MAX, 2];
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke_recurrence(&encoder, &malformed_q, &full_checkpoints, N_TOKENS, &grad_q)
            .expect_err("malformed packed shape must fail without panicking");
        encoder.end();

        const CONV_N_V: usize = 2;
        let conv_v_elements = CONV_N_V * HEAD_DIM;
        let conv_dim = 2 * qk_elements + conv_v_elements;
        let conv_state_elements = 3 * conv_dim;
        let qkv = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_dim) as u64]).unwrap();
        let conv_initial_state =
            MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
        let conv_full_checkpoints =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_state_elements) as u64]).unwrap();
        let conv_weight = MetalTensor::zeros_f32(&ctx, vec![(4 * conv_dim) as u64]).unwrap();
        let conv_grad_q =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let conv_grad_k =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let conv_grad_v =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_v_elements) as u64]).unwrap();
        let conv_grad_final_state =
            MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
        let grad_qkv = MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * conv_dim) as u64]).unwrap();
        let conv_grad_initial_state =
            MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
        let conv_state_scratch_a =
            MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
        let conv_state_scratch_b =
            MetalTensor::zeros_f32(&ctx, vec![conv_state_elements as u64]).unwrap();
        let invoke_conv = |encoder: &KernelEncoder, grad_qkv_output: &MetalTensor| {
            encode_ssm_conv_silu_split_packed_vjp_f32(
                &ctx,
                encoder,
                &qkv,
                &conv_initial_state,
                &conv_full_checkpoints,
                N_TOKENS,
                &conv_weight,
                &conv_grad_q,
                &conv_grad_k,
                &conv_grad_v,
                &conv_grad_final_state,
                grad_qkv_output,
                &conv_grad_initial_state,
                &conv_state_scratch_a,
                &conv_state_scratch_b,
                N_TOKENS,
                N_K,
                CONV_N_V,
                HEAD_DIM,
            )
        };
        one_shot(&ctx, |encoder| invoke_conv(encoder, &grad_qkv)).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        invoke_conv(&encoder, &qkv).expect_err("packed conv gradient/input alias must fail");
        encoder.end();
    }

    #[test]
    fn gdn_scalar_chain_vjps_match_piecewise_oracles() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let source = [-15.0f32, -3.0, -0.25, 0.0, 0.75, 4.0, 15.0];
        let sigmoid_output: Vec<f32> = source
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect();
        let sigmoid_grad: Vec<f32> = (0..source.len())
            .map(|index| index as f32 * 0.07 - 0.19)
            .collect();
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let sigmoid_output_t = tensor(&sigmoid_output);
        let sigmoid_grad_t = tensor(&sigmoid_grad);
        let sigmoid_source_grad_t =
            MetalTensor::zeros_f32(&ctx, vec![source.len() as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_sigmoid_output_vjp_f32(
                &ctx,
                encoder,
                &sigmoid_output_t,
                &sigmoid_grad_t,
                &sigmoid_source_grad_t,
            )
        })
        .unwrap();
        let actual_sigmoid = read_back_f32(&sigmoid_source_grad_t.buffer, source.len());
        for index in 0..source.len() {
            let expected =
                sigmoid_grad[index] * sigmoid_output[index] * (1.0 - sigmoid_output[index]);
            assert!((actual_sigmoid[index] - expected).abs() < 2e-7);
            let epsilon = 1e-4f64;
            let objective = |delta: f64| {
                let value = f64::from(source[index]) + delta;
                f64::from(sigmoid_grad[index]) / (1.0 + (-value).exp())
            };
            let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual_sigmoid[index])).abs() < 2e-6);
        }

        let totals = [
            -25.0f32, -20.25, -20.0, -19.75, 0.0, 19.75, 20.0, 20.25, 25.0,
        ];
        let dt_bias: Vec<f32> = (0..totals.len())
            .map(|index| (index as f32 - 4.0) * 0.125)
            .collect();
        let alpha: Vec<f32> = totals
            .iter()
            .zip(&dt_bias)
            .map(|(total, bias)| total - bias)
            .collect();
        let a_log: Vec<f32> = (0..totals.len())
            .map(|index| -0.015 - index as f32 * 0.004)
            .collect();
        let grad_decay: Vec<f32> = (0..totals.len())
            .map(|index| index as f32 * 0.031 - 0.11)
            .collect();
        let alpha_t = tensor(&alpha);
        let dt_t = tensor(&dt_bias);
        let a_log_t = tensor(&a_log);
        let decay_t = MetalTensor::zeros_f32(&ctx, vec![totals.len() as u64]).unwrap();
        let grad_decay_t = tensor(&grad_decay);
        let grad_alpha_t = MetalTensor::zeros_f32(&ctx, vec![totals.len() as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_gdn_decay_chain_f32(&ctx, encoder, &alpha_t, &dt_t, &a_log_t, &decay_t)?;
            encode_gdn_decay_chain_vjp_f32(
                &ctx,
                encoder,
                &alpha_t,
                &dt_t,
                &a_log_t,
                &decay_t,
                &grad_decay_t,
                &grad_alpha_t,
            )
        })
        .unwrap();
        let decay = read_back_f32(&decay_t.buffer, totals.len());
        let actual_alpha = read_back_f32(&grad_alpha_t.buffer, totals.len());
        for index in 0..totals.len() {
            let value = totals[index];
            let softplus_derivative = if value > 20.0 {
                1.0
            } else if value < -20.0 {
                value.exp()
            } else {
                let exponential = value.exp();
                exponential / (1.0 + exponential)
            };
            let expected = grad_decay[index] * decay[index] * a_log[index] * softplus_derivative;
            assert!(
                (actual_alpha[index] - expected).abs() < 2e-6,
                "decay chain index {index}: {} != {expected}",
                actual_alpha[index]
            );
        }
        for &index in &[0usize, 1, 3, 4, 5, 7, 8] {
            let epsilon = 1e-4f64;
            let objective = |delta: f64| {
                let value = f64::from(alpha[index]) + f64::from(dt_bias[index]) + delta;
                let softplus = if value > 20.0 {
                    value
                } else if value < -20.0 {
                    value.exp()
                } else {
                    (1.0 + value.exp()).ln()
                };
                f64::from(grad_decay[index]) * (softplus * f64::from(a_log[index])).exp()
            };
            let finite_difference = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual_alpha[index])).abs() < 2e-5);
        }
    }

    #[test]
    fn ssm_conv_silu_vjp_matches_shifted_state_oracle() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_K: usize = 1;
        const N_V: usize = 1;
        const HEAD_DIM: usize = 128;
        const CONV_DIM: usize = (2 * N_K + N_V) * HEAD_DIM;
        let qkv: Vec<f32> = (0..CONV_DIM)
            .map(|index| ((index * 7 + 2) % 31) as f32 * 0.009 - 0.13)
            .collect();
        let state: Vec<f32> = (0..3 * CONV_DIM)
            .map(|index| ((index * 11 + 5) % 37) as f32 * 0.006 - 0.105)
            .collect();
        let weight: Vec<f32> = (0..4 * CONV_DIM)
            .map(|index| ((index * 13 + 1) % 41) as f32 * 0.004 - 0.077)
            .collect();
        let grad_out: Vec<f32> = (0..CONV_DIM)
            .map(|index| ((index * 17 + 3) % 43) as f32 * 0.008 - 0.16)
            .collect();
        let grad_state_out: Vec<f32> = (0..3 * CONV_DIM)
            .map(|index| ((index * 19 + 7) % 47) as f32 * 0.005 - 0.11)
            .collect();
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let qkv_t = tensor(&qkv);
        let state_t = tensor(&state);
        let weight_t = tensor(&weight);
        let grad_out_t = tensor(&grad_out);
        let grad_state_out_t = tensor(&grad_state_out);
        let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![CONV_DIM as u64]).unwrap();
        let grad_state_t = MetalTensor::zeros_f32(&ctx, vec![(3 * CONV_DIM) as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_ssm_conv_silu_vjp_f32(
                &ctx,
                encoder,
                &qkv_t,
                &state_t,
                &weight_t,
                &grad_out_t,
                &grad_state_out_t,
                &grad_qkv_t,
                &grad_state_t,
                CONV_DIM,
            )
        })
        .unwrap();
        let actual_qkv = read_back_f32(&grad_qkv_t.buffer, CONV_DIM);
        let actual_state = read_back_f32(&grad_state_t.buffer, 3 * CONV_DIM);
        let split_grad_q_t = tensor(&grad_out[..HEAD_DIM]);
        let split_grad_k_t = tensor(&grad_out[HEAD_DIM..2 * HEAD_DIM]);
        let split_grad_v_t = tensor(&grad_out[2 * HEAD_DIM..]);
        let split_grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![CONV_DIM as u64]).unwrap();
        let split_grad_state_t = MetalTensor::zeros_f32(&ctx, vec![(3 * CONV_DIM) as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_ssm_conv_silu_split_vjp_f32(
                &ctx,
                encoder,
                &qkv_t,
                &state_t,
                &weight_t,
                &split_grad_q_t,
                &split_grad_k_t,
                &split_grad_v_t,
                &grad_state_out_t,
                &split_grad_qkv_t,
                &split_grad_state_t,
                N_K,
                N_V,
                HEAD_DIM,
            )
        })
        .unwrap();
        let split_qkv = read_back_f32(&split_grad_qkv_t.buffer, CONV_DIM);
        let split_state = read_back_f32(&split_grad_state_t.buffer, 3 * CONV_DIM);
        let split_qkv_error = split_qkv
            .iter()
            .zip(&actual_qkv)
            .map(|(split, joined)| (split - joined).abs())
            .fold(0.0f32, f32::max);
        let split_state_error = split_state
            .iter()
            .zip(&actual_state)
            .map(|(split, joined)| (split - joined).abs())
            .fold(0.0f32, f32::max);
        assert!(split_qkv_error < 2e-7, "split QKV error {split_qkv_error}");
        assert!(
            split_state_error < 2e-7,
            "split state error {split_state_error}"
        );
        let mut expected_qkv = vec![0.0f64; CONV_DIM];
        let mut expected_state = vec![0.0f64; 3 * CONV_DIM];
        for channel in 0..CONV_DIM {
            let preactivation = (0..3)
                .map(|row| {
                    f64::from(weight[channel * 4 + row])
                        * f64::from(state[row * CONV_DIM + channel])
                })
                .sum::<f64>()
                + f64::from(weight[channel * 4 + 3]) * f64::from(qkv[channel]);
            let sigmoid = 1.0 / (1.0 + (-preactivation).exp());
            let derivative = sigmoid * (1.0 + preactivation * (1.0 - sigmoid));
            let grad_preactivation = f64::from(grad_out[channel]) * derivative;
            expected_qkv[channel] = grad_preactivation * f64::from(weight[channel * 4 + 3])
                + f64::from(grad_state_out[2 * CONV_DIM + channel]);
            expected_state[channel] = grad_preactivation * f64::from(weight[channel * 4]);
            expected_state[CONV_DIM + channel] = grad_preactivation
                * f64::from(weight[channel * 4 + 1])
                + f64::from(grad_state_out[channel]);
            expected_state[2 * CONV_DIM + channel] = grad_preactivation
                * f64::from(weight[channel * 4 + 2])
                + f64::from(grad_state_out[CONV_DIM + channel]);
        }
        let max_qkv = actual_qkv
            .iter()
            .zip(&expected_qkv)
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        let max_state = actual_state
            .iter()
            .zip(&expected_state)
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(max_qkv < 2e-6, "conv qkv VJP error {max_qkv}");
        assert!(max_state < 2e-6, "conv state VJP error {max_state}");

        let objective = |qkv: &[f64], state: &[f64]| {
            let mut value = 0.0f64;
            for channel in 0..CONV_DIM {
                let preactivation = (0..3)
                    .map(|row| {
                        f64::from(weight[channel * 4 + row]) * state[row * CONV_DIM + channel]
                    })
                    .sum::<f64>()
                    + f64::from(weight[channel * 4 + 3]) * qkv[channel];
                value +=
                    f64::from(grad_out[channel]) * preactivation / (1.0 + (-preactivation).exp());
                value += f64::from(grad_state_out[channel]) * state[CONV_DIM + channel];
                value +=
                    f64::from(grad_state_out[CONV_DIM + channel]) * state[2 * CONV_DIM + channel];
                value += f64::from(grad_state_out[2 * CONV_DIM + channel]) * qkv[channel];
            }
            value
        };
        let qkv64 = qkv.iter().copied().map(f64::from).collect::<Vec<_>>();
        let state64 = state.iter().copied().map(f64::from).collect::<Vec<_>>();
        let epsilon = 1e-5;
        for &index in &[0usize, 128, CONV_DIM - 1] {
            let mut plus = qkv64.clone();
            let mut minus = qkv64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&plus, &state64) - objective(&minus, &state64)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual_qkv[index])).abs() < 2e-5);
        }
        for &index in &[
            0usize,
            CONV_DIM - 1,
            CONV_DIM,
            2 * CONV_DIM + 128,
            3 * CONV_DIM - 1,
        ] {
            let mut plus = state64.clone();
            let mut minus = state64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&qkv64, &plus) - objective(&qkv64, &minus)) / (2.0 * epsilon);
            assert!((finite_difference - f64::from(actual_state[index])).abs() < 2e-5);
        }
    }

    #[test]
    fn ssm_conv_silu_vjp_replays_forward_accumulation_order() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let qkv = [1.0f32];
        let state = [1.0e8f32, -1.0e8, 1.0];
        let weight = [1.0f32; 4];
        let grad_out = [1.0f32];
        let grad_state_out = [0.0f32; 3];
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let qkv_t = tensor(&qkv);
        let forward_state_t = tensor(&state);
        let weight_t = tensor(&weight);
        let forward_out_t = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_ssm_conv_silu_f32(
                &ctx,
                encoder,
                &qkv_t,
                &forward_state_t,
                &weight_t,
                &forward_out_t,
                1,
            )
        })
        .unwrap();
        let forward_out = read_back_f32(&forward_out_t.buffer, 1)[0];
        let expected_forward = 2.0f32 / (1.0 + (-2.0f32).exp());
        assert!((forward_out - expected_forward).abs() < 2e-6);

        let state_t = tensor(&state);
        let grad_out_t = tensor(&grad_out);
        let grad_state_out_t = tensor(&grad_state_out);
        let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let grad_state_t = MetalTensor::zeros_f32(&ctx, vec![3]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_ssm_conv_silu_vjp_f32(
                &ctx,
                encoder,
                &qkv_t,
                &state_t,
                &weight_t,
                &grad_out_t,
                &grad_state_out_t,
                &grad_qkv_t,
                &grad_state_t,
                1,
            )
        })
        .unwrap();
        let sigmoid = 1.0f32 / (1.0 + (-2.0f32).exp());
        let expected_gradient = sigmoid * (1.0 + 2.0 * (1.0 - sigmoid));
        let grad_qkv = read_back_f32(&grad_qkv_t.buffer, 1)[0];
        let grad_state = read_back_f32(&grad_state_t.buffer, 3);
        assert!((grad_qkv - expected_gradient).abs() < 2e-6);
        for gradient in grad_state {
            assert!((gradient - expected_gradient).abs() < 2e-6);
        }
    }

    #[test]
    fn ssm_conv_silu_split_packed_vjp_matches_temporal_oracle_and_adjoint() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_TOKENS: usize = 4;
        const N_CHECKPOINTS: usize = N_TOKENS - 1;
        const N_K: usize = 1;
        const N_V: usize = 2;
        const HEAD_DIM: usize = 128;
        let qk_elements = N_K * HEAD_DIM;
        let v_elements = N_V * HEAD_DIM;
        let conv_dim = 2 * qk_elements + v_elements;
        let state_elements = 3 * conv_dim;
        let qkv: Vec<f32> = (0..N_TOKENS * conv_dim)
            .map(|index| ((index * 7 + 3) % 41) as f32 * 0.0023 - 0.045)
            .collect();
        let initial_state: Vec<f32> = (0..state_elements)
            .map(|index| ((index * 11 + 5) % 43) as f32 * 0.0019 - 0.039)
            .collect();
        let weight: Vec<f32> = (0..4 * conv_dim)
            .map(|index| ((index * 13 + 1) % 47) as f32 * 0.0031 - 0.071)
            .collect();
        let grad_q: Vec<f32> = (0..N_TOKENS * qk_elements)
            .map(|index| ((index * 17 + 7) % 53) as f32 * 0.0017 - 0.043)
            .collect();
        let grad_k: Vec<f32> = (0..N_TOKENS * qk_elements)
            .map(|index| ((index * 19 + 2) % 59) as f32 * 0.0015 - 0.041)
            .collect();
        let grad_v: Vec<f32> = (0..N_TOKENS * v_elements)
            .map(|index| ((index * 23 + 4) % 61) as f32 * 0.0013 - 0.037)
            .collect();
        let grad_final_state: Vec<f32> = (0..state_elements)
            .map(|index| ((index * 29 + 11) % 67) as f32 * 0.00031 - 0.009)
            .collect();
        let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
        let qkv64 = as_f64(&qkv);
        let initial_state64 = as_f64(&initial_state);
        let weight64 = as_f64(&weight);
        let grad_q64 = as_f64(&grad_q);
        let grad_k64 = as_f64(&grad_k);
        let grad_v64 = as_f64(&grad_v);
        let grad_final_state64 = as_f64(&grad_final_state);
        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let qkv_t = tensor(&qkv);
        let initial_state_t = tensor(&initial_state);
        let forward_state_t = tensor(&initial_state);
        let checkpoints_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_CHECKPOINTS * state_elements) as u64]).unwrap();
        let weight_t = tensor(&weight);
        let forward_q_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let forward_k_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * qk_elements) as u64]).unwrap();
        let forward_v_t =
            MetalTensor::zeros_f32(&ctx, vec![(N_TOKENS * v_elements) as u64]).unwrap();
        let grad_q_t = tensor(&grad_q);
        let grad_k_t = tensor(&grad_k);
        let grad_v_t = tensor(&grad_v);
        let grad_final_state_t = tensor(&grad_final_state);
        let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![qkv.len() as u64]).unwrap();
        let grad_initial_state_t =
            MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_state_a_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_state_b_t = MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        one_shot(&ctx, |encoder| {
            encode_gdn_prep_packed_ckpt_f32(
                &ctx,
                encoder,
                &qkv_t,
                &forward_state_t,
                &weight_t,
                &forward_q_t,
                &forward_k_t,
                &forward_v_t,
                &checkpoints_t,
                N_TOKENS,
                N_CHECKPOINTS,
                N_K,
                N_V,
                HEAD_DIM,
            )
        })
        .unwrap();
        let checkpoints = read_back_f32(&checkpoints_t.buffer, N_CHECKPOINTS * state_elements);
        one_shot(&ctx, |encoder| {
            encode_ssm_conv_silu_split_packed_vjp_f32(
                &ctx,
                encoder,
                &qkv_t,
                &initial_state_t,
                &checkpoints_t,
                N_CHECKPOINTS,
                &weight_t,
                &grad_q_t,
                &grad_k_t,
                &grad_v_t,
                &grad_final_state_t,
                &grad_qkv_t,
                &grad_initial_state_t,
                &grad_state_a_t,
                &grad_state_b_t,
                N_TOKENS,
                N_K,
                N_V,
                HEAD_DIM,
            )
        })
        .unwrap();
        let actual = SsmConvSequenceVjpReference {
            grad_qkv: read_back_f32(&grad_qkv_t.buffer, qkv.len())
                .into_iter()
                .map(f64::from)
                .collect(),
            grad_state: read_back_f32(&grad_initial_state_t.buffer, state_elements)
                .into_iter()
                .map(f64::from)
                .collect(),
        };
        let expected = ssm_conv_sequence_vjp_f64(
            &qkv64,
            &initial_state64,
            &weight64,
            &grad_q64,
            &grad_k64,
            &grad_v64,
            &grad_final_state64,
            N_TOKENS,
            qk_elements,
            v_elements,
        );
        for (name, gpu, cpu) in [
            ("qkv", &actual.grad_qkv, &expected.grad_qkv),
            ("initial_state", &actual.grad_state, &expected.grad_state),
        ] {
            let max_abs = gpu
                .iter()
                .zip(cpu)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0f64, f64::max);
            assert!(max_abs < 3e-6, "{name} temporal conv VJP error {max_abs}");
        }

        let objective = |qkv: &[f64], state: &[f64]| {
            ssm_conv_sequence_objective_f64(
                qkv,
                state,
                &weight64,
                &grad_q64,
                &grad_k64,
                &grad_v64,
                &grad_final_state64,
                N_TOKENS,
                qk_elements,
                v_elements,
            )
        };
        let epsilon = 1e-5;
        for &index in &[0usize, conv_dim - 1, qkv64.len() - 1] {
            let mut plus = qkv64.clone();
            let mut minus = qkv64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference = (objective(&plus, &initial_state64)
                - objective(&minus, &initial_state64))
                / (2.0 * epsilon);
            assert!((finite_difference - actual.grad_qkv[index]).abs() < 2e-5);
        }
        for &index in &[0usize, conv_dim, initial_state64.len() - 1] {
            let mut plus = initial_state64.clone();
            let mut minus = initial_state64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let finite_difference =
                (objective(&qkv64, &plus) - objective(&qkv64, &minus)) / (2.0 * epsilon);
            assert!((finite_difference - actual.grad_state[index]).abs() < 2e-5);
        }

        let direction = |len: usize, stride: usize| {
            (0..len)
                .map(|index| ((index * stride + 3) % 31) as f64 * 0.0009 - 0.013)
                .collect::<Vec<_>>()
        };
        let dqkv = direction(qkv64.len(), 5);
        let dstate = direction(initial_state64.len(), 7);
        let inner = |gradient: &[f64], tangent: &[f64]| {
            gradient
                .iter()
                .zip(tangent)
                .map(|(gradient, tangent)| gradient * tangent)
                .sum::<f64>()
        };
        let reverse_directional =
            inner(&actual.grad_qkv, &dqkv) + inner(&actual.grad_state, &dstate);
        let shift = |base: &[f64], tangent: &[f64], amount: f64| {
            base.iter()
                .zip(tangent)
                .map(|(base, tangent)| base + amount * tangent)
                .collect::<Vec<_>>()
        };
        let forward_directional = (objective(
            &shift(&qkv64, &dqkv, epsilon),
            &shift(&initial_state64, &dstate, epsilon),
        ) - objective(
            &shift(&qkv64, &dqkv, -epsilon),
            &shift(&initial_state64, &dstate, -epsilon),
        )) / (2.0 * epsilon);
        assert!(
            (forward_directional - reverse_directional).abs() < 2e-5,
            "temporal conv adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
        );
        for (tensor, original) in [
            (&qkv_t, &qkv),
            (&initial_state_t, &initial_state),
            (&checkpoints_t, &checkpoints),
            (&weight_t, &weight),
            (&grad_q_t, &grad_q),
            (&grad_k_t, &grad_k),
            (&grad_v_t, &grad_v),
            (&grad_final_state_t, &grad_final_state),
        ] {
            assert_eq!(
                read_back_f32(&tensor.buffer, original.len())
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn gdn_envelope_vjps_compose_to_full_step_adjoint() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const N_K: usize = 1;
        const N_V: usize = 2;
        const HEAD_DIM: usize = 128;
        const L2_EPS: f32 = 1e-6;
        const RMS_EPS: f32 = HEAD_DIM as f32 * 1e-6;
        let qk_elements = N_K * HEAD_DIM;
        let v_elements = N_V * HEAD_DIM;
        let conv_dim = 2 * qk_elements + v_elements;
        let state_elements = N_V * HEAD_DIM * HEAD_DIM;
        let qkv_now: Vec<f32> = (0..conv_dim)
            .map(|index| ((index * 7 + 3) % 41) as f32 * 0.006 - 0.115)
            .collect();
        let conv_state: Vec<f32> = (0..3 * conv_dim)
            .map(|index| ((index * 11 + 5) % 43) as f32 * 0.004 - 0.083)
            .collect();
        let conv_weight: Vec<f32> = (0..4 * conv_dim)
            .map(|index| ((index * 13 + 1) % 47) as f32 * 0.003 - 0.069)
            .collect();
        let alpha_source = [-0.7f32, 0.45];
        let dt_bias = [0.12f32, -0.08];
        let a_log = [-0.09f32, -0.14];
        let beta_source = [-0.35f32, 0.8];
        let recurrence_state: Vec<f32> = (0..state_elements)
            .map(|index| ((index * 17 + 7) % 53) as f32 * 0.0008 - 0.021)
            .collect();
        let z: Vec<f32> = (0..v_elements)
            .map(|index| ((index * 19 + 2) % 59) as f32 * 0.05 - 1.35)
            .collect();
        let norm_weight: Vec<f32> = (0..HEAD_DIM)
            .map(|index| 0.62 + (index % 17) as f32 * 0.029)
            .collect();
        let grad_y: Vec<f32> = (0..v_elements)
            .map(|index| ((index * 23 + 3) % 61) as f32 * 0.004 - 0.12)
            .collect();
        let grad_recurrence_state: Vec<f32> = (0..state_elements)
            .map(|index| ((index * 29 + 11) % 67) as f32 * 0.00006 - 0.002)
            .collect();
        let grad_conv_state: Vec<f32> = (0..3 * conv_dim)
            .map(|index| ((index * 31 + 13) % 71) as f32 * 0.0017 - 0.052)
            .collect();

        let mut conv_output = vec![0.0f32; conv_dim];
        for channel in 0..conv_dim {
            let mut preactivation = 0.0f32;
            for row in 0..3 {
                preactivation +=
                    conv_weight[channel * 4 + row] * conv_state[row * conv_dim + channel];
            }
            preactivation += conv_weight[channel * 4 + 3] * qkv_now[channel];
            conv_output[channel] = preactivation / (1.0 + (-preactivation).exp());
        }
        let q_raw = conv_output[..qk_elements].to_vec();
        let k_raw = conv_output[qk_elements..2 * qk_elements].to_vec();
        let v = conv_output[2 * qk_elements..].to_vec();
        let normalize = |input: &[f32]| {
            let mut output = input.to_vec();
            for head in 0..N_K {
                let base = head * HEAD_DIM;
                let radius = input[base..base + HEAD_DIM]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(L2_EPS);
                for index in 0..HEAD_DIM {
                    output[base + index] /= radius;
                }
            }
            output
        };
        let q = normalize(&q_raw);
        let k = normalize(&k_raw);
        let decay: Vec<f32> = (0..N_V)
            .map(|head| {
                let value = alpha_source[head] + dt_bias[head];
                let softplus = if value > 20.0 {
                    value
                } else if value < -20.0 {
                    value.exp()
                } else {
                    (1.0 + value.exp()).ln()
                };
                (softplus * a_log[head]).exp()
            })
            .collect();
        let beta: Vec<f32> = beta_source
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect();
        let mut recurrence_output = vec![0.0f32; v_elements];
        for hi in 0..N_V {
            let hk = hi % N_K;
            for dv in 0..HEAD_DIM {
                let vector_index = hi * HEAD_DIM + dv;
                let row_offset = vector_index * HEAD_DIM;
                let prediction = (0..HEAD_DIM)
                    .map(|dk| decay[hi] * recurrence_state[row_offset + dk] * k[hk * HEAD_DIM + dk])
                    .sum::<f32>();
                let correction = beta[hi] * (v[vector_index] - prediction);
                recurrence_output[vector_index] = (0..HEAD_DIM)
                    .map(|dk| {
                        (decay[hi] * recurrence_state[row_offset + dk]
                            + correction * k[hk * HEAD_DIM + dk])
                            * q[hk * HEAD_DIM + dk]
                    })
                    .sum();
            }
        }

        let tensor = |values: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(values),
                vec![values.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let qkv_t = tensor(&qkv_now);
        let conv_state_t = tensor(&conv_state);
        let conv_weight_t = tensor(&conv_weight);
        let q_raw_t = tensor(&q_raw);
        let k_raw_t = tensor(&k_raw);
        let q_t = tensor(&q);
        let k_t = tensor(&k);
        let v_t = tensor(&v);
        let alpha_t = tensor(&alpha_source);
        let dt_t = tensor(&dt_bias);
        let a_log_t = tensor(&a_log);
        let decay_t = tensor(&decay);
        let beta_t = tensor(&beta);
        let recurrence_state_t = tensor(&recurrence_state);
        let recurrence_output_t = tensor(&recurrence_output);
        let z_t = tensor(&z);
        let norm_weight_t = tensor(&norm_weight);
        let grad_y_t = tensor(&grad_y);
        let grad_recurrence_state_t = tensor(&grad_recurrence_state);
        let grad_conv_state_t = tensor(&grad_conv_state);

        let grad_recurrence_output_t =
            MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
        let grad_z_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
        let grad_q_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let grad_k_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let grad_v_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
        let grad_decay_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_beta_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_recurrence_state_in_t =
            MetalTensor::zeros_f32(&ctx, vec![state_elements as u64]).unwrap();
        let grad_c_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
        let residual_t = MetalTensor::zeros_f32(&ctx, vec![v_elements as u64]).unwrap();
        let grad_q_raw_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let grad_k_raw_t = MetalTensor::zeros_f32(&ctx, vec![qk_elements as u64]).unwrap();
        let grad_alpha_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_beta_source_t = MetalTensor::zeros_f32(&ctx, vec![N_V as u64]).unwrap();
        let grad_qkv_t = MetalTensor::zeros_f32(&ctx, vec![conv_dim as u64]).unwrap();
        let grad_conv_state_in_t =
            MetalTensor::zeros_f32(&ctx, vec![(3 * conv_dim) as u64]).unwrap();

        one_shot(&ctx, |encoder| {
            encode_rmsnorm_gated_vjp_f32(
                &ctx,
                encoder,
                &recurrence_output_t,
                &norm_weight_t,
                &z_t,
                &grad_y_t,
                &grad_recurrence_output_t,
                &grad_z_t,
                N_V,
                HEAD_DIM,
                RMS_EPS,
            )?;
            encode_gdn_step_decay_vjp_f32(
                &ctx,
                encoder,
                &q_t,
                &k_t,
                &v_t,
                &decay_t,
                &beta_t,
                &recurrence_state_t,
                &grad_recurrence_output_t,
                &grad_recurrence_state_t,
                &grad_q_t,
                &grad_k_t,
                &grad_v_t,
                &grad_decay_t,
                &grad_beta_t,
                &grad_recurrence_state_in_t,
                &grad_c_t,
                &residual_t,
                N_V,
                N_K,
                HEAD_DIM,
            )?;
            encode_l2_norm_vjp_batched_f32(
                &ctx,
                encoder,
                &q_raw_t,
                &grad_q_t,
                &grad_q_raw_t,
                N_K,
                HEAD_DIM,
                L2_EPS,
            )?;
            encode_l2_norm_vjp_batched_f32(
                &ctx,
                encoder,
                &k_raw_t,
                &grad_k_t,
                &grad_k_raw_t,
                N_K,
                HEAD_DIM,
                L2_EPS,
            )?;
            encode_gdn_decay_chain_vjp_f32(
                &ctx,
                encoder,
                &alpha_t,
                &dt_t,
                &a_log_t,
                &decay_t,
                &grad_decay_t,
                &grad_alpha_t,
            )?;
            encode_sigmoid_output_vjp_f32(
                &ctx,
                encoder,
                &beta_t,
                &grad_beta_t,
                &grad_beta_source_t,
            )?;
            encode_ssm_conv_silu_split_vjp_f32(
                &ctx,
                encoder,
                &qkv_t,
                &conv_state_t,
                &conv_weight_t,
                &grad_q_raw_t,
                &grad_k_raw_t,
                &grad_v_t,
                &grad_conv_state_t,
                &grad_qkv_t,
                &grad_conv_state_in_t,
                N_K,
                N_V,
                HEAD_DIM,
            )
        })
        .unwrap();

        let grad_qkv = read_back_f32(&grad_qkv_t.buffer, conv_dim);
        let grad_conv_state_in = read_back_f32(&grad_conv_state_in_t.buffer, 3 * conv_dim);
        let grad_alpha = read_back_f32(&grad_alpha_t.buffer, N_V);
        let grad_beta_source = read_back_f32(&grad_beta_source_t.buffer, N_V);
        let grad_recurrence_state_in =
            read_back_f32(&grad_recurrence_state_in_t.buffer, state_elements);
        let grad_z = read_back_f32(&grad_z_t.buffer, v_elements);

        let as_f64 = |values: &[f32]| values.iter().copied().map(f64::from).collect::<Vec<_>>();
        let qkv64 = as_f64(&qkv_now);
        let conv_state64 = as_f64(&conv_state);
        let conv_weight64 = as_f64(&conv_weight);
        let alpha64 = as_f64(&alpha_source);
        let dt64 = as_f64(&dt_bias);
        let a_log64 = as_f64(&a_log);
        let beta_source64 = as_f64(&beta_source);
        let recurrence_state64 = as_f64(&recurrence_state);
        let z64 = as_f64(&z);
        let norm_weight64 = as_f64(&norm_weight);
        let grad_y64 = as_f64(&grad_y);
        let grad_recurrence_state64 = as_f64(&grad_recurrence_state);
        let grad_conv_state64 = as_f64(&grad_conv_state);
        let objective = |qkv: &[f64],
                         conv_state: &[f64],
                         alpha: &[f64],
                         beta_source: &[f64],
                         recurrence_state: &[f64],
                         z: &[f64]| {
            gdn_envelope_objective_f64(
                qkv,
                conv_state,
                &conv_weight64,
                alpha,
                &dt64,
                &a_log64,
                beta_source,
                recurrence_state,
                z,
                &norm_weight64,
                &grad_y64,
                &grad_recurrence_state64,
                &grad_conv_state64,
                N_V,
                N_K,
                HEAD_DIM,
                f64::from(L2_EPS),
                f64::from(RMS_EPS),
            )
        };
        let epsilon = 1e-5;
        let finite_difference = |values: &[f64], index: usize, evaluate: &dyn Fn(&[f64]) -> f64| {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon)
        };
        for &index in &[0usize, qk_elements, 2 * qk_elements, conv_dim - 1] {
            let fd = finite_difference(&qkv64, index, &|candidate| {
                objective(
                    candidate,
                    &conv_state64,
                    &alpha64,
                    &beta_source64,
                    &recurrence_state64,
                    &z64,
                )
            });
            assert!((fd - f64::from(grad_qkv[index])).abs() < 2e-4);
        }
        for &index in &[0usize, conv_dim, 2 * conv_dim + 17, 3 * conv_dim - 1] {
            let fd = finite_difference(&conv_state64, index, &|candidate| {
                objective(
                    &qkv64,
                    candidate,
                    &alpha64,
                    &beta_source64,
                    &recurrence_state64,
                    &z64,
                )
            });
            assert!((fd - f64::from(grad_conv_state_in[index])).abs() < 2e-4);
        }
        for index in 0..N_V {
            let fd = finite_difference(&alpha64, index, &|candidate| {
                objective(
                    &qkv64,
                    &conv_state64,
                    candidate,
                    &beta_source64,
                    &recurrence_state64,
                    &z64,
                )
            });
            assert!((fd - f64::from(grad_alpha[index])).abs() < 2e-4);
            let fd = finite_difference(&beta_source64, index, &|candidate| {
                objective(
                    &qkv64,
                    &conv_state64,
                    &alpha64,
                    candidate,
                    &recurrence_state64,
                    &z64,
                )
            });
            assert!((fd - f64::from(grad_beta_source[index])).abs() < 2e-4);
        }
        for &index in &[
            0usize,
            127,
            HEAD_DIM * HEAD_DIM + 31 * HEAD_DIM + 32,
            state_elements - 1,
        ] {
            let fd = finite_difference(&recurrence_state64, index, &|candidate| {
                objective(
                    &qkv64,
                    &conv_state64,
                    &alpha64,
                    &beta_source64,
                    candidate,
                    &z64,
                )
            });
            assert!((fd - f64::from(grad_recurrence_state_in[index])).abs() < 2e-4);
        }
        for &index in &[0usize, 127, v_elements - 1] {
            let fd = finite_difference(&z64, index, &|candidate| {
                objective(
                    &qkv64,
                    &conv_state64,
                    &alpha64,
                    &beta_source64,
                    &recurrence_state64,
                    candidate,
                )
            });
            assert!((fd - f64::from(grad_z[index])).abs() < 2e-4);
        }

        let direction = |len: usize, stride: usize| {
            (0..len)
                .map(|index| ((index * stride + 3) % 37) as f64 * 0.0007 - 0.012)
                .collect::<Vec<_>>()
        };
        let dqkv = direction(qkv64.len(), 5);
        let dconv = direction(conv_state64.len(), 7);
        let dalpha = direction(alpha64.len(), 11);
        let dbeta = direction(beta_source64.len(), 13);
        let dstate = direction(recurrence_state64.len(), 17);
        let dz = direction(z64.len(), 19);
        let inner = |gradient: &[f32], tangent: &[f64]| {
            gradient
                .iter()
                .zip(tangent)
                .map(|(gradient, tangent)| f64::from(*gradient) * tangent)
                .sum::<f64>()
        };
        let reverse_directional = inner(&grad_qkv, &dqkv)
            + inner(&grad_conv_state_in, &dconv)
            + inner(&grad_alpha, &dalpha)
            + inner(&grad_beta_source, &dbeta)
            + inner(&grad_recurrence_state_in, &dstate)
            + inner(&grad_z, &dz);
        let shift = |base: &[f64], tangent: &[f64], amount: f64| {
            base.iter()
                .zip(tangent)
                .map(|(base, tangent)| base + amount * tangent)
                .collect::<Vec<_>>()
        };
        let plus = objective(
            &shift(&qkv64, &dqkv, epsilon),
            &shift(&conv_state64, &dconv, epsilon),
            &shift(&alpha64, &dalpha, epsilon),
            &shift(&beta_source64, &dbeta, epsilon),
            &shift(&recurrence_state64, &dstate, epsilon),
            &shift(&z64, &dz, epsilon),
        );
        let minus = objective(
            &shift(&qkv64, &dqkv, -epsilon),
            &shift(&conv_state64, &dconv, -epsilon),
            &shift(&alpha64, &dalpha, -epsilon),
            &shift(&beta_source64, &dbeta, -epsilon),
            &shift(&recurrence_state64, &dstate, -epsilon),
            &shift(&z64, &dz, -epsilon),
        );
        let forward_directional = (plus - minus) / (2.0 * epsilon);
        assert!(
            (forward_directional - reverse_directional).abs() < 3e-4,
            "full GDN envelope adjoint mismatch forward={forward_directional} reverse={reverse_directional}"
        );
    }

    #[test]
    fn gdn_envelope_vjps_reject_unsafe_contracts() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let sigmoid = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let grad = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let invoke_sigmoid = |input: &MetalTensor, destination: &MetalTensor| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let result = encode_sigmoid_output_vjp_f32(&ctx, &encoder, input, &grad, destination);
            encoder.end();
            result
        };

        let mut overflow_shape = sigmoid.clone();
        overflow_shape.shape = vec![u64::MAX, 2];
        invoke_sigmoid(&overflow_shape, &output).expect_err("overflowing shape must fail");
        let f16 = MetalTensor::zeros_f16(&ctx, vec![4]).unwrap();
        invoke_sigmoid(&f16, &output).expect_err("non-F32 input must fail");
        invoke_sigmoid(&sigmoid, &sigmoid).expect_err("input/output alias must fail");
        let mut read_only = output.clone();
        read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        invoke_sigmoid(&sigmoid, &read_only).expect_err("read-only output must fail");
        let mut misaligned = output.clone();
        misaligned.offset = 2;
        invoke_sigmoid(&sigmoid, &misaligned).expect_err("misaligned output must fail");
        let mut out_of_range = output.clone();
        out_of_range.offset = 4;
        invoke_sigmoid(&sigmoid, &out_of_range).expect_err("short physical range must fail");

        let o = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let weight = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let z = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let grad_y = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let shared_output = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_rmsnorm_gated_vjp_f32(
            &ctx,
            &encoder,
            &o,
            &weight,
            &z,
            &grad_y,
            &shared_output,
            &shared_output,
            1,
            4,
            1e-6,
        )
        .expect_err("gradient outputs must not alias");
        encoder.end();
    }
}

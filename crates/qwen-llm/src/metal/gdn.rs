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

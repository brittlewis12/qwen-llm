//! RoPE kernels.

use super::*;

/// In-place partial RoPE with NEOX pairing. Rotates the first `n_rot`
/// dims of each head; leaves `[n_rot, head_dim)` untouched. For text-only
/// positions, IMROPE/MROPE collapse to this plain form — sections only
/// differ for vision/video.
///
/// `n_rot` should be `head_dim * partial_rotary_factor` (= 64 for
/// Qwen3.5/3.6 with head_dim=256, factor=0.25).
///
/// CPU oracle: `forward::rope_in_place`.
pub fn encode_rope_neox_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    buf: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if buf.n_elements() as usize != n_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox",
            detail: format!(
                "buf.n={} != n_heads*head_dim={}",
                buf.n_elements(),
                n_heads * head_dim
            ),
        });
    }
    if !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox",
            detail: format!("n_rot={n_rot} must be even and ≤ head_dim={head_dim}"),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
        n_rot: u32,
        position: u32,
        theta_base: f32,
    }
    let pso = ctx.pipeline("kernel_rope_neox_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            n_rot: n_rot as u32,
            position,
            theta_base,
        },
    );
    enc.set_tensor(1, buf);

    let total_pairs = n_heads * (n_rot / 2);
    let tg_threads = 64usize;
    let n_tg = total_pairs.div_ceil(tg_threads);
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

pub(crate) const ROPE_MINIMAX_MAX_VALIDATED_POSITION: u32 = 1_048_575;

pub(crate) fn validate_rope_minimax_position_span(
    start_position: u32,
    n_tokens: usize,
    kernel: &'static str,
) -> Result<(), MetalError> {
    if n_tokens == 0 {
        return Ok(());
    }
    let span = if start_position <= ROPE_MINIMAX_MAX_VALIDATED_POSITION {
        (ROPE_MINIMAX_MAX_VALIDATED_POSITION - start_position) as usize + 1
    } else {
        0
    };
    if n_tokens > span {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "benchmark-only minimax reduction is validated only through position {ROPE_MINIMAX_MAX_VALIDATED_POSITION}"
            ),
        });
    }
    Ok(())
}

/// In-place NEOX RoPE for Q and K using one dispatch. Both tensors share the
/// same position/frequency calculation; the shorter head set is handled by
/// the same threads that rotate the common prefix.
pub fn encode_rope_neox_pair_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    encode_rope_neox_pair_f32_impl(
        ctx,
        enc,
        q,
        k,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        position,
        theta_base,
        "kernel_rope_neox_pair_f32",
        false,
    )
}

/// Head-parallel Q/K RoPE using Metal's combined `sincos` intrinsic.
pub fn encode_rope_neox_pair_sincos_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    encode_rope_neox_pair_f32_impl(
        ctx,
        enc,
        q,
        k,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        position,
        theta_base,
        "kernel_rope_neox_pair_sincos_f32",
        false,
    )
}

/// Q/K RoPE with one lane per rotary pair, sharing coefficients across heads.
pub fn encode_rope_neox_pair_shared_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    encode_rope_neox_pair_f32_impl(
        ctx,
        enc,
        q,
        k,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        position,
        theta_base,
        "kernel_rope_neox_pair_shared_f32",
        true,
    )
}

/// Shared-head Q/K RoPE using the investigated degree-9/10 minimax pair.
pub fn encode_rope_neox_pair_shared_minimax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    validate_rope_minimax_position_span(position, 1, "rope_neox_pair_minimax")?;
    encode_rope_neox_pair_f32_impl(
        ctx,
        enc,
        q,
        k,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        position,
        theta_base,
        "kernel_rope_neox_pair_shared_minimax_f32",
        true,
    )
}

pub(crate) fn encode_rope_neox_pair_f32_impl(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    theta_base: f32,
    kernel_name: &'static str,
    shared_across_heads: bool,
) -> Result<(), MetalError> {
    if q.dtype != GgmlType::F32 || k.dtype != GgmlType::F32 || !q.is_writable() || !k.is_writable()
    {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: format!(
                "Q/K must be writable F32 tensors, got {:?}/{:?} writable={}/{}",
                q.dtype,
                k.dtype,
                q.is_writable(),
                k.is_writable(),
            ),
        });
    }
    if n_q_heads == 0 || n_k_heads == 0 || head_dim == 0 || n_rot == 0 {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "head counts, head_dim, and n_rot must be nonzero".into(),
        });
    }
    if !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: format!("n_rot={n_rot} must be even and <= head_dim={head_dim}"),
        });
    }
    if !theta_base.is_finite() || theta_base <= 1.0 {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "theta_base must be finite and greater than one".into(),
        });
    }
    if tensor_ranges_overlap(q, k) {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "writable Q and K tensors must not overlap".into(),
        });
    }
    let q_want = n_q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "q head shape overflow".into(),
        })?;
    let k_want = n_k_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "k head shape overflow".into(),
        })?;
    if q.n_elements() as usize != q_want || k.n_elements() as usize != k_want {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: format!(
                "q/k expected {q_want}/{k_want} elements, got {}/{}",
                q.n_elements(),
                k.n_elements()
            ),
        });
    }
    let q_bytes = q_want
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "Q byte size overflow".into(),
        })?;
    let k_bytes = k_want
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "K byte size overflow".into(),
        })?;
    let f32_alignment = std::mem::align_of::<f32>() as u64;
    if !tensor_physical_range_valid(q, q_bytes, f32_alignment)
        || !tensor_physical_range_valid(k, k_bytes, f32_alignment)
    {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "Q/K byte range is unaligned or outside its Metal buffer".into(),
        });
    }
    u32::try_from(q_want).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "Q index range exceeds u32".into(),
    })?;
    u32::try_from(k_want).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "K index range exceeds u32".into(),
    })?;
    let n_q_heads_u32 = u32::try_from(n_q_heads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "n_q_heads exceeds u32".into(),
    })?;
    let n_k_heads_u32 = u32::try_from(n_k_heads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "n_k_heads exceeds u32".into(),
    })?;
    let head_dim_u32 = u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "head_dim exceeds u32".into(),
    })?;
    let n_rot_u32 = u32::try_from(n_rot).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "n_rot exceeds u32".into(),
    })?;

    let heads_per_pair = if shared_across_heads {
        1
    } else {
        n_q_heads.max(n_k_heads)
    };
    let total_pairs =
        heads_per_pair
            .checked_mul(n_rot / 2)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "rope_neox_pair",
                detail: "dispatch width overflow".into(),
            })?;
    let tg_threads = if shared_across_heads { 32 } else { 64 };
    let n_tg = total_pairs.div_ceil(tg_threads);
    let dispatched_threads = n_tg
        .checked_mul(tg_threads)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair",
            detail: "dispatch grid overflow".into(),
        })?;
    u32::try_from(dispatched_threads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair",
        detail: "dispatch grid exceeds u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_k_heads: u32,
        head_dim: u32,
        n_rot: u32,
        position: u32,
        theta_base: f32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads_u32,
            n_k_heads: n_k_heads_u32,
            head_dim: head_dim_u32,
            n_rot: n_rot_u32,
            position,
            theta_base,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);

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

pub fn encode_rope_neox_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    buf: &MetalTensor,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if buf.dtype != GgmlType::F32 || !buf.is_writable() {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: format!(
                "buffer must be writable F32, got {:?} writable={}",
                buf.dtype,
                buf.is_writable(),
            ),
        });
    }
    if n_tokens == 0 || n_heads == 0 || head_dim == 0 || n_rot == 0 {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "token, head, head_dim, and n_rot counts must be nonzero".into(),
        });
    }
    if !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: format!("n_rot={n_rot} must be even and <= head_dim={head_dim}"),
        });
    }
    if !theta_base.is_finite() || theta_base <= 1.0 {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "theta_base must be finite and greater than one".into(),
        });
    }
    let want = n_tokens
        .checked_mul(n_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "packed tensor shape overflow".into(),
        })?;
    if buf.n_elements() as usize != want {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: format!(
                "buf.n={} != n_tokens*n_heads*head_dim={}",
                buf.n_elements(),
                want
            ),
        });
    }
    let want_bytes = want
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "buffer byte size overflow".into(),
        })?;
    if !tensor_physical_range_valid(buf, want_bytes, std::mem::align_of::<f32>() as u64) {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "buffer byte range is unaligned or outside its Metal buffer".into(),
        });
    }
    u32::try_from(want).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "buffer index range exceeds u32".into(),
    })?;
    let n_tokens_u32 = u32::try_from(n_tokens).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "n_tokens exceeds u32".into(),
    })?;
    let n_heads_u32 = u32::try_from(n_heads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "n_heads exceeds u32".into(),
    })?;
    let head_dim_u32 = u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "head_dim exceeds u32".into(),
    })?;
    let n_rot_u32 = u32::try_from(n_rot).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "n_rot exceeds u32".into(),
    })?;
    let last_token = u32::try_from(n_tokens - 1).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "n_tokens exceeds u32".into(),
    })?;
    start_position
        .checked_add(last_token)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "position span exceeds u32".into(),
        })?;
    let total_pairs = n_tokens
        .checked_mul(n_heads)
        .and_then(|count| count.checked_mul(n_rot / 2))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "dispatch width overflow".into(),
        })?;
    let tg_threads = 64usize;
    let n_tg = total_pairs.div_ceil(tg_threads);
    let dispatched_threads = n_tg
        .checked_mul(tg_threads)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_packed_consecutive",
            detail: "dispatch grid overflow".into(),
        })?;
    u32::try_from(dispatched_threads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_packed_consecutive",
        detail: "dispatch grid exceeds u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
        n_rot: u32,
        start_position: u32,
        theta_base: f32,
    }
    let pso = ctx.pipeline("kernel_rope_neox_f32_packed_consecutive")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens_u32,
            n_heads: n_heads_u32,
            head_dim: head_dim_u32,
            n_rot: n_rot_u32,
            start_position,
            theta_base,
        },
    );
    enc.set_tensor(1, buf);

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

/// Packed consecutive-position Q/K RoPE in one head-parallel dispatch.
pub fn encode_rope_neox_pair_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    encode_rope_neox_pair_f32_packed_consecutive_impl(
        ctx,
        enc,
        q,
        k,
        n_tokens,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        start_position,
        theta_base,
        "kernel_rope_neox_pair_f32_packed_consecutive",
        false,
    )
}

/// Packed Q/K RoPE with one lane per token and rotary pair.
pub fn encode_rope_neox_pair_shared_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    encode_rope_neox_pair_f32_packed_consecutive_impl(
        ctx,
        enc,
        q,
        k,
        n_tokens,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        start_position,
        theta_base,
        "kernel_rope_neox_pair_shared_f32_packed_consecutive",
        true,
    )
}

/// Selects the measured packed Q/K geometry: head-parallel for short batches,
/// then coefficient sharing once the token axis supplies enough parallelism.
pub fn encode_rope_neox_pair_adaptive_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    if n_tokens >= 128 {
        encode_rope_neox_pair_shared_f32_packed_consecutive(
            ctx,
            enc,
            q,
            k,
            n_tokens,
            n_q_heads,
            n_k_heads,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        )
    } else {
        encode_rope_neox_pair_f32_packed_consecutive(
            ctx,
            enc,
            q,
            k,
            n_tokens,
            n_q_heads,
            n_k_heads,
            head_dim,
            n_rot,
            start_position,
            theta_base,
        )
    }
}

/// Packed shared-head Q/K RoPE using the degree-9/10 minimax pair.
pub fn encode_rope_neox_pair_shared_minimax_f32_packed_consecutive(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
) -> Result<(), MetalError> {
    validate_rope_minimax_position_span(start_position, n_tokens, "rope_neox_pair_packed_minimax")?;
    encode_rope_neox_pair_f32_packed_consecutive_impl(
        ctx,
        enc,
        q,
        k,
        n_tokens,
        n_q_heads,
        n_k_heads,
        head_dim,
        n_rot,
        start_position,
        theta_base,
        "kernel_rope_neox_pair_shared_minimax_f32_packed_consecutive",
        true,
    )
}

pub(crate) fn encode_rope_neox_pair_f32_packed_consecutive_impl(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    n_tokens: usize,
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    n_rot: usize,
    start_position: u32,
    theta_base: f32,
    kernel_name: &'static str,
    shared_across_heads: bool,
) -> Result<(), MetalError> {
    if q.dtype != GgmlType::F32 || k.dtype != GgmlType::F32 || !q.is_writable() || !k.is_writable()
    {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: format!(
                "Q/K must be writable F32 tensors, got {:?}/{:?} writable={}/{}",
                q.dtype,
                k.dtype,
                q.is_writable(),
                k.is_writable(),
            ),
        });
    }
    if n_tokens == 0 || n_q_heads == 0 || n_k_heads == 0 || head_dim == 0 || n_rot == 0 {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "token, head, head_dim, and n_rot counts must be nonzero".into(),
        });
    }
    if !n_rot.is_multiple_of(2) || n_rot > head_dim {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: format!("n_rot={n_rot} must be even and <= head_dim={head_dim}"),
        });
    }
    if !theta_base.is_finite() || theta_base <= 1.0 {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "theta_base must be finite and greater than one".into(),
        });
    }
    if tensor_ranges_overlap(q, k) {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "writable Q and K tensors must not overlap".into(),
        });
    }
    let q_want = n_tokens
        .checked_mul(n_q_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "packed Q shape overflow".into(),
        })?;
    let k_want = n_tokens
        .checked_mul(n_k_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "packed K shape overflow".into(),
        })?;
    if q.n_elements() as usize != q_want || k.n_elements() as usize != k_want {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: format!(
                "nonempty q/k expected {q_want}/{k_want} elements, got {}/{}",
                q.n_elements(),
                k.n_elements()
            ),
        });
    }
    let q_bytes = q_want
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "Q byte size overflow".into(),
        })?;
    let k_bytes = k_want
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "K byte size overflow".into(),
        })?;
    let f32_alignment = std::mem::align_of::<f32>() as u64;
    if !tensor_physical_range_valid(q, q_bytes, f32_alignment)
        || !tensor_physical_range_valid(k, k_bytes, f32_alignment)
    {
        return Err(MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "Q/K byte range is unaligned or outside its Metal buffer".into(),
        });
    }
    u32::try_from(q_want).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "Q index range exceeds u32".into(),
    })?;
    u32::try_from(k_want).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "K index range exceeds u32".into(),
    })?;
    let n_tokens_u32 = u32::try_from(n_tokens).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "n_tokens exceeds u32".into(),
    })?;
    let n_q_heads_u32 = u32::try_from(n_q_heads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "n_q_heads exceeds u32".into(),
    })?;
    let n_k_heads_u32 = u32::try_from(n_k_heads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "n_k_heads exceeds u32".into(),
    })?;
    let head_dim_u32 = u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "head_dim exceeds u32".into(),
    })?;
    let n_rot_u32 = u32::try_from(n_rot).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "n_rot exceeds u32".into(),
    })?;
    let last_token = u32::try_from(n_tokens - 1).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "n_tokens exceeds u32".into(),
    })?;
    start_position
        .checked_add(last_token)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "position span exceeds u32".into(),
        })?;

    let heads_per_pair = if shared_across_heads {
        1
    } else {
        n_q_heads.max(n_k_heads)
    };
    let total_pairs = n_tokens
        .checked_mul(heads_per_pair)
        .and_then(|count| count.checked_mul(n_rot / 2))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "dispatch width overflow".into(),
        })?;
    let tg_threads = if shared_across_heads { 32 } else { 64 };
    let n_tg = total_pairs.div_ceil(tg_threads);
    let dispatched_threads = n_tg
        .checked_mul(tg_threads)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "rope_neox_pair_packed_consecutive",
            detail: "dispatch grid overflow".into(),
        })?;
    u32::try_from(dispatched_threads).map_err(|_| MetalError::BadShape {
        kernel: "rope_neox_pair_packed_consecutive",
        detail: "dispatch grid exceeds u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        n_q_heads: u32,
        n_k_heads: u32,
        head_dim: u32,
        n_rot: u32,
        start_position: u32,
        theta_base: f32,
    }
    let pso = ctx.pipeline(kernel_name)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_tokens: n_tokens_u32,
            n_q_heads: n_q_heads_u32,
            n_k_heads: n_k_heads_u32,
            head_dim: head_dim_u32,
            n_rot: n_rot_u32,
            start_position,
            theta_base,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn rope_neox_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        // Real shapes from Qwen3.5 family:
        //   0.8B: 8 Q heads, 256 head_dim, 64 rotated dims
        //   27B:  24 Q heads or 4 KV heads, 256 head_dim, 64 rotated dims
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        for &n_heads in &[8usize, 24, 4] {
            for &position in &[0u32, 1, 7, 100] {
                let total = n_heads * head_dim;
                let buf_init: Vec<f32> =
                    (0..total).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();

                let mut buf_cpu = buf_init.clone();
                rope_neox_cpu_ref(&mut buf_cpu, n_heads, head_dim, n_rot, position, theta_base);

                let buf_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&buf_init),
                    vec![total as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_rope_neox_f32(
                        &ctx, enc, &buf_t, n_heads, head_dim, n_rot, position, theta_base,
                    )
                })
                .unwrap();
                let gpu = read_back_f32(&buf_t.buffer, total);

                let max_abs = gpu
                    .iter()
                    .zip(buf_cpu.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_abs < 1e-5,
                    "rope_neox n_heads={n_heads} pos={position}: max|Δ|={max_abs}"
                );
            }
        }
    }

    #[test]
    fn rope_neox_pair_matches_cpu() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        for &(n_q, n_k, position) in &[(24usize, 4usize, 0u32), (8, 8, 37)] {
            let q_len = n_q * head_dim;
            let k_len = n_k * head_dim;
            let q_init: Vec<f32> = (0..q_len).map(|i| ((i % 19) as f32 - 9.0) * 0.05).collect();
            let k_init: Vec<f32> = (0..k_len)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
                .collect();
            let mut q_cpu = q_init.clone();
            let mut k_cpu = k_init.clone();
            rope_neox_cpu_ref(&mut q_cpu, n_q, head_dim, n_rot, position, theta_base);
            rope_neox_cpu_ref(&mut k_cpu, n_k, head_dim, n_rot, position, theta_base);

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q_init),
                vec![q_len as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&k_init),
                vec![k_len as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_rope_neox_pair_f32(
                    &ctx, enc, &q_t, &k_t, n_q, n_k, head_dim, n_rot, position, theta_base,
                )
            })
            .unwrap();

            let q_gpu = read_back_f32(&q_t.buffer, q_len);
            let k_gpu = read_back_f32(&k_t.buffer, k_len);
            let q_max = q_gpu
                .iter()
                .zip(q_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let k_max = k_gpu
                .iter()
                .zip(k_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(q_max < 1e-5, "rope pair Q drift: {q_max}");
            assert!(k_max < 1e-5, "rope pair K drift: {k_max}");
        }
    }

    #[test]
    fn rope_neox_pair_optimized_variants_match_baseline() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let n_q = 24;
        let n_k = 4;
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        let q_init: Vec<f32> = (0..n_q * head_dim)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
            .collect();
        let k_init: Vec<f32> = (0..n_k * head_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.046875)
            .collect();

        for position in [0, 1, 127, 65_535, 65_536, 262_143, 1_048_575] {
            let baseline = run_rope_pair_variant(
                &ctx, "baseline", &q_init, &k_init, n_q, n_k, head_dim, n_rot, position, theta_base,
            );
            assert_finite(&baseline.0, "baseline Q");
            assert_finite(&baseline.1, "baseline K");
            for variant in ["sincos", "shared"] {
                let candidate = run_rope_pair_variant(
                    &ctx, variant, &q_init, &k_init, n_q, n_k, head_dim, n_rot, position,
                    theta_base,
                );
                assert_finite(&candidate.0, &format!("RoPE {variant} Q at {position}"));
                assert_finite(&candidate.1, &format!("RoPE {variant} K at {position}"));
                assert_bitwise_equal(
                    &baseline.0,
                    &candidate.0,
                    &format!("RoPE {variant} Q at {position}"),
                );
                assert_bitwise_equal(
                    &baseline.1,
                    &candidate.1,
                    &format!("RoPE {variant} K at {position}"),
                );
            }

            let minimax = run_rope_pair_variant(
                &ctx, "minimax", &q_init, &k_init, n_q, n_k, head_dim, n_rot, position, theta_base,
            );
            assert_finite(&minimax.0, &format!("RoPE minimax Q at {position}"));
            assert_finite(&minimax.1, &format!("RoPE minimax K at {position}"));
            let q_max = max_abs_diff(&baseline.0, &minimax.0);
            let k_max = max_abs_diff(&baseline.1, &minimax.1);
            assert!(
                q_max <= 5.0e-3 && k_max <= 5.0e-3,
                "RoPE minimax drift at position {position}: q={q_max} k={k_max} tolerance=0.005"
            );
        }
    }

    #[test]
    fn rope_optimized_wrappers_reject_zero_rotary_and_overlap() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let (n_tokens, n_heads, head_dim, n_rot) = (1usize, 2usize, 8usize, 4usize);
        let theta = 10_000_000.0f32;
        let same = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let other = MetalTensor::zeros_f32(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        let cmd = ctx
            .queue
            .commandBuffer()
            .expect("validation command buffer");
        let enc = KernelEncoder::begin(&cmd);

        assert!(matches!(
            encode_rope_neox_pair_f32(
                &ctx, &enc, &same, &other, n_heads, n_heads, head_dim, 0, 0, theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        assert!(matches!(
            encode_rope_neox_pair_f32(
                &ctx, &enc, &same, &same, n_heads, n_heads, head_dim, n_rot, 0, theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        let wrong_dtype = MetalTensor::zeros_f16(&ctx, vec![(n_heads * head_dim) as u64]).unwrap();
        assert!(matches!(
            encode_rope_neox_pair_f32(
                &ctx,
                &enc,
                &same,
                &wrong_dtype,
                n_heads,
                n_heads,
                head_dim,
                n_rot,
                0,
                theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        let mut read_only = other.clone();
        read_only.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(matches!(
            encode_rope_neox_pair_f32(
                &ctx, &enc, &same, &read_only, n_heads, n_heads, head_dim, n_rot, 0, theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        assert!(matches!(
            encode_rope_neox_pair_shared_minimax_f32(
                &ctx,
                &enc,
                &same,
                &other,
                n_heads,
                n_heads,
                head_dim,
                n_rot,
                ROPE_MINIMAX_MAX_VALIDATED_POSITION + 1,
                theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        assert!(matches!(
            encode_rope_neox_pair_f32_packed_consecutive(
                &ctx, &enc, &same, &same, n_tokens, n_heads, n_heads, head_dim, n_rot, 0, theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        let packed_q =
            MetalTensor::zeros_f32(&ctx, vec![(2 * n_tokens * n_heads * head_dim) as u64]).unwrap();
        let packed_k =
            MetalTensor::zeros_f32(&ctx, vec![(2 * n_tokens * n_heads * head_dim) as u64]).unwrap();
        assert!(matches!(
            encode_rope_neox_pair_shared_minimax_f32_packed_consecutive(
                &ctx,
                &enc,
                &packed_q,
                &packed_k,
                2,
                n_heads,
                n_heads,
                head_dim,
                n_rot,
                ROPE_MINIMAX_MAX_VALIDATED_POSITION,
                theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        assert!(matches!(
            encode_rope_neox_f32_packed_consecutive(
                &ctx, &enc, &same, n_tokens, n_heads, head_dim, 0, 0, theta,
            ),
            Err(MetalError::BadShape { .. })
        ));

        let n_q = 2usize;
        let n_k = 1usize;
        let q_source =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_q * 2 * head_dim) as u64]).unwrap();
        let q_output_overlap = q_source.view_subrange(0, vec![(n_tokens * n_q * head_dim) as u64]);
        let k_source =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_k * head_dim) as u64]).unwrap();
        let k_output =
            MetalTensor::zeros_f32(&ctx, vec![(n_tokens * n_k * head_dim) as u64]).unwrap();
        let q_weight = MetalTensor::zeros_f32(&ctx, vec![head_dim as u64]).unwrap();
        let k_weight = MetalTensor::zeros_f32(&ctx, vec![head_dim as u64]).unwrap();
        assert!(matches!(
            encode_qk_rms_norm_rope_f32_packed_consecutive(
                &ctx,
                &enc,
                &q_source,
                &q_weight,
                &q_output_overlap,
                &k_source,
                &k_weight,
                &k_output,
                n_tokens,
                n_q,
                n_k,
                head_dim,
                n_rot,
                0,
                1e-6,
                theta,
            ),
            Err(MetalError::BadShape { .. })
        ));
        let q_output_f16 =
            MetalTensor::zeros_f16(&ctx, vec![(n_tokens * n_q * head_dim) as u64]).unwrap();
        assert!(matches!(
            encode_qk_rms_norm_rope_f32_packed_consecutive(
                &ctx,
                &enc,
                &q_source,
                &q_weight,
                &q_output_f16,
                &k_source,
                &k_weight,
                &k_output,
                n_tokens,
                n_q,
                n_k,
                head_dim,
                n_rot,
                0,
                1e-6,
                theta,
            ),
            Err(MetalError::BadShape { .. })
        ));

        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).expect("command buffer completed");
        assert!(cmd.error().is_none(), "command failed: {:?}", cmd.error());
    }

    #[test]
    fn rope_neox_packed_consecutive_matches_cpu() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        for &(n_tokens, n_heads, start_position) in &[(5usize, 4usize, 0u32), (3, 24, 17)] {
            let total = n_tokens * n_heads * head_dim;
            let buf_init: Vec<f32> = (0..total)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect();

            let mut buf_cpu = buf_init.clone();
            for tok in 0..n_tokens {
                let start = tok * n_heads * head_dim;
                let end = start + n_heads * head_dim;
                rope_neox_cpu_ref(
                    &mut buf_cpu[start..end],
                    n_heads,
                    head_dim,
                    n_rot,
                    start_position + tok as u32,
                    theta_base,
                );
            }

            let buf_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&buf_init),
                vec![total as u64],
                GgmlType::F32,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_rope_neox_f32_packed_consecutive(
                    &ctx,
                    enc,
                    &buf_t,
                    n_tokens,
                    n_heads,
                    head_dim,
                    n_rot,
                    start_position,
                    theta_base,
                )
            })
            .unwrap();
            let gpu = read_back_f32(&buf_t.buffer, total);

            let max_abs = gpu
                .iter()
                .zip(buf_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 1e-5,
                "rope_neox_packed n_tokens={n_tokens} n_heads={n_heads} start={start_position}: max|Δ|={max_abs}"
            );
        }
    }

    #[test]
    fn rope_neox_packed_pair_variants_match_baseline() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let n_q = 24;
        let n_k = 4;
        let head_dim = 256;
        let n_rot = 64;
        let theta_base = 10_000_000.0f32;
        for &(n_tokens, start_position) in
            &[(5usize, 0u32), (8, 65_531), (128, 65_536), (8, 1_048_568)]
        {
            let q_init: Vec<f32> = (0..n_tokens * n_q * head_dim)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.03125)
                .collect();
            let k_init: Vec<f32> = (0..n_tokens * n_k * head_dim)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.046875)
                .collect();
            let baseline = run_rope_packed_pair_variant(
                &ctx,
                "baseline",
                &q_init,
                &k_init,
                n_tokens,
                n_q,
                n_k,
                head_dim,
                n_rot,
                start_position,
                theta_base,
            );
            assert_finite(&baseline.0, "packed baseline Q");
            assert_finite(&baseline.1, "packed baseline K");
            for variant in ["paired", "shared", "adaptive"] {
                let candidate = run_rope_packed_pair_variant(
                    &ctx,
                    variant,
                    &q_init,
                    &k_init,
                    n_tokens,
                    n_q,
                    n_k,
                    head_dim,
                    n_rot,
                    start_position,
                    theta_base,
                );
                assert_finite(
                    &candidate.0,
                    &format!("packed RoPE {variant} Q at {start_position}"),
                );
                assert_finite(
                    &candidate.1,
                    &format!("packed RoPE {variant} K at {start_position}"),
                );
                assert_bitwise_equal(
                    &baseline.0,
                    &candidate.0,
                    &format!("packed RoPE {variant} Q at {start_position}"),
                );
                assert_bitwise_equal(
                    &baseline.1,
                    &candidate.1,
                    &format!("packed RoPE {variant} K at {start_position}"),
                );
            }

            let minimax = run_rope_packed_pair_variant(
                &ctx,
                "minimax",
                &q_init,
                &k_init,
                n_tokens,
                n_q,
                n_k,
                head_dim,
                n_rot,
                start_position,
                theta_base,
            );
            assert_finite(
                &minimax.0,
                &format!("packed RoPE minimax Q at {start_position}"),
            );
            assert_finite(
                &minimax.1,
                &format!("packed RoPE minimax K at {start_position}"),
            );
            let q_max = max_abs_diff(&baseline.0, &minimax.0);
            let k_max = max_abs_diff(&baseline.1, &minimax.1);
            assert!(
                q_max <= 5.0e-3 && k_max <= 5.0e-3,
                "packed RoPE minimax drift at start {start_position}: q={q_max} k={k_max} tolerance=0.005"
            );
        }
    }
}

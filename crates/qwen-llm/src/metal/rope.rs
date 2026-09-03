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

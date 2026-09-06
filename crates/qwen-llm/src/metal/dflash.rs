//! DFlash drafter attention and DFlash2 kernels.

use super::*;

/// DFlash 2 two-tap dynamic depthwise convolution over the noise block
/// (kernels/dflash2.metal). `y[t][c] = Σ_tap (base[side][tap][c] +
/// dyn[t][side][tap][group(c)]) · x[t-tap][c]`, zero-padded before the
/// block start. `y` must not alias `x` (taps read neighboring rows).
///
/// Layouts (all F32):
/// * `x`, `y`: `[n_tokens, h]` row-major
/// * `dyn`:    `[n_tokens, 2 · kernel_size · n_groups]` — per-token conv
///   projection output, `(group, tap, side)` fastest-to-slowest
/// * `base`:   `[h, kernel_size, 2]` GGUF tensor — `(channel, tap, side)`
#[allow(clippy::too_many_arguments)]
pub fn encode_dflash2_conv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    x: &MetalTensor,
    dynamic: &MetalTensor,
    base: &MetalTensor,
    y: &MetalTensor,
    n_tokens: usize,
    h: usize,
    kernel_size: usize,
    group_size: usize,
    side: u32,
) -> Result<(), MetalError> {
    if side > 1 || group_size == 0 || kernel_size == 0 || !h.is_multiple_of(group_size) {
        return Err(MetalError::BadShape {
            kernel: "dflash2_conv",
            detail: format!("bad params side={side} kernel={kernel_size} group={group_size} h={h}"),
        });
    }
    let n_groups = h / group_size;
    let dyn_stride = 2 * kernel_size * n_groups;
    if x.n_elements() as usize != n_tokens * h || y.n_elements() as usize != n_tokens * h {
        return Err(MetalError::BadShape {
            kernel: "dflash2_conv",
            detail: format!(
                "x/y elements {}/{} != n_tokens*h={}",
                x.n_elements(),
                y.n_elements(),
                n_tokens * h
            ),
        });
    }
    if (dynamic.n_elements() as usize) < n_tokens * dyn_stride {
        return Err(MetalError::BadShape {
            kernel: "dflash2_conv",
            detail: format!(
                "dyn elements {} < n_tokens*dyn_stride={}",
                dynamic.n_elements(),
                n_tokens * dyn_stride
            ),
        });
    }
    if (base.n_elements() as usize) < h * kernel_size * 2 {
        return Err(MetalError::BadShape {
            kernel: "dflash2_conv",
            detail: format!(
                "base elements {} < h*kernel*2={}",
                base.n_elements(),
                h * kernel_size * 2
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        h: u32,
        n_tokens: u32,
        kernel_size: u32,
        group_size: u32,
        n_groups: u32,
        side: u32,
        dyn_stride: u32,
    }
    let pso = ctx.pipeline("kernel_dflash2_conv_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            h: h as u32,
            n_tokens: n_tokens as u32,
            kernel_size: kernel_size as u32,
            group_size: group_size as u32,
            n_groups: n_groups as u32,
            side,
            dyn_stride: dyn_stride as u32,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, dynamic);
    enc.set_tensor(3, base);
    enc.set_tensor(4, y);

    let total = n_tokens * h;
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: total.div_ceil(tg_threads),
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

crate::env_flag!(default_off dflash_noncausal_noise_enabled, "QWEN_DFLASH_NONCAUSAL_NOISE");

/// **DFlash drafter attention** (v0.72.1) — fused small-N attention
/// with per-layer SWA mask. Replaces the CPU phase-3 attention in
/// `draft_block`. See `kernels/dflash_attn.metal` for design.
///
/// Threadgroup grid: `(n_q_heads, N)` per drafter layer per outer step.
/// Threads per TG: 32 (one simdgroup; head_dim/32=4 dims per lane).
pub fn encode_dflash_attn_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k: &MetalTensor,
    v: &MetalTensor,
    pos_k: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_kv_total: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> Result<(), MetalError> {
    if !head_dim.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn",
            detail: format!("head_dim={head_dim} not divisible by 32"),
        });
    }
    // Codex code-review v0.72.2: kernel uses `q_reg[8]` / `o_acc[8]`
    // sized for head_dim ≤ 256 (8 × 32 lanes = 256 dims). Reject
    // larger head_dim explicitly so future model variants don't
    // silently stack-OOB inside the kernel.
    if head_dim > 256 {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn",
            detail: format!(
                "head_dim={head_dim} > 256: kernel registers q_reg/o_acc are sized for head_dim ≤ 256"
            ),
        });
    }
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn",
            detail: format!("n_q_heads={n_q_heads} not divisible by n_kv_heads={n_kv_heads}"),
        });
    }
    if q.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.q",
            detail: format!(
                "q.n_elements={} != N*n_q*head_dim={}",
                q.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }
    let kv_stride = n_kv_heads * head_dim;
    if k.n_elements() as usize != n_kv_total * kv_stride
        || v.n_elements() as usize != n_kv_total * kv_stride
    {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.kv",
            detail: format!(
                "k/v expected n_kv_total*kv_stride = {}*{} = {} elements",
                n_kv_total,
                kv_stride,
                n_kv_total * kv_stride
            ),
        });
    }
    if pos_k.n_elements() as usize != n_kv_total {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.pos_k",
            detail: format!(
                "pos_k.n_elements={} != n_kv_total={n_kv_total}",
                pos_k.n_elements()
            ),
        });
    }
    if o.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "dflash_attn.o",
            detail: format!(
                "o.n_elements={} != N*n_q*head_dim={}",
                o.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }

    let pso = ctx.pipeline("kernel_dflash_attn_f32")?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32, // codex v0.72.2: real q_idx bound. Was previously
        // a bogus `n_q_heads * 16` placeholder; dispatch-shape bug
        // would silently OOB without this.
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
        noncausal_noise: u32,
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_kv_total: n_kv_total as u32,
            ctx_len: ctx_len as u32,
            n_rows: n as u32,
            noise_start_pos,
            swa_window,
            ctx_scan_start: 0,
            scale,
            noncausal_noise: dflash_noncausal_noise_enabled() as u32,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k);
    enc.set_tensor(3, v);
    enc.set_tensor(4, pos_k);
    enc.set_tensor(5, o);

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: n,
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

/// DFlash drafter attention over two K/V ranges: committed context cache
/// plus the current noise block. Same math as `encode_dflash_attn_f32`, but
/// skips the per-layer materialization of `k_full/v_full = ctx || noise`.
pub fn encode_dflash_attn_two_range_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
) -> Result<(), MetalError> {
    encode_dflash_attn_two_range_pipeline(
        ctx,
        enc,
        q,
        k_ctx,
        v_ctx,
        k_noise,
        v_noise,
        pos_ctx,
        o,
        n,
        n_q_heads,
        n_kv_heads,
        head_dim,
        ctx_len,
        noise_start_pos,
        swa_window,
        0,
        "dflash_attn_two_range",
        "kernel_dflash_attn_two_range_f32",
    )
}

/// Online-softmax DFlash attention with an optional SWA context scan start.
/// `ctx_scan_start` is ignored for full-attention layers (`swa_window == 0`).
pub fn encode_dflash_attn_online_two_range_scan_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    ctx_scan_start: usize,
) -> Result<(), MetalError> {
    encode_dflash_attn_two_range_pipeline(
        ctx,
        enc,
        q,
        k_ctx,
        v_ctx,
        k_noise,
        v_noise,
        pos_ctx,
        o,
        n,
        n_q_heads,
        n_kv_heads,
        head_dim,
        ctx_len,
        noise_start_pos,
        swa_window,
        ctx_scan_start,
        "dflash_attn_online_two_range",
        "kernel_dflash_attn_online_two_range_f32",
    )
}

pub(crate) fn encode_dflash_attn_two_range_pipeline(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    ctx_scan_start: usize,
    kernel_label: &'static str,
    pipeline_name: &'static str,
) -> Result<(), MetalError> {
    if !head_dim.is_multiple_of(32) {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!("head_dim={head_dim} not divisible by 32"),
        });
    }
    if head_dim > 256 {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "head_dim={head_dim} > 256: kernel registers q_reg/o_acc are sized for head_dim <= 256"
            ),
        });
    }
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!("n_q_heads={n_q_heads} not divisible by n_kv_heads={n_kv_heads}"),
        });
    }
    if q.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "q.n_elements={} != N*n_q*head_dim={}",
                q.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }
    let kv_stride = n_kv_heads * head_dim;
    let ctx_elems = ctx_len * kv_stride;
    if (k_ctx.n_elements() as usize) < ctx_elems || (v_ctx.n_elements() as usize) < ctx_elems {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "ctx k/v need at least ctx_len*kv_stride = {ctx_len}*{kv_stride} = {ctx_elems} elements"
            ),
        });
    }
    let noise_elems = n * kv_stride;
    if k_noise.n_elements() as usize != noise_elems || v_noise.n_elements() as usize != noise_elems
    {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "noise k/v expected N*kv_stride = {n}*{kv_stride} = {noise_elems} elements"
            ),
        });
    }
    if (pos_ctx.n_elements() as usize) < ctx_len {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "pos_ctx.n_elements={} < ctx_len={ctx_len}",
                pos_ctx.n_elements()
            ),
        });
    }
    if ctx_scan_start > ctx_len {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!("ctx_scan_start={ctx_scan_start} > ctx_len={ctx_len}"),
        });
    }
    if o.n_elements() as usize != n * n_q_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: kernel_label,
            detail: format!(
                "o.n_elements={} != N*n_q*head_dim={}",
                o.n_elements(),
                n * n_q_heads * head_dim
            ),
        });
    }

    let pso = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32,
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
        noncausal_noise: u32,
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_kv_total: (ctx_len + n) as u32,
            ctx_len: ctx_len as u32,
            n_rows: n as u32,
            noise_start_pos,
            swa_window,
            ctx_scan_start: ctx_scan_start as u32,
            scale,
            noncausal_noise: dflash_noncausal_noise_enabled() as u32,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_ctx);
    enc.set_tensor(3, v_ctx);
    enc.set_tensor(4, k_noise);
    enc.set_tensor(5, v_noise);
    enc.set_tensor(6, pos_ctx);
    enc.set_tensor(7, o);

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: n,
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

/// Fixed-shape GQA-sharing split-4 path for the Qwen3.6 DFlash full-attention
/// layer. The serial main/reduce pair writes exact online-softmax partials.
pub fn encode_dflash_attn_full_gqa_split4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ctx_len: usize,
) -> Result<(), MetalError> {
    const N: usize = 16;
    const N_Q: usize = 32;
    const N_KV: usize = 8;
    const HEAD_DIM: usize = 128;
    const GROUP: usize = 4;
    const SPLIT: usize = 4;
    const KERNEL: &str = "dflash_attn_full_gqa_split4";

    if enc.concurrent {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "main/reduce dependency requires a serial encoder".into(),
        });
    }
    if (n, n_q_heads, n_kv_heads, head_dim) != (N, N_Q, N_KV, HEAD_DIM) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "expected N/n_q/n_kv/head_dim={N}/{N_Q}/{N_KV}/{HEAD_DIM}, got \
                 {n}/{n_q_heads}/{n_kv_heads}/{head_dim}"
            ),
        });
    }
    let tensors = [q, k_ctx, v_ctx, k_noise, v_noise, o_partial, ml_partial, o];
    if tensors.iter().any(|tensor| tensor.dtype != GgmlType::F32) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "all inputs, scratch, and output must be F32".into(),
        });
    }
    let q_elems = N * N_Q * HEAD_DIM;
    let kv_stride = N_KV * HEAD_DIM;
    let ctx_elems = ctx_len
        .checked_mul(kv_stride)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "context element count overflow".into(),
        })?;
    let noise_elems = N * kv_stride;
    let o_partial_elems = N * N_KV * SPLIT * GROUP * HEAD_DIM;
    let ml_partial_elems = N * N_KV * SPLIT * GROUP * 2;
    if q.n_elements() as usize != q_elems
        || o.n_elements() as usize != q_elems
        || (k_ctx.n_elements() as usize) < ctx_elems
        || (v_ctx.n_elements() as usize) < ctx_elems
        || k_noise.n_elements() as usize != noise_elems
        || v_noise.n_elements() as usize != noise_elems
        || (o_partial.n_elements() as usize) < o_partial_elems
        || (ml_partial.n_elements() as usize) < ml_partial_elems
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "shape mismatch: q={} o={} k_ctx={} v_ctx={} k_noise={} v_noise={} \
                 o_partial={} ml_partial={} ctx_len={ctx_len}",
                q.n_elements(),
                o.n_elements(),
                k_ctx.n_elements(),
                v_ctx.n_elements(),
                k_noise.n_elements(),
                v_noise.n_elements(),
                o_partial.n_elements(),
                ml_partial.n_elements(),
            ),
        });
    }
    let total_rows = ctx_len.checked_add(N).ok_or_else(|| MetalError::BadShape {
        kernel: KERNEL,
        detail: "ctx_len + N overflow".into(),
    })?;
    let n_kv_total = u32::try_from(total_rows).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: "ctx_len + N does not fit u32".into(),
    })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32,
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
        noncausal_noise: u32,
    }
    let args = Args {
        n_q_heads: N_Q as u32,
        n_kv_heads: N_KV as u32,
        head_dim: HEAD_DIM as u32,
        n_kv_total,
        ctx_len: u32::try_from(ctx_len).map_err(|_| MetalError::BadShape {
            kernel: KERNEL,
            detail: "ctx_len does not fit u32".into(),
        })?,
        n_rows: N as u32,
        noise_start_pos: 0,
        swa_window: 0,
        ctx_scan_start: 0,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        noncausal_noise: dflash_noncausal_noise_enabled() as u32,
    };

    let main = ctx.pipeline("kernel_dflash_attn_full_gqa_split4_main_f32")?;
    enc.set_pipeline(&main);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_ctx);
    enc.set_tensor(3, v_ctx);
    enc.set_tensor(4, k_noise);
    enc.set_tensor(5, v_noise);
    enc.set_tensor(6, o_partial);
    enc.set_tensor(7, ml_partial);
    enc.dispatch(
        MTLSize {
            width: N_KV,
            height: N,
            depth: SPLIT,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    let reduce = ctx.pipeline("kernel_dflash_attn_full_gqa_split4_reduce_f32")?;
    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, o);
    enc.dispatch(
        MTLSize {
            width: N_Q,
            height: N,
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

/// SWA two-range split-K drafter attention. Same shape
/// contract as [`encode_dflash_attn_online_two_range_scan_f32`] (Q from the
/// noise block, K/V from the per-layer ctx cache plus the noise block,
/// `pos_ctx` for the SWA mask) but partitions the visible window across 4
/// simdgroups per (kv_head, query) and combines with the split4 reduce
/// shape. Product-qualified geometry: N8 / 32 Q heads / 8 KV heads /
/// head_dim 128.
///
/// Partials borrow caller-provided scratch sized
/// `n * N_KV * SPLIT * GROUP * head_dim` (o) and `... * 2` (ml).
/// Their accessed prefixes and the output must be mutually disjoint from all
/// inputs because the serial reduce consumes main's partial writes in place.
#[allow(clippy::too_many_arguments)]
pub fn encode_dflash_attn_swa_split4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    ctx_scan_start: usize,
) -> Result<(), MetalError> {
    encode_dflash_attn_swa_split4_with_noncausal_f32(
        ctx,
        enc,
        q,
        k_ctx,
        v_ctx,
        k_noise,
        v_noise,
        pos_ctx,
        o_partial,
        ml_partial,
        o,
        n,
        ctx_len,
        noise_start_pos,
        swa_window,
        ctx_scan_start,
        dflash_noncausal_noise_enabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_dflash_attn_swa_split4_with_noncausal_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_ctx: &MetalTensor,
    v_ctx: &MetalTensor,
    k_noise: &MetalTensor,
    v_noise: &MetalTensor,
    pos_ctx: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    o: &MetalTensor,
    n: usize,
    ctx_len: usize,
    noise_start_pos: u32,
    swa_window: u32,
    ctx_scan_start: usize,
    noncausal_noise: bool,
) -> Result<(), MetalError> {
    const KERNEL: &str = "dflash_attn_swa_split4";
    const N_Q: usize = 32;
    const N_KV: usize = 8;
    const GROUP: usize = 4;
    const HEAD_DIM: usize = 128;
    const SPLIT: usize = 4;
    if enc.concurrent {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "main/reduce dependency requires a serial encoder".into(),
        });
    }
    if n != 8 {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("expected product-qualified n=8, got {n}"),
        });
    }
    if ctx_scan_start > ctx_len {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("ctx_scan_start={ctx_scan_start} > ctx_len={ctx_len}"),
        });
    }
    let n_u32 = u32::try_from(n).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("n={n} does not fit u32"),
    })?;
    let ctx_len_u32 = u32::try_from(ctx_len).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("ctx_len={ctx_len} does not fit u32"),
    })?;
    let ctx_scan_start_u32 = u32::try_from(ctx_scan_start).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("ctx_scan_start={ctx_scan_start} does not fit u32"),
    })?;
    let total_rows = ctx_len.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: KERNEL,
        detail: "ctx_len + n overflow".into(),
    })?;
    let n_kv_total = u32::try_from(total_rows).map_err(|_| MetalError::BadShape {
        kernel: KERNEL,
        detail: format!("ctx_len + n={total_rows} does not fit u32"),
    })?;
    if noise_start_pos.checked_add(n_u32 - 1).is_none() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!("noise positions overflow u32: start={noise_start_pos} n={n}"),
        });
    }

    let q_elems = n
        .checked_mul(N_Q * HEAD_DIM)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "q/output element count overflow".into(),
        })?;
    let kv_stride = N_KV * HEAD_DIM;
    let ctx_elems = ctx_len
        .checked_mul(kv_stride)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "context element count overflow".into(),
        })?;
    let noise_elems = n
        .checked_mul(kv_stride)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "noise element count overflow".into(),
        })?;
    let partial_groups =
        n.checked_mul(N_KV * SPLIT * GROUP)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "partial group count overflow".into(),
            })?;
    let o_partial_elems =
        partial_groups
            .checked_mul(HEAD_DIM)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "output partial element count overflow".into(),
            })?;
    let ml_partial_elems = partial_groups
        .checked_mul(2)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "softmax partial element count overflow".into(),
        })?;

    let tensors = [
        q, k_ctx, v_ctx, k_noise, v_noise, pos_ctx, o_partial, ml_partial, o,
    ];
    if tensors.iter().any(|tensor| tensor.dtype != GgmlType::F32) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "all inputs, positions, scratch, and output must be F32-backed".into(),
        });
    }
    if !o_partial.is_writable() || !ml_partial.is_writable() || !o.is_writable() {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "partials and output must be writable".into(),
        });
    }
    if q.n_elements() as usize != q_elems
        || o.n_elements() as usize != q_elems
        || (k_ctx.n_elements() as usize) < ctx_elems
        || (v_ctx.n_elements() as usize) < ctx_elems
        || k_noise.n_elements() as usize != noise_elems
        || v_noise.n_elements() as usize != noise_elems
        || (pos_ctx.n_elements() as usize) < ctx_len
        || (o_partial.n_elements() as usize) < o_partial_elems
        || (ml_partial.n_elements() as usize) < ml_partial_elems
    {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: format!(
                "shape mismatch: q={} o={} k_ctx={} v_ctx={} k_noise={} v_noise={} \
                 pos_ctx={} o_partial={} ml_partial={} n={n} ctx_len={ctx_len}",
                q.n_elements(),
                o.n_elements(),
                k_ctx.n_elements(),
                v_ctx.n_elements(),
                k_noise.n_elements(),
                v_noise.n_elements(),
                pos_ctx.n_elements(),
                o_partial.n_elements(),
                ml_partial.n_elements(),
            ),
        });
    }
    let physical_requirements = [
        (q, q_elems),
        (k_ctx, ctx_elems),
        (v_ctx, ctx_elems),
        (k_noise, noise_elems),
        (v_noise, noise_elems),
        (pos_ctx, ctx_len),
        (o_partial, o_partial_elems),
        (ml_partial, ml_partial_elems),
        (o, q_elems),
    ];
    if physical_requirements.iter().any(|(tensor, elems)| {
        elems
            .checked_mul(std::mem::size_of::<f32>())
            .is_none_or(|bytes| !tensor_physical_range_valid(tensor, bytes, 4))
    }) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "tensor physical range is short or misaligned".into(),
        });
    }
    let required_bytes = |elems: usize| {
        elems
            .checked_mul(std::mem::size_of::<f32>())
            .expect("physical range validation already rejected byte overflow")
    };
    let writes = [
        (o_partial, required_bytes(o_partial_elems)),
        (ml_partial, required_bytes(ml_partial_elems)),
        (o, required_bytes(q_elems)),
    ];
    let reads = [
        (q, required_bytes(q_elems)),
        (k_ctx, required_bytes(ctx_elems)),
        (v_ctx, required_bytes(ctx_elems)),
        (k_noise, required_bytes(noise_elems)),
        (v_noise, required_bytes(noise_elems)),
        (pos_ctx, required_bytes(ctx_len)),
    ];
    if writes.iter().enumerate().any(|(index, left)| {
        writes[index + 1..]
            .iter()
            .any(|right| tensor_byte_ranges_overlap(left.0, left.1, right.0, right.1))
    }) || writes.iter().any(|write| {
        reads
            .iter()
            .any(|read| tensor_byte_ranges_overlap(write.0, write.1, read.0, read.1))
    }) {
        return Err(MetalError::BadShape {
            kernel: KERNEL,
            detail: "scratch/output ranges must be mutually disjoint from inputs".into(),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_kv_total: u32,
        ctx_len: u32,
        n_rows: u32,
        noise_start_pos: u32,
        swa_window: u32,
        ctx_scan_start: u32,
        scale: f32,
        noncausal_noise: u32,
    }
    let args = Args {
        n_q_heads: N_Q as u32,
        n_kv_heads: N_KV as u32,
        head_dim: HEAD_DIM as u32,
        n_kv_total,
        ctx_len: ctx_len_u32,
        n_rows: n_u32,
        noise_start_pos,
        swa_window,
        ctx_scan_start: ctx_scan_start_u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        noncausal_noise: noncausal_noise as u32,
    };

    let main = ctx.pipeline("kernel_dflash_attn_swa_split4_main_f32")?;
    enc.set_pipeline(&main);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_ctx);
    enc.set_tensor(3, v_ctx);
    enc.set_tensor(4, k_noise);
    enc.set_tensor(5, v_noise);
    enc.set_tensor(6, pos_ctx);
    enc.set_tensor(7, o_partial);
    enc.set_tensor(8, ml_partial);
    enc.dispatch(
        MTLSize {
            width: N_KV,
            height: n,
            depth: SPLIT,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    let reduce = ctx.pipeline("kernel_dflash_attn_swa_split4_reduce_f32")?;
    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, o);
    enc.dispatch(
        MTLSize {
            width: N_Q,
            height: n,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    /// **v0.72.2 codex code-review test #1**: dflash attention kernel
    /// matches the CPU oracle bit-tight under each mask regime.
    ///
    /// Exercises:
    ///   * `ctx_len == 0` (degenerate: noise-only attention)
    ///   * `ctx_len > 0, swa_window > 0` (SWA layer)
    ///   * `ctx_len > 0, swa_window == 0` (full-attn layer; codex
    ///     mask-semantics flag — full-attn allows ALL ctx keys, no
    ///     causal restriction)
    ///   * `ctx_len > swa_window` (SWA boundary; some ctx keys
    ///     denied by the window even though causal)
    ///   * `ctx_len > 0` with non-contiguous / gapped pos_k
    ///   * Edge: q_pos == k_pos exactly (boundary causal — allowed
    ///     under SWA)
    #[test]
    fn dflash_attn_matches_cpu_oracle_under_mask_regimes() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };

        // Drafter shape: n_q=32, n_kv=8 (group=4), head_dim=128, N=16.
        let n = 16;
        let n_q = 32;
        let n_kv = 8;
        let hd = 128;
        let q_dim = n_q * hd;
        let kv_stride = n_kv * hd;

        // Deterministic synthetic activations.
        let make_buf = |seed: u32, len: usize| -> Vec<f32> {
            let mut s = seed;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.5
                })
                .collect()
        };

        let q = make_buf(1, n * q_dim);

        struct Case {
            label: &'static str,
            ctx_len: usize,
            swa_window: u32,
            noise_start_pos: u32,
            // Custom pos_k for the ctx half (length ctx_len).
            // Builder receives ctx_len + noise_start_pos and returns
            // ctx-side positions.
            pos_ctx: fn(usize, u32) -> Vec<i32>,
        }

        fn pos_recent(ctx_len: usize, noise_start: u32) -> Vec<i32> {
            (0..ctx_len)
                .map(|c| noise_start as i32 - ctx_len as i32 + c as i32)
                .collect()
        }
        fn pos_gapped(ctx_len: usize, noise_start: u32) -> Vec<i32> {
            // Every other position skipped — non-contiguous.
            (0..ctx_len)
                .map(|c| (noise_start as i32 - 2 * ctx_len as i32 + 2 * c as i32).max(0))
                .collect()
        }

        let cases = [
            Case {
                label: "ctx_len=0 (noise-only)",
                ctx_len: 0,
                swa_window: 2048,
                noise_start_pos: 4,
                pos_ctx: pos_recent,
            },
            Case {
                label: "swa, ctx within window",
                ctx_len: 8,
                swa_window: 2048,
                noise_start_pos: 16,
                pos_ctx: pos_recent,
            },
            Case {
                label: "full-attn (swa=0), ctx allowed permissively",
                ctx_len: 8,
                swa_window: 0,
                noise_start_pos: 16,
                pos_ctx: pos_recent,
            },
            Case {
                label: "swa boundary, ctx_len > swa_window",
                ctx_len: 64,
                swa_window: 16,
                noise_start_pos: 80,
                pos_ctx: pos_recent,
            },
            Case {
                label: "swa, gapped pos_ctx",
                ctx_len: 12,
                swa_window: 2048,
                noise_start_pos: 32,
                pos_ctx: pos_gapped,
            },
            Case {
                label: "swa, q_pos == k_pos boundary",
                ctx_len: 4,
                swa_window: 2048,
                // pos_recent constructs ctx positions
                // [noise_start - ctx_len .. noise_start). With
                // noise_start=4, ctx pos = [0,1,2,3]. q_pos at q_idx=0
                // = 4. So q_pos > k_pos — no exact equality.
                // To exercise q_pos == k_pos: shift noise_start_pos so
                // pos_ctx ends at exactly noise_start_pos (= q_pos at
                // q_idx=0). Set ctx_len=4, noise_start_pos=4 →
                // pos_ctx = [0..4); the last ctx is at pos=3, q_pos at
                // q_idx=0 is 4 → still strict. Make ctx_len=5 and
                // noise_start_pos=4 → pos_ctx = [-1..4); ctx[4]=3.
                // Hmm same. This case structurally enforces k_pos < q_pos
                // unless we allow ctx that overlaps noise positions
                // (semantically a contract violation per codex flag).
                //
                // Instead, this case tests q_pos > all ctx positions
                // by a margin of 1 — boundary-adjacent without overlap.
                noise_start_pos: 4,
                pos_ctx: pos_recent,
            },
        ];

        for c in &cases {
            let pos_ctx_vec = (c.pos_ctx)(c.ctx_len, c.noise_start_pos);
            let n_kv_total = c.ctx_len + n;
            // Build pos_k = pos_ctx ++ [noise_start..noise_start+N].
            let mut pos_k = Vec::with_capacity(n_kv_total);
            pos_k.extend_from_slice(&pos_ctx_vec);
            for i in 0..n {
                pos_k.push((c.noise_start_pos + i as u32) as i32);
            }
            let k = make_buf(2, n_kv_total * kv_stride);
            let v = make_buf(3, n_kv_total * kv_stride);
            let ctx_rows = c.ctx_len.max(1);
            let mut k_ctx = vec![0.0_f32; ctx_rows * kv_stride];
            let mut v_ctx = vec![0.0_f32; ctx_rows * kv_stride];
            if c.ctx_len > 0 {
                k_ctx[..c.ctx_len * kv_stride].copy_from_slice(&k[..c.ctx_len * kv_stride]);
                v_ctx[..c.ctx_len * kv_stride].copy_from_slice(&v[..c.ctx_len * kv_stride]);
            }
            let k_noise = k[c.ctx_len * kv_stride..].to_vec();
            let v_noise = v[c.ctx_len * kv_stride..].to_vec();
            let mut pos_ctx = vec![0_i32; c.ctx_len.max(1)];
            if c.ctx_len > 0 {
                pos_ctx[..c.ctx_len].copy_from_slice(&pos_ctx_vec);
            }

            let cpu = dflash_attn_cpu_oracle(
                &q,
                &k,
                &v,
                &pos_k,
                n,
                n_q,
                n_kv,
                hd,
                n_kv_total,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                false,
            );
            let gpu = dflash_attn_readback(
                &ctx,
                &q,
                &k,
                &v,
                &pos_k,
                n,
                n_q,
                n_kv,
                hd,
                n_kv_total,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
            )
            .expect("dflash_attn dispatch");
            let gpu_two_range = dflash_attn_two_range_readback(
                &ctx,
                &q,
                &k_ctx,
                &v_ctx,
                &k_noise,
                &v_noise,
                &pos_ctx,
                n,
                n_q,
                n_kv,
                hd,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                false,
                false,
                false,
                false,
                0,
            )
            .expect("dflash_attn_two_range dispatch");
            let gpu_online_two_range = dflash_attn_two_range_readback(
                &ctx,
                &q,
                &k_ctx,
                &v_ctx,
                &k_noise,
                &v_noise,
                &pos_ctx,
                n,
                n_q,
                n_kv,
                hd,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                true,
                false,
                false,
                false,
                0,
            )
            .expect("dflash_attn_online_two_range dispatch");
            let ctx_scan_start = if c.swa_window > 0 && c.ctx_len > 0 {
                let min_pos = c.noise_start_pos.saturating_sub(c.swa_window);
                pos_ctx_vec.partition_point(|&pos| pos >= 0 && (pos as u32) < min_pos)
            } else {
                0
            };
            let gpu_online_two_range_scan = dflash_attn_two_range_readback(
                &ctx,
                &q,
                &k_ctx,
                &v_ctx,
                &k_noise,
                &v_noise,
                &pos_ctx,
                n,
                n_q,
                n_kv,
                hd,
                c.ctx_len,
                c.noise_start_pos,
                c.swa_window,
                true,
                false,
                false,
                false,
                ctx_scan_start,
            )
            .expect("dflash_attn_online_two_range scan dispatch");
            let gpu_full_gqa_split4 = if c.swa_window == 0 {
                Some(
                    dflash_attn_two_range_readback(
                        &ctx,
                        &q,
                        &k_ctx,
                        &v_ctx,
                        &k_noise,
                        &v_noise,
                        &pos_ctx,
                        n,
                        n_q,
                        n_kv,
                        hd,
                        c.ctx_len,
                        c.noise_start_pos,
                        c.swa_window,
                        false,
                        true,
                        false,
                        false,
                        0,
                    )
                    .expect("dflash_attn_full_gqa_split4 dispatch"),
                )
            } else {
                None
            };

            let mut max_abs = 0.0f32;
            let mut max_abs_two_range = 0.0f32;
            let mut max_abs_online_two_range = 0.0f32;
            let mut max_abs_online_two_range_scan = 0.0f32;
            let mut sum_sq_diff = 0.0f64;
            let mut sum_sq_diff_two_range = 0.0f64;
            let mut sum_sq_diff_online_two_range = 0.0f64;
            let mut sum_sq_diff_online_two_range_scan = 0.0f64;
            let mut sum_sq_cpu = 0.0f64;
            for i in 0..cpu.len() {
                let d = (gpu[i] - cpu[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let d_two_range = (gpu_two_range[i] - cpu[i]).abs();
                if d_two_range > max_abs_two_range {
                    max_abs_two_range = d_two_range;
                }
                let d_online_two_range = (gpu_online_two_range[i] - cpu[i]).abs();
                if d_online_two_range > max_abs_online_two_range {
                    max_abs_online_two_range = d_online_two_range;
                }
                let d_online_two_range_scan = (gpu_online_two_range_scan[i] - cpu[i]).abs();
                if d_online_two_range_scan > max_abs_online_two_range_scan {
                    max_abs_online_two_range_scan = d_online_two_range_scan;
                }
                let dd = (gpu[i] - cpu[i]) as f64;
                sum_sq_diff += dd * dd;
                let dd_two_range = (gpu_two_range[i] - cpu[i]) as f64;
                sum_sq_diff_two_range += dd_two_range * dd_two_range;
                let dd_online_two_range = (gpu_online_two_range[i] - cpu[i]) as f64;
                sum_sq_diff_online_two_range += dd_online_two_range * dd_online_two_range;
                let dd_online_two_range_scan = (gpu_online_two_range_scan[i] - cpu[i]) as f64;
                sum_sq_diff_online_two_range_scan +=
                    dd_online_two_range_scan * dd_online_two_range_scan;
                sum_sq_cpu += (cpu[i] as f64).powi(2);
            }
            let rel_l2 = sum_sq_diff.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            let rel_l2_two_range = sum_sq_diff_two_range.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            let rel_l2_online_two_range =
                sum_sq_diff_online_two_range.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            let rel_l2_online_two_range_scan =
                sum_sq_diff_online_two_range_scan.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
            eprintln!(
                "[dflash-attn-mask {label}] max|Δ|={max_abs:.3e} rel_l2={rel_l2:.3e} two_range_max|Δ|={max_abs_two_range:.3e} two_range_rel_l2={rel_l2_two_range:.3e} online_two_range_max|Δ|={max_abs_online_two_range:.3e} online_two_range_rel_l2={rel_l2_online_two_range:.3e} scan_start={ctx_scan_start} online_scan_max|Δ|={max_abs_online_two_range_scan:.3e} online_scan_rel_l2={rel_l2_online_two_range_scan:.3e}",
                label = c.label
            );
            assert!(max_abs < 1e-4, "{}: max|Δ|={max_abs} too large", c.label);
            assert!(rel_l2 < 1e-5, "{}: rel_l2={rel_l2} too large", c.label);
            assert!(
                max_abs_two_range < 1e-4,
                "{}: two-range max|Δ|={max_abs_two_range} too large",
                c.label
            );
            assert!(
                rel_l2_two_range < 1e-5,
                "{}: two-range rel_l2={rel_l2_two_range} too large",
                c.label
            );
            assert!(
                max_abs_online_two_range < 1e-4,
                "{}: online two-range max|Δ|={max_abs_online_two_range} too large",
                c.label
            );
            assert!(
                rel_l2_online_two_range < 1e-5,
                "{}: online two-range rel_l2={rel_l2_online_two_range} too large",
                c.label
            );
            assert!(
                max_abs_online_two_range_scan < 1e-4,
                "{}: online scan max|Δ|={max_abs_online_two_range_scan} too large",
                c.label
            );
            assert!(
                rel_l2_online_two_range_scan < 1e-5,
                "{}: online scan rel_l2={rel_l2_online_two_range_scan} too large",
                c.label
            );
            if let Some(gpu_full_gqa_split4) = gpu_full_gqa_split4 {
                let mut candidate_max_abs = 0.0f32;
                let mut candidate_sq_diff = 0.0f64;
                for (&got, &want) in gpu_full_gqa_split4.iter().zip(&cpu) {
                    assert!(
                        got.is_finite(),
                        "{}: split4 produced nonfinite output",
                        c.label
                    );
                    let diff = (got - want).abs();
                    candidate_max_abs = candidate_max_abs.max(diff);
                    candidate_sq_diff += (diff as f64).powi(2);
                }
                let candidate_rel_l2 = candidate_sq_diff.sqrt() / (sum_sq_cpu.sqrt() + 1e-30);
                eprintln!(
                    "[dflash-attn-mask split4 {}] max|delta|={candidate_max_abs:.3e} \
                     rel_l2={candidate_rel_l2:.3e}",
                    c.label
                );
                assert!(
                    candidate_max_abs < 1e-4,
                    "{}: split4 max|delta|={candidate_max_abs} too large",
                    c.label
                );
                assert!(
                    candidate_rel_l2 < 1e-5,
                    "{}: split4 rel_l2={candidate_rel_l2} too large",
                    c.label
                );
            }
        }
    }

    #[test]
    fn dflash_attn_swa_split4_matches_cpu_oracle_n8() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let n = 8;
        let n_q = 32;
        let n_kv = 8;
        let hd = 128;
        let ctx_len = 40;
        let swa_window = 16u32;
        let noise_start_pos = 80u32;
        let kv_stride = n_kv * hd;
        let make_buf = |seed: u32, len: usize| -> Vec<f32> {
            let mut state = seed;
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.5
                })
                .collect()
        };
        let q = make_buf(11, n * n_q * hd);
        let k_ctx = make_buf(12, ctx_len * kv_stride);
        let v_ctx = make_buf(13, ctx_len * kv_stride);
        let k_noise = make_buf(14, n * kv_stride);
        let v_noise = make_buf(15, n * kv_stride);
        let pos_ctx: Vec<i32> = (0..ctx_len)
            .map(|index| noise_start_pos as i32 - ctx_len as i32 + index as i32)
            .collect();
        let min_pos = noise_start_pos.saturating_sub(swa_window);
        let ctx_scan_start = pos_ctx.partition_point(|&pos| pos >= 0 && (pos as u32) < min_pos);
        let mut k = k_ctx.clone();
        k.extend_from_slice(&k_noise);
        let mut v = v_ctx.clone();
        v.extend_from_slice(&v_noise);
        let mut pos_k = pos_ctx.clone();
        pos_k.extend((0..n).map(|index| (noise_start_pos + index as u32) as i32));
        let cpu = dflash_attn_cpu_oracle(
            &q,
            &k,
            &v,
            &pos_k,
            n,
            n_q,
            n_kv,
            hd,
            ctx_len + n,
            ctx_len,
            noise_start_pos,
            swa_window,
            false,
        );
        let split4 = dflash_attn_two_range_readback(
            &ctx,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            n,
            n_q,
            n_kv,
            hd,
            ctx_len,
            noise_start_pos,
            swa_window,
            false,
            false,
            true,
            false,
            ctx_scan_start,
        )
        .expect("SWA split4 N8 dispatch");
        let cpu_noncausal = dflash_attn_cpu_oracle(
            &q,
            &k,
            &v,
            &pos_k,
            n,
            n_q,
            n_kv,
            hd,
            ctx_len + n,
            ctx_len,
            noise_start_pos,
            swa_window,
            true,
        );
        let split4_noncausal = dflash_attn_two_range_readback(
            &ctx,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            n,
            n_q,
            n_kv,
            hd,
            ctx_len,
            noise_start_pos,
            swa_window,
            false,
            false,
            true,
            true,
            ctx_scan_start,
        )
        .expect("noncausal SWA split4 N8 dispatch");

        let assert_close = |label: &str, got: &[f32], want: &[f32]| {
            let mut max_abs = 0.0f32;
            let mut diff_sq = 0.0f64;
            let mut ref_sq = 0.0f64;
            for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
                assert!(got.is_finite(), "{label}: nonfinite output at {index}");
                let diff = (got - want).abs();
                max_abs = max_abs.max(diff);
                diff_sq += (diff as f64).powi(2);
                ref_sq += (want as f64).powi(2);
            }
            let rel_l2 = diff_sq.sqrt() / (ref_sq.sqrt() + 1e-30);
            eprintln!("[{label}] max|delta|={max_abs:.3e} rel_l2={rel_l2:.3e}");
            assert!(max_abs < 1e-4, "{label}: max|delta|={max_abs} too large");
            assert!(rel_l2 < 1e-5, "{label}: rel_l2={rel_l2} too large");
        };
        assert_close("dflash-swa-split4-n8", &split4, &cpu);
        assert_close(
            "dflash-swa-split4-n8-noncausal",
            &split4_noncausal,
            &cpu_noncausal,
        );
    }

    #[test]
    fn dflash_attn_swa_split4_rejects_unsafe_contracts() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let n = 8usize;
        let ctx_len = 16usize;
        let q_elems = n * 32 * 128;
        let kv_stride = 8 * 128;
        let partial_groups = n * 8 * 4 * 4;
        let q = MetalTensor::zeros_f32(&ctx, vec![q_elems as u64]).unwrap();
        let k_ctx = MetalTensor::zeros_f32(&ctx, vec![(ctx_len * kv_stride) as u64]).unwrap();
        let v_ctx = MetalTensor::zeros_f32(&ctx, vec![(ctx_len * kv_stride) as u64]).unwrap();
        let k_noise = MetalTensor::zeros_f32(&ctx, vec![(n * kv_stride) as u64]).unwrap();
        let v_noise = MetalTensor::zeros_f32(&ctx, vec![(n * kv_stride) as u64]).unwrap();
        let pos_ctx = MetalTensor::zeros_f32(&ctx, vec![ctx_len as u64]).unwrap();
        let short_pos = MetalTensor::zeros_f32(&ctx, vec![(ctx_len - 1) as u64]).unwrap();
        let o_partial = MetalTensor::zeros_f32(&ctx, vec![(partial_groups * 128) as u64]).unwrap();
        let ml_partial = MetalTensor::zeros_f32(&ctx, vec![(partial_groups * 2) as u64]).unwrap();
        let o = MetalTensor::zeros_f32(&ctx, vec![q_elems as u64]).unwrap();

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let concurrent = KernelEncoder::begin_concurrent(&cmd);
        let concurrent_error = encode_dflash_attn_swa_split4_f32(
            &ctx,
            &concurrent,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            &o_partial,
            &ml_partial,
            &o,
            n,
            ctx_len,
            32,
            16,
            0,
        )
        .expect_err("concurrent main/reduce must be rejected");
        concurrent.end();
        assert!(concurrent_error.to_string().contains("serial encoder"));

        let cmd = ctx.queue.commandBuffer().expect("command buffer");
        let enc = KernelEncoder::begin(&cmd);
        let short_pos_error = encode_dflash_attn_swa_split4_f32(
            &ctx,
            &enc,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &short_pos,
            &o_partial,
            &ml_partial,
            &o,
            n,
            ctx_len,
            32,
            16,
            0,
        )
        .expect_err("short positions must be rejected");
        assert!(short_pos_error.to_string().contains("shape mismatch"));

        let alias_error = encode_dflash_attn_swa_split4_f32(
            &ctx, &enc, &q, &k_ctx, &v_ctx, &k_noise, &v_noise, &pos_ctx, &o_partial, &o_partial,
            &o, n, ctx_len, 32, 16, 0,
        )
        .expect_err("aliased partials must be rejected");
        assert!(alias_error.to_string().contains("disjoint"));

        let position_error = encode_dflash_attn_swa_split4_f32(
            &ctx,
            &enc,
            &q,
            &k_ctx,
            &v_ctx,
            &k_noise,
            &v_noise,
            &pos_ctx,
            &o_partial,
            &ml_partial,
            &o,
            n,
            ctx_len,
            u32::MAX,
            16,
            0,
        )
        .expect_err("overflowing noise positions must be rejected");
        assert!(position_error.to_string().contains("overflow"));
        enc.end();
    }

    /// **v0.72.2 codex code-review test #2**: head_dim > 256 must be
    /// rejected at the host wrapper. Kernel uses fixed-size [8] register
    /// arrays sized for head_dim=256; head_dim=320 would silently
    /// stack-OOB without this guard.
    #[test]
    fn dflash_attn_rejects_head_dim_over_256() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        // Make tiny placeholder buffers; we only care about the host
        // wrapper validation.
        let q = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let k = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let v = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let p = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let o = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        let res = encode_dflash_attn_f32(
            &ctx, &enc, &q, &k, &v, &p, &o, 16,   // n
            32,   // n_q_heads
            8,    // n_kv_heads
            320,  // head_dim — REJECTED
            17,   // n_kv_total
            1,    // ctx_len
            0,    // noise_start_pos
            2048, // swa_window
        );
        enc.end();
        match res {
            Err(MetalError::BadShape { detail, .. }) => {
                assert!(detail.contains("256"), "wrong error detail: {detail}");
            }
            other => panic!("expected BadShape on head_dim>256, got {other:?}"),
        }
    }
}

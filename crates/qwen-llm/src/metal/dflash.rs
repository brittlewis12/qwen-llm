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

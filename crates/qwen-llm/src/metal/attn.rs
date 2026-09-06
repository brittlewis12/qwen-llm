//! Attention: naive decode, flash-attn v4 decode/prefill, matrix sidecar.

use super::*;

/// Split the gated-attention Q-projection output into separate Q and
/// gate tensors. Input layout per head: `[head_dim Q, head_dim gate]`,
/// total length `n_heads * 2 * head_dim`. Outputs are
/// `[n_heads, head_dim]` each.
///
/// v0.432: production attention paths read the interleaved layout
/// directly (strided q-norm + strided gate sigmoid_mul); this kernel
/// remains for tests and as the reference for the interleave layout.
pub fn encode_split_q_gate_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_full: &MetalTensor,
    q: &MetalTensor,
    gate: &MetalTensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    let want_full = (n_heads * 2 * head_dim) as u64;
    let want_each = (n_heads * head_dim) as u64;
    if q_full.n_elements() != want_full {
        return Err(MetalError::BadShape {
            kernel: "split_q_gate",
            detail: format!("q_full expected {want_full} elements"),
        });
    }
    if q.n_elements() != want_each || gate.n_elements() != want_each {
        return Err(MetalError::BadShape {
            kernel: "split_q_gate",
            detail: format!("q/gate expected {want_each} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        head_dim: u32,
    }
    let pso = ctx.pipeline("kernel_split_q_gate_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
        },
    );
    enc.set_tensor(1, q_full);
    enc.set_tensor(2, q);
    enc.set_tensor(3, gate);

    let total = n_heads * head_dim;
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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
    Ok(())
}

pub fn encode_split_qkv_fused_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    q_full: &MetalTensor,
    k_out: &MetalTensor,
    v_out: &MetalTensor,
    n_rows: usize,
    q_full_dim: usize,
    kv_dim: usize,
) -> Result<(), MetalError> {
    let fused_stride = q_full_dim + 2 * kv_dim;
    let want_src = (n_rows * fused_stride) as u64;
    let want_q = (n_rows * q_full_dim) as u64;
    let want_kv = (n_rows * kv_dim) as u64;
    if src.n_elements() != want_src {
        return Err(MetalError::BadShape {
            kernel: "split_qkv_fused",
            detail: format!("src expected {want_src} elements"),
        });
    }
    if q_full.n_elements() != want_q
        || k_out.n_elements() != want_kv
        || v_out.n_elements() != want_kv
    {
        return Err(MetalError::BadShape {
            kernel: "split_qkv_fused",
            detail: format!("q/k/v expected {want_q}/{want_kv}/{want_kv} elements"),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_rows: u32,
        q_full_dim: u32,
        kv_dim: u32,
        fused_stride: u32,
    }
    let pso = ctx.pipeline("kernel_split_qkv_fused_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_rows: n_rows as u32,
            q_full_dim: q_full_dim as u32,
            kv_dim: kv_dim as u32,
            fused_stride: fused_stride as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, q_full);
    enc.set_tensor(3, k_out);
    enc.set_tensor(4, v_out);

    let total = n_rows * fused_stride;
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
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
    Ok(())
}

/// Naive fused single-token attention over an F16 KV cache (scores in
/// threadgroup memory, so the host caps `n_pos` at 7,168 positions). Halves attention bandwidth at long
/// context (saves ~4 GB of reads/token at 4K positions on 27B). Q is
/// still F32; output is F32.
///
/// Caller is responsible for ensuring `k_cache` and `v_cache` are F16-typed
/// MetalTensors (typically allocated via `MetalTensor::zeros_f16`).
pub fn encode_attn_decode_f16kv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    out: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
) -> Result<(), MetalError> {
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
            detail: format!(
                "k/v expected F16 dtype, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want || out.n_elements() != want {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
            detail: format!("q/out expected {want} elements"),
        });
    }
    let kv_stride = n_kv_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_attn_decode_f16kv")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, out);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_simdgroups = tg_threads.div_ceil(32);
    let scores_bytes = n_pos * std::mem::size_of::<f32>();
    let shred_bytes = (n_simdgroups * std::mem::size_of::<f32>()).max(32);
    if scores_bytes > 28 * 1024 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_f16kv",
            detail: format!(
                "n_pos={n_pos} requires {scores_bytes} B threadgroup memory; max ~28 KB"
            ),
        });
    }
    enc.set_threadgroup_memory(0, scores_bytes);
    enc.set_threadgroup_memory(1, shred_bytes);
    enc.dispatch(
        MTLSize {
            width: n_q_heads,
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

/// Compute v4-friendly NWG (split-K partition count) for a given context length.
///
/// Heuristic determined empirically on M4 Max via whole-model phase/ctx sweeps:
///   - Long-context dense group=6 now uses NWG=64; it cuts dense 27B attention
///     by ~4.5 ms at 16K and ~9.3 ms at 32K versus NWG=32.
///   - NWG=16 is competitive for n_pos < 256 (very-cold-start regime).
///   - NWG=1 is catastrophically under-occupied (4 TGs total) — DO NOT
///     ship as performance config; useful only for correctness debugging.
///
/// Long-context group=8 can use NWG=256 with the two-threadgroup reduce path;
/// smaller overrides such as 128/192 remain useful A/B knobs when validating
/// the main-vs-reduce split.
/// `QWEN_ATTN_V4_NWG=1..ATTN_V4_NWG_MAX` is an A/B knob for whole-model sweeps.
/// `QWEN_ATTN_V4_SUBGROUP_MIN_POS=4096` restores the older long-only threshold.
pub fn attn_v4_choose_nwg(n_pos: usize, group: usize) -> usize {
    static NWG_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(nwg) = *NWG_OVERRIDE.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_NWG")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| (1..=ATTN_V4_NWG_MAX).contains(v))
    }) {
        return nwg;
    }

    if group == 8 && n_pos >= 16_384 {
        256
    } else if (matches!(group, 8 | 16) && n_pos >= attn_v4_subgroup_min_pos())
        || (matches!(group, 4 | 6) && n_pos >= 4096)
    {
        // 2026-08-22 audit: for group 4|6 the old 64 partitions
        // under-parallelize at depth (56 GB/s at 130K against the 474
        // GB/s stream). The synthetic sweep picked 128 at 8-64K and
        // 512 at 130K (1.6-2.1x); groups 8|16 keep their table.
        if matches!(group, 4 | 6) {
            if n_pos >= 98_304 { 512 } else { 128 }
        } else {
            64
        }
    } else if n_pos < 256 {
        16
    } else {
        32
    }
}

/// Pick the v4 tile size (KV positions per inner softmax tile) for a given
/// context and GQA group size.
///
/// Current tuning:
/// - group=6 (27B dense): keep C=32 until we have fresh long-ctx sweep data.
/// - group in {8,16} (35B A3B / 122B A10B): C=64 is the current best-known
///   medium/long-context choice. C=128 is experimental and should only be
///   enabled if fresh sweeps beat 64 on real hardware.
///
/// `QWEN_ATTN_V4_TILE_C={16,32,64,128}` is an A/B knob for whole-model sweeps.
pub fn attn_v4_choose_tile_c(n_pos: usize, group: usize) -> usize {
    if group == 8
        && let Some(tile_c) = attn_v4_g8_vstage_c()
    {
        return tile_c;
    }
    static TILE_C_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(tile_c) = *TILE_C_OVERRIDE.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_TILE_C")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| matches!(*v, 16 | 32 | 64 | 128))
    }) {
        return tile_c;
    }

    if group == 16 && n_pos >= 32768 {
        128
    } else if matches!(group, 8 | 16) && n_pos >= attn_v4_subgroup_min_pos() {
        64
    } else {
        32
    }
}

pub(crate) fn attn_v4_g8_vstage_c() -> Option<usize> {
    static VSTAGE_C: OnceLock<Option<usize>> = OnceLock::new();
    *VSTAGE_C.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_G8_VSTAGE_C")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| matches!(*v, 16 | 32))
    })
}

pub(crate) fn attn_v4_g8_bcast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("QWEN_ATTN_V4_G8_BCAST").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

/// W1b partition-packing opt-in: `QWEN_ATTN_V4_PACK=4` packs 4 partitions
/// (one simdgroup each) into 128-thread TGs for the g8/t4/C64 F16 decode
/// main. Default off; see docs/bench/2026-07-06-w1b-attn-partition-pack/.
pub(crate) fn attn_v4_pack() -> usize {
    static PACK: OnceLock<usize> = OnceLock::new();
    *PACK.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_PACK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v == 4)
            .unwrap_or(1)
    })
}

/// Group-tile subgroup size for v4's decode main pass.
///
/// The 122B A10B shape (`GROUP=16`) is faster when split across multiple
/// threadgroups: tile4 trades extra K/V reads for much higher occupancy and
/// lower register pressure. Keep `QWEN_ATTN_V4_G16_TILE` as a kill switch / A/B
/// knob (`4`, `8`, or `16`), but default medium/long-context group16 to tile4.
///
/// The 35B A3B shape (`GROUP=8`) has the same long-context signature with a
/// smaller best split. Keep `QWEN_ATTN_V4_G8_TILE` as a kill switch / A/B knob
/// (`2`, `4`, or `8`): default medium-context group8 decode to tile2, then use
/// tile4 at true-long contexts where NWG=256 recovers enough occupancy.
pub(crate) fn attn_v4_g8_tile_override(var: &str) -> Option<usize> {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| matches!(*v, 2 | 4 | 8))
}

pub(crate) fn attn_v4_subgroup_min_pos() -> usize {
    static MIN_POS: OnceLock<usize> = OnceLock::new();
    *MIN_POS.get_or_init(|| {
        std::env::var("QWEN_ATTN_V4_SUBGROUP_MIN_POS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(ATTN_V4_SUBGROUP_MIN_POS_DEFAULT)
    })
}

pub fn attn_v4_choose_group_tile(n_pos: usize, group: usize) -> usize {
    if let Some(tile) = ATTN_V4_GROUP_TILE_OVERRIDE.with(|cell| cell.get()) {
        return tile;
    }
    if n_pos < attn_v4_subgroup_min_pos() {
        return group;
    }
    if group == 8 {
        static G8_TILE: OnceLock<Option<usize>> = OnceLock::new();
        let default_tile = if n_pos >= 16_384 { 4 } else { 2 };
        return G8_TILE
            .get_or_init(|| attn_v4_g8_tile_override("QWEN_ATTN_V4_G8_TILE"))
            .unwrap_or(default_tile);
    }
    if group != 16 {
        return group;
    }
    static G16_TILE: OnceLock<Option<usize>> = OnceLock::new();
    G16_TILE
        .get_or_init(|| {
            std::env::var("QWEN_ATTN_V4_G16_TILE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|v| matches!(*v, 4 | 8 | 16))
        })
        .unwrap_or(4)
}

pub fn attn_v4_choose_group_tile_prefill(n_pos: usize, group: usize) -> usize {
    if n_pos < 4096 {
        return group;
    }
    if group == 8 {
        static G8_PREFILL_TILE: OnceLock<Option<usize>> = OnceLock::new();
        return G8_PREFILL_TILE
            .get_or_init(|| {
                attn_v4_g8_tile_override("QWEN_ATTN_V4_G8_PREFILL_TILE")
                    .or_else(|| attn_v4_g8_tile_override("QWEN_ATTN_V4_G8_TILE"))
            })
            .unwrap_or(8);
    }
    attn_v4_choose_group_tile(n_pos, group)
}

pub fn with_attn_v4_group_tile_override<T>(tile: usize, f: impl FnOnce() -> T) -> T {
    let prev = ATTN_V4_GROUP_TILE_OVERRIDE.with(|cell| {
        let prev = cell.get();
        cell.set(Some(tile));
        prev
    });
    let out = f();
    ATTN_V4_GROUP_TILE_OVERRIDE.with(|cell| cell.set(prev));
    out
}

/// Encode v4 main kernel + reduce kernel in sequence.
/// Hardcoded constants (must match `kernels/attn_v4.metal`):
///   DK = DV = 256, lanes = 32, GROUP in {4, 6, 8, 16}.
/// Tile size `tile_c ∈ {16, 32, 64, 128}` selects the kernel variant.
/// Use `attn_v4_choose_tile_c(n_pos, group)` for the empirically-tuned choice
/// or pass 32 (default; backward-compat) if unsure.
pub fn encode_attn_decode_v4_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
) -> Result<(), MetalError> {
    // Hardcoded shape preconditions.
    const DK: usize = 256;
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("head_dim={head_dim} but kernel hardcodes {DK}"),
        });
    }
    if !matches!(group, 4 | 6 | 8 | 16) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("group={group} unsupported; expected one of {{4, 6, 8, 16}}"),
        });
    }
    if k_cache.dtype != v_cache.dtype || !matches!(k_cache.dtype, GgmlType::F16 | GgmlType::Q8_0) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!(
                "k/v expected matching F16 or Q8_0 dtypes, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want_q || out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("q/out expected {want_q} elements"),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let group_tile = attn_v4_choose_group_tile(n_pos, group);
    if group_tile == 0 || !group.is_multiple_of(group_tile) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("group_tile={group_tile} must divide group={group}"),
        });
    }
    if group_tile != group
        && !(group == 16 && matches!(group_tile, 4 | 8)
            || group == 8 && matches!(group_tile, 2 | 4))
    {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!("unsupported subgroup: group={group} group_tile={group_tile}"),
        });
    }
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if o_partial.n_elements() < want_o_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!(
                "o_partial too small: have {}, need ≥ {want_o_partial}",
                o_partial.n_elements()
            ),
        });
    }
    if ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4",
            detail: format!(
                "ml_partial too small: have {}, need ≥ {want_ml_partial}",
                ml_partial.n_elements()
            ),
        });
    }

    let kv_stride = n_kv_heads * head_dim;
    // Pre-multiply scale by log2(e) so kernel uses exp2 (Apple GPU fast path).
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));

    // -------- Main kernel ------------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        scale: f32,
    }
    let use_g8_bcast = k_cache.dtype == GgmlType::F16
        && group == 8
        && matches!(group_tile, 2 | 4)
        && tile_c == 64
        && attn_v4_g8_bcast_enabled();
    let use_g8_vstage = !use_g8_bcast
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && attn_v4_g8_vstage_c() == Some(tile_c);
    // W1b: partition-packed main (opt-in; exact same per-simdgroup dataflow)
    let use_pack4 = !use_g8_bcast
        && !use_g8_vstage
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && tile_c == 64
        && attn_v4_pack() == 4;
    let pipeline_name = if use_pack4 {
        "kernel_attn_decode_v4_g8_t4_c64_pack4_f32"
    } else if use_g8_bcast {
        match group_tile {
            2 => "kernel_attn_decode_v4_g8_t2_c64_bcast_f32",
            4 => "kernel_attn_decode_v4_g8_t4_c64_bcast_f32",
            _ => unreachable!(),
        }
    } else if use_g8_vstage {
        match tile_c {
            16 => "kernel_attn_decode_v4_g8_t4_c16_vstage_f32",
            32 => "kernel_attn_decode_v4_g8_t4_c32_vstage_f32",
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!("vstage C={tile_c} unsupported; expected 16 or 32"),
                });
            }
        }
    } else if group_tile == group {
        match (k_cache.dtype, group, tile_c) {
            (GgmlType::F16, 4, 16) => "kernel_attn_decode_v4_g4_c16_f32",
            (GgmlType::F16, 4, 32) => "kernel_attn_decode_v4_g4_f32",
            (GgmlType::F16, 4, 64) => "kernel_attn_decode_v4_g4_c64_f32",
            (GgmlType::F16, 4, 128) => "kernel_attn_decode_v4_g4_c128_f32",
            (GgmlType::Q8_0, 6, 16) => "kernel_attn_decode_v4_q8_c16_f32",
            (GgmlType::Q8_0, 6, 32) => "kernel_attn_decode_v4_q8_f32",
            (GgmlType::Q8_0, 6, 64) => "kernel_attn_decode_v4_q8_c64_f32",
            (GgmlType::Q8_0, 6, 128) => "kernel_attn_decode_v4_q8_c128_f32",
            (GgmlType::F16, 6, 16) => "kernel_attn_decode_v4_c16_f32",
            (GgmlType::F16, 6, 32) => "kernel_attn_decode_v4_f32",
            (GgmlType::F16, 6, 64) => "kernel_attn_decode_v4_c64_f32",
            (GgmlType::F16, 6, 128) => "kernel_attn_decode_v4_c128_f32",
            (GgmlType::F16, 8, 16) => "kernel_attn_decode_v4_g8_c16_f32",
            (GgmlType::F16, 8, 32) => "kernel_attn_decode_v4_g8_f32",
            (GgmlType::F16, 8, 64) => "kernel_attn_decode_v4_g8_c64_f32",
            (GgmlType::F16, 8, 128) => "kernel_attn_decode_v4_g8_c128_f32",
            (GgmlType::Q8_0, 8, 16) => "kernel_attn_decode_v4_q8_g8_c16_f32",
            (GgmlType::Q8_0, 8, 32) => "kernel_attn_decode_v4_q8_g8_f32",
            (GgmlType::Q8_0, 8, 64) => "kernel_attn_decode_v4_q8_g8_c64_f32",
            (GgmlType::Q8_0, 8, 128) => "kernel_attn_decode_v4_q8_g8_c128_f32",
            (GgmlType::F16, 16, 16) => "kernel_attn_decode_v4_g16_c16_f32",
            (GgmlType::F16, 16, 32) => "kernel_attn_decode_v4_g16_f32",
            (GgmlType::F16, 16, 64) => "kernel_attn_decode_v4_g16_c64_f32",
            (GgmlType::F16, 16, 128) => "kernel_attn_decode_v4_g16_c128_f32",
            (GgmlType::Q8_0, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "Q8_0 KV main kernels currently support only group=6/group=8; got group={group}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "unsupported (group={group}, tile_c={tile_c}); tile_c must be 16/32/64/128"
                    ),
                });
            }
        }
    } else {
        match (k_cache.dtype, group, group_tile, tile_c) {
            (GgmlType::F16, 8, 4, 16) => "kernel_attn_decode_v4_g8_t4_c16_f32",
            (GgmlType::F16, 8, 4, 32) => "kernel_attn_decode_v4_g8_t4_f32",
            (GgmlType::F16, 8, 4, 64) => "kernel_attn_decode_v4_g8_t4_c64_f32",
            (GgmlType::F16, 8, 4, 128) => "kernel_attn_decode_v4_g8_t4_c128_f32",
            (GgmlType::F16, 8, 2, 16) => "kernel_attn_decode_v4_g8_t2_c16_f32",
            (GgmlType::F16, 8, 2, 32) => "kernel_attn_decode_v4_g8_t2_f32",
            (GgmlType::F16, 8, 2, 64) => "kernel_attn_decode_v4_g8_t2_c64_f32",
            (GgmlType::F16, 8, 2, 128) => "kernel_attn_decode_v4_g8_t2_c128_f32",
            (GgmlType::Q8_0, 8, 4, 16) => "kernel_attn_decode_v4_q8_g8_t4_c16_f32",
            (GgmlType::Q8_0, 8, 4, 32) => "kernel_attn_decode_v4_q8_g8_t4_f32",
            (GgmlType::Q8_0, 8, 4, 64) => "kernel_attn_decode_v4_q8_g8_t4_c64_f32",
            (GgmlType::Q8_0, 8, 4, 128) => "kernel_attn_decode_v4_q8_g8_t4_c128_f32",
            (GgmlType::Q8_0, 8, 2, 16) => "kernel_attn_decode_v4_q8_g8_t2_c16_f32",
            (GgmlType::Q8_0, 8, 2, 32) => "kernel_attn_decode_v4_q8_g8_t2_f32",
            (GgmlType::Q8_0, 8, 2, 64) => "kernel_attn_decode_v4_q8_g8_t2_c64_f32",
            (GgmlType::Q8_0, 8, 2, 128) => "kernel_attn_decode_v4_q8_g8_t2_c128_f32",
            (GgmlType::F16, 16, 8, 16) => "kernel_attn_decode_v4_g16_t8_c16_f32",
            (GgmlType::F16, 16, 8, 32) => "kernel_attn_decode_v4_g16_t8_f32",
            (GgmlType::F16, 16, 8, 64) => "kernel_attn_decode_v4_g16_t8_c64_f32",
            (GgmlType::F16, 16, 8, 128) => "kernel_attn_decode_v4_g16_t8_c128_f32",
            (GgmlType::F16, 16, 4, 16) => "kernel_attn_decode_v4_g16_t4_c16_f32",
            (GgmlType::F16, 16, 4, 32) => "kernel_attn_decode_v4_g16_t4_f32",
            (GgmlType::F16, 16, 4, 64) => "kernel_attn_decode_v4_g16_t4_c64_f32",
            (GgmlType::F16, 16, 4, 128) => "kernel_attn_decode_v4_g16_t4_c128_f32",
            (GgmlType::Q8_0, _, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "Q8_0 KV subgroup kernels currently support only group=8 tile2/tile4; got group={group}, group_tile={group_tile}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4",
                    detail: format!(
                        "unsupported (group={group}, group_tile={group_tile}, tile_c={tile_c})"
                    ),
                });
            }
        }
    };
    let pso_main = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);

    // Threadgroup memory:
    //   threadgroup(0) sq[group_tile * DK halves]
    //   threadgroup(1) ss[group_tile * C floats]
    // The vstage proof fuses both group8/tile4 simdgroups into one TG, so it
    // allocates both simdgroups' Q/score scratch plus a shared V tile.
    if use_g8_bcast {
        let sq_bytes = group_tile * DK * 2;
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else if use_pack4 {
        const PACK: usize = 4;
        // sq is SHARED across the packed partitions; ss is per-simdgroup.
        enc.set_threadgroup_memory(0, group_tile * DK * 2);
        enc.set_threadgroup_memory(1, PACK * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg.div_ceil(PACK),
            },
            MTLSize {
                width: 32 * PACK,
                height: 1,
                depth: 1,
            },
        );
    } else if use_g8_vstage {
        enc.set_threadgroup_memory(0, 2 * group_tile * DK * 2);
        enc.set_threadgroup_memory(1, 2 * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.set_threadgroup_memory(2, tile_c * head_dim * 2);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: 1,
                depth: nwg,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let sq_bytes = group_tile * DK * 2; // f16 = 2 bytes per element
        let ss_bytes = group_tile * tile_c * std::mem::size_of::<f32>();
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.set_threadgroup_memory(1, ss_bytes);

        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    // -------- Reduce kernel ----------------------------------------------
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let reduce_h2 = nwg >= 128;
    let red_pipeline_name = match (group, reduce_h2) {
        (4, false) => "kernel_attn_decode_v4_reduce_g4_f32",
        (6, false) => "kernel_attn_decode_v4_reduce_f32",
        (8, false) => "kernel_attn_decode_v4_reduce_g8_f32",
        (16, false) => "kernel_attn_decode_v4_reduce_g16_f32",
        (4, true) => "kernel_attn_decode_v4_reduce_h2_g4_f32",
        (6, true) => "kernel_attn_decode_v4_reduce_h2_g6_f32",
        (8, true) => "kernel_attn_decode_v4_reduce_h2_g8_f32",
        (16, true) => "kernel_attn_decode_v4_reduce_h2_g16_f32",
        _ => unreachable!(),
    };
    let pso_red = ctx.pipeline(red_pipeline_name)?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());

    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: if reduce_h2 { 2 } else { 1 },
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

/// Encode only the v4 main kernel into `enc`, writing partials but not the
/// final reduced output. Useful for profiling where the split-K main pass and
/// reduction need to be measured separately.
pub fn encode_attn_decode_v4_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
) -> Result<(), MetalError> {
    const DK: usize = 256;
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("head_dim={head_dim} but kernel hardcodes {DK}"),
        });
    }
    if !matches!(group, 4 | 6 | 8 | 16) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("group={group} unsupported; expected one of {{4, 6, 8, 16}}"),
        });
    }
    if k_cache.dtype != v_cache.dtype || !matches!(k_cache.dtype, GgmlType::F16 | GgmlType::Q8_0) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!(
                "k/v expected matching F16 or Q8_0 dtypes, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    if q.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("q expected {want_q} elements"),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let group_tile = attn_v4_choose_group_tile(n_pos, group);
    if group_tile == 0 || !group.is_multiple_of(group_tile) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("group_tile={group_tile} must divide group={group}"),
        });
    }
    if group_tile != group
        && !(group == 16 && matches!(group_tile, 4 | 8)
            || group == 8 && matches!(group_tile, 2 | 4))
    {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!("unsupported subgroup: group={group} group_tile={group_tile}"),
        });
    }
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let kv_stride = n_kv_heads * head_dim;
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        scale: f32,
    }
    let use_g8_bcast = k_cache.dtype == GgmlType::F16
        && group == 8
        && matches!(group_tile, 2 | 4)
        && tile_c == 64
        && attn_v4_g8_bcast_enabled();
    let use_g8_vstage = !use_g8_bcast
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && attn_v4_g8_vstage_c() == Some(tile_c);
    // W1b: partition-packed main (opt-in; exact same per-simdgroup dataflow)
    let use_pack4 = !use_g8_bcast
        && !use_g8_vstage
        && k_cache.dtype == GgmlType::F16
        && group == 8
        && group_tile == 4
        && tile_c == 64
        && attn_v4_pack() == 4;
    let pipeline_name = if use_pack4 {
        "kernel_attn_decode_v4_g8_t4_c64_pack4_f32"
    } else if use_g8_bcast {
        match group_tile {
            2 => "kernel_attn_decode_v4_g8_t2_c64_bcast_f32",
            4 => "kernel_attn_decode_v4_g8_t4_c64_bcast_f32",
            _ => unreachable!(),
        }
    } else if use_g8_vstage {
        match tile_c {
            16 => "kernel_attn_decode_v4_g8_t4_c16_vstage_f32",
            32 => "kernel_attn_decode_v4_g8_t4_c32_vstage_f32",
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!("vstage C={tile_c} unsupported; expected 16 or 32"),
                });
            }
        }
    } else if group_tile == group {
        match (k_cache.dtype, group, tile_c) {
            (GgmlType::F16, 4, 16) => "kernel_attn_decode_v4_g4_c16_f32",
            (GgmlType::F16, 4, 32) => "kernel_attn_decode_v4_g4_f32",
            (GgmlType::F16, 4, 64) => "kernel_attn_decode_v4_g4_c64_f32",
            (GgmlType::F16, 4, 128) => "kernel_attn_decode_v4_g4_c128_f32",
            (GgmlType::Q8_0, 6, 16) => "kernel_attn_decode_v4_q8_c16_f32",
            (GgmlType::Q8_0, 6, 32) => "kernel_attn_decode_v4_q8_f32",
            (GgmlType::Q8_0, 6, 64) => "kernel_attn_decode_v4_q8_c64_f32",
            (GgmlType::Q8_0, 6, 128) => "kernel_attn_decode_v4_q8_c128_f32",
            (GgmlType::F16, 6, 16) => "kernel_attn_decode_v4_c16_f32",
            (GgmlType::F16, 6, 32) => "kernel_attn_decode_v4_f32",
            (GgmlType::F16, 6, 64) => "kernel_attn_decode_v4_c64_f32",
            (GgmlType::F16, 6, 128) => "kernel_attn_decode_v4_c128_f32",
            (GgmlType::F16, 8, 16) => "kernel_attn_decode_v4_g8_c16_f32",
            (GgmlType::F16, 8, 32) => "kernel_attn_decode_v4_g8_f32",
            (GgmlType::F16, 8, 64) => "kernel_attn_decode_v4_g8_c64_f32",
            (GgmlType::F16, 8, 128) => "kernel_attn_decode_v4_g8_c128_f32",
            (GgmlType::Q8_0, 8, 16) => "kernel_attn_decode_v4_q8_g8_c16_f32",
            (GgmlType::Q8_0, 8, 32) => "kernel_attn_decode_v4_q8_g8_f32",
            (GgmlType::Q8_0, 8, 64) => "kernel_attn_decode_v4_q8_g8_c64_f32",
            (GgmlType::Q8_0, 8, 128) => "kernel_attn_decode_v4_q8_g8_c128_f32",
            (GgmlType::F16, 16, 16) => "kernel_attn_decode_v4_g16_c16_f32",
            (GgmlType::F16, 16, 32) => "kernel_attn_decode_v4_g16_f32",
            (GgmlType::F16, 16, 64) => "kernel_attn_decode_v4_g16_c64_f32",
            (GgmlType::F16, 16, 128) => "kernel_attn_decode_v4_g16_c128_f32",
            (GgmlType::Q8_0, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "Q8_0 KV main kernels currently support only group=6/group=8; got group={group}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "unsupported (group={group}, tile_c={tile_c}); tile_c must be 16/32/64/128"
                    ),
                });
            }
        }
    } else {
        match (k_cache.dtype, group, group_tile, tile_c) {
            (GgmlType::F16, 8, 4, 16) => "kernel_attn_decode_v4_g8_t4_c16_f32",
            (GgmlType::F16, 8, 4, 32) => "kernel_attn_decode_v4_g8_t4_f32",
            (GgmlType::F16, 8, 4, 64) => "kernel_attn_decode_v4_g8_t4_c64_f32",
            (GgmlType::F16, 8, 4, 128) => "kernel_attn_decode_v4_g8_t4_c128_f32",
            (GgmlType::F16, 8, 2, 16) => "kernel_attn_decode_v4_g8_t2_c16_f32",
            (GgmlType::F16, 8, 2, 32) => "kernel_attn_decode_v4_g8_t2_f32",
            (GgmlType::F16, 8, 2, 64) => "kernel_attn_decode_v4_g8_t2_c64_f32",
            (GgmlType::F16, 8, 2, 128) => "kernel_attn_decode_v4_g8_t2_c128_f32",
            (GgmlType::Q8_0, 8, 4, 16) => "kernel_attn_decode_v4_q8_g8_t4_c16_f32",
            (GgmlType::Q8_0, 8, 4, 32) => "kernel_attn_decode_v4_q8_g8_t4_f32",
            (GgmlType::Q8_0, 8, 4, 64) => "kernel_attn_decode_v4_q8_g8_t4_c64_f32",
            (GgmlType::Q8_0, 8, 4, 128) => "kernel_attn_decode_v4_q8_g8_t4_c128_f32",
            (GgmlType::Q8_0, 8, 2, 16) => "kernel_attn_decode_v4_q8_g8_t2_c16_f32",
            (GgmlType::Q8_0, 8, 2, 32) => "kernel_attn_decode_v4_q8_g8_t2_f32",
            (GgmlType::Q8_0, 8, 2, 64) => "kernel_attn_decode_v4_q8_g8_t2_c64_f32",
            (GgmlType::Q8_0, 8, 2, 128) => "kernel_attn_decode_v4_q8_g8_t2_c128_f32",
            (GgmlType::F16, 16, 8, 16) => "kernel_attn_decode_v4_g16_t8_c16_f32",
            (GgmlType::F16, 16, 8, 32) => "kernel_attn_decode_v4_g16_t8_f32",
            (GgmlType::F16, 16, 8, 64) => "kernel_attn_decode_v4_g16_t8_c64_f32",
            (GgmlType::F16, 16, 8, 128) => "kernel_attn_decode_v4_g16_t8_c128_f32",
            (GgmlType::F16, 16, 4, 16) => "kernel_attn_decode_v4_g16_t4_c16_f32",
            (GgmlType::F16, 16, 4, 32) => "kernel_attn_decode_v4_g16_t4_f32",
            (GgmlType::F16, 16, 4, 64) => "kernel_attn_decode_v4_g16_t4_c64_f32",
            (GgmlType::F16, 16, 4, 128) => "kernel_attn_decode_v4_g16_t4_c128_f32",
            (GgmlType::Q8_0, _, _, _) => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "Q8_0 KV subgroup kernels currently support only group=8 tile2/tile4; got group={group}, group_tile={group_tile}, tile_c={tile_c}"
                    ),
                });
            }
            _ => {
                return Err(MetalError::BadShape {
                    kernel: "attn_decode_v4_main",
                    detail: format!(
                        "unsupported (group={group}, group_tile={group_tile}, tile_c={tile_c})"
                    ),
                });
            }
        }
    };
    let pso_main = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    if use_g8_bcast {
        let sq_bytes = group_tile * DK * 2;
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else if use_pack4 {
        const PACK: usize = 4;
        // sq is SHARED across the packed partitions; ss is per-simdgroup.
        enc.set_threadgroup_memory(0, group_tile * DK * 2);
        enc.set_threadgroup_memory(1, PACK * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg.div_ceil(PACK),
            },
            MTLSize {
                width: 32 * PACK,
                height: 1,
                depth: 1,
            },
        );
    } else if use_g8_vstage {
        enc.set_threadgroup_memory(0, 2 * group_tile * DK * 2);
        enc.set_threadgroup_memory(1, 2 * group_tile * tile_c * std::mem::size_of::<f32>());
        enc.set_threadgroup_memory(2, tile_c * head_dim * 2);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: 1,
                depth: nwg,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let sq_bytes = group_tile * DK * 2;
        let ss_bytes = group_tile * tile_c * std::mem::size_of::<f32>();
        enc.set_threadgroup_memory(0, sq_bytes);
        enc.set_threadgroup_memory(1, ss_bytes);
        enc.dispatch(
            MTLSize {
                width: n_kv_heads,
                height: group / group_tile,
                depth: nwg,
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

pub fn encode_attn_stage_floor_g16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    checksum: &MetalTensor,
    n_pos: usize,
    n_kv_heads: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const HD: usize = 256;
    const ROW_BYTES: usize = 288;
    let kernel = "attn_stage_floor_g16";
    let expected_bytes = n_pos
        .checked_mul(n_kv_heads)
        .and_then(|rows| rows.checked_mul(ROW_BYTES))
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "compressed KV byte count overflow".into(),
        })?;
    let expected_output = n_kv_heads
        .checked_mul(nwg)
        .and_then(|groups| groups.checked_mul(HD))
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: "checksum element count overflow".into(),
        })?;
    if n_pos != 32768
        || n_kv_heads != 2
        || nwg != 256
        || k_cache.dtype != GgmlType::F16
        || v_cache.dtype != GgmlType::F16
        || k_cache.n_bytes() as usize != expected_bytes
        || v_cache.n_bytes() as usize != expected_bytes
        || checksum.dtype != GgmlType::F32
        || checksum.n_elements() as usize != expected_output
    {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "requires n_pos=32768 n_kv=2 nwg=256 and exact buffers; got \
                 n_pos={n_pos} n_kv={n_kv_heads} nwg={nwg} bytes={}/{} out={}",
                k_cache.n_bytes(),
                v_cache.n_bytes(),
                checksum.n_elements()
            ),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_pos: u32,
        n_kv_heads: u32,
        n_partitions: u32,
        rows_per_partition: u32,
    }
    enc.set_pipeline(&ctx.pipeline("kernel_attn_stage_floor_g16")?);
    enc.set_bytes(
        0,
        &Args {
            n_pos: n_pos as u32,
            n_kv_heads: n_kv_heads as u32,
            n_partitions: nwg as u32,
            rows_per_partition: n_pos.div_ceil(nwg) as u32,
        },
    );
    enc.set_tensor(1, k_cache);
    enc.set_tensor(2, v_cache);
    enc.set_tensor(3, checksum);
    enc.set_threadgroup_memory(0, 32 * HD * std::mem::size_of::<u16>());
    enc.dispatch(
        MTLSize {
            width: n_kv_heads,
            height: 1,
            depth: nwg,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Synthetic head-major F16 KV sidecar for v4 long-context attention proofing.
///
/// This intentionally supports only the current long MoE subgroup shapes:
/// group8/tile2/C64 plus group16/tile4/C64/C128. `k_cache` and `v_cache` are
/// laid out as `[kv_head, n_pos, head_dim]` with exactly `n_pos` rows per head.
pub fn encode_attn_decode_v4_main_only_f32_head_major(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    nwg: usize,
    tile_c: usize,
) -> Result<(), MetalError> {
    const DK: usize = 256;
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK || !matches!((group, tile_c), (8, 64) | (16, 64 | 128)) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!(
                "expected head_dim=256 and group/tile_c in {{8/64,16/64,16/128}}; got head_dim={head_dim} tile_c={tile_c} group={group}"
            ),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!(
                "expected F16 K/V, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    let want_kv = (n_kv_heads * n_pos * head_dim) as u64;
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if q.n_elements() != want_q
        || k_cache.n_elements() < want_kv
        || v_cache.n_elements() < want_kv
        || o_partial.n_elements() < want_o_partial
        || ml_partial.n_elements() < want_ml_partial
    {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!(
                "shape mismatch q={} want {want_q}, k={} v={} want >= {want_kv}, o={} want >= {want_o_partial}, ml={} want >= {want_ml_partial}",
                q.n_elements(),
                k_cache.n_elements(),
                v_cache.n_elements(),
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_main_hm",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }

    let group_tile = attn_v4_choose_group_tile(n_pos, group);
    let pipeline_name = match (group, group_tile, tile_c) {
        (8, 2, 64) => "kernel_attn_decode_v4_g8_t2_c64_hm_f32",
        (16, 4, 64) => "kernel_attn_decode_v4_g16_t4_c64_hm_f32",
        (16, 4, 128) => "kernel_attn_decode_v4_g16_t4_c128_hm_f32",
        _ => {
            return Err(MetalError::BadShape {
                kernel: "attn_decode_v4_main_hm",
                detail: format!(
                    "unsupported group/group_tile/tile_c for HM proof: {group}/{group_tile}/{tile_c}"
                ),
            });
        }
    };

    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        scale: f32,
    }

    let pso_main = ctx.pipeline(pipeline_name)?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_pos: n_pos as u32,
            kv_stride: head_dim as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            scale,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, group_tile * DK * 2);
    enc.set_threadgroup_memory(1, group_tile * tile_c * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_kv_heads,
            height: group / group_tile,
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encode only the v4 reduce kernel into `enc`, consuming previously-written
/// partials and producing the final attention output.
pub fn encode_attn_decode_v4_reduce_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const DK: usize = 256;
    if !n_q_heads.is_multiple_of(n_kv_heads) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("n_q_heads={n_q_heads} not multiple of n_kv_heads={n_kv_heads}"),
        });
    }
    let group = n_q_heads / n_kv_heads;
    if head_dim != DK {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("head_dim={head_dim} but kernel hardcodes {DK}"),
        });
    }
    if !matches!(group, 4 | 6 | 8 | 16) {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("group={group} unsupported; expected one of {{4, 6, 8, 16}}"),
        });
    }
    let want_q = (n_q_heads * head_dim) as u64;
    if out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("out expected {want_q} elements"),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_o_partial = (n_kv_heads * nwg * group * head_dim) as u64;
    let want_ml_partial = (n_kv_heads * nwg * group * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_decode_v4_reduce",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let reduce_h2 = nwg >= 128;
    let red_pipeline_name = match (group, reduce_h2) {
        (4, false) => "kernel_attn_decode_v4_reduce_g4_f32",
        (6, false) => "kernel_attn_decode_v4_reduce_f32",
        (8, false) => "kernel_attn_decode_v4_reduce_g8_f32",
        (16, false) => "kernel_attn_decode_v4_reduce_g16_f32",
        (4, true) => "kernel_attn_decode_v4_reduce_h2_g4_f32",
        (6, true) => "kernel_attn_decode_v4_reduce_h2_g6_f32",
        (8, true) => "kernel_attn_decode_v4_reduce_h2_g8_f32",
        (16, true) => "kernel_attn_decode_v4_reduce_h2_g16_f32",
        _ => unreachable!(),
    };
    let pso_red = ctx.pipeline(red_pipeline_name)?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_q_heads,
            height: if reduce_h2 { 2 } else { 1 },
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

/// Packed two-query attention for dense Qwen's 24Q/4KV group-6 shape.
/// Both causal rows share each K/V load while retaining independent online
/// softmax state and outputs.
pub fn encode_attn_prefill_v4_g6_q2_c32_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
    q_f32: bool,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 24;
    const N_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 6;
    const QT: usize = 2;
    const TILE_C: usize = 32;

    if enc.concurrent {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: "dependent main/reduce dispatches require a serial encoder".into(),
        });
    }
    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: "n_rows must be > 0".into(),
        });
    }
    if q_rows.dtype != GgmlType::F32
        || o_partial.dtype != GgmlType::F32
        || ml_partial.dtype != GgmlType::F32
        || out.dtype != GgmlType::F32
        || !o_partial.is_writable()
        || !ml_partial.is_writable()
        || !out.is_writable()
    {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!(
                "expected F32 q and writable F32 partials/out, got {:?}/{:?}/{:?}/{:?} writable={}/{}/{}",
                q_rows.dtype,
                o_partial.dtype,
                ml_partial.dtype,
                out.dtype,
                o_partial.is_writable(),
                ml_partial.is_writable(),
                out.is_writable(),
            ),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    let range_fits = |tensor: &MetalTensor, element_bytes: u64| {
        tensor.offset.is_multiple_of(element_bytes)
            && tensor
                .n_elements()
                .checked_mul(element_bytes)
                .and_then(|bytes| tensor.offset.checked_add(bytes))
                .is_some_and(|end| end <= tensor.buffer.length() as u64)
    };
    for (name, tensor, element_bytes) in [
        ("q", q_rows, 4),
        ("k_cache", k_cache, 2),
        ("v_cache", v_cache, 2),
        ("o_partial", o_partial, 4),
        ("ml_partial", ml_partial, 4),
        ("out", out, 4),
    ] {
        if !range_fits(tensor, element_bytes) {
            return Err(MetalError::BadShape {
                kernel: "attn_prefill_v4_g6_q2_c32",
                detail: format!("{name} has an unaligned or out-of-buffer byte range"),
            });
        }
    }
    let nwg_max = crate::metal_forward::ATTN_V4_MAX_NWG;
    if nwg == 0 || nwg > nwg_max {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!("nwg={nwg} out of range [1, {nwg_max}]"),
        });
    }
    let checked_product = |factors: &[usize], label: &str| -> Result<u64, MetalError> {
        let elements = factors
            .iter()
            .try_fold(1usize, |product, factor| product.checked_mul(*factor));
        elements
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| MetalError::BadShape {
                kernel: "attn_prefill_v4_g6_q2_c32",
                detail: format!("{label} element count overflow"),
            })
    };
    let want_q = checked_product(&[n_rows, N_Q_HEADS, HEAD_DIM], "q/out")?;
    if q_rows.n_elements() != want_q || out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!(
                "q/out expected {want_q} elements, got {}/{}",
                q_rows.n_elements(),
                out.n_elements()
            ),
        });
    }
    let n_pos = base_pos
        .checked_add(n_rows)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: "base_pos + n_rows overflow".into(),
        })?;
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let want_kv = checked_product(&[n_pos, kv_stride], "KV")?;
    if k_cache.n_elements() < want_kv || v_cache.n_elements() < want_kv {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!(
                "KV cache too small: need {want_kv}, got {}/{}",
                k_cache.n_elements(),
                v_cache.n_elements()
            ),
        });
    }
    let want_o_partial = checked_product(
        &[n_rows, N_KV_HEADS, nwg, GROUP, HEAD_DIM],
        "output partial",
    )?;
    let want_ml_partial = checked_product(&[n_rows, N_KV_HEADS, nwg, GROUP, 2], "softmax partial")?;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let rows_per_partition = n_pos.div_ceil(nwg);
    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| MetalError::BadShape {
        kernel: "attn_prefill_v4_g6_q2_c32",
        detail: format!("n_rows={n_rows} does not fit u32"),
    })?;
    let n_pos_u32 = u32::try_from(n_pos).map_err(|_| MetalError::BadShape {
        kernel: "attn_prefill_v4_g6_q2_c32",
        detail: format!("n_pos={n_pos} does not fit u32"),
    })?;
    let rows_per_partition_u32 =
        u32::try_from(rows_per_partition).map_err(|_| MetalError::BadShape {
            kernel: "attn_prefill_v4_g6_q2_c32",
            detail: format!("rows_per_partition={rows_per_partition} does not fit u32"),
        })?;
    let base_pos_u32 = u32::try_from(base_pos).map_err(|_| MetalError::BadShape {
        kernel: "attn_prefill_v4_g6_q2_c32",
        detail: format!("base_pos={base_pos} does not fit u32"),
    })?;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        q_f32: u32,
        scale: f32,
    }
    let main = ctx.pipeline("kernel_attn_prefill_v4_g6_q2_c32_f32")?;
    enc.set_pipeline(&main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows_u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos_u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition_u32,
            base_pos: base_pos_u32,
            q_f32: u32::from(q_f32),
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    // sq shmem: F16 (QT*GROUP*HEAD_DIM*2 bytes) or F32 (x2) depending on
    // q_f32; size for the F32 case unconditionally (12KB).
    enc.set_threadgroup_memory(0, QT * GROUP * HEAD_DIM * 4);
    enc.set_threadgroup_memory(1, QT * GROUP * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let reduce = ctx.pipeline("kernel_attn_prefill_v4_reduce_rows_g6_f32")?;
    enc.set_pipeline(&reduce);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_rows: n_rows_u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_Q_HEADS,
            height: n_rows,
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

pub fn encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 8;
    const GROUP_TILE: usize = 2;
    const QT: usize = 2;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g8_t2_q2_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 8;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64_reduce",
            detail: "n_rows must be > 0".into(),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64_reduce",
            detail: format!("out expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q2_c64_reduce",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let pso_red = ctx.pipeline("kernel_attn_prefill_v4_reduce_rows_g8_f32")?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_Q_HEADS,
            height: n_rows,
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

/// Prompt-native packed attention microproof for the A3B attention shape.
///
/// This is intentionally narrow and only meant to answer whether batching
/// multiple consecutive prompt queries against the same K/V tiles can beat the
/// current repeated decode-shaped attention body.
pub fn encode_attn_prefill_v4_g8_t2_q2_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g8_t2_q2_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

pub fn encode_attn_prefill_v4_g8_t2_q4_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 8;
    const GROUP_TILE: usize = 2;
    const QT: usize = 4;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g8_t2_q4_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g8_t2_q4_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g8_t2_q4_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g8_t2_q4_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g8_t2_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 16;
    const GROUP_TILE: usize = 4;
    const QT: usize = 2;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g16_t4_q2_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 16;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64_reduce",
            detail: "n_rows must be > 0".into(),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if out.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64_reduce",
            detail: format!("out expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q2_c64_reduce",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_partitions: u32,
    }
    let pso_red = ctx.pipeline("kernel_attn_prefill_v4_reduce_rows_g16_f32")?;
    enc.set_pipeline(&pso_red);
    enc.set_bytes(
        0,
        &ReduceArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_partitions: nwg as u32,
        },
    );
    enc.set_tensor(1, o_partial);
    enc.set_tensor(2, ml_partial);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(1, nwg * std::mem::size_of::<f32>());
    enc.set_threadgroup_memory(2, nwg * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_Q_HEADS,
            height: n_rows,
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

pub fn encode_attn_prefill_v4_g16_t4_q2_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g16_t4_q2_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q4_c64_main_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    const N_Q_HEADS: usize = 32;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 16;
    const GROUP_TILE: usize = 4;
    const QT: usize = 4;
    const TILE_C: usize = 64;

    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: "n_rows must be > 0".into(),
        });
    }
    if k_cache.dtype != GgmlType::F16 || v_cache.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!(
                "expected F16 KV cache, got {:?}/{:?}",
                k_cache.dtype, v_cache.dtype
            ),
        });
    }
    if nwg == 0 || nwg > ATTN_V4_NWG_MAX {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!("nwg={nwg} out of range [1, {ATTN_V4_NWG_MAX}]"),
        });
    }
    let want_q = (n_rows * N_Q_HEADS * HEAD_DIM) as u64;
    if q_rows.n_elements() != want_q {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!("q expected {want_q} elements"),
        });
    }
    let want_o_partial = (n_rows * N_KV_HEADS * nwg * GROUP * HEAD_DIM) as u64;
    let want_ml_partial = (n_rows * N_KV_HEADS * nwg * GROUP * 2) as u64;
    if o_partial.n_elements() < want_o_partial || ml_partial.n_elements() < want_ml_partial {
        return Err(MetalError::BadShape {
            kernel: "attn_prefill_v4_g16_t4_q4_c64",
            detail: format!(
                "partials too small: o have {} need >= {want_o_partial}, ml have {} need >= {want_ml_partial}",
                o_partial.n_elements(),
                ml_partial.n_elements()
            ),
        });
    }

    let n_pos = base_pos + n_rows;
    let rows_per_partition = n_pos.div_ceil(nwg.max(1));
    let kv_stride = N_KV_HEADS * HEAD_DIM;
    let scale = (1.0f32 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MainArgs {
        n_rows: u32,
        n_q_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        n_pos: u32,
        kv_stride: u32,
        n_partitions: u32,
        rows_per_partition: u32,
        base_pos: u32,
        scale: f32,
    }
    let pso_main = ctx.pipeline("kernel_attn_prefill_v4_g16_t4_q4_c64_f32")?;
    enc.set_pipeline(&pso_main);
    enc.set_bytes(
        0,
        &MainArgs {
            n_rows: n_rows as u32,
            n_q_heads: N_Q_HEADS as u32,
            n_kv_heads: N_KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            n_pos: n_pos as u32,
            kv_stride: kv_stride as u32,
            n_partitions: nwg as u32,
            rows_per_partition: rows_per_partition as u32,
            base_pos: base_pos as u32,
            scale,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, v_cache);
    enc.set_tensor(4, o_partial);
    enc.set_tensor(5, ml_partial);
    enc.set_threadgroup_memory(0, QT * GROUP_TILE * HEAD_DIM * 2);
    enc.set_threadgroup_memory(1, QT * GROUP_TILE * TILE_C * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: N_KV_HEADS,
            height: n_rows.div_ceil(QT) * (GROUP / GROUP_TILE),
            depth: nwg,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    Ok(())
}

pub fn encode_attn_prefill_v4_g16_t4_q4_c64_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    v_cache: &MetalTensor,
    o_partial: &MetalTensor,
    ml_partial: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    nwg: usize,
) -> Result<(), MetalError> {
    encode_attn_prefill_v4_g16_t4_q4_c64_main_only_f32(
        ctx, enc, q_rows, k_cache, v_cache, o_partial, ml_partial, n_rows, base_pos, nwg,
    )?;
    encode_attn_prefill_v4_g16_t4_q2_c64_reduce_only_f32(
        ctx, enc, o_partial, ml_partial, out, n_rows, nwg,
    )?;
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AttnMatrixArgs {
    pub(crate) n_rows: u32,
    pub(crate) n_pos: u32,
    pub(crate) base_pos: u32,
    pub(crate) kv_stride: u32,
    pub(crate) vt_stride: u32,
    pub(crate) n_q_heads: u32,
    pub(crate) n_kv_heads: u32,
    pub(crate) group: u32,
    pub(crate) head_dim: u32,
    pub(crate) scale: f32,
    pub(crate) causal_skip: u32,
}

pub(crate) fn validate_attn_matrix_common(
    kernel: &'static str,
    n_rows: usize,
    n_pos: usize,
    base_pos: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 {
        return Err(MetalError::BadShape {
            kernel,
            detail: "n_rows must be > 0".into(),
        });
    }
    if n_pos < base_pos + n_rows {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!("n_pos={n_pos} < base_pos+n_rows={}", base_pos + n_rows),
        });
    }
    if n_kv_heads == 0 || group == 0 || n_q_heads != n_kv_heads * group || head_dim != 256 {
        return Err(MetalError::BadShape {
            kernel,
            detail: format!(
                "unsupported matrix-attn shape n_q={n_q_heads} n_kv={n_kv_heads} group={group} head_dim={head_dim}"
            ),
        });
    }
    Ok(())
}

crate::env_flag!(
    default_on attn_matrix_vt_compact_dispatch_env_enabled,
    "QWEN_ATTN_MATRIX_VT_COMPACT_DISPATCH"
);

pub(crate) fn attn_matrix_vt_compact_dispatch_enabled() -> Result<bool, MetalError> {
    let current = std::thread::current().id();
    let active_owner = *attn_matrix_vt_override_owner().lock();
    if let Some(owner) = active_owner.as_ref()
        && owner != &current
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_vt_override",
            detail: format!("dispatch on {current:?} while override is owned by {owner:?}"),
        });
    }
    let scoped = ATTN_MATRIX_VT_COMPACT_DISPATCH_OVERRIDE.with(Cell::get);
    if active_owner.is_some() && scoped.is_none() {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_vt_override",
            detail: "active override owner has no thread-local treatment".into(),
        });
    }
    Ok(scoped.unwrap_or_else(attn_matrix_vt_compact_dispatch_env_enabled))
}

pub(crate) const ATTN_MATRIX_VT_THREADS: usize = 256;

pub(crate) fn record_attn_matrix_vt_dispatch(
    base_pos: usize,
    n_rows: usize,
    n_pos: usize,
    total: usize,
    threadgroups: usize,
    compact: bool,
) -> Result<(), MetalError> {
    let current = std::thread::current().id();
    let active_owner = *attn_matrix_vt_capture_owner().lock();
    match active_owner.as_ref() {
        None => {
            if ATTN_MATRIX_VT_CAPTURE_ACTIVE.with(Cell::get) {
                return Err(MetalError::BadShape {
                    kernel: "attn_matrix_vt_capture",
                    detail: "thread-local capture is active without a process owner".into(),
                });
            }
            return Ok(());
        }
        Some(owner) if owner != &current => {
            return Err(MetalError::BadShape {
                kernel: "attn_matrix_vt_capture",
                detail: format!("dispatch on {current:?} while capture is owned by {owner:?}"),
            });
        }
        Some(_) => {}
    }
    if !ATTN_MATRIX_VT_CAPTURE_ACTIVE.with(Cell::get) {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_vt_capture",
            detail: "capture owner has no thread-local accumulator".into(),
        });
    }
    let to_u64 = |name: &'static str, value: usize| {
        u64::try_from(value).map_err(|_| MetalError::BadShape {
            kernel: "attn_matrix_vt_capture",
            detail: format!("{name}={value} does not fit u64"),
        })
    };
    let row_sum = to_u64("n_rows", n_rows)?;
    let element_sum = to_u64("total", total)?;
    let threadgroup_sum = to_u64("threadgroups", threadgroups)?;
    let base_pos_sum = to_u64("base_pos", base_pos)?;
    let n_pos_sum = to_u64("n_pos", n_pos)?;
    ATTN_MATRIX_VT_CAPTURE_STATS.with(|slot| {
        let mut stats = slot.get();
        let add = |name: &'static str, lhs: u64, rhs: u64| {
            lhs.checked_add(rhs).ok_or_else(|| MetalError::BadShape {
                kernel: "attn_matrix_vt_capture",
                detail: format!("{name} counter overflow"),
            })
        };
        stats.calls = add("calls", stats.calls, 1)?;
        stats.row_sum = add("row_sum", stats.row_sum, row_sum)?;
        stats.element_sum = add("element_sum", stats.element_sum, element_sum)?;
        stats.threadgroup_sum = add("threadgroup_sum", stats.threadgroup_sum, threadgroup_sum)?;
        stats.base_pos_sum = add("base_pos_sum", stats.base_pos_sum, base_pos_sum)?;
        stats.n_pos_sum = add("n_pos_sum", stats.n_pos_sum, n_pos_sum)?;
        if compact {
            stats.compact_calls = add("compact_calls", stats.compact_calls, 1)?;
        } else {
            stats.legacy_calls = add("legacy_calls", stats.legacy_calls, 1)?;
        }
        slot.set(stats);
        Ok(())
    })
}

pub(crate) fn attn_matrix_vt_threadgroups(
    total: usize,
    compact: bool,
) -> Result<usize, MetalError> {
    if total == 0 || total > u32::MAX as usize {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("total threads {total} do not fit nonzero shader uint range"),
        });
    }
    let groups = if compact {
        total.div_ceil(ATTN_MATRIX_VT_THREADS)
    } else {
        total
    };
    let padded_threads =
        groups
            .checked_mul(ATTN_MATRIX_VT_THREADS)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "attn_matrix_transpose_v",
                detail: format!("threadgroups {groups} * {ATTN_MATRIX_VT_THREADS} overflows usize"),
            })?;
    if padded_threads - 1 > u32::MAX as usize {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "maximum global thread id {} does not fit shader uint",
                padded_threads - 1
            ),
        });
    }
    Ok(groups)
}

pub(crate) fn encode_attn_matrix_transpose_v_f16_mode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    v_cache: &MetalTensor,
    v_t: &MetalTensor,
    base_pos: usize,
    n_rows: usize,
    n_pos: usize,
    kv_stride: usize,
    vt_stride: usize,
    n_kv_heads: usize,
    head_dim: usize,
    compact_dispatch: bool,
) -> Result<(), MetalError> {
    if v_cache.dtype != GgmlType::F16 || v_t.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "expected F16 tensors, got {:?}/{:?}",
                v_cache.dtype, v_t.dtype
            ),
        });
    }
    let span_end = base_pos
        .checked_add(n_rows)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("base_pos={base_pos} + n_rows={n_rows} overflows"),
        })?;
    if n_rows == 0 || span_end > n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "invalid V transpose span base_pos={base_pos} n_rows={n_rows} n_pos={n_pos}"
            ),
        });
    }
    if vt_stride < n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("vt_stride={vt_stride} < n_pos={n_pos}"),
        });
    }
    let kv_dim = n_kv_heads
        .checked_mul(head_dim)
        .filter(|&value| value > 0)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("invalid n_kv_heads={n_kv_heads} * head_dim={head_dim}"),
        })?;
    if kv_stride < kv_dim {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("kv_stride={kv_stride} < kv_dim={kv_dim}"),
        });
    }
    let last_component = kv_dim - 1;
    let last_pos = span_end - 1;
    let want_vt = last_component
        .checked_mul(vt_stride)
        .and_then(|value| value.checked_add(last_pos))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "V_T max index overflows for kv_dim={kv_dim} vt_stride={vt_stride} span_end={span_end}"
            ),
        })?;
    if v_t.n_elements() < want_vt as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!("v_t has {} elements, need >= {want_vt}", v_t.n_elements()),
        });
    }
    let want_cache = last_pos
        .checked_mul(kv_stride)
        .and_then(|value| value.checked_add(last_component))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "V cache max index overflows for kv_stride={kv_stride} kv_dim={kv_dim} span_end={span_end}"
            ),
        })?;
    if v_cache.n_elements() < want_cache as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "v_cache has {} elements, need >= {want_cache}",
                v_cache.n_elements()
            ),
        });
    }
    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("n_rows={n_rows} does not fit shader uint"),
    })?;
    let n_pos_u32 = u32::try_from(n_pos).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("n_pos={n_pos} does not fit shader uint"),
    })?;
    let base_pos_u32 = u32::try_from(base_pos).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("base_pos={base_pos} does not fit shader uint"),
    })?;
    let kv_stride_u32 = u32::try_from(kv_stride).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("kv_stride={kv_stride} does not fit shader uint"),
    })?;
    let vt_stride_u32 = u32::try_from(vt_stride).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("vt_stride={vt_stride} does not fit shader uint"),
    })?;
    let n_kv_heads_u32 = u32::try_from(n_kv_heads).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("n_kv_heads={n_kv_heads} does not fit shader uint"),
    })?;
    let head_dim_u32 = u32::try_from(head_dim).map_err(|_| MetalError::BadShape {
        kernel: "attn_matrix_transpose_v",
        detail: format!("head_dim={head_dim} does not fit shader uint"),
    })?;
    let total = kv_dim
        .checked_mul(n_rows)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "attn_matrix_transpose_v",
            detail: format!(
                "n_kv_heads={n_kv_heads} * head_dim={head_dim} * n_rows={n_rows} overflows"
            ),
        })?;
    let threadgroups = attn_matrix_vt_threadgroups(total, compact_dispatch)?;
    let pso = ctx.pipeline("kernel_attn_matrix_transpose_v_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows_u32,
            n_pos: n_pos_u32,
            base_pos: base_pos_u32,
            kv_stride: kv_stride_u32,
            vt_stride: vt_stride_u32,
            n_q_heads: 0,
            n_kv_heads: n_kv_heads_u32,
            group: 0,
            head_dim: head_dim_u32,
            scale: 0.0,
            causal_skip: 0,
        },
    );
    enc.set_tensor(1, v_cache);
    enc.set_tensor(2, v_t);
    record_attn_matrix_vt_dispatch(
        base_pos,
        n_rows,
        n_pos,
        total,
        threadgroups,
        compact_dispatch,
    )?;
    enc.dispatch(
        MTLSize {
            width: threadgroups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ATTN_MATRIX_VT_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_attn_matrix_transpose_v_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    v_cache: &MetalTensor,
    v_t: &MetalTensor,
    base_pos: usize,
    n_rows: usize,
    n_pos: usize,
    kv_stride: usize,
    vt_stride: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    encode_attn_matrix_transpose_v_f16_mode(
        ctx,
        enc,
        v_cache,
        v_t,
        base_pos,
        n_rows,
        n_pos,
        kv_stride,
        vt_stride,
        n_kv_heads,
        head_dim,
        attn_matrix_vt_compact_dispatch_enabled()?,
    )
}

pub fn encode_attn_matrix_kq_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    scores: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    kv_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kq",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if q_rows.dtype != GgmlType::F32
        || scores.dtype != GgmlType::F32
        || k_cache.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq",
            detail: format!(
                "expected q/scores F32 and k F16, got {:?}/{:?}/{:?}",
                q_rows.dtype, scores.dtype, k_cache.dtype
            ),
        });
    }
    let want_q = n_rows * n_q_heads * head_dim;
    let want_scores = n_kv_heads * n_rows * group * n_pos;
    if q_rows.n_elements() != want_q as u64 || scores.n_elements() < want_scores as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq",
            detail: format!(
                "bad q/scores sizes: q have {} need {want_q}, scores have {} need >= {want_scores}",
                q_rows.n_elements(),
                scores.n_elements()
            ),
        });
    }
    let full_tiles =
        n_pos.is_multiple_of(64) && (n_rows * group).is_multiple_of(32) && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kq_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kq_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: kv_stride as u32,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, scores);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: n_pos.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_attn_matrix_softmax_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_softmax",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    let want_scores = n_rows * n_q_heads * n_pos;
    if scores.dtype != GgmlType::F32 || scores.n_elements() < want_scores as u64 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_softmax",
            detail: format!("scores have {} need >= {want_scores}", scores.n_elements()),
        });
    }
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let pso = ctx.pipeline("kernel_attn_matrix_softmax_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: 0,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale,
            causal_skip: 0,
        },
    );
    enc.set_tensor(1, scores);
    enc.set_threadgroup_memory(0, 8 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_rows * n_q_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Number of F32 elements required for the per-(query, 64-pos-tile) (m, l)
/// sidecar consumed by the two-pass online matrix attention kernels.
pub fn attn_matrix_ml_elems(n_rows: usize, n_q_heads: usize, n_pos: usize) -> usize {
    n_rows * n_q_heads * n_pos.div_ceil(64) * 2
}

/// KQ with fused online softmax (`kernel_attn_matrix_kq_online_f32` in
/// kernels/attn_matrix_online.metal): same GEMM as [`encode_attn_matrix_kq_f32`]
/// but the epilogue applies scale + causal mask and stores
/// `P~ = exp2(s*scale - m_tile)` as F16 plus a per-(query, tile) (m, l)
/// sidecar. Replaces the separate softmax dispatch; pair with
/// [`encode_attn_matrix_kqv_norm_f32`].
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_matrix_kq_online_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_rows: &MetalTensor,
    k_cache: &MetalTensor,
    scores_h: &MetalTensor,
    ml: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    kv_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kq_online",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if q_rows.dtype != GgmlType::F32
        || scores_h.dtype != GgmlType::F16
        || ml.dtype != GgmlType::F32
        || k_cache.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq_online",
            detail: format!(
                "expected q F32, scores F16, ml F32, k F16; got {:?}/{:?}/{:?}/{:?}",
                q_rows.dtype, scores_h.dtype, ml.dtype, k_cache.dtype
            ),
        });
    }
    let want_q = n_rows * n_q_heads * head_dim;
    let want_scores = n_kv_heads * n_rows * group * n_pos;
    let want_ml = attn_matrix_ml_elems(n_rows, n_q_heads, n_pos);
    if q_rows.n_elements() != want_q as u64
        || scores_h.n_elements() < want_scores as u64
        || ml.n_elements() < want_ml as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kq_online",
            detail: format!(
                "bad sizes: q have {} need {want_q}, scores have {} need >= {want_scores}, ml have {} need >= {want_ml}",
                q_rows.n_elements(),
                scores_h.n_elements(),
                ml.n_elements()
            ),
        });
    }
    let scale = (1.0f32 / (head_dim as f32).sqrt()) * std::f32::consts::LOG2_E;
    let full_tiles =
        n_pos.is_multiple_of(64) && (n_rows * group).is_multiple_of(32) && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kq_online_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kq_online_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: kv_stride as u32,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, q_rows);
    enc.set_tensor(2, k_cache);
    enc.set_tensor(3, scores_h);
    enc.set_tensor(4, ml);
    // AMO_KQ_TG_BYTES in kernels/attn_matrix_online.metal.
    enc.set_threadgroup_memory(0, 9728);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: n_pos.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// KQV with fused normalization (`kernel_attn_matrix_kqv_norm_f32` in
/// kernels/attn_matrix_online.metal): same GEMM as
/// [`encode_attn_matrix_kqv_f32`] but reads the F16 `P~` produced by
/// [`encode_attn_matrix_kq_online_f32`], rescales it by
/// `exp2(m_tile - m_glob)` during staging, and divides by the global `l` in
/// the epilogue.
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_matrix_kqv_norm_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probs_h: &MetalTensor,
    ml: &MetalTensor,
    v_t: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    vt_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kqv_norm",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if vt_stride < n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_norm",
            detail: format!("vt_stride={vt_stride} < n_pos={n_pos}"),
        });
    }
    let want_probs = n_rows * n_q_heads * n_pos;
    let want_ml = attn_matrix_ml_elems(n_rows, n_q_heads, n_pos);
    let want_vt = n_kv_heads * head_dim * vt_stride;
    let want_out = n_rows * n_q_heads * head_dim;
    if probs_h.dtype != GgmlType::F16
        || ml.dtype != GgmlType::F32
        || out.dtype != GgmlType::F32
        || v_t.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_norm",
            detail: format!(
                "expected probs F16, ml F32, out F32, v_t F16; got {:?}/{:?}/{:?}/{:?}",
                probs_h.dtype, ml.dtype, out.dtype, v_t.dtype
            ),
        });
    }
    if probs_h.n_elements() < want_probs as u64
        || ml.n_elements() < want_ml as u64
        || v_t.n_elements() < want_vt as u64
        || out.n_elements() != want_out as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_norm",
            detail: format!(
                "bad sizes: probs {} need >= {want_probs}, ml {} need >= {want_ml}, v_t {} need >= {want_vt}, out {} need {want_out}",
                probs_h.n_elements(),
                ml.n_elements(),
                v_t.n_elements(),
                out.n_elements()
            ),
        });
    }
    let full_tiles =
        n_pos.is_multiple_of(32) && (n_rows * group).is_multiple_of(32) && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kqv_norm_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kqv_norm_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: 0,
            vt_stride: vt_stride as u32,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, probs_h);
    enc.set_tensor(2, ml);
    enc.set_tensor(3, v_t);
    enc.set_tensor(4, out);
    // AMO_KQV_TG_BYTES in kernels/attn_matrix_online.metal.
    enc.set_threadgroup_memory(0, 8320);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: head_dim.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_attn_matrix_kqv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probs: &MetalTensor,
    v_t: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    vt_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kqv",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if vt_stride < n_pos {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv",
            detail: format!("vt_stride={vt_stride} < n_pos={n_pos}"),
        });
    }
    let want_probs = n_rows * n_q_heads * n_pos;
    let want_vt = n_kv_heads * head_dim * vt_stride;
    let want_out = n_rows * n_q_heads * head_dim;
    if probs.dtype != GgmlType::F32 || out.dtype != GgmlType::F32 || v_t.dtype != GgmlType::F16 {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv",
            detail: format!(
                "expected probs/out F32 and v_t F16, got {:?}/{:?}/{:?}",
                probs.dtype, out.dtype, v_t.dtype
            ),
        });
    }
    if probs.n_elements() < want_probs as u64
        || v_t.n_elements() < want_vt as u64
        || out.n_elements() != want_out as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv",
            detail: format!(
                "bad sizes: probs {} need >= {want_probs}, v_t {} need >= {want_vt}, out {} need {want_out}",
                probs.n_elements(),
                v_t.n_elements(),
                out.n_elements()
            ),
        });
    }
    let full_tiles =
        n_pos.is_multiple_of(32) && (n_rows * group).is_multiple_of(32) && head_dim == 256;
    let kernel = if full_tiles {
        "kernel_attn_matrix_kqv_f32_full_tiles"
    } else {
        "kernel_attn_matrix_kqv_f32"
    };
    let pso = ctx.pipeline(kernel)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: 0,
            vt_stride: vt_stride as u32,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, probs);
    enc.set_tensor(2, v_t);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: head_dim.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Direct-V KQV (`kernel_attn_matrix_kqv_direct_v_f32`): the probs x V GEMM
/// reading the V cache directly (strided column-major staging) instead of a
/// pre-transposed v_t, deleting the transpose pass for the decode verify
/// reader. Same grid, staging, and MMA shape as the v_t variant.
#[allow(clippy::too_many_arguments)]
pub fn encode_attn_matrix_kqv_direct_v_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    probs: &MetalTensor,
    v_cache: &MetalTensor,
    out: &MetalTensor,
    n_rows: usize,
    base_pos: usize,
    n_pos: usize,
    kv_stride: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    group: usize,
    head_dim: usize,
    causal_skip: bool,
) -> Result<(), MetalError> {
    validate_attn_matrix_common(
        "attn_matrix_kqv_direct_v",
        n_rows,
        n_pos,
        base_pos,
        n_q_heads,
        n_kv_heads,
        group,
        head_dim,
    )?;
    if kv_stride < n_kv_heads * head_dim {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_direct_v",
            detail: format!("kv_stride={kv_stride} < n_kv_heads*head_dim"),
        });
    }
    let want_probs = n_rows * n_q_heads * n_pos;
    let want_v = n_pos * kv_stride;
    let want_out = n_rows * n_q_heads * head_dim;
    if probs.dtype != GgmlType::F32 || out.dtype != GgmlType::F32 || v_cache.dtype != GgmlType::F16
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_direct_v",
            detail: format!(
                "expected probs/out F32 and v_cache F16, got {:?}/{:?}/{:?}",
                probs.dtype, out.dtype, v_cache.dtype
            ),
        });
    }
    if probs.n_elements() < want_probs as u64
        || v_cache.n_elements() < want_v as u64
        || out.n_elements() != want_out as u64
    {
        return Err(MetalError::BadShape {
            kernel: "attn_matrix_kqv_direct_v",
            detail: format!(
                "bad sizes: probs {} need >= {want_probs}, v_cache {} need >= {want_v}, out {} need {want_out}",
                probs.n_elements(),
                v_cache.n_elements(),
                out.n_elements()
            ),
        });
    }
    let pso = ctx.pipeline("kernel_attn_matrix_kqv_direct_v_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &AttnMatrixArgs {
            n_rows: n_rows as u32,
            n_pos: n_pos as u32,
            base_pos: base_pos as u32,
            kv_stride: kv_stride as u32,
            vt_stride: 0,
            n_q_heads: n_q_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group: group as u32,
            head_dim: head_dim as u32,
            scale: 0.0,
            causal_skip: causal_skip as u32,
        },
    );
    enc.set_tensor(1, probs);
    enc.set_tensor(2, v_cache);
    enc.set_tensor(3, out);
    enc.set_threadgroup_memory(0, 8192);
    enc.dispatch(
        MTLSize {
            width: (n_rows * group).div_ceil(32),
            height: head_dim.div_ceil(64),
            depth: n_kv_heads,
        },
        MTLSize {
            width: 128,
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
    fn attn_v4_q8_kv_close_to_f16_kv() {
        run_attn_v4_q8_kv_compare("g6-main", 24, 4, 4096, 64, 32, None);
    }

    #[test]
    fn attn_v4_q8_group8_main_close_to_f16_kv() {
        run_attn_v4_q8_kv_compare("g8-main", 16, 2, 2048, 32, 32, Some(8));
    }

    #[test]
    fn attn_v4_q8_group8_subgroup_close_to_f16_kv() {
        run_attn_v4_q8_kv_compare("g8-t2", 16, 2, 8192, 64, 64, Some(2));
        run_attn_v4_q8_kv_compare("g8-t4", 16, 2, 16384, 128, 64, Some(4));
    }

    /// v0.433 triage repro for the `attn_v4_matches_naive_f16kv` load-flake:
    /// NaN-prime the o/ml partials scratch before dispatch at the exact
    /// config that failed under parallel-suite load (`group=4 n_pos=1024
    /// nwg=64 C=16`, cos=0.9662). `zeros_f32` is documented-uninitialized,
    /// so isolated runs see fresh zero pages while loaded runs see recycled
    /// garbage; if any kernel cell is read without being written, this test
    /// fails deterministically instead of 50%-of-suite-runs.
    #[test]
    fn attn_v4_partials_fully_written_nan_prime() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let hd = 256usize;
        // The observed failing config plus its close neighbors.
        let cases: &[(usize, usize, usize, usize, usize)] = &[
            // (n_q, n_kv, n_pos, nwg, tile_c)
            (8, 2, 1024, 64, 16),
            (8, 2, 1024, 64, 32),
            (8, 2, 1024, 128, 16),
            (8, 2, 1024, 256, 16),
            (24, 4, 1024, 64, 16),
        ];
        for &(n_q, n_kv, n_pos, nwg, tile_c) in cases {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();
            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_f16kv_f32(
                    &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                )
            })
            .unwrap();
            let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
            let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            // NaN-prime everything the kernels are supposed to fully write.
            unsafe {
                for t in [&o_partial, &ml_partial, &y_v4_t] {
                    let p = t.buffer.contents().as_ptr() as *mut f32;
                    for i in 0..t.n_elements() as usize {
                        *p.add(i) = f32::NAN;
                    }
                }
            }
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial,
                    &ml_partial,
                    &y_v4_t,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
            .unwrap();
            let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
            let nan_count = y_v4.iter().filter(|x| x.is_nan()).count();
            let max_abs = y_v4
                .iter()
                .zip(y_naive.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "[v4-nan-prime group={group} n_pos={n_pos} nwg={nwg} C={tile_c}] \
                 nans={nan_count} max|Δ|={max_abs:.2e}"
            );
            assert_eq!(
                nan_count, 0,
                "v4 output contains NaN after NaN-priming partials: some partial \
                 cell is read without being written (group={group} n_pos={n_pos} \
                 nwg={nwg} C={tile_c})"
            );
            assert!(
                max_abs < 5e-3,
                "v4 diverged from naive with NaN-primed partials: max|Δ|={max_abs} \
                 (group={group} n_pos={n_pos} nwg={nwg} C={tile_c})"
            );
        }
    }

    /// v4 flash-attn (GQA-dedup + online softmax + split-K) vs production
    /// `attn_decode_f16kv_f32`. Same F16 K/V inputs, multiple n_pos and NWG
    /// settings. Must produce numerically equivalent outputs (cos > 0.9999;
    /// max|Δ| ~1e-3 — the bound expected from F32 reorder noise across
    /// completely different reduction orderings).
    #[test]
    fn attn_v4_matches_naive_f16kv() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let hd = 256usize;
        // Cover the currently-supported specializations:
        // small dense (GROUP=4), 27B dense (GROUP=6), 35B A3B (GROUP=8),
        // 122B A10B (GROUP=16).
        let shapes: &[(usize, usize)] = &[(8, 2), (24, 4), (16, 2), (32, 2)];

        for &(n_q, n_kv) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            let cases: &[(usize, usize)] = &[
                (1, 1),
                (32, 1),
                (32, 2),
                (64, 1),
                (256, 4),
                (1024, 8),
                (1024, 64),
                (1024, 128),
                (1024, 256),
                (4096, 16),
                (4096, 64),
            ];

            for &(n_pos, nwg) in cases {
                // Synthesize Q (F32) and K, V (F32 scratch → F16 cache).
                let q: Vec<f32> = (0..n_q * hd)
                    .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                    .collect();
                let cap = n_pos.max(64);
                let k_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                    .collect();
                let v_f32: Vec<f32> = (0..cap * kv_dim)
                    .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                // Build F16 KV cache by scattering F32 source into a F16 dest.
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
                // Use scatter to convert F32 → F16 in cache.
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }

                // --- Reference: naive f16kv kernel ---
                let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
                one_shot(&ctx, |enc| {
                    encode_attn_decode_f16kv_f32(
                        &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                    )
                })
                .unwrap();
                let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

                // --- v4: allocate partials, dispatch main + reduce ---
                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64]).unwrap();
                let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

                // Sweep all three tile-C variants — each must match naive within
                // fp32 reorder noise (cos > 0.9999, max|Δ| < 5e-3).
                for &tile_c in &[16usize, 32, 64, 128] {
                    one_shot(&ctx, |enc| {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_v4_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            tile_c,
                        )
                    })
                    .unwrap();
                    let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
                    let max_abs = y_v4
                        .iter()
                        .zip(y_naive.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let dot: f64 = y_v4
                        .iter()
                        .zip(y_naive.iter())
                        .map(|(a, b)| (*a as f64) * (*b as f64))
                        .sum();
                    let na: f64 = y_v4.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                    let nb: f64 = y_naive
                        .iter()
                        .map(|x| (*x as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let cos = dot / (na * nb);
                    eprintln!(
                        "[v4 group={group:>2} n_q={n_q:>2} n_kv={n_kv:>2} n_pos={n_pos:>4} nwg={nwg:>2} C={tile_c:>2}] max|Δ|={max_abs:.2e}  cos={cos:.6}"
                    );
                    assert!(
                        cos > 0.9999,
                        "v4(group={group}, C={tile_c}) vs naive cos too low at n_pos={n_pos} nwg={nwg}: cos={cos}"
                    );
                    assert!(
                        max_abs < 5e-3,
                        "v4(group={group}, C={tile_c}) vs naive max|Δ| too high at n_pos={n_pos} nwg={nwg}: {max_abs}"
                    );
                }
            }
        }
    }

    #[test]
    fn attn_matrix_vt_dispatch_groups_cover_exact_thread_range() {
        for total in [1usize, 255, 256, 257] {
            assert_eq!(attn_matrix_vt_threadgroups(total, false).unwrap(), total);
            assert_eq!(
                attn_matrix_vt_threadgroups(total, true).unwrap(),
                total.div_ceil(ATTN_MATRIX_VT_THREADS)
            );
        }
        assert!(attn_matrix_vt_threadgroups(0, false).is_err());
        let legacy_max = (u32::MAX as usize + 1) / ATTN_MATRIX_VT_THREADS;
        assert_eq!(
            attn_matrix_vt_threadgroups(legacy_max, false).unwrap(),
            legacy_max
        );
        assert!(attn_matrix_vt_threadgroups(legacy_max + 1, false).is_err());
        assert!(attn_matrix_vt_threadgroups(u32::MAX as usize, true).is_ok());
        assert!(attn_matrix_vt_threadgroups(u32::MAX as usize + 1, true).is_err());
    }

    #[test]
    fn attn_matrix_vt_scoped_override_restores_and_rejects_nesting() {
        let _serial = ATTN_MATRIX_VT_SCOPE_TEST_LOCK.lock().unwrap();
        let baseline = attn_matrix_vt_compact_dispatch_enabled().unwrap();
        assert_eq!(
            with_attn_matrix_vt_compact_dispatch_override(!baseline, || {
                attn_matrix_vt_compact_dispatch_enabled()
            })
            .unwrap()
            .unwrap(),
            !baseline
        );
        assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);

        let nested = with_attn_matrix_vt_compact_dispatch_override(true, || {
            with_attn_matrix_vt_compact_dispatch_override(false, || ())
        })
        .unwrap();
        assert!(nested.is_err());
        assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);

        let cross_thread = with_attn_matrix_vt_compact_dispatch_override(true, || {
            std::thread::spawn(attn_matrix_vt_compact_dispatch_enabled)
                .join()
                .unwrap()
        })
        .unwrap();
        assert!(cross_thread.is_err());
        assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);

        let panicked = std::panic::catch_unwind(|| {
            let _ = with_attn_matrix_vt_compact_dispatch_override(!baseline, || {
                panic!("exercise override unwind restoration")
            });
        });
        assert!(panicked.is_err());
        assert_eq!(attn_matrix_vt_compact_dispatch_enabled().unwrap(), baseline);
    }

    #[test]
    fn attn_matrix_vt_dispatch_capture_is_exact_and_scoped() {
        let _serial = ATTN_MATRIX_VT_SCOPE_TEST_LOCK.lock().unwrap();
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const ROWS: usize = 257;
        let cache = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![0x3555u16; ROWS]),
            vec![ROWS as u64],
            GgmlType::F16,
        )
        .unwrap();
        let make_vt = || {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&vec![0x3aaau16; ROWS]),
                vec![ROWS as u64],
                GgmlType::F16,
            )
            .unwrap()
        };

        for (compact, expected_groups) in [(false, ROWS), (true, ROWS.div_ceil(256))] {
            let vt = make_vt();
            let (encoded, capture) = with_attn_matrix_vt_compact_dispatch_override(compact, || {
                capture_attn_matrix_vt_dispatches(|| {
                    one_shot(&ctx, |enc| {
                        encode_attn_matrix_transpose_v_f16(
                            &ctx, enc, &cache, &vt, 0, ROWS, ROWS, 1, ROWS, 1, 1,
                        )
                    })
                })
            })
            .unwrap()
            .unwrap();
            encoded.unwrap();
            assert!(capture.owner_thread.starts_with("ThreadId("));
            assert_eq!(
                capture.stats,
                AttnMatrixVtDispatchStats {
                    calls: 1,
                    row_sum: ROWS as u64,
                    element_sum: ROWS as u64,
                    threadgroup_sum: expected_groups as u64,
                    compact_calls: u64::from(compact),
                    legacy_calls: u64::from(!compact),
                    base_pos_sum: 0,
                    n_pos_sum: ROWS as u64,
                }
            );
        }

        let (nested, outer) =
            capture_attn_matrix_vt_dispatches(|| capture_attn_matrix_vt_dispatches(|| ())).unwrap();
        assert!(nested.is_err());
        assert_eq!(outer.stats, AttnMatrixVtDispatchStats::default());

        let (cross_thread, capture) = capture_attn_matrix_vt_dispatches(|| {
            std::thread::spawn(|| record_attn_matrix_vt_dispatch(0, 1, 1, 1, 1, true))
                .join()
                .unwrap()
        })
        .unwrap();
        assert!(cross_thread.is_err());
        assert_eq!(capture.stats, AttnMatrixVtDispatchStats::default());

        let panicked = std::panic::catch_unwind(|| {
            let _ = capture_attn_matrix_vt_dispatches(|| panic!("exercise capture unwind"));
        });
        assert!(panicked.is_err());
        let (_, capture) = capture_attn_matrix_vt_dispatches(|| ()).unwrap();
        assert_eq!(capture.stats, AttnMatrixVtDispatchStats::default());
    }

    #[test]
    fn attn_matrix_vt_compact_dispatch_matches_legacy_nonzero_span() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const SENTINEL: u16 = 0x3555;

        // Exact thread totals 255, 256, and 257. The first two also exercise
        // multiple KV heads and dimensions; all use nonzero base and padding.
        for &(n_kv, head_dim, n_rows) in &[(3usize, 5usize, 17usize), (2, 8, 16), (1, 1, 257)] {
            let base_pos = 2usize;
            let n_pos = base_pos + n_rows + 1;
            let vt_stride = n_pos + 3;
            let kv_dim = n_kv * head_dim;
            let total = kv_dim * n_rows;
            assert!([255, 256, 257].contains(&total));
            let cache: Vec<u16> = (0..n_pos * kv_dim)
                .map(|i| half::f16::from_f32(((i % 31) as f32 - 15.0) * 0.03125).to_bits())
                .collect();
            let cache_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&cache),
                vec![cache.len() as u64],
                GgmlType::F16,
            )
            .unwrap();
            let make_vt = || {
                MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&vec![SENTINEL; kv_dim * vt_stride]),
                    vec![(kv_dim * vt_stride) as u64],
                    GgmlType::F16,
                )
                .unwrap()
            };
            let legacy = make_vt();
            let compact = make_vt();

            for (dst, compact_dispatch) in [(&legacy, false), (&compact, true)] {
                one_shot(&ctx, |enc| {
                    encode_attn_matrix_transpose_v_f16_mode(
                        &ctx,
                        enc,
                        &cache_t,
                        dst,
                        base_pos,
                        n_rows,
                        n_pos,
                        kv_dim,
                        vt_stride,
                        n_kv,
                        head_dim,
                        compact_dispatch,
                    )
                })
                .unwrap();
            }

            let mut expected = vec![SENTINEL; kv_dim * vt_stride];
            for pos in base_pos..base_pos + n_rows {
                for flat_d in 0..kv_dim {
                    expected[flat_d * vt_stride + pos] = cache[pos * kv_dim + flat_d];
                }
            }
            assert_eq!(read_back_u16(&legacy), expected);
            assert_eq!(read_back_u16(&compact), expected);

            let cmd = ctx
                .queue
                .commandBuffer()
                .expect("validation command buffer");
            let enc = KernelEncoder::begin(&cmd);
            assert!(
                encode_attn_matrix_transpose_v_f16_mode(
                    &ctx,
                    &enc,
                    &cache_t,
                    &compact,
                    base_pos,
                    n_rows,
                    n_pos,
                    kv_dim - 1,
                    vt_stride,
                    n_kv,
                    head_dim,
                    true,
                )
                .is_err()
            );
            let mut short_cache = cache_t.clone();
            short_cache.shape = vec![((base_pos + n_rows) * kv_dim - 1) as u64];
            assert!(
                encode_attn_matrix_transpose_v_f16_mode(
                    &ctx,
                    &enc,
                    &short_cache,
                    &compact,
                    base_pos,
                    n_rows,
                    n_pos,
                    kv_dim,
                    vt_stride,
                    n_kv,
                    head_dim,
                    true,
                )
                .is_err()
            );
            let mut short_vt = compact.clone();
            let exact_vt_end = (kv_dim - 1) * vt_stride + base_pos + n_rows;
            short_vt.shape = vec![(exact_vt_end - 1) as u64];
            assert!(
                encode_attn_matrix_transpose_v_f16_mode(
                    &ctx, &enc, &cache_t, &short_vt, base_pos, n_rows, n_pos, kv_dim, vt_stride,
                    n_kv, head_dim, true,
                )
                .is_err()
            );
            assert!(
                encode_attn_matrix_transpose_v_f16_mode(
                    &ctx,
                    &enc,
                    &cache_t,
                    &compact,
                    usize::MAX,
                    1,
                    n_pos,
                    kv_dim,
                    vt_stride,
                    n_kv,
                    head_dim,
                    true,
                )
                .is_err()
            );
            enc.end();
        }
    }

    #[test]
    fn attn_matrix_vt_prefix_rebuild_preserves_scattered_suffix() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const PREFIX: usize = 5;
        const CHUNK: usize = 3;
        const N_KV: usize = 2;
        const HEAD_DIM: usize = 8;
        const VT_PADDING: usize = 3;
        const SENTINELS: [u16; 3] = [0x3555, 0x3aaa, 0x3999];

        let n_pos = PREFIX + CHUNK;
        let vt_stride = n_pos + VT_PADDING;
        let kv_dim = N_KV * HEAD_DIM;
        let cache_elems = n_pos * kv_dim;
        let vt_elems = kv_dim * vt_stride;
        let initial_cache: Vec<u16> = (0..cache_elems)
            .map(|i| half::f16::from_f32(((i % 29) as f32 - 14.0) * 0.03125).to_bits())
            .collect();
        let current_f32: Vec<f32> = (0..CHUNK * kv_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
            .collect();
        let current_f16: Vec<u16> = current_f32
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect();
        let current = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&current_f32),
            vec![current_f32.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        let mut expected_cache = initial_cache.clone();
        expected_cache[PREFIX * kv_dim..].copy_from_slice(&current_f16);

        let make_cache = || {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&initial_cache),
                vec![cache_elems as u64],
                GgmlType::F16,
            )
            .unwrap()
        };
        let mut arms = Vec::new();
        for &sentinel in &SENTINELS {
            let cache_k = make_cache();
            let cache_v = make_cache();
            let vt = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&vec![sentinel; vt_elems]),
                vec![vt_elems as u64],
                GgmlType::F16,
            )
            .unwrap();
            one_shot(&ctx, |enc| {
                encode_scatter_offset_f32_to_f16_kv_vt(
                    &ctx,
                    enc,
                    &current,
                    &current,
                    &cache_k,
                    &cache_v,
                    &vt,
                    PREFIX * kv_dim,
                    CHUNK * kv_dim,
                    PREFIX,
                    kv_dim,
                    HEAD_DIM,
                    vt_stride,
                )
            })
            .unwrap();
            assert_eq!(read_back_u16(&cache_k), expected_cache);
            assert_eq!(read_back_u16(&cache_v), expected_cache);
            let scattered_vt = read_back_u16(&vt);
            for row in 0..CHUNK {
                for flat_d in 0..kv_dim {
                    assert_eq!(
                        scattered_vt[flat_d * vt_stride + PREFIX + row],
                        current_f16[row * kv_dim + flat_d]
                    );
                }
            }
            arms.push((cache_v, vt, sentinel));
        }

        let divergent_suffix: Vec<f32> = current_f32.iter().map(|&value| value + 1.0).collect();
        let divergent_suffix_f16: Vec<u16> = divergent_suffix
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect();
        let divergent_suffix_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&divergent_suffix),
            vec![divergent_suffix.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_f16(
                &ctx,
                enc,
                &divergent_suffix_t,
                &arms[2].0,
                PREFIX * kv_dim,
                divergent_suffix.len(),
            )
        })
        .unwrap();
        assert_eq!(
            &read_back_u16(&arms[2].0)[PREFIX * kv_dim..],
            divergent_suffix_f16.as_slice()
        );

        for (index, (cache_v, vt, _)) in arms.iter().enumerate() {
            let (rows, compact) = match index {
                0 => (n_pos, false),
                1 => (n_pos, true),
                _ => (PREFIX, true),
            };
            one_shot(&ctx, |enc| {
                encode_attn_matrix_transpose_v_f16_mode(
                    &ctx, enc, cache_v, vt, 0, rows, n_pos, kv_dim, vt_stride, N_KV, HEAD_DIM,
                    compact,
                )
            })
            .unwrap();
        }

        for (index, (_, vt, sentinel)) in arms.iter().enumerate() {
            let mut expected_vt = vec![*sentinel; vt_elems];
            for pos in 0..n_pos {
                for flat_d in 0..kv_dim {
                    expected_vt[flat_d * vt_stride + pos] = if index == 2 && pos >= PREFIX {
                        current_f16[(pos - PREFIX) * kv_dim + flat_d]
                    } else {
                        expected_cache[pos * kv_dim + flat_d]
                    };
                }
            }
            assert_eq!(read_back_u16(vt), expected_vt);
        }
    }

    #[test]
    fn attn_matrix_prefix_only_vt_rebuild_matches_full_attention() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        const PREFIX: usize = 47;
        const CHUNK: usize = 17;
        const N_Q: usize = 24;
        const N_KV: usize = 4;
        const HEAD_DIM: usize = 256;
        const SENTINEL: u16 = 0x3555;

        let n_pos = PREFIX + CHUNK;
        let group = N_Q / N_KV;
        let kv_dim = N_KV * HEAD_DIM;
        let vt_stride = n_pos + 3;
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
        let q: Vec<f32> = (0..CHUNK * N_Q * HEAD_DIM)
            .map(|i| round_f16(((i % 31) as f32 - 15.0) * 0.01))
            .collect();
        let k: Vec<f32> = (0..n_pos * kv_dim)
            .map(|i| round_f16(((i % 23) as f32 - 11.0) * 0.015))
            .collect();
        let v: Vec<f32> = (0..n_pos * kv_dim)
            .map(|i| round_f16(((i % 17) as f32 - 8.0) * 0.02))
            .collect();
        let k_f16: Vec<u16> = k
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect();
        let v_f16: Vec<u16> = v
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect();
        let mut restored_k = vec![SENTINEL; n_pos * kv_dim];
        let mut restored_v = vec![SENTINEL; n_pos * kv_dim];
        restored_k[..PREFIX * kv_dim].copy_from_slice(&k_f16[..PREFIX * kv_dim]);
        restored_v[..PREFIX * kv_dim].copy_from_slice(&v_f16[..PREFIX * kv_dim]);

        let tensor_f32 = |data: &[f32]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(data),
                vec![data.len() as u64],
                GgmlType::F32,
            )
            .unwrap()
        };
        let tensor_f16 = |data: &[u16]| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(data),
                vec![data.len() as u64],
                GgmlType::F16,
            )
            .unwrap()
        };
        let q_t = tensor_f32(&q);
        let current_k = tensor_f32(&k[PREFIX * kv_dim..]);
        let current_v = tensor_f32(&v[PREFIX * kv_dim..]);
        let full_k = tensor_f16(&k_f16);
        let full_v = tensor_f16(&v_f16);
        let rebuilt_k = tensor_f16(&restored_k);
        let rebuilt_v = tensor_f16(&restored_v);
        let make_vt = || tensor_f16(&vec![SENTINEL; kv_dim * vt_stride]);
        let full_vt = make_vt();
        let rebuilt_vt = make_vt();

        let make_scores =
            || MetalTensor::zeros_f16(&ctx, vec![(CHUNK * N_Q * n_pos) as u64]).unwrap();
        let make_ml = || {
            MetalTensor::zeros_f32(&ctx, vec![attn_matrix_ml_elems(CHUNK, N_Q, n_pos) as u64])
                .unwrap()
        };
        let full_scores = make_scores();
        let full_ml = make_ml();
        let full_out = MetalTensor::zeros_f32(&ctx, vec![(CHUNK * N_Q * HEAD_DIM) as u64]).unwrap();
        one_shot(&ctx, |enc| {
            encode_attn_matrix_transpose_v_f16(
                &ctx, enc, &full_v, &full_vt, 0, n_pos, n_pos, kv_dim, vt_stride, N_KV, HEAD_DIM,
            )?;
            encode_attn_matrix_kq_online_f32(
                &ctx,
                enc,
                &q_t,
                &full_k,
                &full_scores,
                &full_ml,
                CHUNK,
                PREFIX,
                n_pos,
                kv_dim,
                N_Q,
                N_KV,
                group,
                HEAD_DIM,
                true,
            )?;
            encode_attn_matrix_kqv_norm_f32(
                &ctx,
                enc,
                &full_scores,
                &full_ml,
                &full_vt,
                &full_out,
                CHUNK,
                PREFIX,
                n_pos,
                vt_stride,
                N_Q,
                N_KV,
                group,
                HEAD_DIM,
                true,
            )
        })
        .unwrap();

        let rebuilt_scores = make_scores();
        let rebuilt_ml = make_ml();
        let rebuilt_out =
            MetalTensor::zeros_f32(&ctx, vec![(CHUNK * N_Q * HEAD_DIM) as u64]).unwrap();
        let rebuilt_prefix_rows =
            crate::metal_dflash::attn_matrix_vt_prefix_rebuild_rows(true, 0, PREFIX)
                .expect("restored prefix requires a V_T rebuild");
        one_shot(&ctx, |enc| {
            encode_scatter_offset_f32_to_f16_kv_vt(
                &ctx,
                enc,
                &current_k,
                &current_v,
                &rebuilt_k,
                &rebuilt_v,
                &rebuilt_vt,
                PREFIX * kv_dim,
                CHUNK * kv_dim,
                PREFIX,
                kv_dim,
                HEAD_DIM,
                vt_stride,
            )?;
            encode_attn_matrix_transpose_v_f16(
                &ctx,
                enc,
                &rebuilt_v,
                &rebuilt_vt,
                0,
                rebuilt_prefix_rows,
                n_pos,
                kv_dim,
                vt_stride,
                N_KV,
                HEAD_DIM,
            )?;
            encode_attn_matrix_kq_online_f32(
                &ctx,
                enc,
                &q_t,
                &rebuilt_k,
                &rebuilt_scores,
                &rebuilt_ml,
                CHUNK,
                PREFIX,
                n_pos,
                kv_dim,
                N_Q,
                N_KV,
                group,
                HEAD_DIM,
                true,
            )?;
            encode_attn_matrix_kqv_norm_f32(
                &ctx,
                enc,
                &rebuilt_scores,
                &rebuilt_ml,
                &rebuilt_vt,
                &rebuilt_out,
                CHUNK,
                PREFIX,
                n_pos,
                vt_stride,
                N_Q,
                N_KV,
                group,
                HEAD_DIM,
                true,
            )
        })
        .unwrap();

        assert_eq!(read_back_u16(&rebuilt_k), k_f16);
        assert_eq!(read_back_u16(&rebuilt_v), v_f16);
        assert_eq!(read_back_u16(&rebuilt_vt), read_back_u16(&full_vt));
        assert_eq!(
            read_back_f32(&rebuilt_out.buffer, CHUNK * N_Q * HEAD_DIM),
            read_back_f32(&full_out.buffer, CHUNK * N_Q * HEAD_DIM)
        );
    }

    /// Model-free screen preregistered in
    /// `docs/bench/2026-08-17-qwen-vt-rebuild-ceiling/README.md`.
    #[test]
    #[ignore]
    fn attn_matrix_vt_environment_probe() {
        let ctx = MetalContext::new().expect("Metal context for V_T environment probe");
        println!(
            "VT_ENV_JSON {}",
            serde_json::json!({
                "schema_version": 1,
                "test": "metal::tests::attn_matrix_vt_environment_probe",
                "device_registry_id": ctx.device.registryID(),
                "device": ctx.device.name().to_string(),
                "max_buffer_length": ctx.device.maxBufferLength(),
                "recommended_max_working_set_size": ctx.recommended_max_working_set_size(),
            })
        );
    }

    /// Model-free screen preregistered in
    /// `docs/bench/2026-08-17-qwen-vt-rebuild-ceiling/README.md`.
    #[test]
    #[ignore]
    fn attn_matrix_vt_rebuild_screen() {
        const N_LAYERS: usize = 16;
        const N_KV: usize = 4;
        const HEAD_DIM: usize = 256;

        struct Bank {
            src: Vec<MetalTensor>,
            dst: Vec<MetalTensor>,
        }

        fn parse_usize(name: &str) -> usize {
            let raw = std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"));
            raw.parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid {name}={raw:?}"))
        }

        fn fill_tensor(tensor: &MetalTensor, byte: u8) {
            assert_eq!(tensor.dtype, GgmlType::F16);
            assert_eq!(tensor.buffer.storageMode(), MTLStorageMode::Shared);
            assert_eq!(tensor.offset, 0);
            let n_bytes = tensor.n_bytes() as usize;
            assert!(n_bytes <= tensor.buffer.length());
            unsafe {
                std::ptr::write_bytes(tensor.buffer.contents().as_ptr() as *mut u8, byte, n_bytes);
            }
        }

        fn allocate_bank(
            ctx: &MetalContext,
            name: &str,
            elems_per_layer: usize,
            bytes_per_layer: usize,
            src_byte: u8,
            dst_byte: u8,
        ) -> Bank {
            let mut src = Vec::with_capacity(N_LAYERS);
            let mut dst = Vec::with_capacity(N_LAYERS);
            for layer in 0..N_LAYERS {
                let src_layer = MetalTensor::zeros_f16(ctx, vec![elems_per_layer as u64])
                    .unwrap_or_else(|error| {
                        panic!(
                            "allocate bank={name} layer={layer} role=src bytes={bytes_per_layer}: {error}"
                        )
                    });
                let dst_layer = MetalTensor::zeros_f16(ctx, vec![elems_per_layer as u64])
                    .unwrap_or_else(|error| {
                        panic!(
                            "allocate bank={name} layer={layer} role=dst bytes={bytes_per_layer}: {error}"
                        )
                    });
                fill_tensor(&src_layer, src_byte);
                fill_tensor(&dst_layer, dst_byte);
                src.push(src_layer);
                dst.push(dst_layer);
            }
            Bank { src, dst }
        }

        fn run_span(
            ctx: &MetalContext,
            bank: &Bank,
            label: &str,
            base_pos: usize,
            rows: usize,
            n_pos: usize,
            kv_dim: usize,
            compact: bool,
        ) -> (f64, f64) {
            let cmd = ctx
                .queue
                .commandBuffer()
                .expect("V_T rebuild command buffer");
            let enc = KernelEncoder::begin(&cmd);
            for layer in 0..N_LAYERS {
                encode_attn_matrix_transpose_v_f16_mode(
                    ctx,
                    &enc,
                    &bank.src[layer],
                    &bank.dst[layer],
                    base_pos,
                    rows,
                    n_pos,
                    kv_dim,
                    n_pos,
                    N_KV,
                    HEAD_DIM,
                    compact,
                )
                .unwrap();
            }
            enc.end();
            let wall_start = std::time::Instant::now();
            cmd.commit();
            cmd.waitUntilCompleted();
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1e3;
            let status = cmd.status();
            let error = cmd.error();
            assert!(
                status == objc2_metal::MTLCommandBufferStatus::Completed && error.is_none(),
                "V_T command failed arm={label} status={status:?} error={error:?}"
            );
            let gpu_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            assert!(wall_ms.is_finite() && wall_ms > 0.0);
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            (wall_ms, gpu_ms)
        }

        fn prep_overlap(
            ctx: &MetalContext,
            bank: &Bank,
            prefix: usize,
            chunk: usize,
            n_pos: usize,
            kv_dim: usize,
        ) {
            let _ = run_span(ctx, bank, "PREP_D2", prefix, chunk, n_pos, kv_dim, true);
        }

        let mode = std::env::var("QWEN_VT_REBUILD_MODE")
            .unwrap_or_else(|_| panic!("missing QWEN_VT_REBUILD_MODE"));
        let prefix = parse_usize("QWEN_VT_REBUILD_PREFIX");
        let chunk = parse_usize("QWEN_VT_REBUILD_CHUNK");
        match mode.as_str() {
            "dispatch" => {
                assert!([512, 2048, 8192].contains(&prefix));
                assert_eq!(chunk, 128);
            }
            "compact" => {
                assert!([8192, 16384, 32768].contains(&prefix));
                assert_eq!(chunk, 128);
            }
            "overlap" => {
                assert_eq!(prefix, 32768);
                assert_eq!(chunk, 1024);
            }
            _ => panic!("invalid QWEN_VT_REBUILD_MODE={mode:?}"),
        }

        let n_pos = prefix.checked_add(chunk).unwrap();
        let kv_dim = N_KV * HEAD_DIM;
        let elems_per_layer = n_pos.checked_mul(kv_dim).unwrap();
        let bytes_per_layer = elems_per_layer.checked_mul(2).unwrap();
        let total_requested_bytes = bytes_per_layer
            .checked_mul(N_LAYERS)
            .and_then(|value| value.checked_mul(4))
            .unwrap();
        let ctx = MetalContext::new().expect("Metal context for V_T rebuild screen");
        let max_buffer_length = ctx.device.maxBufferLength();
        assert!(
            bytes_per_layer <= max_buffer_length,
            "V_T layer bytes {bytes_per_layer} exceed maxBufferLength {max_buffer_length}"
        );
        let allocated_before = ctx.current_allocated_size();
        let bank_x = allocate_bank(&ctx, "X", elems_per_layer, bytes_per_layer, 0x3c, 0xa5);
        let bank_y = allocate_bank(&ctx, "Y", elems_per_layer, bytes_per_layer, 0x38, 0x5a);
        let allocated_after = ctx.current_allocated_size();

        println!(
            "VT_REBUILD_JSON {}",
            serde_json::json!({
                "kind": "meta",
                "schema_version": 2,
                "mode": mode,
                "prefix": prefix,
                "chunk": chunk,
                "n_pos": n_pos,
                "vt_stride": n_pos,
                "layers": N_LAYERS,
                "n_kv": N_KV,
                "head_dim": HEAD_DIM,
                "kv_dim": kv_dim,
                "device_registry_id": ctx.device.registryID(),
                "device": ctx.device.name().to_string(),
                "max_buffer_length": max_buffer_length,
                "recommended_max_working_set_size": ctx.recommended_max_working_set_size(),
                "bytes_per_layer_buffer": bytes_per_layer,
                "total_requested_bytes": total_requested_bytes,
                "allocated_before": allocated_before,
                "allocated_after": allocated_after,
            })
        );

        let arm_spec = |role: &str| -> (&str, bool, usize) {
            match (mode.as_str(), role) {
                ("dispatch", "A") => ("D0", false, n_pos),
                ("dispatch", "B") => ("D1", true, n_pos),
                ("overlap", "A") => ("D1", true, n_pos),
                ("overlap", "B") => ("D2", true, prefix),
                ("compact", "S") => ("D1", true, n_pos),
                _ => panic!("invalid mode/role {mode}/{role}"),
            }
        };
        let bank = |name: &str| -> &Bank {
            match name {
                "X" => &bank_x,
                "Y" => &bank_y,
                _ => panic!("invalid bank {name}"),
            }
        };
        let run_role = |role: &str, bank_name: &str| -> (f64, f64) {
            let (arm, compact, rows) = arm_spec(role);
            if mode == "overlap" {
                prep_overlap(&ctx, bank(bank_name), prefix, chunk, n_pos, kv_dim);
            }
            run_span(&ctx, bank(bank_name), arm, 0, rows, n_pos, kv_dim, compact)
        };

        let warmups: Vec<(&str, &str)> = match mode.as_str() {
            "dispatch" => vec![("A", "X"), ("B", "Y"), ("A", "Y"), ("B", "X")],
            "compact" => vec![("S", "X"), ("S", "Y")],
            "overlap" => vec![("A", "X"), ("B", "Y"), ("A", "Y"), ("B", "X")],
            _ => unreachable!(),
        };
        for (role, bank_name) in warmups {
            let _ = run_role(role, bank_name);
        }

        let paired_schedule = [
            [("A", "X"), ("B", "Y")],
            [("B", "X"), ("A", "Y")],
            [("B", "Y"), ("A", "X")],
            [("A", "Y"), ("B", "X")],
            [("A", "X"), ("B", "Y")],
            [("B", "X"), ("A", "Y")],
        ];
        let single_banks = ["X", "Y", "Y", "X", "X", "Y"];

        if mode == "compact" {
            for (sample_idx, bank_name) in single_banks.iter().enumerate() {
                let (wall_ms, gpu_ms) = run_role("S", bank_name);
                let (arm, compact, rows) = arm_spec("S");
                let logical_bytes = (N_LAYERS as u64)
                    .checked_mul(kv_dim as u64)
                    .and_then(|value| value.checked_mul(rows as u64))
                    .and_then(|value| value.checked_mul(4))
                    .unwrap();
                let total = kv_dim.checked_mul(rows).unwrap();
                let threadgroups = attn_matrix_vt_threadgroups(total, compact).unwrap();
                println!(
                    "VT_REBUILD_JSON {}",
                    serde_json::json!({
                        "kind": "arm",
                        "schema_version": 2,
                        "mode": mode,
                        "prefix": prefix,
                        "chunk": chunk,
                        "sample": sample_idx + 1,
                        "role": "S",
                        "arm": arm,
                        "bank": bank_name,
                        "base_pos": 0,
                        "rows": rows,
                        "threadgroups_per_layer": threadgroups,
                        "thread_slots_per_layer": threadgroups * ATTN_MATRIX_VT_THREADS,
                        "logical_bytes": logical_bytes,
                        "wall_ms": wall_ms,
                        "gpu_ms": gpu_ms,
                        "gb_s": logical_bytes as f64 / (gpu_ms * 1e6),
                    })
                );
            }
        } else {
            for (pair_idx, pair) in paired_schedule.iter().enumerate() {
                let order = format!("{}{}", pair[0].0, pair[1].0);
                for (sequence_idx, &(role, bank_name)) in pair.iter().enumerate() {
                    let (wall_ms, gpu_ms) = run_role(role, bank_name);
                    let (arm, compact, rows) = arm_spec(role);
                    let logical_bytes = (N_LAYERS as u64)
                        .checked_mul(kv_dim as u64)
                        .and_then(|value| value.checked_mul(rows as u64))
                        .and_then(|value| value.checked_mul(4))
                        .unwrap();
                    let total = kv_dim.checked_mul(rows).unwrap();
                    let threadgroups = attn_matrix_vt_threadgroups(total, compact).unwrap();
                    println!(
                        "VT_REBUILD_JSON {}",
                        serde_json::json!({
                            "kind": "arm",
                            "schema_version": 2,
                            "mode": mode,
                            "prefix": prefix,
                            "chunk": chunk,
                            "pair": pair_idx + 1,
                            "order": order,
                            "sequence": sequence_idx + 1,
                            "role": role,
                            "arm": arm,
                            "bank": bank_name,
                            "base_pos": 0,
                            "rows": rows,
                            "threadgroups_per_layer": threadgroups,
                            "thread_slots_per_layer": threadgroups * ATTN_MATRIX_VT_THREADS,
                            "logical_bytes": logical_bytes,
                            "wall_ms": wall_ms,
                            "gpu_ms": gpu_ms,
                            "gb_s": logical_bytes as f64 / (gpu_ms * 1e6),
                        })
                    );
                }
            }
        }
    }

    /// Micro-oracle for the non-flash matrix-attention sidecar
    /// (`kernel_attn_matrix_{transpose_v,kq,softmax,kqv}_f32`): the packed
    /// prefill attention body. Previously this path had only end-to-end
    /// coverage (27B G6 prefix gate + runtime packed oracle); this gate pins
    /// the kernels in isolation against a CPU f64 reference across all four
    /// production group shapes, full/edge tile geometries, and mid-sequence
    /// `base_pos > 0` chunks (including the tiny-chunk/long-prefix shape).
    ///
    /// Also asserts `causal_skip` on/off produce bitwise-identical output:
    /// skipped KQ tiles are exactly the rows the softmax zero-masks, and
    /// skipped KQV K-tiles multiply exact-zero probs.
    #[test]
    fn attn_matrix_path_matches_cpu_reference() {
        let Some(ctx) = metal_test_context() else {
            return;
        };
        let hd = 256usize;
        // (n_q, n_kv): G4 small dense, G6 27B, G8 A3B, G16 A10B.
        let shapes: &[(usize, usize)] = &[(8, 2), (24, 4), (16, 2), (32, 2)];
        // (n_rows, base_pos); n_pos = base_pos + n_rows as in production
        // (chunk attends to the whole prefix incl. itself).
        //  - (32, 0): first chunk, n_pos < 64 → KQ edge tiles
        //  - (64, 0): full 64-pos KQ tile; N edge depends on group
        //  - (17, 47): odd everything (M/N edge tiles, base_pos > 0)
        //  - (128, 896): full tiles, mid-sequence, n_pos = 1024
        //  - (8, 1016): tiny chunk over long prefix (prefix-gate shape)
        //  - (100, 156): n_pos = 256; N edge for G6/G4, full N for G8/G16
        let cases: &[(usize, usize)] = &[
            (32, 0),
            (64, 0),
            (17, 47),
            (128, 896),
            (8, 1016),
            (100, 156),
        ];

        for &(n_q, n_kv) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            for &(n_rows, base_pos) in cases {
                let n_pos = base_pos + n_rows;
                let round16 = |x: f32| half::f16::from_f32(x).to_f32();
                let q: Vec<f32> = (0..n_rows * n_q * hd)
                    .map(|i| round16(((i % 31) as f32 - 15.0) * 1e-2 + ((i % 7) as f32) * 3e-3))
                    .collect();
                let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                    .map(|i| round16(((i % 23) as f32 - 11.0) * 1.5e-2))
                    .collect();
                let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                    .map(|i| round16(((i % 17) as f32 - 8.0) * 2e-2))
                    .collect();

                let q_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&q),
                    vec![(n_rows * n_q * hd) as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
                for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                    let src_t = MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(src_f32.as_slice()),
                        vec![src_f32.len() as u64],
                        GgmlType::F32,
                    )
                    .unwrap();
                    one_shot(&ctx, |enc| {
                        encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                    })
                    .unwrap();
                }
                let vt_stride = n_pos;
                let v_t =
                    MetalTensor::zeros_f16(&ctx, vec![(n_kv * hd * vt_stride) as u64]).unwrap();
                let scores =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64])
                        .unwrap();
                let out_t = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

                let run_path = |causal_skip: bool| -> Vec<f32> {
                    one_shot(&ctx, |enc| {
                        encode_attn_matrix_transpose_v_f16(
                            &ctx, enc, &v_cache, &v_t, 0, n_pos, n_pos, kv_dim, vt_stride, n_kv, hd,
                        )?;
                        encode_attn_matrix_kq_f32(
                            &ctx,
                            enc,
                            &q_t,
                            &k_cache,
                            &scores,
                            n_rows,
                            base_pos,
                            n_pos,
                            kv_dim,
                            n_q,
                            n_kv,
                            group,
                            hd,
                            causal_skip,
                        )?;
                        encode_attn_matrix_softmax_f32(
                            &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                        )?;
                        encode_attn_matrix_kqv_f32(
                            &ctx,
                            enc,
                            &scores,
                            &v_t,
                            &out_t,
                            n_rows,
                            base_pos,
                            n_pos,
                            vt_stride,
                            n_q,
                            n_kv,
                            group,
                            hd,
                            causal_skip,
                        )
                    })
                    .unwrap();
                    read_back_f32(&out_t.buffer, n_rows * n_q * hd)
                };

                let y_gpu = run_path(true);
                let y_ref = cpu_matrix_attn_reference(
                    &q, &k_f32, &v_f32, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                );

                let max_abs = y_gpu
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let dot: f64 = y_gpu
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let na: f64 = y_gpu
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = y_ref
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let cos = dot / (na * nb);
                eprintln!(
                    "[matrix group={group:>2} n_rows={n_rows:>4} base={base_pos:>4} n_pos={n_pos:>4}] max|Δ|={max_abs:.2e}  cos={cos:.7}"
                );
                assert!(
                    cos > 0.9999,
                    "matrix path vs CPU ref cos too low (group={group} n_rows={n_rows} base={base_pos}): {cos}"
                );
                assert!(
                    max_abs < 5e-3,
                    "matrix path vs CPU ref max|Δ| too high (group={group} n_rows={n_rows} base={base_pos}): {max_abs}"
                );

                // causal_skip must be a pure perf feature: bitwise-identical out.
                if matches!((n_rows, base_pos), (64, 0) | (8, 1016)) {
                    let y_noskip = run_path(false);
                    assert!(
                        y_gpu == y_noskip,
                        "causal_skip changed matrix attention output (group={group} n_rows={n_rows} base={base_pos})"
                    );
                }

                // Two-pass online kernels: KQ folds the softmax into its
                // epilogue (F16 P~ + (m,l) sidecar), KQV normalizes during
                // staging. Must sit in the same envelope vs the CPU reference
                // (the P~ half demotion mirrors the sidecar's half probs).
                let scores_h_t =
                    MetalTensor::zeros_f16(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64])
                        .unwrap();
                let ml_t = MetalTensor::zeros_f32(
                    &ctx,
                    vec![attn_matrix_ml_elems(n_rows, n_q, n_pos) as u64],
                )
                .unwrap();
                let fused_t =
                    MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
                one_shot(&ctx, |enc| {
                    encode_attn_matrix_kq_online_f32(
                        &ctx,
                        enc,
                        &q_t,
                        &k_cache,
                        &scores_h_t,
                        &ml_t,
                        n_rows,
                        base_pos,
                        n_pos,
                        kv_dim,
                        n_q,
                        n_kv,
                        group,
                        hd,
                        true,
                    )?;
                    encode_attn_matrix_kqv_norm_f32(
                        &ctx,
                        enc,
                        &scores_h_t,
                        &ml_t,
                        &v_t,
                        &fused_t,
                        n_rows,
                        base_pos,
                        n_pos,
                        vt_stride,
                        n_q,
                        n_kv,
                        group,
                        hd,
                        true,
                    )
                })
                .unwrap();
                let y_fused = read_back_f32(&fused_t.buffer, n_rows * n_q * hd);
                let fmax_abs = y_fused
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let fdot: f64 = y_fused
                    .iter()
                    .zip(y_ref.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let fna: f64 = y_fused
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let fcos = fdot / (fna * nb);
                let gpu_max_abs = y_fused
                    .iter()
                    .zip(y_gpu.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "[online group={group:>2} n_rows={n_rows:>4} base={base_pos:>4} n_pos={n_pos:>4}] max|Δ|={fmax_abs:.2e}  cos={fcos:.7}  vs3k|Δ|={gpu_max_abs:.2e}"
                );
                assert!(
                    fcos > 0.9999,
                    "online matrix attn vs CPU ref cos too low (group={group} n_rows={n_rows} base={base_pos}): {fcos}"
                );
                assert!(
                    fmax_abs < 5e-3,
                    "online matrix attn vs CPU ref max|Δ| too high (group={group} n_rows={n_rows} base={base_pos}): {fmax_abs}"
                );
                assert!(
                    gpu_max_abs < 5e-3,
                    "online vs 3-kernel matrix attn diverged (group={group} n_rows={n_rows} base={base_pos}): {gpu_max_abs}"
                );

                if (n_rows, base_pos) == (100, 156) {
                    let query_cap = 32usize;
                    let tiled_scores =
                        MetalTensor::zeros_f16(&ctx, vec![(query_cap * n_q * n_pos) as u64])
                            .unwrap();
                    let tiled_ml = MetalTensor::zeros_f32(
                        &ctx,
                        vec![attn_matrix_ml_elems(query_cap, n_q, n_pos) as u64],
                    )
                    .unwrap();
                    let tiled_out =
                        MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
                    one_shot(&ctx, |enc| {
                        for row_base in (0..n_rows).step_by(query_cap) {
                            let rows_n = (n_rows - row_base).min(query_cap);
                            let q_rows = q_t.view_subrange(
                                (row_base * n_q * hd) as u64,
                                vec![(rows_n * n_q * hd) as u64],
                            );
                            let out_rows = tiled_out.view_subrange(
                                (row_base * n_q * hd) as u64,
                                vec![(rows_n * n_q * hd) as u64],
                            );
                            let scores_rows =
                                tiled_scores.view_subrange(0, vec![(rows_n * n_q * n_pos) as u64]);
                            let ml_rows = tiled_ml.view_subrange(
                                0,
                                vec![attn_matrix_ml_elems(rows_n, n_q, n_pos) as u64],
                            );
                            let tile_base_pos = base_pos + row_base;
                            encode_attn_matrix_kq_online_f32(
                                &ctx,
                                enc,
                                &q_rows,
                                &k_cache,
                                &scores_rows,
                                &ml_rows,
                                rows_n,
                                tile_base_pos,
                                n_pos,
                                kv_dim,
                                n_q,
                                n_kv,
                                group,
                                hd,
                                true,
                            )?;
                            encode_attn_matrix_kqv_norm_f32(
                                &ctx,
                                enc,
                                &scores_rows,
                                &ml_rows,
                                &v_t,
                                &out_rows,
                                rows_n,
                                tile_base_pos,
                                n_pos,
                                vt_stride,
                                n_q,
                                n_kv,
                                group,
                                hd,
                                true,
                            )?;
                        }
                        Ok(())
                    })
                    .unwrap();
                    let y_tiled = read_back_f32(&tiled_out.buffer, n_rows * n_q * hd);
                    let tiled_max_abs = y_tiled
                        .iter()
                        .zip(y_fused.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        tiled_max_abs < 5e-5,
                        concat!(
                            "tiled vs untiled online attention diverged ",
                            "(group={}): {}"
                        ),
                        group,
                        tiled_max_abs
                    );
                }
            }
        }
    }

    /// Kill-gate microbench for the two-pass online-softmax matrix attention
    /// kernels vs the three-kernel sidecar (KQ + softmax + KQV) at production
    /// chunk shapes. The promotion bar is >= 1.2x on the summed sidecar time.
    /// Vᵀ transpose/maintenance is excluded from both sides: both variants
    /// consume the same Vᵀ sidecar, so its upkeep cancels.
    ///
    /// `cargo test -p qwen-llm --release attn_matrix_online_vs_sidecar_microbench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attn_matrix_online_vs_sidecar_microbench() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let hd = 256usize;

        fn timed_gpu<F>(ctx: &MetalContext, iters: usize, encode: F) -> f64
        where
            F: Fn(&KernelEncoder) -> Result<(), MetalError>,
        {
            let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&cmd_buf);
            for _ in 0..iters {
                encode(&enc).unwrap();
            }
            enc.end();
            let t0 = std::time::Instant::now();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            t0.elapsed().as_secs_f64() / iters as f64
        }

        // (n_q, n_kv, n_rows, n_pos, label)
        let shapes: &[(usize, usize, usize, usize, &str)] = &[
            (16, 2, 1024, 4096, "G8/A3B chunk@pp4096"),
            (16, 2, 1024, 16384, "G8/A3B chunk@pp16384"),
            (24, 4, 1024, 4096, "G6/27B chunk@pp4096"),
            (24, 4, 1024, 16384, "G6/27B chunk@pp16384"),
            (32, 2, 1024, 1024, "G16/A10B chunk@pp1024"),
            (32, 2, 1024, 4096, "G16/A10B chunk@pp4096"),
        ];

        for &(n_q, n_kv, n_rows, n_pos, label) in shapes {
            let group = n_q / n_kv;
            let kv_dim = n_kv * hd;
            let base_pos = n_pos - n_rows;

            let q: Vec<f32> = (0..n_rows * n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let kv_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_rows * n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for dst in [&k_cache, &v_cache] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(kv_f32.as_slice()),
                    vec![kv_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, kv_f32.len())
                })
                .unwrap();
            }
            let vt_stride = n_pos;
            let v_t = MetalTensor::zeros_f16(&ctx, vec![(n_kv * hd * vt_stride) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_matrix_transpose_v_f16(
                    &ctx, enc, &v_cache, &v_t, 0, n_pos, n_pos, kv_dim, vt_stride, n_kv, hd,
                )
            })
            .unwrap();
            let scores =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
            let scores_h =
                MetalTensor::zeros_f16(&ctx, vec![(n_kv * n_rows * group * n_pos) as u64]).unwrap();
            let ml =
                MetalTensor::zeros_f32(&ctx, vec![attn_matrix_ml_elems(n_rows, n_q, n_pos) as u64])
                    .unwrap();
            let out_3k = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();
            let out_fused = MetalTensor::zeros_f32(&ctx, vec![(n_rows * n_q * hd) as u64]).unwrap();

            let encode_3k = |enc: &KernelEncoder| -> Result<(), MetalError> {
                encode_attn_matrix_kq_f32(
                    &ctx, enc, &q_t, &k_cache, &scores, n_rows, base_pos, n_pos, kv_dim, n_q, n_kv,
                    group, hd, true,
                )?;
                encode_attn_matrix_softmax_f32(
                    &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                )?;
                encode_attn_matrix_kqv_f32(
                    &ctx, enc, &scores, &v_t, &out_3k, n_rows, base_pos, n_pos, vt_stride, n_q,
                    n_kv, group, hd, true,
                )
            };
            let encode_fused = |enc: &KernelEncoder| -> Result<(), MetalError> {
                encode_attn_matrix_kq_online_f32(
                    &ctx, enc, &q_t, &k_cache, &scores_h, &ml, n_rows, base_pos, n_pos, kv_dim,
                    n_q, n_kv, group, hd, true,
                )?;
                encode_attn_matrix_kqv_norm_f32(
                    &ctx, enc, &scores_h, &ml, &v_t, &out_fused, n_rows, base_pos, n_pos,
                    vt_stride, n_q, n_kv, group, hd, true,
                )
            };

            // Warmup + correctness spot at production size.
            one_shot(&ctx, |enc| encode_3k(enc)).unwrap();
            one_shot(&ctx, |enc| encode_fused(enc)).unwrap();
            let y3k = read_back_f32(&out_3k.buffer, n_rows * n_q * hd);
            let yfused = read_back_f32(&out_fused.buffer, n_rows * n_q * hd);
            let max_abs = y3k
                .iter()
                .zip(yfused.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_abs < 5e-3,
                "fused vs sidecar diverged at {label}: max|Δ|={max_abs}"
            );

            let iters = 8usize;
            let mut t3k = f64::INFINITY;
            let mut tfused = f64::INFINITY;
            let mut tkq = f64::INFINITY;
            let mut tsm = f64::INFINITY;
            let mut tkqv = f64::INFINITY;
            for _ in 0..3 {
                t3k = t3k.min(timed_gpu(&ctx, iters, encode_3k));
                tfused = tfused.min(timed_gpu(&ctx, iters, encode_fused));
                tkq = tkq.min(timed_gpu(&ctx, iters, |enc| {
                    encode_attn_matrix_kq_f32(
                        &ctx, enc, &q_t, &k_cache, &scores, n_rows, base_pos, n_pos, kv_dim, n_q,
                        n_kv, group, hd, true,
                    )
                }));
                tsm = tsm.min(timed_gpu(&ctx, iters, |enc| {
                    encode_attn_matrix_softmax_f32(
                        &ctx, enc, &scores, n_rows, base_pos, n_pos, n_q, n_kv, group, hd,
                    )
                }));
                tkqv = tkqv.min(timed_gpu(&ctx, iters, |enc| {
                    encode_attn_matrix_kqv_f32(
                        &ctx, enc, &scores, &v_t, &out_3k, n_rows, base_pos, n_pos, vt_stride, n_q,
                        n_kv, group, hd, true,
                    )
                }));
            }
            eprintln!(
                "[{label:>22}] 3k={:8.3} ms (kq={:.3} sm={:.3} kqv={:.3})  online2p={:8.3} ms  ratio={:.2}x  vs|Δ|={max_abs:.2e}",
                t3k * 1e3,
                tkq * 1e3,
                tsm * 1e3,
                tkqv * 1e3,
                tfused * 1e3,
                t3k / tfused
            );
        }
    }

    /// Focused correctness gate for the A3B group-8 long-context subgroup path.
    ///
    /// Run in a fresh process with one of:
    ///
    /// - `QWEN_ATTN_V4_G8_TILE=4 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
    /// - `QWEN_ATTN_V4_G8_TILE=2 cargo test -p qwen-llm attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
    ///
    /// The env var is intentionally process-global (`OnceLock`) so this test stays
    /// ignored and single-purpose.
    #[test]
    #[ignore]
    fn attn_v4_group8_subgroup_matches_naive_f16kv() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let n_q = 16usize;
        let n_kv = 2usize;
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        for &(n_pos, nwg, tile_c) in &[(4096usize, 64usize, 64usize), (6144, 64, 64)] {
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let cap = n_pos;
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }

            let y_naive_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_f16kv_f32(
                    &ctx, enc, &q_t, &k_cache, &v_cache, &y_naive_t, n_q, n_kv, hd, n_pos,
                )
            })
            .unwrap();
            let y_naive = read_back_f32(&y_naive_t.buffer, n_q * hd);

            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * hd) as u64])
                    .unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * 2) as u64]).unwrap();
            let y_v4_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            one_shot(&ctx, |enc| {
                encode_attn_decode_v4_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial,
                    &ml_partial,
                    &y_v4_t,
                    n_q,
                    n_kv,
                    hd,
                    n_pos,
                    nwg,
                    tile_c,
                )
            })
            .unwrap();
            let y_v4 = read_back_f32(&y_v4_t.buffer, n_q * hd);
            let max_abs = y_v4
                .iter()
                .zip(y_naive.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let dot: f64 = y_v4
                .iter()
                .zip(y_naive.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = y_v4.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = y_naive
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb);
            eprintln!(
                "[v4-g8-subgroup n_pos={n_pos:>5} nwg={nwg:>2} C={tile_c:>2}] max|Δ|={max_abs:.2e} cos={cos:.6}"
            );
            assert!(
                cos > 0.9999,
                "group8 subgroup cos too low at n_pos={n_pos}: {cos}"
            );
            assert!(
                max_abs < 5e-3,
                "group8 subgroup max|Δ| too high at n_pos={n_pos}: {max_abs}"
            );
        }
    }

    /// Prompt-native packed-attention microproof for the A3B long-context shape.
    ///
    /// Compares the new packed multi-query microkernel against repeated
    /// decode-shaped `attn_v4` calls using the same subgroup setting
    /// (`g8_t2`) and the same F16 KV cache.
    #[test]
    #[ignore]
    fn attn_v4_prefill_g8_t2_q2_c64_vs_decode_loop() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        const N_Q: usize = 16;
        const N_KV: usize = 2;
        const HD: usize = 256;
        const N_ROWS: usize = 128;
        const NWG: usize = 64;
        const TILE_C: usize = 64;
        let kv_dim = N_KV * HD;

        for &base_pos in &[16384usize, 32768] {
            let n_pos = base_pos + N_ROWS;
            let q_rows: Vec<f32> = (0..N_ROWS * N_Q * HD)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..n_pos * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q_rows),
                vec![(N_ROWS * N_Q * HD) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }

            let out_baseline =
                MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HD) as u64]).unwrap();
            let out_packed =
                MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HD) as u64]).unwrap();
            let o_partial_row =
                MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * (N_Q / N_KV) * HD) as u64])
                    .unwrap();
            let ml_partial_row =
                MetalTensor::zeros_f32(&ctx, vec![(N_KV * NWG * (N_Q / N_KV) * 2) as u64]).unwrap();
            let o_partial_packed = MetalTensor::zeros_f32(
                &ctx,
                vec![(N_ROWS * N_KV * NWG * (N_Q / N_KV) * HD) as u64],
            )
            .unwrap();
            let ml_partial_packed =
                MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * (N_Q / N_KV) * 2) as u64])
                    .unwrap();

            let t = Instant::now();
            with_attn_v4_group_tile_override(2, || {
                one_shot(&ctx, |enc| {
                    for row in 0..N_ROWS {
                        let q_row =
                            q_t.view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                        let out_row = out_baseline
                            .view_subrange((row * N_Q * HD) as u64, vec![(N_Q * HD) as u64]);
                        encode_attn_decode_v4_f32(
                            &ctx,
                            enc,
                            &q_row,
                            &k_cache,
                            &v_cache,
                            &o_partial_row,
                            &ml_partial_row,
                            &out_row,
                            N_Q,
                            N_KV,
                            HD,
                            base_pos + row + 1,
                            NWG,
                            TILE_C,
                        )
                        .unwrap();
                    }
                    Ok(())
                })
            })
            .unwrap();
            let baseline_wall = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            one_shot(&ctx, |enc| {
                encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                    &ctx,
                    enc,
                    &q_t,
                    &k_cache,
                    &v_cache,
                    &o_partial_packed,
                    &ml_partial_packed,
                    &out_packed,
                    N_ROWS,
                    base_pos,
                    NWG,
                )
                .unwrap();
                Ok(())
            })
            .unwrap();
            let packed_wall = t.elapsed().as_secs_f64() * 1e3;

            let baseline = read_back_f32(&out_baseline.buffer, N_ROWS * N_Q * HD);
            let packed = read_back_f32(&out_packed.buffer, N_ROWS * N_Q * HD);
            let max_abs = packed
                .iter()
                .zip(baseline.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let dot: f64 = packed
                .iter()
                .zip(baseline.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = packed
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = baseline
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb);
            eprintln!(
                "[v4-prefill-a3b base_pos={base_pos:>5} rows={N_ROWS:>3}] decode_loop={baseline_wall:7.2} ms packed={packed_wall:7.2} ms speedup={:.3} max|Δ|={max_abs:.2e} cos={cos:.6}",
                baseline_wall / packed_wall
            );
            assert!(
                cos > 0.99999,
                "prefill packed cos too low at base_pos={base_pos}: {cos}"
            );
            assert!(
                max_abs < 2e-3,
                "prefill packed max|Δ| too high at base_pos={base_pos}: {max_abs}"
            );
        }
    }

    /// Bench: sweep NWG (split-K count) across context lengths to discover
    /// the optimal NWG for our shape on the host GPU. Compares against the
    /// naive f16kv kernel.
    ///
    /// Run with: `cargo test --release --lib -p qwen-llm attn_v4_nwg_sweep
    /// --ignored -- --nocapture`
    #[test]
    #[ignore]
    fn attn_v4_nwg_sweep() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-bench] {}", ctx.describe());

        let n_q = 24usize;
        let n_kv = 4usize;
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        const GROUP: usize = 6;

        let n_iters = 200usize; // chained dispatches per command buffer
        let warmup = 20usize;

        // Each context length we want to characterize.
        // Past 16K we skip naive_f16kv (cap'd) and only run v4 NWG sweep.
        for &n_pos in &[64usize, 256, 1024, 4096, 8192, 16384, 32768, 65536, 131072] {
            let cap = n_pos.max(64);
            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();

            // ----- Naive f16kv baseline (skip if past tg-mem cap ~7000) -----
            let naive_works = n_pos * std::mem::size_of::<f32>() <= 28 * 1024;
            if naive_works {
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_f16kv_f32(
                            &ctx, &enc, &q_t, &k_cache, &v_cache, &y_t, n_q, n_kv, hd, n_pos,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[n_pos={n_pos:>5} {label}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                // Warmup
                bench("naive_f16kv_warmup", warmup);
                bench("naive_f16kv       ", n_iters);
            } else {
                eprintln!("[n_pos={n_pos:>5} naive_f16kv       ] skipped (past tg-mem cap)");
            }

            // ----- v4: sweep NWG -----
            for &nwg in &[1usize, 2, 4, 8, 16, 32] {
                let o_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * hd) as u64]).unwrap();
                let ml_partial =
                    MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * 2) as u64]).unwrap();
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            &enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            32, // tile_c — NWG sweep holds tile constant
                        )
                        .unwrap();
                    }
                    enc.end();
                    let t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[n_pos={n_pos:>5} {label} nwg={nwg:>2}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                bench("v4_warmup           ", warmup);
                bench("v4                  ", n_iters);
            }
            eprintln!();
        }
    }

    /// Bench: sweep TILE-C (KV positions per inner softmax tile) at
    /// production NWG settings. Per Codex's review, GQA-dedup raises
    /// arithmetic intensity per K row, which may shift the optimal C
    /// away from llama.cpp's vec-kernel default of 32.
    ///
    /// Run with: `cargo test --release --lib -p qwen-llm
    /// attn_v4_tile_c_sweep -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn attn_v4_tile_c_sweep() {
        use std::time::Instant;
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        eprintln!("[v4-c-sweep] {}", ctx.describe());

        let n_q = 24usize;
        let n_kv = 4usize;
        let hd = 256usize;
        let kv_dim = n_kv * hd;
        const GROUP: usize = 6;

        let n_iters = 200usize;
        let warmup = 20usize;

        // For each ctx, use the production NWG heuristic (16 below 256, 32 above).
        for &n_pos in &[64usize, 256, 1024, 4096, 16384, 65536, 131072] {
            let nwg = if n_pos < 256 { 16usize } else { 32usize };
            let cap = n_pos.max(64);

            let q: Vec<f32> = (0..n_q * hd)
                .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
                .collect();
            let k_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
                .collect();
            let v_f32: Vec<f32> = (0..cap * kv_dim)
                .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
                .collect();

            let q_t = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&q),
                vec![(n_q * hd) as u64],
                GgmlType::F32,
            )
            .unwrap();
            let k_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            let v_cache = MetalTensor::zeros_f16(&ctx, vec![(cap * kv_dim) as u64]).unwrap();
            for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
                let src_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(src_f32.as_slice()),
                    vec![src_f32.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                one_shot(&ctx, |enc| {
                    encode_scatter_offset_f32_to_f16(&ctx, enc, &src_t, dst, 0, src_f32.len())
                })
                .unwrap();
            }
            let y_t = MetalTensor::zeros_f32(&ctx, vec![(n_q * hd) as u64]).unwrap();
            let o_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * hd) as u64]).unwrap();
            let ml_partial =
                MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * GROUP * 2) as u64]).unwrap();

            for &tile_c in &[16usize, 32, 64, 128] {
                let bench = |label: &str, n: usize| {
                    let cmd = ctx.queue.commandBuffer().expect("cmd");
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..n {
                        encode_attn_decode_v4_f32(
                            &ctx,
                            &enc,
                            &q_t,
                            &k_cache,
                            &v_cache,
                            &o_partial,
                            &ml_partial,
                            &y_t,
                            n_q,
                            n_kv,
                            hd,
                            n_pos,
                            nwg,
                            tile_c,
                        )
                        .unwrap();
                    }
                    enc.end();
                    let t = Instant::now();
                    cmd.commit();
                    cmd.waitUntilCompleted();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                    eprintln!(
                        "[n_pos={n_pos:>6} nwg={nwg:>2} C={tile_c:>2} {label}] {n}× chained: wall={wall:7.2} ms  gpu={gpu:7.2} ms  per-call={:6.3} ms",
                        gpu / n as f64
                    );
                };
                bench("warmup", warmup);
                bench("bench ", n_iters);
            }
            eprintln!();
        }
    }

    /// Attn-v4 decode bandwidth audit (2026-08-22): synthetic session at a
    /// large kv_n_pos, one `encode_attn_decode_v4_f32` call, kernel timing
    /// only. Reports achieved GB/s against the 474 GB/s stream so the
    /// long-context attention anomaly (serial ~130 GB/s, verify ~80 GB/s at
    /// 130K) can be attributed. Sweep with env:
    /// QWEN_ATTN_AUDIT_CTX (default 131072), QWEN_ATTN_AUDIT_MODEL
    /// (default Qwen3.8-27B-Q8_0), QWEN_ATTN_V4_NWG, QWEN_ATTN_V4_TILE_C.
    #[test]
    #[ignore = "slow real-model GPU audit; run explicitly"]
    fn attn_decode_v4_bandwidth_audit_130k() {
        let model_path = std::env::var("QWEN_ATTN_AUDIT_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.8-27B-Q8_0.gguf".into());
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("[attn-audit] skipped — model missing");
            return;
        }
        let n_pos: usize = std::env::var("QWEN_ATTN_AUDIT_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(131_072);
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init failed: {e}"),
        };
        let g = crate::gguf::GgufFile::open(&model_path).expect("open model");
        let m = crate::loader::Model::from_gguf(&g).expect("load model");
        let mm = crate::metal_forward::MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mut sess =
            crate::metal_forward::MetalSession::fresh(&ctx, &mm, n_pos + 16).expect("session");
        for kp in sess.kv_n_pos.iter_mut() {
            *kp = n_pos;
        }
        fill_audit_f16(&sess.kv_k[0], 1);
        fill_audit_f16(&sess.kv_v[0], 2);
        let arch = &m.arch;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let group = n_q / n_kv;
        if head_dim != 256 || !matches!(group, 4 | 6 | 8 | 16) {
            eprintln!("[attn-audit] skipped — unsupported shape");
            return;
        }
        let nwg = crate::metal::attn_v4_choose_nwg(n_pos, group);
        let tile_c = crate::metal::attn_v4_choose_tile_c(n_pos, group);
        let q = MetalTensor::zeros_f32(&ctx, vec![(n_q * head_dim) as u64]).expect("q");
        let attn_o = MetalTensor::zeros_f32(&ctx, vec![(n_q * head_dim) as u64]).expect("o");
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_attn_decode_v4_f32(
            &ctx,
            &enc,
            &q,
            &sess.kv_k[0],
            &sess.kv_v[0],
            &sess.attn_v4_o_partial,
            &sess.attn_v4_ml_partial,
            &attn_o,
            n_q,
            n_kv,
            head_dim,
            n_pos,
            nwg,
            tile_c,
        )
        .expect("encode attn v4");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        let bytes = n_pos as f64 * (n_kv * head_dim * 2) as f64 * 2.0;
        let gbps = bytes / 1e9 / (ms / 1e3);
        eprintln!(
            "[attn-audit] ctx={n_pos} group={group} nwg={nwg} tile_c={tile_c} gpu_ms={ms:.3} gb={:.2} gbps={gbps:.1}",
            bytes / 1e9
        );
    }
}

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

/// F16 KV-cache variant of `encode_attn_decode_f32`. Same algorithm,
/// reads K and V as half-precision. Halves attention bandwidth at long
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

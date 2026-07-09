//! Metal MTP (Multi-Token-Prediction) head and speculative-decode driver.
//!
//! Mirrors `crate::forward::Forward::mtp_step` (CPU oracle) on Metal.
//! See `docs/H4-MTP.md` §1.2 for the algorithmic contract:
//!
//! At sequence slot `position`, the MTP head consumes
//! `(embed(next_tok), prev_hidden, position)` and predicts logits for the
//! token at `position + 2`. Forward path:
//!
//! ```text
//! e_normed = RMSNorm(embed(next_tok), enorm)
//! h_normed = RMSNorm(prev_hidden, hnorm)
//! x        = eh_proj @ concat([e_normed, h_normed])     # 2H → H, vLLM order
//! x       += attn_block(MTP attn, x, position, mtp_kv)  # standard gated attn
//! x       += SwiGLU FFN(x)
//! logits   = lm_head(RMSNorm(x, shared_head_norm))      # shared lm_head
//! ```
//!
//! v1 ships `draft` (greedy logits + token id) and `draft_kv_only`
//! (logits discarded; advances MTP KV by one slot — used by both the
//! inline accept-branch bridge in §1.4 step E and the streamed
//! prompt-time prefill in §1.5).
//!
//! GPU command ordering: `draft` and `draft_kv_only` synchronously commit
//! and wait per call (v1). Future ICB-cached version will overlap with
//! base forward on the same MetalSession command queue.

use crate::codec::dequant_to_f32;
use crate::gguf::GgufFile;
use crate::loader::MtpHead;
use crate::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalError, MetalTensor, attn_v4_choose_nwg,
    attn_v4_choose_tile_c, encode_add_inplace_f32, encode_argmax_f32, encode_attn_decode_f16kv_f32,
    encode_attn_decode_v4_f32, encode_axpy_scalar_f32, encode_ffn_swiglu_q4_K_f32,
    encode_get_rows_f32, encode_moe_down_f32_f32,
    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2, encode_moe_mat_vec_f32,
    encode_moe_swiglu_q4_K_f32, encode_moe_weighted_sum_f32, encode_mtp_draft_affine_q4_gs64_f32,
    encode_mul_f32, encode_rms_norm_batched_f32, encode_rms_norm_mul_f32, encode_rope_neox_f32,
    encode_scatter_offset_f32_to_f16_kv, encode_sigmoid_f32, encode_silu_mul_f32,
    encode_split_q_gate_f32, encode_topk_logits_softmax_dot_sigmoid_f32,
};
use crate::metal_dflash::{
    MetalDFlashLayerMajorScratch, MetalDFlashVerifyScratch, encode_packed_verify_layer_major_inner,
    encode_restore_after_partial_accept_inner,
};
use crate::metal_forward::{
    ATTN_V4_MAX_NWG, MetalAttnBlock, MetalForward, MetalMoeFfn, MetalSession, RMS_EPS,
    checked_u64_div_exact, checked_u64_double, checked_u64_mul, checked_u64_mul4,
    encode_mat_vec_dispatch, encode_scatter_offset_f32, weight_dtype_kept_native,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

#[derive(Debug, thiserror::Error)]
pub enum MtpError {
    #[error("metal: {0}")]
    Metal(#[from] MetalError),
    #[error("metal forward: {0}")]
    MetalForward(#[from] crate::metal_forward::MfError),
    #[error("codec: {0}")]
    Codec(#[from] crate::codec::CodecError),
    #[error("dflash: {0}")]
    DFlash(#[from] crate::metal_dflash::DFlashError),
    #[error("model has no MTP head")]
    NoMtpHead,
    #[error("token {0} out of vocab range {1}")]
    BadToken(i32, u32),
    #[error("invalid QWEN_MTP_MOE_NATIVE_BANKS value {0:?}; expected 0, gate_up, down, 1, or all")]
    InvalidMoeBankPolicy(String),
    #[error(
        "MTP MoE bank policy {policy:?} does not support source gate/up/down={gate:?}/{up:?}/{down:?} shapes={gate_shape:?}/{up_shape:?}/{down_shape:?}"
    )]
    UnsupportedMoeBankPolicy {
        policy: MtpMoeBankPolicy,
        gate: GgmlType,
        up: GgmlType,
        down: GgmlType,
        gate_shape: Vec<u64>,
        up_shape: Vec<u64>,
        down_shape: Vec<u64>,
    },
    #[error("MTP KV invariant: caller passed position={position} but kv_n_pos={kv_n_pos}")]
    KvPositionMismatch { position: u32, kv_n_pos: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpMoeBankPolicy {
    F32,
    GateUp,
    Down,
    All,
}

impl MtpMoeBankPolicy {
    fn parse(raw: Option<&str>) -> Result<Self, MtpError> {
        match raw {
            None | Some("0") => Ok(Self::F32),
            Some("gate_up") => Ok(Self::GateUp),
            Some("down") => Ok(Self::Down),
            Some("1" | "all") => Ok(Self::All),
            Some(other) => Err(MtpError::InvalidMoeBankPolicy(other.to_string())),
        }
    }

    fn from_env() -> Result<Self, MtpError> {
        match std::env::var("QWEN_MTP_MOE_NATIVE_BANKS") {
            Ok(raw) => Self::parse(Some(&raw)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(std::env::VarError::NotUnicode(raw)) => Err(MtpError::InvalidMoeBankPolicy(
                raw.to_string_lossy().into_owned(),
            )),
        }
    }

    fn native_gate_up(self) -> bool {
        matches!(self, Self::GateUp | Self::All)
    }

    fn native_down(self) -> bool {
        matches!(self, Self::Down | Self::All)
    }
}

/// All MTP head weights, resident as `MetalTensor`s. Loaded once at session
/// start. Shape conventions match `crate::loader::MtpHead`.
pub struct MetalMtpHead {
    /// Block index in the GGUF (typically `arch.n_layer`).
    pub block_idx: u32,
    /// Standard full-attention weights at `blk.{block_idx}.*`.
    pub attn: MetalAttnBlock,
    /// `nextn.eh_proj.weight` — `[2H, H]`.
    pub eh_proj: MetalTensor,
    /// `nextn.enorm.weight` — `[H]`.
    pub enorm: MetalTensor,
    /// `nextn.hnorm.weight` — `[H]`.
    pub hnorm: MetalTensor,
    /// `nextn.shared_head_norm.weight` — `[H]`.
    pub shared_head_norm: MetalTensor,
    pub moe_bank_policy: MtpMoeBankPolicy,
}

#[derive(Clone)]
pub struct MtpDraftAffineQ4Head {
    pub weight: MetalTensor,
    pub scales: MetalTensor,
    pub biases: MetalTensor,
    pub n_in: usize,
    pub n_out: usize,
}

pub fn quantize_lm_head_to_q4_1(
    ctx: &MetalContext,
    src: &MetalTensor,
) -> Result<MetalTensor, MtpError> {
    if src.shape.len() != 2 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_1",
            detail: format!("expected rank-2 lm_head, got {:?}", src.shape),
        }));
    }
    let n_in = usize::try_from(src.shape[0]).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_1",
            detail: format!("n_in overflows usize: {}", src.shape[0]),
        })
    })?;
    let n_out = usize::try_from(src.shape[1]).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_1",
            detail: format!("n_out overflows usize: {}", src.shape[1]),
        })
    })?;
    if n_in % 32 != 0 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_1",
            detail: format!("n_in={n_in} is not divisible by 32"),
        }));
    }

    let n_bytes = usize::try_from(src.n_bytes()).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_1",
            detail: format!("source byte length overflows usize: {}", src.n_bytes()),
        })
    })?;
    let bytes = unsafe {
        let ptr = (src.buffer.contents().as_ptr() as *const u8).add(src.offset as usize);
        std::slice::from_raw_parts(ptr, n_bytes)
    };
    let desc = TensorDesc {
        name: "mtp_draft_lm_head_q4_1_src".to_string(),
        shape: src.shape.clone(),
        dtype: src.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: src.n_bytes(),
    };
    let f32 = dequant_to_f32(&desc, bytes)?;
    let blocks_per_row = n_in / 32;
    let mut out = vec![0u8; n_out * blocks_per_row * 20];
    for row in 0..n_out {
        let row_base = row * n_in;
        for block in 0..blocks_per_row {
            let src_base = row_base + block * 32;
            let dst_base = (row * blocks_per_row + block) * 20;
            let vals = &f32[src_base..src_base + 32];
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &v in vals {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            let d = if max_v > min_v {
                (max_v - min_v) / 15.0
            } else {
                0.0
            };
            let m = min_v;
            out[dst_base..dst_base + 2]
                .copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            out[dst_base + 2..dst_base + 4]
                .copy_from_slice(&half::f16::from_f32(m).to_bits().to_le_bytes());
            for i in 0..16 {
                let q0 = if d > 0.0 {
                    ((vals[i] - m) / d).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let q1 = if d > 0.0 {
                    ((vals[i + 16] - m) / d).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                out[dst_base + 4 + i] = q0 | (q1 << 4);
            }
        }
    }
    Ok(MetalTensor::from_bytes(
        ctx,
        &out,
        src.shape.clone(),
        GgmlType::Q4_1,
    )?)
}

pub fn quantize_lm_head_to_q4_0(
    ctx: &MetalContext,
    src: &MetalTensor,
) -> Result<MetalTensor, MtpError> {
    if src.shape.len() != 2 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_0",
            detail: format!("expected rank-2 lm_head, got {:?}", src.shape),
        }));
    }
    let n_in = usize::try_from(src.shape[0]).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_0",
            detail: format!("n_in overflows usize: {}", src.shape[0]),
        })
    })?;
    let n_out = usize::try_from(src.shape[1]).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_0",
            detail: format!("n_out overflows usize: {}", src.shape[1]),
        })
    })?;
    if n_in % 32 != 0 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_0",
            detail: format!("n_in={n_in} is not divisible by 32"),
        }));
    }

    let n_bytes = usize::try_from(src.n_bytes()).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel: "mtp_draft_lm_head_q4_0",
            detail: format!("source byte length overflows usize: {}", src.n_bytes()),
        })
    })?;
    let bytes = unsafe {
        let ptr = (src.buffer.contents().as_ptr() as *const u8).add(src.offset as usize);
        std::slice::from_raw_parts(ptr, n_bytes)
    };
    let desc = TensorDesc {
        name: "mtp_draft_lm_head_q4_0_src".to_string(),
        shape: src.shape.clone(),
        dtype: src.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: src.n_bytes(),
    };
    let f32 = dequant_to_f32(&desc, bytes)?;
    let blocks_per_row = n_in / 32;
    let mut out = vec![0u8; n_out * blocks_per_row * 18];
    for row in 0..n_out {
        let row_base = row * n_in;
        for block in 0..blocks_per_row {
            let src_base = row_base + block * 32;
            let dst_base = (row * blocks_per_row + block) * 18;
            let vals = &f32[src_base..src_base + 32];
            let mut amax = 0.0f32;
            for &v in vals {
                amax = amax.max(v.abs());
            }
            let d = if amax > 0.0 { amax / 7.0 } else { 0.0 };
            out[dst_base..dst_base + 2]
                .copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for i in 0..16 {
                let q0 = if d > 0.0 {
                    (vals[i] / d).round().clamp(-8.0, 7.0) as i32 + 8
                } else {
                    8
                };
                let q1 = if d > 0.0 {
                    (vals[i + 16] / d).round().clamp(-8.0, 7.0) as i32 + 8
                } else {
                    8
                };
                out[dst_base + 2 + i] = (q0 as u8) | ((q1 as u8) << 4);
            }
        }
    }
    Ok(MetalTensor::from_bytes(
        ctx,
        &out,
        src.shape.clone(),
        GgmlType::Q4_0,
    )?)
}

pub fn quantize_lm_head_to_affine_q4_gs64(
    ctx: &MetalContext,
    src: &MetalTensor,
) -> Result<MtpDraftAffineQ4Head, MtpError> {
    const GROUP_SIZE: usize = 64;
    const PACK_FACTOR: usize = 8;
    let kernel = "mtp_draft_lm_head_affine_q4_gs64";
    if src.shape.len() != 2 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel,
            detail: format!("expected rank-2 lm_head, got {:?}", src.shape),
        }));
    }
    let n_in = usize::try_from(src.shape[0]).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel,
            detail: format!("n_in overflows usize: {}", src.shape[0]),
        })
    })?;
    let n_out = usize::try_from(src.shape[1]).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel,
            detail: format!("n_out overflows usize: {}", src.shape[1]),
        })
    })?;
    if n_in % GROUP_SIZE != 0 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel,
            detail: format!("n_in={n_in} is not divisible by {GROUP_SIZE}"),
        }));
    }

    let n_bytes = usize::try_from(src.n_bytes()).map_err(|_| {
        MtpError::Metal(MetalError::BadShape {
            kernel,
            detail: format!("source byte length overflows usize: {}", src.n_bytes()),
        })
    })?;
    let bytes = unsafe {
        let ptr = (src.buffer.contents().as_ptr() as *const u8).add(src.offset as usize);
        std::slice::from_raw_parts(ptr, n_bytes)
    };
    let desc = TensorDesc {
        name: "mtp_draft_lm_head_affine_q4_gs64_src".to_string(),
        shape: src.shape.clone(),
        dtype: src.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: src.n_bytes(),
    };
    let f32 = dequant_to_f32(&desc, bytes)?;
    let packs_per_row = n_in / PACK_FACTOR;
    let groups_per_row = n_in / GROUP_SIZE;
    let mut packed = vec![0u8; n_out * packs_per_row * std::mem::size_of::<u32>()];
    let mut scales = vec![0u8; n_out * groups_per_row * std::mem::size_of::<half::f16>()];
    let mut biases = vec![0u8; n_out * groups_per_row * std::mem::size_of::<half::f16>()];

    for row in 0..n_out {
        let row_base = row * n_in;
        for group in 0..groups_per_row {
            let group_base = row_base + group * GROUP_SIZE;
            let vals = &f32[group_base..group_base + GROUP_SIZE];
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for &v in vals {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            let scale = if max_v > min_v {
                (max_v - min_v) / 15.0
            } else {
                0.0
            };
            let group_dst = row * groups_per_row + group;
            scales[group_dst * 2..group_dst * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            biases[group_dst * 2..group_dst * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(min_v).to_bits().to_le_bytes());

            for pack_in_group in 0..(GROUP_SIZE / PACK_FACTOR) {
                let mut word = 0u32;
                let pack_base = group * GROUP_SIZE / PACK_FACTOR + pack_in_group;
                for j in 0..PACK_FACTOR {
                    let v = vals[pack_in_group * PACK_FACTOR + j];
                    let q = if scale > 0.0 {
                        ((v - min_v) / scale).round().clamp(0.0, 15.0) as u32
                    } else {
                        0
                    };
                    word |= q << (4 * j);
                }
                let dst = (row * packs_per_row + pack_base) * std::mem::size_of::<u32>();
                packed[dst..dst + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }

    Ok(MtpDraftAffineQ4Head {
        weight: MetalTensor::from_bytes(
            ctx,
            &packed,
            vec![n_out as u64, packs_per_row as u64],
            GgmlType::F32,
        )?,
        scales: MetalTensor::from_bytes(
            ctx,
            &scales,
            vec![n_out as u64, groups_per_row as u64],
            GgmlType::F16,
        )?,
        biases: MetalTensor::from_bytes(
            ctx,
            &biases,
            vec![n_out as u64, groups_per_row as u64],
            GgmlType::F16,
        )?,
        n_in,
        n_out,
    })
}

impl MetalMtpHead {
    /// Load the MTP head's weights from the bound `MtpHead` view. Mirrors
    /// `MetalModel::load`'s native-quant policy: kernel-supported quants
    /// (Q4_K, Q5_K, Q6_K, F32) stay in their on-disk dtype; norms +
    /// elementwise weights are dequant'd to F32.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile, mtp: &MtpHead<'_>) -> Result<Self, MtpError> {
        Self::load_with_moe_bank_policy(ctx, gguf, mtp, MtpMoeBankPolicy::from_env()?)
    }

    pub fn load_with_moe_bank_policy(
        ctx: &MetalContext,
        gguf: &GgufFile,
        mtp: &MtpHead<'_>,
        moe_bank_policy: MtpMoeBankPolicy,
    ) -> Result<Self, MtpError> {
        let load_f32 = |desc: &TensorDesc| -> Result<MetalTensor, MtpError> {
            if desc.dtype == GgmlType::F32 {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                let f32 = dequant_to_f32(desc, gguf.slice(desc))?;
                Ok(MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&f32),
                    desc.shape.clone(),
                    GgmlType::F32,
                )?)
            }
        };
        let load_weight = |desc: &TensorDesc| -> Result<MetalTensor, MtpError> {
            if weight_dtype_kept_native(desc.dtype) {
                Ok(MetalTensor::from_gguf_tensor(ctx, desc, gguf.slice(desc))?)
            } else {
                load_f32(desc)
            }
        };
        let load_moe = |moe: &crate::loader::MoeFfn<'_>| -> Result<MetalMoeFfn, MtpError> {
            if (moe_bank_policy.native_gate_up()
                && (moe.gate_exps.dtype != GgmlType::Q4_K
                    || moe.up_exps.dtype != GgmlType::Q4_K
                    || moe.gate_exps.shape.as_slice() != [2048, 512, 256]
                    || moe.up_exps.shape.as_slice() != [2048, 512, 256]))
                || (moe_bank_policy.native_down()
                    && (moe.down_exps.dtype != GgmlType::Q5_K
                        || moe.down_exps.shape.as_slice() != [512, 2048, 256]))
            {
                return Err(MtpError::UnsupportedMoeBankPolicy {
                    policy: moe_bank_policy,
                    gate: moe.gate_exps.dtype,
                    up: moe.up_exps.dtype,
                    down: moe.down_exps.dtype,
                    gate_shape: moe.gate_exps.shape.clone(),
                    up_shape: moe.up_exps.shape.clone(),
                    down_shape: moe.down_exps.shape.clone(),
                });
            }
            Ok(MetalMoeFfn {
                gate_inp: load_f32(moe.gate_inp)?,
                gate_exps: if moe_bank_policy.native_gate_up() {
                    load_weight(moe.gate_exps)?
                } else {
                    load_f32(moe.gate_exps)?
                },
                up_exps: if moe_bank_policy.native_gate_up() {
                    load_weight(moe.up_exps)?
                } else {
                    load_f32(moe.up_exps)?
                },
                down_exps: if moe_bank_policy.native_down() {
                    load_weight(moe.down_exps)?
                } else {
                    load_f32(moe.down_exps)?
                },
                gate_inp_shexp: load_f32(moe.gate_inp_shexp)?,
                gate_inp_cpu: dequant_to_f32(moe.gate_inp, gguf.slice(moe.gate_inp))?,
                gate_inp_shexp_cpu: dequant_to_f32(
                    moe.gate_inp_shexp,
                    gguf.slice(moe.gate_inp_shexp),
                )?,
            })
        };

        Ok(Self {
            block_idx: mtp.block_idx,
            attn: MetalAttnBlock {
                attn_norm: load_f32(mtp.attn.attn_norm)?,
                post_attn_norm: load_f32(mtp.attn.post_attention_norm)?,
                ffn_gate: load_weight(mtp.attn.ffn_gate)?,
                ffn_up: load_weight(mtp.attn.ffn_up)?,
                ffn_down: load_weight(mtp.attn.ffn_down)?,
                q: load_weight(mtp.attn.q)?,
                k: load_weight(mtp.attn.k)?,
                v: load_weight(mtp.attn.v)?,
                qkv_fused: None,
                o: load_weight(mtp.attn.o)?,
                q_norm: load_f32(mtp.attn.q_norm)?,
                k_norm: load_f32(mtp.attn.k_norm)?,
                ffn_moe: mtp.attn.ffn_moe.as_ref().map(load_moe).transpose()?,
            },
            eh_proj: load_weight(mtp.eh_proj)?,
            enorm: load_f32(mtp.enorm)?,
            hnorm: load_f32(mtp.hnorm)?,
            shared_head_norm: load_f32(mtp.shared_head_norm)?,
            moe_bank_policy,
        })
    }
}

/// Per-sequence MTP state: dedicated KV ring for the single MTP attn
/// layer + scratch buffers for the per-step forward. Independent of the
/// base `MetalSession` so the MTP draft can run without disturbing base
/// session arena.
pub struct MetalMtpSession {
    /// MTP attn KV ring. Single layer (the MTP block's own attn).
    /// F16 storage matching the base attn KV convention.
    pub kv_k: MetalTensor,
    pub kv_v: MetalTensor,
    pub kv_n_pos: usize,
    pub kv_capacity: usize,

    // Pre-eh_proj scratch.
    pub e: MetalTensor,         // [H] — embed(next_tok)
    pub e_normed: MetalTensor,  // [H] — RMSNorm(e, enorm)
    pub h_normed: MetalTensor,  // [H] — RMSNorm(prev_hidden, hnorm)
    pub eh_concat: MetalTensor, // [2H] — concat([e_normed, h_normed])

    // Residual stream + post-norm activations (mirror base session).
    pub x: MetalTensor,         // [H] — residual stream
    pub h: MetalTensor,         // [H] — post-norm activation
    pub ffn_gate: MetalTensor,  // [F]
    pub ffn_up: MetalTensor,    // [F]
    pub ffn_inner: MetalTensor, // [F] — silu(gate) * up
    pub ffn_out: MetalTensor,   // [H]
    pub mixer_out: MetalTensor, // [H] — attn output
    pub moe_router_probs: MetalTensor,
    pub moe_topk_idx: MetalTensor,
    pub moe_topk_weight: MetalTensor,
    pub moe_shared_gate: MetalTensor,
    pub moe_expert_out: MetalTensor,

    // Attn scratch.
    pub attn_q_full: MetalTensor,   // [2 * q_dim] — Q + gate interleaved
    pub attn_q: MetalTensor,        // [q_dim]
    pub attn_gate: MetalTensor,     // [q_dim]
    pub attn_q_normed: MetalTensor, // [q_dim]
    pub attn_k_now: MetalTensor,    // [kv_dim]
    pub attn_v_now: MetalTensor,    // [kv_dim]
    pub attn_k_normed: MetalTensor, // [kv_dim]
    pub attn_o: MetalTensor,        // [q_dim]
    pub attn_v4_o_partial: MetalTensor, // n_kv * NWG_max * GROUP * head_dim
    pub attn_v4_ml_partial: MetalTensor, // n_kv * NWG_max * GROUP * 2

    // Output (shared with base — but kept separate to allow concurrent
    // dispatch in a future ICB world).
    pub logits: MetalTensor,       // [V]
    pub draft_argmax: MetalTensor, // [1] i32 in F32 buffer
    pub draft_ids: MetalTensor,    // [16] i32 in F32 buffer
    pub ids_buf: MetalTensor,      // i32 token id (in F32 buffer)
}

impl MetalMtpSession {
    /// Allocate a fresh MTP session. `kv_capacity` is the max number of
    /// MTP KV slots (typically `prompt_len + max_new_tokens`).
    pub fn fresh(
        ctx: &MetalContext,
        head: &MetalMtpHead,
        arch: &crate::model::Arch,
        kv_capacity: usize,
    ) -> Result<Self, MtpError> {
        let h = arch.hidden_size as u64;
        let mtp_moe = head.attn.ffn_moe.is_some();
        let topk = arch.expert_used_count.max(1).min(arch.expert_count.max(1)) as u64;
        let f_exp = arch.expert_feed_forward_length as u64;
        let f_shared = arch.expert_shared_feed_forward_length as u64;
        let f = if mtp_moe {
            checked_u64_mul(topk, f_exp.max(1), "mtp moe ffn scratch overflow")?
                .max(f_shared.max(1))
        } else {
            arch.intermediate_size as u64
        };
        let head_dim = arch.attn_head_dim as u64;
        let n_q = arch.n_q_heads as u64;
        let n_kv = arch.n_kv_heads as u64;
        let q_dim = checked_u64_mul(n_q, head_dim, "mtp q_dim overflow")?;
        let kv_dim = checked_u64_mul(n_kv, head_dim, "mtp kv_dim overflow")?;
        let q_group = checked_u64_div_exact(n_q, n_kv, "mtp n_q_heads / n_kv_heads invalid")?;
        let kv_cache_elems =
            checked_u64_mul(kv_capacity as u64, kv_dim, "mtp kv cache size overflow")?;
        let eh_concat_elems = checked_u64_double(h, "mtp 2 * hidden_size overflow")?;
        let attn_q_full_elems = checked_u64_double(q_dim, "mtp 2 * q_dim overflow")?;
        let attn_v4_o_partial_elems = checked_u64_mul4(
            n_kv,
            ATTN_V4_MAX_NWG as u64,
            q_group,
            head_dim,
            "mtp attn_v4_o_partial size overflow",
        )?;
        let attn_v4_ml_partial_elems = checked_u64_mul4(
            n_kv,
            ATTN_V4_MAX_NWG as u64,
            q_group,
            2,
            "mtp attn_v4_ml_partial size overflow",
        )?;
        let moe_router_n = if mtp_moe {
            arch.expert_count.max(1) as u64
        } else {
            1
        };
        let moe_topk_n = if mtp_moe { topk } else { 1 };
        let moe_expert_out_n = if mtp_moe {
            checked_u64_mul(topk, h, "mtp moe expert out overflow")?
        } else {
            1
        };

        Ok(Self {
            kv_k: MetalTensor::zeros_f16(ctx, vec![kv_cache_elems])?,
            kv_v: MetalTensor::zeros_f16(ctx, vec![kv_cache_elems])?,
            kv_n_pos: 0,
            kv_capacity,
            e: MetalTensor::zeros_f32(ctx, vec![h])?,
            e_normed: MetalTensor::zeros_f32(ctx, vec![h])?,
            h_normed: MetalTensor::zeros_f32(ctx, vec![h])?,
            eh_concat: MetalTensor::zeros_f32(ctx, vec![eh_concat_elems])?,
            x: MetalTensor::zeros_f32(ctx, vec![h])?,
            h: MetalTensor::zeros_f32(ctx, vec![h])?,
            ffn_gate: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_up: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_inner: MetalTensor::zeros_f32(ctx, vec![f])?,
            ffn_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            mixer_out: MetalTensor::zeros_f32(ctx, vec![h])?,
            moe_router_probs: MetalTensor::zeros_f32(ctx, vec![moe_router_n])?,
            moe_topk_idx: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_topk_weight: MetalTensor::zeros_f32(ctx, vec![moe_topk_n])?,
            moe_shared_gate: MetalTensor::zeros_f32(ctx, vec![1])?,
            moe_expert_out: MetalTensor::zeros_f32(ctx, vec![moe_expert_out_n])?,
            attn_q_full: MetalTensor::zeros_f32(ctx, vec![attn_q_full_elems])?,
            attn_q: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_gate: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_q_normed: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_k_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_v_now: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_k_normed: MetalTensor::zeros_f32(ctx, vec![kv_dim])?,
            attn_o: MetalTensor::zeros_f32(ctx, vec![q_dim])?,
            attn_v4_o_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_o_partial_elems])?,
            attn_v4_ml_partial: MetalTensor::zeros_f32(ctx, vec![attn_v4_ml_partial_elems])?,
            logits: MetalTensor::zeros_f32(ctx, vec![arch.vocab_size as u64])?,
            draft_argmax: MetalTensor::zeros_f32(ctx, vec![1])?,
            draft_ids: MetalTensor::zeros_f32(ctx, vec![16])?,
            ids_buf: MetalTensor::zeros_f32(ctx, vec![1])?,
        })
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftReadback {
    None,
    ArgmaxOnly,
    FullLogits,
}

#[cfg_attr(not(test), allow(dead_code))]
struct DraftResult {
    logits: Option<Vec<f32>>,
    argmax: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
pub enum PackedDraftPlan<'a> {
    /// Replay draft token vectors recorded from the native MTP drafter.
    Recorded(&'a [RecordedDraftStep]),
    /// Use the no-spec greedy target stream as a perfect draft oracle.
    Oracle(&'a [i32]),
}

#[derive(Clone, Debug)]
pub struct RecordedDraftStep {
    pub carry_tok: i32,
    /// Base-model packed verify start position for this speculative step.
    /// The native drafter may use a different start position for its own
    /// KV history (for committed history this is one slot earlier), but
    /// replay probes consume recorded draft ids in the verifier timeline.
    pub start_position: u32,
    pub drafts: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordedMtpWork {
    /// Run recursive MTP bodies with recorded ids, but skip draft lm_head/argmax.
    BodyNoLmHead,
    /// Only maintain canonical MTP KV for the carry/accepted-prefix bridges.
    BridgeOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpRecursiveHiddenVariant {
    /// Feed the residual stream before MTP shared-head norm into the next draft.
    PreNorm,
    /// Feed the MTP shared-head-normalized hidden into the next draft. MTPLX's
    /// default native-MTP contract uses this variant.
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpBaseHiddenVariant {
    /// Feed the base residual stream before final output norm into MTP.
    PreNorm,
    /// Feed the base final-output-normalized hidden into MTP. MTPLX's default
    /// Qwen contract uses this variant.
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpHistoryMode {
    /// Maintain a committed MTP KV history alongside the target cache.
    Committed,
    /// Keep accepted draft-chain KV instead of repairing it from target hiddens.
    DraftAccepted,
    /// Match MTPLX's cycle-style draft cache: each speculative step starts with
    /// an empty MTP KV cache, then keeps only the within-chain draft keys.
    Cycle,
}

#[derive(Clone, Debug)]
pub struct MtpRankRow {
    pub step: usize,
    pub depth: usize,
    pub rank: usize,
    pub accepted: bool,
    pub draft_tok: i32,
    pub target_tok: i32,
    pub target_logit: f32,
    pub top_tokens: Vec<i32>,
    pub top_logits: Vec<f32>,
}

fn rank_and_topk(logits: &[f32], target_idx: usize, k: usize) -> (usize, Vec<i32>, Vec<f32>) {
    let target_logit = logits[target_idx];
    let mut rank = 1usize;
    let mut top_tokens = vec![-1i32; k];
    let mut top_logits = vec![f32::NEG_INFINITY; k];
    for (idx, &v) in logits.iter().enumerate() {
        if v > target_logit {
            rank += 1;
        }
        if v <= top_logits[k - 1] {
            continue;
        }
        let mut pos = k - 1;
        while pos > 0 && v > top_logits[pos - 1] {
            top_logits[pos] = top_logits[pos - 1];
            top_tokens[pos] = top_tokens[pos - 1];
            pos -= 1;
        }
        top_logits[pos] = v;
        top_tokens[pos] = idx as i32;
    }
    (rank, top_tokens, top_logits)
}

fn copy_f32_tensor(src: &MetalTensor, dst: &MetalTensor) -> Result<(), MtpError> {
    if src.dtype != GgmlType::F32 || dst.dtype != GgmlType::F32 {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "copy_f32_tensor",
            detail: format!("expected F32/F32, got {:?}/{:?}", src.dtype, dst.dtype),
        }));
    }
    if src.n_elements() != dst.n_elements() {
        return Err(MtpError::Metal(MetalError::BadShape {
            kernel: "copy_f32_tensor",
            detail: format!(
                "element mismatch: src={} dst={}",
                src.n_elements(),
                dst.n_elements()
            ),
        }));
    }
    let n_bytes = (src.n_elements() as usize) * std::mem::size_of::<f32>();
    unsafe {
        let src_ptr = (src.buffer.contents().as_ptr() as *const u8).add(src.offset as usize);
        let dst_ptr = (dst.buffer.contents().as_ptr() as *mut u8).add(dst.offset as usize);
        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, n_bytes);
    }
    Ok(())
}

/// Top-level speculative-decode driver. Owns nothing the base path needs;
/// borrows base for the actual base-forward calls.
pub struct SpeculativeDecoder<'a> {
    pub base: &'a MetalForward<'a>,
    pub mtp_head: &'a MetalMtpHead,
    pub mtp_session: MetalMtpSession,
    draft_token_embd_head: bool,
    draft_lm_head_override: Option<MetalTensor>,
    draft_affine_q4_head_override: Option<MtpDraftAffineQ4Head>,
    recursive_hidden_variant: MtpRecursiveHiddenVariant,
    base_hidden_variant: MtpBaseHiddenVariant,
    history_mode: MtpHistoryMode,
}

impl<'a> SpeculativeDecoder<'a> {
    pub fn new(
        base: &'a MetalForward<'a>,
        mtp_head: &'a MetalMtpHead,
        mtp_session: MetalMtpSession,
    ) -> Self {
        Self {
            base,
            mtp_head,
            mtp_session,
            draft_token_embd_head: false,
            draft_lm_head_override: None,
            draft_affine_q4_head_override: None,
            recursive_hidden_variant: MtpRecursiveHiddenVariant::PreNorm,
            base_hidden_variant: MtpBaseHiddenVariant::PreNorm,
            history_mode: MtpHistoryMode::Committed,
        }
    }

    pub fn set_draft_token_embd_head(&mut self, enabled: bool) {
        self.draft_token_embd_head = enabled;
    }

    pub fn set_draft_lm_head_override(&mut self, head: Option<MetalTensor>) {
        self.draft_lm_head_override = head;
    }

    pub fn set_draft_affine_q4_head_override(&mut self, head: Option<MtpDraftAffineQ4Head>) {
        self.draft_affine_q4_head_override = head;
    }

    fn draft_lm_head(&self) -> &MetalTensor {
        if self.draft_token_embd_head {
            &self.base.model.token_embd
        } else if let Some(head) = &self.draft_lm_head_override {
            head
        } else {
            &self.base.model.lm_head
        }
    }

    fn encode_draft_lm_head_logits(
        &self,
        enc: &KernelEncoder,
        h: usize,
        vocab_size: usize,
    ) -> Result<(), MtpError> {
        let ctx = self.base.ctx;
        if let Some(head) = &self.draft_affine_q4_head_override {
            if head.n_in != h || head.n_out != vocab_size {
                return Err(MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_draft_affine_q4_head",
                    detail: format!(
                        "head dims {}/{} do not match expected {h}/{vocab_size}",
                        head.n_in, head.n_out
                    ),
                }));
            }
            encode_mtp_draft_affine_q4_gs64_f32(
                ctx,
                enc,
                &head.weight,
                &head.scales,
                &head.biases,
                &self.mtp_session.h,
                &self.mtp_session.logits,
                head.n_in,
                head.n_out,
            )?;
        } else {
            let draft_lm_head = self.draft_lm_head();
            encode_mat_vec_dispatch(
                ctx,
                enc,
                draft_lm_head,
                &self.mtp_session.h,
                &self.mtp_session.logits,
                h,
                vocab_size,
            )?;
        }
        Ok(())
    }

    pub fn set_recursive_hidden_variant(&mut self, variant: MtpRecursiveHiddenVariant) {
        self.recursive_hidden_variant = variant;
    }

    pub fn set_base_hidden_variant(&mut self, variant: MtpBaseHiddenVariant) {
        self.base_hidden_variant = variant;
    }

    pub fn set_history_mode(&mut self, mode: MtpHistoryMode) {
        self.history_mode = mode;
    }

    fn wants_base_post_norm(&self) -> bool {
        self.base_hidden_variant == MtpBaseHiddenVariant::PostNorm
    }

    fn uses_cycle_mtp_history(&self) -> bool {
        self.history_mode == MtpHistoryMode::Cycle
    }

    fn uses_draft_accepted_mtp_history(&self) -> bool {
        self.history_mode == MtpHistoryMode::DraftAccepted
    }

    fn write_base_hidden_variant(
        &self,
        pre_norm_hidden: &MetalTensor,
        dst: &MetalTensor,
    ) -> Result<(), MtpError> {
        if !self.wants_base_post_norm() {
            copy_f32_tensor(pre_norm_hidden, dst)?;
            return Ok(());
        }
        let cmd = self.base.ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin(&cmd);
        encode_rms_norm_mul_f32(
            self.base.ctx,
            &enc,
            pre_norm_hidden,
            &self.base.model.output_norm,
            dst,
            RMS_EPS,
        )?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    fn encode_mtp_ffn(&mut self, enc: &KernelEncoder) -> Result<(), MtpError> {
        if let Some(moe) = self.mtp_head.attn.ffn_moe.as_ref() {
            self.encode_mtp_moe_ffn(enc, moe)
        } else {
            self.encode_mtp_dense_ffn(enc)
        }
    }

    fn encode_mtp_dense_ffn(&mut self, enc: &KernelEncoder) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;
        let f = arch.intermediate_size as usize;
        let g_w = &self.mtp_head.attn.ffn_gate;
        let u_w = &self.mtp_head.attn.ffn_up;
        let d_w = &self.mtp_head.attn.ffn_down;
        let ffn_fused = g_w.dtype == GgmlType::Q4_K && u_w.dtype == GgmlType::Q4_K;
        if ffn_fused {
            encode_ffn_swiglu_q4_K_f32(
                ctx,
                enc,
                g_w,
                u_w,
                &self.mtp_session.h,
                &self.mtp_session.ffn_inner,
                h,
                f,
            )?;
        } else {
            encode_mat_vec_dispatch(
                ctx,
                enc,
                g_w,
                &self.mtp_session.h,
                &self.mtp_session.ffn_gate,
                h,
                f,
            )?;
            encode_mat_vec_dispatch(
                ctx,
                enc,
                u_w,
                &self.mtp_session.h,
                &self.mtp_session.ffn_up,
                h,
                f,
            )?;
            encode_silu_mul_f32(
                ctx,
                enc,
                &self.mtp_session.ffn_gate,
                &self.mtp_session.ffn_up,
                &self.mtp_session.ffn_inner,
            )?;
        }
        encode_mat_vec_dispatch(
            ctx,
            enc,
            d_w,
            &self.mtp_session.ffn_inner,
            &self.mtp_session.ffn_out,
            f,
            h,
        )?;
        encode_add_inplace_f32(ctx, enc, &self.mtp_session.x, &self.mtp_session.ffn_out)?;
        Ok(())
    }

    fn encode_mtp_moe_ffn(
        &mut self,
        enc: &KernelEncoder,
        moe: &MetalMoeFfn,
    ) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;
        let f_exp = arch.expert_feed_forward_length as usize;
        let f_shared = arch.expert_shared_feed_forward_length as usize;
        let n_expert = arch.expert_count as usize;
        let topk = arch.expert_used_count.min(arch.expert_count) as usize;

        let router_probs = self
            .mtp_session
            .moe_router_probs
            .view_subrange(0, vec![n_expert as u64]);
        let topk_idx = self
            .mtp_session
            .moe_topk_idx
            .view_subrange(0, vec![topk as u64]);
        let topk_w = self
            .mtp_session
            .moe_topk_weight
            .view_subrange(0, vec![topk as u64]);
        let routed_gate = self
            .mtp_session
            .ffn_gate
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let routed_up = self
            .mtp_session
            .ffn_up
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let routed_inner = self
            .mtp_session
            .ffn_inner
            .view_subrange(0, vec![(topk * f_exp) as u64]);
        let routed_out = self
            .mtp_session
            .moe_expert_out
            .view_subrange(0, vec![(topk * h) as u64]);

        encode_mat_vec_dispatch(
            ctx,
            enc,
            &moe.gate_inp,
            &self.mtp_session.h,
            &router_probs,
            h,
            n_expert,
        )?;
        encode_topk_logits_softmax_dot_sigmoid_f32(
            ctx,
            enc,
            &router_probs,
            &moe.gate_inp_shexp,
            &self.mtp_session.h,
            &topk_idx,
            &topk_w,
            &self.mtp_session.moe_shared_gate,
            n_expert,
            topk,
            h,
        )?;
        match (moe.gate_exps.dtype, moe.up_exps.dtype) {
            (GgmlType::F32, GgmlType::F32) => {
                encode_moe_mat_vec_f32(
                    ctx,
                    enc,
                    &moe.gate_exps,
                    &self.mtp_session.h,
                    &topk_idx,
                    &routed_gate,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_moe_mat_vec_f32(
                    ctx,
                    enc,
                    &moe.up_exps,
                    &self.mtp_session.h,
                    &topk_idx,
                    &routed_up,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
                encode_silu_mul_f32(ctx, enc, &routed_gate, &routed_up, &routed_inner)?;
            }
            (GgmlType::Q4_K, GgmlType::Q4_K) => encode_moe_swiglu_q4_K_f32(
                ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &self.mtp_session.h,
                &topk_idx,
                &routed_inner,
                h,
                f_exp,
                n_expert,
                topk,
            )?,
            (gate, up) => {
                return Err(MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_moe_gate_up",
                    detail: format!("unsupported expert gate/up dtypes {gate:?}/{up:?}"),
                }));
            }
        }

        match moe.down_exps.dtype {
            GgmlType::F32 => {
                encode_moe_down_f32_f32(
                    ctx,
                    enc,
                    &moe.down_exps,
                    &routed_inner,
                    &topk_idx,
                    &routed_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )?;
                encode_moe_weighted_sum_f32(
                    ctx,
                    enc,
                    &routed_out,
                    &topk_w,
                    &self.mtp_session.mixer_out,
                    h,
                    topk,
                )?;
            }
            GgmlType::Q5_K if f_exp == 512 => {
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                    ctx,
                    enc,
                    &moe.down_exps,
                    &routed_inner,
                    &topk_idx,
                    &topk_w,
                    &self.mtp_session.mixer_out,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    1,
                )?;
            }
            down => {
                return Err(MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_moe_down",
                    detail: format!("unsupported expert down dtype/shape {down:?}/K{f_exp}"),
                }));
            }
        }

        let shared_gate = self
            .mtp_session
            .ffn_gate
            .view_subrange(0, vec![f_shared as u64]);
        let shared_up = self
            .mtp_session
            .ffn_up
            .view_subrange(0, vec![f_shared as u64]);
        let shared_inner = self
            .mtp_session
            .ffn_inner
            .view_subrange(0, vec![f_shared as u64]);
        let shared_out = self.mtp_session.ffn_out.view_subrange(0, vec![h as u64]);
        encode_mat_vec_dispatch(
            ctx,
            enc,
            &self.mtp_head.attn.ffn_gate,
            &self.mtp_session.h,
            &shared_gate,
            h,
            f_shared,
        )?;
        encode_mat_vec_dispatch(
            ctx,
            enc,
            &self.mtp_head.attn.ffn_up,
            &self.mtp_session.h,
            &shared_up,
            h,
            f_shared,
        )?;
        encode_silu_mul_f32(ctx, enc, &shared_gate, &shared_up, &shared_inner)?;
        encode_mat_vec_dispatch(
            ctx,
            enc,
            &self.mtp_head.attn.ffn_down,
            &shared_inner,
            &shared_out,
            f_shared,
            h,
        )?;
        encode_axpy_scalar_f32(
            ctx,
            enc,
            &shared_out,
            &self.mtp_session.moe_shared_gate,
            &self.mtp_session.mixer_out,
        )?;
        encode_add_inplace_f32(ctx, enc, &self.mtp_session.x, &self.mtp_session.mixer_out)?;
        Ok(())
    }

    /// Draft a single token. The MTP head at slot `position` consumes
    /// `(embed(next_tok), prev_hidden)` and predicts the token at
    /// `position + 2` (greedy v1).
    ///
    /// `prev_hidden` is the GPU-resident pre-output_norm hidden from the
    /// base model at slot `position`. Caller must ensure
    /// `mtp_session.kv_n_pos == position` (next-sequential append).
    /// Side effect: appends one entry to MTP KV at slot `position`.
    ///
    /// GPU command ordering: this call commits + waits before returning,
    /// so `prev_hidden` may be safely reused or overwritten by subsequent
    /// caller code.
    pub fn draft(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
    ) -> Result<i32, MtpError> {
        let result =
            self.draft_inner(next_tok, prev_hidden, position, DraftReadback::ArgmaxOnly)?;
        Ok(result.argmax.expect("draft argmax missing"))
    }

    /// Same as `draft` but discards logits. Used by the inline accept-branch
    /// bridge (`docs/H4-MTP.md` §1.4 step E) and prompt-time prefill (§1.5).
    /// Identical GPU command ordering contract.
    pub fn draft_kv_only(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
    ) -> Result<(), MtpError> {
        let _ = self.draft_inner(next_tok, prev_hidden, position, DraftReadback::None)?;
        Ok(())
    }

    /// Draft + (optionally) read back logits. Internal helper.
    fn draft_inner(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
        readback: DraftReadback,
    ) -> Result<DraftResult, MtpError> {
        let arch = &self.base.model.arch;
        if next_tok < 0 || (next_tok as u32) >= arch.vocab_size {
            return Err(MtpError::BadToken(next_tok, arch.vocab_size));
        }
        if self.mtp_session.kv_n_pos as u32 != position {
            return Err(MtpError::KvPositionMismatch {
                position,
                kv_n_pos: self.mtp_session.kv_n_pos,
            });
        }
        let h = arch.hidden_size as usize;
        if prev_hidden.n_elements() != h as u64 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_draft.prev_hidden",
                detail: format!(
                    "prev_hidden.n_elements()={} != hidden_size={}",
                    prev_hidden.n_elements(),
                    h,
                ),
            }));
        }

        let ctx = self.base.ctx;

        // Stage next_tok into ids_buf.
        unsafe {
            let ptr = self.mtp_session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = next_tok;
        }

        // Build a single command buffer for the whole MTP step.
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        let enc = KernelEncoder::begin(&cmd);

        // (1) Embedding lookup → e.
        encode_get_rows_f32(
            ctx,
            &enc,
            &self.base.model.token_embd,
            &self.mtp_session.ids_buf,
            &self.mtp_session.e,
            1,
            h,
        )?;

        // (2) RMSNorm both inputs.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.e,
            &self.mtp_head.enorm,
            &self.mtp_session.e_normed,
            RMS_EPS,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            prev_hidden,
            &self.mtp_head.hnorm,
            &self.mtp_session.h_normed,
            RMS_EPS,
        )?;

        // (3) Concat [e_normed, h_normed] → [2H]. vLLM canonical order:
        // embed first (cols 0..H), hidden second (cols H..2H).
        // Implemented as two scatter_offset writes since we don't have a
        // dedicated concat kernel.
        encode_scatter_offset_f32(
            ctx,
            &enc,
            &self.mtp_session.e_normed,
            &self.mtp_session.eh_concat,
            0,
            h,
        )?;
        encode_scatter_offset_f32(
            ctx,
            &enc,
            &self.mtp_session.h_normed,
            &self.mtp_session.eh_concat,
            h,
            h,
        )?;

        // (4) eh_proj: [2H, H] → [H]. Result lands in `x` (residual stream).
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            &self.mtp_head.eh_proj,
            &self.mtp_session.eh_concat,
            &self.mtp_session.x,
            2 * h,
            h,
        )?;

        // ----- MTP transformer block: attn + FFN with residuals -----

        // (5) Pre-attn RMSNorm: x → h.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.x,
            &self.mtp_head.attn.attn_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        // (6) Attn block (standard gated-attn, indexes our dedicated MTP KV).
        // KV-only path for draft_kv_only: append MTP KV for this slot but skip
        // the expensive attention decode / FFN / lm_head tail entirely.
        if matches!(readback, DraftReadback::None) {
            self.encode_mtp_kv_only(&enc, position)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            self.mtp_session.kv_n_pos = position as usize + 1;
            return Ok(DraftResult {
                logits: None,
                argmax: None,
            });
        }
        self.encode_mtp_attn(&enc, position)?;

        // (7) Residual #1: x += mixer_out.
        encode_add_inplace_f32(ctx, &enc, &self.mtp_session.x, &self.mtp_session.mixer_out)?;

        // (8) Pre-FFN RMSNorm: x → h.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.x,
            &self.mtp_head.attn.post_attn_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        // (9-10) FFN + residual. Dense MTP uses SwiGLU; MoE MTP runs
        // router+routed/shared experts against the same residual stream.
        self.encode_mtp_ffn(&enc)?;

        // (11) shared_head_norm: x → h.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &self.mtp_session.x,
            &self.mtp_head.shared_head_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        // (12) lm_head: [H] → [V]. Default reuses the base lm_head; bench
        // probes can substitute draft-only alternatives.
        self.encode_draft_lm_head_logits(&enc, h, arch.vocab_size as usize)?;

        if !matches!(readback, DraftReadback::None) {
            encode_argmax_f32(
                ctx,
                &enc,
                &self.mtp_session.logits,
                &self.mtp_session.draft_argmax,
                1,
                arch.vocab_size as usize,
            )?;
        }

        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        // KV side-effect was committed by encode_mtp_attn; bump our counter.
        self.mtp_session.kv_n_pos = position as usize + 1;

        let argmax = if matches!(
            readback,
            DraftReadback::ArgmaxOnly | DraftReadback::FullLogits
        ) {
            unsafe {
                let src = self.mtp_session.draft_argmax.buffer.contents().as_ptr() as *const i32;
                Some(*src)
            }
        } else {
            None
        };

        if matches!(readback, DraftReadback::FullLogits) {
            let mut out = vec![0.0f32; arch.vocab_size as usize];
            unsafe {
                let src = self.mtp_session.logits.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
            }
            Ok(DraftResult {
                logits: Some(out),
                argmax,
            })
        } else {
            Ok(DraftResult {
                logits: None,
                argmax,
            })
        }
    }

    fn encode_mtp_draft_step(
        &mut self,
        enc: &KernelEncoder,
        id_tensor: &MetalTensor,
        prev_hidden: &MetalTensor,
        position: u32,
        emit_head: bool,
    ) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;

        encode_get_rows_f32(
            ctx,
            enc,
            &self.base.model.token_embd,
            id_tensor,
            &self.mtp_session.e,
            1,
            h,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &self.mtp_session.e,
            &self.mtp_head.enorm,
            &self.mtp_session.e_normed,
            RMS_EPS,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            prev_hidden,
            &self.mtp_head.hnorm,
            &self.mtp_session.h_normed,
            RMS_EPS,
        )?;
        encode_scatter_offset_f32(
            ctx,
            enc,
            &self.mtp_session.e_normed,
            &self.mtp_session.eh_concat,
            0,
            h,
        )?;
        encode_scatter_offset_f32(
            ctx,
            enc,
            &self.mtp_session.h_normed,
            &self.mtp_session.eh_concat,
            h,
            h,
        )?;
        encode_mat_vec_dispatch(
            ctx,
            enc,
            &self.mtp_head.eh_proj,
            &self.mtp_session.eh_concat,
            &self.mtp_session.x,
            2 * h,
            h,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &self.mtp_session.x,
            &self.mtp_head.attn.attn_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;
        self.encode_mtp_attn(enc, position)?;
        encode_add_inplace_f32(ctx, enc, &self.mtp_session.x, &self.mtp_session.mixer_out)?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &self.mtp_session.x,
            &self.mtp_head.attn.post_attn_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;

        self.encode_mtp_ffn(enc)?;
        if !emit_head {
            if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                encode_rms_norm_mul_f32(
                    ctx,
                    enc,
                    &self.mtp_session.x,
                    &self.mtp_head.shared_head_norm,
                    &self.mtp_session.h,
                    RMS_EPS,
                )?;
            }
            return Ok(());
        }
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &self.mtp_session.x,
            &self.mtp_head.shared_head_norm,
            &self.mtp_session.h,
            RMS_EPS,
        )?;
        self.encode_draft_lm_head_logits(enc, h, arch.vocab_size as usize)?;
        encode_argmax_f32(
            ctx,
            enc,
            &self.mtp_session.logits,
            &self.mtp_session.draft_argmax,
            1,
            arch.vocab_size as usize,
        )?;
        Ok(())
    }

    fn draft_chain_single_cb(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        start_position: u32,
        n_drafts: usize,
    ) -> Result<Vec<i32>, MtpError> {
        let arch = &self.base.model.arch;
        if !(1..=15).contains(&n_drafts) {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_draft_chain_single_cb",
                detail: format!("n_drafts={n_drafts} must be in [1, 15]"),
            }));
        }
        if next_tok < 0 || (next_tok as u32) >= arch.vocab_size {
            return Err(MtpError::BadToken(next_tok, arch.vocab_size));
        }
        if self.mtp_session.kv_n_pos as u32 != start_position {
            return Err(MtpError::KvPositionMismatch {
                position: start_position,
                kv_n_pos: self.mtp_session.kv_n_pos,
            });
        }
        let h = arch.hidden_size as usize;
        if prev_hidden.n_elements() != h as u64 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_draft_chain_single_cb.prev_hidden",
                detail: format!(
                    "prev_hidden.n_elements()={} != hidden_size={}",
                    prev_hidden.n_elements(),
                    h,
                ),
            }));
        }

        unsafe {
            let ptr = self.mtp_session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ptr = next_tok;
        }

        let ctx = self.base.ctx;
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        for j in 0..n_drafts {
            let id_tensor = if j == 0 {
                self.mtp_session.ids_buf.clone()
            } else {
                self.mtp_session.draft_argmax.clone()
            };
            let hidden_tensor = if j == 0 {
                prev_hidden.clone()
            } else if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                self.mtp_session.h.clone()
            } else {
                self.mtp_session.x.clone()
            };
            let enc = KernelEncoder::begin(&cmd);
            self.encode_mtp_draft_step(
                &enc,
                &id_tensor,
                &hidden_tensor,
                start_position + j as u32,
                true,
            )?;
            enc.end();

            let blit = BlitEncoder::begin(&cmd);
            blit.copy_buffer(
                &self.mtp_session.draft_argmax.buffer,
                self.mtp_session.draft_argmax.offset,
                &self.mtp_session.draft_ids.buffer,
                self.mtp_session.draft_ids.offset + (j as u64) * std::mem::size_of::<i32>() as u64,
                std::mem::size_of::<i32>() as u64,
            );
            blit.end();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        self.mtp_session.kv_n_pos = start_position as usize + n_drafts;
        let mut out = vec![0i32; n_drafts];
        unsafe {
            let src = (self.mtp_session.draft_ids.buffer.contents().as_ptr() as *const u8)
                .add(self.mtp_session.draft_ids.offset as usize)
                as *const i32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_drafts);
        }
        Ok(out)
    }

    fn draft_chain_recorded_body_only(
        &mut self,
        carry_tok: i32,
        drafts: &[i32],
        prev_hidden: &MetalTensor,
        start_position: u32,
    ) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        if !(1..=15).contains(&drafts.len()) {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_draft_chain_recorded_body_only",
                detail: format!("n_drafts={} must be in [1, 15]", drafts.len()),
            }));
        }
        if carry_tok < 0 || (carry_tok as u32) >= arch.vocab_size {
            return Err(MtpError::BadToken(carry_tok, arch.vocab_size));
        }
        for &tok in drafts {
            if tok < 0 || (tok as u32) >= arch.vocab_size {
                return Err(MtpError::BadToken(tok, arch.vocab_size));
            }
        }
        if self.mtp_session.kv_n_pos as u32 != start_position {
            return Err(MtpError::KvPositionMismatch {
                position: start_position,
                kv_n_pos: self.mtp_session.kv_n_pos,
            });
        }
        let h = arch.hidden_size as usize;
        if prev_hidden.n_elements() != h as u64 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_draft_chain_recorded_body_only.prev_hidden",
                detail: format!(
                    "prev_hidden.n_elements()={} != hidden_size={}",
                    prev_hidden.n_elements(),
                    h,
                ),
            }));
        }

        unsafe {
            let ids_ptr = self.mtp_session.ids_buf.buffer.contents().as_ptr() as *mut i32;
            *ids_ptr = carry_tok;
            let draft_ptr = (self.mtp_session.draft_ids.buffer.contents().as_ptr() as *mut u8)
                .add(self.mtp_session.draft_ids.offset as usize)
                as *mut i32;
            std::ptr::copy_nonoverlapping(drafts.as_ptr(), draft_ptr, drafts.len());
        }

        let ctx = self.base.ctx;
        let cmd = ctx.queue.commandBuffer().expect("cmd buf");
        for j in 0..drafts.len() {
            let id_tensor = if j == 0 {
                self.mtp_session.ids_buf.clone()
            } else {
                self.mtp_session
                    .draft_ids
                    .view_subrange((j - 1) as u64, vec![1])
            };
            let hidden_tensor = if j == 0 {
                prev_hidden.clone()
            } else if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                self.mtp_session.h.clone()
            } else {
                self.mtp_session.x.clone()
            };
            let enc = KernelEncoder::begin(&cmd);
            self.encode_mtp_draft_step(
                &enc,
                &id_tensor,
                &hidden_tensor,
                start_position + j as u32,
                false,
            )?;
            enc.end();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        self.mtp_session.kv_n_pos = start_position as usize + drafts.len();
        Ok(())
    }

    /// MTP-specific attn step. Mirrors `MetalForward::encode_attn` but
    /// indexes the dedicated MTP KV ring (single layer) instead of
    /// `MetalSession::kv_*[attn_idx]`.
    fn encode_mtp_kv_only(&mut self, enc: &KernelEncoder, position: u32) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_kv = arch.n_kv_heads as usize;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        let ab = &self.mtp_head.attn;
        let s = &mut self.mtp_session;

        encode_mat_vec_dispatch(ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;
        encode_rope_neox_f32(
            ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_scatter_offset_f32_to_f16_kv(
            ctx,
            enc,
            &s.attn_k_normed,
            &s.attn_v_now,
            &s.kv_k,
            &s.kv_v,
            (position as usize) * kv_dim,
            kv_dim,
        )?;
        Ok(())
    }

    /// MTP-specific attn step. Mirrors `MetalForward::encode_attn` but
    /// indexes the dedicated MTP KV ring (single layer) instead of
    /// `MetalSession::kv_*[attn_idx]`.
    fn encode_mtp_attn(&mut self, enc: &KernelEncoder, position: u32) -> Result<(), MtpError> {
        let arch = &self.base.model.arch;
        let ctx = self.base.ctx;
        let h = arch.hidden_size as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_q = arch.n_q_heads as usize;
        let n_kv = arch.n_kv_heads as usize;
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        let ab = &self.mtp_head.attn;
        let s = &mut self.mtp_session;

        // (1) Q projection: outputs 2 * q_dim (Q + gate interleaved).
        encode_mat_vec_dispatch(ctx, enc, &ab.q, &s.h, &s.attn_q_full, h, 2 * q_dim)?;

        // (2) Split Q from gate.
        encode_split_q_gate_f32(
            ctx,
            enc,
            &s.attn_q_full,
            &s.attn_q,
            &s.attn_gate,
            n_q,
            head_dim,
        )?;

        // (3) Q-norm (per-head RMSNorm).
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &s.attn_q,
            &ab.q_norm,
            &s.attn_q_normed,
            n_q,
            head_dim,
            RMS_EPS,
        )?;

        // (4) K, V projections.
        encode_mat_vec_dispatch(ctx, enc, &ab.k, &s.h, &s.attn_k_now, h, kv_dim)?;
        encode_mat_vec_dispatch(ctx, enc, &ab.v, &s.h, &s.attn_v_now, h, kv_dim)?;

        // (5) K-norm (per-head).
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &s.attn_k_now,
            &ab.k_norm,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            RMS_EPS,
        )?;

        // (6) Partial RoPE on Q and K.
        encode_rope_neox_f32(
            ctx,
            enc,
            &s.attn_q_normed,
            n_q,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;
        encode_rope_neox_f32(
            ctx,
            enc,
            &s.attn_k_normed,
            n_kv,
            head_dim,
            n_rot,
            position,
            arch.rope_theta,
        )?;

        // (7) Append (K, V) to the MTP KV ring at slot `position`.
        // Strict-monotonic — the caller-side check in draft_inner already
        // asserted kv_n_pos == position.
        encode_scatter_offset_f32_to_f16_kv(
            ctx,
            enc,
            &s.attn_k_normed,
            &s.attn_v_now,
            &s.kv_k,
            &s.kv_v,
            (position as usize) * kv_dim,
            kv_dim,
        )?;
        // KV n_pos counter is bumped in draft_inner after waitUntilCompleted.

        // (8) Fused attention decode against the prefix [0..=position].
        // n_pos is `position + 1` because slot `position` was just appended.
        let n_pos = position as usize + 1;
        const V4_HEAD_DIM: usize = 256;
        const V4_GROUP: usize = 6;
        let use_v4 = head_dim == V4_HEAD_DIM && n_q == n_kv * V4_GROUP;
        if use_v4 {
            let nwg = attn_v4_choose_nwg(n_pos, V4_GROUP);
            let tile_c = attn_v4_choose_tile_c(n_pos, V4_GROUP);
            encode_attn_decode_v4_f32(
                ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k,
                &s.kv_v,
                &s.attn_v4_o_partial,
                &s.attn_v4_ml_partial,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                n_pos,
                nwg,
                tile_c,
            )?;
        } else {
            encode_attn_decode_f16kv_f32(
                ctx,
                enc,
                &s.attn_q_normed,
                &s.kv_k,
                &s.kv_v,
                &s.attn_o,
                n_q,
                n_kv,
                head_dim,
                n_pos,
            )?;
        }

        // (9) Apply gated-attention sigmoid gate: attn_o *= sigmoid(gate).
        // Reuse attn_q (no longer needed) as scratch for sigmoid output.
        encode_sigmoid_f32(ctx, enc, &s.attn_gate, &s.attn_q)?;
        encode_mul_f32(ctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?;

        // (10) Output projection: q_dim → hidden. Result lands in mixer_out.
        encode_mat_vec_dispatch(ctx, enc, &ab.o, &s.attn_o, &s.mixer_out, q_dim, h)?;
        Ok(())
    }

    /// Greedy generation with MTP-augmented speculative decoding.
    /// Implements `docs/H4-MTP.md` §1.4 (lazy sequential verify with
    /// inline accept-branch bridge).
    ///
    /// `prompt_ids` are processed first via `prefill_prompt` (§1.5),
    /// which:
    /// * runs base forward for each prompt token
    /// * streams MTP-KV prefill for slots 0..n-2 immediately after
    ///   each base step (so `h_i` is consumed before invalidation)
    /// * retains `h_{n-1}` for bootstrap
    ///
    /// Then runs the per-step loop until `eos` or `max_new_tokens` is
    /// emitted.
    ///
    /// **Terminal-return contract:** when this returns, the session is
    /// NOT resumable for further generation (the loop may early-exit
    /// after emitting D_tok without running step E bridge or step F base
    /// forward, leaving base/MTP state inconsistent). Caller must use a
    /// fresh session for any continuation.
    pub fn decode(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        stop_tokens: &[i32],
        base_session: &mut MetalSession,
    ) -> Result<DecodeOutput, MtpError> {
        let arch = &self.base.model.arch;
        let h = arch.hidden_size as usize;
        let t_start = std::time::Instant::now();
        let mut stats = SpecStats::default();
        let mut tokens: Vec<i32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        tokens.extend_from_slice(prompt_ids);

        if prompt_ids.is_empty() {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "decode",
                detail: "prompt_ids must be non-empty".into(),
            }));
        }
        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                tokens,
                stats: stats.into_finalized(t_start),
            });
        }

        // Allocate a persistent hidden-carry buffer in the MTP session.
        // We keep two buffers so the bridge step can hold onto h_P while
        // the next single_token_with_hidden writes h_D into the other.
        let hidden_a = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let hidden_b = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;

        // ---------- Prompt prefill ----------
        // Per §1.5: stream MTP-KV prefill during base prompt forward.
        // For i in 0..n: run base on prompt[i] → h_i (in hidden_a or hidden_b
        // depending on parity). For i in 0..n-1: also run MTP draft_kv_only
        // with (prompt[i+1], h_i, i). We read back only argmax tokens,
        // not full logits, since the speculative path needs just the
        // bootstrap next-token id.
        let n_prompt = prompt_ids.len();
        let mut next_bootstrap_tok: i32 = 0;
        let mut h_last_is_a = true; // which buffer holds the most recent hidden
        let t_prefill = std::time::Instant::now();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            let dst = if h_last_is_a { &hidden_a } else { &hidden_b };
            next_bootstrap_tok = self.base.single_token_argmax_with_hidden(
                tid,
                i as u32,
                base_session,
                dst,
                self.wants_base_post_norm(),
            )?;
            stats.base_forward_calls += 1;

            if i + 1 < n_prompt {
                // MTP slot i: pair (prompt[i+1], h_i, i).
                self.draft_kv_only(prompt_ids[i + 1], dst, i as u32)?;
                stats.mtp_calls += 1;
            }
            // Alternate buffers so the next base call doesn't overwrite
            // `dst` while we still might need it. (For prefill, we
            // actually consume `dst` immediately via draft_kv_only above
            // before the next iteration's single_token_with_hidden, so
            // alternation isn't strictly required, but it keeps the
            // buffer policy consistent with the steady-state loop.)
            h_last_is_a = !h_last_is_a;
        }
        stats.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        // After prefill loop: the LAST hidden (h_{n-1}) is in the OPPOSITE
        // buffer from the one h_last_is_a now points to (since we toggled
        // after consumption). Restore the pointer.
        h_last_is_a = !h_last_is_a;
        let hidden_at_proc = if h_last_is_a { &hidden_a } else { &hidden_b };
        // sentinel: track which buffer is which without another bool indirection
        let _ = hidden_at_proc;

        // Bootstrap: argmax(last_logits) is the first token to emit.
        let mut emit_tok = next_bootstrap_tok;
        let mut processed_pos = (n_prompt - 1) as u32;
        // mtp_processed_pos is implicitly tracked by self.mtp_session.kv_n_pos.
        // After prefill it should be n - 1 (slots 0..n-2 = n-1 entries).
        debug_assert_eq!(
            self.mtp_session.kv_n_pos as u32,
            (n_prompt - 1) as u32,
            "after prefill: mtp_kv should have n-1 entries"
        );

        // ---------- Per-step decode loop ----------
        let mut emitted_count: usize = 0;
        loop {
            // A. Emit the carried token. Stop if EOS / limit.
            tokens.push(emit_tok);
            emitted_count += 1;
            if stop_tokens.contains(&emit_tok) || emitted_count >= max_new_tokens {
                break;
            }

            let p_tok = emit_tok;
            let p_pos = processed_pos + 1;

            // B. MTP draft for slot processed_pos with (P_tok, hidden_at_proc, processed_pos).
            // Predicts a candidate for position p_pos+1.
            // Precondition: mtp_session.kv_n_pos == processed_pos.
            // Recompute hidden_at_proc reference from h_last_is_a.
            let hidden_at_proc_ref = if h_last_is_a { &hidden_a } else { &hidden_b };
            let d_tok = self.draft(p_tok, hidden_at_proc_ref, processed_pos)?;
            stats.mtp_calls += 1;
            stats.drafts_attempted += 1;

            // C. Base forward on P_tok at p_pos. Writes new hidden into
            // the OTHER buffer so we can keep hidden_at_proc alive for
            // the inline bridge. After: that other buffer holds h_{p_pos}.
            let dst_for_p = if h_last_is_a { &hidden_b } else { &hidden_a };
            let target_next = self.base.single_token_argmax_with_hidden(
                p_tok,
                p_pos,
                base_session,
                dst_for_p,
                self.wants_base_post_norm(),
            )?;
            stats.base_forward_calls += 1;

            // D. Lazy sequential verify.
            if d_tok == target_next {
                // ACCEPT branch.
                stats.accepted += 1;
                tokens.push(d_tok);
                emitted_count += 1;
                if stop_tokens.contains(&d_tok) || emitted_count >= max_new_tokens {
                    // Terminal return: state inconsistent (no bridge, no
                    // step F). Per the H4-MTP §1.4 contract.
                    break;
                }
                let d_pos = p_pos + 1;

                // E. Inline MTP bridge for slot p_pos with (D_tok, h_p, p_pos).
                // h_p is in dst_for_p. This MUST run BEFORE step F's
                // single_token_with_hidden, which would overwrite the
                // hidden buffer (we'll write into the OTHER one to be
                // safe — alternation policy).
                self.draft_kv_only(d_tok, dst_for_p, p_pos)?;
                stats.mtp_calls += 1;

                // F. Base forward on D_tok at d_pos. Write h_d into the
                // buffer we just freed up (the one that previously held
                // hidden_at_proc — which has now been consumed by step B).
                let dst_for_d = if h_last_is_a { &hidden_a } else { &hidden_b };
                let next_emit = self.base.single_token_argmax_with_hidden(
                    d_tok,
                    d_pos,
                    base_session,
                    dst_for_d,
                    self.wants_base_post_norm(),
                )?;
                stats.base_forward_calls += 1;

                // Update for next iter. The buffer holding the latest
                // hidden has now flipped: previously hidden_at_proc was
                // in `(h_last_is_a ? a : b)`; now h_d is in `(h_last_is_a ? a : b)`
                // (yes, the same one, because we wrote it back into the
                // buffer hidden_at_proc was previously in — and we
                // consumed dst_for_p's contents in step E so dst_for_p is
                // free for the next iter's hidden_at_proc to invalidate).
                // Wait — that means h_last_is_a stays the same. Let me
                // re-derive:
                //   start of iter: hidden_at_proc in (h_last_is_a ? a : b)
                //   step C writes h_p into dst_for_p = (h_last_is_a ? b : a)
                //   step E consumes h_p (still in dst_for_p)
                //   step F writes h_d into dst_for_d = (h_last_is_a ? a : b)
                //                        which is the SAME buffer hidden_at_proc was in.
                //   So h_last_is_a stays the same and the next iter's
                //   hidden_at_proc reads from the same buffer.
                processed_pos = d_pos;
                emit_tok = next_emit;
                stats.steps += 1;
            } else {
                // REJECT branch.
                // Base sits at p_pos, MTP sits at processed_pos == p_pos - 1.
                // No bridge or step F runs. h_p is now the new hidden_at_proc;
                // it's in dst_for_p = (h_last_is_a ? b : a).
                processed_pos = p_pos;
                emit_tok = target_next;
                // Flip h_last_is_a so hidden_at_proc reads from dst_for_p next iter.
                h_last_is_a = !h_last_is_a;
                stats.steps += 1;
            }
        }

        Ok(DecodeOutput {
            tokens,
            stats: stats.into_finalized(t_start),
        })
    }

    /// Experimental MTP-N path. Drafts `spec_tokens` proposals by recursively
    /// feeding the MTP head, then verifies `[carry, drafts...]` in one packed
    /// base-model forward using the DFlash packed-verify machinery.
    ///
    /// This is bench-only for now: recursive draft slots beyond the first use
    /// the previous MTP hidden (`mtp_session.x`) as a surrogate for the exact
    /// base hidden. Correctness is still preserved because the target packed
    /// verify remains authoritative and we rebuild canonical MTP KV for the
    /// accepted prefix from captured base hiddens before continuing.
    pub fn decode_packed_n(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        stop_tokens: &[i32],
        base_session: &mut MetalSession,
        spec_tokens: usize,
        verify_scratch: &mut MetalDFlashVerifyScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
    ) -> Result<DecodeOutput, MtpError> {
        self.decode_packed_n_recording(
            prompt_ids,
            max_new_tokens,
            stop_tokens,
            base_session,
            spec_tokens,
            verify_scratch,
            layer_scratch,
            None,
            None,
            false,
        )
    }

    pub fn decode_packed_n_recording(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        stop_tokens: &[i32],
        base_session: &mut MetalSession,
        spec_tokens: usize,
        verify_scratch: &mut MetalDFlashVerifyScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
        mut draft_trace: Option<&mut Vec<RecordedDraftStep>>,
        mut rank_rows: Option<&mut Vec<MtpRankRow>>,
        single_cb_draft: bool,
    ) -> Result<DecodeOutput, MtpError> {
        if !(2..=15).contains(&spec_tokens) {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n",
                detail: format!("spec_tokens={spec_tokens} must be in [2, 15]"),
            }));
        }

        let arch = &self.base.model.arch;
        let h = arch.hidden_size as usize;
        let t_start = std::time::Instant::now();
        let mut stats = SpecStats::default();
        let mut tokens: Vec<i32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        tokens.extend_from_slice(prompt_ids);

        if prompt_ids.is_empty() {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n",
                detail: "prompt_ids must be non-empty".into(),
            }));
        }
        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                tokens,
                stats: stats.into_finalized(t_start),
            });
        }

        let logical_verify_n = spec_tokens + 1;
        let physical_verify_n = verify_scratch.n as usize;
        if physical_verify_n < logical_verify_n {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n.verify_scratch",
                detail: format!(
                    "verify_scratch.n={} < logical_verify_n={logical_verify_n}",
                    verify_scratch.n
                ),
            }));
        }
        if verify_scratch.k_target_layers != 1 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n.verify_scratch",
                detail: format!(
                    "verify_scratch.k_target_layers={} != 1",
                    verify_scratch.k_target_layers
                ),
            }));
        }
        if layer_scratch.n as usize != physical_verify_n {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n.layer_scratch",
                detail: format!(
                    "layer_scratch.n={} != physical_verify_n={physical_verify_n}",
                    layer_scratch.n
                ),
            }));
        }

        let hidden_cur = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let recursive_hidden = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let n_prompt = prompt_ids.len();
        let mut next_bootstrap_tok: i32 = 0;

        // Prompt prefill: identical streaming contract as H4, but only argmax
        // readback from the base path.
        let t_prefill = std::time::Instant::now();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            next_bootstrap_tok = self.base.single_token_argmax_with_hidden(
                tid,
                i as u32,
                base_session,
                &hidden_cur,
                self.wants_base_post_norm(),
            )?;
            stats.base_forward_calls += 1;

            if i + 1 < n_prompt && !self.uses_cycle_mtp_history() {
                self.draft_kv_only(prompt_ids[i + 1], &hidden_cur, i as u32)?;
                stats.mtp_calls += 1;
            }
        }
        stats.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        if self.uses_cycle_mtp_history() {
            self.mtp_session.kv_n_pos = 0;
        } else {
            debug_assert_eq!(
                self.mtp_session.kv_n_pos as u32,
                (n_prompt - 1) as u32,
                "after prefill: mtp_kv should have n-1 entries"
            );
        }

        let last_layer = [(self.base.model.blocks.len() - 1) as u32];
        let mut emit_tok = next_bootstrap_tok;
        let mut processed_pos = (n_prompt - 1) as u32;
        let mut emitted_count: usize = 0;

        'outer: loop {
            tokens.push(emit_tok);
            emitted_count += 1;
            if stop_tokens.contains(&emit_tok) || emitted_count >= max_new_tokens {
                break;
            }

            let carry_tok = emit_tok;
            let draft_start_position = if self.uses_cycle_mtp_history() {
                self.mtp_session.kv_n_pos = 0;
                0
            } else {
                processed_pos
            };
            let start_position = processed_pos + 1;

            // Draft chain. First slot uses exact base hidden. Subsequent slots
            // recursively consume the previous MTP hidden as an approximation.
            let mut draft_logits: Vec<Vec<f32>> = Vec::new();
            let t_draft = std::time::Instant::now();
            let drafts: Vec<i32> = if rank_rows.is_some() {
                let mut drafts: Vec<i32> = Vec::with_capacity(spec_tokens);
                let first = self.draft_inner(
                    carry_tok,
                    &hidden_cur,
                    draft_start_position,
                    DraftReadback::FullLogits,
                )?;
                drafts.push(first.argmax.expect("draft argmax missing"));
                draft_logits.push(first.logits.expect("draft logits missing"));
                stats.mtp_calls += 1;
                stats.drafts_attempted += 1;
                let hidden_src =
                    if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                        &self.mtp_session.h
                    } else {
                        &self.mtp_session.x
                    };
                copy_f32_tensor(hidden_src, &recursive_hidden)?;
                for j in 1..spec_tokens {
                    let result = self.draft_inner(
                        drafts[j - 1],
                        &recursive_hidden,
                        draft_start_position + j as u32,
                        DraftReadback::FullLogits,
                    )?;
                    drafts.push(result.argmax.expect("draft argmax missing"));
                    draft_logits.push(result.logits.expect("draft logits missing"));
                    stats.mtp_calls += 1;
                    stats.drafts_attempted += 1;
                    let hidden_src =
                        if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                            &self.mtp_session.h
                        } else {
                            &self.mtp_session.x
                        };
                    copy_f32_tensor(hidden_src, &recursive_hidden)?;
                }
                drafts
            } else if single_cb_draft {
                let drafts = self.draft_chain_single_cb(
                    carry_tok,
                    &hidden_cur,
                    draft_start_position,
                    spec_tokens,
                )?;
                stats.mtp_calls += 1;
                stats.drafts_attempted += drafts.len() as u32;
                drafts
            } else {
                let mut drafts: Vec<i32> = Vec::with_capacity(spec_tokens);
                let first = self.draft(carry_tok, &hidden_cur, draft_start_position)?;
                drafts.push(first);
                stats.mtp_calls += 1;
                stats.drafts_attempted += 1;
                let hidden_src =
                    if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                        &self.mtp_session.h
                    } else {
                        &self.mtp_session.x
                    };
                copy_f32_tensor(hidden_src, &recursive_hidden)?;
                for j in 1..spec_tokens {
                    let d = self.draft(
                        drafts[j - 1],
                        &recursive_hidden,
                        draft_start_position + j as u32,
                    )?;
                    drafts.push(d);
                    stats.mtp_calls += 1;
                    stats.drafts_attempted += 1;
                    let hidden_src =
                        if self.recursive_hidden_variant == MtpRecursiveHiddenVariant::PostNorm {
                            &self.mtp_session.h
                        } else {
                            &self.mtp_session.x
                        };
                    copy_f32_tensor(hidden_src, &recursive_hidden)?;
                }
                drafts
            };
            stats.draft_ms += t_draft.elapsed().as_secs_f64() * 1e3;

            if let Some(trace) = draft_trace.as_mut() {
                (*trace).push(RecordedDraftStep {
                    carry_tok,
                    start_position,
                    drafts: drafts.clone(),
                });
            }

            let mut verify_input: Vec<i32> = Vec::with_capacity(physical_verify_n);
            verify_input.push(carry_tok);
            verify_input.extend_from_slice(&drafts);
            while verify_input.len() < physical_verify_n {
                verify_input.push(carry_tok);
            }

            let t_verify = std::time::Instant::now();
            let verify_argmax = encode_packed_verify_layer_major_inner(
                self.base,
                &last_layer,
                &verify_input,
                start_position,
                verify_scratch,
                layer_scratch,
                base_session,
                None,
                None,
            )?;
            stats.verify_ms += t_verify.elapsed().as_secs_f64() * 1e3;
            stats.base_forward_calls += 1;

            if let Some(rows) = rank_rows.as_mut() {
                for (j, logits) in draft_logits.iter().enumerate() {
                    let target_tok = verify_argmax[j];
                    let target_idx = target_tok as usize;
                    let target_logit = logits[target_idx];
                    let (rank, top_tokens, top_logits) = rank_and_topk(logits, target_idx, 16);
                    let accepted = drafts[j] == target_tok;
                    (*rows).push(MtpRankRow {
                        step: stats.steps as usize,
                        depth: j,
                        rank,
                        accepted,
                        draft_tok: drafts[j],
                        target_tok,
                        target_logit,
                        top_tokens,
                        top_logits,
                    });
                    if !accepted {
                        break;
                    }
                }
            }

            let mut n_accepted = 0usize;
            let mut stop_now = false;
            for (j, &draft_tok) in drafts.iter().enumerate() {
                if draft_tok != verify_argmax[j] {
                    break;
                }
                stats.accepted += 1;
                n_accepted += 1;
                tokens.push(draft_tok);
                emitted_count += 1;
                if stop_tokens.contains(&draft_tok) || emitted_count >= max_new_tokens {
                    stop_now = true;
                    break;
                }
            }

            let n_keep = (1 + n_accepted) as u32;
            if n_keep < physical_verify_n as u32 {
                let t_restore = std::time::Instant::now();
                encode_restore_after_partial_accept_inner(
                    self.base,
                    verify_scratch,
                    n_keep,
                    start_position,
                    base_session,
                    None,
                )?;
                stats.restore_ms += t_restore.elapsed().as_secs_f64() * 1e3;
            }

            let t_bridge = std::time::Instant::now();
            if self.uses_cycle_mtp_history() {
                self.mtp_session.kv_n_pos = 0;
            } else if self.uses_draft_accepted_mtp_history() {
                self.mtp_session.kv_n_pos = processed_pos as usize + 1 + n_accepted;
            } else {
                // Recursive draft slots beyond the first are approximate. Rebuild
                // the canonical MTP KV for the accepted prefix from captured base
                // hiddens.
                self.mtp_session.kv_n_pos = processed_pos as usize + 1;
                // Co-indexed dispatch across two Metal scratch slots and the
                // drafts[] array; `j` is the slot id, not just an index.
                #[allow(clippy::needless_range_loop)]
                for j in 0..n_accepted {
                    let prev_hidden = verify_scratch.hidden_capture_n_slot(j as u32);
                    let bridge_position = processed_pos + 1 + j as u32;
                    if self.wants_base_post_norm() {
                        self.write_base_hidden_variant(&prev_hidden, &hidden_cur)?;
                        self.draft_kv_only(drafts[j], &hidden_cur, bridge_position)?;
                    } else {
                        self.draft_kv_only(drafts[j], &prev_hidden, bridge_position)?;
                    }
                    stats.mtp_calls += 1;
                }
            }

            let next_hidden = verify_scratch.hidden_capture_n_slot(n_accepted as u32);
            self.write_base_hidden_variant(&next_hidden, &hidden_cur)?;
            stats.bridge_ms += t_bridge.elapsed().as_secs_f64() * 1e3;

            processed_pos += 1 + n_accepted as u32;
            emit_tok = verify_argmax[n_accepted];
            stats.steps += 1;

            if stop_now {
                break 'outer;
            }
        }

        Ok(DecodeOutput {
            tokens,
            stats: stats.into_finalized(t_start),
        })
    }

    pub fn decode_packed_n_recorded_mtp_work(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        stop_tokens: &[i32],
        base_session: &mut MetalSession,
        spec_tokens: usize,
        verify_scratch: &mut MetalDFlashVerifyScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
        trace: &[RecordedDraftStep],
        work: RecordedMtpWork,
    ) -> Result<DecodeOutput, MtpError> {
        if !(1..=15).contains(&spec_tokens) {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_recorded_mtp_work",
                detail: format!("spec_tokens={spec_tokens} must be in [1, 15]"),
            }));
        }

        let arch = &self.base.model.arch;
        let h = arch.hidden_size as usize;
        let t_start = std::time::Instant::now();
        let mut stats = SpecStats::default();
        let mut tokens: Vec<i32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        tokens.extend_from_slice(prompt_ids);

        if prompt_ids.is_empty() {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_recorded_mtp_work",
                detail: "prompt_ids must be non-empty".into(),
            }));
        }
        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                tokens,
                stats: stats.into_finalized(t_start),
            });
        }

        let logical_verify_n = spec_tokens + 1;
        let physical_verify_n = verify_scratch.n as usize;
        if physical_verify_n < logical_verify_n || layer_scratch.n as usize != physical_verify_n {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_recorded_mtp_work.scratch",
                detail: format!(
                    "scratch N mismatch: verify={} layer={} logical_min={logical_verify_n}",
                    verify_scratch.n, layer_scratch.n
                ),
            }));
        }
        if verify_scratch.k_target_layers != 1 {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_recorded_mtp_work.scratch",
                detail: format!(
                    "verify_scratch.k_target_layers={} != 1",
                    verify_scratch.k_target_layers
                ),
            }));
        }

        let hidden_cur = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let n_prompt = prompt_ids.len();
        let mut next_bootstrap_tok: i32 = 0;

        // Match the current native path's prompt-side MTP KV prefill so this
        // probe prices decode work, not a different cache state.
        let t_prefill = std::time::Instant::now();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            next_bootstrap_tok = self.base.single_token_argmax_with_hidden(
                tid,
                i as u32,
                base_session,
                &hidden_cur,
                self.wants_base_post_norm(),
            )?;
            stats.base_forward_calls += 1;

            if i + 1 < n_prompt {
                self.draft_kv_only(prompt_ids[i + 1], &hidden_cur, i as u32)?;
                stats.mtp_calls += 1;
            }
        }
        stats.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        let last_layer = [(self.base.model.blocks.len() - 1) as u32];
        let mut emit_tok = next_bootstrap_tok;
        let mut processed_pos = (n_prompt - 1) as u32;
        let mut emitted_count: usize = 0;
        let mut step_idx: usize = 0;

        'outer: loop {
            tokens.push(emit_tok);
            emitted_count += 1;
            if stop_tokens.contains(&emit_tok) || emitted_count >= max_new_tokens {
                break;
            }

            let remaining = max_new_tokens.saturating_sub(emitted_count);
            let n_draft = spec_tokens.min(remaining);
            if n_draft == 0 {
                break;
            }
            let physical_draft_slots = physical_verify_n - 1;
            let carry_tok = emit_tok;
            let start_position = processed_pos + 1;
            let row = trace.get(step_idx).ok_or_else(|| {
                MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_decode_packed_n_recorded_mtp_work.recorded",
                    detail: format!("missing draft row for step {step_idx}"),
                })
            })?;
            if row.carry_tok != carry_tok || row.start_position != start_position {
                return Err(MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_decode_packed_n_recorded_mtp_work.recorded",
                    detail: format!(
                        "row {step_idx} alignment mismatch: carry {} vs {carry_tok}, \
                         pos {} vs {start_position}",
                        row.carry_tok, row.start_position,
                    ),
                }));
            }
            if row.drafts.len() < n_draft {
                return Err(MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_decode_packed_n_recorded_mtp_work.recorded",
                    detail: format!(
                        "draft row {step_idx} len {} < needed {n_draft}",
                        row.drafts.len()
                    ),
                }));
            }
            let drafts: Vec<i32> = row.drafts[..n_draft].to_vec();
            stats.drafts_attempted += drafts.len() as u32;

            let t_draft = std::time::Instant::now();
            match work {
                RecordedMtpWork::BodyNoLmHead => {
                    self.draft_chain_recorded_body_only(
                        carry_tok,
                        &drafts,
                        &hidden_cur,
                        processed_pos,
                    )?;
                    stats.mtp_calls += 1;
                }
                RecordedMtpWork::BridgeOnly => {
                    self.draft_kv_only(carry_tok, &hidden_cur, processed_pos)?;
                    stats.mtp_calls += 1;
                }
            }
            stats.draft_ms += t_draft.elapsed().as_secs_f64() * 1e3;

            let mut verify_input: Vec<i32> = Vec::with_capacity(physical_verify_n);
            verify_input.push(carry_tok);
            verify_input.extend_from_slice(&drafts);
            for pad_j in drafts.len()..physical_draft_slots {
                verify_input.push(row.drafts.get(pad_j).copied().unwrap_or(carry_tok));
            }
            let n_eff = physical_verify_n as u32;

            let t_verify = std::time::Instant::now();
            let verify_argmax = encode_packed_verify_layer_major_inner(
                self.base,
                &last_layer,
                &verify_input,
                start_position,
                verify_scratch,
                layer_scratch,
                base_session,
                None,
                Some(n_eff),
            )?;
            stats.verify_ms += t_verify.elapsed().as_secs_f64() * 1e3;
            stats.base_forward_calls += 1;

            let mut n_accepted = 0usize;
            let mut stop_now = false;
            for (j, &draft_tok) in drafts.iter().enumerate() {
                if draft_tok != verify_argmax[j] {
                    break;
                }
                stats.accepted += 1;
                n_accepted += 1;
                tokens.push(draft_tok);
                emitted_count += 1;
                if stop_tokens.contains(&draft_tok) || emitted_count >= max_new_tokens {
                    stop_now = true;
                    break;
                }
            }

            let n_keep = (1 + n_accepted) as u32;
            if n_keep < n_eff {
                let t_restore = std::time::Instant::now();
                encode_restore_after_partial_accept_inner(
                    self.base,
                    verify_scratch,
                    n_keep,
                    start_position,
                    base_session,
                    Some(n_eff),
                )?;
                stats.restore_ms += t_restore.elapsed().as_secs_f64() * 1e3;
            }

            let t_bridge = std::time::Instant::now();
            if self.uses_draft_accepted_mtp_history() {
                self.mtp_session.kv_n_pos = processed_pos as usize + 1 + n_accepted;
            } else {
                self.mtp_session.kv_n_pos = processed_pos as usize + 1;
                #[allow(clippy::needless_range_loop)]
                for j in 0..n_accepted {
                    let prev_hidden = verify_scratch.hidden_capture_n_slot(j as u32);
                    let bridge_position = processed_pos + 1 + j as u32;
                    if self.wants_base_post_norm() {
                        self.write_base_hidden_variant(&prev_hidden, &hidden_cur)?;
                        self.draft_kv_only(drafts[j], &hidden_cur, bridge_position)?;
                    } else {
                        self.draft_kv_only(drafts[j], &prev_hidden, bridge_position)?;
                    }
                    stats.mtp_calls += 1;
                }
            }

            let next_hidden = verify_scratch.hidden_capture_n_slot(n_accepted as u32);
            self.write_base_hidden_variant(&next_hidden, &hidden_cur)?;
            stats.bridge_ms += t_bridge.elapsed().as_secs_f64() * 1e3;

            processed_pos += 1 + n_accepted as u32;
            emit_tok = verify_argmax[n_accepted];
            stats.steps += 1;
            step_idx += 1;

            if stop_now {
                break 'outer;
            }
        }

        Ok(DecodeOutput {
            tokens,
            stats: stats.into_finalized(t_start),
        })
    }

    pub fn decode_packed_n_planned(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        stop_tokens: &[i32],
        base_session: &mut MetalSession,
        spec_tokens: usize,
        verify_scratch: &mut MetalDFlashVerifyScratch,
        layer_scratch: &mut MetalDFlashLayerMajorScratch,
        plan: PackedDraftPlan<'_>,
    ) -> Result<DecodeOutput, MtpError> {
        if !(1..=15).contains(&spec_tokens) {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_planned",
                detail: format!("spec_tokens={spec_tokens} must be in [1, 15]"),
            }));
        }

        let arch = &self.base.model.arch;
        let h = arch.hidden_size as usize;
        let t_start = std::time::Instant::now();
        let mut stats = SpecStats::default();
        let mut tokens: Vec<i32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        tokens.extend_from_slice(prompt_ids);

        if prompt_ids.is_empty() {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_planned",
                detail: "prompt_ids must be non-empty".into(),
            }));
        }
        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                tokens,
                stats: stats.into_finalized(t_start),
            });
        }

        let logical_verify_n = spec_tokens + 1;
        let physical_verify_n = verify_scratch.n as usize;
        if physical_verify_n < logical_verify_n || layer_scratch.n as usize != physical_verify_n {
            return Err(MtpError::Metal(MetalError::BadShape {
                kernel: "mtp_decode_packed_n_planned.scratch",
                detail: format!(
                    "scratch N mismatch: verify={} layer={} logical_min={logical_verify_n}",
                    verify_scratch.n, layer_scratch.n
                ),
            }));
        }

        let hidden_cur = MetalTensor::zeros_f32(self.base.ctx, vec![h as u64])?;
        let n_prompt = prompt_ids.len();
        let mut next_bootstrap_tok: i32 = 0;

        // Prompt prefill: only the target base state is needed for replay or
        // oracle drafts, so this intentionally skips MTP KV prefill.
        let t_prefill = std::time::Instant::now();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            next_bootstrap_tok = self.base.single_token_argmax_with_hidden(
                tid,
                i as u32,
                base_session,
                &hidden_cur,
                self.wants_base_post_norm(),
            )?;
            stats.base_forward_calls += 1;
        }
        stats.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        if let PackedDraftPlan::Oracle(oracle) = plan {
            if oracle.first().copied() != Some(next_bootstrap_tok) {
                return Err(MtpError::Metal(MetalError::BadShape {
                    kernel: "mtp_decode_packed_n_planned.oracle",
                    detail: format!(
                        "oracle first token {:?} != bootstrap {next_bootstrap_tok}",
                        oracle.first()
                    ),
                }));
            }
        }

        let last_layer = [(self.base.model.blocks.len() - 1) as u32];
        let mut emit_tok = next_bootstrap_tok;
        let mut processed_pos = (n_prompt - 1) as u32;
        let mut emitted_count: usize = 0;
        let mut step_idx: usize = 0;

        'outer: loop {
            tokens.push(emit_tok);
            emitted_count += 1;
            if stop_tokens.contains(&emit_tok) || emitted_count >= max_new_tokens {
                break;
            }

            let remaining = max_new_tokens.saturating_sub(emitted_count);
            let n_draft = spec_tokens.min(remaining);
            if n_draft == 0 {
                break;
            }
            let physical_draft_slots = physical_verify_n - 1;
            let carry_tok = emit_tok;
            let start_position = processed_pos + 1;
            let drafts: Vec<i32> = match plan {
                PackedDraftPlan::Recorded(trace) => {
                    let row = trace.get(step_idx).ok_or_else(|| {
                        MtpError::Metal(MetalError::BadShape {
                            kernel: "mtp_decode_packed_n_planned.recorded",
                            detail: format!("missing draft row for step {step_idx}"),
                        })
                    })?;
                    if row.carry_tok != carry_tok || row.start_position != start_position {
                        return Err(MtpError::Metal(MetalError::BadShape {
                            kernel: "mtp_decode_packed_n_planned.recorded",
                            detail: format!(
                                "row {step_idx} alignment mismatch: \
                                 carry {} vs {carry_tok}, pos {} vs {start_position}",
                                row.carry_tok, row.start_position,
                            ),
                        }));
                    }
                    if row.drafts.len() < n_draft {
                        return Err(MtpError::Metal(MetalError::BadShape {
                            kernel: "mtp_decode_packed_n_planned.recorded",
                            detail: format!(
                                "draft row {step_idx} len {} < needed {n_draft}",
                                row.drafts.len()
                            ),
                        }));
                    }
                    row.drafts[..n_draft].to_vec()
                }
                PackedDraftPlan::Oracle(oracle) => {
                    if emitted_count + n_draft > oracle.len() {
                        return Err(MtpError::Metal(MetalError::BadShape {
                            kernel: "mtp_decode_packed_n_planned.oracle",
                            detail: format!(
                                "oracle len {} exhausted at emitted={} need={n_draft}",
                                oracle.len(),
                                emitted_count
                            ),
                        }));
                    }
                    oracle[emitted_count..emitted_count + n_draft].to_vec()
                }
            };
            stats.drafts_attempted += drafts.len() as u32;

            let mut verify_input: Vec<i32> = Vec::with_capacity(physical_verify_n);
            verify_input.push(carry_tok);
            verify_input.extend_from_slice(&drafts);
            for pad_j in drafts.len()..physical_draft_slots {
                let pad_tok = match plan {
                    PackedDraftPlan::Recorded(trace) => trace
                        .get(step_idx)
                        .and_then(|row| row.drafts.get(pad_j))
                        .copied()
                        .unwrap_or(carry_tok),
                    PackedDraftPlan::Oracle(oracle) => oracle
                        .get(emitted_count + pad_j)
                        .copied()
                        .unwrap_or(carry_tok),
                };
                verify_input.push(pad_tok);
            }
            let n_eff = physical_verify_n as u32;

            let t_verify = std::time::Instant::now();
            let verify_argmax = encode_packed_verify_layer_major_inner(
                self.base,
                &last_layer,
                &verify_input,
                start_position,
                verify_scratch,
                layer_scratch,
                base_session,
                None,
                Some(n_eff),
            )?;
            stats.verify_ms += t_verify.elapsed().as_secs_f64() * 1e3;
            stats.base_forward_calls += 1;

            let mut n_accepted = 0usize;
            let mut stop_now = false;
            for (j, &draft_tok) in drafts.iter().enumerate() {
                if draft_tok != verify_argmax[j] {
                    break;
                }
                stats.accepted += 1;
                n_accepted += 1;
                tokens.push(draft_tok);
                emitted_count += 1;
                if stop_tokens.contains(&draft_tok) || emitted_count >= max_new_tokens {
                    stop_now = true;
                    break;
                }
            }

            let n_keep = (1 + n_accepted) as u32;
            if n_keep < n_eff {
                let t_restore = std::time::Instant::now();
                encode_restore_after_partial_accept_inner(
                    self.base,
                    verify_scratch,
                    n_keep,
                    start_position,
                    base_session,
                    Some(n_eff),
                )?;
                stats.restore_ms += t_restore.elapsed().as_secs_f64() * 1e3;
            }

            let t_bridge = std::time::Instant::now();
            let next_hidden = verify_scratch.hidden_capture_n_slot(n_accepted as u32);
            self.write_base_hidden_variant(&next_hidden, &hidden_cur)?;
            stats.bridge_ms += t_bridge.elapsed().as_secs_f64() * 1e3;

            processed_pos += 1 + n_accepted as u32;
            emit_tok = verify_argmax[n_accepted];
            stats.steps += 1;
            step_idx += 1;

            if stop_now {
                break 'outer;
            }
        }

        Ok(DecodeOutput {
            tokens,
            stats: stats.into_finalized(t_start),
        })
    }
}

/// Result of [`SpeculativeDecoder::decode`].
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    /// Full sequence: prompt tokens + emitted tokens.
    pub tokens: Vec<i32>,
    pub stats: SpecStats,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    /// Number of MTP-augmented decode iterations (after prompt prefill).
    pub steps: u32,
    /// Drafts that matched target's argmax.
    pub accepted: u32,
    /// Total drafted tokens proposed for acceptance. For MTP-1 this equals
    /// `steps`; for experimental MTP-N it is `steps * N` minus any terminal
    /// short-circuit on the last outer step.
    pub drafts_attempted: u32,
    /// Total base forward calls (prompt prefill + decode loop).
    pub base_forward_calls: u32,
    /// Total MTP draft / draft_kv_only calls (prefill + decode + bridges).
    pub mtp_calls: u32,
    /// Prompt prefill wall time, including MTP-history prefill when this path
    /// maintains one. Used to normalize against MTPLX decode-only reporting.
    pub prefill_ms: f64,
    /// Decode-loop wall time after prompt/MTP-history prefill.
    pub decode_ms: f64,
    /// Decode-loop wall time spent producing draft tokens or recorded draft work.
    pub draft_ms: f64,
    /// Decode-loop wall time spent in packed target verification.
    pub verify_ms: f64,
    /// Decode-loop wall time spent rolling back rejected packed slots.
    pub restore_ms: f64,
    /// Decode-loop wall time spent repairing/carrying hidden or MTP KV state.
    pub bridge_ms: f64,
    /// Wall time in milliseconds.
    pub wall_ms: f64,
}

impl SpecStats {
    fn into_finalized(mut self, t_start: std::time::Instant) -> Self {
        self.wall_ms = t_start.elapsed().as_secs_f64() * 1e3;
        self.decode_ms = (self.wall_ms - self.prefill_ms).max(0.0);
        self
    }

    /// Acceptance rate α = accepted / steps. Returns 0.0 if no steps ran.
    pub fn acceptance_rate(&self) -> f64 {
        let denom = if self.drafts_attempted > 0 {
            self.drafts_attempted
        } else {
            self.steps
        };
        if denom == 0 {
            0.0
        } else {
            self.accepted as f64 / denom as f64
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn argmax_i32(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

// ----- Tests -----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{Forward, GdnState, KvCache};
    use crate::loader::Model;
    use crate::metal_forward::MetalModel;

    #[test]
    fn mtp_moe_bank_policy_parser_is_strict() {
        assert_eq!(
            MtpMoeBankPolicy::parse(None).unwrap(),
            MtpMoeBankPolicy::F32
        );
        assert_eq!(
            MtpMoeBankPolicy::parse(Some("0")).unwrap(),
            MtpMoeBankPolicy::F32
        );
        assert_eq!(
            MtpMoeBankPolicy::parse(Some("gate_up")).unwrap(),
            MtpMoeBankPolicy::GateUp
        );
        assert_eq!(
            MtpMoeBankPolicy::parse(Some("down")).unwrap(),
            MtpMoeBankPolicy::Down
        );
        assert_eq!(
            MtpMoeBankPolicy::parse(Some("1")).unwrap(),
            MtpMoeBankPolicy::All
        );
        assert_eq!(
            MtpMoeBankPolicy::parse(Some("all")).unwrap(),
            MtpMoeBankPolicy::All
        );
        assert!(MtpMoeBankPolicy::parse(Some("true")).is_err());
    }

    #[test]
    #[ignore = "requires local A3B Q4_K_M MTP fixture and several GiB of Metal memory"]
    fn mtp_moe_native_bank_fixture_ledger() {
        let path =
            "/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mtp-moe-bank-ledger] skipped - fixture missing");
            return;
        }
        let ctx = MetalContext::new().expect("metal context");
        let g = GgufFile::open(path).expect("open fixture");
        let m = Model::from_gguf(&g).expect("load model");
        let mtp = m.mtp.as_ref().expect("MTP head");
        assert_eq!(mtp.block_idx, 40);
        let source = mtp.attn.ffn_moe.as_ref().expect("MoE MTP block");
        assert_eq!(source.gate_exps.dtype, GgmlType::Q4_K);
        assert_eq!(source.up_exps.dtype, GgmlType::Q4_K);
        assert_eq!(source.down_exps.dtype, GgmlType::Q5_K);
        assert_eq!(source.gate_exps.shape, vec![2048, 512, 256]);
        assert_eq!(source.up_exps.shape, vec![2048, 512, 256]);
        assert_eq!(source.down_exps.shape, vec![512, 2048, 256]);

        let expected = [
            (
                MtpMoeBankPolicy::F32,
                [GgmlType::F32, GgmlType::F32, GgmlType::F32],
                3_221_225_472u64,
            ),
            (
                MtpMoeBankPolicy::GateUp,
                [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::F32],
                1_375_731_712u64,
            ),
            (
                MtpMoeBankPolicy::Down,
                [GgmlType::F32, GgmlType::F32, GgmlType::Q5_K],
                2_332_033_024u64,
            ),
            (
                MtpMoeBankPolicy::All,
                [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q5_K],
                486_539_264u64,
            ),
        ];
        for (policy, dtypes, bytes) in expected {
            let head = MetalMtpHead::load_with_moe_bank_policy(&ctx, &g, mtp, policy)
                .expect("load MTP bank policy");
            let moe = head.attn.ffn_moe.as_ref().expect("loaded MoE MTP block");
            assert_eq!(
                [moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype],
                dtypes
            );
            let loaded_bytes =
                moe.gate_exps.n_bytes() + moe.up_exps.n_bytes() + moe.down_exps.n_bytes();
            assert_eq!(loaded_bytes, bytes);
        }
    }

    #[test]
    #[ignore = "requires local A3B Q4_K_M MTP fixture"]
    fn mtp_moe_native_banks_match_f32_draft_logits() {
        let path =
            "/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mtp-moe-native-oracle] skipped - fixture missing");
            return;
        }
        let ctx = MetalContext::new().expect("metal context");
        let g = GgufFile::open(path).expect("open fixture");
        let m = Model::from_gguf(&g).expect("load model");
        let mtp = m.mtp.as_ref().expect("MTP head");
        let mm = MetalModel::load(&ctx, &g, &m).expect("load base model");
        let f32_head =
            MetalMtpHead::load_with_moe_bank_policy(&ctx, &g, mtp, MtpMoeBankPolicy::F32)
                .expect("load F32 MTP head");
        let native_head =
            MetalMtpHead::load_with_moe_bank_policy(&ctx, &g, mtp, MtpMoeBankPolicy::All)
                .expect("load native MTP head");
        let mf = MetalForward::new(&ctx, &mm);
        let tok = crate::tokenizer::Tokenizer::open(path).expect("tokenizer");
        let ids = tok
            .encode("Write a Python function that parses a GGUF header.", false)
            .expect("tokenize prompt");
        assert!(ids.len() >= 4);

        let h = m.arch.hidden_size as u64;
        let mut base_session = MetalSession::fresh(&ctx, &mm, 16).expect("base session");
        let mut hiddens = Vec::new();
        for (position, &token) in ids.iter().take(3).enumerate() {
            let hidden = MetalTensor::zeros_f32(&ctx, vec![h]).expect("hidden capture");
            mf.single_token_argmax_with_hidden(
                token,
                position as u32,
                &mut base_session,
                &hidden,
                true,
            )
            .expect("base hidden capture");
            hiddens.push(hidden);
        }

        let f32_session =
            MetalMtpSession::fresh(&ctx, &f32_head, &m.arch, 16).expect("F32 session");
        let native_session =
            MetalMtpSession::fresh(&ctx, &native_head, &m.arch, 16).expect("native session");
        let mut f32_spec = SpeculativeDecoder::new(&mf, &f32_head, f32_session);
        let mut native_spec = SpeculativeDecoder::new(&mf, &native_head, native_session);

        for step in 0..3 {
            let next_tok = ids[step + 1];
            let f32_logits = f32_spec
                .draft_inner(
                    next_tok,
                    &hiddens[step],
                    step as u32,
                    DraftReadback::FullLogits,
                )
                .expect("F32 MTP draft")
                .logits
                .expect("F32 logits");
            let native_logits = native_spec
                .draft_inner(
                    next_tok,
                    &hiddens[step],
                    step as u32,
                    DraftReadback::FullLogits,
                )
                .expect("native MTP draft")
                .logits
                .expect("native logits");

            let mut dot = 0.0f64;
            let mut f32_norm = 0.0f64;
            let mut native_norm = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut top1 = (0usize, f32::NEG_INFINITY);
            let mut top2 = f32::NEG_INFINITY;
            for (idx, (&reference, &candidate)) in
                f32_logits.iter().zip(native_logits.iter()).enumerate()
            {
                max_abs = max_abs.max((reference - candidate).abs());
                dot += reference as f64 * candidate as f64;
                f32_norm += (reference as f64).powi(2);
                native_norm += (candidate as f64).powi(2);
                if reference > top1.1 {
                    top2 = top1.1;
                    top1 = (idx, reference);
                } else if reference > top2 {
                    top2 = reference;
                }
            }
            let cosine = dot / (f32_norm.sqrt() * native_norm.sqrt() + 1e-30);
            let f32_argmax = argmax_i32(&f32_logits);
            let native_argmax = argmax_i32(&native_logits);
            let margin = top1.1 - top2;
            eprintln!(
                "[mtp-moe-native-oracle] step={step} cos={cosine:.6} max_abs={max_abs:.6} margin={margin:.6} argmax={f32_argmax}/{native_argmax}"
            );
            assert!(cosine > 0.9999, "step {step}: cosine {cosine}");
            assert!(max_abs < 0.05, "step {step}: max_abs {max_abs}");
            if margin > 2.0 * max_abs {
                assert_eq!(f32_argmax, native_argmax, "step {step}: stable argmax");
            }
        }
    }

    /// H4.2 cosine test: Metal MTP draft logits must match CPU MTP draft
    /// logits at cosine ≥ 0.9999 for identical `(next_tok, prev_hidden,
    /// position)` inputs across multiple sequential steps.
    ///
    /// Approach: build `prev_hidden` once via the CPU base forward
    /// (`Forward::single_token_with_hidden`), upload as a Metal tensor,
    /// then run BOTH `SpeculativeDecoder::draft` and `Forward::mtp_step`
    /// on the same input quadruple. Both impls share the same GGUF
    /// weights; the only variance is the kernel substrate. Drift should
    /// be at the quant-noise floor (~1e-3 for Q4_K).
    ///
    /// Exercises:
    /// * step 0: empty MTP KV (attention against self only)
    /// * step 1: 1-entry KV (first-position drift through encode_mtp_attn)
    /// * step 2: 2-entry KV (KV reads + RoPE position increment)
    #[test]
    fn metal_mtp_draft_matches_cpu() {
        let path = "/Users/tito/models/h4-smoke-test/Qwen3.5-0.8B/qwen3.5-0.8b.Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mtp-metal-cosine] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init: {e}"),
        };

        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        assert!(m.mtp.is_some(), "model must have MTP head");
        let mtp_view = m.mtp.as_ref().unwrap();

        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).expect("metal mtp load");
        let mtp_session = MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, 64).expect("session");

        let mf = MetalForward::new(&ctx, &mm);
        let cpu = Forward::new(&g, &m);
        let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);

        // Run a 3-token base prompt forward on CPU to get hiddens
        // h_0, h_1, h_2 — these become `prev_hidden` for our MTP test.
        // We use CPU base forward here to keep the test self-contained
        // (Metal-vs-CPU-base parity is covered elsewhere). The MTP path
        // operates on the same hidden values regardless.
        let tok = crate::tokenizer::Tokenizer::open(path).expect("tok");
        let ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tok");
        eprintln!("[mtp-metal-cosine] prompt tokens: {ids:?}");
        assert!(ids.len() >= 3, "need ≥3 prompt tokens");

        let mut state = GdnState::fresh(&m);
        let mut kv = KvCache::new(&m);
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(3);
        for (i, &tid) in ids.iter().take(3).enumerate() {
            let (_logits, hidden) = cpu
                .single_token_with_hidden(tid, i as u32, &mut state, &mut kv)
                .expect("base forward");
            hiddens.push(hidden);
        }

        // For each of 3 steps, run both CPU mtp_step and Metal draft on
        // matching inputs. Use position=i (the slot of h_i) and a
        // synthetic next_tok per the §1.2 contract (just pick the next
        // prompt id; the value is arbitrary for a kernel-cosine test as
        // long as both paths see the same one).
        let h = m.arch.hidden_size as usize;
        let mut cpu_mtp_kv = cpu.fresh_mtp_kv(64);
        for step in 0..3 {
            let position = step as u32;
            let next_tok = ids[(step + 1).min(ids.len() - 1)];
            let prev_hidden_cpu: &[f32] = &hiddens[step];

            // CPU MTP step.
            let cpu_logits = cpu
                .mtp_step(next_tok, prev_hidden_cpu, position, &mut cpu_mtp_kv)
                .expect("cpu mtp_step");
            assert_eq!(cpu_logits.len(), m.arch.vocab_size as usize);

            // Metal MTP draft on the same inputs.
            // Upload prev_hidden as a fresh F32 tensor (same lifetime as
            // the call — we don't reuse it).
            let prev_hidden_metal = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(prev_hidden_cpu),
                vec![h as u64],
                GgmlType::F32,
            )
            .expect("upload prev_hidden");
            let metal_logits = spec
                .draft_inner(
                    next_tok,
                    &prev_hidden_metal,
                    position,
                    DraftReadback::FullLogits,
                )
                .expect("metal mtp draft")
                .logits
                .expect("full logits");
            assert_eq!(metal_logits.len(), cpu_logits.len());

            let mut max_abs = 0.0f32;
            let mut argmax_metal = 0usize;
            let mut argmax_cpu = 0usize;
            let mut max_metal = f32::NEG_INFINITY;
            let mut max_cpu = f32::NEG_INFINITY;
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..cpu_logits.len() {
                let d = (metal_logits[i] - cpu_logits[i]).abs();
                max_abs = max_abs.max(d);
                if metal_logits[i] > max_metal {
                    max_metal = metal_logits[i];
                    argmax_metal = i;
                }
                if cpu_logits[i] > max_cpu {
                    max_cpu = cpu_logits[i];
                    argmax_cpu = i;
                }
                dot += metal_logits[i] as f64 * cpu_logits[i] as f64;
                na += (metal_logits[i] as f64).powi(2);
                nb += (cpu_logits[i] as f64).powi(2);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            eprintln!(
                "[mtp-metal-cosine] step={step} pos={position} next_tok={next_tok}: \
                 metal_argmax={argmax_metal}({:.4}) cpu_argmax={argmax_cpu}({:.4}) \
                 max|Δ|={max_abs:.2e} cos={cos:.6}",
                max_metal, max_cpu
            );

            assert_eq!(argmax_metal, argmax_cpu, "step {step}: argmax disagreement");
            // Q4_K weights through 1 attn + 1 FFN block. ~1e-3 max|Δ|
            // is in line with the existing single-token base oracle test.
            assert!(cos > 0.9999, "step {step}: cos {cos} below threshold");
            assert!(max_abs < 0.05, "step {step}: max|Δ| {max_abs} above floor");
        }
    }

    /// H4.3 greedy generation equivalence: with MTP=on and MTP=off,
    /// generated token sequences must be IDENTICAL up to max_new_tokens.
    /// This is the real correctness gate for the speculative decode loop
    /// (cursor accounting, inline bridge, prompt prefill, accept/reject
    /// logic). Under greedy verify both paths emit `argmax(target_logits)`
    /// at every position, so any deviation indicates a bug — most likely
    /// in the cursor state machine, prefill streaming, or inline bridge
    /// ordering.
    ///
    /// Reports acceptance rate α as a sanity signal. If α << 0.3 on
    /// natural prose, the MTP forward distribution is broken (not
    /// causing token corruption since target wins on reject, but
    /// silently negating any speculative speedup). For this 0.8B-MTP
    /// smoke test we just assert α > 0 to confirm the MTP forward
    /// produces SOMETHING reasonable.
    #[test]
    fn greedy_equivalence_0_8b_mtp() {
        let path = "/Users/tito/models/h4-smoke-test/Qwen3.5-0.8B/qwen3.5-0.8b.Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("[mtp-equiv] skipped — fixture missing");
            return;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(e) => panic!("init: {e}"),
        };

        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mtp_view = m.mtp.as_ref().expect("MTP head present");

        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).expect("mtp load");
        let mf = MetalForward::new(&ctx, &mm);

        let tok = crate::tokenizer::Tokenizer::open(path).expect("tok");
        let prompt_ids = tok
            .encode("The quick brown fox jumps over the lazy dog", false)
            .expect("tok");
        let max_new_tokens = 16usize;
        // Source the stop set from the GGUF itself (single source of
        // truth). For the 0.8B instruct fixture this resolves to
        // [248046] (`<|im_end|>`).
        let stop_tokens = g.stop_token_ids().expect("declared stop tokens");
        assert_eq!(stop_tokens, vec![248046], "0.8B instruct stop set");

        // ---------- Reference: MTP=off greedy generation ----------
        // Run base alone, capturing the last logits each iter so we can
        // argmax-greedy the next token.
        let mut ref_session =
            MetalSession::fresh(&ctx, &mm, prompt_ids.len() + max_new_tokens + 8).expect("session");
        let mut ref_tokens = prompt_ids.clone();
        let mut last_logits = Vec::new();
        for (i, &tid) in prompt_ids.iter().enumerate() {
            last_logits = mf
                .single_token(tid, i as u32, &mut ref_session)
                .expect("base forward");
        }
        let mut next_tok = argmax_i32(&last_logits);
        let mut pos = (prompt_ids.len() - 1) as u32;
        for _ in 0..max_new_tokens {
            ref_tokens.push(next_tok);
            if stop_tokens.contains(&next_tok) {
                break;
            }
            pos += 1;
            let logits = mf
                .single_token(next_tok, pos, &mut ref_session)
                .expect("base forward decode");
            next_tok = argmax_i32(&logits);
        }
        let ref_generated = &ref_tokens[prompt_ids.len()..];
        eprintln!(
            "[mtp-equiv] ref (MTP=off) generated {} tokens: {:?}",
            ref_generated.len(),
            ref_generated
        );

        // ---------- Test: MTP=on speculative-decode generation ----------
        let mtp_session = MetalMtpSession::fresh(
            &ctx,
            &mtp_head,
            &m.arch,
            prompt_ids.len() + max_new_tokens + 8,
        )
        .expect("mtp session");
        let mut spec_session =
            MetalSession::fresh(&ctx, &mm, prompt_ids.len() + max_new_tokens + 8).expect("session");
        let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
        let result = spec
            .decode(&prompt_ids, max_new_tokens, &stop_tokens, &mut spec_session)
            .expect("spec decode");
        let mtp_generated = &result.tokens[prompt_ids.len()..];
        eprintln!(
            "[mtp-equiv] spec (MTP=on) generated {} tokens: {:?}",
            mtp_generated.len(),
            mtp_generated
        );
        eprintln!(
            "[mtp-equiv] stats: steps={} accepted={} α={:.3} base_calls={} mtp_calls={} wall={:.1}ms",
            result.stats.steps,
            result.stats.accepted,
            result.stats.acceptance_rate(),
            result.stats.base_forward_calls,
            result.stats.mtp_calls,
            result.stats.wall_ms,
        );

        // Token sequences MUST be identical (greedy + greedy verify ⇒
        // emitted = argmax(target) at every position).
        assert_eq!(
            mtp_generated.len(),
            ref_generated.len(),
            "spec generated {} tokens but ref generated {}",
            mtp_generated.len(),
            ref_generated.len()
        );
        for (i, (s, r)) in mtp_generated.iter().zip(ref_generated.iter()).enumerate() {
            assert_eq!(s, r, "token mismatch at gen-position {i}: spec={s} ref={r}",);
        }

        // Sanity: MTP forward did SOMETHING. α=0 on a 16-token sample
        // would mean every draft missed — possible for a small model
        // but worth flagging.
        assert!(
            result.stats.steps > 0,
            "no decode steps ran (max_new_tokens or prompt issue)"
        );
        eprintln!("[mtp-equiv] greedy equivalence: PASS");
    }
}

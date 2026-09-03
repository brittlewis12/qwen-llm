//! Bench helpers chaining many dispatches into one command buffer, plus the
//! bench-only trellis3 decode-floor probe.

use super::*;

/// Chain `n_dispatches` Q4_K mat-vecs into one command buffer, one wait at
/// the end. Used for kernel benchmarks; the production forward pass uses
/// `encode_*` directly with its own command-buffer orchestration.
pub fn bench_q4_k_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_vec_q4_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same but for Q5_K. Used by the v0.73a.0 A-lite go/no-go gate
/// (`q5_k_mat_mat_amortization_vs_n_mat_vec`).
pub fn bench_q5_k_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_vec_q5_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same chained bench harness as `bench_q4_k_mat_mat_chained`, for Q5_K.
/// Used by the v0.73a.0 A-lite go/no-go gate to compare amortized
/// weight-BW of mat-mat (one panel-reload per K-step shared across
/// N_QUERY cols) vs N_QUERY successive mat-vec.
pub fn bench_q5_k_mat_mat_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_mat_q5_k_f32(ctx, &enc, weight, x, y, n_in, n_out, n_query)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same but for Q6_K.
pub fn bench_q6_k_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_vec_q6_k_f32(ctx, &enc, weight, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Same chained bench harness as `bench_q4_k_mat_mat_chained`, for Q6_K.
pub fn bench_q6_k_mat_mat_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_mat_q6_k_f32(ctx, &enc, weight, x, y, n_in, n_out, n_query)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Chain `n_dispatches` Q4_K **mat-mat** (with N_QUERY columns) into one
/// command buffer. Used to bench the lifted llama mat-mat tile against
/// theoretical peak BW and against `n_dispatches × n_query` mat-vec
/// calls.
///
/// Per H5.3b plan rev 6: this is the H5.3b.1–3 perf gate. We expect:
///   * GiB/s should approach the chained64 mat-vec ceiling (77-88% peak
///     for our existing fast Q4_K kernel) because mat-mat amortizes
///     weight reads across N_QUERY columns rather than re-reading.
///   * Wall time vs `n_dispatches × n_query mat_vec` should be ~N_QUERY×
///     less if the BW ceiling is the same — that's the entire point of
///     mat-mat.
pub fn bench_q4_k_mat_mat_chained(
    ctx: &MetalContext,
    weight: &MetalTensor,
    x: &MetalTensor, // [n_query, n_in] row-major F32
    y: &MetalTensor, // [n_out, n_query] col-major F32
    n_in: usize,
    n_out: usize,
    n_query: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_mat_q4_k_f32(ctx, &enc, weight, x, y, n_in, n_out, n_query)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Decode-code variant for the trellis floor kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trellis3Variant {
    /// Per-weight computed code: LCG + fp16 field mask, half2 sum.
    ThreeInst,
    /// One LCG hash yields two weights (V=2): half2 lanes are the values.
    ThreeInstV2,
    /// Per-weight two 256-entry threadgroup-memory LUT lookups, summed.
    Lut8x2,
    /// QTIP HYB-style V=2: one hash + one device-cached half2 lookup per
    /// two weights, sign of .y flipped by hash bit 15 (pre-run amendment
    /// per cx research session 019f7c4b).
    HybV2,
    /// T=256 group-ring window discipline (T6), V=1 per-weight code.
    ThreeInstG,
    /// T=256 group-ring window discipline (T6), V=2 split code.
    ThreeInstV2G,
    /// T6 tuning iteration 1: NSG=4 occupancy variant of ThreeInstG.
    ThreeInstGNsg4,
    /// T6 tuning iteration 3: NR0=4 row-ILP variant of ThreeInstG.
    ThreeInstGNr4,
    /// T8b: dual-3INST group-ring V=2 (rotl+xor second hash word).
    ThreeInstDG,
}

impl Trellis3Variant {
    pub fn kernel_name(self) -> &'static str {
        match self {
            Trellis3Variant::ThreeInst => "kernel_mat_vec_trellis3_3inst_f32",
            Trellis3Variant::ThreeInstV2 => "kernel_mat_vec_trellis3_3inst_v2_f32",
            Trellis3Variant::Lut8x2 => "kernel_mat_vec_trellis3_lut8x2_f32",
            Trellis3Variant::HybV2 => "kernel_mat_vec_trellis3_hyb_v2_f32",
            Trellis3Variant::ThreeInstG => "kernel_mat_vec_trellis3g_3inst_f32",
            Trellis3Variant::ThreeInstV2G => "kernel_mat_vec_trellis3g_3inst_v2_f32",
            Trellis3Variant::ThreeInstGNsg4 => "kernel_mat_vec_trellis3g_3inst_nsg4_f32",
            Trellis3Variant::ThreeInstGNr4 => "kernel_mat_vec_trellis3g_3inst_nr4_f32",
            Trellis3Variant::ThreeInstDG => "kernel_mat_vec_trellis3g_3inst_d_f32",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Trellis3Variant::ThreeInst => "3inst",
            Trellis3Variant::ThreeInstV2 => "3inst_v2",
            Trellis3Variant::Lut8x2 => "lut8x2",
            Trellis3Variant::HybV2 => "hyb_v2",
            Trellis3Variant::ThreeInstG => "3inst_g256",
            Trellis3Variant::ThreeInstV2G => "3inst_v2_g256",
            Trellis3Variant::ThreeInstGNsg4 => "3inst_g256_nsg4",
            Trellis3Variant::ThreeInstGNr4 => "3inst_g256_nr4",
            Trellis3Variant::ThreeInstDG => "3inst_d_g256",
        }
    }
}

pub(crate) const T3_GROUP_BYTES: usize = 96; // 8 spans x 12 B
const T3_GROUP_W: usize = 256;

pub(crate) const T3_LCG_A: u32 = 89226354;

pub(crate) const T3_LCG_B: u32 = 64248484;

pub(crate) const T3_MASK: u32 = 0x8FFF_8FFF;

pub(crate) const T3_FIXED: u32 = 0x3B60_3B60 & !T3_MASK;

/// Bytes of packed trellis weights + scales for a logical [n_in, n_out]
/// matrix (the number the bench charges as "compressed bytes moved").
pub fn trellis3_compressed_bytes(n_in: usize, n_out: usize) -> u64 {
    let nb = n_in / T3_GROUP_W;
    (n_out * nb) as u64 * (T3_GROUP_BYTES as u64 + 2)
}

/// Synthetic trellis tensor: seeded random bitstream + scales + LUT.
/// Throughput/mechanical-correctness fixture only; carries no quality claim.
pub struct Trellis3Synthetic {
    /// `n_out * (n_in/256) * 96` bytes of span words (LSB-first packing).
    pub weight: Vec<u8>,
    /// `n_out * (n_in/256)` fp16 bit patterns (per-group scales).
    pub scales_f16: Vec<u16>,
    /// 1024 fp16 bit patterns. Lut8x2 reads entries 0..512 (256 hi-byte,
    /// then 256 lo-byte). HybV2 reads all 1024 as 512 half2 pairs.
    pub lut_f16: Vec<u16>,
}

pub(crate) fn t3_splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn trellis3_synthetic(n_in: usize, n_out: usize, seed: u64) -> Trellis3Synthetic {
    assert_eq!(n_in % T3_GROUP_W, 0, "n_in must be a multiple of 256");
    let nb = n_in / T3_GROUP_W;
    let mut s = seed;
    let n_weight_bytes = n_out * nb * T3_GROUP_BYTES;
    let mut weight = vec![0u8; n_weight_bytes];
    for chunk in weight.chunks_mut(8) {
        let v = t3_splitmix64(&mut s).to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&v[..n]);
    }
    // Scales in ~[0.5, 1.5); LUT entries in ~[-1, 1).
    let mut scales_f16 = Vec::with_capacity(n_out * nb);
    for _ in 0..n_out * nb {
        let u = (t3_splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32;
        scales_f16.push(half::f16::from_f32(0.5 + u).to_bits());
    }
    let mut lut_f16 = Vec::with_capacity(1024);
    for _ in 0..1024 {
        let u = (t3_splitmix64(&mut s) >> 40) as f32 / (1u64 << 24) as f32;
        lut_f16.push(half::f16::from_f32(2.0 * u - 1.0).to_bits());
    }
    Trellis3Synthetic {
        weight,
        scales_f16,
        lut_f16,
    }
}

/// 16-bit window ending at ring bit `end` (exclusive) of a 768-bit
/// group ring stored LSB-first in 24 words (T=256 discipline).
pub(crate) fn t3_group_ring_state(words: &[u32; 24], end: u32) -> u32 {
    let mut st = 0u32;
    let start = (end + 768 - 16) % 768;
    for i in 0..16 {
        let b = ((start + i) % 768) as usize;
        st |= ((words[b >> 5] >> (b & 31)) & 1) << i;
    }
    st
}

pub(crate) fn t3_ring_extract16(w0: u32, w1: u32, w2: u32, o: u32) -> u32 {
    let a = o >> 5;
    let s = o & 31;
    let wa = match a {
        0 => w0,
        1 => w1,
        _ => w2,
    };
    let wb = match a {
        0 => w1,
        1 => w2,
        _ => w0,
    };
    let v = if s == 0 {
        wa
    } else {
        (wa >> s) | (wb << (32 - s))
    };
    v & 0xFFFF
}

pub(crate) fn t3_hash_half2(st: u32) -> (half::f16, half::f16) {
    let x = st.wrapping_mul(T3_LCG_A).wrapping_add(T3_LCG_B);
    let hb = (x & T3_MASK) | T3_FIXED;
    (
        half::f16::from_bits((hb & 0xFFFF) as u16),
        half::f16::from_bits((hb >> 16) as u16),
    )
}

/// Half FMA with Metal `fma(half, half, half)` semantics: infinitely
/// precise multiply-add, single rounding to f16. f64 holds the exact
/// intermediate for all f16 inputs.
pub(crate) fn t3_fma_h(w: half::f16, y: half::f16, acc: half::f16) -> half::f16 {
    half::f16::from_f64(w.to_f64() * y.to_f64() + acc.to_f64())
}

/// CPU reference GEMV, bit-faithful to the Metal kernels' per-span
/// accumulation (half accumulate within a span, f32 scale fold per group).
/// Cross-span/group f32 summation order differs from the GPU's simd tree;
/// the correctness gate uses a tolerance for that.
pub fn trellis3_cpu_reference(
    variant: Trellis3Variant,
    syn: &Trellis3Synthetic,
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), n_in);
    let nb = n_in / T3_GROUP_W;
    let lut = |i: usize| half::f16::from_bits(syn.lut_f16[i]);
    let mut y = vec![0f32; n_out];
    for (row, yr) in y.iter_mut().enumerate() {
        let mut sum = 0f32;
        for ib in 0..nb {
            let sc = half::f16::from_bits(syn.scales_f16[row * nb + ib]).to_f32();
            let gbase = row * nb * T3_GROUP_BYTES + ib * T3_GROUP_BYTES;
            let mut gwords = [0u32; 24];
            for (i, gw) in gwords.iter_mut().enumerate() {
                *gw = u32::from_le_bytes(
                    syn.weight[gbase + 4 * i..gbase + 4 * i + 4]
                        .try_into()
                        .unwrap(),
                );
            }
            let gwords = &gwords;
            for span in 0..8 {
                let base = gbase + span * 12;
                let wle = |i: usize| {
                    u32::from_le_bytes(
                        syn.weight[base + 4 * i..base + 4 * i + 4]
                            .try_into()
                            .unwrap(),
                    )
                };
                let (w0, w1, w2) = (wle(0), wle(1), wle(2));
                let mut acc = half::f16::from_f32(0.0);
                match variant {
                    Trellis3Variant::ThreeInst => {
                        for j in 0..32u32 {
                            let o = (3 * j + 83) % 96;
                            let st = t3_ring_extract16(w0, w1, w2, o);
                            let (hx, hy) = t3_hash_half2(st);
                            let w = half::f16::from_f32(hx.to_f32() + hy.to_f32());
                            let yv =
                                half::f16::from_f32(x[ib * T3_GROUP_W + span * 32 + j as usize]);
                            acc = t3_fma_h(w, yv, acc);
                        }
                    }
                    Trellis3Variant::ThreeInstV2 => {
                        for t in 0..16u32 {
                            let o = (6 * t + 86) % 96;
                            let st = t3_ring_extract16(w0, w1, w2, o);
                            let (hx, hy) = t3_hash_half2(st);
                            let y0 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize],
                            );
                            let y1 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize + 1],
                            );
                            acc = t3_fma_h(hx, y0, acc);
                            acc = t3_fma_h(hy, y1, acc);
                        }
                    }
                    Trellis3Variant::Lut8x2 => {
                        for j in 0..32u32 {
                            let o = (3 * j + 83) % 96;
                            let st = t3_ring_extract16(w0, w1, w2, o) as usize;
                            let w = half::f16::from_f32(
                                lut(st >> 8).to_f32() + lut(256 + (st & 255)).to_f32(),
                            );
                            let yv =
                                half::f16::from_f32(x[ib * T3_GROUP_W + span * 32 + j as usize]);
                            acc = t3_fma_h(w, yv, acc);
                        }
                    }
                    Trellis3Variant::HybV2 => {
                        for t in 0..16u32 {
                            let o = (6 * t + 86) % 96;
                            let st = t3_ring_extract16(w0, w1, w2, o);
                            let h = st.wrapping_mul(st).wrapping_add(st);
                            let idx = ((h >> 6) & 511) as usize;
                            let vx = lut(2 * idx);
                            let vy_bits = lut(2 * idx + 1).to_bits() ^ (h & 0x8000) as u16;
                            let vy = half::f16::from_bits(vy_bits);
                            let y0 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize],
                            );
                            let y1 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize + 1],
                            );
                            acc = t3_fma_h(vx, y0, acc);
                            acc = t3_fma_h(vy, y1, acc);
                        }
                    }
                    Trellis3Variant::ThreeInstG
                    | Trellis3Variant::ThreeInstGNsg4
                    | Trellis3Variant::ThreeInstGNr4 => {
                        for l in 0..32u32 {
                            let j = span as u32 * 32 + l;
                            let st = t3_group_ring_state(gwords, 3 * (j + 1));
                            let (hx, hy) = t3_hash_half2(st);
                            let w = half::f16::from_f32(hx.to_f32() + hy.to_f32());
                            let yv =
                                half::f16::from_f32(x[ib * T3_GROUP_W + span * 32 + l as usize]);
                            acc = t3_fma_h(w, yv, acc);
                        }
                    }
                    Trellis3Variant::ThreeInstDG => {
                        for t in 0..16u32 {
                            let tj = span as u32 * 16 + t;
                            let st = t3_group_ring_state(gwords, 6 * (tj + 1));
                            let h = st.wrapping_mul(T3_LCG_A).wrapping_add(T3_LCG_B);
                            let g = h ^ h.rotate_left(13);
                            let ha = (h & T3_MASK) | T3_FIXED;
                            let hb = (g & T3_MASK) | T3_FIXED;
                            let ax = half::f16::from_bits((ha & 0xFFFF) as u16).to_f32();
                            let ay = half::f16::from_bits((ha >> 16) as u16).to_f32();
                            let bx = half::f16::from_bits((hb & 0xFFFF) as u16).to_f32();
                            let by = half::f16::from_bits((hb >> 16) as u16).to_f32();
                            let w0 = half::f16::from_f32(ax + ay);
                            let w1 = half::f16::from_f32(bx + by);
                            let y0 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize],
                            );
                            let y1 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize + 1],
                            );
                            acc = t3_fma_h(w0, y0, acc);
                            acc = t3_fma_h(w1, y1, acc);
                        }
                    }
                    Trellis3Variant::ThreeInstV2G => {
                        for t in 0..16u32 {
                            let tj = span as u32 * 16 + t;
                            let st = t3_group_ring_state(gwords, 6 * (tj + 1));
                            let (hx, hy) = t3_hash_half2(st);
                            let y0 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize],
                            );
                            let y1 = half::f16::from_f32(
                                x[ib * T3_GROUP_W + span * 32 + 2 * t as usize + 1],
                            );
                            acc = t3_fma_h(hx, y0, acc);
                            acc = t3_fma_h(hy, y1, acc);
                        }
                    }
                }
                sum = sc.mul_add(acc.to_f32(), sum);
            }
        }
        *yr = sum;
    }
    y
}

pub fn encode_mat_vec_trellis3_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    variant: Trellis3Variant,
    weight: &MetalTensor,
    scales: &MetalTensor,
    lut: Option<&MetalTensor>,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MetalError> {
    if !n_in.is_multiple_of(T3_GROUP_W) {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_trellis3",
            detail: format!("n_in={n_in} not divisible by 256"),
        });
    }
    if variant == Trellis3Variant::Lut8x2 && lut.is_none() {
        return Err(MetalError::BadShape {
            kernel: "mat_vec_trellis3",
            detail: "lut8x2 variant requires a LUT tensor".to_string(),
        });
    }
    let pso = ctx.pipeline(variant.kernel_name())?;
    enc.set_pipeline(&pso);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
    }
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            n_out: n_out as u32,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, scales);
    enc.set_tensor(3, x);
    enc.set_tensor(4, y);
    if let Some(l) = lut {
        enc.set_tensor(5, l);
    }

    let nr0: usize = match variant {
        Trellis3Variant::ThreeInstGNr4 => 4,
        _ => 2,
    };
    let nsg: usize = match variant {
        Trellis3Variant::ThreeInstGNsg4 => 4,
        _ => 2,
    };
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(nr0 * nsg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: nsg * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn bench_trellis3_chained(
    ctx: &MetalContext,
    variant: Trellis3Variant,
    weight: &MetalTensor,
    scales: &MetalTensor,
    lut: Option<&MetalTensor>,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_dispatches: usize,
) -> Result<(), MetalError> {
    let cmd_buf = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd_buf);
    for _ in 0..n_dispatches {
        encode_mat_vec_trellis3_f32(ctx, &enc, variant, weight, scales, lut, x, y, n_in, n_out)?;
    }
    enc.end();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(())
}

/// Upload a synthetic trellis fixture as Metal tensors.
/// The weight byte blob is carried as an F32-typed buffer (bench-only
/// convenience; the kernel reads raw bytes and no dtype validation applies).
pub fn trellis3_upload(
    ctx: &MetalContext,
    syn: &Trellis3Synthetic,
) -> Result<(MetalTensor, MetalTensor, MetalTensor), MetalError> {
    assert_eq!(syn.weight.len() % 4, 0);
    let w_t = MetalTensor::from_bytes(
        ctx,
        &syn.weight,
        vec![(syn.weight.len() / 4) as u64],
        GgmlType::F32,
    )?;
    let s_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&syn.scales_f16),
        vec![syn.scales_f16.len() as u64],
        GgmlType::F16,
    )?;
    let l_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&syn.lut_f16),
        vec![syn.lut_f16.len() as u64],
        GgmlType::F16,
    )?;
    Ok((w_t, s_t, l_t))
}

pub fn mat_vec_trellis3_f32_readback_for_test(
    ctx: &MetalContext,
    variant: Trellis3Variant,
    syn: &Trellis3Synthetic,
    x: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MetalError> {
    let (w_t, s_t, l_t) = trellis3_upload(ctx, syn)?;
    let x_t = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(x),
        vec![n_in as u64],
        GgmlType::F32,
    )?;
    let y_t = MetalTensor::zeros_f32(ctx, vec![n_out as u64])?;
    one_shot(ctx, |enc| {
        encode_mat_vec_trellis3_f32(
            ctx,
            enc,
            variant,
            &w_t,
            &s_t,
            Some(&l_t),
            &x_t,
            &y_t,
            n_in,
            n_out,
        )
    })?;
    Ok(read_back_f32(&y_t.buffer, n_out))
}

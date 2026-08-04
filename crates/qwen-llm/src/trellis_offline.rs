//! Tooling-only offline trellis encoder (T7+, docs/bench/2026-07-19-
//! trellis3-real-weight-oracle/). Exact Viterbi over the L=16 bitshift
//! trellis at K=3 bits/weight, T=256 group-ring windows, matching the
//! bench-only decode kernels in kernels/trellis_gemv_floor.metal
//! bit-for-bit (bridge-tested against `metal::trellis3_cpu_reference`).
//!
//! **Not a production path.** No GPU, no model mutation — this exists so
//! fidelity oracles and (later) a PTQ pipeline can encode real tensors
//! against the exact shipped decode functions.

const L: usize = 16;
const NSTATES: usize = 1 << L;
const GROUP_W: usize = 256;
const RING_BITS: usize = GROUP_W * 3; // 768
const GROUP_WORDS: usize = RING_BITS / 32; // 24

// Must match kernels/trellis_gemv_floor.metal (bridge test enforces).
const LCG_A: u32 = 89_226_354;
const LCG_B: u32 = 64_248_484;
const MASK: u32 = 0x8FFF_8FFF;
const FIXED: u32 = 0x3B60_3B60 & !MASK;

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn halves(bits: u32) -> (f32, f32) {
    (
        half::f16::from_bits((bits & 0xFFFF) as u16).to_f32(),
        half::f16::from_bits((bits >> 16) as u16).to_f32(),
    )
}

/// Decode-value tables for one code over all 65,536 states.
pub struct TrellisCode {
    /// weights per trellis step (1 or 2); step consumes 3*v bits
    pub v: usize,
    vx: Vec<f32>,
    vy: Vec<f32>,
}

impl TrellisCode {
    /// V=1 per-weight code shipped as `3inst`/`3inst_g256`:
    /// half(lo) + half(hi) of the masked LCG hash, f16-rounded.
    pub fn v1_maskor() -> Self {
        let mut vx = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let x = st.wrapping_mul(LCG_A).wrapping_add(LCG_B);
            let hb = (x & MASK) | FIXED;
            let (hx, hy) = halves(hb);
            vx[st as usize] = f16r(hx + hy);
        }
        TrellisCode {
            v: 1,
            vx,
            vy: Vec::new(),
        }
    }

    /// V=2 split code shipped as `3inst_v2`/`3inst_v2_g256`: the two
    /// halves of the hash ARE the two weight values.
    pub fn v2_split_maskor() -> Self {
        let mut vx = vec![0f32; NSTATES];
        let mut vy = vec![0f32; NSTATES];
        for st in 0..NSTATES as u32 {
            let x = st.wrapping_mul(LCG_A).wrapping_add(LCG_B);
            let hb = (x & MASK) | FIXED;
            let (hx, hy) = halves(hb);
            vx[st as usize] = hx;
            vy[st as usize] = hy;
        }
        TrellisCode { v: 2, vx, vy }
    }

    /// Arbitrary code from explicit per-state value tables (T8 code
    /// search; v in {1,2}).
    pub fn from_tables(v: usize, vx: Vec<f32>, vy: Vec<f32>) -> Self {
        assert!(v == 1 || v == 2);
        assert_eq!(vx.len(), NSTATES);
        if v == 2 {
            assert_eq!(vy.len(), NSTATES);
        }
        TrellisCode { v, vx, vy }
    }

    /// RMS of the code's value distribution (for initial scale fits).
    pub fn value_rms(&self) -> f64 {
        let it = self.vx.iter().chain(self.vy.iter());
        let n = if self.v == 2 { 2 * NSTATES } else { NSTATES };
        (it.map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64).sqrt()
    }
}

/// One Viterbi pass over a 256-weight group (right-shift trellis:
/// new = (old >> k) | (fresh << (L-k))). `pin_lead` constrains the
/// initial state's low (L-k) bits (tail-biting pass 2).
fn viterbi_pass(
    x: &[f32],
    scales: &[f32],
    sub: usize,
    code: &TrellisCode,
    pin_lead: Option<u32>,
) -> (Vec<u32>, f32) {
    let k = 3 * code.v;
    let steps = GROUP_W / code.v;
    let n_pred = 1usize << k;
    let lead_mask = ((1u32 << (L - k)) - 1) as usize;

    let cost = |step: usize, st: usize| -> f32 {
        if code.v == 1 {
            let d = x[step] - scales[step / sub] * self_get(&code.vx, st);
            d * d
        } else {
            let sc = scales[(2 * step) / sub];
            let d0 = x[2 * step] - sc * self_get(&code.vx, st);
            let d1 = x[2 * step + 1] - sc * self_get(&code.vy, st);
            d0 * d0 + d1 * d1
        }
    };

    let mut dp_old = vec![f32::INFINITY; NSTATES];
    let mut dp_new = vec![f32::INFINITY; NSTATES];
    let mut bp = vec![0u8; steps * NSTATES];

    for (st, slot) in dp_old.iter_mut().enumerate() {
        let ok = match pin_lead {
            Some(l) => (st & lead_mask) as u32 == l,
            None => true,
        };
        *slot = if ok { cost(0, st) } else { f32::INFINITY };
    }

    for step in 1..steps {
        let bprow = &mut bp[step * NSTATES..(step + 1) * NSTATES];
        for ns in 0..NSTATES {
            let base = (ns & lead_mask) << k;
            let mut best = f32::INFINITY;
            let mut best_t = 0u8;
            for t in 0..n_pred {
                let c = dp_old[base | t];
                if c < best {
                    best = c;
                    best_t = t as u8;
                }
            }
            dp_new[ns] = best + cost(step, ns);
            bprow[ns] = best_t;
        }
        std::mem::swap(&mut dp_old, &mut dp_new);
    }

    let (mut st, total) = dp_old
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, c)| (i, *c))
        .unwrap();
    let mut path = vec![0u32; steps];
    for step in (1..steps).rev() {
        path[step] = st as u32;
        let t = bp[step * NSTATES + st] as usize;
        st = ((st & lead_mask) << k) | t;
    }
    path[0] = st as u32;
    (path, total)
}

#[inline]
fn self_get(v: &[f32], i: usize) -> f32 {
    // (helper keeps the closure monomorphic and index-checked once)
    v[i]
}

fn pack_path(path: &[u32], k: usize) -> [u32; GROUP_WORDS] {
    let mut words = [0u32; GROUP_WORDS];
    let mut bitpos = 0usize;
    for &st in path {
        let fresh = (st >> (L - k)) & ((1u32 << k) - 1);
        for i in 0..k {
            if (fresh >> i) & 1 == 1 {
                words[(bitpos + i) >> 5] |= 1 << ((bitpos + i) & 31);
            }
        }
        bitpos += k;
    }
    words
}

fn ring_state(words: &[u32; GROUP_WORDS], end: usize) -> u32 {
    let mut st = 0u32;
    let start = (end + RING_BITS - L) % RING_BITS;
    for i in 0..L {
        let b = (start + i) % RING_BITS;
        st |= ((words[b >> 5] >> (b & 31)) & 1) << i;
    }
    st
}

/// Decode a packed group's 256 (unscaled) values through the ring
/// windows — bit-honest twin of the group-ring kernels.
pub fn decode_group(words: &[u32; GROUP_WORDS], code: &TrellisCode) -> Vec<f32> {
    let mut out = vec![0f32; GROUP_W];
    if code.v == 1 {
        for (j, o) in out.iter_mut().enumerate() {
            let st = ring_state(words, (3 * (j + 1)) % RING_BITS) as usize;
            *o = code.vx[st];
        }
    } else {
        for t in 0..GROUP_W / 2 {
            let st = ring_state(words, (6 * (t + 1)) % RING_BITS) as usize;
            out[2 * t] = code.vx[st];
            out[2 * t + 1] = code.vy[st];
        }
    }
    out
}

/// Encode one 256-weight group (two-pass tail-biting, keep the better
/// TRUE ring-decoded stream) at a fixed scale.
fn encode_group_at(
    x: &[f32],
    scales: &[f32],
    sub: usize,
    code: &TrellisCode,
) -> ([u32; GROUP_WORDS], f64) {
    let k = 3 * code.v;
    let (p1, _) = viterbi_pass(x, scales, sub, code, None);
    let lead = p1[p1.len() - 1] >> k;
    let (p2, _) = viterbi_pass(x, scales, sub, code, Some(lead));
    let w1 = pack_path(&p1, k);
    let w2 = pack_path(&p2, k);
    let err = |w: &[u32; GROUP_WORDS]| -> f64 {
        let vals = decode_group(w, code);
        x.iter()
            .zip(&vals)
            .enumerate()
            .map(|(i, (xx, vv))| {
                let d = *xx as f64 - scales[i / sub] as f64 * *vv as f64;
                d * d
            })
            .sum()
    };
    let (e1, e2) = (err(&w1), err(&w2));
    if e2 <= e1 { (w2, e2) } else { (w1, e1) }
}

/// Result of encoding one group with a fitted fp16 scale.
pub struct EncodedGroup {
    pub words: [u32; GROUP_WORDS],
    pub scale_f16: u16,
    pub sq_err: f64,
}

/// Encode a 256-weight group: initial rms-matched scale, one
/// least-squares refit, keep the better of the two encodes.
pub fn encode_group(x: &[f32], code: &TrellisCode) -> EncodedGroup {
    assert_eq!(x.len(), GROUP_W);
    let rms_x = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / GROUP_W as f64).sqrt();
    let s0 = f16r((rms_x / code.value_rms().max(1e-12)) as f32);
    let sv0 = [s0; 1];
    let (w0, e0) = encode_group_at(x, &sv0, GROUP_W, code);
    // LS refit against the decoded values, then re-encode.
    let v0 = decode_group(&w0, code);
    let num: f64 = x
        .iter()
        .zip(&v0)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let den: f64 = v0.iter().map(|b| (*b as f64) * (*b as f64)).sum();
    let s1 = f16r(if den > 0.0 { (num / den) as f32 } else { s0 });
    let sv1 = [s1; 1];
    let (w1, e1) = encode_group_at(x, &sv1, GROUP_W, code);
    if e1 <= e0 {
        EncodedGroup {
            words: w1,
            scale_f16: half::f16::from_f32(s1).to_bits(),
            sq_err: e1,
        }
    } else {
        EncodedGroup {
            words: w0,
            scale_f16: half::f16::from_f32(s0).to_bits(),
            sq_err: e0,
        }
    }
}

/// Sub-scale variant (T8/B8): `n_sub` fp16 scales per 256-weight group
/// (chunk = 256/n_sub). Same encode -> LS-refit-per-chunk -> re-encode
/// discipline. Returns the group squared error only.
pub fn encode_group_sub(x: &[f32], code: &TrellisCode, n_sub: usize) -> f64 {
    encode_group_sub_full(x, code, n_sub).sq_err
}

/// Full result of a sub-scaled group encode (T9 LDLQ needs the actual
/// reconstruction, not just the error).
pub struct EncodedGroupSub {
    pub words: [u32; GROUP_WORDS],
    pub scales_f16: Vec<u16>,
    pub sq_err: f64,
}

impl EncodedGroupSub {
    /// Scaled reconstruction of the 256 weights.
    pub fn reconstruct(&self, code: &TrellisCode) -> Vec<f32> {
        let vals = decode_group(&self.words, code);
        let sub = GROUP_W / self.scales_f16.len();
        vals.iter()
            .enumerate()
            .map(|(i, v)| half::f16::from_bits(self.scales_f16[i / sub]).to_f32() * v)
            .collect()
    }
}

/// Sub-scale encode returning the packed stream + fitted scales.
/// Identical arithmetic to the T8-era `encode_group_sub` (the two-encode
/// min is preserved by keeping whichever encode won).
pub fn encode_group_sub_full(x: &[f32], code: &TrellisCode, n_sub: usize) -> EncodedGroupSub {
    assert_eq!(x.len(), GROUP_W);
    assert!(GROUP_W.is_multiple_of(n_sub));
    let sub = GROUP_W / n_sub;
    let crms = code.value_rms().max(1e-12);
    let scales0: Vec<f32> = x
        .chunks(sub)
        .map(|c| {
            let rms =
                (c.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / sub as f64).sqrt();
            f16r((rms / crms) as f32)
        })
        .collect();
    let (w0, e0) = encode_group_at(x, &scales0, sub, code);
    let v0 = decode_group(&w0, code);
    let mut scales1 = scales0.clone();
    for (ci, chunk) in x.chunks(sub).enumerate() {
        let vs = &v0[ci * sub..(ci + 1) * sub];
        let num: f64 = chunk
            .iter()
            .zip(vs)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum();
        let den: f64 = vs.iter().map(|b| (*b as f64) * (*b as f64)).sum();
        if den > 0.0 {
            scales1[ci] = f16r((num / den) as f32);
        }
    }
    let (w1, e1) = encode_group_at(x, &scales1, sub, code);
    if e1 <= e0 {
        EncodedGroupSub {
            words: w1,
            scales_f16: scales1
                .iter()
                .map(|s| half::f16::from_f32(*s).to_bits())
                .collect(),
            sq_err: e1,
        }
    } else {
        EncodedGroupSub {
            words: w0,
            scales_f16: scales0
                .iter()
                .map(|s| half::f16::from_f32(*s).to_bits())
                .collect(),
            sq_err: e0,
        }
    }
}

/// In-place blockwise fast Walsh-Hadamard transform along 128-element
/// blocks (orthonormal: 1/sqrt(128) each application; self-inverse).
pub fn fwht128_blocks(x: &mut [f32]) {
    assert_eq!(x.len() % 128, 0);
    let norm = 1.0 / (128f32).sqrt();
    for blk in x.chunks_mut(128) {
        let mut h = 1;
        while h < 128 {
            let mut i = 0;
            while i < 128 {
                for j in i..i + h {
                    let a = blk[j];
                    let b = blk[j + h];
                    blk[j] = a + b;
                    blk[j + h] = a - b;
                }
                i += h << 1;
            }
            h <<= 1;
        }
        for v in blk.iter_mut() {
            *v *= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bridge: an offline-encoded group must decode identically through
    /// `metal::trellis3_cpu_reference` (which is itself GPU-validated).
    #[test]
    fn offline_encode_bridges_to_kernel_reference() {
        let mut s = 0x00B2_1D6E_u64;
        let mut next = move || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for code in [TrellisCode::v1_maskor(), TrellisCode::v2_split_maskor()] {
            let x: Vec<f32> = (0..GROUP_W)
                .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
                .collect();
            let enc = encode_group(&x, &code);
            let vals = decode_group(&enc.words, &code);
            let scale = half::f16::from_bits(enc.scale_f16).to_f32();

            // Feed the packed group through the metal-side CPU reference
            // as a 1-row [256, 1] GEMV with x = e_j to read back values.
            let mut weight = Vec::with_capacity(96);
            for w in enc.words {
                weight.extend_from_slice(&w.to_le_bytes());
            }
            let syn = crate::metal::Trellis3Synthetic {
                weight,
                scales_f16: vec![enc.scale_f16],
                lut_f16: vec![0u16; 1024],
            };
            let variant = if code.v == 1 {
                crate::metal::Trellis3Variant::ThreeInstG
            } else {
                crate::metal::Trellis3Variant::ThreeInstV2G
            };
            // One dense probe vector: y = sum_j w_j * x_j must match.
            let xv: Vec<f32> = (0..GROUP_W).map(|i| ((i as f32) * 0.017).sin()).collect();
            let y = crate::metal::trellis3_cpu_reference(variant, &syn, &xv, GROUP_W, 1);
            let mut expect = 0f64;
            for j in 0..GROUP_W {
                // reference uses f16 activation + f16 fma chain; tolerance
                // below absorbs that (values themselves are exact f16).
                expect += (scale as f64) * (vals[j] as f64) * (xv[j] as f64);
            }
            let rel = ((y[0] as f64 - expect) / expect.abs().max(1e-9)).abs();
            assert!(
                rel < 5e-3,
                "bridge mismatch v={}: y={} expect={} rel={rel}",
                code.v,
                y[0],
                expect
            );
            // And the decoded VALUES must be exactly representable f16.
            for v in &vals {
                assert_eq!(*v, f16r(*v));
            }
        }
    }

    #[test]
    fn fwht_is_orthonormal_involution() {
        let mut x: Vec<f32> = (0..256)
            .map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5)
            .collect();
        let orig = x.clone();
        let n0: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        fwht128_blocks(&mut x);
        let n1: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(((n1 - n0) / n0).abs() < 1e-6);
        fwht128_blocks(&mut x);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

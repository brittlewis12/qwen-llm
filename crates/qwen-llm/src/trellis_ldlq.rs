//! T9 offline calibration machinery: two-sided block-RHT, f64 Hessian
//! accumulation, 256-block LDL, BlockLDLQ driver around the T=256 span
//! encoder, and the held-out r_H scorer.
//!
//! Preregistration: docs/bench/2026-07-20-trellis3-t9-ldlq-pilot/.
//! Tooling-only (no GPU, no production path). Geometry: T_x=1, T_y=256
//! (LDL feedback block = the Viterbi span; declared 16x coarser than
//! QTIP's g=16 — see the prereg's budgeted priors).

use crate::trellis_offline::{TrellisCode, encode_group_sub_full, fwht128_blocks};

pub const SPAN: usize = 256;

// ---------------------------------------------------------------------
// Two-sided block-RHT (EXL3-style, pure orthogonal): W~ = U W V^T with
// U = S_u * Hbd(128) on the output dim, V = S_v * Hbd(128) on the input
// dim. Signs are Rademacher from a seeded xorshift. Self-inverse up to
// sign re-application (H orthonormal involution per block).
// ---------------------------------------------------------------------

fn sign_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s & 1 == 1 { 1.0f32 } else { -1.0f32 }
        })
        .collect()
}

/// Two-sided rotation of row-major `w` [n_out x n_in], in place.
/// `forward=true` applies W~ = U W V^T; `false` applies the exact
/// inverse (signs after/before Hadamard swap accordingly).
pub fn rht_two_sided(w: &mut [f32], n_out: usize, n_in: usize, seed: u64, forward: bool) {
    assert_eq!(w.len(), n_out * n_in);
    assert!(n_in % 128 == 0 && n_out % 128 == 0);
    let s_in = sign_vec(n_in, seed ^ 0x1157);
    let s_out = sign_vec(n_out, seed ^ 0x2263);
    // Input side (rows): x -> Hbd (S x). Inverse: x -> S (Hbd x).
    for row in w.chunks_mut(n_in) {
        if forward {
            for (v, s) in row.iter_mut().zip(&s_in) {
                *v *= s;
            }
            fwht128_blocks(row);
        } else {
            fwht128_blocks(row);
            for (v, s) in row.iter_mut().zip(&s_in) {
                *v *= s;
            }
        }
    }
    // Output side (columns): same op along columns via transpose.
    let mut col = vec![0f32; n_out];
    for j in 0..n_in {
        for (i, c) in col.iter_mut().enumerate() {
            *c = w[i * n_in + j];
        }
        if forward {
            for (v, s) in col.iter_mut().zip(&s_out) {
                *v *= s;
            }
            fwht128_blocks(&mut col);
        } else {
            fwht128_blocks(&mut col);
            for (v, s) in col.iter_mut().zip(&s_out) {
                *v *= s;
            }
        }
        for (i, c) in col.iter().enumerate() {
            w[i * n_in + j] = *c;
        }
    }
}

/// Input-side-only rotation of an activation vector x -> Hbd (S x),
/// matching the input side of [`rht_two_sided`] (needed to score r_H in
/// the rotated domain: E~ x~ = U E x, and ||U z|| = ||z||).
pub fn rht_rotate_activation(x: &mut [f32], seed: u64) {
    assert!(x.len() % 128 == 0);
    let s_in = sign_vec(x.len(), seed ^ 0x1157);
    for (v, s) in x.iter_mut().zip(&s_in) {
        *v *= s;
    }
    fwht128_blocks(x);
}

// ---------------------------------------------------------------------
// f64 Hessian accumulation + damping.
// ---------------------------------------------------------------------

/// Streaming Gram accumulator: H += x x^T over f32 activation rows,
/// accumulated in f64. Row-major upper storage is full dense d x d for
/// simplicity (Stage A d <= 3584 -> <= 103 MB f64).
pub struct GramAccumulator {
    pub d: usize,
    pub n_samples: u64,
    pub h: Vec<f64>,
}

impl GramAccumulator {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            n_samples: 0,
            h: vec![0f64; d * d],
        }
    }

    pub fn add(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.d);
        let d = self.d;
        for i in 0..d {
            let xi = x[i] as f64;
            if xi == 0.0 {
                continue;
            }
            let row = &mut self.h[i * d..(i + 1) * d];
            for (j, v) in row.iter_mut().enumerate() {
                *v += xi * x[j] as f64;
            }
        }
        self.n_samples += 1;
    }

    /// Batched rank-k update H += sum_t x_t x_t^T, parallelized over
    /// output rows with scoped threads (dependency-free). `xs` is
    /// row-major [n_rows x d].
    pub fn add_batch(&mut self, xs: &[f32], n_threads: usize) {
        assert_eq!(xs.len() % self.d, 0);
        let d = self.d;
        let n_rows = xs.len() / d;
        let nt = n_threads.max(1).min(d);
        let rows_per = d.div_ceil(nt);
        let h_chunks: Vec<&mut [f64]> = self.h.chunks_mut(rows_per * d).collect();
        std::thread::scope(|scope| {
            for (ci, chunk) in h_chunks.into_iter().enumerate() {
                let i0 = ci * rows_per;
                scope.spawn(move || {
                    let n_i = chunk.len() / d;
                    for t in 0..n_rows {
                        let x = &xs[t * d..(t + 1) * d];
                        for li in 0..n_i {
                            let xi = x[i0 + li] as f64;
                            if xi == 0.0 {
                                continue;
                            }
                            let row = &mut chunk[li * d..(li + 1) * d];
                            for (j, v) in row.iter_mut().enumerate() {
                                *v += xi * x[j] as f64;
                            }
                        }
                    }
                });
            }
        });
        self.n_samples += n_rows as u64;
    }

    /// Symmetrize (guards accumulated asymmetry) and apply the frozen
    /// damping H + c*mean(diag)*I. Returns the damped matrix.
    pub fn damped(&self, c: f64) -> Vec<f64> {
        let d = self.d;
        let mut h = self.h.clone();
        for i in 0..d {
            for j in (i + 1)..d {
                let m = 0.5 * (h[i * d + j] + h[j * d + i]);
                h[i * d + j] = m;
                h[j * d + i] = m;
            }
        }
        let mean_diag = (0..d).map(|i| h[i * d + i]).sum::<f64>() / d as f64;
        let lambda = c * mean_diag;
        for i in 0..d {
            h[i * d + i] += lambda;
        }
        h
    }
}

// ---------------------------------------------------------------------
// Dense f64 Cholesky (lower) + 256-block unit-lower extraction.
// H = C C^T; block-LDL: Lbar_ij = C_ij * C_jj^{-1} (right triangular
// solve), Lbar block-diagonal = I. A = Lbar - I is what BlockLDLQ needs.
// ---------------------------------------------------------------------

/// In-place dense lower Cholesky of a row-major symmetric positive
/// definite matrix. Returns Err(pivot_index) on a non-positive pivot.
pub fn cholesky_lower(h: &mut [f64], d: usize) -> Result<(), usize> {
    for j in 0..d {
        let mut s = h[j * d + j];
        for k in 0..j {
            s -= h[j * d + k] * h[j * d + k];
        }
        if s <= 0.0 {
            return Err(j);
        }
        let piv = s.sqrt();
        h[j * d + j] = piv;
        let inv = 1.0 / piv;
        for i in (j + 1)..d {
            let mut v = h[i * d + j];
            for k in 0..j {
                v -= h[i * d + k] * h[j * d + k];
            }
            h[i * d + j] = v * inv;
        }
    }
    // Zero the strict upper triangle for cleanliness.
    for i in 0..d {
        for j in (i + 1)..d {
            h[i * d + j] = 0.0;
        }
    }
    Ok(())
}

/// From the dense lower Cholesky factor C, build A = Lbar - I where
/// Lbar_ij = C_ij C_jj^{-1} per 256-block column (unit block diagonal).
/// Stored dense row-major d x d (strictly-below-block-diagonal band is
/// the only nonzero region).
pub fn block_unit_lower_a(c: &[f64], d: usize) -> Vec<f64> {
    assert_eq!(d % SPAN, 0);
    let nb = d / SPAN;
    let mut a = vec![0f64; d * d];
    for jb in 0..nb {
        let j0 = jb * SPAN;
        // We need X = C_block * inv(C_jj) where C_jj is LOWER-triangular
        // 256x256, i.e. solve X C_jj = C_block. Since C_jj[k,q] != 0 for
        // k >= q, fixing a row i gives, per column q:
        //   X[i,q] C_jj[q,q] = C_block[i,q] - sum_{k>q} X[i,k] C_jj[k,q]
        // -> backward substitution over columns (last to first).
        for i in (j0 + SPAN)..d {
            let mut xrow = [0f64; SPAN];
            for q in (0..SPAN).rev() {
                let mut v = c[i * d + j0 + q];
                for k in (q + 1)..SPAN {
                    v -= xrow[k] * c[(j0 + k) * d + j0 + q];
                }
                xrow[q] = v / c[(j0 + q) * d + j0 + q];
            }
            for q in 0..SPAN {
                a[i * d + j0 + q] = xrow[q];
            }
        }
    }
    a
}

// ---------------------------------------------------------------------
// BlockLDLQ driver.
// ---------------------------------------------------------------------

pub struct LdlqRowResult {
    /// Scaled reconstruction of the full row (length d).
    pub w_hat: Vec<f32>,
    /// Sum over spans of the span-encoder squared error vs the ADJUSTED
    /// targets (diagnostic only — NOT the plain weight error).
    pub adjusted_sq_err: f64,
    /// Per-span diagnostic traces, reverse-span-index order (prereg R5
    /// instrumentation): (target_rms, target_kurtosis_excess).
    pub span_traces: Vec<(f64, f64)>,
}

/// Core BlockLDLQ driver, generic over the span quantizer (the mock
/// path lets tests validate the feedback algebra independently of the
/// trellis encoder). Reverse-order 256-block feedback:
/// Z_j = W_j + (W_later - What_later) A_later,j; What_j = quant(Z_j).
/// `a` None = plain per-span encode (LDLQ-off arms).
pub fn ldlq_quantize_row_with<Q>(
    w_row: &[f32],
    a: Option<&[f64]>,
    d: usize,
    quant_span: Q,
) -> LdlqRowResult
where
    Q: Fn(&[f32]) -> (Vec<f32>, f64),
{
    assert_eq!(w_row.len(), d);
    assert_eq!(d % SPAN, 0);
    let nb = d / SPAN;
    let mut w_hat = vec![0f32; d];
    let mut err = vec![0f64; d]; // W - What, filled right-to-left
    let mut adjusted_sq_err = 0.0f64;
    let mut span_traces = Vec::with_capacity(nb);
    for jb in (0..nb).rev() {
        let j0 = jb * SPAN;
        let mut z = [0f32; SPAN];
        for q in 0..SPAN {
            let mut v = w_row[j0 + q] as f64;
            if let Some(a) = a {
                // += sum_{i > block end} err[i] * A[i, j0+q]
                for (i, e) in err.iter().enumerate().skip(j0 + SPAN) {
                    if *e != 0.0 {
                        v += *e * a[i * d + j0 + q];
                    }
                }
            }
            z[q] = v as f32;
        }
        // R5 trace: adjusted-target moments.
        let mean = z.iter().map(|v| *v as f64).sum::<f64>() / SPAN as f64;
        let var = z.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / SPAN as f64;
        let m4 = z.iter().map(|v| (*v as f64 - mean).powi(4)).sum::<f64>() / SPAN as f64;
        let kurt = if var > 0.0 {
            m4 / (var * var) - 3.0
        } else {
            0.0
        };
        span_traces.push((var.sqrt(), kurt));

        let (rec, sq) = quant_span(&z);
        adjusted_sq_err += sq;
        for q in 0..SPAN {
            w_hat[j0 + q] = rec[q];
            err[j0 + q] = w_row[j0 + q] as f64 - rec[q] as f64;
        }
    }
    LdlqRowResult {
        w_hat,
        adjusted_sq_err,
        span_traces,
    }
}

/// Trellis-encoder BlockLDLQ (the production T9 path).
pub fn ldlq_quantize_row(
    w_row: &[f32],
    a: Option<&[f64]>,
    d: usize,
    code: &TrellisCode,
    n_sub: usize,
) -> LdlqRowResult {
    ldlq_quantize_row_with(w_row, a, d, |z| {
        let enc = encode_group_sub_full(z, code, n_sub);
        (enc.reconstruct(code), enc.sq_err)
    })
}

// ---------------------------------------------------------------------
// Scoring.
// ---------------------------------------------------------------------

/// Plain relative Frobenius ||W - What||_F / ||W||_F.
pub fn rel_frobenius(w: &[f32], w_hat: &[f32]) -> f64 {
    assert_eq!(w.len(), w_hat.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in w.iter().zip(w_hat) {
        let d = *a as f64 - *b as f64;
        num += d * d;
        den += (*a as f64) * (*a as f64);
    }
    (num / den.max(1e-300)).sqrt()
}

/// Streaming r_H accumulator: r_H = ||E X^T||_F / ||W X^T||_F over
/// held-out activation rows x (in the SAME domain as E and W — rotate x
/// with [`rht_rotate_activation`] for rotated-domain arms).
pub struct RhScorer {
    num: f64,
    den: f64,
}

impl RhScorer {
    pub fn new() -> Self {
        Self { num: 0.0, den: 0.0 }
    }

    /// Add one activation row. `e` = W - What (row-major n_out x d),
    /// `w` = reference weights. O(n_out * d) per call.
    pub fn add(&mut self, w: &[f32], w_hat: &[f32], n_out: usize, d: usize, x: &[f32]) {
        assert_eq!(x.len(), d);
        for r in 0..n_out {
            let wr = &w[r * d..(r + 1) * d];
            let hr = &w_hat[r * d..(r + 1) * d];
            let mut we = 0f64;
            let mut ww = 0f64;
            for j in 0..d {
                let xj = x[j] as f64;
                we += (wr[j] as f64 - hr[j] as f64) * xj;
                ww += wr[j] as f64 * xj;
            }
            self.num += we * we;
            self.den += ww * ww;
        }
    }

    pub fn value(&self) -> f64 {
        (self.num / self.den.max(1e-300)).sqrt()
    }
}

impl Default for RhScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift_f32(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (((*s >> 40) as f32) / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn rht_roundtrip_and_frobenius_invariance() {
        let (n_out, n_in) = (256, 512);
        let mut s = 0xABCDu64;
        let w: Vec<f32> = (0..n_out * n_in).map(|_| xorshift_f32(&mut s)).collect();
        let mut wr = w.clone();
        rht_two_sided(&mut wr, n_out, n_in, 7, true);
        let n0: f64 = w.iter().map(|v| (*v as f64).powi(2)).sum();
        let n1: f64 = wr.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(
            ((n1 - n0) / n0).abs() < 1e-5,
            "two-sided RHT must preserve Frobenius"
        );
        rht_two_sided(&mut wr, n_out, n_in, 7, false);
        for (a, b) in w.iter().zip(&wr) {
            assert!((a - b).abs() < 1e-4, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn cholesky_and_block_a_reconstruct() {
        // Small SPD matrix: H = B B^T + d*I, d = 512 (2 blocks).
        let d = 512usize;
        let mut s = 0x1234u64;
        let b: Vec<f64> = (0..d * d).map(|_| xorshift_f32(&mut s) as f64).collect();
        let mut h = vec![0f64; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut v = 0.0;
                for k in 0..d {
                    v += b[i * d + k] * b[j * d + k];
                }
                h[i * d + j] = v;
                h[j * d + i] = v;
            }
            h[i * d + i] += d as f64;
        }
        let h_orig = h.clone();
        cholesky_lower(&mut h, d).expect("SPD");
        // C C^T == H.
        let mut max_rel = 0f64;
        for i in 0..d {
            for j in 0..=i {
                let mut v = 0.0;
                for k in 0..=j.min(i) {
                    v += h[i * d + k] * h[j * d + k];
                }
                let r = (v - h_orig[i * d + j]).abs() / h_orig[i * d + j].abs().max(1.0);
                max_rel = max_rel.max(r);
            }
        }
        assert!(max_rel < 1e-9, "cholesky reconstruction rel err {max_rel}");
        // A = Lbar - I: check Lbar * blockdiag(C) == C on the lower band,
        // i.e. for the (1,0) block: A_10 * C_00 == C_10.
        let a = block_unit_lower_a(&h, d);
        let mut max_rel2 = 0f64;
        for i in SPAN..d {
            for q in 0..SPAN {
                let mut v = 0.0;
                for k in 0..SPAN {
                    v += a[i * d + k] * h[k * d + q]; // C_00 rows are h[k*d+q], k<SPAN
                }
                let r = (v - h[i * d + q]).abs() / h[i * d + q].abs().max(1e-9);
                max_rel2 = max_rel2.max(r);
            }
        }
        assert!(max_rel2 < 1e-8, "block A validation rel err {max_rel2}");
    }

    /// Generate activations with CROSS-BLOCK (lag-256) correlation —
    /// structure the g=256 feedback can actually exploit. (Adjacent-lag
    /// correlation lives INSIDE a span, invisible to block feedback:
    /// the original AR(1) version of this test proved only that.)
    fn lag256_activation(s: &mut u64, d: usize, rho: f32) -> Vec<f32> {
        let mut z = vec![0f32; d];
        for v in z.iter_mut() {
            *v = xorshift_f32(s);
        }
        let mut x = z.clone();
        for j in SPAN..d {
            x[j] = z[j] + rho * z[j - SPAN];
        }
        x
    }

    /// Driver-algebra validation with a MOCK quantizer (elementwise
    /// rounding to a coarse grid): on lag-256-correlated activations,
    /// block feedback must materially improve held-out r_H. Isolates
    /// the LDLQ algebra from trellis-encoder interactions.
    #[test]
    fn ldlq_algebra_beats_plain_with_mock_quantizer() {
        let d = 768usize; // 3 blocks
        let n_out = 6usize;
        let mut s = 0x77AAu64;
        let w: Vec<f32> = (0..n_out * d).map(|_| xorshift_f32(&mut s)).collect();
        let n_cal = 4096usize;
        let mut acc = GramAccumulator::new(d);
        let mut xs_test: Vec<Vec<f32>> = Vec::new();
        for t in 0..(n_cal + 512) {
            let x = lag256_activation(&mut s, d, 0.9);
            if t < n_cal {
                acc.add(&x);
            } else {
                xs_test.push(x);
            }
        }
        let mut hd = acc.damped(0.01);
        cholesky_lower(&mut hd, d).expect("SPD");
        let a = block_unit_lower_a(&hd, d);
        let grid = 0.35f32; // coarse rounding grid ~3-bit-ish
        let mock = |z: &[f32]| -> (Vec<f32>, f64) {
            let rec: Vec<f32> = z.iter().map(|v| (v / grid).round() * grid).collect();
            let sq = z
                .iter()
                .zip(&rec)
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            (rec, sq)
        };
        let mut plain_score = RhScorer::new();
        let mut ldlq_score = RhScorer::new();
        let mut w_hat_plain = vec![0f32; n_out * d];
        let mut w_hat_ldlq = vec![0f32; n_out * d];
        for r in 0..n_out {
            let row = &w[r * d..(r + 1) * d];
            let p = ldlq_quantize_row_with(row, None, d, mock);
            let l = ldlq_quantize_row_with(row, Some(&a), d, mock);
            w_hat_plain[r * d..(r + 1) * d].copy_from_slice(&p.w_hat);
            w_hat_ldlq[r * d..(r + 1) * d].copy_from_slice(&l.w_hat);
        }
        for x in &xs_test {
            plain_score.add(&w, &w_hat_plain, n_out, d, x);
            ldlq_score.add(&w, &w_hat_ldlq, n_out, d, x);
        }
        let (rp, rl) = (plain_score.value(), ldlq_score.value());
        assert!(
            rl < rp * 0.95,
            "mock-quantizer LDLQ must improve held-out r_H by >=5% on \
             lag-256-correlated H: plain={rp:.5} ldlq={rl:.5}"
        );
    }

    /// Same setup through the real trellis encoder. Weaker bar: the
    /// trellis span solve interacts with adjusted targets (prereg R5
    /// risk), so this asserts a >=2% held-out gain.
    #[test]
    fn ldlq_beats_plain_on_weighted_error() {
        let d = 512usize;
        let n_out = 8usize;
        let mut s = 0x66BBu64;
        let w: Vec<f32> = (0..n_out * d).map(|_| xorshift_f32(&mut s)).collect();
        let n_cal = 4096usize;
        let mut acc = GramAccumulator::new(d);
        let mut xs_test: Vec<Vec<f32>> = Vec::new();
        for t in 0..(n_cal + 512) {
            let x = lag256_activation(&mut s, d, 0.9);
            if t < n_cal {
                acc.add(&x);
            } else {
                xs_test.push(x);
            }
        }
        let mut hd = acc.damped(0.01);
        cholesky_lower(&mut hd, d).expect("SPD");
        let a = block_unit_lower_a(&hd, d);
        let code = TrellisCode::v1_maskor();

        let mut plain_score = RhScorer::new();
        let mut ldlq_score = RhScorer::new();
        let mut w_hat_plain = vec![0f32; n_out * d];
        let mut w_hat_ldlq = vec![0f32; n_out * d];
        for r in 0..n_out {
            let row = &w[r * d..(r + 1) * d];
            let p = ldlq_quantize_row(row, None, d, &code, 4);
            let l = ldlq_quantize_row(row, Some(&a), d, &code, 4);
            w_hat_plain[r * d..(r + 1) * d].copy_from_slice(&p.w_hat);
            w_hat_ldlq[r * d..(r + 1) * d].copy_from_slice(&l.w_hat);
        }
        for x in &xs_test {
            plain_score.add(&w, &w_hat_plain, n_out, d, x);
            ldlq_score.add(&w, &w_hat_ldlq, n_out, d, x);
        }
        let (rp, rl) = (plain_score.value(), ldlq_score.value());
        assert!(
            rl < rp * 0.98,
            "trellis LDLQ must improve held-out r_H by >=2% on \
             lag-256-correlated H: plain={rp:.5} ldlq={rl:.5}"
        );
    }
}

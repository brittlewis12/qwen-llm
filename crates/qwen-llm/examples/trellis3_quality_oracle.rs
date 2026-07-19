//! T5: trellis3 Gaussian source-coding quality oracle (CPU-only).
//!
//! Preregistration + gates: docs/bench/2026-07-19-trellis3-quality-oracle/.
//! Prices the throughput winner's two unpublished deviations (V=2 split
//! computed code, T=32 span tail-biting) against canonical QTIP codes, the
//! optimal-scalar (Lloyd-Max) floor, and llama.cpp's 3-bit formats on an
//! i.i.d. standard-normal source.
//!
//! Run: cargo run --release -p qwen-llm --example trellis3_quality_oracle
//!
//! Exact Viterbi over the L=16 bitshift trellis (65,536 states), two-pass
//! tail-biting. MSE is always measured by ring-decoding the PACKED
//! bitstream with the same window math as the shipped kernels — encoder
//! approximations can cost optimality but can never fake fidelity.

use std::time::Instant;

const L: usize = 16;
const NSTATES: usize = 1 << L;
const GROUP_W: usize = 256;
const N_GROUPS: usize = 64;
const SEED: u64 = 0x7E11_15;

const LCG_A: u32 = 89_226_354;
const LCG_B: u32 = 64_248_484;
const MASK: u32 = 0x8FFF_8FFF;
const FIXED: u32 = 0x3B60_3B60 & !MASK;
const XOR_MAGIC: u32 = 0x3B60_3B60;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gaussian_sample(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = ((splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        let u2 = (splitmix64(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        if out.len() < n {
            out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
        }
    }
    out
}

fn f16r(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// Decode-value tables over all 65,536 states.
#[derive(Clone)]
struct Code {
    /// bits consumed per trellis step (3 for V=1, 6 for V=2)
    k: usize,
    /// weights emitted per step (1 or 2)
    v: usize,
    /// value of weight 0 at each state
    vx: Vec<f32>,
    /// value of weight 1 at each state (V=2 only)
    vy: Vec<f32>,
    _name: &'static str,
}

fn halves(bits: u32) -> (f32, f32) {
    (
        half::f16::from_bits((bits & 0xFFFF) as u16).to_f32(),
        half::f16::from_bits((bits >> 16) as u16).to_f32(),
    )
}

fn build_code(kind: &'static str) -> Code {
    let mut vx = vec![0f32; NSTATES];
    let mut vy = vec![0f32; NSTATES];
    let (k, v) = match kind {
        "v2_split_maskor" | "v2_sumdiff_maskor" => (6, 2),
        "v1_maskor" | "v1_canon3inst" => (3, 1),
        _ => panic!("unknown code {kind}"),
    };
    for st in 0..NSTATES as u32 {
        let x = st.wrapping_mul(LCG_A).wrapping_add(LCG_B);
        match kind {
            "v2_split_maskor" => {
                let hb = (x & MASK) | FIXED;
                let (hx, hy) = halves(hb);
                vx[st as usize] = hx;
                vy[st as usize] = hy;
            }
            "v2_sumdiff_maskor" => {
                let hb = (x & MASK) | FIXED;
                let (hx, hy) = halves(hb);
                vx[st as usize] = f16r(hx + hy);
                vy[st as usize] = f16r(hx - hy);
            }
            "v1_maskor" => {
                let hb = (x & MASK) | FIXED;
                let (hx, hy) = halves(hb);
                vx[st as usize] = f16r(hx + hy);
            }
            "v1_canon3inst" => {
                let hb = (x & MASK) ^ XOR_MAGIC;
                let (hx, hy) = halves(hb);
                vx[st as usize] = f16r(hx + hy);
            }
            _ => unreachable!(),
        }
    }
    Code {
        k,
        v,
        vx,
        vy,
        name: kind,
    }
}

/// One Viterbi pass over a span. `pin_lead`: if Some(l), only initial
/// states whose low (16-k) bits equal l are allowed (tail-biting pass 2).
/// Returns (state path, total cost).
fn viterbi_pass(x: &[f32], scale: f32, code: &Code, pin_lead: Option<u32>) -> (Vec<u32>, f32) {
    let k = code.k;
    let steps = x.len() / code.v;
    let n_pred = 1usize << k;
    let lead_mask = (1u32 << (L - k)) - 1;

    let cost = |step: usize, st: usize| -> f32 {
        if code.v == 1 {
            let d = x[step] - scale * code.vx[st];
            d * d
        } else {
            let d0 = x[2 * step] - scale * code.vx[st];
            let d1 = x[2 * step + 1] - scale * code.vy[st];
            d0 * d0 + d1 * d1
        }
    };

    let mut dp_old = vec![f32::INFINITY; NSTATES];
    let mut dp_new = vec![f32::INFINITY; NSTATES];
    let mut bp = vec![0u8; steps * NSTATES];

    for st in 0..NSTATES {
        let ok = match pin_lead {
            Some(l) => (st as u32 & lead_mask) == l,
            None => true,
        };
        dp_old[st] = if ok { cost(0, st) } else { f32::INFINITY };
    }

    // Right-shift bitshift trellis (matches the kernel window convention:
    // fresh bits enter at the HIGH end): new = (old >> k) | (fresh << (L-k)),
    // so old = ((new & lead_mask) << k) | u over the 2^k dropped low bits u.
    for step in 1..steps {
        let bprow = &mut bp[step * NSTATES..(step + 1) * NSTATES];
        for ns in 0..NSTATES {
            let base = (ns & lead_mask as usize) << k;
            let mut best = f32::INFINITY;
            let mut best_t = 0u8;
            for t in 0..n_pred {
                let p = base | t;
                let c = dp_old[p];
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
        st = ((st & lead_mask as usize) << k) | t;
    }
    path[0] = st as u32;
    (path, total)
}

/// Pack a state path's fresh bits into the LSB-first word ring.
fn pack_path(path: &[u32], k: usize, n_words: usize) -> Vec<u32> {
    let mut words = vec![0u32; n_words];
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

/// Ring-extract the 16-bit window ending at bit `end` (exclusive) of an
/// R-bit ring stored LSB-first in words.
fn ring_state(words: &[u32], ring_bits: usize, end: usize) -> u32 {
    let mut st = 0u32;
    let start = (end + ring_bits - L) % ring_bits;
    for i in 0..L {
        let b = (start + i) % ring_bits;
        st |= ((words[b >> 5] >> (b & 31)) & 1) << i;
    }
    st
}

/// Decode a packed span through the ring windows (bit-honest, same math
/// as the shipped kernels) and return reconstructed values (unscaled).
fn decode_span(words: &[u32], code: &Code, span_w: usize) -> Vec<f32> {
    let ring = span_w * 3; // 3 bits/weight at K=3 regardless of V
    let mut out = vec![0f32; span_w];
    if code.v == 1 {
        for j in 0..span_w {
            let st = ring_state(words, ring, (3 * (j + 1)) % ring) as usize;
            out[j] = code.vx[st];
        }
    } else {
        for t in 0..span_w / 2 {
            let st = ring_state(words, ring, (6 * (t + 1)) % ring) as usize;
            out[2 * t] = code.vx[st];
            out[2 * t + 1] = code.vy[st];
        }
    }
    out
}

/// Encode one span (two-pass tail-biting), return packed words.
fn encode_span(x: &[f32], scale: f32, code: &Code) -> Vec<u32> {
    let (p1, _) = viterbi_pass(x, scale, code, None);
    let lead = p1[p1.len() - 1] >> code.k; // last (16-k) stream bits
    let (p2, c2) = viterbi_pass(x, scale, code, Some(lead));
    let n_words = x.len() * 3 / 32;
    // Guard: pass 2 can in principle be worse than an unconstrained
    // wrap-violating pass 1; both decode legally through the ring, so
    // keep whichever has lower TRUE ring-decoded error.
    let w2 = pack_path(&p2, code.k, n_words);
    let w1 = pack_path(&p1, code.k, n_words);
    let err = |w: &Vec<u32>| -> f32 {
        decode_span(w, code, x.len())
            .iter()
            .zip(x)
            .map(|(v, xx)| {
                let d = xx - scale * v;
                d * d
            })
            .sum()
    };
    let (e1, e2) = (err(&w1), err(&w2));
    let _ = c2;
    if e2 <= e1 { w2 } else { w1 }
}

/// Encode a 256-weight group: fit fp16 scale (one refit), spans of
/// `span_w`, return summed squared error over the group.
fn encode_group(x: &[f32], code: &Code, span_w: usize) -> f64 {
    let fit = |scale: f32| -> (f32, f64) {
        // encode all spans at `scale`, return (ls-refit scale, sq err)
        let mut num = 0f64;
        let mut den = 0f64;
        let mut err = 0f64;
        for sp in x.chunks(span_w) {
            let words = encode_span(sp, scale, code);
            let v = decode_span(&words, code, span_w);
            for (xx, vv) in sp.iter().zip(v.iter()) {
                num += (*xx as f64) * (*vv as f64);
                den += (*vv as f64) * (*vv as f64);
                let d = *xx as f64 - scale as f64 * *vv as f64;
                err += d * d;
            }
        }
        let s_ls = if den > 0.0 { (num / den) as f32 } else { scale };
        (f16r(s_ls), err)
    };
    // initial scale: match rms of source to rms of code values
    let rms_x = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64).sqrt();
    let rms_c = (code
        .vx
        .iter()
        .chain(code.vy.iter().filter(|_| code.v == 2))
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / (NSTATES * code.v) as f64)
        .sqrt();
    let s0 = f16r((rms_x / rms_c.max(1e-9)) as f32);
    let (s1, e0) = fit(s0);
    let (_, e1) = fit(s1);
    e0.min(e1)
}

fn lloyd_max_row(x: &[f32], centroids: &[f32], block: usize) -> f64 {
    // per-block fp16 scale, 2 refits, nearest-centroid quantization
    let mut err = 0f64;
    for blk in x.chunks(block) {
        let mut s = f16r(
            ((blk.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / blk.len() as f64).sqrt()
                / (centroids
                    .iter()
                    .map(|c| (*c as f64) * (*c as f64))
                    .sum::<f64>()
                    / centroids.len() as f64)
                    .sqrt()) as f32,
        );
        let mut best_err = f64::INFINITY;
        for _ in 0..3 {
            let mut num = 0f64;
            let mut den = 0f64;
            let mut e = 0f64;
            for xx in blk {
                let mut bc = centroids[0];
                let mut bd = f32::INFINITY;
                for &c in centroids {
                    let d = (xx - s * c).abs();
                    if d < bd {
                        bd = d;
                        bc = c;
                    }
                }
                num += (*xx as f64) * (bc as f64);
                den += (bc as f64) * (bc as f64);
                let dd = *xx as f64 - s as f64 * bc as f64;
                e += dd * dd;
            }
            best_err = best_err.min(e);
            s = f16r(if den > 0.0 { (num / den) as f32 } else { s });
        }
        err += best_err;
    }
    err
}

fn ggml_row(x: &[f32], dtype: u32, n_per_row: usize) -> f64 {
    let nrows = x.len() / n_per_row;
    unsafe {
        llama_cpp_sys_2::ggml_quantize_init(dtype);
        let mut buf = vec![0u8; x.len() * 2]; // over-sized scratch
        let sz = llama_cpp_sys_2::ggml_quantize_chunk(
            dtype,
            x.as_ptr(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            nrows as i64,
            n_per_row as i64,
            std::ptr::null(),
        );
        buf.truncate(sz);
        let traits = llama_cpp_sys_2::ggml_get_type_traits(dtype);
        let to_float = (*traits).to_float.expect("to_float");
        let mut back = vec![0f32; x.len()];
        to_float(
            buf.as_ptr() as *const std::ffi::c_void,
            back.as_mut_ptr(),
            x.len() as i64,
        );
        x.iter()
            .zip(back.iter())
            .map(|(a, b)| {
                let d = (*a - *b) as f64;
                d * d
            })
            .sum()
    }
}

fn run_trellis(label: &str, x: &[f32], code: &Code, span_w: usize, bpw: f64) -> f64 {
    let t0 = Instant::now();
    let n_threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8);
    let groups: Vec<&[f32]> = x.chunks(GROUP_W).collect();
    let err: f64 = std::thread::scope(|scope| {
        let chunk = groups.len().div_ceil(n_threads);
        let handles: Vec<_> = groups
            .chunks(chunk)
            .map(|gs| {
                scope.spawn(move || {
                    gs.iter()
                        .map(|g| encode_group(g, code, span_w))
                        .sum::<f64>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    let mse = err / x.len() as f64;
    println!(
        "{label:<34} bpw={bpw:.4}  MSE={mse:.6}  ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    mse
}

fn main() {
    let n = N_GROUPS * GROUP_W;
    let x = gaussian_sample(n, SEED);
    let var = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64;
    println!("source: n={n} var={var:.4}  D_R(3 bit)={:.6}\n", var / 64.0);

    // TQ3_1S-class centroids (llama-cpp-turboquant tq3_centroids).
    let lm8: [f32; 8] = [
        -1.996684, -1.291398, -0.740341, -0.247508, 0.230106, 0.725222, 1.277503, 1.988943,
    ];

    let a = build_code("v2_split_maskor");
    let a2 = build_code("v2_sumdiff_maskor");
    let b = build_code("v1_maskor");
    let c = build_code("v1_canon3inst");

    let mse_a = run_trellis("A  v2_split_maskor  T=32", &x, &a, 32, 3.0625);
    let mse_a2 = run_trellis("A2 v2_sumdiff_maskor T=32", &x, &a2, 32, 3.0625);
    let mse_b = run_trellis("B  v1_maskor        T=32", &x, &b, 32, 3.0625);
    let mse_c = run_trellis("C  v1_canon3inst    T=256", &x, &c, 256, 3.0625);
    let mse_c2 = run_trellis("C2 v1_maskor        T=256", &x, &b, 256, 3.0625);
    let mse_d = run_trellis("D  v2_split_maskor  T=256", &x, &a, 256, 3.0625);
    let mse_a2t = run_trellis("A2' v2_sumdiff_maskor T=256", &x, &a2, 256, 3.0625);

    let e = lloyd_max_row(&x, &lm8, 32) / n as f64;
    println!("{:<34} bpw=3.5000  MSE={e:.6}", "E  lloyd-max scalar /32");
    let ep = lloyd_max_row(&x, &lm8, 256) / n as f64;
    println!("{:<34} bpw=3.0625  MSE={ep:.6}", "E' lloyd-max scalar /256");

    let f = ggml_row(&x, 11, GROUP_W) / n as f64; // Q3_K
    println!("{:<34} bpw=3.4375  MSE={f:.6}", "F  Q3_K (ggml)");
    let g = ggml_row(&x, 18, GROUP_W) / n as f64; // IQ3_XXS
    println!("{:<34} bpw=3.0625  MSE={g:.6}", "G  IQ3_XXS (ggml)");

    println!("\n== gate evaluation (docs/bench/2026-07-19-trellis3-quality-oracle) ==");
    println!("self-check: C ({mse_c:.6}) must be well below E' ({ep:.6}) or the encoder is broken");
    for (name, m) in [
        ("A", mse_a),
        ("A2", mse_a2),
        ("B", mse_b),
        ("A2'@256", mse_a2t),
        ("D@256", mse_d),
    ] {
        let pass = m <= 0.85 * ep && m <= 1.20 * mse_c;
        println!(
            "{name}: MSE={m:.6}  vs E' {:.3}x  vs C {:.3}x  -> {}",
            m / ep,
            m / mse_c,
            if pass { "PASS" } else { "fail" }
        );
    }
    println!(
        "diagnostics: T-effect(A/D)={:.3}x  code-form(C2/C)={:.3}x",
        mse_a / mse_d,
        mse_c2 / mse_c
    );
}

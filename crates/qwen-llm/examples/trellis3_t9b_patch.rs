//! T9b variant builder (docs/bench/2026-07-20-trellis3-t9b-e2e-arbiter/).
//!
//! Copies the F32 source GGUF and overwrites the 72 FFN tensor data
//! ranges in place with quantize->dequantize output per recipe:
//!   f32  — plain copy (baseline + surgery-path control)
//!   q3k / q4k — ggml_quantize_chunk -> to_float
//!   a0   — trellis V1, T7a input-side incoherence, plain span encode
//!   a3   — same + BlockLDLQ (per-block per-space Hessians, damp 0.01)
//!
//! Env: T9B_SRC, T9B_DST, T9B_RECIPE, T9B_HESS (a3), T9B_THREADS (12),
//! T9B_DAMP (0.01), T9B_NSUB (1).

use qwen_llm::gguf::GgufFile;
use qwen_llm::trellis_ldlq::{
    GramAccumulator, block_unit_lower_a, cholesky_lower, ldlq_quantize_row,
    rotate_hessian_input_f64,
};
use qwen_llm::trellis_offline::{TrellisCode, fwht128_blocks};
use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// T7a's exact input-side sign recipe.
fn t7a_signs(n_in: usize) -> Vec<f32> {
    let mut sgn = vec![1f32; n_in];
    let mut ss = 0x51_6E5u64 ^ (n_in as u64);
    for v in sgn.iter_mut() {
        if splitmix64(&mut ss) & 1 == 1 {
            *v = -1.0;
        }
    }
    sgn
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_f64_file(path: &str, n: usize) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len(), n * 8, "{path} size");
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ggml_roundtrip(rows: &[f32], dtype: u32, n_per_row: usize) -> Vec<f32> {
    let nrows = rows.len() / n_per_row;
    let mut back = vec![0f32; rows.len()];
    unsafe {
        llama_cpp_sys_2::ggml_quantize_init(dtype);
        let mut buf = vec![0u8; rows.len() * 2];
        let sz = llama_cpp_sys_2::ggml_quantize_chunk(
            dtype,
            rows.as_ptr(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            0,
            nrows as i64,
            n_per_row as i64,
            std::ptr::null(),
        );
        buf.truncate(sz);
        let traits = llama_cpp_sys_2::ggml_get_type_traits(dtype);
        let to_float = (*traits).to_float.expect("to_float");
        to_float(
            buf.as_ptr() as *const std::ffi::c_void,
            back.as_mut_ptr(),
            rows.len() as i64,
        );
    }
    back
}

/// Trellis roundtrip of full tensor rows in the T7a input basis, with
/// optional BlockLDLQ feedback. Returns reconstruction in the ORIGINAL
/// basis. Threaded over rows.
fn trellis_roundtrip(
    w: &[f32],
    n_rows: usize,
    d: usize,
    signs: &[f32],
    a: Option<&[f64]>,
    code: &TrellisCode,
    n_sub: usize,
    n_threads: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    let rows_in: Vec<&[f32]> = w.chunks(d).collect();
    let rows_out: Vec<&mut [f32]> = out.chunks_mut(d).collect();
    let mut pairs: Vec<(&[f32], &mut [f32])> = rows_in.into_iter().zip(rows_out).collect();
    let chunk = n_rows.div_ceil(n_threads.max(1));
    std::thread::scope(|scope| {
        for prs in pairs.chunks_mut(chunk) {
            scope.spawn(move || {
                for (rin, rout) in prs.iter_mut() {
                    // Forward input-side basis: signs then Hbd.
                    let mut rb: Vec<f32> = rin.iter().zip(signs).map(|(v, s)| v * s).collect();
                    fwht128_blocks(&mut rb);
                    let res = ldlq_quantize_row(&rb, a, d, code, n_sub);
                    // Inverse basis: Hbd (involution) then signs.
                    let mut rec = res.w_hat;
                    fwht128_blocks(&mut rec);
                    for (v, s) in rec.iter_mut().zip(signs) {
                        *v *= s;
                    }
                    rout.copy_from_slice(&rec);
                }
            });
        }
    });
    out
}

fn main() {
    let src = std::env::var("T9B_SRC")
        .unwrap_or_else(|_| qwen_llm::test_fixtures::QWEN35_0_8B_F32.path().into());
    let dst = std::env::var("T9B_DST").expect("T9B_DST");
    let recipe = std::env::var("T9B_RECIPE").expect("T9B_RECIPE (f32|q3k|q4k|a0|a3)");
    let n_threads = env_or("T9B_THREADS", 12);
    let n_sub = env_or("T9B_NSUB", 1);
    let damp: f64 = std::env::var("T9B_DAMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);

    let t0 = Instant::now();
    std::fs::copy(&src, &dst).expect("copy gguf");
    if recipe == "f32" {
        eprintln!(
            "[t9b-patch] f32 control copy done ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
        return;
    }

    let g = GgufFile::open(&src).expect("open src");
    let code = TrellisCode::v1_maskor();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&dst)
        .expect("open dst rw");

    // LDL factors per (space signature) for a3: keyed by (blk, space d).
    let hess_dir = std::env::var("T9B_HESS").unwrap_or_default();
    let mut lda_cache: std::collections::HashMap<(usize, usize), Option<Vec<f64>>> =
        std::collections::HashMap::new();

    let mut n_patched = 0usize;
    for blk in 0..24usize {
        for kind in ["ffn_gate", "ffn_up", "ffn_down"] {
            let name = format!("blk.{blk}.{kind}.weight");
            let t = g
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let d = t.shape[0] as usize;
            let n_rows = t.shape[1] as usize;
            let w = qwen_llm::codec::dequant_to_f32(t, g.try_slice(t).expect("slice"))
                .expect("dequant");
            let tt = Instant::now();
            let patched: Vec<f32> = match recipe.as_str() {
                "q3k" => ggml_roundtrip(&w, 11, d),
                "q4k" => ggml_roundtrip(&w, 12, d),
                "a0" | "a3" => {
                    let signs = t7a_signs(d);
                    let a = if recipe == "a3" {
                        let space = if kind == "ffn_down" { "inner" } else { "h" };
                        let key = (blk, d);
                        lda_cache
                            .entry(key)
                            .or_insert_with(|| {
                                let path = format!("{hess_dir}/gram_blk{blk}_{space}.f64");
                                let gram = read_f64_file(&path, d * d);
                                let acc = GramAccumulator {
                                    d,
                                    n_samples: 0,
                                    h: gram,
                                };
                                let mut h = acc.damped(damp);
                                rotate_hessian_input_f64(&mut h, d, &signs);
                                match cholesky_lower(&mut h, d) {
                                    Ok(()) => Some(block_unit_lower_a(&h, d)),
                                    Err(p) => {
                                        eprintln!(
                                            "[t9b-patch] WARN cholesky fail {name} pivot {p}"
                                        );
                                        None
                                    }
                                }
                            })
                            .as_deref()
                    } else {
                        None
                    };
                    if recipe == "a3" && a.is_none() {
                        panic!("a3 requires LDL factor for {name}");
                    }
                    trellis_roundtrip(&w, n_rows, d, &signs, a, &code, n_sub, n_threads)
                }
                other => panic!("unknown recipe {other}"),
            };
            assert_eq!(patched.len() * 4, t.n_bytes as usize, "{name} byte size");
            let bytes: Vec<u8> = patched.iter().flat_map(|v| v.to_le_bytes()).collect();
            file.seek(SeekFrom::Start(t.data_offset)).expect("seek");
            file.write_all(&bytes).expect("write");
            n_patched += 1;
            eprintln!(
                "[t9b-patch] {recipe} {name:<24} [{n_rows}x{d}] {:.1}s",
                tt.elapsed().as_secs_f64()
            );
        }
    }
    file.flush().expect("flush");
    eprintln!(
        "[t9b-patch] {recipe}: {n_patched} tensors patched into {dst} ({:.1}s total)",
        t0.elapsed().as_secs_f64()
    );
}

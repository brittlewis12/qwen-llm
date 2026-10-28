use super::*;

/// Opt-in switch for the timing half of `score_keys_match_score_bitwise`: any
/// value prints `attention_score` vs `attention_score_keys` microseconds per
/// dispatch at decode geometry; unset, only exactness runs.
const SCORE_TIMING_ENV: &str = "QWEN_V41_SCORE_TIMING";

fn score_us(ctx: &MetalContext, iters: usize, f: &dyn Fn(&KernelEncoder)) -> f64 {
    let mut samples = Vec::new();
    for round in 0..6 {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        for _ in 0..iters {
            f(&enc);
        }
        enc.end();
        metal::commit_and_wait(&command).unwrap();
        if round > 0 {
            samples.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6 / iters as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// `attention_score_keys` writes exactly the score slab bits of
/// `attention_score` (and leaves the same slots unwritten): decode geometry
/// (64 heads x 512) and ragged toy-like shapes, raw-ring wraparound, and
/// selected IDs past the visible length, whose slots neither kernel writes.
#[test]
fn score_keys_match_score_bitwise() {
    let ctx = MetalContext::new().unwrap();
    let (window, k, length, rows) = (128usize, 512usize, 1500u32, 2048usize);
    let make = |n: usize, seed: usize| {
        (0..n)
            .map(|i| ((i * 2_654_435_761usize + seed) % 1_000_003) as f32 / 250_000.0 - 2.0)
            .collect::<Vec<_>>()
    };
    let ids: Vec<u32> = (0..k as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 21)
        .collect();
    assert!(ids.iter().any(|&id| id >= length) && ids.iter().any(|&id| id < length));
    let selection = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![k as u64],
        GgmlType::I32,
    )
    .unwrap();
    for (heads, dim) in [(64usize, 512usize), (3, 32), (12, 96)] {
        let stride = window + k + 1;
        let up = |v: &[f32]| upload(&ctx, v, vec![v.len() as u64]).unwrap();
        let query = up(&make(heads * dim, 3));
        let raw = up(&make(window * dim, 5));
        let compressed = up(&make(rows * dim, 7));
        let encode = |enc: &KernelEncoder, keys: bool, nr: usize, nk: usize, out: &MetalTensor| {
            enc.set_bytes(0, &[dim as u32, nr as u32, 17u32, window as u32]);
            enc.set_bytes(1, &[nk as u32, stride as u32]);
            for (slot, t) in [(2, &query), (3, &raw), (4, &compressed), (5, &selection)] {
                enc.set_tensor(slot, t);
            }
            enc.set_tensor(7, out);
            enc.set_bytes(9, &length);
            let (name, grid) = if keys {
                enc.set_bytes(14, &(heads as u32));
                (
                    "score_keys",
                    (
                        (nr + nk).div_ceil(SCORE_KEY_BLOCK),
                        heads.div_ceil(SCORE_HEAD_BLOCK),
                    ),
                )
            } else {
                ("score", ((nr + nk).div_ceil(8), heads))
            };
            pipeline(&ctx, enc, name).unwrap();
            enc.dispatch(
                MTLSize {
                    width: grid.0,
                    height: grid.1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        };
        for (nr, nk) in [(1, 0), (5, 7), (17, 16), (128, 0), (128, 129), (128, 512)] {
            let old = up(&vec![f32::NAN; heads * stride]);
            let new = up(&vec![f32::NAN; heads * stride]);
            execute(&ctx, |enc| {
                encode(enc, false, nr, nk, &old);
                encode(enc, true, nr, nk, &new);
            });
            let bits = |t: &MetalTensor| {
                f32_slice(t)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            let (old, new) = (bits(&old), bits(&new));
            assert_eq!(old, new, "{heads}x{dim} {nr}+{nk}");
            let written = old.iter().filter(|b| !f32::from_bits(**b).is_nan()).count();
            let visible = nr + ids[..nk].iter().filter(|&&id| id < length).count();
            assert_eq!(written, heads * visible, "{heads}x{dim} {nr}+{nk}");
        }
        if heads == 64 && std::env::var_os(SCORE_TIMING_ENV).is_some() {
            let out = up(&vec![0.0; heads * stride]);
            for (nr, nk) in [(1, 0), (128, 0), (128, 256), (128, 512)] {
                let old = score_us(&ctx, 200, &|enc| encode(enc, false, nr, nk, &out));
                let new = score_us(&ctx, 200, &|enc| encode(enc, true, nr, nk, &out));
                eprintln!(
                    "score {heads}x{dim} {nr}+{nk} keys: attention_score {old:.2} us, attention_score_keys {new:.2} us ({:.2}x)",
                    old / new
                );
            }
        }
    }
}

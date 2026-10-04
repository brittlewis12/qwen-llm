//! CPU semantic contract for GLM-5.3-Flash operations that have no existing
//! shared contract. Arithmetic is f64; persistent state is stored as f32, as
//! the Metal kernels store it. Cross-checked against FLA's naive recurrence
//! (`scripts/reference/generate_glm53_kda_fixture.py`).

/// KDA head width.
pub const KDA_HEAD_DIM: usize = 128;
/// Conv history positions (kernel size 4, tap 3 is the current token).
pub const KDA_CONV_HISTORY: usize = 3;

/// Per-token KDA inputs for `heads` heads; vectors are `heads * 128` except
/// `raw_beta` (`heads`).
pub struct KdaStepInput<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub raw_gate: &'a [f32],
    pub raw_beta: &'a [f32],
    pub output_gate: &'a [f32],
}

/// Layer weights. Conv taps are channel-major `[channel * 4 + tap]` (GGUF
/// `[4, 1, channels]`); `neg_exp_a_log` is the GGUF `ssm_a = -exp(A_log)`.
pub struct KdaWeights<'a> {
    pub q_conv: &'a [f32],
    pub k_conv: &'a [f32],
    pub v_conv: &'a [f32],
    pub neg_exp_a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub output_norm: &'a [f32],
    pub lower_bound: f32,
    pub norm_eps: f32,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One decode step. `conv_state` is `[q|k|v][3 history][heads * 128]`
/// (oldest first); `state` is `[heads][value][key]`. Both are updated in
/// place; returns `heads * 128` outputs.
pub fn kda_decode_step(
    heads: usize,
    input: &KdaStepInput<'_>,
    weights: &KdaWeights<'_>,
    conv_state: &mut [f32],
    state: &mut [f32],
) -> Vec<f32> {
    const D: usize = KDA_HEAD_DIM;
    let width = heads * D;
    assert_eq!(conv_state.len(), 3 * KDA_CONV_HISTORY * width);
    assert_eq!(state.len(), heads * D * D);
    let mut conv = |which: usize, x: &[f32], taps: &[f32]| -> Vec<f64> {
        let plane = &mut conv_state[which * KDA_CONV_HISTORY * width..][..KDA_CONV_HISTORY * width];
        (0..width)
            .map(|c| {
                let mut acc = x[c] as f64 * taps[c * 4 + 3] as f64;
                for w in 0..KDA_CONV_HISTORY {
                    acc += plane[w * width + c] as f64 * taps[c * 4 + w] as f64;
                }
                for w in 0..KDA_CONV_HISTORY - 1 {
                    plane[w * width + c] = plane[(w + 1) * width + c];
                }
                plane[(KDA_CONV_HISTORY - 1) * width + c] = x[c];
                acc * sigmoid(acc)
            })
            .collect()
    };
    let q = conv(0, input.q, weights.q_conv);
    let k = conv(1, input.k, weights.k_conv);
    let v = conv(2, input.v, weights.v_conv);
    let mut out = vec![0.0f32; width];
    for h in 0..heads {
        let range = h * D..(h + 1) * D;
        let norm = |x: &[f64], scale: f64| -> Vec<f64> {
            let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
            x.iter().map(|v| v * inv * scale).collect()
        };
        let qh = norm(&q[range.clone()], 1.0 / (D as f64).sqrt());
        let kh = norm(&k[range.clone()], 1.0);
        let decay: Vec<f64> = range
            .clone()
            .map(|c| {
                let gate = input.raw_gate[c] as f64 + weights.dt_bias[c] as f64;
                let a = -(weights.neg_exp_a_log[h] as f64);
                (weights.lower_bound as f64 * sigmoid(a * gate)).exp()
            })
            .collect();
        let beta = sigmoid(input.raw_beta[h] as f64);
        let s = &mut state[h * D * D..(h + 1) * D * D];
        let mut o = vec![0.0f64; D];
        for (value, o_value) in o.iter_mut().enumerate() {
            let row = &mut s[value * D..(value + 1) * D];
            let decayed: Vec<f64> = row.iter().zip(&decay).map(|(s, d)| *s as f64 * d).collect();
            let sk: f64 = decayed.iter().zip(&kh).map(|(s, k)| s * k).sum();
            let delta = (v[h * D + value] - sk) * beta;
            let mut readout = 0.0;
            for key in 0..D {
                let updated = decayed[key] + kh[key] * delta;
                row[key] = updated as f32;
                readout += updated * qh[key];
            }
            *o_value = readout;
        }
        let inv_rms = 1.0
            / (o.iter().map(|x| x * x).sum::<f64>() / D as f64 + weights.norm_eps as f64).sqrt();
        for (i, c) in range.enumerate() {
            out[c] = (o[i]
                * inv_rms
                * weights.output_norm[i] as f64
                * sigmoid(input.output_gate[c] as f64)) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val(i: usize, salt: usize, scale: f64) -> f32 {
        ((((i * 7919 + salt * 104_729) % 2003) as f64 / 2003.0 - 0.5) * scale) as f32
    }

    fn series(n: usize, salt: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|i| val(i, salt, scale)).collect()
    }

    /// The CPU contract against FLA's naive recurrence (independent
    /// implementation): two heads, three tokens, nonzero nonsymmetric state,
    /// per-channel decay; output, final state and conv state.
    #[test]
    fn kda_contract_matches_fla_naive_recurrence() {
        const H: usize = 2;
        const T: usize = 3;
        let w = H * KDA_HEAD_DIM;
        let meta: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/glm53_kda_fla_v1.json"))
                .unwrap();
        assert_eq!(
            (meta["heads"].as_u64(), meta["tokens"].as_u64()),
            (Some(2), Some(3))
        );
        let blob: Vec<f32> = include_bytes!("../../tests/fixtures/glm53_kda_fla_v1.f32")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let range = |name: &str| {
            let r = &meta["layout"][name];
            r[0].as_u64().unwrap() as usize..r[1].as_u64().unwrap() as usize
        };
        let q_all = series(T * w, 1, 2.0);
        let k_all = series(T * w, 2, 2.0);
        let v_all = series(T * w, 3, 2.0);
        let gate_all = series(T * w, 4, 6.0);
        let beta_all = series(T * H, 5, 4.0);
        let og_all = series(T * w, 6, 4.0);
        let q_conv = series(w * 4, 7, 1.0);
        let k_conv = series(w * 4, 8, 1.0);
        let v_conv = series(w * 4, 9, 1.0);
        let neg_exp_a_log: Vec<f32> = (0..H)
            .map(|h| {
                (-(0.75 + ((((h * 7919 + 10 * 104_729) % 2003) as f64 / 2003.0) - 0.5))) as f32
            })
            .collect();
        let dt_bias = series(w, 11, 2.0);
        let output_norm: Vec<f32> = (0..KDA_HEAD_DIM)
            .map(|i| {
                (1.0 + ((((i * 7919 + 12 * 104_729) % 2003) as f64 / 2003.0) - 0.5) * 0.5) as f32
            })
            .collect();
        let mut conv_state = series(9 * w, 13, 1.0);
        let mut state = series(H * KDA_HEAD_DIM * KDA_HEAD_DIM, 14, 0.2);
        let weights = KdaWeights {
            q_conv: &q_conv,
            k_conv: &k_conv,
            v_conv: &v_conv,
            neg_exp_a_log: &neg_exp_a_log,
            dt_bias: &dt_bias,
            output_norm: &output_norm,
            lower_bound: -5.0,
            norm_eps: 1e-5,
        };
        let mut out = Vec::with_capacity(T * w);
        for t in 0..T {
            let s = t * w..(t + 1) * w;
            out.extend(kda_decode_step(
                H,
                &KdaStepInput {
                    q: &q_all[s.clone()],
                    k: &k_all[s.clone()],
                    v: &v_all[s.clone()],
                    raw_gate: &gate_all[s.clone()],
                    raw_beta: &beta_all[t * H..(t + 1) * H],
                    output_gate: &og_all[s],
                },
                &weights,
                &mut conv_state,
                &mut state,
            ));
        }
        let check = |label: &str, actual: &[f32], expected: &[f32], relative: f32| {
            let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = actual
                .iter()
                .zip(expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst / scale <= relative,
                "{label}: max abs {worst}, scale {scale}"
            );
        };
        check("output", &out, &blob[range("out")], 1e-5);
        check("state", &state, &blob[range("state")], 1e-5);
        assert_eq!(conv_state, blob[range("conv_state")], "conv state");
    }
}

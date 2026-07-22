//! Deterministic token sampling over a model logits row.
//!
//! Sampling state belongs to one request, not to a model or cached sequence.
//! At request start, reconstructing a sampler from the same configuration
//! produces the same random stream regardless of whether the prompt state was
//! reached by cold prefill, a RAM restore, or a disk restore. Resuming an
//! interrupted generation additionally requires the request-local draw count.

use std::cmp::Ordering;

/// Bump when candidate filtering, probability arithmetic, tie-breaking, or the
/// random-number generator changes.
pub const SAMPLER_ALGORITHM_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    /// Zero selects greedy decoding. Positive values scale retained logits.
    pub temperature: f32,
    /// Zero disables top-k filtering.
    pub top_k: usize,
    /// One disables nucleus filtering. Must be in `(0, 1]`.
    pub top_p: f32,
    /// Zero disables min-p filtering. Must be in `[0, 1]`.
    pub min_p: f32,
    /// Effective per-request seed. Callers choose and record random seeds before
    /// constructing the sampler; zero is an ordinary deterministic seed here.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    /// Sampling preset used by the local Qwen chat/game CLI before this engine
    /// gained a native sampler. The effective request seed remains explicit.
    pub fn qwen_chat(seed: u64) -> Self {
        Self {
            temperature: 0.7,
            top_k: 200,
            top_p: 1.0,
            min_p: 0.05,
            seed,
        }
    }

    pub fn validate(self) -> Result<Self, SamplingError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature(self.temperature));
        }
        if !self.top_p.is_finite() || !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(SamplingError::InvalidTopP(self.top_p));
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingError::InvalidMinP(self.min_p));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledToken {
    pub token: i32,
    /// Zero-based position in descending-logit order after every filter. Tied
    /// logits use ascending token id in sampled mode.
    pub candidate_index: usize,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SamplingError {
    #[error("sampling requires a non-empty logits row")]
    EmptyLogits,
    #[error("sampling temperature must be finite and >= 0, got {0}")]
    InvalidTemperature(f32),
    #[error("top_p must be finite and in (0, 1], got {0}")]
    InvalidTopP(f32),
    #[error("min_p must be finite and in [0, 1], got {0}")]
    InvalidMinP(f32),
    #[error("logit at token {token} is NaN")]
    NanLogit { token: usize },
    #[error("vocabulary size {0} exceeds i32 token ids")]
    VocabularyTooLarge(usize),
    #[error("sampling retained no finite-probability candidates")]
    NoCandidates,
}

#[derive(Clone, Debug)]
pub struct Sampler {
    config: SamplingConfig,
    rng: Xoshiro256PlusPlus,
    draws: usize,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self, SamplingError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            rng: Xoshiro256PlusPlus::from_seed(config.seed),
            draws: 0,
        })
    }

    /// Reconstruct a request-local sampler at an interrupted generation
    /// frontier. Model snapshots intentionally do not carry this count.
    pub fn at_draw(config: SamplingConfig, draws: usize) -> Result<Self, SamplingError> {
        let mut sampler = Self::new(config)?;
        for _ in 0..draws {
            sampler.rng.next_u64();
        }
        sampler.draws = draws;
        Ok(sampler)
    }

    pub fn config(&self) -> SamplingConfig {
        self.config
    }

    pub fn draws(&self) -> usize {
        self.draws
    }

    /// Select one token using the version-1 chain:
    ///
    /// `top-k -> min-p -> temperature -> top-p -> categorical distribution`
    ///
    /// Greedy decoding bypasses the chain and preserves the CLI's existing
    /// highest-token-id tie break. Positive-temperature sampling orders tied
    /// logits by ascending token id and consumes exactly one RNG draw per
    /// successful call, including when filtering leaves a single candidate.
    ///
    /// Filtering intentionally follows the local contract rather than claiming
    /// numerical parity with llama.cpp: min-p uses pre-temperature f64 logits,
    /// while top-p uses the temperature-adjusted f64 distribution. Version-1
    /// replay is scoped to the supported target/build and guarded by golden
    /// vectors for the RNG, f64 draw conversion, and sampled token stream.
    pub fn sample(&mut self, logits: &[f32]) -> Result<SampledToken, SamplingError> {
        if self.config.temperature == 0.0 {
            return Ok(SampledToken {
                token: greedy_token(logits)?,
                candidate_index: 0,
            });
        }

        let mut candidates = sorted_candidates(logits, self.config.top_k)?;

        if self.config.min_p > 0.0 {
            let max_logit = candidates[0].logit;
            if max_logit.is_finite() {
                let threshold = max_logit + f64::from(self.config.min_p).ln();
                candidates.retain(|candidate| candidate.logit >= threshold);
            }
        }
        if candidates.is_empty() {
            return Err(SamplingError::NoCandidates);
        }

        if candidates[0].logit == f64::INFINITY {
            candidates.retain(|candidate| candidate.logit == f64::INFINITY);
        }

        let temperature = f64::from(self.config.temperature);
        for candidate in &mut candidates {
            candidate.logit /= temperature;
        }
        let mut weights = probability_weights(&candidates)?;

        if self.config.top_p < 1.0 {
            let total: f64 = weights.iter().sum();
            let target = total * f64::from(self.config.top_p);
            let mut cumulative = 0.0;
            let mut keep = 0usize;
            for weight in &weights {
                cumulative += *weight;
                keep += 1;
                if cumulative >= target {
                    break;
                }
            }
            candidates.truncate(keep.max(1));
            weights.truncate(candidates.len());
        }

        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(SamplingError::NoCandidates);
        }
        self.draws += 1;
        let target = self.rng.next_unit_f64() * total;
        let mut cumulative = 0.0;
        for (candidate_index, (candidate, weight)) in candidates.iter().zip(&weights).enumerate() {
            cumulative += *weight;
            if target < cumulative {
                return Ok(SampledToken {
                    token: candidate.token,
                    candidate_index,
                });
            }
        }

        let candidate_index = candidates.len() - 1;
        Ok(SampledToken {
            token: candidates[candidate_index].token,
            candidate_index,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    token: i32,
    logit: f64,
}

fn validate_logits_shape(logits: &[f32]) -> Result<(), SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if logits.len() > i32::MAX as usize {
        return Err(SamplingError::VocabularyTooLarge(logits.len()));
    }
    Ok(())
}

fn greedy_token(logits: &[f32]) -> Result<i32, SamplingError> {
    validate_logits_shape(logits)?;
    let mut best_token = 0usize;
    let mut best_logit = logits[0];
    if best_logit.is_nan() {
        return Err(SamplingError::NanLogit { token: 0 });
    }
    for (token, &logit) in logits.iter().enumerate().skip(1) {
        if logit.is_nan() {
            return Err(SamplingError::NanLogit { token });
        }
        if logit.total_cmp(&best_logit) != Ordering::Less {
            best_token = token;
            best_logit = logit;
        }
    }
    Ok(best_token as i32)
}

fn sorted_candidates(logits: &[f32], top_k: usize) -> Result<Vec<Candidate>, SamplingError> {
    validate_logits_shape(logits)?;
    let mut candidates = Vec::with_capacity(logits.len());
    for (token, &logit) in logits.iter().enumerate() {
        if logit.is_nan() {
            return Err(SamplingError::NanLogit { token });
        }
        candidates.push(Candidate {
            token: token as i32,
            logit: f64::from(logit),
        });
    }

    let compare = |a: &Candidate, b: &Candidate| match b.logit.total_cmp(&a.logit) {
        Ordering::Equal => a.token.cmp(&b.token),
        order => order,
    };
    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k, compare);
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(compare);
    Ok(candidates)
}

fn probability_weights(candidates: &[Candidate]) -> Result<Vec<f64>, SamplingError> {
    let n_positive_infinity = candidates
        .iter()
        .filter(|candidate| candidate.logit == f64::INFINITY)
        .count();
    if n_positive_infinity > 0 {
        return Ok(candidates
            .iter()
            .map(|candidate| f64::from(candidate.logit == f64::INFINITY))
            .collect());
    }

    let max_logit = candidates[0].logit;
    if max_logit == f64::NEG_INFINITY {
        return Err(SamplingError::NoCandidates);
    }
    let weights: Vec<_> = candidates
        .iter()
        .map(|candidate| (candidate.logit - max_logit).exp())
        .collect();
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(SamplingError::NoCandidates);
    }
    Ok(weights)
}

/// Fixed request-local RNG. The algorithm is part of
/// [`SAMPLER_ALGORITHM_VERSION`] and deliberately does not depend on `rand`'s
/// evolving `StdRng` contract.
#[derive(Clone, Debug)]
struct Xoshiro256PlusPlus {
    state: [u64; 4],
}

impl Xoshiro256PlusPlus {
    fn from_seed(seed: u64) -> Self {
        let mut splitmix = SplitMix64(seed);
        Self {
            state: [
                splitmix.next(),
                splitmix.next(),
                splitmix.next(),
                splitmix.next(),
            ],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_unit_f64(&mut self) -> f64 {
        const DENOMINATOR: f64 = (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 / DENOMINATOR
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(config: SamplingConfig) -> Sampler {
        Sampler::new(config).expect("valid sampler")
    }

    #[test]
    fn defaults_are_greedy_and_chat_sampling_is_an_explicit_preset() {
        assert_eq!(
            SamplingConfig::default(),
            SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 0,
            }
        );
        assert_eq!(
            SamplingConfig::qwen_chat(42),
            SamplingConfig {
                temperature: 0.7,
                top_k: 200,
                top_p: 1.0,
                min_p: 0.05,
                seed: 42,
            }
        );
        assert_eq!(SAMPLER_ALGORITHM_VERSION, 1);
    }

    #[test]
    fn rejects_invalid_configuration_and_logits() {
        let error = Sampler::new(SamplingConfig {
            temperature: f32::NAN,
            ..SamplingConfig::default()
        })
        .unwrap_err();
        assert!(matches!(
            error,
            SamplingError::InvalidTemperature(value) if value.is_nan()
        ));
        assert!(matches!(
            Sampler::new(SamplingConfig {
                top_p: 0.0,
                ..SamplingConfig::default()
            }),
            Err(SamplingError::InvalidTopP(0.0))
        ));
        for temperature in [-1.0, f32::INFINITY] {
            assert!(matches!(
                Sampler::new(SamplingConfig {
                    temperature,
                    ..SamplingConfig::default()
                }),
                Err(SamplingError::InvalidTemperature(_))
            ));
        }
        for top_p in [-0.1, 1.1, f32::INFINITY] {
            assert!(matches!(
                Sampler::new(SamplingConfig {
                    top_p,
                    ..SamplingConfig::default()
                }),
                Err(SamplingError::InvalidTopP(_))
            ));
        }
        for min_p in [-0.1, 1.1, f32::INFINITY] {
            assert!(matches!(
                Sampler::new(SamplingConfig {
                    min_p,
                    ..SamplingConfig::default()
                }),
                Err(SamplingError::InvalidMinP(_))
            ));
        }
        let mut sampler = sampler(SamplingConfig::default());
        assert_eq!(sampler.sample(&[]), Err(SamplingError::EmptyLogits));
        assert_eq!(
            sampler.sample(&[0.0, f32::NAN]),
            Err(SamplingError::NanLogit { token: 1 })
        );
    }

    #[test]
    fn greedy_preserves_highest_token_id_tie_break() {
        let mut sampler = sampler(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        });
        assert_eq!(sampler.sample(&[1.0, 3.0, 3.0]).unwrap().token, 2);
    }

    #[test]
    fn identical_request_seed_reproduces_the_stream() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0x1234_5678_9abc_def0,
        };
        let logits = [2.0, 1.5, 1.0, 0.5];
        let mut a = sampler(config);
        let mut b = sampler(config);
        let a_tokens: Vec<_> = (0..64).map(|_| a.sample(&logits).unwrap().token).collect();
        let b_tokens: Vec<_> = (0..64).map(|_| b.sample(&logits).unwrap().token).collect();
        assert_eq!(a_tokens, b_tokens);
        assert!(a_tokens.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn rng_and_sample_stream_match_version_one_golden_vectors() {
        let mut splitmix_zero = SplitMix64(0);
        assert_eq!(
            [
                splitmix_zero.next(),
                splitmix_zero.next(),
                splitmix_zero.next(),
                splitmix_zero.next(),
            ],
            [
                0xe220_a839_7b1d_cdaf,
                0x6e78_9e6a_a1b9_65f4,
                0x06c4_5d18_8009_454f,
                0xf88b_b8a8_724c_81ec,
            ]
        );

        let mut rng = Xoshiro256PlusPlus::from_seed(0x1234_5678_9abc_def0);
        assert_eq!(
            [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64()
            ],
            [
                0x4d4f_7607_a97a_1bd6,
                0x9ba0_27c7_6910_d021,
                0x87ad_b062_153a_e0bc,
                0xb750_f7b1_ff94_4783,
            ]
        );

        let mut rng = Xoshiro256PlusPlus::from_seed(0);
        assert_eq!(
            [
                rng.next_unit_f64().to_bits(),
                rng.next_unit_f64().to_bits(),
                rng.next_unit_f64().to_bits(),
                rng.next_unit_f64().to_bits(),
            ],
            [
                0x3fd4_c5d7_5852_42c8,
                0x3fd8_769b_cf70_e034,
                0x3fd7_03f7_e47b_269e,
                0x3f87_75fc_61dd_f2c0,
            ]
        );

        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0x1234_5678_9abc_def0,
        };
        let mut sampler = sampler(config);
        let actual: Vec<_> = (0..16)
            .map(|_| sampler.sample(&[2.0, 1.5, 1.0, 0.5]).unwrap().token)
            .collect();
        assert_eq!(actual, [0, 1, 1, 1, 3, 1, 3, 0, 0, 3, 0, 0, 2, 0, 2, 1]);
    }

    #[test]
    fn draw_count_reconstructs_an_interrupted_sample_stream() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
        };
        let logits = [1.0, 0.5, 0.0];
        let mut uninterrupted = sampler(config);
        for _ in 0..17 {
            uninterrupted.sample(&logits).unwrap();
        }
        let mut resumed = Sampler::at_draw(config, uninterrupted.draws()).unwrap();
        for _ in 0..32 {
            assert_eq!(
                uninterrupted.sample(&logits).unwrap(),
                resumed.sample(&logits).unwrap()
            );
        }
    }

    #[test]
    fn top_k_one_forces_the_argmax_and_has_versioned_draw_consumption() {
        let mut sampler = sampler(SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            seed: 7,
        });
        for draw in 1..=32 {
            assert_eq!(sampler.sample(&[0.0, 3.0, 1.0]).unwrap().token, 1);
            assert_eq!(sampler.draws(), draw);
        }
    }

    #[test]
    fn filtering_order_and_tie_boundaries_are_explicit() {
        let mut min_p = sampler(SamplingConfig {
            temperature: 2.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.5,
            seed: 1,
        });
        for _ in 0..32 {
            assert_eq!(min_p.sample(&[0.0, -1.0]).unwrap().token, 0);
        }

        let mut top_p = sampler(SamplingConfig {
            temperature: 2.0,
            top_k: 0,
            top_p: 0.7,
            min_p: 0.0,
            seed: 1,
        });
        let tokens: Vec<_> = (0..64)
            .map(|_| top_p.sample(&[0.0, -1.0]).unwrap().token)
            .collect();
        assert!(
            tokens.contains(&1),
            "top-p must observe temperature scaling"
        );

        let mut tied_top_k = sampler(SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            seed: 9,
        });
        for _ in 0..64 {
            let sampled = tied_top_k.sample(&[2.0, 2.0, 2.0]).unwrap();
            assert!(matches!(sampled.token, 0 | 1));
            assert_eq!(sampled.candidate_index, sampled.token as usize);
        }

        let mut tied_min_p = sampler(SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 1.0,
            seed: 4,
        });
        let tokens: Vec<_> = (0..64)
            .map(|_| tied_min_p.sample(&[3.0, 3.0, 1.0]).unwrap().token)
            .collect();
        assert!(tokens.contains(&0) && tokens.contains(&1));
    }

    #[test]
    fn partial_top_k_matches_the_total_order_full_sort() {
        let mut splitmix = SplitMix64(0xfeed_face_cafe_beef);
        for len in [2usize, 3, 17, 257] {
            let mut logits: Vec<f32> = (0..len)
                .map(|_| {
                    let bits = (splitmix.next() >> 40) as u32;
                    (bits as f32 / (1u32 << 24) as f32) * 20.0 - 10.0
                })
                .collect();
            logits[0] = 0.0;
            logits[1] = -0.0;
            if len >= 3 {
                logits[2] = logits[0];
            }
            if len >= 17 {
                logits[5] = f32::INFINITY;
                logits[11] = f32::NEG_INFINITY;
            }

            let full = sorted_candidates(&logits, 0).unwrap();
            let mut ks = vec![1, len - 1, len];
            if len > 8 {
                ks.push(8);
            }
            for k in ks {
                let partial = sorted_candidates(&logits, k).unwrap();
                assert_eq!(partial, full[..k], "len={len} k={k}");
            }
        }
    }

    #[test]
    fn top_p_can_reduce_the_distribution_to_the_argmax() {
        let mut sampler = sampler(SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            min_p: 0.0,
            seed: 99,
        });
        for _ in 0..32 {
            assert_eq!(sampler.sample(&[4.0, 0.0, -1.0]).unwrap().token, 0);
        }
    }

    #[test]
    fn positive_infinities_form_a_finite_uniform_support() {
        let mut sampler = sampler(SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 11,
        });
        for _ in 0..64 {
            assert!(matches!(
                sampler
                    .sample(&[f32::INFINITY, 1000.0, f32::INFINITY])
                    .unwrap()
                    .token,
                0 | 2
            ));
        }
    }

    #[test]
    fn negative_infinities_have_zero_weight_and_all_negative_infinity_fails() {
        let mut sampler = sampler(SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 11,
        });
        for _ in 0..32 {
            assert_eq!(
                sampler
                    .sample(&[0.0, f32::NEG_INFINITY, -1000.0])
                    .unwrap()
                    .token,
                0
            );
        }
        assert_eq!(
            sampler.sample(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            Err(SamplingError::NoCandidates)
        );
    }
}

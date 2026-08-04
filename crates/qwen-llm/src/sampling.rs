//! Deterministic token sampling over a model logits row.
//!
//! Sampling state belongs to one request, not to a model or cached sequence.
//! At request start, reconstructing a sampler from the same configuration
//! produces the same random stream regardless of whether the prompt state was
//! reached by cold prefill, a RAM restore, or a disk restore. Resuming an
//! interrupted generation additionally requires the request-local draw count.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BoundedTopKEvidence {
    pub input_logits: usize,
    pub retained_top_k: usize,
    pub max_heap_len: usize,
    pub heap_capacity: usize,
    pub used_bounded_path: bool,
}

/// Opt-in wall attribution for one positive-temperature sampler-v1 call.
///
/// This is diagnostic only. The ordinary [`Sampler::sample`] path remains
/// uninstrumented, and all phase times include exactly the existing work.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingPhaseProfile {
    pub timer_spans: u32,
    pub input_logits: usize,
    pub total_ms: f64,
    pub shape_validation_ms: f64,
    pub candidate_alloc_ms: f64,
    pub candidate_fill_ms: f64,
    pub top_k_order_ms: f64,
    pub min_p_ms: f64,
    pub positive_infinity_ms: f64,
    pub temperature_scale_ms: f64,
    pub probability_weights_ms: f64,
    pub top_p_ms: f64,
    pub categorical_ms: f64,
    pub residual_ms: f64,
    pub candidate_capacity_bytes: usize,
    pub probability_capacity_bytes: usize,
    pub after_top_k: usize,
    pub after_min_p: usize,
    pub after_positive_infinity: usize,
    pub after_top_p: usize,
    pub candidate_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GreedySelection {
    Token(i32),
    NanLogit { token: usize },
}

impl GreedySelection {
    pub(crate) fn from_encoded(raw: i32) -> Self {
        if raw >= 0 {
            Self::Token(raw)
        } else {
            Self::NanLogit {
                token: (!raw) as usize,
            }
        }
    }

    pub fn into_token(self) -> Result<i32, SamplingError> {
        match self {
            Self::Token(token) => Ok(token),
            Self::NanLogit { token } => Err(SamplingError::NanLogit { token }),
        }
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
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

    /// Execute sampler-v1 while retaining only the configured top-k candidates.
    ///
    /// Positive-temperature requests with `0 < top_k < logits.len()` use the
    /// bounded path. Other configurations preserve the ordinary sampler and
    /// report fallback evidence; product callers can reject that evidence.
    pub fn sample_bounded_top_k(
        &mut self,
        logits: &[f32],
    ) -> Result<(SampledToken, BoundedTopKEvidence), SamplingError> {
        if self.config.temperature == 0.0
            || self.config.top_k == 0
            || self.config.top_k >= logits.len()
        {
            let sampled = self.sample(logits)?;
            return Ok((
                sampled,
                BoundedTopKEvidence {
                    input_logits: logits.len(),
                    ..BoundedTopKEvidence::default()
                },
            ));
        }

        validate_logits_shape(logits)?;
        let top_k = self.config.top_k;
        let mut heap = BinaryHeap::with_capacity(top_k);
        let mut max_heap_len = 0usize;
        for (token, &logit) in logits.iter().enumerate() {
            if logit.is_nan() {
                return Err(SamplingError::NanLogit { token });
            }
            let candidate = HeapCandidate(Candidate {
                token: token as i32,
                logit: f64::from(logit),
            });
            if heap.len() < top_k {
                heap.push(candidate);
                max_heap_len = max_heap_len.max(heap.len());
            } else if candidate_better(&candidate.0, &heap.peek().expect("full heap").0) {
                heap.pop();
                heap.push(candidate);
            }
        }
        let heap_capacity = heap.capacity();
        let mut candidates: Vec<Candidate> = heap.into_iter().map(|entry| entry.0).collect();
        candidates.sort_unstable_by(candidate_order);

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
        let evidence = BoundedTopKEvidence {
            input_logits: logits.len(),
            retained_top_k: top_k,
            max_heap_len,
            heap_capacity,
            used_bounded_path: true,
        };
        self.draws += 1;
        let target = self.rng.next_unit_f64() * total;
        let mut cumulative = 0.0;
        for (candidate_index, (candidate, weight)) in candidates.iter().zip(&weights).enumerate() {
            cumulative += *weight;
            if target < cumulative {
                return Ok((
                    SampledToken {
                        token: candidate.token,
                        candidate_index,
                    },
                    evidence,
                ));
            }
        }

        let candidate_index = candidates.len() - 1;
        Ok((
            SampledToken {
                token: candidates[candidate_index].token,
                candidate_index,
            },
            evidence,
        ))
    }

    /// Run the exact sampler-v1 chain with opt-in phase attribution.
    ///
    /// The implementation intentionally mirrors [`Self::sample`] rather than
    /// routing the default path through timers. Equivalence tests pin outcomes,
    /// errors, candidate indices, draw counts, and subsequent RNG state.
    pub fn sample_profiled(
        &mut self,
        logits: &[f32],
    ) -> Result<(SampledToken, SamplingPhaseProfile), SamplingError> {
        if self.config.temperature == 0.0 {
            return Ok((
                SampledToken {
                    token: greedy_token(logits)?,
                    candidate_index: 0,
                },
                SamplingPhaseProfile {
                    input_logits: logits.len(),
                    ..SamplingPhaseProfile::default()
                },
            ));
        }

        let total_t0 = Instant::now();
        let shape_t0 = Instant::now();
        validate_logits_shape(logits)?;
        let shape_validation_ms = elapsed_ms(shape_t0);

        let alloc_t0 = Instant::now();
        let mut candidates = Vec::with_capacity(logits.len());
        let candidate_alloc_ms = elapsed_ms(alloc_t0);
        let candidate_capacity_bytes = candidates
            .capacity()
            .saturating_mul(std::mem::size_of::<Candidate>());

        let fill_t0 = Instant::now();
        for (token, &logit) in logits.iter().enumerate() {
            if logit.is_nan() {
                return Err(SamplingError::NanLogit { token });
            }
            candidates.push(Candidate {
                token: token as i32,
                logit: f64::from(logit),
            });
        }
        let candidate_fill_ms = elapsed_ms(fill_t0);

        let order_t0 = Instant::now();
        let compare = |a: &Candidate, b: &Candidate| match b.logit.total_cmp(&a.logit) {
            Ordering::Equal => a.token.cmp(&b.token),
            order => order,
        };
        if self.config.top_k > 0 && self.config.top_k < candidates.len() {
            candidates.select_nth_unstable_by(self.config.top_k, compare);
            candidates.truncate(self.config.top_k);
        }
        candidates.sort_unstable_by(compare);
        let top_k_order_ms = elapsed_ms(order_t0);
        let after_top_k = candidates.len();

        let min_p_t0 = Instant::now();
        if self.config.min_p > 0.0 {
            let max_logit = candidates[0].logit;
            if max_logit.is_finite() {
                let threshold = max_logit + f64::from(self.config.min_p).ln();
                candidates.retain(|candidate| candidate.logit >= threshold);
            }
        }
        let min_p_ms = elapsed_ms(min_p_t0);
        let after_min_p = candidates.len();
        if candidates.is_empty() {
            return Err(SamplingError::NoCandidates);
        }

        let positive_infinity_t0 = Instant::now();
        if candidates[0].logit == f64::INFINITY {
            candidates.retain(|candidate| candidate.logit == f64::INFINITY);
        }
        let positive_infinity_ms = elapsed_ms(positive_infinity_t0);
        let after_positive_infinity = candidates.len();

        let temperature_t0 = Instant::now();
        let temperature = f64::from(self.config.temperature);
        for candidate in &mut candidates {
            candidate.logit /= temperature;
        }
        let temperature_scale_ms = elapsed_ms(temperature_t0);

        let probability_t0 = Instant::now();
        let mut weights = probability_weights(&candidates)?;
        let probability_weights_ms = elapsed_ms(probability_t0);
        let probability_capacity_bytes = weights
            .capacity()
            .saturating_mul(std::mem::size_of::<f64>());

        let top_p_t0 = Instant::now();
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
        let top_p_ms = elapsed_ms(top_p_t0);
        let after_top_p = candidates.len();

        let categorical_t0 = Instant::now();
        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(SamplingError::NoCandidates);
        }
        self.draws += 1;
        let target = self.rng.next_unit_f64() * total;
        let mut cumulative = 0.0;
        let mut sampled = None;
        for (candidate_index, (candidate, weight)) in candidates.iter().zip(&weights).enumerate() {
            cumulative += *weight;
            if target < cumulative {
                sampled = Some(SampledToken {
                    token: candidate.token,
                    candidate_index,
                });
                break;
            }
        }
        let sampled = sampled.unwrap_or_else(|| {
            let candidate_index = candidates.len() - 1;
            SampledToken {
                token: candidates[candidate_index].token,
                candidate_index,
            }
        });
        let categorical_ms = elapsed_ms(categorical_t0);
        drop(weights);
        drop(candidates);
        let total_ms = elapsed_ms(total_t0);
        let phase_sum_ms = shape_validation_ms
            + candidate_alloc_ms
            + candidate_fill_ms
            + top_k_order_ms
            + min_p_ms
            + positive_infinity_ms
            + temperature_scale_ms
            + probability_weights_ms
            + top_p_ms
            + categorical_ms;

        Ok((
            sampled,
            SamplingPhaseProfile {
                timer_spans: 11,
                input_logits: logits.len(),
                total_ms,
                shape_validation_ms,
                candidate_alloc_ms,
                candidate_fill_ms,
                top_k_order_ms,
                min_p_ms,
                positive_infinity_ms,
                temperature_scale_ms,
                probability_weights_ms,
                top_p_ms,
                categorical_ms,
                residual_ms: total_ms - phase_sum_ms,
                candidate_capacity_bytes,
                probability_capacity_bytes,
                after_top_k,
                after_min_p,
                after_positive_infinity,
                after_top_p,
                candidate_index: sampled.candidate_index,
            },
        ))
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    token: i32,
    logit: f64,
}

fn candidate_order(a: &Candidate, b: &Candidate) -> Ordering {
    match b.logit.total_cmp(&a.logit) {
        Ordering::Equal => a.token.cmp(&b.token),
        order => order,
    }
}

fn candidate_better(a: &Candidate, b: &Candidate) -> bool {
    candidate_order(a, b) == Ordering::Less
}

#[derive(Clone, Copy, Debug)]
struct HeapCandidate(Candidate);

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.token == other.0.token && self.0.logit.to_bits() == other.0.logit.to_bits()
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.logit.total_cmp(&other.0.logit) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.0.token.cmp(&other.0.token),
        }
    }
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

    #[test]
    fn greedy_selection_decodes_tokens_and_nan_indices() {
        assert_eq!(
            GreedySelection::from_encoded(42),
            GreedySelection::Token(42)
        );
        assert_eq!(
            GreedySelection::from_encoded(-1),
            GreedySelection::NanLogit { token: 0 }
        );
        assert_eq!(
            GreedySelection::from_encoded(-248_320),
            GreedySelection::NanLogit { token: 248_319 }
        );
        assert_eq!(GreedySelection::Token(7).into_token().unwrap(), 7);
        assert_eq!(
            GreedySelection::NanLogit { token: 9 }.into_token(),
            Err(SamplingError::NanLogit { token: 9 })
        );
    }

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

    fn assert_bounded_equivalent(config: SamplingConfig, logits: &[f32], continuation: &[f32]) {
        assert!(config.temperature > 0.0);
        assert!(config.top_k > 0 && config.top_k < logits.len());
        assert!(config.top_k < continuation.len());
        let mut ordinary = sampler(config);
        let mut bounded = sampler(config);
        let ordinary_result = ordinary.sample(logits);
        let bounded_result = bounded
            .sample_bounded_top_k(logits)
            .map(|(sampled, evidence)| {
                assert!(evidence.used_bounded_path);
                assert_eq!(evidence.input_logits, logits.len());
                assert_eq!(evidence.retained_top_k, config.top_k);
                assert_eq!(evidence.max_heap_len, config.top_k);
                assert!(evidence.heap_capacity >= config.top_k);
                assert!(evidence.heap_capacity < logits.len());
                sampled
            });
        assert_eq!(bounded_result, ordinary_result);
        assert_eq!(bounded.draws(), ordinary.draws());

        let ordinary_next = ordinary.sample(continuation);
        let bounded_next = bounded
            .sample_bounded_top_k(continuation)
            .map(|(sampled, evidence)| {
                assert!(evidence.used_bounded_path);
                sampled
            });
        assert_eq!(bounded_next, ordinary_next);
        assert_eq!(bounded.draws(), ordinary.draws());
    }

    #[test]
    fn bounded_top_k_matches_sampler_v1() {
        let seeds = [0, 1, 42, u64::MAX];
        let lengths = [2usize, 17, 201, 257, 1024];
        let continuation: Vec<f32> = (0..1025)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 31.0)
            .collect();
        for seed in seeds {
            for len in lengths {
                let mut ks = vec![1, 199, 200, 201, len - 1];
                ks.retain(|&top_k| top_k < len);
                ks.sort_unstable();
                ks.dedup();
                let mut generator = SplitMix64(seed);
                for case in 0..24 {
                    let logits: Vec<f32> = (0..len)
                        .map(|_| {
                            let mut bits = generator.next() as u32;
                            if case < 16 && bits & 0x7f80_0000 == 0x7f80_0000 {
                                bits &= !0x0080_0000;
                            }
                            f32::from_bits(bits)
                        })
                        .collect();
                    for &top_k in &ks {
                        for (temperature, top_p, min_p) in [(0.7, 1.0, 0.05), (1.3, 0.8, 0.0)] {
                            assert_bounded_equivalent(
                                SamplingConfig {
                                    temperature,
                                    top_k,
                                    top_p,
                                    min_p,
                                    seed,
                                },
                                &logits,
                                &continuation,
                            );
                        }
                    }
                }
            }
        }

        for len in [2usize, 17, 201] {
            let mut ks = vec![1, 199, 200, 201, len - 1];
            ks.retain(|&top_k| top_k < len);
            ks.sort_unstable();
            ks.dedup();
            for payload in [0x7fc0_0000, 0x7f80_0001, 0xffc0_0001] {
                for nan_token in 0..len {
                    for &top_k in &ks {
                        let mut logits = vec![0.0; len];
                        logits[nan_token] = f32::from_bits(payload);
                        assert_bounded_equivalent(
                            SamplingConfig {
                                temperature: 0.7,
                                top_k,
                                top_p: 1.0,
                                min_p: 0.05,
                                seed: 42,
                            },
                            &logits,
                            &continuation,
                        );
                    }
                }
            }
        }

        for logits in [
            vec![0.0, -0.0, 0.0, -0.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 1.0],
            vec![f32::INFINITY, 1.0, f32::INFINITY, f32::NEG_INFINITY],
            vec![f32::NEG_INFINITY; 5],
        ] {
            assert_bounded_equivalent(
                SamplingConfig {
                    temperature: 0.7,
                    top_k: 3,
                    top_p: 1.0,
                    min_p: 0.05,
                    seed: 42,
                },
                &logits,
                &continuation,
            );
        }

        for (config, logits) in [
            (
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 3,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed: 0x1234_5678_9abc_def0,
                },
                vec![2.0, 1.5, 1.0, 0.5],
            ),
            (
                SamplingConfig {
                    temperature: 2.0,
                    top_k: 2,
                    top_p: 1.0,
                    min_p: 0.5,
                    seed: 1,
                },
                vec![0.0, -1.0, -2.0],
            ),
            (
                SamplingConfig {
                    temperature: 2.0,
                    top_k: 2,
                    top_p: 0.7,
                    min_p: 0.0,
                    seed: 1,
                },
                vec![0.0, -1.0, -2.0],
            ),
            (
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 2,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed: 9,
                },
                vec![2.0, 2.0, 2.0],
            ),
            (
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 3,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed: 11,
                },
                vec![f32::INFINITY, 1000.0, f32::INFINITY, -1000.0],
            ),
            (
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 3,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed: 11,
                },
                vec![0.0, f32::NEG_INFINITY, -1000.0, f32::NEG_INFINITY],
            ),
        ] {
            let mut ordinary = sampler(config);
            let mut bounded = sampler(config);
            for _ in 0..64 {
                let expected = ordinary.sample(&logits);
                let actual = bounded
                    .sample_bounded_top_k(&logits)
                    .map(|(sampled, evidence)| {
                        assert!(evidence.used_bounded_path);
                        sampled
                    });
                assert_eq!(actual, expected);
                assert_eq!(bounded.draws(), ordinary.draws());

                let expected_next = ordinary.sample(&continuation);
                let actual_next =
                    bounded
                        .sample_bounded_top_k(&continuation)
                        .map(|(sampled, evidence)| {
                            assert!(evidence.used_bounded_path);
                            sampled
                        });
                assert_eq!(actual_next, expected_next);
                assert_eq!(bounded.draws(), ordinary.draws());
            }
        }

        for (len, top_k) in [(4, 0), (4, 4), (4, 5)] {
            let config = SamplingConfig {
                temperature: 0.7,
                top_k,
                top_p: 1.0,
                min_p: 0.05,
                seed: 42,
            };
            let logits = vec![0.0; len];
            let mut ordinary = sampler(config);
            let mut dispatched = sampler(config);
            let expected = ordinary.sample(&logits);
            let (actual, evidence) = dispatched.sample_bounded_top_k(&logits).unwrap();
            assert_eq!(Ok(actual), expected);
            assert!(!evidence.used_bounded_path);
            assert_eq!(ordinary.draws(), dispatched.draws());
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

    fn assert_profiled_equivalent(config: SamplingConfig, logits: &[f32]) {
        let mut ordinary = sampler(config);
        let mut profiled = ordinary.clone();
        let ordinary_result = ordinary.sample(logits);
        let profiled_result = profiled.sample_profiled(logits);
        match (ordinary_result, profiled_result) {
            (Ok(expected), Ok((actual, profile))) => {
                assert_eq!(actual, expected);
                if config.temperature > 0.0 {
                    assert_eq!(profile.timer_spans, 11);
                    assert_eq!(profile.input_logits, logits.len());
                    assert_eq!(profile.candidate_index, actual.candidate_index);
                    assert!(profile.candidate_capacity_bytes > 0);
                    assert!(profile.probability_capacity_bytes > 0);
                    assert!(profile.after_top_k >= profile.after_min_p);
                    assert!(profile.after_min_p >= profile.after_positive_infinity);
                    assert!(profile.after_positive_infinity >= profile.after_top_p);
                }
            }
            (Err(expected), Err(actual)) => assert_eq!(actual, expected),
            (expected, actual) => panic!(
                "ordinary/profiled sampler mismatch: ordinary={expected:?} profiled={actual:?}"
            ),
        }
        assert_eq!(ordinary.draws(), profiled.draws());

        let continuation = [2.0, 1.0, 0.0, -1.0];
        assert_eq!(
            ordinary.sample(&continuation),
            profiled.sample(&continuation),
            "profiled call changed subsequent RNG state"
        );
        assert_eq!(ordinary.draws(), profiled.draws());
    }

    #[test]
    fn profiled_sampler_preserves_version_one_results_errors_and_rng() {
        for seed in [0, 1, 42, u64::MAX] {
            assert_profiled_equivalent(
                SamplingConfig::qwen_chat(seed),
                &[3.0, 2.0, 2.0, 1.0, -0.0, 0.0, f32::NEG_INFINITY],
            );
            assert_profiled_equivalent(
                SamplingConfig {
                    temperature: 2.0,
                    top_k: 0,
                    top_p: 0.7,
                    min_p: 0.25,
                    seed,
                },
                &[f32::INFINITY, 1000.0, f32::INFINITY, -10.0],
            );
            assert_profiled_equivalent(
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 1,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed,
                },
                &[0.0, 3.0, 1.0],
            );
            assert_profiled_equivalent(
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                    seed,
                },
                &[f32::NEG_INFINITY, f32::NEG_INFINITY],
            );
            assert_profiled_equivalent(SamplingConfig::qwen_chat(seed), &[]);
            for nan_position in 0..3 {
                let mut logits = [2.0, 1.0, 0.0];
                logits[nan_position] = f32::NAN;
                assert_profiled_equivalent(SamplingConfig::qwen_chat(seed), &logits);
            }
        }

        assert_profiled_equivalent(SamplingConfig::default(), &[1.0, 3.0, 3.0]);
    }

    #[test]
    fn profiled_sampler_matches_version_one_golden_stream() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0x1234_5678_9abc_def0,
        };
        let logits = [2.0, 1.5, 1.0, 0.5];
        let expected = [0, 1, 1, 1, 3, 1, 3, 0, 0, 3, 0, 0, 2, 0, 2, 1];
        let mut ordinary = sampler(config);
        let mut profiled = sampler(config);
        for expected_token in expected {
            let ordinary_token = ordinary.sample(&logits).unwrap();
            let (profiled_token, profile) = profiled.sample_profiled(&logits).unwrap();
            assert_eq!(ordinary_token, profiled_token);
            assert_eq!(profiled_token.token, expected_token);
            assert_eq!(profile.after_top_p, logits.len());
        }
        assert_eq!(ordinary.draws(), expected.len());
        assert_eq!(profiled.draws(), expected.len());
    }

    #[test]
    fn profiled_sampler_matches_finite_top_p_and_tied_top_k_boundaries() {
        let finite_top_p = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.7,
            min_p: 0.0,
            seed: 19,
        };
        let mut ordinary = sampler(finite_top_p);
        let mut profiled = sampler(finite_top_p);
        for _ in 0..16 {
            let expected = ordinary.sample(&[3.0, 2.0, 1.0, 0.0]).unwrap();
            let (actual, profile) = profiled.sample_profiled(&[3.0, 2.0, 1.0, 0.0]).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(profile.after_top_p, 2);
        }

        let tied_top_k = SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            seed: 23,
        };
        let mut ordinary = sampler(tied_top_k);
        let mut profiled = sampler(tied_top_k);
        for _ in 0..16 {
            let expected = ordinary.sample(&[2.0, 2.0, 2.0]).unwrap();
            let (actual, profile) = profiled.sample_profiled(&[2.0, 2.0, 2.0]).unwrap();
            assert_eq!(actual, expected);
            assert!(matches!(actual.token, 0 | 1));
            assert_eq!(profile.after_top_k, 2);
        }
        assert_eq!(ordinary.draws(), profiled.draws());
    }
}

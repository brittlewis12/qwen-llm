//! CPU F64 summaries of the complete, unfiltered, temperature-one F32 logit row.

use super::{WorkspaceLensError, WorkspaceLensVocabularyScore};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceLensDistributionSummary {
    pub vocab_size: u32,
    pub entropy_nats: f64,
    pub logsumexp: f64,
    pub top_k_mass: f64,
    pub score_max: f64,
    pub score_mean: f64,
    pub score_variance_population: f64,
}

impl WorkspaceLensDistributionSummary {
    /// V2 admits one F64 ULP of JSON parsing drift in the F32-derived maximum.
    pub const VALIDATION_VERSION: u32 = 2;

    /// Validate serialized summaries without inventing statistics for absent rows.
    pub fn validate(
        &self,
        vocab_size: u32,
        top_logits: impl ExactSizeIterator<Item = f32> + Clone,
    ) -> Result<(), &'static str> {
        let k = top_logits.len();
        if vocab_size == 0 || self.vocab_size != vocab_size || k == 0 || k > vocab_size as usize {
            return Err("vocabulary or top-k size mismatch");
        }
        if ![
            self.entropy_nats,
            self.logsumexp,
            self.top_k_mass,
            self.score_max,
            self.score_mean,
            self.score_variance_population,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !top_logits.clone().all(f32::is_finite)
        {
            return Err("non-finite statistics or scores");
        }
        let selected_max = top_logits
            .clone()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        // score_max is an exact F32-to-F64 conversion before serialization, but
        // serde_json's default decimal parser can land on an adjacent F64 value.
        // Bound that representational drift, not an F32 ULP or an absolute epsilon.
        if self.score_max < selected_max.next_down() || self.score_max > selected_max.next_up() {
            return Err("score_max differs from selected F32 maximum by more than one F64 ULP");
        }
        let tolerance = 1e-10;
        let log_v = f64::from(vocab_size).ln();
        let score_tolerance = 8.0 * f64::EPSILON * self.score_max.abs().max(1.0);
        if self.entropy_nats < -tolerance
            || self.entropy_nats > log_v + tolerance
            || self.top_k_mass < k as f64 / f64::from(vocab_size) - tolerance
            || self.top_k_mass > 1.0 + tolerance
            || self.score_variance_population < 0.0
            || self.score_variance_population > f64::from(f32::MAX).powi(2) * (1.0 + tolerance)
            || self.score_mean < -f64::from(f32::MAX)
            || self.score_mean > self.score_max + score_tolerance
            || self.logsumexp < self.score_max - score_tolerance
            || self.logsumexp > self.score_max + log_v + score_tolerance
        {
            return Err("statistics outside full-vocabulary bounds");
        }
        // At huge offsets F64 cannot retain log(S) in logsumexp. Do not validate
        // mass using a subtraction that has already lost those low bits.
        if score_tolerance < 1e-6 {
            let mass = top_logits
                .map(|z| (f64::from(z) - self.score_max).exp())
                .sum::<f64>()
                * (-(self.logsumexp - self.score_max)).exp();
            if (mass - self.top_k_mass).abs() > tolerance + 4.0 * score_tolerance {
                return Err("top-k mass is inconsistent with normalization");
            }
        }
        Ok(())
    }
}

pub(super) fn restore_masked_distribution_row(
    row: &mut [f32],
    first_ids: &[i32],
    first_values: &[f32],
) -> Result<(), WorkspaceLensError> {
    let invalid = WorkspaceLensError::InvalidDistributionSummary;
    if first_ids.len() != super::MPS_FULL_READOUT_TOP_K || first_values.len() != first_ids.len() {
        return Err(invalid(
            "restoration requires exactly the first 16 IDs and values",
        ));
    }
    for (index, (&id, &value)) in first_ids.iter().zip(first_values).enumerate() {
        if id < 0
            || id as usize >= row.len()
            || !value.is_finite()
            || first_ids[..index].contains(&id)
        {
            return Err(invalid("invalid masked ID or saved value"));
        }
    }
    // Only the first MPS pass is masked. The second pass does not mutate logits.
    for (&id, &value) in first_ids.iter().zip(first_values) {
        if row[id as usize] != f32::NEG_INFINITY {
            return Err(invalid(
                "first-pass entry was not masked to negative infinity",
            ));
        }
        row[id as usize] = value;
    }
    if row.iter().any(|value| !value.is_finite()) {
        return Err(invalid("non-finite full-vocabulary tail after restoration"));
    }
    Ok(())
}

pub(super) fn summarize_distribution_row(
    row: &[f32],
    top: &[WorkspaceLensVocabularyScore],
) -> Result<WorkspaceLensDistributionSummary, WorkspaceLensError> {
    let invalid = WorkspaceLensError::InvalidDistributionSummary;
    let vocab_size = u32::try_from(row.len()).map_err(|_| invalid("vocabulary size overflow"))?;
    if row.is_empty() || row.iter().any(|v| !v.is_finite()) {
        return Err(invalid("empty or non-finite vocabulary row"));
    }
    for (index, score) in top.iter().enumerate() {
        if row.get(score.token_id as usize).map(|v| v.to_bits()) != Some(score.logit.to_bits())
            || top[..index]
                .iter()
                .any(|other| other.token_id == score.token_id)
        {
            return Err(invalid("top-k does not match full-vocabulary row"));
        }
    }
    let max = row
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / f64::from(vocab_size);
    let mut sum = 0.0;
    let mut weighted_delta = 0.0;
    let mut squared_deviations = 0.0;
    for &value in row {
        let value = f64::from(value);
        let delta = value - max;
        let weight = delta.exp();
        sum += weight;
        weighted_delta += weight * delta;
        squared_deviations += (value - mean).powi(2);
    }
    let summary = WorkspaceLensDistributionSummary {
        vocab_size,
        entropy_nats: sum.ln() - weighted_delta / sum,
        logsumexp: max + sum.ln(),
        top_k_mass: top
            .iter()
            .map(|score| (f64::from(score.logit) - max).exp())
            .sum::<f64>()
            / sum,
        score_max: max,
        score_mean: mean,
        score_variance_population: squared_deviations / f64::from(vocab_size),
    };
    summary
        .validate(vocab_size, top.iter().map(|score| score.logit))
        .map_err(invalid)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_serialized_real_cell_roundtrip_regression() {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Cell {
            source_layer: u32,
            source_position: usize,
            top_logits: Vec<f32>,
            distribution_summary: WorkspaceLensDistributionSummary,
        }
        let cell: Cell =
            serde_json::from_str(include_str!("distribution_roundtrip_fixture.json")).unwrap();
        assert_eq!((cell.source_layer, cell.source_position), (0, 5));
        let expected = f64::from(cell.top_logits[0]);
        assert_eq!(expected.to_bits(), 0x402f206340000000);
        assert_eq!("15.563257217407227".parse::<f64>().unwrap(), expected);
        // serde_json's default parser yields 0x402f206340000001 here, unlike
        // correctly rounded str::parse. Either parser must accept this artifact.
        assert!(
            cell.distribution_summary
                .score_max
                .to_bits()
                .abs_diff(expected.to_bits())
                <= 1
        );
        cell.distribution_summary
            .validate(248320, cell.top_logits.iter().copied())
            .unwrap();
    }

    #[test]
    fn distribution_max_roundtrip_boundary_rejects_tampering() {
        for value in [
            15.563257f32,
            -15.563257,
            0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MAX,
            -f32::MAX,
        ] {
            let summary = summarize(&[value; 64], 8);
            let max = f64::from(value);
            for admissible in [max, max.next_down(), max.next_up()] {
                let mut parsed = summary.clone();
                parsed.score_max = admissible;
                let before = parsed.clone();
                parsed.validate(64, [value; 8].into_iter()).unwrap();
                assert_eq!(
                    parsed, before,
                    "validation must not normalize stored statistics"
                );
            }
            for tampered in [
                max.next_down().next_down(),
                max.next_up().next_up(),
                f64::from(value.next_up()),
                f64::from(value.next_down()),
            ] {
                let mut parsed = summary.clone();
                parsed.score_max = tampered;
                assert!(
                    parsed.validate(64, [value; 8].into_iter()).is_err(),
                    "accepted {value} -> {tampered}"
                );
            }
        }
    }

    fn summarize(row: &[f32], k: usize) -> WorkspaceLensDistributionSummary {
        let mut top = row
            .iter()
            .enumerate()
            .map(|(id, &logit)| WorkspaceLensVocabularyScore {
                token_id: id as u32,
                logit,
            })
            .collect::<Vec<_>>();
        top.sort_by(|a, b| {
            b.logit
                .total_cmp(&a.logit)
                .then(a.token_id.cmp(&b.token_id))
        });
        summarize_distribution_row(row, &top[..k]).unwrap()
    }

    #[test]
    fn distribution_uniform_extreme_finite_and_shift() {
        for offset in [0.0, 1000.0, f32::MAX, -f32::MAX] {
            let s = summarize(&[offset; 64], 8);
            assert_eq!(s.entropy_nats, 64f64.ln());
            assert_eq!(s.top_k_mass, 0.125);
            assert_eq!(s.score_mean, f64::from(offset));
            assert_eq!(s.score_variance_population, 0.0);
        }
        let a = summarize(&[-2.0, 0.0, 0.0, 2.0], 2);
        let b = summarize(&[998.0, 1000.0, 1000.0, 1002.0], 2);
        assert_eq!(a.entropy_nats, b.entropy_nats);
        assert_eq!(a.top_k_mass, b.top_k_mass);
        assert_eq!(a.score_variance_population, b.score_variance_population);
        assert!((b.logsumexp - a.logsumexp - 1000.0).abs() < 1e-12);
        let extreme = summarize(&[-f32::MAX, f32::MAX], 1);
        assert_eq!(extreme.entropy_nats, 0.0);
        assert_eq!(extreme.top_k_mass, 1.0);
        assert!(extreme.score_variance_population.is_finite());
    }

    #[test]
    fn distribution_restores_exactly_first_pass_and_rejects_bad_tail() {
        let original = (0..64).map(|v| v as f32 - 32.0).collect::<Vec<_>>();
        let ids = (48..64).collect::<Vec<i32>>();
        let values = original[48..].to_vec();
        let mut masked = original.clone();
        masked[48..].fill(f32::NEG_INFINITY);
        restore_masked_distribution_row(&mut masked, &ids, &values).unwrap();
        assert_eq!(masked, original);
        assert_eq!(summarize(&masked, 8), summarize(&original, 8));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            masked[48..].fill(f32::NEG_INFINITY);
            masked[0] = bad;
            assert!(restore_masked_distribution_row(&mut masked, &ids, &values).is_err());
        }
        assert!(
            restore_masked_distribution_row(&mut original.clone(), &[48; 16], &values).is_err()
        );
        assert!(
            restore_masked_distribution_row(&mut original.clone(), &[64; 16], &values).is_err()
        );
        assert!(
            restore_masked_distribution_row(&mut original.clone(), &ids, &[f32::NAN; 16]).is_err()
        );
        assert!(restore_masked_distribution_row(&mut original.clone(), &ids, &values).is_err());
        let mut bit_values = values;
        bit_values[0] = -0.0;
        bit_values[1] = f32::from_bits(1);
        let mut row = original;
        row[48..].fill(f32::NEG_INFINITY);
        restore_masked_distribution_row(&mut row, &ids, &bit_values).unwrap();
        assert_eq!(row[48].to_bits(), (-0.0f32).to_bits());
        assert_eq!(row[49].to_bits(), 1);
    }

    #[test]
    fn distribution_ties_and_schema() {
        let s = summarize(&[1.0; 32], 16);
        assert_eq!(s.top_k_mass, 0.5);
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(
            serde_json::from_value::<WorkspaceLensDistributionSummary>(json.clone()).unwrap(),
            s
        );
        let mut extra = json.clone();
        extra["fake"] = 1.into();
        assert!(serde_json::from_value::<WorkspaceLensDistributionSummary>(extra).is_err());
        let mut missing = json;
        missing.as_object_mut().unwrap().remove("entropy_nats");
        assert!(serde_json::from_value::<WorkspaceLensDistributionSummary>(missing).is_err());
        assert!(s.validate(33, [1.0; 16].into_iter()).is_err());
        let mut bad = s;
        bad.entropy_nats = f64::INFINITY;
        assert!(bad.validate(32, [1.0; 16].into_iter()).is_err());
    }
}

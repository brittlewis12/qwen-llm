use super::super::{Sink, registry::Staged};
use super::*;
use crate::lens_intervention::normalize_direction;
use crate::serve::jobs::state::{Counters, Phase};
use qwen_llm::{
    metal::MetalTensor,
    runtime::{LoadedModel, Sequence},
    tensor::GgmlType,
};
use serde_json::json;

#[derive(Default)]
pub(crate) struct PreparedDirections {
    pub(crate) rows: BTreeMap<(String, u32), MetalTensor>,
}

impl Plan {
    pub(crate) fn memory_bytes(&self, loaded: &LoadedModel) -> Result<(u64, u64)> {
        if self.rows.is_empty() {
            return Ok((0, 0));
        }
        let hidden = u64::from(loaded.arch().hidden_size);
        self.price(hidden, |bytes| -> Result<u64> {
            Ok(loaded.context().shared_buffer_size_and_align(bytes)?.size)
        })
    }

    pub(super) fn price(
        &self,
        hidden: u64,
        mut price: impl FnMut(u64) -> Result<u64>,
    ) -> Result<(u64, u64)> {
        if self.rows.is_empty() {
            return Ok((0, 0));
        }
        let row = hidden
            .checked_mul(4)
            .context("direction row bytes overflow")?;
        let matrix = hidden
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .context("direction matrix bytes overflow")?;
        let retained = price(row)?
            .checked_mul(self.direction_rows as u64)
            .context("direction GPU memory overflow")?;
        let mut temporary = 0u64;
        // Sum selected-row/ID and projection scratch even though their GPU
        // lifetimes partly separate; retained directions overlap both phases.
        for bytes in [matrix, row, row, row, 4] {
            temporary = temporary
                .checked_add(price(bytes)?)
                .context("projection GPU memory overflow")?;
        }
        Ok((
            retained
                .checked_add(temporary)
                .context("direction memory overflow")?,
            row.checked_mul(8)
                .and_then(|n| n.checked_add(4 * 1024 * 1024))
                .context("projection CPU memory overflow")?,
        ))
    }

    pub(crate) fn prepare(
        &self,
        loaded: &LoadedModel,
        sequence: &mut Sequence,
        staged: &Staged,
        sink: &Sink,
        prompt_tokens: usize,
    ) -> Result<PreparedDirections> {
        let mut prepared = PreparedDirections::default();
        if self.rows.is_empty() {
            return Ok(prepared);
        }
        let hidden = loaded.arch().hidden_size as usize;
        let session = loaded.passive_workspace_lens_session(sequence)?;
        let staged_bytes = staged.matrices.values().try_fold(0usize, |total, m| {
            total
                .checked_add(m.capacity())
                .context("staged bytes overflow")
        })?;
        let reserve = staged_bytes
            .checked_add(self.direction_rows * hidden * 4)
            .and_then(|n| n.checked_add(super::super::writer::CPU_UPPER_BYTES as usize))
            .context("projection caller reserve overflow")?;
        ensure!(
            session.f16_transport_readout_query_capacity(reserve)? >= 1,
            "no direction projection capacity"
        );
        for (key, tokens) in &self.rows {
            super::super::checkpoint(sink)?;
            let matrix = staged
                .matrices
                .get(key)
                .context("missing staged direction matrix")?;
            let transport = session.prepare_f16_transport_readouts(matrix)?;
            for (&token, ids) in tokens {
                super::super::checkpoint(sink)?;
                let selected = session.selected_token_readouts(&[token])?;
                super::super::checkpoint(sink)?;
                ensure!(
                    selected.hidden_size == hidden && selected.token_ids == [token],
                    "selected covector identity changed"
                );
                let raw = session.project_prepared_f16_transport_readouts(&transport, &selected)?;
                super::super::checkpoint(sink)?;
                ensure!(raw.len() == hidden, "projected direction shape changed");
                #[cfg(test)]
                let (witness, reference) = projection_witness(matrix, &selected.values, &raw)?;
                for id in ids {
                    super::super::checkpoint(sink)?;
                    let definition = &self.directions[id];
                    let row = normalize_direction(raw.clone(), definition.normalization, id)?;
                    #[cfg(test)]
                    let normalized_witness = {
                        let norm = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let scale = if definition.normalization
                            == crate::lens_intervention::Normalization::UnitL2
                        {
                            norm
                        } else {
                            1.0
                        };
                        let mut max_abs = 0f64;
                        for (&expected, &actual) in reference.iter().zip(&row) {
                            let expected = expected / scale;
                            let delta = (expected - f64::from(actual)).abs();
                            ensure!(
                                delta <= 1e-3 + 1e-4 * expected.abs(),
                                "independent normalized direction mismatch"
                            );
                            max_abs = max_abs.max(delta);
                        }
                        json!({"basis":"normalized_cpu_projection_reference","max_abs_error":max_abs,"within_tolerance":true})
                    };
                    let bytes: Vec<_> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let record = json!({"kind":"direction_prepared","direction_id":id,"lens":key.alias,
                        "source_layer":key.layer,"token_id":token,"normalization":definition.normalization,
                        "target_covector":"deployed_logit_numerator","semantics":"transport_transpose_times_gamma_folded_lm_head_row_not_normalized_logit_gradient",
                        "values_blake3_f32le":blake3::hash(&bytes).to_hex().to_string(),
                        "norm_l2":row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt(),
                        "asset_identity":self.assets[&key.alias]["identity"],"binding_status":self.assets[&key.alias]["transfer"],
                        "target_layer":self.assets[&key.alias]["target_layer"],"method":self.assets[&key.alias]["method"]});
                    #[cfg(test)]
                    let record = {
                        let mut record = record;
                        record["test_projection_witness"] = witness.clone();
                        record["test_normalization_witness"] = normalized_witness;
                        record["test_direction_values"] = json!(row);
                        record
                    };
                    prepared.rows.insert(
                        (id.clone(), key.layer),
                        MetalTensor::from_bytes(
                            loaded.context(),
                            &bytes,
                            vec![hidden as u64],
                            GgmlType::F32,
                        )?,
                    );
                    sink.record(
                        record,
                        Phase::Prefill,
                        &Counters {
                            prompt_tokens: prompt_tokens as u64,
                            ..Default::default()
                        },
                    );
                    super::super::checkpoint(sink)?;
                }
            }
        }
        super::super::checkpoint(sink)?;
        Ok(prepared)
    }
}

#[cfg(test)]
fn projection_witness(
    matrix: &[u8],
    covector: &[f32],
    actual: &[f32],
) -> Result<(Value, Vec<f64>)> {
    let hidden = covector.len();
    ensure!(
        actual.len() == hidden && matrix.len() == hidden * hidden * 2,
        "projection witness shape"
    );
    let mut expected = vec![0f64; hidden];
    for (row, &c) in matrix.chunks_exact(hidden * 2).zip(covector) {
        for (sum, bytes) in expected.iter_mut().zip(row.chunks_exact(2)) {
            *sum +=
                f64::from(half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                    * f64::from(c);
        }
    }
    let mut max_abs = 0f64;
    let transport_effect_l2 = expected
        .iter()
        .zip(covector)
        .map(|(expected, &source)| (expected - f64::from(source)).powi(2))
        .sum::<f64>()
        .sqrt();
    let bypass_tolerance_ratio = expected
        .iter()
        .zip(covector)
        .map(|(expected, &source)| {
            (expected - f64::from(source)).abs() / (1e-3 + 1e-4 * expected.abs())
        })
        .fold(0f64, f64::max);
    for (&expected, &actual) in expected.iter().zip(actual) {
        let delta = (expected - f64::from(actual)).abs();
        ensure!(
            expected.is_finite() && actual.is_finite() && delta <= 1e-3 + 1e-4 * expected.abs(),
            "independent direction projection mismatch"
        );
        max_abs = max_abs.max(delta);
    }
    Ok((
        json!({"basis":"cpu_f64_f16_matrix_transpose_times_deployed_covector","max_abs_error":max_abs,"transport_effect_l2":transport_effect_l2,"bypass_tolerance_ratio":bypass_tolerance_ratio,
        "absolute_tolerance":1e-3,"relative_tolerance":1e-4,"within_tolerance":true}),
        expected,
    ))
}

#[test]
fn transpose_projection_oracle_rejects_bypassed_covectors() {
    let matrix: Vec<_> = [1., 2., 3., 4.]
        .into_iter()
        .flat_map(|x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    let (witness, _) = projection_witness(&matrix, &[1., 2.], &[7., 10.]).unwrap();
    assert!(witness["bypass_tolerance_ratio"].as_f64().unwrap() > 10.);
    assert!(projection_witness(&matrix, &[1., 2.], &[1., 2.]).is_err());
}

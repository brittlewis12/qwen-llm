//! Test-only independent F64 accumulation of the exact original-forward row.

use anyhow::{Result, ensure};
use serde_json::{Value, json};

pub(super) fn transport(matrix: &[u8], source: &[f32], actual: &[f32]) -> Result<Value> {
    let hidden = source.len();
    ensure!(
        hidden > 0
            && actual.len() == hidden
            && hidden.checked_mul(hidden).and_then(|n| n.checked_mul(2)) == Some(matrix.len()),
        "CPU transport witness shape mismatch"
    );
    let mut max_abs = 0.0f64;
    for (row, &got) in matrix.chunks_exact(hidden * 2).zip(actual) {
        let expected: f64 = row
            .chunks_exact(2)
            .zip(source)
            .map(|(bytes, &x)| {
                f64::from(half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                    * f64::from(x)
            })
            .sum();
        let delta = (f64::from(got) - expected).abs();
        ensure!(
            expected.is_finite() && got.is_finite() && delta <= 1e-3 + 1e-4 * expected.abs(),
            "CPU transport witness failed: expected {expected}, GPU {got}, absolute error {delta}"
        );
        max_abs = max_abs.max(delta);
    }
    Ok(
        json!({"basis":"cpu_f64_row_major_f16_matrix_times_original_forward_residual",
        "hidden_size":hidden,"absolute_tolerance":1e-3,"relative_tolerance":1e-4,
        "max_abs_error":max_abs,"within_tolerance":true}),
    )
}

#[test]
fn independent_oracle_detects_transposition_wrong_vectors_and_nonfinite_values() {
    let matrix: Vec<_> = [1., 2., 3., 4.]
        .into_iter()
        .flat_map(|x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    assert!(transport(&matrix, &[2., 3.], &[8., 18.]).is_ok());
    assert!(transport(&matrix, &[2., 3.], &[11., 16.]).is_err());
    assert!(transport(&matrix, &[2., 3.], &[f32::NAN, 18.]).is_err());
    assert!(transport(&matrix, &[2.], &[8.]).is_err());
}

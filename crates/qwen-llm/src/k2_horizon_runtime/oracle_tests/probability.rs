use super::*;

pub(super) fn metrics(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), reference.len());
    assert!(actual.iter().chain(reference).all(|v| v.is_finite()));
    let normalization = |values: &[f32]| {
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let log_sum = values
            .iter()
            .map(|&v| (f64::from(v) - maximum).exp())
            .sum::<f64>()
            .ln();
        (maximum, log_sum)
    };
    let (a_max, a_sum) = normalization(actual);
    let (r_max, r_sum) = normalization(reference);
    let mean_delta = actual
        .iter()
        .zip(reference)
        .map(|(&a, &r)| f64::from(a) - f64::from(r))
        .sum::<f64>()
        / actual.len() as f64;
    let mut kl = 0.0;
    let mut tv = 0.0;
    let mut centered_squared = 0.0;
    for (&a, &r) in actual.iter().zip(reference) {
        let log_q = f64::from(a) - a_max - a_sum;
        let log_p = f64::from(r) - r_max - r_sum;
        let (q, p) = (log_q.exp(), log_p.exp());
        kl += p * (log_p - log_q);
        tv += (p - q).abs();
        centered_squared += (f64::from(a) - f64::from(r) - mean_delta).powi(2);
    }
    let tv = tv * 0.5;
    let centered_rmse = (centered_squared / actual.len() as f64).sqrt();
    assert!(
        [kl, tv, centered_rmse, mean_delta]
            .iter()
            .all(|v| v.is_finite())
    );
    json!({"kl_reference_to_actual":kl,"total_variation":tv,
        "centered_rmse":centered_rmse,"mean_logit_delta":mean_delta})
}

#[test]
fn probability_metrics_are_shift_invariant_and_directional() {
    let shift = metrics(&[1001., 1002., 1003.], &[1., 2., 3.]);
    assert_eq!(shift["kl_reference_to_actual"], 0.0);
    assert_eq!(shift["total_variation"], 0.0);
    assert_eq!(shift["centered_rmse"], 0.0);
    assert_eq!(shift["mean_logit_delta"], 1000.0);
    let log_three = 3.0_f32.ln();
    let m = metrics(&[0., 0.], &[log_three, 0.]);
    let p = f64::from(log_three).exp() / (f64::from(log_three).exp() + 1.);
    let expected = p * (p * 2.).ln() + (1. - p) * ((1. - p) * 2.).ln();
    assert!((m["kl_reference_to_actual"].as_f64().unwrap() - expected).abs() < 1e-14);
    assert!((m["total_variation"].as_f64().unwrap() - (p - 0.5)).abs() < 1e-14);
    let reversed = metrics(&[log_three, 0.], &[0., 0.]);
    assert!(
        (m["kl_reference_to_actual"].as_f64().unwrap()
            - reversed["kl_reference_to_actual"].as_f64().unwrap())
        .abs()
            > 1e-3
    );
}

#[test]
fn probability_metrics_remain_finite_for_sharp_distributions_and_reject_bad_inputs() {
    let m = metrics(&[-10000., 10000.], &[10000., -10000.]);
    assert_eq!(m["kl_reference_to_actual"], 20000.0);
    assert_eq!(m["total_variation"], 1.0);
    for bad in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![0., 1.]] {
        assert!(std::panic::catch_unwind(|| metrics(&bad, &[0.])).is_err());
    }
}

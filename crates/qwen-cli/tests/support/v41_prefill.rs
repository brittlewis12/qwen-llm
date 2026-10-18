use anyhow::{Result, ensure};

// Grouped NativeF32 projection acceptance for this real-model witness only.
// At temperature 1 (default), a 1e-3 absolute logit cap limits any pairwise
// log-odds shift to 2e-3. TV <= 1e-4 targets 0.01 percentage points of probability
// mass; KL(expected || actual) <= 1e-8 and matching top1 are also required.
// This is not scalar-reduction equivalence or an approximate CED guarantee.
fn distribution_failures(max_abs: f64, tv: f64, kl: f64, top1_match: bool) -> Vec<String> {
    let mut failures = Vec::new();
    for (name, value, limit) in [
        ("max_abs", max_abs, 1e-3),
        ("tv", tv, 1e-4),
        ("kl_expected_to_actual", kl, 1e-8),
    ] {
        if !value.is_finite() || value < 0.0 || value > limit {
            failures.push(format!(
                "{name}={value:.9e} exceeds distribution contract [0, {limit}]"
            ));
        }
    }
    if !top1_match {
        failures.push("top1 mismatch".into());
    }
    failures
}

pub(super) fn compare(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    failures: &mut Vec<String>,
) -> Result<()> {
    ensure!(
        actual.len() == 129280 && actual.len() == expected.len(),
        "{label}: logits shape mismatch"
    );
    // Retain the synthetic NativeF32 elementwise bounds as characterization only.
    let (atol, rtol) = (3e-5_f64, 3e-5_f64);
    let (mut max_abs, mut squared_error, mut squared_reference) = (0.0_f64, 0.0, 0.0);
    let mut violations = 0;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && b.is_finite(), "{label}: nonfinite logits");
        let (a, b) = (f64::from(a), f64::from(b));
        let error = (a - b).abs();
        max_abs = max_abs.max(error);
        squared_error += error * error;
        squared_reference += b * b;
        violations += usize::from(error > atol + rtol * b.abs());
    }
    let top1 = |values: &[f32]| {
        values.iter().enumerate().fold(
            0,
            |best, (i, value)| {
                if *value > values[best] { i } else { best }
            },
        )
    };
    let relative_l2 = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    let mean_residual = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| f64::from(a) - f64::from(b))
        .sum::<f64>()
        / actual.len() as f64;
    let mut centered_max_abs = 0.0_f64;
    let mut centered_squared_error = 0.0;
    for (&a, &b) in actual.iter().zip(expected) {
        let residual = f64::from(a) - f64::from(b) - mean_residual;
        centered_max_abs = centered_max_abs.max(residual.abs());
        centered_squared_error += residual * residual;
    }
    let log_probs = |values: &[f32]| {
        let max = values
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = values
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
        values
            .iter()
            .map(|&v| (f64::from(v) - max) - log_sum)
            .collect::<Vec<_>>()
    };
    let actual_logp = log_probs(actual);
    let expected_logp = log_probs(expected);
    let (mut kl, mut tv) = (0.0, 0.0);
    for (&a, &b) in actual_logp.iter().zip(&expected_logp) {
        let p = b.exp();
        let delta = a - b;
        // Generalized KL cancels the first-order term before summation.
        kl += p * (delta.exp_m1() - delta);
        tv += (a.exp() - p).abs() * 0.5;
    }
    let centered_rms = (centered_squared_error / actual.len() as f64).sqrt();
    ensure!(
        [
            relative_l2,
            mean_residual,
            centered_max_abs,
            centered_rms,
            kl,
            tv
        ]
        .iter()
        .all(|v| v.is_finite()),
        "{label}: nonfinite metrics"
    );
    eprintln!(
        "{label}: temperature=1 max_abs={max_abs:.9e} relative_l2={relative_l2:.9e} rms={:.9e} mean_residual={mean_residual:.9e} centered_max_abs={centered_max_abs:.9e} centered_rms={centered_rms:.9e} kl_expected_to_actual={kl:.9e} tv={tv:.9e} top1={}/{} characterization_elementwise_violations={violations} atol={atol} rtol={rtol} acceptance_max_abs=1e-3 acceptance_tv=1e-4 acceptance_kl=1e-8",
        (squared_error / actual.len() as f64).sqrt(),
        top1(actual),
        top1(expected)
    );
    failures.extend(
        distribution_failures(max_abs, tv, kl, top1(actual) == top1(expected))
            .into_iter()
            .map(|failure| format!("{label}: {failure}")),
    );
    Ok(())
}

#[test]
fn distribution_contract_negative_controls() {
    assert!(distribution_failures(1e-3, 1e-4, 1e-8, true).is_empty());
    for (max_abs, tv, kl, top1_match, name) in [
        (1.001e-3, 0.0, 0.0, true, "max_abs"),
        (0.0, 1.001e-4, 0.0, true, "tv"),
        (0.0, 0.0, 1.001e-8, true, "kl_expected_to_actual"),
        (0.0, 0.0, 0.0, false, "top1"),
    ] {
        let failures = distribution_failures(max_abs, tv, kl, top1_match);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].starts_with(name));
    }
}

#[test]
fn comparison_shape_and_nonfinite_are_fatal() {
    let valid = vec![0.0; 129280];
    let mut failures = Vec::new();
    assert!(compare("empty", &[], &[], &mut failures).is_err());
    assert!(compare("wrong equal shape", &[0.0], &[0.0], &mut failures).is_err());
    assert!(compare("mismatch", &valid, &valid[..129279], &mut failures).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = valid.clone();
        bad[23] = value;
        assert!(compare("actual nonfinite", &bad, &valid, &mut failures).is_err());
        assert!(compare("expected nonfinite", &valid, &bad, &mut failures).is_err());
    }
    assert!(failures.is_empty());
}

#[test]
fn comparison_detects_distribution_drift() -> Result<()> {
    let mut expected = vec![-1000.0; 129280];
    expected[..2].fill(0.0);
    for (delta, names) in [
        (0.0003, vec!["kl_expected_to_actual"]),
        (0.0005, vec!["tv", "kl_expected_to_actual"]),
    ] {
        let mut actual = expected.clone();
        actual[0] = delta;
        let mut failures = Vec::new();
        compare("two-way distribution", &actual, &expected, &mut failures)?;
        assert_eq!(failures.len(), names.len(), "{failures:?}");
        for name in names {
            assert!(failures.iter().any(|failure| failure.contains(name)));
        }
    }
    Ok(())
}

#[test]
fn comparison_characterizes_shift_and_accumulates_failures() -> Result<()> {
    let expected = vec![0.0; 129280];
    let mut failures = Vec::new();
    compare("identical", &expected, &expected, &mut failures)?;
    // A uniform shift exceeds the old elementwise bound without changing softmax.
    compare(
        "uniform shift",
        &vec![0.0005; 129280],
        &expected,
        &mut failures,
    )?;
    assert!(failures.is_empty());
    compare("logit cap", &vec![0.002; 129280], &expected, &mut failures)?;
    assert_eq!(failures.len(), 1);
    assert!(failures[0].contains("max_abs"));
    let mut changed_top1 = expected.clone();
    changed_top1[1] = 1e-5;
    compare("top1", &changed_top1, &expected, &mut failures)?;
    assert_eq!(failures.len(), 2);
    assert!(failures[1].contains("top1"));
    Ok(())
}

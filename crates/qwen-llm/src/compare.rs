//! Checked elementwise comparison metrics shared by tests and diagnostics.

/// Largest absolute difference for equal, nonempty, finite f32 inputs.
///
/// Panics with `label` when either side cannot be compared or a difference is
/// non-finite.
pub fn assert_max_abs_diff_f32<'a, 'b, A, E>(label: &str, actual: A, expected: E) -> f32
where
    A: IntoIterator<Item = &'a f32>,
    E: IntoIterator<Item = &'b f32>,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f32;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                assert!(actual.is_finite(), "{label}: non-finite actual");
                assert!(expected.is_finite(), "{label}: non-finite expected");
                let difference = (*actual - *expected).abs();
                assert!(difference.is_finite(), "{label}: non-finite difference");
                maximum = maximum.max(difference);
                count += 1;
            }
            (None, None) => break,
            _ => panic!("{label}: length mismatch"),
        }
    }
    assert!(count > 0, "{label}: empty comparison");
    maximum
}

/// Largest absolute difference for equal, nonempty, finite f64 inputs.
///
/// Panics with `label` when either side cannot be compared or a difference is
/// non-finite.
pub fn assert_max_abs_diff_f64<'a, 'b, A, E>(label: &str, actual: A, expected: E) -> f64
where
    A: IntoIterator<Item = &'a f64>,
    E: IntoIterator<Item = &'b f64>,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f64;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                assert!(actual.is_finite(), "{label}: non-finite actual");
                assert!(expected.is_finite(), "{label}: non-finite expected");
                let difference = (*actual - *expected).abs();
                assert!(difference.is_finite(), "{label}: non-finite difference");
                maximum = maximum.max(difference);
                count += 1;
            }
            (None, None) => break,
            _ => panic!("{label}: length mismatch"),
        }
    }
    assert!(count > 0, "{label}: empty comparison");
    maximum
}

/// Largest absolute difference for report-only f32 comparisons.
///
/// Returns NaN for unequal, empty, non-finite inputs or non-finite differences.
pub fn report_max_abs_diff_f32<'a, 'b, A, E>(actual: A, expected: E) -> f32
where
    A: IntoIterator<Item = &'a f32>,
    E: IntoIterator<Item = &'b f32>,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f32;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                if !actual.is_finite() || !expected.is_finite() {
                    return f32::NAN;
                }
                let difference = (*actual - *expected).abs();
                if !difference.is_finite() {
                    return f32::NAN;
                }
                maximum = maximum.max(difference);
                count += 1;
            }
            (None, None) => break,
            _ => return f32::NAN,
        }
    }
    if count == 0 { f32::NAN } else { maximum }
}

/// Largest absolute difference for report-only f64 comparisons.
///
/// Returns NaN for unequal, empty, non-finite inputs or non-finite differences.
pub fn report_max_abs_diff_f64<'a, 'b, A, E>(actual: A, expected: E) -> f64
where
    A: IntoIterator<Item = &'a f64>,
    E: IntoIterator<Item = &'b f64>,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f64;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                if !actual.is_finite() || !expected.is_finite() {
                    return f64::NAN;
                }
                let difference = (*actual - *expected).abs();
                if !difference.is_finite() {
                    return f64::NAN;
                }
                maximum = maximum.max(difference);
                count += 1;
            }
            (None, None) => break,
            _ => return f64::NAN,
        }
    }
    if count == 0 { f64::NAN } else { maximum }
}

#[cfg(test)]
mod tests {
    use super::{
        assert_max_abs_diff_f32, assert_max_abs_diff_f64, report_max_abs_diff_f32,
        report_max_abs_diff_f64,
    };

    #[test]
    fn assert_helpers_report_largest_finite_difference() {
        assert_eq!(
            assert_max_abs_diff_f32("f32", [1.0, -2.0].iter(), [1.0, -1.5].iter()),
            0.5
        );
        assert_eq!(
            assert_max_abs_diff_f64("f64", [1.0, -2.0].iter(), [1.0, -1.5].iter()),
            0.5
        );
    }

    #[test]
    #[should_panic(expected = "length: length mismatch")]
    fn assert_helper_rejects_unequal_lengths() {
        assert_max_abs_diff_f32("length", [1.0].iter(), [1.0, 2.0].iter());
    }

    #[test]
    #[should_panic(expected = "empty: empty comparison")]
    fn assert_helper_rejects_empty_inputs() {
        assert_max_abs_diff_f32("empty", [].iter(), [].iter());
    }

    #[test]
    #[should_panic(expected = "nan: non-finite actual")]
    fn assert_helper_rejects_nan() {
        assert_max_abs_diff_f32("nan", [f32::NAN].iter(), [0.0].iter());
    }

    #[test]
    #[should_panic(expected = "inf: non-finite expected")]
    fn assert_helper_rejects_infinity() {
        assert_max_abs_diff_f64("inf", [0.0].iter(), [f64::INFINITY].iter());
    }

    #[test]
    fn report_helpers_return_nan_for_uncomparable_inputs() {
        assert_eq!(
            report_max_abs_diff_f32([1.0, -2.0].iter(), [1.0, -1.5].iter()),
            0.5
        );
        assert_eq!(report_max_abs_diff_f64([1.0].iter(), [0.5].iter()), 0.5);
        assert!(report_max_abs_diff_f32([1.0].iter(), [1.0, 2.0].iter()).is_nan());
        assert!(report_max_abs_diff_f64([].iter(), [].iter()).is_nan());
        assert!(report_max_abs_diff_f32([f32::NAN].iter(), [0.0].iter()).is_nan());
        assert!(report_max_abs_diff_f64([f64::INFINITY].iter(), [0.0].iter()).is_nan());
        assert!(report_max_abs_diff_f32([f32::MAX].iter(), [-f32::MAX].iter()).is_nan());
    }
}

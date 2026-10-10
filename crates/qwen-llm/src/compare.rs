//! Checked elementwise comparison metrics shared by tests and diagnostics.

/// Numeric inputs accepted by the f64 comparison functions.
#[doc(hidden)]
pub trait CompareValueF64 {
    fn comparison_value_f64(self) -> f64;
}

/// Numeric inputs accepted by the f32 comparison functions.
#[doc(hidden)]
pub trait CompareValueF32 {
    fn comparison_value_f32(self) -> f32;
}

impl CompareValueF32 for f32 {
    fn comparison_value_f32(self) -> f32 {
        self
    }
}

impl CompareValueF32 for &f32 {
    fn comparison_value_f32(self) -> f32 {
        *self
    }
}

impl CompareValueF64 for f32 {
    fn comparison_value_f64(self) -> f64 {
        f64::from(self)
    }
}

impl CompareValueF64 for f64 {
    fn comparison_value_f64(self) -> f64 {
        self
    }
}

impl CompareValueF64 for &f32 {
    fn comparison_value_f64(self) -> f64 {
        f64::from(*self)
    }
}

impl CompareValueF64 for &f64 {
    fn comparison_value_f64(self) -> f64 {
        *self
    }
}

/// Largest absolute difference for equal, nonempty, finite f32 inputs.
///
/// Panics with `label` when either side cannot be compared or a difference is
/// non-finite.
pub fn assert_max_abs_diff_f32<A, E>(label: &str, actual: A, expected: E) -> f32
where
    A: IntoIterator,
    A::Item: CompareValueF32,
    E: IntoIterator,
    E::Item: CompareValueF32,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f32;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                let actual = actual.comparison_value_f32();
                let expected = expected.comparison_value_f32();
                assert!(actual.is_finite(), "{label}: non-finite actual");
                assert!(expected.is_finite(), "{label}: non-finite expected");
                let difference = (actual - expected).abs();
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
pub fn assert_max_abs_diff_f64<A, E>(label: &str, actual: A, expected: E) -> f64
where
    A: IntoIterator,
    A::Item: CompareValueF64,
    E: IntoIterator,
    E::Item: CompareValueF64,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f64;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                let actual = actual.comparison_value_f64();
                let expected = expected.comparison_value_f64();
                assert!(actual.is_finite(), "{label}: non-finite actual");
                assert!(expected.is_finite(), "{label}: non-finite expected");
                let difference = (actual - expected).abs();
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
pub fn report_max_abs_diff_f32<A, E>(actual: A, expected: E) -> f32
where
    A: IntoIterator,
    A::Item: CompareValueF32,
    E: IntoIterator,
    E::Item: CompareValueF32,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f32;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                let actual = actual.comparison_value_f32();
                let expected = expected.comparison_value_f32();
                if !actual.is_finite() || !expected.is_finite() {
                    return f32::NAN;
                }
                let difference = (actual - expected).abs();
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

/// Largest absolute difference from a report-only stream of already paired f32 values.
///
/// Returns NaN for an empty stream, non-finite values or non-finite differences.
pub fn report_max_abs_pairs_f32<I>(pairs: I) -> f32
where
    I: IntoIterator<Item = (f32, f32)>,
{
    let mut maximum = 0.0f32;
    let mut count = 0usize;
    for (actual, expected) in pairs {
        if !actual.is_finite() || !expected.is_finite() {
            return f32::NAN;
        }
        let difference = (actual - expected).abs();
        if !difference.is_finite() {
            return f32::NAN;
        }
        maximum = maximum.max(difference);
        count += 1;
    }
    if count == 0 { f32::NAN } else { maximum }
}

/// Largest absolute difference for report-only f64 comparisons.
///
/// Returns NaN for unequal, empty, non-finite inputs or non-finite differences.
pub fn report_max_abs_diff_f64<A, E>(actual: A, expected: E) -> f64
where
    A: IntoIterator,
    A::Item: CompareValueF64,
    E: IntoIterator,
    E::Item: CompareValueF64,
{
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f64;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                let actual = actual.comparison_value_f64();
                let expected = expected.comparison_value_f64();
                if !actual.is_finite() || !expected.is_finite() {
                    return f64::NAN;
                }
                let difference = (actual - expected).abs();
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

/// Largest report-only relative difference, using `floor + scale * |expected|`.
///
/// Returns NaN for empty, unequal, non-finite or otherwise invalid comparisons.
pub fn report_max_relative_diff_f64<A, E>(actual: A, expected: E, floor: f64, scale: f64) -> f64
where
    A: IntoIterator,
    A::Item: CompareValueF64,
    E: IntoIterator,
    E::Item: CompareValueF64,
{
    if !floor.is_finite() || !scale.is_finite() || floor < 0.0 || scale < 0.0 {
        return f64::NAN;
    }
    let mut actual = actual.into_iter();
    let mut expected = expected.into_iter();
    let mut maximum = 0.0f64;
    let mut count = 0usize;
    loop {
        match (actual.next(), expected.next()) {
            (Some(actual), Some(expected)) => {
                let actual = actual.comparison_value_f64();
                let expected = expected.comparison_value_f64();
                if !actual.is_finite() || !expected.is_finite() {
                    return f64::NAN;
                }
                let denominator = floor + scale * expected.abs();
                if !denominator.is_finite() || denominator <= 0.0 {
                    return f64::NAN;
                }
                let relative = (actual - expected).abs() / denominator;
                if !relative.is_finite() {
                    return f64::NAN;
                }
                maximum = maximum.max(relative);
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
        report_max_abs_diff_f64, report_max_abs_pairs_f32, report_max_relative_diff_f64,
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
        assert!(
            report_max_abs_diff_f64(std::iter::empty::<&f64>(), std::iter::empty::<&f64>())
                .is_nan()
        );
        assert!(report_max_abs_diff_f32([f32::NAN].iter(), [0.0].iter()).is_nan());
        assert!(report_max_abs_diff_f64([f64::INFINITY].iter(), [0.0].iter()).is_nan());
        assert!(report_max_abs_diff_f32([f32::MAX].iter(), [-f32::MAX].iter()).is_nan());
        assert!(report_max_abs_pairs_f32([]).is_nan());
        assert!(
            (report_max_relative_diff_f64([1.1f64].iter(), [1.0f64].iter(), 0.0, 0.1) - 1.0).abs()
                < 1e-12
        );
        assert!(
            report_max_relative_diff_f64([1.0].iter(), std::iter::empty::<&f64>(), 1e-3, 1e-4)
                .is_nan()
        );
    }
}

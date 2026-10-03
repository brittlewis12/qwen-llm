//! Preserve zero-control intent when authored numbers narrow to native f32.

use serde::{Deserialize, Deserializer, de::Error};

fn checked(value: f32, spelling: &str) -> Result<f32, &'static str> {
    let nonzero = spelling
        .split(['e', 'E'])
        .next()
        .unwrap_or("")
        .bytes()
        .any(|byte| (b'1'..=b'9').contains(&byte));
    if !value.is_finite() || (nonzero && value == 0.0) {
        return Err("coefficient overflows or underflows native f32");
    }
    Ok(value)
}

pub(crate) fn parse_cli(spelling: &str) -> Result<f32, String> {
    let value = spelling.parse::<f32>().map_err(|error| error.to_string())?;
    checked(value, spelling).map_err(str::to_owned)
}

fn number<'de, D: Deserializer<'de>>(deserializer: D) -> Result<serde_json::Number, D::Error> {
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(number) => Ok(number),
        _ => Err(D::Error::custom("coefficient must be a number")),
    }
}

pub(crate) fn deserialize_action<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<f32, D::Error> {
    let number = number(deserializer)?;
    // Internally tagged Action fields buffer integer/f64 visits, unlike plain
    // manifest scalars. Do not double-round integers through f64. The spelling
    // still identifies nonzero input that underflows even at f64.
    let value = if let Some(value) = number.as_u64() {
        value as f32
    } else if let Some(value) = number.as_i64() {
        value as f32
    } else {
        number
            .as_f64()
            .ok_or_else(|| D::Error::custom("coefficient outside numeric range"))? as f32
    };
    checked(value, &number.to_string()).map_err(D::Error::custom)
}

pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    let number = number(deserializer)?;
    let spelling = number.to_string();
    // Unlike internally tagged Actions, manifest scalars narrow directly from
    // arbitrary-precision Number to f32. Keep that rounding behavior distinct.
    let value = serde_json::from_value::<f32>(serde_json::Value::Number(number))
        .map_err(D::Error::custom)?;
    checked(value, &spelling).map_err(D::Error::custom)
}

pub(crate) fn deserialize_many<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<f32>, D::Error> {
    #[derive(Deserialize)]
    struct Coefficient(#[serde(deserialize_with = "deserialize")] f32);
    Ok(Vec::<Coefficient>::deserialize(deserializer)?
        .into_iter()
        .map(|value| value.0)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens_intervention::Action;

    fn actions(coefficient: &str) -> Vec<String> {
        [
            "fixed_add",
            "residual_l2_fraction",
            "projection_ablate",
            "source_to_target",
            "coordinate_swap",
        ]
        .into_iter()
        .map(|kind| {
            let directions = if matches!(kind, "source_to_target" | "coordinate_swap") {
                r#""source":"a","target":"b""#
            } else {
                r#""direction":"a""#
            };
            format!(r#"{{"kind":"{kind}",{directions},"coefficient":{coefficient}}}"#)
        })
        .collect()
    }

    #[test]
    fn every_action_refuses_nonzero_underflow_overflow_and_nonnumbers() {
        for number in [
            "1e-50", "-1e-50", "1e-500", "-1e-500", "7e-46", "-7e-46", "1e40", "1e500", "null",
            "true", "\"1\"", "{}",
        ] {
            for text in actions(number) {
                assert!(serde_json::from_str::<Action>(&text).is_err(), "{text}");
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert!(serde_json::from_value::<Action>(value).is_err(), "{text}");
            }
        }
    }

    #[test]
    fn every_action_preserves_signed_zero_subnormals_and_json_rounding() {
        for number in [
            "0",
            "-0.0",
            "0e-999",
            "-0E+999",
            "1e-40",
            "8e-46",
            "-8e-46",
            "3.4028234e38",
            "0.25",
            "-1",
            "1.0000000596046448",
            "1.0000000596046447753906251",
        ] {
            let expected = (number.parse::<f64>().unwrap() as f32).to_bits();
            for text in actions(number) {
                let direct: Action = serde_json::from_str(&text).unwrap();
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                let buffered: Action = serde_json::from_value(value).unwrap();
                assert_eq!(direct.coefficient().to_bits(), expected, "{text}");
                assert_eq!(buffered.coefficient().to_bits(), expected, "{text}");
                let roundtrip: Action =
                    serde_json::from_str(&serde_json::to_string(&buffered).unwrap()).unwrap();
                assert_eq!(roundtrip.coefficient().to_bits(), expected);
            }
        }
        let halfway = "1.0000000596046448";
        assert_ne!(
            halfway.parse::<f32>().unwrap().to_bits(),
            (halfway.parse::<f64>().unwrap() as f32).to_bits()
        );
        assert_eq!(
            parse_cli(halfway).unwrap().to_bits(),
            halfway.parse::<f32>().unwrap().to_bits()
        );
    }

    #[test]
    fn action_narrowing_matches_legacy_serde_for_integer_and_fractional_numbers() {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum LegacyAction {
            FixedAdd { coefficient: f32 },
        }
        for spelling in [
            "0",
            "-0",
            "-0.0",
            "1",
            "9223372586610589697",
            "-4611686293305294849",
            "18446744073709551615",
            "1.0000000596046448",
            "1e-40",
        ] {
            let text = &actions(spelling)[0];
            let value: serde_json::Value = serde_json::from_str(text).unwrap();
            let LegacyAction::FixedAdd {
                coefficient: expected,
            } = serde_json::from_value(value.clone()).unwrap();
            let action: Action = serde_json::from_value(value).unwrap();
            assert_eq!(
                action.coefficient().to_bits(),
                expected.to_bits(),
                "{spelling}"
            );
        }
    }
}

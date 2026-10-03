//! Assertions about advertised identities, not model selection or content authentication.

use super::super::lens_http::{ApiError, input::Request};
use std::collections::BTreeSet;

pub(crate) fn check<'a>(
    request: &Request,
    model_identity: &str,
    asset_identity: impl Fn(&str) -> Option<&'a str>,
) -> Result<(), ApiError> {
    let Some(expected) = &request.preconditions else {
        return Ok(());
    };
    let mut aliases = BTreeSet::new();
    if let Some(diagnostics) = &request.diagnostics {
        for value in diagnostics.directions.iter().chain(&diagnostics.readouts) {
            aliases.insert(
                value
                    .get("lens")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        ApiError::new(
                            400,
                            "invalid_preconditions",
                            "Every referenced direction/readout must declare its lens alias.",
                        )
                    })?,
            );
        }
    }
    if aliases
        != expected
            .asset_identities
            .keys()
            .map(String::as_str)
            .collect()
    {
        return Err(ApiError::new(
            400,
            "invalid_preconditions",
            "Asset preconditions must cover exactly the aliases in directions and readouts, including zero controls and unused directions.",
        ));
    }
    if expected.model_identity != model_identity {
        return Err(ApiError::new(
            412,
            "binding_mismatch",
            "The advertised model metadata identity changed. Review the current deployment before explicitly rebinding new work.",
        ));
    }
    for (alias, identity) in &expected.asset_identities {
        if asset_identity(alias) != Some(identity.as_str()) {
            return Err(ApiError::new(
                412,
                "binding_mismatch",
                format!(
                    "The registered identity for alias {alias:?} changed or is unavailable. No job was accepted."
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn request() -> serde_json::Value {
        let mut value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/lens_http_v1/request.json"
        ))
        .unwrap();
        value["preconditions"] =
            json!({"model_identity":"model-a","asset_identities":{"plain":"model-a"}});
        value
    }
    #[test]
    fn exact_coverage_and_identity_are_checked_including_unused_and_zero_directions() {
        let mut value = request();
        assert!(
            check(&Request::parse(&value).unwrap(), "model-a", |_| Some(
                "model-a"
            ))
            .is_ok()
        );
        assert_eq!(
            check(&Request::parse(&value).unwrap(), "model-b", |_| Some(
                "model-a"
            ))
            .err()
            .unwrap()
            .status,
            412
        );
        assert_eq!(
            check(&Request::parse(&value).unwrap(), "model-a", |_| None)
                .err()
                .unwrap()
                .status,
            412
        );
        value["diagnostics"]["operations"][0]["action"]["coefficient"] = 0.into();
        value["diagnostics"]["directions"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"unused","lens":"j"}));
        assert_eq!(
            check(&Request::parse(&value).unwrap(), "model-a", |_| Some(
                "model-a"
            ))
            .err()
            .unwrap()
            .status,
            400
        );
        value["preconditions"]["asset_identities"]["j"] = "fit-a".into();
        assert!(
            check(&Request::parse(&value).unwrap(), "model-a", |a| {
                if a == "j" {
                    Some("fit-a")
                } else {
                    Some("model-a")
                }
            })
            .is_ok()
        );
        value["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[]});
        assert_eq!(
            check(&Request::parse(&value).unwrap(), "model-a", |_| Some(
                "model-a"
            ))
            .err()
            .unwrap()
            .status,
            400
        );
        value["preconditions"]["asset_identities"] = json!({});
        assert!(check(&Request::parse(&value).unwrap(), "model-a", |_| None).is_ok());
        value.as_object_mut().unwrap().remove("preconditions");
        assert!(check(&Request::parse(&value).unwrap(), "model-b", |_| None).is_ok());
    }
    #[test]
    fn malformed_or_unbounded_preconditions_are_not_coerced() {
        for bad in [
            json!(null),
            json!({}),
            json!({"model_identity":"","asset_identities":{}}),
            json!({"model_identity":"m","asset_identities":{"j":false}}),
            json!({"model_identity":"m","asset_identities":{},"model_path":"x"}),
            json!({"model_identity":"m".repeat(257),"asset_identities":{}}),
            json!({"model_identity":"m","asset_identities":{"../j":"fit"}}),
        ] {
            let mut value = request();
            value["preconditions"] = bad;
            assert!(Request::parse(&value).is_err());
        }
    }
}

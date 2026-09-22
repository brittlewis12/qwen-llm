//! HF's tojson uses Python json.dumps(ensure_ascii=False), including spaces and
//! Python float exponent formatting. Ordinary compact serde JSON is not identical.
use super::*;

pub(super) fn encode(value: &Value) -> Result<String> {
    let mut out = String::new();
    write(value, &mut out)?;
    Ok(out)
}

fn write(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(key).map_err(|e| error(e.to_string()))?);
                out.push_str(": ");
                write(value, out)?;
            }
            out.push('}');
        }
        Value::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                let value = number.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                    error("tool JSON requires finite binary64 floating-point values")
                })?;
                let scientific = format!("{value:e}");
                let (mantissa, exponent) = scientific.split_once('e').unwrap();
                let exponent: i32 = exponent.parse().unwrap();
                if !(-4..16).contains(&exponent) {
                    out.push_str(mantissa);
                    out.push('e');
                    out.push(if exponent < 0 { '-' } else { '+' });
                    out.push_str(&format!("{:02}", exponent.abs()));
                } else {
                    let fixed = value.to_string();
                    out.push_str(&fixed);
                    if !fixed.contains('.') {
                        out.push_str(".0");
                    }
                }
            } else {
                out.push_str(&text);
            }
        }
        _ => out.push_str(&serde_json::to_string(value).map_err(|e| error(e.to_string()))?),
    }
    Ok(())
}

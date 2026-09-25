//! pmset power snapshots recorded beside bench rows.

use super::*;

/// Every engine lever in the environment, recorded on each row: `QWEN_*`
/// plus Flash-Next's `QWEN4EXP_*` rollbacks, which change execution.
pub(crate) fn capture_qwen_env() -> std::collections::BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("QWEN_") || k.starts_with("QWEN4EXP_"))
        .collect()
}

pub(crate) fn pmset_output(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("pmset")
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub(crate) fn parse_pmset_source(line: &str) -> Option<String> {
    let (_, rest) = line.split_once('\'')?;
    let (source, _) = rest.split_once('\'')?;
    Some(source.to_string())
}

pub(crate) fn parse_battery_percent(line: &str) -> Option<u8> {
    let pct_pos = line.find('%')?;
    let digits: String = line[..pct_pos]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse::<u8>().ok()
}

pub(crate) fn parse_battery_state(line: &str) -> Option<String> {
    let mut parts = line.split(';').map(str::trim);
    let _battery_and_percent = parts.next()?;
    parts.next().map(str::to_string).filter(|s| !s.is_empty())
}

pub(crate) fn parse_pmset_custom_powermodes(raw: &str) -> (Option<i32>, Option<i32>) {
    #[derive(Clone, Copy)]
    enum Section {
        Battery,
        Ac,
    }
    let mut section = None;
    let mut battery = None;
    let mut ac = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        match trimmed {
            "Battery Power:" => {
                section = Some(Section::Battery);
                continue;
            }
            "AC Power:" => {
                section = Some(Section::Ac);
                continue;
            }
            _ => {}
        }
        let mut fields = trimmed.split_whitespace();
        if fields.next() != Some("powermode") {
            continue;
        }
        let Some(value) = fields.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        match section {
            Some(Section::Battery) => battery = Some(value),
            Some(Section::Ac) => ac = Some(value),
            None => {}
        }
    }
    (battery, ac)
}

pub(crate) fn note_no_warning(raw: &str, needle: &str) -> Option<bool> {
    if raw.contains(needle) {
        Some(false)
    } else if raw.trim().is_empty() {
        None
    } else {
        Some(true)
    }
}

pub(crate) fn capture_power_snapshot() -> Option<PowerSnapshot> {
    let batt = pmset_output(&["-g", "batt"]);
    let therm = pmset_output(&["-g", "therm"]);
    let custom = pmset_output(&["-g", "custom"]);
    if batt.is_none() && therm.is_none() && custom.is_none() {
        return None;
    }

    let mut snap = PowerSnapshot::default();
    if let Some(raw) = &batt {
        for line in raw.lines() {
            if line.starts_with("Now drawing from") {
                snap.source = parse_pmset_source(line);
            } else if line.contains("InternalBattery") {
                snap.battery_percent = parse_battery_percent(line);
                snap.battery_state = parse_battery_state(line);
            } else if let Some((_, warning)) = line.split_once("Battery Warning:") {
                snap.battery_warning = Some(warning.trim().to_string());
            }
        }
    }
    if let Some(raw) = &custom {
        let (battery, ac) = parse_pmset_custom_powermodes(raw);
        snap.powermode_battery = battery;
        snap.powermode_ac = ac;
    }
    if let Some(raw) = &therm {
        snap.thermal_warning_recorded =
            note_no_warning(raw, "No thermal warning level has been recorded");
        snap.performance_warning_recorded =
            note_no_warning(raw, "No performance warning level has been recorded");
        snap.cpu_power_status_recorded =
            note_no_warning(raw, "No CPU power status has been recorded");
    }
    Some(snap)
}

pub(crate) fn power_snapshot_summary(power: Option<&PowerSnapshot>) -> String {
    let Some(p) = power else {
        return "unavailable".to_string();
    };
    let battery = match (p.battery_percent, p.battery_state.as_deref()) {
        (Some(percent), Some(state)) => format!("{percent}% {state}"),
        (Some(percent), None) => format!("{percent}%"),
        (None, Some(state)) => state.to_string(),
        (None, None) => "unknown".to_string(),
    };
    format!(
        "source={} battery={} warning={} powermode_ac={} powermode_battery={} thermal_warning={} performance_warning={}",
        p.source.as_deref().unwrap_or("unknown"),
        battery,
        p.battery_warning.as_deref().unwrap_or("none"),
        p.powermode_ac.map_or("?".to_string(), |v| v.to_string()),
        p.powermode_battery
            .map_or("?".to_string(), |v| v.to_string()),
        p.thermal_warning_recorded
            .map_or("?".to_string(), |v| v.to_string()),
        p.performance_warning_recorded
            .map_or("?".to_string(), |v| v.to_string()),
    )
}

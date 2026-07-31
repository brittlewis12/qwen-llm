use anyhow::{Context, Result, ensure};
use qwen_llm::pid_metrics::PidSnapshot;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;
use std::time::{Duration, Instant};

const CPU_LOAD_INFO: i32 = 3;
const CPU_STATE_USER: usize = 0;
const CPU_STATE_SYSTEM: usize = 1;
const CPU_STATE_IDLE: usize = 2;
const CPU_STATE_NICE: usize = 3;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RawCommand {
    argv: Vec<String>,
    status: i32,
    stdout: String,
    stderr: String,
    wall_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CpuIdleSample {
    idle_percent: f64,
    wall_ms: f64,
    user_ticks: u64,
    system_ticks: u64,
    idle_ticks: u64,
    nice_ticks: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProcessRecord {
    pid: i32,
    ppid: i32,
    command: String,
    arguments: String,
    ancestor: bool,
    competitor_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct HostSnapshot {
    label: String,
    ac_power: bool,
    power_source: String,
    thermal_warning: bool,
    performance_warning: bool,
    memory_available_percent: f64,
    cpu_idle_samples: Vec<CpuIdleSample>,
    cpu_idle_median_percent: f64,
    processes: Vec<ProcessRecord>,
    process_census_sha256: String,
    competitor_count: usize,
    pmset_batt: RawCommand,
    pmset_therm: RawCommand,
    memory_pressure: RawCommand,
    process_census: RawCommand,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VmCounters {
    pub(crate) swap_used_bytes: u64,
    pub(crate) pageouts: u64,
    pub(crate) compressions: u64,
    pub(crate) swapouts: u64,
    pub(crate) compressor_pages_stored: u64,
    pub(crate) compressor_pages_occupied: u64,
    pub(crate) process_pageins: u64,
    pub(crate) process_disk_bytes_read: u64,
    pub(crate) process_disk_bytes_written: u64,
    vm_stat: RawCommand,
    swapusage: RawCommand,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VmDelta {
    swap_used_bytes: i128,
    pageouts: u64,
    compressions: u64,
    swapouts: u64,
    compressor_pages_stored: i128,
    compressor_pages_occupied: i128,
    process_pageins: u64,
    process_disk_bytes_read: u64,
    process_disk_bytes_written: u64,
    fatal_growth: Vec<String>,
}

impl VmDelta {
    pub(crate) fn between(before: &VmCounters, after: &VmCounters) -> Self {
        let swap_used_bytes =
            i128::from(after.swap_used_bytes) - i128::from(before.swap_used_bytes);
        let compressor_pages_stored =
            i128::from(after.compressor_pages_stored) - i128::from(before.compressor_pages_stored);
        let compressor_pages_occupied = i128::from(after.compressor_pages_occupied)
            - i128::from(before.compressor_pages_occupied);
        let mut fatal_growth = Vec::new();
        if swap_used_bytes > 0 {
            fatal_growth.push("swap_used_bytes".to_string());
        }
        if compressor_pages_stored > 0 {
            fatal_growth.push("compressor_pages_stored".to_string());
        }
        if compressor_pages_occupied > 0 {
            fatal_growth.push("compressor_pages_occupied".to_string());
        }
        Self {
            swap_used_bytes,
            pageouts: after.pageouts.saturating_sub(before.pageouts),
            compressions: after.compressions.saturating_sub(before.compressions),
            swapouts: after.swapouts.saturating_sub(before.swapouts),
            compressor_pages_stored,
            compressor_pages_occupied,
            process_pageins: after.process_pageins.saturating_sub(before.process_pageins),
            process_disk_bytes_read: after
                .process_disk_bytes_read
                .saturating_sub(before.process_disk_bytes_read),
            process_disk_bytes_written: after
                .process_disk_bytes_written
                .saturating_sub(before.process_disk_bytes_written),
            fatal_growth,
        }
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.fatal_growth.is_empty()
    }
}

impl HostSnapshot {
    pub(crate) fn capture(label: impl Into<String>) -> Result<Self> {
        let label = label.into();
        let pmset_batt = run_command("/usr/bin/pmset", &["-g", "batt"])?;
        let pmset_therm = run_command("/usr/bin/pmset", &["-g", "therm"])?;
        let memory_pressure = run_command("/usr/bin/memory_pressure", &["-Q"])?;
        let power_source = parse_power_source(&pmset_batt.stdout)?;
        let ac_power = power_source == "AC Power";
        let thermal_warning = parse_warning_state(
            &pmset_therm.stdout,
            "No thermal warning level has been recorded",
        )?;
        let performance_warning = parse_warning_state(
            &pmset_therm.stdout,
            "No performance warning level has been recorded",
        )?;
        let memory_available_percent = parse_memory_available(&memory_pressure.stdout)?;

        let mut cpu_idle_samples = Vec::with_capacity(3);
        for _ in 0..3 {
            cpu_idle_samples.push(capture_cpu_idle(Duration::from_secs(1))?);
        }
        let mut idle_values: Vec<f64> = cpu_idle_samples
            .iter()
            .map(|sample| sample.idle_percent)
            .collect();
        idle_values.sort_by(f64::total_cmp);
        let cpu_idle_median_percent = idle_values[1];

        let process_census = run_command("/bin/ps", &["-axo", "pid=,ppid=,comm=,args="])?;
        let ancestors = ancestor_pids(&process_census.stdout, std::process::id() as i32)?;
        let processes = parse_processes(&process_census.stdout, &ancestors)?;
        let competitor_count = processes
            .iter()
            .filter(|process| process.competitor_reason.is_some())
            .count();
        let process_census_sha256 =
            format!("{:x}", Sha256::digest(process_census.stdout.as_bytes()));

        Ok(Self {
            label,
            ac_power,
            power_source,
            thermal_warning,
            performance_warning,
            memory_available_percent,
            cpu_idle_samples,
            cpu_idle_median_percent,
            processes,
            process_census_sha256,
            competitor_count,
            pmset_batt,
            pmset_therm,
            memory_pressure,
            process_census,
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(self.ac_power, "{}: machine is not on AC power", self.label);
        ensure!(
            !self.thermal_warning,
            "{}: thermal warning is active or recorded",
            self.label
        );
        ensure!(
            !self.performance_warning,
            "{}: performance warning is active or recorded",
            self.label
        );
        ensure!(
            self.memory_available_percent >= 50.0,
            "{}: memory availability {:.1}% is below 50%",
            self.label,
            self.memory_available_percent
        );
        ensure!(
            self.cpu_idle_median_percent >= 75.0,
            "{}: median CPU idle {:.2}% is below 75%",
            self.label,
            self.cpu_idle_median_percent
        );
        ensure!(
            self.competitor_count == 0,
            "{}: {} competing inference/GPU processes detected",
            self.label,
            self.competitor_count
        );
        Ok(())
    }
}

impl VmCounters {
    pub(crate) fn capture() -> Result<Self> {
        let vm_stat = run_command("/usr/bin/vm_stat", &[])?;
        let swapusage = run_command("/usr/sbin/sysctl", &["-n", "vm.swapusage"])?;
        let labels = parse_vm_stat(&vm_stat.stdout)?;
        let process = PidSnapshot::now().context("capture process counters")?;
        Ok(Self {
            swap_used_bytes: parse_swap_used_bytes(&swapusage.stdout)?,
            pageouts: required_counter(&labels, "Pageouts")?,
            compressions: required_counter(&labels, "Compressions")?,
            swapouts: required_counter(&labels, "Swapouts")?,
            compressor_pages_stored: required_counter(&labels, "Pages stored in compressor")?,
            compressor_pages_occupied: required_counter(&labels, "Pages occupied by compressor")?,
            process_pageins: process.pageins,
            process_disk_bytes_read: process.diskio_bytesread,
            process_disk_bytes_written: process.diskio_byteswritten,
            vm_stat,
            swapusage,
        })
    }
}

fn run_command(program: &str, args: &[&str]) -> Result<RawCommand> {
    let start = Instant::now();
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("run {program}"))?;
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    let status = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8(output.stdout)
        .with_context(|| format!("decode {program} stdout as UTF-8"))?;
    let stderr = String::from_utf8(output.stderr)
        .with_context(|| format!("decode {program} stderr as UTF-8"))?;
    ensure!(
        output.status.success(),
        "{program} failed with status {status}: {stderr}"
    );
    Ok(RawCommand {
        argv: std::iter::once(program.to_string())
            .chain(args.iter().map(|arg| (*arg).to_string()))
            .collect(),
        status,
        stdout,
        stderr,
        wall_ms,
    })
}

fn parse_power_source(raw: &str) -> Result<String> {
    let line = raw
        .lines()
        .find(|line| line.starts_with("Now drawing from"))
        .context("pmset output lacks power source")?;
    let (_, rest) = line.split_once('\'').context("power source lacks quote")?;
    let (source, _) = rest
        .split_once('\'')
        .context("power source lacks closing quote")?;
    Ok(source.to_string())
}

fn parse_warning_state(raw: &str, no_warning: &str) -> Result<bool> {
    ensure!(!raw.trim().is_empty(), "pmset thermal output is empty");
    Ok(!raw.contains(no_warning))
}

fn parse_memory_available(raw: &str) -> Result<f64> {
    let prefix = "System-wide memory free percentage:";
    let line = raw
        .lines()
        .find(|line| line.trim_start().starts_with(prefix))
        .context("memory_pressure output lacks free percentage")?;
    let value = line
        .trim_start()
        .strip_prefix(prefix)
        .context("memory percentage prefix mismatch")?
        .trim()
        .strip_suffix('%')
        .context("memory percentage lacks percent sign")?
        .trim()
        .parse::<f64>()
        .context("parse memory percentage")?;
    ensure!((0.0..=100.0).contains(&value), "invalid memory percentage");
    Ok(value)
}

fn parse_vm_stat(raw: &str) -> Result<BTreeMap<String, u64>> {
    let mut values = BTreeMap::new();
    for line in raw.lines().skip(1) {
        let Some((label, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_end_matches('.');
        if let Ok(value) = value.parse::<u64>() {
            values.insert(label.trim().to_string(), value);
        }
    }
    Ok(values)
}

fn required_counter(values: &BTreeMap<String, u64>, name: &str) -> Result<u64> {
    values
        .get(name)
        .copied()
        .with_context(|| format!("vm_stat lacks {name:?}"))
}

fn parse_swap_used_bytes(raw: &str) -> Result<u64> {
    let used = raw
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(3)
        .find_map(|window| (window[0] == "used" && window[1] == "=").then_some(window[2]))
        .context("vm.swapusage lacks used value")?;
    let split = used
        .find(|byte: char| !byte.is_ascii_digit() && byte != '.')
        .context("vm.swapusage used value lacks unit")?;
    let number = used[..split]
        .parse::<f64>()
        .context("parse vm.swapusage used number")?;
    let multiplier = match &used[split..] {
        "B" => 1.0,
        "K" => 1024.0,
        "M" => 1024.0 * 1024.0,
        "G" => 1024.0 * 1024.0 * 1024.0,
        "T" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        unit => anyhow::bail!("unsupported vm.swapusage unit {unit:?}"),
    };
    let bytes = number * multiplier;
    ensure!(bytes.is_finite() && bytes >= 0.0 && bytes <= u64::MAX as f64);
    Ok(bytes.round() as u64)
}

fn parse_process_rows(raw: &str) -> Result<Vec<(i32, i32, String, String)>> {
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields
                .next()
                .context("process row lacks pid")?
                .parse::<i32>()
                .context("parse process pid")?;
            let ppid = fields
                .next()
                .context("process row lacks ppid")?
                .parse::<i32>()
                .context("parse process ppid")?;
            let command = fields
                .next()
                .context("process row lacks command")?
                .to_string();
            let arguments = fields.collect::<Vec<_>>().join(" ");
            Ok((pid, ppid, command, arguments))
        })
        .collect()
}

fn ancestor_pids(raw: &str, current: i32) -> Result<BTreeSet<i32>> {
    let rows = parse_process_rows(raw)?;
    let parents: BTreeMap<i32, i32> = rows.iter().map(|row| (row.0, row.1)).collect();
    let mut ancestors = BTreeSet::new();
    let mut cursor = current;
    while cursor > 0 && ancestors.insert(cursor) {
        cursor = parents.get(&cursor).copied().unwrap_or(0);
    }
    Ok(ancestors)
}

fn parse_processes(raw: &str, ancestors: &BTreeSet<i32>) -> Result<Vec<ProcessRecord>> {
    parse_process_rows(raw)?
        .into_iter()
        .map(|(pid, ppid, command, arguments)| {
            let ancestor = ancestors.contains(&pid);
            let competitor_reason = if ancestor {
                None
            } else {
                classify_competitor(&command, &arguments)
            };
            Ok(ProcessRecord {
                pid,
                ppid,
                command,
                arguments,
                ancestor,
                competitor_reason,
            })
        })
        .collect()
}

fn classify_competitor(command: &str, arguments: &str) -> Option<String> {
    let basename = command
        .rsplit('/')
        .next()
        .unwrap_or(command)
        .to_ascii_lowercase();
    let args = arguments.to_ascii_lowercase();
    if basename.starts_with("qwen") || basename.starts_with("llama") {
        return Some("local_inference_binary".to_string());
    }
    if basename == "ollama" || args.contains("python -m mlx_lm") || args.contains("mlx_lm.") {
        return Some("known_inference_runtime".to_string());
    }
    let combined = format!("{basename} {args}");
    if (combined.contains("metal") || combined.contains(" gpu"))
        && (combined.contains("bench")
            || combined.contains("profile")
            || combined.contains("experiment"))
    {
        return Some("metal_or_gpu_benchmark".to_string());
    }
    None
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HostCpuLoadInfo {
    ticks: [u32; 4],
}

unsafe extern "C" {
    fn mach_host_self() -> u32;
    fn host_statistics(
        host_priv: u32,
        flavor: i32,
        host_info_out: *mut i32,
        host_info_out_count: *mut u32,
    ) -> i32;
}

fn cpu_ticks() -> Result<[u32; 4]> {
    let mut info = HostCpuLoadInfo { ticks: [0; 4] };
    let mut count = 4u32;
    let host = unsafe { mach_host_self() };
    let status = unsafe {
        host_statistics(
            host,
            CPU_LOAD_INFO,
            (&mut info as *mut HostCpuLoadInfo).cast::<i32>(),
            &mut count,
        )
    };
    ensure!(status == 0, "host_statistics failed with status {status}");
    ensure!(count == 4, "host_statistics returned {count} words");
    Ok(info.ticks)
}

fn capture_cpu_idle(duration: Duration) -> Result<CpuIdleSample> {
    let before = cpu_ticks()?;
    let start = Instant::now();
    std::thread::sleep(duration);
    let after = cpu_ticks()?;
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    let delta = std::array::from_fn::<u64, 4, _>(|index| {
        u64::from(after[index].wrapping_sub(before[index]))
    });
    let total: u64 = delta.iter().sum();
    ensure!(total > 0, "CPU tick sample is empty");
    Ok(CpuIdleSample {
        idle_percent: delta[CPU_STATE_IDLE] as f64 * 100.0 / total as f64,
        wall_ms,
        user_ticks: delta[CPU_STATE_USER],
        system_ticks: delta[CPU_STATE_SYSTEM],
        idle_ticks: delta[CPU_STATE_IDLE],
        nice_ticks: delta[CPU_STATE_NICE],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frozen_host_outputs() {
        assert_eq!(
            parse_power_source("Now drawing from 'AC Power'\n").unwrap(),
            "AC Power"
        );
        assert_eq!(
            parse_memory_available("System-wide memory free percentage: 76%\n").unwrap(),
            76.0
        );
        assert_eq!(
            parse_swap_used_bytes("total = 4096.00M  used = 12.50M  free = 4083.50M\n").unwrap(),
            13_107_200
        );
        let vm = parse_vm_stat(
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPageouts: 12.\nCompressions: 34.\n",
        )
        .unwrap();
        assert_eq!(vm["Pageouts"], 12);
        assert_eq!(vm["Compressions"], 34);
    }

    #[test]
    fn process_classifier_excludes_ancestors_and_finds_inference() {
        let raw = "1 0 /sbin/launchd /sbin/launchd\n100 1 /bin/zsh zsh\n200 100 qwen-bench qwen-bench integrated-grammar-row\n300 1 llama-cli llama-cli -m model\n";
        let ancestors = ancestor_pids(raw, 200).unwrap();
        let processes = parse_processes(raw, &ancestors).unwrap();
        assert!(
            processes
                .iter()
                .find(|row| row.pid == 200)
                .unwrap()
                .ancestor
        );
        assert_eq!(
            processes
                .iter()
                .find(|row| row.pid == 300)
                .unwrap()
                .competitor_reason
                .as_deref(),
            Some("local_inference_binary")
        );
    }

    #[test]
    fn vm_delta_only_fails_on_occupied_state_growth() {
        fn counters(swap: u64, stored: u64, occupied: u64) -> VmCounters {
            let raw = RawCommand {
                argv: Vec::new(),
                status: 0,
                stdout: String::new(),
                stderr: String::new(),
                wall_ms: 0.0,
            };
            VmCounters {
                swap_used_bytes: swap,
                pageouts: 1,
                compressions: 1,
                swapouts: 1,
                compressor_pages_stored: stored,
                compressor_pages_occupied: occupied,
                process_pageins: 1,
                process_disk_bytes_read: 1,
                process_disk_bytes_written: 1,
                vm_stat: raw.clone(),
                swapusage: raw,
            }
        }
        let before = counters(10, 20, 30);
        let advisory = counters(10, 19, 29);
        assert!(VmDelta::between(&before, &advisory).is_valid());
        let fatal = counters(11, 20, 30);
        assert!(!VmDelta::between(&before, &fatal).is_valid());
    }
}

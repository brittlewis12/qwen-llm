//! CPU-only conditional-distribution analysis; never runs a target or drafter.
//! Optional paired costs: QWEN_NEXTN_SERIAL_MS and QWEN_NEXTN_VERIFY2_MS.
//! Streams payloads; bounded to V <= 1,048,576 and 1 <= scored rows <= 4096.
use super::kl_divergence;
use crate::sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig, SamplingUniform};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const COLUMNS: [&str; 8] = [
    "greedy_agreement",
    "full_kl_p_q",
    "full_kl_q_p",
    "filtered_overlap",
    "p_mass_at_q_argmax",
    "p_support",
    "q_support",
    "support_intersection",
];

fn integer(j: &Value, key: &str) -> Result<usize> {
    let n = j[key].as_u64().with_context(|| format!("invalid {key}"))?;
    Ok(usize::try_from(n)?)
}
fn number(j: &Value, key: &str) -> Result<f64> {
    let n = j[key].as_f64().with_context(|| format!("invalid {key}"))?;
    ensure!(n.is_finite() && n >= 0.0, "invalid {key}");
    Ok(n)
}
fn sampling(c: &Value) -> Result<SamplingConfig> {
    let s = &c["sampling"];
    let version = integer(s, "algorithm_version")?;
    ensure!(
        version == SAMPLER_ALGORITHM_VERSION as usize,
        "sampler version mismatch"
    );
    let config = SamplingConfig {
        temperature: number(s, "temperature")? as f32,
        top_k: integer(s, "top_k")?,
        top_p: number(s, "top_p")? as f32,
        min_p: number(s, "min_p")? as f32,
        seed: s["seed"].as_u64().context("missing seed")?,
    };
    ensure!(
        number(s, "top_p")? <= 1.0 && number(s, "min_p")? <= 1.0,
        "invalid filter range"
    );
    Ok(config.validate()?)
}

#[derive(Clone, Copy, serde::Deserialize)]
struct Shape {
    hidden_size: usize,
    vocab_size: usize,
    token_count: usize,
    score_start: usize,
    score_count: usize,
}
impl Shape {
    fn bytes(self, rows: usize, width: usize) -> Result<u64> {
        (rows as u64)
            .checked_mul(width as u64)
            .and_then(|n| n.checked_mul(4))
            .context("extent overflow")
    }
}

fn alignment(c: &Value, p: &Value) -> Result<(Shape, SamplingConfig, Vec<f64>)> {
    for (j, role) in [(c, "capture"), (p, "probe")] {
        let schema = format!("qwen.glm-nextn-{role}.v1");
        ensure!(
            j["schema"] == schema && j["status"] == "complete",
            "incomplete/wrong {role}"
        );
    }
    ensure!(p["phase"] == "complete", "probe phase not complete");
    ensure!(p["capture_manifest"] == *c, "capture manifest mismatch");
    ensure!(c["byte_order"] == "little", "unsupported byte order");
    let kind = p["hidden_kind"].as_str();
    ensure!(
        matches!(kind, Some("post-norm" | "raw")),
        "conditioning kind"
    );
    let d: Shape = serde_json::from_value(c.clone())?;
    ensure!(
        d.hidden_size > 0 && d.score_start > 0 && d.score_count > 0,
        "zero dimensions"
    );
    ensure!(
        (1..=1 << 20).contains(&d.vocab_size) && d.score_count <= 4096,
        "V/T analyzer bounds"
    );
    let n = d
        .score_start
        .checked_add(d.score_count)
        .context("count overflow")?;
    ensure!(n == d.token_count, "N != S+T");
    for (key, expected) in [
        ("history_rows_processed", d.score_start),
        ("scored_rows_returned", d.score_count),
        ("history_requested_logit_rows", 0),
    ] {
        ensure!(integer(p, key)? == expected, "probe {key} mismatch");
    }
    let batch = integer(p, "history_batch")?;
    ensure!((1..=512).contains(&batch), "invalid history batch");
    let steps = p["score_steps"].as_array().context("missing score steps")?;
    ensure!(steps.len() == d.score_count, "step count mismatch");
    let mut times = Vec::with_capacity(steps.len());
    for (row, step) in steps.iter().enumerate() {
        let position = d.score_start + row;
        for (key, expected) in [
            ("position", position),
            ("target_logit_row", row),
            ("hidden_source_position", position - 1),
        ] {
            ensure!(integer(step, key)? == expected, "row {row}: wrong {key}");
        }
        times.push(number(step, "full_head_step_ms")?);
    }
    let sum: f64 = times.iter().sum();
    ensure!(sum.is_finite(), "timing sum overflow");
    let total = number(p, "score_steps_total_ms")?;
    ensure!(
        (total - sum).abs() <= 4.0 * f64::EPSILON * sum.max(1.0) * times.len() as f64,
        "timing total mismatch"
    );
    number(p, "head_history_init_ms")?;
    number(p, "load_and_context_ms")?;
    Ok((d, sampling(c)?, times))
}

fn metrics(sampler: &Sampler, p: &[f32], q: &[f32]) -> Result<[f64; 8]> {
    ensure!(
        !p.is_empty() && p.len() == q.len() && p.iter().chain(q).all(|v| v.is_finite()),
        "invalid logit rows"
    );
    let uniform = SamplingUniform::new(0.0)?;
    let greedy = Sampler::new(SamplingConfig::default())?;
    let agree =
        greedy.diagnose(p, uniform)?.sampled.token == greedy.diagnose(q, uniform)?.sampled.token;
    let (kpq, kqp) = (kl_divergence(p, q).max(0.0), kl_divergence(q, p).max(0.0));
    // Native filtering and tie order, not an independent top-k/top-p/min-p implementation.
    let (pd, qd) = (sampler.diagnose(p, uniform)?, sampler.diagnose(q, uniform)?);
    let mut mass = vec![0.0; p.len()];
    for c in &pd.candidates {
        mass[c.token as usize] = c.weight / pd.total_weight;
    }
    let mut overlap = 0.0;
    let mut intersection = 0;
    for c in &qd.candidates {
        let q = c.weight / qd.total_weight;
        overlap += mass[c.token as usize].min(q);
        intersection += usize::from(q > 0.0 && mass[c.token as usize] > 0.0);
    }
    // Positive-temperature candidates are in descending weight, ascending-ID
    // tie order. Greedy mode has a singleton with native greedy tie semantics.
    let q_mode = qd.candidates.first().context("empty Q support")?.token as usize;
    let mut result = [
        if agree { 1.0 } else { 0.0 },
        kpq,
        kqp,
        overlap,
        mass[q_mode],
        mass.iter().filter(|&&v| v > 0.0).count() as f64,
        qd.candidates.iter().filter(|c| c.weight > 0.0).count() as f64,
        intersection as f64,
    ];
    ensure!(result.iter().all(|v| v.is_finite()), "nonfinite metric");
    // Positive F64 sums/divisions: cover normalization and accumulation with
    // four gamma_V terms, rather than a vocabulary-independent tolerance.
    let nu = p.len() as f64 * f64::EPSILON;
    let roundoff = 4.0 * nu / (1.0 - nu);
    ensure!((0.0..=1.0 + roundoff).contains(&overlap), "invalid overlap");
    result[3] = overlap.min(1.0); // Only remove normalized-sum roundoff above one.
    Ok(result)
}

fn summary(values: &[f64]) -> Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        let x = p * (sorted.len() - 1) as f64;
        let i = x.floor() as usize;
        sorted[i] + (sorted[x.ceil() as usize] - sorted[i]) * (x - i as f64)
    };
    json!({"count":values.len(),"mean":values.iter().sum::<f64>()/values.len() as f64,
        "min":sorted[0],"max":sorted[sorted.len()-1],"p05":percentile(0.05),"p50":percentile(0.5),
        "p95":percentile(0.95),"p99":percentile(0.99)})
}

fn read_json(path: &Path) -> Result<(Value, String)> {
    ensure!(path.metadata()?.len() <= 8 << 20, "manifest exceeds 8 MiB");
    let bytes = std::fs::read(path)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    Ok((serde_json::from_slice(&bytes)?, hash))
}
fn extent(actual: u64, expected: u64) -> Result<()> {
    ensure!(actual == expected, "binary extent: {actual} != {expected}");
    Ok(())
}
fn open_binary(dir: &Path, name: &str, bytes: u64) -> Result<BufReader<File>> {
    let file = File::open(dir.join(name))?;
    ensure!(file.metadata()?.is_file(), "binary is not a regular file");
    extent(file.metadata()?.len(), bytes)?;
    Ok(BufReader::new(file))
}
fn scan(reader: &mut impl Read, mut words: u64, vocab: Option<usize>) -> Result<()> {
    let mut buffer = [0u8; 65536];
    while words > 0 {
        let n = words.min((buffer.len() / 4) as u64) as usize;
        reader.read_exact(&mut buffer[..n * 4])?;
        for word in buffer[..n * 4].chunks_exact(4) {
            let word = word.try_into().unwrap();
            ensure!(
                match vocab {
                    Some(v) => (u32::from_le_bytes(word) as usize) < v,
                    None => f32::from_le_bytes(word).is_finite(),
                },
                "invalid token/nonfinite feature"
            );
        }
        words -= n as u64;
    }
    ensure!(reader.read(&mut [0])? == 0, "unexpected binary suffix");
    Ok(())
}
fn read_row(reader: &mut impl Read, v: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; v.checked_mul(4).context("row extent overflow")?];
    reader.read_exact(&mut bytes)?;
    let row: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(row.iter().all(|v| v.is_finite()), "nonfinite logit row");
    Ok(row)
}

fn user_costs() -> Result<Option<(f64, f64)>> {
    let a = std::env::var_os("QWEN_NEXTN_SERIAL_MS");
    let b = std::env::var_os("QWEN_NEXTN_VERIFY2_MS");
    ensure!(
        a.is_some() == b.is_some(),
        "supply both serial and verify2 costs or neither"
    );
    let parse = |v: std::ffi::OsString| -> Result<f64> {
        let n: f64 = v.to_str().context("invalid cost encoding")?.parse()?;
        ensure!(n.is_finite() && n > 0.0, "cost must be finite and positive");
        Ok(n)
    };
    a.zip(b)
        .map(|(a, b)| -> Result<_> { Ok((parse(a)?, parse(b)?)) })
        .transpose()
}
fn draft_budget(serial: f64, verify2: f64, acceptance: f64) -> f64 {
    (1.0 + acceptance) * serial - verify2
}

fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    // All validation/serialization precedes even a temporary write. Hard-link
    // publication is atomic and cannot replace an existing file or symlink.
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let temp = parent.join(format!(
        ".nextn-analysis-{}-{nonce}.tmp",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::hard_link(&temp, path)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(temp);
    result
}

#[test]
#[ignore = "CPU artifact analysis: QWEN_NEXTN_CAPTURE_DIR, QWEN_NEXTN_PROBE_DIR, new QWEN_NEXTN_ANALYSIS_OUT"]
fn analyze_nextn_capture() -> Result<()> {
    let env = |key| {
        std::env::var_os(key)
            .map(PathBuf::from)
            .with_context(|| format!("set {key}"))
    };
    let (capture_dir, probe_dir, out) = (
        env("QWEN_NEXTN_CAPTURE_DIR")?,
        env("QWEN_NEXTN_PROBE_DIR")?,
        env("QWEN_NEXTN_ANALYSIS_OUT")?,
    );
    ensure!(
        out.symlink_metadata()
            .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
        "output must not exist"
    );
    let costs = user_costs()?;
    let (c, chash) = read_json(&capture_dir.join("manifest.json"))?;
    let (p, phash) = read_json(&probe_dir.join("report.json"))?;
    let (d, config, times) = alignment(&c, &p)?;
    for (name, rows, width, tokens) in [
        ("tokens.u32", d.token_count, 1, true),
        ("hidden.f32", d.token_count, d.hidden_size, false),
        ("hidden-raw.f32", d.token_count, d.hidden_size, false),
        ("target-logits.f32", d.score_count, d.vocab_size, false),
    ] {
        let bytes = d.bytes(rows, width)?;
        extent(
            c["files"][name]["bytes"]
                .as_u64()
                .context("missing declared file extent")?,
            bytes,
        )?;
        let mut file = open_binary(&capture_dir, name, bytes)?;
        if name != "target-logits.f32" {
            scan(&mut file, bytes / 4, tokens.then_some(d.vocab_size))?;
        }
    }
    let bytes = d.bytes(d.score_count, d.vocab_size)?;
    let mut target = open_binary(&capture_dir, "target-logits.f32", bytes)?;
    let mut draft = open_binary(&probe_dir, "draft-logits.f32", bytes)?;
    let sampler = Sampler::new(config)?;
    let mut rows = Vec::with_capacity(d.score_count);
    for _ in 0..d.score_count {
        rows.push(metrics(
            &sampler,
            &read_row(&mut target, d.vocab_size)?,
            &read_row(&mut draft, d.vocab_size)?,
        )?);
    }
    ensure!(
        target.read(&mut [0])? == 0 && draft.read(&mut [0])? == 0,
        "unexpected logits suffix"
    );
    let aggregates: serde_json::Map<_, _> = COLUMNS
        .iter()
        .enumerate()
        .map(|(i, key)| {
            (
                key.to_string(),
                summary(&rows.iter().map(|r| r[i]).collect::<Vec<_>>()),
            )
        })
        .collect();
    let acceptance = rows.iter().map(|r| r[3]).sum::<f64>() / rows.len() as f64;
    let head_mean = times.iter().sum::<f64>() / times.len() as f64;
    let history = number(&p, "head_history_init_ms")?;
    let cost = costs.map(|(serial,verify2)| -> Result<Value> {
        let budget = draft_budget(serial,verify2,acceptance);
        let required = (verify2+head_mean)/serial-1.0;
        let with_history = (verify2+head_mean+history/rows.len() as f64)/serial-1.0;
        ensure!([budget,required,with_history].iter().all(|v|v.is_finite()), "cost arithmetic overflow");
        Ok(json!({"user_serial_ms":serial,"user_verify2_ms":verify2,
        "ideal_expected_tokens_per_two_row_attempt":1.0+acceptance,
        "draft_plus_unpriced_overhead_budget_ms":budget,
        "required_acceptance_at_reference_head_cost":required,
        "required_acceptance_with_amortized_reference_history":with_history,
        "scope":"hypothetical one-proposal break-even only; user cost compatibility unverified; excludes state catch-up/correction/restore/sampling overhead"}))
    }).transpose()?;
    let report = json!({"schema":"qwen.glm-nextn-analysis.v1","status":"complete",
        "score_start":d.score_start,"score_count":d.score_count,"token_count":d.token_count,
        "hidden_size":d.hidden_size,"vocab_size":d.vocab_size,"hidden_kind":p["hidden_kind"],
        "sampling":{"temperature":config.temperature,"top_k":config.top_k,"top_p":config.top_p,"min_p":config.min_p,
            "seed":config.seed,"algorithm_version":SAMPLER_ALGORITHM_VERSION,"distribution_api":"Sampler::diagnose"},
        "interpretation":"target-on-policy one-step conditional expected maximal-coupling acceptance; not recursive multi-step acceptance or actual end-to-end speedup",
        "metric_conventions":"full KL uses unfiltered temperature-1 softmax; supports count positive mass; Q argmax uses native candidate tie order; greedy agreement uses native greedy ties; percentiles interpolate at p*(n-1)",
        "aggregates":aggregates,"rows":rows.iter().enumerate().map(|(i,row)| {
            let mut v: serde_json::Map<_,_> = COLUMNS.iter().zip(row).map(|(k,v)|(k.to_string(),json!(v))).collect();
            v.insert("position".into(),json!(d.score_start+i));
            v.insert("reference_full_head_step_ms".into(),json!(times[i])); v }).collect::<Vec<_>>(),
        "reference_cost":{"full_head_step_ms":summary(&times),"score_steps_total_ms":p["score_steps_total_ms"],
            "head_history_init_ms":history,"history_ms_per_scored_row":history/rows.len() as f64,
            "load_and_context_ms":p["load_and_context_ms"],"history_batch":p["history_batch"],"body_only_ms":null,
            "scope":"reference process+synchronize+full-logit retrieval; excludes scored input setup/output IO; history includes input IO/setup; not native draft cost"},
        "break_even":cost,"provenance":{"capture_manifest_sha256":chash,"probe_report_sha256":phash}});
    publish(&out, &serde_json::to_vec_pretty(&report)?)
}

#[test]
fn nextn_analysis_cpu_distribution_laws() {
    let config = SamplingConfig {
        temperature: 0.5,
        min_p: 0.4,
        top_p: 1.0,
        ..SamplingConfig::glm5_next(42)
    };
    let sampler = Sampler::new(config).unwrap();
    let p = [1.0f32.ln(), 0.5f32.ln(), 0.1f32.ln(), 0.01f32.ln()];
    let q = [p[1], p[0], p[2], p[3]];
    let equal = metrics(&sampler, &p, &p).unwrap();
    assert_eq!(equal[0], 1.0);
    assert_eq!(equal[1], 0.0);
    assert_eq!(equal[2], 0.0);
    assert!((equal[3] - 1.0).abs() < 1e-12);
    let m = metrics(&sampler, &p, &q).unwrap();
    assert!((m[3] - 0.4).abs() < 1e-7 && (m[4] - 0.2).abs() < 1e-7); // min-p is BEFORE temperature
    assert_eq!(&m[5..8], &[2.0, 2.0, 2.0]);
    assert!(m[1] > 0.0 && m[2] > 0.0);
    let filtered = Sampler::new(SamplingConfig {
        top_p: 0.7,
        ..config
    })
    .unwrap();
    let disjoint = metrics(&filtered, &p, &q).unwrap();
    assert_eq!(&disjoint[3..8], &[0.0, 0.0, 1.0, 1.0, 0.0]);
    let topk = Sampler::new(SamplingConfig { top_k: 1, ..config }).unwrap();
    assert_eq!(metrics(&topk, &p, &q).unwrap()[3], 0.0);
    assert!((draft_budget(10.0, 15.0, 0.8) - 3.0).abs() < 1e-12);
    assert_eq!(summary(&[0.0, 1.0])["p50"], json!(0.5));
}

#[test]
fn nextn_analysis_cpu_alignment_and_finite_errors() {
    let c = json!({"schema":"qwen.glm-nextn-capture.v1","status":"complete","byte_order":"little",
        "hidden_size":3,"vocab_size":4,"token_count":4,"score_start":2,"score_count":2,
        "sampling":{"temperature":1.0,"top_k":0,"top_p":0.95,"min_p":0.0,"seed":42,"algorithm_version":SAMPLER_ALGORITHM_VERSION}});
    let p = json!({"schema":"qwen.glm-nextn-probe.v1","status":"complete","phase":"complete","capture_manifest":c,
        "hidden_kind":"post-norm","history_batch":512,"history_rows_processed":2,"scored_rows_returned":2,"history_requested_logit_rows":0,
        "head_history_init_ms":10.0,"load_and_context_ms":5.0,"score_steps_total_ms":3.0,
        "score_steps":[{"position":2,"target_logit_row":0,"hidden_source_position":1,"full_head_step_ms":1.0},
            {"position":3,"target_logit_row":1,"hidden_source_position":2,"full_head_step_ms":2.0}]});
    let (shape, _, _) = alignment(&c, &p).unwrap();
    assert!(shape.bytes(usize::MAX, 4).is_err());
    for key in ["position", "target_logit_row", "hidden_source_position"] {
        let mut bad = p.clone();
        bad["score_steps"][0][key] = json!(99);
        assert!(alignment(&c, &bad).is_err());
    }
    for key in ["temperature", "top_p", "min_p", "top_k"] {
        let mut bad = c.clone();
        bad["sampling"][key] = json!(-1);
        assert!(sampling(&bad).is_err());
    }
    let mut bad = p.clone();
    bad["status"] = json!("failed");
    assert!(alignment(&c, &bad).is_err());
    let mut bad = c.clone();
    bad["token_count"] = json!(5);
    assert!(alignment(&bad, &p).is_err());
    let mut same_bad_shape = p.clone();
    same_bad_shape["capture_manifest"] = bad.clone();
    assert!(alignment(&bad, &same_bad_shape).is_err());
    assert!(extent(7, 8).is_err());
    assert!(read_row(&mut std::io::Cursor::new(vec![0; 7]), 2).is_err());
    assert!(read_row(&mut std::io::Cursor::new(f32::NAN.to_le_bytes()), 1).is_err());
    let mut poison = std::io::Cursor::new(f32::INFINITY.to_le_bytes());
    assert!(scan(&mut poison, 1, None).is_err());
    assert!(scan(&mut std::io::Cursor::new(4u32.to_le_bytes()), 1, Some(4)).is_err());
    let sampler = Sampler::new(SamplingConfig::glm5_next(42)).unwrap();
    assert!(metrics(&sampler, &[0.0], &[f32::INFINITY]).is_err());
}

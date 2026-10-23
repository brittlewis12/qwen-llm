//! Reader for the frozen full-forward archive, independent of Metal.
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

pub(super) const STEPS: usize = 256;
pub(super) const SCHEMA: &str = "v41-ced-original-full-forward-v1";
pub(super) const RUNTIME: &str = "84c465db06384a4c87da7a7304a98e9b479c150b";
pub(super) const HARNESS: &str = "9ee7b7a3c9a3b5b990f88f997045f851b10ca5a3";
const FILES: [&str; 9] = [
    "header.json",
    "input.json",
    "artifact.json",
    "provenance.json",
    "data.bin",
    "regions.jsonl",
    "states.jsonl",
    "dispatch-first-scalar.json",
    "dispatch-last-scalar.json",
];

pub(super) fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn json_sha(value: &Value) -> Result<String> {
    let mut value = value.clone();
    value.sort_all_objects();
    Ok(sha(&serde_json::to_vec(&value)?))
}

pub(super) fn file_sha(file: &mut File) -> Result<String> {
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(format!("{:x}", hash.finalize()));
        }
        hash.update(&buf[..n]);
    }
}

pub(super) fn open(path: &Path) -> Result<File> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    ensure!(
        f.metadata()?.is_file(),
        "not a regular file: {}",
        path.display()
    );
    Ok(f)
}

pub(super) fn number(v: &Value, name: &str) -> Result<usize> {
    usize::try_from(
        v[name]
            .as_u64()
            .with_context(|| format!("missing/invalid {name}"))?,
    )
    .map_err(Into::into)
}

fn digest(v: &Value) -> Result<&str> {
    let s = v.as_str().context("missing digest")?;
    ensure!(
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid SHA256"
    );
    Ok(s)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub prompt: Vec<u32>,
    pub continuation: Vec<u32>,
}

#[derive(Clone)]
pub(super) struct Geometry {
    pub n: usize,
    pub layers: usize,
    pub cut: usize,
    pub window: usize,
    pub hidden: usize,
    pub head_dim: usize,
    pub index_dim: usize,
    pub vocab: usize,
    pub topk: usize,
    pub block: usize,
    pub owners: Vec<(usize, usize)>,
    pub producers: Vec<usize>,
}

impl Geometry {
    pub fn boundary_start(&self) -> usize {
        self.n
            .saturating_sub(1 + (self.layers - self.cut) * (self.window - 1))
    }
    pub fn output_start(&self, layer: usize) -> usize {
        self.n
            .saturating_sub(1 + (self.layers - 1 - layer) * (self.window - 1))
    }
    pub fn producer(&self, layer: usize) -> usize {
        *self.producers.iter().rfind(|&&p| p <= layer).unwrap()
    }
    pub fn state_keys(&self) -> Vec<(String, usize)> {
        let mut keys: Vec<_> = (0..self.layers)
            .map(|l| ("local_ring_physical".into(), l))
            .collect();
        for &(l, _) in &self.owners {
            keys.extend(
                [
                    "owner_completed_kv",
                    "owner_completed_index_keys",
                    "owner_pending_value",
                    "owner_pending_gate",
                ]
                .map(|r| (r.to_owned(), l)),
            );
        }
        keys
    }
    fn state_shape(&self, role: &str, layer: usize, position: usize) -> Result<Vec<usize>> {
        if role == "local_ring_physical" {
            ensure!(layer < self.layers, "raw layer");
            return Ok(vec![self.window, self.head_dim]);
        }
        let ratio = self
            .owners
            .iter()
            .find(|&&(l, _)| l == layer)
            .context("unknown owner")?
            .1;
        Ok(match role {
            "owner_completed_kv" => vec![position / ratio, self.head_dim],
            "owner_completed_index_keys" => vec![position / ratio, self.index_dim],
            "owner_pending_value" | "owner_pending_gate" => vec![self.head_dim],
            _ => bail!("unknown state role {role}"),
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Record {
    pub role: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub layer: Option<usize>,
    pub producer: Value,
    pub absroworigin: Option<usize>,
    pub state_position: Option<usize>,
    pub filename: String,
    pub byte_offset: usize,
    pub byte_length: usize,
    pub sha256: String,
    pub record: usize,
}

type Key = (String, Option<usize>, usize);

pub(super) struct Reference {
    pub header: Value,
    pub artifact: Value,
    pub provenance: Value,
    pub complete: Value,
    pub states: Vec<Value>,
    pub geometry: Geometry,
    records: BTreeMap<Key, Record>,
    data: Mmap,
    retained: Vec<(PathBuf, File, Vec<i128>)>,
}

fn stamp(f: &File) -> Result<Vec<i128>> {
    let m = f.metadata()?;
    Ok(vec![
        m.dev() as i128,
        m.ino() as i128,
        m.len() as i128,
        m.mtime() as i128,
        m.mtime_nsec() as i128,
        m.ctime() as i128,
        m.ctime_nsec() as i128,
    ])
}

/// Charge the entire lazy mapping, JSON allocation expansion, index nodes and
/// transient parsing/hashing buffers before opening the archive payloads.
pub(super) fn cpu_reserve(dir: &Path) -> Result<u64> {
    let mut total = 64u64 * 1024 * 1024;
    for name in FILES.into_iter().chain(["complete.json"]) {
        let bytes = open(&dir.join(name))?.metadata()?.len();
        let charge = bytes
            .checked_mul(if name == "data.bin" { 1 } else { 24 })
            .context("archive reserve overflow")?;
        total = total
            .checked_add(charge)
            .context("archive reserve overflow")?;
    }
    Ok(total)
}

impl Reference {
    pub fn load(dir: &Path, g: Geometry, input_bytes: &[u8], expected_arm: &str) -> Result<Self> {
        ensure!(cfg!(target_endian = "little"), "little endian required");
        ensure!(
            g.n > 0 && g.cut > 0 && g.cut < g.layers && g.window > 0 && g.block > 0 && g.topk > 0,
            "geometry"
        );
        let complete_path = dir.join("complete.json");
        let mut complete_file = open(&complete_path)?;
        let complete_stamp = stamp(&complete_file)?;
        let complete: Value = serde_json::from_reader(&mut complete_file)?;
        ensure!(
            stamp(&complete_file)? == complete_stamp,
            "completion marker changed during read"
        );
        ensure!(
            complete["schema"] == SCHEMA && complete["complete"] == true,
            "missing/corrupt completion marker"
        );
        ensure!(
            number(&complete, "initial_position")? == g.n
                && number(&complete, "final_position")? == g.n + STEPS
                && number(&complete, "state_positions")? == STEPS + 1,
            "complete position counts"
        );
        let mut retained = vec![(complete_path, complete_file, complete_stamp)];
        let mut names = BTreeSet::new();
        for item in complete["files"].as_array().context("manifest files")? {
            let name = item["filename"].as_str().context("manifest filename")?;
            ensure!(
                FILES.contains(&name) && names.insert(name.to_owned()),
                "unexpected/duplicate manifest member"
            );
            let path = dir.join(name);
            let mut file = open(&path)?;
            let before = stamp(&file)?;
            ensure!(
                file.metadata()?.len() == item["bytes"].as_u64().context("manifest length")?,
                "manifest length: {name}"
            );
            ensure!(
                file_sha(&mut file)? == digest(&item["sha256"])?,
                "manifest hash: {name}"
            );
            ensure!(
                stamp(&file)? == before,
                "reference changed while hashing: {name}"
            );
            retained.push((path, file, before));
        }
        ensure!(names.len() == FILES.len(), "missing manifest files");
        let read_json = |name: &str| -> Result<Value> {
            use std::io::{Seek, SeekFrom};
            let mut f = retained
                .iter()
                .find(|(p, _, _)| p.file_name().unwrap() == name)
                .unwrap()
                .1
                .try_clone()?;
            f.seek(SeekFrom::Start(0))?;
            Ok(serde_json::from_reader(f)?)
        };
        let header = read_json("header.json")?;
        ensure!(
            header["schema"] == SCHEMA
                && header["original_runtime_pin"] == RUNTIME
                && header["arm"] == expected_arm,
            "wrong schema/runtime/arm (scalar reference is original token-major F32, not batched)"
        );
        ensure!(
            number(&header, "N")? == g.n
                && number(&header, "capacity")? == g.n + STEPS
                && number(&header, "continuation_steps")? == STEPS
                && number(&header, "decoder_start")? == g.cut
                && number(&header, "boundary_layer")? + 1 == g.cut
                && number(&header, "window")? == g.window
                && number(&header, "prefill_chunk_rows")? == 2048,
            "header geometry"
        );
        ensure!(
            header["B"] == 1 + (g.layers - g.cut) * (g.window - 1)
                && header["boundary_range"] == json!([g.boundary_start(), g.n]),
            "boundary range"
        );
        let ranges: Vec<_> = (g.cut..g.layers).map(|l| json!({"layer": l,
            "range": [g.output_start(l), g.n], "seed_range": [g.output_start(l).saturating_sub(g.window - 1), g.output_start(l)]})).collect();
        ensure!(
            header["decoder_output_ranges"] == json!(ranges),
            "decoder ranges"
        );
        let expected = &header["expected_counts"];
        let control_rows: usize = (g.cut..g.layers).map(|l| g.n - g.output_start(l)).sum();
        ensure!(
            expected["boundary_rows_each_E_P"] == g.n - g.boundary_start()
                && expected["query_control_rows_each_selected_candidates"] == control_rows
                && expected["full_logit_vectors"] == STEPS + 1
                && expected["causal_hash_positions"] == STEPS + 1
                && expected["causal_regions_each_position"] == g.layers + 4 * g.owners.len()
                && expected["full_binary_state_positions"] == json!([g.n, g.n + STEPS]),
            "expected counts"
        );
        ensure!(
            sha(input_bytes) == digest(&header["input_file_sha256"])?,
            "input file SHA mismatch"
        );
        let input: Input = serde_json::from_slice(input_bytes)?;
        ensure!(
            input.prompt.len() == g.n
                && input.continuation.len() == STEPS
                && input
                    .prompt
                    .iter()
                    .chain(&input.continuation)
                    .all(|&t| (t as usize) < g.vocab),
            "input IDs/extents"
        );
        let input = serde_json::to_value(input)?;
        ensure!(
            json_sha(&input)? == digest(&header["input_ids_sha256"])?
                && read_json("input.json")? == input,
            "input IDs SHA mismatch"
        );
        let artifact = read_json("artifact.json")?;
        ensure!(
            json_sha(&artifact["layout"])? == digest(&artifact["layout_sha256"])?
                && artifact["layout_sha256"] == header["artifact_layout_sha256"],
            "artifact layout digest"
        );
        let shards = artifact["shards"].as_array().context("artifact shards")?;
        ensure!(!shards.is_empty(), "no artifact shards");
        for (i, s) in shards.iter().enumerate() {
            ensure!(
                number(s, "shard_idx")? == i && number(s, "size")? > 0,
                "shard identity"
            );
            let alignment = number(s, "alignment")?;
            ensure!(
                alignment.is_power_of_two() && number(s, "tensor_data_start")? % alignment == 0,
                "shard alignment"
            );
            digest(&s["sha256"])?;
        }
        let provenance = read_json("provenance.json")?;
        ensure!(
            provenance["runtime_source_commit"] == HARNESS
                && provenance["original_runtime_pin"] == RUNTIME,
            "unfrozen harness provenance"
        );
        digest(&provenance["binary_sha256"])?;
        let data_file = &retained
            .iter()
            .find(|(p, _, _)| p.file_name().unwrap() == "data.bin")
            .unwrap()
            .1;
        ensure!(data_file.metadata()?.len() > 0, "empty archive data");
        // Frozen archives must remain immutable for the lifetime of the mapping.
        let data = unsafe { Mmap::map(data_file)? };
        let mut result = Self {
            header,
            artifact,
            provenance,
            complete,
            states: Vec::new(),
            geometry: g,
            records: BTreeMap::new(),
            data,
            retained,
        };
        let mut offset = 0;
        for (id, value) in result.lines("regions.jsonl")?.into_iter().enumerate() {
            ensure!(
                value.get("layer").is_some() && value.get("absroworigin").is_some(),
                "missing record identity fields {id}"
            );
            let r: Record = serde_json::from_value(value)?;
            ensure!(
                r.record == id
                    && r.filename == "data.bin"
                    && r.byte_offset == offset
                    && offset.is_multiple_of(4)
                    && r.byte_length.is_multiple_of(4),
                "record order/offset/alignment {id}"
            );
            let elements = r
                .shape
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b).context("shape overflow"))?;
            ensure!(
                elements.checked_mul(4) == Some(r.byte_length),
                "record shape/length {id}"
            );
            offset = offset
                .checked_add(r.byte_length)
                .context("record overflow")?;
            ensure!(
                offset <= result.data.len()
                    && sha(&result.data[r.byte_offset..offset]) == digest(&json!(r.sha256))?,
                "record hash/bounds {id}"
            );
            let key = result.validate_record(&r)?;
            ensure!(
                result.records.insert(key, r).is_none(),
                "duplicate record identity"
            );
        }
        ensure!(
            offset == result.data.len()
                && number(&result.complete, "regions")? == result.records.len(),
            "complete regions/data length"
        );
        result.validate_coverage()?;
        result.states = result.lines("states.jsonl")?;
        result.validate_states()?;
        result.revalidate()?;
        Ok(result)
    }

    fn lines(&self, name: &str) -> Result<Vec<Value>> {
        use std::io::{Seek, SeekFrom};
        let mut f = self
            .retained
            .iter()
            .find(|(p, _, _)| p.file_name().unwrap() == name)
            .unwrap()
            .1
            .try_clone()?;
        f.seek(SeekFrom::Start(0))?;
        BufReader::new(f)
            .lines()
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect()
    }

    fn validate_record(&self, r: &Record) -> Result<Key> {
        let g = &self.geometry;
        let pos;
        let shape = match r.role.as_str() {
            "E" | "P" => {
                pos = r.absroworigin.context("boundary row")?;
                ensure!(
                    r.layer == Some(g.cut - 1)
                        && (g.boundary_start()..g.n).contains(&pos)
                        && r.producer == g.cut - 1
                        && r.state_position.is_none()
                        && r.dtype == "f32-le",
                    "boundary identity"
                );
                if r.role == "E" {
                    vec![4, g.hidden]
                } else {
                    vec![4]
                }
            }
            "selected" | "candidates" => {
                let layer = r.layer.context("control layer")?;
                pos = r.absroworigin.context("control row")?;
                ensure!(
                    (g.cut..g.layers).contains(&layer)
                        && (g.output_start(layer)..g.n).contains(&pos)
                        && r.producer
                            == if r.role == "selected" {
                                g.producer(layer)
                            } else {
                                g.cut
                            }
                        && r.state_position.is_none()
                        && r.dtype == "i32-le",
                    "control identity/producer"
                );
                vec![if r.role == "selected" {
                    (pos + 1).min(g.topk)
                } else {
                    (pos + 1).div_ceil(g.block)
                }]
            }
            "logits" => {
                pos = r.state_position.context("logit position")?;
                ensure!(
                    (g.n..=g.n + STEPS).contains(&pos)
                        && r.layer.is_none()
                        && r.producer == "head"
                        && r.absroworigin == Some(pos - 1)
                        && r.dtype == "f32-le",
                    "logit identity"
                );
                vec![g.vocab]
            }
            _ => {
                pos = r.state_position.context("state position")?;
                let layer = r.layer.context("state layer")?;
                ensure!(
                    [g.n, g.n + STEPS].contains(&pos) && r.producer == layer && r.dtype == "f32-le",
                    "state identity"
                );
                ensure!(
                    r.absroworigin
                        == if r.role.starts_with("owner_completed_") {
                            Some(0)
                        } else {
                            None
                        },
                    "state origin"
                );
                g.state_shape(&r.role, layer, pos)?
            }
        };
        ensure!(
            r.shape == shape,
            "reference shape mismatch: {} {:?} {pos}: {:?} != {:?}",
            r.role,
            r.layer,
            r.shape,
            shape
        );
        if r.dtype == "f32-le" {
            ensure!(
                self.floats(r)?.iter().all(|v| v.is_finite()),
                "nonfinite reference {}",
                r.role
            );
        } else {
            let values: &[i32] = bytemuck::try_cast_slice(self.bytes(r))
                .map_err(|e| anyhow::anyhow!("I32 alignment: {e}"))?;
            if r.role == "selected" {
                ensure!(
                    values.iter().all(|&v| v >= 0 && v as usize <= pos)
                        && values.windows(2).all(|w| w[0] < w[1]),
                    "invalid selected IDs"
                );
            } else {
                ensure!(
                    values.iter().all(|&v| v == 0 || v == 1),
                    "invalid candidate mask"
                );
            }
        }
        Ok((r.role.clone(), r.layer, pos))
    }

    fn validate_coverage(&self) -> Result<()> {
        let g = &self.geometry;
        let mut count = 0;
        for role in ["E", "P"] {
            for row in g.boundary_start()..g.n {
                self.get(role, Some(g.cut - 1), row)?;
                count += 1;
            }
        }
        for layer in g.cut..g.layers {
            for role in ["selected", "candidates"] {
                for row in g.output_start(layer)..g.n {
                    self.get(role, Some(layer), row)?;
                    count += 1;
                }
            }
        }
        for pos in g.n..=g.n + STEPS {
            self.get("logits", None, pos)?;
            count += 1;
        }
        for pos in [g.n, g.n + STEPS] {
            for (role, layer) in g.state_keys() {
                self.get(&role, Some(layer), pos)?;
                count += 1;
            }
        }
        ensure!(
            count == self.records.len(),
            "extraneous/missing archive records"
        );
        Ok(())
    }

    fn validate_states(&self) -> Result<()> {
        let g = &self.geometry;
        ensure!(self.states.len() == STEPS + 1, "state count");
        for (i, s) in self.states.iter().enumerate() {
            let p = g.n + i;
            let full = i == 0 || i == STEPS;
            ensure!(
                s["position"] == p
                    && s["full_binary_state"] == full
                    && s["physical_ring_next_slot"] == p % g.window
                    && s["pending_ratio2_active"] == (p % 2 == 1)
                    && s["attention_cursor"]["next_position"] == p
                    && s["attention_cursor"]["next_layer"] == 0
                    && s["attention_cursor"]["batch"].is_null(),
                "state cursor {p}"
            );
            ensure!(
                s["address"]["position"] == p
                    && json_sha(&s["address"])? == digest(&s["address_sha256"])?,
                "address hash {p}"
            );
            let regions = s["regions"].as_array().context("state regions")?;
            let keys = g.state_keys();
            ensure!(
                regions.len() == keys.len() && number(s, "region_count")? == keys.len(),
                "causal region count"
            );
            for (r, (role, layer)) in regions.iter().zip(keys) {
                ensure!(
                    r["role"] == role
                        && r["layer"] == layer
                        && r["producer"] == layer
                        && r["state_position"] == p
                        && r["dtype"] == "f32-le"
                        && r["shape"] == json!(g.state_shape(&role, layer, p)?),
                    "state region identity/shape {p}"
                );
                digest(&r["sha256"])?;
                if full {
                    let binary = self.get(&role, Some(layer), p)?;
                    ensure!(
                        r["sha256"] == binary.sha256
                            && r["byte_offset"] == binary.byte_offset
                            && r["byte_length"] == binary.byte_length
                            && r["record"] == binary.record
                            && r["filename"] == "data.bin",
                        "state/binary index disagreement"
                    );
                }
            }
        }
        Ok(())
    }

    pub fn get(&self, role: &str, layer: Option<usize>, position: usize) -> Result<&Record> {
        self.records
            .get(&(role.to_owned(), layer, position))
            .with_context(|| {
                format!("missing reference row/region {role} layer={layer:?} absolute={position}")
            })
    }
    pub fn bytes(&self, r: &Record) -> &[u8] {
        &self.data[r.byte_offset..r.byte_offset + r.byte_length]
    }
    pub fn floats(&self, r: &Record) -> Result<&[f32]> {
        ensure!(r.dtype == "f32-le", "not F32");
        bytemuck::try_cast_slice(self.bytes(r)).map_err(|e| anyhow::anyhow!("F32 alignment: {e}"))
    }
    pub fn revalidate(&self) -> Result<()> {
        for (path, file, before) in &self.retained {
            ensure!(
                stamp(file)? == *before && stamp(&open(path)?)? == *before,
                "reference file replaced/changed: {}",
                path.display()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "archive/tests.rs"]
mod tests;

//! Bounded, metadata-only safetensors binding for individual main-model experts.
//! No payload reads, artifact authentication, MTP binding, or runtime admission.

use super::{ExpertProjection, TextConfig};
use serde_json::Value;
use std::{collections::BTreeMap, ops::Range};

pub const MAX_HEADER_BYTES: u64 = 1024 * 1024;

/// Only storage types observed in the captured pinned headers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceDtype {
    Bf16,
    F32,
    I8,
    F8E4M3,
    F8E8M0,
}

impl SourceDtype {
    pub fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I8 | Self::F8E4M3 | Self::F8E8M0 => 1,
        }
    }

    fn parse(value: &Value) -> Result<Self, String> {
        match value.as_str() {
            Some("BF16") => Ok(Self::Bf16),
            Some("F32") => Ok(Self::F32),
            Some("I8") => Ok(Self::I8),
            Some("F8_E4M3") => Ok(Self::F8E4M3),
            Some("F8_E8M0") => Ok(Self::F8E8M0),
            _ => Err("unsupported source dtype".into()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDescriptor {
    pub dtype: SourceDtype,
    pub shape: Vec<u64>,
    /// Relative to the first payload byte.
    pub relative_bytes: Range<u64>,
    /// Relative to the beginning of the source file, including its length prefix.
    pub file_bytes: Range<u64>,
}

#[derive(Debug)]
pub struct SourceHeader {
    source_file: String,
    declared_file_size: u64,
    payload_start: u64,
    tensors: BTreeMap<String, TensorDescriptor>,
}

impl SourceHeader {
    /// `bytes` must contain exactly the eight-byte LE prefix and JSON header.
    /// `declared_file_size` is supplied by the caller, not verified against disk.
    /// The source label is opaque; parsing performs no filesystem operations.
    pub fn parse(source_file: &str, bytes: &[u8], declared_file_size: u64) -> Result<Self, String> {
        let prefix: [u8; 8] = bytes
            .get(..8)
            .ok_or("truncated header prefix")?
            .try_into()
            .map_err(|_| "invalid header prefix")?;
        let length = u64::from_le_bytes(prefix);
        if length == 0 || length > MAX_HEADER_BYTES || length % 8 != 0 {
            return Err("invalid or oversized header length".into());
        }
        let payload_start = length.checked_add(8).ok_or("header length overflow")?;
        if bytes.len() as u64 != payload_start {
            return Err("header prefix/length mismatch".into());
        }
        let payload_size = declared_file_size
            .checked_sub(payload_start)
            .ok_or("file smaller than header")?;
        let text = std::str::from_utf8(&bytes[8..]).map_err(|e| e.to_string())?;
        if !text.starts_with('{') {
            return Err("header must start with an object".into());
        }
        let value = crate::tool_schema::decode_json(text)?;
        let object = value.as_object().ok_or("header must be an object")?;
        let mut tensors = BTreeMap::new();
        let mut intervals = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                let metadata = value.as_object().ok_or("metadata must be an object")?;
                if metadata.values().any(|v| !v.is_string()) {
                    return Err("metadata values must be strings".into());
                }
                continue;
            }
            let descriptor = value.as_object().ok_or("descriptor must be an object")?;
            if descriptor.len() != 3
                || !["dtype", "shape", "data_offsets"]
                    .iter()
                    .all(|k| descriptor.contains_key(*k))
            {
                return Err(format!("invalid descriptor fields: {name}"));
            }
            let dtype = SourceDtype::parse(&descriptor["dtype"])?;
            let shape = unsigned_array(&descriptor["shape"])?;
            let offsets = unsigned_array(&descriptor["data_offsets"])?;
            if offsets.len() != 2 || offsets[0] > offsets[1] {
                return Err(format!("invalid offsets: {name}"));
            }
            let extent = shape.iter().try_fold(dtype.byte_width(), |n, d| {
                n.checked_mul(*d).ok_or("tensor byte extent overflow")
            })?;
            if offsets[1] - offsets[0] != extent {
                return Err(format!("tensor byte extent mismatch: {name}"));
            }
            let start = payload_start
                .checked_add(offsets[0])
                .ok_or("file offset overflow")?;
            let end = payload_start
                .checked_add(offsets[1])
                .ok_or("file offset overflow")?;
            if end > declared_file_size {
                return Err(format!("tensor outside declared file: {name}"));
            }
            intervals.push((offsets[0], offsets[1]));
            tensors.insert(
                name.clone(),
                TensorDescriptor {
                    dtype,
                    shape,
                    relative_bytes: offsets[0]..offsets[1],
                    file_bytes: start..end,
                },
            );
        }
        intervals.sort_unstable();
        let mut end = 0;
        for (start, next) in intervals {
            if start != end {
                return Err("tensor offset overlap or gap".into());
            }
            end = next;
        }
        if end != payload_size {
            return Err("declared file size does not match payload extent".into());
        }
        Ok(Self {
            source_file: source_file.into(),
            declared_file_size,
            payload_start,
            tensors,
        })
    }

    pub fn source_file(&self) -> &str {
        &self.source_file
    }
    pub fn declared_file_size(&self) -> u64 {
        self.declared_file_size
    }
    pub fn payload_start(&self) -> u64 {
        self.payload_start
    }
    pub fn descriptor(&self, key: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(key)
    }
}

fn unsigned_array(value: &Value) -> Result<Vec<u64>, String> {
    value
        .as_array()
        .ok_or("expected unsigned integer array")?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| "expected unsigned integer".into()))
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundTensor {
    pub source_file: String,
    pub key: String,
    pub descriptor: TensorDescriptor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertPair {
    pub layer: usize,
    pub expert: usize,
    pub projection: ExpertProjection,
    pub rows: u64,
    pub input_columns: u64,
    pub weight: BoundTensor,
    pub scale: BoundTensor,
}

/// Bind only the requested main expert. Separate headers permit split-shard pairs;
/// caller-supplied labels and headers are not proof of checkpoint provenance.
pub fn bind_expert_pair(
    config: &TextConfig,
    layer: usize,
    expert: usize,
    projection: ExpertProjection,
    weight_source: &SourceHeader,
    scale_source: &SourceHeader,
) -> Result<ExpertPair, String> {
    if layer >= config.layers().len() || expert >= config.geometry().expert_count() {
        return Err("main expert index outside text profile".into());
    }
    let part = match projection {
        ExpertProjection::Gate => "w1",
        ExpertProjection::Up => "w3",
        ExpertProjection::Down => "w2",
    };
    let prefix = format!("layers.{layer}.ffn.experts.{expert}.{part}");
    let (weight_shape, scale_shape) = projection.source_shapes();
    let bind = |source: &SourceHeader, suffix: &str, dtype, shape: [u64; 2]| {
        let key = format!("{prefix}.{suffix}");
        let descriptor = source
            .descriptor(&key)
            .ok_or_else(|| format!("missing tensor: {key}"))?;
        if descriptor.dtype != dtype || descriptor.shape.as_slice() != shape {
            return Err(format!("invalid expert dtype or shape: {key}"));
        }
        Ok(BoundTensor {
            source_file: source.source_file.clone(),
            key,
            descriptor: descriptor.clone(),
        })
    };
    let weight = bind(weight_source, "weight", SourceDtype::I8, weight_shape)?;
    let scale = bind(scale_source, "scale", SourceDtype::F8E8M0, scale_shape)?;
    let (rows, input_columns) = projection.logical_shape();
    Ok(ExpertPair {
        layer,
        expert,
        projection,
        rows,
        input_columns,
        weight,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::super::ExpertGeometry;
    use super::*;
    use serde_json::json;

    const HEADER: &[u8] = include_bytes!(
        "../../../../scripts/reference/deepseek_v41/fixtures/headers/model-00003-of-00048.safetensors.header"
    );
    const MANIFEST: &str =
        include_str!("../../../../scripts/reference/deepseek_v41/tensor-headers.json");
    const SHARD: &str = "model-00003-of-00048.safetensors";

    fn config(geometry: ExpertGeometry) -> TextConfig {
        // Synthetic profile counts do not establish any REAP artifact exists.
        let mut value: Value = serde_json::from_str(include_str!(
            "../../../../scripts/reference/deepseek_v41/fixtures/config.json"
        ))
        .unwrap();
        value["text_config"]["n_routed_experts"] = json!(geometry.expert_count());
        TextConfig::from_hf_json(&value.to_string(), geometry).unwrap()
    }

    fn raw(text: &str) -> Vec<u8> {
        let mut text = text.as_bytes().to_vec();
        while text.len() % 8 != 0 {
            text.push(b' ');
        }
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend(text);
        bytes
    }

    fn parse(value: Value, payload: u64) -> Result<SourceHeader, String> {
        let bytes = raw(&value.to_string());
        SourceHeader::parse("synthetic", &bytes, bytes.len() as u64 + payload)
    }

    fn pair_value() -> Value {
        json!({
            "layers.0.ffn.experts.0.w1.weight": {
                "dtype":"I8", "shape":[2304,2560], "data_offsets":[0,5898240]
            },
            "layers.0.ffn.experts.0.w1.scale": {
                "dtype":"F8_E8M0", "shape":[2304,160], "data_offsets":[5898240,6266880]
            }
        })
    }

    #[test]
    fn captured_main_pairs_have_exact_ranges() {
        let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
        let size = manifest["shards"][SHARD]["source_content_length"]
            .as_u64()
            .unwrap();
        let header = SourceHeader::parse(SHARD, HEADER, size).unwrap();
        assert_eq!(size, 7_389_759_032);
        assert_eq!(header.payload_start(), 258_144);
        assert_eq!(header.declared_file_size(), size);
        assert_eq!(header.source_file(), SHARD);
        for (projection, weight, scale) in [
            (
                ExpertProjection::Gate,
                594728408..600626648,
                8015576..8384216,
            ),
            (ExpertProjection::Up, 606524888..612423128, 8752856..9121496),
            (
                ExpertProjection::Down,
                600626648..606524888,
                8384216..8752856,
            ),
        ] {
            let pair = bind_expert_pair(
                &config(ExpertGeometry::Base384),
                0,
                0,
                projection,
                &header,
                &header,
            )
            .unwrap();
            let (ws, ss) = projection.source_shapes();
            assert_eq!(pair.weight.descriptor.shape, ws);
            assert_eq!(pair.scale.descriptor.shape, ss);
            assert_eq!((pair.rows, pair.input_columns), projection.logical_shape());
            for (tensor, relative) in [(&pair.weight, weight), (&pair.scale, scale)] {
                assert_eq!(tensor.descriptor.relative_bytes, relative);
                assert_eq!(
                    tensor.descriptor.file_bytes,
                    relative.start + 258144..relative.end + 258144
                );
                assert_eq!(tensor.source_file, SHARD);
                let recorded = &manifest["tensors"][&tensor.key];
                assert_eq!(
                    recorded["descriptor"]["shape"],
                    json!(tensor.descriptor.shape)
                );
                assert_eq!(
                    recorded["source_byte_offsets"],
                    json!([
                        tensor.descriptor.file_bytes.start,
                        tensor.descriptor.file_bytes.end
                    ])
                );
            }
        }
        assert!(SourceHeader::parse(SHARD, HEADER, size - 1).is_err());
        assert!(SourceHeader::parse(SHARD, HEADER, size + 1).is_err());
    }

    #[test]
    fn rejects_bad_pairs_and_main_indices() {
        let good = parse(pair_value(), 6266880).unwrap();
        for geometry in [
            ExpertGeometry::Base384,
            ExpertGeometry::PotentialReap272,
            ExpertGeometry::PotentialReap256,
        ] {
            let cfg = config(geometry);
            assert!(bind_expert_pair(&cfg, 0, 0, ExpertProjection::Gate, &good, &good).is_ok());
            for (layer, expert) in [(40, 0), (0, 384), (0, geometry.expert_count())] {
                assert!(
                    bind_expert_pair(&cfg, layer, expert, ExpertProjection::Gate, &good, &good)
                        .is_err()
                );
            }
        }
        let cfg = config(ExpertGeometry::Base384);
        for (key, field, replacement) in [
            ("weight", "dtype", json!("F8_E4M3")),
            ("scale", "dtype", json!("I8")),
            ("weight", "shape", json!([2560, 2304])),
            ("scale", "shape", json!([160, 2304])),
        ] {
            let mut value = pair_value();
            value[format!("layers.0.ffn.experts.0.w1.{key}")][field] = replacement;
            let bad = parse(value, 6266880).unwrap();
            assert!(bind_expert_pair(&cfg, 0, 0, ExpertProjection::Gate, &bad, &bad).is_err());
        }
        for new_key in [
            "layers.0.ffn.experts.1.w1.scale",
            "layers.0.ffn.experts.0.w3.scale",
            "mtp.0.ffn.experts.0.w1.scale",
        ] {
            let mut value = pair_value();
            let obj = value.as_object_mut().unwrap();
            let scale = obj.remove("layers.0.ffn.experts.0.w1.scale").unwrap();
            obj.insert(new_key.into(), scale);
            let bad = parse(value, 6266880).unwrap();
            assert!(bind_expert_pair(&cfg, 0, 0, ExpertProjection::Gate, &good, &bad).is_err());
        }
    }

    #[test]
    fn split_shard_pair_preserves_source_offsets() {
        let weight_bytes = raw(&json!({
            "layers.0.ffn.experts.0.w1.weight": {
                "dtype":"I8", "shape":[2304,2560], "data_offsets":[0,5898240]
            }
        })
        .to_string());
        let scale_bytes = raw(&json!({
            "layers.0.ffn.experts.0.w1.scale": {
                "dtype":"F8_E8M0", "shape":[2304,160], "data_offsets":[0,368640]
            }
        })
        .to_string());
        let weight = SourceHeader::parse(
            "weight-shard",
            &weight_bytes,
            weight_bytes.len() as u64 + 5898240,
        )
        .unwrap();
        let scale = SourceHeader::parse(
            "scale-shard",
            &scale_bytes,
            scale_bytes.len() as u64 + 368640,
        )
        .unwrap();
        let pair = bind_expert_pair(
            &config(ExpertGeometry::Base384),
            0,
            0,
            ExpertProjection::Gate,
            &weight,
            &scale,
        )
        .unwrap();
        assert_eq!(pair.weight.source_file, "weight-shard");
        assert_eq!(pair.scale.source_file, "scale-shard");
        assert_eq!(
            pair.weight.descriptor.file_bytes.start,
            weight_bytes.len() as u64
        );
        assert_eq!(
            pair.scale.descriptor.file_bytes.start,
            scale_bytes.len() as u64
        );
        assert!(
            bind_expert_pair(
                &config(ExpertGeometry::Base384),
                0,
                0,
                ExpertProjection::Gate,
                &scale,
                &weight
            )
            .is_err()
        );
    }

    #[test]
    fn observed_dtype_widths_are_checked() {
        for (dtype, width) in [
            ("BF16", 2),
            ("F32", 4),
            ("I8", 1),
            ("F8_E4M3", 1),
            ("F8_E8M0", 1),
        ] {
            let value = json!({"x":{"dtype":dtype,"shape":[3],"data_offsets":[0,3*width]}});
            let header = parse(value, 3 * width).unwrap();
            assert_eq!(header.descriptor("x").unwrap().dtype.byte_width(), width);
        }
    }

    #[test]
    fn rejects_malformed_descriptors_and_metadata() {
        let valid = json!({"x":{"dtype":"I8","shape":[8],"data_offsets":[0,8]}});
        for (field, value) in [
            ("dtype", json!("F16")),
            ("dtype", json!(null)),
            ("shape", json!([7])),
            ("shape", json!([-1])),
            ("shape", json!([1.0])),
            ("shape", json!([u64::MAX, 2])),
            ("data_offsets", json!([1, 9])),
            ("data_offsets", json!([8, 0])),
            ("data_offsets", json!([0])),
            ("data_offsets", json!([false, 8])),
            ("extra", json!(0)),
        ] {
            let mut bad = valid.clone();
            bad["x"][field] = value;
            assert!(parse(bad, 8).is_err(), "{field}");
        }
        let mut overlap = valid.clone();
        overlap["y"] = overlap["x"].clone();
        assert!(parse(overlap, 8).is_err());
        for metadata in [json!(null), json!([]), json!({"format":1})] {
            let mut bad = valid.clone();
            bad["__metadata__"] = metadata;
            assert!(parse(bad, 8).is_err());
        }
        let mut good = valid;
        good["__metadata__"] = json!({"format":"pt"});
        assert!(parse(good, 8).is_ok());
        let bytes = raw(
            &json!({"x":{"dtype":"I8","shape":[8],"data_offsets":[u64::MAX-8,u64::MAX]}})
                .to_string(),
        );
        assert!(
            SourceHeader::parse("overflow", &bytes, u64::MAX)
                .unwrap_err()
                .contains("overflow")
        );
    }

    #[test]
    fn rejects_duplicate_keys_and_bad_framing() {
        for text in [
            r#"{"x":{},"x":{}}"#,
            r#"{"x":{"dtype":"I8","dtype":"I8","shape":[0],"data_offsets":[0,0]}}"#,
            r#"{"__metadata__":{"a":"b","a":"c"}}"#,
        ] {
            let bytes = raw(text);
            assert!(SourceHeader::parse("duplicate", &bytes, bytes.len() as u64).is_err());
        }
        for length in [0, 7, MAX_HEADER_BYTES + 8, u64::MAX] {
            assert!(SourceHeader::parse("bad", &length.to_le_bytes(), u64::MAX).is_err());
        }
        let bytes = raw("{}");
        for end in [0, 7, 8, bytes.len() - 1] {
            assert!(SourceHeader::parse("truncated", &bytes[..end], 16).is_err());
        }
        assert!(SourceHeader::parse("short file", &bytes, 15).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(SourceHeader::parse("extra bytes", &extra, 17).is_err());
        assert!(SourceHeader::parse("empty", &bytes, 16).is_ok());
    }
}

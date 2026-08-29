use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const BF16_BYTES: usize = 2;
const I64_BYTES: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TemplateScore {
    pub row_id: usize,
    pub word_id: i64,
    pub text: String,
    pub score: f32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TemplateLensSummary {
    pub path: PathBuf,
    pub model_id: Option<String>,
    pub dtype: String,
    pub layers: Vec<u32>,
    pub rows: usize,
    pub hidden_size: usize,
    pub tensor_bytes: usize,
    pub metadata: BTreeMap<String, String>,
}

pub struct TemplateVocabulary {
    labels: Vec<String>,
}

pub struct TemplateLens {
    path: PathBuf,
    file: File,
    mmap: Mmap,
    metadata: BTreeMap<String, String>,
    layers: Vec<u32>,
    word_ids: Vec<i64>,
    n_rows: usize,
    hidden_size: usize,
    templates_start: usize,
    templates_bytes: usize,
}

#[derive(Debug)]
pub enum TemplateError {
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    Map {
        path: PathBuf,
        source: std::io::Error,
    },
    HeaderJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    Invalid {
        path: PathBuf,
        detail: String,
    },
    LabelsRead {
        path: PathBuf,
        source: std::io::Error,
    },
    LabelsInvalid {
        path: PathBuf,
        detail: String,
    },
    MissingLayer {
        layer: u32,
        available: Vec<u32>,
    },
    MissingRow {
        row_id: usize,
        rows: usize,
    },
    ActivationWidth {
        got: usize,
        expected: usize,
    },
    VocabularyRows {
        got: usize,
        expected: usize,
    },
    InvalidActivationNorm,
    EmptyTopK,
    BackingFileChanged,
    NoFiniteTemplates,
    NonFiniteTemplateRow {
        layer: u32,
        row_id: usize,
        element: usize,
    },
    MissingLabel {
        label: String,
    },
    AmbiguousLabel {
        label: String,
    },
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open { path, source } => {
                write!(
                    formatter,
                    "failed to open template lens {}: {source}",
                    path.display()
                )
            }
            Self::Map { path, source } => {
                write!(
                    formatter,
                    "failed to map template lens {}: {source}",
                    path.display()
                )
            }
            Self::HeaderJson { path, source } => write!(
                formatter,
                "failed to parse template lens header {}: {source}",
                path.display()
            ),
            Self::Invalid { path, detail } => {
                write!(
                    formatter,
                    "invalid template lens {}: {detail}",
                    path.display()
                )
            }
            Self::LabelsRead { path, source } => write!(
                formatter,
                "failed to read template labels {}: {source}",
                path.display()
            ),
            Self::LabelsInvalid { path, detail } => {
                write!(
                    formatter,
                    "invalid template labels {}: {detail}",
                    path.display()
                )
            }
            Self::MissingLayer { layer, available } => write!(
                formatter,
                "template layer {layer} is absent; available layers are {available:?}"
            ),
            Self::MissingRow { row_id, rows } => {
                write!(
                    formatter,
                    "template row {row_id} is out of range for {rows} rows"
                )
            }
            Self::ActivationWidth { got, expected } => {
                write!(
                    formatter,
                    "activation width {got} != template width {expected}"
                )
            }
            Self::VocabularyRows { got, expected } => write!(
                formatter,
                "template vocabulary row count {got} != template row count {expected}"
            ),
            Self::InvalidActivationNorm => {
                formatter.write_str("activation norm is zero or non-finite")
            }
            Self::EmptyTopK => formatter.write_str("top_k must be nonzero"),
            Self::BackingFileChanged => {
                formatter.write_str("template lens backing file changed while mapped")
            }
            Self::NoFiniteTemplates => {
                formatter.write_str("template layer has no finite, nonzero template rows")
            }
            Self::NonFiniteTemplateRow {
                layer,
                row_id,
                element,
            } => write!(
                formatter,
                "template layer {layer} row {row_id} has a non-finite value at element {element}"
            ),
            Self::MissingLabel { label } => {
                write!(formatter, "template label {label:?} is absent")
            }
            Self::AmbiguousLabel { label } => {
                write!(formatter, "template label {label:?} is duplicated")
            }
        }
    }
}

impl std::error::Error for TemplateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. }
            | Self::Map { source, .. }
            | Self::LabelsRead { source, .. } => Some(source),
            Self::HeaderJson { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsHeader {
    #[serde(rename = "__metadata__", default)]
    metadata: BTreeMap<String, String>,
    layers: TensorHeader,
    templates: TensorHeader,
    word_ids: TensorHeader,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl TemplateVocabulary {
    pub fn load(path: &Path, expected_rows: usize) -> Result<Self, TemplateError> {
        let text = std::fs::read_to_string(path).map_err(|source| TemplateError::LabelsRead {
            path: path.to_path_buf(),
            source,
        })?;
        let mut labels = Vec::with_capacity(expected_rows);
        for (line_index, line) in text.lines().enumerate() {
            let (id, label) =
                line.split_once('\t')
                    .ok_or_else(|| TemplateError::LabelsInvalid {
                        path: path.to_path_buf(),
                        detail: format!("line {} has no tab separator", line_index + 1),
                    })?;
            let id: usize = id.parse().map_err(|_| TemplateError::LabelsInvalid {
                path: path.to_path_buf(),
                detail: format!("line {} has invalid row id {id:?}", line_index + 1),
            })?;
            if id != labels.len() {
                return Err(TemplateError::LabelsInvalid {
                    path: path.to_path_buf(),
                    detail: format!("expected row id {}, got {id}", labels.len()),
                });
            }
            if label.is_empty() {
                return Err(TemplateError::LabelsInvalid {
                    path: path.to_path_buf(),
                    detail: format!("row {id} has an empty label"),
                });
            }
            labels.push(label.to_owned());
        }
        if labels.len() != expected_rows {
            return Err(TemplateError::LabelsInvalid {
                path: path.to_path_buf(),
                detail: format!(
                    "label count {} != template row count {expected_rows}",
                    labels.len()
                ),
            });
        }
        Ok(Self { labels })
    }

    pub fn label(&self, row_id: usize) -> Option<&str> {
        self.labels.get(row_id).map(String::as_str)
    }

    pub fn unique_row_id_for_label(&self, label: &str) -> Result<usize, TemplateError> {
        let mut matches = self
            .labels
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.as_str() == label);
        let Some((row_id, _)) = matches.next() else {
            return Err(TemplateError::MissingLabel {
                label: label.to_owned(),
            });
        };
        if matches.next().is_some() {
            return Err(TemplateError::AmbiguousLabel {
                label: label.to_owned(),
            });
        }
        Ok(row_id)
    }
}

impl TemplateLens {
    pub fn open(path: &Path) -> Result<Self, TemplateError> {
        let file = File::open(path).map_err(|source| TemplateError::Open {
            path: path.to_path_buf(),
            source,
        })?;
        if !file
            .metadata()
            .map_err(|source| TemplateError::Open {
                path: path.to_path_buf(),
                source,
            })?
            .is_file()
        {
            return Err(invalid(path, "backing path is not a regular file"));
        }

        // SAFETY: the map is read-only, and `file` is retained for the lifetime of
        // the map. Public reads also reject a changed backing-file length.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| TemplateError::Map {
            path: path.to_path_buf(),
            source,
        })?;
        if mmap.len() < I64_BYTES {
            return Err(invalid(path, "file is shorter than the safetensors prefix"));
        }
        let header_bytes = u64::from_le_bytes(mmap[..I64_BYTES].try_into().unwrap());
        let header_bytes = usize::try_from(header_bytes)
            .map_err(|_| invalid(path, "header length does not fit usize"))?;
        if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
            return Err(invalid(
                path,
                format!("header length {header_bytes} is outside the supported range"),
            ));
        }
        let data_start = I64_BYTES
            .checked_add(header_bytes)
            .ok_or_else(|| invalid(path, "header endpoint overflow"))?;
        if data_start > mmap.len() {
            return Err(invalid(path, "header extends beyond the file"));
        }
        let header: SafetensorsHeader = serde_json::from_slice(&mmap[I64_BYTES..data_start])
            .map_err(|source| TemplateError::HeaderJson {
                path: path.to_path_buf(),
                source,
            })?;

        validate_tensor(
            path,
            &header.layers,
            "layers",
            "I64",
            I64_BYTES,
            data_start,
            mmap.len(),
        )?;
        validate_tensor(
            path,
            &header.word_ids,
            "word_ids",
            "I64",
            I64_BYTES,
            data_start,
            mmap.len(),
        )?;
        validate_tensor(
            path,
            &header.templates,
            "templates",
            "BF16",
            BF16_BYTES,
            data_start,
            mmap.len(),
        )?;

        let mut ranges = [
            (header.layers.data_offsets, "layers"),
            (header.word_ids.data_offsets, "word_ids"),
            (header.templates.data_offsets, "templates"),
        ];
        ranges.sort_unstable_by_key(|(offsets, _)| offsets[0]);
        let mut cursor = 0usize;
        for ([start, end], name) in ranges {
            if start != cursor {
                return Err(invalid(
                    path,
                    format!(
                        "safetensors data is not contiguous before {name}: expected offset {cursor}, got {start}"
                    ),
                ));
            }
            cursor = end;
        }
        if data_start.checked_add(cursor) != Some(mmap.len()) {
            return Err(invalid(
                path,
                "safetensors data does not consume the complete file",
            ));
        }

        let [n_layers, n_rows, hidden_size] = header.templates.shape.as_slice() else {
            return Err(invalid(
                path,
                "templates must have shape [layers, rows, hidden]",
            ));
        };
        if header.layers.shape != [*n_layers] {
            return Err(invalid(
                path,
                format!(
                    "layers shape {:?} does not match template layer count {n_layers}",
                    header.layers.shape
                ),
            ));
        }
        if header.word_ids.shape != [*n_rows] {
            return Err(invalid(
                path,
                format!(
                    "word_ids shape {:?} does not match template row count {n_rows}",
                    header.word_ids.shape
                ),
            ));
        }

        let layers_i64 = read_i64_tensor(&mmap, data_start, &header.layers);
        let mut layer_set = BTreeSet::new();
        let mut layers = Vec::with_capacity(*n_layers);
        for layer in layers_i64 {
            let layer = u32::try_from(layer)
                .map_err(|_| invalid(path, format!("layer id {layer} is outside u32")))?;
            if !layer_set.insert(layer) {
                return Err(invalid(path, format!("layer id {layer} is duplicated")));
            }
            layers.push(layer);
        }
        let word_ids = read_i64_tensor(&mmap, data_start, &header.word_ids);
        let templates_start = data_start
            .checked_add(header.templates.data_offsets[0])
            .ok_or_else(|| invalid(path, "template offset overflow"))?;
        let templates_bytes = header.templates.data_offsets[1] - header.templates.data_offsets[0];

        Ok(Self {
            path: path.to_path_buf(),
            file,
            mmap,
            metadata: header.metadata,
            layers,
            word_ids,
            n_rows: *n_rows,
            hidden_size: *hidden_size,
            templates_start,
            templates_bytes,
        })
    }

    pub fn layers(&self) -> &[u32] {
        &self.layers
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn model_id(&self) -> Option<&str> {
        self.metadata.get("model_id").map(String::as_str)
    }

    pub fn summary(&self) -> TemplateLensSummary {
        TemplateLensSummary {
            path: self.path.clone(),
            model_id: self.model_id().map(str::to_owned),
            dtype: "bf16".to_owned(),
            layers: self.layers.clone(),
            rows: self.n_rows,
            hidden_size: self.hidden_size,
            tensor_bytes: self.templates_bytes,
            metadata: self.metadata.clone(),
        }
    }

    pub fn score(
        &self,
        layer: u32,
        activation: &[f32],
        vocabulary: &TemplateVocabulary,
        top_k: usize,
    ) -> Result<Vec<TemplateScore>, TemplateError> {
        if top_k == 0 {
            return Err(TemplateError::EmptyTopK);
        }
        if activation.len() != self.hidden_size {
            return Err(TemplateError::ActivationWidth {
                got: activation.len(),
                expected: self.hidden_size,
            });
        }
        if vocabulary.labels.len() != self.n_rows {
            return Err(TemplateError::VocabularyRows {
                got: vocabulary.labels.len(),
                expected: self.n_rows,
            });
        }
        self.check_backing_file()?;
        let layer_slot = self.layer_slot(layer)?;
        let activation_norm = activation
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>()
            .sqrt();
        if !activation_norm.is_finite() || activation_norm == 0.0 {
            return Err(TemplateError::InvalidActivationNorm);
        }

        let layer_elements = self
            .n_rows
            .checked_mul(self.hidden_size)
            .expect("validated template dimensions");
        let layer_start = self.templates_start + layer_slot * layer_elements * BF16_BYTES;
        let layer_bytes = &self.mmap[layer_start..layer_start + layer_elements * BF16_BYTES];
        let mut ranked: Vec<(usize, f64)> = (0..self.n_rows)
            .into_par_iter()
            .filter_map(|row_id| {
                let row_start = row_id * self.hidden_size * BF16_BYTES;
                let bytes = &layer_bytes[row_start..row_start + self.hidden_size * BF16_BYTES];
                let mut dot = 0.0f64;
                let mut norm_squared = 0.0f64;
                for (index, &activation_value) in activation.iter().enumerate() {
                    let template_value = f64::from(decode_bf16(bytes, index));
                    dot += f64::from(activation_value) * template_value;
                    norm_squared += template_value * template_value;
                }
                let score = dot / (activation_norm * norm_squared.sqrt());
                score
                    .is_finite()
                    .then_some((row_id, score.clamp(-1.0, 1.0)))
            })
            .collect();
        if ranked.is_empty() {
            return Err(TemplateError::NoFiniteTemplates);
        }
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(top_k.min(ranked.len()));

        Ok(ranked
            .into_iter()
            .map(|(row_id, score)| TemplateScore {
                row_id,
                word_id: self.word_ids[row_id],
                text: vocabulary.labels[row_id].clone(),
                score: score as f32,
            })
            .collect())
    }

    pub fn row_f32(&self, layer: u32, row_id: usize) -> Result<Vec<f32>, TemplateError> {
        if row_id >= self.n_rows {
            return Err(TemplateError::MissingRow {
                row_id,
                rows: self.n_rows,
            });
        }
        self.check_backing_file()?;
        let layer_slot = self.layer_slot(layer)?;
        let row_elements = layer_slot
            .checked_mul(self.n_rows)
            .and_then(|offset| offset.checked_add(row_id))
            .and_then(|offset| offset.checked_mul(self.hidden_size))
            .expect("validated template dimensions");
        let row_start = self.templates_start + row_elements * BF16_BYTES;
        let bytes = &self.mmap[row_start..row_start + self.hidden_size * BF16_BYTES];
        let mut row = Vec::with_capacity(self.hidden_size);
        for element in 0..self.hidden_size {
            let value = decode_bf16(bytes, element);
            if !value.is_finite() {
                return Err(TemplateError::NonFiniteTemplateRow {
                    layer,
                    row_id,
                    element,
                });
            }
            row.push(value);
        }
        Ok(row)
    }

    fn check_backing_file(&self) -> Result<(), TemplateError> {
        if self
            .file
            .metadata()
            .map(|metadata| metadata.len() as usize != self.mmap.len())
            .unwrap_or(true)
        {
            return Err(TemplateError::BackingFileChanged);
        }
        Ok(())
    }

    fn layer_slot(&self, layer: u32) -> Result<usize, TemplateError> {
        self.layers
            .iter()
            .position(|&candidate| candidate == layer)
            .ok_or_else(|| TemplateError::MissingLayer {
                layer,
                available: self.layers.clone(),
            })
    }
}

fn validate_tensor(
    path: &Path,
    tensor: &TensorHeader,
    name: &str,
    dtype: &str,
    element_bytes: usize,
    data_start: usize,
    file_len: usize,
) -> Result<(), TemplateError> {
    if tensor.dtype != dtype {
        return Err(invalid(
            path,
            format!("{name} dtype {:?} != {dtype}", tensor.dtype),
        ));
    }
    if tensor.shape.is_empty() || tensor.shape.contains(&0) {
        return Err(invalid(path, format!("{name} has an empty or zero shape")));
    }
    let expected_bytes = tensor
        .shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| invalid(path, format!("{name} size overflow")))?;
    let [start, end] = tensor.data_offsets;
    if start > end || end - start != expected_bytes {
        return Err(invalid(
            path,
            format!("{name} offsets {start}..{end} do not match {expected_bytes} bytes"),
        ));
    }
    let physical_end = data_start
        .checked_add(end)
        .ok_or_else(|| invalid(path, format!("{name} endpoint overflow")))?;
    if physical_end > file_len {
        return Err(invalid(path, format!("{name} extends beyond the file")));
    }
    Ok(())
}

fn read_i64_tensor(mmap: &[u8], data_start: usize, tensor: &TensorHeader) -> Vec<i64> {
    let start = data_start + tensor.data_offsets[0];
    let end = data_start + tensor.data_offsets[1];
    mmap[start..end]
        .chunks_exact(I64_BYTES)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn decode_bf16(bytes: &[u8], element: usize) -> f32 {
    let byte_index = element * BF16_BYTES;
    let bits = u16::from_le_bytes([bytes[byte_index], bytes[byte_index + 1]]);
    f32::from_bits(u32::from(bits) << 16)
}

fn invalid(path: &Path, detail: impl Into<String>) -> TemplateError {
    TemplateError::Invalid {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixture_path(suffix: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "qwen-template-lens-test-{}-{}-{suffix}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn bf16(value: f32) -> [u8; BF16_BYTES] {
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }

    fn write_lens_fixture(
        path: &Path,
        layers: &[i64],
        word_ids: &[i64],
        hidden_size: usize,
        templates: &[f32],
    ) {
        assert_eq!(templates.len(), layers.len() * word_ids.len() * hidden_size);
        let layers_end = layers.len() * I64_BYTES;
        let word_ids_end = layers_end + word_ids.len() * I64_BYTES;
        let templates_end = word_ids_end + templates.len() * BF16_BYTES;
        let header = json!({
            "__metadata__": {"model_id": "fixture/model", "space": "raw"},
            "layers": {
                "dtype": "I64",
                "shape": [layers.len()],
                "data_offsets": [0, layers_end]
            },
            "word_ids": {
                "dtype": "I64",
                "shape": [word_ids.len()],
                "data_offsets": [layers_end, word_ids_end]
            },
            "templates": {
                "dtype": "BF16",
                "shape": [layers.len(), word_ids.len(), hidden_size],
                "data_offsets": [word_ids_end, templates_end]
            }
        });
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(I64_BYTES) {
            header.push(b' ');
        }
        let mut bytes = Vec::with_capacity(I64_BYTES + header.len() + templates_end);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for &layer in layers {
            bytes.extend_from_slice(&layer.to_le_bytes());
        }
        for &word_id in word_ids {
            bytes.extend_from_slice(&word_id.to_le_bytes());
        }
        for &value in templates {
            bytes.extend_from_slice(&bf16(value));
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn parses_shapes_layers_rows_and_extracts_layer_major_rows() {
        let tensor_path = fixture_path("shape.safetensors");
        write_lens_fixture(
            &tensor_path,
            &[3, 7],
            &[10, 20, 30],
            3,
            &[
                1.0, 0.0, 0.0, // layer 3, row 0
                0.0, 1.0, 0.0, // layer 3, row 1
                -1.0, 0.0, 0.0, // layer 3, row 2
                0.0, 0.0, 1.0, // layer 7, row 0
                1.0, 1.0, 0.0, // layer 7, row 1
                0.0, -1.0, 0.0, // layer 7, row 2
            ],
        );

        let lens = TemplateLens::open(&tensor_path).unwrap();
        assert_eq!(lens.layers(), &[3, 7]);
        assert_eq!(lens.n_rows(), 3);
        assert_eq!(lens.hidden_size(), 3);
        assert_eq!(lens.model_id(), Some("fixture/model"));
        assert_eq!(lens.row_f32(7, 1).unwrap(), vec![1.0, 1.0, 0.0]);
        assert!(matches!(
            lens.row_f32(5, 0),
            Err(TemplateError::MissingLayer { layer: 5, .. })
        ));

        std::fs::remove_file(tensor_path).unwrap();
    }

    #[test]
    fn scores_positive_orthogonal_and_negative_rows_using_phrase_labels() {
        let tensor_path = fixture_path("score.safetensors");
        let labels_path = fixture_path("labels.tsv");
        write_lens_fixture(
            &tensor_path,
            &[11],
            &[101, 202, 303],
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        );
        std::fs::write(
            &labels_path,
            "0\tice cream\n1\tNew Zealand\n2\twhat this means\n",
        )
        .unwrap();

        let lens = TemplateLens::open(&tensor_path).unwrap();
        let vocabulary = TemplateVocabulary::load(&labels_path, lens.n_rows()).unwrap();
        let scores = lens.score(11, &[1.0, 0.0, 0.0], &vocabulary, 3).unwrap();
        assert_eq!(
            scores.iter().map(|score| score.row_id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(scores[0].text, "ice cream");
        assert_eq!(scores[0].word_id, 101);
        assert_eq!(scores[0].score, 1.0);
        assert_eq!(scores[1].text, "New Zealand");
        assert_eq!(scores[1].score, 0.0);
        assert_eq!(scores[2].text, "what this means");
        assert_eq!(scores[2].score, -1.0);

        std::fs::remove_file(tensor_path).unwrap();
        std::fs::remove_file(labels_path).unwrap();
    }

    #[test]
    fn exact_label_lookup_rejects_absent_and_duplicate_labels() {
        let labels_path = fixture_path("ambiguous-labels.tsv");
        std::fs::write(
            &labels_path,
            "0\ta phrase with spaces\n1\tduplicate\n2\tduplicate\n",
        )
        .unwrap();

        let vocabulary = TemplateVocabulary::load(&labels_path, 3).unwrap();
        assert_eq!(vocabulary.label(0), Some("a phrase with spaces"));
        assert_eq!(vocabulary.label(3), None);
        assert_eq!(
            vocabulary
                .unique_row_id_for_label("a phrase with spaces")
                .unwrap(),
            0
        );
        assert!(matches!(
            vocabulary.unique_row_id_for_label("absent"),
            Err(TemplateError::MissingLabel { .. })
        ));
        assert!(matches!(
            vocabulary.unique_row_id_for_label("duplicate"),
            Err(TemplateError::AmbiguousLabel { .. })
        ));

        std::fs::remove_file(labels_path).unwrap();
    }

    #[test]
    fn row_extraction_rejects_non_finite_values_but_allows_zero_norm() {
        let tensor_path = fixture_path("nonfinite.safetensors");
        write_lens_fixture(
            &tensor_path,
            &[2],
            &[10, 20],
            2,
            &[0.0, 0.0, f32::INFINITY, 1.0],
        );

        let lens = TemplateLens::open(&tensor_path).unwrap();
        assert_eq!(lens.row_f32(2, 0).unwrap(), vec![0.0, 0.0]);
        assert!(matches!(
            lens.row_f32(2, 1),
            Err(TemplateError::NonFiniteTemplateRow {
                layer: 2,
                row_id: 1,
                element: 0
            })
        ));

        std::fs::remove_file(tensor_path).unwrap();
    }
}

//! Native and published lens loading and row preparation.

use super::*;

pub(super) struct NativeLens {
    pub(super) method: String,
    pub(super) target_layer: u32,
    pub(super) source_layers: Vec<u32>,
    pub(super) token_ids: Vec<i32>,
    pub(super) hidden_size: usize,
    pub(super) values: Vec<f32>,
}

pub(super) fn load_native_lens(
    artifact: &Path,
    model_layers: u32,
    hidden_size: usize,
    vocab_size: u32,
) -> Result<NativeLens> {
    let manifest_path = if artifact.is_dir() {
        artifact.join("readouts.json")
    } else {
        artifact.to_path_buf()
    };
    let manifest: crate::TokenReadoutManifest = crate::read_json_file(&manifest_path)?;
    ensure!(
        manifest.schema == crate::TOKEN_READOUT_SCHEMA
            && manifest.schema_version == crate::SCHEMA_VERSION
            && manifest.status == "complete",
        "native lens {} is not a completed fit-tokens artifact",
        manifest_path.display()
    );
    let method = match manifest.config.method {
        crate::FitMethod::J => "J",
        crate::FitMethod::R => "R",
    };
    ensure!(
        manifest.config.orientation == crate::TOKEN_ORIENTATION
            && manifest.readouts == manifest.config.readouts
            && manifest.readouts.target_covectors_dtype == "f32_le",
        "native lens {} has unsupported readout metadata",
        manifest_path.display()
    );
    crate::validate_token_readout_spec(&manifest.readouts, &manifest.config)?;
    ensure!(
        manifest.config.n_layers == model_layers
            && manifest.config.hidden_size as usize == hidden_size
            && manifest.config.vocab_size == vocab_size,
        "native lens {} model geometry does not match the loaded model",
        manifest_path.display()
    );
    ensure!(
        manifest.config.target_layer < model_layers,
        "native lens target layer is outside model geometry"
    );
    let source_layers = manifest.config.source_layers.clone();
    ensure!(
        !source_layers.is_empty(),
        "native lens has no source layers"
    );
    ensure!(
        source_layers.windows(2).all(|pair| pair[0] < pair[1])
            && source_layers.iter().all(|&layer| layer < model_layers),
        "native lens source layers are not sorted or are out of bounds"
    );
    let token_ids = manifest
        .readouts
        .token_ids
        .iter()
        .map(|&id| id as i32)
        .collect::<Vec<_>>();
    ensure!(!token_ids.is_empty(), "native lens has no selected tokens");
    ensure!(
        token_ids
            .iter()
            .all(|&id| id >= 0 && (id as u32) < vocab_size)
            && token_ids.iter().copied().collect::<HashSet<_>>().len() == token_ids.len(),
        "native lens selected token IDs are invalid or duplicated"
    );
    let expected_shape = [source_layers.len(), token_ids.len(), hidden_size];
    ensure!(
        manifest.payload.path == crate::TOKEN_PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == expected_shape,
        "native lens payload shape or descriptor is not supported"
    );
    let expected_values = expected_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .context("native lens payload size overflow")?;
    let expected_bytes = expected_values
        .checked_mul(4)
        .context("native lens byte size overflow")?;
    ensure!(
        expected_bytes <= crate::TOKEN_ARTIFACT_MAX_BYTES
            && manifest.payload.byte_length == expected_bytes as u64,
        "native lens payload byte length does not match its shape"
    );
    let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let payload_path = directory.join(&manifest.payload.path);
    let bytes = crate::read_regular_file_exact(&payload_path, expected_bytes)?;
    let mut values = Vec::with_capacity(expected_values);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        ensure!(
            value.is_finite(),
            "native lens payload contains a non-finite value"
        );
        values.push(value);
    }
    Ok(NativeLens {
        method: method.into(),
        target_layer: manifest.config.target_layer,
        source_layers,
        token_ids,
        hidden_size,
        values,
    })
}

pub(super) fn native_lens_from_projected_full(
    projected: crate::full_lens::ProjectedFullTokenDirections,
) -> NativeLens {
    NativeLens {
        method: projected.method,
        target_layer: projected.target_layer,
        source_layers: projected.source_layers,
        token_ids: projected.token_ids,
        hidden_size: projected.hidden_size,
        values: projected.values,
    }
}

pub(super) fn lens_row(
    prepared: &PreparedLens,
    selector: &DirectionRow,
    layer: u32,
) -> Result<Vec<f32>> {
    match (&prepared.lens, selector) {
        (LoadedLens::Native(native), DirectionRow::TokenId { .. }) => {
            native_lens_row(native, selector, layer, &prepared.id)
        }
        (
            LoadedLens::Template {
                lens,
                vocabulary: _,
            },
            DirectionRow::TemplateRowId { template_row_id },
        ) => lens
            .row_f32(layer, *template_row_id)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        (LoadedLens::Template { lens, vocabulary }, DirectionRow::Label { label }) => {
            let row_id = vocabulary
                .unique_row_id_for_label(label)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            lens.row_f32(layer, row_id)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        }
        (LoadedLens::Native(_), _) => bail!("native selected directions require row.kind=token_id"),
        (LoadedLens::Template { .. }, DirectionRow::TokenId { .. }) => {
            bail!("workspace-template directions require row.kind=template_row_id or label")
        }
    }
}

pub(super) fn native_lens_row(
    native: &NativeLens,
    selector: &DirectionRow,
    layer: u32,
    lens_id: &str,
) -> Result<Vec<f32>> {
    let DirectionRow::TokenId { token_id } = selector else {
        bail!("native selected directions require row.kind=token_id")
    };
    let layer_slot = native
        .source_layers
        .iter()
        .position(|&candidate| candidate == layer)
        .with_context(|| format!("native lens {lens_id} has no row for layer {layer}"))?;
    let token_slot = native
        .token_ids
        .iter()
        .position(|&candidate| candidate == *token_id)
        .with_context(|| format!("native lens {lens_id} has no token row {token_id}"))?;
    let offset = native_payload_offset(
        layer_slot,
        token_slot,
        native.token_ids.len(),
        native.hidden_size,
    );
    Ok(native.values[offset..offset + native.hidden_size].to_vec())
}

pub(super) fn native_payload_offset(
    layer_slot: usize,
    token_slot: usize,
    token_count: usize,
    hidden_size: usize,
) -> usize {
    (layer_slot * token_count + token_slot) * hidden_size
}

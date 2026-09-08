//! Published token-direction banks with tiled projection.

use super::*;

pub(crate) struct ProjectedFullTokenDirections {
    pub(crate) producer_metadata: Option<serde_json::Value>,
    pub(crate) method: String,
    pub(crate) target_layer: u32,
    pub(crate) source_layers: Vec<u32>,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) hidden_size: usize,
    pub(crate) values: Vec<f32>,
}

pub(crate) fn projected_full_token_direction_retained_bytes(
    source_layer_count: usize,
    token_count: usize,
    hidden_size: usize,
) -> Result<usize> {
    let value_bytes = source_layer_count
        .checked_mul(token_count)
        .and_then(|values| values.checked_mul(hidden_size))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("published full transport projected direction byte count overflow")?;
    let page_size = host_page_size_bytes().context("query host page size")?;
    ensure!(page_size > 0, "host page size must be positive");
    [
        value_bytes,
        token_count
            .checked_mul(std::mem::size_of::<i32>())
            .context("published full transport token ID byte count overflow")?,
        source_layer_count
            .checked_mul(std::mem::size_of::<u32>())
            .context("published full transport source layer byte count overflow")?,
        128,
    ]
    .into_iter()
    .try_fold(0usize, |total, logical_bytes| {
        let priced = logical_bytes
            .checked_add(page_size - 1)
            .map(|bytes| bytes / page_size * page_size)
            .context("published full transport host allocation pricing overflow")?;
        total
            .checked_add(priced)
            .context("published full transport retained byte count overflow")
    })
}

pub(crate) fn project_full_token_directions(
    artifact: &Path,
    manifest: &mut BoundFullAccess,
    token_ids: &[u32],
    source_layers: &[u32],
    loaded: &qwen_llm::runtime::LoadedModel,
    target_covector: FullTokenTargetCovector,
    allow_unvalidated_transfer: bool,
) -> Result<ProjectedFullTokenDirections> {
    validate_artifact_directory(artifact, "published full transport lens")?;
    manifest.validate_loaded(loaded)?;
    manifest.acknowledge_transfer(allow_unvalidated_transfer)?;
    let arch = loaded.arch();
    ensure!(
        !token_ids.is_empty(),
        "published full transport lens requires selected token IDs"
    );
    let mut unique_tokens = BTreeSet::new();
    ensure!(
        token_ids.iter().all(|&token| token < arch.vocab_size
            && token <= i32::MAX as u32
            && unique_tokens.insert(token)),
        "published full transport token IDs must be unique and inside the model vocabulary"
    );
    drop(unique_tokens);
    ensure!(
        !source_layers.is_empty()
            && source_layers.iter().copied().collect::<BTreeSet<_>>().len() == source_layers.len()
            && (manifest.is_data() || source_layers.windows(2).all(|pair| pair[0] < pair[1]))
            && source_layers
                .iter()
                .all(|layer| manifest.transport.source_layers.contains(layer)),
        "full transport source layers must be nonempty, unique artifact layers in caller order"
    );

    let hidden_size = arch.hidden_size as usize;
    let projected_values = source_layers
        .len()
        .checked_mul(token_ids.len())
        .and_then(|value| value.checked_mul(hidden_size))
        .context("published full transport projected direction count overflow")?;
    let projected_bytes = projected_full_token_direction_retained_bytes(
        source_layers.len(),
        token_ids.len(),
        hidden_size,
    )?;
    ensure!(
        projected_bytes <= TOKEN_ARTIFACT_MAX_BYTES,
        "published full transport projected directions require {projected_bytes} bytes, exceeding retained result budget {TOKEN_ARTIFACT_MAX_BYTES}"
    );
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(1))
        .context("create published full transport projection sequence")?;
    let workspace_lens = loaded
        .passive_workspace_lens_session(&mut sequence)
        .context("open published full transport projection session")?;
    let caller_reserve_bytes = TOKEN_ARTIFACT_MAX_BYTES
        .checked_add(JSON_FILE_MAX_BYTES)
        .context("full-transport projection caller reserve overflow")?;
    let query_capacity = workspace_lens
        .f16_transport_readout_query_capacity(caller_reserve_bytes)
        .context("derive full-transport token projection tile")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(projected_values)
        .context("allocate published full transport projected directions")?;
    values.resize(projected_values, f32::NAN);
    let single_tile_selected = if token_ids.len() <= query_capacity {
        Some(match target_covector {
            FullTokenTargetCovector::DeployedLogitNumerator => workspace_lens
                .selected_token_readouts(token_ids)
                .context("derive deployed-model selected-token score covectors")?,
            FullTokenTargetCovector::RawLmHead => workspace_lens
                .selected_token_raw_lm_head_rows(token_ids)
                .context("derive deployed-model raw LM-head token covectors")?,
        })
    } else {
        None
    };

    for (layer_index, &layer) in source_layers.iter().enumerate() {
        let matrix = manifest.read_matrix(layer)?;
        let prepared_transport = workspace_lens
            .prepare_f16_transport_readouts(&matrix)
            .with_context(|| format!("prepare published full transport source layer {layer}"))?;
        for (tile_index, token_tile) in token_ids.chunks(query_capacity).enumerate() {
            let token_start = tile_index
                .checked_mul(query_capacity)
                .context("published full transport token tile offset overflow")?;
            let tile_selected = if single_tile_selected.is_none() {
                Some(match target_covector {
                    FullTokenTargetCovector::DeployedLogitNumerator => workspace_lens
                        .selected_token_readouts(token_tile)
                        .context("derive deployed-model selected-token score covectors")?,
                    FullTokenTargetCovector::RawLmHead => workspace_lens
                        .selected_token_raw_lm_head_rows(token_tile)
                        .context("derive deployed-model raw LM-head token covectors")?,
                })
            } else {
                None
            };
            let selected = single_tile_selected
                .as_ref()
                .or(tile_selected.as_ref())
                .expect("one selected-token tile is available");
            ensure!(
                selected.hidden_size == hidden_size && selected.token_ids == token_tile,
                "published full transport selected-token projection changed row order"
            );
            let projected = workspace_lens
                .project_prepared_f16_transport_readouts(&prepared_transport, selected)
                .with_context(|| {
                    format!("project published full transport source layer {layer}")
                })?;
            ensure!(
                projected.len() == token_tile.len() * hidden_size,
                "published full transport projection returned an invalid shape"
            );
            write_projected_full_token_tile(
                &mut values,
                layer_index,
                token_ids.len(),
                token_start,
                hidden_size,
                &projected,
            )?;
        }
    }
    ensure!(
        values.len() == projected_values && values.iter().all(|value| value.is_finite()),
        "published full transport projection returned invalid values"
    );

    Ok(ProjectedFullTokenDirections {
        producer_metadata: if manifest.is_data() {
            Some(manifest.readout_artifact(artifact)?)
        } else {
            None
        },
        method: if manifest.is_data() {
            manifest.transport.method.clone()
        } else {
            format!(
                "published_{}_{}",
                manifest.transport.method,
                match target_covector {
                    FullTokenTargetCovector::DeployedLogitNumerator => {
                        "selected_token_numerator"
                    }
                    FullTokenTargetCovector::RawLmHead => "raw_lm_head_token_direction",
                }
            )
        },
        target_layer: manifest.transport.target_layer,
        source_layers: source_layers.to_vec(),
        token_ids: token_ids.iter().map(|&token| token as i32).collect(),
        hidden_size,
        values,
    })
}

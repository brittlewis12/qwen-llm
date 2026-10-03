//! Lens plan schema, validation, and intervention lowering.

use super::*;
#[cfg(test)]
pub(crate) use crate::lens_intervention::action_requires_unit_l2;
pub(crate) use crate::lens_intervention::{
    Action, DirectionRow, DirectionTargetCovector, LensRowDirectionDefinition, OperationDefinition,
    OperationSite, normalize_direction, operation_enabled,
};

pub(super) const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;

pub(super) const MAX_DIRECTIONS: usize = 4096;

pub(super) const MAX_OPERATIONS: usize = 4096;

pub(super) const MAX_RENDERED_SELECTORS_PER_PLAN: usize = 1024;

pub(super) const MAX_RENDERED_SELECTOR_MATCH_WORK: usize = 1_000_000;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensPlan {
    pub(crate) version: u32,
    pub(crate) lenses: Vec<LensDefinition>,
    pub(crate) directions: Vec<DirectionDefinition>,
    pub(crate) operations: Vec<OperationDefinition>,
    #[serde(default)]
    pub(crate) readouts: Vec<ReadoutDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) direction_readouts: Vec<DirectionReadoutDefinition>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum LensDefinition {
    NativeSelected {
        id: String,
        artifact: PathBuf,
    },
    #[serde(
        rename = "published_full_transport",
        alias = "published_full_j",
        alias = "linear_transport"
    )]
    PublishedFullTransport {
        id: String,
        artifact: PathBuf,
        token_ids: Vec<u32>,
        allow_unvalidated_transfer: bool,
    },
    WorkspaceTemplate {
        id: String,
        weights: PathBuf,
        labels: PathBuf,
    },
}

impl LensDefinition {
    pub(super) fn id(&self) -> &str {
        match self {
            Self::NativeSelected { id, .. }
            | Self::PublishedFullTransport { id, .. }
            | Self::WorkspaceTemplate { id, .. } => id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum DirectionDefinition {
    LensRow(LensRowDirectionDefinition),
    NativeHyper(NativeHyperDirectionDefinition),
    // Last: untagged parsing tries the older shapes first, unchanged.
    Raw(RawDirectionDefinition),
}

/// Operator-supplied residual-coordinate vector(s). Only hidden size and layer
/// bounds are checked; nothing binds the payload to a model identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawDirectionDefinition {
    pub(crate) id: String,
    pub(crate) source: RawDirectionSource,
    pub(crate) normalization: Normalization,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum RawDirectionSource {
    RawResidualF32le {
        path: PathBuf,
        layout: RawDirectionLayout,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        layers: Option<Vec<u32>>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RawDirectionLayout {
    /// One vector usable at every layer.
    Shared,
    /// One vector per listed layer, layer-major in listed order.
    PerLayer,
}

impl RawDirectionSource {
    pub(super) fn parts(&self) -> (&Path, RawDirectionLayout, Option<&[u32]>) {
        match self {
            Self::RawResidualF32le {
                path,
                layout,
                layers,
            } => (path, *layout, layers.as_deref()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeHyperDirectionDefinition {
    pub(crate) id: String,
    pub(crate) source: NativeHyperDirectionSource,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum NativeHyperDirectionSource {
    NativeHyperF32 { path: PathBuf, layer: u32 },
}

impl DirectionDefinition {
    pub(super) fn id(&self) -> &str {
        match self {
            Self::LensRow(direction) => &direction.id,
            Self::NativeHyper(direction) => &direction.id,
            Self::Raw(direction) => &direction.id,
        }
    }

    pub(super) fn lens_row(&self) -> Option<&LensRowDirectionDefinition> {
        match self {
            Self::LensRow(direction) => Some(direction),
            Self::NativeHyper(_) | Self::Raw(_) => None,
        }
    }

    pub(super) fn native_hyper(&self) -> Option<&NativeHyperDirectionDefinition> {
        match self {
            Self::NativeHyper(direction) => Some(direction),
            Self::LensRow(_) | Self::Raw(_) => None,
        }
    }

    pub(super) fn raw(&self) -> Option<&RawDirectionDefinition> {
        match self {
            Self::Raw(direction) => Some(direction),
            Self::LensRow(_) | Self::NativeHyper(_) => None,
        }
    }

    pub(super) fn normalization(&self) -> Option<Normalization> {
        match self {
            Self::LensRow(direction) => Some(direction.normalization),
            Self::Raw(direction) => Some(direction.normalization),
            Self::NativeHyper(_) => None,
        }
    }
}

impl NativeHyperDirectionSource {
    pub(super) fn path_and_layer(&self) -> (&Path, u32) {
        match self {
            Self::NativeHyperF32 { path, layer } => (path, *layer),
        }
    }
}

pub(crate) fn bind_plan_positions(
    authored: &LensPlan,
    rendering: &LensInputRendering,
    prompt_len: usize,
) -> Result<BoundLensPlan> {
    validate_plan_position_selectors(authored)?;
    let rendered_selector_count = authored
        .operations
        .iter()
        .map(|operation| operation.scope.rendered_selector_count())
        .chain(
            authored
                .readouts
                .iter()
                .map(|readout| readout.scope.rendered_selector_count()),
        )
        .chain(
            authored
                .direction_readouts
                .iter()
                .map(|readout| readout.scope.rendered_selector_count()),
        )
        .try_fold(0usize, |total, count| total.checked_add(count))
        .context("rendered-selector count overflow")?;
    ensure!(
        rendered_selector_count <= MAX_RENDERED_SELECTORS_PER_PLAN,
        "Lens plan has {rendered_selector_count} rendered selectors; limit is {MAX_RENDERED_SELECTORS_PER_PLAN}"
    );
    if rendered_selector_count > 0 {
        let match_work = rendered_selector_count
            .checked_mul(rendering.spans.len())
            .context("rendered-selector match-work overflow")?;
        ensure!(
            match_work <= MAX_RENDERED_SELECTOR_MATCH_WORK,
            "rendered selectors require {match_work} span comparisons; limit is {MAX_RENDERED_SELECTOR_MATCH_WORK}"
        );
    }
    let mut resolved = authored.clone();
    let mut position_bindings = Vec::new();
    for operation in &mut resolved.operations {
        bind_scope_prefill(
            &mut operation.scope,
            "operation",
            &operation.id,
            rendering,
            prompt_len,
            &mut position_bindings,
        )?;
    }
    for readout in &mut resolved.readouts {
        bind_scope_prefill(
            &mut readout.scope,
            "readout",
            &readout.id,
            rendering,
            prompt_len,
            &mut position_bindings,
        )?;
    }
    for readout in &mut resolved.direction_readouts {
        bind_scope_prefill(
            &mut readout.scope,
            "direction_readout",
            &readout.id,
            rendering,
            prompt_len,
            &mut position_bindings,
        )?;
    }
    Ok(BoundLensPlan {
        authored: authored.clone(),
        resolved,
        authored_plan_canonical_json_blake3: canonical_plan_blake3(authored)?,
        position_bindings,
    })
}

impl Scope {
    pub(super) fn rendered_selector_count(&self) -> usize {
        match &self.prefill {
            Some(Selector::RenderedSpans { selectors }) => selectors.len(),
            _ => 0,
        }
    }
}

pub(super) fn validate_plan_position_selectors(plan: &LensPlan) -> Result<()> {
    ensure!(
        matches!(plan.version, 1 | 2),
        "Lens plan version must be 1 or 2"
    );
    ensure!(
        plan.operations.len() <= MAX_OPERATIONS
            && plan.readouts.len() <= MAX_READOUTS
            && plan.direction_readouts.len() <= MAX_READOUTS,
        "Lens plan has too many operations or readouts"
    );
    for operation in &plan.operations {
        operation
            .scope
            .validate(&format!("operation {} scope", operation.id), plan.version)?;
    }
    for readout in &plan.readouts {
        readout
            .scope
            .validate(&format!("readout {} scope", readout.id), plan.version)?;
    }
    for readout in &plan.direction_readouts {
        readout.scope.validate(
            &format!("direction readout {} scope", readout.id),
            plan.version,
        )?;
    }
    Ok(())
}

pub(super) fn bind_scope_prefill(
    scope: &mut Scope,
    owner_kind: &str,
    owner_id: &str,
    rendering: &LensInputRendering,
    prompt_len: usize,
    bindings: &mut Vec<PositionBinding>,
) -> Result<()> {
    let Some(Selector::RenderedSpans { selectors }) = scope.prefill.as_ref() else {
        return Ok(());
    };
    ensure!(
        !rendering.spans.is_empty(),
        "{owner_kind} {owner_id} rendered-span prefill selectors require renderer-authored spans; raw text and literal token IDs have none"
    );
    let selectors = selectors.clone();
    let mut values = BTreeSet::new();
    for (selector_index, selector) in selectors.into_iter().enumerate() {
        let mut match_count = 0usize;
        let mut first_match = None;
        let mut last_match = None;
        for matched in rendering
            .spans
            .iter()
            .enumerate()
            .filter(|(_, span)| selector.matches(span))
        {
            match_count += 1;
            first_match.get_or_insert(matched);
            last_match = Some(matched);
        }
        ensure!(
            match_count > 0,
            "{owner_kind} {owner_id} rendered selector {selector_index} matched no authored span"
        );
        let (rendering_span_index, span) = match selector.occurrence {
            RenderedSpanOccurrence::Unique => {
                ensure!(
                    match_count == 1,
                    "{owner_kind} {owner_id} rendered selector {selector_index} matched {} spans; choose first or last explicitly",
                    match_count
                );
                first_match.expect("nonempty matches")
            }
            RenderedSpanOccurrence::First => first_match.expect("nonempty matches"),
            RenderedSpanOccurrence::Last => last_match.expect("nonempty matches"),
        };
        let (token_start, token_end) = span
            .token_start
            .zip(span.token_end)
            .with_context(|| {
                format!(
                    "{owner_kind} {owner_id} rendered selector {selector_index} matched span {rendering_span_index} without an exact token range"
                )
            })?;
        ensure!(
            token_start < token_end && token_end <= prompt_len,
            "{owner_kind} {owner_id} rendered selector {selector_index} matched an invalid token range"
        );
        let position = match selector.edge {
            RenderedSpanEdge::Start => token_start,
            RenderedSpanEdge::End => token_end - 1,
        };
        let resolved_index =
            u32::try_from(position).context("resolved prefill position exceeds u32")?;
        ensure!(
            values.insert(resolved_index),
            "{owner_kind} {owner_id} rendered selectors resolve repeatedly to prefill position {position}"
        );
        bindings.push(PositionBinding {
            owner_kind: owner_kind.into(),
            owner_id: owner_id.into(),
            phase: "prefill".into(),
            selector_index,
            selector,
            rendering_span_index,
            matched_span: span.clone(),
            resolved_index,
            absolute_position: position,
        });
    }
    scope.prefill = Some(Selector::Values {
        values: values.into_iter().collect(),
    });
    Ok(())
}

pub(crate) fn canonical_plan_blake3(plan: &LensPlan) -> Result<String> {
    let bytes = serde_json::to_vec(plan).context("serialize canonical Lens plan JSON")?;
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "canonical Lens plan exceeds limit"
    );
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Clone, Debug)]
pub(crate) struct BoundLensPlan {
    pub(crate) authored: LensPlan,
    pub(crate) resolved: LensPlan,
    pub(crate) authored_plan_canonical_json_blake3: String,
    pub(crate) position_bindings: Vec<PositionBinding>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OperationApplication {
    pub(crate) id: String,
    pub(crate) layer: u32,
    pub(crate) phase: &'static str,
    pub(crate) index: usize,
    #[serde(skip_serializing_if = "OperationSite::is_post_block")]
    pub(crate) site: OperationSite,
}

pub(super) struct PreparedDirection {
    pub(super) rows: BTreeMap<u32, MetalTensor>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RawDirectionBinding {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    pub(crate) layout: RawDirectionLayout,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) layers: Option<Vec<u32>>,
    pub(crate) payload_bytes: u64,
    pub(crate) payload_blake3: String,
    pub(crate) semantics: &'static str,
}

pub(super) struct LoadedRawDirection {
    /// None for shared layout; otherwise the strictly increasing listed layers.
    layers: Option<Vec<u32>>,
    /// Normalized vectors, one per listed layer or one shared vector.
    rows: Vec<Vec<f32>>,
    pub(super) binding: RawDirectionBinding,
}

impl LoadedRawDirection {
    pub(super) fn row(&self, layer: u32) -> Option<&[f32]> {
        match &self.layers {
            None => self.rows.first().map(Vec::as_slice),
            Some(layers) => layers
                .binary_search(&layer)
                .ok()
                .map(|slot| self.rows[slot].as_slice()),
        }
    }
}

pub(super) fn validate_raw_direction_syntax(direction: &RawDirectionDefinition) -> Result<()> {
    let id = &direction.id;
    let (path, layout, layers) = direction.source.parts();
    ensure!(
        !path.as_os_str().is_empty(),
        "raw direction {id} path must not be empty"
    );
    match (layout, layers) {
        (RawDirectionLayout::Shared, None) => {}
        (RawDirectionLayout::Shared, Some(_)) => {
            bail!("raw direction {id} with shared layout must not declare layers")
        }
        (RawDirectionLayout::PerLayer, None) => {
            bail!("raw direction {id} with per_layer layout requires layers")
        }
        (RawDirectionLayout::PerLayer, Some(layers)) => ensure!(
            !layers.is_empty() && layers.windows(2).all(|pair| pair[0] < pair[1]),
            "raw direction {id} layers must be nonempty and strictly increasing"
        ),
    }
    Ok(())
}

/// Reads one exact-length little-endian F32 payload without following symlinks.
pub(super) fn load_raw_direction(
    direction: &RawDirectionDefinition,
    plan_dir: &Path,
    n_layer: u32,
    hidden_size: usize,
) -> Result<LoadedRawDirection> {
    validate_raw_direction_syntax(direction)?;
    let id = &direction.id;
    let (path, layout, layers) = direction.source.parts();
    if let Some(layers) = layers {
        ensure!(
            layers.iter().all(|&layer| layer < n_layer),
            "raw direction {id} layers must be below model layer count {n_layer}"
        );
    }
    ensure!(hidden_size > 0, "raw direction {id} hidden size is zero");
    let row_bytes = hidden_size
        .checked_mul(std::mem::size_of::<f32>())
        .context("raw direction row byte count overflow")?;
    let expected_bytes = row_bytes
        .checked_mul(layers.map_or(1, <[u32]>::len))
        .context("raw direction byte count overflow")?;
    let bytes = crate::read_regular_file_exact(&resolve_plan_path(plan_dir, path), expected_bytes)
        .with_context(|| format!("read raw direction {id} from {}", path.display()))?;
    let payload_blake3 = blake3::hash(&bytes).to_hex().to_string();
    let rows = bytes
        .chunks_exact(row_bytes)
        .enumerate()
        .map(|(slot, chunk)| {
            let values = chunk
                .chunks_exact(4)
                .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>();
            let label = match layers {
                Some(layers) => format!("{id} layer {}", layers[slot]),
                None => id.clone(),
            };
            normalize_direction(values, direction.normalization, &label)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LoadedRawDirection {
        layers: layers.map(<[u32]>::to_vec),
        rows,
        binding: RawDirectionBinding {
            id: id.clone(),
            path: path.to_path_buf(),
            layout,
            layers: layers.map(<[u32]>::to_vec),
            payload_bytes: u64::try_from(expected_bytes).context("raw direction byte count")?,
            payload_blake3,
            semantics: "operator_raw_vector",
        },
    })
}

/// Loads every declared raw direction and checks each referencing layer.
pub(super) fn load_raw_directions(
    plan: &LensPlan,
    plan_dir: &Path,
    n_layer: u32,
    hidden_size: usize,
) -> Result<HashMap<String, LoadedRawDirection>> {
    let mut loaded = HashMap::new();
    for direction in plan.directions.iter().filter_map(DirectionDefinition::raw) {
        let raw = load_raw_direction(direction, plan_dir, n_layer, hidden_size)?;
        ensure!(loaded.insert(direction.id.clone(), raw).is_none());
    }
    for operation in &plan.operations {
        ensure_raw_rows(
            &loaded,
            "operation",
            &operation.id,
            &operation.scope,
            &operation.action.direction_ids().collect::<Vec<_>>(),
            n_layer,
        )?;
    }
    for readout in &plan.direction_readouts {
        ensure_raw_rows(
            &loaded,
            "direction readout",
            &readout.id,
            &readout.scope,
            &[readout.direction.as_str()],
            n_layer,
        )?;
    }
    Ok(loaded)
}

fn ensure_raw_rows(
    loaded: &HashMap<String, LoadedRawDirection>,
    kind: &str,
    owner: &str,
    scope: &Scope,
    ids: &[&str],
    n_layer: u32,
) -> Result<()> {
    let layers = scope
        .layers
        .expand(n_layer, &format!("{kind} {owner} layers"))?;
    for &id in ids {
        let Some(raw) = loaded.get(id) else {
            continue;
        };
        for &layer in &layers {
            ensure!(
                raw.row(layer).is_some(),
                "{kind} {owner} selects layer {layer} absent from per_layer raw direction {id}"
            );
        }
    }
    Ok(())
}

/// Module-site operations are implemented by the dense engine only.
pub(super) fn validate_ordinary_sites(plan: &LensPlan, kind: ArchKind) -> Result<()> {
    if kind == ArchKind::Dense {
        return Ok(());
    }
    for operation in &plan.operations {
        ensure!(
            operation.site == OperationSite::PostBlock,
            "operation {} site {} requires an ordinary dense Qwen model; MoE supports post_block only",
            operation.id,
            operation.site.as_str()
        );
    }
    Ok(())
}

pub(super) struct ExecutionPlan {
    pub(super) plan: LensPlan,
    pub(super) lenses: HashMap<String, PreparedLens>,
    pub(super) directions: HashMap<String, PreparedDirection>,
    pub(super) coordinate_swaps: HashMap<String, PreparedDirection>,
    pub(super) n_layer: u32,
    pub(super) hidden_size: usize,
    pub(super) capture: Option<MetalTensor>,
    pub(super) raw_directions: Vec<RawDirectionBinding>,
}

pub(super) struct PreparedNativeHyperDirection {
    pub(super) layer: u32,
    pub(super) values: Vec<f32>,
}

pub(super) fn load_native_hyper_direction(path: &Path, width: usize, id: &str) -> Result<Vec<f32>> {
    let expected_bytes = width
        .checked_mul(std::mem::size_of::<f32>())
        .context("native hyper direction byte count overflow")?;
    let bytes = crate::read_regular_file_exact(path, expected_bytes)
        .with_context(|| format!("read native hyper direction {id} from {}", path.display()))?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    normalize_direction(values, Normalization::AsStored, id)
}

pub(super) fn parse_plan_bytes(bytes: &[u8]) -> Result<LensPlan> {
    // Parse through Value so serde_json's arbitrary-precision number marker is
    // resolved before serde buffers the internally tagged action enum.
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    Ok(serde_json::from_value(value)?)
}

pub(super) fn validate_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        matches!(plan.version, 1 | 2),
        "Lens plan version must be 1 or 2"
    );
    ensure!(
        !plan.lenses.is_empty()
            || !plan.direction_readouts.is_empty()
            || plan
                .directions
                .iter()
                .any(|direction| direction.lens_row().is_none()),
        "Lens plan must declare at least one lens, raw or native hyper direction, or direction readout"
    );
    ensure!(plan.lenses.len() <= MAX_LENSES, "too many lenses");
    ensure!(
        plan.directions.len() <= MAX_DIRECTIONS,
        "too many directions"
    );
    ensure!(
        plan.operations.len() <= MAX_OPERATIONS,
        "too many operations"
    );
    ensure!(plan.readouts.len() <= MAX_READOUTS, "too many readouts");
    ensure!(
        plan.direction_readouts.len() <= MAX_READOUTS,
        "too many direction readouts"
    );
    unique_ids(plan.lenses.iter().map(LensDefinition::id), "lens")?;
    unique_ids(
        plan.directions.iter().map(DirectionDefinition::id),
        "direction",
    )?;
    unique_ids(
        plan.operations.iter().map(|item| item.id.as_str()),
        "operation",
    )?;
    unique_ids(plan.readouts.iter().map(|item| item.id.as_str()), "readout")?;
    unique_ids(
        plan.direction_readouts.iter().map(|item| item.id.as_str()),
        "direction readout",
    )?;
    for lens in &plan.lenses {
        if let LensDefinition::PublishedFullTransport { id, token_ids, .. } = lens {
            let unique = token_ids.iter().copied().collect::<HashSet<_>>();
            ensure!(
                !token_ids.is_empty() && unique.len() == token_ids.len(),
                "published full transport lens {id} requires unique token IDs"
            );
        }
    }
    let lens_ids: HashSet<&str> = plan.lenses.iter().map(LensDefinition::id).collect();
    let published_full_tokens = plan
        .lenses
        .iter()
        .filter_map(|lens| match lens {
            LensDefinition::PublishedFullTransport { id, token_ids, .. } => Some((
                id.as_str(),
                token_ids.iter().copied().collect::<HashSet<_>>(),
            )),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let direction_ids: HashSet<&str> = plan
        .directions
        .iter()
        .map(DirectionDefinition::id)
        .collect();
    let direction_normalization: HashMap<&str, Option<Normalization>> = plan
        .directions
        .iter()
        .map(|direction| (direction.id(), direction.normalization()))
        .collect();
    for direction in &plan.directions {
        match direction {
            DirectionDefinition::LensRow(direction) => {
                ensure!(
                    lens_ids.contains(direction.lens.as_str()),
                    "direction {} references unknown lens {}",
                    direction.id,
                    direction.lens
                );
                if direction.target_covector.is_some() {
                    ensure!(
                        published_full_tokens.contains_key(direction.lens.as_str())
                            && matches!(direction.row, DirectionRow::TokenId { .. }),
                        "direction {} target_covector is supported only for published full transport token rows",
                        direction.id
                    );
                }
                if let Some(token_ids) = published_full_tokens.get(direction.lens.as_str()) {
                    let DirectionRow::TokenId { token_id } = direction.row else {
                        bail!(
                            "published full transport direction {} requires row.kind=token_id",
                            direction.id
                        );
                    };
                    ensure!(
                        token_id >= 0 && token_ids.contains(&(token_id as u32)),
                        "published full transport direction {} selects token {} absent from lens {}",
                        direction.id,
                        token_id,
                        direction.lens
                    );
                }
            }
            DirectionDefinition::NativeHyper(direction) => {
                let (path, _) = direction.source.path_and_layer();
                ensure!(
                    !path.as_os_str().is_empty(),
                    "native hyper direction {} path must not be empty",
                    direction.id
                );
            }
            DirectionDefinition::Raw(direction) => validate_raw_direction_syntax(direction)?,
        }
    }
    for operation in &plan.operations {
        operation
            .scope
            .validate(&format!("operation {} scope", operation.id), plan.version)?;
        crate::lens_intervention::validate_action(&operation.id, &operation.action, |direction| {
            ensure!(
                direction_ids.contains(direction),
                "operation {} references unknown direction {}",
                operation.id,
                direction
            );
            Ok(direction_normalization.get(direction).copied().flatten())
        })?;
        if operation.site == OperationSite::Embedding {
            let layer_zero = match &operation.scope.layers {
                Selector::Values { values } => values.as_slice() == [0],
                Selector::Range { start, end } => *start == 0 && *end == 0,
                Selector::All | Selector::RenderedSpans { .. } => false,
            };
            ensure!(
                layer_zero,
                "operation {} at the embedding site must select exactly layer 0",
                operation.id
            );
        }
    }
    for readout in &plan.direction_readouts {
        readout.scope.validate(
            &format!("direction readout {} scope", readout.id),
            plan.version,
        )?;
        ensure!(
            direction_ids.contains(readout.direction.as_str()),
            "direction readout {} references unknown direction {}",
            readout.id,
            readout.direction
        );
        ensure!(
            direction_normalization[readout.direction.as_str()].is_some(),
            "direction readout {} requires a lens-row or raw direction",
            readout.id
        );
    }
    for readout in &plan.readouts {
        readout
            .scope
            .validate(&format!("readout {} scope", readout.id), plan.version)?;
        ensure!(
            readout.top_k > 0 && readout.top_k <= MAX_TOP_K,
            "readout {} top_k must be in 1..={MAX_TOP_K}",
            readout.id
        );
        ensure!(
            lens_ids.contains(readout.lens.as_str()),
            "readout {} references unknown lens {}",
            readout.id,
            readout.lens
        );
    }
    Ok(())
}

pub(crate) fn plan_with_operation_coefficient(
    source: &LensPlan,
    operation_id: &str,
    coefficient: f32,
) -> Result<LensPlan> {
    ensure!(coefficient.is_finite(), "sweep coefficient must be finite");
    let mut plan = source.clone();
    let operation = plan
        .operations
        .iter_mut()
        .find(|operation| operation.id == operation_id)
        .with_context(|| format!("Lens plan has no operation {operation_id:?}"))?;
    operation.action.set_coefficient(coefficient);
    validate_plan(&plan)?;
    Ok(plan)
}

pub(crate) fn validate_run_artifact_plan(plan: &LensPlan, runtime_kind: &str) -> Result<()> {
    validate_plan(plan)?;
    match runtime_kind {
        "ordinary_qwen" => validate_ordinary_plan(plan),
        "muse_glimmer" => validate_muse_artifact_plan(plan),
        "flash_next" => validate_flash_artifact_plan(plan),
        _ => bail!("run artifact has unsupported runtime kind {runtime_kind:?}"),
    }
}

/// Raw directions, module sites, and direction readouts run on ordinary Qwen only.
pub(crate) fn refuse_ordinary_only_features(plan: &LensPlan, runtime: &str) -> Result<()> {
    if let Some(direction) = plan.directions.iter().find_map(DirectionDefinition::raw) {
        bail!(
            "{runtime} does not support raw_residual_f32le direction {}; raw directions are ordinary-Qwen only",
            direction.id
        );
    }
    if let Some(operation) = plan
        .operations
        .iter()
        .find(|operation| operation.site != OperationSite::PostBlock)
    {
        bail!(
            "{runtime} does not support operation {} site {}; module sites are ordinary dense Qwen only",
            operation.id,
            operation.site.as_str()
        );
    }
    ensure!(
        plan.direction_readouts.is_empty(),
        "{runtime} does not support direction_readouts; they are ordinary-Qwen only"
    );
    Ok(())
}

pub(super) fn validate_muse_artifact_plan(plan: &LensPlan) -> Result<()> {
    refuse_ordinary_only_features(plan, "Muse Glimmer")?;
    ensure!(
        !plan.operations.is_empty() || !plan.readouts.is_empty(),
        "Muse run artifact plan requires an operation or readout"
    );
    ensure!(
        plan.lenses.iter().all(|lens| matches!(
            lens,
            LensDefinition::NativeSelected { .. } | LensDefinition::PublishedFullTransport { .. }
        )),
        "Muse run artifact plan contains an unsupported lens kind"
    );
    ensure!(
        plan.directions.iter().all(|direction| matches!(
            direction,
            DirectionDefinition::LensRow(definition)
                if matches!(definition.row, DirectionRow::TokenId { .. })
                    && definition.target_covector.is_none()
        )),
        "Muse run artifact plan requires selected-token directions without target_covector overrides"
    );
    Ok(())
}

pub(super) fn validate_ordinary_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        !plan.lenses.is_empty()
            || !plan.directions.is_empty()
            || !plan.direction_readouts.is_empty(),
        "ordinary Qwen Lens plans must declare at least one lens, raw direction, or direction readout"
    );
    ensure!(
        plan.directions
            .iter()
            .all(|direction| direction.native_hyper().is_none()),
        "native hyper directions are supported only by Flash-Next"
    );
    Ok(())
}

pub(super) fn ensure_projected_full_direction_bank_budget(
    plan: &LensPlan,
    required_lens_layers: &HashMap<&str, BTreeSet<u32>>,
    raw_lens_layers: &HashMap<&str, BTreeSet<u32>>,
    hidden_size: usize,
) -> Result<()> {
    let mut retained_bytes = 0usize;
    for lens in &plan.lenses {
        let LensDefinition::PublishedFullTransport { id, token_ids, .. } = lens else {
            continue;
        };
        let layers = required_lens_layers
            .get(id.as_str())
            .with_context(|| format!("published full transport lens {id} is not used"))?;
        retained_bytes = retained_bytes
            .checked_add(
                crate::full_lens::projected_full_token_direction_retained_bytes(
                    layers.len(),
                    token_ids.len(),
                    hidden_size,
                )?,
            )
            .context("published full transport retained bank size overflow")?;
        if let Some(raw_layers) = raw_lens_layers.get(id.as_str()) {
            retained_bytes = retained_bytes
                .checked_add(
                    crate::full_lens::projected_full_token_direction_retained_bytes(
                        raw_layers.len(),
                        token_ids.len(),
                        hidden_size,
                    )?,
                )
                .context("published full transport retained raw bank size overflow")?;
        }
    }
    ensure!(
        retained_bytes <= crate::TOKEN_ARTIFACT_MAX_BYTES,
        "published full transport direction banks require {retained_bytes} bytes, exceeding retained result budget {}",
        crate::TOKEN_ARTIFACT_MAX_BYTES
    );
    Ok(())
}

pub(super) fn open_full_transports(
    plan: &LensPlan,
    plan_dir: &Path,
) -> Result<HashMap<String, crate::full_lens::FullAccess>> {
    let mut opened = HashMap::new();
    for lens in &plan.lenses {
        if let LensDefinition::PublishedFullTransport {
            id,
            artifact,
            allow_unvalidated_transfer,
            ..
        } = lens
        {
            let access =
                crate::full_lens::FullAccess::open(&resolve_plan_path(plan_dir, artifact), false)?;
            access.acknowledge_transfer(*allow_unvalidated_transfer)?;
            opened.insert(id.clone(), access);
        }
    }
    Ok(opened)
}

pub(super) struct BoundPlanArtifacts {
    full: HashMap<String, crate::full_lens::BoundFullAccess>,
    ready: HashMap<String, PreparedLens>,
    raw: HashMap<String, LoadedRawDirection>,
}

fn prepare_cpu_lens(
    definition: &LensDefinition,
    plan_dir: &Path,
    arch: qwen_llm::model::Arch,
) -> Result<PreparedLens> {
    let lens = match definition {
        LensDefinition::NativeSelected { artifact, .. } => LoadedLens::Native(load_native_lens(
            &resolve_plan_path(plan_dir, artifact),
            arch.n_layer,
            arch.hidden_size as usize,
            arch.vocab_size,
        )?),
        LensDefinition::WorkspaceTemplate {
            id,
            weights,
            labels,
        } => {
            let lens = TemplateLens::open(&resolve_plan_path(plan_dir, weights))
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let vocabulary =
                TemplateVocabulary::load(&resolve_plan_path(plan_dir, labels), lens.n_rows())
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            ensure!(
                lens.hidden_size() == arch.hidden_size as usize,
                "template lens {id} hidden size does not match the CPU-bound deployment"
            );
            LoadedLens::Template { lens, vocabulary }
        }
        LensDefinition::PublishedFullTransport { .. } => {
            bail!("full transports require retained bound access")
        }
    };
    Ok(PreparedLens {
        id: definition.id().to_owned(),
        lens,
        raw_lm_head: None,
    })
}

pub(super) fn bind_full_transports(
    mut opened: HashMap<String, crate::full_lens::FullAccess>,
    plan: &LensPlan,
    plan_dir: &Path,
    gguf: &GgufFile,
    cache: Option<&Path>,
) -> Result<BoundPlanArtifacts> {
    let arch = qwen_llm::loader::Model::from_gguf(gguf)?.arch;
    let mut bound = HashMap::new();
    let mut ready = HashMap::new();
    for lens in &plan.lenses {
        if let LensDefinition::PublishedFullTransport {
            id,
            token_ids,
            allow_unvalidated_transfer,
            ..
        } = lens
        {
            ensure!(
                token_ids
                    .iter()
                    .all(|&token| token < arch.vocab_size && token <= i32::MAX as u32),
                "full transport token selection is outside the CPU-bound deployment vocabulary"
            );
            let artifact = opened
                .remove(id)
                .context("missing verified full transport")?;
            bound.insert(
                id.clone(),
                artifact.bind_opened(
                    gguf,
                    crate::full_lens::FullExecutionMode::Projection,
                    cache,
                    *allow_unvalidated_transfer,
                )?,
            );
        }
    }
    for definition in &plan.lenses {
        if !matches!(definition, LensDefinition::PublishedFullTransport { .. }) {
            let prepared = prepare_cpu_lens(definition, plan_dir, arch)?;
            ready.insert(prepared.id.clone(), prepared);
        }
    }
    ensure!(opened.is_empty(), "unbound full transports remain");
    let has_layer = |id: &str, layer| -> bool {
        bound
            .get(id)
            .is_some_and(|artifact| artifact.contains_layer(layer))
            || ready
                .get(id)
                .is_some_and(|lens| lens_has_layer(lens, layer))
    };
    for readout in &plan.readouts {
        for layer in readout
            .scope
            .layers
            .expand(arch.n_layer, "readout.layers")?
        {
            ensure!(
                has_layer(&readout.lens, layer),
                "readout {} lens {} has no row for layer {layer}",
                readout.id,
                readout.lens
            );
        }
    }
    let directions = plan
        .directions
        .iter()
        .map(|direction| (direction.id(), direction))
        .collect::<HashMap<_, _>>();
    validate_ordinary_sites(plan, arch.kind)?;
    let raw = load_raw_directions(plan, plan_dir, arch.n_layer, arch.hidden_size as usize)?;
    for (scope, ids) in direction_uses(plan) {
        let layers = scope.layers.expand(arch.n_layer, "direction use layers")?;
        for id in ids {
            let direction = match directions
                .get(id)
                .context("unknown direction in CPU plan preflight")?
            {
                DirectionDefinition::LensRow(direction) => direction,
                // Raw rows were loaded and layer-checked above.
                DirectionDefinition::Raw(_) => continue,
                DirectionDefinition::NativeHyper(_) => {
                    bail!("ordinary runtime cannot use native hyper direction {id}")
                }
            };
            for &layer in &layers {
                ensure!(
                    has_layer(&direction.lens, layer),
                    "direction {id} has no source layer {layer}"
                );
                if let Some(prepared) = ready.get(&direction.lens) {
                    direction_row(prepared, direction, layer)?;
                }
            }
        }
    }
    Ok(BoundPlanArtifacts {
        full: bound,
        ready,
        raw,
    })
}

/// Every scope that consumes prepared direction rows, with its direction IDs.
fn direction_uses(plan: &LensPlan) -> impl Iterator<Item = (&Scope, Vec<&str>)> {
    plan.operations
        .iter()
        .map(|operation| (&operation.scope, operation.action.direction_ids().collect()))
        .chain(
            plan.direction_readouts
                .iter()
                .map(|readout| (&readout.scope, vec![readout.direction.as_str()])),
        )
}

pub(super) fn prepare_execution_plan(
    plan: &LensPlan,
    plan_dir: &Path,
    loaded: &qwen_llm::runtime::LoadedModel,
    full_transports: &mut BoundPlanArtifacts,
) -> Result<ExecutionPlan> {
    let arch = loaded.arch();
    let direction_defs = plan
        .directions
        .iter()
        .map(|direction| (direction.id(), direction))
        .collect::<HashMap<_, _>>();
    let mut direction_layers: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
    let mut required_lens_layers: HashMap<&str, BTreeSet<u32>> = HashMap::new();
    let mut raw_lens_layers: HashMap<&str, BTreeSet<u32>> = HashMap::new();
    validate_ordinary_sites(plan, arch.kind)?;
    for (scope, ids) in direction_uses(plan) {
        let layers = scope.layers.expand(arch.n_layer, "direction use layers")?;
        for direction_id in ids {
            let definition = direction_defs
                .get(direction_id)
                .with_context(|| format!("unknown direction {direction_id}"))?;
            ensure!(
                definition.native_hyper().is_none(),
                "ordinary runtime cannot load native hyper direction {direction_id}"
            );
            direction_layers
                .entry(direction_id)
                .or_default()
                .extend(layers.iter().copied());
            let Some(direction) = definition.lens_row() else {
                continue;
            };
            required_lens_layers
                .entry(direction.lens.as_str())
                .or_default()
                .extend(layers.iter().copied());
            if direction.effective_target_covector()
                != DirectionTargetCovector::DeployedLogitNumerator
            {
                raw_lens_layers
                    .entry(direction.lens.as_str())
                    .or_default()
                    .extend(layers.iter().copied());
            }
        }
    }
    let mut readout_layers = BTreeSet::new();
    for readout in &plan.readouts {
        let layers = readout
            .scope
            .layers
            .expand(arch.n_layer, "readout.layers")?;
        required_lens_layers
            .entry(readout.lens.as_str())
            .or_default()
            .extend(layers.iter().copied());
        readout_layers.extend(layers);
    }
    // Direction readouts share the post-block capture with lens readouts.
    for readout in &plan.direction_readouts {
        readout_layers.extend(
            readout
                .scope
                .layers
                .expand(arch.n_layer, "direction_readout.layers")?,
        );
    }
    ensure_projected_full_direction_bank_budget(
        plan,
        &required_lens_layers,
        &raw_lens_layers,
        arch.hidden_size as usize,
    )?;

    let mut lenses = HashMap::new();
    for definition in &plan.lenses {
        let prepared = match definition {
            LensDefinition::NativeSelected { id, .. }
            | LensDefinition::WorkspaceTemplate { id, .. } => full_transports
                .ready
                .remove(id)
                .context("lens artifact was not bound before model load")?,
            LensDefinition::PublishedFullTransport {
                id,
                artifact,
                token_ids,
                allow_unvalidated_transfer,
            } => {
                let layers = required_lens_layers
                    .get(id.as_str())
                    .with_context(|| format!("published full transport lens {id} is not used"))?
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let projected = crate::full_lens::project_full_token_directions(
                    &resolve_plan_path(plan_dir, artifact),
                    full_transports
                        .full
                        .get_mut(id)
                        .context("full transport was not verified before model load")?,
                    token_ids,
                    &layers,
                    loaded,
                    crate::full_lens::FullTokenTargetCovector::DeployedLogitNumerator,
                    *allow_unvalidated_transfer,
                )?;
                let raw_lm_head = raw_lens_layers
                    .get(id.as_str())
                    .map(|raw_layers| {
                        crate::full_lens::project_full_token_directions(
                            &resolve_plan_path(plan_dir, artifact),
                            full_transports
                                .full
                                .get_mut(id)
                                .context("full transport was not verified before model load")?,
                            token_ids,
                            &raw_layers.iter().copied().collect::<Vec<_>>(),
                            loaded,
                            crate::full_lens::FullTokenTargetCovector::RawLmHead,
                            *allow_unvalidated_transfer,
                        )
                        .map(native_lens_from_projected_full)
                    })
                    .transpose()?;
                PreparedLens {
                    id: id.clone(),
                    lens: LoadedLens::Native(native_lens_from_projected_full(projected)),
                    raw_lm_head,
                }
            }
        };
        ensure!(lenses.insert(prepared.id.clone(), prepared).is_none());
    }

    for readout in &plan.readouts {
        let layers = readout
            .scope
            .layers
            .expand(arch.n_layer, "readout.layers")?;
        let prepared_lens = &lenses[&readout.lens];
        for &layer in &layers {
            ensure!(
                lens_has_layer(prepared_lens, layer),
                "readout {} lens {} has no row for layer {}",
                readout.id,
                readout.lens,
                layer
            );
        }
    }
    let capture_layers: Vec<u32> = readout_layers.iter().copied().collect();

    let mut directions = HashMap::new();
    for (direction_id, layers) in direction_layers {
        let mut rows = BTreeMap::new();
        for layer in layers {
            let normalized = match direction_defs[direction_id] {
                DirectionDefinition::LensRow(definition) => normalize_direction(
                    direction_row(&lenses[&definition.lens], definition, layer)?,
                    definition.normalization,
                    direction_id,
                )?,
                DirectionDefinition::Raw(_) => full_transports
                    .raw
                    .get(direction_id)
                    .and_then(|raw| raw.row(layer))
                    .with_context(|| {
                        format!("raw direction {direction_id} has no row for layer {layer}")
                    })?
                    .to_vec(),
                DirectionDefinition::NativeHyper(_) => {
                    bail!("ordinary runtime cannot load native hyper direction {direction_id}")
                }
            };
            let tensor = MetalTensor::from_bytes(
                loaded.context(),
                bytemuck::cast_slice(&normalized),
                vec![arch.hidden_size as u64],
                GgmlType::F32,
            )?;
            ensure!(rows.insert(layer, tensor).is_none());
        }
        directions.insert(direction_id.to_owned(), PreparedDirection { rows });
    }
    let coordinate_swaps = prepare_coordinate_swaps(
        &plan.operations,
        &directions,
        loaded.context(),
        arch.n_layer,
        arch.hidden_size as usize,
    )?;
    let capture = if capture_layers.is_empty() {
        None
    } else {
        Some(MetalTensor::zeros_f32(
            loaded.context(),
            vec![(capture_layers.len() * arch.hidden_size as usize) as u64],
        )?)
    };
    let raw_directions = plan
        .directions
        .iter()
        .filter_map(DirectionDefinition::raw)
        .map(|direction| {
            full_transports
                .raw
                .get(&direction.id)
                .map(|raw| raw.binding.clone())
                .with_context(|| format!("raw direction {} was not loaded", direction.id))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ExecutionPlan {
        plan: plan.clone(),
        lenses,
        directions,
        coordinate_swaps,
        n_layer: arch.n_layer,
        hidden_size: arch.hidden_size as usize,
        capture,
        raw_directions,
    })
}

pub(super) fn prepare_coordinate_swaps(
    operations: &[OperationDefinition],
    directions: &HashMap<String, PreparedDirection>,
    context: &MetalContext,
    n_layer: u32,
    hidden_size: usize,
) -> Result<HashMap<String, PreparedDirection>> {
    let mut swaps = HashMap::new();
    let mut reflection_cache = HashMap::<(String, String, u32), MetalTensor>::new();
    for operation in operations {
        let Action::CoordinateSwap { source, target, .. } = &operation.action else {
            continue;
        };
        let pair = if source <= target {
            (source.clone(), target.clone())
        } else {
            (target.clone(), source.clone())
        };
        let layers = operation
            .scope
            .layers
            .expand(n_layer, &format!("operation {} layers", operation.id))?;
        let source = directions
            .get(source)
            .with_context(|| format!("coordinate swap {} has no source direction", operation.id))?;
        let target = directions
            .get(target)
            .with_context(|| format!("coordinate swap {} has no target direction", operation.id))?;
        let mut rows = BTreeMap::new();
        for layer in layers {
            let cache_key = (pair.0.clone(), pair.1.clone(), layer);
            if let Some(reflection) = reflection_cache.get(&cache_key) {
                ensure!(rows.insert(layer, reflection.clone()).is_none());
                continue;
            }
            let source = source.rows.get(&layer).with_context(|| {
                format!(
                    "coordinate swap {} source is unavailable at layer {layer}",
                    operation.id
                )
            })?;
            let target = target.rows.get(&layer).with_context(|| {
                format!(
                    "coordinate swap {} target is unavailable at layer {layer}",
                    operation.id
                )
            })?;
            let reflection = coordinate_swap_reflection_direction(
                &read_f32_tensor(source, hidden_size),
                &read_f32_tensor(target, hidden_size),
                &format!("{} at layer {layer}", operation.id),
            )?;
            let reflection = MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(&reflection),
                vec![hidden_size as u64],
                GgmlType::F32,
            )?;
            ensure!(
                reflection_cache
                    .insert(cache_key, reflection.clone())
                    .is_none()
            );
            ensure!(rows.insert(layer, reflection).is_none());
        }
        ensure!(
            swaps
                .insert(operation.id.clone(), PreparedDirection { rows })
                .is_none(),
            "duplicate coordinate-swap operation {}",
            operation.id
        );
    }
    Ok(swaps)
}

pub(crate) fn coordinate_swap_reflection_direction(
    source: &[f32],
    target: &[f32],
    id: &str,
) -> Result<Vec<f32>> {
    ensure!(
        !source.is_empty() && source.len() == target.len(),
        "coordinate swap {id} directions have incompatible shapes"
    );
    ensure!(
        source.iter().chain(target).all(|value| value.is_finite()),
        "coordinate swap {id} contains a non-finite direction"
    );
    let source_sq = source
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let target_sq = target
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let source_norm = source_sq.sqrt();
    let target_norm = target_sq.sqrt();
    ensure!(
        source_norm.is_finite()
            && target_norm.is_finite()
            && source_norm > 0.0
            && target_norm > 0.0,
        "coordinate swap {id} has a zero or non-finite direction norm"
    );
    let cosine = source
        .iter()
        .zip(target)
        .map(|(&source, &target)| f64::from(source) * f64::from(target))
        .sum::<f64>()
        / (source_norm * target_norm);
    let cosine = cosine.clamp(-1.0, 1.0);
    ensure!(
        1.0 - cosine * cosine > 1e-10,
        "coordinate swap {id} directions are linearly dependent"
    );
    let difference = source
        .iter()
        .zip(target)
        .map(|(&source, &target)| {
            (f64::from(source) / source_norm - f64::from(target) / target_norm) as f32
        })
        .collect();
    normalize_direction(difference, Normalization::UnitL2, id)
}

pub(super) fn resolve_plan_path(plan_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        plan_dir.join(path)
    }
}

pub(super) fn direction_row(
    prepared: &PreparedLens,
    definition: &LensRowDirectionDefinition,
    layer: u32,
) -> Result<Vec<f32>> {
    match definition.effective_target_covector() {
        DirectionTargetCovector::DeployedLogitNumerator => {
            lens_row(prepared, &definition.row, layer)
        }
        DirectionTargetCovector::RawLmHead => {
            raw_lm_head_direction_row(prepared, &definition.row, layer, &definition.id)
        }
        DirectionTargetCovector::RawLmHeadOrthogonalToDeployedLogitNumerator => {
            let raw = raw_lm_head_direction_row(prepared, &definition.row, layer, &definition.id)?;
            let deployed = lens_row(prepared, &definition.row, layer)?;
            orthogonal_component(&raw, &deployed, &definition.id)
        }
    }
}

pub(super) fn raw_lm_head_direction_row(
    prepared: &PreparedLens,
    selector: &DirectionRow,
    layer: u32,
    direction_id: &str,
) -> Result<Vec<f32>> {
    let raw = prepared
        .raw_lm_head
        .as_ref()
        .with_context(|| format!("direction {direction_id} requires a raw LM-head projection"))?;
    native_lens_row(raw, selector, layer, &prepared.id)
}

pub(crate) fn validate_reachable_scopes(
    plan: &LensPlan,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let prefill_bound =
        u32::try_from(prompt_len).context("prompt length exceeds selector range")?;
    let decode_bound = u32::try_from(max_new_tokens.saturating_sub(1))
        .context("decode selector range overflow")?;
    for operation in &plan.operations {
        validate_scope_reachable(&operation.scope, prefill_bound, decode_bound, &operation.id)?;
    }
    for readout in &plan.readouts {
        validate_scope_reachable(&readout.scope, prefill_bound, decode_bound, &readout.id)?;
    }
    for readout in &plan.direction_readouts {
        validate_scope_reachable(&readout.scope, prefill_bound, decode_bound, &readout.id)?;
    }
    Ok(())
}

pub(super) fn validate_scope_reachable(
    scope: &Scope,
    prefill_bound: u32,
    decode_bound: u32,
    id: &str,
) -> Result<()> {
    if let Some(selector) = &scope.prefill {
        selector.expand(prefill_bound, &format!("{id}.prefill"))?;
    }
    if let Some(selector) = &scope.decode {
        ensure!(
            decode_bound > 0,
            "{id}.decode cannot reach a transition when max_new_tokens is 1"
        );
        selector.expand(decode_bound, &format!("{id}.decode"))?;
    }
    Ok(())
}

pub(super) fn scope_matches(scope: &Scope, phase: Phase, layer: u32) -> Result<bool> {
    let index = u32::try_from(phase.index()).context("event index exceeds u32")?;
    let phase_matches = match phase {
        Phase::Prefill(_) => match &scope.prefill {
            Some(selector) => selector_contains(selector, index)?,
            None => false,
        },
        Phase::Decode(_) => match &scope.decode {
            Some(selector) => selector_contains(selector, index)?,
            None => false,
        },
    };
    Ok(phase_matches && selector_contains(&scope.layers, layer)?)
}

pub(super) fn selector_contains(selector: &Selector, value: u32) -> Result<bool> {
    Ok(match selector {
        Selector::All => true,
        Selector::Values { values } => values.binary_search(&value).is_ok(),
        Selector::Range { start, end } => (*start..=*end).contains(&value),
        Selector::RenderedSpans { .. } => bail!("execution plan contains an unresolved selector"),
    })
}

pub(super) fn action_to_intervention<'a>(
    operation_id: &str,
    action: &'a Action,
    layer: u32,
    directions: &'a HashMap<String, PreparedDirection>,
    coordinate_swaps: &'a HashMap<String, PreparedDirection>,
) -> Result<PostBlockIntervention<'a>> {
    let direction = |id: &str| -> Result<&'a MetalTensor> {
        directions
            .get(id)
            .and_then(|prepared| prepared.rows.get(&layer))
            .with_context(|| format!("direction {id} has no uploaded row for layer {layer}"))
    };
    crate::lens_intervention::lower(action, layer, direction, || {
        coordinate_swaps
            .get(operation_id)
            .and_then(|prepared| prepared.rows.get(&layer))
            .with_context(|| {
                format!(
                    "coordinate swap {operation_id} has no reflection direction at layer {layer}"
                )
            })
    })
}

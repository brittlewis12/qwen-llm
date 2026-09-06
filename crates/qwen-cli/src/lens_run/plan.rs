//! Lens plan schema, validation, and intervention lowering.

use super::*;

pub(super) const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;

pub(super) const MAX_DIRECTIONS: usize = 4096;

pub(super) const MAX_OPERATIONS: usize = 4096;

pub(super) const MAX_SELECTOR_VALUES: usize = 4096;

pub(super) const MAX_RENDERED_SELECTOR_TEXT_BYTES: usize = 1024;

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
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum LensDefinition {
    NativeSelected {
        id: String,
        artifact: PathBuf,
    },
    #[serde(rename = "published_full_transport", alias = "published_full_j")]
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
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensRowDirectionDefinition {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) row: DirectionRow,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) target_covector: Option<DirectionTargetCovector>,
    pub(crate) normalization: Normalization,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DirectionTargetCovector {
    DeployedLogitNumerator,
    RawLmHead,
    RawLmHeadOrthogonalToDeployedLogitNumerator,
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
        }
    }

    pub(super) fn lens_row(&self) -> Option<&LensRowDirectionDefinition> {
        match self {
            Self::LensRow(direction) => Some(direction),
            Self::NativeHyper(_) => None,
        }
    }

    pub(super) fn native_hyper(&self) -> Option<&NativeHyperDirectionDefinition> {
        match self {
            Self::LensRow(_) => None,
            Self::NativeHyper(direction) => Some(direction),
        }
    }

    pub(super) fn normalization(&self) -> Option<Normalization> {
        self.lens_row().map(|direction| direction.normalization)
    }
}

impl LensRowDirectionDefinition {
    pub(super) fn effective_target_covector(&self) -> DirectionTargetCovector {
        self.target_covector
            .unwrap_or(DirectionTargetCovector::DeployedLogitNumerator)
    }
}

impl NativeHyperDirectionSource {
    pub(super) fn path_and_layer(&self) -> (&Path, u32) {
        match self {
            Self::NativeHyperF32 { path, layer } => (path, *layer),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum DirectionRow {
    TokenId { token_id: i32 },
    TemplateRowId { template_row_id: usize },
    Label { label: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationDefinition {
    pub(crate) id: String,
    pub(crate) scope: Scope,
    pub(crate) action: Action,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Action {
    FixedAdd {
        direction: String,
        coefficient: f32,
    },
    ResidualL2Fraction {
        direction: String,
        coefficient: f32,
    },
    ProjectionAblate {
        direction: String,
        coefficient: f32,
    },
    SourceToTarget {
        source: String,
        target: String,
        coefficient: f32,
    },
    CoordinateSwap {
        source: String,
        target: String,
        coefficient: f32,
    },
}

impl Action {
    pub(crate) fn coefficient(&self) -> f32 {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient,
        }
    }

    pub(super) fn set_coefficient(&mut self, value: f32) {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient = value,
        }
    }

    pub(crate) fn direction_ids<'a>(&'a self) -> impl Iterator<Item = &'a str> + 'a {
        match self {
            Self::FixedAdd { direction, .. }
            | Self::ResidualL2Fraction { direction, .. }
            | Self::ProjectionAblate { direction, .. } => {
                EitherDirectionIds::One(std::iter::once(direction.as_str()))
            }
            Self::SourceToTarget { source, target, .. }
            | Self::CoordinateSwap { source, target, .. } => {
                EitherDirectionIds::Two([source.as_str(), target.as_str()].into_iter())
            }
        }
    }
}

pub(super) enum EitherDirectionIds<'a> {
    One(std::iter::Once<&'a str>),
    Two(std::array::IntoIter<&'a str, 2>),
}

impl<'a> Iterator for EitherDirectionIds<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(iter) => iter.next(),
            Self::Two(iter) => iter.next(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub(crate) layers: Selector,
    #[serde(default)]
    pub(crate) prefill: Option<Selector>,
    #[serde(default)]
    pub(crate) decode: Option<Selector>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Selector {
    All,
    Values {
        values: Vec<u32>,
    },
    Range {
        start: u32,
        end: u32,
    },
    RenderedSpans {
        selectors: Vec<RenderedSpanSelector>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RenderedSpanSelector {
    pub(crate) span_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) message_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    #[serde(default)]
    pub(crate) occurrence: RenderedSpanOccurrence,
    pub(crate) edge: RenderedSpanEdge,
}

impl Selector {
    pub(super) fn validate(&self, name: &str, allow_rendered_spans: bool) -> Result<()> {
        match self {
            Self::All => Ok(()),
            Self::Values { values } => {
                ensure!(!values.is_empty(), "{name} values must not be empty");
                ensure!(
                    values.len() <= MAX_SELECTOR_VALUES,
                    "{name} has too many explicit values"
                );
                ensure!(
                    values.windows(2).all(|pair| pair[0] < pair[1]),
                    "{name} values must be sorted and unique"
                );
                Ok(())
            }
            Self::Range { start, end } => {
                ensure!(
                    start <= end,
                    "{name} range must be inclusive with start <= end"
                );
                ensure!(
                    u64::from(*end) - u64::from(*start) < MAX_SELECTOR_VALUES as u64,
                    "{name} range is too large"
                );
                Ok(())
            }
            Self::RenderedSpans { selectors } => {
                ensure!(
                    allow_rendered_spans,
                    "{name} does not support rendered-span selectors"
                );
                ensure!(
                    !selectors.is_empty() && selectors.len() <= MAX_SELECTOR_VALUES,
                    "{name} rendered_spans requires 1..={MAX_SELECTOR_VALUES} selectors"
                );
                ensure!(
                    selectors.iter().collect::<BTreeSet<_>>().len() == selectors.len(),
                    "{name} repeats an authored rendered-span selector"
                );
                for (index, selector) in selectors.iter().enumerate() {
                    selector.validate(&format!("{name}.selectors[{index}]"))?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn expand(&self, upper_bound: u32, name: &str) -> Result<Vec<u32>> {
        self.validate(name, false)?;
        let values = match self {
            Self::All => (0..upper_bound).collect(),
            Self::Values { values } => values.clone(),
            Self::Range { start, end } => (*start..=*end).collect(),
            Self::RenderedSpans { .. } => unreachable!("validation rejects unresolved selectors"),
        };
        ensure!(
            values.iter().all(|&value| value < upper_bound),
            "{name} contains a value outside 0..{upper_bound}"
        );
        Ok(values)
    }
}

impl RenderedSpanSelector {
    pub(super) fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            is_known_lens_span(&self.span_kind),
            "{name}.span_kind is not a known renderer span"
        );
        ensure!(
            self.message_index
                .is_none_or(|index| index < MAX_SELECTOR_VALUES)
                && self
                    .tool_call_index
                    .is_none_or(|index| index < MAX_SELECTOR_VALUES),
            "{name} message/tool-call index exceeds the selector bound"
        );
        ensure!(
            self.role
                .as_deref()
                .is_none_or(|role| { matches!(role, "system" | "user" | "assistant" | "tool") }),
            "{name}.role is unsupported"
        );
        ensure!(
            self.channel.as_deref().is_none_or(|channel| {
                matches!(channel, "thinking" | "tool_call" | "tool_result")
            }),
            "{name}.channel is unsupported"
        );
        for (field, value) in [
            ("role", self.role.as_deref()),
            ("channel", self.channel.as_deref()),
            ("label", self.label.as_deref()),
        ] {
            ensure!(
                value.is_none_or(|value| {
                    !value.is_empty() && value.len() <= MAX_RENDERED_SELECTOR_TEXT_BYTES
                }),
                "{name}.{field} is empty or too long"
            );
        }
        Ok(())
    }

    pub(super) fn matches(&self, span: &LensRenderedSpan) -> bool {
        span.kind == self.span_kind
            && self
                .message_index
                .is_none_or(|value| span.message_index == Some(value))
            && self
                .tool_call_index
                .is_none_or(|value| span.tool_call_index == Some(value))
            && self
                .role
                .as_deref()
                .is_none_or(|value| span.role.as_deref() == Some(value))
            && self
                .channel
                .as_deref()
                .is_none_or(|value| span.channel.as_deref() == Some(value))
            && self
                .label
                .as_deref()
                .is_none_or(|value| span.label.as_deref() == Some(value))
    }
}

impl Scope {
    pub(super) fn validate(&self, name: &str, plan_version: u32) -> Result<()> {
        self.layers.validate(&format!("{name}.layers"), false)?;
        ensure!(
            self.prefill.is_some() || self.decode.is_some(),
            "{name} must select prefill and/or decode"
        );
        if let Some(selector) = &self.prefill {
            selector.validate(&format!("{name}.prefill"), plan_version == 2)?;
        }
        if let Some(selector) = &self.decode {
            selector.validate(&format!("{name}.decode"), false)?;
        }
        Ok(())
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
        plan.operations.len() <= MAX_OPERATIONS && plan.readouts.len() <= MAX_READOUTS,
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
}

pub(super) struct PreparedDirection {
    pub(super) rows: BTreeMap<u32, MetalTensor>,
}

pub(super) struct ExecutionPlan {
    pub(super) plan: LensPlan,
    pub(super) lenses: HashMap<String, PreparedLens>,
    pub(super) directions: HashMap<String, PreparedDirection>,
    pub(super) coordinate_swaps: HashMap<String, PreparedDirection>,
    pub(super) n_layer: u32,
    pub(super) hidden_size: usize,
    pub(super) capture: Option<MetalTensor>,
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
            || plan
                .directions
                .iter()
                .any(|direction| direction.native_hyper().is_some()),
        "Lens plan must declare at least one lens or native hyper direction"
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
    for lens in &plan.lenses {
        if let LensDefinition::PublishedFullTransport {
            id,
            token_ids,
            allow_unvalidated_transfer,
            ..
        } = lens
        {
            let unique = token_ids.iter().copied().collect::<HashSet<_>>();
            ensure!(
                !token_ids.is_empty() && unique.len() == token_ids.len(),
                "published full transport lens {id} requires unique token IDs"
            );
            ensure!(
                *allow_unvalidated_transfer,
                "published full transport lens {id} requires allow_unvalidated_transfer=true for BF16-to-GGUF use"
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
        }
    }
    for operation in &plan.operations {
        operation
            .scope
            .validate(&format!("operation {} scope", operation.id), plan.version)?;
        let coefficient = operation.action.coefficient();
        ensure!(
            coefficient.is_finite(),
            "operation {} coefficient must be finite",
            operation.id,
        );
        if let Action::CoordinateSwap {
            source,
            target,
            coefficient,
        } = &operation.action
        {
            ensure!(
                source != target,
                "coordinate-swap operation {} requires distinct source and target directions",
                operation.id
            );
            ensure!(
                (2.0 * *coefficient).is_finite(),
                "coordinate-swap operation {} coefficient overflows its reflection scale",
                operation.id
            );
        }
        for direction in operation.action.direction_ids() {
            ensure!(
                direction_ids.contains(direction),
                "operation {} references unknown direction {}",
                operation.id,
                direction
            );
        }
        if action_requires_unit_l2(&operation.action) {
            for direction in operation.action.direction_ids() {
                if let Some(normalization) =
                    direction_normalization.get(direction).copied().flatten()
                {
                    ensure!(
                        normalization == Normalization::UnitL2,
                        "operation {} requires unit_l2 direction {}",
                        operation.id,
                        direction
                    );
                }
            }
        }
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

pub(super) fn validate_muse_artifact_plan(plan: &LensPlan) -> Result<()> {
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
        !plan.lenses.is_empty(),
        "ordinary Qwen Lens plans must declare at least one lens"
    );
    ensure!(
        plan.directions
            .iter()
            .all(|direction| direction.lens_row().is_some()),
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

pub(super) fn prepare_execution_plan(
    plan: &LensPlan,
    plan_dir: &Path,
    loaded: &qwen_llm::runtime::LoadedModel,
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
    for operation in &plan.operations {
        let layers = operation
            .scope
            .layers
            .expand(arch.n_layer, "operation.layers")?;
        for direction_id in operation.action.direction_ids() {
            let direction = direction_defs[direction_id].lens_row().with_context(|| {
                format!("ordinary runtime cannot load native hyper direction {direction_id}")
            })?;
            direction_layers
                .entry(direction_id)
                .or_default()
                .extend(layers.iter().copied());
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
    ensure_projected_full_direction_bank_budget(
        plan,
        &required_lens_layers,
        &raw_lens_layers,
        arch.hidden_size as usize,
    )?;

    let mut lenses = HashMap::new();
    for definition in &plan.lenses {
        let prepared = match definition {
            LensDefinition::NativeSelected { id, artifact } => PreparedLens {
                id: id.clone(),
                lens: LoadedLens::Native(load_native_lens(
                    &resolve_plan_path(plan_dir, artifact),
                    arch.n_layer,
                    arch.hidden_size as usize,
                    arch.vocab_size,
                )?),
                raw_lm_head: None,
            },
            LensDefinition::PublishedFullTransport {
                id,
                artifact,
                token_ids,
                allow_unvalidated_transfer: _,
            } => {
                let layers = required_lens_layers
                    .get(id.as_str())
                    .with_context(|| format!("published full transport lens {id} is not used"))?
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let projected = crate::full_lens::project_full_token_directions(
                    &resolve_plan_path(plan_dir, artifact),
                    token_ids,
                    &layers,
                    loaded,
                    crate::full_lens::FullTokenTargetCovector::DeployedLogitNumerator,
                )?;
                let raw_lm_head = raw_lens_layers
                    .get(id.as_str())
                    .map(|raw_layers| {
                        crate::full_lens::project_full_token_directions(
                            &resolve_plan_path(plan_dir, artifact),
                            token_ids,
                            &raw_layers.iter().copied().collect::<Vec<_>>(),
                            loaded,
                            crate::full_lens::FullTokenTargetCovector::RawLmHead,
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
            LensDefinition::WorkspaceTemplate {
                id,
                weights,
                labels,
            } => {
                let weights_path = resolve_plan_path(plan_dir, weights);
                let lens = TemplateLens::open(&weights_path)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                let vocabulary =
                    TemplateVocabulary::load(&resolve_plan_path(plan_dir, labels), lens.n_rows())
                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                ensure!(
                    lens.hidden_size() == arch.hidden_size as usize,
                    "template lens {} hidden size {} != model {}",
                    id,
                    lens.hidden_size(),
                    arch.hidden_size
                );
                PreparedLens {
                    id: id.clone(),
                    lens: LoadedLens::Template { lens, vocabulary },
                    raw_lm_head: None,
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
        let definition = direction_defs[direction_id];
        let definition = definition.lens_row().with_context(|| {
            format!("ordinary runtime cannot load native hyper direction {direction_id}")
        })?;
        let prepared_lens = &lenses[&definition.lens];
        let mut rows = BTreeMap::new();
        for layer in layers {
            let raw = direction_row(prepared_lens, definition, layer)?;
            let normalized = normalize_direction(raw, definition.normalization, direction_id)?;
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
    Ok(ExecutionPlan {
        plan: plan.clone(),
        lenses,
        directions,
        coordinate_swaps,
        n_layer: arch.n_layer,
        hidden_size: arch.hidden_size as usize,
        capture,
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

pub(super) fn action_requires_unit_l2(action: &Action) -> bool {
    matches!(
        action,
        Action::ResidualL2Fraction { .. }
            | Action::ProjectionAblate { .. }
            | Action::SourceToTarget { .. }
            | Action::CoordinateSwap { .. }
    )
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

pub(crate) fn normalize_direction(
    mut row: Vec<f32>,
    normalization: Normalization,
    id: &str,
) -> Result<Vec<f32>> {
    ensure!(!row.is_empty(), "direction {id} is empty");
    ensure!(
        row.iter().all(|value| value.is_finite()),
        "direction {id} has a non-finite value"
    );
    let norm_squared = row
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let norm = norm_squared.sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "direction {id} has a zero or non-finite norm"
    );
    if normalization == Normalization::UnitL2 {
        for value in &mut row {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    ensure!(
        row.iter().all(|value| value.is_finite()),
        "direction {id} normalization overflowed"
    );
    Ok(row)
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

pub(super) fn operation_enabled(operation: &OperationDefinition) -> bool {
    operation.action.coefficient() != 0.0
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
    Ok(match action {
        Action::FixedAdd {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Fixed {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ResidualL2Fraction {
            direction: id,
            coefficient,
        } => PostBlockIntervention::ResidualL2Relative {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ProjectionAblate {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Projection {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::SourceToTarget {
            source,
            target,
            coefficient,
        } => PostBlockIntervention::SourceToTarget {
            layer,
            source: direction(source)?,
            target: direction(target)?,
            coefficient: *coefficient,
        },
        Action::CoordinateSwap { coefficient, .. } => PostBlockIntervention::Projection {
            layer,
            direction: coordinate_swaps
                .get(operation_id)
                .and_then(|prepared| prepared.rows.get(&layer))
                .with_context(|| {
                    format!(
                        "coordinate swap {operation_id} has no reflection direction at layer {layer}"
                    )
                })?,
            coefficient: 2.0 * *coefficient,
        },
    })
}

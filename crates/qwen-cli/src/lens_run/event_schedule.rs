use super::{LensPlan, Phase, Scope, Selector, operation_enabled};
use anyhow::{Context, Result, bail, ensure};

#[derive(Clone, Debug, Eq, PartialEq)]
enum CompiledSelector {
    All,
    Values(Box<[u32]>),
    Range { start: u32, end: u32 },
}

impl CompiledSelector {
    fn compile(selector: &Selector, upper_bound: Option<u32>) -> Result<Self> {
        let compiled = match selector {
            Selector::All => Self::All,
            Selector::Values { values } => {
                let mut copied = Vec::new();
                copied
                    .try_reserve_exact(values.len())
                    .context("allocate compiled Lens selector")?;
                copied.extend_from_slice(values);
                Self::Values(copied.into_boxed_slice())
            }
            Selector::Range { start, end } => Self::Range {
                start: *start,
                end: *end,
            },
            Selector::RenderedSpans { .. } => {
                bail!("execution plan contains an unresolved selector")
            }
        };
        if let Some(upper_bound) = upper_bound {
            ensure!(
                compiled.values_within(upper_bound),
                "compiled Lens layer selector exceeds model layer count {upper_bound}"
            );
        }
        Ok(compiled)
    }

    fn contains(&self, value: u32) -> bool {
        match self {
            Self::All => true,
            Self::Values(values) => values.binary_search(&value).is_ok(),
            Self::Range { start, end } => (*start..=*end).contains(&value),
        }
    }

    fn values_within(&self, upper_bound: u32) -> bool {
        match self {
            Self::All => true,
            Self::Values(values) => values.iter().all(|&value| value < upper_bound),
            Self::Range { end, .. } => *end < upper_bound,
        }
    }

    fn matches(&self, selector: &Selector) -> bool {
        match (self, selector) {
            (Self::All, Selector::All) => true,
            (Self::Values(compiled), Selector::Values { values }) => {
                compiled.as_ref() == values.as_slice()
            }
            (
                Self::Range {
                    start: compiled_start,
                    end: compiled_end,
                },
                Selector::Range { start, end },
            ) => compiled_start == start && compiled_end == end,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompiledScope {
    layers: CompiledSelector,
    prefill: Option<CompiledSelector>,
    decode: Option<CompiledSelector>,
}

impl CompiledScope {
    fn compile(scope: &Scope, layer_count: u32) -> Result<Self> {
        Ok(Self {
            layers: CompiledSelector::compile(&scope.layers, Some(layer_count))?,
            prefill: scope
                .prefill
                .as_ref()
                .map(|selector| CompiledSelector::compile(selector, None))
                .transpose()?,
            decode: scope
                .decode
                .as_ref()
                .map(|selector| CompiledSelector::compile(selector, None))
                .transpose()?,
        })
    }

    fn phase_matches(&self, phase: Phase) -> Result<bool> {
        let index = u32::try_from(phase.index()).context("event index exceeds u32")?;
        Ok(match phase {
            Phase::Prefill(_) => self
                .prefill
                .as_ref()
                .is_some_and(|selector| selector.contains(index)),
            Phase::Decode(_) => self
                .decode
                .as_ref()
                .is_some_and(|selector| selector.contains(index)),
        })
    }

    fn matches(&self, scope: &Scope) -> bool {
        self.layers.matches(&scope.layers)
            && optional_selector_matches(self.prefill.as_ref(), scope.prefill.as_ref())
            && optional_selector_matches(self.decode.as_ref(), scope.decode.as_ref())
    }
}

fn optional_selector_matches(
    compiled: Option<&CompiledSelector>,
    selector: Option<&Selector>,
) -> bool {
    match (compiled, selector) {
        (Some(compiled), Some(selector)) => compiled.matches(selector),
        (None, None) => true,
        _ => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScheduledDefinition {
    id: String,
    scope: CompiledScope,
}

#[derive(Default)]
pub(super) struct CompiledEvent {
    operation_indices: Vec<usize>,
    operation_topology_indices: Vec<usize>,
    readout_indices: Vec<usize>,
    capture_layers: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PassivePrefillSpan {
    pub(super) start: usize,
    pub(super) end: usize,
}

impl CompiledEvent {
    pub(super) fn operation_indices(&self) -> &[usize] {
        &self.operation_indices
    }

    pub(super) fn operation_topology_indices(&self) -> &[usize] {
        &self.operation_topology_indices
    }

    pub(super) fn readout_indices(&self) -> &[usize] {
        &self.readout_indices
    }

    pub(super) fn capture_layers(&self) -> &[u32] {
        &self.capture_layers
    }
}

pub(super) struct CompiledEventSchedule {
    operations: Vec<ScheduledDefinition>,
    readouts: Vec<ScheduledDefinition>,
    layer_count: u32,
}

impl CompiledEventSchedule {
    pub(super) fn compile(plan: &LensPlan, layer_count: u32) -> Result<Self> {
        let mut operations = Vec::new();
        operations
            .try_reserve_exact(plan.operations.len())
            .context("allocate compiled Lens operation schedule")?;
        for operation in &plan.operations {
            operations.push(ScheduledDefinition {
                id: operation.id.clone(),
                scope: CompiledScope::compile(&operation.scope, layer_count)?,
            });
        }

        let mut readouts = Vec::new();
        readouts
            .try_reserve_exact(plan.readouts.len())
            .context("allocate compiled Lens readout schedule")?;
        for readout in &plan.readouts {
            readouts.push(ScheduledDefinition {
                id: readout.id.clone(),
                scope: CompiledScope::compile(&readout.scope, layer_count)?,
            });
        }

        Ok(Self {
            operations,
            readouts,
            layer_count,
        })
    }

    pub(super) fn bind<'schedule, 'plan>(
        &'schedule self,
        plan: &'plan LensPlan,
    ) -> Result<BoundEventSchedule<'schedule, 'plan>> {
        ensure!(
            self.operations.len() == plan.operations.len()
                && self.readouts.len() == plan.readouts.len(),
            "Lens plan shape changed after event schedule compilation"
        );
        for (compiled, operation) in self.operations.iter().zip(&plan.operations) {
            ensure!(
                compiled.id == operation.id && compiled.scope.matches(&operation.scope),
                "Lens operation topology changed after event schedule compilation"
            );
        }
        for (compiled, readout) in self.readouts.iter().zip(&plan.readouts) {
            ensure!(
                compiled.id == readout.id && compiled.scope.matches(&readout.scope),
                "Lens readout topology changed after event schedule compilation"
            );
        }
        Ok(BoundEventSchedule {
            schedule: self,
            plan,
        })
    }
}

pub(super) struct BoundEventSchedule<'schedule, 'plan> {
    schedule: &'schedule CompiledEventSchedule,
    plan: &'plan LensPlan,
}

impl BoundEventSchedule<'_, '_> {
    pub(super) fn new_event(&self) -> Result<CompiledEvent> {
        let mut event = CompiledEvent::default();
        event
            .operation_indices
            .try_reserve_exact(self.schedule.operations.len())
            .context("allocate Lens event operation indices")?;
        event
            .operation_topology_indices
            .try_reserve_exact(self.schedule.operations.len())
            .context("allocate Lens event operation topology indices")?;
        event
            .readout_indices
            .try_reserve_exact(self.schedule.readouts.len())
            .context("allocate Lens event readout indices")?;
        event
            .capture_layers
            .try_reserve_exact(
                usize::try_from(self.schedule.layer_count)
                    .context("Lens layer count exceeds host address space")?,
            )
            .context("allocate Lens event capture layers")?;
        Ok(event)
    }

    pub(super) fn populate(&self, phase: Phase, event: &mut CompiledEvent) -> Result<()> {
        event.operation_indices.clear();
        event.operation_topology_indices.clear();
        event.readout_indices.clear();
        event.capture_layers.clear();

        for (definition_index, compiled) in self.schedule.operations.iter().enumerate() {
            if compiled.scope.phase_matches(phase)? {
                event.operation_topology_indices.push(definition_index);
                if operation_enabled(&self.plan.operations[definition_index]) {
                    event.operation_indices.push(definition_index);
                }
            }
        }
        for (definition_index, compiled) in self.schedule.readouts.iter().enumerate() {
            if compiled.scope.phase_matches(phase)? {
                event.readout_indices.push(definition_index);
            }
        }
        for layer in 0..self.schedule.layer_count {
            if event.readout_indices.iter().any(|&definition_index| {
                self.schedule.readouts[definition_index]
                    .scope
                    .layers
                    .contains(layer)
            }) {
                event.capture_layers.push(layer);
            }
        }
        Ok(())
    }

    pub(super) fn passive_prefill_spans(
        &self,
        prompt_len: usize,
        min_span_len: usize,
    ) -> Result<Vec<PassivePrefillSpan>> {
        ensure!(prompt_len > 0, "Lens prompt must not be empty");
        ensure!(
            min_span_len > 0,
            "packed prefill minimum span must be positive"
        );
        let final_prompt_index = prompt_len - 1;
        let mut span_start = None;
        let mut spans = Vec::new();
        for index in 0..final_prompt_index {
            let phase = Phase::Prefill(index);
            let mut has_operation = false;
            for (definition_index, compiled) in self.schedule.operations.iter().enumerate() {
                if operation_enabled(&self.plan.operations[definition_index])
                    && compiled.scope.phase_matches(phase)?
                {
                    has_operation = true;
                    break;
                }
            }
            let mut has_readout = false;
            for compiled in &self.schedule.readouts {
                if compiled.scope.phase_matches(phase)? {
                    has_readout = true;
                    break;
                }
            }
            let passive = !has_operation && !has_readout;
            match (span_start, passive) {
                (None, true) => span_start = Some(index),
                (Some(start), false) => {
                    if index - start >= min_span_len {
                        spans.push(PassivePrefillSpan { start, end: index });
                    }
                    span_start = None;
                }
                _ => {}
            }
        }
        if let Some(start) = span_start
            && final_prompt_index - start >= min_span_len
        {
            spans.push(PassivePrefillSpan {
                start,
                end: final_prompt_index,
            });
        }
        Ok(spans)
    }

    pub(super) fn plan(&self) -> &LensPlan {
        self.plan
    }

    pub(super) fn operation_selects_layer(&self, definition_index: usize, layer: u32) -> bool {
        self.schedule.operations[definition_index]
            .scope
            .layers
            .contains(layer)
    }

    pub(super) fn readout_selects_layer(&self, definition_index: usize, layer: u32) -> bool {
        self.schedule.readouts[definition_index]
            .scope
            .layers
            .contains(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens_run::{
        Action, LensDefinition, OperationDefinition, ReadoutDefinition, scope_matches,
    };
    use std::path::PathBuf;

    fn plan() -> LensPlan {
        LensPlan {
            version: 1,
            lenses: vec![LensDefinition::WorkspaceTemplate {
                id: "lens".into(),
                weights: PathBuf::from("weights"),
                labels: PathBuf::from("labels"),
            }],
            directions: Vec::new(),
            operations: vec![
                OperationDefinition {
                    id: "late".into(),
                    scope: Scope {
                        layers: Selector::Values { values: vec![2] },
                        prefill: Some(Selector::All),
                        decode: None,
                    },
                    action: Action::FixedAdd {
                        direction: "a".into(),
                        coefficient: 1.0,
                    },
                },
                OperationDefinition {
                    id: "wide".into(),
                    scope: Scope {
                        layers: Selector::Range { start: 0, end: 2 },
                        prefill: Some(Selector::Values { values: vec![1] }),
                        decode: Some(Selector::Range { start: 0, end: 1 }),
                    },
                    action: Action::FixedAdd {
                        direction: "b".into(),
                        coefficient: 1.0,
                    },
                },
                OperationDefinition {
                    id: "disabled".into(),
                    scope: Scope {
                        layers: Selector::All,
                        prefill: Some(Selector::All),
                        decode: Some(Selector::All),
                    },
                    action: Action::FixedAdd {
                        direction: "c".into(),
                        coefficient: -0.0,
                    },
                },
            ],
            readouts: vec![
                ReadoutDefinition {
                    id: "outer".into(),
                    lens: "lens".into(),
                    scope: Scope {
                        layers: Selector::Values { values: vec![0, 2] },
                        prefill: Some(Selector::Values { values: vec![1] }),
                        decode: None,
                    },
                    top_k: 1,
                },
                ReadoutDefinition {
                    id: "inner".into(),
                    lens: "lens".into(),
                    scope: Scope {
                        layers: Selector::Values { values: vec![1] },
                        prefill: Some(Selector::Values { values: vec![1] }),
                        decode: Some(Selector::Values { values: vec![0] }),
                    },
                    top_k: 1,
                },
            ],
        }
    }

    fn scheduled_operation_sites(
        schedule: &BoundEventSchedule<'_, '_>,
        event: &CompiledEvent,
        layer_count: u32,
    ) -> Vec<(usize, u32)> {
        let mut sites = Vec::new();
        for layer in 0..layer_count {
            for &definition_index in event.operation_indices() {
                if schedule.operation_selects_layer(definition_index, layer) {
                    sites.push((definition_index, layer));
                }
            }
        }
        sites
    }

    fn scheduled_readout_sites(
        schedule: &BoundEventSchedule<'_, '_>,
        event: &CompiledEvent,
    ) -> Vec<(usize, u32)> {
        let mut sites = Vec::new();
        for &definition_index in event.readout_indices() {
            for &layer in event.capture_layers() {
                if schedule.readout_selects_layer(definition_index, layer) {
                    sites.push((definition_index, layer));
                }
            }
        }
        sites
    }

    #[test]
    fn schedule_matches_dynamic_scope_order_for_every_test_event() {
        let plan = plan();
        let compiled = CompiledEventSchedule::compile(&plan, 3).unwrap();
        let schedule = compiled.bind(&plan).unwrap();
        let mut event = schedule.new_event().unwrap();

        for phase in [
            Phase::Prefill(0),
            Phase::Prefill(1),
            Phase::Prefill(2),
            Phase::Decode(0),
            Phase::Decode(1),
            Phase::Decode(2),
        ] {
            schedule.populate(phase, &mut event).unwrap();
            let mut expected_operations = Vec::new();
            for layer in 0..3 {
                for (definition_index, operation) in plan.operations.iter().enumerate() {
                    if operation_enabled(operation)
                        && scope_matches(&operation.scope, phase, layer).unwrap()
                    {
                        expected_operations.push((definition_index, layer));
                    }
                }
            }
            assert_eq!(
                scheduled_operation_sites(&schedule, &event, 3),
                expected_operations
            );

            let mut expected_readouts = Vec::new();
            for (definition_index, readout) in plan.readouts.iter().enumerate() {
                for layer in 0..3 {
                    if scope_matches(&readout.scope, phase, layer).unwrap() {
                        expected_readouts.push((definition_index, layer));
                    }
                }
            }
            assert_eq!(
                scheduled_readout_sites(&schedule, &event),
                expected_readouts
            );
            let expected_capture_layers = (0..3)
                .filter(|layer| {
                    expected_readouts
                        .iter()
                        .any(|(_, selected_layer)| selected_layer == layer)
                })
                .collect::<Vec<_>>();
            assert_eq!(event.capture_layers(), expected_capture_layers);
        }
    }

    #[test]
    fn schedule_reuses_topology_across_zero_and_nonzero_sweep_arms() {
        let source = plan();
        let compiled = CompiledEventSchedule::compile(&source, 3).unwrap();

        for coefficient in [0.0, 0.75, -0.0] {
            let mut arm = source.clone();
            arm.operations[1].action = Action::FixedAdd {
                direction: "b".into(),
                coefficient,
            };
            let schedule = compiled.bind(&arm).unwrap();
            let mut event = schedule.new_event().unwrap();
            schedule.populate(Phase::Prefill(1), &mut event).unwrap();
            assert_eq!(event.operation_indices().contains(&1), coefficient != 0.0);
            assert!(event.operation_topology_indices().contains(&1));
        }
    }

    #[test]
    fn passive_spans_are_maximal_thresholded_and_exclude_the_final_prompt_token() {
        let mut source = plan();
        source.readouts.clear();
        source.operations.truncate(2);
        source.operations[0].scope.prefill = Some(Selector::Values { values: vec![2] });
        source.operations[1].scope.prefill = Some(Selector::Values { values: vec![5] });
        let compiled = CompiledEventSchedule::compile(&source, 3).unwrap();
        let schedule = compiled.bind(&source).unwrap();

        assert_eq!(
            schedule.passive_prefill_spans(8, 2).unwrap(),
            vec![
                PassivePrefillSpan { start: 0, end: 2 },
                PassivePrefillSpan { start: 3, end: 5 },
            ]
        );
        assert_eq!(
            schedule.passive_prefill_spans(8, 1).unwrap(),
            vec![
                PassivePrefillSpan { start: 0, end: 2 },
                PassivePrefillSpan { start: 3, end: 5 },
                PassivePrefillSpan { start: 6, end: 7 },
            ]
        );
        assert!(schedule.passive_prefill_spans(1, 1).unwrap().is_empty());
        assert!(schedule.passive_prefill_spans(0, 1).is_err());
        assert!(schedule.passive_prefill_spans(8, 0).is_err());

        let mut zero_arm = source.clone();
        zero_arm.operations[1].action.set_coefficient(-0.0);
        let zero_schedule = compiled.bind(&zero_arm).unwrap();
        assert_eq!(
            zero_schedule.passive_prefill_spans(8, 2).unwrap(),
            vec![
                PassivePrefillSpan { start: 0, end: 2 },
                PassivePrefillSpan { start: 3, end: 7 },
            ]
        );

        let mut readout_source = plan();
        readout_source.operations.clear();
        readout_source.readouts.truncate(1);
        readout_source.readouts[0].scope.prefill = Some(Selector::Values { values: vec![3] });
        let readout_compiled = CompiledEventSchedule::compile(&readout_source, 3).unwrap();
        let readout_schedule = readout_compiled.bind(&readout_source).unwrap();
        assert_eq!(
            readout_schedule.passive_prefill_spans(7, 2).unwrap(),
            vec![
                PassivePrefillSpan { start: 0, end: 3 },
                PassivePrefillSpan { start: 4, end: 6 },
            ]
        );
    }

    #[test]
    fn schedule_rejects_stale_or_unresolved_topology() {
        let mut plan = plan();
        let compiled = CompiledEventSchedule::compile(&plan, 3).unwrap();

        plan.operations.swap(0, 1);
        assert!(compiled.bind(&plan).is_err());
        plan.operations.swap(0, 1);
        plan.operations[0].scope.layers = Selector::Values { values: vec![1] };
        assert!(compiled.bind(&plan).is_err());

        plan.operations[0].scope.layers = Selector::RenderedSpans {
            selectors: Vec::new(),
        };
        assert!(CompiledEventSchedule::compile(&plan, 3).is_err());
    }
}

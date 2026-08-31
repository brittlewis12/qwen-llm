use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::lens_inspect::{self, Cell, TraceDocument, VectorCell};
use super::lens_run::{self, CoefficientSweepManifest, LensPlan};
use super::{read_regular_file_bounded, read_regular_file_exact};

const COMPARE_MAX_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_LIMIT: usize = 25;
const SWEEP_MANIFEST_MAX_BYTES: usize = 16 * 1024 * 1024;
const SWEEP_INSPECT_MAX_TOTAL_BYTES: usize = 128 * 1024 * 1024;
const SWEEP_INSPECT_MAX_DETAILS: usize = 1024;
const RUN_METADATA_STRING_MAX_BYTES: usize = 16 * 1024;
const RUN_DECODED_TEXT_MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Args)]
pub(crate) struct CompareArgs {
    /// Left regular non-symlink trace or run JSON artifact.
    left: PathBuf,
    /// Right regular non-symlink trace or run JSON artifact.
    right: PathBuf,
    /// Render concise text or a typed comparison result.
    #[arg(long, value_enum, default_value_t = CompareFormat::Text)]
    format: CompareFormat,
    /// Maximum detail rows or cells included in the result.
    #[arg(long, default_value_t = DEFAULT_LIMIT)]
    limit: usize,
}

#[derive(Debug, Args)]
pub(crate) struct InspectSweepArgs {
    /// New-style qwen.lens.coefficient_sweep directory.
    sweep: PathBuf,
    /// Arm compared against every other arm; defaults to the first numeric zero.
    #[arg(long)]
    reference_arm: Option<usize>,
    /// Render concise text or a typed inspection result.
    #[arg(long, value_enum, default_value_t = InspectSweepFormat::Text)]
    format: InspectSweepFormat,
    /// Maximum exact comparison details retained across the complete report.
    #[arg(long, default_value_t = DEFAULT_LIMIT)]
    limit: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CompareFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InspectSweepFormat {
    Text,
    Json,
}

#[derive(Deserialize)]
struct Envelope {
    schema: String,
    schema_version: u32,
}

#[derive(Debug, Serialize)]
#[serde(tag = "comparison_kind", rename_all = "snake_case")]
enum ComparisonResult {
    Trace(TraceComparison),
    Run(RunComparison),
}

pub(crate) fn run(args: CompareArgs) -> Result<()> {
    ensure!(args.limit > 0, "--limit must be positive");
    let left_bytes = read_regular_file_bounded(&args.left, COMPARE_MAX_BYTES)?;
    let right_bytes = read_regular_file_bounded(&args.right, COMPARE_MAX_BYTES)?;
    let left_envelope: Envelope = serde_json::from_slice(&left_bytes)
        .with_context(|| format!("parse artifact envelope {}", args.left.display()))?;
    let right_envelope: Envelope = serde_json::from_slice(&right_bytes)
        .with_context(|| format!("parse artifact envelope {}", args.right.display()))?;
    ensure!(
        left_envelope.schema == right_envelope.schema,
        "mixed schemas are not comparable: {:?} versus {:?}",
        left_envelope.schema,
        right_envelope.schema
    );
    let result = match left_envelope.schema.as_str() {
        "qwen.lens.trace" => {
            ensure!(
                matches!(left_envelope.schema_version, 2 | 3)
                    && matches!(right_envelope.schema_version, 2 | 3),
                "trace comparison supports schema versions 2 and 3 only"
            );
            let left = lens_inspect::parse_trace_bytes(&left_bytes, &args.left)?;
            let right = lens_inspect::parse_trace_bytes(&right_bytes, &args.right)?;
            ComparisonResult::Trace(compare_traces(&left, &right, args.limit)?)
        }
        "qwen.lens.run" => {
            ensure!(
                matches!(left_envelope.schema_version, 1 | 2)
                    && left_envelope.schema_version == right_envelope.schema_version,
                "run comparison supports same-version schema 1 or 2 pairs only"
            );
            let left = parse_run_bytes(&left_bytes, &args.left)?;
            let right = parse_run_bytes(&right_bytes, &args.right)?;
            ComparisonResult::Run(compare_runs(&left, &right, args.limit)?)
        }
        schema => bail!("unsupported comparison schema {schema:?}"),
    };
    match args.format {
        CompareFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
        CompareFormat::Text => print_text(&result),
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct SweepInspection {
    inspection_kind: &'static str,
    sweep_schema_version: u32,
    bundle_integrity: &'static str,
    run_validation: &'static str,
    source_plan_identity: &'static str,
    sweep_root: PathBuf,
    manifest_blake3: String,
    operation_id: String,
    canonical_source_plan_path: PathBuf,
    runtime_kind: String,
    model_path: PathBuf,
    reference_arm: usize,
    reference_coefficient: f32,
    total_child_bytes: u64,
    arms: Vec<SweepInspectionArm>,
    duplicate_coefficient_groups: Vec<DuplicateCoefficientGroup>,
    generation_groups: Vec<SweepGenerationGroup>,
    detail_limit: usize,
    retained_detail_count: usize,
}

#[derive(Debug, Serialize)]
struct SweepInspectionArm {
    index: usize,
    coefficient: f32,
    coefficient_bits: String,
    artifact: String,
    byte_length: u64,
    blake3: String,
    generation_group: usize,
    generated_text: String,
    generated_token_ids: Vec<i32>,
    stop_reason: String,
    selected_operation_application_count: usize,
    total_operation_application_count: usize,
    live_readout_count: usize,
    reference_comparison: Option<SweepReferenceComparison>,
}

#[derive(Debug, Serialize)]
struct SweepReferenceComparison {
    plans_equal: bool,
    first_generated_token_divergence: Option<GeneratedDivergence>,
    stop_reason_changed: bool,
    reference_operation_application_count: usize,
    arm_operation_application_count: usize,
    matched_readout_count: usize,
    changed_readout_count: usize,
    candidate_difference_count: usize,
    unmatched_readout_count: usize,
    changed_readouts: Vec<MatchedReadout>,
    unmatched_readouts: Vec<UnmatchedReadout>,
}

#[derive(Debug, Serialize)]
struct DuplicateCoefficientGroup {
    coefficient: f32,
    coefficient_bits: String,
    arm_indices: Vec<usize>,
    byte_identical: bool,
}

#[derive(Debug, Serialize)]
struct SweepGenerationGroup {
    id: usize,
    arm_indices: Vec<usize>,
    generated_token_ids: Vec<i32>,
    decoded_text: String,
    stop_reason: String,
}

type SweepGenerationKey = (Vec<i32>, String, String);

struct LoadedSweep {
    root: PathBuf,
    manifest_blake3: String,
    manifest: CoefficientSweepManifest,
    arms: Vec<LoadedSweepArm>,
    total_child_bytes: u64,
}

struct LoadedSweepArm {
    index: usize,
    coefficient: f32,
    artifact: String,
    byte_length: u64,
    blake3: String,
    document: RunDocument,
}

pub(crate) fn inspect_sweep(args: InspectSweepArgs) -> Result<()> {
    ensure!(
        args.limit > 0 && args.limit <= SWEEP_INSPECT_MAX_DETAILS,
        "--limit must be in 1..={SWEEP_INSPECT_MAX_DETAILS}"
    );
    let loaded = load_sweep(&args.sweep)?;
    let reference_arm = args.reference_arm.unwrap_or_else(|| {
        loaded
            .arms
            .iter()
            .position(|arm| arm.coefficient == 0.0)
            .unwrap_or(0)
    });
    ensure!(
        reference_arm < loaded.arms.len(),
        "--reference-arm {reference_arm} is outside 0..{}",
        loaded.arms.len()
    );
    let reference = &loaded.arms[reference_arm].document;
    let (generation_groups, generation_group_by_arm) = sweep_generation_groups(&loaded.arms);
    let duplicate_coefficient_groups = duplicate_coefficient_groups(&loaded.arms);
    let operation_id = loaded.manifest.operation_id.clone();
    let mut arms = Vec::with_capacity(loaded.arms.len());
    let mut remaining_details = args.limit;
    for arm in &loaded.arms {
        let selected_operation_application_count = arm
            .document
            .operation_applications
            .iter()
            .filter(|application| application.id == operation_id)
            .count();
        let reference_comparison = (arm.index != reference_arm)
            .then(|| compare_sweep_runs(reference, &arm.document, &mut remaining_details))
            .transpose()?;
        arms.push(SweepInspectionArm {
            index: arm.index,
            coefficient: arm.coefficient,
            coefficient_bits: coefficient_bits(arm.coefficient),
            artifact: arm.artifact.clone(),
            byte_length: arm.byte_length,
            blake3: arm.blake3.clone(),
            generation_group: generation_group_by_arm[arm.index],
            generated_text: arm.document.decoded_text.clone(),
            generated_token_ids: arm.document.generated_token_ids.clone(),
            stop_reason: arm.document.stop_reason.clone(),
            selected_operation_application_count,
            total_operation_application_count: arm.document.operation_applications.len(),
            live_readout_count: arm.document.live_readouts.len(),
            reference_comparison,
        });
    }
    let result = SweepInspection {
        inspection_kind: "coefficient_sweep",
        sweep_schema_version: loaded.manifest.schema_version,
        bundle_integrity: "verified",
        run_validation: "strict_known_qwen_lens_run_v1",
        source_plan_identity: "unverifiable_manifest_v1",
        sweep_root: loaded.root,
        manifest_blake3: loaded.manifest_blake3,
        operation_id,
        canonical_source_plan_path: loaded.manifest.canonical_source_plan_path,
        runtime_kind: reference.runtime_kind.clone(),
        model_path: reference.model_path.clone(),
        reference_arm,
        reference_coefficient: loaded.arms[reference_arm].coefficient,
        total_child_bytes: loaded.total_child_bytes,
        arms,
        duplicate_coefficient_groups,
        generation_groups,
        detail_limit: args.limit,
        retained_detail_count: args.limit - remaining_details,
    };
    match args.format {
        InspectSweepFormat::Text => print_sweep_inspection(&result),
        InspectSweepFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
    }
    Ok(())
}

fn load_sweep(path: &Path) -> Result<LoadedSweep> {
    let root = canonical_real_directory(path, "sweep root")?;
    ensure_directory_entries(
        &root,
        [OsString::from("arms"), OsString::from("manifest.json")]
            .into_iter()
            .collect(),
        "sweep root",
    )?;
    let manifest_path = root.join("manifest.json");
    let manifest_bytes = read_regular_file_bounded(&manifest_path, SWEEP_MANIFEST_MAX_BYTES)?;
    let manifest = lens_run::parse_sweep_manifest_bytes(&manifest_bytes)
        .with_context(|| format!("validate sweep manifest {}", manifest_path.display()))?;
    let manifest_blake3 = blake3::hash(&manifest_bytes).to_hex().to_string();

    let arms_root = root.join("arms");
    require_real_directory(&arms_root, "sweep arms directory")?;
    let expected_arm_entries = manifest
        .arms
        .iter()
        .map(|arm| OsString::from(format!("{:06}", arm.index)))
        .collect();
    ensure_directory_entries(&arms_root, expected_arm_entries, "sweep arms directory")?;

    let mut total_child_bytes = 0usize;
    let mut arms = Vec::with_capacity(manifest.arms.len());
    let mut plans = Vec::with_capacity(manifest.arms.len());
    for arm in &manifest.arms {
        let arm_directory = arms_root.join(format!("{:06}", arm.index));
        require_real_directory(&arm_directory, "sweep arm directory")?;
        ensure_directory_entries(
            &arm_directory,
            [OsString::from("run.json")].into_iter().collect(),
            "sweep arm directory",
        )?;
        let byte_length = usize::try_from(arm.byte_length)
            .context("sweep child byte length does not fit this platform")?;
        ensure!(
            byte_length <= COMPARE_MAX_BYTES,
            "sweep child {} exceeds the {} byte run limit",
            arm.index,
            COMPARE_MAX_BYTES
        );
        total_child_bytes = total_child_bytes
            .checked_add(byte_length)
            .context("sweep child byte total overflow")?;
        ensure!(
            total_child_bytes <= SWEEP_INSPECT_MAX_TOTAL_BYTES,
            "sweep children total {} bytes exceeds inspection limit {}",
            total_child_bytes,
            SWEEP_INSPECT_MAX_TOTAL_BYTES
        );
        let run_path = arm_directory.join("run.json");
        let bytes = read_regular_file_exact(&run_path, byte_length)?;
        ensure!(
            blake3::hash(&bytes).to_hex().as_str() == arm.blake3,
            "sweep child {} BLAKE3 does not match manifest",
            arm.index
        );
        let document = parse_run_bytes(&bytes, &run_path)?;
        let plan = validate_sweep_child(&document, &manifest, arm)?;
        plans.push(plan);
        arms.push(LoadedSweepArm {
            index: arm.index,
            coefficient: arm.coefficient,
            artifact: arm.artifact.clone(),
            byte_length: arm.byte_length,
            blake3: arm.blake3.clone(),
            document,
        });
    }
    let reference_plan = plans
        .first()
        .context("coefficient sweep manifest has no arms")?;
    for (arm, plan) in manifest.arms.iter().zip(&plans) {
        let expected = lens_run::plan_with_operation_coefficient(
            reference_plan,
            &manifest.operation_id,
            arm.coefficient,
        )?;
        ensure!(
            expected == *plan,
            "sweep child {} plan differs beyond the selected operation coefficient",
            arm.index
        );
    }
    let reference = &arms
        .first()
        .context("coefficient sweep manifest has no loaded arms")?
        .document;
    ensure_sweep_readout_semantics(&arms)?;
    for arm in &arms {
        ensure_sweep_run_context(reference, &arm.document)?;
    }
    Ok(LoadedSweep {
        root,
        manifest_blake3,
        manifest,
        arms,
        total_child_bytes: total_child_bytes as u64,
    })
}

fn validate_sweep_child(
    document: &RunDocument,
    manifest: &CoefficientSweepManifest,
    arm: &lens_run::CoefficientSweepArm,
) -> Result<LensPlan> {
    ensure!(
        document.schema_version == 1
            && document.runtime_kind == "ordinary_qwen"
            && document.execution_binding.is_none()
            && document.native_hyper_captures.is_empty(),
        "sweep child {} must be an ordinary qwen.lens.run v1 without native captures",
        arm.index
    );
    ensure!(
        document.decoded_text.len() <= RUN_DECODED_TEXT_MAX_BYTES
            && document
                .operation_applications
                .iter()
                .all(|application| { application.id.len() <= RUN_METADATA_STRING_MAX_BYTES })
            && document.live_readouts.iter().all(|readout| {
                [
                    readout.id.as_str(),
                    readout.lens.as_str(),
                    readout.method.as_str(),
                    readout.score_kind.as_str(),
                    readout.candidate_universe.as_str(),
                ]
                .iter()
                .all(|value| value.len() <= RUN_METADATA_STRING_MAX_BYTES)
                    && readout.scores.iter().all(|score| {
                        score
                            .label
                            .as_ref()
                            .is_none_or(|label| label.len() <= RUN_METADATA_STRING_MAX_BYTES)
                    })
            }),
        "sweep child {} exceeds bounded string lengths",
        arm.index
    );
    ensure!(
        document.canonical_plan_path == manifest.canonical_source_plan_path,
        "sweep child {} canonical source-plan path differs from manifest",
        arm.index
    );
    let plan: LensPlan = serde_json::from_value(document.plan.clone())
        .with_context(|| format!("parse sweep child {} effective plan", arm.index))?;
    lens_run::validate_sweep_effective_plan(&plan, &manifest.operation_id)
        .with_context(|| format!("validate sweep child {} effective plan", arm.index))?;
    let operation = plan
        .operations
        .iter()
        .find(|operation| operation.id == manifest.operation_id)
        .with_context(|| {
            format!(
                "sweep child {} has no selected operation {:?}",
                arm.index, manifest.operation_id
            )
        })?;
    ensure!(
        operation.action.coefficient().to_bits() == arm.coefficient.to_bits(),
        "sweep child {} selected coefficient differs from manifest",
        arm.index
    );
    let requested = serde_json::to_value(&plan.readouts)?;
    ensure!(
        requested == serde_json::Value::Array(document.requested_live_readouts.clone()),
        "sweep child {} requested readouts differ from its effective plan",
        arm.index
    );
    let operations = plan
        .operations
        .iter()
        .map(|operation| (operation.id.as_str(), operation))
        .collect::<BTreeMap<_, _>>();
    let mut application_keys = BTreeSet::new();
    for application in &document.operation_applications {
        let operation = operations.get(application.id.as_str()).with_context(|| {
            format!(
                "sweep child {} records unknown operation {}",
                arm.index, application.id
            )
        })?;
        ensure!(
            sweep_scope_matches(
                &operation.scope,
                &application.phase,
                application.index,
                application.layer,
                document,
            ),
            "sweep child {} operation application is outside its effective scope",
            arm.index
        );
        ensure!(
            application_keys.insert((
                application.id.as_str(),
                application.layer,
                application.phase.as_str(),
                application.index,
            )),
            "sweep child {} repeats an operation application",
            arm.index
        );
    }
    let readout_definitions = plan
        .readouts
        .iter()
        .map(|readout| (readout.id.as_str(), readout))
        .collect::<BTreeMap<_, _>>();
    readout_map(document)?;
    let mut readout_sites = BTreeSet::new();
    let mut readout_semantics = BTreeMap::new();
    for readout in &document.live_readouts {
        let definition = readout_definitions
            .get(readout.id.as_str())
            .with_context(|| {
                format!(
                    "sweep child {} emits unknown readout {}",
                    arm.index, readout.id
                )
            })?;
        ensure!(
            readout.lens == definition.lens
                && readout.scores.len() <= definition.top_k
                && sweep_scope_matches(
                    &definition.scope,
                    &readout.phase,
                    readout.index,
                    readout.source_layer,
                    document,
                ),
            "sweep child {} live readout differs from its effective definition or scope",
            arm.index
        );
        ensure!(
            readout_sites.insert((
                readout.id.as_str(),
                readout.source_layer,
                readout.phase.as_str(),
                readout.index,
            )),
            "sweep child {} repeats a planned readout site",
            arm.index
        );
        let semantics = (
            readout.method.as_str(),
            readout.score_kind.as_str(),
            readout.candidate_universe.as_str(),
            readout.target_layer,
        );
        if let Some(existing) = readout_semantics.insert(readout.id.as_str(), semantics) {
            ensure!(
                existing == semantics,
                "sweep child {} changes one readout's score semantics across sites",
                arm.index
            );
        }
    }
    if arm.coefficient == 0.0 {
        ensure!(
            document
                .operation_applications
                .iter()
                .all(|application| application.id != manifest.operation_id),
            "sweep child {} zero arm records the disabled selected operation",
            arm.index
        );
    }
    Ok(plan)
}

fn sweep_scope_matches(
    scope: &lens_run::Scope,
    phase: &str,
    index: usize,
    layer: u32,
    document: &RunDocument,
) -> bool {
    let Ok(index_u32) = u32::try_from(index) else {
        return false;
    };
    let phase_selector = match phase {
        "prefill" if index < document.prompt_token_ids.len() => scope.prefill.as_ref(),
        "decode" if index < document.generated_token_ids.len().saturating_sub(1) => {
            scope.decode.as_ref()
        }
        _ => return false,
    };
    phase_selector.is_some_and(|selector| selector_matches(selector, index_u32))
        && selector_matches(&scope.layers, layer)
}

fn selector_matches(selector: &lens_run::Selector, value: u32) -> bool {
    match selector {
        lens_run::Selector::All => true,
        lens_run::Selector::Values { values } => values.binary_search(&value).is_ok(),
        lens_run::Selector::Range { start, end } => (*start..=*end).contains(&value),
    }
}

fn ensure_sweep_readout_semantics(arms: &[LoadedSweepArm]) -> Result<()> {
    let mut semantics = BTreeMap::<String, (String, String, String, Option<u32>)>::new();
    for arm in arms {
        for readout in &arm.document.live_readouts {
            let current = (
                readout.method.clone(),
                readout.score_kind.clone(),
                readout.candidate_universe.clone(),
                readout.target_layer,
            );
            if let Some(existing) = semantics.insert(readout.id.clone(), current.clone()) {
                ensure!(
                    existing == current,
                    "sweep readout {} changes score semantics across arms",
                    readout.id
                );
            }
        }
    }
    Ok(())
}

fn ensure_sweep_run_context(reference: &RunDocument, candidate: &RunDocument) -> Result<()> {
    ensure!(
        reference.schema_version == candidate.schema_version
            && reference.input_source == candidate.input_source
            && reference.max_new_tokens == candidate.max_new_tokens,
        "sweep child run schema or input/generation context differs"
    );
    ensure!(
        reference.prompt_token_ids == candidate.prompt_token_ids,
        "sweep child prompt token IDs differ"
    );
    ensure!(
        reference.runtime_kind == candidate.runtime_kind,
        "sweep child runtime kinds differ"
    );
    ensure!(
        reference.model_path == candidate.model_path,
        "sweep child model paths differ"
    );
    let bindings_match = match (&reference.execution_binding, &candidate.execution_binding) {
        (None, None) => true,
        (Some(left), Some(right)) => left.stable_identity_eq(right),
        _ => false,
    };
    ensure!(
        bindings_match,
        "sweep child stable execution bindings differ"
    );
    ensure!(
        reference.sampler == candidate.sampler,
        "sweep child sampler settings or seed differ"
    );
    readout_map(reference)?;
    readout_map(candidate)?;
    Ok(())
}

fn compare_sweep_runs(
    reference: &RunDocument,
    candidate: &RunDocument,
    remaining_details: &mut usize,
) -> Result<SweepReferenceComparison> {
    ensure_sweep_run_context(reference, candidate)?;
    let left_readouts = readout_map(reference)?;
    let right_readouts = readout_map(candidate)?;
    let mut matched_readout_count = 0usize;
    let mut changed_readout_count = 0usize;
    let mut candidate_difference_count = 0usize;
    let mut changed_readouts = Vec::new();
    let mut unmatched_readout_count = 0usize;
    let mut unmatched_readouts = Vec::new();
    for key in left_readouts
        .keys()
        .chain(right_readouts.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        match (left_readouts.get(&key), right_readouts.get(&key)) {
            (Some(left), Some(right)) => {
                matched_readout_count += 1;
                let retain_readout = *remaining_details > 0;
                let candidate_limit = remaining_details.saturating_sub(1);
                let comparison = compare_readout(key, left, right, candidate_limit);
                candidate_difference_count += comparison.candidate_difference_count;
                if comparison.candidate_difference_count > 0 {
                    changed_readout_count += 1;
                    if retain_readout {
                        *remaining_details -= 1;
                        *remaining_details -= comparison.candidate_differences.len();
                        changed_readouts.push(comparison);
                    }
                }
            }
            (Some(_), None) => {
                unmatched_readout_count += 1;
                if *remaining_details > 0 {
                    *remaining_details -= 1;
                    unmatched_readouts.push(UnmatchedReadout {
                        key,
                        side: "reference_only",
                    });
                }
            }
            (None, Some(_)) => {
                unmatched_readout_count += 1;
                if *remaining_details > 0 {
                    *remaining_details -= 1;
                    unmatched_readouts.push(UnmatchedReadout {
                        key,
                        side: "arm_only",
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }
    Ok(SweepReferenceComparison {
        plans_equal: reference.plan == candidate.plan,
        first_generated_token_divergence: first_divergence(
            &reference.generated_token_ids,
            &candidate.generated_token_ids,
        ),
        stop_reason_changed: reference.stop_reason != candidate.stop_reason,
        reference_operation_application_count: reference.operation_applications.len(),
        arm_operation_application_count: candidate.operation_applications.len(),
        matched_readout_count,
        changed_readout_count,
        candidate_difference_count,
        unmatched_readout_count,
        changed_readouts,
        unmatched_readouts,
    })
}

fn canonical_real_directory(path: &Path, label: &str) -> Result<PathBuf> {
    require_real_directory(path, label)?;
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve {label} {}", path.display()))?;
    require_real_directory(&canonical, label)?;
    Ok(canonical)
}

fn require_real_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "{label} {} must be a real non-symlink directory",
        path.display()
    );
    Ok(())
}

fn ensure_directory_entries(path: &Path, expected: BTreeSet<OsString>, label: &str) -> Result<()> {
    let mut actual = BTreeSet::new();
    for entry in
        std::fs::read_dir(path).with_context(|| format!("read {label} {}", path.display()))?
    {
        let name = entry
            .with_context(|| format!("read entry in {label} {}", path.display()))?
            .file_name();
        actual.insert(name);
        ensure!(
            actual.len() <= expected.len(),
            "{label} {} contains more than {} entries",
            path.display(),
            expected.len()
        );
    }
    ensure!(
        actual == expected,
        "{label} {} entries differ: expected {:?}, found {:?}",
        path.display(),
        expected,
        actual
    );
    Ok(())
}

fn duplicate_coefficient_groups(arms: &[LoadedSweepArm]) -> Vec<DuplicateCoefficientGroup> {
    let mut groups: Vec<(u32, Vec<&LoadedSweepArm>)> = Vec::new();
    for arm in arms {
        let bits = arm.coefficient.to_bits();
        if let Some((_, members)) = groups.iter_mut().find(|(candidate, _)| *candidate == bits) {
            members.push(arm);
        } else {
            groups.push((bits, vec![arm]));
        }
    }
    groups
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .map(|(bits, members)| DuplicateCoefficientGroup {
            coefficient: f32::from_bits(bits),
            coefficient_bits: format!("0x{bits:08x}"),
            arm_indices: members.iter().map(|arm| arm.index).collect(),
            byte_identical: members
                .windows(2)
                .all(|pair| pair[0].blake3 == pair[1].blake3),
        })
        .collect()
}

fn sweep_generation_groups(arms: &[LoadedSweepArm]) -> (Vec<SweepGenerationGroup>, Vec<usize>) {
    let mut groups: Vec<(SweepGenerationKey, Vec<usize>)> = Vec::new();
    let mut by_arm = vec![0; arms.len()];
    for arm in arms {
        let key = (
            arm.document.generated_token_ids.clone(),
            arm.document.decoded_text.clone(),
            arm.document.stop_reason.clone(),
        );
        let group_id =
            if let Some(index) = groups.iter().position(|(candidate, _)| *candidate == key) {
                groups[index].1.push(arm.index);
                index
            } else {
                groups.push((key, vec![arm.index]));
                groups.len() - 1
            };
        by_arm[arm.index] = group_id;
    }
    let groups = groups
        .into_iter()
        .enumerate()
        .map(
            |(id, ((generated_token_ids, decoded_text, stop_reason), arm_indices))| {
                SweepGenerationGroup {
                    id,
                    arm_indices,
                    generated_token_ids,
                    decoded_text,
                    stop_reason,
                }
            },
        )
        .collect();
    (groups, by_arm)
}

fn coefficient_bits(coefficient: f32) -> String {
    format!("0x{:08x}", coefficient.to_bits())
}

fn print_sweep_inspection(result: &SweepInspection) {
    println!(
        "coefficient sweep: bundle_integrity={} run_validation={} arms={} operation={} no_inference=true",
        result.bundle_integrity,
        result.run_validation,
        result.arms.len(),
        serde_json::to_string(&result.operation_id).expect("string serialization cannot fail")
    );
    println!(
        "runtime={} model={:?} reference_arm={} coefficient={} source_plan_identity={}",
        result.runtime_kind,
        result.model_path,
        result.reference_arm,
        result.reference_coefficient,
        result.source_plan_identity
    );
    for arm in &result.arms {
        let comparison = arm.reference_comparison.as_ref();
        let divergence = comparison
            .and_then(|comparison| comparison.first_generated_token_divergence.as_ref())
            .map_or_else(|| "none".into(), |divergence| divergence.index.to_string());
        println!(
            "arm={} coefficient={} bits={} generation_group={} generated_text={} stop={} selected_applications={} total_applications={} live_readouts={} divergence={} changed_readouts={} candidate_differences={} unmatched_readouts={}",
            arm.index,
            arm.coefficient,
            arm.coefficient_bits,
            arm.generation_group,
            serde_json::to_string(&arm.generated_text).expect("string serialization cannot fail"),
            serde_json::to_string(&arm.stop_reason).expect("string serialization cannot fail"),
            arm.selected_operation_application_count,
            arm.total_operation_application_count,
            arm.live_readout_count,
            divergence,
            comparison.map_or(0, |comparison| comparison.changed_readout_count),
            comparison.map_or(0, |comparison| comparison.candidate_difference_count),
            comparison.map_or(0, |comparison| comparison.unmatched_readout_count),
        );
    }
    for group in &result.duplicate_coefficient_groups {
        println!(
            "duplicate coefficient={} bits={} arms={:?} byte_identical={}",
            group.coefficient, group.coefficient_bits, group.arm_indices, group.byte_identical
        );
    }
    for group in &result.generation_groups {
        println!(
            "generation_group={} arms={:?} token_ids={:?} text={} stop={}",
            group.id,
            group.arm_indices,
            group.generated_token_ids,
            serde_json::to_string(&group.decoded_text).expect("string serialization cannot fail"),
            serde_json::to_string(&group.stop_reason).expect("string serialization cannot fail")
        );
    }
    println!(
        "manifest_blake3={} total_child_bytes={} retained_details={}/{} sweep={:?}",
        result.manifest_blake3,
        result.total_child_bytes,
        result.retained_detail_count,
        result.detail_limit,
        result.sweep_root
    );
}

#[derive(Debug, Serialize)]
struct TraceComparison {
    alignment: &'static str,
    schema_version: u32,
    left_lens: LensIdentity,
    right_lens: LensIdentity,
    score_semantics: TraceScoreIdentity,
    cell_count: usize,
    top1_changed_cell_count: usize,
    first_changed: Option<CellCoordinate>,
    changed_cell_count: usize,
    changed_cells: Vec<TraceCellDifference>,
    aggregate_difference_count: usize,
    aggregate_differences: Vec<AggregateDifference>,
    vectors: VectorComparison,
    detail_limit: usize,
}

#[derive(Debug, Serialize)]
struct LensIdentity {
    method: String,
    artifact_kind: String,
    source_repository: Option<String>,
    source_revision: Option<String>,
    source_filename: Option<String>,
    payload_blake3: Option<String>,
}

#[derive(Debug, Serialize, PartialEq)]
struct TraceScoreIdentity {
    kind: Option<String>,
    normalization: Option<String>,
    candidate_universe: Option<String>,
    softmax_applied: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct CellCoordinate {
    source_layer: u32,
    source_position: usize,
}

#[derive(Debug, Serialize)]
struct TraceCellDifference {
    coordinate: CellCoordinate,
    top1_changed: bool,
    candidate_difference_count: usize,
    candidates: Vec<TopKCandidateDifference>,
}

#[derive(Debug, Serialize)]
struct TopKCandidateDifference {
    token_id: u32,
    display: String,
    status: CandidateStatus,
    left_rank: Option<usize>,
    right_rank: Option<usize>,
    left_logit: Option<f32>,
    right_logit: Option<f32>,
    logit_delta: Option<f32>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum CandidateStatus {
    PresentBoth,
    EnteredCapturedTopK,
    ExitedCapturedTopK,
}

impl CandidateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::PresentBoth => "present_both",
            Self::EnteredCapturedTopK => "entered_captured_top_k",
            Self::ExitedCapturedTopK => "exited_captured_top_k",
        }
    }
}

#[derive(Debug, Serialize)]
struct AggregateDifference {
    token_id: u32,
    display: String,
    left_count: usize,
    right_count: usize,
    count_delta: i64,
    left_top1_count: usize,
    right_top1_count: usize,
    top1_count_delta: i64,
    left_best_rank: Option<usize>,
    right_best_rank: Option<usize>,
    best_rank_delta: Option<i64>,
}

#[derive(Debug, Default, Serialize)]
struct VectorComparison {
    metadata_compatible: bool,
    metadata_mismatch: Option<String>,
    metadata_incompatible_cell_count: usize,
    metadata_incompatible_cells: Vec<CellCoordinate>,
    matched_cell_count: usize,
    matched_cells: Vec<VectorMetrics>,
    unmatched_cell_count: usize,
    unmatched_cells: Vec<UnmatchedVectorCell>,
    incompatible_dimension_count: usize,
    incompatible_dimensions: Vec<IncompatibleVectorCell>,
}

#[derive(Debug, Serialize)]
struct VectorMetrics {
    coordinate: CellCoordinate,
    dimension: usize,
    cosine: Option<f64>,
    l2_distance: f64,
    left_norm: f64,
    right_norm: f64,
}

#[derive(Debug, Serialize)]
struct UnmatchedVectorCell {
    coordinate: CellCoordinate,
    side: &'static str,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct IncompatibleVectorCell {
    coordinate: CellCoordinate,
    left_dimension: usize,
    right_dimension: usize,
}

fn compare_traces(
    left: &TraceDocument,
    right: &TraceDocument,
    limit: usize,
) -> Result<TraceComparison> {
    ensure!(
        left.schema_version == right.schema_version,
        "trace schema versions differ"
    );
    ensure!(
        left.input_token_ids == right.input_token_ids,
        "trace input token IDs differ"
    );
    ensure!(
        left.selected_layers == right.selected_layers,
        "trace selected layers/order differ"
    );
    ensure!(left.top_k == right.top_k, "trace captured top-k differs");
    let left_coordinates = trace_coordinates(left);
    let right_coordinates = trace_coordinates(right);
    ensure!(
        left_coordinates == right_coordinates,
        "trace cell coordinates/order differ"
    );
    let left_score = trace_score_identity(left);
    let right_score = trace_score_identity(right);
    ensure!(left_score == right_score, "trace score semantics differ");
    if left.schema_version == 3 {
        ensure!(
            left.deployed_model
                .as_ref()
                .and_then(|model| model.locator_id.as_ref())
                == right
                    .deployed_model
                    .as_ref()
                    .and_then(|model| model.locator_id.as_ref()),
            "v3 deployed model locator IDs differ"
        );
        ensure!(
            left.deployed_model
                .as_ref()
                .and_then(|model| model.locator_id.as_ref())
                .is_some(),
            "v3 comparison requires a deployed model locator ID"
        );
        ensure!(
            left.tokenizer
                .as_ref()
                .and_then(|tokenizer| tokenizer.metadata_id.as_ref())
                == right
                    .tokenizer
                    .as_ref()
                    .and_then(|tokenizer| tokenizer.metadata_id.as_ref()),
            "v3 tokenizer metadata IDs differ"
        );
        ensure!(
            left.tokenizer
                .as_ref()
                .and_then(|tokenizer| tokenizer.metadata_id.as_ref())
                .is_some(),
            "v3 comparison requires a tokenizer metadata ID"
        );
    }

    let mut changed_cells = Vec::new();
    let mut top1_changed_cell_count = 0;
    let mut first_changed = None;
    for (left_cell, right_cell) in left.cells.iter().zip(&right.cells) {
        let top1_changed = left_cell.top_k.first().map(|score| score.token_id)
            != right_cell.top_k.first().map(|score| score.token_id);
        top1_changed_cell_count += usize::from(top1_changed);
        let candidates = compare_top_k(left_cell, right_cell);
        if !candidates.is_empty() {
            let coordinate = CellCoordinate {
                source_layer: left_cell.source_layer,
                source_position: left_cell.source_position,
            };
            first_changed.get_or_insert(coordinate);
            changed_cells.push(TraceCellDifference {
                coordinate,
                top1_changed,
                candidate_difference_count: candidates.len(),
                candidates: candidates.into_iter().take(limit).collect(),
            });
        }
    }
    let aggregate_differences = compare_aggregates(left, right);
    let vectors = compare_vectors(left, right, limit);
    Ok(TraceComparison {
        alignment: "exact layer/position and exact token ID; no inferred alignment",
        schema_version: left.schema_version,
        left_lens: lens_identity(left),
        right_lens: lens_identity(right),
        score_semantics: left_score,
        cell_count: left.cells.len(),
        top1_changed_cell_count,
        first_changed,
        changed_cell_count: changed_cells.len(),
        changed_cells: changed_cells.into_iter().take(limit).collect(),
        aggregate_difference_count: aggregate_differences.len(),
        aggregate_differences: aggregate_differences.into_iter().take(limit).collect(),
        vectors,
        detail_limit: limit,
    })
}

fn trace_coordinates(document: &TraceDocument) -> Vec<CellCoordinate> {
    document
        .cells
        .iter()
        .map(|cell| CellCoordinate {
            source_layer: cell.source_layer,
            source_position: cell.source_position,
        })
        .collect()
}

fn trace_score_identity(document: &TraceDocument) -> TraceScoreIdentity {
    document.score_semantics.as_ref().map_or(
        TraceScoreIdentity {
            kind: None,
            normalization: None,
            candidate_universe: None,
            softmax_applied: None,
        },
        |score| TraceScoreIdentity {
            kind: Some(score.kind.clone()),
            normalization: score.normalization.clone(),
            candidate_universe: score.candidate_universe.clone(),
            softmax_applied: Some(score.softmax_applied),
        },
    )
}

fn lens_identity(document: &TraceDocument) -> LensIdentity {
    LensIdentity {
        method: document.lens.method.clone(),
        artifact_kind: document.lens.kind.clone(),
        source_repository: document.lens.source_repository.clone(),
        source_revision: document.lens.source_revision.clone(),
        source_filename: document.lens.source_filename.clone(),
        payload_blake3: document.lens.payload_blake3.clone(),
    }
}

fn compare_top_k(left: &Cell, right: &Cell) -> Vec<TopKCandidateDifference> {
    let left_map: BTreeMap<_, _> = left
        .top_k
        .iter()
        .map(|score| (score.token_id, score))
        .collect();
    let right_map: BTreeMap<_, _> = right
        .top_k
        .iter()
        .map(|score| (score.token_id, score))
        .collect();
    left_map
        .keys()
        .chain(right_map.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|token_id| {
            let left_score = left_map.get(&token_id).copied();
            let right_score = right_map.get(&token_id).copied();
            if let (Some(a), Some(b)) = (left_score, right_score)
                && a.rank == b.rank
                && a.logit == b.logit
            {
                return None;
            }
            let status = match (left_score, right_score) {
                (Some(_), Some(_)) => CandidateStatus::PresentBoth,
                (None, Some(_)) => CandidateStatus::EnteredCapturedTopK,
                (Some(_), None) => CandidateStatus::ExitedCapturedTopK,
                (None, None) => unreachable!(),
            };
            Some(TopKCandidateDifference {
                token_id,
                display: right_score
                    .or(left_score)
                    .unwrap()
                    .token_display_lossy
                    .clone(),
                status,
                left_rank: left_score.map(|score| score.rank),
                right_rank: right_score.map(|score| score.rank),
                left_logit: left_score.map(|score| score.logit),
                right_logit: right_score.map(|score| score.logit),
                logit_delta: left_score.zip(right_score).map(|(a, b)| b.logit - a.logit),
            })
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct AggregateValue {
    count: usize,
    top1: usize,
    best_rank: Option<usize>,
}

fn aggregate(document: &TraceDocument) -> BTreeMap<u32, AggregateValue> {
    let mut output = BTreeMap::new();
    for cell in &document.cells {
        for score in &cell.top_k {
            let row = output
                .entry(score.token_id)
                .or_insert(AggregateValue::default());
            row.count += 1;
            row.top1 += usize::from(score.rank == 0);
            row.best_rank = Some(
                row.best_rank
                    .map_or(score.rank, |rank| rank.min(score.rank)),
            );
        }
    }
    output
}

fn compare_aggregates(left: &TraceDocument, right: &TraceDocument) -> Vec<AggregateDifference> {
    let displays = trace_displays(left, right);
    let left_values = aggregate(left);
    let right_values = aggregate(right);
    let mut rows: Vec<_> = left_values
        .keys()
        .chain(right_values.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|token_id| {
            let a = left_values.get(&token_id).copied().unwrap_or_default();
            let b = right_values.get(&token_id).copied().unwrap_or_default();
            if a.count == b.count && a.top1 == b.top1 && a.best_rank == b.best_rank {
                return None;
            }
            Some(AggregateDifference {
                token_id,
                display: displays.get(&token_id).cloned().unwrap_or_default(),
                left_count: a.count,
                right_count: b.count,
                count_delta: b.count as i64 - a.count as i64,
                left_top1_count: a.top1,
                right_top1_count: b.top1,
                top1_count_delta: b.top1 as i64 - a.top1 as i64,
                left_best_rank: a.best_rank,
                right_best_rank: b.best_rank,
                best_rank_delta: a
                    .best_rank
                    .zip(b.best_rank)
                    .map(|(x, y)| y as i64 - x as i64),
            })
        })
        .collect();
    rows.sort_by_key(|row| (Reverse(row.count_delta.unsigned_abs()), row.token_id));
    rows
}

fn trace_displays(left: &TraceDocument, right: &TraceDocument) -> BTreeMap<u32, String> {
    left.cells
        .iter()
        .chain(&right.cells)
        .flat_map(|cell| &cell.top_k)
        .fold(BTreeMap::new(), |mut displays, score| {
            displays
                .entry(score.token_id)
                .or_insert_with(|| score.token_display_lossy.clone());
            displays
        })
}

fn vector_map(document: &TraceDocument) -> BTreeMap<CellCoordinate, &VectorCell> {
    document
        .vectors
        .as_ref()
        .into_iter()
        .flat_map(|vectors| &vectors.cells)
        .map(|cell| {
            (
                CellCoordinate {
                    source_layer: cell.source_layer,
                    source_position: cell.source_position,
                },
                cell,
            )
        })
        .collect()
}

fn compare_vectors(left: &TraceDocument, right: &TraceDocument, limit: usize) -> VectorComparison {
    let metadata_mismatch = vector_metadata_mismatch(left, right);
    let left = vector_map(left);
    let right = vector_map(right);
    let mut output = VectorComparison {
        metadata_compatible: metadata_mismatch.is_none(),
        metadata_mismatch,
        ..VectorComparison::default()
    };
    if !output.metadata_compatible {
        let coordinates = left
            .keys()
            .chain(right.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        output.metadata_incompatible_cell_count = coordinates.len();
        output.metadata_incompatible_cells = coordinates.into_iter().take(limit).collect();
        return output;
    }
    for coordinate in left
        .keys()
        .chain(right.keys())
        .copied()
        .collect::<BTreeSet<_>>()
    {
        match (left.get(&coordinate), right.get(&coordinate)) {
            (Some(a), Some(b)) if a.values.len() == b.values.len() => {
                let (mut dot, mut left_sq, mut right_sq, mut distance_sq) = (0.0, 0.0, 0.0, 0.0);
                for (&x, &y) in a.values.iter().zip(&b.values) {
                    let (x, y) = (f64::from(x), f64::from(y));
                    dot += x * y;
                    left_sq += x * x;
                    right_sq += y * y;
                    distance_sq += (y - x) * (y - x);
                }
                let left_norm = left_sq.sqrt();
                let right_norm = right_sq.sqrt();
                output.matched_cell_count += 1;
                if output.matched_cells.len() < limit {
                    output.matched_cells.push(VectorMetrics {
                        coordinate,
                        dimension: a.values.len(),
                        cosine: (left_norm > 0.0 && right_norm > 0.0)
                            .then_some(dot / (left_norm * right_norm)),
                        l2_distance: distance_sq.sqrt(),
                        left_norm,
                        right_norm,
                    });
                }
            }
            (Some(a), Some(b)) => {
                output.incompatible_dimension_count += 1;
                if output.incompatible_dimensions.len() < limit {
                    output.incompatible_dimensions.push(IncompatibleVectorCell {
                        coordinate,
                        left_dimension: a.values.len(),
                        right_dimension: b.values.len(),
                    });
                }
            }
            (Some(a), None) | (None, Some(a)) => {
                let side = if left.contains_key(&coordinate) {
                    "left_only"
                } else {
                    "right_only"
                };
                output.unmatched_cell_count += 1;
                if output.unmatched_cells.len() < limit {
                    output.unmatched_cells.push(UnmatchedVectorCell {
                        coordinate,
                        side,
                        dimension: a.values.len(),
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }
    output
}

fn vector_metadata_mismatch(left: &TraceDocument, right: &TraceDocument) -> Option<String> {
    let left_vectors = left.vectors.as_ref();
    let right_vectors = right.vectors.as_ref();
    match (left_vectors, right_vectors) {
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => Some("only one trace contains vector metadata".into()),
        (Some(a), Some(b)) => {
            let left_metadata = (
                left.lens.target_layer,
                a.operation.as_deref(),
                a.stage.as_deref(),
                a.value_dtype.as_deref(),
                a.hidden_coordinate.as_deref(),
                a.hidden_size,
            );
            let right_metadata = (
                right.lens.target_layer,
                b.operation.as_deref(),
                b.stage.as_deref(),
                b.value_dtype.as_deref(),
                b.hidden_coordinate.as_deref(),
                b.hidden_size,
            );
            (left_metadata != right_metadata).then(|| {
                format!(
                    "vector coordinate metadata differs: left={left_metadata:?} right={right_metadata:?}"
                )
            })
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RunSampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunDocument {
    schema: String,
    schema_version: u32,
    runtime_kind: String,
    model_path: PathBuf,
    canonical_plan_path: PathBuf,
    plan: serde_json::Value,
    input_source: String,
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    sampler: RunSampler,
    max_new_tokens: usize,
    decoded_text: String,
    stop_reason: String,
    operation_applications: Vec<RunOperationApplication>,
    requested_live_readouts: Vec<serde_json::Value>,
    live_readouts: Vec<RunReadout>,
    #[serde(default)]
    native_hyper_captures: Vec<serde_json::Value>,
    #[serde(default)]
    execution_binding: Option<RunExecutionBinding>,
}

fn parse_run_bytes(bytes: &[u8], path: &Path) -> Result<RunDocument> {
    let document: RunDocument = serde_json::from_slice(bytes)
        .with_context(|| format!("parse run JSON {}", path.display()))?;
    document
        .validate()
        .with_context(|| format!("validate run JSON {}", path.display()))?;
    Ok(document)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RunExecutionBinding {
    deployed_model_content_blake3: String,
    content_identity_outcome: String,
    weight_bytes_hashed: u64,
    published_lenses: Vec<RunPublishedLensBinding>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RunPublishedLensBinding {
    lens_id: String,
    manifest: PathBuf,
    manifest_canonical_json_blake3: String,
    profile: String,
    method: String,
    target_layer: u32,
    fitted_checkpoint: String,
    fitted_checkpoint_revision: String,
    source_repository: String,
    source_revision: String,
    source_sha256: String,
    payload_blake3: String,
    claims_basis: String,
    transfer_validation_status: String,
    selected_token_ids: Vec<u32>,
    selected_matrices: Vec<RunPublishedMatrixBinding>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RunPublishedMatrixBinding {
    source_layer: u32,
    blake3: String,
}

impl RunDocument {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "qwen.lens.run" && matches!(self.schema_version, 1 | 2),
            "unsupported run schema/version"
        );
        match (self.schema_version, &self.execution_binding) {
            (1, None) => {}
            (2, Some(binding)) => binding.validate()?,
            (1, Some(_)) => bail!("run schema version 1 must not contain execution_binding"),
            (2, None) => bail!("run schema version 2 requires execution_binding"),
            _ => unreachable!(),
        }
        ensure!(
            self.sampler.temperature.is_finite()
                && self.sampler.temperature >= 0.0
                && self.sampler.top_p.is_finite()
                && self.sampler.top_p > 0.0
                && self.sampler.top_p <= 1.0
                && self.sampler.min_p.is_finite()
                && (0.0..=1.0).contains(&self.sampler.min_p),
            "run sampler settings are outside native bounds"
        );
        ensure!(
            matches!(
                self.input_source.as_str(),
                "prompt" | "token_ids" | "messages"
            ) && !self.prompt_token_ids.is_empty()
                && self.prompt_token_ids.len() <= lens_run::MAX_NEW_TOKENS * 16
                && self.max_new_tokens > 0
                && self.max_new_tokens <= lens_run::MAX_NEW_TOKENS,
            "run input source, prompt, or generation bound is invalid"
        );
        ensure!(
            !self.model_path.as_os_str().is_empty()
                && self.canonical_plan_path.is_absolute()
                && self.prompt_token_ids.iter().all(|token| *token >= 0)
                && self.generated_token_ids.iter().all(|token| *token >= 0)
                && self.generated_token_ids.len() <= self.max_new_tokens,
            "run model/plan path or token IDs are invalid"
        );
        match self.stop_reason.as_str() {
            "max_new_tokens" => ensure!(
                self.generated_token_ids.len() == self.max_new_tokens,
                "max_new_tokens run did not generate its declared bound"
            ),
            "stop_token" => ensure!(
                !self.generated_token_ids.is_empty(),
                "stop_token run generated no stop token"
            ),
            _ => bail!("run has unsupported stop reason {:?}", self.stop_reason),
        }
        for application in &self.operation_applications {
            ensure!(
                !application.id.is_empty()
                    && matches!(application.phase.as_str(), "prefill" | "decode"),
                "run operation application has an invalid ID or phase"
            );
        }
        for readout in &self.live_readouts {
            ensure!(
                !readout.id.is_empty()
                    && !readout.lens.is_empty()
                    && !readout.method.is_empty()
                    && !readout.score_kind.is_empty()
                    && !readout.candidate_universe.is_empty()
                    && matches!(readout.phase.as_str(), "prefill" | "decode"),
                "run readout is missing score semantics"
            );
            let mut identities = BTreeSet::new();
            for score in &readout.scores {
                ensure!(
                    score.score.is_finite() && score.token_id.is_none_or(|token| token >= 0),
                    "run readout contains a non-finite score or negative token ID"
                );
                ensure!(
                    identities.insert(score.identity()),
                    "run readout repeats a candidate identity"
                );
            }
        }
        Ok(())
    }
}

impl RunExecutionBinding {
    fn validate(&self) -> Result<()> {
        ensure!(
            is_lower_hex_digest(&self.deployed_model_content_blake3)
                && !self.content_identity_outcome.is_empty()
                && self.weight_bytes_hashed == 0
                && !self.published_lenses.is_empty(),
            "run execution binding has invalid model identity or no published lenses"
        );
        let mut lens_ids = BTreeSet::new();
        for lens in &self.published_lenses {
            ensure!(
                !lens.lens_id.is_empty()
                    && lens_ids.insert(&lens.lens_id)
                    && is_lower_hex_digest(&lens.manifest_canonical_json_blake3)
                    && !lens.profile.is_empty()
                    && matches!(lens.method.as_str(), "J" | "R")
                    && !lens.fitted_checkpoint.is_empty()
                    && !lens.fitted_checkpoint_revision.is_empty()
                    && !lens.source_repository.is_empty()
                    && !lens.source_revision.is_empty()
                    && is_lower_hex_digest(&lens.source_sha256)
                    && is_lower_hex_digest(&lens.payload_blake3)
                    && !lens.claims_basis.is_empty()
                    && !lens.transfer_validation_status.is_empty()
                    && !lens.selected_token_ids.is_empty()
                    && !lens.selected_matrices.is_empty(),
                "run published lens binding is incomplete"
            );
            let mut layers = BTreeSet::new();
            let unique_tokens = lens
                .selected_token_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            ensure!(
                unique_tokens.len() == lens.selected_token_ids.len()
                    && lens.selected_matrices.iter().all(|matrix| {
                        layers.insert(matrix.source_layer) && is_lower_hex_digest(&matrix.blake3)
                    }),
                "run published lens binding has duplicate tokens/layers or invalid matrix digests"
            );
        }
        Ok(())
    }

    fn stable_identity_eq(&self, other: &Self) -> bool {
        self.deployed_model_content_blake3 == other.deployed_model_content_blake3
            && self.weight_bytes_hashed == other.weight_bytes_hashed
            && self.published_lenses == other.published_lenses
    }
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunOperationApplication {
    id: String,
    layer: u32,
    phase: String,
    index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunReadout {
    id: String,
    lens: String,
    method: String,
    score_kind: String,
    candidate_universe: String,
    source_layer: u32,
    target_layer: Option<u32>,
    phase: String,
    index: usize,
    scores: Vec<RunScore>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunScore {
    token_id: Option<i32>,
    row_id: usize,
    word_id: Option<i64>,
    label: Option<String>,
    score: f32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct ReadoutKey {
    id: String,
    lens: String,
    method: String,
    score_kind: String,
    candidate_universe: String,
    source_layer: u32,
    target_layer: Option<u32>,
    phase: String,
    index: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct ScoreIdentity {
    token_id: Option<i32>,
    row_id: usize,
    word_id: Option<i64>,
    label: Option<String>,
}

impl RunReadout {
    fn key(&self) -> ReadoutKey {
        ReadoutKey {
            id: self.id.clone(),
            lens: self.lens.clone(),
            method: self.method.clone(),
            score_kind: self.score_kind.clone(),
            candidate_universe: self.candidate_universe.clone(),
            source_layer: self.source_layer,
            target_layer: self.target_layer,
            phase: self.phase.clone(),
            index: self.index,
        }
    }
}
impl RunScore {
    fn identity(&self) -> ScoreIdentity {
        ScoreIdentity {
            token_id: self.token_id,
            row_id: self.row_id,
            word_id: self.word_id,
            label: self.label.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct RunComparison {
    alignment: &'static str,
    runtime_kind: String,
    model_path: PathBuf,
    plans_equal: bool,
    left_generated_text: String,
    right_generated_text: String,
    left_generated_token_ids: Vec<i32>,
    right_generated_token_ids: Vec<i32>,
    first_generated_token_divergence: Option<GeneratedDivergence>,
    left_stop_reason: String,
    right_stop_reason: String,
    operation_applications: OperationComparison,
    native_capture_counts: SideCounts,
    matched_readout_count: usize,
    matched_readouts: Vec<MatchedReadout>,
    unmatched_readout_count: usize,
    unmatched_readouts: Vec<UnmatchedReadout>,
    detail_limit: usize,
}

#[derive(Debug, Serialize)]
struct GeneratedDivergence {
    index: usize,
    left_token_id: Option<i32>,
    right_token_id: Option<i32>,
    length_only: bool,
}

#[derive(Debug, Serialize)]
struct SideCounts {
    left: usize,
    right: usize,
}

#[derive(Debug, Serialize)]
struct OperationComparison {
    left_total_count: usize,
    right_total_count: usize,
    left_applications: Vec<RunOperationApplication>,
    right_applications: Vec<RunOperationApplication>,
}

#[derive(Debug, Serialize)]
struct MatchedReadout {
    key: ReadoutKey,
    candidate_difference_count: usize,
    candidate_differences: Vec<RunCandidateDifference>,
}

#[derive(Debug, Serialize)]
struct RunCandidateDifference {
    identity: ScoreIdentity,
    status: SelectedCandidateStatus,
    left_score: Option<f32>,
    right_score: Option<f32>,
    score_delta: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SelectedCandidateStatus {
    PresentBoth,
    EnteredReadoutTopK,
    ExitedReadoutTopK,
}

impl SelectedCandidateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::PresentBoth => "present_both",
            Self::EnteredReadoutTopK => "entered_readout_top_k",
            Self::ExitedReadoutTopK => "exited_readout_top_k",
        }
    }
}

#[derive(Debug, Serialize)]
struct UnmatchedReadout {
    key: ReadoutKey,
    side: &'static str,
}

fn compare_runs(left: &RunDocument, right: &RunDocument, limit: usize) -> Result<RunComparison> {
    ensure!(
        left.prompt_token_ids == right.prompt_token_ids,
        "run prompt token IDs differ"
    );
    ensure!(
        left.runtime_kind == right.runtime_kind,
        "run runtime kinds differ"
    );
    ensure!(
        left.model_path == right.model_path,
        "run model paths differ"
    );
    let bindings_match = match (&left.execution_binding, &right.execution_binding) {
        (None, None) => true,
        (Some(left), Some(right)) => left.stable_identity_eq(right),
        _ => false,
    };
    ensure!(bindings_match, "run stable execution bindings differ");
    ensure!(
        left.sampler == right.sampler,
        "run sampler settings or seed differ"
    );
    let divergence = first_divergence(&left.generated_token_ids, &right.generated_token_ids);
    let left_readouts = readout_map(left)?;
    let right_readouts = readout_map(right)?;
    let mut matched_readouts = Vec::new();
    let mut unmatched_readouts = Vec::new();
    for key in left_readouts
        .keys()
        .chain(right_readouts.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        match (left_readouts.get(&key), right_readouts.get(&key)) {
            (Some(a), Some(b)) => matched_readouts.push(compare_readout(key, a, b, limit)),
            (Some(_), None) => unmatched_readouts.push(UnmatchedReadout {
                key,
                side: "left_only",
            }),
            (None, Some(_)) => unmatched_readouts.push(UnmatchedReadout {
                key,
                side: "right_only",
            }),
            (None, None) => unreachable!(),
        }
    }
    Ok(RunComparison {
        alignment: "exact readout key and exact candidate identity; no inferred alignment",
        runtime_kind: left.runtime_kind.clone(),
        model_path: left.model_path.clone(),
        plans_equal: left.plan == right.plan,
        left_generated_text: left.decoded_text.clone(),
        right_generated_text: right.decoded_text.clone(),
        left_generated_token_ids: left.generated_token_ids.clone(),
        right_generated_token_ids: right.generated_token_ids.clone(),
        first_generated_token_divergence: divergence,
        left_stop_reason: left.stop_reason.clone(),
        right_stop_reason: right.stop_reason.clone(),
        operation_applications: OperationComparison {
            left_total_count: left.operation_applications.len(),
            right_total_count: right.operation_applications.len(),
            left_applications: left
                .operation_applications
                .iter()
                .take(limit)
                .cloned()
                .collect(),
            right_applications: right
                .operation_applications
                .iter()
                .take(limit)
                .cloned()
                .collect(),
        },
        native_capture_counts: SideCounts {
            left: left.native_hyper_captures.len(),
            right: right.native_hyper_captures.len(),
        },
        matched_readout_count: matched_readouts.len(),
        matched_readouts: matched_readouts.into_iter().take(limit).collect(),
        unmatched_readout_count: unmatched_readouts.len(),
        unmatched_readouts: unmatched_readouts.into_iter().take(limit).collect(),
        detail_limit: limit,
    })
}

fn first_divergence(left: &[i32], right: &[i32]) -> Option<GeneratedDivergence> {
    let index = left
        .iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))?;
    Some(GeneratedDivergence {
        index,
        left_token_id: left.get(index).copied(),
        right_token_id: right.get(index).copied(),
        length_only: index == left.len().min(right.len()),
    })
}

fn readout_map(document: &RunDocument) -> Result<BTreeMap<ReadoutKey, &RunReadout>> {
    let mut map = BTreeMap::new();
    for readout in &document.live_readouts {
        ensure!(
            map.insert(readout.key(), readout).is_none(),
            "run repeats an exact readout key"
        );
    }
    Ok(map)
}

fn compare_readout(
    key: ReadoutKey,
    left: &RunReadout,
    right: &RunReadout,
    limit: usize,
) -> MatchedReadout {
    let left: BTreeMap<_, _> = left
        .scores
        .iter()
        .map(|score| (score.identity(), score.score))
        .collect();
    let right: BTreeMap<_, _> = right
        .scores
        .iter()
        .map(|score| (score.identity(), score.score))
        .collect();
    let differences: Vec<_> = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|identity| {
            let a = left.get(&identity).copied();
            let b = right.get(&identity).copied();
            if a == b {
                return None;
            }
            Some(RunCandidateDifference {
                identity,
                status: match (a, b) {
                    (Some(_), Some(_)) => SelectedCandidateStatus::PresentBoth,
                    (None, Some(_)) => SelectedCandidateStatus::EnteredReadoutTopK,
                    (Some(_), None) => SelectedCandidateStatus::ExitedReadoutTopK,
                    (None, None) => unreachable!(),
                },
                left_score: a,
                right_score: b,
                score_delta: a.zip(b).map(|(x, y)| f64::from(y) - f64::from(x)),
            })
        })
        .collect();
    MatchedReadout {
        key,
        candidate_difference_count: differences.len(),
        candidate_differences: differences.into_iter().take(limit).collect(),
    }
}

fn print_text(result: &ComparisonResult) {
    match result {
        ComparisonResult::Trace(result) => {
            println!(
                "trace comparison: alignment=exact (layer,position) and token_id; no inference"
            );
            println!(
                "left method={} artifact={} file={:?} digest={:?} right method={} artifact={} file={:?} digest={:?}",
                result.left_lens.method,
                result.left_lens.artifact_kind,
                result.left_lens.source_filename,
                result.left_lens.payload_blake3,
                result.right_lens.method,
                result.right_lens.artifact_kind,
                result.right_lens.source_filename,
                result.right_lens.payload_blake3
            );
            println!(
                "score kind={:?} normalization={:?} universe={:?} softmax={:?}",
                result.score_semantics.kind,
                result.score_semantics.normalization,
                result.score_semantics.candidate_universe,
                result.score_semantics.softmax_applied
            );
            println!(
                "cells={} top1_changed={} changed={} first_changed={}",
                result.cell_count,
                result.top1_changed_cell_count,
                result.changed_cell_count,
                result.first_changed.map_or_else(
                    || "none".into(),
                    |c| format!("{}:{}", c.source_layer, c.source_position)
                )
            );
            for cell in &result.changed_cells {
                println!(
                    "cell {}:{} top1_changed={} candidate_differences={}",
                    cell.coordinate.source_layer,
                    cell.coordinate.source_position,
                    cell.top1_changed,
                    cell.candidate_difference_count
                );
                for candidate in &cell.candidates {
                    println!(
                        "  token={} status={} rank={:?}->{:?} logit={:?}->{:?} delta={:?}",
                        candidate.token_id,
                        candidate.status.as_str(),
                        candidate.left_rank,
                        candidate.right_rank,
                        candidate.left_logit,
                        candidate.right_logit,
                        candidate.logit_delta
                    );
                }
            }
            for row in &result.aggregate_differences {
                println!(
                    "aggregate token={} display={:?} count={}->{} delta={} top1={}->{} best_rank={:?}->{:?}",
                    row.token_id,
                    row.display,
                    row.left_count,
                    row.right_count,
                    row.count_delta,
                    row.left_top1_count,
                    row.right_top1_count,
                    row.left_best_rank,
                    row.right_best_rank
                );
            }
            println!(
                "aggregate_differences={} vectors_metadata_compatible={} vectors_matched={} vectors_unmatched={} vectors_dimension_incompatible={} vectors_metadata_incompatible={}",
                result.aggregate_difference_count,
                result.vectors.metadata_compatible,
                result.vectors.matched_cell_count,
                result.vectors.unmatched_cell_count,
                result.vectors.incompatible_dimension_count,
                result.vectors.metadata_incompatible_cell_count,
            );
            if let Some(mismatch) = &result.vectors.metadata_mismatch {
                println!("vector metadata mismatch: {mismatch}");
            }
            for vector in &result.vectors.matched_cells {
                println!(
                    "vector {}:{} dim={} cosine={:?} l2={} norms={}/{}",
                    vector.coordinate.source_layer,
                    vector.coordinate.source_position,
                    vector.dimension,
                    vector.cosine,
                    vector.l2_distance,
                    vector.left_norm,
                    vector.right_norm
                );
            }
            for vector in &result.vectors.unmatched_cells {
                println!(
                    "unmatched vector {}:{} side={} dim={}",
                    vector.coordinate.source_layer,
                    vector.coordinate.source_position,
                    vector.side,
                    vector.dimension
                );
            }
            for vector in &result.vectors.incompatible_dimensions {
                println!(
                    "incompatible vector {}:{} dimensions={}/{} (not compared)",
                    vector.coordinate.source_layer,
                    vector.coordinate.source_position,
                    vector.left_dimension,
                    vector.right_dimension
                );
            }
        }
        ComparisonResult::Run(result) => {
            println!(
                "run comparison: alignment=exact readout key and candidate identity; no inference"
            );
            println!(
                "runtime={} model={}",
                result.runtime_kind,
                result.model_path.display()
            );
            println!(
                "left generated_text={:?} token_ids={:?} stop={} right generated_text={:?} token_ids={:?} stop={}",
                result.left_generated_text,
                result.left_generated_token_ids,
                result.left_stop_reason,
                result.right_generated_text,
                result.right_generated_token_ids,
                result.right_stop_reason
            );
            println!(
                "first_generated_divergence={}",
                result
                    .first_generated_token_divergence
                    .as_ref()
                    .map_or_else(
                        || "none".into(),
                        |d| format!(
                            "index={} left={:?} right={:?} length_only={}",
                            d.index, d.left_token_id, d.right_token_id, d.length_only
                        )
                    )
            );
            println!(
                "operations left={} right={} native_captures left={} right={}",
                result.operation_applications.left_total_count,
                result.operation_applications.right_total_count,
                result.native_capture_counts.left,
                result.native_capture_counts.right
            );
            println!("plans_equal={}", result.plans_equal);
            for application in &result.operation_applications.left_applications {
                println!(
                    "left operation id={} layer={} phase={} index={}",
                    application.id, application.layer, application.phase, application.index
                );
            }
            for application in &result.operation_applications.right_applications {
                println!(
                    "right operation id={} layer={} phase={} index={}",
                    application.id, application.layer, application.phase, application.index
                );
            }
            println!(
                "readouts matched={} unmatched={}",
                result.matched_readout_count, result.unmatched_readout_count
            );
            for readout in &result.matched_readouts {
                println!(
                    "readout id={} lens={} method={} score_kind={} universe={} layer={} target={:?} phase={} index={} candidate_differences={}",
                    readout.key.id,
                    readout.key.lens,
                    readout.key.method,
                    readout.key.score_kind,
                    readout.key.candidate_universe,
                    readout.key.source_layer,
                    readout.key.target_layer,
                    readout.key.phase,
                    readout.key.index,
                    readout.candidate_difference_count
                );
                for candidate in &readout.candidate_differences {
                    println!(
                        "  candidate token={:?} row={} word={:?} label={:?} status={} score={:?}->{:?} delta={:?}",
                        candidate.identity.token_id,
                        candidate.identity.row_id,
                        candidate.identity.word_id,
                        candidate.identity.label,
                        candidate.status.as_str(),
                        candidate.left_score,
                        candidate.right_score,
                        candidate.score_delta
                    );
                }
            }
            for readout in &result.unmatched_readouts {
                println!(
                    "unmatched readout side={} id={} lens={} method={} score_kind={} universe={} layer={} target={:?} phase={} index={}",
                    readout.side,
                    readout.key.id,
                    readout.key.lens,
                    readout.key.method,
                    readout.key.score_kind,
                    readout.key.candidate_universe,
                    readout.key.source_layer,
                    readout.key.target_layer,
                    readout.key.phase,
                    readout.key.index
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Parser)]
    struct InspectSweepArgsParser {
        #[command(flatten)]
        args: InspectSweepArgs,
    }

    fn sweep_plan(coefficient: f32) -> LensPlan {
        serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [{
                "id":"d",
                "lens":"t",
                "row":{"kind":"template_row_id","template_row_id":0},
                "normalization":"as_stored"
            }],
            "operations": [{
                "id":"swept",
                "scope":{"layers":{"kind":"values","values":[1]},"prefill":{"kind":"all"}},
                "action":{"kind":"fixed_add","direction":"d","coefficient":coefficient}
            }],
            "readouts": [{
                "id":"live",
                "lens":"t",
                "scope":{"layers":{"kind":"values","values":[1]},"prefill":{"kind":"all"}},
                "top_k":1
            }]
        }))
        .unwrap()
    }

    fn sweep_run_document(coefficient: f32) -> RunDocument {
        let plan = sweep_plan(coefficient);
        let requested_live_readouts = serde_json::to_value(&plan.readouts)
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
        RunDocument {
            schema: "qwen.lens.run".into(),
            schema_version: 1,
            runtime_kind: "ordinary_qwen".into(),
            model_path: "/model.gguf".into(),
            canonical_plan_path: "/source/plan.json".into(),
            plan: serde_json::to_value(plan).unwrap(),
            input_source: "token_ids".into(),
            prompt_token_ids: vec![1, 2],
            generated_token_ids: vec![3],
            sampler: RunSampler {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7,
            },
            max_new_tokens: 1,
            decoded_text: "answer".into(),
            stop_reason: "max_new_tokens".into(),
            operation_applications: if coefficient == 0.0 {
                Vec::new()
            } else {
                vec![RunOperationApplication {
                    id: "swept".into(),
                    layer: 1,
                    phase: "prefill".into(),
                    index: 0,
                }]
            },
            requested_live_readouts,
            live_readouts: vec![RunReadout {
                id: "live".into(),
                lens: "t".into(),
                method: "workspace_template_cosine".into(),
                score_kind: "cosine_similarity".into(),
                candidate_universe: "workspace_template_rows".into(),
                source_layer: 1,
                target_layer: None,
                phase: "prefill".into(),
                index: 0,
                scores: vec![RunScore {
                    token_id: None,
                    row_id: 0,
                    word_id: Some(1),
                    label: Some("candidate".into()),
                    score: coefficient,
                }],
            }],
            native_hyper_captures: Vec::new(),
            execution_binding: None,
        }
    }

    fn sweep_fixture(coefficients: &[f32], mutate: impl Fn(usize, &mut RunDocument)) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-inspect-sweep-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let arms_root = root.join("arms");
        std::fs::create_dir(&arms_root).unwrap();
        let mut arms = Vec::new();
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let arm_root = arms_root.join(format!("{index:06}"));
            std::fs::create_dir(&arm_root).unwrap();
            let mut document = sweep_run_document(coefficient);
            mutate(index, &mut document);
            let bytes = serde_json::to_vec(&document).unwrap();
            std::fs::write(arm_root.join("run.json"), &bytes).unwrap();
            arms.push(lens_run::CoefficientSweepArm {
                index,
                coefficient,
                artifact: format!("arms/{index:06}/run.json"),
                byte_length: bytes.len() as u64,
                blake3: blake3::hash(&bytes).to_hex().to_string(),
            });
        }
        let manifest = CoefficientSweepManifest {
            schema: "qwen.lens.coefficient_sweep".into(),
            schema_version: 1,
            producer: lens_run::SweepProducer {
                build_commit: "a".repeat(40),
                build_dirty: "0".into(),
                build_source_state: format!("git-source-sha256-v2:{}", "b".repeat(64)),
            },
            canonical_source_plan_path: "/source/plan.json".into(),
            operation_id: "swept".into(),
            coefficients: coefficients.to_vec(),
            arms,
        };
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        root
    }

    fn rewrite_sweep_manifest(root: &Path, mutate: impl FnOnce(&mut CoefficientSweepManifest)) {
        let path = root.join("manifest.json");
        let mut manifest: CoefficientSweepManifest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        mutate(&mut manifest);
        std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    fn rewrite_sweep_child(root: &Path, index: usize, mutate: impl FnOnce(&mut serde_json::Value)) {
        let path = root.join(format!("arms/{index:06}/run.json"));
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        mutate(&mut value);
        let bytes = serde_json::to_vec(&value).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        rewrite_sweep_manifest(root, |manifest| {
            manifest.arms[index].byte_length = bytes.len() as u64;
            manifest.arms[index].blake3 = blake3::hash(&bytes).to_hex().to_string();
        });
    }

    #[test]
    fn inspect_sweep_cli_accepts_reference_limit_and_json() {
        let args = InspectSweepArgsParser::try_parse_from([
            "test",
            "sweep",
            "--reference-arm",
            "2",
            "--limit",
            "7",
            "--format",
            "json",
        ])
        .unwrap()
        .args;
        assert_eq!(args.sweep, PathBuf::from("sweep"));
        assert_eq!(args.reference_arm, Some(2));
        assert_eq!(args.limit, 7);
        assert!(matches!(args.format, InspectSweepFormat::Json));
    }

    #[test]
    fn inspect_sweep_verifies_controls_groups_and_exact_deltas() {
        let root = sweep_fixture(&[0.0, 0.1, 0.0], |_, _| {});
        let loaded = load_sweep(&root).unwrap();
        assert_eq!(loaded.arms.len(), 3);
        assert_eq!(loaded.arms[0].blake3, loaded.arms[2].blake3);
        let duplicate = duplicate_coefficient_groups(&loaded.arms);
        assert_eq!(duplicate.len(), 1);
        assert_eq!(duplicate[0].arm_indices, vec![0, 2]);
        assert!(duplicate[0].byte_identical);
        let (generations, by_arm) = sweep_generation_groups(&loaded.arms);
        assert_eq!(generations.len(), 1);
        assert_eq!(by_arm, vec![0, 0, 0]);
        let mut detail_budget = 10;
        let comparison = compare_sweep_runs(
            &loaded.arms[0].document,
            &loaded.arms[1].document,
            &mut detail_budget,
        )
        .unwrap();
        assert_eq!(comparison.changed_readout_count, 1);
        assert_eq!(comparison.candidate_difference_count, 1);
        assert_eq!(comparison.arm_operation_application_count, 1);
        assert_eq!(comparison.changed_readouts.len(), 1);
        assert_eq!(
            comparison.changed_readouts[0].candidate_differences.len(),
            1
        );
        assert_eq!(detail_budget, 8);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_sweep_detail_budget_is_global_across_arms() {
        let root = sweep_fixture(&[0.0, 0.1, 0.2], |_, _| {});
        let loaded = load_sweep(&root).unwrap();
        let mut detail_budget = 1;
        let first = compare_sweep_runs(
            &loaded.arms[0].document,
            &loaded.arms[1].document,
            &mut detail_budget,
        )
        .unwrap();
        assert_eq!(first.changed_readouts.len(), 1);
        assert!(first.changed_readouts[0].candidate_differences.is_empty());
        assert_eq!(detail_budget, 0);
        let second = compare_sweep_runs(
            &loaded.arms[0].document,
            &loaded.arms[2].document,
            &mut detail_budget,
        )
        .unwrap();
        assert_eq!(second.changed_readout_count, 1);
        assert!(second.changed_readouts.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_sweep_preserves_distinct_signed_zero_bits() {
        let root = sweep_fixture(&[0.0, -0.0], |_, _| {});
        let loaded = load_sweep(&root).unwrap();
        assert_eq!(loaded.arms[0].coefficient.to_bits(), 0.0_f32.to_bits());
        assert_eq!(loaded.arms[1].coefficient.to_bits(), (-0.0_f32).to_bits());
        assert!(duplicate_coefficient_groups(&loaded.arms).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_child_digest_drift() {
        let root = sweep_fixture(&[0.0], |_, _| {});
        let path = root.join("arms/000000/run.json");
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(path, bytes).unwrap();
        assert!(load_sweep(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_missing_children_and_declared_length_drift() {
        let missing_root = sweep_fixture(&[0.0], |_, _| {});
        std::fs::remove_file(missing_root.join("arms/000000/run.json")).unwrap();
        assert!(load_sweep(&missing_root).is_err());
        std::fs::remove_dir_all(missing_root).unwrap();

        let length_root = sweep_fixture(&[0.0], |_, _| {});
        rewrite_sweep_manifest(&length_root, |manifest| {
            manifest.arms[0].byte_length += 1;
        });
        assert!(load_sweep(&length_root).is_err());
        std::fs::remove_dir_all(length_root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_symlinked_children_and_extra_entries() {
        let symlink_root = sweep_fixture(&[0.0], |_, _| {});
        let run = symlink_root.join("arms/000000/run.json");
        let target = symlink_root.with_extension("target.json");
        std::fs::rename(&run, &target).unwrap();
        symlink(&target, &run).unwrap();
        assert!(load_sweep(&symlink_root).is_err());
        std::fs::remove_dir_all(symlink_root).unwrap();
        std::fs::remove_file(target).unwrap();

        let extra_root = sweep_fixture(&[0.0], |_, _| {});
        std::fs::write(extra_root.join("unexpected"), b"x").unwrap();
        assert!(load_sweep(&extra_root).is_err());
        std::fs::remove_dir_all(extra_root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_wrong_coefficient_and_unrelated_plan_mutation() {
        let coefficient_root = sweep_fixture(&[0.0, 0.1], |index, document| {
            if index == 1 {
                document.plan["operations"][0]["action"]["coefficient"] = json!(0.2);
            }
        });
        assert!(load_sweep(&coefficient_root).is_err());
        std::fs::remove_dir_all(coefficient_root).unwrap();

        let plan_root = sweep_fixture(&[0.0, 0.1], |index, document| {
            if index == 1 {
                document.plan["readouts"][0]["top_k"] = json!(2);
                document.requested_live_readouts =
                    document.plan["readouts"].as_array().unwrap().clone();
            }
        });
        assert!(load_sweep(&plan_root).is_err());
        std::fs::remove_dir_all(plan_root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_cross_arm_context_drift() {
        let root = sweep_fixture(&[0.0, 0.1], |index, document| {
            if index == 1 {
                document.sampler.seed = 8;
            }
        });
        assert!(load_sweep(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();

        let semantics_root = sweep_fixture(&[0.0, 0.1], |index, document| {
            if index == 1 {
                document.live_readouts[0].score_kind = "different_kind".into();
            }
        });
        assert!(load_sweep(&semantics_root).is_err());
        std::fs::remove_dir_all(semantics_root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_unknown_fields_and_malformed_run_semantics() {
        let unknown_root = sweep_fixture(&[0.0], |_, _| {});
        rewrite_sweep_child(&unknown_root, 0, |value| {
            value["unknown"] = json!(true);
        });
        assert!(load_sweep(&unknown_root).is_err());
        std::fs::remove_dir_all(unknown_root).unwrap();

        let token_root = sweep_fixture(&[0.0], |_, document| {
            document.prompt_token_ids[0] = -1;
        });
        assert!(load_sweep(&token_root).is_err());
        std::fs::remove_dir_all(token_root).unwrap();

        let stop_root = sweep_fixture(&[0.0], |_, document| {
            document.stop_reason = "unknown".into();
        });
        assert!(load_sweep(&stop_root).is_err());
        std::fs::remove_dir_all(stop_root).unwrap();

        let length_root = sweep_fixture(&[0.0], |_, document| {
            document.generated_token_ids.push(4);
        });
        assert!(load_sweep(&length_root).is_err());
        std::fs::remove_dir_all(length_root).unwrap();

        let sampler_root = sweep_fixture(&[0.0], |_, document| {
            document.sampler.top_p = 0.0;
        });
        assert!(load_sweep(&sampler_root).is_err());
        std::fs::remove_dir_all(sampler_root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_out_of_scope_applications_and_readouts() {
        let application_root = sweep_fixture(&[0.1], |_, document| {
            document.operation_applications[0].index = 99;
        });
        assert!(load_sweep(&application_root).is_err());
        std::fs::remove_dir_all(application_root).unwrap();

        let readout_root = sweep_fixture(&[0.0], |_, document| {
            document.live_readouts[0].phase = "decode".into();
        });
        assert!(load_sweep(&readout_root).is_err());
        std::fs::remove_dir_all(readout_root).unwrap();

        let unknown_root = sweep_fixture(&[0.0], |_, document| {
            document.live_readouts[0].id = "unknown".into();
        });
        assert!(load_sweep(&unknown_root).is_err());
        std::fs::remove_dir_all(unknown_root).unwrap();

        let duplicate_root = sweep_fixture(&[0.0], |_, document| {
            let mut duplicate = document.live_readouts[0].clone();
            duplicate.method = "contradictory_method".into();
            document.live_readouts.push(duplicate);
        });
        assert!(load_sweep(&duplicate_root).is_err());
        std::fs::remove_dir_all(duplicate_root).unwrap();
    }

    #[test]
    fn inspect_sweep_rejects_applications_from_disabled_zero_operation() {
        let root = sweep_fixture(&[0.0], |_, document| {
            document
                .operation_applications
                .push(RunOperationApplication {
                    id: "swept".into(),
                    layer: 1,
                    phase: "prefill".into(),
                    index: 0,
                });
        });
        assert!(load_sweep(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn trace(top_k: serde_json::Value, vectors: Option<serde_json::Value>) -> TraceDocument {
        let scores = top_k.as_array().unwrap();
        let occurrences: Vec<_> = scores
            .iter()
            .map(|score| {
                json!({
                    "token_id": score["token_id"],
                    "count": 1,
                    "top1_count": usize::from(score["rank"] == 0),
                    "best_rank": score["rank"]
                })
            })
            .collect();
        let value = json!({
            "schema": "qwen.lens.trace",
            "schema_version": 3,
            "producer": {},
            "deployed_model": {"locator_id": "model", "vocab_size": 100},
            "tokenizer": {"metadata_id": "tokenizer"},
            "lens": {"kind": "published_full_transport", "method": "J", "payload_blake3": "artifact"},
            "score_semantics": {"kind": "logit", "normalization": "rmsnorm", "candidate_universe": "full_model_vocabulary", "softmax_applied": false},
            "execution_mode": "passive",
            "input_token_ids": [10],
            "input_tokens": [{"position": 0, "token_id": 10, "token_display_lossy": "a"}],
            "rendering": {"renderer": "test", "spans": []},
            "selected_layers": [2],
            "top_k": 2,
            "cells": [{"source_layer": 2, "source_position": 0, "source_token_id": 10, "predicts_position": 1, "top_k": top_k}],
            "vectors": vectors,
            "timing": {},
            "occurrences": {"global": occurrences, "per_layer": [{"source_layer": 2, "tokens": occurrences}]}
        });
        lens_inspect::parse_trace_bytes(
            &serde_json::to_vec(&value).unwrap(),
            std::path::Path::new("synthetic-trace.json"),
        )
        .unwrap()
    }

    fn scores(first: u32, second: u32, first_logit: f32) -> serde_json::Value {
        json!([
            {"rank": 0, "token_id": first, "token_display_lossy": first.to_string(), "logit": first_logit},
            {"rank": 1, "token_id": second, "token_display_lossy": second.to_string(), "logit": 1.0}
        ])
    }

    fn vectors(values: &[f32]) -> serde_json::Value {
        json!({
            "operation": "test", "stage": "test", "value_dtype": "f32",
            "hidden_size": values.len(), "shape": [1, values.len()],
            "cells": [{"source_layer": 2, "source_position": 0, "source_token_id": 10, "predicts_position": 1, "values": values}]
        })
    }

    #[test]
    fn rejects_incompatible_trace_coordinates() {
        let left = trace(scores(7, 8, 2.0), None);
        let mut right = trace(scores(7, 8, 2.0), None);
        right.selected_layers = vec![3];
        assert!(compare_traces(&left, &right, 10).is_err());
    }

    #[test]
    fn reports_top_k_entry_and_exit_without_invented_values() {
        let comparison = compare_traces(
            &trace(scores(7, 8, 2.0), None),
            &trace(scores(7, 9, 2.0), None),
            10,
        )
        .unwrap();
        let candidates = &comparison.changed_cells[0].candidates;
        let exited = candidates.iter().find(|row| row.token_id == 8).unwrap();
        let entered = candidates.iter().find(|row| row.token_id == 9).unwrap();
        assert!(matches!(exited.status, CandidateStatus::ExitedCapturedTopK));
        assert!(
            exited.right_rank.is_none()
                && exited.right_logit.is_none()
                && exited.logit_delta.is_none()
        );
        assert!(matches!(
            entered.status,
            CandidateStatus::EnteredCapturedTopK
        ));
        assert!(
            entered.left_rank.is_none()
                && entered.left_logit.is_none()
                && entered.logit_delta.is_none()
        );
    }

    #[test]
    fn computes_matching_vector_metrics() {
        let comparison = compare_traces(
            &trace(scores(7, 8, 2.0), Some(vectors(&[1.0, 0.0]))),
            &trace(scores(7, 8, 2.0), Some(vectors(&[0.0, 1.0]))),
            10,
        )
        .unwrap();
        let metrics = &comparison.vectors.matched_cells[0];
        assert_eq!(metrics.cosine, Some(0.0));
        assert!((metrics.l2_distance - 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!((metrics.left_norm, metrics.right_norm), (1.0, 1.0));
    }

    fn run_document(generated: Vec<i32>, scores: Vec<RunScore>) -> RunDocument {
        let max_new_tokens = generated.len().max(1);
        RunDocument {
            schema: "qwen.lens.run".into(),
            schema_version: 1,
            runtime_kind: "ordinary_qwen".into(),
            model_path: "model.gguf".into(),
            canonical_plan_path: "/plan.json".into(),
            plan: serde_json::from_value(json!({"version": 1, "lenses": [], "directions": [], "operations": [], "readouts": []})).unwrap(),
            input_source: "token_ids".into(),
            prompt_token_ids: vec![1, 2],
            generated_token_ids: generated,
            sampler: RunSampler {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7,
            },
            max_new_tokens,
            decoded_text: "text".into(),
            stop_reason: "max_new_tokens".into(),
            operation_applications: Vec::new(),
            requested_live_readouts: Vec::new(),
            live_readouts: vec![RunReadout {
                id: "readout".into(),
                lens: "lens".into(),
                method: "J".into(),
                score_kind: "selected_row_score".into(),
                candidate_universe: "selected_rows".into(),
                source_layer: 2,
                target_layer: Some(3),
                phase: "decode".into(),
                index: 0,
                scores,
            }],
            native_hyper_captures: Vec::new(),
            execution_binding: None,
        }
    }

    fn run_score(token_id: i32, score: f32) -> RunScore {
        RunScore {
            token_id: Some(token_id),
            row_id: token_id as usize,
            word_id: None,
            label: None,
            score,
        }
    }

    fn run_execution_binding() -> RunExecutionBinding {
        RunExecutionBinding {
            deployed_model_content_blake3: "11".repeat(32),
            content_identity_outcome: "Hit".into(),
            weight_bytes_hashed: 0,
            published_lenses: vec![RunPublishedLensBinding {
                lens_id: "published-j".into(),
                manifest: "/lens/lens.json".into(),
                manifest_canonical_json_blake3: "22".repeat(32),
                profile: "eyes-profile".into(),
                method: "J".into(),
                target_layer: 51,
                fitted_checkpoint: "eyes/model".into(),
                fitted_checkpoint_revision: "revision".into(),
                source_repository: "eyes/lens".into(),
                source_revision: "source-revision".into(),
                source_sha256: "33".repeat(32),
                payload_blake3: "44".repeat(32),
                claims_basis: "pinned_repository_model_card".into(),
                transfer_validation_status: "unvalidated".into(),
                selected_token_ids: vec![7],
                selected_matrices: vec![RunPublishedMatrixBinding {
                    source_layer: 25,
                    blake3: "55".repeat(32),
                }],
            }],
        }
    }

    #[test]
    fn version_two_runs_require_and_exactly_match_execution_bindings() {
        let mut left = run_document(vec![3], vec![]);
        left.schema_version = 2;
        left.execution_binding = Some(run_execution_binding());
        let mut right = run_document(vec![3], vec![]);
        right.schema_version = 2;
        right.execution_binding = Some(run_execution_binding());
        right
            .execution_binding
            .as_mut()
            .unwrap()
            .content_identity_outcome = "DeclaredAndStored".into();
        left.validate().unwrap();
        right.validate().unwrap();
        compare_runs(&left, &right, 10).unwrap();

        right.execution_binding.as_mut().unwrap().published_lenses[0].selected_matrices[0].blake3 =
            "66".repeat(32);
        assert!(compare_runs(&left, &right, 10).is_err());
    }

    #[test]
    fn reports_generated_length_only_divergence() {
        let comparison = compare_runs(
            &run_document(vec![3], vec![]),
            &run_document(vec![3, 4], vec![]),
            10,
        )
        .unwrap();
        let divergence = comparison.first_generated_token_divergence.unwrap();
        assert_eq!(divergence.index, 1);
        assert!(divergence.length_only);
        assert_eq!(
            (divergence.left_token_id, divergence.right_token_id),
            (None, Some(4))
        );
    }

    #[test]
    fn aligns_readout_scores_only_by_exact_candidate_identity() {
        let comparison = compare_runs(
            &run_document(vec![3], vec![run_score(7, 1.0), run_score(8, 2.0)]),
            &run_document(vec![3], vec![run_score(7, 1.5), run_score(9, 3.0)]),
            10,
        )
        .unwrap();
        let rows = &comparison.matched_readouts[0].candidate_differences;
        let matched = rows
            .iter()
            .find(|row| row.identity.token_id == Some(7))
            .unwrap();
        assert_eq!(matched.score_delta, Some(0.5));
        let exited = rows
            .iter()
            .find(|row| row.identity.token_id == Some(8))
            .unwrap();
        assert!(matches!(
            exited.status,
            SelectedCandidateStatus::ExitedReadoutTopK
        ));
        assert!(exited.right_score.is_none() && exited.score_delta.is_none());
        let entered = rows
            .iter()
            .find(|row| row.identity.token_id == Some(9))
            .unwrap();
        assert!(matches!(
            entered.status,
            SelectedCandidateStatus::EnteredReadoutTopK
        ));
        assert!(entered.left_score.is_none() && entered.score_delta.is_none());
    }
}

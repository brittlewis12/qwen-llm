//! Test-only comparison against the immutable original full-Forward archive.
//! Parent declares `#[cfg(test)] pub(super) mod qualification;` in ced.rs.
//! Hooks are called only AFTER checked GPU completion and kernel status checks.
//! No hooks, overrides, leases or dispatch changes are installed in normal code.
//! Required: MTL_DEBUG_LAYER=1, QWEN_METAL_LEASE_WAIT=1, V41_CED_GGUF,
//! V41_CED_TOKENS, V41_CED_FROZEN_REFERENCE (one completed original arm), and
//! V41_CED_QUALIFICATION_REPORT (new absolute private JSON outside checkout).
use super::*;
use crate::deepseek_v41::gguf as v41_gguf;
use anyhow::{Context, Result as AResult, ensure};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::rc::Rc;

mod archive;
mod current_pair;
mod metrics;
mod provenance;

use archive::{Geometry, Input, Reference, STEPS};
use metrics::Difference;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    FullProduction,
    FullScalar,
    ShrinkingScalar,
    ShrinkingProduction,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::FullProduction => "current_full_production_matches_frozen_reference",
            Self::FullScalar => "current_full_scalar_matches_frozen_reference",
            Self::ShrinkingScalar => "shrinking_scalar_matches_frozen_reference",
            Self::ShrinkingProduction => "shrinking_production_against_frozen_reference",
        }
    }
    fn scalar(self) -> bool {
        matches!(self, Self::FullScalar | Self::ShrinkingScalar)
    }
    fn shrinking(self) -> bool {
        matches!(self, Self::ShrinkingScalar | Self::ShrinkingProduction)
    }
    fn descriptive(self) -> bool {
        self == Self::ShrinkingProduction
    }
    fn reference_arm(self) -> &'static str {
        if self.scalar() {
            "native-token-major-f32"
        } else {
            "production-full-prefill-2048"
        }
    }
}

#[test]
#[ignore = "release, full stock IQ3 1017, frozen reference, private report and explicit Metal lease required"]
fn current_full_production_matches_frozen_reference() {
    run(Arm::FullProduction).unwrap();
}

#[test]
#[ignore = "release, original token-major F32 reference, private report and explicit Metal lease required"]
fn current_full_scalar_matches_frozen_reference() {
    run(Arm::FullScalar).unwrap();
}

#[test]
#[ignore = "release, strict ScalarNative CED + 256 teacher-forced steps; explicit Metal lease required"]
fn shrinking_scalar_matches_frozen_reference() {
    run(Arm::ShrinkingScalar).unwrap();
}

#[test]
#[ignore = "release, production CED characterization, frozen reference and explicit Metal lease required"]
fn shrinking_production_against_frozen_reference() {
    run(Arm::ShrinkingProduction).unwrap();
}

struct Comparator {
    reference: Rc<Reference>,
    arm: Arm,
    boundary_next: usize,
    controls_next: Vec<usize>,
    published: bool,
    decoder_next: usize,
    differences: BTreeMap<String, Difference>,
}

impl Comparator {
    fn new(reference: Rc<Reference>, arm: Arm) -> Self {
        let g = &reference.geometry;
        Self {
            arm,
            boundary_next: g.boundary_start(),
            controls_next: (g.cut..g.layers).map(|l| g.output_start(l)).collect(),
            published: false,
            decoder_next: g.cut,
            differences: BTreeMap::new(),
            reference,
        }
    }

    fn compare(
        &mut self,
        role: &str,
        layer: Option<usize>,
        coordinate: usize,
        bytes: &[u8],
        float: bool,
        strict: bool,
        stage: &str,
    ) -> AResult<()> {
        let r = self.reference.get(role, layer, coordinate)?;
        let context = json!({"stage": stage, "role": role, "layer": layer, "absolute_row_or_state_position": coordinate,
            "reference_record": r.record, "reference_producer": r.producer});
        self.differences
            .entry(format!(
                "{stage}/{role}/{}",
                layer.map_or_else(|| "head".into(), |l| l.to_string())
            ))
            .or_default()
            .add(self.reference.bytes(r), bytes, float, &context, strict)
    }

    fn boundary(
        &mut self,
        start: usize,
        rows: usize,
        residual: &MetalTensor,
        mix: &MetalTensor,
    ) -> AResult<()> {
        let g = self.reference.geometry.clone();
        let end = start.checked_add(rows).context("boundary overflow")?;
        ensure!(
            rows > 0 && rows <= MAX_PREFILL_CHUNK_ROWS && end <= g.n,
            "boundary chunk range"
        );
        let e = f32_slice(residual).map_err(anyhow::Error::msg)?;
        let p = f32_slice(mix).map_err(anyhow::Error::msg)?;
        ensure!(
            e.len() == rows * 4 * g.hidden && p.len() == rows * 4,
            "boundary tensor shape"
        );
        for row in start.max(g.boundary_start())..end {
            ensure!(
                row == self.boundary_next,
                "missing/duplicate boundary row {row}, expected {}",
                self.boundary_next
            );
            let i = row - start;
            self.compare(
                "E",
                Some(g.cut - 1),
                row,
                bytemuck::cast_slice(&e[i * 4 * g.hidden..(i + 1) * 4 * g.hidden]),
                true,
                true,
                "boundary",
            )?;
            self.compare(
                "P",
                Some(g.cut - 1),
                row,
                bytemuck::cast_slice(&p[i * 4..(i + 1) * 4]),
                true,
                true,
                "boundary",
            )?;
            self.boundary_next += 1;
        }
        Ok(())
    }

    fn controls(
        &mut self,
        layer: usize,
        start: usize,
        rows: usize,
        view: &attention_gpu::QueryControlView,
    ) -> AResult<()> {
        let g = self.reference.geometry.clone();
        let end = start.checked_add(rows).context("control range overflow")?;
        ensure!(
            self.published
                && layer == self.decoder_next
                && (g.cut..g.layers).contains(&layer)
                && rows > 0
                && rows <= PREFILL_CHUNK_ROWS
                && start >= g.output_start(layer)
                && end <= g.n
                && view.absolute_rows() == (start..end),
            "control absolute range/layer"
        );
        let selected = checked_i32(view.selected())?;
        let candidates = checked_i32(view.candidates())?;
        let cw = (g.n + STEPS).div_ceil(g.block);
        ensure!(
            selected.len() == rows * g.topk && candidates.len() == rows * cw,
            "control tensor shape"
        );
        for row in start..end {
            ensure!(
                self.controls_next[layer - g.cut] == row,
                "missing/duplicate control row layer {layer} absolute {row}"
            );
            let i = row - start;
            let visible_selected = &selected[i * g.topk..i * g.topk + (row + 1).min(g.topk)];
            let visible_candidates = &candidates[i * cw..i * cw + (row + 1).div_ceil(g.block)];
            ensure!(
                visible_selected
                    .iter()
                    .all(|&v| v >= 0 && v as usize <= row)
                    && visible_selected.windows(2).all(|v| v[0] < v[1])
                    && visible_candidates.iter().all(|&v| v == 0 || v == 1),
                "invalid causal control values"
            );
            self.compare(
                "selected",
                Some(layer),
                row,
                bytemuck::cast_slice(visible_selected),
                false,
                !self.arm.descriptive(),
                "controls",
            )?;
            self.compare(
                "candidates",
                Some(layer),
                row,
                bytemuck::cast_slice(visible_candidates),
                false,
                !self.arm.descriptive(),
                "controls",
            )?;
            self.controls_next[layer - g.cut] += 1;
        }
        Ok(())
    }

    fn publication(&mut self, attention: &attention_gpu::Workspace, n: usize) -> AResult<()> {
        let g = self.reference.geometry.clone();
        ensure!(
            n == g.n && !self.published && self.boundary_next == n,
            "publication position/coverage/order"
        );
        let regions = attention
            .ced_diagnostic_regions(n)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            regions.len() == 1 + 4 * g.owners.len(),
            "publication inventory"
        );
        let raw = causal_slice(regions[0])?;
        let width = g.window * g.head_dim;
        ensure!(raw.len() == g.layers * width, "physical ring extent");
        for layer in 0..g.cut {
            self.compare(
                "local_ring_physical",
                Some(layer),
                n,
                bytemuck::cast_slice(&raw[layer * width..(layer + 1) * width]),
                true,
                true,
                "publication",
            )?;
        }
        self.globals(&regions, n, "publication")?;
        self.published = true;
        Ok(())
    }

    fn globals(&mut self, regions: &[(&MetalTensor, usize)], n: usize, stage: &str) -> AResult<()> {
        let owners = self.reference.geometry.owners.clone();
        ensure!(
            regions.len() == 1 + 4 * owners.len(),
            "owner region inventory"
        );
        for (slot, (owner, _)) in owners.into_iter().enumerate() {
            for (j, role) in [
                "owner_completed_kv",
                "owner_completed_index_keys",
                "owner_pending_value",
                "owner_pending_gate",
            ]
            .into_iter()
            .enumerate()
            {
                self.compare(
                    role,
                    Some(owner),
                    n,
                    bytemuck::cast_slice(causal_slice(regions[1 + 4 * slot + j])?),
                    true,
                    true,
                    stage,
                )?;
            }
        }
        Ok(())
    }

    fn decoder_layer(
        &mut self,
        attention: &attention_gpu::Workspace,
        layer: usize,
        n: usize,
    ) -> AResult<()> {
        let g = self.reference.geometry.clone();
        ensure!(
            self.published
                && n == g.n
                && layer == self.decoder_next
                && layer < g.layers
                && self.controls_next[layer - g.cut] == n,
            "decoder layer completion/order/controls"
        );
        let regions = attention
            .ced_diagnostic_regions(n)
            .map_err(anyhow::Error::msg)?;
        let raw = causal_slice(regions[0])?;
        let width = g.window * g.head_dim;
        ensure!(raw.len() == g.layers * width, "decoder raw extent");
        let r = self.reference.get("local_ring_physical", Some(layer), n)?;
        // Original and CED use the same physical modulo slots. Only slots for
        // absolute rows max(0,N-window)..N are causal when N < window.
        let valid = n.min(g.window) * g.head_dim;
        let context = json!({"stage": "decoder_layer", "layer": layer, "state_position": n,
            "absolute_valid_rows": [n.saturating_sub(g.window), n], "physical_next_slot": n % g.window});
        self.differences
            .entry(format!("decoder_layer/local_ring_physical/{layer}"))
            .or_default()
            .add(
                &self.reference.bytes(r)[..valid * 4],
                bytemuck::cast_slice(&raw[layer * width..layer * width + valid]),
                true,
                &context,
                !self.arm.descriptive(),
            )?;
        self.globals(&regions, n, "decoder_globals_unchanged")?;
        self.decoder_next += 1;
        Ok(())
    }

    fn finish_hooks(&self) -> AResult<()> {
        let g = &self.reference.geometry;
        ensure!(
            self.boundary_next == g.n
                && self.controls_next.iter().all(|&p| p == g.n)
                && self.published
                && self.decoder_next == g.layers,
            "missing qualification hooks/coverage"
        );
        Ok(())
    }
    fn difference_report(&self) -> Value {
        Value::Object(
            self.differences
                .iter()
                .map(|(k, d)| (k.clone(), d.json()))
                .collect(),
        )
    }

    fn hook_report(&self) -> Value {
        let g = &self.reference.geometry;
        if !self.arm.shrinking() {
            return Value::Null;
        }
        json!({"boundary_absolute_range": [g.boundary_start(), self.boundary_next],
            "publication_completed": self.published, "decoder_layers_completed": self.decoder_next - g.cut,
            "controls": (g.cut..g.layers).map(|layer| json!({"layer": layer,
                "absolute_query_range": [g.output_start(layer), self.controls_next[layer - g.cut]],
                "selected_producer": g.producer(layer), "candidate_producer": g.cut,
                "provenance": "validated stock schedule; frozen records independently require these producer IDs; runtime completion validates retained producer coverage"})).collect::<Vec<_>>()})
    }
}

type Shared = Rc<RefCell<Comparator>>;
thread_local! { static ACTIVE: RefCell<Option<Shared>> = const { RefCell::new(None) }; }

struct Scope(std::marker::PhantomData<Rc<()>>);
impl Scope {
    fn enter(comparator: Shared) -> AResult<Self> {
        ACTIVE.with(|slot| {
            ensure!(slot.borrow().is_none(), "nested CED qualification scope");
            *slot.borrow_mut() = Some(comparator);
            Ok(Self(std::marker::PhantomData))
        })
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
    }
}

fn hook(f: impl FnOnce(&mut Comparator) -> AResult<()>) -> Result<()> {
    let active = ACTIVE.with(|slot| slot.borrow().clone());
    match active {
        Some(c) => f(&mut c.borrow_mut()).map_err(|e| format!("CED qualification: {e:#}")),
        None => Ok(()),
    }
}

pub(in super::super) fn boundary(
    start: usize,
    rows: usize,
    residual: &MetalTensor,
    mix: &MetalTensor,
) -> Result<()> {
    hook(|c| c.boundary(start, rows, residual, mix))
}
pub(in super::super) fn controls(
    layer: usize,
    start: usize,
    rows: usize,
    view: &attention_gpu::QueryControlView,
) -> Result<()> {
    hook(|c| c.controls(layer, start, rows, view))
}
pub(in super::super) fn publication(attention: &attention_gpu::Workspace, n: usize) -> Result<()> {
    hook(|c| c.publication(attention, n))
}
pub(in super::super) fn decoder_layer(
    attention: &attention_gpu::Workspace,
    layer: usize,
    n: usize,
) -> Result<()> {
    hook(|c| c.decoder_layer(attention, layer, n))
}

fn checked_i32(t: &MetalTensor) -> AResult<&[i32]> {
    let count = usize::try_from(t.n_elements())?;
    ensure!(
        t.dtype == GgmlType::I32
            && t.offset.is_multiple_of(4)
            && t.buffer.storageMode() == objc2_metal::MTLStorageMode::Shared
            && t.offset
                .checked_add(
                    (count as u64)
                        .checked_mul(4)
                        .context("I32 extent overflow")?
                )
                .is_some_and(|end| end <= t.buffer.length() as u64),
        "invalid shared I32 view"
    );
    Ok(unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((t.offset / 4) as usize),
            count,
        )
    })
}

fn causal_slice((tensor, count): (&MetalTensor, usize)) -> AResult<&[f32]> {
    f32_slice(tensor)
        .map_err(anyhow::Error::msg)?
        .get(..count)
        .context("causal region extent")
}

fn env_path(name: &str) -> AResult<PathBuf> {
    let path = std::env::var_os(name).with_context(|| format!("required environment: {name}"))?;
    ensure!(!path.is_empty(), "empty {name}");
    Ok(path.into())
}

fn add(values: &[u64]) -> AResult<u64> {
    values.iter().try_fold(0u64, |a, &b| {
        a.checked_add(b).context("qualification memory overflow")
    })
}

fn require_reference_chunk_rows(selected: usize, reference: usize) -> AResult<()> {
    ensure!(
        selected == reference,
        "frozen production Full chunk geometry differs from selected runtime; use a fresh current Full/CED pair"
    );
    Ok(())
}

#[test]
fn production_reference_rejects_a_different_full_chunk_schedule() {
    assert!(require_reference_chunk_rows(2048, 2048).is_ok());
    assert!(require_reference_chunk_rows(4096, 4096).is_ok());
    assert!(require_reference_chunk_rows(4096, 2048).is_err());
    assert!(require_reference_chunk_rows(2048, 4096).is_err());
}
fn admit(ctx: &MetalContext, metal_bytes: u64, cpu_bytes: u64) -> AResult<Value> {
    let s = ctx.memory_signals();
    let a =
        metal::evaluate_metal_memory_admission_with_cpu_bytes(metal_bytes, cpu_bytes, 0, s, true);
    if let Some(reason) = a.refusal() {
        return Err(reason.into());
    }
    Ok(
        json!({"metal_bytes": metal_bytes, "cpu_bytes": cpu_bytes, "reason": a.reason.as_str(),
        "recommended_max_bytes": s.recommended_max_bytes, "current_allocated_bytes": s.current_allocated_bytes,
        "process_limit_remaining_bytes": s.process_limit_remaining_bytes}),
    )
}

fn preweight_memory(
    ctx: &MetalContext,
    p: &v41_gguf::Plan,
    g: &Geometry,
    arm: Arm,
    archive_cpu: u64,
    metadata: u64,
    tensor_count: usize,
) -> AResult<Value> {
    let c = &p.config;
    let capacity = g.n + STEPS;
    let full_chunk_rows = prefill_rows_for_resident(p.bytes.resident_bytes());
    let rows = g.n.min(full_chunk_rows);
    let projections = vec![(6144, 5 * c.hidden); p.engram_hash.layers.len()];
    let policies = if arm.shrinking() {
        vec![
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ]
    } else {
        vec![PrefillExecution::FullForward]
    };
    let (request, full, ced) = preflight::runtime_from_parts(
        c,
        &p.schedule,
        &p.engram_hash,
        &projections,
        g.n,
        capacity,
        &policies,
        full_chunk_rows,
        metal::moe_grouped_tensor_selected(ctx),
        &|bytes| {
            if bytes == 0 {
                return Ok(0);
            }
            ctx.price_shared_buffer_upper(bytes)
                .map(|p| p.priced_upper_bytes)
                .map_err(|e| e.to_string())
        },
    )
    .map_err(anyhow::Error::msg)?;
    let half_scratch = prefill::half_scratch_budget(ctx, c, rows).map_err(anyhow::Error::msg)?;
    let scratch = full
        .0
        .max(ced.as_ref().map_or(0, |c| c.incremental_metal_bytes));
    let runtime_cpu = full
        .1
        .max(ced.as_ref().map_or(0, |c| c.incremental_cpu_bytes));
    let extra = 16 * 1024 * 1024;
    let state_floats = g.layers * c.window * c.head_dim
        + g.owners
            .iter()
            .map(|&(_, ratio)| capacity / ratio * (c.head_dim + c.index_dim) + 2 * c.head_dim)
            .sum::<usize>();
    let cpu = add(&[
        request.cpu_bytes,
        archive_cpu,
        metadata
            .checked_mul(8)
            .context("metadata reserve overflow")?,
        (state_floats as u64)
            .checked_mul(8)
            .context("compare reserve overflow")?,
        c.vocab as u64 * 32,
        runtime_cpu,
        64 * 1024 * 1024,
    ])?;
    let gpu = add(&[
        p.bytes.resident_bytes(),
        (tensor_count as u64)
            .checked_mul(32768)
            .context("tensor overhead overflow")?,
        request.metal_bytes,
        scratch,
        extra,
    ])?;
    let admission = admit(ctx, gpu, cpu)?;
    Ok(
        json!({"preweight_admission": admission, "archive_cpu_reserve_bytes": archive_cpu,
        "full_prefill_chunk_rows": full_chunk_rows,
        "cpu_total_reserve_bytes": cpu, "chunk_scratch_priced_bytes": scratch,
        "half_chunk_scratch_priced_bytes": half_scratch, "ced_extra_priced_bytes": extra,
        "strategy": "normal CPU+Metal admission before resident weight loading; entire reference mmap bytes charged plus 24x JSON/index bytes, full causal CPU compare reserve, 8 logit buffers and parse/hash/upload slack; direct read-only region slices, no full-state clones"}),
    )
}

fn admission_ledger(memory: &CedMemoryEstimate) -> CedMemoryEstimate {
    let mut ledger = memory.clone();
    ledger.process_ru_maxrss_before_bytes = None;
    ledger.process_ru_maxrss_after_bytes = None;
    ledger
}

#[test]
fn memory_ledger_comparison_excludes_runtime_peak_observations() {
    let estimated = CedMemoryEstimate {
        incremental_metal_bytes: 4096,
        process_ru_maxrss_before_bytes: Some(100),
        ..Default::default()
    };
    let mut observed = estimated.clone();
    observed.process_ru_maxrss_before_bytes = Some(200);
    observed.process_ru_maxrss_after_bytes = Some(300);
    assert_eq!(admission_ledger(&observed), admission_ledger(&estimated));
    observed.incremental_metal_bytes += 1;
    assert_ne!(admission_ledger(&observed), admission_ledger(&estimated));
}

pub(in crate::deepseek_v41::forward) fn production_command_count(plan: &CedPlan) -> AResult<usize> {
    let scalar_tail = usize::from(plan.scalar_tail().is_some());
    plan.decoder.iter().try_fold(
        plan.prompt_rows.div_ceil(plan.full_chunk_rows),
        |commands, layer| {
            let packed_rows = layer
                .output_rows
                .len()
                .checked_sub(scalar_tail)
                .context("scalar tail outside decoder outputs")?;
            commands
                .checked_add(layer.seed_rows.len().div_ceil(PREFILL_CHUNK_ROWS))
                .and_then(|n| n.checked_add(packed_rows.div_ceil(PREFILL_CHUNK_ROWS)))
                .and_then(|n| n.checked_add(scalar_tail))
                .and_then(|n| {
                    n.checked_sub(usize::from(
                        layer.fuse_seed_with_packed_output(plan.prompt_rows),
                    ))
                })
                .context("qualification command count overflow")
        },
    )
}

#[test]
fn production_seed_fusion_removes_only_eligible_boundaries_at_native_lengths() {
    for (n, old, fused, seeds, seed_rows) in [
        (128, 22, 21, 1, 127),
        (2048, 38, 21, 17, 2047),
        (8192, 47, 27, 20, 2540),
    ] {
        let plan = CedPlan::shrinking(
            &Config::default(),
            &v41_gguf::Schedule::default(),
            &engram::HashConfig::default(),
            &CedRequest::cold(n, n + STEPS),
        )
        .unwrap();
        let incumbent = n.div_ceil(PREFILL_CHUNK_ROWS)
            + plan
                .decoder
                .iter()
                .map(|layer| {
                    layer.seed_rows.len().div_ceil(PREFILL_CHUNK_ROWS)
                        + layer.output_rows.len().div_ceil(PREFILL_CHUNK_ROWS)
                })
                .sum::<usize>();
        assert_eq!(incumbent, old);
        assert_eq!(production_command_count(&plan).unwrap(), fused);
        assert_eq!(
            plan.decoder
                .iter()
                .filter(|layer| layer.fuse_seed_with_packed_output(n))
                .count(),
            seeds
        );
        assert_eq!(plan.work.decoder_seed_rows, seed_rows);
    }
}

#[test]
fn production_command_count_preserves_genuine_scalar_tail_submissions() {
    for (n, expected) in [
        (1, 21),
        (128, 21),
        (2048, 21),
        (4096, 25),
        (4097, 46),
        (8192, 27),
        (8193, 48),
    ] {
        let plan = CedPlan::shrinking(
            &Config::default(),
            &v41_gguf::Schedule::default(),
            &engram::HashConfig::default(),
            &CedRequest::cold(n, n + STEPS),
        )
        .unwrap();
        assert_eq!(production_command_count(&plan).unwrap(), expected, "N={n}");
        if n == 4097 {
            let layer = &plan.decoder[18];
            assert_eq!(layer.output_rows.len(), 128);
            assert_eq!(
                (layer.output_rows.len() - 1).div_ceil(PREFILL_CHUNK_ROWS) + 1,
                2
            );
        }
    }
}

fn execution_report(
    model: &Forward<'_>,
    r: &PrefillReport,
    g: &Geometry,
    arm: Arm,
) -> AResult<Value> {
    let p = CedPlan::shrinking(
        &model.config,
        &model.schedule,
        &model.engram_hash,
        &CedRequest::cold(g.n, g.n + STEPS).with_full_chunk_rows(model.prefill_rows),
    )?;
    ensure!(
        r.execution == PrefillExecution::ShrinkingCed && r.fallback.is_none(),
        "CED did not execute: {r:?}"
    );
    let counters = r.counters.as_ref().context("missing CED counters")?;
    counters.validate(&p).map_err(anyhow::Error::msg)?;
    let commands = if arm.scalar() {
        g.n + p.work.decoder_seed_rows + p.work.decoder_output_rows
    } else {
        production_command_count(&p)?
    };
    ensure!(
        counters.encoded_commands == commands && counters.completed_commands == commands,
        "CED command coverage"
    );
    for l in &counters.layers {
        let rows = if l.layer < g.cut {
            g.n
        } else {
            p.decoder[l.layer - g.cut].output_rows.len()
        };
        ensure!(
            l.selection_producer_rows
                == if model.schedule.produces_index(l.layer) {
                    rows
                } else {
                    0
                },
            "index producer work coverage"
        );
    }
    let memory = r.memory.as_ref().context("missing accepted CED memory")?;
    let estimated = super::estimate(model, g.n + STEPS, &p).map_err(anyhow::Error::msg)?;
    ensure!(
        admission_ledger(memory) == admission_ledger(&estimated)
            && memory.incremental_cpu_bytes > 0
            && memory.incremental_metal_bytes > 0
            && r.observed_incremental_metal_bytes
                .context("missing observed CED allocation")?
                <= memory.incremental_metal_bytes,
        "CED memory estimate/observation mismatch"
    );
    ensure!(
        memory
            .process_ru_maxrss_before_bytes
            .zip(memory.process_ru_maxrss_after_bytes)
            .is_some_and(|(before, after)| after >= before),
        "missing or nonmonotonic process peak observations"
    );
    ensure!(
        p.boundary_rows.start > 0 && p.work.total_output_rows < p.work.full_forward_output_rows,
        "qualification prompt does not actually shrink dependencies"
    );
    Ok(serde_json::to_value(r)?)
}

fn check_state(cmp: &mut Comparator, model: &Forward<'_>, state: &State) -> AResult<Value> {
    let g = cmp.reference.geometry.clone();
    let pos = state.position();
    ensure!(
        state.is_ready() && (g.n..=g.n + STEPS).contains(&pos),
        "state phase/position"
    );
    let reference_state = cmp.reference.states[pos - g.n].clone();
    let address = json!({"position": pos, "compressed_pad": state.addresses.pad(),
        "compressed_history_oldest_first": state.addresses.history().iter().map(|t| match t { engram::Token::Compressed(id) => json!(id), engram::Token::Dead => Value::Null }).collect::<Vec<_>>()});
    ensure!(
        address == reference_state["address"]
            && archive::json_sha(&address)? == reference_state["address_sha256"],
        "address/history mismatch at {pos}"
    );
    let attention = state
        .gpu_step
        .as_ref()
        .context("missing GPU state")?
        .attention();
    attention
        .validate_snapshot_position(pos)
        .map_err(anyhow::Error::msg)?;
    let regions = attention.causal_regions(pos).map_err(anyhow::Error::msg)?;
    ensure!(
        regions.len() == 1 + 4 * g.owners.len() && model.layers.len() == g.layers,
        "state inventory"
    );
    let raw = causal_slice(regions[0])?;
    let width = g.window * g.head_dim;
    ensure!(raw.len() == g.layers * width, "all physical rings");
    let keys = g.state_keys();
    let mut hashes = Vec::with_capacity(keys.len());
    for (i, (role, layer)) in keys.into_iter().enumerate() {
        let values = if i < g.layers {
            &raw[layer * width..(layer + 1) * width]
        } else {
            causal_slice(regions[1 + i - g.layers])?
        };
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite causal state {role}/{layer}/{pos}"
        );
        let bytes = bytemuck::cast_slice(values);
        let hash = archive::sha(bytes);
        let expected = &reference_state["regions"][i];
        let shape_len = expected["shape"]
            .as_array()
            .context("state shape")?
            .iter()
            .try_fold(1usize, |n, d| {
                n.checked_mul(d.as_u64().context("dimension")? as usize)
                    .context("shape overflow")
            })?;
        ensure!(
            shape_len == values.len(),
            "state shape mismatch at {pos}/{role}/{layer}"
        );
        let strict = !cmp.arm.descriptive() || role != "local_ring_physical" || layer < g.cut;
        let equal = expected["sha256"] == hash;
        if pos == g.n || pos == g.n + STEPS {
            cmp.compare(
                &role,
                Some(layer),
                pos,
                bytes,
                true,
                strict,
                if pos == g.n {
                    "initial_state"
                } else {
                    "final_state"
                },
            )?;
        }
        ensure!(
            !strict || equal,
            "strict causal region hash mismatch at position={pos} role={role} layer={layer}"
        );
        hashes.push(json!({"role": role, "layer": layer, "sha256": hash, "reference_sha256": expected["sha256"], "bitwise_hash_equal": equal, "strict": strict}));
    }
    Ok(
        json!({"position": pos, "address": address, "address_sha256": archive::json_sha(&address)?,
        "physical_ring_next_slot": pos % g.window, "pending_ratio2_active": pos % 2 == 1,
        "attention_cursor": {"next_position": pos, "next_layer": 0, "batch": null,
            "validated_by": "validate_snapshot_position"}, "regions": hashes}),
    )
}

fn run(arm: Arm) -> AResult<()> {
    provenance::environment()?;
    let gguf_path = env_path("V41_CED_GGUF")?;
    let input_path = env_path("V41_CED_TOKENS")?;
    let reference_path = env_path("V41_CED_FROZEN_REFERENCE")?;
    let report_path = env_path("V41_CED_QUALIFICATION_REPORT")?;
    let (checkout, source) = provenance::source()?;
    for path in [&input_path, &reference_path] {
        ensure!(
            !path.canonicalize()?.starts_with(&checkout),
            "private input/reference payloads must be outside checkout"
        );
    }
    ensure!(
        !report_path
            .parent()
            .context("report parent")?
            .canonicalize()?
            .starts_with(reference_path.canonicalize()?),
        "qualification report must not modify frozen reference directory"
    );
    let mut report_file = private_report(&report_path, &checkout)?;
    // Unit-test contexts do not acquire the production benchmark lease.
    let _lease = metal::acquire_metal_benchmark_lease()?;
    let ctx = MetalContext::new()?;
    let archive_cpu = archive::cpu_reserve(&reference_path)?;
    admit(&ctx, 0, add(&[archive_cpu, 128 * 1024 * 1024])?)?;
    let mut input_bytes = Vec::new();
    archive::open(&input_path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut input_bytes)?;
    ensure!(
        input_bytes.len() <= 16 * 1024 * 1024,
        "input exceeds 16 MiB"
    );
    let input: Input = serde_json::from_slice(&input_bytes)?;
    let gguf = Arc::new(crate::gguf::GgufFile::open(&gguf_path)?);
    let planned = v41_gguf::plan(&gguf)?;
    provenance::profile(&gguf, &planned)?;
    let g = Geometry {
        n: input.prompt.len(),
        layers: planned.schedule.layers,
        cut: planned
            .schedule
            .candidate_source_layer
            .context("missing cut")?,
        window: planned.config.window,
        hidden: planned.config.hidden,
        head_dim: planned.config.head_dim,
        index_dim: planned.config.index_dim,
        vocab: planned.config.vocab,
        topk: planned.config.index_topk,
        block: planned.config.candidate_block,
        owners: planned
            .schedule
            .kv_source_layers
            .iter()
            .map(|&l| (l, planned.schedule.ratio(l)))
            .collect(),
        producers: planned.schedule.index_source_layers.clone(),
    };
    ensure!(
        g.n > 0
            && g.n
                .checked_add(STEPS)
                .is_some_and(|n| n as u64 <= planned.engram_hash.max_position),
        "input length/context"
    );
    let reference = Rc::new(Reference::load(
        &reference_path,
        g.clone(),
        &input_bytes,
        arm.reference_arm(),
    )?);
    if !arm.scalar() {
        require_reference_chunk_rows(
            prefill_rows_for_resident(planned.bytes.resident_bytes()),
            archive::number(&reference.header, "prefill_chunk_rows")?,
        )?;
    }
    provenance::frozen(&reference.provenance)?;
    let stamps_before = provenance::artifact(&gguf, &planned, &reference.artifact)?;
    let metadata = serde_json::to_vec(gguf.model.metadata())?.len() as u64;
    let memory = preweight_memory(
        &ctx,
        &planned,
        &g,
        arm,
        archive_cpu,
        metadata,
        gguf.tensors.len(),
    )?;
    drop(input_bytes);
    let weights = ResidentWeights::load(&ctx, Arc::clone(&gguf))?;
    let model = Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
        .map_err(anyhow::Error::msg)?;
    let mut state = model.new_state(g.n + STEPS)?;
    let comparator = Rc::new(RefCell::new(Comparator::new(Rc::clone(&reference), arm)));
    let started = std::time::Instant::now();
    let mut execution = Value::Null;
    let mut logits = match arm {
        Arm::FullProduction => model.prefill(&mut state, &input.prompt, None)?,
        Arm::FullScalar => {
            for &t in &input.prompt[..g.n - 1] {
                model
                    .advance(&mut state, t, None)
                    .map_err(anyhow::Error::msg)?;
            }
            model
                .step(&mut state, input.prompt[g.n - 1], None)
                .map_err(anyhow::Error::msg)?
        }
        Arm::ShrinkingScalar | Arm::ShrinkingProduction => {
            let scope = Scope::enter(Rc::clone(&comparator))?;
            let (values, report) = if arm == Arm::ShrinkingScalar {
                super::run(
                    &model,
                    &mut state,
                    &input.prompt,
                    None,
                    Arithmetic::ScalarNative,
                )?
            } else {
                model.prefill_with_execution(
                    &mut state,
                    &input.prompt,
                    None,
                    PrefillExecution::ShrinkingCed,
                )?
            };
            drop(scope);
            comparator.borrow().finish_hooks()?;
            execution = execution_report(&model, &report, &g, arm)?;
            values
        }
    };
    ensure!(
        state.prefill_origin()
            == if arm.shrinking() {
                PrefillExecution::ShrinkingCed
            } else {
                PrefillExecution::FullForward
            },
        "actual prefill origin"
    );
    let prefill_seconds = started.elapsed().as_secs_f64();
    let mut states = Vec::with_capacity(STEPS + 1);
    let mut logit_rows = Vec::with_capacity(STEPS + 1);
    let mut predictions = Vec::with_capacity(STEPS);
    for i in 0..=STEPS {
        let pos = g.n + i;
        ensure!(
            state.position() == pos && logits.len() == g.vocab,
            "logit/state position/shape"
        );
        let mut row_diff = Difference::default();
        let r = reference.get("logits", None, pos)?;
        let expected = reference.floats(r)?;
        row_diff.add(
            reference.bytes(r),
            bytemuck::cast_slice(&logits),
            true,
            &json!({"role": "logits", "position": pos}),
            !arm.descriptive(),
        )?;
        comparator.borrow_mut().compare(
            "logits",
            None,
            pos,
            bytemuck::cast_slice(&logits),
            true,
            !arm.descriptive(),
            "logits",
        )?;
        let prediction = if i < STEPS {
            let p = metrics::prediction(expected, &logits, input.continuation[i])?;
            predictions.push(p.clone());
            p
        } else {
            Value::Null
        };
        logit_rows.push(json!({"position": pos, "difference": row_diff.json(), "sha256": archive::sha(bytemuck::cast_slice(&logits)),
            "reference_sha256": r.sha256, "prediction": prediction, "comparison_only": i == STEPS}));
        states.push(check_state(&mut comparator.borrow_mut(), &model, &state)?);
        if i < STEPS {
            logits = model
                .step(&mut state, input.continuation[i], None)
                .map_err(anyhow::Error::msg)?;
        }
    }
    let total_seconds = started.elapsed().as_secs_f64();
    let stamps_after = provenance::artifact(&gguf, &planned, &reference.artifact)?;
    ensure!(
        stamps_before == stamps_after,
        "GGUF retained stamps changed during qualification"
    );
    reference.revalidate()?;
    let (_, source_after) = provenance::source()?;
    ensure!(
        source == source_after,
        "source/binary provenance changed during qualification"
    );
    let report = json!({"schema": "v41-ced-qualification-v1", "complete": true, "arm": arm.name(),
        "full_prefill_chunk_rows": model.prefill_rows,
        "N": g.n, "capacity": g.n + STEPS, "continuation_steps": STEPS,
        "reference": {"directory": reference_path, "header": reference.header, "complete": reference.complete,
            "provenance": reference.provenance, "validation": "all manifest and region lengths, SHA256, 4-byte alignment, shape, identity, producer, complete row/state coverage; exact raw input SHA and canonical IDs SHA; retained reference file stamps before/after"},
        "candidate_provenance": source, "artifact_validation": {"method": "stamp-based reuse of earlier frozen full-content SHA256; model shard content NOT rehashed in this run",
            "before": stamps_before, "after": stamps_after, "layout_sha256": reference.artifact["layout_sha256"]},
        "memory": memory, "execution": execution, "differences": comparator.borrow().difference_report(),
        "hook_coverage": comparator.borrow().hook_report(), "gguf_supplied_path": gguf_path,
        "device_facts": ctx.device_facts(), "tensor_ops_status": format!("{:?}", ctx.tensor_ops_status()),
        "original_shard_content_sha256": reference.artifact["shards"].as_array().context("original shards")?.iter()
            .map(|s| json!({"shard_idx": s["shard_idx"], "size": s["size"], "sha256": s["sha256"]})).collect::<Vec<_>>(),
        "states": states, "logits": logit_rows, "held_256": metrics::held_summary(&predictions)?,
        "strict_contract": if arm.descriptive() { "E/P, encoder physical rings, every owner KV/key/pending operand, address history bitwise; publication and every decoder layer also assure owner globals not rewritten. Decoder/control/logit differences have no tolerance or quality pass verdict." }
            else { "strict bitwise: all causal regions and 257 logit rows and per-position causal hashes N..N+256; shrinking scalar additionally E/P, all causal controls, publication and every decoder raw ring" },
        "formal_quality_verdict": Value::Null, "diagnostic_timings": {"prefill_seconds": prefill_seconds, "total_seconds": total_seconds,
            "not_performance_evidence": "includes synchronous comparison/readback/hashing hooks; use separate plain 8K benchmark"},
        "limitations": ["no sampled/argmax-fed continuation", "no automatic tolerance relaxation", "original scalar reference is token-major F32 operators, not all-batched NativeF32", "snapshot/restore/growth and failure/cancellation qualification remain separate tests", "content identity reuse is stamp-based, not a fresh model shard content hash"]});
    serde_json::to_writer(&mut report_file, &report)?;
    report_file.write_all(b"\n")?;
    report_file.sync_all()?;
    File::open(report_path.parent().unwrap())?.sync_all()?;
    Ok(())
}

fn private_report(path: &Path, checkout: &Path) -> AResult<File> {
    ensure!(
        path.is_absolute() && path.file_name().is_some(),
        "report must be a new absolute file outside checkout"
    );
    let parent = path.parent().unwrap().canonicalize()?;
    ensure!(
        !parent.starts_with(checkout.canonicalize()?),
        "private payloads must be outside checkout"
    );
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    Ok(file)
}

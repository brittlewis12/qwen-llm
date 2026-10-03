//! Read shared plain/fitted heads from original-forward captures, never replay.

use super::{
    Sink, checkpoint,
    execute::TokenEngine,
    readouts::{MAX_LABEL_BYTES, Plan},
};
use crate::serve::jobs::state::{Counters, Phase};
use anyhow::{Context, Result, ensure};
use objc2_metal::MTLBuffer;
use qwen_llm::{
    metal::MetalTensor,
    runtime::{LoadedModel, Sequence},
    tokenizer::Tokenizer,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{borrow::Cow, time::Instant};
mod paired;
#[cfg(test)]
mod verification;

pub(super) struct Engine<'a> {
    loaded: &'a LoadedModel,
    tokenizer: &'a Tokenizer,
    sequence: Sequence,
    plan: &'a Plan,
    capture: Option<MetalTensor>,
    layers: Vec<u32>,
    before: Option<MetalTensor>,
    before_layers: Vec<u32>,
    staged: &'a super::registry::Staged,
    fitted: Option<qwen_llm::workspace_lens::WorkspaceLensFullReadoutWorkspace<'a>>,
    interventions: &'a super::interventions::Plan,
    directions: super::interventions::PreparedDirections,
}
impl<'a> Engine<'a> {
    pub(super) fn new(
        loaded: &'a LoadedModel,
        tokenizer: &'a Tokenizer,
        mut sequence: Sequence,
        prepared: &'a super::Prepared,
        staged: &'a super::registry::Staged,
        sink: &Sink,
    ) -> Result<Self> {
        let plan = &prepared.readouts;
        let allocate = |layers: usize| -> Result<Option<MetalTensor>> {
            Ok(if layers == 0 {
                None
            } else {
                let count = layers
                    .checked_mul(loaded.arch().hidden_size as usize)
                    .context("capture elements overflow")?;
                Some(MetalTensor::zeros_f32(
                    loaded.context(),
                    vec![u64::try_from(count)?],
                )?)
            })
        };
        let capture = allocate(plan.max_capture_layers)?;
        let before = allocate(plan.pairs.max_layers)?;
        let directions = prepared.interventions.prepare(
            loaded,
            &mut sequence,
            staged,
            sink,
            prepared.prompt.len(),
        )?;
        checkpoint(sink)?;
        let fitted = if plan.matrices.is_empty() {
            None
        } else {
            Some(
                loaded
                    .passive_workspace_lens_session(&mut sequence)?
                    .full_readout_workspace(1)?,
            )
        };
        Ok(Self {
            loaded,
            tokenizer,
            sequence,
            plan,
            capture,
            layers: Vec::new(),
            before,
            before_layers: Vec::new(),
            staged,
            fitted,
            interventions: &prepared.interventions,
            directions,
        })
    }
}
impl TokenEngine for Engine<'_> {
    fn forward(&mut self, token: i32, position: u32, logits: bool) -> Result<Vec<f32>> {
        self.sequence.check_position(position as usize)?;
        self.sequence.ensure_can_append(1)?;
        self.layers.clear();
        if let Some(event) = self.plan.events.get(&position) {
            self.layers.extend(event.keys().copied());
        }
        self.before_layers.clear();
        if let Some(event) = self.plan.pairs.events.get(&position) {
            self.before_layers.extend(event.keys().copied());
        }
        let before = if self.before_layers.is_empty() {
            None
        } else {
            Some(
                self.before
                    .as_ref()
                    .context("missing admitted before capture")?
                    .view_subrange(
                        0,
                        vec![
                            (self.before_layers.len() * self.loaded.arch().hidden_size as usize)
                                as u64,
                        ],
                    ),
            )
        };
        let capture = if self.layers.is_empty() {
            None
        } else {
            let count = self.layers.len() * self.loaded.arch().hidden_size as usize;
            Some(
                self.capture
                    .as_ref()
                    .context("missing admitted capture")?
                    .view_subrange(0, vec![count as u64]),
            )
        };
        let mut operations = Vec::new();
        if let Some(event) = self.interventions.events.get(&position) {
            for (&layer, indices) in event {
                for &index in indices {
                    operations.push(crate::lens_intervention::lower(
                        &self.interventions.operations[index].action,
                        layer,
                        |id| {
                            self.directions
                                .rows
                                .get(&(id.to_owned(), layer))
                                .context("missing prepared direction")
                        },
                        || anyhow::bail!("unsupported native reflection"),
                    )?);
                }
            }
        }
        let output = crate::ordinary_executor::post_block_forward(
            &self.loaded.forward(),
            unsafe { self.sequence.metal_session_mut() },
            token,
            position,
            logits,
            capture
                .as_ref()
                .map(|buffer| (self.layers.as_slice(), buffer)),
            before
                .as_ref()
                .map(|buffer| (self.before_layers.as_slice(), buffer)),
            &operations,
            &[],
        )?;
        self.sequence.advance_by(1)?;
        Ok(output)
    }
    fn observe(
        &mut self,
        token: i32,
        position: u32,
        generation_logits: &[f32],
        counters: &Counters,
        sink: &Sink,
    ) -> Result<()> {
        self.interventions
            .publish_applications(position, counters, sink)?;
        if !self.plan.events.contains_key(&position) {
            return Ok(());
        }
        checkpoint(sink)?;
        let hidden = self.loaded.arch().hidden_size as usize;
        let values = read_capture(
            self.capture.as_ref().context("missing capture")?,
            self.layers.len() * hidden,
        )?;
        let before = if self.before_layers.is_empty() {
            Vec::new()
        } else {
            read_capture(
                self.before.as_ref().context("missing before capture")?,
                self.before_layers.len() * hidden,
            )?
        };
        for (slot, &layer) in self.layers.iter().enumerate() {
            let applied = self.interventions.applied_ids(position, layer);
            let residual = &values[slot * hidden..(slot + 1) * hidden];
            let before = self
                .before_layers
                .binary_search(&layer)
                .ok()
                .map(|slot| &before[slot * hidden..(slot + 1) * hidden]);
            paired::publish(
                self.plan, token, position, layer, counters, &applied, before, residual, sink,
            )?;
            for lens in self.plan.heads(position, layer).keys().copied() {
                checkpoint(sink)?;
                let start = Instant::now();
                let residual = &values[slot * hidden..(slot + 1) * hidden];
                #[cfg(test)]
                let mut transport_witness = None;
                let full = if lens == "plain" {
                    self.loaded
                        .passive_workspace_lens_session(&mut self.sequence)?
                        .deployed_logits_from_post_block_residual(residual)?
                } else {
                    let key = super::registry::MatrixKey {
                        alias: lens.into(),
                        layer,
                    };
                    let matrix = self
                        .staged
                        .matrices
                        .get(&key)
                        .context("missing admitted matrix")?;
                    let full = self
                        .fitted
                        .as_mut()
                        .context("missing fitted workspace")?
                        .apply_row_f16_transport_logits_with_vector(matrix, residual)?;
                    #[cfg(test)]
                    {
                        transport_witness = Some(verification::transport(
                            matrix,
                            residual,
                            &full.transported_values,
                        )?);
                    }
                    full
                };
                publish_head(
                    self.plan,
                    self.tokenizer,
                    HeadSite {
                        token,
                        position,
                        layer,
                        lens,
                        applied_operation_ids: &applied,
                        #[cfg(test)]
                        transport_witness,
                        #[cfg(test)]
                        source_values: Some(residual),
                        counters,
                        generation_logits: (lens == "plain"
                            && layer + 1 == self.loaded.arch().n_layer
                            && !generation_logits.is_empty())
                        .then_some(generation_logits),
                    },
                    full,
                    sink,
                    start,
                )?;
            }
        }
        Ok(())
    }
}

struct HeadSite<'a> {
    token: i32,
    position: u32,
    layer: u32,
    lens: &'a str,
    applied_operation_ids: &'a [&'a str],
    #[cfg(test)]
    transport_witness: Option<Value>,
    #[cfg(test)]
    source_values: Option<&'a [f32]>,
    counters: &'a Counters,
    generation_logits: Option<&'a [f32]>,
}

fn publish_head(
    plan: &Plan,
    tokenizer: &Tokenizer,
    site: HeadSite<'_>,
    full: qwen_llm::workspace_lens::WorkspaceLensFullVocabularyLogitsWithVector,
    sink: &Sink,
    start: Instant,
) -> Result<()> {
    let HeadSite {
        token,
        position,
        layer,
        lens,
        applied_operation_ids,
        #[cfg(test)]
        transport_witness,
        #[cfg(test)]
        source_values,
        counters,
        generation_logits,
    } = site;
    let phase = if u64::from(position) < counters.prompt_tokens {
        Phase::Prefill
    } else {
        Phase::Decode
    };
    let index = if phase == Phase::Prefill {
        u64::from(position)
    } else {
        u64::from(position) - counters.prompt_tokens
    };
    let witness = generation_logits
        .map(|actual| logit_witness(&full.logits, actual))
        .transpose()?;
    let indices = plan.events[&position][&layer]
        .iter()
        .copied()
        .filter(|&i| plan.readouts[i].lens == lens)
        .collect::<Vec<_>>();
    let asset = if lens == "plain" {
        None
    } else {
        Some(
            plan.registry
                .as_ref()
                .context("missing fitted registry")?
                .asset(lens)?,
        )
    };
    let max_k = indices
        .iter()
        .map(|&i| plan.readouts[i].top_k)
        .max()
        .context("empty readout event")?;
    let source_key = format!("source-{position}-{layer}");
    let logits_key = format!("logits-{position}-{layer}-{lens}");
    if plan
        .retained_heads
        .contains(&(position, layer, lens.to_owned()))
    {
        sink.array(json!({"key":logits_key,"quantity":"readout_logits","lens":lens,"source_key":source_key,
            "phase":phase,"index":index,"position":position,"source_layer":layer,"input_token_id":token,
            "target_layer":asset.map(|a|&a["target_layer"]),"asset_identity":asset.map(|a|&a["identity"]),
            "score_kind":"logit","candidate_universe":"full_vocabulary","capture_stage":"post_block_after_operations",
            "applied_operation_ids":applied_operation_ids,"provenance":"original_forward"}),&full.logits,phase,counters)?;
    }
    let top = full.into_topk(max_k)?;
    let mut label_bytes = 0usize;
    for score in &top.readout.scores {
        label_bytes = label_bytes
            .checked_add(
                tokenizer
                    .try_decode_piece_bytes_exact(score.token_id as i32)?
                    .len(),
            )
            .context("label bytes overflow")?;
        ensure!(
            label_bytes <= MAX_LABEL_BYTES,
            "readout labels exceed retained-record byte allowance"
        );
    }
    let scores = top
        .readout
        .scores
        .iter()
        .map(|score| -> Result<Score<'_>> {
            Ok(Score {
                token_id: score.token_id,
                row_id: score.token_id,
                label: String::from_utf8_lossy(
                    tokenizer.try_decode_piece_bytes_exact(score.token_id as i32)?,
                ),
                score: score.logit,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    for (ordinal, &request) in indices.iter().enumerate() {
        checkpoint(sink)?;
        let readout = &plan.readouts[request];
        sink.record(
            ReadoutRecord {
                kind: "readout",
                readout_id: &readout.id,
                lens,
                phase,
                index,
                position,
                input_token_id: token,
                predicts_position: u64::from(position) + 1,
                source_layer: layer,
                target_layer: asset
                    .and_then(|v| v["target_layer"].as_u64())
                    .map(|v| v as u32),
                asset_identity: asset.and_then(|v| v["identity"].as_str()),
                binding_status: asset.and_then(|v| v["transfer"].as_str()),
                method: asset.and_then(|v| v["method"].as_str()),
                capture_stage: "post_block_after_operations",
                applied_operation_ids,
                provenance: "original_forward",
                score_kind: "logit",
                candidate_universe: "full_vocabulary",
                scores: &scores[..readout.top_k],
                generation_logit_witness: witness.as_ref(),
                #[cfg(test)]
                test_transport_witness: transport_witness.as_ref(),
                #[cfg(test)]
                test_source_values: source_values,
                retained: readout.retain.as_ref().map(|_| Retained {
                    source_key: &source_key,
                    logits_key: &logits_key,
                }),
                cost: Cost {
                    readout_ms: (ordinal == 0).then_some(elapsed),
                    shared_head_position: position,
                    shared_head_layer: layer,
                    shared_head_lens: lens,
                },
            },
            phase,
            counters,
        );
        checkpoint(sink)?;
    }
    Ok(())
}

#[derive(Serialize)]
struct Score<'a> {
    token_id: u32,
    row_id: u32,
    label: Cow<'a, str>,
    score: f32,
}
#[derive(Serialize)]
struct Cost<'a> {
    readout_ms: Option<f64>,
    shared_head_position: u32,
    shared_head_layer: u32,
    shared_head_lens: &'a str,
}
#[derive(Serialize)]
struct ReadoutRecord<'a> {
    kind: &'static str,
    readout_id: &'a str,
    lens: &'a str,
    phase: Phase,
    index: u64,
    position: u32,
    input_token_id: i32,
    predicts_position: u64,
    source_layer: u32,
    target_layer: Option<u32>,
    asset_identity: Option<&'a str>,
    binding_status: Option<&'a str>,
    method: Option<&'a str>,
    capture_stage: &'static str,
    applied_operation_ids: &'a [&'a str],
    provenance: &'static str,
    score_kind: &'static str,
    candidate_universe: &'static str,
    scores: &'a [Score<'a>],
    generation_logit_witness: Option<&'a Value>,
    #[cfg(test)]
    #[serde(skip_serializing_if = "Option::is_none")]
    test_transport_witness: Option<&'a Value>,
    #[cfg(test)]
    #[serde(skip_serializing_if = "Option::is_none")]
    test_source_values: Option<&'a [f32]>,
    retained: Option<Retained<'a>>,
    cost: Cost<'a>,
}

#[derive(Serialize)]
struct Retained<'a> {
    source_key: &'a str,
    logits_key: &'a str,
}

fn publish_source(
    plan: &Plan,
    token: i32,
    position: u32,
    layer: u32,
    counters: &Counters,
    applied: &[&str],
    residual: &[f32],
    sink: &Sink,
) -> Result<()> {
    if !plan.retained_sources.contains(&(position, layer)) {
        return Ok(());
    }
    let phase = if u64::from(position) < counters.prompt_tokens {
        Phase::Prefill
    } else {
        Phase::Decode
    };
    let index = if phase == Phase::Prefill {
        u64::from(position)
    } else {
        u64::from(position) - counters.prompt_tokens
    };
    sink.array(json!({"key":format!("source-{position}-{layer}"),"quantity":"source_residual","phase":phase,"index":index,"position":position,
        "source_layer":layer,"input_token_id":token,"capture_stage":"post_block_after_operations","applied_operation_ids":applied,"provenance":"original_forward"}),residual,phase,counters)
}

fn read_capture(capture: &MetalTensor, count: usize) -> Result<Vec<f32>> {
    let bytes = count.checked_mul(4).context("capture byte overflow")?;
    let offset = usize::try_from(capture.offset)?;
    ensure!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= capture.buffer.length()),
        "capture readback outside buffer"
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .context("allocate capture readback")?;
    // The synchronous original forward completed successfully before this read.
    let source = unsafe {
        std::slice::from_raw_parts(
            capture
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<f32>(),
            count,
        )
    };
    values.extend_from_slice(source);
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite captured residual"
    );
    Ok(values)
}

fn logit_witness(observed: &[f32], actual: &[f32]) -> Result<Value> {
    ensure!(
        !actual.is_empty() && actual.len() == observed.len(),
        "generation/observer logit shape mismatch"
    );
    let mut max_abs = 0.0_f64;
    let mut within = true;
    for (&a, &b) in observed.iter().zip(actual) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite generation/observer logits"
        );
        let delta = (f64::from(a) - f64::from(b)).abs();
        max_abs = max_abs.max(delta);
        within &= delta <= 1e-4 + 1e-4 * f64::from(b).abs();
    }
    Ok(
        json!({"basis":"same_original_forward_generation_logits","vocabulary_size":actual.len(),"max_abs_error":max_abs,"absolute_tolerance":1e-4,"relative_tolerance":1e-4,"within_tolerance":within}),
    )
}

#[cfg(test)]
pub(crate) use tests::run_cpu_readouts;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::native::{CpuFixture, execute, writer};
    use std::sync::Arc;

    #[test]
    fn retained_producer_deduplicates_arrays_and_only_links_requesting_rows() {
        let mut fixture = CpuFixture::new();
        let (_files, registry) = super::super::registry::tests::fitted_fixture();
        let profile = Arc::get_mut(&mut fixture.profile).unwrap();
        profile.plain_readouts = true;
        profile.registry = Some(registry.clone());
        let mut request = fixture.request("arrays");
        request["preconditions"]["asset_identities"] =
            json!({"plain":"cpu-fixture","fit":registry.asset("fit").unwrap()["identity"]});
        let readout = |id, lens, retain| {
            json!({"id":id,"lens":lens,"mode":"full_vocabulary","top_k":2,"retain":retain,
            "scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]},"decode":{"kind":"values","values":[0]}}})
        };
        request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[readout("none","plain",None),readout("a","plain",Some("scores_and_residual")),readout("b","plain",Some("scores_and_residual")),readout("fit","fit",Some("scores_and_residual"))]});
        let prepared = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .ok()
            .unwrap();
        let id = fixture
            .store
            .accept_with_archive("arrays", &request, true, prepared.readouts.archive_bytes)
            .unwrap()
            .status
            .id;
        let writer = writer::Writer::spawn(
            fixture.store.clone(),
            id.clone(),
            fixture.store.control(&id).unwrap(),
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let outcome = run_cpu_readouts(&prepared, writer.sink(), &fixture.profile.tokenizer);
        writer.finish(outcome).unwrap();
        let page = fixture.store.result(&id, None, 128).unwrap();
        let arrays = page
            .records
            .iter()
            .filter(|r| r["kind"] == "retained_array")
            .collect::<Vec<_>>();
        assert_eq!(arrays.len(), 6);
        let total = arrays
            .iter()
            .map(|r| r["array"]["byte_length"].as_u64().unwrap())
            .sum::<u64>();
        assert_eq!(total, prepared.readouts.archive_bytes);
        for array in &arrays {
            let offset = array["array"]["url"]
                .as_str()
                .unwrap()
                .rsplit('/')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let bytes = fixture.store.array(&id, offset).unwrap();
            let values = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            if array["quantity"] == "source_residual" {
                assert_eq!(values, [1., 2.]);
            } else {
                assert_eq!(values.len(), fixture.profile.tokenizer.n_vocab() as usize);
                assert_eq!(values[120], 10.);
                assert_eq!(values[1], 0.);
            }
        }
        for row in page.records.iter().filter(|r| r["kind"] == "readout") {
            if row["readout_id"] == "none" {
                assert!(row["retained"].is_null());
                continue;
            }
            for field in ["source_key", "logits_key"] {
                let array = arrays
                    .iter()
                    .find(|a| a["key"] == row["retained"][field])
                    .unwrap();
                assert!(array["seq"].as_u64().unwrap() < row["seq"].as_u64().unwrap());
            }
        }
        let replacement = Arc::new(
            crate::serve::jobs::store::JobStore::open(
                &fixture.root.join("replacement"),
                crate::serve::jobs::store::Limits::default(),
            )
            .unwrap(),
        );
        drop(std::mem::replace(&mut fixture.store, replacement));
        let reopened = crate::serve::jobs::store::JobStore::open(
            &fixture.root.join("jobs"),
            crate::serve::jobs::store::Limits::default(),
        )
        .unwrap();
        assert_eq!(
            reopened.result(&id, None, 128).unwrap().records,
            page.records
        );
    }

    #[test]
    fn failed_array_write_discards_dependent_records_and_recovers_committed_prefix() {
        let mut fixture = CpuFixture::new();
        Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
        let mut request = fixture.request("array-failure");
        request["preconditions"]["asset_identities"] = json!({"plain":"cpu-fixture"});
        request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[{"id":"retained","lens":"plain","mode":"full_vocabulary","top_k":2,"retain":"scores_and_residual","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]}}}]});
        let prepared = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .ok()
            .unwrap();
        let id = fixture
            .store
            .accept_with_archive(
                "array-failure",
                &request,
                true,
                prepared.readouts.archive_bytes,
            )
            .unwrap()
            .status
            .id;
        let writer = writer::Writer::spawn(
            fixture.store.clone(),
            id.clone(),
            fixture.store.control(&id).unwrap(),
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        fixture
            .store
            .fail_once(crate::serve::jobs::store::FaultPoint::ArrayPartialWrite);
        let outcome = run_cpu_readouts(&prepared, writer.sink(), &fixture.profile.tokenizer);
        writer.finish(outcome).unwrap();
        let status = fixture.store.status(&id).unwrap();
        assert!(status.result.complete);
        assert!(status.result.error.is_some());
        let records = fixture.store.result(&id, None, 128).unwrap().records;
        assert!(
            !records
                .iter()
                .any(|r| r["kind"] == "retained_array" || r["kind"] == "readout")
        );
        let replacement = Arc::new(
            crate::serve::jobs::store::JobStore::open(
                &fixture.root.join("replacement"),
                crate::serve::jobs::store::Limits::default(),
            )
            .unwrap(),
        );
        drop(std::mem::replace(&mut fixture.store, replacement));
        let reopened = crate::serve::jobs::store::JobStore::open(
            &fixture.root.join("jobs"),
            crate::serve::jobs::store::Limits::default(),
        )
        .unwrap();
        assert_eq!(reopened.result(&id, None, 128).unwrap().records, records);
        assert_eq!(
            std::fs::metadata(fixture.root.join("jobs").join(id).join("arrays.bin"))
                .unwrap()
                .len(),
            0
        );
    }

    struct Synthetic<'a> {
        plan: &'a Plan,
        interventions: &'a super::super::interventions::Plan,
        tokenizer: &'a Tokenizer,
        forwards: usize,
        heads: usize,
        staged: Option<&'a super::super::registry::Staged>,
    }
    impl TokenEngine for Synthetic<'_> {
        fn forward(&mut self, _: i32, position: u32, _: bool) -> Result<Vec<f32>> {
            assert_eq!(position as usize, self.forwards);
            self.forwards += 1;
            let mut logits = vec![0.0; self.tokenizer.n_vocab() as usize];
            logits[120] = 10.;
            Ok(logits)
        }
        fn observe(
            &mut self,
            token: i32,
            position: u32,
            logits: &[f32],
            counters: &Counters,
            sink: &Sink,
        ) -> Result<()> {
            assert_eq!(
                self.forwards as u64,
                counters.consumed_prompt_tokens + counters.consumed_generated_tokens
            );
            self.interventions
                .publish_applications(position, counters, sink)?;
            if let Some(event) = self.plan.events.get(&position) {
                for &layer in event.keys() {
                    let applied = self.interventions.applied_ids(position, layer);
                    paired::publish(
                        self.plan,
                        token,
                        position,
                        layer,
                        counters,
                        &applied,
                        self.plan
                            .pairs
                            .events
                            .get(&position)
                            .and_then(|e| e.get(&layer))
                            .map(|_| [1., 2.].as_slice()),
                        &[1., 2.],
                        sink,
                    )?;
                    for lens in self.plan.heads(position, layer).keys().copied() {
                        checkpoint(sink)?;
                        if lens != "plain" {
                            let key = super::super::registry::MatrixKey {
                                alias: lens.into(),
                                layer,
                            };
                            assert!(!self.staged.unwrap().matrices[&key].is_empty());
                        }
                        self.heads += 1;
                        publish_head(
                            self.plan,
                            self.tokenizer,
                            HeadSite {
                                token,
                                position,
                                layer,
                                lens,
                                applied_operation_ids: &applied,
                                transport_witness: None,
                                source_values: None,
                                counters,
                                generation_logits: (lens == "plain"
                                    && layer == 1
                                    && position as u64 + 1 >= counters.prompt_tokens)
                                    .then_some(logits),
                            },
                            qwen_llm::workspace_lens::WorkspaceLensFullVocabularyLogitsWithVector {
                                logits: logits.to_vec(),
                                transported_values: vec![1., 2.],
                                rms_denominator_f64_recomputed: 1.,
                            },
                            sink,
                            Instant::now(),
                        )?;
                    }
                }
            }
            Ok(())
        }
    }
    pub(crate) fn run_cpu_readouts(
        prepared: &crate::serve::native::Prepared,
        sink: &Sink,
        tokenizer: &Tokenizer,
    ) -> crate::serve::native::Outcome {
        super::super::with_staged(
            prepared,
            sink,
            prepared.staging.matrix_bytes + super::super::registry::STAGING_OVERHEAD_BYTES,
            |staged| {
                let mut engine = Synthetic {
                    plan: &prepared.readouts,
                    interventions: &prepared.interventions,
                    tokenizer,
                    forwards: 0,
                    heads: 0,
                    staged: Some(staged),
                };
                execute::run_engine(prepared, sink, &[], &mut engine, |_| Ok(b"x".to_vec()))
            },
        )
    }

    #[test]
    fn original_consumption_drives_shared_heads_and_durable_plain_records() {
        let mut fixture = CpuFixture::new();
        Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
        let mut request = fixture.request("plain-shared");
        let prompt = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .ok()
            .unwrap()
            .prompt
            .len();
        request["preconditions"]["asset_identities"] = json!({"plain":"cpu-fixture"});
        let readout = |id, k| {
            json!({"id":id,"lens":"plain","mode":"full_vocabulary","top_k":k,
            "scope":{"layers":{"kind":"all"},"prefill":{"kind":"values","values":[0,prompt-1]},"decode":{"kind":"all"}}})
        };
        request["diagnostics"] =
            json!({"directions":[],"operations":[],"readouts":[readout("a",3),readout("b",1)]});
        let prepared = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .ok()
            .unwrap();
        let id = fixture
            .store
            .accept("plain-shared", &request, true)
            .unwrap()
            .status
            .id;
        let writer = writer::Writer::spawn(
            Arc::clone(&fixture.store),
            id.clone(),
            fixture.store.control(&id).unwrap(),
            &prepared,
            Default::default(),
        )
        .unwrap();
        writer.wait_ready().unwrap();
        let mut engine = Synthetic {
            plan: &prepared.readouts,
            interventions: &prepared.interventions,
            tokenizer: &fixture.profile.tokenizer,
            forwards: 0,
            heads: 0,
            staged: None,
        };
        let outcome = execute::run_engine(&prepared, writer.sink(), &[], &mut engine, |_| {
            Ok(b"x".to_vec())
        });
        assert_eq!(
            outcome.reason,
            crate::serve::jobs::state::StopReason::TokenLimit
        );
        assert_eq!(engine.forwards, prompt + 2);
        assert_eq!(engine.heads, 8);
        writer.finish(outcome).unwrap();
        let page = serde_json::to_value(fixture.store.result(&id, None, 64).unwrap()).unwrap();
        let rows = page["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == "readout")
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 16);
        for pair in rows.chunks_exact(2) {
            assert_eq!(pair[0]["readout_id"], "a");
            assert_eq!(pair[1]["readout_id"], "b");
            assert_eq!(pair[0]["scores"].as_array().unwrap().len(), 3);
            assert_eq!(pair[1]["scores"].as_array().unwrap().len(), 1);
            assert_eq!(pair[0]["scores"][0]["token_id"], 120);
            assert_eq!(pair[0]["scores"][0]["label"], "x");
            assert!(!pair[0]["cost"]["readout_ms"].is_null());
            assert!(pair[1]["cost"]["readout_ms"].is_null());
            assert!(pair[0]["position"].as_u64().unwrap() < (prompt + 2) as u64);
        }
        let status = fixture.store.status(&id).unwrap();
        assert_eq!(status.observations.committed_records, 16);
        assert_eq!(
            status.observations.state,
            crate::serve::jobs::state::ObservationState::Complete
        );
        assert_eq!(status.generation.counters.consumed_generated_tokens, 2);
    }

    #[test]
    fn interruption_between_shared_publications_stops_before_another_head_or_forward() {
        use crate::serve::jobs::state::{JobState, StopReason};
        for server_stop in [false, true] {
            let mut fixture = CpuFixture::new();
            Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
            let mut request = fixture.request("between-readouts");
            request["preconditions"]["asset_identities"] = json!({"plain":"cpu-fixture"});
            let readout = |id| {
                json!({"id":id,"lens":"plain","mode":"full_vocabulary","top_k":1,
                "scope":{"layers":{"kind":"all"},"prefill":{"kind":"values","values":[0]}}})
            };
            request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[readout("first"),readout("second")]});
            let prepared = fixture
                .profile
                .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
                .ok()
                .unwrap();
            let id = fixture
                .store
                .accept("between-readouts", &request, true)
                .unwrap()
                .status
                .id;
            let mut writer = writer::Writer::spawn(
                Arc::clone(&fixture.store),
                id.clone(),
                fixture.store.control(&id).unwrap(),
                &prepared,
                Default::default(),
            )
            .unwrap();
            writer.after_record(move |sink| {
                if server_stop {
                    sink.server.close();
                } else {
                    sink.fail_recording(
                        "injected_writer_failure",
                        "stop after first shared readout",
                    );
                }
            });
            writer.wait_ready().unwrap();
            let mut engine = Synthetic {
                plan: &prepared.readouts,
                interventions: &prepared.interventions,
                tokenizer: &fixture.profile.tokenizer,
                forwards: 0,
                heads: 0,
                staged: None,
            };
            let outcome = execute::run_engine(&prepared, writer.sink(), &[], &mut engine, |_| {
                panic!("stopped during first prompt forward")
            });
            assert_eq!(engine.forwards, 1);
            assert_eq!(engine.heads, 1);
            assert_eq!(outcome.counters.consumed_prompt_tokens, 1);
            assert_eq!(
                outcome.reason,
                if server_stop {
                    StopReason::ServerRestart
                } else {
                    StopReason::Cancelled
                }
            );
            writer.finish(outcome).unwrap();
            let status = fixture.store.status(&id).unwrap();
            assert!(status.result.complete);
            assert_eq!(
                status.state,
                if server_stop {
                    JobState::Interrupted
                } else {
                    JobState::Failed
                }
            );
            assert_eq!(status.result.error.is_some(), !server_stop);
            let page = serde_json::to_value(fixture.store.result(&id, None, 64).unwrap()).unwrap();
            let rows = page["records"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["kind"] == "readout")
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0]["readout_id"], "first");
        }
    }

    #[test]
    fn aggregate_labels_are_bounded_before_lossy_publication_and_never_truncated() {
        for (top_k, token, expected_label, succeeds) in [
            (1, 261, "x".repeat(MAX_LABEL_BYTES), true),
            (2, 261, String::new(), false),
            (1, 255, String::from_utf8_lossy(&[255]).into_owned(), true),
            (1, 0, "\0".into(), true),
            (1, 34, "\"".into(), true),
        ] {
            let mut fixture =
                CpuFixture::with_extra_tokens(&["x".repeat(MAX_LABEL_BYTES), "yz".into()]);
            Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
            let mut request = fixture.request("labels");
            request["preconditions"]["asset_identities"] = json!({"plain":"cpu-fixture"});
            request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[{"id":"labels","lens":"plain","mode":"full_vocabulary","top_k":top_k,
                "scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]}}}]});
            let prepared = fixture
                .profile
                .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
                .ok()
                .unwrap();
            let id = fixture
                .store
                .accept("labels", &request, true)
                .unwrap()
                .status
                .id;
            let writer = writer::Writer::spawn(
                Arc::clone(&fixture.store),
                id.clone(),
                fixture.store.control(&id).unwrap(),
                &prepared,
                Default::default(),
            )
            .unwrap();
            writer.wait_ready().unwrap();
            let counters = Counters {
                prompt_tokens: prepared.prompt.len() as u64,
                consumed_prompt_tokens: 1,
                ..Default::default()
            };
            let mut logits = vec![0.; 263];
            logits[262] = 5.;
            logits[token] = 10.;
            let result = publish_head(
                &prepared.readouts,
                &fixture.profile.tokenizer,
                HeadSite {
                    token: prepared.prompt[0],
                    position: 0,
                    layer: 0,
                    lens: "plain",
                    applied_operation_ids: &[],
                    transport_witness: None,
                    source_values: None,
                    counters: &counters,
                    generation_logits: None,
                },
                qwen_llm::workspace_lens::WorkspaceLensFullVocabularyLogitsWithVector {
                    logits,
                    transported_values: vec![0.; 2],
                    rms_denominator_f64_recomputed: 1.,
                },
                writer.sink(),
                Instant::now(),
            );
            assert_eq!(result.is_ok(), succeeds, "{result:?}");
            writer
                .finish(crate::serve::native::Outcome::interrupted(counters))
                .unwrap();
            let page = serde_json::to_value(fixture.store.result(&id, None, 64).unwrap()).unwrap();
            let rows = page["records"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["kind"] == "readout")
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), usize::from(succeeds));
            if succeeds {
                assert_eq!(rows[0]["scores"][0]["label"], expected_label);
            }
        }
    }

    #[test]
    fn witness_keeps_mismatches_visible_and_refuses_nonfinite_values() {
        assert_eq!(
            logit_witness(&[1., 2.], &[1., 2.]).unwrap()["within_tolerance"],
            true
        );
        assert_eq!(
            logit_witness(&[1., 3.], &[1., 2.]).unwrap()["within_tolerance"],
            false
        );
        assert!(logit_witness(&[f32::NAN], &[0.]).is_err());
        assert!(logit_witness(&[0.], &[]).is_err());
    }
}

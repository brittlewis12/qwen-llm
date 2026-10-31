//! Actual-Q4 bitwise smoke only: three frozen documents, no performance/tool packet.
use super::*;
use crate::glm5_next::memory::{KdaHalfInputsScope, TensorGateUpScope};
use crate::glm5_next_metal::tensor_q5_down;
use crate::tensor::TensorDesc;

#[path = "kda_half_plain.rs"]
mod plain;

const CAPACITY: usize = 4416;
const HALF_BYTES: u64 = 16 << 20;
const Q4_PIN: &[u8] =
    include_bytes!("../../../../../../docs/bench/2026-10-16-glm-quality-comparison/stock-q4.json");
const CONVERT: &str = "kernel_scatter_offset_f32_to_f16";
const HALF: &str = "kernel_mat_mat_q8_0_f16act_tensor_wide";
const WIDE: &str = "kernel_mat_mat_q8_0_f32_tensor_wide";
const DOWN: &str = "kernel_moe_down_q5_K_f32_grouped_slots_tensor";
const Q6_DOWN_BLOCKS: [usize; 3] = [11, 12, 44];
const Q5_GATE_UP_BLOCKS: [usize; 1] = [11];
const Q8_MLA_BLOCKS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];

pub(super) fn metadata_pins() -> Vec<(String, GgmlType, Vec<u64>)> {
    let mut pins = Vec::new();
    for block in 3..45 {
        let down = if Q6_DOWN_BLOCKS.contains(&block) {
            GgmlType::Q6_K
        } else {
            GgmlType::Q5_K
        };
        let gate_up = if Q5_GATE_UP_BLOCKS.contains(&block) {
            GgmlType::Q5_K
        } else {
            GgmlType::Q4_K
        };
        for (suffix, dtype, shape) in [
            ("down", down, vec![2048, 4096, 288]),
            ("gate", gate_up, vec![4096, 2048, 288]),
            ("up", gate_up, vec![4096, 2048, 288]),
        ] {
            pins.push((
                format!("blk.{block}.ffn_{suffix}_exps.weight"),
                dtype,
                shape,
            ));
        }
    }
    for block in Q8_MLA_BLOCKS.into_iter().chain([45]) {
        for (suffix, shape) in [("k", vec![256, 512, 64]), ("v", vec![512, 256, 64])] {
            pins.push((
                format!("blk.{block}.attn_{suffix}_b.weight"),
                GgmlType::Q8_0,
                shape,
            ));
        }
    }
    pins
}

pub(super) fn assert_metadata_pins(tensors: &[TensorDesc]) {
    for (name, dtype, shape) in metadata_pins() {
        let tensor = tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("missing pinned tensor {name}"));
        assert_eq!(tensor.dtype, dtype, "{name} dtype");
        assert_eq!(tensor.shape, shape, "{name} shape");
    }
    assert_eq!(tensors.iter().filter(|t| t.name.ends_with(".attn_k_b.weight") || t.name.ends_with(".attn_v_b.weight")).count(), 24);
}

fn pin_q4_shards(gguf: &GgufFile) -> String {
    let pin: Value = serde_json::from_slice(Q4_PIN).unwrap();
    assert_eq!(gguf.shards.len(), 6);
    assert_eq!(gguf.tensors.len(), 1412);
    for (shard, expected) in gguf.shards.iter().zip(pin["shards"].as_array().unwrap()) {
        assert_eq!(
            shard.path.file_name().unwrap().to_str().unwrap(),
            expected["name"].as_str().unwrap()
        );
        assert_eq!(shard.mmap_len() as u64, expected["size"].as_u64().unwrap());
    }
    let header_sha = sha256_hex(gguf.primary_shard().mmap_bytes());
    assert_eq!(header_sha, pin["shards"][0]["sha256"].as_str().unwrap());
    header_sha
}

fn cpu_reserve_total(terms: &[(&str, u64)]) -> u64 {
    terms
        .iter()
        .try_fold(0u64, |n, (_, bytes)| n.checked_add(*bytes))
        .unwrap()
}

fn preload_admission(
    peak: u64,
    cpu: u64,
    signals: crate::metal::MetalMemorySignals,
) -> crate::metal::MetalMemoryAdmission {
    crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(peak, cpu, 0, signals, true)
}

fn load_admitted(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &crate::glm5_next::Glm5NextModel<'_>,
    cpu: u64,
    out: &mut Report,
) -> Glm5NextWeights {
    let _allocation = ctx.begin_allocation_transaction();
    let preflight = preflight_session(ctx, gguf, model, CAPACITY, ROWS).unwrap();
    assert!(preflight.ledger.scratch_options().kda_half_inputs);
    let admission = preload_admission(preflight.ledger.peak_bytes(), cpu, ctx.memory_signals());
    out.emit(json!({"event":"preload_admission","ledger":preflight.ledger.terms(),
        "cpu_reserve_bytes":cpu,"required_bytes":admission.required_bytes,"admitted":admission.is_admitted(),
        "reason":admission.reason.as_str(),"cpu_availability_guard":true,
        "cpu_required_bytes":admission.cpu_required_bytes,"cpu_charged_to_nonzero_process_budget":true}));
    assert!(
        admission.is_admitted(),
        "full model/session/CPU admission refused"
    );
    Glm5NextWeights::load(ctx, gguf).unwrap()
}

fn reserve(vocab: u32) -> (Vec<(&'static str, u64)>, u64) {
    // Parent's two 64-row captures plus one extra prefill row per trajectory.
    let mut terms = cpu_terms(vocab, true);
    terms.push((
        "two_extra_prefill_rows_and_headers",
        2 * (u64::from(vocab) * 4 + size_of::<Vec<f32>>() as u64),
    ));
    let total = cpu_reserve_total(&terms);
    (terms, total)
}

#[test]
fn cpu_kda_half_preload_checks_cpu_availability_not_only_combined_budget() {
    let mut signals = crate::metal::MetalMemorySignals {
        recommended_max_bytes: 8 << 30,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(0),
        host_budget: Some(crate::metal::HostMemoryBudget {
            physical_bytes: 16 << 30,
            conservative_available_bytes: 128 << 20,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    };
    let (peak, cpu) = (1 << 30, 256 << 20);
    assert!(
        crate::metal::evaluate_metal_memory_admission(peak + cpu, 0, signals, true).is_admitted()
    );
    let refused = preload_admission(peak, cpu, signals);
    assert_eq!(
        refused.reason,
        crate::metal::MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    assert_eq!(refused.cpu_required_bytes, Some(cpu));
    assert_eq!(refused.required_bytes, Some(peak + cpu));
    signals
        .host_budget
        .as_mut()
        .unwrap()
        .conservative_available_bytes = 512 << 20;
    assert!(preload_admission(peak, cpu, signals).is_admitted());
}

fn bitwise(a: &[Vec<f32>], b: &[Vec<f32>], vocab: usize) {
    assert_eq!(a.len(), CONTINUATION + 1);
    assert_eq!(b.len(), a.len());
    for (row, (a, b)) in a.iter().zip(b).enumerate() {
        check_logits(a, vocab);
        check_logits(b, vocab);
        assert!(
            a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "full-vocab bitwise mismatch at row {row}"
        );
    }
}

fn counts(rows: Vec<DispatchCensusRow>) -> BTreeMap<String, usize> {
    let heap = rows.capacity() * size_of::<DispatchCensusRow>()
        + rows
            .iter()
            .map(|r| r.kernel.capacity() + r.tag.as_ref().map_or(0, String::capacity))
            .sum::<usize>();
    assert!(2 * heap as u64 <= CENSUS_BYTES);
    let mut result = BTreeMap::new();
    for row in &rows {
        *result.entry(row.kernel.clone()).or_default() += 1;
    }
    result
}

fn audit_census(current: &BTreeMap<String, usize>, candidate: &BTreeMap<String, usize>) {
    let count = |map: &BTreeMap<String, usize>, name: &str| map.get(name).copied().unwrap_or(0);
    assert_eq!(count(current, HALF), 0);
    assert_eq!(count(candidate, HALF), 204);
    assert_eq!(count(candidate, CONVERT), count(current, CONVERT) + 34);
    assert_eq!(count(current, WIDE), count(candidate, WIDE) + 204);
    for map in [current, candidate] {
        assert_eq!(count(map, DOWN), 39);
    }
    let unchanged = |map: &BTreeMap<String, usize>| {
        map.iter()
            .filter(|(name, _)| ![CONVERT, HALF, WIDE].contains(&name.as_str()))
            .map(|(name, count)| (name.clone(), *count))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(
        unchanged(current),
        unchanged(candidate),
        "unrelated dispatch counts changed"
    );
}

#[test]
fn cpu_kda_half_smoke_protocol_is_three_documents_and_bounded_captures() {
    let docs = documents();
    assert_eq!(docs.len(), 3);
    assert_eq!(
        docs.iter().map(|d| d.prefix).collect::<Vec<_>>(),
        [2048, 2048, 4352]
    );
    assert_eq!(
        docs.iter().map(|d| d.prefix + CONTINUATION).max(),
        Some(CAPACITY)
    );
    assert_eq!(CONTINUATION, 64);
    assert_eq!(ROWS * 4096 * 2, HALF_BYTES as usize);
    let (terms, cpu) = reserve(154880);
    assert_eq!(
        &terms[..cpu_terms(154880, true).len()],
        cpu_terms(154880, true)
    );
    assert!(cpu > 2 * 65 * 154880 * 4);
    assert!(cpu < 256 << 20);
}

#[test]
fn cpu_kda_half_smoke_census_requires_exact_stage_delta_and_qualified_down() {
    let current = BTreeMap::from([(CONVERT.into(), 11), (WIDE.into(), 300), (DOWN.into(), 39)]);
    let mut candidate = BTreeMap::from([
        (CONVERT.into(), 45),
        (WIDE.into(), 96),
        (HALF.into(), 204),
        (DOWN.into(), 39),
    ]);
    audit_census(&current, &candidate);
    for (name, wrong) in [(CONVERT, 46), (HALF, 203), (DOWN, 38)] {
        let old = candidate.insert(name.into(), wrong).unwrap();
        assert!(std::panic::catch_unwind(|| audit_census(&current, &candidate)).is_err());
        candidate.insert(name.into(), old);
    }
}

#[test]
fn cpu_kda_half_smoke_comparison_requires_finite_full_vocabulary_bits() {
    let a = vec![vec![0.0, 1.0]; CONTINUATION + 1];
    let mut b = a.clone();
    bitwise(&a, &b, 2);
    for (row, col, value) in [
        (64, 1, f32::from_bits(1.0f32.to_bits() + 1)),
        (0, 0, -0.0),
        (32, 1, f32::NAN),
    ] {
        b[row][col] = value;
        assert!(std::panic::catch_unwind(|| bitwise(&a, &b, 2)).is_err());
        b[row][col] = a[row][col];
    }
    let nan = vec![vec![f32::NAN]; CONTINUATION + 1];
    assert!(std::panic::catch_unwind(|| bitwise(&nan, &nan, 1)).is_err());
}

#[test]
fn cpu_kda_half_smoke_metadata_pins_exact_expert_and_serialized_mla_assignments() {
    let pins = metadata_pins();
    let names = |dtype, suffix: &str| {
        pins.iter()
            .filter(|(name, kind, _)| *kind == dtype && name.ends_with(suffix))
            .map(|(name, _, _)| name.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(pins.len(), 150);
    assert_eq!(
        names(GgmlType::Q6_K, ".ffn_down_exps.weight"),
        [
            "blk.11.ffn_down_exps.weight",
            "blk.12.ffn_down_exps.weight",
            "blk.44.ffn_down_exps.weight"
        ]
    );
    assert_eq!(names(GgmlType::Q5_K, ".ffn_down_exps.weight").len(), 39);
    assert_eq!(
        names(GgmlType::Q5_K, ".ffn_gate_exps.weight"),
        ["blk.11.ffn_gate_exps.weight"]
    );
    assert_eq!(
        names(GgmlType::Q5_K, ".ffn_up_exps.weight"),
        ["blk.11.ffn_up_exps.weight"]
    );
    assert_eq!(
        names(GgmlType::Q4_K, ".ffn_gate_exps.weight").len()
            + names(GgmlType::Q4_K, ".ffn_up_exps.weight").len(),
        82
    );
    for suffix in ["k", "v"] {
        assert_eq!(
            names(GgmlType::Q8_0, &format!(".attn_{suffix}_b.weight")),
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45]
                .map(|block| format!("blk.{block}.attn_{suffix}_b.weight"))
        );
    }
}

#[test]
fn cpu_kda_half_smoke_metadata_rejects_count_preserving_swaps_and_mla_drift() {
    let tensors = metadata_pins()
        .into_iter()
        .map(|(name, dtype, shape)| {
            let (_, bytes) = crate::metal::checked_ggml_shape_bytes(&shape, dtype).unwrap();
            TensorDesc {
                name,
                dtype,
                shape,
                shard_idx: 0,
                data_offset: 0,
                n_bytes: bytes as u64,
            }
        })
        .collect::<Vec<_>>();
    assert_metadata_pins(&tensors);
    for (a, b) in [
        ("blk.11.ffn_down_exps.weight", "blk.10.ffn_down_exps.weight"),
        ("blk.11.ffn_gate_exps.weight", "blk.12.ffn_gate_exps.weight"),
        ("blk.11.ffn_up_exps.weight", "blk.12.ffn_up_exps.weight"),
    ] {
        let mut bad = tensors.clone();
        let i = bad.iter().position(|t| t.name == a).unwrap();
        let j = bad.iter().position(|t| t.name == b).unwrap();
        let (left, right) = (bad[i].dtype, bad[j].dtype);
        bad[i].dtype = right;
        bad[j].dtype = left;
        assert!(std::panic::catch_unwind(|| assert_metadata_pins(&bad)).is_err());
    }
    for block in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45] {
        for suffix in ["k", "v"] {
            let name = format!("blk.{block}.attn_{suffix}_b.weight");
            let i = tensors.iter().position(|t| t.name == name).unwrap();
            for change in 0..3 {
                let mut bad = tensors.clone();
                match change {
                    0 => bad[i].name = format!("blk.46.attn_{suffix}_b.weight"),
                    1 => bad[i].dtype = GgmlType::Q6_K,
                    _ => bad[i].shape.swap(0, 1),
                }
                assert!(std::panic::catch_unwind(|| assert_metadata_pins(&bad)).is_err());
            }
        }
    }
    let mut extra = tensors.clone();
    let mut tensor = tensors.last().unwrap().clone();
    tensor.name = "blk.46.attn_v_b.weight".into();
    extra.push(tensor);
    assert!(std::panic::catch_unwind(|| assert_metadata_pins(&extra)).is_err());
}

#[test]
#[ignore = "actual pinned Q4 M5; release, validation ON, production lease, global QWEN_GLM53_KDA_HALF_INPUTS=1, fresh GLM53_KDA_HALF_SMOKE_REPORT"]
fn m5_kda_half_actual_q4_three_document_bitwise_smoke() {
    environment();
    assert_eq!(
        std::env::var("QWEN_GLM53_KDA_HALF_INPUTS").as_deref(),
        Ok("1")
    );
    assert!(memory::kda_half_inputs_enabled());
    let _routed_off = TensorGateUpScope::new(false);
    let _lease = production_lease();
    let _dense = tensor_prefill::Scope::new(true);
    let _down = tensor_down::Scope::new(true);
    let _wide = tensor_prefill::Q8WideScope::new(true);
    let _q5 = tensor_q5_down::Scope::new(true);
    let mut out = Report::new("GLM53_KDA_HALF_SMOKE_REPORT", true);
    let docs = documents();
    let gguf = GgufFile::open(std::env::var_os("GLM53_GGUF").unwrap()).expect("open Q4 artifact");
    let header_sha = pin_q4_shards(&gguf);
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = Glm5NextArtifactLayout::inspect(&gguf).unwrap();
    assert!(artifact.packed_prefill());
    let c = artifact.config();
    assert_eq!(c.blocks.len(), 45);
    assert_eq!(c.leading_dense_block_count, 3);
    assert_metadata_pins(&gguf.tensors);
    let vocab = c.vocab_size as usize;
    assert!(
        docs.iter()
            .all(|d| d.tokens.iter().all(|&t| (t as usize) < vocab))
    );
    assert_eq!(
        docs.iter().map(|d| d.prefix + CONTINUATION).max(),
        Some(CAPACITY)
    );
    let mut audit = Vec::new();
    let mut q5_names = Vec::new();
    for (block, layer) in artifact.model().blocks.iter().enumerate() {
        if let MixerTensors::Kda(kda) = &layer.mixer {
            for (weight, width) in [
                (kda.query, 8192),
                (kda.key, 8192),
                (kda.value, 8192),
                (kda.decay_a, 128),
                (kda.beta, 64),
                (kda.gate_a, 128),
            ] {
                assert_eq!(weight.dtype, GgmlType::Q8_0);
                assert_eq!(weight.shape, [4096, width]);
                audit.push(
                    json!({"block":block,"name":weight.name,"dtype":"Q8_0","shape":weight.shape}),
                );
            }
        }
        if let FfnTensors::Moe(moe) = &layer.ffn {
            if moe.down_experts.dtype == GgmlType::Q5_K {
                assert_eq!(moe.down_experts.shape, [2048, 4096, 288]);
                q5_names.push(moe.down_experts.name.clone());
            }
        }
    }
    assert_eq!(audit.len(), 204);
    assert_eq!(q5_names.len(), 39);
    let (terms, cpu) = reserve(c.vocab_size);
    let ctx = MetalContext::new().unwrap();
    assert!(ctx.device.name().to_string().contains("M5"));
    assert!(ctx.tensor_ops_status().is_available());
    assert!(crate::metal::moe_grouped_tensor_selected(&ctx));
    let half_price = ctx
        .price_shared_buffer_upper(HALF_BYTES)
        .unwrap()
        .priced_upper_bytes;
    out.emit(json!({"event":"header","schema":"glm53.kda_half_bitwise_smoke.v1",
        "candidate_base_commit":"4ace096b47cb1a9e9a5a48c45ededfe599023e1c",
        "binary_sha256":binary_hash(),"test_source_sha256":sha256_hex(include_bytes!("kda_half_inputs.rs")),
        "fixture_sha256":sha256_hex(FROZEN),"artifact_pin_sha256":sha256_hex(Q4_PIN),
        "metadata_shard_sha256":header_sha,"artifact_layout":artifact_layout(&gguf),
        "identity_scope":"pinned shard names/sizes, metadata-only primary SHA, tensor-table digest and retained stamps; weight payload hashes not verified",
        "shard_stamps":stamps.iter().map(|s|json!({"shard":s.shard_idx,"bytes":s.size,"device":s.device,"inode":s.inode,
            "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect::<Vec<_>>(),
        "qualified_wide_q8_and_q5_down":true,"routed_tensor_gate_up":false,"stage_arms":[["CURRENT",false],["KDA_HALF",true]],
        "capacity":CAPACITY,"rows":ROWS,"half_backing_logical_bytes_both_arms":HALF_BYTES,
        "half_backing_priced_upper_bytes_both_arms":half_price,"cpu_terms":terms,"cpu_reserve_bytes":cpu,
        "eligible_kda_input_audit":audit,"q5_down_audit":q5_names,
        "strict_metadata_pins":{"q6_down_blocks":Q6_DOWN_BLOCKS,"q5_gate_up_blocks":Q5_GATE_UP_BLOCKS,
            "executed_q8_mla_blocks":Q8_MLA_BLOCKS,"serialized_mtp_mla_block":45,"named_tensors":150},
        "documents":docs.iter().map(|d|json!({"stratum":d.stratum,"prefix":d.prefix,"tokens_sha256":token_hash(&d.tokens)})).collect::<Vec<_>>(),
        "teacher_forced_steps":CONTINUATION,"compared_rows_per_document":CONTINUATION+1,
        "timed":false,"performance_verdict":null,"state_capture":false}));
    let weights = load_admitted(&ctx, &gguf, artifact.model(), cpu, &mut out);
    let ledger = Glm5NextSession::plan_memory(&ctx, &weights, CAPACITY, ROWS).unwrap();
    let specs = memory::packed_normalized_half_specs(c, ROWS as u64, ledger.scratch_options());
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "normalized_half");
    assert_eq!(specs[0].shape, [4096, 2048]);
    assert_eq!(specs[0].bytes(), Some(HALF_BYTES));
    let mut warm: Option<(BTreeMap<String, usize>, Vec<f32>)> = None;
    for (label, enabled) in [("CURRENT", false), ("KDA_HALF", true)] {
        assert!(
            memory::kda_half_inputs_enabled(),
            "session construction must use global storage reservation"
        );
        let mut s = fresh(
            &ctx,
            &weights,
            CAPACITY,
            cpu,
            label,
            "warm_census",
            &mut out,
        );
        assert_eq!(*s.ledger(), ledger);
        let census = Census::begin();
        let (logits, _) = {
            let _stage = KdaHalfInputsScope::new(enabled);
            observed_prefill(&ctx, &mut s, &docs[0].tokens[..ROWS], DENSE_AND_DOWN)
        };
        let rows = census.finish();
        assert_plain();
        check_logits(&logits, vocab);
        let kernels = counts(rows);
        out.emit(json!({"event":"warm_census","arm":label,"kernel_counts":kernels,
            "expected_kda_conversions_added":if enabled {34} else {0},"expected_half_input_consumers":if enabled {204} else {0},
            "expected_unchanged_q5_down":39,"timed":false}));
        if let Some((current, reference)) = warm.take() {
            audit_census(&current, &kernels);
            assert!(
                reference
                    .iter()
                    .zip(&logits)
                    .all(|(a, b): (&f32, &f32)| a.to_bits() == b.to_bits()),
                "warm logits differ"
            );
        } else {
            warm = Some((kernels, logits));
        }
    }
    for (case, d) in docs.iter().enumerate() {
        let mut trajectories = Vec::with_capacity(2);
        for (label, enabled) in [("CURRENT", false), ("KDA_HALF", true)] {
            assert_plain();
            assert!(memory::kda_half_inputs_enabled());
            let mut s = fresh(&ctx, &weights, CAPACITY, cpu, label, "bitwise", &mut out);
            assert_eq!(*s.ledger(), ledger);
            let (logits, _) = {
                let _stage = KdaHalfInputsScope::new(enabled);
                observed_prefill(&ctx, &mut s, &d.tokens[..d.prefix], DENSE_AND_DOWN)
            };
            assert_plain();
            let mut capture = Vec::with_capacity(CONTINUATION + 1);
            capture.push(logits);
            for (i, &token) in d.tokens[d.prefix..].iter().enumerate() {
                capture.push(s.forward(&ctx, token).unwrap());
                check_session(&s, d.prefix + i + 1);
            }
            assert_eq!(capture.len(), CONTINUATION + 1);
            check_session(&s, d.prefix + CONTINUATION);
            out.emit(
                json!({"event":"trajectory","case":case,"arm":label,"final_position":s.position(),
                "forward_calls":CONTINUATION,"captured_rows":capture.len(),"timed":false}),
            );
            trajectories.push(capture);
        }
        // Sessions/trajectories complete before any full-vocabulary comparison.
        bitwise(&trajectories[0], &trajectories[1], vocab);
        let mut hash = Sha256::new();
        for row in &trajectories[0] {
            hash.update(bytemuck::cast_slice(row));
        }
        out.emit(
            json!({"event":"bitwise_document","case":case,"stratum":d.stratum,"prefix":d.prefix,
            "compared_rows":CONTINUATION+1,"vocab":vocab,"all_finite":true,"all_bits_equal":true,
            "trajectory_sha256":format!("{:x}",hash.finalize()),"timed":false}),
        );
    }
    assert_plain();
    assert!(
        gguf.revalidate_retained_shard_stamps().unwrap() == stamps,
        "retained shard stamps changed"
    );
    out.emit(json!({"event":"complete","execution_complete":true,"documents":3,"arms":2,
        "sessions":8,"census_prefills":2,"teacher_forced_steps_per_arm_document":64,"compared_full_vocab_rows":195,
        "all_finite":true,"all_bits_equal":true,"same_16mib_backing_both_arms":true,
        "retained_shard_stamps_unchanged":true,"observers_clear":true,"performance_verdict":null}));
    out.sync();
}

use super::*;

#[test]
fn full_readout_workspace_plan_is_stable_across_layer_reuse() {
    let plan = full_readout_workspace_allocation_bytes(7, 32, 101).unwrap();
    assert_eq!(plan.len(), 9);
    assert_eq!(plan[0], 32 * 32 * 2);
    assert_eq!(plan[1..4], [7 * 32 * 4; 3]);
    assert_eq!(plan[4], 7 * 101 * 4);
    assert_eq!(plan[5..], [7 * MPS_FULL_READOUT_TOP_K * 4; 4]);

    for _source_layer in 0..32 {
        assert_eq!(
            full_readout_workspace_allocation_bytes(7, 32, 101).unwrap(),
            plan,
            "a layer read must fit the original persistent allocation plan"
        );
    }
}

#[test]
fn reusable_cpu_exact_top_k_matches_pre_workspace_reference() {
    let logits = [-3.0, 8.0, 8.0, 1.25, -0.0, 0.0, 19.0, 7.5, 19.0, -11.0];
    let actual = exact_vocabulary_top_k(&logits, 6).unwrap();
    let mut reference = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logit)| WorkspaceLensVocabularyScore {
            token_id: token_id as u32,
            logit,
        })
        .collect::<Vec<_>>();
    reference.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    reference.truncate(6);
    assert_eq!(actual, reference);
}

#[test]
fn reusable_packed_selection_matches_pre_workspace_two_pass_reference() {
    let token_ids = [71, 72];
    let mut first_ids = vec![0; token_ids.len() * MPS_FULL_READOUT_TOP_K];
    let mut second_ids = vec![0; token_ids.len() * MPS_FULL_READOUT_TOP_K];
    let mut first_values = vec![0.0; first_ids.len()];
    let mut second_values = vec![0.0; second_ids.len()];
    for row in 0..token_ids.len() {
        for slot in 0..MPS_FULL_READOUT_TOP_K {
            let index = row * MPS_FULL_READOUT_TOP_K + slot;
            first_ids[index] = slot as i32;
            second_ids[index] = (MPS_FULL_READOUT_TOP_K + slot) as i32;
            first_values[index] = (row * 100 + slot) as f32;
            second_values[index] = (row * 100 + 50 - slot) as f32;
        }
    }
    let actual = build_packed_vocabulary_positions(
        &token_ids,
        9,
        25,
        64,
        &first_ids,
        &first_values,
        &second_ids,
        &second_values,
    )
    .unwrap();

    for (row, position) in actual.iter().enumerate() {
        let base = row * MPS_FULL_READOUT_TOP_K;
        let mut reference = (0..MPS_FULL_READOUT_TOP_K)
            .flat_map(|slot| {
                [
                    WorkspaceLensVocabularyScore {
                        token_id: first_ids[base + slot] as u32,
                        logit: first_values[base + slot],
                    },
                    WorkspaceLensVocabularyScore {
                        token_id: second_ids[base + slot] as u32,
                        logit: second_values[base + slot],
                    },
                ]
            })
            .collect::<Vec<_>>();
        reference.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        reference.truncate(25);
        assert_eq!(position.scores, reference);
        assert_eq!(position.source_position, 9 + row);
    }
}

#[test]
fn full_readout_transport_validation_requires_exact_matrix_size() {
    let finite_word = half::f16::from_f32(0.5).to_bits().to_le_bytes();
    let transport = finite_word.repeat(4);
    validate_full_readout_transport_size(&transport, 2).unwrap();

    assert!(matches!(
        validate_full_readout_transport_size(&transport[..6], 2).unwrap_err(),
        WorkspaceLensError::InvalidFullReadoutTransportSize {
            got: 6,
            expected: 8
        }
    ));
}

#[test]
fn packed_capture_layers_require_nonempty_unique_caller_order() {
    validate_packed_capture_layers(8, &[5, 1, 7]).unwrap();
    assert!(matches!(
        validate_packed_capture_layers(8, &[]).unwrap_err(),
        WorkspaceLensError::EmptyWorkspaceSourceLayers
    ));
    assert!(matches!(
        validate_packed_capture_layers(8, &[5, 1, 5]).unwrap_err(),
        WorkspaceLensError::DuplicatePackedCaptureLayer { layer: 5 }
    ));
}

#[test]
fn packed_transported_vector_positions_preserve_caller_order() {
    assert_eq!(
        validate_packed_transported_vector_positions(41, 3, &[43, 41, 42]).unwrap(),
        vec![2, 0, 1]
    );
    assert_eq!(
        validate_packed_transported_vector_positions(41, 3, &[]).unwrap(),
        Vec::<usize>::new()
    );
}

#[test]
fn packed_transported_vector_positions_reject_out_of_range_values() {
    assert!(matches!(
        validate_packed_transported_vector_positions(41, 3, &[40]).unwrap_err(),
        WorkspaceLensError::PackedTransportedVectorPositionOutOfRange {
            source_position: 40,
            start_position: 41,
            end_position: 44,
        }
    ));
    assert!(matches!(
        validate_packed_transported_vector_positions(41, 3, &[44]).unwrap_err(),
        WorkspaceLensError::PackedTransportedVectorPositionOutOfRange {
            source_position: 44,
            start_position: 41,
            end_position: 44,
        }
    ));
}

#[test]
fn packed_transported_vector_positions_reject_duplicates() {
    assert!(matches!(
        validate_packed_transported_vector_positions(41, 3, &[42, 41, 42]).unwrap_err(),
        WorkspaceLensError::DuplicatePackedTransportedVectorPosition {
            source_position: 42
        }
    ));
}

#[test]
fn packed_vocabulary_positions_preserve_absolute_next_position_semantics() {
    let mut first_ids = vec![0i32; 2 * MPS_FULL_READOUT_TOP_K];
    let mut first_values = vec![-10.0f32; 2 * MPS_FULL_READOUT_TOP_K];
    let mut second_ids = vec![0i32; 2 * MPS_FULL_READOUT_TOP_K];
    let second_values = vec![-20.0f32; 2 * MPS_FULL_READOUT_TOP_K];
    for row in 0..2 {
        for column in 0..MPS_FULL_READOUT_TOP_K {
            let index = row * MPS_FULL_READOUT_TOP_K + column;
            first_ids[index] = column as i32;
            second_ids[index] = (MPS_FULL_READOUT_TOP_K + column) as i32;
        }
    }
    first_ids.swap(0, 9);
    first_ids.swap(1, 4);
    first_values[0] = 3.5;
    first_values[1] = 2.25;
    let second_row = MPS_FULL_READOUT_TOP_K;
    first_ids.swap(second_row, second_row + 7);
    first_ids.swap(second_row + 1, second_row + 3);
    first_values[second_row] = 4.0;
    first_values[second_row + 1] = 1.5;

    let positions = build_packed_vocabulary_positions(
        &[101, 102],
        41,
        2,
        128,
        &first_ids,
        &first_values,
        &second_ids,
        &second_values,
    )
    .unwrap();
    assert_eq!(positions.len(), 2);
    assert_eq!(positions[0].source_position, 41);
    assert_eq!(positions[0].source_token_id, 101);
    assert_eq!(positions[0].predicts_position, 42);
    assert_eq!(positions[0].scores[0].token_id, 9);
    assert_eq!(positions[0].scores[1].logit, 2.25);
    assert_eq!(positions[1].source_position, 42);
    assert_eq!(positions[1].source_token_id, 102);
    assert_eq!(positions[1].predicts_position, 43);
    assert_eq!(positions[1].scores[0].token_id, 7);
    assert_eq!(positions[1].scores[1].logit, 1.5);
}

#[test]
#[ignore = "requires a real dense model, Metal, and the full F16 transport payload"]
fn reusable_full_readout_matches_legacy_real_model() {
    use crate::runtime::{LoadedModelConfig, ModelLoadIntent, Runtime, SequenceConfig};
    use std::io::{Read, Seek, SeekFrom};

    let model_path = std::env::var("QWEN_WORKSPACE_LENS_MODEL")
        .or_else(|_| std::env::var("QWEN_RESEARCH_MODEL"))
        .expect("QWEN_WORKSPACE_LENS_MODEL");
    let payload_path = std::env::var("QWEN_WORKSPACE_LENS_TRANSPORT_PAYLOAD")
        .or_else(|_| std::env::var("QWEN_RESEARCH_TRANSPORT_PAYLOAD"))
        .expect("QWEN_WORKSPACE_LENS_TRANSPORT_PAYLOAD");
    let source_layer: u32 = std::env::var("QWEN_WORKSPACE_LENS_SOURCE_LAYER")
        .or_else(|_| std::env::var("QWEN_RESEARCH_SOURCE_LAYER"))
        .expect("QWEN_WORKSPACE_LENS_SOURCE_LAYER")
        .parse()
        .expect("numeric source layer");

    let runtime = Runtime::metal().expect("initialize Metal runtime");
    let loaded = runtime
        .load_model_with_intent(
            model_path,
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )
        .expect("load real model");
    let arch = loaded.arch();
    let matrix_bytes = checked_product(
        checked_product(arch.hidden_size as usize, arch.hidden_size as usize).unwrap(),
        2,
    )
    .unwrap();
    let mut payload = std::fs::File::open(payload_path).expect("open transport payload");
    let other_layer = if source_layer == 0 { 1 } else { 0 };
    let mut read_transport = |layer: u32| {
        let mut transport = vec![0u8; matrix_bytes];
        payload
            .seek(SeekFrom::Start(layer as u64 * matrix_bytes as u64))
            .expect("seek source-layer matrix");
        payload
            .read_exact(&mut transport)
            .expect("read source-layer matrix");
        transport
    };
    let transport = read_transport(source_layer);
    let other_transport = read_transport(other_layer);

    let token_ids = (1..=MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS as i32).collect::<Vec<_>>();
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(token_ids.len()))
        .expect("create sequence");
    let mut workspace_lens = loaded
        .workspace_lens_session(&mut sequence)
        .expect("open workspace-lens session");
    let capture = workspace_lens
        .forward_packed_post_block_capture(&token_ids, &[other_layer, source_layer])
        .expect("packed post-block capture");
    let packed = workspace_lens
        .apply_packed_capture_f16_transport_topk_with_vectors(
            &capture,
            source_layer,
            &transport,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        )
        .expect("packed full-vocabulary readout");
    let other_packed = workspace_lens
        .apply_packed_capture_f16_transport_topk_with_vectors(
            &capture,
            other_layer,
            &other_transport,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        )
        .expect("legacy packed other-layer readout");
    let mut reusable = workspace_lens
        .full_readout_workspace(token_ids.len())
        .expect("allocate reusable full-readout workspace");
    assert!(matches!(
        reusable.apply_packed_capture_bound_f16_transport_topk_with_vectors(
            &capture,
            source_layer,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        ),
        Err(WorkspaceLensError::FullReadoutTransportNotBound)
    ));
    reusable
        .bind_f16_transport(&transport)
        .expect("bind reusable source-layer transport");
    let reusable_bound_packed = reusable
        .apply_packed_capture_bound_f16_transport_topk_with_vectors(
            &capture,
            source_layer,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        )
        .expect("bound reusable packed source-layer readout");
    let reusable_packed = reusable
        .apply_packed_capture_f16_transport_topk_with_vectors(
            &capture,
            source_layer,
            &transport,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        )
        .expect("reusable packed source-layer readout");
    let reusable_other_packed = reusable
        .apply_packed_capture_f16_transport_topk_with_vectors(
            &capture,
            other_layer,
            &other_transport,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        )
        .expect("reusable packed other-layer readout");
    let reusable_packed_repeat = reusable
        .apply_packed_capture_f16_transport_topk_with_vectors(
            &capture,
            source_layer,
            &transport,
            MAX_FULL_READOUT_TOP_K,
            &[0, 16, 127],
        )
        .expect("repeated reusable packed source-layer readout");
    let assert_exact_packed =
        |legacy: &WorkspaceLensPackedFullVocabularyReadout,
         candidate: &WorkspaceLensPackedFullVocabularyReadout| {
            assert_eq!(candidate.source_layer, legacy.source_layer);
            assert_eq!(candidate.start_position, legacy.start_position);
            assert_eq!(candidate.position_count, legacy.position_count);
            assert_eq!(candidate.top_k, legacy.top_k);
            assert_eq!(
                candidate.packed_prefill_gpu_ms,
                legacy.packed_prefill_gpu_ms
            );
            assert_eq!(
                candidate.packed_prefill_wall_ms,
                legacy.packed_prefill_wall_ms
            );
            assert_eq!(candidate.positions, legacy.positions);
            assert_eq!(candidate.transported_vectors, legacy.transported_vectors);
        };
    assert_exact_packed(&packed, &reusable_packed);
    assert_exact_packed(&packed, &reusable_bound_packed);
    assert_exact_packed(&other_packed, &reusable_other_packed);
    assert_exact_packed(&packed, &reusable_packed_repeat);
    assert_eq!(packed.transported_vectors.len(), 3);
    for (&source_position, vector) in [0, 16, 127].iter().zip(&packed.transported_vectors) {
        assert_eq!(vector.source_position, source_position);
        assert_eq!(vector.source_token_id, token_ids[source_position]);
        assert_eq!(vector.predicts_position, source_position + 1);
        assert_eq!(vector.values.len(), arch.hidden_size as usize);
        assert!(vector.values.iter().all(|value| value.is_finite()));
    }
    for capture_row in [0, 15, 16, 17, 127] {
        let captured_residual = capture
            .row_for_test(capture_row, source_layer)
            .expect("read exact packed capture row");
        let serial = workspace_lens
            .apply_f16_transport_topk_with_vector(
                &transport,
                &captured_residual,
                MAX_FULL_READOUT_TOP_K,
            )
            .expect("serial full-vocabulary readout");
        let reusable_serial = reusable
            .apply_row_f16_transport_topk_with_vector(
                &transport,
                &captured_residual,
                MAX_FULL_READOUT_TOP_K,
            )
            .expect("reusable serial full-vocabulary readout");
        assert_eq!(reusable_serial, serial);
    }
    let source_residual = capture
        .row_for_test(16, source_layer)
        .expect("read repeated source-layer row");
    let other_residual = capture
        .row_for_test(16, other_layer)
        .expect("read other-layer row");
    let legacy_other_serial = workspace_lens
        .apply_f16_transport_topk_with_vector(
            &other_transport,
            &other_residual,
            MAX_FULL_READOUT_TOP_K,
        )
        .expect("legacy other-layer serial readout");
    let reusable_other_serial = reusable
        .apply_row_f16_transport_topk_with_vector(
            &other_transport,
            &other_residual,
            MAX_FULL_READOUT_TOP_K,
        )
        .expect("reusable other-layer serial readout");
    assert_eq!(reusable_other_serial, legacy_other_serial);
    let legacy_source_repeat = workspace_lens
        .apply_f16_transport_topk_with_vector(&transport, &source_residual, MAX_FULL_READOUT_TOP_K)
        .expect("legacy repeated source-layer serial readout");
    let reusable_source_repeat = reusable
        .apply_row_f16_transport_topk_with_vector(
            &transport,
            &source_residual,
            MAX_FULL_READOUT_TOP_K,
        )
        .expect("reusable repeated source-layer serial readout");
    assert_eq!(reusable_source_repeat, legacy_source_repeat);
    eprintln!(
        "packed timings positions={} prefill_gpu_ms={:.3} prefill_wall_ms={:.3} readout_gpu_ms={:.3} readout_wall_ms={:.3}",
        packed.position_count,
        packed.packed_prefill_gpu_ms,
        packed.packed_prefill_wall_ms,
        packed.readout_gpu_ms,
        packed.readout_wall_ms,
    );
}

fn f32_tensor(context: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(context, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
}

fn cpu_dense_ffn_vjp(
    residual: &[f32],
    norm: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    grad_output: &[f32],
    n_query: usize,
    intermediate_size: usize,
    rule: DenseFfnVjpRule,
) -> Vec<f32> {
    let hidden_size = residual.len();
    let sumsq: f32 = residual.iter().map(|value| value * value).sum();
    let scale = (sumsq / hidden_size as f32 + RMS_EPS).sqrt().recip();
    let normalized: Vec<f32> = residual
        .iter()
        .zip(norm)
        .map(|(value, weight)| value * scale * weight)
        .collect();
    let project = |weight: &[f32], n_in: usize, n_out: usize, input: &[f32]| {
        (0..n_out)
            .map(|output| {
                (0..n_in)
                    .map(|input_index| weight[output * n_in + input_index] * input[input_index])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>()
    };
    let gate = project(gate_weight, hidden_size, intermediate_size, &normalized);
    let up = project(up_weight, hidden_size, intermediate_size, &normalized);
    let mut result = vec![0.0f32; n_query * hidden_size];
    for query in 0..n_query {
        let incoming = &grad_output[query * hidden_size..(query + 1) * hidden_size];
        let mut grad_inner = vec![0.0f32; intermediate_size];
        for input in 0..intermediate_size {
            grad_inner[input] = (0..hidden_size)
                .map(|output| down_weight[output * intermediate_size + input] * incoming[output])
                .sum();
        }
        let mut grad_gate = vec![0.0f32; intermediate_size];
        let mut grad_up = vec![0.0f32; intermediate_size];
        for index in 0..intermediate_size {
            let sigmoid = 1.0 / (1.0 + (-gate[index]).exp());
            let silu = gate[index] * sigmoid;
            match rule {
                DenseFfnVjpRule::Jacobian => {
                    let derivative = sigmoid * (1.0 + gate[index] * (1.0 - sigmoid));
                    grad_gate[index] = grad_inner[index] * up[index] * derivative;
                    grad_up[index] = grad_inner[index] * silu;
                }
                DenseFfnVjpRule::Relp => {
                    grad_gate[index] = 0.5 * grad_inner[index] * up[index] * sigmoid;
                    grad_up[index] = 0.5 * grad_inner[index] * silu;
                }
            }
        }
        let mut grad_norm = vec![0.0f32; hidden_size];
        for input in 0..hidden_size {
            grad_norm[input] = (0..intermediate_size)
                .map(|output| {
                    gate_weight[output * hidden_size + input] * grad_gate[output]
                        + up_weight[output * hidden_size + input] * grad_up[output]
                })
                .sum();
        }
        let dot: f32 = (0..hidden_size)
            .map(|index| residual[index] * grad_norm[index] * norm[index])
            .sum();
        let correction = dot * scale * scale * scale / hidden_size as f32;
        for index in 0..hidden_size {
            let direct = grad_norm[index] * norm[index] * scale;
            let ffn_branch = match rule {
                DenseFfnVjpRule::Jacobian => direct - residual[index] * correction,
                DenseFfnVjpRule::Relp => direct,
            };
            result[query * hidden_size + index] = incoming[index] + ffn_branch;
        }
    }
    result
}

#[test]
fn capture_layer_validation_preserves_unsorted_duplicates() {
    let layers = [7, 1, 7, 0];
    validate_capture_layers(8, &layers).unwrap();
    assert_eq!(layers, [7, 1, 7, 0]);
}

#[test]
fn capture_layer_validation_rejects_first_out_of_range_layer() {
    let error = validate_capture_layers(8, &[1, 8, 9]).unwrap_err();
    assert!(matches!(
        error,
        WorkspaceLensError::InvalidLayer {
            layer: 8,
            n_layers: 8
        }
    ));
}

#[test]
fn workspace_capture_transposes_tokens_into_layer_major_banks() {
    const TOKENS: usize = 3;
    const LAYERS: usize = 2;
    const HIDDEN: usize = 2;
    let token_captures = [
        [0.0, 1.0, 10.0, 11.0],
        [2.0, 3.0, 12.0, 13.0],
        [4.0, 5.0, 14.0, 15.0],
    ];
    let mut bank = vec![0.0; TOKENS * LAYERS * HIDDEN];
    for (token, capture) in token_captures.iter().enumerate() {
        copy_workspace_token_capture(&mut bank, capture, token, TOKENS, LAYERS, HIDDEN).unwrap();
    }
    assert_eq!(
        bank,
        [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
        ]
    );
}

#[test]
fn workspace_batch_accessors_preserve_source_then_query_layout() {
    let batch = WorkspaceLensVjpBatch {
        target_layer: 3,
        source_layers: vec![0, 2],
        n_query: 3,
        n_tokens: 2,
        hidden_size: 2,
        values: (0..24).map(|value| value as f32).collect(),
        diagnostics: Vec::new(),
    };
    assert_eq!(
        batch.source_values(1).unwrap(),
        (12..24).map(|value| value as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        batch.source_query_values(1, 2).unwrap(),
        [20.0, 21.0, 22.0, 23.0]
    );
    assert!(batch.source_values(2).is_none());
    assert!(batch.source_query_values(0, 3).is_none());
}

#[test]
fn workspace_covectors_build_target_bank_and_reduce_kqh_layout() {
    let covectors = [1.0f32, 2.0, 3.0, 4.0];
    let target_bank = build_workspace_target_bank(&covectors, 2, 4, 2, 1..3).unwrap();
    assert_eq!(
        target_bank,
        [
            0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 3.0, 4.0, 0.0, 0.0,
        ]
    );

    let source_zero = (0..16).map(|value| value as f32).collect::<Vec<_>>();
    let source_one = (100..116).map(|value| value as f32).collect::<Vec<_>>();
    let trajectories = [source_zero, source_one].concat();
    let mut values = vec![0.0f32; 2 * 2 * 2];
    reduce_workspace_vjp_readouts(&trajectories, 2, 2, 4, 2, 1..3, &mut values, 2, 0).unwrap();
    assert_eq!(values, [3.0, 4.0, 11.0, 12.0, 103.0, 104.0, 111.0, 112.0]);

    let readouts = WorkspaceLensReadouts {
        target_layer: 4,
        source_layers: vec![0, 2],
        n_query: 2,
        n_tokens: 4,
        n_valid_positions: 2,
        hidden_size: 2,
        values,
        diagnostics: Vec::new(),
    };
    assert_eq!(
        readouts.source_values(1).unwrap(),
        [103.0, 104.0, 111.0, 112.0]
    );
    assert_eq!(readouts.query_values(1, 0).unwrap(), [103.0, 104.0]);
    assert!(readouts.source_values(2).is_none());
    assert!(readouts.query_values(0, 2).is_none());
}

#[test]
fn one_hot_covectors_construct_the_same_targets_as_basis_rows() {
    const TOKENS: usize = 5;
    const HIDDEN: usize = 4;
    let rows = [1usize, 3];
    let mut covectors = vec![0.0f32; rows.len() * HIDDEN];
    for (query, &row) in rows.iter().enumerate() {
        covectors[query * HIDDEN + row] = 1.0;
    }
    let arbitrary =
        build_workspace_target_bank(&covectors, rows.len(), TOKENS, HIDDEN, 1..4).unwrap();
    let mut basis = vec![0.0f32; arbitrary.len()];
    for (query, &row) in rows.iter().enumerate() {
        for position in 1..4 {
            basis[query * TOKENS * HIDDEN + position * HIDDEN + row] = 1.0;
        }
    }
    assert_eq!(arbitrary, basis);
}

#[test]
fn readout_helpers_reject_bad_sizes_non_finite_values_and_token_ids() {
    assert!(matches!(
        build_workspace_target_bank(&[1.0, 2.0, 3.0], 1, 3, 2, 0..2).unwrap_err(),
        WorkspaceLensError::WorkspaceTargetCovectorSize { .. }
    ));
    assert!(matches!(
        build_workspace_target_bank(&[1.0, f32::NAN], 1, 3, 2, 0..2).unwrap_err(),
        WorkspaceLensError::NonFiniteWorkspaceTargetCovector { index: 1 }
    ));
    assert!(matches!(
        validate_selected_token_ids(&[], 10).unwrap_err(),
        WorkspaceLensError::EmptyTokenReadoutSelection
    ));
    assert!(matches!(
        validate_selected_token_ids(&[2, 2], 10).unwrap_err(),
        WorkspaceLensError::DuplicateTokenReadoutId { token_id: 2 }
    ));
    assert!(matches!(
        validate_selected_token_ids(&[10], 10).unwrap_err(),
        WorkspaceLensError::TokenReadoutIdOutOfRange { token_id: 10, .. }
    ));
    assert!(matches!(
        validate_selected_token_request_size(11, 10, 4).unwrap_err(),
        WorkspaceLensError::TokenReadoutCountExceedsVocabulary {
            got: 11,
            vocab_size: 10
        }
    ));
    assert!(matches!(
        validate_selected_token_request_size(70_000_000, u32::MAX, 1).unwrap_err(),
        WorkspaceLensError::WorkspaceLensResultByteBudgetExceeded { .. }
    ));
}

#[test]
fn f16_transport_projection_capacity_is_derived_from_live_bytes() {
    let hidden_size = 5_120;
    let transport_bytes = hidden_size * hidden_size * std::mem::size_of::<half::f16>();
    let capacity = f16_transport_readout_query_capacity(hidden_size, transport_bytes, 0).unwrap();
    assert!(capacity > 32);
    assert!(
        f16_transport_readout_peak_bytes(hidden_size, transport_bytes, capacity).unwrap()
            <= MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES
    );
    assert!(
        f16_transport_readout_peak_bytes(hidden_size, transport_bytes, capacity + 1).unwrap()
            > MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES
    );
    let additional_live_bytes = 144 * 1024 * 1024;
    let reserved_capacity =
        f16_transport_readout_query_capacity(hidden_size, transport_bytes, additional_live_bytes)
            .unwrap();
    assert!(reserved_capacity > 32 && reserved_capacity < capacity);
    assert!(
        f16_transport_readout_peak_bytes(hidden_size, transport_bytes, reserved_capacity,).unwrap()
            + additional_live_bytes
            <= MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES
    );
}

#[test]
fn token_readout_gamma_multiplication_is_rowwise_and_finite() {
    let mut values = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
    multiply_token_readout_gamma_in_place(&mut values, &[0.5, 2.0, -1.0], 2, 3).unwrap();
    assert_eq!(values, [0.5, 4.0, -3.0, -0.5, -4.0, 3.0]);
    let mut bad_size = [1.0, 2.0];
    assert!(matches!(
        multiply_token_readout_gamma_in_place(&mut bad_size, &[1.0], 1, 2).unwrap_err(),
        WorkspaceLensError::ActivationSize { .. }
    ));
    let mut non_finite = [f32::INFINITY];
    assert!(matches!(
        multiply_token_readout_gamma_in_place(&mut non_finite, &[1.0], 1, 1).unwrap_err(),
        WorkspaceLensError::NonFiniteTokenReadoutData { .. }
    ));
}

#[test]
fn workspace_reference_positions_exclude_prefix_and_final_token() {
    assert_eq!(workspace_valid_position_range(8, 4).unwrap(), 4..7);
    assert!(matches!(
        workspace_valid_position_range(5, 4).unwrap_err(),
        WorkspaceLensError::WorkspaceNoValidPositions {
            n_tokens: 5,
            skip_first: 4
        }
    ));

    let source = [0.0f32, 10.0, 2.0, 12.0, 4.0, 14.0, 6.0, 16.0, 100.0, 200.0];
    let mut row = [0.0f32; 2];
    reduce_workspace_source_positions(&source, 5, 2, 1..4, &mut row).unwrap();
    assert_eq!(row, [4.0, 14.0]);
}

#[test]
fn workspace_row_diagnostics_merge_by_schedule_and_maximum() {
    let mut aggregate = vec![WorkspaceLensReplayDiagnostic {
        layer: 3,
        kind: WorkspaceLensBlockKind::Attention,
        residual_replay_max_abs_error: 0.1,
    }];
    merge_workspace_diagnostics(
        &mut aggregate,
        &[WorkspaceLensReplayDiagnostic {
            layer: 3,
            kind: WorkspaceLensBlockKind::Attention,
            residual_replay_max_abs_error: 0.2,
        }],
    )
    .unwrap();
    assert_eq!(aggregate[0].residual_replay_max_abs_error, 0.2);
    assert!(matches!(
        merge_workspace_diagnostics(
            &mut aggregate,
            &[WorkspaceLensReplayDiagnostic {
                layer: 2,
                kind: WorkspaceLensBlockKind::Gdn,
                residual_replay_max_abs_error: 0.0,
            }],
        )
        .unwrap_err(),
        WorkspaceLensError::WorkspaceDiagnosticScheduleMismatch
    ));
    assert!(matches!(
        merge_workspace_diagnostics(
            &mut aggregate,
            &[WorkspaceLensReplayDiagnostic {
                layer: 3,
                kind: WorkspaceLensBlockKind::Attention,
                residual_replay_max_abs_error: f32::NAN,
            }],
        )
        .unwrap_err(),
        WorkspaceLensError::NonFiniteWorkspaceReplayDiagnostic { layer: 3 }
    ));
}

#[test]
fn workspace_readout_budget_and_non_finite_reductions_fail_closed() {
    assert!(matches!(
        enforce_workspace_lens_byte_budget(
            "test result",
            MAX_WORKSPACE_LENS_OWNED_RESULT_BYTES + 1
        )
        .unwrap_err(),
        WorkspaceLensError::WorkspaceLensResultByteBudgetExceeded { .. }
    ));

    let mut destination = [0.0f32; 1];
    assert!(matches!(
        reduce_workspace_source_positions(&[0.0, f32::NAN, 1.0], 3, 1, 0..2, &mut destination,)
            .unwrap_err(),
        WorkspaceLensError::NonFiniteWorkspaceVjpTrajectory { index: 1 }
    ));
    assert!(matches!(
        reduce_workspace_source_positions(
            &[f32::MAX, f32::MAX, 0.0],
            3,
            1,
            0..2,
            &mut destination,
        )
        .unwrap_err(),
        WorkspaceLensError::NonFiniteWorkspaceReduction {
            stage: "sum",
            index: 0
        }
    ));
    assert!(matches!(
        validate_workspace_vjp_finite(
            &[0.0, f32::INFINITY],
            &[WorkspaceLensReplayDiagnostic {
                layer: 2,
                kind: WorkspaceLensBlockKind::Gdn,
                residual_replay_max_abs_error: 0.0,
            }],
        )
        .unwrap_err(),
        WorkspaceLensError::NonFiniteWorkspaceVjpTrajectory { index: 1 }
    ));
}

#[test]
fn workspace_vjp_crosses_sources_in_caller_order_without_reversing_them() {
    let source_layers = [0, 3, 1, 3];
    let target_cotangent = [1.0f32, 2.0, 3.0];
    let mut traversed = Vec::new();
    let (values, diagnostics) = compose_workspace_vjp(
        4,
        &source_layers,
        target_cotangent.len(),
        &target_cotangent,
        |layer, gradient| {
            traversed.push(layer);
            Ok((
                gradient.iter().map(|value| value * layer as f32).collect(),
                WorkspaceLensReplayDiagnostic {
                    layer,
                    kind: if layer == 4 {
                        WorkspaceLensBlockKind::Attention
                    } else {
                        WorkspaceLensBlockKind::Gdn
                    },
                    residual_replay_max_abs_error: layer as f32 * 1e-6,
                },
            ))
        },
    )
    .unwrap();
    assert_eq!(traversed, [4, 3, 2, 1]);
    assert_eq!(
        diagnostics
            .iter()
            .map(|diagnostic| diagnostic.layer)
            .collect::<Vec<_>>(),
        traversed
    );
    assert_eq!(&values[0..3], &[24.0, 48.0, 72.0]);
    assert_eq!(&values[3..6], &[4.0, 8.0, 12.0]);
    assert_eq!(&values[6..9], &[24.0, 48.0, 72.0]);
    assert_eq!(&values[9..12], &[4.0, 8.0, 12.0]);

    let error = compose_workspace_vjp(
        4,
        &[4],
        target_cotangent.len(),
        &target_cotangent,
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        WorkspaceLensError::WorkspaceSourceNotBeforeTarget {
            source_layer: 4,
            target_layer: 4
        }
    ));
}

#[test]
fn replay_diagnostics_fail_closed_on_non_finite_values() {
    assert_eq!(finite_abs_difference(f32::NAN, 0.0), f32::INFINITY);
    assert_eq!(finite_abs_difference(0.0, f32::INFINITY), f32::INFINITY);
    assert_eq!(
        max_abs_difference(&[0.0, f32::NAN], &[0.0, 0.0]),
        f32::INFINITY
    );
}

#[test]
fn causal_gated_attention_and_rope_vjps_match_finite_differences() {
    const TOKENS: usize = 4;
    const N_Q: usize = 4;
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 8;
    const N_ROT: usize = 4;
    let geometry = AttnGeometry {
        hidden_size: 13,
        n_q_heads: N_Q,
        n_kv_heads: N_KV,
        head_dim: HEAD_DIM,
        n_rot: N_ROT,
        q_elements: N_Q * HEAD_DIM,
        q_full_elements: 2 * N_Q * HEAD_DIM,
        kv_elements: N_KV * HEAD_DIM,
        rope_theta: 10_000.0,
    };
    let values = |len: usize, stride: usize, scale: f32, offset: f32| {
        (0..len)
            .map(|index| ((index * stride + 3) % 47) as f32 * scale + offset)
            .collect::<Vec<_>>()
    };
    let q = values(TOKENS * geometry.q_elements, 5, 0.013, -0.27);
    let k = values(TOKENS * geometry.kv_elements, 7, 0.011, -0.23);
    let v = values(TOKENS * geometry.kv_elements, 11, 0.017, -0.35);
    let gate = values(TOKENS * geometry.q_elements, 13, 0.029, -0.61);
    let grad = values(TOKENS * geometry.q_elements, 17, 0.019, -0.41);
    let actual =
        cpu_causal_gated_attention_vjp(&q, &k, &v, &gate, &grad, TOKENS, geometry).unwrap();
    let objective = |q: &[f32], k: &[f32], v: &[f32], gate: &[f32]| {
        cpu_causal_gated_attention_forward(q, k, v, gate, TOKENS, geometry)
            .unwrap()
            .gated_output
            .iter()
            .zip(&grad)
            .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
            .sum::<f64>()
    };
    let epsilon = 2e-3f32;
    let finite_difference = |values: &[f32], index: usize, evaluate: &dyn Fn(&[f32]) -> f64| {
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        (evaluate(&plus) - evaluate(&minus)) / (2.0 * f64::from(epsilon))
    };
    for &index in &[
        2 * geometry.q_elements,
        3 * geometry.q_elements + HEAD_DIM - 1,
    ] {
        let fd = finite_difference(&q, index, &|candidate| objective(candidate, &k, &v, &gate));
        assert!((fd - f64::from(actual.grad_q[index])).abs() < 2e-3);
    }
    for &index in &[0usize, geometry.kv_elements + 3, k.len() - 1] {
        let fd = finite_difference(&k, index, &|candidate| objective(&q, candidate, &v, &gate));
        assert!((fd - f64::from(actual.grad_k[index])).abs() < 2e-3);
        let fd = finite_difference(&v, index, &|candidate| objective(&q, &k, candidate, &gate));
        assert!((fd - f64::from(actual.grad_v[index])).abs() < 2e-3);
    }
    for &index in &[0usize, 2 * geometry.q_elements + 5, gate.len() - 1] {
        let fd = finite_difference(&gate, index, &|candidate| objective(&q, &k, &v, candidate));
        assert!((fd - f64::from(actual.grad_gate[index])).abs() < 2e-3);
    }
    let direction = |len: usize, stride: usize| {
        (0..len)
            .map(|index| ((index * stride + 1) % 31) as f32 * 0.0013 - 0.019)
            .collect::<Vec<_>>()
    };
    let dq = direction(q.len(), 19);
    let dk = direction(k.len(), 23);
    let dv = direction(v.len(), 29);
    let dg = direction(gate.len(), 31);
    let shift = |base: &[f32], tangent: &[f32], amount: f32| {
        base.iter()
            .zip(tangent)
            .map(|(&base, &tangent)| base + amount * tangent)
            .collect::<Vec<_>>()
    };
    let plus = objective(
        &shift(&q, &dq, epsilon),
        &shift(&k, &dk, epsilon),
        &shift(&v, &dv, epsilon),
        &shift(&gate, &dg, epsilon),
    );
    let minus = objective(
        &shift(&q, &dq, -epsilon),
        &shift(&k, &dk, -epsilon),
        &shift(&v, &dv, -epsilon),
        &shift(&gate, &dg, -epsilon),
    );
    let forward_directional = (plus - minus) / (2.0 * f64::from(epsilon));
    let inner = |gradient: &[f32], tangent: &[f32]| {
        gradient
            .iter()
            .zip(tangent)
            .map(|(&gradient, &tangent)| f64::from(gradient) * f64::from(tangent))
            .sum::<f64>()
    };
    let reverse_directional = inner(&actual.grad_q, &dq)
        + inner(&actual.grad_k, &dk)
        + inner(&actual.grad_v, &dv)
        + inner(&actual.grad_gate, &dg);
    assert!((forward_directional - reverse_directional).abs() < 2e-3);

    let rope_input = values(TOKENS * N_Q * HEAD_DIM, 37, 0.021, -0.44);
    let rope_grad = values(TOKENS * N_Q * HEAD_DIM, 41, 0.018, -0.39);
    let mut rotated = rope_input.clone();
    rope_neox_rows_in_place(
        &mut rotated,
        TOKENS,
        N_Q,
        HEAD_DIM,
        N_ROT,
        7,
        geometry.rope_theta,
        false,
    )
    .unwrap();
    let mut transposed = rope_grad.clone();
    rope_neox_rows_in_place(
        &mut transposed,
        TOKENS,
        N_Q,
        HEAD_DIM,
        N_ROT,
        7,
        geometry.rope_theta,
        true,
    )
    .unwrap();
    let left = inner(&rotated, &rope_grad);
    let right = inner(&rope_input, &transposed);
    assert!((left - right).abs() < 2e-5);
    for token in 0..TOKENS {
        for head in 0..N_Q {
            let base = (token * N_Q + head) * HEAD_DIM;
            for index in N_ROT..HEAD_DIM {
                assert_eq!(
                    rotated[base + index].to_bits(),
                    rope_input[base + index].to_bits()
                );
                assert_eq!(
                    transposed[base + index].to_bits(),
                    rope_grad[base + index].to_bits()
                );
            }
        }
    }
}

#[test]
fn hybrid_attention_mixer_vjp_matches_directional_finite_differences() {
    let context = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const TOKENS: usize = 3;
    const HIDDEN: usize = 16;
    const N_Q: usize = 2;
    const N_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    let geometry = AttnGeometry {
        hidden_size: HIDDEN,
        n_q_heads: N_Q,
        n_kv_heads: N_KV,
        head_dim: HEAD_DIM,
        n_rot: 4,
        q_elements: N_Q * HEAD_DIM,
        q_full_elements: 2 * N_Q * HEAD_DIM,
        kv_elements: N_KV * HEAD_DIM,
        rope_theta: 10_000.0,
    };
    let values = |len: usize, stride: usize, scale: f32, offset: f32| {
        (0..len)
            .map(|index| ((index * stride + 3) % 47) as f32 * scale + offset)
            .collect::<Vec<_>>()
    };
    let attn_norm = f32_tensor(
        &context,
        &values(HIDDEN, 5, 0.011, 0.67),
        vec![HIDDEN as u64],
    );
    let q_weight_values = values(HIDDEN * 2 * geometry.q_elements, 7, 0.0017, -0.037);
    let q_weight = f32_tensor(
        &context,
        &q_weight_values,
        vec![HIDDEN as u64, (2 * geometry.q_elements) as u64],
    );
    let k_weight_values = values(HIDDEN * geometry.kv_elements, 11, 0.0021, -0.043);
    let k_weight = f32_tensor(
        &context,
        &k_weight_values,
        vec![HIDDEN as u64, geometry.kv_elements as u64],
    );
    let v_weight_values = values(HIDDEN * geometry.kv_elements, 13, 0.0023, -0.047);
    let v_weight = f32_tensor(
        &context,
        &v_weight_values,
        vec![HIDDEN as u64, geometry.kv_elements as u64],
    );
    let o_weight_values = values(geometry.q_elements * HIDDEN, 17, 0.0019, -0.039);
    let o_weight = f32_tensor(
        &context,
        &o_weight_values,
        vec![geometry.q_elements as u64, HIDDEN as u64],
    );
    let q_norm = f32_tensor(
        &context,
        &values(HEAD_DIM, 19, 0.017, 0.59),
        vec![HEAD_DIM as u64],
    );
    let k_norm = f32_tensor(
        &context,
        &values(HEAD_DIM, 23, 0.019, 0.57),
        vec![HEAD_DIM as u64],
    );
    let weights = AttnMixerWeights {
        attn_norm: &attn_norm,
        q: &q_weight,
        k: &k_weight,
        v: &v_weight,
        o: &o_weight,
        q_norm: &q_norm,
        k_norm: &k_norm,
    };
    let input = values(TOKENS * HIDDEN, 29, 0.009, -0.19);
    let grad = values(TOKENS * HIDDEN, 31, 0.013, -0.27);
    let run = |input: &[f32], grad: &[f32], rule| {
        attn_mixer_replay_vjp_readback(&context, geometry, weights, input, grad, TOKENS, rule)
            .unwrap()
    };
    let actual = run(&input, &grad, AttnBlockVjpRule::Jacobian);
    assert!(actual.grad_input.iter().all(|value| value.is_finite()));
    assert!(actual.grad_input.iter().any(|value| *value != 0.0));
    let objective = |candidate: &[f32]| {
        run(candidate, &grad, AttnBlockVjpRule::Jacobian)
            .mixer_outputs
            .iter()
            .zip(&grad)
            .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
            .sum::<f64>()
    };
    let epsilon = 2e-2f32;
    for index in [0usize, HIDDEN, input.len() - 1] {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let fd = (objective(&plus) - objective(&minus)) / (2.0 * f64::from(epsilon));
        assert!(
            (fd - f64::from(actual.grad_input[index])).abs() < 2e-3,
            "coordinate {index} fd={fd} reverse={}",
            actual.grad_input[index]
        );
    }
    let direction = values(input.len(), 37, 0.0013, -0.021);
    let shift = |amount: f32| {
        input
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value + amount * direction)
            .collect::<Vec<_>>()
    };
    let forward_directional =
        (objective(&shift(epsilon)) - objective(&shift(-epsilon))) / (2.0 * f64::from(epsilon));
    let reverse_directional = actual
        .grad_input
        .iter()
        .zip(&direction)
        .map(|(&gradient, &direction)| f64::from(gradient) * f64::from(direction))
        .sum::<f64>();
    assert!(
        (forward_directional - reverse_directional).abs() < 2e-3,
        "hybrid attention directional mismatch forward={forward_directional} reverse={reverse_directional}"
    );
    let zero = run(&input, &vec![0.0; grad.len()], AttnBlockVjpRule::Jacobian);
    assert!(zero.grad_input.iter().all(|value| *value == 0.0));
    let relp = run(&input, &grad, AttnBlockVjpRule::Relp);
    assert!(
        relp.grad_input
            .iter()
            .zip(&actual.grad_input)
            .any(|(relp, jacobian)| relp.to_bits() != jacobian.to_bits())
    );

    let hidden_elements = TOKENS * HIDDEN;
    for rule in [AttnBlockVjpRule::Jacobian, AttnBlockVjpRule::Relp] {
        for query_batches in [1, 2, 8] {
            let grad_bank: Vec<f32> = (0..query_batches * hidden_elements)
                .map(|index| {
                    let query = index / hidden_elements;
                    let local = index % hidden_elements;
                    ((local * 31 + query * 11 + 5) % 59) as f32 * 0.009 - 0.24
                        + query as f32 * 0.007
                })
                .collect();
            let batched = attn_mixer_replay_vjp_batch_readback(
                &context,
                geometry,
                weights,
                &input,
                &grad_bank,
                TOKENS,
                query_batches,
                rule,
            )
            .unwrap();
            for query in 0..query_batches {
                let start = query * hidden_elements;
                let end = start + hidden_elements;
                let serial = run(&input, &grad_bank[start..end], rule);
                let mixer_error = batched
                    .mixer_outputs
                    .iter()
                    .zip(&serial.mixer_outputs)
                    .map(|(&batched, &serial)| finite_abs_difference(batched, serial))
                    .fold(0.0f32, f32::max);
                let gradient_error = batched.grad_input[start..end]
                    .iter()
                    .zip(&serial.grad_input)
                    .map(|(&batched, &serial)| finite_abs_difference(batched, serial))
                    .fold(0.0f32, f32::max);
                assert!(
                    mixer_error < 1e-6 && gradient_error < 2e-5,
                    "{rule:?} attention batch {query_batches} query {query}: mixer={mixer_error} gradient={gradient_error}"
                );
            }
        }

        const QUERY_BATCHES: usize = 4;
        const ACTIVE_QUERY: usize = 2;
        let mut isolated_grad = vec![0.0f32; QUERY_BATCHES * hidden_elements];
        let active_start = ACTIVE_QUERY * hidden_elements;
        isolated_grad[active_start..active_start + hidden_elements].copy_from_slice(&grad);
        let isolated = attn_mixer_replay_vjp_batch_readback(
            &context,
            geometry,
            weights,
            &input,
            &isolated_grad,
            TOKENS,
            QUERY_BATCHES,
            rule,
        )
        .unwrap();
        let serial = run(&input, &grad, rule);
        for query in 0..QUERY_BATCHES {
            let start = query * hidden_elements;
            let end = start + hidden_elements;
            if query == ACTIVE_QUERY {
                let error = isolated.grad_input[start..end]
                    .iter()
                    .zip(&serial.grad_input)
                    .map(|(&batched, &serial)| finite_abs_difference(batched, serial))
                    .fold(0.0f32, f32::max);
                assert!(error < 2e-5, "{rule:?} isolated attention error {error}");
            } else {
                assert!(
                    isolated.grad_input[start..end]
                        .iter()
                        .all(|value| *value == 0.0),
                    "{rule:?} attention query {query} received another query's cotangent"
                );
            }
        }
    }
}

#[test]
fn gdn_mixer_replay_vjp_matches_directional_finite_differences() {
    let context = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const N_TOKENS: usize = 3;
    const HIDDEN: usize = 32;
    const N_K: usize = 1;
    const N_V: usize = 2;
    const HEAD_DIM: usize = 128;
    let qk_elements = N_K * HEAD_DIM;
    let v_elements = N_V * HEAD_DIM;
    let conv_dim = 2 * qk_elements + v_elements;
    let state_elements = v_elements * HEAD_DIM;
    let conv_state_elements = 3 * conv_dim;
    let geometry = GdnGeometry {
        hidden_size: HIDDEN,
        n_v_heads: N_V,
        n_k_heads: N_K,
        head_dim: HEAD_DIM,
        qk_elements,
        v_elements,
        conv_dim,
        state_elements,
        conv_state_elements,
    };
    let values = |len: usize, stride: usize, scale: f32, offset: f32| {
        (0..len)
            .map(|index| ((index * stride + 3) % 47) as f32 * scale + offset)
            .collect::<Vec<_>>()
    };
    let attn_norm = f32_tensor(
        &context,
        &values(HIDDEN, 5, 0.013, 0.63),
        vec![HIDDEN as u64],
    );
    let qkv_weight_values = values(HIDDEN * conv_dim, 7, 0.0007, -0.015);
    let qkv_weight = f32_tensor(
        &context,
        &qkv_weight_values,
        vec![HIDDEN as u64, conv_dim as u64],
    );
    let z_weight_values = values(HIDDEN * v_elements, 11, 0.0008, -0.017);
    let z_weight = f32_tensor(
        &context,
        &z_weight_values,
        vec![HIDDEN as u64, v_elements as u64],
    );
    let beta_weight_values = values(HIDDEN * N_V, 13, 0.0011, -0.021);
    let beta_weight = f32_tensor(
        &context,
        &beta_weight_values,
        vec![HIDDEN as u64, N_V as u64],
    );
    let alpha_weight_values = values(HIDDEN * N_V, 17, 0.0013, -0.024);
    let alpha_weight = f32_tensor(
        &context,
        &alpha_weight_values,
        vec![HIDDEN as u64, N_V as u64],
    );
    let a_log = f32_tensor(&context, &[-0.08, -0.13], vec![N_V as u64]);
    let dt_bias = f32_tensor(&context, &[0.07, -0.11], vec![N_V as u64]);
    let conv_weight_values = values(4 * conv_dim, 19, 0.0019, -0.041);
    let conv_weight = f32_tensor(&context, &conv_weight_values, vec![4, conv_dim as u64]);
    let internal_norm = f32_tensor(
        &context,
        &values(HEAD_DIM, 23, 0.009, 0.58),
        vec![HEAD_DIM as u64],
    );
    let out_weight_values = values(v_elements * HIDDEN, 29, 0.0009, -0.019);
    let out_weight = f32_tensor(
        &context,
        &out_weight_values,
        vec![v_elements as u64, HIDDEN as u64],
    );
    let weights = GdnMixerWeights {
        attn_norm: &attn_norm,
        in_proj_qkv: &qkv_weight,
        in_proj_z: &z_weight,
        beta_proj: &beta_weight,
        alpha_proj: &alpha_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        conv1d: &conv_weight,
        norm: &internal_norm,
        out_proj: &out_weight,
    };
    let input = values(N_TOKENS * HIDDEN, 31, 0.007, -0.15);
    let initial_conv_state = values(conv_state_elements, 37, 0.0008, -0.018);
    let initial_recurrence_state = values(state_elements, 41, 0.00017, -0.004);
    let grad_output = values(N_TOKENS * HIDDEN, 43, 0.006, -0.13);
    let run = |input: &[f32], grad_output: &[f32], rule| {
        gdn_mixer_replay_vjp_readback(
            &context,
            geometry,
            weights,
            input,
            &initial_conv_state,
            &initial_recurrence_state,
            grad_output,
            N_TOKENS,
            rule,
            true,
        )
        .unwrap()
    };
    let actual = run(&input, &grad_output, GdnMixerVjpRule::Jacobian);
    assert!(actual.grad_input.iter().all(|value| value.is_finite()));
    assert!(actual.mixer_outputs.iter().all(|value| value.is_finite()));
    assert!(actual.grad_input.iter().any(|value| *value != 0.0));
    assert_eq!(actual.final_conv_state.len(), conv_state_elements);
    assert_eq!(actual.final_recurrence_state.len(), state_elements);

    let objective = |candidate: &[f32]| {
        run(candidate, &grad_output, GdnMixerVjpRule::Jacobian)
            .mixer_outputs
            .iter()
            .zip(&grad_output)
            .map(|(&value, &gradient)| f64::from(value) * f64::from(gradient))
            .sum::<f64>()
    };
    let epsilon = 2e-2f32;
    for index in [0usize, HIDDEN, input.len() - 1] {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let finite_difference = (objective(&plus) - objective(&minus)) / (2.0 * f64::from(epsilon));
        assert!(
            (finite_difference - f64::from(actual.grad_input[index])).abs() < 8e-4,
            "coordinate {index} finite_difference={finite_difference} reverse={}",
            actual.grad_input[index]
        );
    }
    let direction = values(input.len(), 47, 0.0011, -0.023);
    let shift = |amount: f32| {
        input
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value + amount * direction)
            .collect::<Vec<_>>()
    };
    let forward_directional =
        (objective(&shift(epsilon)) - objective(&shift(-epsilon))) / (2.0 * f64::from(epsilon));
    let reverse_directional = actual
        .grad_input
        .iter()
        .zip(&direction)
        .map(|(&gradient, &direction)| f64::from(gradient) * f64::from(direction))
        .sum::<f64>();
    assert!(
        (forward_directional - reverse_directional).abs() < 1e-3,
        "directional mismatch forward={forward_directional} reverse={reverse_directional}"
    );

    let zeros = vec![0.0f32; grad_output.len()];
    let zero = run(&input, &zeros, GdnMixerVjpRule::Jacobian);
    assert!(zero.grad_input.iter().all(|value| *value == 0.0));
    assert!(
        zero.grad_initial_conv_state
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(
        zero.grad_initial_recurrence_state
            .iter()
            .all(|value| *value == 0.0)
    );
    let relp = run(&input, &grad_output, GdnMixerVjpRule::Relp);
    assert!(
        relp.grad_input
            .iter()
            .zip(&actual.grad_input)
            .any(|(relp, jacobian)| relp.to_bits() != jacobian.to_bits())
    );

    let hidden_elements = N_TOKENS * HIDDEN;
    let max_error = |left: &[f32], right: &[f32]| {
        if left.len() != right.len() {
            return f32::INFINITY;
        }
        left.iter()
            .zip(right)
            .map(|(&left, &right)| finite_abs_difference(left, right))
            .fold(0.0f32, f32::max)
    };
    for rule in [GdnMixerVjpRule::Jacobian, GdnMixerVjpRule::Relp] {
        for query_batches in [1, 2, 8] {
            let grad_bank: Vec<f32> = (0..query_batches * hidden_elements)
                .map(|index| {
                    let query = index / hidden_elements;
                    let local = index % hidden_elements;
                    ((local * 43 + query * 13 + 7) % 61) as f32 * 0.006 - 0.17
                        + query as f32 * 0.005
                })
                .collect();
            let batched = gdn_mixer_replay_vjp_batch_readback(
                &context,
                geometry,
                weights,
                &input,
                &initial_conv_state,
                &initial_recurrence_state,
                &grad_bank,
                N_TOKENS,
                query_batches,
                rule,
                true,
            )
            .unwrap();
            for query in 0..query_batches {
                let hidden_start = query * hidden_elements;
                let hidden_end = hidden_start + hidden_elements;
                let serial = run(&input, &grad_bank[hidden_start..hidden_end], rule);
                let conv_start = query * conv_state_elements;
                let recurrence_start = query * state_elements;
                let errors = [
                    max_error(&batched.mixer_outputs, &serial.mixer_outputs),
                    max_error(&batched.final_conv_state, &serial.final_conv_state),
                    max_error(
                        &batched.final_recurrence_state,
                        &serial.final_recurrence_state,
                    ),
                    max_error(
                        &batched.grad_input[hidden_start..hidden_end],
                        &serial.grad_input,
                    ),
                    max_error(
                        &batched.grad_initial_conv_state
                            [conv_start..conv_start + conv_state_elements],
                        &serial.grad_initial_conv_state,
                    ),
                    max_error(
                        &batched.grad_initial_recurrence_state
                            [recurrence_start..recurrence_start + state_elements],
                        &serial.grad_initial_recurrence_state,
                    ),
                ];
                let error = errors.into_iter().fold(0.0f32, f32::max);
                assert!(
                    error < 2e-5,
                    "{rule:?} GDN batch {query_batches} query {query} error {error}"
                );
            }
        }

        const QUERY_BATCHES: usize = 4;
        const ACTIVE_QUERY: usize = 2;
        let mut isolated_grad = vec![0.0f32; QUERY_BATCHES * hidden_elements];
        let active_start = ACTIVE_QUERY * hidden_elements;
        isolated_grad[active_start..active_start + hidden_elements].copy_from_slice(&grad_output);
        let isolated = gdn_mixer_replay_vjp_batch_readback(
            &context,
            geometry,
            weights,
            &input,
            &initial_conv_state,
            &initial_recurrence_state,
            &isolated_grad,
            N_TOKENS,
            QUERY_BATCHES,
            rule,
            true,
        )
        .unwrap();
        let serial = run(&input, &grad_output, rule);
        for query in 0..QUERY_BATCHES {
            let hidden_start = query * hidden_elements;
            let hidden_end = hidden_start + hidden_elements;
            let conv_start = query * conv_state_elements;
            let recurrence_start = query * state_elements;
            if query == ACTIVE_QUERY {
                let error = [
                    max_error(
                        &isolated.grad_input[hidden_start..hidden_end],
                        &serial.grad_input,
                    ),
                    max_error(
                        &isolated.grad_initial_conv_state
                            [conv_start..conv_start + conv_state_elements],
                        &serial.grad_initial_conv_state,
                    ),
                    max_error(
                        &isolated.grad_initial_recurrence_state
                            [recurrence_start..recurrence_start + state_elements],
                        &serial.grad_initial_recurrence_state,
                    ),
                ]
                .into_iter()
                .fold(0.0f32, f32::max);
                assert!(error < 2e-5, "{rule:?} isolated GDN error {error}");
            } else {
                assert!(
                    isolated.grad_input[hidden_start..hidden_end]
                        .iter()
                        .chain(
                            &isolated.grad_initial_conv_state
                                [conv_start..conv_start + conv_state_elements],
                        )
                        .chain(
                            &isolated.grad_initial_recurrence_state
                                [recurrence_start..recurrence_start + state_elements],
                        )
                        .all(|value| *value == 0.0),
                    "{rule:?} GDN query {query} received another query's cotangent"
                );
            }
        }
    }
}

#[test]
fn dense_ffn_query_rows_match_serial_batches_and_do_not_cross_talk() {
    let context = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const ROWS: usize = 3;
    const HIDDEN: usize = 11;
    const INTERMEDIATE: usize = 17;
    let hidden_elements = ROWS * HIDDEN;
    let residuals: Vec<f32> = (0..hidden_elements)
        .map(|index| ((index * 7 + 3) % 29) as f32 * 0.027 - 0.36)
        .collect();
    let norm: Vec<f32> = (0..HIDDEN)
        .map(|index| 0.61 + (index % 5) as f32 * 0.08)
        .collect();
    let gate_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 11 + 5) % 37) as f32 * 0.006 - 0.097)
        .collect();
    let up_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 13 + 2) % 41) as f32 * 0.005 - 0.083)
        .collect();
    let down_weight: Vec<f32> = (0..INTERMEDIATE * HIDDEN)
        .map(|index| ((index * 17 + 1) % 43) as f32 * 0.004 - 0.071)
        .collect();
    let norm_tensor = f32_tensor(&context, &norm, vec![HIDDEN as u64]);
    let gate_tensor = f32_tensor(
        &context,
        &gate_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let up_tensor = f32_tensor(
        &context,
        &up_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let down_tensor = f32_tensor(
        &context,
        &down_weight,
        vec![INTERMEDIATE as u64, HIDDEN as u64],
    );

    for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
        for query_batches in [1, 2, 8] {
            let grad_outputs: Vec<f32> = (0..query_batches * hidden_elements)
                .map(|index| {
                    let query = index / hidden_elements;
                    let local = index % hidden_elements;
                    ((local * 19 + query * 7 + 4) % 47) as f32 * 0.009 - 0.18 + query as f32 * 0.013
                })
                .collect();
            let batched = dense_ffn_vjp_query_rows_readback(
                &context,
                3,
                HIDDEN,
                INTERMEDIATE,
                &residuals,
                &norm_tensor,
                &gate_tensor,
                &up_tensor,
                &down_tensor,
                &grad_outputs,
                ROWS,
                query_batches,
                rule,
            )
            .unwrap();
            for query in 0..query_batches {
                let start = query * hidden_elements;
                let end = start + hidden_elements;
                let serial = dense_ffn_vjp_rows_readback(
                    &context,
                    3,
                    HIDDEN,
                    INTERMEDIATE,
                    &residuals,
                    &norm_tensor,
                    &gate_tensor,
                    &up_tensor,
                    &down_tensor,
                    &grad_outputs[start..end],
                    ROWS,
                    rule,
                )
                .unwrap();
                let max_abs = batched[start..end]
                    .iter()
                    .zip(&serial)
                    .map(|(&batched, &serial)| (batched - serial).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 2e-5,
                    "{rule:?} batch {query_batches} query {query} error {max_abs}"
                );
            }
        }

        const QUERY_BATCHES: usize = 4;
        const ACTIVE_QUERY: usize = 2;
        let active_grad: Vec<f32> = (0..hidden_elements)
            .map(|index| ((index * 23 + 9) % 53) as f32 * 0.007 - 0.16)
            .collect();
        let mut isolated_grad = vec![0.0f32; QUERY_BATCHES * hidden_elements];
        let active_start = ACTIVE_QUERY * hidden_elements;
        isolated_grad[active_start..active_start + hidden_elements].copy_from_slice(&active_grad);
        let isolated = dense_ffn_vjp_query_rows_readback(
            &context,
            3,
            HIDDEN,
            INTERMEDIATE,
            &residuals,
            &norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &isolated_grad,
            ROWS,
            QUERY_BATCHES,
            rule,
        )
        .unwrap();
        let serial = dense_ffn_vjp_rows_readback(
            &context,
            3,
            HIDDEN,
            INTERMEDIATE,
            &residuals,
            &norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &active_grad,
            ROWS,
            rule,
        )
        .unwrap();
        for query in 0..QUERY_BATCHES {
            let start = query * hidden_elements;
            let end = start + hidden_elements;
            if query == ACTIVE_QUERY {
                let max_abs = isolated[start..end]
                    .iter()
                    .zip(&serial)
                    .map(|(&batched, &serial)| (batched - serial).abs())
                    .fold(0.0f32, f32::max);
                assert!(max_abs < 2e-5, "{rule:?} isolated query error {max_abs}");
            } else {
                assert!(
                    isolated[start..end].iter().all(|value| *value == 0.0),
                    "{rule:?} query {query} received another query's cotangent"
                );
            }
        }
    }
}

#[test]
fn gdn_block_composition_matches_distinct_row_and_temporal_oracles() {
    let context = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const ROWS: usize = 3;
    const HIDDEN: usize = 11;
    const INTERMEDIATE: usize = 17;
    let residuals: Vec<f32> = (0..ROWS * HIDDEN)
        .map(|index| ((index * 7 + 3) % 29) as f32 * 0.027 - 0.36)
        .collect();
    let norm: Vec<f32> = (0..HIDDEN)
        .map(|index| 0.61 + (index % 5) as f32 * 0.08)
        .collect();
    let gate_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 11 + 5) % 37) as f32 * 0.006 - 0.097)
        .collect();
    let up_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 13 + 2) % 41) as f32 * 0.005 - 0.083)
        .collect();
    let down_weight: Vec<f32> = (0..INTERMEDIATE * HIDDEN)
        .map(|index| ((index * 17 + 1) % 43) as f32 * 0.004 - 0.071)
        .collect();
    let grad_output: Vec<f32> = (0..ROWS * HIDDEN)
        .map(|index| ((index * 19 + 4) % 47) as f32 * 0.009 - 0.18)
        .collect();
    let norm_tensor = f32_tensor(&context, &norm, vec![HIDDEN as u64]);
    let gate_tensor = f32_tensor(
        &context,
        &gate_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let up_tensor = f32_tensor(
        &context,
        &up_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let down_tensor = f32_tensor(
        &context,
        &down_weight,
        vec![INTERMEDIATE as u64, HIDDEN as u64],
    );
    for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
        let actual = dense_ffn_vjp_rows_readback(
            &context,
            3,
            HIDDEN,
            INTERMEDIATE,
            &residuals,
            &norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &grad_output,
            ROWS,
            rule,
        )
        .unwrap();
        let mut expected = Vec::with_capacity(ROWS * HIDDEN);
        for row in 0..ROWS {
            expected.extend(cpu_dense_ffn_vjp(
                &residuals[row * HIDDEN..(row + 1) * HIDDEN],
                &norm,
                &gate_weight,
                &up_weight,
                &down_weight,
                &grad_output[row * HIDDEN..(row + 1) * HIDDEN],
                1,
                INTERMEDIATE,
                rule,
            ));
        }
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 2e-5, "{rule:?} rowwise FFN error {max_abs}");

        let block_rule = match rule {
            DenseFfnVjpRule::Jacobian => GdnBlockVjpRule::Jacobian,
            DenseFfnVjpRule::Relp => GdnBlockVjpRule::Relp,
        };
        let expected_mixer_rule = match rule {
            DenseFfnVjpRule::Jacobian => GdnMixerVjpRule::Jacobian,
            DenseFfnVjpRule::Relp => GdnMixerVjpRule::Relp,
        };
        let composition = compose_gdn_block_vjp(
            &context,
            3,
            HIDDEN,
            INTERMEDIATE,
            &residuals,
            &norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &grad_output,
            ROWS,
            block_rule,
            |incoming, mixer_rule| {
                assert_eq!(mixer_rule, expected_mixer_rule);
                let incoming_error = incoming
                    .iter()
                    .zip(&expected)
                    .map(|(&incoming, &expected)| (incoming - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(incoming_error < 2e-5);
                let mut branch = vec![0.0f32; incoming.len()];
                for row in 0..ROWS {
                    for column in 0..HIDDEN {
                        let index = row * HIDDEN + column;
                        branch[index] = 0.35 * incoming[index]
                            + if row + 1 < ROWS {
                                0.2 * incoming[(row + 1) * HIDDEN + column]
                            } else {
                                0.0
                            };
                    }
                }
                Ok(WorkspaceLensGdnVjp {
                    layer: 3,
                    n_tokens: ROWS,
                    hidden_size: HIDDEN,
                    values: branch,
                    grad_initial_conv_state: Vec::new(),
                    grad_initial_recurrence_state: Vec::new(),
                    replay_mixer_outputs: Vec::new(),
                    residual_replay_max_abs_error: 0.0,
                    final_conv_state_max_abs_error: 0.0,
                    final_recurrence_state_max_abs_error: 0.0,
                })
            },
        )
        .unwrap();
        let mut expected_full = expected.clone();
        for row in 0..ROWS {
            for column in 0..HIDDEN {
                let index = row * HIDDEN + column;
                expected_full[index] += 0.35 * expected[index]
                    + if row + 1 < ROWS {
                        0.2 * expected[(row + 1) * HIDDEN + column]
                    } else {
                        0.0
                    };
            }
        }
        let post_mixer_error = composition
            .grad_post_mixer_residuals
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let full_error = composition
            .values
            .iter()
            .zip(&expected_full)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(post_mixer_error < 2e-5);
        assert!(full_error < 4e-5, "{rule:?} full block error {full_error}");
    }

    let zero_gate = f32_tensor(
        &context,
        &vec![0.0; HIDDEN * INTERMEDIATE],
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let zero_up = f32_tensor(
        &context,
        &vec![0.0; HIDDEN * INTERMEDIATE],
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let zero_down = f32_tensor(
        &context,
        &vec![0.0; INTERMEDIATE * HIDDEN],
        vec![INTERMEDIATE as u64, HIDDEN as u64],
    );
    let identity_only = compose_gdn_block_vjp(
        &context,
        3,
        HIDDEN,
        INTERMEDIATE,
        &residuals,
        &norm_tensor,
        &zero_gate,
        &zero_up,
        &zero_down,
        &grad_output,
        ROWS,
        GdnBlockVjpRule::Jacobian,
        |incoming, _| {
            Ok(WorkspaceLensGdnVjp {
                layer: 3,
                n_tokens: ROWS,
                hidden_size: HIDDEN,
                values: vec![0.0; incoming.len()],
                grad_initial_conv_state: Vec::new(),
                grad_initial_recurrence_state: Vec::new(),
                replay_mixer_outputs: Vec::new(),
                residual_replay_max_abs_error: 0.0,
                final_conv_state_max_abs_error: 0.0,
                final_recurrence_state_max_abs_error: 0.0,
            })
        },
    )
    .unwrap();
    assert_eq!(identity_only.grad_post_mixer_residuals, grad_output);
    assert_eq!(identity_only.values, grad_output);
}

#[test]
fn dense_ffn_vjp_composes_jacobian_and_relp_rules() {
    let context = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const HIDDEN: usize = 11;
    const INTERMEDIATE: usize = 17;
    let residual: Vec<f32> = (0..HIDDEN)
        .map(|index| ((index * 7 + 3) % 19) as f32 * 0.041 - 0.31)
        .collect();
    let norm: Vec<f32> = (0..HIDDEN)
        .map(|index| 0.61 + (index % 5) as f32 * 0.08)
        .collect();
    let gate_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 11 + 5) % 37) as f32 * 0.006 - 0.097)
        .collect();
    let up_weight: Vec<f32> = (0..HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 13 + 2) % 41) as f32 * 0.005 - 0.083)
        .collect();
    let down_weight: Vec<f32> = (0..INTERMEDIATE * HIDDEN)
        .map(|index| ((index * 17 + 1) % 43) as f32 * 0.004 - 0.071)
        .collect();
    let norm_tensor = f32_tensor(&context, &norm, vec![HIDDEN as u64]);
    let gate_tensor = f32_tensor(
        &context,
        &gate_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let up_tensor = f32_tensor(
        &context,
        &up_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let down_tensor = f32_tensor(
        &context,
        &down_weight,
        vec![INTERMEDIATE as u64, HIDDEN as u64],
    );

    for n_query in [1usize, 2, 8] {
        let grad_output: Vec<f32> = (0..n_query * HIDDEN)
            .map(|index| ((index * 19 + 4) % 47) as f32 * 0.009 - 0.18)
            .collect();
        for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
            let expected = cpu_dense_ffn_vjp(
                &residual,
                &norm,
                &gate_weight,
                &up_weight,
                &down_weight,
                &grad_output,
                n_query,
                INTERMEDIATE,
                rule,
            );
            let actual = dense_ffn_vjp_readback(
                &context,
                3,
                HIDDEN,
                INTERMEDIATE,
                &residual,
                &norm_tensor,
                &gate_tensor,
                &up_tensor,
                &down_tensor,
                &grad_output,
                n_query,
                rule,
            )
            .unwrap();
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs < 4e-5,
                "rule={rule:?} n_query={n_query}: max error {max_abs}"
            );
        }
    }
    let mismatch = dense_ffn_vjp_readback(
        &context,
        3,
        HIDDEN + 1,
        INTERMEDIATE,
        &residual,
        &norm_tensor,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
        &vec![0.0; HIDDEN],
        1,
        DenseFfnVjpRule::Jacobian,
    )
    .expect_err("architecture/weight shape mismatch must fail");
    assert!(matches!(
        mismatch,
        WorkspaceLensError::InvalidDenseFfnShape {
            role: LinearRole::FfnGate,
            ..
        }
    ));
}

#[test]
fn dense_ffn_vjp_preserves_the_identity_residual() {
    let context = match MetalContext::new() {
        Ok(context) => context,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(error) => panic!("init failed: {error}"),
    };
    const HIDDEN: usize = 7;
    const INTERMEDIATE: usize = 9;
    const N_QUERY: usize = 2;
    let residual = vec![0.25f32; HIDDEN];
    let norm = vec![1.0f32; HIDDEN];
    let gate_weight = vec![0.03f32; HIDDEN * INTERMEDIATE];
    let up_weight = vec![-0.02f32; HIDDEN * INTERMEDIATE];
    let down_weight = vec![0.0f32; INTERMEDIATE * HIDDEN];
    let grad_output: Vec<f32> = (0..N_QUERY * HIDDEN)
        .map(|index| index as f32 * 0.017 - 0.09)
        .collect();
    let norm = f32_tensor(&context, &norm, vec![HIDDEN as u64]);
    let gate = f32_tensor(
        &context,
        &gate_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let up = f32_tensor(
        &context,
        &up_weight,
        vec![HIDDEN as u64, INTERMEDIATE as u64],
    );
    let down = f32_tensor(
        &context,
        &down_weight,
        vec![INTERMEDIATE as u64, HIDDEN as u64],
    );
    for rule in [DenseFfnVjpRule::Jacobian, DenseFfnVjpRule::Relp] {
        let actual = dense_ffn_vjp_readback(
            &context,
            0,
            HIDDEN,
            INTERMEDIATE,
            &residual,
            &norm,
            &gate,
            &up,
            &down,
            &grad_output,
            N_QUERY,
            rule,
        )
        .unwrap();
        assert_eq!(actual, grad_output, "identity branch changed in {rule:?}");
    }
}
#[test]
#[ignore = "requires Metal and QWEN_WORKSPACE_LENS_MODEL; supports native dense or MoE"]
fn passive_scalar_last_layer_matches_deployed_inference() {
    use crate::runtime::{LoadedModelConfig, ModelLoadIntent, Runtime, SequenceConfig};

    let runtime = Runtime::metal().unwrap();
    let loaded = runtime
        .load_model_with_intent(
            std::env::var("QWEN_WORKSPACE_LENS_MODEL").unwrap(),
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )
        .unwrap();
    let mut observed = loaded.create_sequence(SequenceConfig::new(2)).unwrap();
    let mut ordinary = loaded.create_sequence(SequenceConfig::new(2)).unwrap();
    let mut passive = loaded
        .passive_workspace_lens_session(&mut observed)
        .unwrap();
    let capture = passive
        .forward_prompt_last_post_block_residuals(&[1, 2], &[loaded.arch().n_layer - 1])
        .unwrap();
    let readout = passive
        .deployed_logits_from_post_block_residual(&capture.capture.values)
        .unwrap();
    assert_eq!(readout.transported_values, capture.capture.values);
    let forward = loaded.forward();
    let state = unsafe { ordinary.metal_session_mut() };
    forward.single_token(1, 0, state).unwrap();
    let expected = forward.single_token(2, 1, state).unwrap();
    assert_eq!(readout.logits.len(), loaded.arch().vocab_size as usize);
    for (actual, expected) in readout.logits.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
    }
}

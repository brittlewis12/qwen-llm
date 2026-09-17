use super::*;
use crate::gguf::GgufFile;
use crate::loader::Model;
use crate::metal::{
    MetalContext, encode_gdn_step_decay_f32, encode_l2_norm_batched_f32, encode_rmsnorm_gated_f32,
    encode_ssm_conv_silu_f32,
};
use crate::metal_forward::{MetalModel, MetalSession};
use std::time::Instant;

#[cfg(feature = "dflash-k0s-diagnostics")]
struct K0sSyntheticRowProvider {
    side: DFlashK0sCodebookSide,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
impl DFlashK0sRowProvider for K0sSyntheticRowProvider {
    fn rank(&self) -> usize {
        DFLASH_K0S_RANK
    }

    fn row_count(&self) -> usize {
        DFLASH_K0S_VOCAB
    }

    fn dequant_row_for_k0s(&self, row: usize, out: &mut [f32]) -> Result<(), DFlashError> {
        if out.len() != DFLASH_K0S_RANK || row >= DFLASH_K0S_VOCAB {
            return Err(dflash_k0s_error("synthetic row request is out of range"));
        }
        out.fill(0.0);
        match self.side {
            DFlashK0sCodebookSide::Predecessor => {
                out[0] = (row % 17) as f32;
                out[1] = 1.0;
                out[DFLASH_K0S_RANK - 1] = -0.5;
            }
            DFlashK0sCodebookSide::Successor => {
                out[0] = (row % 13) as f32;
                out[1] = 2.0;
                out[DFLASH_K0S_RANK - 1] = 4.0;
            }
        }
        Ok(())
    }

    fn raw_row_for_k0s(&self, row: usize) -> Result<Vec<u8>, DFlashError> {
        let mut bytes = vec![match self.side {
            DFlashK0sCodebookSide::Predecessor => 0xa0,
            DFlashK0sCodebookSide::Successor => 0xb0,
        }];
        bytes.extend_from_slice(&(row as u32).to_le_bytes());
        Ok(bytes)
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_rows() -> Vec<DFlashK0sLatticeRow> {
    let mut rows = Vec::new();
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        let count = if depth == 1 { 1 } else { DFLASH_K0S_TOP_K };
        for predecessor_position in 0..count {
            let slots = (0..DFLASH_K0S_TOP_K)
                .map(|candidate_slot| DFlashK0sSlot {
                    candidate_slot,
                    token_id: (depth * 100 + predecessor_position * 16 + candidate_slot) as i32,
                    unary_bits: (candidate_slot as f32).to_bits(),
                    score_bits: Some((candidate_slot as f32).to_bits()),
                })
                .collect();
            rows.push(DFlashK0sLatticeRow {
                row_index: rows.len(),
                depth,
                predecessor_slot: (depth > 1).then_some(predecessor_position),
                predecessor_token: predecessor_position as i32,
                slots,
                issues: Vec::new(),
                greedy_slot: 15,
                has_valid_choice: true,
            });
        }
    }
    rows
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_capture() -> DFlashK0sCapture {
    let descriptor = || TensorDesc {
        name: "synthetic".into(),
        shape: vec![1],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: 4,
    };
    let dispatch = DFlashK0sDispatchCensusRow {
        family: "draft".into(),
        tag: Some(DFLASH_K0S_SELECTOR_DISPATCH_TAG.into()),
        encoder_ordinal: 2,
        encoder_concurrent: false,
        kernel: "selector_kernel".into(),
        grid: [8, 256, 1],
        threads: [32, 1, 1],
        grid_threadgroups: 64,
        threadgroup_threads: 32,
    };
    let provenance = DFlashK0sProvenance {
        selector_hidden: DFlashK0sTensorProvenance {
            descriptor: descriptor(),
            full_tensor_sha256: [1; 32],
        },
        predecessor: DFlashK0sTensorProvenance {
            descriptor: descriptor(),
            full_tensor_sha256: [2; 32],
        },
        successor: DFlashK0sTensorProvenance {
            descriptor: descriptor(),
            full_tensor_sha256: [3; 32],
        },
        embedded_metallib_sha256: [4; 32],
    };
    let mut capture = DFlashK0sCapture {
        draft_tokens: (0..8).collect(),
        draft_token_bits: (0..8).collect(),
        depths: (1..8)
            .map(|depth| DFlashK0sDepth {
                depth,
                position: 100 + depth as u32,
                full_logits_bits: vec![depth as u32, 0x8000_0000],
                top_k_ids: vec![depth as i32],
                unary_bits: vec![0x3f80_0000],
                selector_hidden_bits: vec![0x4000_0000],
                top_k_issues: Vec::new(),
            })
            .collect(),
        lattice: Vec::new(),
        production_chain: DFlashK0sChain {
            requested_slots: vec![0; 7],
            visited_row_indices: vec![0; 7],
            tokens: (1..8).collect(),
            event: None,
            terminated: false,
            packet_geometry_valid: true,
        },
        raw_rows: Vec::new(),
        provenance,
        dispatch_census: vec![dispatch.clone()],
        selector_hidden_dispatch: dispatch,
        kernel_trace: crate::metal::KernelTraceCounters {
            encoders: 1,
            concurrent_encoders: 0,
            dispatches: 3,
        },
        state_identity: DFlashK0sStateIdentity {
            carry_token: 7,
            noise_start_position: 100,
            target_context_len: 9,
            context_hidden_watermark: 9,
            kv_context_watermark: 9,
            draft_tokens_sha256: [5; 32],
            noise_input_sha256: [6; 32],
            synchronized_event_sha256: [7; 32],
            diagnostic_state_sha256: [8; 32],
        },
        capture_sha256: [0; 32],
        content_sha256: [0; 32],
    };
    capture.content_sha256 = dflash_k0s_capture_content_sha256(&capture);
    capture.capture_sha256 = dflash_k0s_capture_sha256(&capture);
    capture
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_positional_lattice_has_exact_97_by_16_geometry() {
    let rows = k0s_synthetic_rows();
    assert_eq!(rows.len(), DFLASH_K0S_LATTICE_ROWS);
    assert!(
        rows.iter()
            .enumerate()
            .all(|(global_index, row)| row.row_index == global_index)
    );
    assert!(rows.iter().all(|row| row.slots.len() == DFLASH_K0S_TOP_K));
    assert_eq!(rows.iter().filter(|row| row.depth == 1).count(), 1);
    for depth in 2..DFLASH_K0S_BLOCK_SIZE {
        assert_eq!(rows.iter().filter(|row| row.depth == depth).count(), 16);
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_lattice_builder_core_integrates_all_rows_scores_raw_data_and_chains() {
    let predecessor = K0sSyntheticRowProvider {
        side: DFlashK0sCodebookSide::Predecessor,
    };
    let successor = K0sSyntheticRowProvider {
        side: DFlashK0sCodebookSide::Successor,
    };
    let mut ids = vec![0i32; DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_TOP_K];
    let mut unary = vec![0.0f32; ids.len()];
    for depth in 0..DFLASH_K0S_BLOCK_SIZE {
        for slot in 0..DFLASH_K0S_TOP_K {
            let offset = depth * DFLASH_K0S_TOP_K + slot;
            ids[offset] = (100 + depth * 32 + slot) as i32;
            unary[offset] = slot as f32 * 0.125;
        }
    }
    let mut z = vec![0.0f32; DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_RANK];
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        z[depth * DFLASH_K0S_RANK] = 0.5;
        z[depth * DFLASH_K0S_RANK + 1] = 3.0;
        z[(depth + 1) * DFLASH_K0S_RANK - 1] = 0.25;
    }

    let carry = 7;
    let (rows, raw_rows) =
        dflash_k0s_build_lattice_core(&predecessor, &successor, carry, &ids, &unary, &z).unwrap();
    assert_eq!(rows.len(), 97);
    assert!(rows.iter().all(|row| row.slots.len() == 16));
    assert!(
        rows.iter()
            .enumerate()
            .all(|(global_index, row)| row.row_index == global_index)
    );
    assert_eq!(raw_rows.len(), 97 + 97 * 16);
    assert_eq!(raw_rows[0].side, DFlashK0sCodebookSide::Predecessor);
    assert_eq!(raw_rows[0].depth, 1);
    assert_eq!(raw_rows[0].predecessor_slot, None);
    assert_eq!(raw_rows[0].token_id, carry);
    assert_eq!(raw_rows[0].bytes, [0xa0, 7, 0, 0, 0]);
    assert_eq!(raw_rows[1].side, DFlashK0sCodebookSide::Successor);
    assert_eq!(raw_rows[1].candidate_slot, Some(0));

    let first_candidate = ids[DFLASH_K0S_TOP_K];
    let expected_first_score = 12.5f32;
    assert_eq!(first_candidate, 132);
    assert_eq!(
        rows[0].slots[0].score_bits,
        Some(expected_first_score.to_bits())
    );

    let depth_two_slot_zero = rows
        .iter()
        .find(|row| row.depth == 2 && row.predecessor_slot == Some(0))
        .unwrap();
    let depth_two_slot_one = rows
        .iter()
        .find(|row| row.depth == 2 && row.predecessor_slot == Some(1))
        .unwrap();
    assert_ne!(
        depth_two_slot_zero.slots[0].score_bits,
        depth_two_slot_one.slots[0].score_bits
    );

    let mut greedy_slots = Vec::new();
    let mut predecessor_slot = None;
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        let row = rows
            .iter()
            .find(|row| row.depth == depth && row.predecessor_slot == predecessor_slot)
            .unwrap();
        greedy_slots.push(row.greedy_slot);
        predecessor_slot = Some(row.greedy_slot);
    }
    let production =
        dflash_k0s_traverse_slots_mode(&rows, carry, &greedy_slots, DFlashK0sChainMode::Production);
    let mut draft_tokens = vec![ids[0]];
    draft_tokens.extend_from_slice(&production.tokens);
    dflash_k0s_verify_production_replay(&draft_tokens, &production).unwrap();

    let mut non_greedy_slots = greedy_slots;
    non_greedy_slots[0] = if non_greedy_slots[0] == 0 { 1 } else { 0 };
    let non_greedy = dflash_k0s_traverse_slots(&rows, carry, &non_greedy_slots);
    assert_ne!(
        production.visited_row_indices[1],
        non_greedy.visited_row_indices[1]
    );
    let production_downstream = &rows[production.visited_row_indices[1]].slots[0];
    let non_greedy_downstream = &rows[non_greedy.visited_row_indices[1]].slots[0];
    assert_ne!(
        production_downstream.score_bits,
        non_greedy_downstream.score_bits
    );
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_fixed_non_greedy_chain_is_positional_and_rng_free() {
    let rows = k0s_synthetic_rows();
    let slots = [1, 2, 3, 4, 5, 6, 7];
    let chain = dflash_k0s_traverse_slots(&rows, 42, &slots);
    assert_eq!(chain.requested_slots, slots);
    assert_eq!(chain.tokens.len(), 7);
    assert_eq!(chain.visited_row_indices, vec![0, 2, 19, 36, 53, 70, 87]);
    assert!(chain.event.is_none());
    assert!(chain.packet_geometry_valid);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_topk_orders_boundary_ties_and_signed_zero_by_token_id() {
    let mut logits = vec![-1.0f32; 20];
    for value in &mut logits[..17] {
        *value = 1.0;
    }
    let ids: Vec<i32> = (0..16).collect();
    let unary: Vec<f32> = ids.iter().map(|&id| logits[id as usize]).collect();
    assert!(dflash_k0s_reconstruct_top_k(&logits, &ids, &unary).is_empty());
    let mut zeros = vec![-1.0f32; 18];
    zeros[0] = -0.0;
    zeros[1] = 0.0;
    let mut zero_ids: Vec<i32> = (0..18).collect();
    zero_ids.sort_by_key(|&id| if id < 2 { (0, id) } else { (1, id) });
    zero_ids.truncate(16);
    let zero_unary: Vec<f32> = zero_ids.iter().map(|&id| zeros[id as usize]).collect();
    assert!(dflash_k0s_reconstruct_top_k(&zeros, &zero_ids, &zero_unary).is_empty());
    assert_eq!(&zero_ids[..2], &[0, 1]);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_nonfinite_logits_remain_raw_and_support_is_undefined() {
    for value in [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
    ] {
        let mut logits = vec![0.0; 16];
        logits[3] = value;
        let issues = dflash_k0s_reconstruct_top_k(&logits, &(0..16).collect::<Vec<_>>(), &logits);
        assert_eq!(
            issues,
            vec![DFlashK0sTopKIssue::NonFiniteLogit {
                token_id: 3,
                bits: value.to_bits()
            }]
        );
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_topk_reports_deliberate_id_and_unary_mismatches() {
    let logits: Vec<f32> = (0..20).map(|value| value as f32).collect();
    let mut ids: Vec<i32> = (4..20).rev().collect();
    let mut unary: Vec<f32> = ids.iter().map(|&id| logits[id as usize]).collect();
    ids[0] = 18;
    unary[1] = -123.0;
    let issues = dflash_k0s_reconstruct_top_k(&logits, &ids, &unary);
    assert!(matches!(
        issues[0],
        DFlashK0sTopKIssue::IdMismatch {
            slot: 0,
            expected: 19,
            observed: 18
        }
    ));
    assert!(matches!(
        issues[1],
        DFlashK0sTopKIssue::UnaryMismatch { slot: 1, .. }
    ));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_issue_order_and_nonfinite_classes_are_exact() {
    let ids = [5, 5, -1, -1, 6, 7];
    let nan = f32::from_bits(0x7fc0_4321);
    let scores = [
        Some(0.0),
        Some(nan),
        None,
        None,
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
    ];
    let (issues, best, valid) = dflash_k0s_classify_slots(&ids, &scores, 10);
    assert!(matches!(
        issues[0],
        DFlashK0sSlotIssue::DuplicateId {
            candidate_slot: 1,
            ..
        }
    ));
    assert!(matches!(
        issues[1],
        DFlashK0sSlotIssue::NonFiniteScore {
            class: DFlashK0sNonFiniteClass::Nan,
            score_bits: 0x7fc0_4321,
            ..
        }
    ));
    assert!(matches!(
        issues[2],
        DFlashK0sSlotIssue::Sentinel {
            candidate_slot: 2,
            ..
        }
    ));
    assert!(matches!(
        issues[3],
        DFlashK0sSlotIssue::DuplicateId {
            candidate_slot: 3,
            ..
        }
    ));
    assert!(matches!(
        issues[4],
        DFlashK0sSlotIssue::Sentinel {
            candidate_slot: 3,
            ..
        }
    ));
    assert!(matches!(
        issues[5],
        DFlashK0sSlotIssue::NonFiniteScore {
            class: DFlashK0sNonFiniteClass::PositiveInfinity,
            ..
        }
    ));
    assert!(matches!(
        issues[6],
        DFlashK0sSlotIssue::NonFiniteScore {
            class: DFlashK0sNonFiniteClass::NegativeInfinity,
            ..
        }
    ));
    assert_eq!(best, 4);
    assert!(valid);

    let (issues, _, valid) =
        dflash_k0s_classify_slots(&[1, 2, -1], &[Some(f32::NEG_INFINITY), Some(nan), None], 10);
    assert!(!valid);
    assert_eq!(issues.last(), Some(&DFlashK0sSlotIssue::NoValidChoice));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
#[allow(clippy::assign_op_pattern)]
fn k0s_finite_score_preserves_scalar_operation_bits_and_first_tie() {
    let score = dflash_k0s_scalar_score(&[1.5, -2.0, 0.25], &[2.0, 3.0, 4.0], 0.5);
    let mut expected = 0.0f32;
    expected = expected + 1.5f32 * 2.0f32;
    expected = expected + -2.0f32 * 3.0f32;
    expected = expected + 0.25f32 * 4.0f32;
    expected = 0.5f32 + expected;
    assert_eq!(score.to_bits(), expected.to_bits());
    let (_, best, valid) = dflash_k0s_classify_slots(&[1, 2], &[Some(-0.0), Some(0.0)], 10);
    assert!(valid);
    assert_eq!(best, 0);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_chain_classifies_invalid_carry_missing_row_and_slot_zero() {
    let rows = k0s_synthetic_rows();
    let invalid = dflash_k0s_traverse_slots(&rows, -1, &[0]);
    assert!(matches!(
        invalid.event,
        Some(DFlashK0sChainEvent::InvalidCarry { .. })
    ));
    assert!(invalid.terminated);
    assert!(invalid.tokens.is_empty());

    let mut missing_depth_one = rows.clone();
    missing_depth_one.retain(|row| row.depth != 1);
    let missing_depth_one = dflash_k0s_traverse_slots(&missing_depth_one, 1, &[0]);
    assert!(matches!(
        missing_depth_one.event,
        Some(DFlashK0sChainEvent::MissingPredecessorRow {
            depth: 1,
            predecessor_slot: None,
            ..
        })
    ));

    let mut missing = rows.clone();
    missing.retain(|row| !(row.depth == 2 && row.predecessor_slot == Some(3)));
    let missing = dflash_k0s_traverse_slots(&missing, 1, &[3, 0]);
    assert!(!missing.packet_geometry_valid);
    assert!(matches!(
        missing.event,
        Some(DFlashK0sChainEvent::MissingPredecessorRow {
            depth: 2,
            predecessor_slot: Some(3),
            ..
        })
    ));
    assert!(missing.terminated);

    let mut terminating = rows;
    terminating[0].has_valid_choice = false;
    terminating[0].slots[0].token_id = -1;
    let fixed = dflash_k0s_traverse_slots(&terminating, 1, &[0, 1]);
    assert!(fixed.terminated);
    assert!(fixed.tokens.is_empty());
    assert!(fixed.event.is_none());
    let terminating =
        dflash_k0s_traverse_slots_mode(&terminating, 1, &[0, 1], DFlashK0sChainMode::Production);
    assert!(matches!(
        terminating.event,
        Some(DFlashK0sChainEvent::SlotZeroTermination {
            depth: 1,
            slot: 0,
            ..
        })
    ));
    assert!(terminating.terminated);
    assert!(terminating.tokens.is_empty());

    let mut sentinel = k0s_synthetic_rows();
    sentinel[0].slots[3].token_id = -1;
    let sentinel = dflash_k0s_traverse_slots(&sentinel, 1, &[3, 0]);
    assert!(sentinel.terminated);
    assert!(sentinel.tokens.is_empty());
    assert!(sentinel.event.is_none());
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_production_replay_is_required_despite_unrelated_issues() {
    let mut rows = k0s_synthetic_rows();
    rows[96].issues.push(DFlashK0sSlotIssue::DuplicateId {
        candidate_slot: 1,
        first_slot: 0,
        token_id: 9,
    });
    let slots = vec![1, 2, 3, 4, 5, 6, 7];
    let chain = dflash_k0s_traverse_slots_mode(&rows, 42, &slots, DFlashK0sChainMode::Production);
    let mut draft = vec![999];
    draft.extend_from_slice(&chain.tokens);
    assert!(dflash_k0s_verify_production_replay(&draft, &chain).is_ok());
    draft[7] ^= 1;
    assert!(dflash_k0s_verify_production_replay(&draft, &chain).is_err());
    let mut shortened = chain;
    shortened.terminated = true;
    shortened.tokens.pop();
    assert!(dflash_k0s_verify_production_replay(&draft, &shortened).is_err());
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_positions_use_checked_u32_arithmetic() {
    assert_eq!(dflash_k0s_positions(u32::MAX - 7).unwrap()[6], u32::MAX);
    assert!(dflash_k0s_positions(u32::MAX - 6).is_err());
    assert!(dflash_k0s_positions(u32::MAX).is_err());
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_observer_guard_restores_baseline_on_all_cpu_exit_paths() {
    let baseline = crate::metal::diagnostics_observer_active_counts();
    let guard = DFlashK0sObserverGuard::begin().unwrap();
    let _ = guard.finish().unwrap();
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);

    fn result_error() -> Result<(), DFlashError> {
        let _guard = DFlashK0sObserverGuard::begin()?;
        Err(dflash_k0s_error("synthetic result error"))
    }
    assert!(result_error().is_err());
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);

    let guard = DFlashK0sObserverGuard::begin().unwrap();
    let _ = guard.finish().unwrap();
    let _: Result<(), DFlashError> = Err(dflash_k0s_error("synthetic post-draft error"));
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);

    let unwind = std::panic::catch_unwind(|| {
        let _guard = DFlashK0sObserverGuard::begin().unwrap();
        panic!("synthetic observer unwind");
    });
    assert!(unwind.is_err());
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_live_observation_blocks_draft_before_any_work() {
    assert!(dflash_k0s_require_no_live_event(None).is_ok());
    let error = dflash_k0s_require_no_live_event(Some(77)).unwrap_err();
    assert!(matches!(
        &error,
        DFlashError::K0sObservationLive { event_sequence: 77 }
    ));
    assert_eq!(
        error.to_string(),
        "dflash K0-S observation event 77 is still live; extract or finish it before another draft"
    );
    let source = include_str!("../metal_dflash.rs");
    let draft = source
        .split("pub fn draft_block(")
        .nth(1)
        .unwrap()
        .split("fn select_draft_path")
        .next()
        .unwrap();
    let guard = draft.find("dflash_k0s_require_no_live_event").unwrap();
    assert!(guard < draft.find("let arch =").unwrap());
    for work in ["commandBuffer()", "KernelEncoder::begin", "encode_"] {
        assert!(draft[..guard].find(work).is_none());
    }
    assert!(!draft[..guard].contains("k0s_live_event = None"));
    let drop_impl = ["impl Drop for ", "DFlashK0sProductionObservation"].concat();
    assert!(!source.contains(&drop_impl));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_session_sequence_is_nonzero_monotonic_and_fails_closed() {
    let next = std::sync::atomic::AtomicU64::new(1);
    assert_eq!(dflash_k0s_allocate_session_sequence_from(&next).unwrap(), 1);
    assert_eq!(dflash_k0s_allocate_session_sequence_from(&next).unwrap(), 2);

    let exhausted = std::sync::atomic::AtomicU64::new(u64::MAX);
    assert!(dflash_k0s_allocate_session_sequence_from(&exhausted).is_err());
    assert_eq!(
        exhausted.load(std::sync::atomic::Ordering::Relaxed),
        u64::MAX
    );
    assert!(dflash_k0s_allocate_session_sequence_from(&exhausted).is_err());

    let zero = std::sync::atomic::AtomicU64::new(0);
    assert!(dflash_k0s_allocate_session_sequence_from(&zero).is_err());
    assert_eq!(zero.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_session_sequence_is_private_and_assigned_once_by_fresh() {
    let source = include_str!("../metal_dflash.rs");
    let session_struct = source
        .split("pub struct MetalDFlashSession {")
        .nth(1)
        .unwrap()
        .split("struct DFlashSessionGeometry")
        .next()
        .unwrap();
    assert!(session_struct.contains("k0s_session_sequence: u64"));
    assert!(!session_struct.contains("pub k0s_session_sequence"));

    let fresh = source
        .split("impl MetalDFlashSession {")
        .nth(1)
        .unwrap()
        .split("pub fn fresh(")
        .nth(1)
        .unwrap()
        .split("pub fn enable_phase_timers")
        .next()
        .unwrap();
    assert_eq!(
        fresh
            .matches("dflash_k0s_allocate_session_sequence()?")
            .count(),
        1
    );
    assert!(
        fresh.find("phase_timings: Vec::new()")
            < fresh.find("k0s_session_sequence: dflash_k0s_allocate_session_sequence()?")
    );

    let binding = source
        .split("fn dflash_k0s_session_binding_sha256")
        .nth(1)
        .unwrap()
        .split("fn dflash_k0s_hash_f32le")
        .next()
        .unwrap();
    assert!(binding.find("session.k0s_session_sequence") < binding.find("for tensor in"));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_dispatch_census_v1_cap_is_inclusive_at_256() {
    assert!(dflash_k0s_check_dispatch_census_len(256).is_ok());
    assert!(dflash_k0s_check_dispatch_census_len(257).is_err());
    assert!(dflash_k0s_check_dispatch_census_len(usize::MAX).is_err());
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_capture_hash_commits_every_synchronized_input_family() {
    let capture = k0s_synthetic_capture();
    assert_eq!(capture.capture_sha256, dflash_k0s_capture_sha256(&capture));
    assert_eq!(
        capture.content_sha256,
        dflash_k0s_capture_content_sha256(&capture)
    );
    let original = capture.capture_sha256;
    let original_content = capture.content_sha256;

    let mut envelope_mutation = capture.clone();
    envelope_mutation.state_identity.carry_token ^= 1;
    envelope_mutation.state_identity.noise_start_position ^= 1;
    envelope_mutation.state_identity.synchronized_event_sha256[0] ^= 1;
    assert_ne!(dflash_k0s_capture_sha256(&envelope_mutation), original);
    assert_eq!(
        dflash_k0s_capture_content_sha256(&envelope_mutation),
        original_content
    );

    let mut mutated = capture.clone();
    mutated.depths[0].full_logits_bits[0] ^= 1;
    assert_ne!(dflash_k0s_capture_sha256(&mutated), original);
    assert_ne!(
        dflash_k0s_capture_content_sha256(&mutated),
        original_content
    );
    let mut mutated = capture.clone();
    mutated.depths[0].top_k_ids[0] ^= 1;
    assert_ne!(dflash_k0s_capture_sha256(&mutated), original);
    let mut mutated = capture.clone();
    mutated.depths[0].selector_hidden_bits[0] ^= 1;
    assert_ne!(dflash_k0s_capture_sha256(&mutated), original);
    let mut mutated = capture.clone();
    mutated.dispatch_census[0].grid[0] ^= 1;
    assert_ne!(dflash_k0s_capture_sha256(&mutated), original);
    let mut mutated = capture;
    mutated.provenance.embedded_metallib_sha256[0] ^= 1;
    assert_ne!(dflash_k0s_capture_sha256(&mutated), original);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_compiled_scalar_contract_matches_independent_fixture() {
    let fixture = dflash_k0s_scalar_contract_fixture();
    assert_eq!(fixture.cases.len(), 6);
    assert!(fixture.cases.iter().all(|case| {
        case.a_bits.len() == DFLASH_K0S_RANK
            && case.z_bits.len() == DFLASH_K0S_RANK
            && case.successor_bits.len() == DFLASH_K0S_RANK
    }));
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| (case.name, case.score_bits))
            .collect::<Vec<_>>(),
        vec![
            ("fma_sensitive_cancellation", 0x0000_0000),
            ("subnormal_signed_result", 0x8000_0001),
            ("signed_zero", 0x0000_0000),
            ("overflow_adjacent_finite", 0x7f7f_ffff),
            ("rank_order_cancellation", 0x4050_0000),
            ("halfway_round_to_even", 0x3f80_0000),
        ]
    );
    assert_eq!(fixture.cases[0].a_bits[1], 0x3f80_0001);
    assert_eq!(fixture.cases[1].a_bits[0], 1);
    assert_eq!(fixture.cases[2].unary_bits, 0x8000_0000);
    assert_eq!(fixture.cases[5].a_bits[1], 0x3380_0000);
    assert_eq!(
        fixture.fixture_sha256,
        [
            0x8c, 0x22, 0xbf, 0x3b, 0x4e, 0xe5, 0x1e, 0xfa, 0xf3, 0x15, 0x13, 0x7a, 0x86, 0x6f,
            0xee, 0xf8, 0xfc, 0x80, 0x19, 0xd5, 0xb2, 0x1c, 0x88, 0x74, 0x11, 0xb5, 0x32, 0xbd,
            0x79, 0xfa, 0x60, 0xe9,
        ]
    );
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_state_digest_material_commits_label_dtype_shape_elements_and_bytes() {
    let digest = |label: &str, dtype: GgmlType, shape: &[u64], bytes: &[u8]| {
        let mut hash = Sha256::new();
        hash.update(b"state-material-test");
        dflash_k0s_hash_state_tensor_material(&mut hash, label, dtype, shape, bytes);
        <[u8; 32]>::from(hash.finalize())
    };
    let original = digest("x", GgmlType::F32, &[2, 2], &[0; 16]);
    assert_ne!(original, digest("h", GgmlType::F32, &[2, 2], &[0; 16]));
    assert_ne!(original, digest("x", GgmlType::I32, &[2, 2], &[0; 16]));
    assert_ne!(original, digest("x", GgmlType::F32, &[4], &[0; 16]));
    let mut bytes = [0; 16];
    bytes[15] = 1;
    assert_ne!(original, digest("x", GgmlType::F32, &[2, 2], &bytes));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_state_digest_source_covers_all_session_tensor_families() {
    assert_eq!(
        DFLASH_K0S_REQUIRED_STATE_TENSORS,
        [
            "target_ctx_stacked",
            "pos_ctx",
            "ctx_h",
            "noise_ids",
            "x",
            "h",
            "q_buf",
            "k_noise",
            "v_noise",
            "k_ctx_buf",
            "v_ctx_buf",
            "attn_o",
            "mixer_out",
            "draft_logits",
            "draft_argmax",
            "k_full",
            "v_full",
            "pos_k",
            "attn_o_full",
            "ffn_gate_buf",
            "ffn_up_buf",
            "ffn_inner_buf",
            "ffn_out_buf",
        ]
    );
    let source = include_str!("../metal_dflash.rs");
    let state_digest = source
        .split("fn dflash_k0s_state_sha256")
        .nth(1)
        .unwrap()
        .split("pub fn dflash_k0s_scalar_contract_fixture")
        .next()
        .unwrap();
    for label in DFLASH_K0S_REQUIRED_STATE_TENSORS.iter().copied().chain([
        "k_ctx_cache",
        "v_ctx_cache",
        "conv_buf",
        "conv_dyn_attn",
        "conv_dyn_ffn",
        "topk_ids",
        "topk_vals",
        "sel_h",
        "phase_timings",
    ]) {
        assert!(state_digest.contains(label), "missing state field {label}");
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_codebook_retains_exact_descriptor_hash_and_raw_rows() {
    let desc = TensorDesc {
        name: "selector_test".into(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        shard_idx: 3,
        data_offset: 4096,
        n_bytes: 16,
    };
    let bytes: Vec<u8> = (0..16).collect();
    let codebook = DFlash2Codebook::from_gguf(&desc, &bytes).unwrap();
    assert_eq!(codebook.original_desc.name, desc.name);
    assert_eq!(codebook.original_desc.data_offset, 4096);
    assert_eq!(
        codebook.full_tensor_sha256,
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    assert_eq!(codebook.raw_row(1).unwrap(), bytes[8..].to_vec());
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_rejects_malformed_geometry_before_lattice_allocation() {
    let desc = TensorDesc {
        name: "small".into(),
        shape: vec![2, 1],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: 8,
    };
    let codebook = DFlash2Codebook::from_gguf(&desc, &[0; 8]).unwrap();
    assert!(dflash_k0s_build_lattice(&codebook, &codebook, 0, &[], &[], &[]).is_err());
    assert!(usize::MAX.checked_mul(DFLASH_K0S_TOP_K).is_none());
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K0sMetalLeaseDecision {
    Proceed,
    Skip,
    FailRequired,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_metal_lease_decision(
    lease_wait: Option<&str>,
    require_metal_tests: Option<&str>,
) -> K0sMetalLeaseDecision {
    if lease_wait == Some("1") {
        K0sMetalLeaseDecision::Proceed
    } else if require_metal_tests == Some("1") {
        K0sMetalLeaseDecision::FailRequired
    } else {
        K0sMetalLeaseDecision::Skip
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_metal_test_context() -> Option<MetalContext> {
    match k0s_metal_lease_decision(
        std::env::var("QWEN_METAL_LEASE_WAIT").ok().as_deref(),
        std::env::var("QWEN_REQUIRE_METAL_TESTS").ok().as_deref(),
    ) {
        K0sMetalLeaseDecision::Proceed => {}
        K0sMetalLeaseDecision::Skip => return None,
        K0sMetalLeaseDecision::FailRequired => {
            panic!("Metal tests require QWEN_METAL_LEASE_WAIT=1 before Metal initialization")
        }
    }
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(MetalError::EmptyLibrary | MetalError::NoDevice) => {
            let required = std::env::var("QWEN_REQUIRE_METAL_TESTS").as_deref() == Ok("1");
            if required {
                panic!("Metal is required but unavailable");
            }
            None
        }
        Err(error) => panic!("Metal context: {error}"),
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_metal_lease_guard_decides_before_context_creation() {
    assert_eq!(
        k0s_metal_lease_decision(Some("1"), Some("1")),
        K0sMetalLeaseDecision::Proceed
    );
    assert_eq!(
        k0s_metal_lease_decision(Some("1"), None),
        K0sMetalLeaseDecision::Proceed
    );
    for lease in [None, Some("0"), Some("true"), Some(" 1"), Some("1 ")] {
        assert_eq!(
            k0s_metal_lease_decision(lease, None),
            K0sMetalLeaseDecision::Skip
        );
        assert_eq!(
            k0s_metal_lease_decision(lease, Some("1")),
            K0sMetalLeaseDecision::FailRequired
        );
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_embedded_metallib_identity_is_context_free_and_feature_gated() {
    let bytes = dflash_k0s_embedded_metallib_bytes();
    let identity = dflash_k0s_embedded_metallib_identity();
    assert_eq!(bytes, crate::KERNELS_METALLIB);
    assert_eq!(identity.byte_count, crate::KERNELS_METALLIB.len());
    assert_eq!(
        identity.sha256,
        <[u8; 32]>::from(Sha256::digest(crate::KERNELS_METALLIB))
    );
    let source = include_str!("../metal_dflash.rs");
    for name in [
        "pub fn dflash_k0s_embedded_metallib_bytes",
        "pub fn dflash_k0s_embedded_metallib_identity",
    ] {
        let prefix = source.split(name).next().unwrap();
        assert!(prefix.ends_with("#[cfg(feature = \"dflash-k0s-diagnostics\")]\n"));
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_selector_input_and_event_envelope_known_vectors() {
    let hex = |digest: [u8; 32]| {
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let logits = [1.0, -0.0, f32::from_bits(0x7fc0_1234)];
    let ids = [-1, 0, i32::MAX];
    let unary = [f32::INFINITY, f32::from_bits(1)];
    let selector_hidden = [0.5, -2.25];
    let logits_hash = dflash_k0s_hash_full_logits_f32le(&logits);
    let ids_hash = dflash_k0s_hash_top_k_ids_i32le(&ids);
    let unary_hash = dflash_k0s_hash_unary_f32le(&unary);
    let selector_hidden_hash = dflash_k0s_hash_selector_hidden_f32le(&selector_hidden);
    assert_eq!(
        hex(logits_hash),
        "d7ce69baed8e4e54815db4d1b3da7a63f11a4b7c0753fd3b1402d8868d356929"
    );
    assert_eq!(
        hex(ids_hash),
        "f724f45be88aed244e4a8db9f36c03527834b23dd8d759667b8b3ae611fa909d"
    );
    assert_eq!(
        hex(unary_hash),
        "d6df90d4352ae14812cd7012e834670c8b0d61bf845605f4bb840bb413c28ae5"
    );
    assert_eq!(
        hex(selector_hidden_hash),
        "d68fdd6c4295f556a433846a741fd499700782fd88afcac690687cfb5c710f8d"
    );
    let parity = DFlashK0sParitySummary {
        draft_tokens: vec![1, -2],
        dispatch_census: vec![DFlashK0sDispatchCensusRow {
            family: "fam".into(),
            tag: Some("tag".into()),
            encoder_ordinal: 9,
            encoder_concurrent: true,
            kernel: "k".into(),
            grid: [1, 2, 3],
            threads: [4, 5, 6],
            grid_threadgroups: 7,
            threadgroup_threads: 8,
        }],
        kernel_trace: [10, 11, 12],
        selector_inputs: DFlashK0sSelectorInputIdentity {
            full_logits_count: logits.len(),
            full_logits_sha256_f32le: logits_hash,
            top_k_ids_count: ids.len(),
            top_k_ids_sha256_i32le: ids_hash,
            unary_count: unary.len(),
            unary_sha256_f32le: unary_hash,
            selector_hidden_count: selector_hidden.len(),
            selector_hidden_sha256_f32le: selector_hidden_hash,
        },
        selector_dispatch: DFlashK0sSelectorDispatchIdentity {
            weight_dtype: GgmlType::F32,
            input_dtype: GgmlType::F32,
            output_dtype: GgmlType::F32,
            block_size: 8,
            hidden_size: 5120,
            selector_rank: 256,
        },
        diagnostic_state_sha256: [5; 32],
        carry_token: -7,
        noise_start_position: 42,
        session_binding_sha256: [6; 32],
    };
    assert_eq!(
        hex(dflash_k0s_event_envelope_sha256(&parity, 99)),
        "9ecf1a2f54c9608fbc5110838bf93544d02f68f3fe1c69da6849492dcf0ca39d"
    );
    let mut mutation = parity.clone();
    mutation.selector_dispatch.selector_rank ^= 1;
    assert_ne!(
        dflash_k0s_event_envelope_sha256(&mutation, 99),
        dflash_k0s_event_envelope_sha256(&parity, 99)
    );
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_f32_tensor(ctx: &MetalContext, label: &str, elements: usize) -> MetalTensor {
    let seed = label
        .bytes()
        .fold(1u32, |sum, byte| sum.wrapping_add(byte as u32));
    let values: Vec<f32> = (0..elements)
        .map(|index| seed as f32 * 0.001 + index as f32 * 0.000_001)
        .collect();
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&values),
        vec![elements as u64],
        GgmlType::F32,
    )
    .unwrap()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_i32_tensor(ctx: &MetalContext, label: &str, elements: usize) -> MetalTensor {
    let seed = label
        .bytes()
        .fold(1i32, |sum, byte| sum.wrapping_add(byte as i32));
    let values: Vec<i32> = (0..elements).map(|index| seed + index as i32).collect();
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&values),
        vec![elements as u64],
        GgmlType::I32,
    )
    .unwrap()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_zero_q4_codebook(name: &str) -> DFlash2Codebook {
    let row_bytes = 144usize;
    let raw_len = row_bytes * DFLASH_K0S_VOCAB;
    let raw_words = vec![0u32; raw_len / 4];
    let full_tensor_sha256 = Sha256::digest(bytemuck::cast_slice::<u32, u8>(&raw_words)).into();
    let original_desc = TensorDesc {
        name: name.into(),
        shape: vec![DFLASH_K0S_RANK as u64, DFLASH_K0S_VOCAB as u64],
        dtype: GgmlType::Q4_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: raw_len as u64,
    };
    DFlash2Codebook {
        raw_words,
        raw_len,
        dtype: GgmlType::Q4_K,
        rank: DFLASH_K0S_RANK,
        n_rows: DFLASH_K0S_VOCAB,
        row_bytes,
        row_desc: TensorDesc {
            name: format!("{name}.row"),
            shape: vec![DFLASH_K0S_RANK as u64],
            dtype: GgmlType::Q4_K,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: row_bytes as u64,
        },
        original_desc,
        full_tensor_sha256,
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_head(ctx: &MetalContext) -> MetalDFlashHead {
    let mut hidden_values = vec![0.0f32; DFLASH_K0S_HIDDEN * DFLASH_K0S_RANK];
    for rank in 0..DFLASH_K0S_RANK {
        hidden_values[rank * DFLASH_K0S_HIDDEN] = (rank + 1) as f32;
    }
    let hidden_sha256 = Sha256::digest(bytemuck::cast_slice(&hidden_values)).into();
    let hidden = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&hidden_values),
        vec![DFLASH_K0S_HIDDEN as u64, DFLASH_K0S_RANK as u64],
        GgmlType::F32,
    )
    .unwrap();
    let hidden_desc = TensorDesc {
        name: "synthetic.selector_hidden.weight".into(),
        shape: vec![DFLASH_K0S_HIDDEN as u64, DFLASH_K0S_RANK as u64],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: (hidden_values.len() * 4) as u64,
    };
    drop(hidden_values);
    MetalDFlashHead {
        config: crate::loader::DFlashConfig {
            n_layer: 0,
            hidden_size: DFLASH_K0S_HIDDEN as u32,
            intermediate_size: 1,
            n_q_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            rope_theta: 10_000.0,
            swa_window: 0,
            block_size: DFLASH_K0S_BLOCK_SIZE as u32,
            mask_token_id: 99,
            n_target_features_layers: 0,
            conv_kernel_size: 1,
            conv_group_size: DFLASH_K0S_HIDDEN as u32,
            selector_rank: DFLASH_K0S_RANK as u32,
            selector_top_k: DFLASH_K0S_TOP_K as u32,
        },
        target_layer_ids: Vec::new(),
        fc: k0s_synthetic_f32_tensor(ctx, "head.fc", 1),
        hidden_norm: k0s_synthetic_f32_tensor(ctx, "head.hidden_norm", 1),
        output_norm: k0s_synthetic_f32_tensor(ctx, "head.output_norm", 1),
        layers: Vec::new(),
        selector: Some(MetalDFlash2Selector {
            hidden,
            predecessor: k0s_zero_q4_codebook("synthetic.selector_predecessor.weight"),
            successor: k0s_zero_q4_codebook("synthetic.selector_successor.weight"),
            rank: DFLASH_K0S_RANK,
            top_k: DFLASH_K0S_TOP_K,
            hidden_provenance: DFlashK0sTensorProvenance {
                descriptor: hidden_desc,
                full_tensor_sha256: hidden_sha256,
            },
        }),
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_session(ctx: &MetalContext) -> MetalDFlashSession {
    let mut h_values = vec![0.0f32; DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_HIDDEN];
    for depth in 0..DFLASH_K0S_BLOCK_SIZE {
        h_values[depth * DFLASH_K0S_HIDDEN] = (depth + 1) as f32;
    }
    let h = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&h_values),
        vec![(DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_HIDDEN) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let mut logits = vec![0.0f32; DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_VOCAB];
    for depth in 0..DFLASH_K0S_BLOCK_SIZE {
        for token in 0..DFLASH_K0S_VOCAB {
            logits[depth * DFLASH_K0S_VOCAB + token] = depth as f32 * 1_000_000.0 - token as f32;
        }
    }
    let draft_logits = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&logits),
        vec![logits.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    drop(logits);
    let mut topk_ids = vec![0i32; DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_TOP_K];
    let mut topk_vals = vec![0.0f32; topk_ids.len()];
    for depth in 0..DFLASH_K0S_BLOCK_SIZE {
        for slot in 0..DFLASH_K0S_TOP_K {
            let offset = depth * DFLASH_K0S_TOP_K + slot;
            topk_ids[offset] = slot as i32;
            topk_vals[offset] = depth as f32 * 1_000_000.0 - slot as f32;
        }
    }
    let topk_ids = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&topk_ids),
        vec![(DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_TOP_K) as u64],
        GgmlType::I32,
    )
    .unwrap();
    let topk_vals = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&topk_vals),
        vec![(DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_TOP_K) as u64],
        GgmlType::F32,
    )
    .unwrap();
    let noise: Vec<i32> = std::iter::once(7)
        .chain(std::iter::repeat_n(99, DFLASH_K0S_BLOCK_SIZE - 1))
        .collect();
    MetalDFlashSession {
        target_ctx_stacked: k0s_synthetic_f32_tensor(ctx, "target_ctx_stacked", 37),
        target_ctx_n: 1,
        target_ctx_capacity: 1,
        pos_ctx: k0s_synthetic_i32_tensor(ctx, "pos_ctx", 1),
        ctx_h: k0s_synthetic_f32_tensor(ctx, "ctx_h", DFLASH_K0S_HIDDEN),
        ctx_h_ready_n: 1,
        k_ctx_cache: vec![
            k0s_synthetic_f32_tensor(ctx, "k_ctx_cache.0", 31),
            k0s_synthetic_f32_tensor(ctx, "k_ctx_cache.1", 29),
        ],
        v_ctx_cache: vec![
            k0s_synthetic_f32_tensor(ctx, "v_ctx_cache.0", 31),
            k0s_synthetic_f32_tensor(ctx, "v_ctx_cache.1", 29),
        ],
        kv_ctx_ready_n: 1,
        noise_ids: MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&noise),
            vec![DFLASH_K0S_BLOCK_SIZE as u64],
            GgmlType::I32,
        )
        .unwrap(),
        x: k0s_synthetic_f32_tensor(ctx, "x", DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_HIDDEN),
        h,
        q_buf: k0s_synthetic_f32_tensor(ctx, "q_buf", 43),
        k_noise: k0s_synthetic_f32_tensor(ctx, "k_noise", 47),
        v_noise: k0s_synthetic_f32_tensor(ctx, "v_noise", 53),
        k_ctx_buf: k0s_synthetic_f32_tensor(ctx, "k_ctx_buf", 59),
        v_ctx_buf: k0s_synthetic_f32_tensor(ctx, "v_ctx_buf", 61),
        attn_o: k0s_synthetic_f32_tensor(ctx, "attn_o", 67),
        mixer_out: k0s_synthetic_f32_tensor(ctx, "mixer_out", 71),
        draft_logits,
        draft_argmax: k0s_synthetic_i32_tensor(ctx, "draft_argmax", DFLASH_K0S_BLOCK_SIZE),
        conv_buf: Some(k0s_synthetic_f32_tensor(ctx, "conv_buf", 73)),
        conv_dyn_attn: Some(k0s_synthetic_f32_tensor(ctx, "conv_dyn_attn", 79)),
        conv_dyn_ffn: Some(k0s_synthetic_f32_tensor(ctx, "conv_dyn_ffn", 83)),
        topk_ids: Some(topk_ids),
        topk_vals: Some(topk_vals),
        sel_h: Some(
            MetalTensor::zeros_f32(ctx, vec![(DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_RANK) as u64])
                .unwrap(),
        ),
        k_full: k0s_synthetic_f32_tensor(ctx, "k_full", 89),
        v_full: k0s_synthetic_f32_tensor(ctx, "v_full", 97),
        pos_k: k0s_synthetic_i32_tensor(ctx, "pos_k", 101),
        attn_o_full: k0s_synthetic_f32_tensor(ctx, "attn_o_full", 103),
        ffn_gate_buf: k0s_synthetic_f32_tensor(ctx, "ffn_gate_buf", 107),
        ffn_up_buf: k0s_synthetic_f32_tensor(ctx, "ffn_up_buf", 109),
        ffn_inner_buf: k0s_synthetic_f32_tensor(ctx, "ffn_inner_buf", 113),
        ffn_out_buf: k0s_synthetic_f32_tensor(ctx, "ffn_out_buf", 127),
        enable_phase_timers: false,
        phase_timings: Vec::new(),
        k0s_session_sequence: dflash_k0s_allocate_session_sequence().unwrap(),
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_dispatch(
    ctx: &MetalContext,
    head: &MetalDFlashHead,
    session: &MetalDFlashSession,
) -> (
    Vec<DFlashK0sDispatchCensusRow>,
    crate::metal::KernelTraceCounters,
) {
    let observer = DFlashK0sObserverGuard::begin().unwrap();
    crate::metal::dispatch_census_set_family("dflash_k0s_synthetic");
    let cmd = ctx
        .queue
        .commandBuffer()
        .expect("synthetic K0-S command buffer");
    let enc = KernelEncoder::begin(&cmd);
    {
        let _tag = dispatch_census_tag_scope(|| DFLASH_K0S_SELECTOR_DISPATCH_TAG.to_owned());
        let selector = head.selector.as_ref().unwrap();
        encode_mat_mat_dispatch(
            ctx,
            &enc,
            &selector.hidden,
            &session.h,
            session.sel_h.as_ref().unwrap(),
            DFLASH_K0S_HIDDEN,
            DFLASH_K0S_RANK,
            DFLASH_K0S_BLOCK_SIZE,
        )
        .unwrap();
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    require_prefill_command_completed(&cmd).unwrap();
    let (rows, counters) = observer.finish().unwrap();
    let rows = rows
        .into_iter()
        .map(|row| DFlashK0sDispatchCensusRow {
            family: row.family.to_owned(),
            tag: row.tag,
            encoder_ordinal: row.encoder_ordinal,
            encoder_concurrent: row.encoder_concurrent,
            kernel: row.kernel,
            grid: [row.grid_width, row.grid_height, row.grid_depth],
            threads: [row.threads_width, row.threads_height, row.threads_depth],
            grid_threadgroups: row.grid_tgs,
            threadgroup_threads: row.tg_threads,
        })
        .collect();
    (rows, counters)
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synthetic_tensor_sha256(tensor: &MetalTensor) -> [u8; 32] {
    let mut hash = Sha256::new();
    dflash_k0s_hash_state_tensor(&mut hash, "synthetic_target_state", tensor).unwrap();
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_synchronized_inputs_sha256(session: &MetalDFlashSession, draft: &[i32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"qwen.dflash_k0s.synthetic_inputs.v1");
    for (label, tensor) in [
        ("draft_logits", &session.draft_logits),
        ("topk_ids", session.topk_ids.as_ref().unwrap()),
        ("topk_vals", session.topk_vals.as_ref().unwrap()),
        ("sel_h", session.sel_h.as_ref().unwrap()),
    ] {
        dflash_k0s_hash_state_tensor(&mut hash, label, tensor).unwrap();
    }
    for token in draft {
        hash.update(token.to_le_bytes());
    }
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_stream_zeroes(hash: &mut Sha256, mut bytes: usize) {
    let zeroes = [0u8; 4096];
    while bytes != 0 {
        let chunk = bytes.min(zeroes.len());
        hash.update(&zeroes[..chunk]);
        bytes -= chunk;
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_expected_zero_codebook_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    k0s_stream_zeroes(&mut hash, 144 * DFLASH_K0S_VOCAB);
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_expected_selector_hidden_sha256() -> [u8; 32] {
    let mut hash = Sha256::new();
    for rank in 0..DFLASH_K0S_RANK {
        hash.update(((rank + 1) as f32).to_bits().to_le_bytes());
        k0s_stream_zeroes(&mut hash, (DFLASH_K0S_HIDDEN - 1) * 4);
    }
    hash.finalize().into()
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_assert_synthetic_capture(
    capture: &DFlashK0sCapture,
    head: &MetalDFlashHead,
    state_digest: [u8; 32],
) {
    assert_eq!(capture.draft_tokens, vec![0; DFLASH_K0S_BLOCK_SIZE]);
    assert_eq!(capture.draft_token_bits, vec![0; DFLASH_K0S_BLOCK_SIZE]);
    let mut expected_draft_hash = Sha256::new();
    for _ in 0..DFLASH_K0S_BLOCK_SIZE {
        expected_draft_hash.update(0i32.to_le_bytes());
    }
    let expected_draft_hash: [u8; 32] = expected_draft_hash.finalize().into();
    assert_eq!(
        capture.state_identity.draft_tokens_sha256,
        expected_draft_hash
    );
    assert_eq!(capture.state_identity.carry_token, 7);
    assert_eq!(capture.state_identity.noise_start_position, 200);
    assert_eq!(capture.state_identity.target_context_len, 1);
    assert_eq!(capture.state_identity.context_hidden_watermark, 1);
    assert_eq!(capture.state_identity.kv_context_watermark, 1);
    let mut expected_noise_hash = Sha256::new();
    expected_noise_hash.update(7i32.to_le_bytes());
    for _ in 1..DFLASH_K0S_BLOCK_SIZE {
        expected_noise_hash.update(99i32.to_le_bytes());
    }
    let expected_noise_hash: [u8; 32] = expected_noise_hash.finalize().into();
    assert_eq!(
        capture.state_identity.noise_input_sha256,
        expected_noise_hash
    );
    let mut expected_event_hash = Sha256::new();
    expected_event_hash.update(7i32.to_le_bytes());
    expected_event_hash.update(200u32.to_le_bytes());
    expected_event_hash.update(1u64.to_le_bytes());
    expected_event_hash.update(1u64.to_le_bytes());
    expected_event_hash.update(1u64.to_le_bytes());
    expected_event_hash.update(expected_noise_hash);
    for _depth in 1..DFLASH_K0S_BLOCK_SIZE {
        for slot in 0..DFLASH_K0S_TOP_K {
            expected_event_hash.update((slot as i32).to_le_bytes());
        }
    }
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        for slot in 0..DFLASH_K0S_TOP_K {
            expected_event_hash.update(
                (depth as f32 * 1_000_000.0 - slot as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
    }
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        for rank in 0..DFLASH_K0S_RANK {
            expected_event_hash.update(
                ((depth + 1) as f32 * (rank + 1) as f32)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
    }
    let expected_event_hash: [u8; 32] = expected_event_hash.finalize().into();
    assert_eq!(
        capture.state_identity.synchronized_event_sha256,
        expected_event_hash
    );
    assert_eq!(capture.state_identity.diagnostic_state_sha256, state_digest);
    assert_eq!(capture.depths.len(), 7);
    for depth in 1..DFLASH_K0S_BLOCK_SIZE {
        let row = &capture.depths[depth - 1];
        assert_eq!(row.depth, depth);
        assert_eq!(row.position, 200 + depth as u32);
        assert_eq!(row.full_logits_bits.len(), DFLASH_K0S_VOCAB);
        for (token, &bits) in row.full_logits_bits.iter().enumerate() {
            assert_eq!(
                bits,
                (depth as f32 * 1_000_000.0 - token as f32).to_bits(),
                "depth={depth} token={token}"
            );
        }
        assert_eq!(row.top_k_ids, (0..16).collect::<Vec<_>>());
        for slot in 0..DFLASH_K0S_TOP_K {
            assert_eq!(
                row.unary_bits[slot],
                (depth as f32 * 1_000_000.0 - slot as f32).to_bits()
            );
        }
        assert_eq!(row.selector_hidden_bits.len(), DFLASH_K0S_RANK);
        for rank in 0..DFLASH_K0S_RANK {
            assert_eq!(
                row.selector_hidden_bits[rank],
                ((depth + 1) as f32 * (rank + 1) as f32).to_bits()
            );
        }
        assert!(row.top_k_issues.is_empty());
    }
    assert_eq!(capture.lattice.len(), 97);
    for (row_index, row) in capture.lattice.iter().enumerate() {
        assert_eq!(row.row_index, row_index);
        assert_eq!(row.slots.len(), 16);
        assert!(row.issues.is_empty());
        assert_eq!(row.greedy_slot, 0);
        for slot in &row.slots {
            assert_eq!(slot.score_bits, Some(slot.unary_bits));
        }
    }
    assert_eq!(capture.raw_rows.len(), 97 + 97 * 16);
    let mut raw_index = 0usize;
    for row in &capture.lattice {
        let predecessor = &capture.raw_rows[raw_index];
        raw_index += 1;
        assert_eq!(predecessor.side, DFlashK0sCodebookSide::Predecessor);
        assert_eq!(predecessor.bytes.len(), 144);
        assert_eq!(predecessor.depth, row.depth);
        assert_eq!(predecessor.predecessor_slot, row.predecessor_slot);
        assert_eq!(predecessor.candidate_slot, None);
        assert_eq!(predecessor.token_id, row.predecessor_token);
        assert!(predecessor.bytes.iter().all(|&byte| byte == 0));
        for slot in &row.slots {
            let successor = &capture.raw_rows[raw_index];
            raw_index += 1;
            assert_eq!(successor.side, DFlashK0sCodebookSide::Successor);
            assert_eq!(successor.depth, row.depth);
            assert_eq!(successor.predecessor_slot, row.predecessor_slot);
            assert_eq!(successor.candidate_slot, Some(slot.candidate_slot));
            assert_eq!(successor.token_id, slot.token_id);
            assert_eq!(successor.bytes.len(), 144);
            assert!(successor.bytes.iter().all(|&byte| byte == 0));
        }
    }
    assert_eq!(raw_index, capture.raw_rows.len());
    assert_eq!(capture.production_chain.tokens, vec![0; 7]);
    assert_eq!(capture.production_chain.requested_slots, vec![0; 7]);
    assert!(!capture.production_chain.terminated);
    assert!(capture.production_chain.event.is_none());
    assert_eq!(capture.dispatch_census.len(), 1);
    let dispatch = &capture.dispatch_census[0];
    assert_eq!(dispatch.family, "dflash_k0s_synthetic");
    assert_eq!(dispatch.encoder_ordinal, 0);
    assert!(!dispatch.encoder_concurrent);
    assert_eq!(dispatch.kernel, "kernel_mat_mat_f32_f32");
    assert_eq!(dispatch.grid, [DFLASH_K0S_RANK as u64, 1, 1]);
    assert_eq!(dispatch.threads, [32, 1, 1]);
    assert_eq!(dispatch.grid_threadgroups, DFLASH_K0S_RANK as u64);
    assert_eq!(dispatch.threadgroup_threads, 32);
    assert_eq!(
        capture.selector_hidden_dispatch.tag.as_deref(),
        Some(DFLASH_K0S_SELECTOR_DISPATCH_TAG)
    );
    assert_eq!(&capture.selector_hidden_dispatch, dispatch);
    assert_eq!(capture.kernel_trace.encoders, 1);
    assert_eq!(capture.kernel_trace.concurrent_encoders, 0);
    assert_eq!(capture.kernel_trace.dispatches, 1);
    let selector = head.selector.as_ref().unwrap();
    let expected_hidden_hash = k0s_expected_selector_hidden_sha256();
    let expected_codebook_hash = k0s_expected_zero_codebook_sha256();
    assert_eq!(
        capture.provenance.selector_hidden.full_tensor_sha256,
        expected_hidden_hash
    );
    assert_eq!(
        selector.hidden_provenance.full_tensor_sha256,
        expected_hidden_hash
    );
    assert_eq!(
        capture.provenance.selector_hidden.descriptor.name,
        "synthetic.selector_hidden.weight"
    );
    assert_eq!(
        capture.provenance.selector_hidden.descriptor.dtype,
        GgmlType::F32
    );
    assert_eq!(
        capture.provenance.selector_hidden.descriptor.shape,
        [DFLASH_K0S_HIDDEN as u64, DFLASH_K0S_RANK as u64]
    );
    assert_eq!(capture.provenance.selector_hidden.descriptor.shard_idx, 0);
    assert_eq!(capture.provenance.selector_hidden.descriptor.data_offset, 0);
    assert_eq!(
        capture.provenance.selector_hidden.descriptor.n_bytes,
        (DFLASH_K0S_HIDDEN * DFLASH_K0S_RANK * 4) as u64
    );
    assert_eq!(
        capture.provenance.predecessor.full_tensor_sha256,
        expected_codebook_hash
    );
    assert_eq!(
        capture.provenance.successor.full_tensor_sha256,
        expected_codebook_hash
    );
    assert_eq!(
        selector.predecessor.full_tensor_sha256,
        expected_codebook_hash
    );
    assert_eq!(
        selector.successor.full_tensor_sha256,
        expected_codebook_hash
    );
    assert_eq!(
        capture.provenance.predecessor.descriptor.name,
        "synthetic.selector_predecessor.weight"
    );
    assert_eq!(
        capture.provenance.successor.descriptor.name,
        "synthetic.selector_successor.weight"
    );
    assert_eq!(
        capture.provenance.predecessor.descriptor.dtype,
        GgmlType::Q4_K
    );
    assert_eq!(
        capture.provenance.successor.descriptor.dtype,
        GgmlType::Q4_K
    );
    assert_eq!(
        capture.provenance.predecessor.descriptor.shape,
        [DFLASH_K0S_RANK as u64, DFLASH_K0S_VOCAB as u64]
    );
    assert_eq!(
        capture.provenance.successor.descriptor.shape,
        [DFLASH_K0S_RANK as u64, DFLASH_K0S_VOCAB as u64]
    );
    for descriptor in [
        &capture.provenance.predecessor.descriptor,
        &capture.provenance.successor.descriptor,
    ] {
        assert_eq!(descriptor.shard_idx, 0);
        assert_eq!(descriptor.data_offset, 0);
        assert_eq!(descriptor.n_bytes, (144 * DFLASH_K0S_VOCAB) as u64);
    }
    assert_eq!(
        capture.provenance.embedded_metallib_sha256,
        <[u8; 32]>::from(Sha256::digest(crate::KERNELS_METALLIB))
    );
    assert_eq!(
        capture.content_sha256,
        dflash_k0s_capture_content_sha256(capture)
    );
    assert_eq!(capture.capture_sha256, dflash_k0s_capture_sha256(capture));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct K0sSyntheticArmSummary {
    draft: Vec<i32>,
    first_rows: Vec<DFlashK0sDispatchCensusRow>,
    first_counters: (u64, u64, u64),
    synchronized_inputs: [u8; 32],
    state_before_extraction: [u8; 32],
    state_after_extraction: [u8; 32],
    target_before: [u8; 32],
    target_after: [u8; 32],
    continuation_rows: Vec<DFlashK0sDispatchCensusRow>,
    continuation_counters: (u64, u64, u64),
    continuation_bits: Vec<u32>,
    continuation_token: i32,
    final_state: [u8; 32],
    final_target: [u8; 32],
    capture_sha256: Option<[u8; 32]>,
    content_sha256: Option<[u8; 32]>,
    event_envelope_sha256: Option<[u8; 32]>,
    rng_draws: u64,
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_forge_observation(
    parity: DFlashK0sParitySummary,
    event_sequence: u64,
) -> DFlashK0sProductionObservation {
    let event_envelope_sha256 = dflash_k0s_event_envelope_sha256(&parity, event_sequence);
    DFlashK0sProductionObservation {
        summary: DFlashK0sObservationSummary {
            parity,
            event_sequence,
            event_envelope_sha256,
        },
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_run_synthetic_arm(
    ctx: &MetalContext,
    head: &MetalDFlashHead,
    diagnostic_on: bool,
    event_sequence: u64,
) -> K0sSyntheticArmSummary {
    let baseline = crate::metal::diagnostics_observer_active_counts();
    let session = k0s_synthetic_session(ctx);
    let target_state = k0s_synthetic_f32_tensor(ctx, "target_state", 257);
    let target_values = read_f32_tensor(&target_state);
    let target_seed = "target_state"
        .bytes()
        .fold(1u32, |sum, byte| sum.wrapping_add(byte as u32));
    for (index, value) in target_values.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            (target_seed as f32 * 0.001 + index as f32 * 0.000_001).to_bits()
        );
    }
    let draft = vec![0i32; DFLASH_K0S_BLOCK_SIZE];
    let (first_rows, first_trace) = k0s_synthetic_dispatch(ctx, head, &session);
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);
    let synchronized_inputs = k0s_synchronized_inputs_sha256(&session, &draft);
    let state_before_extraction = dflash_k0s_state_sha256(&session).unwrap();
    let target_before = k0s_synthetic_tensor_sha256(&target_state);
    let parity = dflash_k0s_parity_summary_from_parts(
        head,
        &session,
        7,
        200,
        &draft,
        &first_rows,
        first_trace,
    )
    .unwrap();
    assert_eq!(
        parity.selector_inputs.full_logits_count,
        7 * DFLASH_K0S_VOCAB
    );
    assert_eq!(parity.selector_inputs.top_k_ids_count, 7 * DFLASH_K0S_TOP_K);
    assert_eq!(parity.selector_inputs.unary_count, 7 * DFLASH_K0S_TOP_K);
    assert_eq!(
        parity.selector_inputs.selector_hidden_count,
        7 * DFLASH_K0S_RANK
    );
    assert_eq!(
        parity.selector_dispatch,
        DFlashK0sSelectorDispatchIdentity {
            weight_dtype: GgmlType::F32,
            input_dtype: GgmlType::F32,
            output_dtype: GgmlType::F32,
            block_size: DFLASH_K0S_BLOCK_SIZE,
            hidden_size: DFLASH_K0S_HIDDEN,
            selector_rank: DFLASH_K0S_RANK,
        }
    );
    let capture = diagnostic_on.then(|| {
        let state_before_finish = dflash_k0s_state_sha256(&session).unwrap();
        let finish_sequence = event_sequence + 100;
        let finish_observation = k0s_forge_observation(parity.clone(), finish_sequence);
        let finish_reuse = k0s_forge_observation(parity.clone(), finish_sequence);
        let mut finish_live = Some(finish_sequence);
        let finished =
            dflash_k0s_finish_observation(head, &session, &mut finish_live, finish_observation)
                .unwrap();
        assert_eq!(finished.parity, parity);
        assert_eq!(finish_live, None);
        assert_eq!(
            dflash_k0s_state_sha256(&session).unwrap(),
            state_before_finish
        );
        assert!(
            dflash_k0s_finish_observation(head, &session, &mut finish_live, finish_reuse,).is_err()
        );
        let observation = k0s_forge_observation(parity.clone(), event_sequence);
        assert_eq!(observation.summary().parity, parity);
        let forged_reuse = k0s_forge_observation(parity.clone(), event_sequence);
        let mut live_event = Some(event_sequence);
        let staged = dflash_k0s_consume_observation(
            head,
            &session,
            DFLASH_K0S_VOCAB,
            &mut live_event,
            observation,
        )
        .unwrap();
        assert_eq!(live_event, None);
        assert!(
            dflash_k0s_consume_observation(
                head,
                &session,
                DFLASH_K0S_VOCAB,
                &mut live_event,
                forged_reuse,
            )
            .is_err()
        );
        let composed = DFlashDecoder::extract_k0s_post_sync(
            head,
            &session,
            DFLASH_K0S_VOCAB,
            7,
            200,
            draft.clone(),
            first_rows.clone(),
            first_trace,
        )
        .unwrap();
        assert_eq!(staged.capture_sha256, composed.capture_sha256);
        assert_eq!(staged.content_sha256, composed.content_sha256);
        staged
    });
    let state_after_extraction = dflash_k0s_state_sha256(&session).unwrap();
    let target_after = k0s_synthetic_tensor_sha256(&target_state);
    assert_eq!(state_before_extraction, state_after_extraction);
    assert_eq!(target_before, target_after);
    if let Some(capture) = capture.as_ref() {
        k0s_assert_synthetic_capture(capture, head, state_before_extraction);
    }
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);

    let (continuation_rows, continuation_trace) = k0s_synthetic_dispatch(ctx, head, &session);
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);
    let continuation = read_f32_tensor(session.sel_h.as_ref().unwrap());
    let continuation_bits: Vec<u32> = continuation.iter().map(|value| value.to_bits()).collect();
    let continuation_token =
        (continuation[DFLASH_K0S_RANK].to_bits() % DFLASH_K0S_VOCAB as u32) as i32;
    let final_state = dflash_k0s_state_sha256(&session).unwrap();
    let final_target = k0s_synthetic_tensor_sha256(&target_state);
    let capture_sha256 = capture.as_ref().map(|capture| capture.capture_sha256);
    let content_sha256 = capture.as_ref().map(|capture| capture.content_sha256);
    let event_envelope_sha256 =
        diagnostic_on.then(|| dflash_k0s_event_envelope_sha256(&parity, event_sequence));
    drop(capture);
    K0sSyntheticArmSummary {
        draft,
        first_rows,
        first_counters: (
            first_trace.encoders,
            first_trace.concurrent_encoders,
            first_trace.dispatches,
        ),
        synchronized_inputs,
        state_before_extraction,
        state_after_extraction,
        target_before,
        target_after,
        continuation_rows,
        continuation_counters: (
            continuation_trace.encoders,
            continuation_trace.concurrent_encoders,
            continuation_trace.dispatches,
        ),
        continuation_bits,
        continuation_token,
        final_state,
        final_target,
        capture_sha256,
        content_sha256,
        event_envelope_sha256,
        rng_draws: 0,
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_mutate_tensor_u32(tensor: &MetalTensor, element: usize) {
    assert!(matches!(tensor.dtype, GgmlType::F32 | GgmlType::I32));
    assert!(element < tensor.n_elements() as usize);
    unsafe {
        let ptr = (tensor.buffer.contents().as_ptr() as *mut u8)
            .add(tensor.offset as usize)
            .cast::<u32>()
            .add(element);
        *ptr ^= 1;
    }
}

#[cfg(feature = "dflash-k0s-diagnostics")]
fn k0s_assert_observation_mutation_rejected(
    ctx: &MetalContext,
    head: &MetalDFlashHead,
    mutate: impl FnOnce(&MetalDFlashSession),
) {
    let session = k0s_synthetic_session(ctx);
    let draft = vec![0; DFLASH_K0S_BLOCK_SIZE];
    let (rows, counters) = k0s_synthetic_dispatch(ctx, head, &session);
    let parity =
        dflash_k0s_parity_summary_from_parts(head, &session, 7, 200, &draft, &rows, counters)
            .unwrap();
    let observation = k0s_forge_observation(parity, 11);
    mutate(&session);
    let mut live = Some(11);
    assert!(
        dflash_k0s_consume_observation(head, &session, DFLASH_K0S_VOCAB, &mut live, observation,)
            .is_err()
    );
    assert_eq!(live, None);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_synthetic_metal_diagnostic_parity_both_orders() {
    let Some(ctx) = k0s_metal_test_context() else {
        return;
    };
    let baseline = crate::metal::diagnostics_observer_active_counts();
    let head = k0s_synthetic_head(&ctx);

    let off_a = k0s_run_synthetic_arm(&ctx, &head, false, 1);
    let on_a = k0s_run_synthetic_arm(&ctx, &head, true, 2);
    let on_b = k0s_run_synthetic_arm(&ctx, &head, true, 3);
    let off_b = k0s_run_synthetic_arm(&ctx, &head, false, 4);

    let without_capture = |mut summary: K0sSyntheticArmSummary| {
        summary.capture_sha256 = None;
        summary.content_sha256 = None;
        summary.event_envelope_sha256 = None;
        summary
    };
    assert_eq!(
        without_capture(off_a.clone()),
        without_capture(on_a.clone())
    );
    assert_eq!(
        without_capture(on_b.clone()),
        without_capture(off_b.clone())
    );
    assert_eq!(
        without_capture(off_a.clone()),
        without_capture(off_b.clone())
    );
    assert_eq!(without_capture(on_a.clone()), without_capture(on_b.clone()));
    assert_eq!(on_a.capture_sha256, on_b.capture_sha256);
    assert_eq!(on_a.content_sha256, on_b.content_sha256);
    assert!(on_a.content_sha256.is_some());
    assert_ne!(on_a.event_envelope_sha256, on_b.event_envelope_sha256);
    assert!(on_a.capture_sha256.is_some());
    assert!(off_a.capture_sha256.is_none());
    assert!(off_b.capture_sha256.is_none());
    for summary in [&off_a, &on_a, &on_b, &off_b] {
        assert_eq!(summary.first_rows, summary.continuation_rows);
        assert_eq!(summary.first_counters, (1, 0, 1));
        assert_eq!(summary.continuation_counters, (1, 0, 1));
        assert_eq!(
            summary.state_before_extraction,
            summary.state_after_extraction
        );
        assert_eq!(summary.state_after_extraction, summary.final_state);
        assert_eq!(summary.target_before, summary.target_after);
        assert_eq!(summary.target_after, summary.final_target);
        assert_eq!(summary.rng_draws, 0);
        assert_eq!(
            summary.continuation_bits.len(),
            DFLASH_K0S_BLOCK_SIZE * DFLASH_K0S_RANK
        );
        for depth in 0..DFLASH_K0S_BLOCK_SIZE {
            for rank in 0..DFLASH_K0S_RANK {
                assert_eq!(
                    summary.continuation_bits[depth * DFLASH_K0S_RANK + rank],
                    ((depth + 1) as f32 * (rank + 1) as f32).to_bits()
                );
            }
        }
        assert_eq!(
            summary.continuation_token,
            (2.0f32.to_bits() % DFLASH_K0S_VOCAB as u32) as i32
        );
    }

    k0s_assert_observation_mutation_rejected(&ctx, &head, |session| {
        k0s_mutate_tensor_u32(&session.draft_logits, DFLASH_K0S_VOCAB)
    });
    k0s_assert_observation_mutation_rejected(&ctx, &head, |session| {
        k0s_mutate_tensor_u32(session.topk_ids.as_ref().unwrap(), DFLASH_K0S_TOP_K)
    });
    k0s_assert_observation_mutation_rejected(&ctx, &head, |session| {
        k0s_mutate_tensor_u32(session.topk_vals.as_ref().unwrap(), DFLASH_K0S_TOP_K)
    });
    k0s_assert_observation_mutation_rejected(&ctx, &head, |session| {
        k0s_mutate_tensor_u32(session.sel_h.as_ref().unwrap(), DFLASH_K0S_RANK)
    });
    k0s_assert_observation_mutation_rejected(&ctx, &head, |session| {
        k0s_mutate_tensor_u32(&session.x, 0)
    });

    let draft = vec![0; DFLASH_K0S_BLOCK_SIZE];
    let first_session = k0s_synthetic_session(&ctx);
    let (rows, counters) = k0s_synthetic_dispatch(&ctx, &head, &first_session);
    let parity = dflash_k0s_parity_summary_from_parts(
        &head,
        &first_session,
        7,
        200,
        &draft,
        &rows,
        counters,
    )
    .unwrap();
    let first_sequence = first_session.k0s_session_sequence;
    drop(first_session);
    let second_session = k0s_synthetic_session(&ctx);
    let (second_rows, second_counters) = k0s_synthetic_dispatch(&ctx, &head, &second_session);
    let second_parity = dflash_k0s_parity_summary_from_parts(
        &head,
        &second_session,
        7,
        200,
        &draft,
        &second_rows,
        second_counters,
    )
    .unwrap();
    assert_ne!(first_sequence, second_session.k0s_session_sequence);
    assert_ne!(
        parity.session_binding_sha256,
        second_parity.session_binding_sha256
    );
    let mut first_semantic = parity.clone();
    first_semantic.session_binding_sha256 = [0; 32];
    let mut second_semantic = second_parity;
    second_semantic.session_binding_sha256 = [0; 32];
    assert_eq!(first_semantic, second_semantic);

    let first_session = k0s_synthetic_session(&ctx);
    let (rows, counters) = k0s_synthetic_dispatch(&ctx, &head, &first_session);
    let parity = dflash_k0s_parity_summary_from_parts(
        &head,
        &first_session,
        7,
        200,
        &draft,
        &rows,
        counters,
    )
    .unwrap();
    let mut live = Some(21);
    assert!(
        dflash_k0s_consume_observation(
            &head,
            &second_session,
            DFLASH_K0S_VOCAB,
            &mut live,
            k0s_forge_observation(parity.clone(), 21),
        )
        .is_err()
    );
    assert_eq!(live, None);
    let mut stale = None;
    assert!(
        dflash_k0s_consume_observation(
            &head,
            &first_session,
            DFLASH_K0S_VOCAB,
            &mut stale,
            k0s_forge_observation(parity.clone(), 21),
        )
        .is_err()
    );
    let mut wrong_dispatch = parity.clone();
    wrong_dispatch.selector_dispatch.selector_rank ^= 1;
    let mut live = Some(23);
    assert!(
        dflash_k0s_finish_observation(
            &head,
            &first_session,
            &mut live,
            k0s_forge_observation(wrong_dispatch, 23),
        )
        .is_err()
    );
    let mut bad_envelope = k0s_forge_observation(parity, 22);
    bad_envelope.summary.event_envelope_sha256[0] ^= 1;
    let mut live = Some(22);
    assert!(
        dflash_k0s_consume_observation(
            &head,
            &first_session,
            DFLASH_K0S_VOCAB,
            &mut live,
            bad_envelope,
        )
        .is_err()
    );
    assert_eq!(crate::metal::diagnostics_observer_active_counts(), baseline);
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_wrapper_source_has_one_draft_and_no_metal_tail_work() {
    let source = include_str!("../metal_dflash.rs");
    let wrapper = source
        .split("pub fn draft_block_with_k0s_diagnostic")
        .nth(1)
        .unwrap()
        .split("pub fn draft_block_with_k0s_observation")
        .next()
        .unwrap();
    assert_eq!(
        wrapper.matches("draft_block_with_k0s_observation(").count(),
        1
    );
    assert_eq!(wrapper.matches("extract_k0s_observation(").count(), 1);
    assert_eq!(wrapper.matches("self.draft_block(").count(), 0);
    let staged_observation = source
        .split("pub fn draft_block_with_k0s_observation")
        .nth(1)
        .unwrap()
        .split("pub fn extract_k0s_observation")
        .next()
        .unwrap();
    assert_eq!(
        staged_observation
            .matches("observe_k0s_production_draft(")
            .count(),
        1
    );
    let observation = source
        .split("fn observe_k0s_production_draft")
        .nth(1)
        .unwrap()
        .split("fn extract_k0s_post_sync")
        .next()
        .unwrap();
    assert_eq!(observation.matches("self.draft_block(").count(), 1);
    let extraction = source
        .split("fn extract_k0s_post_sync")
        .nth(1)
        .unwrap()
        .split("pub fn draft_block_with_logits")
        .next()
        .unwrap();
    for forbidden in [
        ".commit()",
        "waitUntilCompleted",
        "KernelEncoder::begin",
        "BlitEncoder::begin",
        "commandBuffer()",
        "self.draft_block(",
        "encode_",
        "Sampler",
        "rng",
    ] {
        assert!(
            !extraction.contains(forbidden),
            "extraction contains {forbidden}"
        );
    }
    let staged_consumer = source
        .split("fn dflash_k0s_consume_observation")
        .nth(1)
        .unwrap()
        .split("pub fn dflash_k0s_scalar_contract_fixture")
        .next()
        .unwrap();
    for forbidden in [
        ".commit()",
        "waitUntilCompleted",
        "KernelEncoder::begin",
        "commandBuffer()",
        "draft_block(",
        "encode_",
        "Sampler",
        "rng",
    ] {
        assert!(
            !staged_consumer.contains(forbidden),
            "staged consumer contains {forbidden}"
        );
    }
    let finish_only = source
        .split("fn dflash_k0s_finish_observation")
        .nth(1)
        .unwrap()
        .split("pub fn dflash_k0s_scalar_contract_fixture")
        .next()
        .unwrap();
    for forbidden in [
        "extract_k0s_post_sync",
        ".commit()",
        "waitUntilCompleted",
        "KernelEncoder::begin",
        "commandBuffer()",
        "draft_block(",
        "encode_",
        "Sampler",
        "rng",
    ] {
        assert!(
            !finish_only.contains(forbidden),
            "finish-only validation contains {forbidden}"
        );
    }
    assert!(
        observation
            .find("dflash_k0s_check_dispatch_census_len")
            .unwrap()
            < observation.find("let dispatch_census:").unwrap()
    );
    assert!(source.contains("DFLASH_K0S_SELECTOR_DISPATCH_TAG.to_owned()"));
    assert!(source.contains("let observer_guard = DFlashK0sObserverGuard::begin()?"));
}

#[cfg(feature = "dflash-k0s-diagnostics")]
#[test]
fn k0s_four_arm_model_head_contract_is_shared_immutable_session_local_state() {
    let source = include_str!("../metal_dflash.rs");
    let decoder = source
        .split("pub struct DFlashDecoder<'a>")
        .nth(1)
        .unwrap()
        .split("impl<'a> DFlashDecoder<'a>")
        .next()
        .unwrap();
    assert!(decoder.contains("pub base: &'a MetalForward<'a>"));
    assert!(decoder.contains("pub head: &'a MetalDFlashHead"));
    assert!(decoder.contains("pub session: MetalDFlashSession"));

    let forward_source = include_str!("../metal_forward/mod.rs");
    let forward = forward_source
        .split("pub struct MetalForward<'a>")
        .nth(1)
        .unwrap()
        .split("impl<'a> MetalForward<'a>")
        .next()
        .unwrap();
    assert!(forward.contains("pub model: &'a MetalModel"));

    let draft = source
        .split("pub fn draft_block(")
        .nth(1)
        .unwrap()
        .split("fn select_draft_path")
        .next()
        .unwrap();
    assert!(draft.contains("if let Some(sel) = self.head.selector.as_ref()"));
    assert!(draft.contains("&sel.hidden"));
    assert!(draft.contains("&self.session.h"));
    assert!(draft.contains("self.session.sel_h.as_ref()"));
    for forbidden in [
        "&mut self.head",
        "self.head.selector.as_mut()",
        "self.head =",
        "self.base =",
    ] {
        assert!(!draft.contains(forbidden));
    }
    assert!(draft.contains("self.session.ctx_h_ready_n = ctx_len"));
    assert!(draft.contains("self.session.kv_ctx_ready_n = ctx_len"));
}

fn memory_estimate_config() -> crate::loader::DFlashConfig {
    crate::loader::DFlashConfig {
        n_layer: 2,
        hidden_size: 8,
        intermediate_size: 16,
        n_q_heads: 2,
        n_kv_heads: 2,
        head_dim: 4,
        rope_theta: 10_000.0,
        swa_window: 0,
        block_size: 4,
        mask_token_id: 0,
        n_target_features_layers: 3,
        conv_kernel_size: 0,
        conv_group_size: 0,
        selector_rank: 0,
        selector_top_k: 0,
    }
}

fn synthetic_q4_k_codebook_bytes(n_rows: usize) -> Vec<u32> {
    let mut words = vec![0u32; n_rows * 144 / 4];
    let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut words);
    for row in 0..n_rows {
        let block = &mut bytes[row * 144..(row + 1) * 144];
        block[..2].copy_from_slice(
            &half::f16::from_f32(0.25 + row as f32 * 0.125)
                .to_bits()
                .to_le_bytes(),
        );
        block[2..4].copy_from_slice(
            &half::f16::from_f32(0.0625 + row as f32 * 0.03125)
                .to_bits()
                .to_le_bytes(),
        );
        for (index, byte) in block[4..].iter_mut().enumerate() {
            *byte = (row.wrapping_mul(37).wrapping_add(index * 13) & 0xff) as u8;
        }
    }
    words
}

#[test]
fn dflash2_q4_k_codebook_rows_match_whole_tensor_dequant() {
    let words = synthetic_q4_k_codebook_bytes(3);
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let desc = TensorDesc {
        name: "selector_q4_k_test".into(),
        shape: vec![256, 3],
        dtype: GgmlType::Q4_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes.len() as u64,
    };
    let expected = dequant_to_f32(&desc, bytes).unwrap();
    let codebook = DFlash2Codebook::from_gguf(&desc, bytes).unwrap();
    assert_eq!((codebook.raw_words.as_ptr() as usize) % 4, 0);

    for row in 0..3 {
        let mut actual = vec![f32::NAN; 256];
        codebook.dequant_row(row, &mut actual).unwrap();
        assert!(
            actual
                .iter()
                .zip(&expected[row * 256..(row + 1) * 256])
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }
}

#[test]
fn dflash2_q4_k_codebook_rejects_bad_geometry_and_payloads() {
    let bad_rank = TensorDesc {
        name: "selector_q4_k_bad_rank".into(),
        shape: vec![128, 2],
        dtype: GgmlType::Q4_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: 144,
    };
    assert!(matches!(
        DFlash2Codebook::from_gguf(&bad_rank, &[0u8; 144]),
        Err(DFlashError::BadDrafter(
            "selector codebook rank is not block-aligned"
        ))
    ));

    let words = synthetic_q4_k_codebook_bytes(2);
    let bytes = bytemuck::cast_slice::<u32, u8>(&words);
    let desc = TensorDesc {
        name: "selector_q4_k_bad_payload".into(),
        shape: vec![256, 2],
        dtype: GgmlType::Q4_K,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes.len() as u64,
    };
    assert!(DFlash2Codebook::from_gguf(&desc, &bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert!(DFlash2Codebook::from_gguf(&desc, &trailing).is_err());

    let codebook = DFlash2Codebook::from_gguf(&desc, bytes).unwrap();
    assert!(codebook.dequant_row(0, &mut [0.0; 255]).is_err());
    assert!(codebook.dequant_row(2, &mut [0.0; 256]).is_err());
}

#[test]
fn dflash2_legacy_codebook_formats_preserve_row_boundaries() {
    let f32_values = [1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75];
    let f32_bytes = bytemuck::cast_slice::<f32, u8>(&f32_values);
    let f32_desc = TensorDesc {
        name: "selector_f32_test".into(),
        shape: vec![3, 2],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: f32_bytes.len() as u64,
    };
    let f32_codebook = DFlash2Codebook::from_gguf(&f32_desc, f32_bytes).unwrap();
    let mut f32_row = [0.0; 3];
    f32_codebook.dequant_row(1, &mut f32_row).unwrap();
    assert_eq!(
        f32_row.map(f32::to_bits),
        [f32_values[3], f32_values[4], f32_values[5]].map(f32::to_bits)
    );

    let f16_values = [
        half::f16::from_f32(1.0).to_bits(),
        half::f16::from_f32(-2.0).to_bits(),
        half::f16::from_f32(3.5).to_bits(),
        half::f16::from_f32(4.25).to_bits(),
        half::f16::from_f32(-5.5).to_bits(),
        half::f16::from_f32(6.75).to_bits(),
    ];
    let f16_bytes = bytemuck::cast_slice::<u16, u8>(&f16_values);
    let f16_desc = TensorDesc {
        name: "selector_f16_test".into(),
        shape: vec![3, 2],
        dtype: GgmlType::F16,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: f16_bytes.len() as u64,
    };
    let f16_codebook = DFlash2Codebook::from_gguf(&f16_desc, f16_bytes).unwrap();
    let mut f16_row = [0.0; 3];
    f16_codebook.dequant_row(1, &mut f16_row).unwrap();
    let f16_expected = [f16_values[3], f16_values[4], f16_values[5]]
        .map(|bits| half::f16::from_bits(bits).to_f32());
    assert_eq!(f16_row.map(f32::to_bits), f16_expected.map(f32::to_bits));

    let bf16_values = [
        (1.0f32.to_bits() >> 16) as u16,
        ((-2.0f32).to_bits() >> 16) as u16,
        (3.5f32.to_bits() >> 16) as u16,
        (4.25f32.to_bits() >> 16) as u16,
        ((-5.5f32).to_bits() >> 16) as u16,
        (6.75f32.to_bits() >> 16) as u16,
    ];
    let bf16_bytes = bytemuck::cast_slice::<u16, u8>(&bf16_values);
    let bf16_desc = TensorDesc {
        name: "selector_bf16_test".into(),
        shape: vec![3, 2],
        dtype: GgmlType::BF16,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bf16_bytes.len() as u64,
    };
    let bf16_codebook = DFlash2Codebook::from_gguf(&bf16_desc, bf16_bytes).unwrap();
    let mut bf16_row = [0.0; 3];
    bf16_codebook.dequant_row(1, &mut bf16_row).unwrap();
    let bf16_expected = [bf16_values[3], bf16_values[4], bf16_values[5]]
        .map(|bits| f32::from_bits((bits as u32) << 16));
    assert_eq!(bf16_row.map(f32::to_bits), bf16_expected.map(f32::to_bits));

    let mut q8_bytes = [0u8; 68];
    for row in 0..2 {
        let block = &mut q8_bytes[row * 34..(row + 1) * 34];
        block[..2].copy_from_slice(
            &half::f16::from_f32(0.25 * (row + 1) as f32)
                .to_bits()
                .to_le_bytes(),
        );
        for (index, value) in block[2..].iter_mut().enumerate() {
            *value = (index as i8 - 12 + row as i8) as u8;
        }
    }
    let q8_desc = TensorDesc {
        name: "selector_q8_test".into(),
        shape: vec![32, 2],
        dtype: GgmlType::Q8_0,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: q8_bytes.len() as u64,
    };
    let q8_codebook = DFlash2Codebook::from_gguf(&q8_desc, &q8_bytes).unwrap();
    let mut q8_row = [0.0; 32];
    q8_codebook.dequant_row(1, &mut q8_row).unwrap();
    for (index, value) in q8_row.iter().enumerate() {
        assert_eq!(value.to_bits(), (0.5 * (index as f32 - 11.0)).to_bits());
    }
}

fn selector_test_codebook(rows: &[&[f32]]) -> DFlash2Codebook {
    let rank = rows.first().expect("at least one row").len();
    let mut raw = Vec::with_capacity(rows.len() * rank * 4);
    for row in rows {
        assert_eq!(row.len(), rank);
        for value in *row {
            raw.extend_from_slice(&value.to_le_bytes());
        }
    }
    let desc = TensorDesc {
        name: "selector_test_codebook".into(),
        shape: vec![rank as u64, rows.len() as u64],
        dtype: GgmlType::F32,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: raw.len() as u64,
    };
    DFlash2Codebook::from_gguf(&desc, &raw).unwrap()
}

#[test]
fn dflash2_selector_diagnostic_scores_and_tracks_predecessors() {
    let predecessor = selector_test_codebook(&[&[1.0, 2.0], &[2.0, 0.0], &[0.0, 3.0], &[1.0, 1.0]]);
    let successor = selector_test_codebook(&[&[0.0, 0.0], &[1.0, 1.0], &[2.0, 0.0], &[0.0, 1.0]]);
    let ids = [0, 0, 1, 2, 3, 1];
    let unary = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let gates = [0.0, 0.0, 2.0, 0.5, 1.0, 2.0];
    let (tokens, depths) =
        diagnose_dflash2_selector_walk(&predecessor, &successor, 2, 2, 0, 3, &ids, &unary, &gates)
            .unwrap();

    assert_eq!(tokens, vec![0, 2, 3]);
    assert_eq!(depths[0].final_scores, vec![Some(3.0), Some(4.0)]);
    assert_eq!(depths[0].greedy_index, Some(1));
    assert_eq!(depths[1].predecessor_token, Some(2));
    assert_eq!(depths[1].predecessor_choice_index, Some(1));
    assert_eq!(depths[1].final_scores, vec![Some(6.0), Some(6.0)]);
    assert_eq!(depths[1].greedy_index, Some(0));
}

#[test]
fn dflash2_selector_diagnostic_keeps_first_tie_and_flags_candidates() {
    let predecessor = selector_test_codebook(&[&[1.0], &[1.0]]);
    let successor = selector_test_codebook(&[&[0.0], &[1.0]]);
    let ids = [0, 0, 0, 0, 1, 1, -1, 9];
    let unary = [0.0; 8];
    let gates = [0.0, 1.0];
    let (_, depths) =
        diagnose_dflash2_selector_walk(&predecessor, &successor, 4, 1, 0, 2, &ids, &unary, &gates)
            .unwrap();
    let depth = &depths[0];

    assert_eq!(depth.greedy_index, Some(0));
    assert_eq!(depth.greedy_token, Some(1));
    assert!(depth.issues.contains(&DFlash2SelectorIssue::DuplicateId {
        candidate_index: 1,
        first_index: 0,
        token_id: 1,
    }));
    assert!(depth.issues.contains(&DFlash2SelectorIssue::Sentinel {
        candidate_index: 2,
        token_id: -1,
    }));
    assert!(depth.issues.contains(&DFlash2SelectorIssue::Sentinel {
        candidate_index: 3,
        token_id: 9,
    }));
    assert_eq!(depth.final_scores, vec![Some(1.0), Some(1.0), None, None]);
}

#[test]
fn dflash2_selector_diagnostic_reports_nonfinite_and_no_choice() {
    let predecessor = selector_test_codebook(&[&[1.0], &[1.0], &[1.0]]);
    let successor = selector_test_codebook(&[&[0.0], &[0.0], &[0.0]]);
    let ids = [0, 0, 1, 2, 1, 2];
    let unary = [0.0, 0.0, f32::INFINITY, 0.0, f32::NAN, f32::NEG_INFINITY];
    let gates = [0.0, 1.0, 1.0];
    let (_, depths) =
        diagnose_dflash2_selector_walk(&predecessor, &successor, 2, 1, 0, 3, &ids, &unary, &gates)
            .unwrap();

    assert_eq!(depths[0].greedy_index, Some(0));
    assert!(matches!(
        depths[0].issues.as_slice(),
        [DFlash2SelectorIssue::NonFiniteScore {
            candidate_index: 0,
            token_id: 1,
            score,
        }] if score.is_infinite() && score.is_sign_positive()
    ));
    assert_eq!(depths[1].greedy_index, Some(0));
    assert_eq!(depths[1].greedy_token, Some(1));
    assert_eq!(depths[1].greedy_score, Some(f32::NEG_INFINITY));
    assert!(depths[1]
        .issues
        .iter()
        .any(|issue| matches!(issue, DFlash2SelectorIssue::NonFiniteScore { score, .. } if score.is_nan())));
    assert!(
        depths[1]
            .issues
            .contains(&DFlash2SelectorIssue::NoValidChoice)
    );
}

#[test]
fn dflash2_selector_diagnostic_returns_first_replay_mismatch() {
    let error = verify_dflash2_selector_replay(&[10, 20, 30], &[10, 21, 31]).unwrap_err();
    assert!(matches!(
        error,
        DFlashError::SelectorDiagnosticMismatch {
            depth: 1,
            production: 20,
            reconstructed: 21,
        }
    ));
}

#[test]
fn dflash2_selector_diagnostic_replay_rejects_invalid_predecessor() {
    let predecessor = selector_test_codebook(&[&[1.0], &[1.0]]);
    let successor = selector_test_codebook(&[&[0.0], &[0.0], &[1.0]]);
    let error = diagnose_dflash2_selector_walk(
        &predecessor,
        &successor,
        1,
        1,
        0,
        2,
        &[0, 2],
        &[0.0, 1.0],
        &[0.0, 1.0],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DFlashError::BadDrafter("selector codebook row out of range")
    ));
}

#[test]
fn dflash2_selector_diagnostic_checked_layout_honors_view_offset() {
    assert_eq!(
        checked_selector_read_layout(
            GgmlType::F32,
            GgmlType::F32,
            2,
            2,
            8,
            16,
            4,
            4,
            "synthetic_view",
        )
        .unwrap(),
        (8, 8)
    );
    assert!(
        checked_selector_read_layout(
            GgmlType::F32,
            GgmlType::F32,
            2,
            2,
            12,
            16,
            4,
            4,
            "synthetic_view",
        )
        .is_err()
    );
    assert!(
        checked_selector_read_layout(
            GgmlType::I32,
            GgmlType::F32,
            2,
            2,
            8,
            16,
            4,
            4,
            "synthetic_view",
        )
        .is_err()
    );
}

#[test]
fn dflash_context_estimate_is_exact_and_monotonic() {
    let cfg = memory_estimate_config();
    let at_10 = DFlashSessionGeometry::new(&cfg, 3, 16, 100, 10)
        .unwrap()
        .context_logical_bytes(cfg.n_layer)
        .unwrap();
    // 122 context-scaled F32 elements per position plus 68 fixed
    // elements from the two full-context buffers and position vector.
    assert_eq!(at_10, (122 * 10 + 68) * 4);
    let at_11 = DFlashSessionGeometry::new(&cfg, 3, 16, 100, 11)
        .unwrap()
        .context_logical_bytes(cfg.n_layer)
        .unwrap();
    assert_eq!(at_11 - at_10, 122 * 4);
    assert!(at_11 > at_10);
    assert!(DFlashSessionGeometry::new(&cfg, usize::MAX, u64::MAX, 100, 10).is_err());
}

#[test]
fn dflash_capture_window_span_and_complete_edges() {
    assert_eq!(
        dflash_capture_window_span(100, DFLASH_CAPTURE_WINDOW),
        (0, 100)
    );
    assert_eq!(
        dflash_capture_window_span(DFLASH_CAPTURE_WINDOW, DFLASH_CAPTURE_WINDOW),
        (0, DFLASH_CAPTURE_WINDOW)
    );
    assert_eq!(
        dflash_capture_window_span(3076, DFLASH_CAPTURE_WINDOW),
        (1028, DFLASH_CAPTURE_WINDOW)
    );
    assert_eq!(dflash_capture_window_span(3076, usize::MAX), (0, 3076));
    assert!(dflash_capture_window_complete(
        3076,
        1028,
        2048,
        DFLASH_CAPTURE_WINDOW
    ));
    assert!(!dflash_capture_window_complete(
        3076,
        0,
        2048,
        DFLASH_CAPTURE_WINDOW
    ));
    assert!(!dflash_capture_window_complete(
        3076,
        1028,
        2047,
        DFLASH_CAPTURE_WINDOW
    ));
    assert!(dflash_capture_window_complete(3076, 0, 3076, usize::MAX));
}

#[test]
fn dflash_swa_split4_scope_matches_measured_product_cell() {
    assert!(dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 2048, 0, true, 2048, 16
    ));
    assert!(dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 8853, 6805, true, 2048, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        false, 8, 32, 8, 128, 8853, 6805, true, 2048, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 16, 32, 8, 128, 8853, 6805, true, 2048, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 2047, 0, true, 2048, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 8853, 0, true, 2048, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 8853, 8837, true, 16, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 8853, 6805, true, 2048, 0
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 8, 24, 4, 256, 8853, 6805, true, 2048, 16
    ));
    assert!(!dflash_swa_split4_eligible(
        true, 8, 32, 8, 128, 8853, 6805, false, 2048, 16
    ));

    let exact_positions: Vec<i32> = (0..8853).collect();
    assert!(dflash_swa_exact_visible_suffix(
        &exact_positions,
        8853,
        6805,
        8853,
        2048,
    ));
    let mut gapped_positions = exact_positions;
    gapped_positions[7000] += 1;
    assert!(!dflash_swa_exact_visible_suffix(
        &gapped_positions,
        8853,
        6805,
        8853,
        2048,
    ));
}

#[test]
fn matrix_query_tiles_cover_nonzero_prefix_and_tail() {
    let chunk_start = 3072usize;
    let tiles: Vec<_> = attn_matrix_query_tiles(2050, 1024)
        .map(|(base, rows)| (base, rows, chunk_start + base))
        .collect();
    assert_eq!(
        tiles,
        vec![(0, 1024, 3072), (1024, 1024, 4096), (2048, 2, 5120)]
    );
}

#[test]
fn matrix_query_tiles_preserve_uncapped_shape() {
    assert_eq!(
        attn_matrix_query_tiles(1024, 4096).collect::<Vec<_>>(),
        vec![(0, 1024)]
    );
}

#[test]
fn matrix_vt_prefix_rebuild_span_is_exact() {
    assert_eq!(attn_matrix_vt_prefix_rebuild_rows(false, 0, 47), None);
    assert_eq!(attn_matrix_vt_prefix_rebuild_rows(true, 0, 0), None);
    assert_eq!(attn_matrix_vt_prefix_rebuild_rows(true, 0, 47), Some(47));
    assert_eq!(attn_matrix_vt_prefix_rebuild_rows(true, 46, 47), Some(47));
    assert_eq!(attn_matrix_vt_prefix_rebuild_rows(true, 47, 47), None);
    assert_eq!(attn_matrix_vt_prefix_rebuild_rows(true, 64, 47), None);
}

#[test]
fn prefill_command_completion_check_fails_closed() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(crate::metal::MetalError::EmptyLibrary | crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(error) => panic!("Metal context: {error}"),
    };
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    assert!(require_prefill_command_completed(&cmd).is_err());
    cmd.commit();
    cmd.waitUntilCompleted();
    require_prefill_command_completed(&cmd).unwrap();
}

#[test]
fn matrix_query_cap_parser_fails_closed() {
    assert_eq!(parse_prefill_attn_matrix_query_cap(None), Ok(None));
    assert_eq!(parse_prefill_attn_matrix_query_cap(Some("4")), Ok(Some(4)));
    for invalid in ["0", "-1", "junk", " 4", "4 ", "184467440737095516160"] {
        assert!(
            parse_prefill_attn_matrix_query_cap(Some(invalid)).is_err(),
            "accepted invalid query cap {invalid:?}"
        );
    }
}

#[test]
fn configured_matrix_query_cap_is_explicit_and_bounded() {
    let configured = Some(PrefillScratchConfig {
        matrix_query_cap: Some(1024),
    });
    let unavailable = || Err("invalid environment fallback".into());
    assert_eq!(
        resolve_prefill_attn_matrix_query_cap_with(false, configured, unavailable),
        Ok(Some(1024))
    );
    assert_eq!(
        resolve_prefill_attn_matrix_query_cap_with(true, configured, unavailable),
        Ok(None)
    );
    let zero = Some(PrefillScratchConfig {
        matrix_query_cap: Some(0),
    });
    assert!(resolve_prefill_attn_matrix_query_cap_with(false, zero, unavailable).is_err());
    assert_eq!(
        resolve_prefill_attn_matrix_query_cap_with(false, None, || Ok(Some(256))),
        Ok(Some(256))
    );
}

#[test]
fn prefill_scratch_overlay_layout_matches_product_profiles() {
    let cases = [
        (
            1024u64 * 32 * 11287,
            1024u64 * 32 * 177 * 2,
            [
                4096u64 * 12288,
                4096u64 * 8192,
                4096u64 * 64,
                4096u64 * 64,
                4096u64 * 2048,
                4096u64 * 2048,
                4096u64 * 8192,
                4096u64 * 8192,
                4096u64 * 8192,
            ],
            (786_104_320, 807_403_520, 807_403_520),
        ),
        (
            1024u64 * 16 * 11287,
            1024u64 * 16 * 177 * 2,
            [
                2048u64 * 8192,
                2048u64 * 4096,
                2048u64 * 32,
                2048u64 * 32,
                2048u64 * 2048,
                2048u64 * 2048,
                2048u64 * 4096,
                2048u64 * 4096,
                2048u64 * 4096,
            ],
            (393_052_160, 235_405_312, 393_052_160),
        ),
    ];
    for (scores, ml, gdn, expected) in cases {
        let layout = prefill_scratch_overlay_layout(scores, ml, gdn).unwrap();
        assert_eq!(
            (
                layout.attention_bytes,
                layout.gdn_bytes,
                layout.backing_bytes
            ),
            expected
        );
        assert_eq!(
            layout.attention_bytes + layout.gdn_bytes - layout.backing_bytes,
            layout.attention_bytes.min(layout.gdn_bytes)
        );
        let gdn_ranges = [
            layout.gdn_qkv,
            layout.gdn_z,
            layout.gdn_beta,
            layout.gdn_alpha,
            layout.gdn_q_norm,
            layout.gdn_k_norm,
            layout.gdn_v,
            layout.gdn_out,
            layout.gdn_normed,
        ];
        for (index, range) in gdn_ranges.iter().enumerate() {
            assert_eq!(range.offset % PREFILL_SCRATCH_OVERLAY_ALIGNMENT, 0);
            assert!(range.offset + range.bytes <= layout.backing_bytes);
            for other in &gdn_ranges[index + 1..] {
                assert!(range.offset + range.bytes <= other.offset);
            }
        }
        assert_eq!(layout.scores_h.offset, 0);
        assert!(layout.scores_h.offset + layout.scores_h.bytes <= layout.ml.offset);
        assert!(layout.ml.offset + layout.ml.bytes <= layout.backing_bytes);
    }
}

fn product_moe_plan_arch(a10b: bool) -> crate::model::Arch {
    let mut arch = crate::model::QWEN3_27B;
    arch.kind = crate::model::ArchKind::Moe;
    arch.n_layer = if a10b { 48 } else { 40 };
    arch.hidden_size = if a10b { 3072 } else { 2048 };
    arch.intermediate_size = 0;
    arch.n_q_heads = if a10b { 32 } else { 16 };
    arch.n_kv_heads = 2;
    arch.gdn_n_v_heads = if a10b { 64 } else { 32 };
    arch.expert_count = 256;
    arch.expert_used_count = 8;
    arch.expert_feed_forward_length = if a10b { 1024 } else { 512 };
    arch.expert_shared_feed_forward_length = if a10b { 1024 } else { 512 };
    arch.mtp_n_hidden_layers = 0;
    arch
}

fn product_moe_plan(
    arch: &crate::model::Arch,
    block_size: u32,
    configured_topology: bool,
) -> PrefillScratchPlan {
    let modes = PrefillScratchPlanModes {
        enable_attn_packed: true,
        enable_attn_fused_qkv_g8: false,
        enable_attn_matrix: true,
        attn_matrix_max_pos: 11_287,
        attn_matrix_online: true,
        attn_matrix_query_cap: configured_topology.then_some(1024),
        overlay_allowed: configured_topology,
        single_chunk_vt: false,
    };
    build_prefill_scratch_plan_from_arch(
        arch,
        u64::from(arch.n_layer / arch.full_attention_interval),
        true,
        block_size,
        false,
        modes,
    )
    .unwrap()
}

#[test]
fn single_chunk_vt_plan_shares_only_layer_storage() {
    let arch = crate::model::QWEN3_27B;
    let base = product_moe_plan(&arch, 32, true);
    for end in [8192, 8840, 32768] {
        let mut modes = base.modes;
        modes.attn_matrix_max_pos = end;
        let full = build_prefill_scratch_plan_from_arch(&arch, 16, true, 32, false, modes).unwrap();
        modes.single_chunk_vt = true;
        let single =
            build_prefill_scratch_plan_from_arch(&arch, 16, true, 32, false, modes).unwrap();
        assert_eq!(
            full.logical_bytes - single.logical_bytes,
            15 * 4 * 256 * end * 2
        );
        assert_eq!(full.allocations.len(), single.allocations.len());
        for (a, b) in full.allocations.iter().zip(&single.allocations) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.dtype, b.dtype);
            if a.name == "attn_matrix_vt_pack" {
                assert_eq!(a.logical_bytes, 16 * b.logical_bytes);
            } else {
                assert_eq!(a, b);
            }
        }
        assert_eq!(full.deferred_allocations, single.deferred_allocations);
        assert_eq!(full.matrix_query_rows, single.matrix_query_rows);
        assert_eq!(full.matrix_max_pos, single.matrix_max_pos);
        assert_eq!(modes.vt_layer(15), 0);
        assert_eq!(full.modes.vt_layer(15), 15);
        assert!(build_prefill_scratch_plan_from_arch(&arch, 16, true, 32, true, modes).is_err());
        assert!(build_prefill_scratch_plan_from_arch(&arch, 16, true, 0, false, modes).is_err());
        assert!(
            build_prefill_scratch_plan_from_arch(
                &product_moe_plan_arch(false),
                16,
                true,
                32,
                false,
                modes
            )
            .is_err()
        );
        modes.enable_attn_matrix = false;
        assert!(build_prefill_scratch_plan_from_arch(&arch, 16, true, 32, false, modes).is_err());
    }
}

#[test]
fn prefill_scratch_plan_matches_product_residuals() {
    for (a10b, wide_chunk, logical_residual, observed_residual) in [
        (false, 2048, 90_083_328u64, 90_079_232u64),
        (true, 4096, 876_916_736u64, 876_920_832u64),
    ] {
        let arch = product_moe_plan_arch(a10b);
        let baseline = product_moe_plan(&arch, 1024, false);
        let wide = product_moe_plan(&arch, wide_chunk, true);
        assert_eq!(
            wide.logical_bytes - baseline.logical_bytes,
            logical_residual
        );
        assert_eq!(logical_residual.abs_diff(observed_residual), 4096);
        assert_eq!(wide.matrix_query_rows, 1024);
        assert!(wide.overlay.is_some());
        let mut names = wide
            .allocations
            .iter()
            .chain(&wide.deferred_allocations)
            .map(|allocation| allocation.name)
            .collect::<Vec<_>>();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
        assert_eq!(
            wide.allocations
                .iter()
                .find(|allocation| allocation.name == "moe_group_slot_idx_pack")
                .expect("grouped gather IDs are planned")
                .dtype,
            GgmlType::I32
        );
        assert_eq!(
            wide.logical_bytes,
            wide.allocations
                .iter()
                .map(|allocation| allocation.logical_bytes)
                .sum::<u64>()
        );
        let mut priced_calls = 0usize;
        let priced = wide
            .priced_upper_bound(|bytes| {
                priced_calls += 1;
                bytes.checked_add(4096).ok_or_else(|| MetalError::BadShape {
                    kernel: "prefill_scratch_plan_test",
                    detail: "synthetic pricing overflow".into(),
                })
            })
            .unwrap();
        let maximum_count = wide.allocation_count() + wide.deferred_allocations.len();
        assert_eq!(priced_calls, maximum_count);
        assert_eq!(
            priced,
            wide.maximum_logical_bytes().unwrap() + 4096 * u64::try_from(maximum_count).unwrap()
        );
    }
}

#[test]
fn fresh_single_chunk_scratch_architecture_screen() {
    let arch = crate::model::QWEN3_27B;
    for width in [19, 32, 48] {
        let mut modes = resolve_prefill_scratch_plan_modes(
            &arch,
            width,
            false,
            Some(width as usize),
            Some(PrefillScratchConfig::default()),
        )
        .unwrap();
        modes.single_chunk_vt = true;
        let plan =
            build_prefill_scratch_plan_from_arch(&arch, 16, true, width, false, modes).unwrap();
        // A synthetic 16 KiB rounding screen, not driver pricing or admission.
        let rounded = plan
            .priced_upper_bound(|bytes| Ok(bytes.div_ceil(16384) * 16384))
            .unwrap();
        assert!(rounded <= 128 * 1024 * 1024);
        assert_eq!(plan.matrix_max_pos, width as u64);
        assert_eq!(plan.matrix_query_rows, width);
        eprintln!(
            "fresh-plan width={width} logical_bytes={} synthetic_rounded_bytes={rounded}",
            plan.logical_bytes
        );
    }
}

#[test]
fn prefill_overlay_backing_plan_allocates_with_matching_type() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal init: {e}"),
    };
    let arch = product_moe_plan_arch(false);
    let plan = product_moe_plan(&arch, 1, true);
    let allocation = plan.allocations.first().expect("overlay backing plan");
    assert_eq!(allocation.name, "attn_gdn_overlay_backing");
    assert_eq!(allocation.dtype, GgmlType::F32);
    assert!(allocation.logical_bytes.is_multiple_of(4));
    let mut allocator = PrefillScratchAllocator::new(&plan);
    let backing = allocator
        .f32(&ctx, allocation.name, vec![allocation.logical_bytes / 4])
        .expect("allocate overlay backing");
    assert_eq!(backing.dtype, GgmlType::F32);
    assert!(backing.buffer.length() as u64 >= allocation.logical_bytes);
}

fn write_tensor_f32(t: &MetalTensor, data: &[f32]) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

fn read_tensor_i32_f32buf(t: &MetalTensor) -> Vec<i32> {
    assert_eq!(t.dtype, GgmlType::F32);
    let n = t.n_elements() as usize;
    let mut out = vec![0i32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn read_tensor_i32(t: &MetalTensor) -> Vec<i32> {
    assert_eq!(t.dtype, GgmlType::I32);
    let n = t.n_elements() as usize;
    let mut out = vec![0i32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn write_tensor_i32_f32buf(t: &MetalTensor, data: &[i32]) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

fn write_tensor_i32(t: &MetalTensor, data: &[i32]) {
    assert_eq!(t.dtype, GgmlType::I32);
    assert_eq!(t.n_elements() as usize, data.len());
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }
}

#[derive(Clone, Copy, Debug)]
struct ExpertGroupRange {
    expert: usize,
    start: usize,
    len: usize,
}

fn build_expert_slot_groups(
    topk_idx: &[i32],
    topk_weight: &[f32],
    topk: usize,
    n_expert: usize,
) -> (Vec<ExpertGroupRange>, Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut buckets: Vec<Vec<(i32, i32, f32)>> = vec![Vec::new(); n_expert];
    for (slot, &expert_i) in topk_idx.iter().enumerate() {
        if expert_i < 0 {
            continue;
        }
        let expert = expert_i as usize;
        if expert >= n_expert {
            continue;
        }
        let token = (slot / topk) as i32;
        buckets[expert].push((slot as i32, token, topk_weight[slot]));
    }

    let mut ranges = Vec::new();
    let mut slot_ids = Vec::with_capacity(topk_idx.len());
    let mut token_ids = Vec::with_capacity(topk_idx.len());
    let mut weights = Vec::with_capacity(topk_idx.len());
    for (expert, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let start = slot_ids.len();
        for (slot_id, token_id, weight) in bucket {
            slot_ids.push(slot_id);
            token_ids.push(token_id);
            weights.push(weight);
        }
        ranges.push(ExpertGroupRange {
            expert,
            start,
            len: slot_ids.len() - start,
        });
    }
    (ranges, slot_ids, token_ids, weights)
}

fn timed_gpu_cmd<F>(ctx: &MetalContext, f: F) -> f64
where
    F: FnOnce(&KernelEncoder),
{
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    f(&enc);
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
}

fn run_packed_moe_tail_profile(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[packed-moe-tail-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, g_w, u_w, d_w, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => (
            &g.post_attn_norm,
            &g.ffn_gate,
            &g.ffn_up,
            &g.ffn_down,
            g.ffn_moe.as_ref().expect("moe block"),
        ),
        crate::metal_forward::MetalBlock::Attn(a) => (
            &a.post_attn_norm,
            &a.ffn_gate,
            &a.ffn_up,
            &a.ffn_down,
            a.ffn_moe.as_ref().expect("moe block"),
        ),
    };

    let mut session = MetalSession::fresh(&ctx, &mm, chunk_p + 16).expect("session");
    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    // Warmup pipeline cache.
    write_tensor_f32(&x_pack, &x_init);
    {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_rms_norm_batched_f32(
            &ctx,
            &enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("warmup postnorm");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    for n_idx in 0..chunk_p.min(2) {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_copy_offset_f32(&ctx, &enc, &x_pack, n_idx * h, &session.x, h).expect("copy x");
        encode_copy_offset_f32(&ctx, &enc, &h_pack, n_idx * h, &session.h, h).expect("copy h");
        mf.encode_moe_route_prepare(&enc, &mut session, moe)
            .expect("route");
        mf.encode_moe_routed_ffn_gpu(&enc, &mut session, moe)
            .expect("routed");
        mf.encode_moe_shared_ffn_gpu(&enc, &mut session, g_w, u_w, d_w)
            .expect("shared");
        encode_add_inplace_f32(&ctx, &enc, &session.x, &session.mixer_out).expect("resid");
        encode_scatter_offset_f32(&ctx, &enc, &session.x, &x_pack, n_idx * h, h).expect("scatter");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let mut postnorm_ms = 0.0f64;
    let mut route_ms = 0.0f64;
    let mut routed_ms = 0.0f64;
    let mut resid_scatter_ms = 0.0f64;
    let mut total_wall_ms = 0.0f64;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);

        let wall = Instant::now();
        {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            encode_rms_norm_batched_f32(
                &ctx,
                &enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            postnorm_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        }
        for n_idx in 0..chunk_p {
            {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                encode_copy_offset_f32(&ctx, &enc, &x_pack, n_idx * h, &session.x, h)
                    .expect("copy x");
                encode_copy_offset_f32(&ctx, &enc, &h_pack, n_idx * h, &session.h, h)
                    .expect("copy h");
                mf.encode_moe_route_prepare(&enc, &mut session, moe)
                    .expect("route");
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                route_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }
            {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_routed_ffn_gpu(&enc, &mut session, moe)
                    .expect("routed");
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                routed_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }
            {
                let cmd = ctx.queue.commandBuffer().expect("cmd");
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_shared_ffn_gpu(&enc, &mut session, g_w, u_w, d_w)
                    .expect("shared");
                encode_add_inplace_f32(&ctx, &enc, &session.x, &session.mixer_out).expect("resid");
                encode_scatter_offset_f32(&ctx, &enc, &session.x, &x_pack, n_idx * h, h)
                    .expect("scatter");
                enc.end();
                cmd.commit();
                cmd.waitUntilCompleted();
                resid_scatter_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
            }
        }
        total_wall_ms += wall.elapsed().as_secs_f64() * 1e3;
    }

    let denom = n_runs as f64;
    let postnorm_ms = postnorm_ms / denom;
    let route_ms = route_ms / denom;
    let routed_ms = routed_ms / denom;
    let resid_scatter_ms = resid_scatter_ms / denom;
    let total_gpu = postnorm_ms + route_ms + routed_ms + resid_scatter_ms;
    eprintln!(
        "[packed-moe-tail-{label}] chunk_p={chunk_p} avg wall={:.2} ms",
        total_wall_ms / denom
    );
    eprintln!(
        "[packed-moe-tail-{label}]   postnorm          {:6.2} ms ({:5.1}%)",
        postnorm_ms,
        postnorm_ms / total_gpu * 100.0
    );
    eprintln!(
        "[packed-moe-tail-{label}]   route+copy        {:6.2} ms ({:5.1}%)",
        route_ms,
        route_ms / total_gpu * 100.0
    );
    eprintln!(
        "[packed-moe-tail-{label}]   routed_ffn        {:6.2} ms ({:5.1}%)",
        routed_ms,
        routed_ms / total_gpu * 100.0
    );
    eprintln!(
        "[packed-moe-tail-{label}]   shared+resid+copy {:6.2} ms ({:5.1}%)",
        resid_scatter_ms,
        resid_scatter_ms / total_gpu * 100.0
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_packed_moe_tail_profile() {
    run_packed_moe_tail_profile(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 8, 4);
}

#[test]
#[ignore]
fn metal_122b_a10b_packed_moe_tail_profile() {
    run_packed_moe_tail_profile(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b", 8, 3);
}

fn run_packed_moe_tail_ab_profile(
    model_path: &str,
    label: &str,
    chunk_ps: &[usize],
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[moe-tail-ab-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, g_w, u_w, d_w, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => (
            &g.post_attn_norm,
            &g.ffn_gate,
            &g.ffn_up,
            &g.ffn_down,
            g.ffn_moe.as_ref().expect("moe block"),
        ),
        crate::metal_forward::MetalBlock::Attn(a) => (
            &a.post_attn_norm,
            &a.ffn_gate,
            &a.ffn_up,
            &a.ffn_down,
            a.ffn_moe.as_ref().expect("moe block"),
        ),
    };

    for &chunk_p in chunk_ps {
        let mut session = MetalSession::fresh(&ctx, &mm, chunk_p + 16).expect("session");
        let scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
        let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
        let router_probs_pack = scratch
            .moe_router_probs_pack
            .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
        let topk_idx_pack = scratch
            .moe_topk_idx_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let topk_weight_pack = scratch
            .moe_topk_weight_pack
            .view_subrange(0, vec![(chunk_p * topk) as u64]);
        let shared_gate_pack = scratch
            .moe_shared_gate_pack
            .view_subrange(0, vec![chunk_p as u64]);
        let moe_inner_pack = scratch
            .moe_inner_pack
            .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
        let mixer_out_pack = scratch
            .mixer_out_pack
            .view_subrange(0, vec![(chunk_p * h) as u64]);
        let x_init: Vec<f32> = (0..chunk_p * h)
            .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
            .collect();

        let mut old_postnorm = 0.0f64;
        let mut old_route = 0.0f64;
        let mut old_routed = 0.0f64;
        let mut old_shared = 0.0f64;
        let mut old_wall = 0.0f64;
        let mut new_postnorm = 0.0f64;
        let mut new_route = 0.0f64;
        let mut new_swiglu = 0.0f64;
        let mut new_down = 0.0f64;
        let mut new_shared = 0.0f64;
        let mut new_wall = 0.0f64;
        let mut old_one_cb = 0.0f64;
        let mut new_one_cb = 0.0f64;

        for run_idx in 0..=n_runs {
            let sample = run_idx > 0;

            write_tensor_f32(&x_pack, &x_init);
            let old_wall_start = Instant::now();
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("old postnorm");
            });
            if sample {
                old_postnorm += ms;
            }
            for n_idx in 0..chunk_p {
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                        .expect("old copy x");
                    encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                        .expect("old copy h");
                    mf.encode_moe_route_prepare(enc, &mut session, moe)
                        .expect("old route");
                });
                if sample {
                    old_route += ms;
                }
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    mf.encode_moe_routed_ffn_gpu(enc, &mut session, moe)
                        .expect("old routed");
                });
                if sample {
                    old_routed += ms;
                }
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                        .expect("old shared");
                    encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                        .expect("old resid");
                    encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                        .expect("old scatter");
                });
                if sample {
                    old_shared += ms;
                }
            }
            if sample {
                old_wall += old_wall_start.elapsed().as_secs_f64() * 1e3;
            }

            write_tensor_f32(&x_pack, &x_init);
            let new_wall_start = Instant::now();
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("new postnorm");
            });
            if sample {
                new_postnorm += ms;
            }
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("new router matmat");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("new packed topk/shared gate");
            });
            if sample {
                new_route += ms;
            }
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("new packed swiglu");
            });
            if sample {
                new_swiglu += ms;
            }
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("new packed down+sum");
            });
            if sample {
                new_down += ms;
            }
            for n_idx in 0..chunk_p {
                let mixer_n = mixer_out_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                let shared_gate_n = shared_gate_pack.view_subrange(n_idx as u64, vec![1]);
                let ms = timed_gpu_cmd(&ctx, |enc| {
                    encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                        .expect("new copy x");
                    encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                        .expect("new copy h");
                    encode_copy_offset_f32(&ctx, enc, &mixer_n, 0, &session.mixer_out, h)
                        .expect("new copy routed out");
                    encode_copy_offset_f32(
                        &ctx,
                        enc,
                        &shared_gate_n,
                        0,
                        &session.moe_shared_gate,
                        1,
                    )
                    .expect("new copy shared gate");
                    mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                        .expect("new shared");
                    encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                        .expect("new resid");
                    encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                        .expect("new scatter");
                });
                if sample {
                    new_shared += ms;
                }
            }
            if sample {
                new_wall += new_wall_start.elapsed().as_secs_f64() * 1e3;
            }

            write_tensor_f32(&x_pack, &x_init);
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("old-cb postnorm");
                for n_idx in 0..chunk_p {
                    encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                        .expect("old-cb copy x");
                    encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                        .expect("old-cb copy h");
                    mf.encode_moe_route_prepare(enc, &mut session, moe)
                        .expect("old-cb route");
                    mf.encode_moe_routed_ffn_gpu(enc, &mut session, moe)
                        .expect("old-cb routed");
                    mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                        .expect("old-cb shared");
                    encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                        .expect("old-cb resid");
                    encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                        .expect("old-cb scatter");
                }
            });
            if sample {
                old_one_cb += ms;
            }

            write_tensor_f32(&x_pack, &x_init);
            let ms = timed_gpu_cmd(&ctx, |enc| {
                encode_rms_norm_batched_f32(
                    &ctx,
                    enc,
                    &x_pack,
                    post_norm,
                    &h_pack,
                    chunk_p,
                    h,
                    crate::metal_forward::RMS_EPS,
                )
                .expect("new-cb postnorm");
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    &moe.gate_inp,
                    &h_pack,
                    &router_probs_pack,
                    h,
                    n_expert,
                    chunk_p,
                )
                .expect("new-cb router matmat");
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("new-cb packed topk/shared gate");
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &topk_idx_pack,
                    &moe_inner_pack,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("new-cb packed swiglu");
                encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &moe_inner_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &mixer_out_pack,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("new-cb packed down+sum");
                for n_idx in 0..chunk_p {
                    let mixer_n = mixer_out_pack.view_subrange((n_idx * h) as u64, vec![h as u64]);
                    let shared_gate_n = shared_gate_pack.view_subrange(n_idx as u64, vec![1]);
                    encode_copy_offset_f32(&ctx, enc, &x_pack, n_idx * h, &session.x, h)
                        .expect("new-cb copy x");
                    encode_copy_offset_f32(&ctx, enc, &h_pack, n_idx * h, &session.h, h)
                        .expect("new-cb copy h");
                    encode_copy_offset_f32(&ctx, enc, &mixer_n, 0, &session.mixer_out, h)
                        .expect("new-cb copy routed out");
                    encode_copy_offset_f32(
                        &ctx,
                        enc,
                        &shared_gate_n,
                        0,
                        &session.moe_shared_gate,
                        1,
                    )
                    .expect("new-cb copy shared gate");
                    mf.encode_moe_shared_ffn_gpu(enc, &mut session, g_w, u_w, d_w)
                        .expect("new-cb shared");
                    encode_add_inplace_f32(&ctx, enc, &session.x, &session.mixer_out)
                        .expect("new-cb resid");
                    encode_scatter_offset_f32(&ctx, enc, &session.x, &x_pack, n_idx * h, h)
                        .expect("new-cb scatter");
                }
            });
            if sample {
                new_one_cb += ms;
            }
        }

        let denom = n_runs as f64;
        let old_postnorm = old_postnorm / denom;
        let old_route = old_route / denom;
        let old_routed = old_routed / denom;
        let old_shared = old_shared / denom;
        let old_wall = old_wall / denom;
        let new_postnorm = new_postnorm / denom;
        let new_route = new_route / denom;
        let new_swiglu = new_swiglu / denom;
        let new_down = new_down / denom;
        let new_shared = new_shared / denom;
        let new_wall = new_wall / denom;
        let old_one_cb = old_one_cb / denom;
        let new_one_cb = new_one_cb / denom;
        let old_gpu = old_postnorm + old_route + old_routed + old_shared;
        let new_gpu = new_postnorm + new_route + new_swiglu + new_down + new_shared;
        eprintln!(
            "[moe-tail-ab-{label}] P={chunk_p} split_old_gpu={old_gpu:.2} ms split_new_gpu={new_gpu:.2} ms split_speedup={:.3} old_wall={old_wall:.2} ms new_wall={new_wall:.2} ms",
            old_gpu / new_gpu
        );
        eprintln!(
            "[moe-tail-ab-{label}]   one_cb old={old_one_cb:.2} ms new={new_one_cb:.2} ms speedup={:.3}",
            old_one_cb / new_one_cb
        );
        eprintln!(
            "[moe-tail-ab-{label}]   old postnorm={old_postnorm:.2} route+copy={old_route:.2} routed={old_routed:.2} shared+resid+copy={old_shared:.2}"
        );
        eprintln!(
            "[moe-tail-ab-{label}]   new postnorm={new_postnorm:.2} packed_route={new_route:.2} swiglu={new_swiglu:.2} down_sum={new_down:.2} shared+resid+copy={new_shared:.2}"
        );
    }
}

#[test]
#[ignore]
fn metal_35b_a3b_packed_moe_tail_ab_profile() {
    run_packed_moe_tail_ab_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b",
        &[8, 16, 64, 128, 320],
        3,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_packed_moe_tail_ab_profile() {
    run_packed_moe_tail_ab_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        &[8, 64, 128, 320],
        2,
    );
}

fn run_live_packed_moe_tail_phase_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[live-packed-moe-tail-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let f_shared = arch.expert_shared_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, g_w, u_w, d_w, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => (
            &g.post_attn_norm,
            &g.ffn_gate,
            &g.ffn_up,
            &g.ffn_down,
            g.ffn_moe.as_ref().expect("moe block"),
        ),
        crate::metal_forward::MetalBlock::Attn(a) => (
            &a.post_attn_norm,
            &a.ffn_gate,
            &a.ffn_up,
            &a.ffn_down,
            a.ffn_moe.as_ref().expect("moe block"),
        ),
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let mixer_out_pack = scratch
        .mixer_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let shared_ffn_gate_pack = scratch
        .moe_shared_ffn_gate_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_ffn_up_pack = scratch
        .moe_shared_ffn_up_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_ffn_inner_pack = scratch
        .moe_shared_ffn_inner_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_ffn_out_pack = scratch
        .moe_shared_ffn_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    write_tensor_f32(&x_pack, &x_init);
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("warmup postnorm");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("warmup route");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("warmup topk/shared");
        encode_moe_swiglu_q4_K_f32_packed_slots(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &topk_idx_pack,
            &moe_inner_pack,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("warmup swiglu");
        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
            &ctx,
            enc,
            &moe.down_exps,
            &moe_inner_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &mixer_out_pack,
            f_exp,
            h,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("warmup down");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            g_w,
            &h_pack,
            &shared_ffn_gate_pack,
            h,
            f_shared,
            chunk_p,
        )
        .expect("warmup shared gate");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            u_w,
            &h_pack,
            &shared_ffn_up_pack,
            h,
            f_shared,
            chunk_p,
        )
        .expect("warmup shared up");
        encode_silu_mul_f32(
            &ctx,
            enc,
            &shared_ffn_gate_pack,
            &shared_ffn_up_pack,
            &shared_ffn_inner_pack,
        )
        .expect("warmup shared silu");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            d_w,
            &shared_ffn_inner_pack,
            &shared_ffn_out_pack,
            f_shared,
            h,
            chunk_p,
        )
        .expect("warmup shared down");
        encode_axpy_rowwise_f32(
            &ctx,
            enc,
            &shared_ffn_out_pack,
            &shared_gate_pack,
            &mixer_out_pack,
            h,
            chunk_p,
        )
        .expect("warmup shared axpy");
        encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("warmup resid");
    });

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("hist postnorm");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("hist route");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("hist topk/shared");
    });
    let idxs = read_tensor_i32_f32buf(&topk_idx_pack);
    let mut counts = vec![0usize; n_expert];
    for &idx in &idxs {
        if idx >= 0 && (idx as usize) < n_expert {
            counts[idx as usize] += 1;
        }
    }
    let active = counts.iter().filter(|&&c| c > 0).count();
    let mut nonzero: Vec<usize> = counts.iter().copied().filter(|&c| c > 0).collect();
    nonzero.sort_unstable();
    let max_slots = nonzero.last().copied().unwrap_or(0);
    let p50 = nonzero.get(nonzero.len() / 2).copied().unwrap_or(0);
    let p90 = nonzero
        .get((nonzero.len().saturating_sub(1) * 9) / 10)
        .copied()
        .unwrap_or(0);
    let mean_active = if active > 0 {
        idxs.len() as f64 / active as f64
    } else {
        0.0
    };
    eprintln!(
        "[live-packed-moe-tail-{label}] expert bucket stats: slots={} active_experts={} mean_active={mean_active:.2} p50={} p90={} max={}",
        idxs.len(),
        active,
        p50,
        p90,
        max_slots
    );

    let mut postnorm_ms = 0.0f64;
    let mut route_ms = 0.0f64;
    let mut routed_swiglu_ms = 0.0f64;
    let mut routed_down_ms = 0.0f64;
    let mut shared_gateup_silu_ms = 0.0f64;
    let mut shared_down_ms = 0.0f64;
    let mut shared_axpy_resid_ms = 0.0f64;
    let mut one_cb_ms = 0.0f64;
    let mut wall_ms = 0.0f64;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let wall = Instant::now();

        postnorm_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
        });

        route_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route logits");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared gate");
        });

        routed_swiglu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("routed swiglu");
        });

        routed_down_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &mixer_out_pack,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("routed down");
        });

        shared_gateup_silu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                g_w,
                &h_pack,
                &shared_ffn_gate_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("shared gate");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                u_w,
                &h_pack,
                &shared_ffn_up_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("shared up");
            encode_silu_mul_f32(
                &ctx,
                enc,
                &shared_ffn_gate_pack,
                &shared_ffn_up_pack,
                &shared_ffn_inner_pack,
            )
            .expect("shared silu");
        });

        shared_down_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                d_w,
                &shared_ffn_inner_pack,
                &shared_ffn_out_pack,
                f_shared,
                h,
                chunk_p,
            )
            .expect("shared down");
        });

        shared_axpy_resid_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_axpy_rowwise_f32(
                &ctx,
                enc,
                &shared_ffn_out_pack,
                &shared_gate_pack,
                &mixer_out_pack,
                h,
                chunk_p,
            )
            .expect("shared axpy");
            encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("resid add");
        });

        one_cb_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("one-cb postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("one-cb route logits");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("one-cb topk/shared gate");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("one-cb routed swiglu");
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &mixer_out_pack,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("one-cb routed down");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                g_w,
                &h_pack,
                &shared_ffn_gate_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("one-cb shared gate");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                u_w,
                &h_pack,
                &shared_ffn_up_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("one-cb shared up");
            encode_silu_mul_f32(
                &ctx,
                enc,
                &shared_ffn_gate_pack,
                &shared_ffn_up_pack,
                &shared_ffn_inner_pack,
            )
            .expect("one-cb shared silu");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                d_w,
                &shared_ffn_inner_pack,
                &shared_ffn_out_pack,
                f_shared,
                h,
                chunk_p,
            )
            .expect("one-cb shared down");
            encode_axpy_rowwise_f32(
                &ctx,
                enc,
                &shared_ffn_out_pack,
                &shared_gate_pack,
                &mixer_out_pack,
                h,
                chunk_p,
            )
            .expect("one-cb shared axpy");
            encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("one-cb resid");
        });

        wall_ms += wall.elapsed().as_secs_f64() * 1e3;
    }

    let denom = n_runs as f64;
    let postnorm_ms = postnorm_ms / denom;
    let route_ms = route_ms / denom;
    let routed_swiglu_ms = routed_swiglu_ms / denom;
    let routed_down_ms = routed_down_ms / denom;
    let shared_gateup_silu_ms = shared_gateup_silu_ms / denom;
    let shared_down_ms = shared_down_ms / denom;
    let shared_axpy_resid_ms = shared_axpy_resid_ms / denom;
    let one_cb_ms = one_cb_ms / denom;
    let wall_ms = wall_ms / denom;
    let split_gpu = postnorm_ms
        + route_ms
        + routed_swiglu_ms
        + routed_down_ms
        + shared_gateup_silu_ms
        + shared_down_ms
        + shared_axpy_resid_ms;
    eprintln!(
        "[live-packed-moe-tail-{label}] chunk_p={chunk_p} avg wall={wall_ms:.2} ms split_gpu={split_gpu:.2} ms one_cb={one_cb_ms:.2} ms"
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   postnorm           {:6.2} ms ({:5.1}%)",
        postnorm_ms,
        postnorm_ms / split_gpu * 100.0
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   route+topk+sgate   {:6.2} ms ({:5.1}%)",
        route_ms,
        route_ms / split_gpu * 100.0
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   routed_swiglu      {:6.2} ms ({:5.1}%)",
        routed_swiglu_ms,
        routed_swiglu_ms / split_gpu * 100.0
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   routed_down_sum    {:6.2} ms ({:5.1}%)",
        routed_down_ms,
        routed_down_ms / split_gpu * 100.0
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   shared_gateup_silu {:6.2} ms ({:5.1}%)",
        shared_gateup_silu_ms,
        shared_gateup_silu_ms / split_gpu * 100.0
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   shared_down        {:6.2} ms ({:5.1}%)",
        shared_down_ms,
        shared_down_ms / split_gpu * 100.0
    );
    eprintln!(
        "[live-packed-moe-tail-{label}]   shared_axpy+resid  {:6.2} ms ({:5.1}%)",
        shared_axpy_resid_ms,
        shared_axpy_resid_ms / split_gpu * 100.0
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_live_packed_moe_tail_phase_profile() {
    run_live_packed_moe_tail_phase_profile(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 320, 3);
}

#[test]
#[ignore]
fn metal_122b_a10b_live_packed_moe_tail_phase_profile() {
    run_live_packed_moe_tail_phase_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        2,
    );
}

fn run_live_grouped_moe_tail_phase_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[live-grouped-moe-tail-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let f_shared = arch.expert_shared_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, g_w, u_w, d_w, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => (
            &g.post_attn_norm,
            &g.ffn_gate,
            &g.ffn_up,
            &g.ffn_down,
            g.ffn_moe.as_ref().expect("moe block"),
        ),
        crate::metal_forward::MetalBlock::Attn(a) => (
            &a.post_attn_norm,
            &a.ffn_gate,
            &a.ffn_up,
            &a.ffn_down,
            a.ffn_moe.as_ref().expect("moe block"),
        ),
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_group_count_pack = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let moe_group_ids_pack = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let moe_group_inner_pack = scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let moe_group_out_pack = scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let mixer_out_pack = scratch
        .mixer_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let shared_ffn_gate_pack = scratch
        .moe_shared_ffn_gate_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_ffn_up_pack = scratch
        .moe_shared_ffn_up_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_ffn_inner_pack = scratch
        .moe_shared_ffn_inner_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_ffn_out_pack = scratch
        .moe_shared_ffn_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    let fused_route_bucket = prefill_moe_route_bucket_fused_auto_enabled(h, chunk_p);

    let mut postnorm_ms = 0.0f64;
    let mut route_logits_ms = 0.0f64;
    let mut route_select_ms = 0.0f64;
    let mut route_bucket_ms = 0.0f64;
    let mut grouped_swiglu_ms = 0.0f64;
    let mut grouped_down_ms = 0.0f64;
    let mut grouped_reduce_ms = 0.0f64;
    let mut shared_gateup_silu_ms = 0.0f64;
    let mut shared_down_ms = 0.0f64;
    let mut shared_axpy_resid_ms = 0.0f64;
    let mut wall_ms = 0.0f64;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let wall = Instant::now();

        postnorm_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
        });

        route_logits_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_route_logits_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route logits");
        });

        if fused_route_bucket {
            route_select_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_fill_f32(&ctx, enc, &moe_group_count_pack, 0.0)
                    .expect("zero fused route counts");
                crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("fused topk+bucket");
            });
        } else {
            route_select_ms += timed_gpu_cmd(&ctx, |enc| {
                encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                    &ctx,
                    enc,
                    &router_probs_pack,
                    &moe.gate_inp_shexp,
                    &h_pack,
                    &topk_idx_pack,
                    &topk_weight_pack,
                    &shared_gate_pack,
                    n_expert,
                    topk,
                    h,
                    chunk_p,
                )
                .expect("topk/shared gate");
            });
            route_bucket_ms += timed_gpu_cmd(&ctx, |enc| {
                crate::metal::encode_moe_route_bucket_slots_f32(
                    &ctx,
                    enc,
                    &topk_idx_pack,
                    &moe_group_count_pack,
                    &moe_group_ids_pack,
                    n_expert,
                    chunk_p,
                    topk,
                )
                .expect("bucket slots");
            });
        }

        grouped_swiglu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &moe_group_inner_pack, 0.0).expect("zero grouped inner");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &moe_group_count_pack,
                &moe_group_ids_pack,
                &moe_group_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu");
        });

        grouped_down_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &moe_group_out_pack, 0.0).expect("zero grouped out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_group_inner_pack,
                &moe_group_count_pack,
                &moe_group_ids_pack,
                &moe_group_out_pack,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down");
        });

        grouped_reduce_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &moe_group_out_pack,
                &topk_weight_pack,
                &mixer_out_pack,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce");
        });

        shared_gateup_silu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                g_w,
                &h_pack,
                &shared_ffn_gate_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("shared gate");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                u_w,
                &h_pack,
                &shared_ffn_up_pack,
                h,
                f_shared,
                chunk_p,
            )
            .expect("shared up");
            encode_silu_mul_f32(
                &ctx,
                enc,
                &shared_ffn_gate_pack,
                &shared_ffn_up_pack,
                &shared_ffn_inner_pack,
            )
            .expect("shared silu");
        });

        shared_down_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                d_w,
                &shared_ffn_inner_pack,
                &shared_ffn_out_pack,
                f_shared,
                h,
                chunk_p,
            )
            .expect("shared down");
        });

        shared_axpy_resid_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_axpy_rowwise_f32(
                &ctx,
                enc,
                &shared_ffn_out_pack,
                &shared_gate_pack,
                &mixer_out_pack,
                h,
                chunk_p,
            )
            .expect("shared axpy");
            encode_add_inplace_f32(&ctx, enc, &x_pack, &mixer_out_pack).expect("resid");
        });

        wall_ms += wall.elapsed().as_secs_f64() * 1e3;
    }

    let denom = n_runs as f64;
    let total = postnorm_ms
        + route_logits_ms
        + route_select_ms
        + route_bucket_ms
        + grouped_swiglu_ms
        + grouped_down_ms
        + grouped_reduce_ms
        + shared_gateup_silu_ms
        + shared_down_ms
        + shared_axpy_resid_ms;
    eprintln!(
        "[live-grouped-moe-tail-{label}] chunk_p={chunk_p} avg wall={:.2} ms split_gpu={:.2} ms",
        wall_ms / denom,
        total / denom,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   postnorm           {:6.2} ms ({:5.1}%)",
        postnorm_ms / denom,
        postnorm_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   route_logits       {:6.2} ms ({:5.1}%)",
        route_logits_ms / denom,
        route_logits_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   route_select       {:6.2} ms ({:5.1}%)",
        route_select_ms / denom,
        route_select_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   route_bucket       {:6.2} ms ({:5.1}%)",
        route_bucket_ms / denom,
        route_bucket_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   grouped_swiglu     {:6.2} ms ({:5.1}%)",
        grouped_swiglu_ms / denom,
        grouped_swiglu_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   grouped_down       {:6.2} ms ({:5.1}%)",
        grouped_down_ms / denom,
        grouped_down_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   grouped_reduce     {:6.2} ms ({:5.1}%)",
        grouped_reduce_ms / denom,
        grouped_reduce_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   shared_gateup_silu {:6.2} ms ({:5.1}%)",
        shared_gateup_silu_ms / denom,
        shared_gateup_silu_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   shared_down        {:6.2} ms ({:5.1}%)",
        shared_down_ms / denom,
        shared_down_ms / total * 100.0,
    );
    eprintln!(
        "[live-grouped-moe-tail-{label}]   shared_axpy+resid  {:6.2} ms ({:5.1}%)",
        shared_axpy_resid_ms / denom,
        shared_axpy_resid_ms / total * 100.0,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_live_grouped_moe_tail_phase_profile_320() {
    run_live_grouped_moe_tail_phase_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-320",
        320,
        2,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_live_grouped_moe_tail_phase_profile_320() {
    run_live_grouped_moe_tail_phase_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-320",
        320,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_live_grouped_moe_tail_phase_profile_512() {
    run_live_grouped_moe_tail_phase_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-512",
        512,
        2,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_live_grouped_moe_tail_phase_profile_512() {
    run_live_grouped_moe_tail_phase_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-512",
        512,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_live_grouped_moe_tail_phase_profile_1024() {
    run_live_grouped_moe_tail_phase_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-1024",
        1024,
        2,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_live_grouped_moe_tail_phase_profile_1024() {
    run_live_grouped_moe_tail_phase_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-1024",
        1024,
        2,
    );
}

fn run_moe_route_bucket_fused_oracle(model_path: &str, label: &str, chunk_p: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[moe-route-bucket-fused-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };
    assert_eq!(moe.gate_exps.dtype, GgmlType::Q4_K, "gate dtype");
    assert_eq!(moe.up_exps.dtype, GgmlType::Q4_K, "up dtype");
    assert_eq!(moe.down_exps.dtype, GgmlType::Q5_K, "down dtype");

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let split_idx = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let split_w = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let split_gate = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let split_counts = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let split_ids = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let fused_idx = scratch
        .moe_group_slot_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let fused_w = scratch
        .moe_group_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let fused_gate = scratch
        .moe_group_token_idx_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let fused_counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("fused_counts");
    let fused_ids =
        MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("fused_ids");
    let f_exp = arch.expert_feed_forward_length as usize;
    let split_inner = scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let fused_inner = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let split_out = scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let fused_out = scratch
        .moe_expert_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let split_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("split_reduced");
    let fused_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("fused_reduced");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    write_tensor_f32(&x_pack, &x_init);

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("postnorm");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("route logits");
    });

    let _ = timed_gpu_cmd(&ctx, |enc| {
        crate::metal::encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &split_idx,
            &split_w,
            &split_gate,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("split topk/shared");
        crate::metal::encode_moe_route_bucket_slots_f32(
            &ctx,
            enc,
            &split_idx,
            &split_counts,
            &split_ids,
            n_expert,
            chunk_p,
            topk,
        )
        .expect("split bucket");
    });

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_fill_f32(&ctx, enc, &fused_counts, 0.0).expect("zero fused counts");
        crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &fused_idx,
            &fused_w,
            &fused_gate,
            &fused_counts,
            &fused_ids,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("fused topk+bucket");
    });

    let split_idx_cpu = read_tensor_i32_f32buf(&split_idx);
    let fused_idx_cpu = read_tensor_i32(&fused_idx);
    let split_w_cpu = read_tensor_f32(&split_w);
    let fused_w_cpu = read_tensor_f32(&fused_w);
    let split_gate_cpu = read_tensor_f32(&split_gate);
    let fused_gate_cpu = read_tensor_f32(&fused_gate);
    let split_counts_cpu = cpu_read_i32_f32buf(&split_counts);
    let fused_counts_cpu = cpu_read_i32_f32buf(&fused_counts);
    let split_ids_cpu = cpu_read_i32_f32buf(&split_ids);
    let fused_ids_cpu = cpu_read_i32_f32buf(&fused_ids);

    assert_eq!(split_idx_cpu, fused_idx_cpu, "topk idx mismatch");
    assert_eq!(split_counts_cpu, fused_counts_cpu, "bucket counts mismatch");
    let slot_count = chunk_p * topk;
    let total_count: usize = split_counts_cpu.iter().map(|&c| c.max(0) as usize).sum();
    let active_experts = split_counts_cpu.iter().filter(|&&c| c > 0).count();
    for (expert, &count) in split_counts_cpu.iter().enumerate().take(n_expert) {
        let count = count as usize;
        let base = expert * chunk_p;
        let split_slice = &split_ids_cpu[base..base + count];
        let fused_slice = &fused_ids_cpu[base..base + count];
        for &slot in fused_slice {
            assert!(
                slot >= 0 && (slot as usize) < slot_count,
                "fused bucket slot out of range: expert={expert} slot={slot} slot_count={slot_count}"
            );
            assert_eq!(
                fused_idx_cpu[slot as usize], expert as i32,
                "fused bucket expert mismatch: expert={expert} slot={slot} slot_expert={} count={count}",
                fused_idx_cpu[slot as usize]
            );
        }
        let mut split_sorted = split_slice.to_vec();
        let mut fused_sorted = fused_slice.to_vec();
        split_sorted.sort_unstable();
        fused_sorted.sort_unstable();
        assert_eq!(
            split_sorted,
            fused_sorted,
            "bucket content mismatch for expert {expert}: split={:?} fused={:?}",
            &split_sorted[..split_sorted.len().min(16)],
            &fused_sorted[..fused_sorted.len().min(16)]
        );
    }
    let cos_w = cosine_f32(&split_w_cpu, &fused_w_cpu);
    let cos_gate = cosine_f32(&split_gate_cpu, &fused_gate_cpu);
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_fill_f32(&ctx, enc, &split_inner, 0.0).expect("zero split inner");
        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &split_counts,
            &split_ids,
            &split_inner,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("split swiglu");
        encode_fill_f32(&ctx, enc, &split_out, 0.0).expect("zero split out");
        crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
            &ctx,
            enc,
            &moe.down_exps,
            &split_inner,
            &split_counts,
            &split_ids,
            &split_out,
            f_exp,
            h,
            n_expert,
            chunk_p,
        )
        .expect("split down");
        crate::metal::encode_moe_weighted_sum_packed_f32(
            &ctx,
            enc,
            &split_out,
            &split_w,
            &split_reduced,
            h,
            topk,
            chunk_p,
        )
        .expect("split reduce");
    });
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_fill_f32(&ctx, enc, &fused_inner, 0.0).expect("zero fused inner");
        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &fused_counts,
            &fused_ids,
            &fused_inner,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("fused swiglu");
        encode_fill_f32(&ctx, enc, &fused_out, 0.0).expect("zero fused out");
        crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
            &ctx,
            enc,
            &moe.down_exps,
            &fused_inner,
            &fused_counts,
            &fused_ids,
            &fused_out,
            f_exp,
            h,
            n_expert,
            chunk_p,
        )
        .expect("fused down");
        crate::metal::encode_moe_weighted_sum_packed_f32(
            &ctx,
            enc,
            &fused_out,
            &fused_w,
            &fused_reduced,
            h,
            topk,
            chunk_p,
        )
        .expect("fused reduce");
    });
    let split_reduced_cpu = read_tensor_f32(&split_reduced);
    let fused_reduced_cpu = read_tensor_f32(&fused_reduced);
    let cos_reduced = cosine_f32(&split_reduced_cpu, &fused_reduced_cpu);
    eprintln!(
        "[moe-route-bucket-fused-{label}] total_count={} active_experts={} cos(topk_w)={cos_w:.6} cos(shared_gate)={cos_gate:.6} cos(reduced)={cos_reduced:.6}",
        total_count, active_experts,
    );
    assert!(cos_w > 0.999999, "topk_w cos too low: {cos_w}");
    assert!(cos_gate > 0.999999, "shared_gate cos too low: {cos_gate}");
    assert!(cos_reduced > 0.999999, "reduced cos too low: {cos_reduced}");
}

fn run_moe_route_logits_e8p32_oracle(model_path: &str, label: &str, chunk_p: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[moe-route-e8p32-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };
    assert_eq!(
        moe.gate_inp.dtype,
        GgmlType::F32,
        "route logits oracle expects F32 router"
    );
    assert_eq!(
        n_expert % 8,
        0,
        "route logits oracle expects expert_count % 8 == 0"
    );
    assert_eq!(h % 4, 0, "route logits oracle expects hidden % 4 == 0");

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let probs_generic = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let probs_e8 =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * n_expert) as u64]).expect("probs_e8");
    let idx_generic = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let w_generic = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let gate_generic = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let idx_e8 = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64]).expect("idx_e8");
    let w_e8 = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64]).expect("w_e8");
    let gate_e8 = MetalTensor::zeros_f32(&ctx, vec![chunk_p as u64]).expect("gate_e8");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    write_tensor_f32(&x_pack, &x_init);

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("postnorm");
    });

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &probs_generic,
            h,
            n_expert,
            chunk_p,
        )
        .expect("generic route logits");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &probs_generic,
            &moe.gate_inp_shexp,
            &h_pack,
            &idx_generic,
            &w_generic,
            &gate_generic,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("generic topk");
    });

    let _ = timed_gpu_cmd(&ctx, |enc| {
        crate::metal::encode_mat_mat_f32_router_e8p32(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &probs_e8,
            h,
            n_expert,
            chunk_p,
        )
        .expect("e8p32 route logits");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &probs_e8,
            &moe.gate_inp_shexp,
            &h_pack,
            &idx_e8,
            &w_e8,
            &gate_e8,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("e8p32 topk");
    });

    let probs_generic_cpu = read_tensor_f32(&probs_generic);
    let probs_e8_cpu = read_tensor_f32(&probs_e8);
    let idx_generic_cpu = read_tensor_i32_f32buf(&idx_generic);
    let idx_e8_cpu = read_tensor_i32_f32buf(&idx_e8);
    let w_generic_cpu = read_tensor_f32(&w_generic);
    let w_e8_cpu = read_tensor_f32(&w_e8);
    let gate_generic_cpu = read_tensor_f32(&gate_generic);
    let gate_e8_cpu = read_tensor_f32(&gate_e8);

    let probs_cos = cosine_f32(&probs_generic_cpu, &probs_e8_cpu);
    let w_cos = cosine_f32(&w_generic_cpu, &w_e8_cpu);
    let gate_cos = cosine_f32(&gate_generic_cpu, &gate_e8_cpu);
    let mismatch_count = idx_generic_cpu
        .iter()
        .zip(&idx_e8_cpu)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[moe-route-e8p32-{label}] probs_cos={probs_cos:.6} topk_mismatches={mismatch_count} w_cos={w_cos:.6} gate_cos={gate_cos:.6}"
    );
    assert_eq!(mismatch_count, 0, "topk mismatch count {mismatch_count}");
    assert!(
        probs_cos > 0.999999,
        "router probs cos too low: {probs_cos}"
    );
    assert!(w_cos > 0.999999, "topk weight cos too low: {w_cos}");
    assert!(gate_cos > 0.999999, "shared gate cos too low: {gate_cos}");
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_bucket_fused_oracle_128() {
    run_moe_route_bucket_fused_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-128", 128);
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_bucket_fused_oracle_320() {
    run_moe_route_bucket_fused_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-320", 320);
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_bucket_fused_oracle_512() {
    run_moe_route_bucket_fused_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 512);
}

#[test]
#[ignore]
fn metal_122b_a10b_moe_route_bucket_fused_oracle_512() {
    run_moe_route_bucket_fused_oracle(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b", 512);
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_logits_e8p32_oracle_128() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-128", 128);
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_logits_e8p32_oracle_320() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-320", 320);
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_logits_e8p32_oracle_512() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-512", 512);
}

#[test]
#[ignore]
fn metal_35b_a3b_moe_route_logits_e8p32_oracle_1024() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-1024", 1024);
}

#[test]
#[ignore]
fn metal_122b_a10b_moe_route_logits_e8p32_oracle_320() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b-320", 320);
}

#[test]
#[ignore]
fn metal_122b_a10b_moe_route_logits_e8p32_oracle_512() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b-512", 512);
}

#[test]
#[ignore]
fn metal_122b_a10b_moe_route_logits_e8p32_oracle_1024() {
    run_moe_route_logits_e8p32_oracle(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b-1024", 1024);
}

fn run_gpu_compacted_grouped_down_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[gpu-grouped-down-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let slot_count = chunk_p * topk;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![slot_count as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![slot_count as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(slot_count * f_exp) as u64]);
    let grouped_slot_out = scratch
        .moe_expert_out_pack
        .view_subrange(0, vec![(slot_count * h) as u64]);
    let slot_ref = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("slot ref");
    let current_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("current reduced");
    let grouped_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("grouped reduced");
    let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
    let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut current_ms = 0.0f64;
    let mut map_ms = 0.0f64;
    let mut grouped_down_ms = 0.0f64;
    let mut grouped_reduce_ms = 0.0f64;
    let mut grouped_total_ms = 0.0f64;
    let mut active_experts = 0usize;
    let mut max_count = 0i32;
    let mut total_count = 0i32;
    let mut covered_slots = 0usize;
    let mut count_mismatches = 0usize;
    let mut id_mismatches = 0usize;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;
    let mut slot_cos_min = f64::INFINITY;
    let mut slot_max_abs = 0.0f32;
    let mut lane_dot = [0.0f64; 4];
    let mut lane_na = [0.0f64; 4];
    let mut lane_nb = [0.0f64; 4];

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("swiglu");
        });

        current_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &current_reduced,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current down");
        });

        let total_start = Instant::now();
        map_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &counts,
                &ids,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
        });
        let counts_cpu = read_tensor_i32_f32buf(&counts);
        active_experts = counts_cpu.iter().filter(|&&c| c > 0).count();
        max_count = counts_cpu.iter().copied().max().unwrap_or(0);
        total_count = counts_cpu.iter().sum();
        let ids_cpu = read_tensor_i32_f32buf(&ids);
        let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
        let topk_weight_cpu = read_tensor_f32(&topk_weight_pack);
        let (cpu_groups, cpu_slot_ids, _, _) =
            build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
        let mut cpu_counts = vec![0usize; n_expert];
        for group in &cpu_groups {
            cpu_counts[group.expert] = group.len;
        }
        count_mismatches = (0..n_expert)
            .filter(|&e| cpu_counts[e] as i32 != counts_cpu[e])
            .count();
        let mut cpu_ids = vec![-1i32; n_expert * chunk_p];
        for group in &cpu_groups {
            let dst = &mut cpu_ids[group.expert * chunk_p..group.expert * chunk_p + group.len];
            let src = &cpu_slot_ids[group.start..group.start + group.len];
            dst.copy_from_slice(src);
        }
        id_mismatches = 0;
        for expert in 0..n_expert {
            let count = counts_cpu[expert].max(0) as usize;
            for j in 0..count.min(chunk_p) {
                if cpu_ids[expert * chunk_p + j] != ids_cpu[expert * chunk_p + j] {
                    id_mismatches += 1;
                }
            }
        }
        let mut seen = vec![false; slot_count];
        for expert in 0..n_expert {
            let count = counts_cpu[expert].max(0) as usize;
            for j in 0..count.min(chunk_p) {
                let slot = ids_cpu[expert * chunk_p + j];
                if slot >= 0 && (slot as usize) < slot_count {
                    seen[slot as usize] = true;
                }
            }
        }
        covered_slots = seen.iter().filter(|&&b| b).count();
        grouped_down_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_slot_out, 0.0).expect("zero grouped slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &counts,
                &ids,
                &grouped_slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down slots");
        });
        grouped_reduce_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &grouped_slot_out,
                &topk_weight_pack,
                &grouped_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce");
        });

        let _ = timed_gpu_cmd(&ctx, |enc| {
            for token in 0..chunk_p {
                let inner_n = moe_inner_pack
                    .view_subrange((token * topk * f_exp) as u64, vec![(topk * f_exp) as u64]);
                let idx_n = topk_idx_pack.view_subrange((token * topk) as u64, vec![topk as u64]);
                let out_n =
                    slot_ref.view_subrange((token * topk * h) as u64, vec![(topk * h) as u64]);
                crate::metal::encode_moe_down_q5_K_f32(
                    &ctx,
                    enc,
                    &moe.down_exps,
                    &inner_n,
                    &idx_n,
                    &out_n,
                    f_exp,
                    h,
                    n_expert,
                    topk,
                )
                .expect("slot ref down");
            }
        });
        grouped_total_ms += total_start.elapsed().as_secs_f64() * 1e3;

        let cur = read_tensor_f32(&current_reduced);
        let grp = read_tensor_f32(&grouped_reduced);
        cos_min = cos_min.min(cosine_f32(&cur, &grp));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - grp[i]).abs());
        }
        let slot_cur = read_tensor_f32(&slot_ref);
        let slot_grp = read_tensor_f32(&grouped_slot_out);
        slot_cos_min = slot_cos_min.min(cosine_f32(&slot_cur, &slot_grp));
        for i in 0..slot_cur.len() {
            slot_max_abs = slot_max_abs.max((slot_cur[i] - slot_grp[i]).abs());
        }
        for expert in 0..n_expert {
            let count = counts_cpu[expert].max(0) as usize;
            for j in 0..count.min(chunk_p) {
                let slot = ids_cpu[expert * chunk_p + j] as usize;
                if slot >= slot_count {
                    continue;
                }
                let lane = j % 4;
                let off = slot * h;
                for c in 0..h {
                    let a = slot_cur[off + c] as f64;
                    let b = slot_grp[off + c] as f64;
                    lane_dot[lane] += a * b;
                    lane_na[lane] += a * a;
                    lane_nb[lane] += b * b;
                }
            }
        }
    }

    let lane_cos = |lane: usize| -> f64 {
        lane_dot[lane] / (lane_na[lane].sqrt() * lane_nb[lane].sqrt() + 1e-30)
    };

    let denom = n_runs as f64;
    eprintln!(
        "[gpu-grouped-down-{label}] chunk_p={chunk_p} active_experts={} max_count={} total_count={} covered_slots={} count_mismatches={} id_mismatches={} current={:.2} ms map={:.2} ms grouped_down={:.2} ms grouped_reduce={:.2} ms grouped_total={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e} slot_cos_min={:.6} slot_max_abs={:.3e} lane_cos=[{:.4},{:.4},{:.4},{:.4}]",
        active_experts,
        max_count,
        total_count,
        covered_slots,
        count_mismatches,
        id_mismatches,
        current_ms / denom,
        map_ms / denom,
        grouped_down_ms / denom,
        grouped_reduce_ms / denom,
        grouped_total_ms / denom,
        (current_ms / denom) / (grouped_total_ms / denom),
        cos_min,
        max_abs,
        slot_cos_min,
        slot_max_abs,
        lane_cos(0),
        lane_cos(1),
        lane_cos(2),
        lane_cos(3)
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_gpu_compacted_grouped_down_profile() {
    run_gpu_compacted_grouped_down_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_gpu_compacted_grouped_down_profile() {
    run_gpu_compacted_grouped_down_profile(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 320, 2);
}

fn run_grouped_q5_down_vs_matmat_oracle(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    min_group: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-q5-oracle-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    write_tensor_f32(&x_pack, &x_init);
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("postnorm");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("route");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("topk/shared");
        encode_moe_swiglu_q4_K_f32_packed_slots(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &topk_idx_pack,
            &moe_inner_pack,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("swiglu");
    });

    let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
    let topk_weight_cpu = read_tensor_f32(&topk_weight_pack);
    let (groups, slot_ids, _, _) =
        build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
    let group = groups
        .iter()
        .copied()
        .max_by_key(|g| g.len)
        .expect("at least one expert group");
    assert!(group.len >= min_group, "largest group too small");

    let count_t = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_t");
    let ids_t = MetalTensor::zeros_f32(&ctx, vec![(n_expert * group.len) as u64]).expect("ids_t");
    let inner_t = MetalTensor::zeros_f32(&ctx, vec![(group.len * f_exp) as u64]).expect("inner_t");
    let grouped_out_t =
        MetalTensor::zeros_f32(&ctx, vec![(group.len * h) as u64]).expect("grouped_out_t");
    let matmat_out_t =
        MetalTensor::zeros_f32(&ctx, vec![(group.len * h) as u64]).expect("matmat_out_t");
    let slot_ids_t = MetalTensor::zeros_i32(&ctx, vec![group.len as u64]).expect("slot_ids_t");

    let mut counts = vec![0i32; n_expert];
    counts[group.expert] = group.len as i32;
    let mut ids = vec![-1i32; n_expert * group.len];
    for j in 0..group.len {
        ids[group.expert * group.len + j] = j as i32;
    }
    cpu_write_i32_f32buf(&count_t, &counts);
    cpu_write_i32_f32buf(&ids_t, &ids);
    cpu_write_i32buf(&slot_ids_t, &slot_ids[group.start..group.start + group.len]);

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_get_rows_f32(
            &ctx,
            enc,
            &moe_inner_pack,
            &slot_ids_t,
            &inner_t,
            group.len,
            f_exp,
        )
        .expect("gather inner");
    });

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_fill_f32(&ctx, enc, &grouped_out_t, 0.0).expect("zero grouped out");
        crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
            &ctx,
            enc,
            &moe.down_exps,
            &inner_t,
            &count_t,
            &ids_t,
            &grouped_out_t,
            f_exp,
            h,
            n_expert,
            group.len,
        )
        .expect("grouped slots");
    });

    let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
    let expert_w = moe
        .down_exps
        .view_bytes(group.expert as u64 * expert_bytes, vec![(h * f_exp) as u64]);
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &expert_w,
            &inner_t,
            &matmat_out_t,
            f_exp,
            h,
            group.len,
        )
        .expect("matmat oracle");
    });

    let grouped = read_tensor_f32(&grouped_out_t);
    let oracle = read_tensor_f32(&matmat_out_t);
    let cos = cosine_f32(&grouped, &oracle);
    let mut grouped_t = vec![0.0f32; grouped.len()];
    for q in 0..group.len {
        for r in 0..h {
            grouped_t[q * h + r] = grouped[r * group.len + q];
        }
    }
    let cos_t = cosine_f32(&grouped_t, &oracle);
    let mut max_abs = 0.0f32;
    for i in 0..grouped.len() {
        max_abs = max_abs.max((grouped[i] - oracle[i]).abs());
    }
    let tile_rows = h.min(64);
    let tile_cols = group.len.min(32);
    let mut row_mod8_abs = [0.0f64; 8];
    let mut row_mod8_n = [0usize; 8];
    let mut col_mod4_abs = [0.0f64; 4];
    let mut col_mod4_n = [0usize; 4];
    for q in 0..tile_cols {
        for r in 0..tile_rows {
            let d = (grouped[q * h + r] - oracle[q * h + r]).abs() as f64;
            row_mod8_abs[r % 8] += d;
            row_mod8_n[r % 8] += 1;
            col_mod4_abs[q % 4] += d;
            col_mod4_n[q % 4] += 1;
        }
    }
    let row_stat = |i: usize| row_mod8_abs[i] / row_mod8_n[i].max(1) as f64;
    let col_stat = |i: usize| col_mod4_abs[i] / col_mod4_n[i].max(1) as f64;
    eprintln!(
        "[grouped-q5-oracle-{label}] expert={} count={} cos={cos:.6} cos_t={cos_t:.6} max_abs={max_abs:.3e} row_mod8=[{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e}] col_mod4=[{:.2e},{:.2e},{:.2e},{:.2e}]",
        group.expert,
        group.len,
        row_stat(0),
        row_stat(1),
        row_stat(2),
        row_stat(3),
        row_stat(4),
        row_stat(5),
        row_stat(6),
        row_stat(7),
        col_stat(0),
        col_stat(1),
        col_stat(2),
        col_stat(3),
    );

    for &probe_n in &[17usize, 32] {
        if group.len < probe_n {
            continue;
        }
        let count_probe = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_probe");
        let ids_probe =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * probe_n) as u64]).expect("ids_probe");
        let inner_probe = inner_t.view_subrange(0, vec![(probe_n * f_exp) as u64]);
        let grouped_probe =
            MetalTensor::zeros_f32(&ctx, vec![(probe_n * h) as u64]).expect("grouped_probe");
        let oracle_probe =
            MetalTensor::zeros_f32(&ctx, vec![(probe_n * h) as u64]).expect("oracle_probe");
        let mut probe_counts = vec![0i32; n_expert];
        probe_counts[group.expert] = probe_n as i32;
        let mut probe_ids = vec![-1i32; n_expert * probe_n];
        for j in 0..probe_n {
            probe_ids[group.expert * probe_n + j] = j as i32;
        }
        cpu_write_i32_f32buf(&count_probe, &probe_counts);
        cpu_write_i32_f32buf(&ids_probe, &probe_ids);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_probe, 0.0).expect("zero grouped probe");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &inner_probe,
                &count_probe,
                &ids_probe,
                &grouped_probe,
                f_exp,
                h,
                n_expert,
                probe_n,
            )
            .expect("grouped probe");
        });
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &expert_w,
                &inner_probe,
                &oracle_probe,
                f_exp,
                h,
                probe_n,
            )
            .expect("oracle probe");
        });
        let gp = read_tensor_f32(&grouped_probe);
        let op = read_tensor_f32(&oracle_probe);
        let probe_cos = cosine_f32(&gp, &op);
        let split16_cos = if probe_n >= 32 {
            let mut a0 = Vec::with_capacity(16 * h);
            let mut b0 = Vec::with_capacity(16 * h);
            let mut a1 = Vec::with_capacity(16 * h);
            let mut b1 = Vec::with_capacity(16 * h);
            for q in 0..16 {
                a0.extend_from_slice(&gp[q * h..(q + 1) * h]);
                b0.extend_from_slice(&op[q * h..(q + 1) * h]);
            }
            for q in 16..32 {
                a1.extend_from_slice(&gp[q * h..(q + 1) * h]);
                b1.extend_from_slice(&op[q * h..(q + 1) * h]);
            }
            format!(
                " first16={:.6} second16={:.6}",
                cosine_f32(&a0, &b0),
                cosine_f32(&a1, &b1)
            )
        } else {
            String::new()
        };
        let mut probe_max = 0.0f32;
        for i in 0..gp.len() {
            probe_max = probe_max.max((gp[i] - op[i]).abs());
        }
        eprintln!(
            "[grouped-q5-oracle-{label}] probe_n={} cos={probe_cos:.6} max_abs={probe_max:.3e}{}",
            probe_n, split16_cos,
        );
    }
}

fn run_grouped_q5_swiglu_vs_matmat_oracle(
    model_path: &str,
    label: &str,
    layer_idx: usize,
    chunk_p: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-q5-swiglu-oracle-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = mf.model.blocks.get(layer_idx).expect("layer exists");
    let moe = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => g.ffn_moe.as_ref().expect("moe block"),
        crate::metal_forward::MetalBlock::Attn(a) => a.ffn_moe.as_ref().expect("moe block"),
    };
    assert_eq!(moe.gate_exps.dtype, GgmlType::Q5_K, "gate dtype");
    assert_eq!(moe.up_exps.dtype, GgmlType::Q5_K, "up dtype");

    let slot_count = chunk_p * topk;
    let mut count_plan = vec![1usize, 15, 16, 17, 31, 32, 33, 48, 64, 64, 64, 64];
    let used: usize = count_plan.iter().sum();
    assert!(used <= slot_count, "count plan too large");
    count_plan.push(slot_count - used);
    assert!(
        count_plan.iter().all(|&c| c <= chunk_p),
        "count plan exceeds grouped id stride"
    );

    let selected_experts: Vec<usize> = (0..count_plan.len())
        .map(|i| (17 * i + 3) % n_expert)
        .collect();
    let mut counts = vec![0i32; n_expert];
    let mut ids = vec![-1i32; n_expert * chunk_p];
    let slots: Vec<i32> = (0..slot_count)
        .map(|i| ((i * 37) % slot_count) as i32)
        .collect();
    let mut cursor = 0usize;
    let mut expert_slots: Vec<(usize, Vec<i32>)> = Vec::new();
    for (&expert, &count) in selected_experts.iter().zip(count_plan.iter()) {
        counts[expert] = count as i32;
        let mut group_slots = Vec::with_capacity(count);
        for j in 0..count {
            let slot = slots[cursor + j];
            ids[expert * chunk_p + j] = slot;
            group_slots.push(slot);
        }
        cursor += count;
        expert_slots.push((expert, group_slots));
    }
    assert_eq!(cursor, slot_count);

    let h_pack = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("h_pack");
    let counts_t = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
    let ids_t = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
    let actual_inner =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("actual_inner");
    let h_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| (((i * 13 + 7) % 97) as f32 - 48.0) * 0.0075)
        .collect();
    write_tensor_f32(&h_pack, &h_init);
    cpu_write_i32_f32buf(&counts_t, &counts);
    cpu_write_i32_f32buf(&ids_t, &ids);

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_fill_f32(&ctx, enc, &actual_inner, -777.0).expect("poison actual");
        crate::metal::encode_moe_swiglu_q5_K_f32_grouped_slots_n16(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &counts_t,
            &ids_t,
            &actual_inner,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("grouped q5 swiglu");
    });

    let gate_bytes = moe.gate_exps.n_bytes() / n_expert as u64;
    let up_bytes = moe.up_exps.n_bytes() / n_expert as u64;
    let mut expected = vec![0.0f32; slot_count * f_exp];
    for (expert, group_slots) in &expert_slots {
        if group_slots.is_empty() {
            continue;
        }
        let n = group_slots.len();
        let token_ids: Vec<i32> = group_slots.iter().map(|slot| slot / topk as i32).collect();
        let token_ids_t = MetalTensor::zeros_i32(&ctx, vec![n as u64]).expect("token_ids");
        let h_group = MetalTensor::zeros_f32(&ctx, vec![(n * h) as u64]).expect("h_group");
        let gate_out = MetalTensor::zeros_f32(&ctx, vec![(n * f_exp) as u64]).expect("gate_out");
        let up_out = MetalTensor::zeros_f32(&ctx, vec![(n * f_exp) as u64]).expect("up_out");
        let inner_group =
            MetalTensor::zeros_f32(&ctx, vec![(n * f_exp) as u64]).expect("inner_group");
        cpu_write_i32buf(&token_ids_t, &token_ids);
        let gate_w = moe
            .gate_exps
            .view_bytes(*expert as u64 * gate_bytes, vec![(h * f_exp) as u64]);
        let up_w = moe
            .up_exps
            .view_bytes(*expert as u64 * up_bytes, vec![(h * f_exp) as u64]);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_get_rows_f32(&ctx, enc, &h_pack, &token_ids_t, &h_group, n, h)
                .expect("gather h");
            encode_mat_mat_dispatch(&ctx, enc, &gate_w, &h_group, &gate_out, h, f_exp, n)
                .expect("gate oracle");
            encode_mat_mat_dispatch(&ctx, enc, &up_w, &h_group, &up_out, h, f_exp, n)
                .expect("up oracle");
            encode_silu_mul_f32(&ctx, enc, &gate_out, &up_out, &inner_group).expect("silu oracle");
        });
        let group_cpu = read_tensor_f32(&inner_group);
        for (j, &slot) in group_slots.iter().enumerate() {
            let dst = slot as usize * f_exp;
            let src = j * f_exp;
            expected[dst..dst + f_exp].copy_from_slice(&group_cpu[src..src + f_exp]);
        }
    }

    let actual = read_tensor_f32(&actual_inner);
    let cos = cosine_f32(&actual, &expected);
    let mut max_abs = 0.0f32;
    let mut poison_count = 0usize;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_abs = max_abs.max((a - e).abs());
        if *a == -777.0 {
            poison_count += 1;
        }
    }
    eprintln!(
        "[grouped-q5-swiglu-oracle-{label}] layer={layer_idx} chunk_p={chunk_p} experts={} slots={} cos={cos:.6} max_abs={max_abs:.3e} poison_count={poison_count}",
        expert_slots.len(),
        slot_count,
    );
    assert_eq!(poison_count, 0, "grouped q5 swiglu left poisoned slots");
    assert!(cos > 0.999, "grouped q5 swiglu cos too low: {cos}");
    assert!(
        max_abs < 2.5e-1,
        "grouped q5 swiglu max_abs too high: {max_abs}"
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q5_down_vs_matmat_oracle() {
    run_grouped_q5_down_vs_matmat_oracle(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        32,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q5_swiglu_vs_matmat_oracle_layer46() {
    run_grouped_q5_swiglu_vs_matmat_oracle(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-layer46",
        46,
        64,
    );
}

fn run_grouped_q4_swiglu_vs_packed_oracle(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    min_group: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-q4-oracle-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    write_tensor_f32(&x_pack, &x_init);
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("postnorm");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("route");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("topk/shared");
        encode_moe_swiglu_q4_K_f32_packed_slots(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &topk_idx_pack,
            &moe_inner_pack,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("packed swiglu");
    });

    let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
    let topk_weight_cpu = read_tensor_f32(&topk_weight_pack);
    let (groups, slot_ids, _, _) =
        build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
    let group = groups
        .iter()
        .copied()
        .max_by_key(|g| g.len)
        .expect("at least one expert group");
    assert!(group.len >= min_group, "largest group too small");

    let count_t = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_t");
    let ids_t = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids_t");
    let grouped_inner_out = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
        .expect("grouped_inner_out");
    let slot_ids_t = MetalTensor::zeros_i32(&ctx, vec![group.len as u64]).expect("slot_ids_t");
    let packed_probe =
        MetalTensor::zeros_f32(&ctx, vec![(group.len * f_exp) as u64]).expect("packed_probe");
    let grouped_probe =
        MetalTensor::zeros_f32(&ctx, vec![(group.len * f_exp) as u64]).expect("grouped_probe");

    let mut counts = vec![0i32; n_expert];
    counts[group.expert] = group.len as i32;
    let mut ids = vec![-1i32; n_expert * chunk_p];
    for j in 0..group.len {
        ids[group.expert * chunk_p + j] = slot_ids[group.start + j];
    }
    cpu_write_i32_f32buf(&count_t, &counts);
    cpu_write_i32_f32buf(&ids_t, &ids);
    cpu_write_i32buf(&slot_ids_t, &slot_ids[group.start..group.start + group.len]);

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_fill_f32(&ctx, enc, &grouped_inner_out, 0.0).expect("zero grouped q4 out");
        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &count_t,
            &ids_t,
            &grouped_inner_out,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("grouped q4 swiglu");
        encode_get_rows_f32(
            &ctx,
            enc,
            &moe_inner_pack,
            &slot_ids_t,
            &packed_probe,
            group.len,
            f_exp,
        )
        .expect("gather packed probe");
        encode_get_rows_f32(
            &ctx,
            enc,
            &grouped_inner_out,
            &slot_ids_t,
            &grouped_probe,
            group.len,
            f_exp,
        )
        .expect("gather grouped probe");
    });

    let packed = read_tensor_f32(&packed_probe);
    let grouped = read_tensor_f32(&grouped_probe);
    let cos = cosine_f32(&packed, &grouped);
    let mut max_abs = 0.0f32;
    for i in 0..packed.len() {
        max_abs = max_abs.max((packed[i] - grouped[i]).abs());
    }
    eprintln!(
        "[grouped-q4-oracle-{label}] expert={} count={} cos={cos:.6} max_abs={max_abs:.3e}",
        group.expert, group.len,
    );

    for &probe_n in &[16usize, 17, 32] {
        if group.len < probe_n {
            continue;
        }
        let count_probe = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("count_probe");
        let ids_probe =
            MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids_probe");
        let grouped_probe_all = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
            .expect("grouped_probe_all");
        let packed_probe_n =
            MetalTensor::zeros_f32(&ctx, vec![(probe_n * f_exp) as u64]).expect("packed_probe_n");
        let grouped_probe_n =
            MetalTensor::zeros_f32(&ctx, vec![(probe_n * f_exp) as u64]).expect("grouped_probe_n");
        let slot_ids_probe =
            MetalTensor::zeros_i32(&ctx, vec![probe_n as u64]).expect("slot_ids_probe");
        let mut probe_counts = vec![0i32; n_expert];
        probe_counts[group.expert] = probe_n as i32;
        let mut probe_ids = vec![-1i32; n_expert * chunk_p];
        for j in 0..probe_n {
            probe_ids[group.expert * chunk_p + j] = slot_ids[group.start + j];
        }
        cpu_write_i32_f32buf(&count_probe, &probe_counts);
        cpu_write_i32_f32buf(&ids_probe, &probe_ids);
        cpu_write_i32buf(
            &slot_ids_probe,
            &slot_ids[group.start..group.start + probe_n],
        );

        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_probe_all, 0.0).expect("zero grouped q4 probe all");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &count_probe,
                &ids_probe,
                &grouped_probe_all,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped q4 probe");
            encode_get_rows_f32(
                &ctx,
                enc,
                &moe_inner_pack,
                &slot_ids_probe,
                &packed_probe_n,
                probe_n,
                f_exp,
            )
            .expect("gather packed probe n");
            encode_get_rows_f32(
                &ctx,
                enc,
                &grouped_probe_all,
                &slot_ids_probe,
                &grouped_probe_n,
                probe_n,
                f_exp,
            )
            .expect("gather grouped probe n");
        });

        let packed_n = read_tensor_f32(&packed_probe_n);
        let grouped_n = read_tensor_f32(&grouped_probe_n);
        let probe_cos = cosine_f32(&packed_n, &grouped_n);
        let mut probe_max = 0.0f32;
        for i in 0..packed_n.len() {
            probe_max = probe_max.max((packed_n[i] - grouped_n[i]).abs());
        }
        eprintln!(
            "[grouped-q4-oracle-{label}] probe_n={} cos={probe_cos:.6} max_abs={probe_max:.3e}",
            probe_n,
        );
    }
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q4_swiglu_vs_packed_oracle() {
    run_grouped_q4_swiglu_vs_packed_oracle(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        32,
    );
}

fn run_grouped_swiglu_down_backend_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-swiglu-down-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let slot_count = chunk_p * topk;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let atomic_topk_idx_pack =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64]).expect("atomic_topk_idx_pack");
    let atomic_topk_weight_pack = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk) as u64])
        .expect("atomic_topk_weight_pack");
    let atomic_shared_gate_pack =
        MetalTensor::zeros_f32(&ctx, vec![chunk_p as u64]).expect("atomic_shared_gate_pack");
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let packed_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("packed_reduced");
    let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
    let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
    let atomic_counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("atomic_counts");
    let atomic_ids =
        MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("atomic_ids");
    let live_grouped_inner = MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64])
        .expect("live_grouped_inner");
    let live_grouped_slot_out =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("live_grouped_slot_out");
    let live_grouped_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("live_grouped_reduced");
    let atomic_grouped_inner = MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64])
        .expect("atomic_grouped_inner");
    let atomic_grouped_slot_out = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64])
        .expect("atomic_grouped_slot_out");
    let atomic_grouped_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("atomic_grouped_reduced");
    let fused_gate_up =
        fuse_q4k_gate_up_expert_banks(&ctx, &moe.gate_exps, &moe.up_exps, h, f_exp, n_expert);
    let fused_grouped_inner = MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64])
        .expect("fused_grouped_inner");
    let fused_grouped_slot_out = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64])
        .expect("fused_grouped_slot_out");
    let fused_grouped_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("fused_grouped_reduced");
    let grouped_gate =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped_gate");
    let grouped_up =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped_up");
    let grouped_inner =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped_inner");
    let grouped_slot_out =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("grouped_slot_out");
    let grouped_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("grouped_reduced");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut packed_tail_ms = 0.0f64;
    let mut live_grouped_tail_ms = 0.0f64;
    let mut atomic_grouped_tail_ms = 0.0f64;
    let mut fused_grouped_tail_ms = 0.0f64;
    let mut map_ms = 0.0f64;
    let mut grouped_swiglu_gpu_ms = 0.0f64;
    let mut grouped_down_ms = 0.0f64;
    let mut grouped_reduce_ms = 0.0f64;
    let mut grouped_total_wall_ms = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;
    let mut atomic_cos_min = f64::INFINITY;
    let mut atomic_max_abs = 0.0f32;
    let mut fused_cos_min = f64::INFINITY;
    let mut fused_max_abs = 0.0f32;
    let mut active_experts = 0usize;
    let mut max_count = 0usize;
    let mut p50_count = 0usize;
    let mut p90_count = 0usize;
    let mut experts_ge16 = 0usize;
    let mut experts_ge32 = 0usize;
    let mut experts_ge48 = 0usize;
    let mut scan_back_edges = 0usize;
    let mut scan_edges = 0usize;
    let mut atomic_back_edges = 0usize;
    let mut atomic_edges = 0usize;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            encode_fill_f32(&ctx, enc, &atomic_counts, 0.0).expect("zero atomic counts");
            crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &atomic_topk_idx_pack,
                &atomic_topk_weight_pack,
                &atomic_shared_gate_pack,
                &atomic_counts,
                &atomic_ids,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared fused bucket");
        });

        packed_tail_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current swiglu");
            crate::metal::encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &packed_reduced,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current down");
        });

        map_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &counts,
                &ids,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
        });

        live_grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &live_grouped_inner, 0.0).expect("zero live grouped inner");
            if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &live_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("live grouped hot n32");
                    if hot_threshold > 0 {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &live_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("live grouped cold n16");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &live_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("live grouped n16 fallback");
                }
            } else {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &live_grouped_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("live grouped n16");
            }
            encode_fill_f32(&ctx, enc, &live_grouped_slot_out, 0.0)
                .expect("zero live grouped slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &live_grouped_inner,
                &counts,
                &ids,
                &live_grouped_slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("live grouped down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &live_grouped_slot_out,
                &topk_weight_pack,
                &live_grouped_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("live grouped reduce");
        });

        atomic_grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &atomic_grouped_inner, 0.0)
                .expect("zero atomic grouped inner");
            if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &atomic_counts,
                        &atomic_ids,
                        &atomic_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("atomic grouped hot n32");
                    if hot_threshold > 0 {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &atomic_counts,
                            &atomic_ids,
                            &atomic_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("atomic grouped cold n16");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &atomic_counts,
                        &atomic_ids,
                        &atomic_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("atomic grouped n16 fallback");
                }
            } else {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &atomic_counts,
                    &atomic_ids,
                    &atomic_grouped_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("atomic grouped n16");
            }
            encode_fill_f32(&ctx, enc, &atomic_grouped_slot_out, 0.0)
                .expect("zero atomic grouped slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &atomic_grouped_inner,
                &atomic_counts,
                &atomic_ids,
                &atomic_grouped_slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("atomic grouped down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &atomic_grouped_slot_out,
                &atomic_topk_weight_pack,
                &atomic_grouped_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("atomic grouped reduce");
        });

        fused_grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &fused_grouped_inner, 0.0)
                .expect("zero fused grouped inner");
            if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n32_range(
                        &ctx,
                        enc,
                        &fused_gate_up,
                        &h_pack,
                        &counts,
                        &ids,
                        &fused_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("fused grouped hot n32");
                    if hot_threshold > 0 {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                            &ctx,
                            enc,
                            &fused_gate_up,
                            &h_pack,
                            &counts,
                            &ids,
                            &fused_grouped_inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("fused grouped cold n16");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                        &ctx,
                        enc,
                        &fused_gate_up,
                        &h_pack,
                        &counts,
                        &ids,
                        &fused_grouped_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        0,
                        i32::MAX as u32,
                    )
                    .expect("fused grouped n16 fallback");
                }
            } else {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                    &ctx,
                    enc,
                    &fused_gate_up,
                    &h_pack,
                    &counts,
                    &ids,
                    &fused_grouped_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    0,
                    i32::MAX as u32,
                )
                .expect("fused grouped n16");
            }
            encode_fill_f32(&ctx, enc, &fused_grouped_slot_out, 0.0)
                .expect("zero fused grouped slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &fused_grouped_inner,
                &counts,
                &ids,
                &fused_grouped_slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("fused grouped down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &fused_grouped_slot_out,
                &topk_weight_pack,
                &fused_grouped_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("fused grouped reduce");
        });

        let wall = Instant::now();
        let counts_cpu = read_tensor_i32_f32buf(&counts);
        let ids_cpu = read_tensor_i32_f32buf(&ids);
        let atomic_counts_cpu = read_tensor_i32_f32buf(&atomic_counts);
        let atomic_ids_cpu = read_tensor_i32_f32buf(&atomic_ids);
        let mut active_counts: Vec<usize> = counts_cpu
            .iter()
            .filter_map(|&c| (c > 0).then_some(c as usize))
            .collect();
        active_counts.sort_unstable();
        active_experts = active_counts.len();
        max_count = counts_cpu.iter().copied().max().unwrap_or(0).max(0) as usize;
        if !active_counts.is_empty() {
            p50_count = active_counts[active_counts.len() / 2];
            p90_count = active_counts[(active_counts.len() * 9 / 10).min(active_counts.len() - 1)];
        }
        experts_ge16 = active_counts.iter().filter(|&&c| c >= 16).count();
        experts_ge32 = active_counts.iter().filter(|&&c| c >= 32).count();
        experts_ge48 = active_counts.iter().filter(|&&c| c >= 48).count();
        let bucket_back_edges = |counts: &[i32], ids: &[i32]| -> (usize, usize) {
            let mut back = 0usize;
            let mut edges = 0usize;
            for expert in 0..n_expert {
                let count = counts[expert].max(0) as usize;
                let mut prev_token = None;
                for j in 0..count.min(chunk_p) {
                    let slot = ids[expert * chunk_p + j].max(0) as usize;
                    let token = slot / topk;
                    if let Some(prev) = prev_token {
                        edges += 1;
                        if token < prev {
                            back += 1;
                        }
                    }
                    prev_token = Some(token);
                }
            }
            (back, edges)
        };
        (scan_back_edges, scan_edges) = bucket_back_edges(&counts_cpu, &ids_cpu);
        (atomic_back_edges, atomic_edges) = bucket_back_edges(&atomic_counts_cpu, &atomic_ids_cpu);

        grouped_swiglu_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
            if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                    crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &grouped_gate,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("split gate hot n32");
                    crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n32_range(
                        &ctx,
                        enc,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &grouped_up,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("split up hot n32");
                    if hot_threshold > 0 {
                        crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &grouped_gate,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("split gate cold n16");
                        crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16_range(
                            &ctx,
                            enc,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            &grouped_up,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("split up cold n16");
                    }
                } else {
                    crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &grouped_gate,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("split gate n16 fallback");
                    crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &grouped_up,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("split up n16 fallback");
                }
            } else {
                crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &grouped_gate,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("split gate n16");
                crate::metal::encode_moe_matmul_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &grouped_up,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("split up n16");
            }
            encode_silu_mul_f32(&ctx, enc, &grouped_gate, &grouped_up, &grouped_inner)
                .expect("grouped silu");
        });

        grouped_down_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_slot_out, 0.0).expect("zero grouped slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &grouped_inner,
                &counts,
                &ids,
                &grouped_slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down slots");
        });
        grouped_reduce_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &grouped_slot_out,
                &topk_weight_pack,
                &grouped_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce");
        });
        grouped_total_wall_ms += wall.elapsed().as_secs_f64() * 1e3;

        let cur = read_tensor_f32(&live_grouped_reduced);
        let grp = read_tensor_f32(&grouped_reduced);
        cos_min = cos_min.min(cosine_f32(&cur, &grp));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - grp[i]).abs());
        }
        let atomic = read_tensor_f32(&atomic_grouped_reduced);
        atomic_cos_min = atomic_cos_min.min(cosine_f32(&cur, &atomic));
        for i in 0..cur.len() {
            atomic_max_abs = atomic_max_abs.max((cur[i] - atomic[i]).abs());
        }
        let fus = read_tensor_f32(&fused_grouped_reduced);
        fused_cos_min = fused_cos_min.min(cosine_f32(&cur, &fus));
        for i in 0..cur.len() {
            fused_max_abs = fused_max_abs.max((cur[i] - fus[i]).abs());
        }
    }

    let denom = n_runs as f64;
    let split_down_reduce_ms = (grouped_down_ms + grouped_reduce_ms) / denom;
    eprintln!(
        "[grouped-swiglu-down-{label}] chunk_p={chunk_p} active_experts={} p50_count={} p90_count={} max_count={} experts_ge16={} experts_ge32={} experts_ge48={} scan_back_edges={}/{} atomic_back_edges={}/{} packed_tail={:.2} ms live_grouped_tail={:.2} ms atomic_grouped_tail={:.2} ms fused_grouped_tail={:.2} ms map={:.2} ms split_gate_up={:.2} ms split_down={:.2} ms split_reduce={:.2} ms split_down_reduce={:.2} ms split_wall={:.2} vs_live={:.3} vs_atomic_live={:.3} vs_fused_live={:.3} vs_packed={:.3} cos_min={:.6} max_abs={:.3e} atomic_cos_min={:.6} atomic_max_abs={:.3e} fused_cos_min={:.6} fused_max_abs={:.3e}",
        active_experts,
        p50_count,
        p90_count,
        max_count,
        experts_ge16,
        experts_ge32,
        experts_ge48,
        scan_back_edges,
        scan_edges,
        atomic_back_edges,
        atomic_edges,
        packed_tail_ms / denom,
        live_grouped_tail_ms / denom,
        atomic_grouped_tail_ms / denom,
        fused_grouped_tail_ms / denom,
        map_ms / denom,
        grouped_swiglu_gpu_ms / denom,
        grouped_down_ms / denom,
        grouped_reduce_ms / denom,
        split_down_reduce_ms,
        grouped_total_wall_ms / denom,
        (live_grouped_tail_ms / denom) / (grouped_total_wall_ms / denom),
        (live_grouped_tail_ms / denom) / (atomic_grouped_tail_ms / denom),
        (live_grouped_tail_ms / denom) / (fused_grouped_tail_ms / denom),
        (packed_tail_ms / denom) / (grouped_total_wall_ms / denom),
        cos_min,
        max_abs,
        atomic_cos_min,
        atomic_max_abs,
        fused_cos_min,
        fused_max_abs,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_swiglu_down_backend_profile() {
    run_grouped_swiglu_down_backend_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_swiglu_down_backend_profile() {
    run_grouped_swiglu_down_backend_profile(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 320, 2);
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_swiglu_down_backend_profile_512() {
    run_grouped_swiglu_down_backend_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-512",
        512,
        2,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_swiglu_down_backend_profile_512() {
    run_grouped_swiglu_down_backend_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-512",
        512,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_swiglu_down_backend_profile_1024() {
    run_grouped_swiglu_down_backend_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-1024",
        1024,
        2,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_swiglu_down_backend_profile_1024() {
    run_grouped_swiglu_down_backend_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-1024",
        1024,
        2,
    );
}

fn run_grouped_swiglu_fused_bank_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-swiglu-fused-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };
    assert_eq!(moe.gate_exps.dtype, GgmlType::Q4_K);
    assert_eq!(moe.up_exps.dtype, GgmlType::Q4_K);

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
    let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
    let current_inner =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64]).expect("current_inner");
    let fused_inner =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64]).expect("fused_inner");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    let hot_threshold = prefill_moe_hot_expert_min_slots().unwrap_or(1) as u32;
    let fused_gate_up =
        fuse_q4k_gate_up_expert_banks(&ctx, &moe.gate_exps, &moe.up_exps, h, f_exp, n_expert);

    let mut current_ms = 0.0f64;
    let mut fused_ms = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;
    let mut hot_experts = 0usize;
    let mut hot_slots = 0usize;
    let mut max_count = 0usize;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
        });
        let _ = timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &counts,
                &ids,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
        });

        let counts_cpu = read_tensor_i32_f32buf(&counts);
        hot_experts = counts_cpu
            .iter()
            .filter(|&&c| c >= hot_threshold as i32)
            .count();
        hot_slots = counts_cpu
            .iter()
            .filter(|&&c| c >= hot_threshold as i32)
            .map(|&c| c as usize)
            .sum();
        max_count = counts_cpu.iter().copied().max().unwrap_or(0).max(0) as usize;
        if hot_experts == 0 {
            eprintln!("[grouped-swiglu-fused-{label}] no experts above threshold={hot_threshold}");
            return;
        }

        current_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &current_inner, 0.0).expect("zero current inner");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &counts,
                &ids,
                &current_inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                hot_threshold,
                i32::MAX as u32,
            )
            .expect("current hot n32");
            if hot_threshold > 0 {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &current_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    0,
                    hot_threshold.saturating_sub(1),
                )
                .expect("current cold n16");
            }
        });

        fused_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &fused_inner, 0.0).expect("zero fused inner");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n32_range(
                &ctx,
                enc,
                &fused_gate_up,
                &h_pack,
                &counts,
                &ids,
                &fused_inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
                hot_threshold,
                i32::MAX as u32,
            )
            .expect("fused hot n32");
            if hot_threshold > 0 {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_fused_n16_range(
                    &ctx,
                    enc,
                    &fused_gate_up,
                    &h_pack,
                    &counts,
                    &ids,
                    &fused_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    0,
                    hot_threshold.saturating_sub(1),
                )
                .expect("fused cold n16");
            }
        });

        let cur = read_tensor_f32(&current_inner);
        let fus = read_tensor_f32(&fused_inner);
        cos_min = cos_min.min(cosine_f32(&cur, &fus));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - fus[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[grouped-swiglu-fused-{label}] chunk_p={chunk_p} hot_threshold={} hot_experts={} hot_slots={} max_count={} current_total={:.2} ms fused_total={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
        hot_threshold,
        hot_experts,
        hot_slots,
        max_count,
        current_ms / denom,
        fused_ms / denom,
        (current_ms / denom) / (fused_ms / denom),
        cos_min,
        max_abs,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_swiglu_fused_bank_profile_320() {
    run_grouped_swiglu_fused_bank_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-320",
        320,
        3,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_swiglu_fused_bank_profile_320() {
    run_grouped_swiglu_fused_bank_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-320",
        320,
        3,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_swiglu_fused_bank_profile_512() {
    run_grouped_swiglu_fused_bank_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-512",
        512,
        3,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_swiglu_fused_bank_profile_512() {
    run_grouped_swiglu_fused_bank_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-512",
        512,
        3,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_swiglu_fused_bank_profile_1024() {
    run_grouped_swiglu_fused_bank_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b-1024",
        1024,
        3,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_swiglu_fused_bank_profile_1024() {
    run_grouped_swiglu_fused_bank_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-1024",
        1024,
        3,
    );
}

fn assert_moe_grouped_slot_coverage(
    label: &str,
    counts: &[i32],
    ids: &[i32],
    topk_idx: &[i32],
    n_expert: usize,
    topk: usize,
    chunk_p: usize,
) {
    assert_eq!(counts.len(), n_expert, "{label}: counts length");
    assert_eq!(ids.len(), n_expert * chunk_p, "{label}: ids length");
    assert_eq!(topk_idx.len(), chunk_p * topk, "{label}: topk length");

    let mut per_token_seen = vec![false; n_expert];
    for token in 0..chunk_p {
        per_token_seen.fill(false);
        for k in 0..topk {
            let slot = token * topk + k;
            let expert = topk_idx[slot];
            assert!(expert >= 0, "{label}: negative expert at slot {slot}");
            let expert = expert as usize;
            assert!(
                expert < n_expert,
                "{label}: expert {expert} out of range at slot {slot}"
            );
            assert!(
                !per_token_seen[expert],
                "{label}: duplicate expert {expert} for token {token}"
            );
            per_token_seen[expert] = true;
        }
    }

    let slot_count = chunk_p * topk;
    let mut seen = vec![0u8; slot_count];
    let mut total = 0usize;
    for (expert, &count) in counts.iter().enumerate().take(n_expert) {
        assert!(count >= 0, "{label}: negative count for expert {expert}");
        let count = count as usize;
        assert!(
            count <= chunk_p,
            "{label}: count {count} exceeds chunk_p for expert {expert}"
        );
        total += count;
        let base = expert * chunk_p;
        for j in 0..count {
            let slot = ids[base + j];
            assert!(slot >= 0, "{label}: negative id for expert {expert} j={j}");
            let slot = slot as usize;
            assert!(
                slot < slot_count,
                "{label}: id {slot} out of slot range for expert {expert} j={j}"
            );
            seen[slot] = seen[slot].saturating_add(1);
        }
    }
    assert_eq!(total, slot_count, "{label}: total routed slot count");
    for (slot, &n) in seen.iter().enumerate() {
        assert_eq!(n, 1, "{label}: slot {slot} appears {n} times");
    }
}

fn run_grouped_zero_fill_coverage_oracle(model_path: &str, label: &str, chunk_p: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-zero-fill-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let slot_count = chunk_p * topk;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };
    assert_eq!(moe.gate_exps.dtype, GgmlType::Q4_K, "gate dtype");
    assert_eq!(moe.up_exps.dtype, GgmlType::Q4_K, "up dtype");
    assert_eq!(moe.down_exps.dtype, GgmlType::Q5_K, "down dtype");

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let counts = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let ids = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let inner_zero =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("inner_zero");
    let out_zero = MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("out_zero");
    let reduced_zero =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_zero");
    let inner_poison =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("inner_poison");
    let out_poison =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("out_poison");
    let reduced_poison =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_poison");

    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    write_tensor_f32(&x_pack, &x_init);

    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("postnorm");
        encode_moe_route_logits_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("route logits");
        encode_fill_f32(&ctx, enc, &counts, 0.0).expect("zero route counts");
        crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            &counts,
            &ids,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("topk+bucket");
    });

    let counts_cpu = read_tensor_i32_f32buf(&counts);
    let ids_cpu = read_tensor_i32_f32buf(&ids);
    let topk_idx_cpu = read_tensor_i32_f32buf(&topk_idx_pack);
    assert_moe_grouped_slot_coverage(
        label,
        &counts_cpu,
        &ids_cpu,
        &topk_idx_cpu,
        n_expert,
        topk,
        chunk_p,
    );

    let run_grouped = |inner: &MetalTensor, out: &MetalTensor, reduced: &MetalTensor, fill: f32| {
        timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, inner, fill).expect("fill grouped inner");
            if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
                if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        hot_threshold as u32,
                        i32::MAX as u32,
                    )
                    .expect("hot grouped swiglu");
                    if hot_threshold > 0 {
                        crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            &h_pack,
                            &counts,
                            &ids,
                            inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            chunk_p,
                            0,
                            hot_threshold.saturating_sub(1) as u32,
                        )
                        .expect("cold grouped swiglu");
                    }
                } else {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                    )
                    .expect("grouped swiglu");
                }
            } else {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("grouped swiglu");
            }
            encode_fill_f32(&ctx, enc, out, fill).expect("fill grouped out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                inner,
                &counts,
                &ids,
                out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                out,
                &topk_weight_pack,
                reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("weighted sum");
        })
    };

    let zero_ms = run_grouped(&inner_zero, &out_zero, &reduced_zero, 0.0);
    let poison_ms = run_grouped(&inner_poison, &out_poison, &reduced_poison, f32::NAN);
    let zero = read_tensor_f32(&reduced_zero);
    let poison = read_tensor_f32(&reduced_poison);
    assert!(
        zero.iter().all(|v| v.is_finite()),
        "{label}: zero output has non-finite"
    );
    assert!(
        poison.iter().all(|v| v.is_finite()),
        "{label}: poison output has non-finite"
    );
    let cos = cosine_f32(&zero, &poison);
    let mut max_abs = 0.0f32;
    for i in 0..zero.len() {
        max_abs = max_abs.max((zero[i] - poison[i]).abs());
    }
    eprintln!(
        "[grouped-zero-fill-{label}] chunk_p={chunk_p} zero={zero_ms:.2} ms poison={poison_ms:.2} ms cos={cos:.9} max_abs={max_abs:.3e}"
    );
    assert!(cos > 0.999999, "{label}: cosine {cos:.9} below gate");
    assert!(max_abs < 1e-5, "{label}: max_abs {max_abs:.3e} above gate");
}

fn run_grouped_moe_overlap_falsifier(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-overlap-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let f_shared = arch.expert_shared_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, g_w, u_w, d_w, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => (
            &g.post_attn_norm,
            &g.ffn_gate,
            &g.ffn_up,
            &g.ffn_down,
            g.ffn_moe.as_ref().expect("moe block"),
        ),
        crate::metal_forward::MetalBlock::Attn(a) => (
            &a.post_attn_norm,
            &a.ffn_gate,
            &a.ffn_up,
            &a.ffn_down,
            a.ffn_moe.as_ref().expect("moe block"),
        ),
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let counts = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let ids = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let routed_inner = scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let routed_out = scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let mixer_out = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("mixer_out");
    let shared_gate_ffn = scratch
        .moe_shared_ffn_gate_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_up_ffn = scratch
        .moe_shared_ffn_up_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_inner_ffn = scratch
        .moe_shared_ffn_inner_pack
        .view_subrange(0, vec![(chunk_p * f_shared) as u64]);
    let shared_out_ffn = scratch
        .moe_shared_ffn_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let serial_final =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("serial_final");
    let concurrent_final =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("concurrent_final");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let _ = timed_gpu_cmd(&ctx, |enc| {
        write_tensor_f32(&x_pack, &x_init);
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("postnorm");
        encode_moe_route_logits_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("route logits");
        encode_fill_f32(&ctx, enc, &counts, 0.0).expect("zero counts");
        crate::metal::encode_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            &counts,
            &ids,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("route+bucket");
    });

    let encode_routed = |enc: &KernelEncoder| {
        if prefill_moe_grouped_hot_q4_n32_enabled(chunk_p) {
            if let Some(hot_threshold) = prefill_moe_hot_expert_min_slots() {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32_range(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &routed_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                    hot_threshold as u32,
                    i32::MAX as u32,
                )
                .expect("routed hot n32");
                if hot_threshold > 0 {
                    crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16_range(
                        &ctx,
                        enc,
                        &moe.gate_exps,
                        &moe.up_exps,
                        &h_pack,
                        &counts,
                        &ids,
                        &routed_inner,
                        h,
                        f_exp,
                        n_expert,
                        topk,
                        chunk_p,
                        0,
                        hot_threshold.saturating_sub(1) as u32,
                    )
                    .expect("routed cold n16");
                }
            } else {
                crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &h_pack,
                    &counts,
                    &ids,
                    &routed_inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    chunk_p,
                )
                .expect("routed n16 fallback");
            }
        } else {
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &counts,
                &ids,
                &routed_inner,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("routed n16");
        }
        encode_fill_f32(&ctx, enc, &routed_out, 0.0).expect("zero routed out");
        crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
            &ctx,
            enc,
            &moe.down_exps,
            &routed_inner,
            &counts,
            &ids,
            &routed_out,
            f_exp,
            h,
            n_expert,
            chunk_p,
        )
        .expect("routed down");
        crate::metal::encode_moe_weighted_sum_packed_f32(
            &ctx,
            enc,
            &routed_out,
            &topk_weight_pack,
            &mixer_out,
            h,
            topk,
            chunk_p,
        )
        .expect("routed reduce");
    };

    let encode_shared = |enc: &KernelEncoder| {
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            g_w,
            &h_pack,
            &shared_gate_ffn,
            h,
            f_shared,
            chunk_p,
        )
        .expect("shared gate");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            u_w,
            &h_pack,
            &shared_up_ffn,
            h,
            f_shared,
            chunk_p,
        )
        .expect("shared up");
        encode_silu_mul_f32(
            &ctx,
            enc,
            &shared_gate_ffn,
            &shared_up_ffn,
            &shared_inner_ffn,
        )
        .expect("shared silu");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            d_w,
            &shared_inner_ffn,
            &shared_out_ffn,
            f_shared,
            h,
            chunk_p,
        )
        .expect("shared down");
    };

    let mut serial_gpu = 0.0f64;
    let mut concurrent_gpu = 0.0f64;
    let mut serial_wall = 0.0f64;
    let mut concurrent_wall = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;

    for _ in 0..n_runs {
        write_tensor_f32(&serial_final, &x_init);
        let wall = Instant::now();
        let cmd = ctx.queue.commandBuffer().expect("serial cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_fill_f32(&ctx, &enc, &mixer_out, 0.0).expect("zero mixer");
        encode_routed(&enc);
        encode_shared(&enc);
        encode_axpy_rowwise_f32(
            &ctx,
            &enc,
            &shared_out_ffn,
            &shared_gate_pack,
            &mixer_out,
            h,
            chunk_p,
        )
        .expect("shared axpy");
        encode_add_inplace_f32(&ctx, &enc, &serial_final, &mixer_out).expect("serial add");
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        serial_gpu += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        serial_wall += wall.elapsed().as_secs_f64() * 1e3;

        write_tensor_f32(&concurrent_final, &x_init);
        let wall = Instant::now();
        let cmd = ctx.queue.commandBuffer().expect("concurrent cmd");
        {
            let enc = KernelEncoder::begin(&cmd);
            encode_fill_f32(&ctx, &enc, &mixer_out, 0.0).expect("zero mixer concurrent");
            enc.end();
        }
        {
            let enc = KernelEncoder::begin_concurrent(&cmd);
            encode_routed(&enc);
            encode_shared(&enc);
            enc.end();
        }
        {
            let enc = KernelEncoder::begin(&cmd);
            encode_axpy_rowwise_f32(
                &ctx,
                &enc,
                &shared_out_ffn,
                &shared_gate_pack,
                &mixer_out,
                h,
                chunk_p,
            )
            .expect("shared axpy concurrent");
            encode_add_inplace_f32(&ctx, &enc, &concurrent_final, &mixer_out)
                .expect("concurrent add");
            enc.end();
        }
        cmd.commit();
        cmd.waitUntilCompleted();
        concurrent_gpu += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
        concurrent_wall += wall.elapsed().as_secs_f64() * 1e3;

        let serial = read_tensor_f32(&serial_final);
        let concurrent = read_tensor_f32(&concurrent_final);
        cos_min = cos_min.min(cosine_f32(&serial, &concurrent));
        for i in 0..serial.len() {
            max_abs = max_abs.max((serial[i] - concurrent[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[grouped-overlap-{label}] chunk_p={chunk_p} serial_gpu={:.2} ms concurrent_gpu={:.2} ms serial_wall={:.2} ms concurrent_wall={:.2} ms speedup_gpu={:.3} speedup_wall={:.3} cos_min={:.6} max_abs={:.3e}",
        serial_gpu / denom,
        concurrent_gpu / denom,
        serial_wall / denom,
        concurrent_wall / denom,
        (serial_gpu / denom) / (concurrent_gpu / denom),
        (serial_wall / denom) / (concurrent_wall / denom),
        cos_min,
        max_abs,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_overlap_falsifier_512() {
    run_grouped_moe_overlap_falsifier(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-512", 512, 3);
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_overlap_falsifier_512() {
    run_grouped_moe_overlap_falsifier(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "a10b-512",
        512,
        3,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_zero_fill_coverage_oracle_512() {
    run_grouped_zero_fill_coverage_oracle(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-512", 512);
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_zero_fill_coverage_oracle_512() {
    run_grouped_zero_fill_coverage_oracle(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "a10b-512",
        512,
    );
}

fn run_grouped_q4_n32_proof(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-q4-n32-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let group_count_pack = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let group_ids_pack = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let inner_n16 = scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let inner_n32 = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let slot_out = scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let reduced_n16 =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_n16");
    let reduced_n32 =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_n32");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut n16_gpu = 0.0f64;
    let mut n32_gpu = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;
    let mut active_experts = 0usize;
    let mut max_count = 0usize;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &group_count_pack,
                &group_ids_pack,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
        });
        let count_cpu = cpu_read_i32_f32buf(&group_count_pack);
        active_experts = count_cpu.iter().filter(|&&c| c > 0).count();
        max_count = count_cpu.iter().copied().max().unwrap_or(0).max(0) as usize;

        n16_gpu += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &inner_n16, 0.0).expect("zero inner n16");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &group_count_pack,
                &group_ids_pack,
                &inner_n16,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu n16");
            encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &inner_n16,
                &group_count_pack,
                &group_ids_pack,
                &slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down n16");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &slot_out,
                &topk_weight_pack,
                &reduced_n16,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce n16");
        });

        n32_gpu += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &inner_n32, 0.0).expect("zero inner n32");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &group_count_pack,
                &group_ids_pack,
                &inner_n32,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu n32");
            encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out 2");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &inner_n32,
                &group_count_pack,
                &group_ids_pack,
                &slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down n32");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &slot_out,
                &topk_weight_pack,
                &reduced_n32,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce n32");
        });

        let cur = read_tensor_f32(&reduced_n16);
        let alt = read_tensor_f32(&reduced_n32);
        cos_min = cos_min.min(cosine_f32(&cur, &alt));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - alt[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[grouped-q4-n32-{label}] chunk_p={chunk_p} active_experts={} max_count={} n16_gpu={:.2} ms n32_gpu={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
        active_experts,
        max_count,
        n16_gpu / denom,
        n32_gpu / denom,
        (n16_gpu / denom) / (n32_gpu / denom),
        cos_min,
        max_abs,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_q4_n32_proof_512() {
    run_grouped_q4_n32_proof(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 512, 2);
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q4_n32_proof_512() {
    run_grouped_q4_n32_proof(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b", 512, 2);
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_q4_n32_proof_1024() {
    run_grouped_q4_n32_proof(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b-1024", 1024, 2);
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q4_n32_proof_1024() {
    run_grouped_q4_n32_proof(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b-1024",
        1024,
        2,
    );
}

fn run_grouped_q4_hot_n32_proof(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    threshold: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-q4-hot-n32-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let count_pack = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let ids_pack = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let hot_count_pack = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("hot_count");
    let cold_count_pack = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("cold_count");
    let hot_ids_pack =
        MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("hot_ids");
    let cold_ids_pack =
        MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("cold_ids");
    let inner_n16 = scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let inner_mix = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let slot_out = scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let reduced_n16 =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_n16");
    let reduced_mix =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("reduced_mix");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut n16_gpu = 0.0f64;
    let mut mix_gpu = 0.0f64;
    let mut cpu_split_ms = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;
    let mut hot_experts = 0usize;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &count_pack,
                &ids_pack,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
        });

        let t_cpu = Instant::now();
        let counts_cpu = cpu_read_i32_f32buf(&count_pack);
        let ids_cpu = cpu_read_i32_f32buf(&ids_pack);
        let mut hot_counts = vec![0i32; n_expert];
        let mut cold_counts = vec![0i32; n_expert];
        let mut hot_ids = vec![-1i32; n_expert * chunk_p];
        let mut cold_ids = vec![-1i32; n_expert * chunk_p];
        hot_experts = 0;
        for expert in 0..n_expert {
            let len = counts_cpu[expert].max(0) as usize;
            if len >= threshold {
                hot_experts += 1;
                hot_counts[expert] = len as i32;
                hot_ids[expert * chunk_p..expert * chunk_p + len]
                    .copy_from_slice(&ids_cpu[expert * chunk_p..expert * chunk_p + len]);
            } else if len > 0 {
                cold_counts[expert] = len as i32;
                cold_ids[expert * chunk_p..expert * chunk_p + len]
                    .copy_from_slice(&ids_cpu[expert * chunk_p..expert * chunk_p + len]);
            }
        }
        cpu_write_i32_f32buf(&hot_count_pack, &hot_counts);
        cpu_write_i32_f32buf(&cold_count_pack, &cold_counts);
        cpu_write_i32_f32buf(&hot_ids_pack, &hot_ids);
        cpu_write_i32_f32buf(&cold_ids_pack, &cold_ids);
        cpu_split_ms += t_cpu.elapsed().as_secs_f64() * 1e3;

        n16_gpu += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &inner_n16, 0.0).expect("zero inner n16");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &count_pack,
                &ids_pack,
                &inner_n16,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu n16");
            encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &inner_n16,
                &count_pack,
                &ids_pack,
                &slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down n16");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &slot_out,
                &topk_weight_pack,
                &reduced_n16,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce n16");
        });

        mix_gpu += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &inner_mix, 0.0).expect("zero inner mix");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n32(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &hot_count_pack,
                &hot_ids_pack,
                &inner_mix,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu n32 hot");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &cold_count_pack,
                &cold_ids_pack,
                &inner_mix,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu n16 cold");
            encode_fill_f32(&ctx, enc, &slot_out, 0.0).expect("zero slot out 2");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &inner_mix,
                &count_pack,
                &ids_pack,
                &slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down mix");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &slot_out,
                &topk_weight_pack,
                &reduced_mix,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce mix");
        });

        let cur = read_tensor_f32(&reduced_n16);
        let alt = read_tensor_f32(&reduced_mix);
        cos_min = cos_min.min(cosine_f32(&cur, &alt));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - alt[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[grouped-q4-hot-n32-{label}] chunk_p={chunk_p} threshold={threshold} hot_experts={} n16_gpu={:.2} ms mix_gpu={:.2} ms cpu_split={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
        hot_experts,
        n16_gpu / denom,
        mix_gpu / denom,
        cpu_split_ms / denom,
        (n16_gpu / denom) / (mix_gpu / denom),
        cos_min,
        max_abs,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_q4_hot_n32_proof_512() {
    run_grouped_q4_hot_n32_proof(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 512, 32, 2);
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q4_hot_n32_proof_512() {
    run_grouped_q4_hot_n32_proof(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        512,
        32,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_q4_hot_n32_threshold_scan_512() {
    for &threshold in &[48usize, 64, 96, 128] {
        run_grouped_q4_hot_n32_proof(
            crate::test_fixtures::A3B_Q4_K_M.path(),
            &format!("a3b-th{threshold}"),
            512,
            threshold,
            2,
        );
    }
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_q4_hot_n32_threshold_scan_512() {
    for &threshold in &[48usize, 64, 96, 128] {
        run_grouped_q4_hot_n32_proof(
            crate::test_fixtures::A10B_Q4_K_XL.path(),
            &format!("122b-th{threshold}"),
            512,
            threshold,
            2,
        );
    }
}

fn run_gpu_owned_grouped_routed_backend_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[gpu-owned-grouped-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let grouped_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("grouped_reduced");
    let current_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("current_reduced");
    let counts = MetalTensor::zeros_f32(&ctx, vec![n_expert as u64]).expect("counts");
    let ids = MetalTensor::zeros_f32(&ctx, vec![(n_expert * chunk_p) as u64]).expect("ids");
    let grouped_inner_slot_major =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * f_exp) as u64])
            .expect("grouped_inner_slot_major");
    let grouped_slot_out =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * topk * h) as u64]).expect("grouped_slot_out");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut current_tail_ms = 0.0f64;
    let mut map_ms = 0.0f64;
    let mut grouped_tail_ms = 0.0f64;
    let mut grouped_wall_ms = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
        });

        current_tail_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current swiglu");
            crate::metal::encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &current_reduced,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current down");
        });

        let wall = Instant::now();
        map_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &counts,
                &ids,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
        });

        grouped_tail_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &grouped_inner_slot_major, 0.0)
                .expect("zero grouped inner slot-major");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &counts,
                &ids,
                &grouped_inner_slot_major,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu n16");
            encode_fill_f32(&ctx, enc, &grouped_slot_out, 0.0).expect("zero grouped slot out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &grouped_inner_slot_major,
                &counts,
                &ids,
                &grouped_slot_out,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down slots");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &grouped_slot_out,
                &topk_weight_pack,
                &grouped_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce");
        });
        grouped_wall_ms += wall.elapsed().as_secs_f64() * 1e3;

        let cur = read_tensor_f32(&current_reduced);
        let grp = read_tensor_f32(&grouped_reduced);
        cos_min = cos_min.min(cosine_f32(&cur, &grp));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - grp[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[gpu-owned-grouped-{label}] chunk_p={chunk_p} current_tail={:.2} ms map={:.2} ms grouped_tail={:.2} ms grouped_wall={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
        current_tail_ms / denom,
        map_ms / denom,
        grouped_tail_ms / denom,
        grouped_wall_ms / denom,
        (current_tail_ms / denom) / (grouped_wall_ms / denom),
        cos_min,
        max_abs,
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_gpu_owned_grouped_routed_backend_profile() {
    run_gpu_owned_grouped_routed_backend_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_gpu_owned_grouped_routed_backend_profile() {
    run_gpu_owned_grouped_routed_backend_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b",
        320,
        2,
    );
}

fn run_grouped_routed_down_accum_proof(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-down-accum-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_group_count_pack = scratch
        .moe_group_count_pack
        .view_subrange(0, vec![n_expert as u64]);
    let moe_group_ids_pack = scratch
        .moe_group_ids_pack
        .view_subrange(0, vec![(n_expert * chunk_p) as u64]);
    let moe_group_slot_idx_pack = scratch
        .moe_group_slot_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let moe_group_token_idx_pack = scratch
        .moe_group_token_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let moe_group_weight_pack = scratch
        .moe_group_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let moe_group_inner_pack = scratch
        .moe_group_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let moe_group_out_pack = scratch
        .moe_group_out_pack
        .view_subrange(0, vec![(chunk_p * topk * h) as u64]);
    let current_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("current_reduced");
    let accum_reduced =
        MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("accum_reduced");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut current_gpu_ms = 0.0f64;
    let mut accum_gpu_ms = 0.0f64;
    let mut cpu_group_ms = 0.0f64;
    let mut accum_wall_ms = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            crate::metal::encode_moe_route_bucket_slots_f32(
                &ctx,
                enc,
                &topk_idx_pack,
                &moe_group_count_pack,
                &moe_group_ids_pack,
                n_expert,
                chunk_p,
                topk,
            )
            .expect("bucket slots");
            encode_fill_f32(&ctx, enc, &moe_group_inner_pack, 0.0).expect("zero grouped inner");
            crate::metal::encode_moe_swiglu_q4_K_f32_grouped_slots_n16(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &moe_group_count_pack,
                &moe_group_ids_pack,
                &moe_group_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("grouped swiglu");
        });

        current_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &moe_group_out_pack, 0.0).expect("zero grouped out");
            crate::metal::encode_moe_down_q5_K_f32_grouped_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_group_inner_pack,
                &moe_group_count_pack,
                &moe_group_ids_pack,
                &moe_group_out_pack,
                f_exp,
                h,
                n_expert,
                chunk_p,
            )
            .expect("grouped down");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &moe_group_out_pack,
                &topk_weight_pack,
                &current_reduced,
                h,
                topk,
                chunk_p,
            )
            .expect("grouped reduce");
        });

        let t_cpu = Instant::now();
        let topk_idx_cpu = cpu_read_i32_f32buf(&topk_idx_pack);
        let topk_weight_cpu = cpu_read_f32buf(&topk_weight_pack);
        let count_cpu = cpu_read_i32_f32buf(&moe_group_count_pack);
        let ids_cpu = cpu_read_i32_f32buf(&moe_group_ids_pack);
        let (groups, slot_ids, token_ids, weights) =
            build_expert_slot_groups_cpu(&topk_idx_cpu, &topk_weight_cpu, topk, n_expert);
        let mut gpu_order = Vec::new();
        for expert in 0..n_expert {
            let len = count_cpu[expert].max(0) as usize;
            for j in 0..len.min(chunk_p) {
                gpu_order.push(ids_cpu[expert * chunk_p + j]);
            }
        }
        assert_eq!(slot_ids, gpu_order, "grouped slot ordering diverged");
        cpu_write_i32buf(&moe_group_slot_idx_pack, &slot_ids);
        cpu_write_i32_f32buf(&moe_group_token_idx_pack, &token_ids);
        cpu_write_f32buf(&moe_group_weight_pack, &weights);
        cpu_group_ms += t_cpu.elapsed().as_secs_f64() * 1e3;

        let wall = Instant::now();
        accum_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &accum_reduced, 0.0).expect("zero accum reduced");
            for group in &groups {
                let token_ids_n = moe_group_token_idx_pack
                    .view_subrange(group.start as u64, vec![group.len as u64]);
                let weights_n =
                    moe_group_weight_pack.view_subrange(group.start as u64, vec![group.len as u64]);
                let inner_n = moe_group_inner_pack.view_subrange(
                    (group.start * f_exp) as u64,
                    vec![(group.len * f_exp) as u64],
                );
                let out_n = moe_group_out_pack
                    .view_subrange((group.start * h) as u64, vec![(group.len * h) as u64]);
                let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
                let expert_w = moe
                    .down_exps
                    .view_bytes(group.expert as u64 * expert_bytes, vec![(h * f_exp) as u64]);
                encode_mat_mat_dispatch(
                    &ctx, enc, &expert_w, &inner_n, &out_n, f_exp, h, group.len,
                )
                .expect("expert grouped down");
                crate::metal::encode_scatter_axpy_rows_unique_f32(
                    &ctx,
                    enc,
                    &out_n,
                    &token_ids_n,
                    &weights_n,
                    &accum_reduced,
                    h,
                    group.len,
                )
                .expect("scatter accum");
            }
        });
        accum_wall_ms += wall.elapsed().as_secs_f64() * 1e3;

        let cur = read_tensor_f32(&current_reduced);
        let acc = read_tensor_f32(&accum_reduced);
        cos_min = cos_min.min(cosine_f32(&cur, &acc));
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - acc[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[grouped-down-accum-{label}] chunk_p={chunk_p} current_gpu={:.2} ms accum_gpu={:.2} ms cpu_group={:.2} ms accum_wall={:.2} ms gpu_speedup={:.3} cos_min={:.6} max_abs={:.3e}",
        current_gpu_ms / denom,
        accum_gpu_ms / denom,
        cpu_group_ms / denom,
        accum_wall_ms / denom,
        (current_gpu_ms / denom) / (accum_gpu_ms / denom),
        cos_min,
        max_abs,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_routed_down_accum_proof_512() {
    run_grouped_routed_down_accum_proof(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 512, 2);
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_routed_down_accum_proof_512() {
    run_grouped_routed_down_accum_proof(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b", 512, 2);
}

fn run_fused_routed_tail_profile(model_path: &str, label: &str, chunk_p: usize, n_runs: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[fused-routed-tail-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(chunk_p * topk) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(chunk_p * topk * f_exp) as u64]);
    let current_out = scratch
        .mixer_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let fused_out = scratch
        .moe_shared_ffn_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    write_tensor_f32(&x_pack, &x_init);
    let _ = timed_gpu_cmd(&ctx, |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            post_norm,
            &h_pack,
            chunk_p,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .expect("warmup postnorm");
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &moe.gate_inp,
            &h_pack,
            &router_probs_pack,
            h,
            n_expert,
            chunk_p,
        )
        .expect("warmup route");
        encode_topk_logits_softmax_dot_sigmoid_packed_f32(
            &ctx,
            enc,
            &router_probs_pack,
            &moe.gate_inp_shexp,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &shared_gate_pack,
            n_expert,
            topk,
            h,
            chunk_p,
        )
        .expect("warmup topk/shared");
        encode_moe_swiglu_q4_K_f32_packed_slots(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &h_pack,
            &topk_idx_pack,
            &moe_inner_pack,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("warmup swiglu");
        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
            &ctx,
            enc,
            &moe.down_exps,
            &moe_inner_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &current_out,
            f_exp,
            h,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("warmup current down");
        encode_fill_f32(&ctx, enc, &fused_out, 0.0).expect("warmup fused zero");
        crate::metal::encode_moe_fused_routed_q4q5_token_f32(
            &ctx,
            enc,
            &moe.gate_exps,
            &moe.up_exps,
            &moe.down_exps,
            &h_pack,
            &topk_idx_pack,
            &topk_weight_pack,
            &fused_out,
            h,
            f_exp,
            n_expert,
            topk,
            chunk_p,
        )
        .expect("warmup fused routed");
    });

    let mut current_ms = 0.0f64;
    let mut fused_ms = 0.0f64;
    let mut cos_min = f64::INFINITY;
    let mut max_abs = 0.0f32;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
        });

        current_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current swiglu");
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &current_out,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current down");
        });

        fused_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &fused_out, 0.0).expect("fused zero");
            crate::metal::encode_moe_fused_routed_q4q5_token_f32(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &fused_out,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("fused routed");
        });

        let cur = read_tensor_f32(&current_out);
        let fus = read_tensor_f32(&fused_out);
        let cos = cosine_f32(&cur, &fus);
        cos_min = cos_min.min(cos);
        for i in 0..cur.len() {
            max_abs = max_abs.max((cur[i] - fus[i]).abs());
        }
    }

    let denom = n_runs as f64;
    eprintln!(
        "[fused-routed-tail-{label}] chunk_p={chunk_p} current={:.2} ms fused={:.2} ms speedup={:.3} cos_min={:.6} max_abs={:.3e}",
        current_ms / denom,
        fused_ms / denom,
        (current_ms / denom) / (fused_ms / denom),
        cos_min,
        max_abs
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_fused_routed_tail_profile() {
    run_fused_routed_tail_profile(crate::test_fixtures::A10B_Q4_K_XL.path(), "122b", 320, 2);
}

#[test]
#[ignore]
fn metal_35b_a3b_fused_routed_tail_profile() {
    run_fused_routed_tail_profile(crate::test_fixtures::A3B_Q4_K_M.path(), "a3b", 320, 2);
}

fn run_grouped_routed_down_experiment_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[grouped-down-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let slot_count = chunk_p * topk;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![(slot_count) as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![(slot_count) as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(slot_count * f_exp) as u64]);
    let mixer_out_pack = scratch
        .mixer_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let token_major_out = scratch
        .moe_expert_out_pack
        .view_subrange(0, vec![(slot_count * h) as u64]);
    let group_slot_ids = MetalTensor::zeros_i32(&ctx, vec![slot_count as u64]).expect("group ids");
    let group_token_ids =
        MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group tokens");
    let group_expert_ids =
        MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group experts");
    let group_weights =
        MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group weights");
    let grouped_inner =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped inner");
    let grouped_out =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("grouped out");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut current_down_gpu = 0.0f64;
    let mut grouped_cpu_ms = 0.0f64;
    let mut grouped_gpu_ms = 0.0f64;
    let mut grouped_wall_ms = 0.0f64;
    let mut group_count = 0usize;
    let mut max_group = 0usize;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("swiglu");
        });

        current_down_gpu += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &mixer_out_pack,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current down");
        });

        let prep_start = Instant::now();
        let topk_idx = read_tensor_i32_f32buf(&topk_idx_pack);
        let topk_weight = read_tensor_f32(&topk_weight_pack);
        let (groups, slot_ids, token_ids, weights) =
            build_expert_slot_groups(&topk_idx, &topk_weight, topk, n_expert);
        let mut expert_ids = Vec::with_capacity(slot_ids.len());
        for group in &groups {
            expert_ids.extend(std::iter::repeat_n(group.expert as i32, group.len));
        }
        group_count = groups.len();
        max_group = groups.iter().map(|g| g.len).max().unwrap_or(0);
        write_tensor_i32(&group_slot_ids, &slot_ids);
        write_tensor_i32_f32buf(&group_token_ids, &token_ids);
        write_tensor_i32_f32buf(&group_expert_ids, &expert_ids);
        write_tensor_f32(&group_weights, &weights);
        grouped_cpu_ms += prep_start.elapsed().as_secs_f64() * 1e3;

        let wall = Instant::now();
        grouped_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
            crate::metal::encode_get_rows_f32(
                &ctx,
                enc,
                &moe_inner_pack,
                &group_slot_ids,
                &grouped_inner,
                slot_count,
                f_exp,
            )
            .expect("gather grouped inner");
            crate::metal::encode_moe_down_q5_K_f32_grouped_rows(
                &ctx,
                enc,
                &moe.down_exps,
                &grouped_inner,
                &group_expert_ids,
                &grouped_out,
                f_exp,
                h,
                n_expert,
                slot_count,
            )
            .expect("grouped down matmat");
            crate::metal::encode_scatter_rows_f32_unique(
                &ctx,
                enc,
                &grouped_out,
                &group_slot_ids,
                &token_major_out,
                h,
                slot_count,
            )
            .expect("scatter grouped rows");
            crate::metal::encode_moe_weighted_sum_packed_f32(
                &ctx,
                enc,
                &token_major_out,
                &topk_weight_pack,
                &mixer_out_pack,
                h,
                topk,
                chunk_p,
            )
            .expect("packed weighted sum");
        });
        grouped_wall_ms += wall.elapsed().as_secs_f64() * 1e3;
    }

    let denom = n_runs as f64;
    eprintln!(
        "[grouped-down-{label}] chunk_p={chunk_p} groups={} max_group={} current_down_gpu={:.2} ms grouped_cpu={:.2} ms grouped_gpu={:.2} ms grouped_total={:.2} ms grouped_wall={:.2} ms speedup={:.3}",
        group_count,
        max_group,
        current_down_gpu / denom,
        grouped_cpu_ms / denom,
        grouped_gpu_ms / denom,
        (grouped_cpu_ms + grouped_gpu_ms) / denom,
        grouped_wall_ms / denom,
        (current_down_gpu / denom) / ((grouped_cpu_ms + grouped_gpu_ms) / denom)
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_grouped_routed_down_experiment_profile() {
    run_grouped_routed_down_experiment_profile(
        crate::test_fixtures::A10B_Q4_K_XL.path(),
        "122b",
        320,
        2,
    );
}

#[test]
#[ignore]
fn metal_35b_a3b_grouped_routed_down_experiment_profile() {
    run_grouped_routed_down_experiment_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b",
        320,
        2,
    );
}

fn run_hot_expert_routed_down_hybrid_profile(
    model_path: &str,
    label: &str,
    chunk_p: usize,
    hot_threshold: usize,
    n_runs: usize,
) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[hot-hybrid-{label}] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    assert_eq!(arch.kind, crate::model::ArchKind::Moe);
    let h = arch.hidden_size as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let slot_count = chunk_p * topk;

    let block = &mf.model.blocks[0];
    let (post_norm, moe) = match block {
        crate::metal_forward::MetalBlock::Gdn(g) => {
            (&g.post_attn_norm, g.ffn_moe.as_ref().expect("moe block"))
        }
        crate::metal_forward::MetalBlock::Attn(a) => {
            (&a.post_attn_norm, a.ffn_moe.as_ref().expect("moe block"))
        }
    };

    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");
    let x_pack = scratch.x_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(chunk_p * h) as u64]);
    let router_probs_pack = scratch
        .moe_router_probs_pack
        .view_subrange(0, vec![(chunk_p * n_expert) as u64]);
    let topk_idx_pack = scratch
        .moe_topk_idx_pack
        .view_subrange(0, vec![slot_count as u64]);
    let topk_weight_pack = scratch
        .moe_topk_weight_pack
        .view_subrange(0, vec![slot_count as u64]);
    let shared_gate_pack = scratch
        .moe_shared_gate_pack
        .view_subrange(0, vec![chunk_p as u64]);
    let moe_inner_pack = scratch
        .moe_inner_pack
        .view_subrange(0, vec![(slot_count * f_exp) as u64]);
    let hot_mixer_out = scratch
        .mixer_out_pack
        .view_subrange(0, vec![(chunk_p * h) as u64]);
    let group_slot_ids = MetalTensor::zeros_i32(&ctx, vec![slot_count as u64]).expect("group ids");
    let group_token_ids =
        MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group tokens");
    let group_weights =
        MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("group weights");
    let grouped_inner =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * f_exp) as u64]).expect("grouped inner");
    let grouped_out =
        MetalTensor::zeros_f32(&ctx, vec![(slot_count * h) as u64]).expect("grouped out");
    let cold_idx_pack = MetalTensor::zeros_f32(&ctx, vec![slot_count as u64]).expect("cold idx");
    let cold_out = MetalTensor::zeros_f32(&ctx, vec![(chunk_p * h) as u64]).expect("cold out");
    let x_init: Vec<f32> = (0..chunk_p * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();

    let mut current_down_gpu = 0.0f64;
    let mut hybrid_cpu_ms = 0.0f64;
    let mut hybrid_gpu_ms = 0.0f64;
    let mut hybrid_wall_ms = 0.0f64;
    let mut hot_groups_n = 0usize;
    let mut hot_slots_n = 0usize;

    for _ in 0..n_runs {
        write_tensor_f32(&x_pack, &x_init);
        let _ = timed_gpu_cmd(&ctx, |enc| {
            encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack,
                post_norm,
                &h_pack,
                chunk_p,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .expect("postnorm");
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &moe.gate_inp,
                &h_pack,
                &router_probs_pack,
                h,
                n_expert,
                chunk_p,
            )
            .expect("route");
            encode_topk_logits_softmax_dot_sigmoid_packed_f32(
                &ctx,
                enc,
                &router_probs_pack,
                &moe.gate_inp_shexp,
                &h_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &shared_gate_pack,
                n_expert,
                topk,
                h,
                chunk_p,
            )
            .expect("topk/shared");
            encode_moe_swiglu_q4_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.gate_exps,
                &moe.up_exps,
                &h_pack,
                &topk_idx_pack,
                &moe_inner_pack,
                h,
                f_exp,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("swiglu");
        });

        current_down_gpu += timed_gpu_cmd(&ctx, |enc| {
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &topk_idx_pack,
                &topk_weight_pack,
                &hot_mixer_out,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("current down");
        });

        let prep_start = Instant::now();
        let topk_idx = read_tensor_i32_f32buf(&topk_idx_pack);
        let topk_weight = read_tensor_f32(&topk_weight_pack);
        let (groups, slot_ids, token_ids, weights) =
            build_expert_slot_groups(&topk_idx, &topk_weight, topk, n_expert);
        let hot_groups: Vec<_> = groups
            .iter()
            .copied()
            .filter(|g| g.len >= hot_threshold)
            .collect();
        hot_groups_n = hot_groups.len();
        hot_slots_n = hot_groups.iter().map(|g| g.len).sum();
        let mut cold_idx = topk_idx.clone();
        for group in &hot_groups {
            for &slot_id in &slot_ids[group.start..group.start + group.len] {
                cold_idx[slot_id as usize] = -1;
            }
        }
        write_tensor_i32(&group_slot_ids, &slot_ids);
        write_tensor_i32_f32buf(&group_token_ids, &token_ids);
        write_tensor_f32(&group_weights, &weights);
        write_tensor_i32_f32buf(&cold_idx_pack, &cold_idx);
        hybrid_cpu_ms += prep_start.elapsed().as_secs_f64() * 1e3;

        let wall = Instant::now();
        hybrid_gpu_ms += timed_gpu_cmd(&ctx, |enc| {
            encode_fill_f32(&ctx, enc, &hot_mixer_out, 0.0).expect("zero hot");
            for group in &hot_groups {
                let slot_ids_n =
                    group_slot_ids.view_subrange(group.start as u64, vec![group.len as u64]);
                let token_ids_n =
                    group_token_ids.view_subrange(group.start as u64, vec![group.len as u64]);
                let weights_n =
                    group_weights.view_subrange(group.start as u64, vec![group.len as u64]);
                let inner_n = grouped_inner.view_subrange(
                    (group.start * f_exp) as u64,
                    vec![(group.len * f_exp) as u64],
                );
                let out_n = grouped_out
                    .view_subrange((group.start * h) as u64, vec![(group.len * h) as u64]);
                crate::metal::encode_get_rows_f32(
                    &ctx,
                    enc,
                    &moe_inner_pack,
                    &slot_ids_n,
                    &inner_n,
                    group.len,
                    f_exp,
                )
                .expect("gather hot inner");
                let expert_bytes = moe.down_exps.n_bytes() / n_expert as u64;
                let expert_w = moe
                    .down_exps
                    .view_bytes(group.expert as u64 * expert_bytes, vec![(h * f_exp) as u64]);
                encode_mat_mat_dispatch(
                    &ctx, enc, &expert_w, &inner_n, &out_n, f_exp, h, group.len,
                )
                .expect("hot grouped matmat");
                crate::metal::encode_scatter_axpy_rows_unique_f32(
                    &ctx,
                    enc,
                    &out_n,
                    &token_ids_n,
                    &weights_n,
                    &hot_mixer_out,
                    h,
                    group.len,
                )
                .expect("scatter hot out");
            }
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                enc,
                &moe.down_exps,
                &moe_inner_pack,
                &cold_idx_pack,
                &topk_weight_pack,
                &cold_out,
                f_exp,
                h,
                n_expert,
                topk,
                chunk_p,
            )
            .expect("cold down");
            encode_add_inplace_f32(&ctx, enc, &hot_mixer_out, &cold_out).expect("merge hot/cold");
        });
        hybrid_wall_ms += wall.elapsed().as_secs_f64() * 1e3;
    }

    let denom = n_runs as f64;
    eprintln!(
        "[hot-hybrid-{label}] chunk_p={chunk_p} threshold={hot_threshold} hot_groups={} hot_slots={} current_down_gpu={:.2} ms hybrid_cpu={:.2} ms hybrid_gpu={:.2} ms hybrid_total={:.2} ms hybrid_wall={:.2} ms speedup={:.3}",
        hot_groups_n,
        hot_slots_n,
        current_down_gpu / denom,
        hybrid_cpu_ms / denom,
        hybrid_gpu_ms / denom,
        (hybrid_cpu_ms + hybrid_gpu_ms) / denom,
        hybrid_wall_ms / denom,
        (current_down_gpu / denom) / ((hybrid_cpu_ms + hybrid_gpu_ms) / denom)
    );
}

#[test]
#[ignore]
fn metal_122b_a10b_hot_expert_routed_down_hybrid_profile() {
    for threshold in [64usize, 96, 128] {
        run_hot_expert_routed_down_hybrid_profile(
            crate::test_fixtures::A10B_Q4_K_XL.path(),
            "122b",
            320,
            threshold,
            2,
        );
    }
}

#[test]
#[ignore]
fn metal_35b_a3b_hot_expert_routed_down_hybrid_profile() {
    run_hot_expert_routed_down_hybrid_profile(
        crate::test_fixtures::A3B_Q4_K_M.path(),
        "a3b",
        320,
        64,
        2,
    );
}

fn run_packed_dense_prefill_phase_profile(model_path: &str, prompt: &str, chunk_p: usize) {
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[packed-prefill-phase] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    assert_eq!(m.arch.kind, crate::model::ArchKind::Dense);
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok.encode(prompt, false).expect("tokenize");
    let total_n = ids.len();
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let f = arch.intermediate_size as usize;
    let cap = total_n + 16;
    let mut sess = MetalSession::fresh(&ctx, &mm, cap).expect("session");
    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, chunk_p as u32).expect("scratch");

    let ids_buf = MetalTensor::zeros_i32(&ctx, vec![chunk_p as u64]).expect("ids buf");
    unsafe {
        let p = ids_buf.buffer.contents().as_ptr() as *mut i32;
        for (i, &t) in ids.iter().enumerate() {
            *p.add(i) = t;
        }
    }

    let x_pack_p = scratch.x_pack.view_subrange(0, vec![(total_n * h) as u64]);
    let h_pack_p = scratch.h_pack.view_subrange(0, vec![(total_n * h) as u64]);
    let mixer_out_pack_p = scratch
        .mixer_out_pack
        .view_subrange(0, vec![(total_n * h) as u64]);

    let timed =
        |label: &str,
         cb: &mut dyn FnMut(&KernelEncoder) -> Result<(), crate::metal_forward::MfError>|
         -> f64 {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            cb(&enc).expect(label);
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
        };

    let mut embed_ms = 0.0f64;
    let mut gdn_front_ms = 0.0f64;
    let mut gdn_alpha_beta_ms = 0.0f64;
    let mut gdn_tail_ms = 0.0f64;
    let mut gdn_back_ms = 0.0f64;
    let mut attn_front_ms = 0.0f64;
    let mut attn_decode_ms = 0.0f64;
    let mut attn_back_ms = 0.0f64;
    let mut ffn_ms = 0.0f64;
    let mut tail_ms = 0.0f64;

    embed_ms += timed("embed", &mut |enc| {
        let ids_view = ids_buf.view_subrange(0, vec![total_n as u64]);
        crate::metal::encode_get_rows_f32(
            &ctx,
            enc,
            &mf.model.token_embd,
            &ids_view,
            &x_pack_p,
            total_n,
            h,
        )
        .map_err(crate::metal_forward::MfError::from)
    });

    let gdn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
    let attn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;
    let ffn_mat_mat_eligible = prefill_mat_mat_dispatch_eligible;

    let mut gdn_idx = 0usize;
    let mut attn_idx = 0usize;
    for block in &mf.model.blocks {
        let attn_norm = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => &g.attn_norm,
            crate::metal_forward::MetalBlock::Attn(a) => &a.attn_norm,
        };
        let post_norm = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => &g.post_attn_norm,
            crate::metal_forward::MetalBlock::Attn(a) => &a.post_attn_norm,
        };
        let (g_w, u_w, d_w) = match block {
            crate::metal_forward::MetalBlock::Gdn(g) => (&g.ffn_gate, &g.ffn_up, &g.ffn_down),
            crate::metal_forward::MetalBlock::Attn(a) => (&a.ffn_gate, &a.ffn_up, &a.ffn_down),
        };

        timed("pre_mixer_norm", &mut |enc| {
            crate::metal::encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack_p,
                attn_norm,
                &h_pack_p,
                total_n,
                h,
                crate::metal_forward::RMS_EPS,
            )
            .map_err(crate::metal_forward::MfError::from)
        });

        match block {
            crate::metal_forward::MetalBlock::Gdn(g) => {
                let gi = gdn_idx;
                gdn_idx += 1;
                let n_k_u = arch.gdn_n_k_heads as usize;
                let n_v_u = arch.gdn_n_v_heads as usize;
                let head_dim_u = arch.gdn_head_dim as usize;
                let conv_dim = (2 * n_k_u + n_v_u) * head_dim_u;
                let v_dim = n_v_u * head_dim_u;
                let gdn_qkv_pack_p = scratch
                    .gdn_qkv_pack
                    .view_subrange(0, vec![(total_n * conv_dim) as u64]);
                let gdn_z_pack_p = scratch
                    .gdn_z_pack
                    .view_subrange(0, vec![(total_n * v_dim) as u64]);
                let gdn_beta_pack_p = scratch
                    .gdn_beta_pack
                    .view_subrange(0, vec![(total_n * n_v_u) as u64]);
                let gdn_alpha_pack_p = scratch
                    .gdn_alpha_pack
                    .view_subrange(0, vec![(total_n * n_v_u) as u64]);
                let gdn_q_norm_pack_p = scratch
                    .gdn_q_norm_pack
                    .view_subrange(0, vec![(total_n * n_k_u * head_dim_u) as u64]);
                let gdn_k_norm_pack_p = scratch
                    .gdn_k_norm_pack
                    .view_subrange(0, vec![(total_n * n_k_u * head_dim_u) as u64]);
                let gdn_v_pack_p = scratch
                    .gdn_v_pack
                    .view_subrange(0, vec![(total_n * v_dim) as u64]);
                let gdn_out_pack_p = scratch
                    .gdn_out_pack
                    .view_subrange(0, vec![(total_n * v_dim) as u64]);
                let gdn_normed_pack_p = scratch
                    .gdn_normed_pack
                    .view_subrange(0, vec![(total_n * v_dim) as u64]);
                let gdn_batched = gdn_mat_mat_eligible(g.in_proj_qkv.dtype)
                    && gdn_mat_mat_eligible(g.in_proj_z.dtype)
                    && gdn_mat_mat_eligible(g.out_proj.dtype);
                if gdn_batched {
                    gdn_front_ms += timed("gdn_front", &mut |enc| {
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &g.in_proj_qkv,
                            &h_pack_p,
                            &gdn_qkv_pack_p,
                            h,
                            conv_dim,
                            total_n,
                        )?;
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &g.in_proj_z,
                            &h_pack_p,
                            &gdn_z_pack_p,
                            h,
                            v_dim,
                            total_n,
                        )?;
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &g.beta_proj,
                            &h_pack_p,
                            &gdn_beta_pack_p,
                            h,
                            n_v_u,
                            total_n,
                        )?;
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &g.alpha_proj,
                            &h_pack_p,
                            &gdn_alpha_pack_p,
                            h,
                            n_v_u,
                            total_n,
                        )
                    });
                    gdn_alpha_beta_ms += timed("gdn_alpha_beta", &mut |enc| {
                        crate::metal::encode_sigmoid_f32(
                            &ctx,
                            enc,
                            &gdn_beta_pack_p,
                            &gdn_beta_pack_p,
                        )?;
                        encode_gdn_decay_chain_batched_f32(
                            &ctx,
                            enc,
                            &gdn_alpha_pack_p,
                            &g.dt_bias,
                            &g.a_log,
                            &gdn_alpha_pack_p,
                            total_n,
                            n_v_u,
                        )
                        .map_err(crate::metal_forward::MfError::from)
                    });
                    if dense_packed_gdn_step_enabled() {
                        let mut conv_stage_ms = 0.0f64;
                        let mut step_stage_ms = 0.0f64;
                        let mut rms_stage_ms = 0.0f64;
                        conv_stage_ms += timed("gdn_tail_pre_step", &mut |enc| {
                            for n_idx in 0..total_n {
                                let qkv_n = gdn_qkv_pack_p.view_subrange(
                                    (n_idx * conv_dim) as u64,
                                    vec![conv_dim as u64],
                                );
                                encode_ssm_conv_silu_f32(
                                    &ctx,
                                    enc,
                                    &qkv_n,
                                    &sess.gdn_conv[gi],
                                    &g.conv1d,
                                    &sess.gdn_qkv_conv,
                                    conv_dim,
                                )?;
                                let q_view = sess
                                    .gdn_qkv_conv
                                    .view_subrange(0, vec![(n_k_u * head_dim_u) as u64]);
                                let k_view = sess.gdn_qkv_conv.view_subrange(
                                    (n_k_u * head_dim_u) as u64,
                                    vec![(n_k_u * head_dim_u) as u64],
                                );
                                let v_view = sess.gdn_qkv_conv.view_subrange(
                                    (2 * n_k_u * head_dim_u) as u64,
                                    vec![v_dim as u64],
                                );
                                encode_l2_norm_batched_f32(
                                    &ctx,
                                    enc,
                                    &q_view,
                                    &sess.gdn_q_norm,
                                    n_k_u,
                                    head_dim_u,
                                    crate::metal_forward::RMS_EPS,
                                )?;
                                encode_l2_norm_batched_f32(
                                    &ctx,
                                    enc,
                                    &k_view,
                                    &sess.gdn_k_norm,
                                    n_k_u,
                                    head_dim_u,
                                    crate::metal_forward::RMS_EPS,
                                )?;
                                encode_scatter_offset_f32(
                                    &ctx,
                                    enc,
                                    &sess.gdn_q_norm,
                                    &gdn_q_norm_pack_p,
                                    n_idx * n_k_u * head_dim_u,
                                    n_k_u * head_dim_u,
                                )?;
                                encode_scatter_offset_f32(
                                    &ctx,
                                    enc,
                                    &sess.gdn_k_norm,
                                    &gdn_k_norm_pack_p,
                                    n_idx * n_k_u * head_dim_u,
                                    n_k_u * head_dim_u,
                                )?;
                                encode_scatter_offset_f32(
                                    &ctx,
                                    enc,
                                    &v_view,
                                    &gdn_v_pack_p,
                                    n_idx * v_dim,
                                    v_dim,
                                )?;
                            }
                            Ok(())
                        });
                        step_stage_ms += timed("gdn_tail_step", &mut |enc| {
                            encode_gdn_step_decay_packed_f32(
                                &ctx,
                                enc,
                                &gdn_q_norm_pack_p,
                                &gdn_k_norm_pack_p,
                                &gdn_v_pack_p,
                                &gdn_alpha_pack_p,
                                &gdn_beta_pack_p,
                                &sess.gdn_state[gi],
                                &gdn_out_pack_p,
                                total_n,
                                n_v_u,
                                n_k_u,
                                head_dim_u,
                            )
                            .map_err(crate::metal_forward::MfError::from)
                        });
                        rms_stage_ms += timed("gdn_tail_post_step", &mut |enc| {
                            for n_idx in 0..total_n {
                                let out_n = gdn_out_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                let z_n = gdn_z_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                let normed_n = gdn_normed_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                encode_rmsnorm_gated_f32(
                                    &ctx,
                                    enc,
                                    &out_n,
                                    &g.norm,
                                    &z_n,
                                    &normed_n,
                                    n_v_u,
                                    head_dim_u,
                                    crate::metal_forward::RMS_EPS * head_dim_u as f32,
                                )?;
                            }
                            Ok(())
                        });
                        gdn_tail_ms += conv_stage_ms + step_stage_ms + rms_stage_ms;
                    } else {
                        gdn_tail_ms += timed("gdn_tail", &mut |enc| {
                            for n_idx in 0..total_n {
                                let qkv_n = gdn_qkv_pack_p.view_subrange(
                                    (n_idx * conv_dim) as u64,
                                    vec![conv_dim as u64],
                                );
                                let z_n = gdn_z_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                let beta_n = gdn_beta_pack_p
                                    .view_subrange((n_idx * n_v_u) as u64, vec![n_v_u as u64]);
                                let alpha_n = gdn_alpha_pack_p
                                    .view_subrange((n_idx * n_v_u) as u64, vec![n_v_u as u64]);
                                let normed_n = gdn_normed_pack_p
                                    .view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
                                mf.encode_gdn_tail(
                                    enc, g, gi, &mut sess, &qkv_n, &z_n, &alpha_n, &beta_n,
                                    &normed_n,
                                )?;
                            }
                            Ok(())
                        });
                    }
                    gdn_back_ms += timed("gdn_back", &mut |enc| {
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &g.out_proj,
                            &gdn_normed_pack_p,
                            &mixer_out_pack_p,
                            v_dim,
                            h,
                            total_n,
                        )?;
                        crate::metal::encode_add_inplace_f32(
                            &ctx,
                            enc,
                            &x_pack_p,
                            &mixer_out_pack_p,
                        )
                        .map_err(crate::metal_forward::MfError::from)
                    });
                } else {
                    gdn_tail_ms += timed("gdn_fallback", &mut |enc| {
                        for n_idx in 0..total_n {
                            encode_copy_offset_f32(&ctx, enc, &h_pack_p, n_idx * h, &sess.h, h)?;
                            mf.encode_gdn(enc, g, gi, &mut sess)?;
                            crate::metal_forward::encode_scatter_offset_f32(
                                &ctx,
                                enc,
                                &sess.mixer_out,
                                &mixer_out_pack_p,
                                n_idx * h,
                                h,
                            )?;
                        }
                        Ok(())
                    });
                    gdn_back_ms += timed("gdn_resid", &mut |enc| {
                        crate::metal::encode_add_inplace_f32(
                            &ctx,
                            enc,
                            &x_pack_p,
                            &mixer_out_pack_p,
                        )
                        .map_err(crate::metal_forward::MfError::from)
                    });
                }
            }
            crate::metal_forward::MetalBlock::Attn(a) => {
                let ai = attn_idx;
                attn_idx += 1;
                let head_dim = arch.attn_head_dim as usize;
                let n_q = arch.n_q_heads as usize;
                let n_kv = arch.n_kv_heads as usize;
                let q_dim = n_q * head_dim;
                let kv_dim = n_kv * head_dim;
                let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
                let q_full_pack_p = scratch
                    .attn_q_full_pack
                    .view_subrange(0, vec![(total_n * 2 * q_dim) as u64]);
                // v0.447: the production packs are stubbed; this profile
                // helper times the OLD split path, so it allocates its
                // own buffers.
                let q_pack_p =
                    MetalTensor::zeros_f32(&ctx, vec![(total_n * q_dim) as u64]).unwrap();
                let gate_pack_p =
                    MetalTensor::zeros_f32(&ctx, vec![(total_n * q_dim) as u64]).unwrap();
                let q_normed_pack_p = scratch
                    .attn_q_normed_pack
                    .view_subrange(0, vec![(total_n * q_dim) as u64]);
                let k_now_pack_p = scratch
                    .attn_k_now_pack
                    .view_subrange(0, vec![(total_n * kv_dim) as u64]);
                let v_now_pack_p = scratch
                    .attn_v_now_pack
                    .view_subrange(0, vec![(total_n * kv_dim) as u64]);
                let k_normed_pack_p = scratch
                    .attn_k_normed_pack
                    .view_subrange(0, vec![(total_n * kv_dim) as u64]);
                let attn_o_pack_p = scratch
                    .attn_o_pack
                    .view_subrange(0, vec![(total_n * q_dim) as u64]);
                let attn_batched = attn_mat_mat_eligible(a.q.dtype)
                    && attn_mat_mat_eligible(a.k.dtype)
                    && attn_mat_mat_eligible(a.v.dtype)
                    && attn_mat_mat_eligible(a.o.dtype);
                if attn_batched {
                    attn_front_ms += timed("attn_front", &mut |enc| {
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &a.q,
                            &h_pack_p,
                            &q_full_pack_p,
                            h,
                            2 * q_dim,
                            total_n,
                        )?;
                        crate::metal::encode_split_q_gate_f32(
                            &ctx,
                            enc,
                            &q_full_pack_p,
                            &q_pack_p,
                            &gate_pack_p,
                            total_n * n_q,
                            head_dim,
                        )?;
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &a.k,
                            &h_pack_p,
                            &k_now_pack_p,
                            h,
                            kv_dim,
                            total_n,
                        )?;
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &a.v,
                            &h_pack_p,
                            &v_now_pack_p,
                            h,
                            kv_dim,
                            total_n,
                        )?;
                        crate::metal::encode_rms_norm_batched_f32(
                            &ctx,
                            enc,
                            &q_pack_p,
                            &a.q_norm,
                            &q_normed_pack_p,
                            total_n * n_q,
                            head_dim,
                            crate::metal_forward::RMS_EPS,
                        )?;
                        crate::metal::encode_rms_norm_batched_f32(
                            &ctx,
                            enc,
                            &k_now_pack_p,
                            &a.k_norm,
                            &k_normed_pack_p,
                            total_n * n_kv,
                            head_dim,
                            crate::metal_forward::RMS_EPS,
                        )
                        .map_err(crate::metal_forward::MfError::from)
                    });
                    attn_decode_ms += timed("attn_decode", &mut |enc| {
                        crate::metal::encode_rope_neox_f32_packed_consecutive(
                            &ctx,
                            enc,
                            &q_normed_pack_p,
                            total_n,
                            n_q,
                            head_dim,
                            n_rot,
                            0,
                            arch.rope_theta,
                        )?;
                        crate::metal::encode_rope_neox_f32_packed_consecutive(
                            &ctx,
                            enc,
                            &k_normed_pack_p,
                            total_n,
                            n_kv,
                            head_dim,
                            n_rot,
                            0,
                            arch.rope_theta,
                        )?;
                        crate::metal::encode_scatter_offset_f32_to_f16_kv(
                            &ctx,
                            enc,
                            &k_normed_pack_p,
                            &v_now_pack_p,
                            &sess.kv_k[ai],
                            &sess.kv_v[ai],
                            0,
                            total_n * kv_dim,
                        )?;
                        for n_idx in 0..total_n {
                            let q_normed_n = q_normed_pack_p
                                .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                            let attn_o_n = attn_o_pack_p
                                .view_subrange((n_idx * q_dim) as u64, vec![q_dim as u64]);
                            sess.kv_n_pos[ai] = n_idx + 1;
                            let group = n_q / n_kv;
                            let nwg = crate::metal::attn_v4_choose_nwg(sess.kv_n_pos[ai], group);
                            let tile_c =
                                crate::metal::attn_v4_choose_tile_c(sess.kv_n_pos[ai], group);
                            crate::metal::encode_attn_decode_v4_f32(
                                &ctx,
                                enc,
                                &q_normed_n,
                                &sess.kv_k[ai],
                                &sess.kv_v[ai],
                                &sess.attn_v4_o_partial,
                                &sess.attn_v4_ml_partial,
                                &attn_o_n,
                                n_q,
                                n_kv,
                                head_dim,
                                sess.kv_n_pos[ai],
                                nwg,
                                tile_c,
                            )?;
                        }
                        Ok(())
                    });
                    attn_back_ms += timed("attn_back", &mut |enc| {
                        crate::metal::encode_sigmoid_f32(&ctx, enc, &gate_pack_p, &q_pack_p)?;
                        crate::metal::encode_mul_f32(
                            &ctx,
                            enc,
                            &attn_o_pack_p,
                            &q_pack_p,
                            &attn_o_pack_p,
                        )?;
                        encode_mat_mat_dispatch(
                            &ctx,
                            enc,
                            &a.o,
                            &attn_o_pack_p,
                            &mixer_out_pack_p,
                            q_dim,
                            h,
                            total_n,
                        )?;
                        crate::metal::encode_add_inplace_f32(
                            &ctx,
                            enc,
                            &x_pack_p,
                            &mixer_out_pack_p,
                        )
                        .map_err(crate::metal_forward::MfError::from)
                    });
                } else {
                    attn_decode_ms += timed("attn_fallback", &mut |enc| {
                        for n_idx in 0..total_n {
                            encode_copy_offset_f32(&ctx, enc, &h_pack_p, n_idx * h, &sess.h, h)?;
                            mf.encode_attn(enc, a, ai, n_idx as u32, &mut sess)?;
                            crate::metal_forward::encode_scatter_offset_f32(
                                &ctx,
                                enc,
                                &sess.mixer_out,
                                &mixer_out_pack_p,
                                n_idx * h,
                                h,
                            )?;
                        }
                        Ok(())
                    });
                    attn_back_ms += timed("attn_resid", &mut |enc| {
                        crate::metal::encode_add_inplace_f32(
                            &ctx,
                            enc,
                            &x_pack_p,
                            &mixer_out_pack_p,
                        )
                        .map_err(crate::metal_forward::MfError::from)
                    });
                }
            }
        }

        let mat_mat_path = ffn_mat_mat_eligible(g_w.dtype)
            && ffn_mat_mat_eligible(u_w.dtype)
            && ffn_mat_mat_eligible(d_w.dtype);
        let ffn_gate_pack_p = scratch
            .ffn_gate_pack
            .view_subrange(0, vec![(total_n * f) as u64]);
        let ffn_up_pack_p = scratch
            .ffn_up_pack
            .view_subrange(0, vec![(total_n * f) as u64]);
        let ffn_inner_pack_p = scratch
            .ffn_inner_pack
            .view_subrange(0, vec![(total_n * f) as u64]);
        let ffn_out_pack_p = scratch
            .ffn_out_pack
            .view_subrange(0, vec![(total_n * h) as u64]);
        ffn_ms += timed("ffn", &mut |enc| {
            crate::metal::encode_rms_norm_batched_f32(
                &ctx,
                enc,
                &x_pack_p,
                post_norm,
                &h_pack_p,
                total_n,
                h,
                crate::metal_forward::RMS_EPS,
            )?;
            if mat_mat_path {
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    g_w,
                    &h_pack_p,
                    &ffn_gate_pack_p,
                    h,
                    f,
                    total_n,
                )?;
                encode_mat_mat_dispatch(&ctx, enc, u_w, &h_pack_p, &ffn_up_pack_p, h, f, total_n)?;
                crate::metal::encode_silu_mul_f32(
                    &ctx,
                    enc,
                    &ffn_gate_pack_p,
                    &ffn_up_pack_p,
                    &ffn_inner_pack_p,
                )?;
                encode_mat_mat_dispatch(
                    &ctx,
                    enc,
                    d_w,
                    &ffn_inner_pack_p,
                    &ffn_out_pack_p,
                    f,
                    h,
                    total_n,
                )?;
            } else {
                for n_idx in 0..total_n {
                    let h_n = h_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                    let gate_n = ffn_gate_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                    let up_n = ffn_up_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                    let inner_n =
                        ffn_inner_pack_p.view_subrange((n_idx * f) as u64, vec![f as u64]);
                    let out_n = ffn_out_pack_p.view_subrange((n_idx * h) as u64, vec![h as u64]);
                    crate::metal_forward::encode_mat_vec_dispatch(
                        &ctx, enc, g_w, &h_n, &gate_n, h, f,
                    )?;
                    crate::metal_forward::encode_mat_vec_dispatch(
                        &ctx, enc, u_w, &h_n, &up_n, h, f,
                    )?;
                    crate::metal::encode_silu_mul_f32(&ctx, enc, &gate_n, &up_n, &inner_n)?;
                    crate::metal_forward::encode_mat_vec_dispatch(
                        &ctx, enc, d_w, &inner_n, &out_n, f, h,
                    )?;
                }
            }
            crate::metal::encode_add_inplace_f32(&ctx, enc, &x_pack_p, &ffn_out_pack_p)
                .map_err(crate::metal_forward::MfError::from)
        });
    }

    tail_ms += timed("tail", &mut |enc| {
        let x_last = x_pack_p.view_subrange(((total_n - 1) * h) as u64, vec![h as u64]);
        crate::metal::encode_rms_norm_mul_f32(
            &ctx,
            enc,
            &x_last,
            &mf.model.output_norm,
            &sess.h,
            crate::metal_forward::RMS_EPS,
        )?;
        crate::metal_forward::encode_mat_vec_dispatch(
            &ctx,
            enc,
            &mf.model.lm_head,
            &sess.h,
            &sess.logits,
            h,
            arch.vocab_size as usize,
        )
    });

    let total = embed_ms
        + gdn_front_ms
        + gdn_alpha_beta_ms
        + gdn_tail_ms
        + gdn_back_ms
        + attn_front_ms
        + attn_decode_ms
        + attn_back_ms
        + ffn_ms
        + tail_ms;
    eprintln!(
        "[packed-prefill-phase] prompt_tokens={} chunk_p={chunk_p} phase_sum={total:.2} ms",
        total_n
    );
    eprintln!(
        "[packed-prefill-phase]   embed         {:7.2} ms ({:5.1}%)",
        embed_ms,
        embed_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   gdn_front     {:7.2} ms ({:5.1}%)",
        gdn_front_ms,
        gdn_front_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   gdn_alpha_beta{:7.2} ms ({:5.1}%)",
        gdn_alpha_beta_ms,
        gdn_alpha_beta_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   gdn_tail      {:7.2} ms ({:5.1}%)",
        gdn_tail_ms,
        gdn_tail_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   gdn_back      {:7.2} ms ({:5.1}%)",
        gdn_back_ms,
        gdn_back_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   attn_front    {:7.2} ms ({:5.1}%)",
        attn_front_ms,
        attn_front_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   attn_decode   {:7.2} ms ({:5.1}%)",
        attn_decode_ms,
        attn_decode_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   attn_back     {:7.2} ms ({:5.1}%)",
        attn_back_ms,
        attn_back_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   ffn           {:7.2} ms ({:5.1}%)",
        ffn_ms,
        ffn_ms / total * 100.0
    );
    eprintln!(
        "[packed-prefill-phase]   tail          {:7.2} ms ({:5.1}%)",
        tail_ms,
        tail_ms / total * 100.0
    );
}

#[test]
#[ignore]
fn metal_27b_packed_prefill_phase_profile() {
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(32);
    run_packed_dense_prefill_phase_profile(
        crate::test_fixtures::QWEN36_27B_Q4_K_M.path(),
        &prompt,
        321,
    );
}

#[test]
#[ignore]
fn metal_27b_packed_gdn_tail_profile() {
    let model_path = crate::test_fixtures::QWEN36_27B_Q4_K_M.path();
    if !std::path::Path::new(model_path).exists() {
        eprintln!("[packed-gdn-tail] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(model_path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let tok = crate::tokenizer::Tokenizer::open(model_path).expect("tok");
    let ids = tok
        .encode(
            &"The quick brown fox jumps over the lazy dog. ".repeat(32),
            false,
        )
        .expect("tokenize");
    let total_n = ids.len();
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let g = match &mf.model.blocks[0] {
        crate::metal_forward::MetalBlock::Gdn(g) => g,
        _ => panic!("expected block 0 gdn"),
    };

    let sess = MetalSession::fresh(&ctx, &mm, total_n + 16).expect("session");
    let scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, total_n as u32).expect("scratch");
    let ids_buf = MetalTensor::zeros_i32(&ctx, vec![total_n as u64]).expect("ids buf");
    unsafe {
        let p = ids_buf.buffer.contents().as_ptr() as *mut i32;
        for (i, &t) in ids.iter().enumerate() {
            *p.add(i) = t;
        }
    }

    let x_pack = scratch.x_pack.view_subrange(0, vec![(total_n * h) as u64]);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(total_n * h) as u64]);
    let gdn_qkv_pack = scratch
        .gdn_qkv_pack
        .view_subrange(0, vec![(total_n * conv_dim) as u64]);
    let gdn_z_pack = scratch
        .gdn_z_pack
        .view_subrange(0, vec![(total_n * v_dim) as u64]);
    let gdn_beta_pack = scratch
        .gdn_beta_pack
        .view_subrange(0, vec![(total_n * n_v) as u64]);
    let gdn_alpha_pack = scratch
        .gdn_alpha_pack
        .view_subrange(0, vec![(total_n * n_v) as u64]);
    let gdn_normed_pack = scratch
        .gdn_normed_pack
        .view_subrange(0, vec![(total_n * v_dim) as u64]);

    let timed =
        |label: &str,
         cb: &mut dyn FnMut(&KernelEncoder) -> Result<(), crate::metal_forward::MfError>|
         -> f64 {
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = KernelEncoder::begin(&cmd);
            cb(&enc).expect(label);
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3
        };

    // Stage packed inputs for one representative GDN block.
    let _ = timed("warm_embed", &mut |enc| {
        let ids_view = ids_buf.view_subrange(0, vec![total_n as u64]);
        encode_get_rows_f32(
            &ctx,
            enc,
            &mf.model.token_embd,
            &ids_view,
            &x_pack,
            total_n,
            h,
        )
        .map_err(crate::metal_forward::MfError::from)
    });
    let _ = timed("warm_norm", &mut |enc| {
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &x_pack,
            &g.attn_norm,
            &h_pack,
            total_n,
            h,
            crate::metal_forward::RMS_EPS,
        )
        .map_err(crate::metal_forward::MfError::from)
    });
    let _ = timed("warm_front", &mut |enc| {
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &g.in_proj_qkv,
            &h_pack,
            &gdn_qkv_pack,
            h,
            conv_dim,
            total_n,
        )?;
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &g.in_proj_z,
            &h_pack,
            &gdn_z_pack,
            h,
            v_dim,
            total_n,
        )?;
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &g.beta_proj,
            &h_pack,
            &gdn_beta_pack,
            h,
            n_v,
            total_n,
        )?;
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &g.alpha_proj,
            &h_pack,
            &gdn_alpha_pack,
            h,
            n_v,
            total_n,
        )
    });
    let _ = timed("warm_alpha_beta", &mut |enc| {
        encode_sigmoid_f32(&ctx, enc, &gdn_beta_pack, &gdn_beta_pack)?;
        encode_gdn_decay_chain_batched_f32(
            &ctx,
            enc,
            &gdn_alpha_pack,
            &g.dt_bias,
            &g.a_log,
            &gdn_alpha_pack,
            total_n,
            n_v,
        )
        .map_err(crate::metal_forward::MfError::from)
    });

    let mut conv_ms = 0.0f64;
    let mut l2_ms = 0.0f64;
    let mut step_ms = 0.0f64;
    let mut rms_ms = 0.0f64;
    let mut out_ms = 0.0f64;
    for n_idx in 0..total_n {
        let qkv_n = gdn_qkv_pack.view_subrange((n_idx * conv_dim) as u64, vec![conv_dim as u64]);
        let z_n = gdn_z_pack.view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);
        let beta_n = gdn_beta_pack.view_subrange((n_idx * n_v) as u64, vec![n_v as u64]);
        let alpha_n = gdn_alpha_pack.view_subrange((n_idx * n_v) as u64, vec![n_v as u64]);
        let normed_n = gdn_normed_pack.view_subrange((n_idx * v_dim) as u64, vec![v_dim as u64]);

        conv_ms += timed("conv", &mut |enc| {
            encode_ssm_conv_silu_f32(
                &ctx,
                enc,
                &qkv_n,
                &sess.gdn_conv[0],
                &g.conv1d,
                &sess.gdn_qkv_conv,
                conv_dim,
            )
            .map_err(crate::metal_forward::MfError::from)
        });
        let q_view = sess
            .gdn_qkv_conv
            .view_subrange(0, vec![(n_k * head_dim) as u64]);
        let k_view = sess
            .gdn_qkv_conv
            .view_subrange((n_k * head_dim) as u64, vec![(n_k * head_dim) as u64]);
        let v_view = sess
            .gdn_qkv_conv
            .view_subrange((2 * n_k * head_dim) as u64, vec![(n_v * head_dim) as u64]);
        l2_ms += timed("l2", &mut |enc| {
            encode_l2_norm_batched_f32(
                &ctx,
                enc,
                &q_view,
                &sess.gdn_q_norm,
                n_k,
                head_dim,
                crate::metal_forward::RMS_EPS,
            )?;
            encode_l2_norm_batched_f32(
                &ctx,
                enc,
                &k_view,
                &sess.gdn_k_norm,
                n_k,
                head_dim,
                crate::metal_forward::RMS_EPS,
            )
            .map_err(crate::metal_forward::MfError::from)
        });
        step_ms += timed("step", &mut |enc| {
            encode_gdn_step_decay_f32(
                &ctx,
                enc,
                &sess.gdn_q_norm,
                &sess.gdn_k_norm,
                &v_view,
                &alpha_n,
                &beta_n,
                &sess.gdn_state[0],
                &sess.gdn_out,
                n_v,
                n_k,
                head_dim,
            )
            .map_err(crate::metal_forward::MfError::from)
        });
        rms_ms += timed("rmsnorm_gated", &mut |enc| {
            encode_rmsnorm_gated_f32(
                &ctx,
                enc,
                &sess.gdn_out,
                &g.norm,
                &z_n,
                &normed_n,
                n_v,
                head_dim,
                crate::metal_forward::RMS_EPS * head_dim as f32,
            )
            .map_err(crate::metal_forward::MfError::from)
        });
        out_ms += timed("out_proj", &mut |enc| {
            crate::metal_forward::encode_mat_vec_dispatch(
                &ctx,
                enc,
                &g.out_proj,
                &normed_n,
                &sess.mixer_out,
                v_dim,
                h,
            )
        });
    }

    let total = conv_ms + l2_ms + step_ms + rms_ms + out_ms;
    eprintln!(
        "[packed-gdn-tail] prompt_tokens={} one-layer total={total:.2} ms",
        total_n
    );
    eprintln!(
        "[packed-gdn-tail]   conv           {:7.2} ms ({:5.1}%)",
        conv_ms,
        conv_ms / total * 100.0
    );
    eprintln!(
        "[packed-gdn-tail]   l2             {:7.2} ms ({:5.1}%)",
        l2_ms,
        l2_ms / total * 100.0
    );
    eprintln!(
        "[packed-gdn-tail]   step_decay     {:7.2} ms ({:5.1}%)",
        step_ms,
        step_ms / total * 100.0
    );
    eprintln!(
        "[packed-gdn-tail]   rmsnorm_gated  {:7.2} ms ({:5.1}%)",
        rms_ms,
        rms_ms / total * 100.0
    );
    eprintln!(
        "[packed-gdn-tail]   out_proj       {:7.2} ms ({:5.1}%)",
        out_ms,
        out_ms / total * 100.0
    );
}

/// H5.3a foundation: verify `MetalDFlashVerifyScratch` allocates
/// correctly-sized buffers, and that `slot_view` helpers land at
/// the right offsets with the right shapes. Uses the 0.8B oracle
/// (24 layers, all GDN — so n_gdn = n_layer = 24, smaller than
/// 27B's 48). Loads in <1 s.
#[test]
fn dflash_verify_scratch_slots_and_offsets() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[dflash-scratch] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    // Pretend we have a DFlash drafter with N=8, K=3 (synthetic;
    // doesn't have to match a real drafter — we're only testing the
    // scratch struct's offset arithmetic against the 0.8B target arch).
    let n: u32 = 8;
    let k: u32 = 3;
    let scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, n, k).expect("scratch alloc");

    // Sanity on cached dims.
    let arch = &mm.arch;
    assert_eq!(scratch.n, n);
    assert_eq!(scratch.k_target_layers, k);
    assert_eq!(scratch.hidden_size, arch.hidden_size as u64);
    let expected_ssm =
        (arch.gdn_n_v_heads as u64) * (arch.gdn_head_dim as u64) * (arch.gdn_head_dim as u64);
    let expected_conv = ((arch.gdn_conv_kernel as u64) - 1)
        * (2 * (arch.gdn_n_k_heads as u64) + (arch.gdn_n_v_heads as u64))
        * (arch.gdn_head_dim as u64);
    assert_eq!(scratch.ssm_state_elems, expected_ssm);
    assert_eq!(scratch.conv_state_elems, expected_conv);
    let expected_n_gdn = mm
        .blocks
        .iter()
        .filter(|b| matches!(b, crate::metal_forward::MetalBlock::Gdn(_)))
        .count() as u32;
    assert_eq!(scratch.n_gdn_layers, expected_n_gdn);
    eprintln!(
        "[dflash-scratch] H={} n_gdn={} ssm_elems={} conv_elems={}",
        scratch.hidden_size,
        scratch.n_gdn_layers,
        scratch.ssm_state_elems,
        scratch.conv_state_elems
    );

    // -- backing buffer sizes --
    let f32_size = std::mem::size_of::<f32>() as u64;
    assert_eq!(
        scratch.gdn_ckpt.shape,
        vec![scratch.n_gdn_layers as u64, n as u64, expected_ssm]
    );
    assert_eq!(
        scratch.gdn_ckpt.n_elements(),
        scratch.n_gdn_layers as u64 * n as u64 * expected_ssm
    );
    assert_eq!(
        scratch.conv_ckpt.shape,
        vec![scratch.n_gdn_layers as u64, n as u64, expected_conv]
    );
    // v0.71: layout switched from [K, N, H] to [N, K, H] for
    // contiguous-by-N reads (target_ctx append in H5.5 outer loop).
    assert_eq!(
        scratch.hidden_capture.shape,
        vec![n as u64, k as u64, scratch.hidden_size]
    );
    assert_eq!(scratch.packed_ids_buf.shape, vec![n as u64]);
    assert_eq!(scratch.verify_argmax.shape, vec![n as u64]);
    assert_eq!(scratch.packed_ids_buf.dtype, GgmlType::I32);
    assert_eq!(scratch.verify_argmax.dtype, GgmlType::I32);

    // -- gdn_ckpt_slot offsets --
    // Slot (layer, n) should land at offset (layer * N + n) * ssm_elems
    // F32 elements. View shape = [ssm_elems].
    for layer in 0..scratch.n_gdn_layers {
        for nn in 0..n {
            let slot = scratch.gdn_ckpt_slot(layer, nn);
            let expected_elem_off = (layer as u64 * n as u64 + nn as u64) * expected_ssm;
            let expected_byte_off = expected_elem_off * f32_size;
            assert_eq!(
                slot.shape,
                vec![expected_ssm],
                "gdn slot ({layer},{nn}) shape"
            );
            assert_eq!(
                slot.offset, expected_byte_off,
                "gdn slot ({layer},{nn}) byte offset"
            );
            // Slot must share the underlying buffer with the parent.
            let slot_buf_ptr: *const _ = &*slot.buffer;
            let parent_buf_ptr: *const _ = &*scratch.gdn_ckpt.buffer;
            assert_eq!(
                slot_buf_ptr, parent_buf_ptr,
                "gdn slot does not share buffer with parent"
            );
        }
    }

    // -- conv_ckpt_slot offsets --
    for layer in 0..scratch.n_gdn_layers.min(4) {
        for nn in [0, n / 2, n - 1] {
            let slot = scratch.conv_ckpt_slot(layer, nn);
            let expected_elem_off = (layer as u64 * n as u64 + nn as u64) * expected_conv;
            assert_eq!(slot.shape, vec![expected_conv]);
            assert_eq!(slot.offset, expected_elem_off * f32_size);
        }
    }

    // -- hidden_capture_slot offsets (v0.71: [N, K, H] layout) --
    for kk in 0..k {
        for nn in 0..n {
            let slot = scratch.hidden_capture_slot(kk, nn);
            let expected_elem_off = (nn as u64 * k as u64 + kk as u64) * scratch.hidden_size;
            assert_eq!(slot.shape, vec![scratch.hidden_size]);
            assert_eq!(slot.offset, expected_elem_off * f32_size);
        }
    }
    // -- hidden_capture_n_slot (NEW v0.71): K*H contiguous per token --
    for nn in 0..n {
        let n_slot = scratch.hidden_capture_n_slot(nn);
        let kh = k as u64 * scratch.hidden_size;
        assert_eq!(n_slot.shape, vec![kh]);
        assert_eq!(n_slot.offset, (nn as u64) * kh * f32_size);
    }

    // -- token_slot / argmax_slot — single-element views --
    for nn in 0..n {
        let tok_slot = scratch.token_slot(nn);
        assert_eq!(tok_slot.shape, vec![1]);
        assert_eq!(tok_slot.offset, (nn as u64) * f32_size);
        assert_eq!(tok_slot.dtype, GgmlType::I32);
        let am_slot = scratch.argmax_slot(nn);
        assert_eq!(am_slot.shape, vec![1]);
        assert_eq!(am_slot.offset, (nn as u64) * f32_size);
        assert_eq!(am_slot.dtype, GgmlType::I32);
    }

    // -- write/read round-trip via a slot, to confirm the underlying
    //    buffer offset actually addresses what we think it does. We
    //    write a sentinel through gdn_ckpt_slot(layer=2, n=3) and
    //    read it back through the parent's contents() pointer at
    //    the same byte offset.
    {
        let layer = 2u32;
        let nn = 3u32;
        let slot = scratch.gdn_ckpt_slot(layer, nn);
        // Write 'sentinel' as the FIRST element of the slot.
        unsafe {
            let p =
                (slot.buffer.contents().as_ptr() as *mut u8).add(slot.offset as usize) as *mut f32;
            *p = 1234.5;
        }
        // Read through the PARENT buffer at the computed byte offset.
        let parent_byte_off = ((layer as u64 * n as u64 + nn as u64) * expected_ssm) * f32_size;
        unsafe {
            let p = (scratch.gdn_ckpt.buffer.contents().as_ptr() as *const u8)
                .add(parent_byte_off as usize) as *const f32;
            assert!(
                (*p - 1234.5).abs() < 1e-9,
                "round-trip via slot got {} expected 1234.5",
                *p
            );
        }
    }
}

/// H5.3a gate G1 (lite) + G5: packed_verify produces the SAME
/// argmax tokens as N successive `single_token` calls from a fresh
/// session. Headline correctness signal for the H5.3a scaffold —
/// proves:
///   * packed semantics (residual stream evolution, KV append,
///     GDN+conv state evolution) match N single-token decode
///   * GPU argmax (lowest-index tie policy) matches CPU argmax
///   * packed_ids_buf reads the right slot per block (the codex Q7
///     mitigation; if this were broken, every get_rows would read
///     the same stale token id and all argmaxes would equal each
///     other or be silently wrong)
///   * codex Q2 design Y (batched-end-of-token blits) doesn't
///     break correctness — if the blit pass were perturbing later
///     tokens, the second/third token argmaxes would diverge
///
/// Also pulls in gate G2 lite: post-packed `gdn_state[k]`,
/// `gdn_conv[k]`, and `kv_n_pos` must match post-N-single-token
/// session state (proves the per-token blits captured the same
/// bytes the in-place updates produced).
///
/// Does NOT yet verify (later H5.3a gates):
///   * checkpoint slot CONTENTS at intermediate n (G3 — needs
///     restore primitive to validate)
///   * hidden capture layout (G4 — separate test)
///   * cosine ≥ 0.9999 on raw logits (G1 full — needs _with_logits)
///
/// 0.8B-F32, N=4. Loads in ~500 ms; total runtime ≤ 2 s on M4 Max.
#[test]
fn dflash_packed_verify_argmax_matches_n_single_tokens() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[dflash-packed-verify] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    // Baseline: N successive single_token calls on a fresh session.
    let n: u32 = 4;
    let start_position: u32 = 0;
    let tokens: Vec<i32> = vec![9419, 1, 5, 1234];
    assert_eq!(tokens.len() as u32, n);

    let mut single_session = MetalSession::fresh(&ctx, &mm, 64).expect("session 1");
    let mut single_argmaxes = Vec::with_capacity(n as usize);
    for (i, &tok) in tokens.iter().enumerate() {
        let logits = mf
            .single_token(tok, start_position + i as u32, &mut single_session)
            .expect("single token");
        // CPU argmax with lowest-index tie (matches kernel_argmax_f32).
        let mut best = f32::NEG_INFINITY;
        let mut idx: i32 = 0;
        for (j, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                idx = j as i32;
            }
        }
        single_argmaxes.push(idx);
    }
    eprintln!("[dflash-packed-verify] single argmaxes: {single_argmaxes:?}");

    // Packed: one packed_verify call from a FRESH session.
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target_layers = target_layer_ids.len() as u32;
    let mut packed_session = MetalSession::fresh(&ctx, &mm, 64).expect("session 2");
    let mut scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, n, k_target_layers).expect("scratch");

    let packed_argmaxes = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &tokens,
        start_position,
        &mut scratch,
        &mut packed_session,
    )
    .expect("packed verify");
    eprintln!("[dflash-packed-verify] packed argmaxes: {packed_argmaxes:?}");

    assert_eq!(
        packed_argmaxes, single_argmaxes,
        "G1/G5: packed_verify argmaxes must match N successive single_token argmaxes"
    );

    // Bonus: gate G2 lite — post-packed session state matches
    // post-N-single-token session state.
    //
    // **Bitwise equality, NOT a slack tolerance.** Per codex
    // open-ended review: this path is literally identical
    // dispatch order and identical state evolution (packed_verify
    // is N sequential single_token encodes inside one cmd
    // buffer; same kernels, same args, same bind order). If
    // bitwise eq fails, that is a real signal — not noise to
    // be papered over with `< 1e-5`. Keep the bar.
    for (i, (s_state, p_state)) in single_session
        .gdn_state
        .iter()
        .zip(packed_session.gdn_state.iter())
        .enumerate()
    {
        unsafe {
            let s = s_state.buffer.contents().as_ptr() as *const u32;
            let p = p_state.buffer.contents().as_ptr() as *const u32;
            let n_elems = s_state.n_elements() as usize;
            for j in 0..n_elems {
                let sv = *s.add(j);
                let pv = *p.add(j);
                if sv != pv {
                    let sf = f32::from_bits(sv);
                    let pf = f32::from_bits(pv);
                    panic!(
                        "G2: gdn_state[{i}][{j}] bitwise mismatch: \
                         single={sf} (0x{sv:08x}) packed={pf} (0x{pv:08x}) \
                         Δ={}",
                        sf - pf
                    );
                }
            }
        }
    }
    for (i, (s_conv, p_conv)) in single_session
        .gdn_conv
        .iter()
        .zip(packed_session.gdn_conv.iter())
        .enumerate()
    {
        unsafe {
            let s = s_conv.buffer.contents().as_ptr() as *const u32;
            let p = p_conv.buffer.contents().as_ptr() as *const u32;
            let n_elems = s_conv.n_elements() as usize;
            for j in 0..n_elems {
                let sv = *s.add(j);
                let pv = *p.add(j);
                if sv != pv {
                    let sf = f32::from_bits(sv);
                    let pf = f32::from_bits(pv);
                    panic!(
                        "G2: gdn_conv[{i}][{j}] bitwise mismatch: \
                         single={sf} (0x{sv:08x}) packed={pf} (0x{pv:08x}) \
                         Δ={}",
                        sf - pf
                    );
                }
            }
        }
    }
    assert_eq!(
        single_session.kv_n_pos, packed_session.kv_n_pos,
        "G2: kv_n_pos diverged"
    );
}

/// H5.3b.4-5 headline correctness gate: layer-major
/// `packed_verify` produces identical argmax tokens and
/// tight-tolerance-equal logits + session state vs the
/// token-major oracle on the same inputs.
///
/// HISTORY OF THE COMPARISON STRENGTH (v0.425 triage): this gate
/// was originally BIT-EXACT on everything, and legitimately so —
/// both paths ran the same per-token F32 mat-vec kernels in a
/// different dispatch order, and same-stream Metal dispatches are
/// deterministic. That premise died at v0.154 (`b053d18`), which
/// added `GgmlType::F32` to `prefill_mat_mat_dispatch_eligible`:
/// since then layer-major uses batched F32 simdgroup-matrix
/// mat-mat for GDN/attention projections, FFN, and the lm_head
/// tail, while token-major still runs per-token mat-vec. Both
/// stage in F32 (no half-precision casting — see
/// `kernel_mat_mat_f32_f32`); the divergence is purely FP32
/// reduction-order. Worst-case reorder envelope for K=5120 dots
/// is on the order of `gamma_K ~= K * 2^-24 ~= 3e-4` relative to
/// the absolute-value dot mass; measured deltas on this fixture
/// are `logits max|Δ|=7.7e-4 / min_cos=0.9999999978`,
/// `gdn_state max|Δ|<=1.6e-4`, `gdn_conv max|Δ|<=7.3e-4`.
/// Note the batched projections feed the GDN recurrence, so
/// session state diverges at the same reorder scale as logits —
/// state comparisons cannot be bitwise either.
///
/// Per codex Q4 + the codex layer-major partner-session failure-
/// mode prediction: "argmax + final-state gates can mask shape-
/// only bugs on lucky logits." Therefore this test ALSO compares
/// raw `[N, V]` logits row-by-row via the `_with_logits` debug
/// variants. A transposed/wrong-stride layout bug destroys row
/// cosine and max|Δ| by many orders of magnitude, so the
/// tolerance gate preserves the original shape-bug coverage;
/// tolerances are ~5x the measured reorder envelope so real bugs
/// (which produce deltas at the 1e-1..1e+1 scale) cannot hide.
/// Argmax tokens and `kv_n_pos` remain exact-equality gates.
///
/// 0.8B-F32, M=2 prime + N=4 verify. ≤ 2 s.
#[test]
fn dflash_packed_verify_layer_major_matches_token_major() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    const M: u32 = 2;
    const N: u32 = 4;
    let prime_tokens: [i32; M as usize] = [9419, 1];
    let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target = target_layer_ids.len() as u32;

    // Two identically-primed sessions.
    let mut sess_tok = MetalSession::fresh(&ctx, &mm, 64).expect("sess tok");
    let mut sess_lm = MetalSession::fresh(&ctx, &mm, 64).expect("sess lm");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_tok)
            .expect("prime tok");
        mf.single_token(tok, i as u32, &mut sess_lm)
            .expect("prime lm");
    }

    // Token-major path with logits.
    let mut dbg_tok =
        MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target).expect("dbg scratch tok");
    let argmax_tok = encode_packed_verify_with_logits_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        &mut dbg_tok,
        &mut sess_tok,
    )
    .expect("token-major");

    // Layer-major path with logits.
    let mut dbg_lm =
        MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target).expect("dbg scratch lm");
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N).expect("layer scratch");
    let MetalDFlashDebugScratch {
        verify: lm_verify,
        debug_logits: lm_debug,
    } = &mut dbg_lm;
    let argmax_lm = encode_packed_verify_layer_major_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        lm_verify,
        &mut layer_scratch,
        &mut sess_lm,
        Some(lm_debug),
        None, // n_eff_override (test always uses full N)
    )
    .expect("layer-major");

    eprintln!(
        "[layer-major-vs-token-major] argmax_tok={argmax_tok:?} \
         argmax_lm={argmax_lm:?}"
    );

    // Bit-exact argmax tokens.
    assert_eq!(
        argmax_tok, argmax_lm,
        "layer-major argmax tokens diverge from token-major"
    );

    // Test-premise guard: the tolerance gate below is only justified
    // while layer-major actually takes the batched F32 mat-mat tail.
    // If F32 ever leaves the eligibility set, this comparison should
    // be restored to bitwise (see doc comment).
    assert!(
        prefill_mat_mat_dispatch_eligible(mm.lm_head.dtype),
        "test premise changed: lm_head dtype {:?} no longer mat-mat \
         eligible; restore the bitwise logits/state gates",
        mm.lm_head.dtype
    );

    // Tight-tolerance raw logits (codex's intermediate-layer paranoia
    // gate; argmax alone could pass even if intermediate layouts
    // were silently transposed for some shapes). Bounds are ~5x the
    // measured v0.424 FP32 reduction-order envelope (max|Δ|=7.7e-4,
    // min_cos=0.9999999978); real layout bugs blow through both by
    // orders of magnitude.
    const LOGITS_MAX_ABS: f32 = 4e-3;
    const LOGITS_MIN_COS: f64 = 0.999_999_9;
    let v = m.arch.vocab_size as usize;
    unsafe {
        let p_tok = dbg_tok.debug_logits.buffer.contents().as_ptr() as *const f32;
        let p_lm = dbg_lm.debug_logits.buffer.contents().as_ptr() as *const f32;
        let mut max_abs = 0f32;
        let mut min_cos = f64::INFINITY;
        for n_idx in 0..(N as usize) {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for i in 0..v {
                let a = *p_tok.add(n_idx * v + i);
                let b = *p_lm.add(n_idx * v + i);
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
                dot += (a as f64) * (b as f64);
                na += (a as f64) * (a as f64);
                nb += (b as f64) * (b as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
            if cos < min_cos {
                min_cos = cos;
            }
        }
        eprintln!(
            "[layer-major-vs-token-major] logits max|Δ|={max_abs:.3e} \
             min_cos={min_cos:.10}"
        );
        assert!(
            max_abs <= LOGITS_MAX_ABS,
            "logits max|Δ|={max_abs:.3e} > {LOGITS_MAX_ABS:.0e}: beyond the \
             FP32 reduction-order envelope — likely a real layout/kernel bug"
        );
        assert!(
            min_cos >= LOGITS_MIN_COS,
            "logits min per-row cos={min_cos:.10} < {LOGITS_MIN_COS}"
        );
    }

    // Tight-tolerance GDN session state. NOT bitwise since v0.154:
    // batched F32 mat-mat front projections feed the GDN recurrence
    // in layer-major, so reorder noise propagates into state (see
    // doc comment; measured gdn_state max|Δ|<=1.6e-4, gdn_conv
    // max|Δ|<=7.3e-4). Divergence beyond ~5x that envelope is a
    // real correctness failure.
    const GDN_STATE_MAX_ABS: f32 = 1e-3;
    const GDN_CONV_MAX_ABS: f32 = 4e-3;
    for (i, (a, b)) in sess_tok
        .gdn_state
        .iter()
        .zip(sess_lm.gdn_state.iter())
        .enumerate()
    {
        unsafe {
            let pa = a.buffer.contents().as_ptr() as *const f32;
            let pb = b.buffer.contents().as_ptr() as *const f32;
            let n_elems = a.n_elements() as usize;
            let mut max_abs = 0f32;
            for j in 0..n_elems {
                let d = (*pa.add(j) - *pb.add(j)).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            assert!(
                max_abs <= GDN_STATE_MAX_ABS,
                "gdn_state[{i}] max|Δ|={max_abs:.3e} > {GDN_STATE_MAX_ABS:.0e} \
                 between token-major and layer-major after the same N-token batch"
            );
        }
    }
    for (i, (a, b)) in sess_tok
        .gdn_conv
        .iter()
        .zip(sess_lm.gdn_conv.iter())
        .enumerate()
    {
        unsafe {
            let pa = a.buffer.contents().as_ptr() as *const f32;
            let pb = b.buffer.contents().as_ptr() as *const f32;
            let n_elems = a.n_elements() as usize;
            let mut max_abs = 0f32;
            for j in 0..n_elems {
                let d = (*pa.add(j) - *pb.add(j)).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            assert!(
                max_abs <= GDN_CONV_MAX_ABS,
                "gdn_conv[{i}] max|Δ|={max_abs:.3e} > {GDN_CONV_MAX_ABS:.0e} \
                 between token-major and layer-major"
            );
        }
    }
    assert_eq!(
        sess_tok.kv_n_pos, sess_lm.kv_n_pos,
        "kv_n_pos diverges between token-major and layer-major"
    );
}

/// **v0.76 adaptive-N back-off correctness gate**: confirm that
/// `encode_packed_verify_layer_major_inner` with `n_eff_override
/// = Some(n_eff)` produces argmaxes EQUAL to a fresh full-N=block
/// run on the first `n_eff` tokens, AND advances session state
/// (gdn_state, gdn_conv, kv_n_pos for attn layers) consistently
/// with running a single_token loop on those `n_eff` tokens.
///
/// The greedy-equivalence story for adaptive N hinges on this: the
/// verify math doesn't change, we just process fewer tokens. The
/// argmax tokens accepted should be identical OVER THE FIRST n_eff
/// SLOTS regardless of whether N=16 or N=8 was used.
///
/// Uses 0.8B-F32 (lib loop, fast). 27B integration test follows in
/// dflash_correctness.rs.
#[test]
fn dflash_packed_verify_n_eff_override_equiv() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[n_eff-equiv] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    // Two identically-primed sessions: M=2 priming tokens.
    const M: u32 = 2;
    const N_BLOCK: u32 = 16;
    let prime_tokens: [i32; M as usize] = [9419, 1];
    // 8 verify tokens (we'll run them via n_eff_override=Some(8)).
    let verify_tokens_8: [i32; 8] = [1234, 7, 999, 42, 11, 22, 33, 44];

    let mut sess_full = MetalSession::fresh(&ctx, &mm, 64).expect("sess full");
    let mut sess_eff = MetalSession::fresh(&ctx, &mm, 64).expect("sess eff");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_full)
            .expect("prime full");
        mf.single_token(tok, i as u32, &mut sess_eff)
            .expect("prime eff");
    }

    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target = target_layer_ids.len() as u32;

    // Path A: full N=8 scratch, no override (baseline behavior).
    let mut verify_8 = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 8, k_target).expect("verify_8");
    let mut layer_8 = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, 8).expect("layer_8");
    let argmax_full_8 = encode_packed_verify_layer_major_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens_8,
        M,
        &mut verify_8,
        &mut layer_8,
        &mut sess_full,
        None,
        None, // no override; verify N=8 from scratch shape
    )
    .expect("packed_verify N=8 full");

    // Path B: N=16 scratch, n_eff_override=Some(8), only 8 verify tokens.
    let mut verify_16 =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, N_BLOCK, k_target).expect("verify_16");
    let mut layer_16 = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N_BLOCK).expect("layer_16");
    let argmax_eff_8 = encode_packed_verify_layer_major_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens_8,
        M,
        &mut verify_16,
        &mut layer_16,
        &mut sess_eff,
        None,
        Some(8), // override truncates effective chain to 8
    )
    .expect("packed_verify N=16 scratch with n_eff=8 override");

    eprintln!("[n_eff-equiv] argmax_full_8={argmax_full_8:?} argmax_eff_8={argmax_eff_8:?}");

    // Bit-exact argmax — same math, just allocated differently.
    assert_eq!(
        argmax_full_8, argmax_eff_8,
        "n_eff=8 override produces different argmaxes than fresh N=8 scratch"
    );
    // Returned vec length matches n_eff, NOT n_block.
    assert_eq!(argmax_eff_8.len(), 8);

    // Bit-exact session state (GDN state, conv, kv_n_pos).
    for (i, (a, b)) in sess_full
        .gdn_state
        .iter()
        .zip(sess_eff.gdn_state.iter())
        .enumerate()
    {
        unsafe {
            let pa = a.buffer.contents().as_ptr() as *const u32;
            let pb = b.buffer.contents().as_ptr() as *const u32;
            let n_elems = a.n_elements() as usize;
            for j in 0..n_elems {
                if *pa.add(j) != *pb.add(j) {
                    panic!("gdn_state[{i}][{j}] diverges between full-N=8 and N=16+override=8");
                }
            }
        }
    }
    for (i, (a, b)) in sess_full
        .gdn_conv
        .iter()
        .zip(sess_eff.gdn_conv.iter())
        .enumerate()
    {
        unsafe {
            let pa = a.buffer.contents().as_ptr() as *const u32;
            let pb = b.buffer.contents().as_ptr() as *const u32;
            let n_elems = a.n_elements() as usize;
            for j in 0..n_elems {
                if *pa.add(j) != *pb.add(j) {
                    panic!("gdn_conv[{i}][{j}] diverges between full-N=8 and N=16+override=8");
                }
            }
        }
    }
    assert_eq!(
        sess_full.kv_n_pos, sess_eff.kv_n_pos,
        "kv_n_pos diverges between full-N=8 and N=16+override=8"
    );

    // Edge cases on the override itself.
    // n_eff=0 → error.
    let err = encode_packed_verify_layer_major_inner(
        &mf,
        &target_layer_ids,
        &[],
        M + 8, // start_position past the prior call
        &mut verify_16,
        &mut layer_16,
        &mut sess_eff,
        None,
        Some(0),
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(
                detail.contains("n=0") || detail.contains("[1, n_block"),
                "expected n_eff=0 → BadShape with range error: {detail}"
            );
        }
        other => panic!("expected BadShape on n_eff=0, got {other:?}"),
    }

    // n_eff > n_block → error.
    let err = encode_packed_verify_layer_major_inner(
        &mf,
        &target_layer_ids,
        &[1i32; 17],
        M + 8,
        &mut verify_16,
        &mut layer_16,
        &mut sess_eff,
        None,
        Some(17), // > N_BLOCK=16
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(
                detail.contains("[1, n_block"),
                "expected n_eff=17 → BadShape with range error: {detail}"
            );
        }
        other => panic!("expected BadShape on n_eff>n_block, got {other:?}"),
    }
}

/// H5.3a guard-wall test (codex failure-mode mitigation): if the
/// scratch was allocated with a different `block_size` /
/// `target_layer_ids.len()` / model arch, packed_verify must
/// FAIL LOUDLY at entry, not silently corrupt downstream blits.
/// Catches the scratch/model/session dimensional drift class
/// codex flagged.
#[test]
fn dflash_packed_verify_dim_guard_wall() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let mut session = MetalSession::fresh(&ctx, &mm, 64).expect("session");
    let target_layer_ids: Vec<u32> = vec![5, 10, 15];
    let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 4, 3).expect("scratch");

    // (a) tokens.len() != scratch.n
    let bad_tokens = vec![1i32, 2, 3];
    let err = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &bad_tokens,
        0,
        &mut scratch,
        &mut session,
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(detail.contains("tokens.len()"), "wrong error: {detail}");
        }
        other => panic!("expected BadShape on tokens.len mismatch, got {other:?}"),
    }

    // (b) target_layer_ids.len() != scratch.k_target_layers
    let bad_layers: Vec<u32> = vec![5, 15];
    let err = encode_packed_verify_inner(
        &mf,
        &bad_layers,
        &[1i32, 2, 3, 4],
        0,
        &mut scratch,
        &mut session,
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(detail.contains("k_target_layers"), "wrong error: {detail}");
        }
        other => panic!("expected BadShape on k mismatch, got {other:?}"),
    }

    // (c) start_position + N > kv_capacity
    let err = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &[1i32, 2, 3, 4],
        61, // 61 + 4 = 65 > 64
        &mut scratch,
        &mut session,
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(detail.contains("kv_capacity"), "wrong error: {detail}");
        }
        other => panic!("expected BadShape on kv overflow, got {other:?}"),
    }

    // (d) bad token id
    let bad_token: i32 = m.arch.vocab_size as i32 + 100;
    let err = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &[1i32, 2, bad_token, 4],
        0,
        &mut scratch,
        &mut session,
    );
    match err {
        Err(DFlashError::BadToken(t, _)) => assert_eq!(t, bad_token),
        other => panic!("expected BadToken, got {other:?}"),
    }

    // (e) target_layer_id out of range
    let bad_layers: Vec<u32> = vec![5, 999, 15]; // 999 > 0.8B's 24 layers
    let mut scratch2 = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 4, 3).expect("scratch2");
    let err = encode_packed_verify_inner(
        &mf,
        &bad_layers,
        &[1i32, 2, 3, 4],
        0,
        &mut scratch2,
        &mut session,
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(detail.contains("layer id"), "wrong error: {detail}");
        }
        other => panic!("expected BadShape on layer id, got {other:?}"),
    }
}

/// H5.3a guard test (codex biggest miss): the session must
/// represent the prefix ending at `start_position`. A misaligned
/// session (kv_n_pos != start_position) must be rejected loudly,
/// not silently produce wrong results.
///
/// Two scenarios:
///   (a) Fresh session (kv_n_pos=0) called with start_position>0
///       — should fail.
///   (b) Stale session (kv_n_pos=K from prior decode) called with
///       start_position != K — should fail.
#[test]
fn dflash_packed_verify_kv_n_pos_guard() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let target_layer_ids: Vec<u32> = vec![5, 15];
    let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, 4, 2).expect("scratch");

    // Scenario (a): fresh session (kv_n_pos all 0), start_position=5.
    let mut fresh_session = MetalSession::fresh(&ctx, &mm, 64).expect("session");
    let err = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &[1i32, 2, 3, 4],
        5,
        &mut scratch,
        &mut fresh_session,
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(detail.contains("kv_n_pos"), "wrong error: {detail}");
            assert!(detail.contains("start_position"), "wrong error: {detail}");
        }
        other => panic!("expected BadShape on fresh-session/start>0, got {other:?}"),
    }

    // 0.8B has all GDN layers — no attn — so `kv_n_pos` is empty.
    // Skip the stale-session check; it's better exercised at 27B.
    // But we DO want to confirm the guard exits cleanly when the
    // vec is empty (passes through trivially: no entries to
    // disagree). I.e. with no attn layers, fresh session at
    // start_position=0 passes the guard.
    eprintln!(
        "[dflash-kv-n-pos-guard] 0.8B fresh kv_n_pos.len={} (all GDN layers)",
        fresh_session.kv_n_pos.len()
    );
}

/// H5.3a G2++ via PRIMED session: prove packed_verify works
/// correctly when the session is partway through a generation,
/// i.e. the kv_n_pos==start_position guard isn't masking a bug
/// where we silently DROP previously-encoded state.
///
/// Setup:
///   1. Run M=2 single_token calls on session_A starting from
///      tokens[0..M]. Session_A.kv_n_pos == M after.
///   2. Run packed_verify(tokens[M..M+N], start_position=M)
///      against session_A. Expected: argmaxes match
///      tokens[M+1..M+N+1]'s argmax under continued single_token
///      decode.
///   3. Compare against single_token continued for N more steps
///      on session_B (also primed identically through M).
///
/// This is the test codex specifically called out as more
/// important than the cosine gate before writing restore.
#[test]
fn dflash_packed_verify_with_primed_session() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    // Prime BOTH sessions identically through M tokens.
    const M: u32 = 3; // priming length
    const N: u32 = 4; // packed verify length
    let prime_tokens: [i32; M as usize] = [9419, 1, 5];
    let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];

    let mut session_a = MetalSession::fresh(&ctx, &mm, 64).expect("session A");
    let mut session_b = MetalSession::fresh(&ctx, &mm, 64).expect("session B");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut session_a)
            .expect("prime A");
        mf.single_token(tok, i as u32, &mut session_b)
            .expect("prime B");
    }
    // Note: 0.8B has no attn layers, so kv_n_pos is empty — the
    // guard trivially passes regardless of M. The test still
    // proves the GDN+conv state evolution is correct under
    // start_position > 0 on packed_verify.

    // Continue B with N single_token calls; collect argmaxes.
    let mut single_argmaxes = Vec::with_capacity(N as usize);
    for (i, &tok) in verify_tokens.iter().enumerate() {
        let logits = mf
            .single_token(tok, M + i as u32, &mut session_b)
            .expect("continue B");
        let mut best = f32::NEG_INFINITY;
        let mut idx: i32 = 0;
        for (j, &v) in logits.iter().enumerate() {
            if v > best {
                best = v;
                idx = j as i32;
            }
        }
        single_argmaxes.push(idx);
    }

    // Run packed_verify on A starting at start_position=M.
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, 2).expect("scratch");
    let packed_argmaxes = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        &mut scratch,
        &mut session_a,
    )
    .expect("packed verify with primed session");

    eprintln!(
        "[dflash-primed] M={M} N={N} \
         single_argmaxes={single_argmaxes:?} \
         packed_argmaxes={packed_argmaxes:?}"
    );
    assert_eq!(
        packed_argmaxes, single_argmaxes,
        "packed_verify on primed session must match continued single_token"
    );

    // Bitwise GDN+conv state equivalence post-packed vs post-single.
    for (i, (s_state, p_state)) in session_b
        .gdn_state
        .iter()
        .zip(session_a.gdn_state.iter())
        .enumerate()
    {
        unsafe {
            let s = s_state.buffer.contents().as_ptr() as *const u32;
            let p = p_state.buffer.contents().as_ptr() as *const u32;
            let n_elems = s_state.n_elements() as usize;
            for j in 0..n_elems {
                if *s.add(j) != *p.add(j) {
                    panic!(
                        "primed-G2: gdn_state[{i}][{j}] bitwise mismatch \
                         after primed packed_verify"
                    );
                }
            }
        }
    }
}

/// H5.3a gate G3 (checkpoint replay equivalence) + G6 (restore
/// boundary cases): the headline correctness gate for the
/// rollback primitive. Per codex H5.3a review: cosine is independent
/// of restore; restore unblocks G3+G6, which prove the checkpoint
/// CONTENTS at intermediate n are correct (not just final state).
///
/// Setup: prime two fresh sessions identically through M tokens.
/// Run packed_verify(verify_tokens, start_position=M) on session_A.
/// For each n_keep ∈ {1, N/2, N}:
///   * Restore session_A to n_keep.
///   * Run one single_token at position M + n_keep on session_A
///     with a marker token.
///   * On session_B (separately primed), run n_keep single_tokens
///     of verify_tokens[0..n_keep], then one single_token of the
///     marker. session_A and session_B should now have BIT-EXACT
///     gdn_state, gdn_conv, kv_n_pos, AND argmax token.
///
/// This is the strongest possible test of the rollback semantics.
/// If checkpoint slot CONTENTS are wrong (e.g., off-by-one indexing),
/// session_A's post-restore state diverges from session_B's
/// "ground-truth" sequential state and we catch it.
///
/// 0.8B-F32, M=2 prime + N=4 verify. ≤ 5 s on M4 Max.
#[test]
fn dflash_restore_after_partial_accept_replay_equivalence() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    const M: u32 = 2;
    const N: u32 = 4;
    let prime_tokens: [i32; M as usize] = [9419, 1];
    let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
    let marker_token: i32 = 555; // post-restore single_token input

    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target_layers = target_layer_ids.len() as u32;

    // G6 boundary set: n_keep = 1 (full reject; carry only),
    // n_keep = N/2 (typical partial), n_keep = N (full accept).
    for &n_keep in &[1u32, N / 2, N] {
        // -- session_A: prime + packed_verify + restore + one
        //    single_token at M + n_keep.
        let mut sess_a = MetalSession::fresh(&ctx, &mm, 64).expect("sess A");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_a)
                .expect("prime A");
        }
        let mut scratch =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, k_target_layers).expect("scratch");
        let _packed = encode_packed_verify_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            &mut scratch,
            &mut sess_a,
        )
        .expect("packed verify");

        encode_restore_after_partial_accept_inner(&mf, &scratch, n_keep, M, &mut sess_a, None)
            .expect("restore");

        let logits_a = mf
            .single_token(marker_token, M + n_keep, &mut sess_a)
            .expect("marker on A");
        let mut argmax_a: i32 = 0;
        let mut best = f32::NEG_INFINITY;
        for (j, &v) in logits_a.iter().enumerate() {
            if v > best {
                best = v;
                argmax_a = j as i32;
            }
        }

        // -- session_B: prime + n_keep single_tokens through
        //    verify_tokens[0..n_keep] + one single_token of marker.
        //    This is the "ground truth" sequential trajectory.
        let mut sess_b = MetalSession::fresh(&ctx, &mm, 64).expect("sess B");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_b)
                .expect("prime B");
        }
        for i in 0..n_keep {
            mf.single_token(verify_tokens[i as usize], M + i, &mut sess_b)
                .expect("kept verify token on B");
        }
        let logits_b = mf
            .single_token(marker_token, M + n_keep, &mut sess_b)
            .expect("marker on B");
        let mut argmax_b: i32 = 0;
        let mut best = f32::NEG_INFINITY;
        for (j, &v) in logits_b.iter().enumerate() {
            if v > best {
                best = v;
                argmax_b = j as i32;
            }
        }

        eprintln!(
            "[restore-replay n_keep={n_keep}] argmax_A={argmax_a} \
             argmax_B={argmax_b}"
        );

        // G3: argmax tokens must match (covers logits-after-restore
        // equivalence at the argmax-coarsened level).
        assert_eq!(
            argmax_a, argmax_b,
            "G3: argmax post-restore differs at n_keep={n_keep}: \
             A={argmax_a} B={argmax_b}"
        );

        // G3 (stronger): bitwise equality on gdn_state / gdn_conv
        // after the marker single_token. The marker is processed
        // identically on both paths so divergence indicates a
        // restore bug, not a forward bug.
        for (i, (a_state, b_state)) in sess_a
            .gdn_state
            .iter()
            .zip(sess_b.gdn_state.iter())
            .enumerate()
        {
            unsafe {
                let a = a_state.buffer.contents().as_ptr() as *const u32;
                let b = b_state.buffer.contents().as_ptr() as *const u32;
                let n_elems = a_state.n_elements() as usize;
                for j in 0..n_elems {
                    if *a.add(j) != *b.add(j) {
                        let af = f32::from_bits(*a.add(j));
                        let bf = f32::from_bits(*b.add(j));
                        panic!(
                            "G3: gdn_state[{i}][{j}] post-restore-then-marker \
                             differs at n_keep={n_keep}: A={af} B={bf}"
                        );
                    }
                }
            }
        }
        for (i, (a_conv, b_conv)) in sess_a
            .gdn_conv
            .iter()
            .zip(sess_b.gdn_conv.iter())
            .enumerate()
        {
            unsafe {
                let a = a_conv.buffer.contents().as_ptr() as *const u32;
                let b = b_conv.buffer.contents().as_ptr() as *const u32;
                let n_elems = a_conv.n_elements() as usize;
                for j in 0..n_elems {
                    if *a.add(j) != *b.add(j) {
                        panic!(
                            "G3: gdn_conv[{i}][{j}] post-restore-then-marker \
                             differs at n_keep={n_keep}"
                        );
                    }
                }
            }
        }

        // kv_n_pos must equal M + n_keep + 1 on both (after marker
        // single_token).
        let expected_kv = (M as usize) + (n_keep as usize) + 1;
        for (i, &a_pos) in sess_a.kv_n_pos.iter().enumerate() {
            assert_eq!(
                a_pos, expected_kv,
                "G3: sess_A kv_n_pos[{i}]={a_pos} != expected {expected_kv}"
            );
            assert_eq!(
                sess_b.kv_n_pos[i], expected_kv,
                "G3: sess_B kv_n_pos[{i}] != expected {expected_kv}"
            );
        }
    }
}

#[test]
fn dflash_layer_major_packed_gdn_checkpoint_replay_equivalence() {
    if !dflash_verify_packed_gdn_enabled() {
        eprintln!("[packed-gdn-restore] skipped — packed GDN disabled");
        return;
    }
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    const M: u32 = 2;
    const N: u32 = 4;
    let prime_tokens: [i32; M as usize] = [9419, 1];
    let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target = target_layer_ids.len() as u32;

    for n_keep in 1..N {
        let mut sess_restored = MetalSession::fresh(&ctx, &mm, 64).expect("restored session");
        let mut sess_prefix = MetalSession::fresh(&ctx, &mm, 64).expect("prefix session");
        for (i, &tok) in prime_tokens.iter().enumerate() {
            mf.single_token(tok, i as u32, &mut sess_restored)
                .expect("prime restored");
            mf.single_token(tok, i as u32, &mut sess_prefix)
                .expect("prime prefix");
        }

        let mut verify_full =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, k_target).expect("verify full");
        let mut layer_full = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N).expect("layer full");
        encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens,
            M,
            &mut verify_full,
            &mut layer_full,
            &mut sess_restored,
            None,
            None,
        )
        .expect("full packed verify");
        encode_restore_after_partial_accept_inner(
            &mf,
            &verify_full,
            n_keep,
            M,
            &mut sess_restored,
            None,
        )
        .expect("restore packed checkpoint");

        let mut verify_prefix =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, k_target).expect("verify prefix");
        let mut layer_prefix =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, N).expect("layer prefix");
        encode_packed_verify_layer_major_inner(
            &mf,
            &target_layer_ids,
            &verify_tokens[..n_keep as usize],
            M,
            &mut verify_prefix,
            &mut layer_prefix,
            &mut sess_prefix,
            None,
            Some(n_keep),
        )
        .expect("prefix packed verify");

        assert_eq!(
            sess_restored.kv_n_pos, sess_prefix.kv_n_pos,
            "kv position mismatch after restoring {n_keep} rows"
        );
        let mut state_max_abs = 0.0f32;
        for (restored, prefix) in sess_restored
            .gdn_state
            .iter()
            .zip(sess_prefix.gdn_state.iter())
        {
            unsafe {
                let a = restored.buffer.contents().as_ptr() as *const f32;
                let b = prefix.buffer.contents().as_ptr() as *const f32;
                for i in 0..restored.n_elements() as usize {
                    state_max_abs = state_max_abs.max((*a.add(i) - *b.add(i)).abs());
                }
            }
        }
        let mut conv_max_abs = 0.0f32;
        for (restored, prefix) in sess_restored
            .gdn_conv
            .iter()
            .zip(sess_prefix.gdn_conv.iter())
        {
            unsafe {
                let a = restored.buffer.contents().as_ptr() as *const f32;
                let b = prefix.buffer.contents().as_ptr() as *const f32;
                for i in 0..restored.n_elements() as usize {
                    conv_max_abs = conv_max_abs.max((*a.add(i) - *b.add(i)).abs());
                }
            }
        }
        assert!(
            state_max_abs <= 1e-3,
            "GDN state drift {state_max_abs:.3e} after restoring {n_keep} rows"
        );
        assert!(
            conv_max_abs <= 4e-3,
            "GDN conv drift {conv_max_abs:.3e} after restoring {n_keep} rows"
        );

        let marker = 555;
        let logits_restored = mf
            .single_token(marker, M + n_keep, &mut sess_restored)
            .expect("marker after restore");
        let logits_prefix = mf
            .single_token(marker, M + n_keep, &mut sess_prefix)
            .expect("marker after prefix");
        let mut dot = 0.0f64;
        let mut norm_restored = 0.0f64;
        let mut norm_prefix = 0.0f64;
        for (&a, &b) in logits_restored.iter().zip(&logits_prefix) {
            dot += (a as f64) * (b as f64);
            norm_restored += (a as f64).powi(2);
            norm_prefix += (b as f64).powi(2);
        }
        let cosine = dot / (norm_restored.sqrt() * norm_prefix.sqrt() + 1e-30);
        let argmax = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        eprintln!(
            "[packed-gdn-restore] n_keep={n_keep} state_max={state_max_abs:.3e} \
             conv_max={conv_max_abs:.3e} continuation_cos={cosine:.10}"
        );
        assert_eq!(
            argmax(&logits_restored),
            argmax(&logits_prefix),
            "continuation argmax mismatch after restoring {n_keep} rows"
        );
        assert!(
            cosine >= 0.999_999_9,
            "continuation cosine {cosine:.10} after restoring {n_keep} rows"
        );
    }
}

/// H5.3a gate G1 (FULL): cosine ≥ 0.9999 between
/// `packed_verify_with_logits` row-n logits and N successive
/// `single_token` logits. The strongest correctness signal at
/// the LOGITS layer (not just argmax-coarsened).
///
/// G1 lite (in dflash_packed_verify_argmax_matches_n_single_tokens)
/// only checks argmax tokens — it would pass if all rows shifted
/// by a constant. G1 full catches:
///   * subtle accumulation differences in the lm_head mat-vec
///   * any per-vocab-row bias from a wrong scatter offset
///   * cosine that's strong but not perfect (e.g. F16 KV
///     accumulation paths in attn-v4) — G1 full's threshold of
///     0.9999 is the H5.3 plan gate per docs/H5-DFLASH.md §3 H5.3.
///
/// 0.8B-F32, M=2 prime + N=4 verify. Uses
/// MetalDFlashDebugScratch (allocates the [N, V] buffer; debug-
/// only path).
#[test]
fn dflash_packed_verify_with_logits_cosine_match() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    const M: u32 = 2;
    const N: u32 = 4;
    let prime_tokens: [i32; M as usize] = [9419, 1];
    let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target_layers = target_layer_ids.len() as u32;
    let v = m.arch.vocab_size as usize;

    // -- Reference: N successive single_token on a primed session.
    let mut sess_b = MetalSession::fresh(&ctx, &mm, 64).expect("sess B");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_b)
            .expect("prime B");
    }
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(N as usize);
    for (i, &tok) in verify_tokens.iter().enumerate() {
        let logits = mf
            .single_token(tok, M + i as u32, &mut sess_b)
            .expect("single token");
        reference.push(logits);
    }

    // -- Packed with logits dump.
    let mut sess_a = MetalSession::fresh(&ctx, &mm, 64).expect("sess A");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_a)
            .expect("prime A");
    }
    let mut dbg_scratch =
        MetalDFlashDebugScratch::fresh(&ctx, &mm, N, k_target_layers).expect("dbg scratch");
    let _ = encode_packed_verify_with_logits_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        &mut dbg_scratch,
        &mut sess_a,
    )
    .expect("packed verify w logits");

    // -- Per-row cosine vs reference. F32 path; we expect bit-exact
    //    actually, but the H5.3 plan threshold is 0.9999 because
    //    quantized paths will round-trip differently. Test both
    //    bounds.
    let dump_n_elems = dbg_scratch.debug_logits.n_elements() as usize;
    let mut packed_dump = vec![0.0f32; dump_n_elems];
    unsafe {
        let src = dbg_scratch.debug_logits.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, packed_dump.as_mut_ptr(), dump_n_elems);
    }

    for n in 0..N as usize {
        let packed_row = &packed_dump[n * v..(n + 1) * v];
        let ref_row = &reference[n];
        // Cosine.
        let mut dot = 0.0f64;
        let mut np = 0.0f64;
        let mut nr = 0.0f64;
        for i in 0..v {
            let p = packed_row[i] as f64;
            let r = ref_row[i] as f64;
            dot += p * r;
            np += p * p;
            nr += r * r;
        }
        let cos = dot / (np.sqrt() * nr.sqrt() + 1e-30);
        // Max abs diff.
        let mut max_abs = 0.0f32;
        for i in 0..v {
            let d = (packed_row[i] - ref_row[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        // Bitwise equality count (sanity — F32 path should be
        // mostly bit-exact but some atomic ordering can diverge).
        let mut bit_eq = 0usize;
        for i in 0..v {
            if packed_row[i].to_bits() == ref_row[i].to_bits() {
                bit_eq += 1;
            }
        }
        eprintln!(
            "[g1-full n={n}] cos={cos:.10} max|Δ|={max_abs:.3e} \
             bit_eq={}/{} ({:.2}%)",
            bit_eq,
            v,
            100.0 * (bit_eq as f64) / (v as f64)
        );
        assert!(cos >= 0.9999, "G1 full: row {n} cosine {cos} < 0.9999");
    }
}

/// H5.3a gate G4: hidden capture LAYOUT.
///
/// Codex flagged this gap: G1 / G2 / G3 all check argmax tokens
/// or final session state, but `hidden_capture[k, n, :]` could
/// have wrong dim-order (e.g., stored as [N, K, H] instead of
/// [K, N, H]) and the rest of the test suite would still pass.
/// The dim-order bug only surfaces downstream when the drafter
/// reads target_ctx and produces garbage logits.
///
/// Setup: prime fresh session through M tokens. Run packed_verify
/// on session_A through N tokens; collect scratch.hidden_capture.
/// Separately, run `single_token_with_multi_hidden` N times on
/// session_B (primed identically), capturing per-token hiddens
/// into a `[K, H]` buffer per call. Stack into a `[N, K, H]`
/// reference. Compare against scratch.hidden_capture (which is
/// layout `[K, N, H]`) under the documented permutation.
///
/// Bitwise F32 match required. Catches:
///   * (k, n) → linear-index transpose bugs in slot_view
///   * scatter dst offset miscomputation in packed_verify
///   * the wrong target_layer being captured at index k
///
/// 0.8B-F32, M=2 prime + N=4 verify, K=2 layers. ≤ 4 s.
#[test]
// Hidden-capture layout test: `n` is a multi-purpose row index used
// for `reference[n][...]`, `(n * K + k) * H + i` stride math, and
// failure-message position. Iterator rewrite would lose all three.
#[allow(clippy::needless_range_loop)]
fn dflash_packed_verify_hidden_capture_layout() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    const M: u32 = 2;
    const N: u32 = 4;
    let prime_tokens: [i32; M as usize] = [9419, 1];
    let verify_tokens: [i32; N as usize] = [1234, 7, 999, 42];

    // Pick target_layer_ids such that they MUST be captured
    // distinctly — different blocks (5, 15) on 0.8B's 24-layer
    // schedule. If layout is K↔N transposed, the two layers'
    // hiddens get confused at different (n, k) pairs.
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let k_target_layers = target_layer_ids.len() as u32;
    let h = m.arch.hidden_size as usize;

    // -- session_A: packed_verify, capture into scratch.hidden_capture.
    let mut sess_a = MetalSession::fresh(&ctx, &mm, 64).expect("sess A");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_a)
            .expect("prime A");
    }
    let mut scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, k_target_layers).expect("scratch");
    let _ = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &verify_tokens,
        M,
        &mut scratch,
        &mut sess_a,
    )
    .expect("packed verify");

    // -- session_B: prime identically, then for each n in 0..N call
    //    single_token_with_multi_hidden. The hidden_dst is shape
    //    [K, H], laid out as `[k * h .. (k+1) * h]` per layer
    //    (matches MetalForward::single_token_with_multi_hidden).
    let mut sess_b = MetalSession::fresh(&ctx, &mm, 64).expect("sess B");
    for (i, &tok) in prime_tokens.iter().enumerate() {
        mf.single_token(tok, i as u32, &mut sess_b)
            .expect("prime B");
    }
    let single_hidden_buf =
        MetalTensor::zeros_f32(&ctx, vec![k_target_layers as u64 * h as u64]).expect("hidden dst");
    // [N][K * H] — flat reference dump per token.
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(N as usize);
    for (i, &tok) in verify_tokens.iter().enumerate() {
        let _ = mf
            .single_token_with_multi_hidden(
                tok,
                M + i as u32,
                &mut sess_b,
                &target_layer_ids,
                &single_hidden_buf,
            )
            .expect("single token w multi hidden");
        let n_elems = k_target_layers as usize * h;
        let mut row = vec![0.0f32; n_elems];
        unsafe {
            let src = single_hidden_buf.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, row.as_mut_ptr(), n_elems);
        }
        reference.push(row);
    }

    // -- Compare scratch.hidden_capture (layout [K, N, H]) against
    //    reference (layout [N, K * H]) under the documented
    //    permutation. For each (k, n): scratch[k*N*H + n*H + i]
    //    == reference[n][k*H + i].
    let scratch_buf_n_elems = scratch.hidden_capture.n_elements() as usize;
    let mut scratch_dump = vec![0.0f32; scratch_buf_n_elems];
    unsafe {
        let src = scratch.hidden_capture.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, scratch_dump.as_mut_ptr(), scratch_buf_n_elems);
    }

    // v0.71: layout is [N, K, H], so scratch index = (n * K + k) * H + i.
    for k in 0..k_target_layers as usize {
        for n in 0..N as usize {
            for i in 0..h {
                let scratch_idx = (n * (k_target_layers as usize) + k) * h + i;
                let ref_idx_in_row = k * h + i;
                let s = scratch_dump[scratch_idx];
                let r = reference[n][ref_idx_in_row];
                if s.to_bits() != r.to_bits() {
                    panic!(
                        "G4: hidden_capture[n={n}, k={k}, i={i}] differs: \
                         scratch={s} (0x{:08x}) reference={r} (0x{:08x})",
                        s.to_bits(),
                        r.to_bits()
                    );
                }
            }
        }
    }
    eprintln!(
        "[hidden-capture-layout] M={M} N={N} K={k_target_layers} \
         H={h}: bitwise match across all (k, n, i)"
    );

    // ALSO: confirm the two captured layers are NOT trivially
    // identical. If they were, a K↔N layout bug would silently
    // pass. We require the L2 distance between layer 5 and layer
    // 15 captures at n=0 to be substantial.
    let mut l2 = 0.0f64;
    for i in 0..h {
        let a = reference[0][i] as f64; // n=0, k=0 (layer 5)
        let b = reference[0][h + i] as f64; // n=0, k=1 (layer 15)
        l2 += (a - b).powi(2);
    }
    l2 = l2.sqrt();
    eprintln!(
        "[hidden-capture-layout] ||layer5_at_n0 - layer15_at_n0||_2 = {l2:.4} \
         (must be substantially nonzero or the test is degenerate)"
    );
    assert!(
        l2 > 0.1,
        "test is degenerate: the two captured layers are nearly identical, \
         a K↔N layout bug would pass silently. Pick more-different layers."
    );
}

/// H5.3a guard tests for restore primitive (codex failure-mode
/// mitigation): n_keep=0 must fail loudly; n_keep > N must fail;
/// stale kv_n_pos must fail.
#[test]
fn dflash_restore_after_partial_accept_guard_wall() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    const N: u32 = 4;
    let mut sess = MetalSession::fresh(&ctx, &mm, 64).expect("sess");
    let target_layer_ids: Vec<u32> = vec![5, 15];
    let mut scratch = MetalDFlashVerifyScratch::fresh(&ctx, &mm, N, 2).expect("scratch");

    // Run packed_verify so session is in the post-packed-verify state.
    let _ = encode_packed_verify_inner(
        &mf,
        &target_layer_ids,
        &[1i32, 2, 3, 4],
        0,
        &mut scratch,
        &mut sess,
    )
    .expect("packed verify");

    // (a) n_keep = 0
    let err = encode_restore_after_partial_accept_inner(&mf, &scratch, 0, 0, &mut sess, None);
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(detail.contains("n_keep=0"), "wrong error: {detail}");
        }
        other => panic!("expected BadShape on n_keep=0, got {other:?}"),
    }

    // (b) n_keep > N
    let err = encode_restore_after_partial_accept_inner(&mf, &scratch, N + 1, 0, &mut sess, None);
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(
                detail.contains(&format!("n_keep={}", N + 1)),
                "wrong error: {detail}"
            );
        }
        other => panic!("expected BadShape on n_keep>N, got {other:?}"),
    }

    // (c) wrong start_position (kv_n_pos contract violation).
    // 0.8B has no attn layers so kv_n_pos.len() == 0 — the loop
    // is trivially satisfied. Note in stderr; the contract is
    // exercised on 27B (different test).
    if !sess.kv_n_pos.is_empty() {
        let err = encode_restore_after_partial_accept_inner(&mf, &scratch, 2, 99, &mut sess, None);
        match err {
            Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
                assert!(detail.contains("kv_n_pos"), "wrong error: {detail}");
            }
            other => {
                panic!("expected BadShape on kv_n_pos mismatch, got {other:?}")
            }
        }
    } else {
        eprintln!(
            "[restore-guard] 0.8B has no attn layers; \
             kv_n_pos contract is exercised at 27B (separate test)"
        );
    }
}

/// Verify `MetalDFlashDebugScratch` builds correctly and `logits_slot`
/// returns properly-aligned views into the [N, V] buffer.
#[test]
fn dflash_debug_scratch_logits_slots() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[dflash-debug-scratch] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");

    let n: u32 = 4;
    let k: u32 = 2;
    let dbg = MetalDFlashDebugScratch::fresh(&ctx, &mm, n, k).expect("debug scratch alloc");

    let v = m.arch.vocab_size as u64;
    let f32_size = std::mem::size_of::<f32>() as u64;
    assert_eq!(dbg.debug_logits.shape, vec![n as u64, v]);
    for nn in 0..n {
        let slot = dbg.logits_slot(nn);
        assert_eq!(slot.shape, vec![v]);
        assert_eq!(slot.offset, (nn as u64) * v * f32_size);
    }
    // Verify the wrapped verify scratch is independently usable.
    assert_eq!(dbg.verify.n, n);
    assert_eq!(dbg.verify.k_target_layers, k);
}

/// **v0.75.1 correctness gate** (fast, lib-loop variant on
/// Qwen3.5-0.8B-F32). Oracle: a sequential `single_token_with_
/// multi_hidden` loop. Experimental: one `prefill_tokens_with_
/// multi_hidden` call.
///
/// Cosine equivalence ≥ 0.999 required on:
///   * final logits (last prompt token's vocab vector)
///   * accumulated per-token multi-hidden capture
///   * GDN state + conv tensors per layer
///
/// 0.8B-F32 has both GDN and attention blocks, so this gates the GDN
/// recurrence, dense FFN mat-mat, attention projection mat-mat, and tail.
/// The larger 27B integration test in `tests/dflash_correctness.rs` still
/// gates production-shape attention profiling.
///
/// Edge cases tested via subroutine: T<P, T==P, T==P+r, T==2P.
#[test]
fn prefill_tokens_matches_single_token_loop_0_8b() {
    let path = crate::test_fixtures::QWEN35_0_8B_F32.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[prefill-tokens-vs-single-token] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let arch = &mm.arch;
    let h = arch.hidden_size as usize;

    // K=4 capture layers spanning the network. 0.8B-F32 has 24
    // layers so we hit GDN at varied depths (all-GDN model).
    let capture_layers: Vec<u32> = vec![0, 7, 14, 21];
    let k = capture_layers.len();

    // Run each scenario as a closure so the same comparator covers
    // T<P, T==P, T==P+r, T==2P, T=P+1 (chunk_p==1), and the
    // start_position > 0 case (extending an already-advanced session).
    //
    // `prefix_len` primes both sessions with `prefix_len` tokens via
    // sequential `single_token` first; prefill then runs from
    // `start_position = prefix_len` over `total_n` more tokens. The
    // total number of tokens forwarded by both paths is
    // `prefix_len + total_n`. The final logits / hidden capture
    // comparison is over the LAST `total_n` tokens.
    let run_scenario = |label: &str, total_n: usize, p: usize, prefix_len: usize| {
        assert!(total_n >= 1 && p >= 1, "{label}: need n,p ≥ 1");

        // Synthesize a deterministic prefix + suffix sequence.
        let n_total_with_prefix = prefix_len + total_n;
        let all_tokens: Vec<i32> = (0..n_total_with_prefix)
            .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
            .collect();
        let prefix_tokens = &all_tokens[..prefix_len];
        let token_ids = &all_tokens[prefix_len..];

        // ---- Oracle path: sequential single_token_with_multi_hidden ----
        // (prefix uses single_token to advance state, suffix uses
        // single_token_with_multi_hidden so we capture the same
        // hidden states the prefill captures.)
        let cap = n_total_with_prefix + 4;
        let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
        for (i, &tid) in prefix_tokens.iter().enumerate() {
            mf.single_token(tid, i as u32, &mut sess_a)
                .expect("oracle prefix advance");
        }
        let h_dst_a = MetalTensor::zeros_f32(&ctx, vec![(k * h) as u64]).expect("h_dst_a");
        let mut accum_a = vec![0.0f32; total_n * k * h];
        let mut last_a = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last_a = mf
                .single_token_with_multi_hidden(
                    tid,
                    (prefix_len + i) as u32,
                    &mut sess_a,
                    &capture_layers,
                    &h_dst_a,
                )
                .expect("oracle forward");
            unsafe {
                let src = h_dst_a.buffer.contents().as_ptr() as *const f32;
                std::ptr::copy_nonoverlapping(
                    src,
                    accum_a[i * k * h..(i + 1) * k * h].as_mut_ptr(),
                    k * h,
                );
            }
        }

        // ---- Experimental path: prefill_tokens_with_multi_hidden ----
        // Same prefix advance via single_token, then one prefill call
        // starting at start_position = prefix_len.
        let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
        for (i, &tid) in prefix_tokens.iter().enumerate() {
            mf.single_token(tid, i as u32, &mut sess_b)
                .expect("experimental prefix advance");
        }
        let scratch_plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            &mm,
            p as u32,
            n_total_with_prefix,
            PrefillScratchConfig::default(),
        )
        .expect("scratch plan");
        if label.starts_with("T<P") {
            let mut invalid_plan = scratch_plan.clone();
            invalid_plan.block_size += 1;
            assert!(
                MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &mm, invalid_plan,)
                    .is_err()
            );
        }
        let expected_plan = scratch_plan.clone();
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &mm, scratch_plan)
                .expect("layer scratch from plan");
        assert_eq!(layer_scratch.prefill_scratch_plan(), &expected_plan);
        let h_dst_b =
            MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst_b");
        let last_b = prefill_tokens_with_multi_hidden(
            &mf,
            token_ids,
            prefix_len as u32,
            &mut sess_b,
            &mut layer_scratch,
            &capture_layers,
            Some(&h_dst_b),
        )
        .expect("prefill");
        // ---- Compare final logits (cos ≥ 0.999). ----
        assert_eq!(last_a.len(), last_b.len(), "{label}: logits len mismatch");
        let cos_logits = cosine_f32(&last_a, &last_b);
        eprintln!(
            "[prefill-vs-single] {label}: T={total_n} P={p} prefix={prefix_len} chunks={} cos(logits)={cos_logits:.6}",
            total_n.div_ceil(p)
        );
        assert!(
            cos_logits >= 0.999,
            "{label}: logits cos={cos_logits} < 0.999"
        );

        // ---- Compare accumulated multi-hidden (cos per token slot). ----
        let mut accum_b = vec![0.0f32; total_n * k * h];
        unsafe {
            let src = h_dst_b.buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(src, accum_b.as_mut_ptr(), total_n * k * h);
        }
        // Compare per-token-per-capture-layer (most diagnostic) AND
        // overall (compact summary).
        let mut min_cos = f64::INFINITY;
        let mut worst_pos = (0usize, 0usize);
        for t in 0..total_n {
            for k_idx in 0..k {
                let off = (t * k + k_idx) * h;
                let a_slice = &accum_a[off..off + h];
                let b_slice = &accum_b[off..off + h];
                let c = cosine_f32(a_slice, b_slice);
                if c < min_cos {
                    min_cos = c;
                    worst_pos = (t, k_idx);
                }
            }
        }
        eprintln!(
            "[prefill-vs-single] {label}: hidden cos_min={min_cos:.6} \
             (at token={}, capture_layer={})",
            worst_pos.0, worst_pos.1
        );
        assert!(
            min_cos >= 0.999,
            "{label}: hidden capture cos_min={min_cos} < 0.999 \
             (worst at token={}, capture_layer={})",
            worst_pos.0,
            worst_pos.1
        );

        // ---- Compare GDN state + conv per layer (cos ≥ 0.999). ----
        assert_eq!(
            sess_a.gdn_state.len(),
            sess_b.gdn_state.len(),
            "GDN state vec length mismatch"
        );
        for gi in 0..sess_a.gdn_state.len() {
            let a_state = read_tensor_f32(&sess_a.gdn_state[gi]);
            let b_state = read_tensor_f32(&sess_b.gdn_state[gi]);
            let cs = cosine_f32(&a_state, &b_state);
            let a_conv = read_tensor_f32(&sess_a.gdn_conv[gi]);
            let b_conv = read_tensor_f32(&sess_b.gdn_conv[gi]);
            let cc = cosine_f32(&a_conv, &b_conv);
            assert!(cs >= 0.999, "{label}: GDN[{gi}] state cos={cs} < 0.999");
            assert!(cc >= 0.999, "{label}: GDN[{gi}] conv cos={cc} < 0.999");
        }

        // ---- KV state (only for attn layers; 0.8B has none). ----
        assert_eq!(sess_a.kv_n_pos.len(), sess_b.kv_n_pos.len());
        for ai in 0..sess_a.kv_n_pos.len() {
            assert_eq!(
                sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai],
                "{label}: kv_n_pos[{ai}] mismatch ({} vs {})",
                sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai]
            );
        }
    };

    // Edge cases. Format: (label, total_n, p, prefix_len).
    run_scenario("T<P (T=3,P=8)", 3, 8, 0);
    run_scenario("T==P (T=8,P=8)", 8, 8, 0);
    run_scenario("T==P+r (T=11,P=8)", 11, 8, 0);
    run_scenario("T==2P (T=16,P=8)", 16, 8, 0);
    // chunk_p == 1 final chunk (codex pre-commit ask): T=9, P=8.
    run_scenario("T=P+1 chunk_p=1 (T=9,P=8)", 9, 8, 0);
    // start_position > 0 (codex pre-commit ask): prime with prefix=4
    // single_tokens, then run prefill at start_position=4 over T=10
    // suffix tokens. Exercises the "extend an already-advanced
    // session" public API guarantee that prior tests didn't hit.
    run_scenario("start_position>0 (prefix=4, T=10,P=8)", 10, 8, 4);

    // ---- T=0 must error. ----
    let mut sess_e = MetalSession::fresh(&ctx, &mm, 8).expect("sess E");
    let mut layer_scratch_e = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, 4).expect("scratch E");
    let h_dst_e = MetalTensor::zeros_f32(&ctx, vec![1]).expect("h_dst_e");
    let err = prefill_tokens_with_multi_hidden(
        &mf,
        &[],
        0,
        &mut sess_e,
        &mut layer_scratch_e,
        &capture_layers,
        Some(&h_dst_e),
    );
    match err {
        Err(DFlashError::Metal(crate::metal::MetalError::BadShape { detail, .. })) => {
            assert!(
                detail.contains("empty"),
                "expected 'empty' in T=0 error: {detail}"
            );
        }
        other => panic!("expected BadShape on T=0, got {other:?}"),
    }
}

#[test]
#[ignore]
fn prefill_bf16_bfloat_act_matches_exact_0_8b() {
    let path = "/Users/tito/models/Qwen3.5-0.8B-BF16.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[bf16-bfloat-act-model] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let capture_layers: Vec<u32> = vec![0, 7, 14, 21];
    let k = capture_layers.len();
    let total_n = 16usize;
    let p = 16usize;
    let ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();

    let run = |enabled: bool| {
        crate::metal_forward::with_matmat_bf16_bfloat_act_override(enabled, || {
            let mut session = MetalSession::fresh(&ctx, &mm, total_n + 4).expect("session");
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, p as u32).expect("scratch");
            let h_dst =
                MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst");
            let logits = prefill_tokens_with_multi_hidden(
                &mf,
                &ids,
                0,
                &mut session,
                &mut scratch,
                &capture_layers,
                Some(&h_dst),
            )
            .expect("prefill");
            let hidden = read_tensor_f32(&h_dst);
            let gdn_state: Vec<Vec<f32>> = session.gdn_state.iter().map(read_tensor_f32).collect();
            let gdn_conv: Vec<Vec<f32>> = session.gdn_conv.iter().map(read_tensor_f32).collect();
            (
                logits,
                hidden,
                gdn_state,
                gdn_conv,
                session.kv_n_pos.clone(),
            )
        })
    };

    let exact = run(false);
    let approx = run(true);
    let logits_cos = cosine_f32(&exact.0, &approx.0);
    let hidden_cos = cosine_f32(&exact.1, &approx.1);
    eprintln!("[bf16-bfloat-act-model] logits_cos={logits_cos:.6} hidden_cos={hidden_cos:.6}");
    assert!(logits_cos >= 0.999, "logits cos={logits_cos}");
    assert!(hidden_cos >= 0.999, "hidden cos={hidden_cos}");
    assert_eq!(exact.4, approx.4, "kv positions differ");
    for (i, (a, b)) in exact.2.iter().zip(approx.2.iter()).enumerate() {
        let cos = cosine_f32(a, b);
        assert!(cos >= 0.999, "gdn_state[{i}] cos={cos}");
    }
    for (i, (a, b)) in exact.3.iter().zip(approx.3.iter()).enumerate() {
        let cos = cosine_f32(a, b);
        assert!(cos >= 0.999, "gdn_conv[{i}] cos={cos}");
    }
}

#[test]
#[ignore]
fn prefill_f16_half_act_matches_exact_0_8b() {
    let path = "/Users/tito/models/Qwen3.5-0.8B.f16.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[f16-half-act-model] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let capture_layers: Vec<u32> = vec![0, 7, 14, 21];
    let k = capture_layers.len();
    let total_n = 16usize;
    let p = 16usize;
    let ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();

    let run = |enabled: bool| {
        crate::metal::with_matmat_f16_half_act_override(enabled, || {
            let mut session = MetalSession::fresh(&ctx, &mm, total_n + 4).expect("session");
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, p as u32).expect("scratch");
            let h_dst =
                MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst");
            let logits = prefill_tokens_with_multi_hidden(
                &mf,
                &ids,
                0,
                &mut session,
                &mut scratch,
                &capture_layers,
                Some(&h_dst),
            )
            .expect("prefill");
            let hidden = read_tensor_f32(&h_dst);
            let gdn_state: Vec<Vec<f32>> = session.gdn_state.iter().map(read_tensor_f32).collect();
            let gdn_conv: Vec<Vec<f32>> = session.gdn_conv.iter().map(read_tensor_f32).collect();
            (
                logits,
                hidden,
                gdn_state,
                gdn_conv,
                session.kv_n_pos.clone(),
            )
        })
    };

    let exact = run(false);
    let approx = run(true);
    let logits_cos = cosine_f32(&exact.0, &approx.0);
    let hidden_cos = cosine_f32(&exact.1, &approx.1);
    eprintln!("[f16-half-act-model] logits_cos={logits_cos:.6} hidden_cos={hidden_cos:.6}");
    assert!(logits_cos >= 0.999, "logits cos={logits_cos}");
    assert!(hidden_cos >= 0.999, "hidden cos={hidden_cos}");
    assert_eq!(exact.4, approx.4, "kv positions differ");
    for (i, (a, b)) in exact.2.iter().zip(approx.2.iter()).enumerate() {
        let cos = cosine_f32(a, b);
        assert!(cos >= 0.999, "gdn_state[{i}] cos={cos}");
    }
    for (i, (a, b)) in exact.3.iter().zip(approx.3.iter()).enumerate() {
        let cos = cosine_f32(a, b);
        assert!(cos >= 0.999, "gdn_conv[{i}] cos={cos}");
    }
}

#[test]
#[ignore]
fn prefill_q4_legacy_mm_matches_exact_0_8b() {
    for path in [
        "/Users/tito/models/Qwen3.5-0.8B-Q4_0.gguf",
        "/Users/tito/models/Qwen3.5-0.8B-Q4_1.gguf",
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!("[q4-legacy-mm-model] skipped missing fixture {path}");
            continue;
        }
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(crate::metal::MetalError::EmptyLibrary)
            | Err(crate::metal::MetalError::NoDevice) => return,
            Err(e) => panic!("metal init: {e}"),
        };
        let g = GgufFile::open(path).expect("open");
        let m = Model::from_gguf(&g).expect("load");
        let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
        let mf = MetalForward::new(&ctx, &mm);
        let arch = &mm.arch;
        let h = arch.hidden_size as usize;
        let capture_layers: Vec<u32> = vec![0, 7, 14, 21];
        let k = capture_layers.len();
        let total_n = 16usize;
        let p = 16usize;
        let ids: Vec<i32> = (0..total_n)
            .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
            .collect();

        let run = |enabled: bool| {
            crate::metal::with_matmat_q4_legacy_mm_override(enabled, || {
                let mut session = MetalSession::fresh(&ctx, &mm, total_n + 4).expect("session");
                let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, p as u32)
                    .expect("scratch");
                let h_dst =
                    MetalTensor::zeros_f32(&ctx, vec![(total_n * k * h) as u64]).expect("h_dst");
                let logits = prefill_tokens_with_multi_hidden(
                    &mf,
                    &ids,
                    0,
                    &mut session,
                    &mut scratch,
                    &capture_layers,
                    Some(&h_dst),
                )
                .expect("prefill");
                let hidden = read_tensor_f32(&h_dst);
                let gdn_state: Vec<Vec<f32>> =
                    session.gdn_state.iter().map(read_tensor_f32).collect();
                let gdn_conv: Vec<Vec<f32>> =
                    session.gdn_conv.iter().map(read_tensor_f32).collect();
                (
                    logits,
                    hidden,
                    gdn_state,
                    gdn_conv,
                    session.kv_n_pos.clone(),
                )
            })
        };

        let exact = run(false);
        let approx = run(true);
        let logits_cos = cosine_f32(&exact.0, &approx.0);
        let hidden_cos = cosine_f32(&exact.1, &approx.1);
        eprintln!(
            "[q4-legacy-mm-model] {path} logits_cos={logits_cos:.6} hidden_cos={hidden_cos:.6}"
        );
        assert!(logits_cos >= 0.999, "{path} logits cos={logits_cos}");
        assert!(hidden_cos >= 0.999, "{path} hidden cos={hidden_cos}");
        assert_eq!(exact.4, approx.4, "{path} kv positions differ");
        for (i, (a, b)) in exact.2.iter().zip(approx.2.iter()).enumerate() {
            let cos = cosine_f32(a, b);
            assert!(cos >= 0.999, "{path} gdn_state[{i}] cos={cos}");
        }
        for (i, (a, b)) in exact.3.iter().zip(approx.3.iter()).enumerate() {
            let cos = cosine_f32(a, b);
            assert!(cos >= 0.999, "{path} gdn_conv[{i}] cos={cos}");
        }
    }
}

#[test]
#[ignore]
fn prefill_bf16_bfloat_act_a3b_moe_drift_smoke() {
    let path = "/Users/tito/models/BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("[bf16-bfloat-act-a3b] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let total_n = 32usize;
    let p = 32usize;
    let ids: Vec<i32> = (0..total_n)
        .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
        .collect();

    let run = |enabled: bool| {
        crate::metal_forward::with_matmat_bf16_bfloat_act_override(enabled, || {
            let mut session = MetalSession::fresh(&ctx, &mm, total_n + 4).expect("session");
            let mut scratch =
                MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, p as u32).expect("scratch");
            let logits = prefill_tokens_with_multi_hidden(
                &mf,
                &ids,
                0,
                &mut session,
                &mut scratch,
                &[],
                None,
            )
            .expect("prefill");
            let gdn_state: Vec<Vec<f32>> = session.gdn_state.iter().map(read_tensor_f32).collect();
            let gdn_conv: Vec<Vec<f32>> = session.gdn_conv.iter().map(read_tensor_f32).collect();
            (logits, gdn_state, gdn_conv, session.kv_n_pos.clone())
        })
    };

    let exact = run(false);
    let approx = run(true);
    let logits_cos = cosine_f32(&exact.0, &approx.0);
    assert!(logits_cos >= 0.999, "logits cos={logits_cos}");
    assert_eq!(exact.3, approx.3, "kv positions differ");
    let mut min_state_cos = f64::INFINITY;
    let mut min_state_idx = 0usize;
    for (i, (a, b)) in exact.1.iter().zip(approx.1.iter()).enumerate() {
        let cos = cosine_f32(a, b);
        if cos < min_state_cos {
            min_state_cos = cos;
            min_state_idx = i;
        }
    }
    let mut min_conv_cos = f64::INFINITY;
    let mut min_conv_idx = 0usize;
    for (i, (a, b)) in exact.2.iter().zip(approx.2.iter()).enumerate() {
        let cos = cosine_f32(a, b);
        if cos < min_conv_cos {
            min_conv_cos = cos;
            min_conv_idx = i;
        }
    }
    eprintln!(
        "[bf16-bfloat-act-a3b] logits_cos={logits_cos:.6} \
         min_state_cos={min_state_cos:.6}@{min_state_idx} \
         min_conv_cos={min_conv_cos:.6}@{min_conv_idx}"
    );
    assert!(min_state_cos >= 0.995, "gdn_state min cos={min_state_cos}");
    assert!(min_conv_cos >= 0.995, "gdn_conv min cos={min_conv_cos}");
}

/// **MoE prefill correctness gate** (Qwen3.6-35B-A3B-UD-Q4_K_M, with
/// gate/up expert banks in Q4_K and down expert banks in a mixed Q5_K/Q6_K
/// set). Oracle: a sequential `single_token` loop. Experimental: one
/// `prefill_tokens_with_multi_hidden` call
/// (with empty capture-layers since the per-layer hidden-capture
/// helper returns `UnsupportedMoe` on MoE arches; the lr1 / packed
/// MoE bugs we're guarding against still propagate to final logits
/// and GDN/KV session state, so dropping the per-layer capture
/// loses debug locality but does not weaken the gate).
///
/// `prefill_moe_grouped_enabled()` and `prefill_moe_packed_routed_
/// enabled()` both latch into process-wide `OnceLock`s on first
/// call, so a single test invocation can only validate whichever
/// branch wins per process. To exercise both:
/// ```bash
/// cargo test ..._35b_a3b_moe --release -- --ignored --nocapture           # grouped (default)
/// QWEN_PREFILL_MOE_GROUPED=0 cargo test ..._35b_a3b_moe --release -- --ignored --nocapture  # packed-routed
/// ```
/// Without this gate, the grouped path was default-on but had no
/// non-`#[ignore]` MoE prefill correctness coverage; the May 2026
/// Q4 grouped kernel `lr1` clamp bug (`kernels/moe.metal:686`,
/// missing clamp to `nr1 - 1`) would have been caught here.
///
/// Cosine equivalence ≥ 0.999 required on logits, GDN state+conv
/// per layer, and exact equality on KV pos counters.
///
/// `#[ignore]` because A3B-Q4 dequant + 30B-param prefill is several
/// minutes per scenario.
#[test]
#[ignore]
fn prefill_tokens_matches_single_token_loop_35b_a3b_moe() {
    let path = crate::test_fixtures::A3B_Q4_K_M.path();
    if !std::path::Path::new(path).exists() {
        eprintln!("[moe-prefill-vs-single] skipped — fixture missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(crate::metal::MetalError::EmptyLibrary) | Err(crate::metal::MetalError::NoDevice) => {
            return;
        }
        Err(e) => panic!("metal init: {e}"),
    };
    let g = GgufFile::open(path).expect("open");
    let m = Model::from_gguf(&g).expect("load");
    assert_eq!(
        m.arch.kind,
        crate::model::ArchKind::Moe,
        "expected A3B MoE arch; got {:?}",
        m.arch.kind
    );
    let mm = MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mf = MetalForward::new(&ctx, &mm);

    let arch = &mm.arch;

    // Branch labeling is informational: see test-level doc — the
    // MoE-prefill path is locked per-process by OnceLock on first
    // call. The label here reflects whichever env-flag state was
    // active at process start.
    let active_branch = if std::env::var("QWEN_PREFILL_MOE_GROUPED")
        .as_deref()
        .map(|v| matches!(v, "0" | "false" | "FALSE" | "no" | "NO"))
        .unwrap_or(false)
    {
        "packed-routed (grouped disabled via env)"
    } else {
        "grouped (default)"
    };
    eprintln!("[moe-prefill-vs-single] active branch: {active_branch}");

    // NOTE: `single_token_with_multi_hidden` returns UnsupportedMoe,
    // so the oracle for MoE has to be plain `single_token` and the
    // experimental path passes `&[]` to skip multi-hidden capture.
    // The lr1 clamp bug we're guarding against propagates from the
    // MoE swiglu output → next layer residual → final logits and
    // GDN/KV state — so dropping the per-layer hidden capture
    // loses some debug locality but does NOT weaken the gate.

    let run_scenario = |scenario_label: &str, total_n: usize, p: usize, prefix_len: usize| {
        let branch_label = format!("{scenario_label} | {active_branch}");
        assert!(total_n >= 1 && p >= 1, "{branch_label}: need n,p ≥ 1");

        // Deterministic synthetic token sequence (skip 0 to avoid
        // any specialness around <bos>; vocab on A3B is ~150k).
        let n_total_with_prefix = prefix_len + total_n;
        let all_tokens: Vec<i32> = (0..n_total_with_prefix)
            .map(|i| ((i * 13 + 7) % (arch.vocab_size as usize - 1)) as i32 + 1)
            .collect();
        let prefix_tokens = &all_tokens[..prefix_len];
        let token_ids = &all_tokens[prefix_len..];

        // ---- Oracle path: sequential single_token loop. ----
        let cap = n_total_with_prefix + 4;
        let mut sess_a = MetalSession::fresh(&ctx, &mm, cap).expect("sess A");
        for (i, &tid) in prefix_tokens.iter().enumerate() {
            mf.single_token(tid, i as u32, &mut sess_a)
                .expect("oracle prefix advance");
        }
        let mut last_a = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last_a = mf
                .single_token(tid, (prefix_len + i) as u32, &mut sess_a)
                .expect("oracle forward");
        }

        // ---- Experimental path: one prefill_tokens call. ----
        let mut sess_b = MetalSession::fresh(&ctx, &mm, cap).expect("sess B");
        for (i, &tid) in prefix_tokens.iter().enumerate() {
            mf.single_token(tid, i as u32, &mut sess_b)
                .expect("experimental prefix advance");
        }
        let scratch_plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            &mm,
            p as u32,
            n_total_with_prefix,
            PrefillScratchConfig {
                matrix_query_cap: prefill_attn_matrix_query_cap().unwrap(),
            },
        )
        .expect("scratch plan");
        let expected_plan = scratch_plan.clone();
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(&ctx, &mm, scratch_plan)
                .expect("layer scratch from plan");
        assert_eq!(layer_scratch.prefill_scratch_plan(), &expected_plan);
        if branch_label.starts_with("T=5") {
            layer_scratch
                .ensure_moe_packed_fallback(&ctx)
                .expect("planned fallback growth");
            assert_eq!(
                layer_scratch.moe_inner_pack.n_elements(),
                layer_scratch.moe_inner_full_elems
            );
            assert_eq!(
                layer_scratch.moe_expert_out_pack.n_elements(),
                layer_scratch.moe_out_full_elems
            );
        }
        let last_b = prefill_tokens_with_multi_hidden(
            &mf,
            token_ids,
            prefix_len as u32,
            &mut sess_b,
            &mut layer_scratch,
            &[],
            None,
        )
        .expect("prefill");
        if let Some(query_cap) = prefill_attn_matrix_query_cap().unwrap() {
            assert_eq!(layer_scratch.attn_matrix_query_rows(), p.min(query_cap));
            if token_ids.len().min(p) > query_cap {
                assert!(
                    layer_scratch.attn_matrix_tiled_layer_calls() > 0,
                    "{branch_label}: query-cap gate did not tile matrix attention"
                );
            }
            if prefill_attn_gdn_scratch_overlay_enabled()
                && !prefill_scratch_overlay_diagnostic_mode_present()
            {
                assert!(
                    layer_scratch.prefill_scratch_overlay_stats().is_some(),
                    "{branch_label}: overlay gate constructed separate scratch"
                );
            }
        }

        // ---- Compare final logits (cos ≥ 0.999). ----
        assert_eq!(
            last_a.len(),
            last_b.len(),
            "{branch_label}: logits len mismatch"
        );
        let cos_logits = cosine_f32(&last_a, &last_b);
        eprintln!(
            "[moe-prefill-vs-single] {branch_label}: T={total_n} P={p} prefix={prefix_len} chunks={} cos(logits)={cos_logits:.6}",
            total_n.div_ceil(p)
        );
        assert!(
            cos_logits >= 0.999,
            "{branch_label}: logits cos={cos_logits} < 0.999"
        );

        // ---- Compare GDN state + conv per layer (cos ≥ 0.999). ----
        assert_eq!(
            sess_a.gdn_state.len(),
            sess_b.gdn_state.len(),
            "{branch_label}: GDN state vec length mismatch"
        );
        let mut min_gdn_cos = f64::INFINITY;
        let mut worst_gdn = 0usize;
        for gi in 0..sess_a.gdn_state.len() {
            let a_state = read_tensor_f32(&sess_a.gdn_state[gi]);
            let b_state = read_tensor_f32(&sess_b.gdn_state[gi]);
            let cs = cosine_f32(&a_state, &b_state);
            let a_conv = read_tensor_f32(&sess_a.gdn_conv[gi]);
            let b_conv = read_tensor_f32(&sess_b.gdn_conv[gi]);
            let cc = cosine_f32(&a_conv, &b_conv);
            let layer_min = cs.min(cc);
            if layer_min < min_gdn_cos {
                min_gdn_cos = layer_min;
                worst_gdn = gi;
            }
            assert!(
                cs >= 0.999,
                "{branch_label}: GDN[{gi}] state cos={cs} < 0.999"
            );
            assert!(
                cc >= 0.999,
                "{branch_label}: GDN[{gi}] conv cos={cc} < 0.999"
            );
        }
        eprintln!(
            "[moe-prefill-vs-single] {branch_label}: gdn cos_min={min_gdn_cos:.6} (layer={worst_gdn})"
        );

        // ---- KV pos counters (A3B has attn layers). ----
        assert_eq!(sess_a.kv_n_pos.len(), sess_b.kv_n_pos.len());
        for ai in 0..sess_a.kv_n_pos.len() {
            assert_eq!(
                sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai],
                "{branch_label}: kv_n_pos[{ai}] mismatch ({} vs {})",
                sess_a.kv_n_pos[ai], sess_b.kv_n_pos[ai]
            );
        }
    };

    // Scenarios: kept small because A3B prefill is heavy (~30B
    // active during MoE swiglu/down across all 8-of-128 experts).
    // P=8 matches the production drafter block_size for A3B.
    run_scenario("T=5 single-chunk", 5, 8, 0);
    run_scenario("T=11/P=8 multi-chunk", 11, 8, 0);
    run_scenario("start_position>0 (prefix=2, T=6)", 6, 8, 2);
    run_scenario(
        "A3B packed-attn active-shape (prefix=4096, T=4, P=8)",
        4,
        8,
        4096,
    );
    run_scenario(
        "A3B packed-attn threshold-cross (prefix=4095, T=2, P=8)",
        2,
        8,
        4095,
    );
    run_scenario(
        "A3B packed-attn single-row (prefix=4096, T=1, P=8)",
        1,
        8,
        4096,
    );
    run_scenario(
        "A3B packed-attn full-tile (prefix=4096, T=8, P=8)",
        8,
        8,
        4096,
    );
    run_scenario(
        "A3B packed-attn multi-tile (prefix=4096, T=16, P=16)",
        16,
        16,
        4096,
    );
    run_scenario(
        "A3B packed-attn late-prefix (prefix=8191, T=2, P=8)",
        2,
        8,
        8191,
    );
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn read_tensor_f32(t: &MetalTensor) -> Vec<f32> {
    let n = t.n_elements() as usize;
    let mut out = vec![0.0f32; n];
    unsafe {
        let src = t.buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn read_tensor_bytes(t: &MetalTensor) -> Vec<u8> {
    let n = t.n_bytes() as usize;
    let mut out = vec![0u8; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }
    out
}

fn fuse_q4k_gate_up_expert_banks(
    ctx: &MetalContext,
    gate: &MetalTensor,
    up: &MetalTensor,
    n_hidden: usize,
    n_ffn: usize,
    n_expert: usize,
) -> MetalTensor {
    const Q4K_BYTES: usize = 144;
    let gate_bytes = read_tensor_bytes(gate);
    let up_bytes = read_tensor_bytes(up);
    let row_bytes = (n_hidden / 256) * Q4K_BYTES;
    let blocks_per_row = row_bytes / Q4K_BYTES;
    let per_expert_bytes = row_bytes * n_ffn;
    assert_eq!(gate_bytes.len(), per_expert_bytes * n_expert);
    assert_eq!(up_bytes.len(), per_expert_bytes * n_expert);
    let mut fused = Vec::with_capacity(gate_bytes.len() * 2);
    for expert in 0..n_expert {
        let g_exp = &gate_bytes[expert * per_expert_bytes..(expert + 1) * per_expert_bytes];
        let u_exp = &up_bytes[expert * per_expert_bytes..(expert + 1) * per_expert_bytes];
        for row in 0..n_ffn {
            let row_off = row * row_bytes;
            for block in 0..blocks_per_row {
                let off = row_off + block * Q4K_BYTES;
                fused.extend_from_slice(&g_exp[off..off + Q4K_BYTES]);
                fused.extend_from_slice(&u_exp[off..off + Q4K_BYTES]);
            }
        }
    }
    let fused_elems = (fused.len() / Q4K_BYTES) * 256;
    MetalTensor::from_bytes(ctx, &fused, vec![fused_elems as u64], GgmlType::Q4_K)
        .expect("fused q4k gate/up")
}

/// F1 — windowed drafter-context bit-identity oracle (2026-08-20).
///
/// Preregistered gate; see
/// `docs/bench/2026-08-20-windowed-dflash-pre-gates/README.md`. Lives
/// in-lib so it uses the per-PID test lease directory and can run
/// alongside the production serve daemon without contending its
/// process-exclusive lease.
///
/// Run:
/// ```bash
/// cargo test --release -p qwen-llm windowed_dflash_bit_identity_3076 \
///   -- --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "slow real-model GPU gate; run explicitly"]
fn windowed_dflash_bit_identity_3076() {
    use crate::loader::open_dflash_drafter;
    use crate::metal::MetalError;
    use crate::tokenizer::Tokenizer;

    const PROMPT_TOKENS: usize = 3076;
    const WINDOW: usize = 2048;
    const SLACK: usize = 64;
    const TARGET_GGUF: &str = "/Users/tito/models/Qwen3.8-27B-Q8_0.gguf";
    const DRAFTER_GGUF: &str = "/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf";
    const FIXTURE_TEXT: &str = "The quick brown fox jumps over the lazy dog while the \
        server quietly processes a long agentic conversation about software \
        engineering, performance analysis, and the correct way to serve large \
        language models on Apple silicon without disturbing the production queue. ";

    if !std::path::Path::new(TARGET_GGUF).exists() || !std::path::Path::new(DRAFTER_GGUF).exists() {
        eprintln!("[f1] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("[f1] metal init: {e}"),
    };

    eprintln!("[f1] loading target + drafter…");
    let target_g = GgufFile::open(TARGET_GGUF).expect("open target");
    let target_m = Model::from_gguf(&target_g).expect("load target");
    let drafter_g = GgufFile::open(DRAFTER_GGUF).expect("open drafter");
    let head = open_dflash_drafter(&drafter_g, &target_m).expect("bind drafter");

    eprintln!(
        "[f1] drafter: layers={} swa_all={} block={} target_layers={:?}",
        head.config.n_layer,
        head.layers.iter().all(|l| l.is_swa),
        head.config.block_size,
        head.target_layer_ids,
    );
    assert!(
        head.layers.iter().all(|l| l.is_swa),
        "premise requires an all-SWA drafter"
    );

    let h = target_m.arch.hidden_size as usize;
    let v = target_m.arch.vocab_size as usize;
    let n_features = head.target_layer_ids.len() * h;

    let tok = Tokenizer::open(TARGET_GGUF).expect("tokenizer");
    let mut text = String::new();
    while tok.encode(&text, false).map(|ids| ids.len()).unwrap_or(0) < PROMPT_TOKENS {
        text.push_str(FIXTURE_TEXT);
    }
    let mut prompt_ids = tok.encode(&text, false).expect("encode prompt");
    prompt_ids.truncate(PROMPT_TOKENS);
    eprintln!("[f1] prompt {PROMPT_TOKENS} tokens");

    let mm = MetalModel::load(&ctx, &target_g, &target_m).expect("metal target load");
    let mf = MetalForward::new(&ctx, &mm);
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).expect("metal drafter load");

    let capture = MetalTensor::zeros_f32(&ctx, vec![(PROMPT_TOKENS * n_features) as u64])
        .expect("capture buffer");
    let mut layer_scratch = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, PROMPT_TOKENS as u32)
        .expect("prefill layer scratch");
    let mut target_session =
        MetalSession::fresh(&ctx, &mm, PROMPT_TOKENS + SLACK).expect("target session");

    eprintln!("[f1] metal prefill with multi-hidden capture…");
    prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut target_session,
        &mut layer_scratch,
        &head.target_layer_ids,
        Some(&capture),
    )
    .expect("capture prefill");

    let carry = prompt_ids[PROMPT_TOKENS - 1];
    let noise_start_pos = PROMPT_TOKENS as u32;
    let n = head.config.block_size as usize;

    let seed = |count: usize, start_pos: u32| -> DFlashDecoder<'_> {
        let capacity = count + n + 32;
        let mut dsess = MetalDFlashSession::fresh(&ctx, &mhead, h as u64, v as u64, capacity)
            .expect("drafter session");
        let offset = (PROMPT_TOKENS - count) * n_features;
        let view = capture.view_subrange(offset as u64, vec![(count * n_features) as u64]);
        dsess
            .append_target_ctx_columns_contiguous_now(&ctx, &view, start_pos, count, n_features)
            .expect("seed ctx columns");
        DFlashDecoder::new(&mf, &mhead, dsess)
    };

    let mut dec_a = seed(PROMPT_TOKENS, 0);
    let mut dec_b = seed(WINDOW, (PROMPT_TOKENS - WINDOW) as u32);
    let mut dec_c = seed(WINDOW - 1, (PROMPT_TOKENS - (WINDOW - 1)) as u32);

    let logits_a = dec_a
        .draft_block_with_logits(carry, noise_start_pos)
        .expect("draft A");
    let logits_b = dec_b
        .draft_block_with_logits(carry, noise_start_pos)
        .expect("draft B");
    let logits_c = dec_c
        .draft_block_with_logits(carry, noise_start_pos)
        .expect("draft C");

    assert_eq!(logits_a.len(), n * v);
    assert_eq!(logits_b.len(), n * v);
    assert_eq!(logits_c.len(), n * v);

    let argmaxes = |logits: &[f32]| -> Vec<usize> {
        (0..n)
            .map(|row| {
                let slice = &logits[row * v..(row + 1) * v];
                slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect()
    };
    let arg_a = argmaxes(&logits_a);
    let arg_b = argmaxes(&logits_b);
    let arg_c = argmaxes(&logits_c);

    let mut max_ab = 0.0f32;
    let mut max_ac = 0.0f32;
    for row in 0..n {
        for j in 0..v {
            let a = logits_a[row * v + j];
            let b = logits_b[row * v + j];
            let c = logits_c[row * v + j];
            max_ab = max_ab.max((a - b).abs());
            max_ac = max_ac.max((a - c).abs());
        }
        eprintln!(
            "[f1] row {row}: argmax A={} B={} C={}",
            arg_a[row], arg_b[row], arg_c[row]
        );
    }
    eprintln!("[f1] max|A-B|={max_ab:.3e} max|A-C|={max_ac:.3e}");

    assert_eq!(
        max_ab, 0.0,
        "windowed session B diverges from full session A (max|A-B|={max_ab})"
    );
    assert_eq!(
        arg_a, arg_b,
        "argmax mismatch between full (A) and windowed (B) sessions"
    );
    assert!(
        max_ac > 0.0,
        "negative control C equals full session A — harness is insensitive to window truncation"
    );
    eprintln!("[f1] PASS: windowed == full bitwise; truncated control diverges");
}

/// F2-adjacent split-capture equivalence: the 1a production mechanics
/// (plain prefill outside the capture window, capture sub-span inside)
/// must produce a window buffer bit-identical to the suffix of a
/// full-span capture, and the resulting windowed session must draft
/// bit-identically to the full session.
#[test]
#[ignore = "slow real-model GPU gate; run explicitly"]
fn windowed_split_capture_equals_full_capture() {
    use crate::loader::open_dflash_drafter;
    use crate::metal::MetalError;
    use crate::tokenizer::Tokenizer;

    const PROMPT_TOKENS: usize = 3076;
    const WINDOW: usize = DFLASH_CAPTURE_WINDOW;
    const SLACK: usize = 64;
    const TARGET_GGUF: &str = "/Users/tito/models/Qwen3.8-27B-Q8_0.gguf";
    const DRAFTER_GGUF: &str = "/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf";
    const FIXTURE_TEXT: &str = "The quick brown fox jumps over the lazy dog while the \
        server quietly processes a long agentic conversation about software \
        engineering, performance analysis, and the correct way to serve large \
        language models on Apple silicon without disturbing the production queue. ";

    if !std::path::Path::new(TARGET_GGUF).exists() || !std::path::Path::new(DRAFTER_GGUF).exists() {
        eprintln!("[f2-split] skipped — fixtures missing");
        return;
    }
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("[f2-split] metal init: {e}"),
    };

    eprintln!("[f2-split] loading target + drafter…");
    let target_g = GgufFile::open(TARGET_GGUF).expect("open target");
    let target_m = Model::from_gguf(&target_g).expect("load target");
    let drafter_g = GgufFile::open(DRAFTER_GGUF).expect("open drafter");
    let head = open_dflash_drafter(&drafter_g, &target_m).expect("bind drafter");
    assert_eq!(
        dflash_capture_window_limit(
            &MetalDFlashHead::load(&ctx, &drafter_g, &head).expect("metal head")
        ),
        WINDOW,
        "premise requires the all-SWA head to window at 2048"
    );

    let h = target_m.arch.hidden_size as usize;
    let v = target_m.arch.vocab_size as usize;
    let n_features = head.target_layer_ids.len() * h;

    let tok = Tokenizer::open(TARGET_GGUF).expect("tokenizer");
    let mut text = String::new();
    while tok.encode(&text, false).map(|ids| ids.len()).unwrap_or(0) < PROMPT_TOKENS {
        text.push_str(FIXTURE_TEXT);
    }
    let mut prompt_ids = tok.encode(&text, false).expect("encode prompt");
    prompt_ids.truncate(PROMPT_TOKENS);
    eprintln!("[f2-split] prompt {PROMPT_TOKENS} tokens");

    let mm = MetalModel::load(&ctx, &target_g, &target_m).expect("metal target load");
    let mf = MetalForward::new(&ctx, &mm);
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).expect("metal drafter load");

    // Full-span capture (reference).
    let capture_full = MetalTensor::zeros_f32(&ctx, vec![(PROMPT_TOKENS * n_features) as u64])
        .expect("full capture buffer");
    let mut scratch_full = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, PROMPT_TOKENS as u32)
        .expect("full layer scratch");
    let mut sess_full =
        MetalSession::fresh(&ctx, &mm, PROMPT_TOKENS + SLACK).expect("full target session");
    prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut sess_full,
        &mut scratch_full,
        &head.target_layer_ids,
        Some(&capture_full),
    )
    .expect("full capture prefill");

    // Windowed split: plain [0, wstart) + capture [wstart, PROMPT_TOKENS).
    let wstart = PROMPT_TOKENS - WINDOW;
    let capture_win = MetalTensor::zeros_f32(&ctx, vec![(WINDOW * n_features) as u64])
        .expect("window capture buffer");
    let mut scratch_win = MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, PROMPT_TOKENS as u32)
        .expect("window layer scratch");
    let mut sess_win =
        MetalSession::fresh(&ctx, &mm, PROMPT_TOKENS + SLACK).expect("window target session");
    prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids[..wstart],
        0,
        &mut sess_win,
        &mut scratch_win,
        &[],
        None,
    )
    .expect("plain prefix prefill");
    prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids[wstart..],
        wstart as u32,
        &mut sess_win,
        &mut scratch_win,
        &head.target_layer_ids,
        Some(&capture_win),
    )
    .expect("window capture prefill");

    // Bitwise: window buffer == suffix of full buffer.
    let full = read_tensor_f32(&capture_full);
    let win = read_tensor_f32(&capture_win);
    let suffix = &full[wstart * n_features..];
    assert_eq!(win.len(), suffix.len());
    let mut max_delta = 0.0f32;
    for (a, b) in win.iter().zip(suffix) {
        max_delta = max_delta.max((a - b).abs());
    }
    eprintln!("[f2-split] max|capture_win - full_suffix|={max_delta:.3e}");
    assert_eq!(
        max_delta, 0.0,
        "windowed split capture diverges from full-capture suffix"
    );

    // Draft bit-identity: windowed session vs full session.
    let carry = prompt_ids[PROMPT_TOKENS - 1];
    let noise_start_pos = PROMPT_TOKENS as u32;
    let n = head.config.block_size as usize;
    let seed = |count: usize, start_pos: u32, src: &MetalTensor| -> DFlashDecoder<'_> {
        let capacity = count + n + 32;
        let mut dsess = MetalDFlashSession::fresh(&ctx, &mhead, h as u64, v as u64, capacity)
            .expect("drafter session");
        let offset = (PROMPT_TOKENS - count) * n_features;
        let view = src.view_subrange(offset as u64, vec![(count * n_features) as u64]);
        dsess
            .append_target_ctx_columns_contiguous_now(&ctx, &view, start_pos, count, n_features)
            .expect("seed ctx columns");
        DFlashDecoder::new(&mf, &mhead, dsess)
    };
    let mut dec_full = seed(PROMPT_TOKENS, 0, &capture_full);
    let mut dec_win = seed(WINDOW, wstart as u32, &capture_full);
    let logits_full = dec_full
        .draft_block_with_logits(carry, noise_start_pos)
        .expect("draft full");
    let logits_win = dec_win
        .draft_block_with_logits(carry, noise_start_pos)
        .expect("draft windowed");
    let mut max_delta = 0.0f32;
    for (a, b) in logits_full.iter().zip(&logits_win) {
        max_delta = max_delta.max((a - b).abs());
    }
    eprintln!("[f2-split] max|draft_full - draft_windowed|={max_delta:.3e}");
    assert_eq!(
        max_delta, 0.0,
        "windowed-split session drafts diverge from full session"
    );
    eprintln!("[f2-split] PASS: split capture == full suffix; drafts bit-identical");
}

mod prompt_lookup_layout_gate {
    use super::*;
    use crate::model::{QWEN3_0_8B, QWEN3_27B};

    fn q4km_block() -> PromptLookupBlockLayout {
        PromptLookupBlockLayout {
            gate: GgmlType::Q4_K,
            up: GgmlType::Q4_K,
            down: GgmlType::Q6_K,
            moe: false,
        }
    }

    #[test]
    fn qualified_q4km_layout_passes() {
        let blocks = (0..64).map(|i| PromptLookupBlockLayout {
            down: if i % 2 == 0 {
                GgmlType::Q4_K
            } else {
                GgmlType::Q6_K
            },
            ..q4km_block()
        });
        assert_eq!(
            check_prompt_lookup_n8_layout(&QWEN3_27B, GgmlType::Q6_K, blocks),
            Ok(())
        );
    }

    #[test]
    fn non_27b_architecture_is_refused_before_blocks_are_inspected() {
        let err = check_prompt_lookup_n8_layout(&QWEN3_0_8B, GgmlType::Q6_K, std::iter::empty())
            .unwrap_err();
        assert!(err.contains("dense 27B architecture"), "{err}");
    }

    #[test]
    fn q8_lm_head_is_refused() {
        let err = check_prompt_lookup_n8_layout(&QWEN3_27B, GgmlType::Q8_0, std::iter::empty())
            .unwrap_err();
        assert!(err.contains("lm_head is Q8_0"), "{err}");
    }

    #[test]
    fn a_single_offending_block_names_its_index_and_dtypes() {
        let mut blocks = vec![q4km_block(); 64];
        blocks[17].gate = GgmlType::Q8_0;
        let err = check_prompt_lookup_n8_layout(&QWEN3_27B, GgmlType::Q6_K, blocks).unwrap_err();
        assert!(
            err.contains("block 17 has gate/up/down=Q8_0/Q4_K/Q6_K, moe=false"),
            "{err}"
        );
    }

    #[test]
    fn routed_experts_are_refused() {
        let mut blocks = vec![q4km_block(); 64];
        blocks[0].moe = true;
        let err = check_prompt_lookup_n8_layout(&QWEN3_27B, GgmlType::Q6_K, blocks).unwrap_err();
        assert!(err.contains("moe=true"), "{err}");
    }
}

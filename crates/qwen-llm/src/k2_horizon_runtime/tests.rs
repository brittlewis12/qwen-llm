use super::*;

fn config() -> K2HorizonConfig {
    K2HorizonConfig {
        layer_count: 36,
        context_length: 524288,
        hidden_size: 4096,
        feed_forward_size: 12288,
        vocab_size: 250624,
        query_head_count: 32,
        kv_head_count: 8,
        key_head_dim: 128,
        value_head_dim: 128,
        norm_groups: 4,
        rms_epsilon: 1e-6,
        rope_dimension_count: 128,
        rope_theta: 10_000_000.0,
    }
}

#[test]
fn session_memory_is_capacity_shaped_with_one_logits_row() {
    for capacity in [1, 17, 7168] {
        let plan = SessionMemoryPlan::new(&config(), capacity).unwrap();
        assert_eq!(plan.specs().len(), 13);
        assert_eq!(
            plan.buffer_bytes().iter().sum::<u64>(),
            147456 * u64::from(capacity) + 1_240_068
        );
        assert_eq!(plan.specs()[11], (GgmlType::F32, vec![250624]));
        assert_eq!(
            plan.specs()[12],
            (GgmlType::F16, vec![plan.cache_bytes / 2])
        );
    }
    assert!(SessionMemoryPlan::new(&config(), 7169).is_err());
}

#[test]
fn ledger_validates_all_ids_and_bounds_without_state_changes() {
    let mut ledger = Ledger::default();
    for tokens in [vec![], vec![0, 250624], vec![u32::MAX], vec![0; 5]] {
        assert!(ledger.begin(&tokens, 250624, 4).is_err());
        assert_eq!(ledger.prefix(), 0);
        assert!(!ledger.is_poisoned());
    }
    assert!(ledger.begin(&[i32::MAX as u32 + 1], u32::MAX, 4).is_err());
    drop(ledger.begin(&[250623], 250624, 4).unwrap());
    assert!(!ledger.is_poisoned());
}

#[test]
fn ledger_commits_once_only_after_all_synchronous_checks() {
    let mut ledger = Ledger::default();
    let mut transaction = ledger.begin(&[1, 2, 3], 10, 4).unwrap();
    for _ in 0..3 {
        assert_eq!(transaction.old_prefix(), 0);
        transaction.submitting().unwrap();
        transaction.checked().unwrap();
    }
    assert!(transaction.submitting().is_err());
    transaction.commit().unwrap();
    assert_eq!(ledger.prefix(), 3);
    let mut transaction = ledger.begin(&[4], 10, 4).unwrap();
    assert_eq!(transaction.old_prefix(), 3);
    transaction.submitting().unwrap();
    transaction.checked().unwrap();
    transaction.commit().unwrap();
    assert_eq!(ledger.prefix(), 4);
    assert!(ledger.begin(&[0], 10, 4).is_err());
}

#[test]
fn abandoned_submission_or_finite_failure_poisons_without_partial_prefix() {
    for checked in [0, 1, 2] {
        let mut ledger = Ledger::default();
        let mut initial = ledger.begin(&[0], 10, 8).unwrap();
        initial.submitting().unwrap();
        initial.checked().unwrap();
        initial.commit().unwrap();
        {
            let mut append = ledger.begin(&[1, 2, 3], 10, 8).unwrap();
            for _ in 0..checked {
                append.submitting().unwrap();
                append.checked().unwrap();
            }
            append.submitting().unwrap();
            // A command failure, finite-check failure, cancellation, or unwind.
        }
        assert_eq!(ledger.prefix(), 1);
        assert!(ledger.is_poisoned());
        assert!(matches!(
            ledger.begin(&[0], 10, 8),
            Err(K2RuntimeError::Poisoned)
        ));
    }
}

#[test]
fn encoding_failure_before_submission_is_retryable_but_partial_append_is_not() {
    let mut ledger = Ledger::default();
    {
        let mut append = ledger.begin(&[0, 1], 10, 8).unwrap();
        assert!(append.checked().is_err());
    }
    assert!(!ledger.is_poisoned());
    {
        let mut append = ledger.begin(&[0, 1], 10, 8).unwrap();
        append.submitting().unwrap();
        assert!(append.submitting().is_err());
        append.checked().unwrap();
        // Encoding the second command fails after a successful first command.
    }
    assert_eq!(ledger.prefix(), 0);
    assert!(ledger.is_poisoned());
}

#[test]
fn premature_commit_cannot_expose_staged_rows() {
    let mut ledger = Ledger::default();
    assert!(ledger.begin(&[0], 10, 4).unwrap().commit().is_err());
    assert!(!ledger.is_poisoned());
    let mut append = ledger.begin(&[0, 1], 10, 4).unwrap();
    append.submitting().unwrap();
    append.checked().unwrap();
    assert!(append.commit().is_err());
    assert_eq!(ledger.prefix(), 0);
    assert!(ledger.is_poisoned());
}

#[test]
fn cpu_inspected_session_allocations_require_shared_zero_offset_bounds() {
    assert!(validate_cpu_layout(true, 0, 16, 16).is_ok());
    assert!(validate_cpu_layout(true, 0, 16, 32).is_ok());
    for (shared, offset, bytes, length) in [
        (false, 0, 16, 16),
        (true, 2, 16, 32),
        (true, 0, 0, 16),
        (true, 0, 32, 16),
        (true, 0, u64::MAX, u64::MAX),
    ] {
        assert!(validate_cpu_layout(shared, offset, bytes, length).is_err());
    }
}

#[test]
fn session_permit_rejects_concurrency_and_releases_on_drop() {
    let active = Cell::new(false);
    let first = SessionPermit::acquire(&active).unwrap();
    assert!(SessionPermit::acquire(&active).is_err());
    assert!(active.get());
    drop(first);
    assert!(!active.get());
    let replacement = SessionPermit::acquire(&active).unwrap();
    drop(replacement);
    assert!(!active.get());
}

#[test]
#[ignore = "CPU/header only, requires K2_GGUF; no Metal context or weight execution"]
fn inspect_downloaded_runtime_plan_without_metal() {
    let path = std::env::var("K2_GGUF").expect("K2_GGUF");
    let source = GgufFile::open(path).unwrap();
    let plan = K2RuntimePlan::inspect(&source, 32, 16384, 8 * 1024 * 1024 * 1024).unwrap();
    assert_eq!(plan.weight_payload_bytes(), 9_562_505_216);
    assert_eq!(plan.config().layer_count, 36);
    assert_eq!(source.tensors.len(), 327);
    let physical = plan.weight_buffer_bytes().unwrap().iter().sum::<u64>();
    assert!(physical >= plan.weight_payload_bytes());
    assert!(physical < plan.weight_payload_bytes() + 1024 * 1024);
    assert_eq!(
        plan.session_buffer_bytes().iter().sum::<u64>(),
        32 * 147456 + 1_240_068
    );
    plan.revalidate_source().unwrap();
}

#[test]
#[ignore = "UNQUALIFIED full checkpoint GPU execution; requires separate explicit authorization and K2_GGUF"]
fn gpu_checkpoint_forward_and_split_prefill_smoke() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let path = std::env::var("K2_GGUF").expect("K2_GGUF");
    let source = GgufFile::open(path).unwrap();
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 4).unwrap();
    let mut full = model.create_session(0).unwrap();
    let expected = full.append(&[0, 42, 17]).unwrap();
    assert_eq!(expected.len(), 250624);
    assert_eq!(full.committed_len(), 3);
    assert!(model.create_session(0).is_err());
    drop(full);
    let mut split = model.create_session(0).unwrap();
    split.append(&[0, 42]).unwrap();
    let actual = split.append(&[17]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(split.committed_len(), 3);
    assert!(split.append(&[250624]).is_err());
    assert_eq!(split.committed_len(), 3);
    assert!(!split.is_poisoned());
    split.append(&[19]).unwrap();
    assert!(split.append(&[20]).is_err());
    assert_eq!(split.committed_len(), 4);
}

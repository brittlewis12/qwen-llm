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
fn application_default_is_online_without_capacity_or_cache_layout_change() {
    assert_eq!(DEFAULT_ATTENTION_BACKEND, AttentionBackend::Online);
    let plan = SessionMemoryPlan::new(&config(), 256, K2KvStorage::F16).unwrap();
    assert_eq!(plan.cache_bytes, 147456 * 256);
    assert_eq!(
        plan.buffer_bytes().iter().sum::<u64>(),
        147456 * 256 + 1_240_068
    );
}

#[test]
fn compact_session_replaces_the_arena_without_hidden_float_history() {
    for capacity in [1, 32, 256, 7168] {
        let f16 = SessionMemoryPlan::new(&config(), capacity, K2KvStorage::F16).unwrap();
        let q8 = SessionMemoryPlan::new(&config(), capacity, K2KvStorage::Q8_0).unwrap();
        assert_eq!(q8.specs().len(), 13);
        assert_eq!(&q8.specs()[..12], &f16.specs()[..12]);
        assert_eq!(q8.cache_bytes, 78336 * u64::from(capacity));
        assert_eq!(q8.specs()[12], (GgmlType::I8, vec![q8.cache_bytes]));
        assert_eq!(
            q8.buffer_bytes().iter().sum::<u64>(),
            q8.cache_bytes + 1_240_068
        );
        assert_eq!(
            f16.cache_bytes - q8.cache_bytes,
            69120 * u64::from(capacity)
        );
    }
}

#[test]
fn session_memory_is_capacity_shaped_with_one_logits_row() {
    for capacity in [1, 17, 7168, 7169, 8192, 524288] {
        let plan = SessionMemoryPlan::new(&config(), capacity, K2KvStorage::F16).unwrap();
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
    assert!(SessionMemoryPlan::new(&config(), 524289, K2KvStorage::F16).is_err());
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
fn readout_transaction_completes_once_without_advancing_even_at_capacity() {
    let mut ledger = Ledger::default();
    let mut append = ledger.begin(&[0], 10, 1).unwrap();
    append.submitting().unwrap();
    append.checked().unwrap();
    append.commit().unwrap();
    let mut readout = ledger.begin_readout().unwrap();
    assert_eq!(readout.old_prefix(), 1);
    assert!(readout.checked().is_err());
    readout.submitting().unwrap();
    assert!(readout.submitting().is_err());
    readout.checked().unwrap();
    assert!(readout.checked().is_err());
    assert!(readout.submitting().is_err());
    readout.commit().unwrap();
    assert_eq!(ledger.prefix(), 1);
    assert!(!ledger.is_poisoned());
    assert!(ledger.begin(&[], 10, 1).is_err());
    assert!(ledger.begin(&[0], 10, 1).is_err());
}

#[test]
fn readout_abandonment_after_submission_poisons_without_advancing() {
    for completed in [false, true] {
        let mut ledger = Ledger::default();
        drop(ledger.begin_readout().unwrap());
        assert!(ledger.begin_readout().unwrap().commit().is_err());
        assert!(!ledger.is_poisoned());
        {
            let mut readout = ledger.begin_readout().unwrap();
            readout.submitting().unwrap();
            if completed {
                readout.checked().unwrap();
            }
        }
        assert_eq!(ledger.prefix(), 0);
        assert!(ledger.is_poisoned());
        assert!(matches!(
            ledger.begin_readout(),
            Err(K2RuntimeError::Poisoned)
        ));
        assert!(matches!(
            ledger.begin(&[0], 10, 1),
            Err(K2RuntimeError::Poisoned)
        ));
    }
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
    assert_eq!(model.attention, AttentionBackend::Online);
    assert_eq!(model.prefill, PrefillMode::BatchQ8);
    assert_eq!(model.prefill_info(3).chunk_tokens, 3);
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

#[test]
#[ignore = "full checkpoint GPU lens correctness; requires production lease and K2_GGUF"]
fn gpu_checkpoint_captures_and_readout_preserve_continuation() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let source = GgufFile::open(std::env::var("K2_GGUF").expect("K2_GGUF")).unwrap();
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 4).unwrap();
    let layers = (0..36).collect::<Vec<_>>();
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let cache_bits = |session: &K2Session<'_, '_>| unsafe {
        // Private checked shared F16 arena; no command is in flight here.
        std::slice::from_raw_parts(
            session
                .buffers
                .cache
                .buffer
                .contents()
                .as_ptr()
                .cast::<u16>(),
            session.buffers.cache.n_elements() as usize,
        )
        .to_vec()
    };

    let mut session = model.create_session(37).unwrap();
    let empty_cache = cache_bits(&session);
    assert!(
        session
            .readout(&vec![0.0; 4096])
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert_eq!(session.committed_len(), 0);
    assert_eq!(cache_bits(&session), empty_cache);
    let captured = session.append_with_captures(&[0, 42, 17], &layers).unwrap();
    assert_eq!(captured.absolute_position, 39);
    assert_eq!(captured.post_block_layers, layers);
    assert_eq!(captured.residuals.len(), 36 * 4096);
    assert!(captured.residuals.iter().all(|v| v.is_finite()));
    assert_ne!(
        &captured.residuals[..4096],
        &captured.residuals[35 * 4096..]
    );
    assert_eq!(
        &captured.residuals[35 * 4096..],
        read_f32(&session.buffers.residual)
    );
    let cache = cache_bits(&session);
    let final_readout = session.readout(&captured.residuals[35 * 4096..]).unwrap();
    assert_eq!(bits(&final_readout), bits(&captured.logits));
    assert_eq!(session.committed_len(), 3);
    assert_eq!(cache_bits(&session), cache);
    assert!(
        session
            .readout(&vec![0.0; 4096])
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    for sites in [vec![], vec![0, 0], vec![35, 0], vec![36]] {
        assert!(session.append_with_captures(&[19], &sites).is_err());
    }
    assert!(session.append_with_captures(&[250624], &[35]).is_err());
    for row in [
        vec![],
        vec![0.0; 4095],
        vec![0.0; 4097],
        vec![f32::NAN; 4096],
    ] {
        assert!(session.readout(&row).is_err());
    }
    assert_eq!(session.committed_len(), 3);
    assert!(!session.is_poisoned());
    assert_eq!(cache_bits(&session), cache);
    let continuation = session.append_with_captures(&[19], &[0, 17, 35]).unwrap();
    assert_eq!(continuation.absolute_position, 40);
    let full_cache = cache_bits(&session);
    assert_eq!(
        bits(
            &session
                .readout(&continuation.residuals[2 * 4096..])
                .unwrap()
        ),
        bits(&continuation.logits)
    );
    assert_eq!(session.committed_len(), 4);
    assert_eq!(cache_bits(&session), full_cache);
    drop(session);

    let mut split = model.create_session(37).unwrap();
    split.append(&[0, 42]).unwrap();
    let split_captured = split.append_with_captures(&[17], &layers).unwrap();
    assert_eq!(split_captured.absolute_position, captured.absolute_position);
    assert_eq!(bits(&split_captured.residuals), bits(&captured.residuals));
    assert_eq!(bits(&split_captured.logits), bits(&captured.logits));
    let all_continuation = split.append_with_captures(&[19], &layers).unwrap();
    for (row, layer) in [0, 17, 35].into_iter().enumerate() {
        assert_eq!(
            bits(&continuation.residuals[row * 4096..(row + 1) * 4096]),
            bits(&all_continuation.residuals[layer * 4096..(layer + 1) * 4096]),
        );
    }
    drop(split);

    let mut plain = model.create_session(37).unwrap();
    assert_eq!(
        bits(&plain.append(&[0, 42, 17]).unwrap()),
        bits(&captured.logits)
    );
    assert_eq!(
        bits(&plain.append(&[19]).unwrap()),
        bits(&continuation.logits)
    );
    assert_eq!(cache_bits(&plain), full_cache);
}

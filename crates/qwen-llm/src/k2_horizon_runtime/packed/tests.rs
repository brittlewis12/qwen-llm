use super::*;

#[test]
fn automatic_selection_requires_complete_q8_projection_inventory_and_lcpp() {
    assert!(eligible(std::iter::repeat_n(GgmlType::Q8_0, 252), true));
    assert!(!eligible(std::iter::repeat_n(GgmlType::Q8_0, 252), false));
    for count in [0, 1, 251, 253] {
        assert!(!eligible(std::iter::repeat_n(GgmlType::Q8_0, count), true));
    }
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ] {
        assert!(!eligible(
            std::iter::repeat_n(GgmlType::Q8_0, 251).chain([dtype]),
            true
        ));
    }
    for tokens in [0usize, 1, 2, 31, 32, 33, 256] {
        let serial = PrefillMode::Serial.info(tokens);
        assert_eq!(serial.mode, "serial_single_token");
        assert_eq!(serial.commands, tokens);
        assert_eq!(serial.temporary_activation_bytes, 0);
        let packed = PrefillMode::BatchQ8.info(tokens);
        assert_eq!(packed.chunk_tokens, tokens.min(32));
        assert_eq!(packed.commands, tokens.div_ceil(32));
        assert_eq!(
            packed.temporary_activation_bytes,
            if tokens > 1 {
                237572 * tokens.min(32) as u64
            } else {
                0
            }
        );
    }
}

#[test]
fn scratch_is_bounded_and_excludes_persistent_cache_and_head() {
    for rows in [2, 3, 31, 32] {
        let specs = scratch_specs(rows).unwrap();
        assert_eq!(specs.len(), 11);
        assert_eq!(specs[0], (GgmlType::I32, vec![rows as u64]));
        assert_eq!(
            specs.iter().map(|(_, s)| s[0] * 4).sum::<u64>(),
            rows as u64 * 237572
        );
    }
    for rows in [0, 1, 33, usize::MAX] {
        assert!(scratch_specs(rows).is_err());
    }
}

#[test]
fn chunked_ledger_counts_commands_but_commits_tokens_once() {
    for count in [1, 2, 31, 32, 33, 63, 64, 65, 255, 256] {
        let mut ledger = Ledger::default();
        let tokens = vec![42; count];
        let mut transaction = ledger.begin_chunked(&tokens, 250624, 256, 32).unwrap();
        for _ in 0..count.div_ceil(32) {
            assert_eq!(transaction.old_prefix(), 0);
            transaction.submitting().unwrap();
            transaction.checked().unwrap();
        }
        assert!(transaction.submitting().is_err());
        transaction.commit().unwrap();
        assert_eq!(ledger.prefix(), count as u32);
        assert!(!ledger.is_poisoned());
    }
    let mut ledger = Ledger::default();
    for chunk in [0, 33, usize::MAX] {
        assert!(ledger.begin_chunked(&[0], 10, 256, chunk).is_err());
    }
    assert!(ledger.begin_chunked(&[0; 257], 10, 256, 32).is_err());
    let mut invalid = vec![0; 65];
    invalid[64] = 250624;
    assert!(ledger.begin_chunked(&invalid, 250624, 256, 32).is_err());
    assert_eq!(ledger.prefix(), 0);
    assert!(!ledger.is_poisoned());
}

#[test]
fn abandoned_chunks_poison_only_after_submission_and_never_publish_partial_prefix() {
    for complete in [0, 1, 2] {
        let mut ledger = Ledger::default();
        {
            let mut transaction = ledger.begin_chunked(&[0; 65], 10, 256, 32).unwrap();
            for _ in 0..complete {
                transaction.submitting().unwrap();
                transaction.checked().unwrap();
            }
        }
        assert_eq!(ledger.prefix(), 0);
        assert_eq!(ledger.is_poisoned(), complete > 0);
    }
    let mut ledger = Ledger::default();
    let mut transaction = ledger.begin_chunked(&[0; 65], 10, 256, 32).unwrap();
    transaction.submitting().unwrap();
    transaction.checked().unwrap();
    assert!(transaction.commit().is_err());
    assert_eq!(ledger.prefix(), 0);
    assert!(ledger.is_poisoned());
}

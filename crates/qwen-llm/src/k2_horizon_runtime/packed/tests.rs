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
    for rows in [2, 3, 31, 32, 128, 256] {
        let specs = scratch_specs(rows).unwrap();
        assert_eq!(specs.len(), 11);
        assert_eq!(specs[0], (GgmlType::I32, vec![rows as u64]));
        assert_eq!(
            specs.iter().map(|(_, s)| s[0] * 4).sum::<u64>(),
            rows as u64 * 237572
        );
        assert_eq!(
            scratch_bytes(rows).unwrap().iter().sum::<u64>(),
            rows as u64 * ACTIVATION_BYTES_PER_ROW
        );
    }
    for rows in [0, 1, 257, usize::MAX] {
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
    for chunk in [0, 257, usize::MAX] {
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

/// llama.cpp Q4_K_M mix: Q4_K everywhere except V and down, which alternate
/// Q6_K/Q4_K by layer (the "use more bits" layers).
fn q4_k_m_like() -> Vec<GgmlType> {
    (0..36)
        .flat_map(|layer| {
            let wide = if layer % 2 == 0 {
                GgmlType::Q6_K
            } else {
                GgmlType::Q4_K
            };
            [
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                wide,
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                wide,
            ]
        })
        .collect()
}

fn uniform(dtype: GgmlType) -> Vec<GgmlType> {
    vec![dtype; 252]
}

#[test]
fn prefill_request_parsing_is_explicit() {
    for value in [None, Some(""), Some("general"), Some(" general ")] {
        assert_eq!(
            PrefillRequest::parse(value).unwrap(),
            PrefillRequest::General
        );
    }
    assert_eq!(
        PrefillRequest::parse(Some("q8_lcpp")).unwrap(),
        PrefillRequest::Q8Lcpp
    );
    assert_eq!(
        PrefillRequest::parse(Some("serial")).unwrap(),
        PrefillRequest::Serial
    );
    for value in ["auto", "batch", "Q8_LCPP", "1"] {
        assert!(PrefillRequest::parse(Some(value)).is_err(), "{value}");
    }
}

#[test]
fn general_batched_prefill_is_default_for_every_admitted_projection_dtype() {
    let mut mixed = q4_k_m_like();
    mixed[5] = GgmlType::Q5_K;
    mixed[9] = GgmlType::BF16;
    mixed[20] = GgmlType::F16;
    mixed[33] = GgmlType::F32;
    for (name, dtypes) in [
        ("q4_k_m", q4_k_m_like()),
        ("q8_0", uniform(GgmlType::Q8_0)),
        ("q6_k", uniform(GgmlType::Q6_K)),
        ("q5_k", uniform(GgmlType::Q5_K)),
        ("f16", uniform(GgmlType::F16)),
        ("bf16", uniform(GgmlType::BF16)),
        ("mixed", mixed),
    ] {
        for lcpp in [false, true] {
            let mut asked = Vec::new();
            let selection = select(PrefillRequest::General, &dtypes, lcpp, |rows| {
                asked.push(rows);
                Ok(true)
            })
            .unwrap();
            assert_eq!(
                selection.mode,
                PrefillMode::General { chunk: 256 },
                "{name}"
            );
            assert_eq!(selection.reason, "default");
            assert_eq!(asked, [256], "{name}: priced only the admitted chunk");
        }
    }
}

#[test]
fn general_batched_prefill_shrinks_chunk_under_memory_pricing_before_serial() {
    let dtypes = q4_k_m_like();
    let mut asked = Vec::new();
    let selection = select(PrefillRequest::General, &dtypes, true, |rows| {
        asked.push(rows);
        Ok(rows <= 64)
    })
    .unwrap();
    assert_eq!(selection.mode, PrefillMode::General { chunk: 64 });
    assert_eq!(asked, [256, 128, 64]);
    assert!(
        selection.reason.contains("from 256 to 64"),
        "{}",
        selection.reason
    );

    let mut asked = Vec::new();
    let selection = select(PrefillRequest::General, &dtypes, true, |rows| {
        asked.push(rows);
        Ok(false)
    })
    .unwrap();
    assert_eq!(selection.mode, PrefillMode::Serial);
    assert_eq!(asked, [256, 128, 64, 32, 16, 8, 4, 2]);
    assert!(
        selection.reason.contains("memory admission denied"),
        "{}",
        selection.reason
    );
    // Pricing failures propagate instead of silently choosing a topology.
    assert!(
        select(PrefillRequest::General, &dtypes, true, |_| Err(invalid(
            "x"
        )))
        .is_err()
    );
}

#[test]
fn general_selection_reports_unsupported_dtypes_and_rejects_bad_inventory() {
    for dtype in [GgmlType::Q4_0, GgmlType::IQ4_NL, GgmlType::Q2_K] {
        let mut dtypes = q4_k_m_like();
        dtypes[100] = dtype;
        let selection = select(PrefillRequest::General, &dtypes, true, |_| {
            panic!("no pricing for an unsupported inventory")
        })
        .unwrap();
        assert_eq!(selection.mode, PrefillMode::Serial);
        assert!(
            selection.reason.contains(&format!("{dtype:?}")),
            "{}",
            selection.reason
        );
    }
    for count in [0, 251, 253] {
        assert!(
            select(
                PrefillRequest::General,
                &vec![GgmlType::Q8_0; count],
                true,
                |_| Ok(true)
            )
            .is_err()
        );
    }
}

#[test]
fn explicit_q8_lcpp_and_serial_requests_never_substitute() {
    let q8 = uniform(GgmlType::Q8_0);
    let selection = select(PrefillRequest::Q8Lcpp, &q8, true, |rows| {
        assert_eq!(rows, 32);
        Ok(true)
    })
    .unwrap();
    assert_eq!(selection.mode, PrefillMode::BatchQ8);
    assert!(select(PrefillRequest::Q8Lcpp, &q8, false, |_| Ok(true)).is_err());
    assert!(select(PrefillRequest::Q8Lcpp, &q8, true, |_| Ok(false)).is_err());
    assert!(select(PrefillRequest::Q8Lcpp, &q4_k_m_like(), true, |_| Ok(true)).is_err());
    for dtypes in [q8, q4_k_m_like()] {
        let selection = select(PrefillRequest::Serial, &dtypes, true, |_| {
            panic!("serial needs no scratch")
        })
        .unwrap();
        assert_eq!(selection.mode, PrefillMode::Serial);
    }
}

#[test]
fn general_prefill_info_matches_chunked_commands_and_pricing() {
    let mode = PrefillMode::General { chunk: 256 };
    for (tokens, chunk, commands) in [
        (0usize, 0usize, 0usize),
        (1, 1, 1),
        (2, 2, 1),
        (255, 255, 1),
        (256, 256, 1),
        (257, 256, 2),
        (4408, 256, 18),
    ] {
        let info = mode.info(tokens);
        assert_eq!(info.chunk_tokens, chunk, "{tokens}");
        assert_eq!(info.commands, commands, "{tokens}");
        assert_eq!(
            info.mode,
            if chunk > 1 {
                "general_matmat_batch"
            } else {
                "serial_single_token"
            }
        );
        assert_eq!(
            info.temporary_activation_bytes,
            if chunk > 1 {
                ACTIVATION_BYTES_PER_ROW * chunk as u64
            } else {
                0
            }
        );
    }
    assert_eq!(PrefillMode::General { chunk: 64 }.info(4408).commands, 69);
    assert!(PrefillMode::General { chunk: 256 }.is_general());
    assert!(!PrefillMode::BatchQ8.is_general());
    assert!(!PrefillMode::Serial.is_general());
}

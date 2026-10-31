use super::*;

#[test]
fn cpu_kda_half_live_non_tensor_and_default_off_footprints_are_unchanged() {
    let c = config();
    let price = memory::apple_16k_price;
    for enabled in [false, true] {
        let _scope = memory::KdaHalfInputsScope::new(enabled);
        for tensor_tier in [false, true] {
            let mut options = memory::ScratchOptions::for_tensor_tier(false, tensor_tier);
            options.tensor_gate_up = false;
            assert_eq!(options.kda_half_inputs, enabled && tensor_tier);
            for rows in [0, 512, 1024, 2048] {
                let live =
                    Glm5NextMemoryLedger::new_with_options(&c, 0, 4096, rows, options, &price)
                        .unwrap();
                let base = Glm5NextMemoryLedger::new_with_options(
                    &c,
                    0,
                    4096,
                    rows,
                    memory::ScratchOptions::default(),
                    &price,
                )
                .unwrap();
                assert_eq!(
                    live.peak_bytes() - base.peak_bytes(),
                    if enabled && tensor_tier && rows >= 1024 {
                        8192 * rows
                    } else {
                        0
                    }
                );
            }
        }
    }
}

#[test]
fn cpu_kda_half_ledger_prices_one_owned_buffer_at_admitted_rows() {
    let c = config();
    assert_eq!(c.block_count(MixerKind::Kda), 34);
    // Deliberately non-linear pricing proves one physical allocation is added.
    let price = |bytes| memory::apple_16k_price(bytes).and_then(|n| n.checked_add(16384));
    for generic_hc in [false, true] {
        for rows in [0, 512, 1023, 1024, 1025, 2048] {
            let options = |selected| memory::ScratchOptions {
                generic_hc,
                kda_half_inputs: selected,
                tensor_gate_up: false,
            };
            let base = Glm5NextMemoryLedger::new_with_options(
                &c,
                1234,
                4096,
                rows,
                options(false),
                &price,
            )
            .unwrap();
            let on =
                Glm5NextMemoryLedger::new_with_options(&c, 1234, 4096, rows, options(true), &price)
                    .unwrap();
            let specs = half_specs(&c, rows, true);
            if rows > 0 {
                let owned = memory::packed_scratch_specs_with_options(&c, rows, options(true));
                let names: std::collections::HashSet<_> =
                    owned.iter().map(|spec| spec.name).collect();
                assert_eq!(owned.len(), names.len());
                assert_eq!(names.contains("normalized_half"), rows >= 1024);
                assert_eq!(names.contains("hc_normalized"), generic_hc);
                let packed_price = on
                    .terms()
                    .into_iter()
                    .find(|(name, _)| *name == "packed_scratch")
                    .unwrap()
                    .1;
                assert_eq!(memory::priced(&owned, &price).unwrap(), packed_price);
            }
            let extra = memory::priced(&specs, &price).unwrap();
            assert_eq!(specs.len(), usize::from(rows >= 1024));
            if let Some(spec) = specs.first() {
                assert_eq!(spec.name, "normalized_half");
                assert_eq!(spec.dtype, BufferType::F16);
                assert_eq!(spec.shape, [4096, rows]);
                assert_eq!(spec.bytes(), Some(8192 * rows));
            }
            assert_eq!(on.peak_bytes() - base.peak_bytes(), extra);
            assert_eq!(
                on.session_buffer_bytes() - base.session_buffer_bytes(),
                extra
            );
            for ((name, a), (other, b)) in on.terms().into_iter().zip(base.terms()) {
                assert_eq!(name, other);
                assert_eq!(
                    a - b,
                    if ["packed_scratch", "session_buffers"].contains(&name) {
                        extra
                    } else {
                        0
                    }
                );
            }
            assert_eq!(on.scratch_options(), options(true));
        }
    }
    assert_eq!(half_specs(&c, 1024, true)[0].bytes(), Some(8 << 20));
    assert_eq!(half_specs(&c, 2048, true)[0].bytes(), Some(16 << 20));
    assert!(half_specs(&c, u64::MAX, true)[0].bytes().is_none());
    assert!(memory::priced(&half_specs(&c, 1024, true), &|_| None).is_err());
}

#[test]
fn cpu_kda_half_device_free_entrypoints_and_capacity_advice_never_underprice() {
    let c = config();
    let price = memory::apple_16k_price;
    let _on = memory::KdaHalfInputsScope::new(true);
    for generic_hc in [false, true] {
        for rows in [1024, 1025, 2048] {
            let options = memory::ScratchOptions::device_free(generic_hc);
            assert!(options.kda_half_inputs);
            let expected =
                Glm5NextMemoryLedger::new_with_options(&c, 0, 4096, rows, options, &price).unwrap();
            assert_eq!(
                Glm5NextMemoryLedger::new_with_generic_hc(&c, 0, 4096, rows, generic_hc, &price)
                    .unwrap(),
                expected
            );
            if !generic_hc {
                assert_eq!(
                    Glm5NextMemoryLedger::new(&c, 0, 4096, rows, &price).unwrap(),
                    expected
                );
            }
            assert_eq!(
                Glm5NextMemoryLedger::max_capacity_with_generic_hc(
                    &c,
                    0,
                    rows,
                    expected.peak_bytes(),
                    generic_hc,
                    &price
                )
                .unwrap(),
                Some(4096)
            );
            let minimum =
                Glm5NextMemoryLedger::new_with_options(&c, 0, rows, rows, options, &price).unwrap();
            assert_eq!(
                Glm5NextMemoryLedger::max_capacity_with_options(
                    &c,
                    0,
                    rows,
                    minimum.peak_bytes() - 1,
                    options,
                    &price
                )
                .unwrap(),
                None
            );
        }
    }
}

#[test]
fn cpu_gate_up_shared_half_owners_price_once_without_kda_metadata() {
    let mut c = config();
    for has_kda in [true, false] {
        if !has_kda {
            for block in &mut c.blocks {
                block.mixer = MixerKind::Mla;
            }
        }
        for kda_half_inputs in [false, true] {
            for tensor_gate_up in [false, true] {
                let options = memory::ScratchOptions {
                    generic_hc: false,
                    kda_half_inputs,
                    tensor_gate_up,
                };
                for rows in [0, 512, 1024, 1025, 2048] {
                    let specs = memory::packed_normalized_half_specs(&c, rows, options);
                    let owns = rows >= 1024 && ((kda_half_inputs && has_kda) || tensor_gate_up);
                    assert_eq!(specs.len(), usize::from(owns));
                    if owns {
                        assert_eq!(specs[0].name, "normalized_half");
                        assert_eq!(specs[0].bytes(), Some(8192 * rows));
                    }
                    let price =
                        |bytes| memory::apple_16k_price(bytes).and_then(|n| n.checked_add(16384));
                    let packed_base =
                        memory::packed_scratch_specs_with_options(&c, rows, Default::default());
                    let packed_on = memory::packed_scratch_specs_with_options(&c, rows, options);
                    assert_eq!(
                        memory::priced(&packed_on, &price).unwrap()
                            - memory::priced(&packed_base, &price).unwrap(),
                        memory::priced(&specs, &price).unwrap()
                    );
                    // Full-session validation pins the release KDA/MLA layout;
                    // no-KDA metadata is a buffer-list/allocator contract only.
                    if !has_kda {
                        continue;
                    }
                    let base = Glm5NextMemoryLedger::new_with_options(
                        &c,
                        0,
                        4096,
                        rows,
                        Default::default(),
                        &price,
                    )
                    .unwrap();
                    let on =
                        Glm5NextMemoryLedger::new_with_options(&c, 0, 4096, rows, options, &price)
                            .unwrap();
                    assert_eq!(
                        on.peak_bytes() - base.peak_bytes(),
                        memory::priced(&specs, &price).unwrap()
                    );
                    assert_eq!(on.scratch_options(), options);
                }
            }
        }
    }
    assert!(!memory::ScratchOptions::for_tensor_tier(false, false).tensor_gate_up);
    assert_eq!(
        memory::ScratchOptions::device_free(false).tensor_gate_up,
        crate::env_flag::read_default_on("QWEN_GLM53_TENSOR_GATE_UP")
    );
}

fn half_specs(c: &Glm5NextConfig, rows: u64, selected: bool) -> Vec<memory::BufferSpec> {
    memory::packed_normalized_half_specs(
        c,
        rows,
        memory::ScratchOptions {
            kda_half_inputs: selected,
            ..Default::default()
        },
    )
}

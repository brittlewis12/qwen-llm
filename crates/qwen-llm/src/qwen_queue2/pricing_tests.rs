use super::*;
use crate::metal::{MetalBufferSizeAndAlign, price_shared_buffer_upper};
use crate::model::QWEN3_27B;

const PAGE: u64 = 16 * 1024;

// Independent inventory of MetalSession::fresh for Dense27B, F16 KV, capacity
// 135 (the retained Saluki qualification). Labels identify constructor fields;
// this is not a synthetic list of conveniently page-aligned allocations.
fn dense27b_inventory() -> Vec<(&'static str, u64, u64)> {
    let mut inventory = vec![
        ("gdn_conv", 122_880, 48),
        ("gdn_state", 3_145_728, 48),
        ("kv_k/kv_v", 276_480, 32),
    ];
    for (name, elements) in [
        ("x", 5120),
        ("h", 5120),
        ("ffn_gate", 17408),
        ("ffn_up", 17408),
        ("ffn_inner", 17408),
        ("ffn_out", 5120),
        ("gdn_qkv", 10240),
        ("gdn_qkv_conv", 10240),
        ("gdn_z", 6144),
        ("gdn_b", 48),
        ("gdn_beta", 48),
        ("gdn_a", 48),
        ("gdn_alpha", 48),
        ("gdn_q_norm", 2048),
        ("gdn_k_norm", 2048),
        ("gdn_out", 6144),
        ("gdn_normed", 6144),
        ("mixer_out", 5120),
        ("attn_q_full", 12288),
        ("attn_q", 6144),
        ("attn_gate", 6144),
        ("attn_q_normed", 6144),
        ("attn_k_now", 1024),
        ("attn_v_now", 1024),
        ("attn_k_normed", 1024),
        ("attn_o", 6144),
        ("attn_v4_o_partial", 6_291_456),
        ("attn_v4_ml_partial", 49_152),
        ("logits", 248_320),
        ("moe_router_probs", 1),
        ("moe_topk_idx", 1),
        ("moe_topk_weight", 1),
        ("moe_shared_gate", 1),
        ("moe_inner", 1),
        ("moe_expert_out", 1),
    ] {
        inventory.push((name, elements * 4, 1));
    }
    inventory.push(("argmax_tok/ids_buf", 4, 2));
    inventory
}

#[test]
fn session_pricing_dense27b_inventory_rounds_each_buffer() {
    let inventory = dense27b_inventory();
    assert_eq!(
        inventory.iter().map(|(_, _, count)| count).sum::<u64>(),
        165
    );
    let logical = inventory
        .iter()
        .map(|(_, bytes, count)| bytes * count)
        .sum::<u64>();
    assert_eq!(logical, 192_719_648);

    let mut quoted_bytes = Vec::new();
    let upper = session_upper_bytes_with_pricer(&QWEN3_27B, 135, 48, 16, GgmlType::F16, |bytes| {
        quoted_bytes.push(bytes);
        price_shared_buffer_upper(
            bytes,
            MetalBufferSizeAndAlign {
                size: bytes,
                alignment: 256,
            },
            PAGE,
            u64::MAX,
        )
        .map(|priced| priced.priced_upper_bytes)
    })
    .unwrap();
    assert_eq!(
        quoted_bytes,
        inventory
            .iter()
            .map(|(_, bytes, _)| *bytes)
            .collect::<Vec<_>>()
    );
    let independently_rounded = inventory
        .iter()
        .map(|(_, bytes, count)| bytes.div_ceil(PAGE) * PAGE * count)
        .sum::<u64>();
    assert_eq!(upper, independently_rounded);
    assert_eq!(upper, 193_593_344);
    assert!(
        upper >= 193_314_816,
        "must cover the retained allocation delta"
    );
    let rounded_after_count = inventory
        .iter()
        .map(|(_, bytes, count)| (bytes * count).div_ceil(PAGE) * PAGE)
        .sum::<u64>();
    assert!(upper > rounded_after_count);
    assert!(upper > logical.div_ceil(PAGE) * PAGE);

    // Unit prices count actual allocations, rather than the 39 pricing calls.
    let allocation_count =
        session_upper_bytes_with_pricer(&QWEN3_27B, 135, 48, 16, GgmlType::F16, |_| Ok(1)).unwrap();
    assert_eq!(allocation_count, 165);
    assert_eq!(QWEN_QUEUE2_DYNAMIC_RESERVE_BYTES, 2 * 1024 * 1024 * 1024);
}

#[test]
fn session_pricing_skips_absent_layer_allocations() {
    let mut quoted_bytes = Vec::new();
    let count = session_upper_bytes_with_pricer(&QWEN3_27B, 135, 0, 0, GgmlType::F16, |bytes| {
        quoted_bytes.push(bytes);
        Ok(1)
    })
    .unwrap();
    assert_eq!(count, 37); // 35 F32 scratch buffers plus two I32 buffers.
    assert_eq!(
        quoted_bytes,
        dense27b_inventory()
            .iter()
            .skip(3)
            .map(|(_, bytes, _)| *bytes)
            .collect::<Vec<_>>()
    );
}

#[test]
fn session_pricing_propagates_shared_buffer_errors() {
    let error = session_upper_bytes_with_pricer(&QWEN3_27B, 135, 48, 16, GgmlType::F16, |bytes| {
        price_shared_buffer_upper(
            bytes,
            MetalBufferSizeAndAlign {
                size: bytes,
                alignment: 256,
            },
            PAGE,
            1024,
        )
        .map(|priced| priced.priced_upper_bytes)
    })
    .unwrap_err();
    assert!(matches!(
        error,
        QwenQueue2Error::SharedBufferPricing {
            logical_bytes: 122_880,
            source: SharedBufferPricingError::ExceedsDeviceMaximum {
                logical_bytes: 122_880,
                max_buffer_length: 1024,
            },
        }
    ));

    // A later quote failure must stop the inventory walk and retain the typed
    // pricing error; no partial total or MetalError conversion is acceptable.
    let mut calls = 0;
    let error = session_upper_bytes_with_pricer(&QWEN3_27B, 135, 48, 16, GgmlType::F16, |_| {
        calls += 1;
        if calls == 4 {
            Err(SharedBufferPricingError::Overflow)
        } else {
            Ok(1)
        }
    })
    .unwrap_err();
    assert_eq!(calls, 4);
    assert!(matches!(
        error,
        QwenQueue2Error::SharedBufferPricing {
            logical_bytes: 20_480,
            source: SharedBufferPricingError::Overflow,
        }
    ));
}

#[test]
fn session_pricing_checks_repeat_and_total_overflow() {
    for (gdn_layers, expected_calls, expected_error) in [
        (2, 1, "priced session allocations overflow"),
        (1, 2, "session allocation total overflow"),
    ] {
        let mut calls = 0;
        let error =
            session_upper_bytes_with_pricer(&QWEN3_27B, 135, gdn_layers, 0, GgmlType::F16, |_| {
                calls += 1;
                Ok(u64::MAX)
            })
            .unwrap_err();
        assert_eq!(calls, expected_calls);
        assert!(
            matches!(error, QwenQueue2Error::Validation(ref message) if message == expected_error)
        );
    }
}

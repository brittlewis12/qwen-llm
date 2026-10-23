use super::*;

#[test]
fn half_staging_default_geometry_is_202_mib() {
    let c = Config::default();
    let attention = attention_half_scratch_elements(&c, 2048).unwrap();
    let [input, inner] = ffn_half_scratch_elements(&c, 2048).unwrap();
    let bytes = [attention, input, inner].map(|n| HalfScratch::logical_bytes(n).unwrap());
    assert_eq!(bytes, [128u64, 20, 54].map(|mib| mib * 1024 * 1024));
    assert_eq!(bytes.into_iter().sum::<u64>(), 202 * 1024 * 1024);
    assert_eq!(attention_half_scratch_elements(&c, 0).unwrap(), 0);
    assert_eq!(ffn_half_scratch_elements(&c, 0).unwrap(), [0, 0]);
    assert_eq!(
        ffn_half_scratch_elements(&Config { topk: 0, ..c }, 3).unwrap(),
        [3 * 5120, 3 * 2304]
    );
}

#[test]
fn half_staging_geometry_checks_products_and_bytes() {
    let mut c = Config {
        hidden: 3,
        query_rank: 5,
        heads: 2,
        head_dim: 7,
        groups: 3,
        output_rank: 11,
        intermediate: 13,
        topk: 2,
        ..Config::default()
    };
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 33 * 17);
    assert_eq!(
        ffn_half_scratch_elements(&c, 17).unwrap(),
        [3 * 17, 26 * 17]
    );
    c.query_rank = 37;
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 37 * 17);
    c.hidden = 41;
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 41 * 17);
    c.heads = 7;
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 49 * 17);
    assert!(attention_half_scratch_elements(&c, usize::MAX).is_err());
    for invalid in [
        Config {
            heads: usize::MAX,
            head_dim: 2,
            ..c.clone()
        },
        Config {
            groups: usize::MAX,
            output_rank: 2,
            ..c.clone()
        },
        Config {
            query_rank: usize::MAX,
            ..c.clone()
        },
    ] {
        assert!(attention_half_scratch_elements(&invalid, 2).is_err());
    }
    for invalid in [
        Config {
            hidden: usize::MAX,
            ..c.clone()
        },
        Config {
            intermediate: usize::MAX,
            ..c.clone()
        },
        Config {
            topk: usize::MAX,
            ..c
        },
    ] {
        assert!(ffn_half_scratch_elements(&invalid, 2).is_err());
    }
    assert_eq!(HalfScratch::logical_bytes(0).unwrap(), 0);
    assert!(HalfScratch::logical_bytes(usize::MAX).is_err());
    assert!(half_scratch_extent(&[usize::MAX / 2 + 1, 1]).is_err());
}

#[test]
#[ignore = "self-leased tiny Metal half allocation ledger; MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1; no model"]
fn half_staging_allocation_ledger_counts_unused_override_buffers() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert!(!crate::env_flag::read_default_off(
        "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
    ));
    let _lease =
        metal::acquire_metal_benchmark_lease().expect("production lease and ordinary wired gate");
    objc2::rc::autoreleasepool(|_| {
        const EXPLICIT_BUFFER_BOUND: u64 = 3 * 16384;
        let ctx = MetalContext::new().unwrap();
        let c = Config {
            hidden: 33,
            query_rank: 41,
            heads: 3,
            head_dim: 7,
            groups: 2,
            output_rank: 23,
            intermediate: 17,
            topk: 3,
            ..Config::default()
        };
        let selected = HalfScratch::selected(&ctx);
        if !selected {
            eprintln!(
                "half staging selected=false: unsupported tensor tier (including M4); \
                 checking zero pricing/no allocation only, not GPU half-buffer coverage"
            );
        }
        assert!(HalfScratch::new(&ctx, 0).unwrap().is_none());
        assert_eq!(half_scratch_budget(&ctx, &c, 0).unwrap(), 0);
        for rows in [1, 33, 129] {
            let [input, inner] = ffn_half_scratch_elements(&c, rows).unwrap();
            let elements = [
                attention_half_scratch_elements(&c, rows).unwrap(),
                input,
                inner,
            ];
            let budget = half_scratch_budget(&ctx, &c, rows).unwrap();
            let expected: u64 = if selected {
                elements
                    .into_iter()
                    .map(|n| {
                        ctx.price_shared_buffer_upper(HalfScratch::logical_bytes(n).unwrap())
                            .unwrap()
                            .priced_upper_bytes
                    })
                    .sum()
            } else {
                0
            };
            assert_eq!(budget, expected);
            assert!(
                budget <= EXPLICIT_BUFFER_BOUND,
                "explicit half buffers {budget}"
            );
            with_f32_linears(|| {
                assert!(F32_LINEAR_ROWS.with(|flag| flag.get()));
                assert_eq!(half_scratch_budget(&ctx, &c, rows).unwrap(), budget);
                let before = ctx.memory_signals().current_allocated_bytes;
                let scratch = elements.map(|n| HalfScratch::new(&ctx, n).unwrap());
                let mut buffer_bytes = 0u64;
                for (buffer, n) in scratch.iter().zip(elements) {
                    assert_eq!(buffer.is_some(), selected);
                    if let Some(buffer) = buffer {
                        assert_eq!(buffer.tensor.dtype, GgmlType::F16);
                        assert_eq!(buffer.tensor.n_elements(), n as u64);
                        assert_eq!(
                            buffer.tensor.n_bytes(),
                            HalfScratch::logical_bytes(n).unwrap()
                        );
                        buffer_bytes += buffer.tensor.buffer.length() as u64;
                    }
                }
                assert!(
                    buffer_bytes <= budget,
                    "{buffer_bytes} buffer bytes exceed {budget}"
                );
                let allocated = ctx
                    .memory_signals()
                    .current_allocated_bytes
                    .saturating_sub(before);
                assert!(
                    allocated <= budget,
                    "{allocated} allocated exceeds {budget} priced bytes"
                );
                eprintln!(
                    "half staging rows={rows} selected={selected} priced={budget} \
                     buffer_bytes={buffer_bytes} device_delta={allocated} \
                     explicit_buffer_bound={EXPLICIT_BUFFER_BOUND}; context/driver separate"
                );
                drop(scratch);
            });
        }
    });
}

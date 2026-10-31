use super::*;
use objc2::rc::{Weak, autoreleasepool};
use std::collections::BTreeMap;

#[test]
#[ignore = "self-leased bounded M5 cache allocation/lifetime witness; <=1 GiB combined toy quote; debug validation required"]
fn bounded_gpu_cache_backings_and_group_views_match_independent_quote() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert!(!crate::env_flag::read_default_off(
        "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
    ));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.device_facts().name.contains("M5"),
        "M5 witness required"
    );
    let price = |bytes| {
        ctx.price_shared_buffer_upper(bytes)
            .map(|p| p.priced_upper_bytes)
            .map_err(|e| e.to_string())
    };
    let specs = [
        LinearSpec {
            name: "output_a".into(),
            input: 256,
            output: 256,
        },
        LinearSpec {
            name: "head".into(),
            input: 256,
            output: 3,
        },
    ];
    let plan = BindingMemoryPlan::new(specs, |_| Ok(GgmlType::Q5_K), true, &price).unwrap();
    let controls = 2 * price(8).unwrap();
    let caches = price(256 * 256 * 2).unwrap() + price(256 * 3 * 2).unwrap();
    let sources = price(256 * 176).unwrap() + price(3 * 176).unwrap();
    let cpu = 128 * 1024 * 1024;
    assert_eq!(plan.control_metal_bytes(), controls);
    assert_eq!(plan.cache_metal_bytes(), caches);
    assert!(sources + controls + caches + cpu <= 1 << 30);
    let a = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        sources + controls + caches,
        cpu,
        0,
        ctx.memory_signals(),
        true,
    );
    assert!(a.is_admitted(), "{a:?}");
    let baseline = ctx.current_allocated_size();
    let (weights, groups, head, weak) = autoreleasepool(|_| {
        let mut weights = BTreeMap::new();
        weights.insert(
            "output_a".into(),
            MetalTensor::zeros_dtype_unstaged(&ctx, vec![256, 256], GgmlType::Q5_K).unwrap(),
        );
        weights.insert(
            "head".into(),
            MetalTensor::zeros_dtype_unstaged(&ctx, vec![256, 3], GgmlType::Q5_K).unwrap(),
        );
        let (groups, head, weak) = plan
            .with_admission(cpu, 0, 0, ctx.memory_signals(), |plan| {
                let b = Binder {
                    ctx: &ctx,
                    weights: &weights,
                    plan,
                };
                let output = b.linear("output_a", 256, 256)?;
                let head = b.linear("head", 256, 3)?;
                let cache = output.half_weight.as_ref().unwrap();
                assert!(cache.is_writable());
                assert!(!std::ptr::eq(&*cache.buffer, &*output.tensor.buffer));
                assert!(!std::ptr::eq(
                    &*cache.buffer,
                    &*head.half_weight.as_ref().unwrap().buffer
                ));
                let weak = Weak::from_retained(&cache.buffer);
                let groups: Vec<_> = (0..8)
                    .map(|g| cache.view_bytes(g * 32 * 256 * 2, vec![256, 32]))
                    .collect();
                assert!(
                    groups
                        .iter()
                        .all(|view| std::ptr::eq(&*view.buffer, &*cache.buffer))
                );
                assert!(std::ptr::eq(
                    &*output.tensor.buffer,
                    &*weights["output_a"].buffer
                ));
                assert!(ctx.current_allocated_size() - baseline <= sources + controls + caches);
                let logical = 259 * 176 + 259 * 256 * 2 + 2 * 8;
                assert!(ctx.current_allocated_size() - baseline >= logical);
                Ok((groups, head.half_weight.unwrap(), weak))
            })
            .unwrap();
        (weights, groups, head, weak)
    });
    assert!(weak.load().is_some());
    assert!(ctx.current_allocated_size() - baseline <= sources + controls + caches);
    autoreleasepool(|_| {
        drop(groups);
        drop(head);
        drop(weights);
    });
    assert!(weak.load().is_none());
    assert!(ctx.current_allocated_size() <= baseline);
}

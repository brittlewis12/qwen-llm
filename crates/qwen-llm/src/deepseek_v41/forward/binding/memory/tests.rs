use super::*;
use std::cell::Cell;

fn spec(name: &str, input: usize, output: usize) -> LinearSpec {
    LinearSpec {
        name: name.into(),
        input,
        output,
    }
}

fn price(bytes: u64) -> Result<u64> {
    metal::price_shared_buffer_upper(
        bytes,
        metal::MetalBufferSizeAndAlign {
            size: bytes,
            alignment: 256,
        },
        16_384,
        1 << 30,
    )
    .map(|p| p.priced_upper_bytes)
    .map_err(|e| e.to_string())
}

fn plan(enabled: bool) -> BindingMemoryPlan {
    BindingMemoryPlan::new(
        [spec("a", 256, 33), spec("b", 256, 33), spec("c", 256, 33)],
        |s| {
            Ok(if s.name == "c" {
                GgmlType::Q8_0
            } else {
                GgmlType::Q5_K
            })
        },
        enabled,
        &price,
    )
    .unwrap()
}

fn signals() -> metal::MetalMemorySignals {
    metal::MetalMemorySignals {
        recommended_max_bytes: 1 << 30,
        current_allocated_bytes: 1024,
        process_limit_remaining_bytes: Some(1 << 30),
        host_budget: Some(metal::HostMemoryBudget {
            physical_bytes: 1 << 30,
            conservative_available_bytes: 1 << 30,
            external_gap_bytes: 2048,
            private_shadow_bytes: 4096,
        }),
    }
}

#[test]
fn cache_selection_is_frozen_and_each_buffer_is_priced_separately() {
    let on = plan(true);
    let off = plan(false);
    assert!(on.cache_enabled());
    assert!(!off.cache_enabled());
    assert_eq!(on.cache_count(), 2);
    assert_eq!(off.cache_count(), 0);
    assert_eq!(on.control_metal_bytes(), 3 * 16_384);
    assert_eq!(on.cache_metal_bytes(), 2 * 32_768);
    assert_eq!(off.cache_metal_bytes(), 0);
    assert!(on.cache_metal_bytes() > price(2 * 256 * 33 * 2).unwrap());
    assert!(on.cache_for("a", 256, 33, GgmlType::Q5_K).unwrap());
    assert!(!off.cache_for("a", 256, 33, GgmlType::Q5_K).unwrap());
    assert!(!on.cache_for("c", 256, 33, GgmlType::Q8_0).unwrap());
    for (name, input, output, dtype) in [
        ("a", 256, 32, GgmlType::Q5_K),
        ("a", 256, 33, GgmlType::Q8_0),
        ("missing", 256, 33, GgmlType::Q5_K),
    ] {
        assert!(on.cache_for(name, input, output, dtype).is_err());
    }
}

#[test]
fn cache_plan_rejects_overflow_device_limit_unsupported_and_duplicate_inventory() {
    for (input, output) in [
        (usize::MAX, 2),
        (1 << 30, 1 << 30),
        (0, 256),
        (256, 0),
        (256, (1 << 30) / 512 + 1),
    ] {
        assert!(
            BindingMemoryPlan::new(
                [spec("a", input, output)],
                |_| Ok(GgmlType::Q5_K),
                true,
                &price
            )
            .is_err()
        );
    }
    assert!(
        BindingMemoryPlan::new([spec("a", 256, 256)], |_| Ok(GgmlType::I32), true, &price).is_err()
    );
    assert!(
        BindingMemoryPlan::new(
            [spec("a", 256, 256); 1],
            |_| Ok(GgmlType::Q5_K),
            true,
            &|_| Err("pricing refused".into())
        )
        .is_err()
    );
    assert!(
        BindingMemoryPlan::new(
            [spec("a", 256, 256), spec("a", 256, 256)],
            |_| Ok(GgmlType::F32),
            true,
            &price
        )
        .is_err()
    );
    assert!(
        BindingMemoryPlan::new(
            [spec("a", 256, 256), spec("b", 256, 256)],
            |_| Ok(GgmlType::Q5_K),
            true,
            &|_| Ok(u64::MAX)
        )
        .is_err()
    );
}

#[test]
fn binding_aggregate_refuses_before_factory_at_each_one_byte_boundary() {
    let p = plan(true);
    let metal = p.cache_metal_bytes() + p.control_metal_bytes() + 1024;
    let cpu = 8192 + 16384;
    let mut exact = signals();
    let live = exact.effective_device_usage().unwrap();
    exact.recommended_max_bytes = live + metal;
    exact.process_limit_remaining_bytes = Some(metal + cpu);
    let host = exact.host_budget.as_mut().unwrap();
    host.physical_bytes = live + host.private_shadow_bytes + metal + cpu;
    host.conservative_available_bytes = host.private_shadow_bytes + cpu;
    let calls = Cell::new(0);
    let run = |s| {
        p.clone().with_admission(8192, 1024, 16384, s, |frozen| {
            calls.set(calls.get() + 1);
            assert_eq!(frozen, p);
            Ok(())
        })
    };
    run(exact).unwrap();
    assert_eq!(calls.get(), 1);
    for field in 0..4 {
        let mut short = exact;
        match field {
            0 => short.recommended_max_bytes -= 1,
            1 => *short.process_limit_remaining_bytes.as_mut().unwrap() -= 1,
            2 => short.host_budget.as_mut().unwrap().physical_bytes -= 1,
            3 => {
                short
                    .host_budget
                    .as_mut()
                    .unwrap()
                    .conservative_available_bytes -= 1
            }
            _ => unreachable!(),
        }
        assert!(run(short).is_err(), "field {field}");
        assert_eq!(calls.get(), 1);
    }
    assert!(
        p.clone()
            .with_admission::<()>(u64::MAX, 0, 1, signals(), |_| panic!(
                "overflow reached factory"
            ))
            .is_err()
    );
    assert!(
        p.with_admission::<()>(0, u64::MAX, 0, signals(), |_| panic!(
            "overflow reached factory"
        ))
        .is_err()
    );
}

#[test]
fn direct_binding_admits_only_construction_and_runtime_counts_live_cache_once() {
    let p = plan(true);
    let binding = p.cache_metal_bytes() + p.control_metal_bytes();
    let mut s = signals();
    let live = s.effective_device_usage().unwrap();
    s.recommended_max_bytes = live + binding;
    p.clone().with_admission(128, 0, 0, s, |_| Ok(())).unwrap();
    assert!(
        p.clone()
            .with_admission::<()>(128, 1, 0, s, |_| panic!("known future workspace omitted"))
            .is_err()
    );
    let runtime = 4096;
    s.current_allocated_bytes += binding;
    s.recommended_max_bytes += runtime;
    assert!(
        metal::evaluate_metal_memory_admission_with_cpu_bytes(runtime, 128, 0, s, true)
            .is_admitted()
    );
    assert!(
        !metal::evaluate_metal_memory_admission_with_cpu_bytes(runtime + binding, 128, 0, s, true)
            .is_admitted()
    );
}

#[test]
fn grouped_output_has_one_cache_backing_for_eight_views() {
    let c = Config::default();
    assert_eq!(c.groups, 8);
    let schedule = Schedule {
        layers: 1,
        kv_source_layers: vec![],
        index_source_layers: vec![],
        compression_ratios: vec![0],
        candidate_source_layer: None,
    };
    let mut hash = engram::HashConfig::default();
    hash.layers.clear();
    let p = BindingMemoryPlan::new(
        linear_inventory(&c, &schedule, &hash),
        |s| {
            Ok(if s.name == "layers.0.attn.wo_a.weight" {
                GgmlType::Q5_K
            } else {
                GgmlType::F32
            })
        },
        true,
        &price,
    )
    .unwrap();
    assert_eq!(p.cache_count(), 1);
    assert_eq!(
        p.cache_metal_bytes(),
        (c.heads * c.head_dim / c.groups * c.groups * c.output_rank * 2) as u64
    );
}

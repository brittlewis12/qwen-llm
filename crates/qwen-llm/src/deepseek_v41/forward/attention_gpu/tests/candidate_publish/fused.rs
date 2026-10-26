use super::*;

const FUSED_PROTOCOL: &str = "v41-candidate-publish-fused-v1";

#[test]
fn mask_fused_cpu_policy_boundaries_and_forced_precedence() {
    let mut cases = 0;
    for domain in [0, 1, 2048, 2049, 131072, 131073, usize::MAX] {
        for topk in [0, 1, 2047, 2048, 2049, 4096, 131072, usize::MAX] {
            for concurrent in [false, true] {
                for capacity in [0, 16383, 16384, 32768] {
                    let expected = if !concurrent
                        && domain.min(topk) >= 2048
                        && domain > domain.min(topk)
                        && domain <= 131072
                        && capacity >= 16384
                    {
                        Arm::Fused
                    } else {
                        Arm::Incumbent
                    };
                    assert_eq!(
                        candidate_publication_arm(domain, topk, concurrent, capacity, None),
                        expected
                    );
                    for forced in [Arm::Incumbent, Arm::ClearScatter, Arm::Fused] {
                        assert_eq!(
                            candidate_publication_arm(
                                domain,
                                topk,
                                concurrent,
                                capacity,
                                Some(forced)
                            ),
                            forced
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 448);
    assert_eq!(
        candidate_publication_arm(2049, 2048, false, 16384, None),
        Arm::Fused
    );
    assert_eq!(
        candidate_publication_arm(2049, usize::MAX, false, 16384, None),
        Arm::Incumbent
    );
}

#[test]
fn mask_fused_cpu_word_rounding_width_capacity_and_budget() {
    for (domain, words) in [
        (0, 0),
        (1, 1),
        (31, 1),
        (32, 1),
        (33, 2),
        (255, 8),
        (256, 8),
        (257, 9),
        (2048, 64),
        (2049, 65),
        (131071, 4096),
        (131072, 4096),
    ] {
        assert_eq!(candidate_fused_words(domain), Some(words));
    }
    for domain in [131073, u32::MAX as usize, usize::MAX] {
        assert_eq!(candidate_fused_words(domain), None);
    }
    assert_eq!(MASK_FUSED_TG_BYTES, 16384);
    for max_threads in [0, 1, 7, 32, 128, 255, 256, 512, usize::MAX] {
        assert_eq!(
            candidate_fused_width(max_threads, 16384, 16384),
            max_threads.min(256)
        );
        assert_eq!(candidate_fused_width(max_threads, 16384, 16383), 0);
        assert_eq!(candidate_fused_width(max_threads, 16385, 16384), 0);
        assert_eq!(candidate_fused_width(max_threads, usize::MAX, 32768), 0);
    }
    // On-chip scratch does not enter the unchanged three-buffer/CPU inventory.
    assert_eq!(buffer_words(131072, 2048), [2097, 131105, 131137]);
    assert_eq!(cpu_price(131072, 2048), 9511520);
    assert!(
        buffer_words(131072, 2048).iter().sum::<usize>() as u64 * 4
            + 3 * 65536
            + cpu_price(131072, 2048)
            < LIMIT
    );
}

fn cooperative_mask(domain: usize, input: &[u32], width: usize) -> Vec<u32> {
    assert!(width > 0);
    let words = candidate_fused_words(domain).unwrap();
    let mut bits = vec![POISON; 4096];
    for tid in 0..width {
        for j in (tid..words).step_by(width) {
            bits[j] = 0;
        }
    }
    for tid in 0..width {
        for j in (tid..input.len()).step_by(width) {
            let id = input[j] as usize;
            if id < domain {
                bits[id >> 5] |= 1 << (id & 31);
            }
        }
    }
    let mut dense = vec![POISON; domain + TAIL];
    for tid in 0..width {
        for j in (tid..domain).step_by(width) {
            dense[j] = (bits[j >> 5] >> (j & 31)) & 1;
        }
    }
    assert!(bits[words..].iter().all(|&v| v == POISON));
    assert!(dense[domain..].iter().all(|&v| v == POISON));
    if !domain.is_multiple_of(32) {
        assert_eq!(bits[words - 1] >> (domain % 32), 0);
    }
    dense.truncate(domain);
    dense
}

#[test]
fn mask_fused_cpu_strided_bitset_membership_grid() {
    let mut cases = 0;
    for domain in DOMAINS {
        for count in COUNTS {
            for pattern in PATTERNS {
                let input = input_ids(domain, count, pattern);
                let expected = reference(domain, &input);
                for width in [1, 7, 32, 128, 256] {
                    assert_eq!(cooperative_mask(domain, &input, width), expected);
                }
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 144);
    for domain in [31, 32, 33, 131071, 131072] {
        let input = [
            0,
            31,
            32,
            domain as u32 - 1,
            domain as u32 - 1,
            domain as u32,
            u32::MAX,
        ];
        for width in [1, 7, 128, 256] {
            assert_eq!(
                cooperative_mask(domain, &input, width),
                reference(domain, &input)
            );
        }
    }
}

#[test]
fn mask_fused_cpu_single_dispatch_dense_source_contract() {
    let source = include_str!("../../../../../../../../kernels/deepseek_v41_attention.metal");
    let kernel = source
        .split("kernel void kernel_deepseek_v41_attention_keep_fused(")
        .nth(1)
        .unwrap()
        .split("// Experimental only:")
        .next()
        .unwrap();
    assert!(kernel.contains("threadgroup atomic_uint bits[4096]"));
    assert!(kernel.contains("[[threads_per_threadgroup]]"));
    assert_eq!(
        kernel
            .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
            .count(),
        2
    );
    assert!(!kernel.contains("return"));
    assert!(!kernel.contains("device atomic_uint"));
    assert!(kernel.find("a.x <= 131072u").unwrap() < kernel.find("a.x + 31u").unwrap());
    assert!(kernel.contains("if (id < a.x)"));
    assert!(kernel.contains("atomic_fetch_or_explicit(bits + (id >> 5), 1u << (id & 31u)"));
    assert!(kernel.contains("mask[j] = (atomic_load_explicit"));
    assert_eq!(Arm::Fused.dispatches(), Arm::Incumbent.dispatches());
    assert_eq!(Arm::ClearScatter.dispatches(), 2);
    assert_eq!(Arm::Fused.name(), "fused_onchip");
    assert_ne!(FUSED_PROTOCOL, PROTOCOL);
    let host = include_str!("../../../attention_gpu.rs");
    let publish = host
        .split("    fn publish_candidates(")
        .nth(1)
        .unwrap()
        .split("    fn sort(")
        .next()
        .unwrap();
    assert_eq!(publish.matches("enc.set_bytes(").count(), 1);
    assert_eq!(publish.matches("enc.set_tensor(").count(), 2);
    assert_eq!(publish.matches("enc.set_pipeline(").count(), 1);
    assert!(!publish.contains("std::env"));
    assert!(!publish.contains("MetalTensor::"));
}

#[test]
#[ignore = "parent-only self-leased current-device 144-case incumbent/clear-scatter/fused correctness; MTL_DEBUG_LAYER=1; <64 MiB"]
fn candidate_publish_fused_current_device_correctness() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("production lease and wired guard");
    let ctx = MetalContext::new().expect("explicit correctness requires Metal, never skip");
    assert!(!metal::dispatch_census_is_active());
    eprintln!(
        "{}",
        json!({"protocol": FUSED_PROTOCOL, "event": "context", "device": ctx.describe(),
        "arms": [Arm::Incumbent.name(), Arm::ClearScatter.name(), Arm::Fused.name()],
        "cases": 144, "threadgroup_bytes": MASK_FUSED_TG_BYTES,
        "msl_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../../../../../../../kernels/deepseek_v41_attention.metal")))})
    );
    let p = Pipelines::new(&ctx);
    correctness_grid(&ctx, &p);
    eprintln!(
        "{}",
        json!({"protocol": FUSED_PROTOCOL, "event": "correctness_complete", "cases": 144,
        "arms": [Arm::Incumbent.name(), Arm::ClearScatter.name(), Arm::Fused.name()]})
    );
}

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
fn q8_byte_plans_preserve_causal_coordinates_without_f16_reinterpretation() {
    for capacity in [1, 32, 33, 256, 257, 7168] {
        let plan =
            K2ShortContextPlan::with_storage(config(), 8191, capacity, K2KvStorage::Q8_0).unwrap();
        assert_eq!(plan.storage(), K2KvStorage::Q8_0);
        assert_eq!(plan.row_bytes(), 1088);
        assert_eq!(plan.arena_bytes(), 78336 * u64::from(capacity));
        assert_eq!(plan.arena_bytes() * 32, 147456 * u64::from(capacity) * 17);
        let append = plan.append(0, 8191, capacity).unwrap();
        let token = append.token(capacity - 1).unwrap();
        assert_eq!(token.absolute_position(), 8191 + capacity - 1);
        assert_eq!(token.visible_positions(), capacity);
        assert_eq!(token.storage(), K2KvStorage::Q8_0);
        let mut cursor = 0;
        for layer in 0..36 {
            let planes = plan.layer_planes(layer).unwrap();
            let reads = token.read_ranges(layer).unwrap();
            let writes = token.write_ranges(layer).unwrap();
            for (plane, read, write) in [
                (planes.key, reads.key, writes.key),
                (planes.value, reads.value, writes.value),
            ] {
                assert_eq!(plane.start, cursor);
                assert_eq!(plane.start % 34, 0);
                assert_eq!(read, plane);
                assert_eq!(write.end, plane.end);
                assert_eq!(write.end - write.start, 1088);
                cursor = plane.end;
            }
        }
        assert_eq!(cursor, plan.arena_bytes());
        assert!(plan.append(capacity, 8191 + capacity, 1).is_err());
    }
    for capacity in [0, 7169, u32::MAX] {
        assert!(
            K2ShortContextPlan::with_storage(config(), 0, capacity, K2KvStorage::Q8_0).is_err()
        );
    }
}

#[test]
fn request_capacity_is_not_declared_context_or_qualification() {
    let plan = K2ShortContextPlan::new(config(), 131069, 7168).unwrap();
    assert_eq!(plan.declared_context(), 524288);
    assert_eq!(plan.capacity(), 7168);
    assert_eq!(plan.row_bytes(), 2048);
    assert_eq!(plan.arena_bytes(), 147456 * 7168);
    assert!(K2ShortContextPlan::new(config(), 0, 7169).is_err());
    assert!(K2ShortContextPlan::new(config(), 0, 0).is_err());
    assert!(K2ShortContextPlan::new(config(), u32::MAX, 1).is_err());
    assert!(K2ShortContextPlan::new(config(), 524287, 1).is_ok());
    assert!(K2ShortContextPlan::new(config(), 524287, 2).is_err());
    let mut pretrain = config();
    pretrain.context_length = 8192;
    pretrain.rope_theta = 500000.0;
    assert!(K2ShortContextPlan::new(pretrain.clone(), 8191, 1).is_ok());
    assert!(K2ShortContextPlan::new(pretrain, 8191, 2).is_err());
}

#[test]
fn arena_planes_are_disjoint_complete_and_f16_aligned() {
    for capacity in [1, 17, 7168] {
        let plan = K2ShortContextPlan::new(config(), 0, capacity).unwrap();
        let mut cursor = 0;
        for layer in 0..36 {
            let planes = plan.layer_planes(layer).unwrap();
            for plane in [planes.key, planes.value] {
                assert_eq!(plane.start, cursor);
                assert_eq!(plane.start % 2, 0);
                assert_eq!(plane.end - plane.start, u64::from(capacity) * 2048);
                cursor = plane.end;
            }
        }
        assert_eq!(cursor, plan.arena_bytes());
        assert!(plan.layer_planes(36).is_err());
    }
}

#[test]
fn continuation_ranges_include_current_row_and_exclude_future_rows() {
    let plan = K2ShortContextPlan::new(config(), 7919, 8).unwrap();
    let append = plan.append(3, 7922, 4).unwrap();
    assert_eq!(append.final_prefix(), 7);
    for i in 0..4 {
        let token = append.token(i).unwrap();
        assert_eq!(token.absolute_position(), 7922 + i);
        assert_eq!(token.visible_positions(), 4 + i);
        assert_eq!(token.rope().position(), 7922 + i);
        for layer in [0, 1, 35] {
            let planes = plan.layer_planes(layer).unwrap();
            let writes = token.write_ranges(layer).unwrap();
            let reads = token.read_ranges(layer).unwrap();
            for (plane, write, read) in [
                (planes.key, writes.key, reads.key),
                (planes.value, writes.value, reads.value),
            ] {
                assert_eq!(write.start, plane.start + u64::from(3 + i) * 2048);
                assert_eq!(write.end - write.start, 2048);
                assert_eq!(read.start, plane.start);
                assert_eq!(read.end, write.end);
                assert!(read.end < plane.end);
            }
        }
    }
    assert!(append.token(4).is_err());
    for (prefix, absolute, tokens) in [
        (3, 7921, 1),
        (3, 7923, 1),
        (3, 7922, 0),
        (3, 7922, 6),
        (9, 7928, 1),
        (u32::MAX, 0, 2),
    ] {
        assert!(plan.append(prefix, absolute, tokens).is_err());
    }
}

#[test]
fn scratch_boundary_matches_the_candidate_kernel_helper() {
    let plan = K2ShortContextPlan::new(config(), 0, 7168).unwrap();
    let append = plan.append(0, 0, 7168).unwrap();
    for positions in [1, 3, 4, 5, 127, 128, 7167, 7168] {
        let token = append.token(positions - 1).unwrap();
        let (scores, reduce) = crate::metal::materialized_attention_scratch_bytes(
            "k2_plan_test",
            positions as usize,
            32,
        )
        .unwrap();
        assert_eq!(token.score_scratch_bytes() as usize, scores);
        assert_eq!(reduce, 128);
        assert!(scores <= 28 * 1024);
    }
    assert_eq!(append.token(7167).unwrap().score_scratch_bytes(), 28 * 1024);
}

#[test]
fn full_neox_geometry_cannot_silently_become_qwen_or_muse() {
    for modify in [
        |c: &mut K2HorizonConfig| c.rope_dimension_count = 64,
        |c: &mut K2HorizonConfig| c.kv_head_count = 2,
        |c: &mut K2HorizonConfig| c.query_head_count = 16,
        |c: &mut K2HorizonConfig| c.key_head_dim = 256,
        |c: &mut K2HorizonConfig| c.rope_theta = 1.0,
    ] {
        let mut c = config();
        modify(&mut c);
        assert!(K2ShortContextPlan::new(c, 0, 1).is_err());
    }
    for theta in [500000.0, 1000000.0, 10000000.0] {
        let mut c = config();
        c.rope_theta = theta;
        let plan = K2ShortContextPlan::new(c, 37, 2).unwrap();
        let append = plan.append(0, 37, 2).unwrap();
        let token = append.token(1).unwrap();
        let rope = token.rope();
        assert_eq!(rope.position(), 38);
        assert_eq!(rope.theta(), theta);
        assert_eq!(FullNeoxRope::QUERY_HEADS / FullNeoxRope::KV_HEADS, 4);
        assert_eq!(FullNeoxRope::HEAD_DIM, 128);
        assert_eq!(FullNeoxRope::ROTARY_DIM, 128);
    }
}

#[test]
fn norm_plan_preserves_per_group_gamma_coordinates() {
    let plan = K2ShortContextPlan::new(config(), 0, 1).unwrap();
    let x = (0..4096)
        .map(|i| 0.1 + (i % 71) as f64 / 20.0 + (i / 1024) as f64)
        .collect::<Vec<_>>();
    let gamma = (0..4096)
        .map(|i| 0.3 + i as f64 / 4096.0)
        .collect::<Vec<_>>();
    let mut actual = vec![0.0; 4096];
    for group in plan.norm_groups() {
        let range = group.elements.start as usize..group.elements.end as usize;
        let inverse = (x[range.clone()].iter().map(|v| v * v).sum::<f64>() / range.len() as f64
            + f64::from(group.epsilon))
        .sqrt()
        .recip();
        for i in range {
            actual[i] = x[i] * inverse * gamma[i];
        }
    }
    let whole_inverse = (x.iter().map(|v| v * v).sum::<f64>() / 4096.0 + 1e-6)
        .sqrt()
        .recip();
    let mut whole_delta = 0.0_f64;
    let mut gamma_delta = 0.0_f64;
    for i in 0..4096 {
        let start = i / 1024 * 1024;
        let mean = (start..start + 1024).map(|j| x[j].powi(2)).sum::<f64>() / 1024.0;
        let expected = x[i] / (mean + f64::from(1e-6_f32)).sqrt() * gamma[i];
        assert!((actual[i] - expected).abs() < 1e-12);
        whole_delta = whole_delta.max((actual[i] - x[i] * whole_inverse * gamma[i]).abs());
        gamma_delta =
            gamma_delta.max((actual[i] - expected * gamma[(i + 1024) % 4096] / gamma[i]).abs());
    }
    assert!(whole_delta > 0.1);
    assert!(gamma_delta > 0.1);
}

use super::*;
use qwen_llm::snapshot_policy::SnapshotPolicyConfig;

#[derive(Clone, Debug, PartialEq)]
struct State {
    position: usize,
    digest: u64,
}
fn state(ids: &[u32]) -> State {
    State {
        position: ids.len(),
        digest: ids
            .iter()
            .fold(0u64, |s, &x| s.wrapping_mul(101).wrapping_add(x as u64)),
    }
}
fn insert(cache: &mut SnapshotCache<Checkpoint<State>>, ids: &[u32], logits: Vec<f32>) {
    let bytes =
        SnapshotCache::<Checkpoint<State>>::entry_bytes(ids.len(), 16 + (logits.len() * 4) as u64)
            .unwrap();
    assert!(cache.insert_strict(
        ids.to_vec(),
        Checkpoint {
            snapshot: state(ids),
            logits
        },
        bytes
    ));
}
fn head(c: &Checkpoint<State>) -> bool {
    c.logits.len() == 3 && c.logits.iter().all(|x| x.is_finite())
}

#[test]
fn resident_key_requires_the_entire_consumed_prefix_and_a_ready_owner() {
    let prompt = vec![1, 2, 3];
    let generated = vec![4, 5, 6];
    crate::derive_completed_checkpoint_boundary(prompt.len(), &generated, 2, 5).unwrap();
    let mut consumed = prompt;
    consumed.extend([4, 5]);
    assert_eq!(
        live_prefix_len(&consumed, &[1, 2, 3, 4, 5, 6, 7], 5, true),
        Some(5)
    );
    assert_eq!(live_prefix_len(&consumed, &consumed, 5, true), Some(5));
    for (ids, pos, ready) in [
        (&[1, 2, 3, 4, 9, 6][..], 5, true),
        (&[1, 2, 3][..], 5, true),
        (&consumed[..], 4, true),
        (&consumed[..], 5, false),
    ] {
        assert_eq!(live_prefix_len(&consumed, ids, pos, ready), None);
    }
    assert_eq!(live_prefix_len(&[], &[1], 0, true), None);
}

#[test]
fn endpoint_checkpoint_is_bounded_and_preserves_cold_and_header_only_cuts() {
    for (on, matched, total, cut, rows, want) in [
        (true, 65024, 66624, Some(66619), Some(2048), true),
        (false, 65024, 66624, Some(66619), Some(2048), false),
        (true, 0, 1600, Some(1595), Some(2048), false),
        (true, 65024, 65029, Some(65024), Some(2048), false),
        (true, 65024, 65535, Some(65530), Some(2048), false),
        (true, 65024, 67072, Some(67067), Some(2048), true),
        (true, 65024, 67073, Some(67068), Some(2048), false),
        (true, 65024, 66624, Some(66615), Some(2048), false),
        (true, 65024, 66624, None, Some(2048), false),
        (true, 65024, 66624, Some(66619), Some(512), false),
        (true, 65024, 66624, Some(66619), None, false),
        (true, 65024, 1, Some(0), Some(2048), false),
    ] {
        assert_eq!(endpoint_checkpoint(on, matched, total, cut, rows), want);
    }
}

#[test]
fn endpoint_forks_and_retries_keep_actual_state_and_logit_bits() {
    let base: Vec<u32> = (0..65024).collect();
    let full: Vec<u32> = (0..66624).collect();
    let mut cache = SnapshotCache::new(16 << 20, SnapshotPolicyConfig::default())
        .with_checkpoint_ladder(super::super::super::checkpoint_ladder::Config::testing(
            8 << 20,
        ));
    let conversation = super::super::super::checkpoint_ladder::Conversation::testing(3);
    let ids = &base[..63488];
    let bytes = SnapshotCache::<Checkpoint<State>>::entry_bytes(ids.len(), 16).unwrap();
    assert!(cache.insert_ladder_in(
        conversation,
        (),
        ids.to_vec(),
        Checkpoint {
            snapshot: state(ids),
            logits: vec![]
        },
        bytes
    ));
    let bits = [0u32, 0x80000000, 0x3f800001];
    insert(&mut cache, &full, bits.map(f32::from_bits).to_vec());
    cache.record_tip(Some(conversation), &(), &full);
    assert!(
        cache.entry_for(&full[..full.len() - 5]).is_none(),
        "no pre-header alias"
    );
    assert_eq!(
        live_prefix_len(&base, &full, base.len(), true),
        Some(base.len())
    );
    let mut fork = base[..64961].to_vec();
    fork[64900] = 999999;
    let hit = checkpoint_candidate(&mut cache, &fork, true, head).unwrap();
    assert_eq!(hit.prefix_len, 63488);
    let mut runner = state(&[]);
    let matched = restore_prefix_hit(
        &mut cache,
        Some(hit),
        &mut runner,
        |r, c| {
            *r = c.snapshot.clone();
            Ok::<_, &str>(r.position)
        },
        |r| {
            *r = state(&[]);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(matched, 63488);
    assert_eq!(runner, state(ids));
    for _ in 0..2 {
        let hit = checkpoint_candidate(&mut cache, &full, true, head).unwrap();
        let immutable = hit.value.clone();
        let mut restored = vec![];
        let n = restore_prefix_hit(
            &mut cache,
            Some(hit),
            &mut runner,
            |r, c| {
                *r = c.snapshot.clone();
                restored = c.logits.iter().map(|x| x.to_bits()).collect();
                Ok::<_, &str>(r.position)
            },
            |r| {
                *r = state(&[]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(n, full.len());
        assert_eq!(restored, bits);
        runner.position += 1;
        runner.digest = 0;
        assert_eq!(immutable.snapshot, state(&full));
        assert_eq!(
            immutable
                .logits
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
    }
    insert(&mut cache, &base[..60000], vec![]);
    assert!(
        checkpoint_candidate(&mut cache, &base[..60000], true, head).is_none(),
        "state-only exact hit has no observation"
    );
    assert_eq!(
        checkpoint_candidate(&mut cache, &full, false, head)
            .unwrap()
            .prefix_len,
        63488
    );
}

#[test]
fn failed_exact_restore_is_removed_before_cold_reset() {
    let ids = [1, 2, 3];
    let mut cache = SnapshotCache::new(1024, Default::default());
    insert(&mut cache, &ids, vec![0., -0., 1.]);
    let hit = checkpoint_candidate(&mut cache, &ids, true, head);
    let mut runner = state(&[9]);
    assert_eq!(
        restore_prefix_hit(
            &mut cache,
            hit,
            &mut runner,
            |r, _| {
                r.position = 2;
                Ok::<_, &str>(2)
            },
            |r| {
                *r = state(&[]);
                Ok(())
            }
        )
        .unwrap(),
        0
    );
    assert_eq!(runner, state(&[]));
    assert!(cache.entry_for(&ids).is_none());
}

use super::*;

fn lineage() -> CheckpointLineage {
    CheckpointLineage {
        session: 7,
        branch: 2,
        append_epoch: 3,
    }
}

#[test]
fn device_checkpoint_cpu_lineage_allows_only_same_branch_appends() {
    let mut current = lineage();
    let saved = Stamp {
        lineage: current,
        length: 3,
    };
    current.validate_restore(saved, 3).unwrap();
    current.record_append().unwrap();
    current.validate_restore(saved, 7).unwrap();
    assert!(current.validate_restore(saved, 2).is_err());
    let other = CheckpointLineage {
        session: 8,
        ..current
    };
    assert!(other.validate_restore(saved, 7).is_err());
    let future = Stamp {
        lineage: CheckpointLineage {
            append_epoch: current.append_epoch + 1,
            ..current
        },
        length: 7,
    };
    assert!(current.validate_restore(future, 7).is_err());
}

#[test]
fn device_checkpoint_cpu_lineage_reset_and_restore_invalidate_siblings() {
    let current = lineage();
    let saved = Stamp {
        lineage: current,
        length: 3,
    };
    let next = current.next_branch().unwrap();
    assert_eq!(next.session, current.session);
    assert_eq!(next.append_epoch, 0);
    assert!(next.validate_restore(saved, 3).is_err());
    let mut replayed = next;
    for _ in 0..8 {
        replayed.record_append().unwrap();
    }
    assert!(replayed.validate_restore(saved, 11).is_err());
    replayed
        .validate_restore(
            Stamp {
                lineage: next,
                length: 3,
            },
            11,
        )
        .unwrap();
}

#[test]
fn device_checkpoint_cpu_lineage_overflow_refuses_without_mutation() {
    let mut current = CheckpointLineage {
        append_epoch: u64::MAX,
        ..lineage()
    };
    let before = current;
    assert!(current.record_append().is_err());
    assert_eq!(current, before);
    let current = CheckpointLineage {
        branch: u64::MAX,
        ..lineage()
    };
    assert!(current.next_branch().is_err());
}

#[test]
fn device_checkpoint_cpu_payload_is_fixed_not_a_prefix_snapshot() {
    let config = Qwen4ExpConfig::flash_next_reference();
    for capacity in [64, 8192, 32768] {
        let g = Qwen4ExpTextSessionMetalGeometry::from_config(&config, capacity).unwrap();
        assert_eq!(g.device_checkpoint_bytes(), 118_063_104);
        assert!(g.snapshot_bytes(1).unwrap() > g.device_checkpoint_bytes());
        assert!(g.snapshot_bytes(capacity).unwrap() > g.snapshot_bytes(1).unwrap());
    }
}

#[test]
#[ignore = "exclusive GPU lease; synthetic compact checkpoint blits, no model weights"]
fn device_checkpoint_metal_restores_only_bounded_state_and_rejects_poison() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let g =
        Qwen4ExpTextSessionMetalGeometry::from_config(&Qwen4ExpConfig::flash_next_reference(), 64)
            .unwrap();
    let mut workspace = Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, g).unwrap();
    let mut bank = workspace.allocate_device_checkpoint_bank(&ctx).unwrap();
    let mut sibling = workspace.allocate_device_checkpoint_bank(&ctx).unwrap();
    assert!(
        workspace
            .restore_device_checkpoint(&ctx, &mut bank)
            .is_err()
    );
    workspace
        .capture_device_checkpoint(&ctx, &mut bank)
        .unwrap();
    workspace
        .capture_device_checkpoint(&ctx, &mut sibling)
        .unwrap();
    assert_eq!(bank.captured_position(), Some(0));
    assert_eq!(bank.allocation().logical_bytes, 118_063_104);
    let regions = workspace.compact_checkpoint_regions(&ctx).unwrap();
    assert_eq!(regions.len(), 85);
    for region in &regions {
        // SAFETY: idle, private, writable F32 state with validated extents.
        unsafe {
            std::slice::from_raw_parts_mut(
                region
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(region.offset as usize),
                region.n_bytes() as usize,
            )
            .fill(0x31);
        }
    }
    let cache_regions: Vec<_> = workspace
        .qsa_persistent_state_tensors()
        .into_iter()
        .flat_map(|(_, regions)| regions.into_iter().skip(1))
        .collect();
    for region in &cache_regions {
        // SAFETY: idle test workspace; fill deliberately unreachable suffix bytes.
        unsafe {
            std::slice::from_raw_parts_mut(
                region
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(region.offset as usize),
                region.n_bytes() as usize,
            )
            .fill(0x42);
        }
    }
    workspace.state_poisoned = true;
    assert!(
        workspace
            .restore_device_checkpoint(&ctx, &mut bank)
            .is_err()
    );
    assert_eq!(bank.captured_position(), Some(0));
    workspace.state_poisoned = false; // synthetic flag only; no failed GPU command
    workspace
        .restore_device_checkpoint(&ctx, &mut bank)
        .unwrap();
    assert_eq!(bank.captured_position(), None);
    assert!(workspace.logits().is_err());
    for region in &regions {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                region
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(region.offset as usize),
                region.n_bytes() as usize,
            )
        };
        assert!(bytes.iter().all(|&v| v == 0));
    }
    for region in &cache_regions {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                region
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(region.offset as usize),
                region.n_bytes() as usize,
            )
        };
        assert!(bytes.iter().all(|&v| v == 0x42));
    }
    assert!(
        workspace
            .restore_device_checkpoint(&ctx, &mut bank)
            .is_err()
    );
    assert!(
        workspace
            .restore_device_checkpoint(&ctx, &mut sibling)
            .is_err()
    );
    workspace
        .capture_device_checkpoint(&ctx, &mut bank)
        .unwrap();
    workspace.reset().unwrap();
    assert!(
        workspace
            .restore_device_checkpoint(&ctx, &mut bank)
            .is_err()
    );
}

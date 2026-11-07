use super::*;

/// Prefill A, snapshot, keep going with B and a short greedy decode; then
/// restore the snapshot into the reset (dirty) workspace and replay B and
/// the decode: logits and all 121 persistent tensors must match bit for
/// bit. A cold prefill of A+B must agree within the released packed
/// tolerance (its packed chunking differs from the split).
#[test]
#[ignore = "production lease + released UD-Q3_K_XL; snapshot restore replays continuation bit-exactly"]
fn snapshot_restore_replays_continuation_bit_exactly() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let bytes = include_bytes!(
        "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
    );
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    // A short prefix with a partial index block, and one ending just below
    // the QSA dense end so B's packed range crosses into selection.
    for (prefix, suffix) in [(13, 6), (2_047, 40)] {
        let total = prefix + suffix;
        assert!(tokens.len() >= total);
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, total + 8).unwrap();
        let mut loaded =
            Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, total).unwrap();
        let bindings = (loaded.guarded_topk_enabled(), loaded.hc_up_mix_enabled());
        let mut runner = loaded.create_runner(&ctx).unwrap();
        let (a, b) = (&tokens[..prefix], &tokens[prefix..total]);

        runner.prefill(a).unwrap();
        let estimate = runner.snapshot_bytes().unwrap();
        let snapshot = runner.capture_snapshot().unwrap();
        assert_eq!(snapshot.length(), prefix);
        assert_eq!(snapshot.payload_bytes(), estimate);
        let replay = |runner: &mut Qwen4ExpTextRunner<'_, '_, '_>, forced: Option<&[u32]>| {
            let mut rows = vec![
                runner
                    .prefill_continuation_with_command_checkpoint(b, || Ok(()))
                    .unwrap()
                    .to_vec(),
            ];
            let mut decoded = Vec::new();
            for step in 0..4 {
                let token = forced.map_or_else(|| argmax(rows.last().unwrap()) as u32, |f| f[step]);
                decoded.push(token);
                rows.push(runner.forward_token(token).unwrap().to_vec());
            }
            (rows, decoded, snapshot_persistent_state(runner))
        };
        let (baseline, decoded, baseline_state) = replay(&mut runner, None);

        runner.restore_snapshot(&snapshot).unwrap();
        assert_eq!(runner.next_position(), prefix);
        assert!(runner.logits().is_err(), "a snapshot carries no logits");
        assert_eq!(
            (
                runner.workspace.guarded_topk_enabled(),
                runner.workspace.hc_up_mix_enabled()
            ),
            bindings,
            "restore must not touch configured math"
        );
        let (restored, _, restored_state) = replay(&mut runner, Some(&decoded));
        for (index, (expected, actual)) in baseline.iter().zip(&restored).enumerate() {
            assert_f32_bits_eq(&format!("prefix={prefix} row {index}"), expected, actual);
        }
        assert_eq!(baseline_state.len(), 121);
        for (index, (expected, actual)) in baseline_state.iter().zip(&restored_state).enumerate() {
            assert!(
                expected == actual,
                "prefix={prefix} persistent tensor {index}"
            );
        }

        runner.reset().unwrap();
        let cold = runner.prefill(&tokens[..total]).unwrap().to_vec();
        assert_released_packed_logit_arms_close(
            &format!("prefix={prefix} split vs cold"),
            &cold,
            &baseline[0],
        );
        eprintln!(
            "snapshot prefix={prefix} suffix={suffix} bytes={estimate} bit-exact rows={}",
            baseline.len()
        );
    }
}

#[test]
#[ignore = "production lease + QWEN4EXP_Q3_K_XL_GGUF override; committed live-trajectory observer and bit-exact restore; model GPU"]
fn live_checkpoint_observer_restores_exact_state() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let bytes = include_bytes!(
        "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
    );
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::from_gguf(&gguf).unwrap();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 2210).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, 128).unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert_eq!(runner.packed_prefill_capacity(), Some(128));
    runner.prefill(&tokens[..97]).unwrap();
    for &token in &tokens[97..100] {
        runner.forward_token(token).unwrap();
    }
    let initial = runner.capture_snapshot().unwrap();
    let mut callbacks = 0;
    assert!(
        runner
            .prefill_continuation_with_observer(
                &tokens[100..356],
                || Err(Qwen4ExpRuntimeError::Checkpoint(
                    "cancel before command".into()
                )),
                &mut |_, _| callbacks += 1
            )
            .is_err()
    );
    assert_eq!(callbacks, 0, "no observation before a successful commit");
    assert_eq!(runner.next_position(), 100);
    let reference = runner
        .prefill_continuation_with_command_checkpoint(&tokens[100..2200], || Ok(()))
        .unwrap()
        .to_vec();
    let reference_state = snapshot_persistent_state(&runner);
    runner.restore_snapshot(&initial).unwrap();
    let mut checkpoint = None;
    let observed = runner
        .prefill_continuation_with_observer(&tokens[100..2200], || Ok(()), &mut |runner, range| {
            assert_eq!(runner.next_position(), range.end);
            if range.end == 2179 {
                checkpoint = Some(runner.capture_snapshot().unwrap());
            }
        })
        .unwrap()
        .to_vec();
    assert_f32_bits_eq("observer changes no prompt logits", &reference, &observed);
    assert_eq!(reference_state, snapshot_persistent_state(&runner));
    let mut tail = Vec::new();
    for &token in &tokens[2200..2204] {
        tail.push(runner.forward_token(token).unwrap().to_vec());
    }
    let end = snapshot_persistent_state(&runner);
    runner
        .restore_snapshot(&checkpoint.expect("committed full chunk after sparse frontier"))
        .unwrap();
    let restored = runner
        .prefill_continuation_with_command_checkpoint(&tokens[2179..2200], || Ok(()))
        .unwrap()
        .to_vec();
    assert_f32_bits_eq("restored prompt logits", &reference, &restored);
    for (i, &token) in tokens[2200..2204].iter().enumerate() {
        assert_f32_bits_eq(
            "teacher-forced continuation",
            &tail[i],
            &runner.forward_token(token).unwrap().to_vec(),
        );
    }
    assert_eq!(
        end,
        snapshot_persistent_state(&runner),
        "all persistent tensors"
    );
}

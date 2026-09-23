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

use super::*;

fn visible_state(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> Vec<Vec<u8>> {
    runner
        .workspace
        .checkpoint_visible_regions_for_tests()
        .unwrap()
        .iter()
        .map(|t| {
            // All producer commands are released; views are checked allocation subranges.
            unsafe {
                std::slice::from_raw_parts(
                    t.buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(t.offset as usize),
                    t.n_bytes() as usize,
                )
            }
            .to_vec()
        })
        .collect()
}

#[test]
#[ignore = "exclusive Metal lease + Flash-Next fixture; retained workspace ownership and bitwise continuation"]
fn released_retained_workspace_preserves_state_and_rejects_foreign_owner() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::from_gguf(&gguf).unwrap();
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    let ids: Vec<u32> = tokenizer
        .encode(
            &"fn checkpoint(state: &mut State) { state.commit(); }\n".repeat(4),
            false,
        )
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert!(ids.len() >= 16);
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 64).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, 32).unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    runner.prefill(&ids[..8]).unwrap();
    let before = runner.logits().unwrap().to_vec();
    let expected = runner
        .prefill_continuation_with_command_checkpoint(&ids[8..16], || Ok(()))
        .unwrap()
        .to_vec();
    let expected_state = visible_state(&runner);
    let expected_history = runner.workspace.ple_prior_tokens().to_vec();
    runner.reset().unwrap();
    runner.prefill(&ids[..8]).unwrap();
    assert_eq!(
        before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        runner
            .logits()
            .unwrap()
            .as_slice()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    let foreign =
        Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, runner.workspace.geometry().clone())
            .unwrap();
    let owned = runner.into_workspace();
    assert!(
        loaded
            .retain_workspace(foreign)
            .unwrap_err()
            .to_string()
            .contains("does not belong")
    );
    loaded.retain_workspace(owned).unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert_eq!(runner.next_position(), 8);
    assert_eq!(
        before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        runner
            .logits()
            .unwrap()
            .as_slice()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    let actual = runner
        .prefill_continuation_with_command_checkpoint(&ids[8..16], || Ok(()))
        .unwrap();
    assert_eq!(
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        actual
            .as_slice()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(expected_state, visible_state(&runner));
    assert_eq!(expected_history, runner.workspace.ple_prior_tokens());
    let workspace = runner.into_workspace();
    loaded.restore_workspace(workspace).unwrap();
    let runner = loaded.create_runner(&ctx).unwrap();
    assert_eq!(runner.next_position(), 0);
    assert!(runner.logits().is_err());
    let workspace = runner.into_workspace();
    loaded.restore_workspace(workspace).unwrap();
}

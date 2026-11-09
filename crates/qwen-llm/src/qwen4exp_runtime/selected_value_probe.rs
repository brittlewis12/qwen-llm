use super::*;
use crate::tokenizer::Tokenizer;
use sha2::{Digest, Sha256};

fn hash(values: &[f32]) -> String {
    assert!(values.iter().all(|v| v.is_finite()));
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

#[test]
#[ignore = "local Flash target and text; production lease; product cancellation and retry"]
fn product_selected_cancellation_retry() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let text = std::fs::read_to_string(std::env::var_os("QWEN_QSA_TEXT").unwrap()).unwrap();
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tokens: Vec<u32> = Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|v| u32::try_from(v).unwrap())
        .take(10240)
        .collect();
    assert_eq!(tokens.len(), 10240);
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, tokens.len()).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, tokens.len()).unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    crate::metal::dispatch_census_begin();
    let expected = hash(runner.prefill(&tokens).unwrap().as_slice());
    let census = crate::metal::dispatch_census_take();
    assert!(
        census
            .iter()
            .any(|r| r.kernel.contains("gqa12") && !r.kernel.ends_with("_test")),
        "product route must actually run"
    );
    let state = super::strict_inject_probe::state(&runner);
    runner.reset().unwrap();
    let mut checkpoints = 0;
    let error = runner
        .prefill_with_command_checkpoint(&tokens, || {
            checkpoints += 1;
            if checkpoints == 4 {
                Err(Qwen4ExpRuntimeError::Checkpoint(
                    "selected boundary cancellation".into(),
                ))
            } else {
                Ok(())
            }
        })
        .err()
        .expect("cancellation must stop prefill");
    assert!(matches!(error, Qwen4ExpRuntimeError::Prefill { .. }));
    assert_eq!(runner.next_position(), 4099);
    assert!(!runner.workspace.is_poisoned());
    let actual = hash(
        runner
            .prefill_continuation_with_command_checkpoint(&tokens[4099..], || Ok(()))
            .unwrap()
            .as_slice(),
    );
    assert_eq!(actual, expected);
    assert_eq!(super::strict_inject_probe::state(&runner), state);
}

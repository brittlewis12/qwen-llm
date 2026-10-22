use super::*;
use crate::qwen4exp_metal::strict_inject_probe::Scope;
use serde_json::json;
use sha2::{Digest, Sha256};

fn fingerprint(values: &[f32]) -> String {
    assert!(values.iter().all(|v| v.is_finite()));
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

fn state(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> Vec<String> {
    assert!(!runner.workspace.is_poisoned());
    let position = runner.next_position();
    let mut active_qsa = Vec::new();
    for (layer, tensors) in runner.workspace.qsa_persistent_state_tensors() {
        assert_eq!(tensors.len(), 4);
        let ratio = runner
            .weights
            .geometry
            .post_ple()
            .iter()
            .find(|block| block.layer() == layer)
            .unwrap()
            .mixer()
            .qsa()
            .unwrap()
            .compression_ratio();
        for (part, tensor) in tensors.into_iter().enumerate() {
            let bytes = match part {
                0 => tensor.n_bytes() as usize,
                1 => tensor.shape[0] as usize * (position / ratio) * 2,
                2 | 3 => tensor.shape[0] as usize * tensor.shape[1] as usize * position * 2,
                _ => unreachable!(),
            };
            active_qsa.push((tensor, bytes));
        }
    }
    let mut result = vec![
        format!("{}", runner.next_position()),
        format!("{:?}", runner.workspace.qsa_committed_lengths()),
        format!("{:?}", runner.workspace.ple_prior_tokens()),
    ];
    let mut total_bytes = 0;
    for tensor in runner.workspace.persistent_state_tensors() {
        let active_bytes = active_qsa
            .iter()
            .find(|(q, _)| {
                std::ptr::addr_eq(
                    Retained::as_ptr(&q.buffer),
                    Retained::as_ptr(&tensor.buffer),
                ) && q.offset == tensor.offset
            })
            .map_or(tensor.n_bytes() as usize, |(_, bytes)| *bytes);
        assert!(active_bytes <= tensor.n_bytes() as usize);
        total_bytes += active_bytes;
        // Called only after synchronous commands have completed and released.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize),
                active_bytes,
            )
        };
        result.push(format!("{:x}", Sha256::digest(bytes)));
    }
    assert_eq!(total_bytes as u64, runner.snapshot_bytes().unwrap());
    result
}

#[test]
#[ignore = "local Flash target; production lease; ordinary prefill strict HC qualification"]
fn ordinary_prefill_exact_and_abba() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let report = std::env::var_os("QWEN_HC_STRICT_REPORT").expect("new report path required");
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)
        .unwrap();
    let bytes = include_bytes!(
        "../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97"
    );
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    crate::metal::hc4_strict::prepare_hc4_strict_test(&ctx).unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 2052).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, 2048).unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let plan = runner.workspace.memory_plan().clone();
    let mut records = Vec::new();
    for rows in [128, 512, 2048] {
        let mut baseline = None;
        for (index, enabled) in [false, true, false, true, true, false]
            .into_iter()
            .enumerate()
        {
            runner.reset().unwrap();
            let _scope = Scope::new(enabled);
            if index < 2 {
                crate::metal::dispatch_census_begin();
            }
            let logits = fingerprint(runner.prefill(&tokens[..rows]).unwrap().as_slice());
            let timing = runner.last_prefill_timing().unwrap();
            assert_eq!(timing.packed_token_count, rows);
            assert_eq!(timing.command_count, 1);
            assert!(!timing.contains_selection);
            let hyper = fingerprint(&runner.workspace.final_packed_hyper_for_tests(rows));
            let prefill_state = state(&runner);
            if index < 2 {
                let census = crate::metal::dispatch_census_take();
                let count = census
                    .iter()
                    .filter(|r| r.kernel == "kernel_mat_mat_f32_hc4p32_strict_test")
                    .count();
                assert_eq!(count, if enabled { 96 } else { 0 });
            }
            let mut continuation = Vec::new();
            for &token in &tokens[rows..rows + 4] {
                continuation.push(fingerprint(runner.forward_token(token).unwrap().as_slice()));
            }
            let observed = (logits, hyper, prefill_state, continuation, state(&runner));
            if let Some(expected) = &baseline {
                assert_eq!(
                    &observed, expected,
                    "rows={rows} index={index} strict={enabled}"
                );
            } else {
                baseline = Some(observed);
            }
            assert_eq!(runner.workspace.memory_plan(), &plan);
            let record = json!({"rows":rows,"index":index,"strict":enabled,
                "measured":index >= 2,"wall_ms":timing.total_wall_ms,
                "gpu_ms":timing.complete_gpu_ms().unwrap(),"exact":true});
            eprintln!("strict_hc_native {record}");
            records.push(record);
        }
    }
    serde_json::to_writer_pretty(&mut output, &json!({"records":records})).unwrap();
}

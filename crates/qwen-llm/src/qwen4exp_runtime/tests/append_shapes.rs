//! Released-artifact append packet. Only this ignored test reads FLASH_APPEND_PLAN.
use super::*;
use crate::qwen4exp_prefill_tensor::shapes;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::io::Write;
use std::time::Instant;

fn emit(out: &mut std::fs::File, row: Value) {
    writeln!(out, "{row}").unwrap();
    out.flush().unwrap();
}

fn tokens(path: &str, tok: &Tokenizer) -> Vec<u32> {
    tok.encode(
        &std::fs::read_to_string(path)
            .unwrap()
            .replace("<|", "‹|")
            .replace("<｜", "‹｜"),
        false,
    )
    .unwrap()
    .into_iter()
    .map(|v| u32::try_from(v).unwrap())
    .collect()
}

fn run_append(r: &mut Qwen4ExpTextRunner<'_, '_, '_>, input: &[u32], candidate: bool) -> Vec<f32> {
    shapes::with_enabled(candidate, || {
        r.prefill_continuation_with_command_checkpoint(input, || Ok(()))
            .unwrap()
            .to_vec()
    })
}

fn distributions(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    input: &[u32],
    candidate: bool,
    noise: bool,
    targets: usize,
) -> Vec<Vec<f32>> {
    crate::metal::with_gdn_prefill_numeric_control(!noise, || {
        let endpoint = run_append(r, input, candidate);
        let hyper = r
            .workspace
            .completed_final_hyper_rows_for_tests(input.len());
        let width = r.weights.geometry.hyper_width();
        let selected: Vec<f32> = (0..targets)
            .flat_map(|i| {
                let row = i * (input.len() - 1) / (targets - 1);
                hyper[row * width..(row + 1) * width].iter().copied()
            })
            .collect();
        let values = native_readout_of_saved_hyper_rows(r, &selected, targets);
        assert_f32_bits_eq(
            "selected native endpoint",
            &endpoint,
            values.last().unwrap(),
        );
        values
    })
}

fn lse(x: &[f32]) -> f64 {
    assert!(x.iter().all(|v| v.is_finite()));
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + x
        .iter()
        .map(|&v| (f64::from(v) - max).exp())
        .sum::<f64>()
        .ln()
}

fn compare(a: &[f32], b: &[f32], target: usize) -> Value {
    assert_eq!(a.len(), b.len());
    let (la, lb) = (lse(a), lse(b));
    let kl: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let p = f64::from(x) - la;
            p.exp() * (p - (f64::from(y) - lb))
        })
        .sum();
    assert!(kl.is_finite() && kl >= -1e-10);
    json!({"nll_a":la-f64::from(a[target]),"nll_b":lb-f64::from(b[target]),
        "dnll":lb-f64::from(b[target])-(la-f64::from(a[target])),"kl":kl.max(0.0),
        "argmax_same":argmax(a)==argmax(b),"bitwise":bytemuck::cast_slice::<f32,u8>(a)==bytemuck::cast_slice::<f32,u8>(b)})
}

#[test]
#[ignore = "released model, canonical lease, NEW FLASH_PREFILL_OUT and frozen FLASH_APPEND_PLAN"]
fn native_append_shape_packet() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let plan: Value = serde_json::from_slice(
        &std::fs::read(std::env::var("FLASH_APPEND_PLAN").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var("FLASH_PREFILL_OUT").unwrap())
        .unwrap();
    let gguf = GgufFile::open(std::env::var("FLASH_PREFILL_MODEL").unwrap()).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let prefix = tokens(plan["prefix"].as_str().unwrap(), &tok);
    let depth = plan["depth"].as_u64().unwrap() as usize;
    let rows: Vec<usize> = plan["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect();
    assert!(depth <= prefix.len());
    let ctx = MetalContext::new().unwrap();
    assert!(
        ctx.tensor_ops_status().is_available(),
        "packet requires the tensor tier"
    );
    let config = Qwen4ExpConfig::from_gguf(&gguf).unwrap();
    let cap = Qwen4ExpSessionCapacity::for_forward_limit(&config, depth + 4096).unwrap();
    let mut model =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, cap, depth + 4096).unwrap();
    let mut r = model.create_runner(&ctx).unwrap();
    let start = Instant::now();
    shapes::with_enabled(false, || {
        r.prefill(&prefix[..depth]).unwrap();
    });
    let entry = r.capture_snapshot().unwrap();
    let candidate_entry = if plan["mode"] == "timing" {
        None
    } else {
        r.reset().unwrap();
        shapes::with_enabled(true, || {
            r.prefill(&prefix[..depth]).unwrap();
        });
        Some(r.capture_snapshot().unwrap())
    };
    let noise_entry = if plan["mode"] == "timing" {
        None
    } else {
        r.reset().unwrap();
        crate::metal::with_gdn_prefill_numeric_control(false, || {
            shapes::with_enabled(false, || {
                r.prefill(&prefix[..depth]).unwrap();
            });
        });
        Some(r.capture_snapshot().unwrap())
    };
    emit(
        &mut out,
        json!({"kind":"prefix","depth":depth,"wall_ms":start.elapsed().as_secs_f64()*1e3,
        "input_hash":blake3::hash(bytemuck::cast_slice(&prefix[..depth])).to_hex().to_string(),
        "plan":plan,"model":std::env::var("FLASH_PREFILL_MODEL").unwrap()}),
    );
    if plan["mode"] == "timing" {
        for &n in &rows {
            let input = &prefix[depth..depth + n];
            for on in [false, true] {
                r.restore_snapshot(&entry).unwrap();
                crate::metal::dispatch_census_begin();
                run_append(&mut r, input, on);
                let mut counts = BTreeMap::<String, usize>::new();
                for row in crate::metal::dispatch_census_take() {
                    *counts.entry(row.kernel).or_default() += 1;
                }
                emit(
                    &mut out,
                    json!({"kind":"routes","rows":n,"candidate":on,"counts":counts,"timing_eligible":false}),
                );
            }
            let mut expected = BTreeMap::new();
            for (block, order) in [[false, true, true, false], [true, false, false, true]]
                .into_iter()
                .enumerate()
            {
                for (arm, on) in order.into_iter().enumerate() {
                    r.restore_snapshot(&entry).unwrap();
                    let start = Instant::now();
                    let logits = run_append(&mut r, input, on);
                    let wall = start.elapsed().as_secs_f64() * 1e3;
                    let t = r.last_prefill_timing().unwrap();
                    assert_eq!(t.command_count, 1);
                    let hash = blake3::hash(bytemuck::cast_slice(&logits))
                        .to_hex()
                        .to_string();
                    assert_eq!(
                        expected.entry(on).or_insert_with(|| hash.clone()),
                        &hash,
                        "same-policy snapshot replay differs"
                    );
                    emit(
                        &mut out,
                        json!({"kind":"native","rows":n,"block":block,"arm":arm,"candidate":on,
                        "wall_ms":wall,"gpu_ms":t.gpu_ms,"encode_ms":t.encode_cpu_ms,
                        "hash":hash}),
                    );
                }
            }
        }
        return;
    }
    let targets = plan["targets"].as_u64().unwrap() as usize;
    assert!((2..=256).contains(&targets));
    // Targets are independent; keep each vocabulary reduction in its original
    // F64 order and bound CPU parallelism while the model remains resident.
    let statistics_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for doc in plan["documents"].as_array().unwrap() {
        let data = tokens(doc["path"].as_str().unwrap(), &tok);
        let offset = doc["offset"].as_u64().unwrap_or(0) as usize;
        for &n in &rows {
            assert!(offset + n < data.len());
            let input = &data[offset..offset + n];
            let mut arms = Vec::new();
            for (candidate, noise) in [(false, false), (true, false), (false, false), (false, true)]
            {
                r.restore_snapshot(if noise {
                    noise_entry.as_ref().unwrap()
                } else if candidate {
                    candidate_entry.as_ref().unwrap_or(&entry)
                } else {
                    &entry
                })
                .unwrap();
                arms.push(distributions(&mut r, input, candidate, noise, targets));
            }
            let values: Vec<Value> = statistics_pool.install(|| {
                (0..targets)
                    .into_par_iter()
                    .map(|i| {
                        assert_f32_bits_eq("baseline repeat noise", &arms[0][i], &arms[2][i]);
                        let target = data[offset + i * (n - 1) / (targets - 1) + 1] as usize;
                        json!({"position":i*(n-1)/(targets-1),"target":target,
                    "candidate":compare(&arms[0][i],&arms[1][i],target),
                    "noise_control":compare(&arms[0][i],&arms[3][i],target)})
                    })
                    .collect()
            });
            emit(
                &mut out,
                json!({"kind":"quality","depth":depth,"rows":n,"document":doc["id"],
                "input_hash":blake3::hash(bytemuck::cast_slice(input)).to_hex().to_string(),"targets":values,"repeat_bitwise":true}),
            );
        }
    }
}

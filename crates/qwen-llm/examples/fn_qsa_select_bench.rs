//! Isolated singleton QSA selector latency: serial chains of identical
//! dispatches in one command buffer, GPU-timestamp median per dispatch.
//! [VISIBLE=2048,8192,32768] [CHAIN=64] [TRIALS=7]. Run under the Metal lease.
#[cfg(feature = "perf-ablations")]
mod probe {
    use anyhow::Result;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLSize};
    use qwen_llm::{metal::*, tensor::GgmlType};
    use serde_json::json;

    fn scores(n: usize) -> Vec<u8> {
        // Positive, distinct-ish pseudorandom scores: radix passes run deep.
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        (0..n)
            .flat_map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32) / (1u64 << 24) as f32 * 3.0).to_le_bytes()
            })
            .collect()
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        let visible: Vec<usize> = args
            .get(1)
            .map_or("2048,8192,32768", String::as_str)
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let chain: usize = args.get(2).map_or(Ok(64), |s| s.parse())?;
        let trials: usize = args.get(3).map_or(Ok(7), |s| s.parse())?;
        let ctx = MetalContext::new()?;
        for n in visible {
            let row = MetalTensor::from_bytes(&ctx, &scores(n), vec![n as u64], GgmlType::F32)?;
            let vis =
                MetalTensor::from_bytes(&ctx, &(n as i32).to_le_bytes(), vec![1], GgmlType::I32)?;
            let ids = MetalTensor::zeros_i32(&ctx, vec![2048])?;
            let count = MetalTensor::zeros_i32(&ctx, vec![1])?;
            let status = MetalTensor::zeros_i32(&ctx, vec![1])?;
            let mut arms: Vec<(String, usize, usize)> = vec![
                ("kernel_qwen4exp_qsa_fill_block_ids_i32".into(), 1, 0),
                (
                    "kernel_deepseek_v4_select_top_k_radix4_ids_f32".into(),
                    256,
                    1024,
                ),
                ("kernel_qwen4exp_qsa_select_scan_f32".into(), 256, 1024),
            ];
            for keys in [8usize, 16, 32] {
                for threads in [256usize, 512, 1024] {
                    if keys * threads >= n {
                        arms.push((
                            format!("kernel_qwen4exp_qsa_select_decode{keys}_f32"),
                            threads,
                            0,
                        ));
                    }
                }
            }
            for (name, threads, dynamic) in arms {
                let pso = ctx.pipeline(&name)?;
                if threads > pso.maxTotalThreadsPerThreadgroup() {
                    println!(
                        "{}",
                        json!({"visible":n,"kernel":name,"threads":threads,"skipped":pso.maxTotalThreadsPerThreadgroup()})
                    );
                    continue;
                }
                let mut samples = Vec::new();
                for _ in 0..trials + 1 {
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    for _ in 0..chain {
                        enc.set_pipeline(&pso);
                        if dynamic == 0 && threads == 1 {
                            // Fill: block_budget, visible, ratio, length, width.
                            enc.set_bytes(0, &[512u32, n as u32, 4, 4 * n as u32, 2048]);
                            enc.set_tensor(1, &ids);
                            enc.set_tensor(2, &count);
                            enc.set_tensor(3, &status);
                        } else {
                            enc.set_bytes(0, &[n as u32, 512, 1, 0]);
                            enc.set_tensor(1, &row);
                            enc.set_tensor(2, &vis);
                            enc.set_tensor(3, &ids);
                            enc.set_tensor(4, &count);
                            enc.set_tensor(5, &status);
                        }
                        if dynamic > 0 {
                            enc.set_threadgroup_memory(0, dynamic);
                            enc.set_threadgroup_memory(1, dynamic);
                        }
                        enc.dispatch(
                            MTLSize {
                                width: 1,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: threads,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    enc.end();
                    cmd.commit();
                    wait_completed(&cmd)?;
                    samples.push((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6 / chain as f64);
                }
                samples.remove(0);
                samples.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    json!({"visible":n,"kernel":name,"threads":threads,"us_per_dispatch_median":samples[samples.len()/2],"min":samples[0],"max":samples[samples.len()-1]})
                );
            }
        }
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    panic!("requires --features perf-ablations")
}

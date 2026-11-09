//! Bounded replay/fusion discriminator. Enable perf-ablations; no product knob.
#[cfg(feature = "perf-ablations")]
mod probe {
    use qwen_llm::{
        gguf::GgufFile,
        metal::{KernelEncoder, MetalContext},
        qwen4exp::Qwen4ExpConfig,
        qwen4exp_runtime::{
            Qwen4ExpLoadedModel, Qwen4ExpPostLayerHyperRequest, Qwen4ExpSessionCapacity,
        },
    };
    use serde_json::json;
    fn ids(n: usize, v: u32, mut s: u64) -> Vec<u32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % u64::from(v)) as u32
            })
            .collect()
    }
    fn error(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0., f64::max)
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let a: Vec<_> = std::env::args().collect();
        let depth: usize = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8192);
        let ctx = MetalContext::new()?;
        let g = GgufFile::open(&a[1])?;
        let c = Qwen4ExpConfig::from_gguf(&g)?;
        let cap = Qwen4ExpSessionCapacity::for_forward_limit(&c, depth + 192)?;
        let mut model = Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &g, cap, depth + 192)?;
        let mut r = model.create_runner(&ctx)?;
        r.configure_causal_executor(false);
        r.prefill(&ids(64, c.vocab_size, 1))?;
        r.reset()?;
        r.prefill(&ids(depth, c.vocab_size, 1 ^ 0x9e3779b97f4a7c15))?;
        for t in ids(64, c.vocab_size, 7) {
            r.forward_token(t)?;
        }
        let entry = r.capture_snapshot()?;
        let token = ids(1, c.vocab_size, 13)[0];
        let layers = [1, 2, 3, 4, 7, 15, 31, 47];
        let mut baseline = Vec::new();
        for layer in layers {
            r.restore_snapshot(&entry)?;
            r.configure_causal_executor(false);
            baseline.push(
                r.forward_token_with_post_layer_hyper_capture(
                    token,
                    Qwen4ExpPostLayerHyperRequest {
                        layer,
                        fixed_add: None,
                    },
                )?
                .values,
            );
        }
        for mode in 0..4 {
            let _mode = KernelEncoder::diagnostic_causal_replay(mode);
            for (i, layer) in layers.into_iter().enumerate() {
                r.restore_snapshot(&entry)?;
                r.configure_causal_executor(true);
                let got = r.forward_token_with_post_layer_hyper_capture(
                    token,
                    Qwen4ExpPostLayerHyperRequest {
                        layer,
                        fixed_add: None,
                    },
                )?;
                println!(
                    "{}",
                    json!({"kind":"layer_discriminator","mode":mode,"layer":layer,"max_abs":error(&baseline[i],&got.values)})
                );
            }
        }
        Ok(())
    }
}
#[cfg(feature = "perf-ablations")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    probe::run()
}
#[cfg(not(feature = "perf-ablations"))]
fn main() {
    eprintln!("fn_executor_debug requires --features perf-ablations");
    std::process::exit(2);
}

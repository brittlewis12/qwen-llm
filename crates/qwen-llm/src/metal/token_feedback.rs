//! Model-independent GPU sampling and asynchronous token readback. Each request
//! owns its sampler and buffers; neither weights nor another member's state is
//! stored here. Candidate arithmetic is F32, unlike CPU sampler-v1's F64 weights.
use super::*;
use crate::sampling::SamplingConfig;
use std::sync::mpsc::{Receiver, channel};

pub const GPU_SAMPLER_ALGORITHM_VERSION: u32 = 2;

pub struct GpuTokenSampler {
    config: SamplingConfig,
    vocab: usize,
    scratch: [MetalTensor; 2],
    radix_chunk_hist: MetalTensor,
    radix_chunk_base: MetalTensor,
    radix_bucket_totals: MetalTensor,
    radix_bucket_offsets: MetalTensor,
    weights: MetalTensor,
    block_sums: MetalTensor,
    block_prefix: MetalTensor,
    total: MetalTensor,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SortArgs {
    n: u32,
    keep: u32,
    first: u32,
    run: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RadixArgs {
    n: u32,
    shift: u32,
    buckets: u32,
    mask: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleArgs {
    n: u32,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    uniform: f32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WeightArgs {
    n: u32,
    temperature: f32,
    min_p: f32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockArgs {
    n: u32,
    block: u32,
    blocks: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleBlockArgs {
    n: u32,
    top_p: f32,
    uniform: f32,
    block: u32,
    blocks: u32,
    pad: u32,
}
const BLOCK: usize = 512;

impl GpuTokenSampler {
    pub fn new(
        ctx: &MetalContext,
        vocab: usize,
        config: SamplingConfig,
    ) -> Result<Self, MetalError> {
        let config = config.validate().map_err(|e| MetalError::BadShape {
            kernel: "gpu_sampler",
            detail: e.to_string(),
        })?;
        if vocab == 0 || vocab > i32::MAX as usize {
            return Err(MetalError::BadShape {
                kernel: "gpu_sampler",
                detail: "invalid vocabulary width".into(),
            });
        }
        for kernel in [
            "kernel_token_sort_tiles",
            "kernel_token_merge_runs",
            "kernel_token_sample_sorted",
            "kernel_argmax_f32_greedy",
            "kernel_token_radix_build",
            "kernel_token_radix_chunk_hist",
            "kernel_token_radix_chunk_base",
            "kernel_token_radix_bucket_prefix",
            "kernel_token_radix_stable_scatter",
            "kernel_token_radix_pairs",
            "kernel_token_weight_sorted",
            "kernel_token_sample_weighted",
            "kernel_token_block_sums",
            "kernel_token_block_prefix",
            "kernel_token_sample_blocks",
        ] {
            ctx.pipeline(kernel)?;
        }
        let groups = vocab.div_ceil(256);
        let n = groups * 256 * 2;
        Ok(Self {
            config,
            vocab,
            scratch: [
                MetalTensor::zeros_f32(ctx, vec![n as u64])?,
                MetalTensor::zeros_f32(ctx, vec![n as u64])?,
            ],
            radix_chunk_hist: MetalTensor::zeros_i32(ctx, vec![(groups * 2048) as u64])?,
            radix_chunk_base: MetalTensor::zeros_i32(ctx, vec![(groups * 2048) as u64])?,
            radix_bucket_totals: MetalTensor::zeros_i32(ctx, vec![2048])?,
            radix_bucket_offsets: MetalTensor::zeros_i32(ctx, vec![2048])?,
            weights: MetalTensor::zeros_f32(ctx, vec![vocab as u64])?,
            block_sums: MetalTensor::zeros_f32(ctx, vec![vocab.div_ceil(BLOCK) as u64])?,
            block_prefix: MetalTensor::zeros_f32(ctx, vec![vocab.div_ceil(BLOCK) as u64])?,
            total: MetalTensor::zeros_f32(ctx, vec![1])?,
        })
    }
    /// The caller must retain this sampler, logits and output until completion.
    /// Pre-encoding a later command is safe; commands sharing this scratch must
    /// be committed in order, with at most one executing at a time.
    pub fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        logits: &MetalTensor,
        output: &MetalTensor,
        uniform: f32,
    ) -> Result<(), MetalError> {
        checks::require_serial("gpu_sampler", enc)?;
        checks::check_tensor(
            "gpu_sampler",
            logits,
            GgmlType::F32,
            &[self.vocab as u64],
            false,
            "logits",
        )?;
        checks::check_tensor("gpu_sampler", output, GgmlType::I32, &[1], true, "token")?;
        checks::check_disjoint("gpu_sampler", output, &[(logits, "logits")])?;
        let device = ctx.device.registryID();
        if enc.ended
            || enc.parent_command_buffer().device().registryID() != device
            || [
                logits,
                output,
                &self.scratch[0],
                &self.radix_chunk_hist,
                &self.radix_chunk_base,
                &self.radix_bucket_totals,
                &self.radix_bucket_offsets,
                &self.weights,
                &self.block_sums,
                &self.block_prefix,
                &self.total,
            ]
            .iter()
            .any(|t| t.buffer.device().registryID() != device)
        {
            return Err(MetalError::BadShape {
                kernel: "gpu_sampler",
                detail:
                    "encoder or sampler bindings belong to another device, or encoder has ended"
                        .into(),
            });
        }
        if self.config.temperature == 0. {
            return encode_argmax_f32_greedy(ctx, enc, logits, output, 1, self.vocab);
        }
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err(MetalError::BadShape {
                kernel: "gpu_sampler",
                detail: "uniform must be in [0,1)".into(),
            });
        }
        let bounded = self.config.top_k > 0 && self.config.top_k <= 128;
        let mut target = 0;
        let mut n;
        if bounded {
            let keep = self.config.top_k.min(self.vocab);
            n = self.vocab;
            loop {
                enc.set_pipeline(&ctx.pipeline("kernel_token_sort_tiles")?);
                enc.set_bytes(
                    0,
                    &SortArgs {
                        n: n as u32,
                        keep: keep as u32,
                        first: u32::from(n == self.vocab && target == 0),
                        run: 0,
                    },
                );
                enc.set_tensor(1, logits);
                enc.set_tensor(2, &self.scratch[1 - target]);
                enc.set_tensor(3, &self.scratch[target]);
                enc.dispatch(
                    MTLSize {
                        width: n.div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                let next = n.div_ceil(256) * keep;
                if n <= 256 {
                    n = next.min(n);
                    break;
                }
                n = next;
                target = 1 - target;
            }
        } else {
            let groups = self.vocab.div_ceil(256);
            let threads = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            let mut src = 0usize;
            enc.set_pipeline(&ctx.pipeline("kernel_token_radix_build")?);
            enc.set_bytes(
                2,
                &RadixArgs {
                    n: self.vocab as u32,
                    shift: 0,
                    buckets: 0,
                    mask: 0,
                    groups: groups as u32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                },
            );
            enc.set_tensor(0, logits);
            enc.set_tensor(1, &self.scratch[0]);
            enc.dispatch(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                threads,
            );
            enc.buffer_barrier();
            for (shift, bits) in [(32u32, 11u32), (43, 11), (54, 10)] {
                let buckets = 1u32 << bits;
                let args = RadixArgs {
                    n: self.vocab as u32,
                    shift,
                    buckets,
                    mask: buckets - 1,
                    groups: groups as u32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                };
                enc.set_pipeline(&ctx.pipeline("kernel_token_radix_chunk_hist")?);
                enc.set_bytes(2, &args);
                enc.set_tensor(0, &self.scratch[src]);
                enc.set_tensor(1, &self.radix_chunk_hist);
                enc.dispatch(
                    MTLSize {
                        width: groups,
                        height: 1,
                        depth: 1,
                    },
                    threads,
                );
                enc.buffer_barrier();
                enc.set_pipeline(&ctx.pipeline("kernel_token_radix_chunk_base")?);
                enc.set_bytes(3, &args);
                enc.set_tensor(0, &self.radix_chunk_hist);
                enc.set_tensor(1, &self.radix_chunk_base);
                enc.set_tensor(2, &self.radix_bucket_totals);
                enc.dispatch(
                    MTLSize {
                        width: (buckets as usize).div_ceil(1024),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.buffer_barrier();
                enc.set_pipeline(&ctx.pipeline("kernel_token_radix_bucket_prefix")?);
                enc.set_bytes(2, &args);
                enc.set_tensor(0, &self.radix_bucket_totals);
                enc.set_tensor(1, &self.radix_bucket_offsets);
                enc.dispatch(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.buffer_barrier();
                enc.set_pipeline(&ctx.pipeline("kernel_token_radix_stable_scatter")?);
                enc.set_bytes(4, &args);
                enc.set_tensor(0, &self.scratch[src]);
                enc.set_tensor(1, &self.scratch[1 - src]);
                enc.set_tensor(2, &self.radix_chunk_base);
                enc.set_tensor(3, &self.radix_bucket_offsets);
                enc.dispatch(
                    MTLSize {
                        width: groups,
                        height: 1,
                        depth: 1,
                    },
                    threads,
                );
                enc.buffer_barrier();
                src = 1 - src;
            }
            enc.set_pipeline(&ctx.pipeline("kernel_token_radix_pairs")?);
            enc.set_bytes(
                2,
                &RadixArgs {
                    n: self.vocab as u32,
                    shift: 0,
                    buckets: 0,
                    mask: 0,
                    groups: groups as u32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                },
            );
            enc.set_tensor(0, &self.scratch[src]);
            enc.set_tensor(1, &self.scratch[1 - src]);
            enc.dispatch(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                threads,
            );
            target = 1 - src;
            n = if self.config.top_k == 0 {
                self.vocab
            } else {
                self.config.top_k.min(self.vocab)
            };
        }
        enc.buffer_barrier();
        enc.set_pipeline(&ctx.pipeline("kernel_token_weight_sorted")?);
        enc.set_bytes(
            0,
            &WeightArgs {
                n: n as u32,
                temperature: self.config.temperature,
                min_p: self.config.min_p,
                pad: 0,
            },
        );
        enc.set_tensor(1, &self.scratch[target]);
        enc.set_tensor(2, &self.weights);
        enc.dispatch(
            MTLSize {
                width: n.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        let blocks = n.div_ceil(BLOCK);
        enc.buffer_barrier();
        enc.set_pipeline(&ctx.pipeline("kernel_token_block_sums")?);
        enc.set_bytes(
            0,
            &BlockArgs {
                n: n as u32,
                block: BLOCK as u32,
                blocks: blocks as u32,
                pad: 0,
            },
        );
        enc.set_tensor(1, &self.weights);
        enc.set_tensor(2, &self.block_sums);
        enc.dispatch(
            MTLSize {
                width: blocks.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.buffer_barrier();
        enc.set_pipeline(&ctx.pipeline("kernel_token_block_prefix")?);
        enc.set_bytes(
            0,
            &BlockArgs {
                n: n as u32,
                block: BLOCK as u32,
                blocks: blocks as u32,
                pad: 0,
            },
        );
        enc.set_tensor(1, &self.block_sums);
        enc.set_tensor(2, &self.block_prefix);
        enc.set_tensor(3, &self.total);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        enc.buffer_barrier();
        enc.set_pipeline(&ctx.pipeline("kernel_token_sample_blocks")?);
        enc.set_bytes(
            0,
            &SampleBlockArgs {
                n: n as u32,
                top_p: self.config.top_p,
                uniform: uniform.min(f32::from_bits(0x3f7fffff)),
                block: BLOCK as u32,
                blocks: blocks as u32,
                pad: 0,
            },
        );
        enc.set_tensor(1, &self.scratch[target]);
        enc.set_tensor(2, &self.weights);
        enc.set_tensor(3, &self.block_sums);
        enc.set_tensor(4, &self.block_prefix);
        enc.set_tensor(5, &self.total);
        enc.set_tensor(6, output);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{Sampler, SamplingUniform};

    #[test]
    fn gpu_sampler_rejects_unordered_and_aliasing_bindings() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        let gpu = GpuTokenSampler::new(
            &ctx,
            513,
            SamplingConfig {
                temperature: 0.8,
                top_k: 20,
                ..Default::default()
            },
        )
        .unwrap();
        let x = MetalTensor::zeros_f32(&ctx, vec![513]).unwrap();
        let y = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin_concurrent(&command);
        assert!(gpu.encode(&ctx, &enc, &x, &y, 0.5).is_err());
        enc.end();
        let enc = KernelEncoder::begin(&command);
        let mut alias = x.view_subrange(0, vec![1]);
        alias.dtype = GgmlType::I32;
        assert!(gpu.encode(&ctx, &enc, &x, &alias, 0.5).is_err());
        enc.end();
    }

    #[test]
    fn gpu_sampler_orders_candidates_filters_and_async_reads() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        let logits: Vec<f32> = (0..513)
            .map(|i| ((i * 167 % 521) as f32 - 260.) / 19.)
            .collect();
        let x = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&logits),
            vec![logits.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        for top_k in [1, 20, 128, 129, 0] {
            let config = SamplingConfig {
                temperature: 0.8,
                top_k,
                top_p: 0.95,
                min_p: 0.02,
                seed: 1,
            };
            let cpu = Sampler::new(config).unwrap();
            let gpu = GpuTokenSampler::new(&ctx, logits.len(), config).unwrap();
            for uniform in [0.0f32, 0.013, 0.1, 0.51, 0.9, 0.9999] {
                let output = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                gpu.encode(&ctx, &enc, &x, &output, uniform).unwrap();
                enc.end();
                let readback = AsyncTokenReadback::attach(&command, &output);
                command.commit();
                let token = readback.receive().unwrap();
                wait_completed(&command).unwrap();
                let expected = cpu
                    .diagnose(&logits, SamplingUniform::new(f64::from(uniform)).unwrap())
                    .unwrap()
                    .sampled
                    .token;
                assert_eq!(token, expected, "top_k={top_k} uniform={uniform}");
            }
        }
    }

    #[test]
    fn gpu_sampler_greedy_ties_nan_and_infinity_contracts() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        for values in [
            vec![-0., 0., 0.],
            vec![1., f32::INFINITY, f32::INFINITY],
            vec![0., f32::NAN, 2.],
            vec![f32::NEG_INFINITY; 3],
        ] {
            for config in [
                SamplingConfig::default(),
                SamplingConfig {
                    temperature: 0.8,
                    top_k: 20,
                    top_p: 0.95,
                    min_p: 0.02,
                    ..Default::default()
                },
                SamplingConfig {
                    temperature: 0.8,
                    top_k: 0,
                    top_p: 0.95,
                    min_p: 0.02,
                    ..Default::default()
                },
            ] {
                let gpu = GpuTokenSampler::new(&ctx, values.len(), config).unwrap();
                let x = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&values),
                    vec![values.len() as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let y = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                gpu.encode(&ctx, &enc, &x, &y, 0.).unwrap();
                enc.end();
                let readback = AsyncTokenReadback::attach(&command, &y);
                command.commit();
                let actual = readback.receive().unwrap();
                wait_completed(&command).unwrap();
                assert_eq!(
                    actual,
                    if values[1].is_nan() {
                        -2
                    } else if values[0] == f32::NEG_INFINITY && config.temperature > 0. {
                        i32::MIN
                    } else if values[0] == f32::NEG_INFINITY {
                        0
                    } else {
                        1
                    }
                );
            }
        }
    }

    #[test]
    fn gpu_sampler_unbounded_matches_host_across_distributions_and_uniforms() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        let vocab = 8192usize;
        let distributions: Vec<(&str, Vec<f32>)> = vec![
            ("flat", vec![0.0; vocab]),
            (
                "peaked",
                (0..vocab)
                    .map(|i| if i == 123 { 10.0 } else { -10.0 })
                    .collect(),
            ),
            ("ties", (0..vocab).map(|i| (i % 3) as f32).collect()),
            (
                "signed_zero",
                (0..vocab)
                    .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
                    .collect(),
            ),
            (
                "ramp",
                (0..vocab).map(|i| (i as f32) / 100.0 - 40.0).collect(),
            ),
            (
                "sparse_inf",
                (0..vocab)
                    .map(|i| {
                        if i % 1000 == 0 {
                            f32::INFINITY
                        } else {
                            (i as f32) * 0.001
                        }
                    })
                    .collect(),
            ),
        ];
        let presets = [
            (
                "v41",
                SamplingConfig {
                    temperature: 1.,
                    top_k: 0,
                    top_p: 1.,
                    min_p: 0.,
                    seed: 1,
                },
            ),
            (
                "glm",
                SamplingConfig {
                    temperature: 1.,
                    top_k: 0,
                    top_p: 0.95,
                    min_p: 0.,
                    seed: 1,
                },
            ),
            (
                "nucleus_minp",
                SamplingConfig {
                    temperature: 1.,
                    top_k: 0,
                    top_p: 0.9,
                    min_p: 0.01,
                    seed: 1,
                },
            ),
            (
                "top20",
                SamplingConfig {
                    temperature: 1.,
                    top_k: 20,
                    top_p: 0.95,
                    min_p: 0.,
                    seed: 1,
                },
            ),
        ];
        let mut uniforms: Vec<f32> = (0..64).map(|k| k as f32 / 64.0).collect();
        uniforms.extend([0.0, 0.5, 0.9999]);
        let (mut total, mut mismatches) = (0usize, 0usize);
        for (name, logits) in &distributions {
            let x = MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(logits),
                vec![vocab as u64],
                GgmlType::F32,
            )
            .unwrap();
            for (preset, config) in presets {
                let cpu = Sampler::new(config).unwrap();
                let gpu = GpuTokenSampler::new(&ctx, vocab, config).unwrap();
                for &uniform in &uniforms {
                    let output = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
                    let command = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&command);
                    gpu.encode(&ctx, &enc, &x, &output, uniform).unwrap();
                    enc.end();
                    let readback = AsyncTokenReadback::attach(&command, &output);
                    command.commit();
                    let token = readback.receive().unwrap();
                    wait_completed(&command).unwrap();
                    let expected = cpu
                        .diagnose(logits, SamplingUniform::new(f64::from(uniform)).unwrap())
                        .unwrap()
                        .sampled
                        .token;
                    total += 1;
                    if token != expected {
                        mismatches += 1;
                    }
                }
            }
        }
        // F32 candidate arithmetic can flip a token exactly at a cumulative
        // boundary versus the host's F64 sums; the strict exactness test above
        // pins the standard presets. Guard against a broken sort or scan.
        assert!(
            mismatches * 50 <= total,
            "f32 sampler boundary mismatches {mismatches}/{total}"
        );
    }
}

/// A dedicated output slot is retained by the completion block. Encoding the
/// next step never races the readback; that command is committed only after the
/// receiver accepts the current token and its stop/stream decision.
pub struct AsyncTokenReadback {
    receiver: Receiver<Result<i32, ()>>,
}
impl AsyncTokenReadback {
    pub fn attach(command: &ProtocolObject<dyn MTLCommandBuffer>, token: &MetalTensor) -> Self {
        assert_eq!(token.dtype, GgmlType::I32);
        assert_eq!(token.n_elements(), 1);
        assert!(
            token.offset.is_multiple_of(4) && token.offset as usize + 4 <= token.buffer.length()
        );
        assert_eq!(
            token.buffer.storageMode(),
            MTLStorageMode::Shared,
            "token readback requires shared storage"
        );
        let (sender, receiver) = channel();
        let buffer = token.buffer.clone();
        let offset = token.offset as usize;
        let block = RcBlock::new(
            move |command: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let command = unsafe { command.as_ref() };
                let value = if command.status() == objc2_metal::MTLCommandBufferStatus::Completed {
                    Ok(unsafe {
                        *buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(offset)
                            .cast::<i32>()
                    })
                } else {
                    Err(())
                };
                let _ = sender.send(value);
            },
        );
        unsafe {
            command.addCompletedHandler(&*block as *const _ as *mut _);
        }
        Self { receiver }
    }
    pub fn receive(&self) -> Result<i32, MetalError> {
        self.receiver
            .recv()
            .ok()
            .and_then(Result::ok)
            .ok_or_else(|| {
                MetalError::Counter("GPU token command failed or completion was lost".into())
            })
    }
}

/// Fast, distribution-exact categorical draw for the **unforced** agent preset
/// (`temperature > 0`, `top_p = 1`, `top_k = 0`, `min_p = 0`): Gumbel-max over
/// the whole vocabulary, `O(V)`, one member-axis dispatch plus a small final
/// reduce. Every candidate's Gumbel noise is a counter-based hash of
/// `(seed, member, index)`, so a seeded run is deterministic on this path.
///
/// This is **not** token-identical with [`GpuTokenSampler`] or the host sampler
/// at the same seed; it is statistically exact from `softmax(logits / T)`. The
/// exact path stays available through [`fast_sampler_allowed`]'s callers.
pub struct FastGpuSampler {
    vocab: usize,
    members: usize,
    block: usize,
    blocks: usize,
    partial: MetalTensor,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GumbelArgs {
    seed: u64,
    vocab: u32,
    members: u32,
    blocks: u32,
    block: u32,
    inv_temp: f32,
    pad: u32,
}

impl FastGpuSampler {
    pub const BLOCK: usize = 512;

    pub fn new(ctx: &MetalContext, vocab: usize, members: usize) -> Result<Self, MetalError> {
        if vocab == 0 || vocab > i32::MAX as usize || members == 0 {
            return Err(MetalError::BadShape {
                kernel: "fast_sampler",
                detail: "invalid vocabulary width or member count".into(),
            });
        }
        ctx.pipeline("kernel_token_gumbel_block")?;
        ctx.pipeline("kernel_token_gumbel_final")?;
        let blocks = vocab.div_ceil(Self::BLOCK);
        // token_pair is {float value; uint id}; two i32 words per entry.
        let partial = MetalTensor::zeros_i32(ctx, vec![(2 * blocks * members) as u64])?;
        Ok(Self {
            vocab,
            members,
            block: Self::BLOCK,
            blocks,
            partial,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn members(&self) -> usize {
        self.members
    }

    /// `logits` is `[members * vocab]` F32 (member rows contiguous), `tokens`
    /// `[members]` I32. All member rows are sampled under the same `seed`.
    pub fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        logits: &MetalTensor,
        tokens: &MetalTensor,
        seed: u64,
        temperature: f32,
    ) -> Result<(), MetalError> {
        checks::require_serial("fast_sampler", enc)?;
        checks::check_tensor(
            "fast_sampler",
            logits,
            GgmlType::F32,
            &[(self.members * self.vocab) as u64],
            false,
            "logits",
        )?;
        checks::check_tensor(
            "fast_sampler",
            tokens,
            GgmlType::I32,
            &[self.members as u64],
            true,
            "tokens",
        )?;
        checks::check_disjoint("fast_sampler", tokens, &[(logits, "logits")])?;
        if !(temperature > 0.0 && temperature.is_finite()) {
            return Err(MetalError::BadShape {
                kernel: "fast_sampler",
                detail: "temperature must be finite and positive".into(),
            });
        }
        let args = GumbelArgs {
            seed,
            vocab: self.vocab as u32,
            members: self.members as u32,
            blocks: self.blocks as u32,
            block: self.block as u32,
            inv_temp: 1.0 / temperature,
            pad: 0,
        };
        enc.set_pipeline(&ctx.pipeline("kernel_token_gumbel_block")?);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, logits);
        enc.set_tensor(2, &self.partial);
        enc.dispatch(
            MTLSize {
                width: self.blocks,
                height: self.members,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.set_pipeline(&ctx.pipeline("kernel_token_gumbel_final")?);
        enc.set_bytes(0, &args);
        enc.set_tensor(2, &self.partial);
        enc.set_tensor(3, tokens);
        enc.dispatch(
            MTLSize {
                width: self.members,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

/// Whether the fast Gumbel sampler may serve this request. Only the unforced
/// agent preset qualifies (`T>0`, `top_k=0`, `top_p=1`, `min_p=0`) and only when
/// the request carries **no explicit seed**; an explicit seed, `QWEN_FAST_SAMPLER=0`
/// or `QWEN_GPU_SAMPLER_EXACT=1` keeps the exact sampler. The serve default seed
/// (42) must not be confused with an explicit one; callers pass `explicit_seed`
/// from `request.seed.is_some()`.
pub fn fast_sampler_allowed(config: &SamplingConfig, explicit_seed: bool) -> bool {
    config.temperature > 0.0
        && config.temperature.is_finite()
        && config.top_k == 0
        && config.top_p >= 1.0
        && config.min_p == 0.0
        && !explicit_seed
        && std::env::var("QWEN_FAST_SAMPLER").as_deref() != Ok("0")
        && std::env::var("QWEN_GPU_SAMPLER_EXACT").as_deref() != Ok("1")
}

#[cfg(test)]
mod fast_sampler_tests {
    use super::*;

    fn context() -> Option<MetalContext> {
        crate::test_fixtures::metal_context_or_skip()
    }

    fn draw(
        ctx: &MetalContext,
        logits: &[f32],
        vocab: usize,
        members: usize,
        seed: u64,
        temperature: f32,
    ) -> Vec<i32> {
        let sampler = FastGpuSampler::new(ctx, vocab, members).unwrap();
        let x = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(logits),
            vec![(members * vocab) as u64],
            GgmlType::F32,
        )
        .unwrap();
        let tokens = MetalTensor::zeros_i32(ctx, vec![members as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        sampler
            .encode(ctx, &enc, &x, &tokens, seed, temperature)
            .unwrap();
        enc.end();
        crate::metal::commit_and_wait(&command).unwrap();
        crate::metal::test_support::tensor_f32_at_offset(&tokens)
            .iter()
            .map(|v| v.to_bits() as i32)
            .collect()
    }

    fn repeat_row(row: &[f32], members: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(row.len() * members);
        for _ in 0..members {
            v.extend_from_slice(row);
        }
        v
    }

    /// Chi-square goodness of fit against the exact softmax for flat, peaked
    /// and tied logits. Members are independent hashes of the same seed.
    #[test]
    fn fast_sampler_matches_softmax_distribution() {
        let Some(ctx) = context() else {
            return;
        };
        for (name, logits) in [
            ("flat", vec![0.0f32; 8]),
            ("peaked", vec![0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0]),
            ("sloped", vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
        ] {
            let vocab = logits.len();
            let members = 8192usize;
            let draws = draw(
                &ctx,
                &repeat_row(&logits, members),
                vocab,
                members,
                0x5EED_1234,
                1.0,
            );
            let mut counts = vec![0usize; vocab];
            for &t in &draws {
                assert!((0..vocab as i32).contains(&t), "{name}: token {t}");
                counts[t as usize] += 1;
            }
            let max = logits.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f64> = logits.iter().map(|v| ((v - max) as f64).exp()).collect();
            let total: f64 = exps.iter().sum();
            let n = draws.len() as f64;
            let mut chi2 = 0.0;
            for (c, e) in counts.iter().zip(&exps) {
                let expected = e / total * n;
                chi2 += (*c as f64 - expected).powi(2) / expected;
            }
            // df = 7: p > 0.001 needs chi2 < ~24.3; allow generous headroom.
            assert!(chi2 < 40.0, "{name}: chi^2 {chi2:.1} counts {counts:?}");
        }
    }

    #[test]
    fn fast_sampler_seed_is_deterministic_and_seed_sensitive() {
        let Some(ctx) = context() else {
            return;
        };
        let logits = repeat_row(&[0.0, 0.25, 1.0, 2.0, 0.5, -1.0], 64);
        let a = draw(&ctx, &logits, 6, 64, 7, 1.0);
        let b = draw(&ctx, &logits, 6, 64, 7, 1.0);
        assert_eq!(a, b, "same seed must reproduce the same tokens");
        let c = draw(&ctx, &logits, 6, 64, 8, 1.0);
        assert_ne!(a, c, "a different seed should move some draws");
    }

    #[test]
    fn fast_sampler_large_vocab_is_deterministic() {
        let Some(ctx) = context() else {
            return;
        };
        let vocab = 129_280usize;
        let logits: Vec<f32> = (0..vocab)
            .map(|i| (((i.wrapping_mul(2_654_435_761) % 1000) as f32) - 500.) / 37.)
            .collect();
        let row = repeat_row(&logits, 4);
        let a = draw(&ctx, &row, vocab, 4, 99, 1.0);
        let b = draw(&ctx, &row, vocab, 4, 99, 1.0);
        assert_eq!(a, b);
        assert!(a.iter().all(|&t| (0..vocab as i32).contains(&t)));
    }

    #[test]
    fn fast_sampler_policy_is_unforced_and_filter_free_only() {
        let base = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 42,
        };
        assert!(fast_sampler_allowed(&base, false));
        assert!(
            !fast_sampler_allowed(&base, true),
            "explicit seed forces exact"
        );
        let mut filtered = base;
        filtered.top_p = 0.95;
        assert!(!fast_sampler_allowed(&filtered, false));
        filtered = base;
        filtered.top_k = 200;
        assert!(!fast_sampler_allowed(&filtered, false));
        filtered = base;
        filtered.temperature = 0.0;
        assert!(!fast_sampler_allowed(&filtered, false));
    }
}

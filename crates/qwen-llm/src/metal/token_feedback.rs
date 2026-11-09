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
struct SampleArgs {
    n: u32,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    uniform: f32,
}

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
        ] {
            ctx.pipeline(kernel)?;
        }
        let n = vocab.div_ceil(256) * 256 * 2;
        Ok(Self {
            config,
            vocab,
            scratch: [
                MetalTensor::zeros_f32(ctx, vec![n as u64])?,
                MetalTensor::zeros_f32(ctx, vec![n as u64])?,
            ],
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
            || [logits, output, &self.scratch[0]]
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
        let keep = if bounded {
            self.config.top_k.min(self.vocab)
        } else {
            256
        };
        let mut n = self.vocab;
        let mut target = 0;
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
            if !bounded || n <= 256 {
                n = next.min(n);
                break;
            }
            n = next;
            target = 1 - target;
        }
        if !bounded {
            // Preserve padding sentinels during merges, then retain only real
            // vocabulary entries. Total-order ties include signed zero.
            n = self.vocab.div_ceil(256) * 256;
            let mut run = 256;
            while run < n {
                enc.set_pipeline(&ctx.pipeline("kernel_token_merge_runs")?);
                enc.set_bytes(
                    0,
                    &SortArgs {
                        n: n as u32,
                        keep: 0,
                        first: 0,
                        run: run as u32,
                    },
                );
                enc.set_tensor(1, &self.scratch[target]);
                enc.set_tensor(2, &self.scratch[1 - target]);
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
                target = 1 - target;
                run *= 2;
            }
            n = if self.config.top_k == 0 {
                self.vocab
            } else {
                self.config.top_k.min(self.vocab)
            };
        }
        enc.set_pipeline(&ctx.pipeline("kernel_token_sample_sorted")?);
        enc.set_bytes(
            0,
            &SampleArgs {
                n: n as u32,
                temperature: self.config.temperature,
                min_p: self.config.min_p,
                top_p: self.config.top_p,
                uniform: uniform.min(f32::from_bits(0x3f7fffff)),
            },
        );
        enc.set_tensor(1, &self.scratch[target]);
        enc.set_tensor(2, output);
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

//! Immutable PLE banks use the normal GGUF import/admission path and retain
//! their mmap. Per-sequence hashes, history and controls belong to the loop.
use super::*;
use crate::metal::{Buffer, MetalGgufBacking};
use crate::qwen4exp::PleConfig;
use objc2_metal::{
    MTLArgumentEncoder, MTLComputeCommandEncoder, MTLFunction, MTLLibrary, MTLResourceUsage,
};
use std::sync::{Arc, Mutex, Weak};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    table_offset: u64,
    tail_begin: u64,
    row_bytes: u32,
    ngram: u32,
    heads: u32,
    heads_per_ngram: u32,
    eos: u32,
    controls: u32,
    position: u32,
    pad: u32,
}

struct GpuPleBanks {
    source: Arc<memmap2::Mmap>,
    source_start: usize,
    source_len: usize,
    device: u64,
    table_offset: u64,
    tail_begin: u64,
    banks: Vec<MetalGgufBacking>,
    tail: Buffer,
    banks_arg: Buffer,
}

/// Factory siblings share only immutable registrations, never sequence state.
#[derive(Clone, Default)]
pub(crate) struct GpuPleBankCache(Arc<Mutex<Weak<GpuPleBanks>>>);

pub(super) struct GpuInputs {
    ple: Arc<GpuPleBanks>,
    controls_arg: Buffer,
    metadata: Buffer,
    kinds: Buffer,
    controls: Vec<MetalTensor>,
    history: MetalTensor,
    input: MetalTensor,
    packed: MetalTensor,
    args: Args,
}

impl GpuPleBankCache {
    pub(super) fn ready(&self) -> bool {
        self.0.lock().is_ok_and(|cached| cached.upgrade().is_some())
    }

    fn get(
        &self,
        ctx: &MetalContext,
        table: PleIq4NlTable<'_>,
    ) -> Result<Arc<GpuPleBanks>, Qwen4ExpTextSessionError> {
        let (mmap, shard, start) = table.gpu_backing().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("GPU PLE requires a retained GGUF mapping".into())
        })?;
        let mut cached = self
            .0
            .lock()
            .map_err(|_| Qwen4ExpTextSessionError::Invalid("GPU PLE bank cache poisoned".into()))?;
        if let Some(banks) = cached.upgrade() {
            if banks.device != ctx.device.registryID()
                || !Arc::ptr_eq(&banks.source, mmap)
                || banks.source_start != start
                || banks.source_len != table.gpu_bytes().len()
            {
                return invalid("GPU PLE bank cache belongs to another table or device");
            }
            return Ok(banks);
        }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let base = start / page * page;
        let data_end = start + table.gpu_bytes().len();
        let end = (data_end.div_ceil(page) * page).min(mmap.len() / page * page);
        const BANK: usize = 1 << 30;
        if (data_end - base).div_ceil(BANK) > 32 {
            return invalid("GPU PLE mapping exceeds bank/page bounds");
        }
        let mut banks = Vec::new();
        for offset in (base..end).step_by(BANK) {
            banks.push(ctx.gguf_no_copy_window(
                mmap.clone(),
                shard,
                offset,
                (end - offset).min(BANK),
                16,
            )?);
        }
        // The existing import path requires complete mapped pages. Copy only a
        // final partial page, never a large table/window, when the tensor ends
        // at EOF. This also supports small GGUF fixtures safely.
        let has_tail = data_end > end;
        let tail = ctx.buffer_from(if has_tail {
            &mmap[end..data_end]
        } else {
            &[0u8]
        })?;
        let bank_function = ctx
            .library
            .newFunctionWithName(&objc2_foundation::NSString::from_str(
                "kernel_qwen4exp_gpu_ple_rows",
            ))
            .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("GPU PLE kernel missing".into()))?;
        let bank_encoder = unsafe { bank_function.newArgumentEncoderWithBufferIndex(0) };
        let banks_arg = ctx.buffer_uninit(bank_encoder.encodedLength())?;
        unsafe {
            bank_encoder.setArgumentBuffer_offset(Some(&banks_arg), 0);
        }
        for (i, b) in banks.iter().enumerate() {
            unsafe {
                bank_encoder.setBuffer_offset_atIndex(Some(&b.buffer), 0, i);
            }
        }
        unsafe {
            bank_encoder.setBuffer_offset_atIndex(Some(&tail), 0, 32);
        }
        let ple = Arc::new(GpuPleBanks {
            source: mmap.clone(),
            source_start: start,
            source_len: table.gpu_bytes().len(),
            device: ctx.device.registryID(),
            table_offset: (start - base) as u64,
            tail_begin: if has_tail {
                (end - base) as u64
            } else {
                u64::MAX
            },
            banks,
            tail,
            banks_arg,
        });
        // Failed per-member setup and dropped siblings must release their
        // registrations; the factory itself retains no admitted GPU memory.
        *cached = Arc::downgrade(&ple);
        Ok(ple)
    }
}

impl GpuInputs {
    pub(super) fn reset_history(&self, workspace: &Qwen4ExpTextSessionMetalWorkspace) {
        let prior = workspace.zero_one.prior_tokens();
        let mut values = vec![0i32; self.args.ngram as usize];
        values[0] = prior.len() as i32;
        for (dst, &src) in values[1..].iter_mut().zip(prior) {
            *dst = src as i32;
        }
        // No committed loop command exists while resetting this cached input.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.history
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.history.offset as usize)
                    .cast::<i32>(),
                values.len(),
            );
        }
    }
    pub(super) fn new(
        ctx: &MetalContext,
        table: PleIq4NlTable<'_>,
        config: &PleConfig,
        workspace: &Qwen4ExpTextSessionMetalWorkspace,
    ) -> Result<Self, Qwen4ExpTextSessionError> {
        if !(2..=16).contains(&config.ngram_size) {
            return invalid("GPU PLE requires ngram size 2..=16");
        }
        let ple = workspace.gpu_ple_banks.get(ctx, table)?;
        let mut controls: Vec<_> = workspace
            .zero_one
            .gpu_route_statuses()
            .into_iter()
            .map(|t| (t, 0))
            .collect();
        for block in &workspace.post_ple {
            controls.extend(block.gpu_control_tensors());
        }
        if controls.len() > 96 {
            return invalid("GPU decode control count exceeds argument capacity");
        }
        let control_function = ctx
            .library
            .newFunctionWithName(&objc2_foundation::NSString::from_str(
                "kernel_qwen4exp_gpu_input_controls",
            ))
            .ok_or_else(|| {
                Qwen4ExpTextSessionError::Invalid("GPU input-control kernel missing".into())
            })?;
        let control_encoder = unsafe { control_function.newArgumentEncoderWithBufferIndex(0) };
        let controls_arg = ctx.buffer_uninit(control_encoder.encodedLength())?;
        unsafe {
            control_encoder.setArgumentBuffer_offset(Some(&controls_arg), 0);
        }
        for (i, (t, _)) in controls.iter().enumerate() {
            unsafe {
                control_encoder.setBuffer_offset_atIndex(Some(&t.buffer), t.offset as usize, i);
            }
        }
        let kinds = ctx.buffer_from(&controls.iter().map(|(_, k)| *k).collect::<Vec<u32>>())?;
        let metadata: Vec<u64> = config
            .multipliers
            .iter()
            .chain(&config.head_offsets)
            .chain(&config.head_vocab_sizes)
            .copied()
            .collect();
        let metadata = ctx.buffer_from(&metadata)?;
        let prior = workspace.zero_one.prior_tokens();
        let mut history = vec![0i32; config.ngram_size as usize];
        history[0] = prior.len() as i32;
        for (dst, &src) in history[1..].iter_mut().zip(prior) {
            *dst = src as i32;
        }
        let history = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&history),
            vec![history.len() as u64],
            GgmlType::I32,
        )?;
        let (input, packed) = workspace.zero_one.gpu_input_buffers();
        let args = Args {
            table_offset: ple.table_offset,
            tail_begin: ple.tail_begin,
            row_bytes: table.row_bytes() as u32,
            ngram: config.ngram_size,
            heads: config.head_count()? as u32,
            heads_per_ngram: config.heads_per_ngram,
            eos: config.eos_token_id,
            controls: controls.len() as u32,
            position: 0,
            pad: 0,
        };
        for name in [
            "kernel_qwen4exp_gpu_ple_rows",
            "kernel_qwen4exp_gpu_input_controls",
        ] {
            ctx.pipeline(name)?;
        }
        Ok(Self {
            ple,
            controls_arg,
            metadata,
            kinds,
            controls: controls.into_iter().map(|(t, _)| t).collect(),
            history,
            input,
            packed,
            args,
        })
    }

    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        feedback: &MetalTensor,
        position: usize,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        // Ordered direct prefix, before any recorded model scopes. Argument
        // pointees need explicit residency and fences on a concurrent pass.
        assert!(enc.recording.borrow().is_none());
        let args = Args {
            position: u32::try_from(position).map_err(|_| {
                Qwen4ExpTextSessionError::Invalid("GPU position exceeds u32".into())
            })?,
            ..self.args
        };
        enc.set_pipeline(&ctx.pipeline("kernel_qwen4exp_gpu_ple_rows")?);
        enc.set_buffer(0, &self.ple.banks_arg, 0);
        enc.set_buffer(1, &self.metadata, 0);
        enc.set_bytes(2, &args);
        enc.set_tensor(3, feedback);
        enc.set_tensor(4, &self.history);
        enc.set_tensor(5, &self.packed);
        for b in &self.ple.banks {
            enc.use_resource_read(&b.buffer);
        }
        enc.use_resource_read(&self.ple.tail);
        enc.dispatch(
            objc2_metal::MTLSize {
                width: args.heads as usize,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        if enc.causal_mode {
            enc.buffer_barrier();
        }
        enc.set_pipeline(&ctx.pipeline("kernel_qwen4exp_gpu_input_controls")?);
        enc.set_buffer(0, &self.controls_arg, 0);
        enc.set_buffer(1, &self.kinds, 0);
        enc.set_bytes(2, &args);
        enc.set_tensor(3, feedback);
        enc.set_tensor(4, &self.history);
        enc.set_tensor(5, &self.input);
        for t in &self.controls {
            let resource = ProtocolObject::<dyn objc2_metal::MTLResource>::from_ref(&*t.buffer);
            enc.raw
                .borrow()
                .useResource_usage(resource, MTLResourceUsage::Write);
        }
        enc.dispatch(
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        if enc.causal_mode {
            enc.buffer_barrier();
        }
        Ok(())
    }
}

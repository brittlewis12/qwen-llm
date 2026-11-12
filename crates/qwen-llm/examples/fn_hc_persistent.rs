#![cfg_attr(not(feature = "perf-ablations"), allow(unused))]
//! Isolated Flash-Next HC persistent read-site probe.
//!
//! Usage: `fn_hc_persistent MODEL [REPS=9] [CHAIN_SITES=47]`
//! Run under the canonical queued Metal lease. The probe imports only block-0
//! HC tensors, compares the incumbent norm/down+injection-prep/up+mix chain
//! with one MSL 3.2 task-ordered dispatch, and checks mixed and residual bits.
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState,
    MTLDevice, MTLLanguageVersion, MTLMathMode, MTLSize,
};
use qwen_llm::{
    gguf::GgufFile,
    metal::{Buffer, KernelEncoder, MetalContext, MetalTensor, Pipeline, commit_and_wait},
    qwen4exp_loader::Qwen4ExpModel,
    tensor::GgmlType,
};
use std::{path::Path, process::Command};

type Error = Box<dyn std::error::Error>;
const BRANCHES: usize = 4;
const HIDDEN: usize = 2560;
const RANK: usize = 320;
const HYPER: usize = BRANCHES * HIDDEN;
const UP_WORKERS: usize = 256;
const UP_START: usize = 4 + 1 + RANK + BRANCHES;
const TASKS: usize = UP_START + UP_WORKERS;
const THREADS: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PersistentArgs {
    task_count: u32,
    branch_count: u32,
    hidden_size: u32,
    low_rank: u32,
    eps: f32,
    prefetch_up: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormArgs {
    branch_count: u32,
    hidden_size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BranchArgs {
    branch_count: u32,
    hidden_size: u32,
}

struct HcWeights {
    norm: MetalTensor,
    down: MetalTensor,
    up: MetalTensor,
    inject: MetalTensor,
    inject_source_dtype: GgmlType,
}

struct Scratch {
    normalized: MetalTensor,
    low: MetalTensor,
    raw_gate: MetalTensor,
    injection: MetalTensor,
    mixed_trace: MetalTensor,
}

struct SyncStorage {
    next: Buffer,
    ready: Buffer,
    sites: usize,
}

#[derive(Clone)]
struct ArmOutput {
    gpu_us: f64,
    mixed: Vec<f32>,
    residual: Vec<f32>,
    normalized: Vec<f32>,
    low: Vec<f32>,
    raw_gate: Vec<f32>,
    injection: Vec<f32>,
}

fn thermal() -> Option<u32> {
    let output = Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .last()?
        .parse()
        .ok()
}

fn qwen_model_processes() -> Result<Vec<String>, Error> {
    let output = Command::new("ps")
        .args(["-axo", "pid=,comm=,args="])
        .output()?;
    let mut found = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let mut fields = line.split_whitespace();
        let Some(pid) = fields.next() else { continue };
        let Some(comm) = fields.next() else { continue };
        if Path::new(comm).file_name().and_then(|s| s.to_str()) == Some("qwen") {
            found.push(format!("pid={pid} command={line}"));
        }
    }
    Ok(found)
}

fn mix_values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 200_001) as i32 - 100_000) as f32 * scale / 100_000.0
        })
        .collect()
}

fn import_tensor(
    ctx: &MetalContext,
    gguf: &GgufFile,
    desc: &qwen_llm::tensor::TensorDesc,
) -> Result<MetalTensor, Error> {
    Ok(MetalTensor::from_gguf_tensor(
        ctx,
        desc,
        gguf.try_slice(desc)?,
    )?)
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let count = tensor.n_elements() as usize;
    let offset = tensor.offset as usize;
    // SAFETY: callers wait for command completion before reading shared output.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<f32>(),
            count,
        )
        .to_vec()
    }
}

fn write_f32(tensor: &MetalTensor, values: &[f32]) {
    assert_eq!(tensor.n_elements() as usize, values.len());
    // SAFETY: the previous command was completed; this is a shared writable
    // activation buffer with a validated element count.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.len(),
        );
    }
}

unsafe fn fill_words(buffer: &Buffer, words: usize, value: u32) {
    // SAFETY: buffer is four-byte aligned and no prior command is in flight.
    let dst =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u32>(), words) };
    dst.fill(value);
}

fn compile_candidate(ctx: &mut MetalContext) -> Result<Pipeline, Error> {
    let original = ctx.library.clone();
    let result = (|| {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        options.setMathMode(MTLMathMode::Fast);
        ctx.library = ctx
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("fn_hc_persistent.metal")),
                Some(&options),
            )
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok::<_, Error>(ctx.pipeline("kernel_fn_hc_persistent")?)
    })();
    ctx.library = original;
    result
}

fn new_scratch(ctx: &MetalContext, sites: usize) -> Result<Scratch, Error> {
    Ok(Scratch {
        normalized: MetalTensor::zeros_f32(ctx, vec![HYPER as u64])?,
        low: MetalTensor::zeros_f32(ctx, vec![RANK as u64])?,
        raw_gate: MetalTensor::zeros_f32(ctx, vec![HYPER as u64])?,
        injection: MetalTensor::zeros_f32(ctx, vec![BRANCHES as u64])?,
        mixed_trace: MetalTensor::zeros_f32(ctx, vec![(sites * HIDDEN) as u64])?,
    })
}

fn new_sync_storage(ctx: &MetalContext, sites: usize) -> Result<SyncStorage, Error> {
    Ok(SyncStorage {
        // Per site: topological task ticket, then down completion count.
        next: ctx.buffer_uninit(sites * 8)?,
        ready: ctx.buffer_uninit(sites * TASKS * 4)?,
        sites,
    })
}

fn reset_sync(sync: &SyncStorage) {
    unsafe {
        fill_words(&sync.next, sync.sites * 2, 0);
        fill_words(&sync.ready, sync.sites * TASKS, 0);
    }
}

fn encode_norm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &HcWeights,
    hyper: &MetalTensor,
    scratch: &Scratch,
    eps: f32,
) -> Result<(), Error> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_rms_norm_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &NormArgs {
            branch_count: BRANCHES as u32,
            hidden_size: HIDDEN as u32,
            eps,
        },
    );
    enc.set_tensor(1, hyper);
    enc.set_tensor(2, &weights.norm);
    enc.set_tensor(3, &scratch.normalized);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    let simdgroups = threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: BRANCHES,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_down_inject(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &HcWeights,
    scratch: &Scratch,
) -> Result<(), Error> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_down_silu_inject_bf16")?;
    enc.set_pipeline(&pipeline);
    enc.set_tensor(0, &weights.down);
    enc.set_tensor(1, &scratch.normalized);
    enc.set_tensor(2, &scratch.low);
    enc.set_tensor(3, &weights.inject);
    enc.set_tensor(4, &scratch.injection);
    enc.set_threadgroup_memory(0, 0);
    enc.dispatch(
        MTLSize {
            width: RANK + BRANCHES,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_up(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &HcWeights,
    scratch: &Scratch,
    mixed: &MetalTensor,
) -> Result<(), Error> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_up_mix_bf16_k320")?;
    enc.set_pipeline(&pipeline);
    enc.set_tensor(0, &weights.up);
    enc.set_tensor(1, &scratch.low);
    enc.set_tensor(2, &scratch.normalized);
    enc.set_tensor(3, &scratch.raw_gate);
    enc.set_tensor(4, mixed);
    enc.dispatch(
        MTLSize {
            width: HIDDEN / 4,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_inject(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    block_output: &MetalTensor,
    injection: &MetalTensor,
    hyper: &MetalTensor,
) -> Result<(), Error> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_inject_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &BranchArgs {
            branch_count: BRANCHES as u32,
            hidden_size: HIDDEN as u32,
        },
    );
    enc.set_tensor(1, block_output);
    enc.set_tensor(2, injection);
    enc.set_tensor(3, hyper);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024) as usize;
    enc.dispatch(
        MTLSize {
            width: HIDDEN.div_ceil(threads),
            height: BRANCHES,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_incumbent_site(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &HcWeights,
    hyper: &MetalTensor,
    block_output: &MetalTensor,
    scratch: &Scratch,
    mixed: &MetalTensor,
    eps: f32,
) -> Result<(), Error> {
    encode_norm(ctx, enc, weights, hyper, scratch, eps)?;
    encode_down_inject(ctx, enc, weights, scratch)?;
    encode_up(ctx, enc, weights, scratch, mixed)?;
    encode_inject(ctx, enc, block_output, &scratch.injection, hyper)
}

fn encode_persistent_site(
    enc: &KernelEncoder,
    pipeline: &Pipeline,
    weights: &HcWeights,
    hyper: &MetalTensor,
    scratch: &Scratch,
    mixed: &MetalTensor,
    sync: &SyncStorage,
    site: usize,
    eps: f32,
    prefetch_up: bool,
) {
    enc.set_pipeline(pipeline);
    enc.set_buffer(0, &sync.next, (site * 8) as u64);
    enc.set_buffer(1, &sync.ready, (site * TASKS * 4) as u64);
    enc.set_tensor(2, hyper);
    enc.set_tensor(3, &weights.norm);
    enc.set_tensor(4, &scratch.normalized);
    enc.set_tensor(5, &weights.down);
    enc.set_tensor(6, &scratch.low);
    enc.set_tensor(7, &weights.inject);
    enc.set_tensor(8, &scratch.injection);
    enc.set_tensor(9, &weights.up);
    enc.set_tensor(10, &scratch.raw_gate);
    enc.set_tensor(11, mixed);
    enc.set_bytes(
        12,
        &PersistentArgs {
            task_count: TASKS as u32,
            branch_count: BRANCHES as u32,
            hidden_size: HIDDEN as u32,
            low_rank: RANK as u32,
            eps,
            prefetch_up: u32::from(prefetch_up),
        },
    );
    enc.dispatch(
        MTLSize {
            width: TASKS,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
}

fn mismatch(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

fn run_arm(
    ctx: &MetalContext,
    candidate: Option<&Pipeline>,
    prefetch_up: bool,
    weights: &HcWeights,
    hyper: &MetalTensor,
    block_output: &MetalTensor,
    scratch: &Scratch,
    sync: &SyncStorage,
    sites: usize,
    eps: f32,
) -> Result<ArmOutput, Error> {
    reset_sync(sync);
    let command = ctx.queue.commandBuffer().ok_or("no HC command buffer")?;
    let enc = KernelEncoder::begin(&command);
    for site in 0..sites {
        let mixed = scratch
            .mixed_trace
            .view_subrange((site * HIDDEN) as u64, vec![HIDDEN as u64]);
        if let Some(pipeline) = candidate {
            encode_persistent_site(
                &enc,
                pipeline,
                weights,
                hyper,
                scratch,
                &mixed,
                sync,
                site,
                eps,
                prefetch_up,
            );
            encode_inject(ctx, &enc, block_output, &scratch.injection, hyper)?;
        } else {
            encode_incumbent_site(
                ctx,
                &enc,
                weights,
                hyper,
                block_output,
                scratch,
                &mixed,
                eps,
            )?;
        }
    }
    enc.end();
    commit_and_wait(&command)?;
    let elapsed_us = (command.GPUEndTime() - command.GPUStartTime()) * 1e6;
    let mixed = scratch
        .mixed_trace
        .view_subrange(0, vec![(sites * HIDDEN) as u64]);
    Ok(ArmOutput {
        gpu_us: elapsed_us,
        mixed: read_f32(&mixed),
        residual: read_f32(hyper),
        normalized: read_f32(&scratch.normalized),
        low: read_f32(&scratch.low),
        raw_gate: read_f32(&scratch.raw_gate),
        injection: read_f32(&scratch.injection),
    })
}

fn emit_mismatch(label: &str, incumbent: &[f32], candidate: &[f32]) -> Result<(), Error> {
    if let Some(index) = mismatch(incumbent, candidate) {
        return Err(format!(
            "{label} differs at {index}: incumbent={:#010x}, candidate={:#010x}",
            incumbent[index].to_bits(),
            candidate[index].to_bits()
        )
        .into());
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().collect();
    let model_path = args.get(1).ok_or("MODEL [REPS=31] [CHAIN_SITES=47]")?;
    let reps = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(31usize);
    let chain_sites = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(47usize);
    if reps < 3 || chain_sites < 1 || chain_sites > 47 {
        return Err("reps must be >=3 and chain sites must be in1..=47".into());
    }
    let other_models = qwen_model_processes()?;
    if !other_models.is_empty() {
        return Err(format!(
            "refusing Flash-Next load while another Qwen model process is resident: {}",
            other_models.join("; ")
        )
        .into());
    }
    let ctx = MetalContext::new()?;
    let gguf = GgufFile::open(model_path)?;
    let model = Qwen4ExpModel::from_gguf(&gguf)?;
    let block = model.blocks.first().ok_or("missing block zero")?;
    let desc = &block.attention_residual;
    let inject_source_dtype = desc.inject.dtype;
    let injection = if inject_source_dtype == GgmlType::BF16 {
        let bytes = gguf.try_slice(&desc.inject)?;
        let bf16: &[u16] = bytemuck::cast_slice(bytes);
        let f32_values: Vec<f32> = bf16
            .iter()
            .map(|bits| f32::from_bits(u32::from(*bits) << 16))
            .collect();
        MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&f32_values),
            desc.inject.shape.clone(),
            GgmlType::F32,
        )?
    } else {
        import_tensor(&ctx, &gguf, &desc.inject)?
    };
    let weights = HcWeights {
        norm: import_tensor(&ctx, &gguf, &desc.norm)?,
        down: import_tensor(&ctx, &gguf, &desc.down)?,
        up: import_tensor(&ctx, &gguf, &desc.up)?,
        inject: injection,
        inject_source_dtype,
    };
    if (
        weights.norm.dtype,
        weights.down.dtype,
        weights.up.dtype,
        weights.inject.dtype,
    ) != (GgmlType::F32, GgmlType::BF16, GgmlType::BF16, GgmlType::F32)
    {
        return Err(format!(
            "block-0 HC tensors {} / {} / {} / {} bind as {:?}/{:?}/{:?}/{:?}, expected F32/BF16/BF16/F32 (raw injection {:?})",
            desc.norm.name,
            desc.down.name,
            desc.up.name,
            desc.inject.name,
            weights.norm.dtype,
            weights.down.dtype,
            weights.up.dtype,
            weights.inject.dtype,
            weights.inject_source_dtype,
        )
        .into());
    }
    let mut ctx = ctx;
    let pipeline = compile_candidate(&mut ctx)?;
    let eps = model.config.rms_norm_eps;
    let initial_hyper = mix_values(HYPER, 0x243f_6a88, 0.8);
    let block_data = mix_values(HIDDEN, 0x1319_8a2e, 0.3);
    let block_output = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&block_data),
        vec![HIDDEN as u64],
        GgmlType::F32,
    )?;
    let mut reference_hyper = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&initial_hyper),
        vec![HYPER as u64],
        GgmlType::F32,
    )?;
    let mut candidate_hyper = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&initial_hyper),
        vec![HYPER as u64],
        GgmlType::F32,
    )?;
    let reference_scratch = new_scratch(&ctx, chain_sites)?;
    let candidate_scratch = new_scratch(&ctx, chain_sites)?;
    let reference_sync = new_sync_storage(&ctx, chain_sites)?;
    let candidate_sync = new_sync_storage(&ctx, chain_sites)?;

    write_f32(&reference_hyper, &initial_hyper);
    let diagnostic_reference = run_arm(
        &ctx,
        None,
        false,
        &weights,
        &reference_hyper,
        &block_output,
        &reference_scratch,
        &reference_sync,
        1,
        eps,
    )?;
    write_f32(&candidate_hyper, &initial_hyper);
    let diagnostic_global_up = run_arm(
        &ctx,
        Some(&pipeline),
        false,
        &weights,
        &candidate_hyper,
        &block_output,
        &candidate_scratch,
        &candidate_sync,
        1,
        eps,
    )?;
    emit_mismatch(
        "diagnostic mixed without weight prefetch",
        &diagnostic_reference.mixed,
        &diagnostic_global_up.mixed,
    )?;
    emit_mismatch(
        "diagnostic residual without weight prefetch",
        &diagnostic_reference.residual,
        &diagnostic_global_up.residual,
    )?;
    emit_mismatch(
        "diagnostic gate without weight prefetch",
        &diagnostic_reference.raw_gate,
        &diagnostic_global_up.raw_gate,
    )?;
    println!(
        "{}",
        serde_json::json!({"kind":"fn_hc_persistent_no_prefetch_exact","bitwise":true})
    );

    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_hc_persistent_header",
            "device": ctx.describe(),
            "model": model_path,
            "block": 0,
            "weight_names": [desc.norm.name, desc.down.name, desc.up.name, desc.inject.name],
            "weight_dtypes": [format!("{:?}", weights.norm.dtype), format!("{:?}", weights.down.dtype), format!("{:?}", weights.up.dtype), format!("{:?}", weights.inject.dtype)],
            "injection_source_dtype": format!("{:?}", weights.inject_source_dtype),
            "weight_bytes": [weights.norm.n_bytes(), weights.down.n_bytes(), weights.up.n_bytes(), weights.inject.n_bytes()],
            "geometry": {"branches": BRANCHES, "hidden": HIDDEN, "rank": RANK},
            "persistent_tasks": TASKS,
            "up_worker_groups": UP_WORKERS,
            "threadgroup_threads": THREADS,
            "sites": chain_sites,
            "reps": reps,
            "msl": "3.2 runtime library",
            "thermal": thermal(),
        })
    );

    let mut single_reference = Vec::with_capacity(reps);
    let mut single_candidate = Vec::with_capacity(reps);
    let mut chain_reference = Vec::with_capacity(reps);
    let mut chain_candidate = Vec::with_capacity(reps);
    let mut site_reference_bits: Option<ArmOutput> = None;
    let mut site_candidate_bits: Option<ArmOutput> = None;
    let mut chain_reference_bits: Option<ArmOutput> = None;
    let mut chain_candidate_bits: Option<ArmOutput> = None;
    for round in 0..(reps + 2) {
        for candidate in [round % 2 == 1, round % 2 == 0] {
            let (hyper, scratch, sync) = if candidate {
                (&mut candidate_hyper, &candidate_scratch, &candidate_sync)
            } else {
                (&mut reference_hyper, &reference_scratch, &reference_sync)
            };
            write_f32(hyper, &initial_hyper);
            let site = run_arm(
                &ctx,
                candidate.then_some(&pipeline),
                candidate,
                &weights,
                hyper,
                &block_output,
                scratch,
                sync,
                1,
                eps,
            )?;
            write_f32(hyper, &initial_hyper);
            let chain = run_arm(
                &ctx,
                candidate.then_some(&pipeline),
                candidate,
                &weights,
                hyper,
                &block_output,
                scratch,
                sync,
                chain_sites,
                eps,
            )?;
            if round == 0 {
                if candidate {
                    site_candidate_bits = Some(site.clone());
                    chain_candidate_bits = Some(chain.clone());
                } else {
                    site_reference_bits = Some(site.clone());
                    chain_reference_bits = Some(chain.clone());
                }
            }
            if round >= 2 {
                if candidate {
                    single_candidate.push(site.gpu_us);
                    chain_candidate.push(chain.gpu_us);
                } else {
                    single_reference.push(site.gpu_us);
                    chain_reference.push(chain.gpu_us);
                }
            }
        }
    }
    let ref_site = site_reference_bits.ok_or("missing baseline site")?;
    let can_site = site_candidate_bits.ok_or("missing candidate site")?;
    let ref_chain = chain_reference_bits.ok_or("missing baseline chain")?;
    let can_chain = chain_candidate_bits.ok_or("missing candidate chain")?;
    for (label, a, b) in [
        (
            "site normalized",
            &ref_site.normalized,
            &can_site.normalized,
        ),
        ("site low", &ref_site.low, &can_site.low),
        ("site injection", &ref_site.injection, &can_site.injection),
        ("site raw gate", &ref_site.raw_gate, &can_site.raw_gate),
        ("site mixed", &ref_site.mixed, &can_site.mixed),
        ("site residual", &ref_site.residual, &can_site.residual),
        (
            "chain normalized",
            &ref_chain.normalized,
            &can_chain.normalized,
        ),
        ("chain low", &ref_chain.low, &can_chain.low),
        (
            "chain injection",
            &ref_chain.injection,
            &can_chain.injection,
        ),
        ("chain raw gate", &ref_chain.raw_gate, &can_chain.raw_gate),
        ("chain mixed", &ref_chain.mixed, &can_chain.mixed),
        ("chain residual", &ref_chain.residual, &can_chain.residual),
    ] {
        emit_mismatch(label, a, b)?;
    }
    single_reference.sort_by(f64::total_cmp);
    single_candidate.sort_by(f64::total_cmp);
    chain_reference.sort_by(f64::total_cmp);
    chain_candidate.sort_by(f64::total_cmp);
    let single_base = single_reference[reps / 2];
    let single_fast = single_candidate[reps / 2];
    let chain_base = chain_reference[reps / 2];
    let chain_fast = chain_candidate[reps / 2];
    println!(
        "{}",
        serde_json::json!({
            "kind": "fn_hc_persistent_result",
            "site_baseline_us": single_base,
            "site_candidate_us": single_fast,
            "site_speedup": single_base / single_fast,
            "chain_sites": chain_sites,
            "chain_baseline_us": chain_base,
            "chain_candidate_us": chain_fast,
            "chain_speedup": chain_base / chain_fast,
            "site_mixed_bitwise": true,
            "site_residual_bitwise": true,
            "chain_mixed_bitwise": true,
            "chain_residual_bitwise": true,
            "reps": reps,
            "thermal": thermal(),
            "site_samples_us": {"baseline": single_reference, "candidate": single_candidate},
            "chain_samples_us": {"baseline": chain_reference, "candidate": chain_candidate},
        })
    );
    Ok(())
}

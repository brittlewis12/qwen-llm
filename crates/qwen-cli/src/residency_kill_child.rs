//! Child process for the bounded kill check of ordinary command-buffer
//! wiring (PERF-ROADMAP 2026-10-05 #1). It maps a disposable file, wraps it
//! in a no-copy Metal buffer, and holds it wired the way serve's idle
//! keep-alive does, so an external observer can SIGKILL it and watch system
//! wired memory recover. It never uses a residency set. Holds the production
//! Metal lease like every engine process.

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice,
    MTLResourceOptions,
};
use qwen_llm::metal::{
    KeepAlivePulse, KernelEncoder, MetalContext, MetalTensor, ResidencyKeepAlive, encode_fill_f32,
};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    /// Pulse every 500 ms until killed (the keep-alive state).
    Pulse,
    /// Pulse a few times, then stop and idle until killed (unwired state).
    Stopped,
    /// Keep a long command in flight (marking the buffer used) until killed.
    Execute,
    /// Pulse for `--seconds`, then exit normally (control).
    Normal,
}

#[derive(Parser, Debug)]
pub struct ResidencyKillChildArgs {
    /// Disposable file to map (page-aligned length).
    #[arg(long)]
    file: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    /// JSON-lines status file the observer reads.
    #[arg(long)]
    status: PathBuf,
    /// Normal mode's pulsing duration.
    #[arg(long, default_value = "10")]
    seconds: u64,
}

fn status(path: &PathBuf, value: serde_json::Value) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{value}")?;
    file.sync_data()?;
    Ok(())
}

pub fn run(args: ResidencyKillChildArgs) -> Result<()> {
    let started = Instant::now();
    let since = || started.elapsed().as_secs_f64();
    let file = std::fs::File::open(&args.file)?;
    // SAFETY: a disposable file this test owns; read-only mapping.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let page = 16 * 1024;
    ensure!(
        mmap.len() >= page && mmap.len().is_multiple_of(page),
        "file length must be a positive multiple of 16 KiB"
    );
    let ctx = MetalContext::new().context("Metal context (production lease)")?;
    let ptr =
        std::ptr::NonNull::new(mmap.as_ptr() as *mut std::ffi::c_void).context("null mapping")?;
    // SAFETY: page-aligned read-only mapping that outlives the buffer (both
    // live until this function returns or the process dies).
    let buffer = unsafe {
        ctx.device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                mmap.len(),
                MTLResourceOptions::StorageModeShared,
                None,
            )
    }
    .context("no-copy buffer")?;
    status(
        &args.status,
        serde_json::json!({"phase": "ready", "pid": std::process::id(), "bytes": buffer.length(), "t": since()}),
    )?;
    let mut keep_alive = ResidencyKeepAlive::new(&ctx)?;
    let pulse = |keep_alive: &mut ResidencyKeepAlive| -> Result<()> {
        match keep_alive.pulse(&ctx, &[&buffer])? {
            KeepAlivePulse::Submitted | KeepAlivePulse::StillInFlight => Ok(()),
        }
    };
    match args.mode {
        Mode::Pulse | Mode::Normal => {
            let end = Duration::from_secs(args.seconds);
            let mut announced = false;
            loop {
                pulse(&mut keep_alive)?;
                if !announced {
                    status(
                        &args.status,
                        serde_json::json!({"phase": "pulsing", "t": since()}),
                    )?;
                    announced = true;
                }
                std::thread::sleep(Duration::from_millis(500));
                if matches!(args.mode, Mode::Normal) && started.elapsed() > end {
                    keep_alive.settle(Duration::from_secs(5))?;
                    status(
                        &args.status,
                        serde_json::json!({"phase": "exiting", "t": since()}),
                    )?;
                    return Ok(());
                }
            }
        }
        Mode::Stopped => {
            for _ in 0..6 {
                pulse(&mut keep_alive)?;
                std::thread::sleep(Duration::from_millis(500));
            }
            keep_alive.settle(Duration::from_secs(5))?;
            status(
                &args.status,
                serde_json::json!({"phase": "stopped", "t": since()}),
            )?;
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        Mode::Execute => {
            // Long commands: mark the buffer used, then many fills over a
            // small owned scratch. Resubmit as each completes.
            let scratch = MetalTensor::zeros_f32(&ctx, vec![4 << 20])?;
            loop {
                let command = ctx.queue.commandBuffer().context("command buffer")?;
                let enc = KernelEncoder::begin(&command);
                enc.use_resource_read(&buffer);
                for _ in 0..40_000 {
                    encode_fill_f32(&ctx, &enc, &scratch, 1.0)?;
                }
                enc.end();
                command.commit();
                status(
                    &args.status,
                    serde_json::json!({"phase": "executing", "committed_t": since(),
                        "running": command.status() != MTLCommandBufferStatus::Completed}),
                )?;
                qwen_llm::metal::wait_completed(&command)?;
                let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
                status(
                    &args.status,
                    serde_json::json!({"phase": "completed", "gpu_ms": gpu_ms, "t": since()}),
                )?;
            }
        }
    }
}

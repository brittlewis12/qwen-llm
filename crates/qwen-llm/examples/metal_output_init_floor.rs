use block2::RcBlock;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputePipelineState,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use qwen_llm::metal::{Buffer, KernelEncoder, MetalContext};
use serde_json::json;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct FillSample {
    wall_ms: f64,
    gpu_ms: f64,
}

fn parse_usize(flag: &str, default: usize) -> usize {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == flag {
            return args
                .next()
                .unwrap_or_else(|| panic!("{flag} requires a value"))
                .parse()
                .unwrap_or_else(|_| panic!("{flag} requires a positive integer"));
        }
    }
    default
}

fn parse_string(flag: &str, default: &str) -> String {
    let mut args = std::env::args();
    while let Some(argument) = args.next() {
        if argument == flag {
            return args
                .next()
                .unwrap_or_else(|| panic!("{flag} requires a value"));
        }
    }
    default.to_string()
}

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|argument| argument == flag)
}

fn host_page_size() -> usize {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(page > 0, "host page size query failed");
    page as usize
}

fn anonymous_no_copy_buffer(ctx: &MetalContext, bytes: usize) -> Buffer {
    let page = host_page_size();
    assert!(bytes > 0 && bytes.is_multiple_of(page));
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(raw, libc::MAP_FAILED, "anonymous mmap failed");
    let pointer = NonNull::new(raw).expect("anonymous mmap returned null");
    assert!((pointer.as_ptr() as usize).is_multiple_of(page));
    let deallocator: RcBlock<dyn Fn(NonNull<c_void>, usize)> =
        RcBlock::new(|pointer: NonNull<c_void>, length: usize| {
            let _ = unsafe { libc::munmap(pointer.as_ptr(), length) };
        });
    let buffer = unsafe {
        ctx.device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                pointer,
                bytes,
                MTLResourceOptions::StorageModeShared,
                Some(&deallocator),
            )
    };
    match buffer {
        Some(buffer) => buffer,
        None => {
            let result = unsafe { libc::munmap(pointer.as_ptr(), bytes) };
            assert_eq!(result, 0, "failed anonymous Metal backing cleanup");
            panic!("Metal rejected anonymous no-copy buffer");
        }
    }
}

fn fill(ctx: &MetalContext, buffer: &Buffer, elements: usize, value: f32) -> FillSample {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        value: f32,
    }

    let pipeline = ctx.pipeline("kernel_fill_f32").expect("fill pipeline");
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    let command = ctx.queue.commandBuffer().expect("fill command buffer");
    let encoder = KernelEncoder::begin(&command);
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(
        0,
        &Args {
            n: u32::try_from(elements).expect("element count fits u32"),
            value,
        },
    );
    encoder.set_buffer(1, buffer, 0);
    encoder.dispatch(
        MTLSize {
            width: elements.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end();

    let started = Instant::now();
    command.commit();
    command.waitUntilCompleted();
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
    FillSample { wall_ms, gpu_ms }
}

fn verify(buffer: &Buffer, elements: usize, expected: f32) -> u64 {
    let values =
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<f32>(), elements) };
    let stride = (elements / 4096).max(1);
    values.iter().step_by(stride).enumerate().fold(
        0xcbf2_9ce4_8422_2325u64,
        |state, (index, value)| {
            assert_eq!(value.to_bits(), expected.to_bits());
            state.wrapping_mul(0x0000_0100_0000_01b3) ^ u64::from(value.to_bits()) ^ index as u64
        },
    )
}

fn verify_full(buffer: &Buffer, elements: usize, expected: f32) -> String {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            buffer.contents().as_ptr().cast::<u8>(),
            elements * std::mem::size_of::<f32>(),
        )
    };
    let actual = blake3::hash(bytes);
    let chunk = vec![expected; 256 * 1024];
    let chunk_bytes = bytemuck::cast_slice::<f32, u8>(&chunk);
    let mut expected_hash = blake3::Hasher::new();
    let mut remaining = elements;
    while remaining >= chunk.len() {
        expected_hash.update(chunk_bytes);
        remaining -= chunk.len();
    }
    expected_hash.update(bytemuck::cast_slice(&chunk[..remaining]));
    let expected_hash = expected_hash.finalize();
    assert_eq!(actual, expected_hash, "full output digest mismatch");
    actual.to_hex().to_string()
}

fn main() {
    let mib = parse_usize("--mib", 512);
    let arm = parse_string("--arm", "device-zeroed");
    let full_verify = has_flag("--full-verify");
    assert!(mib > 0, "--mib must be positive");
    let bytes = mib.checked_mul(1024 * 1024).expect("byte count overflow");
    assert!(bytes.is_multiple_of(host_page_size()));
    assert!(bytes.is_multiple_of(std::mem::size_of::<f32>()));
    let elements = bytes / std::mem::size_of::<f32>();
    assert!(
        u32::try_from(elements).is_ok(),
        "fill kernel uses u32 indexing"
    );

    let ctx = MetalContext::new().expect("Metal context");
    let allocation_started = Instant::now();
    let buffer = match arm.as_str() {
        "device-zeroed" => ctx.buffer_uninit(bytes).expect("device buffer"),
        "wrapped-anon" => anonymous_no_copy_buffer(&ctx, bytes),
        _ => panic!("--arm must be device-zeroed or wrapped-anon"),
    };
    let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(buffer.length(), bytes);

    let first_value = 3.25f32;
    let first = fill(&ctx, &buffer, elements, first_value);
    let first_checksum = verify(&buffer, elements, first_value);
    let first_digest = full_verify.then(|| verify_full(&buffer, elements, first_value));
    let second_value = -1.5f32;
    let second = fill(&ctx, &buffer, elements, second_value);
    let second_checksum = verify(&buffer, elements, second_value);
    let second_digest = full_verify.then(|| verify_full(&buffer, elements, second_value));

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "qwen-metal-output-init-floor/v1",
            "arm": arm,
            "mib": mib,
            "bytes": bytes,
            "elements": elements,
            "allocation_ms": allocation_ms,
            "first_fill": {
                "wall_ms": first.wall_ms,
                "gpu_ms": first.gpu_ms,
                "checksum": first_checksum,
                "digest": first_digest,
            },
            "second_fill": {
                "wall_ms": second.wall_ms,
                "gpu_ms": second.gpu_ms,
                "checksum": second_checksum,
                "digest": second_digest,
            },
        }))
        .expect("serialize result")
    );
}

use super::*;
use std::time::Instant;

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    // Darwin ru_maxrss is bytes and is cumulative for this process, not a
    // sampled allocation peak or a claim about Metal physical residency.
    u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).unwrap()
}

fn assert_bytes(tensor: &MetalTensor, value: u8) {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<u8>(),
            tensor.n_bytes() as usize,
        )
    };
    assert!(bytes.iter().all(|&b| b == value));
}

#[test]
#[ignore = "synthetic Metal allocation correctness; production lease/wired gate, no weights"]
fn gpu_unstaged_zero_bytes_ownership_and_partial_failure_cleanup() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    admit(&ctx, 4 * 1024 * 1024).unwrap();
    for (dtype, count) in [
        (GgmlType::I8, 32771),
        (GgmlType::I32, 4099),
        (GgmlType::F16, 8195),
        (GgmlType::F32, 4099),
        (GgmlType::Q8_0, 64),
    ] {
        let tensor = MetalTensor::zeros_dtype_unstaged(&ctx, vec![count], dtype).unwrap();
        assert!(tensor.is_writable());
        assert_eq!(tensor.offset, 0);
        assert_eq!(tensor.dtype, dtype);
        assert_eq!(tensor.buffer.storageMode(), MTLStorageMode::Shared);
        assert_bytes(&tensor, 0);
        unsafe {
            std::ptr::write_bytes(
                tensor.buffer.contents().as_ptr().cast::<u8>(),
                0xa5,
                tensor.n_bytes() as usize,
            )
        };
        let second = MetalTensor::zeros_dtype_unstaged(&ctx, vec![count], dtype).unwrap();
        assert_ne!(second.buffer.contents(), tensor.buffer.contents());
        assert_bytes(&second, 0);
        assert_bytes(&tensor, 0xa5);
    }
    assert!(MetalTensor::zeros_dtype_unstaged(&ctx, vec![u64::MAX], GgmlType::F32).is_err());
    assert!(MetalTensor::zeros_dtype_unstaged(&ctx, vec![33], GgmlType::Q8_0).is_err());
    let plan = SessionMemoryPlan {
        cache_bytes: 147456,
        storage: K2KvStorage::F16,
    };
    let before = ctx.current_allocated_size();
    let mut constructed = 0;
    let tensors = plan
        .specs()
        .into_iter()
        .enumerate()
        .map(|(i, (dtype, shape))| {
            if i == 4 {
                return Err(invalid("injected allocation failure"));
            }
            constructed += 1;
            Ok(MetalTensor::zeros_dtype_unstaged(&ctx, shape, dtype)?)
        });
    assert!(SessionBuffers::from_tensors(tensors).is_err());
    assert_eq!(constructed, 4);
    assert_eq!(
        ctx.current_allocated_size(),
        before,
        "partial construction leaked storage"
    );
    let buffers = SessionBuffers::new(&ctx, &plan).unwrap();
    drop(buffers);
    assert_eq!(ctx.current_allocated_size(), before);
}

#[test]
#[ignore = "substantial allocation stress probe; fresh process, production lease/gate; default 32768-token 4.5GiB KV, no weights"]
fn gpu_k2_session_allocation_high_water_probe() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let capacity: u64 = std::env::var("K2_ALLOCATION_CAPACITY")
        .unwrap_or_else(|_| "32768".into())
        .parse()
        .unwrap();
    assert!(
        (1..=32768).contains(&capacity),
        "stress-probe bound, not a runtime limit"
    );
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let plan = SessionMemoryPlan {
        cache_bytes: 147456 * capacity,
        storage: K2KvStorage::F16,
    };
    let price = price_buffers(&ctx, &plan.buffer_bytes()).unwrap();
    let _transaction = ctx.begin_allocation_transaction();
    admit(&ctx, price).unwrap();
    let before_metal = ctx.current_allocated_size();
    let before_peak = peak_rss_bytes();
    let started = Instant::now();
    let buffers = SessionBuffers::new(&ctx, &plan).unwrap();
    let initialization_ms = started.elapsed().as_secs_f64() * 1e3;
    let after_peak = peak_rss_bytes();
    reconcile(&ctx, before_metal, price).unwrap();
    let metal_delta = ctx.current_allocated_size() - before_metal;
    assert!(buffers.cache.is_writable());
    assert_eq!(buffers.cache.n_bytes(), plan.cache_bytes);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            buffers.cache.buffer.contents().as_ptr().cast::<u8>(),
            plan.cache_bytes as usize,
        )
    };
    for offset in [0, bytes.len() / 2, bytes.len() - 16384] {
        assert!(bytes[offset..offset + 16384].iter().all(|&b| b == 0));
    }
    let report = serde_json::json!({
        "status":"passed", "scope":"synthetic_session_allocation_not_model_execution",
        "capacity":capacity, "logical_kv_bytes":plan.cache_bytes,
        "tensor_sized_host_staging_bytes":0, "metal_priced_upper_bytes":price,
        "observed_metal_delta_bytes":metal_delta, "initialization_ms":initialization_ms,
        "process_ru_maxrss_before_bytes":before_peak, "process_ru_maxrss_after_bytes":after_peak,
        "rss_policy":"Darwin_process_cumulative_high_water_not_exact_allocation_or_physical_residency",
        "paired_performance_claim":false,
    });
    drop(buffers);
    assert_eq!(ctx.current_allocated_size(), before_metal);
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Ok(path) = std::env::var("K2_ALLOCATION_EVIDENCE") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

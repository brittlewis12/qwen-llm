use super::*;

fn geometry() -> Qwen4ExpTextSessionMetalGeometry {
    Qwen4ExpTextSessionMetalGeometry::from_config(&Qwen4ExpConfig::flash_next_reference(), 8192)
        .unwrap()
}

#[test]
fn capture_cpu_layout_preserves_caller_tap_order() {
    let plan = Qwen4ExpCapturePlan::new(&geometry(), vec![45, 5, 25, 15, 35], 2048).unwrap();
    assert_eq!(plan.layers(), &[45, 5, 25, 15, 35]);
    assert_eq!(plan.feature_width(), 12800);
    assert_eq!(plan.capacity(), 2048);
    for row in [0, 1, 2047] {
        for tap in 0..5 {
            assert_eq!(plan.scatter_row(tap, row) * 2560, row * 12800 + tap * 2560);
        }
    }
    assert_eq!(plan.logical_bytes(), 2048 * (12800 + 2560 + 4 + 5) * 4);
    assert_eq!(plan.allocations()[0].1, 104_857_600);
}

#[test]
fn capture_cpu_layout_rejects_invalid_extents_and_layers() {
    let g = geometry();
    for layers in [vec![], vec![0], vec![48], vec![u32::MAX], vec![5, 5]] {
        assert!(Qwen4ExpCapturePlan::new(&g, layers, 8).is_err());
    }
    for capacity in [0, usize::MAX, i32::MAX as usize] {
        assert!(Qwen4ExpCapturePlan::new(&g, vec![5], capacity).is_err());
    }
    assert!(Qwen4ExpCapturePlan::new(&g, vec![47, 1], 1).is_ok());
}

#[test]
fn capture_cpu_layout_window_has_absolute_positions() {
    let p = Qwen4ExpCapturePlan::new(&geometry(), vec![5], 2048).unwrap();
    assert_eq!(p.tail_range(0, 1).unwrap(), 0..1);
    assert_eq!(p.tail_range(0, 2048).unwrap(), 0..2048);
    assert_eq!(p.tail_range(0, 4099).unwrap(), 2051..4099);
    assert_eq!(p.tail_range(4096, 7).unwrap(), 4096..4103);
    assert_eq!(p.tail_range(4096, 4099).unwrap(), 6147..8195);
    assert!(p.tail_range(0, 0).is_err());
    assert!(p.tail_range(usize::MAX, 1).is_err());
    assert!(p.tail_range(i32::MAX as usize, 1).is_err());
}

#[test]
fn capture_cpu_layout_chunk_intersections_cover_each_feature_once() {
    let p = Qwen4ExpCapturePlan::new(&geometry(), vec![45, 5, 25], 7).unwrap();
    let window = p.tail_range(20, 17).unwrap();
    assert_eq!(window, 30..37);
    let mut visited = vec![0; 7 * 3];
    // Earlier chunk, partial overlap, whole packed chunk, scalar tail.
    for chunk in [20..27, 27..32, 32..36, 36..37] {
        for layer in [5, 25, 45] {
            let tap = p.layers.iter().position(|&id| id == layer).unwrap();
            for position in overlap(&window, chunk.start, chunk.len()) {
                visited[p.scatter_row(tap, position - window.start)] += 1;
            }
        }
    }
    assert_eq!(visited, vec![1; 21]);
    assert!(overlap(&window, 37, 8).is_empty());
    assert!(overlap(&window, 0, 4).is_empty());
}

#[test]
#[ignore = "opt-in synthetic Metal capture; requires exclusive GPU lease"]
fn capture_metal_scalar_and_packed_means_match_token_major_layout() {
    use objc2_metal::MTLCommandQueue;
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let plan = Qwen4ExpCapturePlan::new(&geometry(), vec![45, 5, 25], 5).unwrap();
    let h = plan.hidden;
    let mut scalar = Qwen4ExpCaptureBank::allocate(&ctx, plan.clone()).unwrap();
    let mut packed = Qwen4ExpCaptureBank::allocate(&ctx, plan.clone()).unwrap();
    let mut inputs = Vec::new();
    for layer in [5, 25, 45] {
        let values: Vec<f32> = (0..8)
            .flat_map(|row| {
                (0..4).flat_map(move |stream| {
                    (0..h).map(move |d| {
                        layer as f32 + row as f32 * 8.0 + stream as f32 + d as f32 / 4096.0
                    })
                })
            })
            .collect();
        // Nonzero backing offset exercises source-view addressing too.
        let mut padded = vec![-999.0; 16];
        padded.extend_from_slice(&values);
        padded.extend([-999.0; 16]);
        let tensor = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&padded),
            vec![padded.len() as u64],
            GgmlType::F32,
        )
        .unwrap();
        inputs.push((layer, tensor, padded));
    }
    let scalar_sink = scalar.sink(103..108);
    let packed_sink = packed.sink(103..108);
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    for (layer, tensor, _) in &inputs {
        let hyper = tensor.view_subrange(16, vec![h as u64, 4, 8]);
        for row in 0..8 {
            let one = hyper.view_subrange((row * h * 4) as u64, vec![h as u64 * 4]);
            scalar_sink
                .encode(&ctx, &enc, *layer, &one, 100 + row, 1)
                .unwrap();
        }
        // Partial tail-window overlap, then scalar tail after the packed block.
        let batch = hyper.view_subrange(0, vec![h as u64 * 4, 7]);
        packed_sink
            .encode(&ctx, &enc, *layer, &batch, 100, 7)
            .unwrap();
        let last = hyper.view_subrange((7 * h * 4) as u64, vec![h as u64 * 4]);
        packed_sink
            .encode(&ctx, &enc, *layer, &last, 107, 1)
            .unwrap();
    }
    enc.end();
    command.commit();
    crate::metal::wait_completed(&command).unwrap();
    let scalar = scalar_sink.complete();
    let packed = packed_sink.complete();
    let read = |t: &MetalTensor| {
        // SAFETY: the sole command has completed; these views are in-bounds F32.
        unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize)
                    .cast::<f32>(),
                t.n_elements() as usize,
            )
            .to_vec()
        }
    };
    let a = read(scalar.tensor());
    let b = read(packed.tensor());
    assert_eq!(a, b);
    assert_eq!(packed.positions(), 103..108);
    for row in 0..5 {
        for (tap, &layer) in plan.layers().iter().enumerate() {
            for d in 0..h {
                let expected = layer as f32 + (row + 3) as f32 * 8.0 + 1.5 + d as f32 / 4096.0;
                assert_eq!(b[(row * 3 + tap) * h + d], expected);
            }
        }
    }
    for (_, tensor, original) in &inputs {
        assert_eq!(read(tensor), *original);
    }
}

#[test]
#[ignore = "opt-in allocation-only Metal regression; exclusive lease, no weights"]
fn capture_metal_allocation_sequence_stays_within_owned_buffer_prices() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let _transaction = ctx.begin_allocation_transaction();
    let g = geometry();
    let before = ctx.current_allocated_size();
    let full = Qwen4ExpCaptureBank::allocate(
        &ctx,
        Qwen4ExpCapturePlan::new(&g, vec![45, 5, 25, 15, 35], 4).unwrap(),
    )
    .unwrap();
    let describe = |bank: &Qwen4ExpCaptureBank| {
        [&bank.features, &bank.mean, &bank.quarters, &bank.indices]
            .iter()
            .zip(bank.allocations())
            .map(|(tensor, allocation)| {
                (
                    allocation.name.clone(),
                    allocation.logical_bytes,
                    allocation.priced_bytes,
                    tensor.buffer.length(),
                    tensor.buffer.allocatedSize(),
                )
            })
            .collect::<Vec<_>>()
    };
    eprintln!(
        "full capture: device_delta={} priced={} buffers(name,logical,priced,length,allocated)={:?}",
        ctx.current_allocated_size().saturating_sub(before),
        full.priced_upper_bytes(),
        describe(&full)
    );
    let mut underpriced = full.allocations.clone();
    underpriced[0].priced_bytes = full.features.buffer.allocatedSize() as u64 - 1;
    let error = validate_optional_owned_allocations(
        &ctx,
        &[&full.features, &full.mean, &full.quarters, &full.indices],
        &underpriced,
        ctx.current_allocated_size(),
    )
    .unwrap_err()
    .to_string();
    for field in [
        "capture.features",
        "owned_allocated_bytes=",
        "priced_upper_bytes=",
        "device_delta=",
    ] {
        assert!(error.contains(field), "{error}");
    }
    let before = ctx.current_allocated_size();
    let logits =
        Qwen4ExpRowLogitsBank::allocate(&ctx, Qwen4ExpRowLogitsPlan::new(&g, 4).unwrap()).unwrap();
    eprintln!(
        "row logits: device_delta={} priced={}",
        ctx.current_allocated_size().saturating_sub(before),
        logits.allocation().priced_bytes
    );
    // Keep resources live to cross allocator-pool/validation growth points.
    // At most 16 MiB of short capture banks; no command submission or model state.
    let mut short = Vec::new();
    let mut zero_deltas = 0;
    let mut over_price_deltas = 0;
    for index in 0..256 {
        let before = ctx.current_allocated_size();
        let bank =
            Qwen4ExpCaptureBank::allocate(&ctx, Qwen4ExpCapturePlan::new(&g, vec![5], 1).unwrap())
                .unwrap_or_else(|error| panic!("short capture {index}: {error}"));
        let after = ctx.current_allocated_size();
        let resources = describe(&bank);
        let owned: usize = resources.iter().map(|entry| entry.4).sum();
        let delta = after.saturating_sub(before);
        zero_deltas += usize::from(delta == 0);
        over_price_deltas += usize::from(delta > bank.priced_upper_bytes());
        if index < 4 || delta > bank.priced_upper_bytes() {
            eprintln!(
                "short capture {index}: device_delta={} owned={owned} priced={} buffers={resources:?}",
                after.saturating_sub(before),
                bank.priced_upper_bytes()
            );
        }
        for (name, logical, priced, length, allocated) in resources {
            assert_eq!(logical, length as u64, "{name}");
            assert!(
                allocated as u64 <= priced,
                "{name}: allocated={allocated} priced={priced}"
            );
        }
        short.push(bank);
    }
    assert_eq!(short.len(), 256);
    assert_eq!(full.plan().capacity(), 4);
    assert_eq!(logits.plan().capacity(), 4);
    eprintln!(
        "short bank summary: count={} zero_device_deltas={zero_deltas} device_deltas_above_owned_price={over_price_deltas}",
        short.len()
    );

    let mut config = Qwen4ExpConfig::flash_next_reference();
    config.vocab_size = 16;
    let ple = config.ple.as_mut().unwrap();
    ple.token_vocab_size = 16;
    ple.eos_token_id = 0;
    ple.image_token_id = None;
    let small_geometry = Qwen4ExpTextSessionMetalGeometry::from_config(&config, 64).unwrap();
    let mut small_readouts = Vec::new();
    let mut over_price_deltas = 0;
    for index in 0..1024 {
        let before = ctx.current_allocated_size();
        let bank = Qwen4ExpRowLogitsBank::allocate(
            &ctx,
            Qwen4ExpRowLogitsPlan::new(&small_geometry, 1).unwrap(),
        )
        .unwrap();
        let delta = ctx.current_allocated_size().saturating_sub(before);
        let priced = bank.allocation().priced_bytes;
        if delta > priced {
            over_price_deltas += 1;
            eprintln!("small readout {index}: device_delta={delta} priced={priced}");
        }
        small_readouts.push(bank);
    }
    eprintln!(
        "small readout summary: count={} device_deltas_above_owned_price={over_price_deltas}",
        small_readouts.len()
    );
}

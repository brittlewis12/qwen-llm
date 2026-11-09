use super::*;
use crate::metal::test_support::*;

#[test]
fn diagnostics_observers_return_to_inactive_state() {
    assert_eq!(diagnostics_observer_thread_active_counts(), [0, 0, 0]);
    {
        let _trace = kernel_trace_begin();
        assert_eq!(diagnostics_observer_thread_active_counts(), [1, 0, 0]);
    }
    dispatch_census_begin();
    assert_eq!(diagnostics_observer_thread_active_counts(), [0, 1, 0]);
    assert!(dispatch_census_take().is_empty());
    assert_eq!(diagnostics_observer_thread_active_counts(), [0, 0, 0]);
}

#[test]
fn tensor_ops_matmul2d_witness_selects_and_matches_portable() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const N_IN: usize = 32;
    const N_OUT: usize = 16;
    const N_QUERY: usize = 16;
    let left_values = (0..N_OUT * N_IN)
        .map(|index| half::f16::from_f32(((index * 13 % 29) as f32 - 14.0) / 32.0))
        .collect::<Vec<_>>();
    let right_values = (0..N_QUERY * N_IN)
        .map(|index| half::f16::from_f32(((index * 7 % 31) as f32 - 15.0) / 37.0))
        .collect::<Vec<_>>();
    let left = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&left_values),
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::F16,
    )
    .unwrap();
    let right = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&right_values),
        vec![N_IN as u64, N_QUERY as u64],
        GgmlType::F16,
    )
    .unwrap();
    let selected = MetalTensor::zeros_f32(&ctx, vec![N_QUERY as u64, N_OUT as u64]).unwrap();
    let portable = MetalTensor::zeros_f32(&ctx, vec![N_QUERY as u64, N_OUT as u64]).unwrap();

    dispatch_census_begin();
    let selected_cmd = ctx.queue.commandBuffer().expect("selected command buffer");
    let selected_enc = KernelEncoder::begin(&selected_cmd);
    encode_matmul2d_f16_f32(
        &ctx,
        &selected_enc,
        &left,
        &right,
        &selected,
        N_IN,
        N_OUT,
        N_QUERY,
    )
    .unwrap();
    selected_enc.end();
    selected_cmd.commit();
    wait_completed(&selected_cmd).expect("selected matmul2d command buffer");
    let selected_census = dispatch_census_take();

    dispatch_census_begin();
    let portable_cmd = ctx.queue.commandBuffer().expect("portable command buffer");
    let portable_enc = KernelEncoder::begin(&portable_cmd);
    crate::metal::mat_mat::encode_matmul2d_f16_f32_portable(
        &ctx,
        &portable_enc,
        &left,
        &right,
        &portable,
        N_IN,
        N_OUT,
        N_QUERY,
    )
    .unwrap();
    portable_enc.end();
    portable_cmd.commit();
    wait_completed(&portable_cmd).expect("portable matmul2d command buffer");
    let portable_census = dispatch_census_take();

    let selected_kernel = selected_census
        .first()
        .expect("selected dispatch census row")
        .kernel
        .as_str();
    let portable_kernel = portable_census
        .first()
        .expect("portable dispatch census row")
        .kernel
        .as_str();
    let facts = ctx.device_facts();
    if facts.gpu_families.metal4
        && facts.gpu_families.apple10
        && facts.os_is_at_least(crate::TENSOR_METALLIB_TARGET) == Some(true)
    {
        assert_eq!(ctx.tensor_ops_status(), &TensorOpsStatus::Available);
    }
    let expected_selected = if ctx.tensor_ops_status().is_available() {
        "kernel_tensor_matmul2d_f16_f32"
    } else {
        "kernel_matmul2d_f16_f32_portable"
    };
    assert_eq!(selected_census.len(), 1);
    assert_eq!(portable_census.len(), 1);
    if !facts.gpu_families.metal4 || !facts.gpu_families.apple10 {
        assert!(matches!(
            ctx.tensor_ops_status(),
            TensorOpsStatus::Unavailable {
                reason: TensorOpsUnavailableReason::GpuFamilyAbsent { .. }
            }
        ));
    }
    assert_eq!(selected_kernel, expected_selected);
    assert_eq!(portable_kernel, "kernel_matmul2d_f16_f32_portable");
    let selected_values = tensor_f32_at_offset(&selected);
    let portable_values = tensor_f32_at_offset(&portable);
    let diff =
        crate::compare::report_max_abs_diff_f32(selected_values.iter(), portable_values.iter());
    assert!(
        diff <= 0.02,
        "F16 matmul2d max abs diff {diff} exceeds 0.02"
    );
    eprintln!(
        "tensor_ops_witness status={:?} selected_kernel={} portable_kernel={} max_abs_diff={diff:.6} bound=0.02",
        ctx.tensor_ops_status(),
        selected_kernel,
        portable_kernel,
    );
}

/// Serve's idle residency renews only after a request that submitted compute
/// work, measured by this counter: every compute encoder constructor counts
/// (serial and concurrent), blit encoders do not.
#[test]
fn compute_encoder_counter_counts_every_compute_encoder_and_no_blit() {
    const CHILD: &str = "QWEN_ENCODER_COUNTER_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let qualified = concat!(
            module_path!(),
            "::compute_encoder_counter_counts_every_compute_encoder_and_no_blit"
        );
        let name = qualified.split_once("::").unwrap().1;
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--test-threads=1"])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success(), "isolated encoder counter test");
        return;
    }

    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let before = compute_encoders_begun();
    KernelEncoder::begin(&cmd).end();
    let after_serial = compute_encoders_begun();
    assert!(after_serial > before, "a serial compute encoder");
    KernelEncoder::begin_concurrent(&cmd).end();
    assert!(compute_encoders_begun() > after_serial, "a concurrent one");
    let blit_cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let before_blit = compute_encoders_begun();
    BlitEncoder::begin(&blit_cmd).end();
    assert_eq!(compute_encoders_begun(), before_blit, "a blit counted");
    cmd.commit();
    blit_cmd.commit();
    wait_completed(&cmd).expect("compute command buffer");
    wait_completed(&blit_cmd).expect("blit command buffer");
}

#[test]
fn mxfp4_f32_matrix_tile_matches_scalar_envelope_and_guards() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    const N_IN: usize = 64;
    const N_OUT: usize = 65;
    let mut weight_bytes = Vec::new();
    for row in 0..N_OUT {
        for block in 0..N_IN / 32 {
            let mut indices = [0u8; 32];
            for (index, value) in indices.iter_mut().enumerate() {
                *value = ((row * 11 + block * 7 + index * 3) % 16) as u8;
            }
            weight_bytes.extend_from_slice(&encode_mxfp4_block(
                120 + ((row + block) % 8) as u8,
                &indices,
            ));
        }
    }
    let weight = offset_tensor(
        &ctx,
        7,
        &weight_bytes,
        13,
        vec![N_IN as u64, N_OUT as u64],
        GgmlType::MXFP4,
    );

    for n_batch in [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let x_values = (0..n_batch * N_IN)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 19.0)
            .collect::<Vec<_>>();
        let x = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&x_values),
            20,
            vec![N_IN as u64, n_batch as u64],
            GgmlType::F32,
        );
        let output_bytes = vec![0u8; n_batch * N_OUT * size_of::<f32>()];
        let control = offset_tensor(
            &ctx,
            32,
            &output_bytes,
            28,
            vec![N_OUT as u64, n_batch as u64],
            GgmlType::F32,
        );
        let candidate = offset_tensor(
            &ctx,
            24,
            &output_bytes,
            36,
            vec![N_OUT as u64, n_batch as u64],
            GgmlType::F32,
        );
        let repeat = offset_tensor(
            &ctx,
            40,
            &output_bytes,
            44,
            vec![N_OUT as u64, n_batch as u64],
            GgmlType::F32,
        );
        let weight_before = tensor_backing_bytes(&weight);
        let x_before = tensor_backing_bytes(&x);

        let command = ctx
            .queue
            .commandBuffer()
            .expect("MXFP4 matrix command buffer");
        let encoder = KernelEncoder::begin(&command);
        for row in 0..n_batch {
            encode_mat_vec_mxfp4_f32(
                &ctx,
                &encoder,
                &weight,
                &x.view_subrange((row * N_IN) as u64, vec![N_IN as u64]),
                &control.view_subrange((row * N_OUT) as u64, vec![N_OUT as u64]),
                N_IN,
                N_OUT,
            )
            .expect("encode scalar MXFP4 control");
        }
        encode_mat_mat_mxfp4_f32_mm64x32(
            &ctx, &encoder, &weight, &x, &candidate, N_IN, N_OUT, n_batch,
        )
        .expect("encode MXFP4 matrix candidate");
        encode_mat_mat_mxfp4_f32_mm64x32(
            &ctx, &encoder, &weight, &x, &repeat, N_IN, N_OUT, n_batch,
        )
        .expect("encode repeated MXFP4 matrix candidate");
        encoder.end();
        command.commit();
        crate::metal::wait_completed(&command).expect("command buffer completed");
        assert!(command.error().is_none());

        let control_values = tensor_f32_at_offset(&control);
        let candidate_values = tensor_f32_at_offset(&candidate);
        let repeat_values = tensor_f32_at_offset(&repeat);
        assert_eq!(
            candidate_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            repeat_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "n_batch={n_batch} repeat"
        );
        let mut diff_sq = 0.0f64;
        let mut control_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut candidate_sq = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut max_control = 0.0f32;
        for (&got, &want) in candidate_values.iter().zip(&control_values) {
            let diff = f64::from(got) - f64::from(want);
            diff_sq += diff * diff;
            control_sq += f64::from(want) * f64::from(want);
            candidate_sq += f64::from(got) * f64::from(got);
            dot += f64::from(got) * f64::from(want);
            max_abs = max_abs.max((got - want).abs());
            max_control = max_control.max(want.abs());
        }
        let relative_rms = (diff_sq / control_sq).sqrt();
        let cosine = dot / (control_sq * candidate_sq).sqrt();
        let normalized_max = f64::from(max_abs / max_control.max(1.0));
        assert!(
            relative_rms <= 2.0e-5,
            "n_batch={n_batch} relative_rms={relative_rms}"
        );
        assert!(cosine >= 0.999_999_9, "n_batch={n_batch} cosine={cosine}");
        assert!(
            normalized_max <= 1.0e-4,
            "n_batch={n_batch} normalized_max={normalized_max}"
        );
        assert!(candidate_values.iter().all(|value| value.is_finite()));
        assert_eq!(tensor_backing_bytes(&weight), weight_before);
        assert_eq!(tensor_backing_bytes(&x), x_before);
        assert_offset_guards(&control, 32, 28);
        assert_offset_guards(&candidate, 24, 36);
        assert_offset_guards(&repeat, 40, 44);
    }
}

#[test]
#[ignore]
fn mxfp4_f32_matrix_tile_k216_n2048_bucket_floor() {
    run_mxfp4_f32_matrix_tile_k216_bucket_floor(12_288, [160, 166], 77, 0.60);
}

#[test]
fn owned_weight_view_is_read_only_and_bounds_checked() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let buffer = ctx.buffer_uninit(96).expect("owned backing");
    let view = MetalTensor::owned_weight_view(buffer.clone(), 32, vec![16], GgmlType::F32, 32)
        .expect("valid owned weight view");
    assert_eq!(
        view.provenance(),
        MetalTensorProvenance::OwnedWeightReadOnly
    );
    assert!(!view.is_writable());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            view.assert_writable("test write")
        }))
        .is_err()
    );
    assert!(
        MetalTensor::owned_weight_view(buffer.clone(), 1, vec![1], GgmlType::F32, 32,).is_err()
    );
    assert!(MetalTensor::owned_weight_view(buffer, 64, vec![16], GgmlType::F32, 32).is_err());
}

#[test]
fn q6_k_row_bank_view_fixes_geometry_alignment_and_provenance() {
    let ctx = match metal_test_context() {
        Some(ctx) => ctx,
        None => return,
    };
    let buffer = ctx.buffer_uninit(480).expect("Q6_K row-bank backing");
    let view = MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 256, 2)
        .expect("valid Q6_K row-bank view");
    assert_eq!(view.shape, [256, 2]);
    assert_eq!(view.dtype, GgmlType::Q6_K);
    assert_eq!(view.offset, 32);
    assert_eq!(view.n_bytes(), 420);
    assert_eq!(
        view.provenance(),
        MetalTensorProvenance::OwnedWeightReadOnly
    );
    assert!(!view.is_writable());

    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 1, 256, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 0, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 128, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 32, 256, 0).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 64, 256, 2).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer.clone(), 0, usize::MAX, 1).is_err());
    assert!(MetalTensor::q6_k_row_bank_weight_view(buffer, 0, 256, usize::MAX).is_err());
}

#[test]
fn checked_shape_bytes_rejects_product_overflow() {
    let err = checked_shape_bytes(&[u64::MAX, 2], std::mem::size_of::<f32>())
        .expect_err("shape product must overflow");
    assert!(matches!(err, MetalError::TensorSizeOverflow { .. }));
}

#[test]
fn checked_shape_bytes_rejects_byte_overflow() {
    let err = checked_shape_bytes(&[usize::MAX as u64 / 4 + 1], std::mem::size_of::<f32>())
        .expect_err("byte count must overflow usize");
    assert!(matches!(err, MetalError::TensorSizeOverflow { .. }));
}

#[test]
fn checked_ggml_shape_bytes_rejects_bad_q8_block() {
    let err = checked_ggml_shape_bytes(&[31], GgmlType::Q8_0)
        .expect_err("Q8_0 element count must align to a 32-element block");
    assert!(matches!(err, MetalError::BadShape { .. }));
}

#[test]
fn i32_subrange_preserves_type_and_byte_offset() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let ids = MetalTensor::zeros_i32(&ctx, vec![8]).unwrap();
    let view = ids.view_subrange(3, vec![2]);
    assert_eq!(view.dtype, GgmlType::I32);
    assert_eq!(view.shape, vec![2]);
    assert_eq!(view.offset, ids.offset + 3 * size_of::<i32>() as u64);
    assert_eq!(
        Retained::as_ptr(&view.buffer),
        Retained::as_ptr(&ids.buffer)
    );
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_allows_disjoint_and_shared_reads() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let shared_in = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let out_a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let out_b = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    // The production concurrent-pass shape: shared read-only input,
    // pairwise-disjoint outputs. Must not panic.
    enc.note_read(&shared_in);
    enc.note_write(&out_a);
    enc.note_read(&shared_in);
    enc.note_write(&out_b);
    enc.end();
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_panics_on_read_after_write() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let a = MetalTensor::zeros_f32(&ctx, vec![64]).unwrap();
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    enc.note_write(&a);
    let hazard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        enc.note_read(&a);
    }));
    assert!(
        hazard.is_err(),
        "read of a tensor written in the same Concurrent pass must panic"
    );
    enc.end();
}

#[cfg(debug_assertions)]
#[test]
fn concurrent_hazard_guard_allows_disjoint_views_of_one_buffer() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let arena = MetalTensor::zeros_f32(&ctx, vec![128]).unwrap();
    let lo = arena.view_subrange(0, vec![64]);
    let hi = arena.view_subrange(64, vec![64]);
    let cmd = ctx.queue.commandBuffer().expect("cmd buf");
    let enc = KernelEncoder::begin_concurrent(&cmd);
    // Disjoint sub-views of one arena are the packed-scratch pattern;
    // byte-range tracking (not buffer identity) must permit this.
    enc.note_write(&lo);
    enc.note_write(&hi);
    // But an overlapping second write must panic.
    let overlap = arena.view_subrange(32, vec![64]);
    let hazard = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        enc.note_write(&overlap);
    }));
    assert!(hazard.is_err(), "overlapping concurrent writes must panic");
    enc.end();
}

/// Ensures the chained-encoding API is correctness-equivalent to one-shot.
/// (The bench's only structural difference vs `encode_*` is that it loops
/// `encode_*` inside the same encoder; if it diverges, the kernel is
/// reading non-deterministic state — a bug.)
#[test]
fn chained_encoding_is_correct() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let n_in = 1024;
    let n_out = 4096;
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|i| ((i % 17) as f32 - 8.0) * 1e-3)
        .collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i % 7) as f32 - 3.0) * 1e-2).collect();
    let cpu = crate::forward::mat_vec_pub(&w, n_in, n_out, &x);

    let w_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&w),
        vec![n_in as u64, n_out as u64],
        GgmlType::F32,
    )
    .unwrap();
    let x_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&x),
        vec![n_in as u64],
        GgmlType::F32,
    )
    .unwrap();
    let y_t = MetalTensor::zeros_f32(&ctx, vec![n_out as u64]).unwrap();

    // Chain 8 dispatches; final result should equal one dispatch (each
    // overwrites the previous).
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    for _ in 0..8 {
        encode_mat_vec_f32(&ctx, &enc, &w_t, &x_t, &y_t, n_in, n_out).unwrap();
    }
    enc.end();
    cmd.commit();
    wait_completed(&cmd).expect("Metal command buffer failed");
    let gpu = read_back_f32(&y_t.buffer, n_out);

    let max_abs = crate::compare::assert_max_abs_diff_f32(
        concat!(file!(), ":", line!()),
        gpu.iter(),
        cpu.iter(),
    );
    assert!(
        max_abs < 1e-3,
        "chained encoding diverged: max|Δ|={max_abs}"
    );
}

mod dispatch_latency;
mod f32_narrow_tiles;
mod q6_k_f32_tile;

/// Half-subnormal / F32->half probe. Run on M4 and M5:
/// `cargo test -p qwen-llm --release --lib half_range_probe -- --ignored --nocapture`
#[test]
#[ignore = "GPU half-range probe; run on M4 and M5 with --ignored --nocapture"]
fn half_range_probe() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    eprintln!("half-range-probe device={}", ctx.device.name());

    let values = [2f32.powi(-24), 2f32.powi(-20), 2f32.powi(-15), 1.0];

    // 1. simdgroup half-operand MMA, F32 accumulator.
    let mma = ctx.pipeline("kernel_half_range_mma_probe").unwrap();
    for &v in &values {
        let a = vec![half::f16::from_f32(v); 64];
        let b = vec![half::f16::from_f32(1.0); 64];
        let a_t = MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&a), vec![64], GgmlType::F16)
            .unwrap();
        let b_t = MetalTensor::from_bytes(&ctx, bytemuck::cast_slice(&b), vec![64], GgmlType::F16)
            .unwrap();
        let out = MetalTensor::zeros_f32(&ctx, vec![1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        enc.set_pipeline(&mma);
        enc.set_tensor(0, &a_t);
        enc.set_tensor(1, &b_t);
        enc.set_tensor(2, &out);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.end();
        commit_and_wait(&command).unwrap();
        let got = read_back_f32(&out.buffer, 1)[0];
        let expected = 8.0 * v;
        let verdict = if got == expected {
            "preserve"
        } else if got == 0.0 {
            "flush"
        } else {
            panic!("simdgroup MMA returned an unrecognized value {got} for {v}");
        };
        eprintln!(
            "half-range-probe simdgroup-mma v={v:.9e} got={got:.9e} expected={expected:.9e} verdict={verdict}"
        );
    }

    // 2. Metal 4 matmul2d half operands, F32 accumulator.
    match ctx.tensor_pipeline("kernel_tensor_matmul2d_f16_f32") {
        Ok(tensor) => {
            for &v in &values {
                let left = vec![half::f16::from_f32(v); 16 * 32];
                let right = vec![half::f16::from_f32(1.0); 16 * 32];
                let left_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&left),
                    vec![32, 16],
                    GgmlType::F16,
                )
                .unwrap();
                let right_t = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&right),
                    vec![32, 16],
                    GgmlType::F16,
                )
                .unwrap();
                let out = MetalTensor::zeros_f32(&ctx, vec![16 * 16]).unwrap();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                enc.set_pipeline(&tensor);
                enc.set_tensor(0, &left_t);
                enc.set_tensor(1, &right_t);
                enc.set_tensor(2, &out);
                enc.dispatch(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.end();
                commit_and_wait(&command).unwrap();
                let got = read_back_f32(&out.buffer, 1)[0];
                let expected = 32.0 * v;
                let verdict = if got == expected {
                    "preserve"
                } else if got == 0.0 {
                    "flush"
                } else {
                    panic!("matmul2d returned an unrecognized value {got} for {v}");
                };
                eprintln!(
                    "half-range-probe matmul2d v={v:.9e} got={got:.9e} expected={expected:.9e} verdict={verdict}"
                );
            }
        }
        Err(error) => {
            if ctx.tensor_ops_status().is_available() {
                panic!("tensor tier is available but matmul2d pipeline failed: {error:?}");
            }
            eprintln!(
                "half-range-probe matmul2d unavailable status={:?}: {error:?}",
                ctx.tensor_ops_status()
            );
        }
    }

    // 3. F32 -> half conversion at and above the overflow boundary.
    let convert_values = [
        65_504.0f32,
        65_520.0,
        65_536.0,
        100_000.0,
        2f32.powi(-24),
        2f32.powi(-20),
        2f32.powi(-15),
    ];
    let n = convert_values.len() as u32;
    let input = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&convert_values),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap();
    let output = MetalTensor::zeros_dtype(&ctx, vec![n as u64], GgmlType::F16).unwrap();
    let pso = ctx.pipeline("kernel_f32_to_half_probe").unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &n);
    enc.set_tensor(1, &input);
    enc.set_tensor(2, &output);
    enc.dispatch(
        MTLSize {
            width: n.div_ceil(32) as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end();
    commit_and_wait(&command).unwrap();
    let bits: &[u16] = unsafe {
        std::slice::from_raw_parts(output.buffer.contents().as_ptr().cast::<u16>(), n as usize)
    };
    for (&v, &got) in convert_values.iter().zip(bits) {
        let oracle = half::f16::from_f32(v).to_bits();
        eprintln!(
            "half-range-probe f32->half v={v:.1} bits=0x{got:04x} value={:.9e} oracle=0x{oracle:04x} {}",
            half::f16::from_bits(got).to_f32(),
            if got == oracle { "match" } else { "MISMATCH" }
        );
        assert_eq!(
            got, oracle,
            "device conversion differs from half::f16 for {v}"
        );
    }
}

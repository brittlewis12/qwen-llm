use super::*;
use crate::metal::test_support::*;

#[test]
fn diagnostics_observers_return_to_inactive_state() {
    assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
    {
        let _trace = kernel_trace_begin();
        assert_eq!(diagnostics_observer_active_counts(), [1, 0, 0]);
    }
    dispatch_census_begin();
    assert_eq!(diagnostics_observer_active_counts(), [0, 1, 0]);
    assert!(dispatch_census_take().is_empty());
    assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
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
    run_mxfp4_f32_matrix_tile_k216_bucket_floor(12_288, [160, 166], 77, 250.0, 200.0, 0.50);
}

#[test]
fn owned_weight_view_is_read_only_and_bounds_checked() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(MetalError::NoDevice | MetalError::EmptyLibrary) => return,
        Err(error) => panic!("Metal context: {error}"),
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
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
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
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
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
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
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
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
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
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
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

    let max_abs = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "chained encoding diverged: max|Δ|={max_abs}"
    );
}

/// Boundary oracle for the packed g6_q2 attention reduce (2026-08-22):
/// model-free comparison of `encode_attn_prefill_v4_g6_q2_c32_f32`
/// (n_rows=8, nwg=128 — above the historical 64-partition cap) against
/// eight per-row `encode_attn_decode_v4_f32` calls on the same synthetic
/// Q/KV. Two 64-partition couplings shipped behind the old cap — a
/// fixed-64 shmem sizing in the encoder and a two-pass staging loop in
/// the reduce — both silently corrupted at nwg > 64; a cosine gate here
/// discriminates that corruption from reorder noise (corruption moved
/// logits by 5-11 absolute, reorder stays < 1e-3 cosine distance).
#[test]
#[ignore = "slow GPU oracle; run explicitly"]
fn packed_q2_attention_matches_per_row_at_high_nwg() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    const N_ROWS: usize = 8;
    const N_Q: usize = 24;
    const N_KV: usize = 4;
    const HEAD_DIM: usize = 256;
    const GROUP: usize = 6;
    let ctx_len: usize = std::env::var("QWEN_PACKED_Q2_ORACLE_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let oracle_ctx_len: usize = ctx_len;
    let base_pos: usize = oracle_ctx_len - N_ROWS;
    const NWG: usize = 128;

    let mut rng_state: u32 = 0x1234_5678;
    let mut rand_f32 = move || {
        rng_state = rng_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((rng_state >> 8) as f32) / ((1u32 << 24) as f32) - 0.5
    };
    let q: Vec<f32> = (0..N_ROWS * N_Q * HEAD_DIM).map(|_| rand_f32()).collect();
    let kv_elems = oracle_ctx_len * N_KV * HEAD_DIM;
    let k_f32: Vec<f32> = (0..kv_elems).map(|_| rand_f32()).collect();
    let v_f32: Vec<f32> = (0..kv_elems).map(|_| rand_f32()).collect();
    let k_half: Vec<half::f16> = k_f32.iter().map(|v| half::f16::from_f32(*v)).collect();
    let v_half: Vec<half::f16> = v_f32.iter().map(|v| half::f16::from_f32(*v)).collect();

    let q_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&q),
        vec![(N_ROWS * N_Q * HEAD_DIM) as u64],
        GgmlType::F32,
    )
    .expect("q");
    let k_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&k_half),
        vec![kv_elems as u64],
        GgmlType::F16,
    )
    .expect("k");
    let v_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&v_half),
        vec![kv_elems as u64],
        GgmlType::F16,
    )
    .expect("v");
    let o_packed =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HEAD_DIM) as u64]).expect("o packed");
    let o_partial =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * GROUP * HEAD_DIM) as u64])
            .expect("o partial");
    let ml_partial = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_KV * NWG * GROUP * 2) as u64])
        .expect("ml partial");

    {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        encode_attn_prefill_v4_g6_q2_c32_f32(
            &ctx,
            &enc,
            &q_t,
            &k_t,
            &v_t,
            &o_partial,
            &ml_partial,
            &o_packed,
            N_ROWS,
            base_pos,
            NWG,
            true,
        )
        .expect("packed encode");
        enc.end();
        cmd.commit();
        wait_completed(&cmd).expect("Metal command buffer failed");
    }

    let o_per_row =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * N_Q * HEAD_DIM) as u64]).expect("o per row");
    let o_partial_1 = MetalTensor::zeros_f32(&ctx, vec![(N_KV * 1024 * GROUP * HEAD_DIM) as u64])
        .expect("o partial 1");
    let ml_partial_1 =
        MetalTensor::zeros_f32(&ctx, vec![(N_KV * 1024 * GROUP * 2) as u64]).expect("ml partial 1");
    {
        let cmd = ctx.queue.commandBuffer().expect("cmd");
        let enc = KernelEncoder::begin(&cmd);
        for row in 0..N_ROWS {
            let q_row =
                q_t.view_subrange((row * N_Q * HEAD_DIM) as u64, vec![(N_Q * HEAD_DIM) as u64]);
            let o_row = o_per_row
                .view_subrange((row * N_Q * HEAD_DIM) as u64, vec![(N_Q * HEAD_DIM) as u64]);
            encode_attn_decode_v4_f32(
                &ctx,
                &enc,
                &q_row,
                &k_t,
                &v_t,
                &o_partial_1,
                &ml_partial_1,
                &o_row,
                N_Q,
                N_KV,
                HEAD_DIM,
                base_pos + row + 1,
                NWG,
                32,
            )
            .expect("per-row encode");
        }
        enc.end();
        cmd.commit();
        wait_completed(&cmd).expect("Metal command buffer failed");
    }

    unsafe {
        let a = o_packed.buffer.contents().as_ptr() as *const f32;
        let b = o_per_row.buffer.contents().as_ptr() as *const f32;
        let n = N_ROWS * N_Q * HEAD_DIM;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..n {
            let av = *a.add(i);
            let bv = *b.add(i);
            dot += (av as f64) * (bv as f64);
            na += (av as f64).powi(2);
            nb += (bv as f64).powi(2);
            max_abs = max_abs.max((av - bv).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("[packed-q2-oracle] cos={cos:.6} max|delta|={max_abs:.3e} nwg={NWG}");
        assert!(
            cos > 0.9999,
            "packed q2 attention diverges from per-row at nwg={NWG} (cos={cos})"
        );
    }
}

/// Perf audit for the packed g6 q2 shared-KV attention at depth
/// (2026-08-22): synthetic session at a large kv_n_pos, one
/// `encode_attn_prefill_v4_g6_q2_c32_f32` call over n_rows=8 with the
/// selector's nwg, kernel timing only. Reports ms and effective GB/s
/// (KV read once per row-pair = 4x the per-layer KV bytes). The per-row
/// baseline at 130K/nwg=512 is ~4.81ms per row (~38.5ms per layer for
/// 8 rows); the packed path is the default-on perf gate.
/// Env: QWEN_ATTN_AUDIT_CTX, QWEN_ATTN_AUDIT_MODEL.
#[test]
#[ignore = "slow real-model GPU audit; run explicitly"]
fn packed_q2_attention_perf_audit_130k() {
    let model_path = std::env::var("QWEN_ATTN_AUDIT_MODEL")
        .unwrap_or_else(|_| "/Users/tito/models/Qwen3.8-27B-Q8_0.gguf".into());
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("[packed-audit] skipped — model missing");
        return;
    }
    let n_pos: usize = std::env::var("QWEN_ATTN_AUDIT_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(131_072);
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
        Err(e) => panic!("init failed: {e}"),
    };
    let g = crate::gguf::GgufFile::open(&model_path).expect("open model");
    let m = crate::loader::Model::from_gguf(&g).expect("load model");
    let mm = crate::metal_forward::MetalModel::load(&ctx, &g, &m).expect("metal load");
    let mut sess =
        crate::metal_forward::MetalSession::fresh(&ctx, &mm, n_pos + 16).expect("session");
    for kp in sess.kv_n_pos.iter_mut() {
        *kp = n_pos;
    }
    fill_audit_f16(&sess.kv_k[0], 1);
    fill_audit_f16(&sess.kv_v[0], 2);
    let arch = &m.arch;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let group = n_q / n_kv;
    if head_dim != 256 || group != 6 {
        eprintln!("[packed-audit] skipped — unsupported shape");
        return;
    }
    const N_ROWS: usize = 8;
    const NWG_MAX: usize = 1024;
    let nwg = crate::metal::attn_v4_choose_nwg(n_pos, 6);
    let base_pos = n_pos - N_ROWS;
    let q = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_q * head_dim) as u64]).expect("q");
    let o = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_q * head_dim) as u64]).expect("o");
    let per_row_o_partial =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * NWG_MAX * group * head_dim) as u64])
            .expect("per-row o partial");
    let per_row_ml_partial =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * NWG_MAX * group * 2) as u64])
            .expect("per-row ml partial");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    for row in 0..N_ROWS {
        let q_row = q.view_subrange((row * n_q * head_dim) as u64, vec![(n_q * head_dim) as u64]);
        let o_row = o.view_subrange((row * n_q * head_dim) as u64, vec![(n_q * head_dim) as u64]);
        encode_attn_decode_v4_f32(
            &ctx,
            &enc,
            &q_row,
            &sess.kv_k[0],
            &sess.kv_v[0],
            &per_row_o_partial,
            &per_row_ml_partial,
            &o_row,
            n_q,
            n_kv,
            head_dim,
            base_pos + row + 1,
            crate::metal::attn_v4_choose_nwg(base_pos + row + 1, group),
            crate::metal::attn_v4_choose_tile_c(base_pos + row + 1, group),
        )
        .expect("per-row encode");
    }
    enc.end();
    cmd.commit();
    wait_completed(&cmd).expect("Metal command buffer failed");
    let per_row_chain_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    eprintln!("[per-row-chain-audit] ctx={n_pos} rows={N_ROWS} gpu_ms={per_row_chain_ms:.3}");
    let o_partial = MetalTensor::zeros_f32(
        &ctx,
        vec![(N_ROWS * n_kv * NWG_MAX * group * head_dim) as u64],
    )
    .expect("o partial");
    let ml_partial =
        MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_kv * NWG_MAX * group * 2) as u64])
            .expect("ml partial");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    encode_attn_prefill_v4_g6_q2_c32_f32(
        &ctx,
        &enc,
        &q,
        &sess.kv_k[0],
        &sess.kv_v[0],
        &o_partial,
        &ml_partial,
        &o,
        N_ROWS,
        base_pos,
        nwg,
        true,
    )
    .expect("packed encode");
    enc.end();
    cmd.commit();
    wait_completed(&cmd).expect("Metal command buffer failed");
    let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    // KV read once per row-pair: ceil(N_ROWS/2) passes over the layer KV.
    let passes = N_ROWS.div_ceil(2) as f64;
    let bytes = passes * n_pos as f64 * (n_kv * head_dim * 2 * 2) as f64;
    let gbps = bytes / 1e9 / (ms / 1e3);
    eprintln!(
        "[packed-audit] ctx={n_pos} nwg={nwg} gpu_ms={ms:.3} gb={:.2} gbps={gbps:.1} (per-row baseline ~38.5 ms/layer at 130K)",
        bytes / 1e9
    );

    // Tier-3 matrix reader timing: KQ (MMA) -> softmax -> direct-V KQV.
    let scores = MetalTensor::zeros_f32(&ctx, vec![(N_ROWS * n_q * n_pos) as u64]).expect("scores");
    let cmd = ctx.queue.commandBuffer().expect("cmd");
    let enc = KernelEncoder::begin(&cmd);
    encode_attn_matrix_kq_f32(
        &ctx,
        &enc,
        &q,
        &sess.kv_k[0],
        &scores,
        N_ROWS,
        base_pos,
        n_pos,
        n_kv * head_dim,
        n_q,
        n_kv,
        group,
        head_dim,
        true,
    )
    .expect("matrix kq");
    encode_attn_matrix_softmax_f32(
        &ctx, &enc, &scores, N_ROWS, base_pos, n_pos, n_q, n_kv, group, head_dim,
    )
    .expect("matrix softmax");
    encode_attn_matrix_kqv_direct_v_f32(
        &ctx,
        &enc,
        &scores,
        &sess.kv_v[0],
        &o,
        N_ROWS,
        base_pos,
        n_pos,
        n_kv * head_dim,
        n_q,
        n_kv,
        group,
        head_dim,
        true,
    )
    .expect("matrix kqv direct v");
    enc.end();
    cmd.commit();
    wait_completed(&cmd).expect("Metal command buffer failed");
    let ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
    eprintln!(
        "[matrix-audit] ctx={n_pos} gpu_ms={ms:.3} (per-row baseline ~38.5 ms/layer at 130K)"
    );
}

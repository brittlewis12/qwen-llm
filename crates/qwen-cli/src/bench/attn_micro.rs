//! Attention front/prefill/layer/intra microbenchmarks.

use super::*;

pub(crate) fn run_attn_front_micro(args: AttnFrontMicroArgs) -> Result<()> {
    let AttnFrontMicroArgs { model, rows } = args;
    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let block = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .ok_or_else(|| anyhow!("model has no full-attention block"))?;
    let h = mm.arch.hidden_size as usize;
    let head_dim = mm.arch.attn_head_dim as usize;
    let n_q = mm.arch.n_q_heads as usize;
    let n_kv = mm.arch.n_kv_heads as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let fused_out = 2 * q_dim + 2 * kv_dim;

    let x: Vec<f32> = (0..rows * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    let as_bytes = |xs: &[f32]| unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
    };
    let x_t = MetalTensor::from_bytes(&ctx, as_bytes(&x), vec![(rows * h) as u64], GgmlType::F32)?;

    let read_tensor_bytes = |t: &MetalTensor| -> Vec<u8> {
        let n = t.n_bytes() as usize;
        let mut out = vec![0u8; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let mut fused_bytes = Vec::new();
    fused_bytes.extend_from_slice(&read_tensor_bytes(&block.q));
    fused_bytes.extend_from_slice(&read_tensor_bytes(&block.k));
    fused_bytes.extend_from_slice(&read_tensor_bytes(&block.v));
    let fused_w = MetalTensor::from_bytes(
        &ctx,
        &fused_bytes,
        vec![h as u64, fused_out as u64],
        GgmlType::Q8_0,
    )?;

    let q_out = MetalTensor::zeros_f32(&ctx, vec![(rows * 2 * q_dim) as u64])?;
    let k_out = MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?;
    let v_out = MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?;
    let fused_out_t = MetalTensor::zeros_f32(&ctx, vec![(rows * fused_out) as u64])?;

    let t = Instant::now();
    {
        let cmd = ctx.queue.commandBuffer().context("separate cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_mat_dispatch(&ctx, &enc, &block.q, &x_t, &q_out, h, 2 * q_dim, rows)?;
        encode_mat_mat_dispatch(&ctx, &enc, &block.k, &x_t, &k_out, h, kv_dim, rows)?;
        encode_mat_mat_dispatch(&ctx, &enc, &block.v, &x_t, &v_out, h, kv_dim, rows)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let separate_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    {
        let cmd = ctx.queue.commandBuffer().context("fused cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_mat_dispatch(&ctx, &enc, &fused_w, &x_t, &fused_out_t, h, fused_out, rows)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let fused_ms = t.elapsed().as_secs_f64() * 1e3;

    let read_back = |t: &MetalTensor| -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let q = read_back(&q_out);
    let k = read_back(&k_out);
    let v = read_back(&v_out);
    let fused = read_back(&fused_out_t);
    let row_q = 2 * q_dim;
    let row_kv = kv_dim;
    let row_f = fused_out;
    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let q_off = row * row_q;
        let k_off = row * row_kv;
        let f_off = row * row_f;
        for i in 0..row_q {
            max_abs = max_abs.max((q[q_off + i] - fused[f_off + i]).abs());
        }
        for i in 0..row_kv {
            max_abs = max_abs.max((k[k_off + i] - fused[f_off + row_q + i]).abs());
            max_abs = max_abs.max((v[k_off + i] - fused[f_off + row_q + row_kv + i]).abs());
        }
    }
    println!(
        "[attn-front-micro] rows={} separate_ms={:.2} fused_ms={:.2} speedup={:.3} max|Δ|={:.2e}",
        rows,
        separate_ms,
        fused_ms,
        separate_ms / fused_ms,
        max_abs
    );
    Ok(())
}

pub(crate) fn run_attn_prefill_micro(args: AttnPrefillMicroArgs) -> Result<()> {
    let AttnPrefillMicroArgs {
        model,
        base_pos,
        rows,
        nwg,
        qt,
    } = args;
    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    const HD: usize = 256;
    let n_q: usize = m.arch.n_q_heads as usize;
    let n_kv: usize = m.arch.n_kv_heads as usize;
    let (group_tile, tile_c) = match (n_q, n_kv) {
        (16, 2) => (2, 64),
        (24, 4) => (6, 32),
        (32, 2) => (4, 64),
        _ => (0, 0),
    };
    if group_tile == 0 {
        anyhow::bail!(
            "attn-prefill-micro unsupported shape n_q={} n_kv={}",
            n_q,
            n_kv
        );
    }
    if m.arch.attn_head_dim as usize != HD {
        anyhow::bail!(
            "attn-prefill-micro only supports head_dim=256, got {}",
            m.arch.attn_head_dim
        );
    }
    let kv_dim = n_kv * HD;
    let n_pos = base_pos + rows;

    let q_rows: Vec<f32> = (0..rows * n_q * HD)
        .map(|i| ((i % 31) as f32 - 15.0) * 1e-2)
        .collect();
    let k_f32: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
        .collect();
    let v_f32: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
        .collect();

    let as_bytes = |xs: &[f32]| unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
    };

    let q_t = MetalTensor::from_bytes(
        &ctx,
        as_bytes(&q_rows),
        vec![(rows * n_q * HD) as u64],
        GgmlType::F32,
    )?;
    let k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    for (src_f32, dst) in [(&k_f32, &k_cache), (&v_f32, &v_cache)] {
        let src_t = MetalTensor::from_bytes(
            &ctx,
            as_bytes(src_f32.as_slice()),
            vec![src_f32.len() as u64],
            GgmlType::F32,
        )?;
        let cmd = ctx.queue.commandBuffer().context("prep cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_scatter_offset_f32_to_f16(&ctx, &enc, &src_t, dst, 0, src_f32.len())?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let out_baseline = MetalTensor::zeros_f32(&ctx, vec![(rows * n_q * HD) as u64])?;
    let out_packed = MetalTensor::zeros_f32(&ctx, vec![(rows * n_q * HD) as u64])?;
    let o_partial_row =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * HD) as u64])?;
    let ml_partial_row =
        MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * (n_q / n_kv) * 2) as u64])?;
    let o_partial_packed =
        MetalTensor::zeros_f32(&ctx, vec![(rows * n_kv * nwg * (n_q / n_kv) * HD) as u64])?;
    let ml_partial_packed =
        MetalTensor::zeros_f32(&ctx, vec![(rows * n_kv * nwg * (n_q / n_kv) * 2) as u64])?;

    let run_baseline = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        with_attn_v4_group_tile_override(group_tile, || {
            for row in 0..rows {
                let q_row = q_t.view_subrange((row * n_q * HD) as u64, vec![(n_q * HD) as u64]);
                let out_row =
                    out_baseline.view_subrange((row * n_q * HD) as u64, vec![(n_q * HD) as u64]);
                encode_attn_decode_v4_f32(
                    &ctx,
                    &enc,
                    &q_row,
                    &k_cache,
                    &v_cache,
                    &o_partial_row,
                    &ml_partial_row,
                    &out_row,
                    n_q,
                    n_kv,
                    HD,
                    base_pos + row + 1,
                    nwg,
                    tile_c,
                )
                .expect("decode-shaped oracle attention");
            }
        });
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };
    run_baseline()?;
    let t = Instant::now();
    run_baseline()?;
    let baseline_wall = t.elapsed().as_secs_f64() * 1e3;

    let run_packed = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("packed cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        match (n_q, n_kv, qt) {
            (24, 4, 2) => encode_attn_prefill_v4_g6_q2_c32_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
                false,
            )?,
            (16, 2, 2) => encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            (16, 2, 4) => encode_attn_prefill_v4_g8_t2_q4_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 2) => encode_attn_prefill_v4_g16_t4_q2_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 4) => encode_attn_prefill_v4_g16_t4_q4_c64_f32(
                &ctx,
                &enc,
                &q_t,
                &k_cache,
                &v_cache,
                &o_partial_packed,
                &ml_partial_packed,
                &out_packed,
                rows,
                base_pos,
                nwg,
            )?,
            _ => anyhow::bail!(
                "attn-prefill-micro unsupported shape n_q={} n_kv={} qt={}",
                n_q,
                n_kv,
                qt
            ),
        }
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };
    run_packed()?;
    let t = Instant::now();
    run_packed()?;
    let packed_wall = t.elapsed().as_secs_f64() * 1e3;

    let read_back = |t: &MetalTensor| -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let baseline = read_back(&out_baseline);
    let packed = read_back(&out_packed);
    let max_abs = packed
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = packed
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = packed
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = baseline
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    println!(
        "[attn-prefill-micro] base_pos={} rows={} nwg={} qt={} decode_loop_ms={:.2} packed_ms={:.2} speedup={:.3} max|Δ|={:.2e} cos={:.6}",
        base_pos,
        rows,
        nwg,
        qt,
        baseline_wall,
        packed_wall,
        baseline_wall / packed_wall,
        max_abs,
        cos
    );
    Ok(())
}

pub(crate) fn run_attn_layer_micro(args: AttnLayerMicroArgs) -> Result<()> {
    let AttnLayerMicroArgs {
        model,
        base_pos,
        rows,
        nwg,
        qt,
    } = args;
    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let block = mm
        .blocks
        .iter()
        .find_map(|b| match b {
            MetalBlock::Attn(a) => Some(a),
            _ => None,
        })
        .ok_or_else(|| anyhow!("model has no full-attention block"))?;
    const HD: usize = 256;
    let n_q = mm.arch.n_q_heads as usize;
    let n_kv = mm.arch.n_kv_heads as usize;
    let h = mm.arch.hidden_size as usize;
    let q_dim = n_q * HD;
    let kv_dim = n_kv * HD;
    let group = n_q / n_kv;
    let (group_tile, tile_c) = match (n_q, n_kv) {
        (16, 2) => (2, 64),
        (24, 4) => (6, 32),
        (32, 2) => (4, 64),
        _ => anyhow::bail!(
            "attn-layer-micro unsupported shape n_q={} n_kv={}",
            n_q,
            n_kv
        ),
    };
    if mm.arch.attn_head_dim as usize != HD {
        anyhow::bail!(
            "attn-layer-micro only supports head_dim=256, got {}",
            mm.arch.attn_head_dim
        );
    }
    if !matches!(qt, 2 | 4) {
        anyhow::bail!("attn-layer-micro only supports qt=2 or 4, got {qt}");
    }
    let n_rot = (HD as f32 * mm.arch.partial_rotary_factor) as usize;
    let n_pos = base_pos + rows;

    let h_rows: Vec<f32> = (0..rows * h)
        .map(|i| ((i % 37) as f32 - 18.0) * 1e-2)
        .collect();
    let prefix_k: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 1.5e-2)
        .collect();
    let prefix_v: Vec<f32> = (0..n_pos * kv_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 2e-2)
        .collect();
    let as_bytes = |xs: &[f32]| unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
    };
    let h_t = MetalTensor::from_bytes(
        &ctx,
        as_bytes(&h_rows),
        vec![(rows * h) as u64],
        GgmlType::F32,
    )?;

    #[derive(Clone)]
    struct StackScratch {
        q_full: MetalTensor,
        q: MetalTensor,
        gate: MetalTensor,
        q_normed: MetalTensor,
        k_now: MetalTensor,
        v_now: MetalTensor,
        k_normed: MetalTensor,
        attn_o: MetalTensor,
        mixer_out: MetalTensor,
        o_partial: MetalTensor,
        ml_partial: MetalTensor,
    }

    let make_baseline_scratch = || -> Result<StackScratch> {
        Ok(StackScratch {
            q_full: MetalTensor::zeros_f32(&ctx, vec![(rows * 2 * q_dim) as u64])?,
            q: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            gate: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            q_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            k_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            v_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            k_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            attn_o: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            mixer_out: MetalTensor::zeros_f32(&ctx, vec![(rows * h) as u64])?,
            o_partial: MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * HD) as u64])?,
            ml_partial: MetalTensor::zeros_f32(&ctx, vec![(n_kv * nwg * group * 2) as u64])?,
        })
    };
    let make_packed_scratch = || -> Result<StackScratch> {
        Ok(StackScratch {
            q_full: MetalTensor::zeros_f32(&ctx, vec![(rows * 2 * q_dim) as u64])?,
            q: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            gate: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            q_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            k_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            v_now: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            k_normed: MetalTensor::zeros_f32(&ctx, vec![(rows * kv_dim) as u64])?,
            attn_o: MetalTensor::zeros_f32(&ctx, vec![(rows * q_dim) as u64])?,
            mixer_out: MetalTensor::zeros_f32(&ctx, vec![(rows * h) as u64])?,
            o_partial: MetalTensor::zeros_f32(&ctx, vec![(rows * n_kv * nwg * group * HD) as u64])?,
            ml_partial: MetalTensor::zeros_f32(&ctx, vec![(rows * n_kv * nwg * group * 2) as u64])?,
        })
    };

    let baseline = make_baseline_scratch()?;
    let packed = make_packed_scratch()?;
    let baseline_k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let baseline_v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let packed_k_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;
    let packed_v_cache = MetalTensor::zeros_f16(&ctx, vec![(n_pos * kv_dim) as u64])?;

    let seed_cache = |dst: &MetalTensor, src_f32: &[f32]| -> Result<()> {
        let src_t = MetalTensor::from_bytes(
            &ctx,
            as_bytes(src_f32),
            vec![src_f32.len() as u64],
            GgmlType::F32,
        )?;
        let cmd = ctx.queue.commandBuffer().context("seed cache cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_scatter_offset_f32_to_f16(&ctx, &enc, &src_t, dst, 0, src_f32.len())?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };
    for dst in [&baseline_k_cache, &packed_k_cache] {
        seed_cache(dst, &prefix_k)?;
    }
    for dst in [&baseline_v_cache, &packed_v_cache] {
        seed_cache(dst, &prefix_v)?;
    }

    let run_front = |enc: &KernelEncoder,
                     scratch: &StackScratch,
                     k_cache: &MetalTensor,
                     v_cache: &MetalTensor|
     -> Result<()> {
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &block.q,
            &h_t,
            &scratch.q_full,
            h,
            2 * q_dim,
            rows,
        )?;
        encode_mat_mat_dispatch(&ctx, enc, &block.k, &h_t, &scratch.k_now, h, kv_dim, rows)?;
        encode_mat_mat_dispatch(&ctx, enc, &block.v, &h_t, &scratch.v_now, h, kv_dim, rows)?;
        encode_split_q_gate_f32(
            &ctx,
            enc,
            &scratch.q_full,
            &scratch.q,
            &scratch.gate,
            rows * n_q,
            HD,
        )?;
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &scratch.q,
            &block.q_norm,
            &scratch.q_normed,
            rows * n_q,
            HD,
            RMS_EPS,
        )?;
        encode_rms_norm_batched_f32(
            &ctx,
            enc,
            &scratch.k_now,
            &block.k_norm,
            &scratch.k_normed,
            rows * n_kv,
            HD,
            RMS_EPS,
        )?;
        encode_rope_neox_f32_packed_consecutive(
            &ctx,
            enc,
            &scratch.q_normed,
            rows,
            n_q,
            HD,
            n_rot,
            base_pos as u32,
            mm.arch.rope_theta,
        )?;
        encode_rope_neox_f32_packed_consecutive(
            &ctx,
            enc,
            &scratch.k_normed,
            rows,
            n_kv,
            HD,
            n_rot,
            base_pos as u32,
            mm.arch.rope_theta,
        )?;
        encode_scatter_offset_f32_to_f16_kv(
            &ctx,
            enc,
            &scratch.k_normed,
            &scratch.v_now,
            k_cache,
            v_cache,
            base_pos * kv_dim,
            rows * kv_dim,
        )?;
        Ok(())
    };

    let run_tail = |enc: &KernelEncoder, scratch: &StackScratch| -> Result<()> {
        encode_sigmoid_f32(&ctx, enc, &scratch.gate, &scratch.q)?;
        encode_mul_f32(&ctx, enc, &scratch.attn_o, &scratch.q, &scratch.attn_o)?;
        encode_mat_mat_dispatch(
            &ctx,
            enc,
            &block.o,
            &scratch.attn_o,
            &scratch.mixer_out,
            q_dim,
            h,
            rows,
        )?;
        Ok(())
    };

    let run_baseline = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("baseline layer cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        run_front(&enc, &baseline, &baseline_k_cache, &baseline_v_cache)?;
        with_attn_v4_group_tile_override(group_tile, || {
            for row in 0..rows {
                let q_row = baseline
                    .q_normed
                    .view_subrange((row * q_dim) as u64, vec![q_dim as u64]);
                let out_row = baseline
                    .attn_o
                    .view_subrange((row * q_dim) as u64, vec![q_dim as u64]);
                encode_attn_decode_v4_f32(
                    &ctx,
                    &enc,
                    &q_row,
                    &baseline_k_cache,
                    &baseline_v_cache,
                    &baseline.o_partial,
                    &baseline.ml_partial,
                    &out_row,
                    n_q,
                    n_kv,
                    HD,
                    base_pos + row + 1,
                    nwg,
                    tile_c,
                )
                .expect("baseline decode attention");
            }
        });
        run_tail(&enc, &baseline)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };

    let run_packed = || -> Result<()> {
        let cmd = ctx.queue.commandBuffer().context("packed layer cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        run_front(&enc, &packed, &packed_k_cache, &packed_v_cache)?;
        match (n_q, n_kv, qt) {
            (24, 4, 2) => encode_attn_prefill_v4_g6_q2_c32_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
                false,
            )?,
            (16, 2, 2) => encode_attn_prefill_v4_g8_t2_q2_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            (16, 2, 4) => encode_attn_prefill_v4_g8_t2_q4_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 2) => encode_attn_prefill_v4_g16_t4_q2_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            (32, 2, 4) => encode_attn_prefill_v4_g16_t4_q4_c64_f32(
                &ctx,
                &enc,
                &packed.q_normed,
                &packed_k_cache,
                &packed_v_cache,
                &packed.o_partial,
                &packed.ml_partial,
                &packed.attn_o,
                rows,
                base_pos,
                nwg,
            )?,
            _ => anyhow::bail!(
                "attn-layer-micro unsupported shape n_q={} n_kv={} qt={}",
                n_q,
                n_kv,
                qt
            ),
        }
        run_tail(&enc, &packed)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    };

    run_baseline()?;
    let t = Instant::now();
    run_baseline()?;
    let baseline_wall = t.elapsed().as_secs_f64() * 1e3;
    run_packed()?;
    let t = Instant::now();
    run_packed()?;
    let packed_wall = t.elapsed().as_secs_f64() * 1e3;

    let read_back = |t: &MetalTensor| -> Vec<f32> {
        let n = t.n_elements() as usize;
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    };
    let baseline_out = read_back(&baseline.mixer_out);
    let packed_out = read_back(&packed.mixer_out);
    let max_abs = packed_out
        .iter()
        .zip(baseline_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f64 = packed_out
        .iter()
        .zip(baseline_out.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let na: f64 = packed_out
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = baseline_out
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb);
    println!(
        "[attn-layer-micro] base_pos={} rows={} nwg={} qt={} baseline_ms={:.2} packed_ms={:.2} speedup={:.3} max|Δ|={:.2e} cos={:.6}",
        base_pos,
        rows,
        nwg,
        qt,
        baseline_wall,
        packed_wall,
        baseline_wall / packed_wall,
        max_abs,
        cos,
    );
    Ok(())
}

pub(crate) fn run_attn_intra(args: AttnIntraArgs) -> Result<()> {
    let AttnIntraArgs {
        model,
        ctx: target,
        runs,
        block,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be > 0"));
    }

    let mctx = MetalContext::new()?;
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&mctx, &g, &m)?;
    let mf = MetalForward::new(&mctx, &mm);

    let total_attn = mm
        .blocks
        .iter()
        .filter(|b| matches!(b, MetalBlock::Attn(_)))
        .count();
    let mut attn_seen = 0usize;
    let mut selected: Option<(usize, usize)> = None;
    for (idx, b) in mm.blocks.iter().enumerate() {
        if matches!(b, MetalBlock::Attn(_)) {
            if block.is_none_or(|want| want == idx) {
                selected = Some((idx, attn_seen));
                break;
            }
            attn_seen += 1;
        }
    }
    let (block_idx, attn_idx_in_session) = selected.ok_or_else(|| match block {
        Some(idx) => anyhow!("block {idx} is not a full-attention block"),
        None => anyhow!("model has no full-attention blocks"),
    })?;
    let ab = match &mm.blocks[block_idx] {
        MetalBlock::Attn(a) => a,
        _ => unreachable!(),
    };

    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let group = n_q / n_kv;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
    if head_dim != 256 || !matches!(group, 4 | 6 | 8 | 16) {
        return Err(anyhow!(
            "attn-intra only supports attn_v4 shapes; got head_dim={head_dim} group={group}"
        ));
    }

    {
        let mut s = MetalSession::fresh(&mctx, &mm, 32)?;
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s)?;
        }
    }
    let mut s = MetalSession::fresh(&mctx, &mm, target + runs + 16)?;
    if target > 1 {
        // prefill-warm (v0.494 pattern): production packed prefill instead of
        // the token-by-token decode ramp; validated gpu_ms parity at ctx16384.
        let ids = vec![0i32; target];
        let chunk = default_prefill_chunk(mm.arch.kind, target);
        let mut scratch = fresh_prefill_scratch_for_prompt(&mctx, &mm, chunk, ids.len())
            .context("attn-intra prefill scratch")?;
        let t0 = Instant::now();
        prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
            .context("attn-intra prefill warm")?;
        eprintln!(
            "[attn-intra] prefill-warm to {target} in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    }

    let timed = |label: &str,
                 cb: &dyn Fn(&KernelEncoder) -> Result<()>,
                 phases: &mut Vec<(String, f64)>|
     -> Result<()> {
        let cmd = mctx.queue.commandBuffer().context("attn-intra cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        cb(&enc)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();
        phases.push((
            label.to_string(),
            (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3,
        ));
        Ok(())
    };

    let sigmoid_mul = env_flag_default_on("QWEN_DECODE_ATTN_SIGMOID_MUL");
    let rope_pair = env_flag_default_on("QWEN_DECODE_ROPE_PAIR");
    let fused_qk_norm_rope =
        env_flag_default_on("QWEN_DECODE_QK_NORM_ROPE_FUSED") && rope_pair && sigmoid_mul;
    let mut agg: Vec<(String, f64)> = Vec::new();
    for run in 0..runs {
        shutdown::checkpoint()?;
        let position = target as u32 + run as u32;
        let mut phases: Vec<(String, f64)> = Vec::new();
        timed(
            "pre_norm (rms_norm)",
            &|enc| {
                Ok(encode_rms_norm_mul_f32(
                    &mctx,
                    enc,
                    &s.x,
                    &ab.attn_norm,
                    &s.h,
                    RMS_EPS,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "q_proj_2x (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.q,
                    &s.h,
                    &s.attn_q_full,
                    h,
                    2 * q_dim,
                )?)
            },
            &mut phases,
        )?;
        if !sigmoid_mul {
            timed(
                "split_q_gate",
                &|enc| {
                    Ok(encode_split_q_gate_f32(
                        &mctx,
                        enc,
                        &s.attn_q_full,
                        &s.attn_q,
                        &s.attn_gate,
                        n_q,
                        head_dim,
                    )?)
                },
                &mut phases,
            )?;
        }
        timed(
            "k_proj (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.k,
                    &s.h,
                    &s.attn_k_now,
                    h,
                    kv_dim,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "v_proj (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.v,
                    &s.h,
                    &s.attn_v_now,
                    h,
                    kv_dim,
                )?)
            },
            &mut phases,
        )?;
        if fused_qk_norm_rope {
            timed(
                "qk_norm_rope (fused)",
                &|enc| {
                    Ok(encode_qk_rms_norm_rope_f32_packed_consecutive(
                        &mctx,
                        enc,
                        &s.attn_q_full,
                        &ab.q_norm,
                        &s.attn_q_normed,
                        &s.attn_k_now,
                        &ab.k_norm,
                        &s.attn_k_normed,
                        1,
                        n_q,
                        n_kv,
                        head_dim,
                        n_rot,
                        position,
                        RMS_EPS,
                        arch.rope_theta,
                    )?)
                },
                &mut phases,
            )?;
        } else {
            timed(
                "q_norm (batched rms)",
                &|enc| {
                    if sigmoid_mul {
                        Ok(encode_rms_norm_batched_src_strided_f32(
                            &mctx,
                            enc,
                            &s.attn_q_full,
                            &ab.q_norm,
                            &s.attn_q_normed,
                            n_q,
                            head_dim,
                            2 * head_dim,
                            0,
                            RMS_EPS,
                        )?)
                    } else {
                        Ok(encode_rms_norm_batched_f32(
                            &mctx,
                            enc,
                            &s.attn_q,
                            &ab.q_norm,
                            &s.attn_q_normed,
                            n_q,
                            head_dim,
                            RMS_EPS,
                        )?)
                    }
                },
                &mut phases,
            )?;
            timed(
                "k_norm (batched rms)",
                &|enc| {
                    Ok(encode_rms_norm_batched_f32(
                        &mctx,
                        enc,
                        &s.attn_k_now,
                        &ab.k_norm,
                        &s.attn_k_normed,
                        n_kv,
                        head_dim,
                        RMS_EPS,
                    )?)
                },
                &mut phases,
            )?;
            if rope_pair {
                timed(
                    "rope Q+K (paired)",
                    &|enc| {
                        Ok(encode_rope_neox_pair_f32(
                            &mctx,
                            enc,
                            &s.attn_q_normed,
                            &s.attn_k_normed,
                            n_q,
                            n_kv,
                            head_dim,
                            n_rot,
                            position,
                            arch.rope_theta,
                        )?)
                    },
                    &mut phases,
                )?;
            } else {
                timed(
                    "rope Q",
                    &|enc| {
                        Ok(encode_rope_neox_f32(
                            &mctx,
                            enc,
                            &s.attn_q_normed,
                            n_q,
                            head_dim,
                            n_rot,
                            position,
                            arch.rope_theta,
                        )?)
                    },
                    &mut phases,
                )?;
                timed(
                    "rope K",
                    &|enc| {
                        Ok(encode_rope_neox_f32(
                            &mctx,
                            enc,
                            &s.attn_k_normed,
                            n_kv,
                            head_dim,
                            n_rot,
                            position,
                            arch.rope_theta,
                        )?)
                    },
                    &mut phases,
                )?;
            }
        }
        timed(
            "kv scatter (fused)",
            &|enc| match s.kv_k[attn_idx_in_session].dtype {
                GgmlType::F16 => Ok(encode_scatter_offset_f32_to_f16_kv(
                    &mctx,
                    enc,
                    &s.attn_k_normed,
                    &s.attn_v_now,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    (position as usize) * kv_dim,
                    kv_dim,
                )?),
                GgmlType::Q8_0 => Ok(encode_scatter_offset_f32_to_q8_0_kv(
                    &mctx,
                    enc,
                    &s.attn_k_normed,
                    &s.attn_v_now,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    (position as usize) * kv_dim,
                    kv_dim,
                )?),
                other => Err(anyhow!("unsupported KV dtype for attn-intra: {other:?}")),
            },
            &mut phases,
        )?;
        s.kv_n_pos[attn_idx_in_session] = position as usize + 1;
        let n_pos = s.kv_n_pos[attn_idx_in_session];
        let nwg = attn_v4_choose_nwg(n_pos, group);
        let tile_c = attn_v4_choose_tile_c(n_pos, group);
        timed(
            "attn_decode_v4_main",
            &|enc| {
                Ok(encode_attn_decode_v4_main_only_f32(
                    &mctx,
                    enc,
                    &s.attn_q_normed,
                    &s.kv_k[attn_idx_in_session],
                    &s.kv_v[attn_idx_in_session],
                    &s.attn_v4_o_partial,
                    &s.attn_v4_ml_partial,
                    n_q,
                    n_kv,
                    head_dim,
                    n_pos,
                    nwg,
                    tile_c,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "attn_decode_v4_reduce",
            &|enc| {
                Ok(encode_attn_decode_v4_reduce_only_f32(
                    &mctx,
                    enc,
                    &s.attn_v4_o_partial,
                    &s.attn_v4_ml_partial,
                    &s.attn_o,
                    n_q,
                    n_kv,
                    head_dim,
                    nwg,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "gate sigmoid + mul",
            &|enc| {
                if sigmoid_mul {
                    Ok(encode_sigmoid_mul_gate_strided_f32(
                        &mctx,
                        enc,
                        &s.attn_q_full,
                        &s.attn_o,
                        &s.attn_o,
                        n_q,
                        head_dim,
                        2 * head_dim,
                        head_dim,
                    )?)
                } else {
                    encode_sigmoid_f32(&mctx, enc, &s.attn_gate, &s.attn_q)?;
                    Ok(encode_mul_f32(&mctx, enc, &s.attn_o, &s.attn_q, &s.attn_o)?)
                }
            },
            &mut phases,
        )?;
        timed(
            "o_proj (mat_vec)",
            &|enc| {
                Ok(encode_mat_vec_dispatch(
                    &mctx,
                    enc,
                    &ab.o,
                    &s.attn_o,
                    &s.mixer_out,
                    q_dim,
                    h,
                )?)
            },
            &mut phases,
        )?;
        timed(
            "residual_add #1",
            &|enc| Ok(encode_add_inplace_f32(&mctx, enc, &s.x, &s.mixer_out)?),
            &mut phases,
        )?;
        timed(
            "post_norm (rms_norm)",
            &|enc| {
                Ok(encode_rms_norm_mul_f32(
                    &mctx,
                    enc,
                    &s.x,
                    &ab.post_attn_norm,
                    &s.h,
                    RMS_EPS,
                )?)
            },
            &mut phases,
        )?;

        if agg.is_empty() {
            agg = phases;
        } else {
            for ((_, total), (_, ms)) in agg.iter_mut().zip(phases) {
                *total += ms;
            }
        }
    }

    let n_pos_est = target + runs;
    let nwg = attn_v4_choose_nwg(n_pos_est, group);
    let tile_c = attn_v4_choose_tile_c(n_pos_est, group);
    let group_tile = attn_v4_choose_group_tile(n_pos_est, group);
    let subgroups = group / group_tile.max(1);
    let logical_kv_bytes = n_kv * n_pos_est * (head_dim + head_dim) * 2;
    let subgroup_kv_bytes = logical_kv_bytes * subgroups;
    let partial_bytes = n_kv * nwg * group * (head_dim * 4 + 2 * 4);
    let reduce_bytes = partial_bytes + n_q * head_dim * 4;

    let avgs: Vec<(String, f64)> = agg
        .into_iter()
        .map(|(name, ms)| (name, ms / runs as f64))
        .collect();
    let total: f64 = avgs.iter().map(|(_, ms)| *ms).sum();
    println!(
        "[attn-intra ctx={target}] block={block_idx} attn_idx={attn_idx_in_session} \
         attn_layers={total_attn} runs={runs} n_q={n_q} n_kv={n_kv} group={group} \
         group_tile={group_tile} nwg={nwg} tile_c={tile_c}"
    );
    println!(
        "[attn-intra ctx={target}] bytes_est main_gb={:.4} reduce_gb={:.4} \
         logical_kv_gb={:.4} subgroup_kv_gb={:.4}",
        (subgroup_kv_bytes + partial_bytes) as f64 / 1e9,
        reduce_bytes as f64 / 1e9,
        logical_kv_bytes as f64 / 1e9,
        subgroup_kv_bytes as f64 / 1e9,
    );
    println!(
        "[attn-intra ctx={target}] one_layer_avg={total:.4} ms extrapolated={:.4} ms",
        total * total_attn as f64
    );
    println!("phase\tavg_ms\tpct\textrapolated_ms\test_gb\test_gb_s");
    for (name, ms) in &avgs {
        let est_bytes = match name.as_str() {
            "attn_decode_v4_main" => subgroup_kv_bytes + partial_bytes,
            "attn_decode_v4_reduce" => reduce_bytes,
            _ => 0,
        };
        if est_bytes > 0 && *ms > 0.0 {
            let gb = est_bytes as f64 / 1e9;
            println!(
                "{name}\t{ms:.4}\t{:.2}\t{:.4}\t{gb:.4}\t{:.1}",
                ms / total * 100.0,
                ms * total_attn as f64,
                gb / (*ms / 1000.0)
            );
        } else {
            println!(
                "{name}\t{ms:.4}\t{:.2}\t{:.4}\t\t",
                ms / total * 100.0,
                ms * total_attn as f64
            );
        }
    }

    Ok(())
}

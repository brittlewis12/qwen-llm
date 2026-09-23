//! Projection and small-N matmat microbenchmarks.

use super::*;

#[derive(Clone, Copy)]
pub(crate) struct ProjectionWeight<'a> {
    pub(crate) weight: &'a MetalTensor,
    pub(crate) n_out: usize,
}

pub(crate) struct ProjectionBatchBench<'a> {
    pub(crate) name: &'static str,
    pub(crate) n_in: usize,
    pub(crate) max_out: usize,
    pub(crate) input_pack_count: usize,
    pub(crate) weights: Vec<ProjectionWeight<'a>>,
    pub(crate) x_single: MetalTensor,
    pub(crate) x_batch: MetalTensor,
    pub(crate) x_slots: Vec<MetalTensor>,
    pub(crate) y_single: Vec<MetalTensor>,
    pub(crate) y_batch: Vec<MetalTensor>,
    pub(crate) y_sink: MetalTensor,
    pub(crate) weight_bytes: u64,
}

impl<'a> ProjectionBatchBench<'a> {
    pub(crate) fn new(
        ctx: &MetalContext,
        name: &'static str,
        n_in: usize,
        max_tokens: usize,
        input_pack_count: usize,
        weights: Vec<ProjectionWeight<'a>>,
    ) -> Result<Self> {
        if weights.is_empty() {
            return Err(anyhow!("projection group {name} has no weights"));
        }
        let max_out = weights.iter().map(|w| w.n_out).max().unwrap_or(1);
        let weight_bytes = weights.iter().map(|w| w.weight.n_bytes()).sum();
        let mut y_single = Vec::with_capacity(weights.len());
        let mut y_batch = Vec::with_capacity(weights.len());
        for w in &weights {
            y_single.push(MetalTensor::zeros_f32(ctx, vec![w.n_out as u64])?);
            y_batch.push(MetalTensor::zeros_f32(
                ctx,
                vec![(max_tokens * w.n_out) as u64],
            )?);
        }
        let mut x_slots = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            x_slots.push(MetalTensor::zeros_f32(ctx, vec![n_in as u64])?);
        }
        Ok(Self {
            name,
            n_in,
            max_out,
            input_pack_count,
            weights,
            x_single: MetalTensor::zeros_f32(ctx, vec![n_in as u64])?,
            x_batch: MetalTensor::zeros_f32(ctx, vec![(max_tokens * n_in) as u64])?,
            x_slots,
            y_single,
            y_batch,
            y_sink: MetalTensor::zeros_f32(ctx, vec![(max_tokens * max_out) as u64])?,
            weight_bytes,
        })
    }
}

pub(crate) fn blit_projection_group_pack(
    blit: &BlitEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) {
    let row_bytes = (group.n_in * 4) as u64;
    for _ in 0..group.input_pack_count {
        for tok in 0..tokens {
            let dst_offset = group.x_batch.offset + tok as u64 * row_bytes;
            blit.copy_buffer(
                &group.x_slots[tok].buffer,
                group.x_slots[tok].offset,
                &group.x_batch.buffer,
                dst_offset,
                row_bytes,
            );
        }
    }
}

pub(crate) fn blit_projection_group_scatter(
    blit: &BlitEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) {
    for (i, w) in group.weights.iter().enumerate() {
        let row_bytes = (w.n_out * 4) as u64;
        let sink_row_bytes = (group.max_out * 4) as u64;
        for tok in 0..tokens {
            let src_offset = group.y_batch[i].offset + tok as u64 * row_bytes;
            let dst_offset = group.y_sink.offset + tok as u64 * sink_row_bytes;
            blit.copy_buffer(
                &group.y_batch[i].buffer,
                src_offset,
                &group.y_sink.buffer,
                dst_offset,
                row_bytes,
            );
        }
    }
}

pub(crate) fn projection_group_pack_bytes(group: &ProjectionBatchBench<'_>) -> u64 {
    (group.input_pack_count * group.n_in * 4) as u64
}

pub(crate) fn projection_group_scatter_bytes(group: &ProjectionBatchBench<'_>) -> u64 {
    group.weights.iter().map(|w| (w.n_out * 4) as u64).sum()
}

pub(crate) fn encode_projection_group_matvec_seq(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) -> Result<()> {
    for _ in 0..tokens {
        for (i, w) in group.weights.iter().enumerate() {
            encode_mat_vec_dispatch(
                ctx,
                enc,
                w.weight,
                &group.x_single,
                &group.y_single[i],
                group.n_in,
                w.n_out,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn encode_projection_group_matmat_batch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    group: &ProjectionBatchBench<'_>,
    tokens: usize,
) -> Result<()> {
    let x_batch = group
        .x_batch
        .view_subrange(0, vec![(tokens * group.n_in) as u64]);
    for (i, w) in group.weights.iter().enumerate() {
        let y_batch = group.y_batch[i].view_subrange(0, vec![(tokens * w.n_out) as u64]);
        encode_mat_mat_dispatch(
            ctx, enc, w.weight, &x_batch, &y_batch, group.n_in, w.n_out, tokens,
        )?;
    }
    Ok(())
}

pub(crate) fn projection_fill_inputs(
    ctx: &MetalContext,
    groups: &[ProjectionBatchBench<'_>],
) -> Result<()> {
    let cmd = ctx.queue.commandBuffer().context("projection fill cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    for (i, group) in groups.iter().enumerate() {
        let v = 0.03125 + (i as f32) * 0.00390625;
        encode_fill_f32(ctx, &enc, &group.x_single, v)?;
        encode_fill_f32(ctx, &enc, &group.x_batch, v)?;
        for slot in &group.x_slots {
            encode_fill_f32(ctx, &enc, slot, v)?;
        }
    }
    enc.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    Ok(())
}

pub(crate) fn run_gdn_proj_micro(args: GdnProjMicroArgs) -> Result<()> {
    let GdnProjMicroArgs {
        model,
        iters,
        warmup,
        tokens,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let s = MetalSession::fresh(&ctx, &mm, 32).context("session")?;

    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let gdn_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            MetalBlock::Attn(_) => None,
        })
        .collect();
    if gdn_blocks.is_empty() {
        return Err(anyhow!("model has no GDN blocks"));
    }

    let h_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;
    let gdn_normed_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * v_dim) as u64])?;
    let qkv_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * conv_dim) as u64])?;
    let z_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * v_dim) as u64])?;
    let out_batch = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;

    {
        let cmd = ctx.queue.commandBuffer().context("init fill cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_fill_f32(&ctx, &enc, &s.h, 0.125)?;
        encode_fill_f32(&ctx, &enc, &s.gdn_normed, 0.0625)?;
        encode_fill_f32(&ctx, &enc, &h_batch, 0.125)?;
        encode_fill_f32(&ctx, &enc, &gdn_normed_batch, 0.0625)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
    }

    let qkv_bytes: u64 = gdn_blocks.iter().map(|gb| gb.in_proj_qkv.n_bytes()).sum();
    let z_bytes: u64 = gdn_blocks.iter().map(|gb| gb.in_proj_z.n_bytes()).sum();
    let beta_bytes: u64 = gdn_blocks.iter().map(|gb| gb.beta_proj.n_bytes()).sum();
    let out_bytes: u64 = gdn_blocks.iter().map(|gb| gb.out_proj.n_bytes()).sum();
    let beta_f32 = gdn_blocks
        .iter()
        .all(|gb| gb.beta_proj.dtype == GgmlType::F32);

    println!(
        "[gdn-proj-micro] model={} layers={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        gdn_blocks.len(),
        h,
        conv_dim,
        v_dim,
        tokens,
        warmup,
        iters
    );
    println!(
        "phase\tmode\ttokens\tbytes_gb_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\teff_weight_gb_s"
    );

    let report = |label: &str, mode: &str, bytes_per_token: u64, wall_ms: f64, gpu_ms: f64| {
        let bytes_gb = bytes_per_token as f64 / 1e9;
        let eff_gb = bytes_gb * tokens as f64;
        let gpu_per_tok = gpu_ms / tokens as f64;
        let gb_s = eff_gb / (gpu_ms / 1e3);
        println!(
            "{label}\t{mode}\t{tokens}\t{bytes_gb:.4}\t{wall_ms:.4}\t{gpu_ms:.4}\t{gpu_per_tok:.4}\t{gb_s:.1}"
        );
    };

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_qkv, &s.h, &s.gdn_qkv, h, conv_dim)?;
            }
        }
        Ok(())
    })?;
    report("qkv", "matvec_seq", qkv_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_qkv,
                &h_batch,
                &qkv_batch,
                h,
                conv_dim,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("qkv", "matmat_batch", qkv_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
            }
        }
        Ok(())
    })?;
    report("z", "matvec_seq", z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_z,
                &h_batch,
                &z_batch,
                h,
                v_dim,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("z", "matmat_batch", z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_qkv, &s.h, &s.gdn_qkv, h, conv_dim)?;
                encode_mat_vec_dispatch(&ctx, enc, &gb.in_proj_z, &s.h, &s.gdn_z, h, v_dim)?;
            }
        }
        Ok(())
    })?;
    report("qkv+z", "matvec_seq", qkv_bytes + z_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_qkv,
                &h_batch,
                &qkv_batch,
                h,
                conv_dim,
                tokens,
            )?;
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.in_proj_z,
                &h_batch,
                &z_batch,
                h,
                v_dim,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("qkv+z", "matmat_batch", qkv_bytes + z_bytes, wall, gpu);

    if beta_f32 {
        let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
            for _ in 0..tokens {
                for gb in &gdn_blocks {
                    encode_mat_vec_dispatch(&ctx, enc, &gb.beta_proj, &s.h, &s.gdn_b, h, n_v)?;
                    encode_sigmoid_f32(&ctx, enc, &s.gdn_b, &s.gdn_beta)?;
                }
            }
            Ok(())
        })?;
        report("beta", "matvec+sigmoid", beta_bytes, wall, gpu);

        let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
            for _ in 0..tokens {
                for gb in &gdn_blocks {
                    encode_mat_vec_f32_sigmoid(
                        &ctx,
                        enc,
                        &gb.beta_proj,
                        &s.h,
                        &s.gdn_beta,
                        h,
                        n_v,
                    )?;
                }
            }
            Ok(())
        })?;
        report("beta", "matvec_sigmoid_fused", beta_bytes, wall, gpu);
    } else {
        println!("beta\tskipped\t{tokens}\t0.0000\t0.0000\t0.0000\t0.0000\t0.0");
    }

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for _ in 0..tokens {
            for gb in &gdn_blocks {
                encode_mat_vec_dispatch(
                    &ctx,
                    enc,
                    &gb.out_proj,
                    &s.gdn_normed,
                    &s.mixer_out,
                    v_dim,
                    h,
                )?;
            }
        }
        Ok(())
    })?;
    report("out", "matvec_seq", out_bytes, wall, gpu);

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for gb in &gdn_blocks {
            encode_mat_mat_dispatch(
                &ctx,
                enc,
                &gb.out_proj,
                &gdn_normed_batch,
                &out_batch,
                v_dim,
                h,
                tokens,
            )?;
        }
        Ok(())
    })?;
    report("out", "matmat_batch", out_bytes, wall, gpu);

    Ok(())
}

/// v0.77 small-N mat-mat variant sweep at N=8 on production shapes.
///
/// Motivation: the interleaved verify microbench showed packed-verify cost
/// is per-dispatch kernel efficiency c(n) on the weight sweep (fixed
/// overhead ≈ 0), so the whole Spec(8) premium sits in the v0.501 table's
/// N=8 picks (Q4_K `r1c1k64_sg2` c 2.25-2.78, Q6_K `r1c1k128` c 1.6-1.8,
/// Q5_K generic). This times every drop-in alternative per weight family;
/// winners get promoted into `encode_mat_mat_dispatch`'s table.
pub(crate) fn run_matmat_smalln_micro(args: MatmatSmallnMicroArgs) -> Result<()> {
    let MatmatSmallnMicroArgs {
        model,
        iters,
        warmup,
        tensors_per_family,
    } = args;
    if iters == 0 || tensors_per_family == 0 {
        return Err(anyhow!("--iters and --tensors-per-family must be >= 1"));
    }
    const N_COLS: usize = 8;

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;

    // Synthetic Q8_0 tensors at the DFlash 2 drafter's production shapes:
    // the drafter GGUF is a different arch (`dflash`) this subcommand
    // can't load, and timing is content-independent for memory-bound
    // kernels, so valid-format synthetic blocks stand in. Created before
    // `families` so the borrows below outlive it.
    let q8_shapes: [(&'static str, usize, usize); 6] = [
        ("q8_attn_q", 5120, 4096),
        ("q8_attn_kv", 5120, 1024),
        ("q8_attn_o", 4096, 5120),
        ("q8_ffn_gate", 5120, 17408),
        ("q8_ffn_down", 17408, 5120),
        ("q8_conv_proj", 5120, 1280),
    ];
    let mut q8_synth: Vec<(&'static str, MetalTensor)> = Vec::new();
    for (label, n_in, n_out) in q8_shapes {
        // block_q8_0: f16 scale + 32 i8, 34 bytes / 32 elems.
        let n_blocks = n_in * n_out / 32;
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        for (i, b) in block[2..].iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_sub(16);
        }
        let bytes: Vec<u8> = block.iter().copied().cycle().take(n_blocks * 34).collect();
        let t = MetalTensor::from_bytes(
            &ctx,
            &bytes,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        )?;
        q8_synth.push((label, t));
    }

    // Weight families in verify-path dispatch order. gate/up share shape +
    // dtype, so gate stands in for both.
    // Group by (family, dtype): Q4_K_M quantizes some instances of a
    // family at Q4_K and others at Q6_K — mixing them blurs per-shape
    // winner attribution (the first sweep run hit exactly that).
    let mut families: Vec<(String, Vec<&MetalTensor>)> = vec![(
        format!("lm_head[{:?}]", mm.lm_head.dtype),
        vec![&mm.lm_head],
    )];
    for (label, t) in &q8_synth {
        families.push((format!("{label}[Q8_0]"), vec![t]));
    }
    fn push_family<'a>(
        families: &mut Vec<(String, Vec<&'a MetalTensor>)>,
        cap: usize,
        label: &str,
        t: &'a MetalTensor,
    ) {
        let key = format!("{label}[{:?}]", t.dtype);
        match families.iter_mut().find(|(l, _)| *l == key) {
            Some((_, v)) => {
                if v.len() < cap {
                    v.push(t);
                }
            }
            None => families.push((key, vec![t])),
        }
    }
    let cap = tensors_per_family;
    for b in &mm.blocks {
        match b {
            MetalBlock::Gdn(gb) => {
                push_family(&mut families, cap, "gdn_qkv", &gb.in_proj_qkv);
                push_family(&mut families, cap, "gdn_z", &gb.in_proj_z);
                push_family(&mut families, cap, "gdn_out", &gb.out_proj);
                push_family(&mut families, cap, "ffn_gate", &gb.ffn_gate);
                push_family(&mut families, cap, "ffn_down", &gb.ffn_down);
            }
            MetalBlock::Attn(ab) => {
                push_family(&mut families, cap, "attn_q", &ab.q);
                push_family(&mut families, cap, "attn_o", &ab.o);
            }
        }
    }
    let mut max_in = 0usize;
    let mut max_out = 0usize;
    for (_, ts) in &families {
        for t in ts {
            let [n_in, n_out] = t.shape.as_slice() else {
                continue;
            };
            max_in = max_in.max(*n_in as usize);
            max_out = max_out.max(*n_out as usize);
        }
    }
    let x8 = MetalTensor::zeros_f32(&ctx, vec![(N_COLS * max_in) as u64])?;
    let y8 = MetalTensor::zeros_f32(&ctx, vec![(N_COLS * max_out) as u64])?;
    {
        let cmd = ctx.queue.commandBuffer().context("fill cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_fill_f32(&ctx, &enc, &x8, 0.125)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
    }

    println!(
        "[matmat-smalln-micro] model={} n_cols={N_COLS} warmup={warmup} iters={iters} tensors_per_family={tensors_per_family}",
        model.display()
    );
    println!("family\tdtype\tn_in\tn_out\tcount\tcandidate\tgpu_ms_per_dispatch\teff_weight_gb_s");

    for (label, tensors) in &families {
        let Some(first) = tensors.first() else {
            continue;
        };
        let [n_in, n_out] = first.shape.as_slice() else {
            continue;
        };
        let (n_in, n_out) = (*n_in as usize, *n_out as usize);
        let dtype = first.dtype;
        let count = tensors.len();
        let bytes_per_dispatch: u64 =
            tensors.iter().map(|t| t.n_bytes()).sum::<u64>() / count as u64;
        let x = x8.view_subrange(0, vec![(N_COLS * n_in) as u64]);
        let y = y8.view_subrange(0, vec![(N_COLS * n_out) as u64]);

        let candidates: Vec<&'static str> = match dtype {
            GgmlType::Q4_K => vec![
                "table",
                "generic",
                "seq8",
                "nc8",
                "mma8_r2c1k64",
                "mma8_r1c1k128",
                "mma8_r1c1k128_vec4",
                "mma8_r1c1k64_sg2",
                "mma8_r1c1k64_sg2_vec4",
                "mma8_r2c1k64_vec4",
                "mma8_r2c1k128",
                "mma8_r4c1k64",
            ],
            GgmlType::Q6_K => vec![
                "table",
                "generic",
                "seq8",
                "nc8",
                "mma8_r2c1k64",
                "mma8_r1c1k128",
                "mma8_r1c1k64_sg2",
                "mma8_r2c1k128",
                "mma8_r4c1k64",
            ],
            // v0.77: Q5_K/Q8_0 mma8v variants exist now (no nc kernels).
            GgmlType::Q5_K | GgmlType::Q8_0 => vec![
                "table",
                "generic",
                "seq8",
                "mma8_r2c1k64",
                "mma8_r1c1k128",
                "mma8_r1c1k64_sg2",
                "mma8_r2c1k128",
                "mma8_r4c1k64",
            ],
            _ => vec!["table", "seq8"],
        };

        for cand in candidates {
            let result = time_gpu_reps(&ctx, warmup, iters, |enc| {
                for w in tensors {
                    match cand {
                        "table" => {
                            encode_mat_mat_dispatch(&ctx, enc, w, &x, &y, n_in, n_out, N_COLS)?
                        }
                        "generic" => match dtype {
                            GgmlType::Q4_K => qwen_llm::metal::encode_mat_mat_q4_k_f32(
                                &ctx, enc, w, &x, &y, n_in, n_out, N_COLS,
                            )?,
                            GgmlType::Q5_K => qwen_llm::metal::encode_mat_mat_q5_k_f32(
                                &ctx, enc, w, &x, &y, n_in, n_out, N_COLS,
                            )?,
                            GgmlType::Q6_K => qwen_llm::metal::encode_mat_mat_q6_k_f32(
                                &ctx, enc, w, &x, &y, n_in, n_out, N_COLS,
                            )?,
                            GgmlType::Q8_0 => qwen_llm::metal::encode_mat_mat_q8_0_f32(
                                &ctx, enc, w, &x, &y, n_in, n_out, N_COLS,
                            )?,
                            other => return Err(anyhow!("no generic arm for {other:?}")),
                        },
                        "seq8" => {
                            for r in 0..N_COLS {
                                let xr = x.view_subrange((r * n_in) as u64, vec![n_in as u64]);
                                let yr = y.view_subrange((r * n_out) as u64, vec![n_out as u64]);
                                encode_mat_vec_dispatch(&ctx, enc, w, &xr, &yr, n_in, n_out)?;
                            }
                        }
                        "nc8" => qwen_llm::metal::encode_mat_vec_nc_dispatch(
                            &ctx, enc, w, &x, &y, n_in, n_out, N_COLS,
                        )?,
                        v => {
                            let variant = v.strip_prefix("mma8_").expect("mma8 candidate");
                            qwen_llm::metal::encode_mat_mat_mma8_variant(
                                &ctx, enc, w, &x, &y, n_in, n_out, variant,
                            )?
                        }
                    }
                }
                Ok(())
            });
            match result {
                Ok((_wall, gpu)) => {
                    let gpu_per_dispatch = gpu / count as f64;
                    let gb_s = (bytes_per_dispatch as f64 / 1e9) / (gpu_per_dispatch / 1e3);
                    println!(
                        "{label}\t{dtype:?}\t{n_in}\t{n_out}\t{count}\t{cand}\t{gpu_per_dispatch:.4}\t{gb_s:.1}"
                    );
                }
                Err(e) => {
                    println!("{label}\t{dtype:?}\t{n_in}\t{n_out}\t{count}\t{cand}\tn/a\t({e})");
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn run_decode_proj_batch(args: DecodeProjBatchArgs) -> Result<()> {
    let DecodeProjBatchArgs {
        model,
        mut tokens,
        iters,
        warmup,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens.is_empty() || tokens.contains(&0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let head_dim = arch.attn_head_dim as usize;
    let n_q = arch.n_q_heads as usize;
    let n_kv = arch.n_kv_heads as usize;
    let q_dim = n_q * head_dim;
    let kv_dim = n_kv * head_dim;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let gdn_head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * gdn_head_dim;
    let v_dim = n_v * gdn_head_dim;
    let ffn_dim = if arch.kind == qwen_llm::model::ArchKind::Moe {
        arch.expert_shared_feed_forward_length as usize
    } else {
        arch.intermediate_size as usize
    };

    let gdn_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => Some(g),
            MetalBlock::Attn(_) => None,
        })
        .collect();
    let attn_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(_) => None,
            MetalBlock::Attn(a) => Some(a),
        })
        .collect();

    let mut groups = Vec::new();
    if !gdn_blocks.is_empty() {
        let mut weights = Vec::with_capacity(gdn_blocks.len() * 2);
        for gb in &gdn_blocks {
            weights.push(ProjectionWeight {
                weight: &gb.in_proj_qkv,
                n_out: conv_dim,
            });
            weights.push(ProjectionWeight {
                weight: &gb.in_proj_z,
                n_out: v_dim,
            });
        }
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "gdn_qkv_z",
            h,
            max_tokens,
            gdn_blocks.len(),
            weights,
        )?);

        let weights = gdn_blocks
            .iter()
            .map(|gb| ProjectionWeight {
                weight: &gb.out_proj,
                n_out: h,
            })
            .collect();
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "gdn_out",
            v_dim,
            max_tokens,
            gdn_blocks.len(),
            weights,
        )?);
    }

    if !attn_blocks.is_empty() {
        let mut weights = Vec::with_capacity(attn_blocks.len() * 3);
        for ab in &attn_blocks {
            weights.push(ProjectionWeight {
                weight: &ab.q,
                n_out: 2 * q_dim,
            });
            weights.push(ProjectionWeight {
                weight: &ab.k,
                n_out: kv_dim,
            });
            weights.push(ProjectionWeight {
                weight: &ab.v,
                n_out: kv_dim,
            });
        }
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "attn_qkv",
            h,
            max_tokens,
            attn_blocks.len(),
            weights,
        )?);

        let weights = attn_blocks
            .iter()
            .map(|ab| ProjectionWeight {
                weight: &ab.o,
                n_out: h,
            })
            .collect();
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "attn_o",
            q_dim,
            max_tokens,
            attn_blocks.len(),
            weights,
        )?);
    }

    if ffn_dim > 0 {
        let mut gate_up = Vec::with_capacity(mm.blocks.len() * 2);
        let mut down = Vec::with_capacity(mm.blocks.len());
        for block in &mm.blocks {
            match block {
                MetalBlock::Gdn(gb) => {
                    gate_up.push(ProjectionWeight {
                        weight: &gb.ffn_gate,
                        n_out: ffn_dim,
                    });
                    gate_up.push(ProjectionWeight {
                        weight: &gb.ffn_up,
                        n_out: ffn_dim,
                    });
                    down.push(ProjectionWeight {
                        weight: &gb.ffn_down,
                        n_out: h,
                    });
                }
                MetalBlock::Attn(ab) => {
                    gate_up.push(ProjectionWeight {
                        weight: &ab.ffn_gate,
                        n_out: ffn_dim,
                    });
                    gate_up.push(ProjectionWeight {
                        weight: &ab.ffn_up,
                        n_out: ffn_dim,
                    });
                    down.push(ProjectionWeight {
                        weight: &ab.ffn_down,
                        n_out: h,
                    });
                }
            }
        }
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "ffn_gate_up_dense_or_shared",
            h,
            max_tokens,
            mm.blocks.len(),
            gate_up,
        )?);
        groups.push(ProjectionBatchBench::new(
            &ctx,
            "ffn_down_dense_or_shared",
            ffn_dim,
            max_tokens,
            mm.blocks.len(),
            down,
        )?);
    }

    groups.push(ProjectionBatchBench::new(
        &ctx,
        "lm_head",
        h,
        max_tokens,
        1,
        vec![ProjectionWeight {
            weight: &mm.lm_head,
            n_out: arch.vocab_size as usize,
        }],
    )?);

    projection_fill_inputs(&ctx, &groups)?;

    println!(
        "[decode-proj-batch] model={} kind={:?} layers={} gdn_layers={} attn_layers={} h={} q_dim={} kv_dim={} conv_dim={} v_dim={} ffn_dim={} vocab={} tokens={} warmup={} iters={}",
        model.display(),
        arch.kind,
        mm.blocks.len(),
        gdn_blocks.len(),
        attn_blocks.len(),
        h,
        q_dim,
        kv_dim,
        conv_dim,
        v_dim,
        ffn_dim,
        arch.vocab_size,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );
    println!(
        "component\tmode\ttokens\tn_in\tmax_out\tweights\tweight_gb_per_token\tdispatches_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\teff_weight_gb_s\tsaving_ms_per_tok\tsaving_pct"
    );

    for &n_tokens in &tokens {
        let mut sum_seq_gpu = 0.0f64;
        let mut sum_batch_gpu = 0.0f64;
        let mut total_weight_bytes = 0u64;
        let mut total_weights = 0usize;

        for group in &groups {
            let seq = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
                encode_projection_group_matvec_seq(&ctx, enc, group, n_tokens)
            })?;
            let batch = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
                encode_projection_group_matmat_batch(&ctx, enc, group, n_tokens)
            })?;
            let seq_per_tok = seq.avg_gpu_ms / n_tokens as f64;
            let batch_per_tok = batch.avg_gpu_ms / n_tokens as f64;
            let save = seq_per_tok - batch_per_tok;
            let pct = if seq_per_tok > 0.0 {
                save / seq_per_tok * 100.0
            } else {
                0.0
            };
            let weight_gb = group.weight_bytes as f64 / 1e9;
            let seq_gb_s = weight_gb * n_tokens as f64 / (seq.avg_gpu_ms / 1e3);
            let batch_gb_s = weight_gb * n_tokens as f64 / (batch.avg_gpu_ms / 1e3);
            let seq_dispatch = group.weights.len() as f64;
            let batch_dispatch = group.weights.len() as f64 / n_tokens as f64;
            let pack_bytes = projection_group_pack_bytes(group);
            let scatter_bytes = projection_group_scatter_bytes(group);
            let pack = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_pack(&blit, group, n_tokens);
                blit.end();
                Ok(())
            })?;
            let scatter = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_scatter(&blit, group, n_tokens);
                blit.end();
                Ok(())
            })?;
            let with_layout = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_pack(&blit, group, n_tokens);
                blit.end();
                let enc = KernelEncoder::begin(cmd);
                encode_projection_group_matmat_batch(&ctx, &enc, group, n_tokens)?;
                enc.end();
                let blit = BlitEncoder::begin(cmd);
                blit_projection_group_scatter(&blit, group, n_tokens);
                blit.end();
                Ok(())
            })?;
            let pack_gb = pack_bytes as f64 / 1e9;
            let scatter_gb = scatter_bytes as f64 / 1e9;
            let pack_gb_s = pack_gb * n_tokens as f64 / (pack.avg_gpu_ms / 1e3);
            let scatter_gb_s = scatter_gb * n_tokens as f64 / (scatter.avg_gpu_ms / 1e3);
            let with_layout_per_tok = with_layout.avg_gpu_ms / n_tokens as f64;
            let with_layout_save = seq_per_tok - with_layout_per_tok;
            let with_layout_pct = if seq_per_tok > 0.0 {
                with_layout_save / seq_per_tok * 100.0
            } else {
                0.0
            };
            println!(
                "{}\tmatvec_seq\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                weight_gb,
                seq_dispatch,
                seq.avg_wall_ms,
                seq.avg_gpu_ms,
                seq_per_tok,
                seq.p50_gpu_ms / n_tokens as f64,
                seq.p90_gpu_ms / n_tokens as f64,
                seq.max_gpu_ms / n_tokens as f64,
                seq_gb_s,
                0.0,
                0.0
            );
            println!(
                "{}\tmatmat_batch\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                weight_gb,
                batch_dispatch,
                batch.avg_wall_ms,
                batch.avg_gpu_ms,
                batch_per_tok,
                batch.p50_gpu_ms / n_tokens as f64,
                batch.p90_gpu_ms / n_tokens as f64,
                batch.max_gpu_ms / n_tokens as f64,
                batch_gb_s,
                save,
                pct
            );
            println!(
                "{}\tlayout_pack\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.input_pack_count,
                pack_gb,
                group.input_pack_count as f64,
                pack.avg_wall_ms,
                pack.avg_gpu_ms,
                pack.avg_gpu_ms / n_tokens as f64,
                pack.p50_gpu_ms / n_tokens as f64,
                pack.p90_gpu_ms / n_tokens as f64,
                pack.max_gpu_ms / n_tokens as f64,
                pack_gb_s,
                0.0,
                0.0
            );
            println!(
                "{}\tlayout_scatter\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                scatter_gb,
                group.weights.len() as f64,
                scatter.avg_wall_ms,
                scatter.avg_gpu_ms,
                scatter.avg_gpu_ms / n_tokens as f64,
                scatter.p50_gpu_ms / n_tokens as f64,
                scatter.p90_gpu_ms / n_tokens as f64,
                scatter.max_gpu_ms / n_tokens as f64,
                scatter_gb_s,
                0.0,
                0.0
            );
            println!(
                "{}\tmatmat_with_layout\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
                group.name,
                n_tokens,
                group.n_in,
                group.max_out,
                group.weights.len(),
                weight_gb + pack_gb + scatter_gb,
                batch_dispatch + group.input_pack_count as f64 + group.weights.len() as f64,
                with_layout.avg_wall_ms,
                with_layout.avg_gpu_ms,
                with_layout_per_tok,
                with_layout.p50_gpu_ms / n_tokens as f64,
                with_layout.p90_gpu_ms / n_tokens as f64,
                with_layout.max_gpu_ms / n_tokens as f64,
                (weight_gb + pack_gb + scatter_gb) * n_tokens as f64
                    / (with_layout.avg_gpu_ms / 1e3),
                with_layout_save,
                with_layout_pct
            );
            sum_seq_gpu += seq.avg_gpu_ms;
            sum_batch_gpu += batch.avg_gpu_ms;
            total_weight_bytes += group.weight_bytes;
            total_weights += group.weights.len();
        }

        let seq_all = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
            for group in &groups {
                encode_projection_group_matvec_seq(&ctx, enc, group, n_tokens)?;
            }
            Ok(())
        })?;
        let batch_all = time_gpu_reps_stats(&ctx, warmup, iters, |enc| {
            for group in &groups {
                encode_projection_group_matmat_batch(&ctx, enc, group, n_tokens)?;
            }
            Ok(())
        })?;
        let with_layout_all = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            let blit = BlitEncoder::begin(cmd);
            for group in &groups {
                blit_projection_group_pack(&blit, group, n_tokens);
            }
            blit.end();
            let enc = KernelEncoder::begin(cmd);
            for group in &groups {
                encode_projection_group_matmat_batch(&ctx, &enc, group, n_tokens)?;
            }
            enc.end();
            let blit = BlitEncoder::begin(cmd);
            for group in &groups {
                blit_projection_group_scatter(&blit, group, n_tokens);
            }
            blit.end();
            Ok(())
        })?;

        let isolated_save = (sum_seq_gpu - sum_batch_gpu) / n_tokens as f64;
        let one_encoder_seq_per_tok = seq_all.avg_gpu_ms / n_tokens as f64;
        let one_encoder_batch_per_tok = batch_all.avg_gpu_ms / n_tokens as f64;
        let one_encoder_save = one_encoder_seq_per_tok - one_encoder_batch_per_tok;
        let with_layout_per_tok = with_layout_all.avg_gpu_ms / n_tokens as f64;
        let with_layout_save = one_encoder_seq_per_tok - with_layout_per_tok;
        let one_encoder_pct = if one_encoder_seq_per_tok > 0.0 {
            one_encoder_save / one_encoder_seq_per_tok * 100.0
        } else {
            0.0
        };
        let with_layout_pct = if one_encoder_seq_per_tok > 0.0 {
            with_layout_save / one_encoder_seq_per_tok * 100.0
        } else {
            0.0
        };
        let weight_gb = total_weight_bytes as f64 / 1e9;
        let layout_bytes: u64 = groups
            .iter()
            .map(|group| projection_group_pack_bytes(group) + projection_group_scatter_bytes(group))
            .sum();
        let layout_gb = layout_bytes as f64 / 1e9;
        let seq_gb_s = weight_gb * n_tokens as f64 / (seq_all.avg_gpu_ms / 1e3);
        let batch_gb_s = weight_gb * n_tokens as f64 / (batch_all.avg_gpu_ms / 1e3);
        println!(
            "aggregate_isolated\tsummed_groups\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t0.0000\t{:.4}\t{:.4}\t0.0000\t0.0000\t0.0000\t0.0\t{:.4}\t0.0",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64,
            sum_seq_gpu,
            sum_seq_gpu / n_tokens as f64,
            isolated_save
        );
        println!(
            "aggregate_isolated\tsummed_matmat\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t0.0000\t{:.4}\t{:.4}\t0.0000\t0.0000\t0.0000\t0.0\t{:.4}\t{:.1}",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64 / n_tokens as f64,
            sum_batch_gpu,
            sum_batch_gpu / n_tokens as f64,
            isolated_save,
            if sum_seq_gpu > 0.0 {
                (sum_seq_gpu - sum_batch_gpu) / sum_seq_gpu * 100.0
            } else {
                0.0
            }
        );
        println!(
            "aggregate_one_encoder\tmatvec_seq\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t0.0",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64,
            seq_all.avg_wall_ms,
            seq_all.avg_gpu_ms,
            one_encoder_seq_per_tok,
            seq_all.p50_gpu_ms / n_tokens as f64,
            seq_all.p90_gpu_ms / n_tokens as f64,
            seq_all.max_gpu_ms / n_tokens as f64,
            seq_gb_s,
            0.0
        );
        println!(
            "aggregate_one_encoder\tmatmat_batch\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
            n_tokens,
            total_weights,
            weight_gb,
            total_weights as f64 / n_tokens as f64,
            batch_all.avg_wall_ms,
            batch_all.avg_gpu_ms,
            one_encoder_batch_per_tok,
            batch_all.p50_gpu_ms / n_tokens as f64,
            batch_all.p90_gpu_ms / n_tokens as f64,
            batch_all.max_gpu_ms / n_tokens as f64,
            batch_gb_s,
            one_encoder_save,
            one_encoder_pct
        );
        println!(
            "aggregate_one_encoder\tmatmat_with_layout\t{}\t0\t0\t{}\t{:.4}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.4}\t{:.1}",
            n_tokens,
            total_weights,
            weight_gb + layout_gb,
            total_weights as f64 / n_tokens as f64,
            with_layout_all.avg_wall_ms,
            with_layout_all.avg_gpu_ms,
            with_layout_per_tok,
            with_layout_all.p50_gpu_ms / n_tokens as f64,
            with_layout_all.p90_gpu_ms / n_tokens as f64,
            with_layout_all.max_gpu_ms / n_tokens as f64,
            (weight_gb + layout_gb) * n_tokens as f64 / (with_layout_all.avg_gpu_ms / 1e3),
            with_layout_save,
            with_layout_pct
        );
    }

    Ok(())
}

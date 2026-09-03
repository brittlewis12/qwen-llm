//! Metal counters, pipeline audit, and dispatch census subcommands.

use super::*;

pub(crate) fn run_metal_counters(_args: MetalCountersArgs) -> Result<()> {
    let ctx = MetalContext::new()?;
    let caps = ctx.counter_capabilities();
    println!("device\t{}", ctx.describe());
    println!(
        "sampling\tstage={}\tdispatch={}\tblit={}",
        caps.supports_stage_boundary, caps.supports_dispatch_boundary, caps.supports_blit_boundary
    );
    println!("counter_sets\t{}", caps.sets.len());
    for set in &caps.sets {
        println!(
            "set\t{}\tcounters={}\tsample_buffer={}",
            set.name,
            set.counters.len(),
            set.sample_buffer_status
        );
        for counter in &set.counters {
            println!("counter\t{}\t{}", set.name, counter);
        }
    }
    Ok(())
}

pub(crate) const HOT_DECODE_PIPELINE_AUDIT: &[&str] = &[
    "kernel_attn_decode_v4_g8_t4_c64_f32",
    "kernel_attn_decode_v4_g8_t4_c128_f32",
    "kernel_attn_decode_v4_g8_t2_c64_f32",
    "kernel_attn_decode_v4_g8_t2_c128_f32",
    "kernel_attn_decode_v4_g16_t4_c64_f32",
    "kernel_attn_decode_v4_g16_t4_c128_f32",
    "kernel_attn_decode_v4_reduce_h2_g8_f32",
    "kernel_attn_decode_v4_reduce_h2_g16_f32",
    "kernel_gdn_prep_parallel_state_f32",
    "kernel_gdn_decay_chain_f32",
    "kernel_l2_norm_pair_hd128_r4_f32",
    "kernel_gdn_step_decay_f32",
    "kernel_gdn_step_decay_packed_f32",
    "kernel_gdn_step_decay_packed_nsg4_f32",
    "kernel_rmsnorm_gated_hd128_r4_f32",
    "kernel_mat_vec_q4_K_f32",
    "kernel_mat_vec_q5_K_f32",
    "kernel_mat_vec_q6_K_f32",
    "kernel_mat_vec_q8_0_f32",
    "kernel_moe_swiglu_q4_K_f32_grouped_slots_n16",
    "kernel_moe_swiglu_q5_K_f32_grouped_slots_n16",
    "kernel_moe_down_q5_K_f32_grouped_slots",
    "kernel_moe_down_q5_K_f32_grouped_slots_tiny8_r16",
    "kernel_moe_down_q6_K_f32_grouped_slots",
    "kernel_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2",
    "kernel_sigmoid_mul_gate_strided_f32",
    "kernel_argmax_f32",
];

pub(crate) fn run_metal_pipelines(args: MetalPipelinesArgs) -> Result<()> {
    let ctx = MetalContext::new()?;
    let names: Vec<String> = if args.kernels.is_empty() {
        HOT_DECODE_PIPELINE_AUDIT
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        args.kernels
    };
    println!("kernel\tthread_width\tmax_threads_per_tg\tstatic_tg_mem\ticb");
    for name in names {
        match ctx.pipeline_info(&name) {
            Ok(info) => println!(
                "{}\t{}\t{}\t{}\t{}",
                info.name,
                info.thread_execution_width,
                info.max_total_threads_per_threadgroup,
                info.static_threadgroup_memory_length,
                info.supports_indirect_command_buffers
            ),
            Err(e) => eprintln!("missing\t{name}\t{e}"),
        }
    }
    Ok(())
}

pub(crate) fn run_dispatch_census(args: DispatchCensusArgs) -> Result<()> {
    use qwen_llm::metal::{dispatch_census_begin, dispatch_census_take};
    let ctx = MetalContext::new()?;
    eprintln!("[census] device: {}", ctx.describe());
    let g = GgufFile::open(&args.model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;
    let mf = MetalForward::new(&ctx, &mm);

    let mut s = MetalSession::fresh(&ctx, &mm, args.ctx + 32)?;
    // PSO warm on the exact profiled path (all splits on for finest labels)
    for i in 0..3 {
        let _ = mf.single_token_argmax_stage_profiled_concurrent_gdn_moe(
            0, i as u32, &mut s, true, true, true,
        )?;
    }
    let mut s = MetalSession::fresh(&ctx, &mm, args.ctx + 32)?;
    if args.ctx > 1 {
        if args.decode_ramp_warm {
            let _ = mf.single_token(0, 0, &mut s)?;
            for p in 1..(args.ctx as u32) {
                let _ = mf.single_token(0, p, &mut s)?;
            }
        } else {
            let ids = vec![0i32; args.ctx];
            let chunk = default_prefill_chunk(mm.arch.kind, args.ctx);
            let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, chunk, ids.len())
                .context("census prefill scratch")?;
            let t0 = Instant::now();
            prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
                .context("census prefill warm")?;
            eprintln!(
                "[census] prefill-warm to {} in {:.1}s",
                args.ctx,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    dispatch_census_begin();
    let (_tok, profile) = mf.single_token_argmax_stage_profiled_concurrent_gdn_moe(
        0,
        args.ctx as u32,
        &mut s,
        true,
        true,
        true,
    )?;
    let rows = dispatch_census_take();

    // family time totals from the same token
    let mut fam_ms: BTreeMap<String, (f64, f64)> = BTreeMap::new(); // ms, pct
    for st in &profile.stages {
        let e = fam_ms.entry(st.family.clone()).or_default();
        e.0 += st.duration_ms_scaled;
        e.1 += st.fraction_of_gpu * 100.0;
    }
    // dispatch shape aggregation per (family, kernel, grid, tg)
    let mut agg: BTreeMap<(String, String, u64, u64), u64> = BTreeMap::new();
    for r in &rows {
        *agg.entry((
            r.family.to_string(),
            r.kernel.clone(),
            r.grid_tgs,
            r.tg_threads,
        ))
        .or_default() += 1;
    }

    println!(
        "[census] ctx={} dispatches={} families={} (stage-profiled token; shapes exact, times +~19% perturbed - use shares)",
        args.ctx,
        rows.len(),
        fam_ms.len()
    );
    println!("family\tms\tpct_gpu\tkernel\tcount\tgrid_tgs\ttg_threads\tsimdgroups\tcore_fill_pct");
    let mut fam_sorted: Vec<_> = fam_ms.iter().collect();
    fam_sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let mut json_rows = Vec::new();
    for (fam, (ms, pct)) in &fam_sorted {
        let mut first = true;
        for ((f, kernel, grid, tg), count) in agg.iter() {
            if f != *fam {
                continue;
            }
            let sg = grid * (tg / 32).max(1);
            // one-simdgroup-per-TG fill estimate vs 40 cores
            let fill = (*grid as f64 / TP_GPU_CORES * 100.0).min(100.0);
            println!(
                "{}\t{:.3}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{:.0}",
                if first { fam.as_str() } else { "" },
                if first { *ms } else { 0.0 },
                if first { *pct } else { 0.0 },
                kernel,
                count,
                grid,
                tg,
                sg,
                fill
            );
            json_rows.push(serde_json::json!({
                "family": fam, "family_ms": ms, "family_pct": pct,
                "kernel": kernel, "count": count, "grid_tgs": grid,
                "tg_threads": tg, "simdgroups_per_dispatch": sg,
            }));
            first = false;
        }
    }
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        &args.out,
        serde_json::to_string_pretty(&serde_json::json!({
            "model": args.model.display().to_string(),
            "ctx": args.ctx,
            "gpu_ms_perturbed": profile.token.gpu_kernel_ms,
            "rows": json_rows,
        }))?,
    )?;
    println!("wrote {}", args.out.display());
    Ok(())
}

//! GPU topology probe (bench-only microbenchmark).

use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TpParams {
    pub(crate) spin_iters: u32,
    pub(crate) grid_tgs: u32,
    pub(crate) stages: u32,
    pub(crate) epochs: u32,
    pub(crate) cadence_iters: u32,
    pub(crate) spin_budget: u32,
    pub(crate) traffic_mask: u32,
    pub(crate) sample_every: u32,
    pub(crate) n_elems: u32,
}

pub(crate) const TP_RING_SAMPLES: usize = 64; // must match TP_LAT_SAMPLES in the kernel
pub(crate) const TP_LAT_UNUSED: u32 = 0xFFFF_FFFF;

pub(crate) const TP_GPU_CORES: f64 = 40.0; // M4 Max

pub(crate) struct TpCal {
    fma_ns: f64,
    poll_ns: f64,
}

pub(crate) type TpBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub(crate) fn tp_zero(buf: &TpBuffer, n_bytes: usize) {
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, n_bytes) }
}

pub(crate) fn tp_fill_u32(buf: &TpBuffer, n: usize, v: u32) {
    let p = buf.contents().as_ptr() as *mut u32;
    for i in 0..n {
        unsafe { p.add(i).write(v) }
    }
}

pub(crate) fn tp_read_u32(buf: &TpBuffer, n: usize) -> Vec<u32> {
    let p = buf.contents().as_ptr() as *const u32;
    (0..n).map(|i| unsafe { p.add(i).read() }).collect()
}

pub(crate) fn tp_read_f32_bits(buf: &TpBuffer, n: usize) -> Vec<u32> {
    tp_read_u32(buf, n)
}

/// One serial-encoder command buffer; returns GPU ms.
pub(crate) fn tp_run_timed<F>(ctx: &MetalContext, encode: F) -> Result<f64>
where
    F: FnOnce(&KernelEncoder) -> Result<()>,
{
    let cmd = ctx.queue.commandBuffer().context("tp cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    encode(&enc)?;
    enc.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3)
}

pub(crate) fn tp_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kernel: &str,
    params: &TpParams,
    buffers: &[&TpBuffer],
    tgs: usize,
    w: usize,
) -> Result<()> {
    let pso = ctx.pipeline(kernel)?;
    anyhow::ensure!(
        pso.maxTotalThreadsPerThreadgroup() >= w,
        "{kernel}: maxTotalThreadsPerThreadgroup {} < requested width {w} \
         (itself a residency datum; record and skip)",
        pso.maxTotalThreadsPerThreadgroup()
    );
    enc.set_pipeline(&pso);
    enc.set_bytes(0, params);
    for (i, b) in buffers.iter().enumerate() {
        enc.set_buffer(i + 1, b, 0);
    }
    enc.dispatch(
        MTLSize {
            width: tgs,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: w,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(crate) fn tp_calibrate(ctx: &MetalContext, seed: &TpBuffer) -> Result<TpCal> {
    // FMA chain rate: 1 TG x 32 threads, dependent chain, min of 3.
    let out = ctx.buffer_uninit(32 * 4)?;
    let iters = 1u32 << 22;
    let mut fma_ms = f64::INFINITY;
    for _ in 0..3 {
        let p = TpParams {
            spin_iters: iters,
            ..Default::default()
        };
        let ms = tp_run_timed(ctx, |enc| {
            tp_dispatch(ctx, enc, "tp_calibrate", &p, &[seed, &out], 1, 32)
        })?;
        fma_ms = fma_ms.min(ms);
    }
    // Poll rate: 1 TG x 32 threads polling a never-changing atomic.
    let flag = ctx.buffer_uninit(4)?;
    tp_zero(&flag, 4);
    let pout = ctx.buffer_uninit(32 * 4)?;
    let polls = 1u32 << 22;
    let mut poll_ms = f64::INFINITY;
    for _ in 0..3 {
        let p = TpParams {
            spin_budget: polls,
            ..Default::default()
        };
        let ms = tp_run_timed(ctx, |enc| {
            tp_dispatch(ctx, enc, "tp_poll_calibrate", &p, &[&flag, &pout], 1, 32)
        })?;
        poll_ms = poll_ms.min(ms);
    }
    Ok(TpCal {
        fma_ns: fma_ms * 1e6 / iters as f64,
        poll_ns: poll_ms * 1e6 / polls as f64,
    })
}

pub(crate) fn tp_percentile_u32(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub(crate) struct TpArmROut {
    pub(crate) rows: Vec<serde_json::Value>,
    /// (variant, traffic, w) -> (max_alive at longest dwell, steady p10,
    /// plateaued?)
    pub(crate) caps: BTreeMap<(String, bool, usize), (u32, u32, bool)>,
}

pub(crate) fn tp_arm_r(
    ctx: &MetalContext,
    cal: &TpCal,
    seed: &TpBuffer,
    traffic: &TpBuffer,
    traffic_mask: u32,
    quick: bool,
) -> Result<TpArmROut> {
    let g_tgs: usize = if quick { 2560 } else { 10240 };
    let dwells_us_full: &[f64] = if quick {
        &[200.0, 1000.0]
    } else {
        &[200.0, 1000.0, 5000.0]
    };
    // traffic rows are qualitative (does residency change under load?) and
    // memory-slow; cap their dwell sweep to keep the budget sane
    let dwells_us_tr: &[f64] = &[200.0, 1000.0];
    let widths: &[usize] = &[32, 64, 128, 256];
    let variants: &[(&str, u32, bool)] = &[
        ("tp_residency_lo", 1, false),
        ("tp_residency_hi8", 8, false),
        ("tp_residency_hi32", 32, false),
        ("tp_residency_hi64", 64, false),
        ("tp_residency_lo_tr", 1, true),
        ("tp_residency_hi8_tr", 8, true),
        ("tp_residency_hi32_tr", 32, true),
        ("tp_residency_hi64_tr", 64, true),
    ];

    let census = ctx.buffer_uninit(8)?;
    let entry = ctx.buffer_uninit(g_tgs * 4)?;
    let out = ctx.buffer_uninit(g_tgs * 4)?;
    let dummy = ctx.buffer_uninit(4)?;
    tp_zero(&dummy, 4);

    let mut rows = Vec::new();
    let mut caps = BTreeMap::new();
    println!("\n== arm R: residency census (grid {g_tgs} TGs) ==");
    println!("variant\ttraffic\tW\tdwell_us\tmax_alive\tper_core\tsteady_p10\twaves\tgpu_ms");
    for &(name, _nacc, tr) in variants {
        // EMPIRICAL per-variant iteration cost (1 TG, uncontended): the
        // NACC accumulator chains pipeline, so scaling fma_ns by NACC
        // over-sizes dwell by up to ~8x (smoke finding). Traffic variants
        // are memory-dominated and calibrate much slower.
        let iter_ns = {
            let cal_spin: u32 = if tr { 1 << 13 } else { 1 << 16 };
            let p = TpParams {
                spin_iters: cal_spin,
                grid_tgs: 1,
                traffic_mask: if tr { traffic_mask } else { 0 },
                ..Default::default()
            };
            tp_zero(&census, 8);
            let tbuf = if tr { traffic } else { &dummy };
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let ms = tp_run_timed(ctx, |enc| {
                    tp_dispatch(
                        ctx,
                        enc,
                        name,
                        &p,
                        &[seed, &census, &entry, &out, tbuf],
                        1,
                        32,
                    )
                })?;
                best = best.min(ms);
            }
            best * 1e6 / cal_spin as f64
        };
        let dwells_us = if tr { dwells_us_tr } else { dwells_us_full };
        let _ = cal; // grid-level fma calibration retained for arms S/D
        for &w in widths {
            let mut per_dwell: Vec<(f64, u32, u32, f64)> = Vec::new();
            for &dw in dwells_us {
                let spin = ((dw * 1e3 / iter_ns).max(1.0)) as u32;
                tp_zero(&census, 8);
                tp_zero(&entry, g_tgs * 4);
                let p = TpParams {
                    spin_iters: spin,
                    grid_tgs: g_tgs as u32,
                    traffic_mask: if tr { traffic_mask } else { 0 },
                    ..Default::default()
                };
                let tbuf = if tr { traffic } else { &dummy };
                let ms = tp_run_timed(ctx, |enc| {
                    tp_dispatch(
                        ctx,
                        enc,
                        name,
                        &p,
                        &[seed, &census, &entry, &out, tbuf],
                        g_tgs,
                        w,
                    )
                })?;
                let max_alive = tp_read_u32(&census, 2)[1];
                let ea = tp_read_u32(&entry, g_tgs);
                // steady distribution: skip the first ramp wave
                let skip = (max_alive as usize).min(g_tgs.saturating_sub(1));
                let mut steady: Vec<u32> = ea[skip..].to_vec();
                steady.sort_unstable();
                let p10 = tp_percentile_u32(&steady, 0.10);
                let p50 = tp_percentile_u32(&steady, 0.50);
                let p90 = tp_percentile_u32(&steady, 0.90);
                let waves = (g_tgs as f64 / max_alive.max(1) as f64).ceil();
                println!(
                    "{name}\t{tr}\t{w}\t{dw:.0}\t{max_alive}\t{:.1}\t{p10}\t{waves:.0}\t{ms:.1}",
                    max_alive as f64 / TP_GPU_CORES
                );
                rows.push(serde_json::json!({
                    "arm": "r", "variant": name, "traffic": tr,
                    "w": w, "dwell_us_nominal": dw, "spin_iters": spin,
                    "iter_ns_1tg": iter_ns,
                    "gpu_ms": ms, "max_alive": max_alive,
                    "per_core": max_alive as f64 / TP_GPU_CORES,
                    "steady_p10": p10, "steady_p50": p50, "steady_p90": p90,
                    "waves": waves,
                }));
                per_dwell.push((dw, max_alive, p10, ms));
            }
            // plateau: last two dwells within 10%
            let n = per_dwell.len();
            let plateaued = n >= 2 && {
                let a = per_dwell[n - 2].1 as f64;
                let b = per_dwell[n - 1].1 as f64;
                (a - b).abs() / a.max(1.0) <= 0.10
            };
            let last = per_dwell[n - 1];
            caps.insert((name.to_string(), tr, w), (last.1, last.2, plateaued));
        }
    }
    Ok(TpArmROut { rows, caps })
}

pub(crate) fn tp_lat_stats(
    all: &[Vec<u32>],
    poll_ns: f64,
    cadence_ns: f64,
) -> (serde_json::Value, Vec<usize>) {
    // all[c][s] = poll counts per sampled epoch slot (aligned across
    // consumers). Global-stall slots: >= 80% of consumers spike >= 10x the
    // overall median sample.
    let mut flat: Vec<u32> = all
        .iter()
        .flat_map(|v| v.iter().copied())
        .filter(|&x| x != TP_LAT_UNUSED)
        .collect();
    if flat.is_empty() {
        return (serde_json::json!({"n": 0}), vec![]);
    }
    flat.sort_unstable();
    let med = flat[flat.len() / 2].max(1);
    let n_slots = all[0].len();
    let mut stall_slots = Vec::new();
    for s in 0..n_slots {
        let mut n_seen = 0usize;
        let mut n_spike = 0usize;
        for c in all {
            let v = c[s];
            if v != TP_LAT_UNUSED {
                n_seen += 1;
                if v >= med.saturating_mul(10) {
                    n_spike += 1;
                }
            }
        }
        if n_seen > 0 && (n_spike as f64) / (n_seen as f64) >= 0.8 {
            stall_slots.push(s);
        }
    }
    let filtered: Vec<u32> = {
        let mut v: Vec<u32> = Vec::new();
        for c in all {
            for (s, &x) in c.iter().enumerate() {
                if x != TP_LAT_UNUSED && !stall_slots.contains(&s) {
                    v.push(x);
                }
            }
        }
        v.sort_unstable();
        v
    };
    let pct = |v: &[u32], q: f64| tp_percentile_u32(v, q) as f64 * poll_ns / 1e3; // us
    // The sampled wait spans a full inter-epoch interval (the consumer
    // starts waiting right after observing the previous epoch), so raw
    // waits legitimately cluster near the producer cadence. The
    // propagation-relevant metric is the EXCESS over cadence; kill gates
    // read filtered_excess_p99_us (smoke finding).
    let excess = |v: &[u32]| -> Vec<u32> {
        let cad_polls = (cadence_ns / poll_ns) as i64;
        let mut e: Vec<u32> = v
            .iter()
            .map(|&x| ((x as i64 - cad_polls).max(0)) as u32)
            .collect();
        e.sort_unstable();
        e
    };
    let flat_ex = excess(&flat);
    let filt_ex = excess(&filtered);
    let j = serde_json::json!({
        "n": flat.len(),
        "raw_p50_us": pct(&flat, 0.50),
        "raw_p95_us": pct(&flat, 0.95),
        "raw_p99_us": pct(&flat, 0.99),
        "raw_max_us": *flat.last().unwrap() as f64 * poll_ns / 1e3,
        "n_stall_slots": stall_slots.len(),
        "filtered_p50_us": pct(&filtered, 0.50),
        "filtered_p95_us": pct(&filtered, 0.95),
        "filtered_p99_us": pct(&filtered, 0.99),
        "excess_p50_us": pct(&flat_ex, 0.50),
        "excess_p99_us": pct(&flat_ex, 0.99),
        "filtered_excess_p50_us": pct(&filt_ex, 0.50),
        "filtered_excess_p95_us": pct(&filt_ex, 0.95),
        "filtered_excess_p99_us": pct(&filt_ex, 0.99),
    });
    (j, stall_slots)
}

/// Launch a long-running streaming kernel on a second in-process queue to
/// generate device traffic under the signal window. Returns (queue, cmd).
pub(crate) fn tp_background_traffic(
    ctx: &MetalContext,
    cal: &TpCal,
    seed: &TpBuffer,
    traffic: &TpBuffer,
    traffic_mask: u32,
    target_ms: f64,
    resident_est: u32,
) -> Result<(MetalQueue, MetalCommand)> {
    let q2 = ctx.device.newCommandQueue().context("bg queue")?;
    let g: usize = 10240;
    let waves = (g as f64 / resident_est.max(1) as f64).ceil().max(1.0);
    let dwell_ms = (target_ms * 1.5 / waves).max(0.5);
    let spin = ((dwell_ms * 1e6) / cal.fma_ns).max(1.0) as u32;
    let census = ctx.buffer_uninit(8)?;
    tp_zero(&census, 8);
    let entry = ctx.buffer_uninit(g * 4)?;
    let out = ctx.buffer_uninit(g * 4)?;
    let p = TpParams {
        spin_iters: spin,
        grid_tgs: g as u32,
        traffic_mask,
        ..Default::default()
    };
    let cmd = q2.commandBuffer().context("bg cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    tp_dispatch(
        ctx,
        &enc,
        "tp_residency_lo_tr",
        &p,
        &[seed, &census, &entry, &out, traffic],
        g,
        32,
    )?;
    enc.end();
    cmd.commit();
    Ok((q2, cmd))
}

pub(crate) fn tp_arm_s(
    ctx: &MetalContext,
    cal: &TpCal,
    seed: &TpBuffer,
    traffic: &TpBuffer,
    traffic_mask: u32,
    resident_cap: u32,
    quick: bool,
) -> Result<Vec<serde_json::Value>> {
    let epochs: u32 = if quick { 1_000 } else { 10_000 };
    let mut consumer_set: Vec<usize> = vec![40, 160];
    let cap = (resident_cap.saturating_sub(1) as usize).min(1024);
    if cap > 160 {
        consumer_set.push(cap);
    }
    let cadences: &[(&str, f64)] = &[("tight", 0.0), ("25us", 25_000.0), ("100us", 100_000.0)];
    let mut rows = Vec::new();
    println!("\n== arm S: bounded one-way signaling (epochs {epochs}) ==");
    println!(
        "consumers\tcadence\ttraffic\tdelivered\ttimeouts\tcorrupt\texcess_p50/p95/p99_us\tstalls"
    );

    let ring = ctx.buffer_uninit(TP_RING_SAMPLES * 4)?; // TP_RING == 64 slots
    let sink = ctx.buffer_uninit(4)?;
    for &consumers in &consumer_set {
        let stats = ctx.buffer_uninit(consumers * 4 * 4)?;
        let lat = ctx.buffer_uninit(consumers * TP_RING_SAMPLES * 4)?;
        for &(cad_name, cad_ns) in cadences {
            for tr in [false, true] {
                let cadence_iters = (cad_ns / cal.fma_ns) as u32;
                let window_ns = epochs as f64 * cad_ns.max(1_000.0);
                let budget = (((window_ns * 2.0 + 100e6) / cal.poll_ns) as u64)
                    .min(u32::MAX as u64 - 1) as u32;
                tp_zero(&ring, TP_RING_SAMPLES * 4);
                tp_zero(&stats, consumers * 4 * 4);
                tp_fill_u32(&lat, consumers * TP_RING_SAMPLES, TP_LAT_UNUSED);
                let p = TpParams {
                    epochs,
                    cadence_iters,
                    spin_budget: budget,
                    sample_every: (epochs / TP_RING_SAMPLES as u32).max(1),
                    grid_tgs: (consumers + 1) as u32,
                    ..Default::default()
                };
                let bg = if tr {
                    Some(tp_background_traffic(
                        ctx,
                        cal,
                        seed,
                        traffic,
                        traffic_mask,
                        window_ns / 1e6 + 100.0,
                        resident_cap,
                    )?)
                } else {
                    None
                };
                let ms = tp_run_timed(ctx, |enc| {
                    tp_dispatch(
                        ctx,
                        enc,
                        "tp_signal",
                        &p,
                        &[seed, &ring, &stats, &lat, &sink],
                        consumers + 1,
                        32,
                    )
                })?;
                let mut bg_ms = 0.0;
                if let Some((_q2, cmd)) = bg {
                    qwen_llm::metal::wait_completed(&cmd)?;
                    bg_ms = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;
                }
                let st = tp_read_u32(&stats, consumers * 4);
                let delivered = (0..consumers).filter(|&c| st[c * 4] == epochs).count();
                let timeouts: u32 = (0..consumers).map(|c| st[c * 4 + 2]).sum();
                let corrupt: u32 = (0..consumers).map(|c| st[c * 4 + 1]).sum();
                let lat_all: Vec<Vec<u32>> = (0..consumers)
                    .map(|c| {
                        tp_read_u32(&lat, consumers * TP_RING_SAMPLES)
                            [c * TP_RING_SAMPLES..(c + 1) * TP_RING_SAMPLES]
                            .to_vec()
                    })
                    .collect();
                let (lat_j, stalls) = tp_lat_stats(&lat_all, cal.poll_ns, cad_ns);
                println!(
                    "{consumers}\t{cad_name}\t{tr}\t{delivered}/{consumers}\t{timeouts}\t{corrupt}\t{:.1}/{:.1}/{:.1}\t{}",
                    lat_j["filtered_excess_p50_us"].as_f64().unwrap_or(0.0),
                    lat_j["filtered_excess_p95_us"].as_f64().unwrap_or(0.0),
                    lat_j["filtered_excess_p99_us"].as_f64().unwrap_or(0.0),
                    stalls.len(),
                );
                rows.push(serde_json::json!({
                    "arm": "s", "kind": "signal", "consumers": consumers,
                    "cadence": cad_name, "traffic": tr, "epochs": epochs,
                    "spin_budget": budget, "gpu_ms": ms, "bg_gpu_ms": bg_ms,
                    "delivered": delivered, "timeouts": timeouts,
                    "corrupt": corrupt, "latency": lat_j,
                    "latency_unit_note":
                        "poll counts x no-traffic poll_ns; traffic rows are \
                         indicative only (kill gate reads no-traffic rows)",
                }));
            }
        }
    }

    // Cross-object reordering probe (>= 25 us cadence only, per design).
    let re_epochs: u32 = if quick { 5_000 } else { 40_000 };
    let consumers = 160usize;
    let stats = ctx.buffer_uninit(consumers * 4 * 4)?;
    let ab = ctx.buffer_uninit(64 * 4)?;
    for tr in [false, true] {
        let cad_ns = 25_000.0;
        let budget = (((re_epochs as f64 * cad_ns * 2.0 + 100e6) / cal.poll_ns) as u64)
            .min(u32::MAX as u64 - 1) as u32;
        tp_zero(&ab, 64 * 4);
        tp_zero(&stats, consumers * 4 * 4);
        let p = TpParams {
            epochs: re_epochs,
            cadence_iters: (cad_ns / cal.fma_ns) as u32,
            spin_budget: budget,
            sample_every: 1,
            grid_tgs: (consumers + 1) as u32,
            ..Default::default()
        };
        let bg = if tr {
            Some(tp_background_traffic(
                ctx,
                cal,
                seed,
                traffic,
                traffic_mask,
                re_epochs as f64 * cad_ns / 1e6 + 100.0,
                resident_cap,
            )?)
        } else {
            None
        };
        let _ms = tp_run_timed(ctx, |enc| {
            tp_dispatch(
                ctx,
                enc,
                "tp_reorder",
                &p,
                &[seed, &ab, &stats, &sink],
                consumers + 1,
                32,
            )
        })?;
        if let Some((_q2, cmd)) = bg {
            qwen_llm::metal::wait_completed(&cmd)?;
        }
        let st = tp_read_u32(&stats, consumers * 4);
        let fresh: u64 = (0..consumers).map(|c| st[c * 4] as u64).sum();
        let stale: u64 = (0..consumers).map(|c| st[c * 4 + 1] as u64).sum();
        let corrupt: u64 = (0..consumers).map(|c| st[c * 4 + 2] as u64).sum();
        let timeouts: u32 = (0..consumers).map(|c| st[c * 4 + 3]).sum();
        println!(
            "reorder\ttraffic={tr}\tfresh={fresh}\tstale={stale}\trate={:.2e}\tcorrupt={corrupt}\ttimeouts={timeouts}",
            stale as f64 / fresh.max(1) as f64
        );
        rows.push(serde_json::json!({
            "arm": "s", "kind": "reorder", "traffic": tr,
            "epochs": re_epochs, "consumers": consumers,
            "fresh": fresh, "stale": stale,
            "stale_rate": stale as f64 / fresh.max(1) as f64,
            "corrupt": corrupt, "timeouts": timeouts,
        }));
    }
    Ok(rows)
}

pub(crate) fn tp_median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

pub(crate) fn tp_arm_d(
    ctx: &MetalContext,
    cal: &TpCal,
    m_tgs_by_w: &BTreeMap<usize, usize>,
    runs: usize,
    quick: bool,
) -> Result<Vec<serde_json::Value>> {
    let ks: &[u32] = if quick { &[8, 110] } else { &[8, 32, 110] };
    let works_us: &[f64] = if quick {
        &[0.0, 15.0]
    } else {
        &[0.0, 5.0, 15.0, 30.0]
    };
    let widths: &[usize] = &[32, 64];
    let mut rows = Vec::new();
    println!("\n== arm D: boundary drain vs persistence ==");
    println!(
        "W\tm\tK\twork_us\tladder_ms\tlocal_mem_ms\tlocal_reg_ms\tglobal_ms\tglobal_wide_ms\tboundary_us\tbarrier_us\taborts"
    );

    for &w in widths {
        let &cap = m_tgs_by_w
            .get(&w)
            .with_context(|| format!("no arm-R low-water for W={w}; pass --grid-tgs"))?;
        // m-axis: production glue dispatches are NARROW (few TGs on a mostly
        // idle machine); the wide row keeps the design's low-water shape.
        let mut ms_list: Vec<usize> = if quick {
            vec![4, cap]
        } else {
            vec![4, 64, cap]
        };
        ms_list.sort_unstable();
        ms_list.dedup();
        let n_max = cap.max(64) * w;
        let input = ctx.buffer_uninit(n_max * 4)?;
        {
            let p = input.contents().as_ptr() as *mut f32;
            for i in 0..n_max {
                unsafe { p.add(i).write(0.5 + (i % 1024) as f32 * 1e-3) }
            }
        }
        let out = ctx.buffer_uninit(n_max * 4)?;
        let scratch = ctx.buffer_uninit(n_max * 4)?;
        let bar = ctx.buffer_uninit(8)?;
        let abort = ctx.buffer_uninit(4)?;

        // Mixed-shape ladders (timing-only): consecutive stages alternate TG
        // counts, modeling production's narrow-glue -> wide-projection
        // heterogeneity where a wide dispatch cannot start until the narrow
        // one's LAST TG drains. No persistent twin / checksum; per-boundary
        // cost is derived against the m=4 local_mem baseline (same
        // per-thread work; each shape fits one wave).
        for (mix_name, m_pat) in [
            ("mix4_64", vec![4usize, 64]),
            ("mix4_cap", vec![4usize, cap]),
        ] {
            for &k in ks {
                for &work in works_us {
                    let spin = ((work * 1e3 / cal.fma_ns).max(1.0)) as u32;
                    let mut t_mixed = Vec::new();
                    for rep in 0..=runs {
                        let ms = tp_run_timed(ctx, |enc| {
                            let mut src = &input;
                            let bufs = [&out, &scratch];
                            for s in 0..k as usize {
                                let m_s = m_pat[s % m_pat.len()];
                                let p = TpParams {
                                    spin_iters: spin,
                                    grid_tgs: m_s as u32,
                                    stages: k,
                                    n_elems: (m_s * w) as u32,
                                    ..Default::default()
                                };
                                let dst = bufs[s % 2];
                                let kn = if s % 2 == 0 {
                                    "tp_chain_stage"
                                } else {
                                    "tp_chain_stage_b"
                                };
                                tp_dispatch(ctx, enc, kn, &p, &[src, dst], m_s, w)?;
                                src = dst;
                            }
                            Ok(())
                        })?;
                        if rep > 0 {
                            t_mixed.push(ms);
                        }
                    }
                    let ml = tp_median(t_mixed);
                    println!("{w}\t{mix_name}\t{k}\t{work:.0}\t{ml:.3}\t-\t-\t-\t-\t-\t-\t-");
                    rows.push(serde_json::json!({
                        "arm": "d", "w": w, "mixed": mix_name, "cap_tgs": cap,
                        "k": k, "work_us_nominal": work, "spin_iters": spin,
                        "ladder_ms": ml,
                    }));
                }
            }
        }

        for &m in &ms_list {
            let n = m * w;
            for &k in ks {
                for &work in works_us {
                    let spin = ((work * 1e3 / cal.fma_ns).max(1.0)) as u32;
                    let p = TpParams {
                        spin_iters: spin,
                        grid_tgs: m as u32,
                        stages: k,
                        spin_budget: ((10e6 / cal.poll_ns) as u64).min(u32::MAX as u64) as u32,
                        n_elems: n as u32,
                        ..Default::default()
                    };

                    // ladder alternates two identical PSOs to match
                    // production's per-dispatch state changes
                    let ladder_encode = |enc: &KernelEncoder| -> Result<()> {
                        let mut src = &input;
                        let bufs = [&out, &scratch];
                        for s in 0..k as usize {
                            let dst = bufs[s % 2];
                            let kn = if s % 2 == 0 {
                                "tp_chain_stage"
                            } else {
                                "tp_chain_stage_b"
                            };
                            tp_dispatch(ctx, enc, kn, &p, &[src, dst], m, w)?;
                            src = dst;
                        }
                        Ok(())
                    };
                    let final_ladder = if k as usize % 2 == 1 { &out } else { &scratch };

                    let mut checksum: Option<Vec<u32>> = None;
                    let mut mismatch = false;

                    let mut t_ladder = Vec::new();
                    for rep in 0..=runs {
                        let ms = tp_run_timed(ctx, ladder_encode)?;
                        if rep == 0 {
                            checksum = Some(tp_read_f32_bits(final_ladder, n));
                            continue;
                        }
                        t_ladder.push(ms);
                    }
                    let checksum = checksum.unwrap();

                    let mut run_variant = |kernel: &str,
                                           bufs: &[&TpBuffer],
                                           launch_tgs: usize,
                                           grid_tgs_param: usize,
                                           needs_bar: bool|
                     -> Result<(Vec<f64>, u32)> {
                        let pv = TpParams {
                            grid_tgs: grid_tgs_param as u32,
                            ..p
                        };
                        let mut ts = Vec::new();
                        let mut aborts = 0u32;
                        for rep in 0..=runs {
                            if needs_bar {
                                tp_zero(&bar, 8);
                                tp_zero(&abort, 4);
                            }
                            let ms = tp_run_timed(ctx, |enc| {
                                tp_dispatch(ctx, enc, kernel, &pv, bufs, launch_tgs, w)
                            })?;
                            if needs_bar && tp_read_u32(&abort, 1)[0] != 0 {
                                aborts += 1;
                                continue; // timing not admitted
                            }
                            if rep == 0 {
                                if tp_read_f32_bits(&out, n) != checksum {
                                    mismatch = true;
                                }
                                continue;
                            }
                            ts.push(ms);
                        }
                        Ok((ts, aborts))
                    };

                    let (t_lmem, _) = run_variant(
                        "tp_chain_persistent_local_mem",
                        &[&input, &out, &scratch],
                        m,
                        m,
                        false,
                    )?;
                    let (t_lreg, _) = run_variant(
                        "tp_chain_persistent_local_reg",
                        &[&input, &out],
                        m,
                        m,
                        false,
                    )?;
                    let (t_glob, aborts) = run_variant(
                        "tp_chain_persistent_global",
                        &[&input, &out, &bar, &abort, &scratch],
                        m,
                        m,
                        true,
                    )?;
                    // persistent HOST shape: full low-water grid carries
                    // narrow stages; idle TGs still pay every barrier
                    let (t_gwide, aborts_w) = if m < cap {
                        run_variant(
                            "tp_chain_persistent_global",
                            &[&input, &out, &bar, &abort, &scratch],
                            cap,
                            cap,
                            true,
                        )?
                    } else {
                        (Vec::new(), 0)
                    };

                    let (ml, mm, mr, mg) = (
                        tp_median(t_ladder),
                        tp_median(t_lmem),
                        tp_median(t_lreg),
                        tp_median(t_glob),
                    );
                    let mgw = if t_gwide.is_empty() {
                        f64::NAN
                    } else {
                        tp_median(t_gwide)
                    };
                    let boundaries = (k - 1).max(1) as f64;
                    let boundary_us = (ml - mm) * 1e3 / boundaries;
                    let barrier_us = (mg - mm) * 1e3 / boundaries;
                    println!(
                        "{w}\t{m}\t{k}\t{work:.0}\t{ml:.3}\t{mm:.3}\t{mr:.3}\t{mg:.3}\t{mgw:.3}\t{boundary_us:.1}\t{barrier_us:.1}\t{}{}",
                        aborts + aborts_w,
                        if mismatch { "\tCHECKSUM-MISMATCH" } else { "" }
                    );
                    rows.push(serde_json::json!({
                        "arm": "d", "w": w, "m_tgs": m, "cap_tgs": cap, "k": k,
                        "work_us_nominal": work, "spin_iters": spin,
                        "ladder_ms": ml, "local_mem_ms": mm, "local_reg_ms": mr,
                        "global_ms": mg,
                        "global_wide_ms": if mgw.is_nan() { serde_json::Value::Null } else { serde_json::json!(mgw) },
                        "aborts": aborts, "aborts_wide": aborts_w,
                        "checksum_ok": !mismatch,
                        "boundary_us_per": boundary_us,
                        "barrier_us_per": barrier_us,
                        "barrier_wide_us_per": if mgw.is_nan() { serde_json::Value::Null } else { serde_json::json!((mgw - mm) * 1e3 / boundaries) },
                        "recovery_local_mem": 1.0 - mm / ml.max(1e-9),
                        "recovery_local_reg": 1.0 - mr / ml.max(1e-9),
                        "recovery_global": 1.0 - mg / ml.max(1e-9),
                        "recovery_global_wide": if mgw.is_nan() { serde_json::Value::Null } else { serde_json::json!(1.0 - mgw / ml.max(1e-9)) },
                    }));
                }
            }
        }
    }
    Ok(rows)
}

pub(crate) fn run_topology_probe(args: TopologyProbeArgs) -> Result<()> {
    let ctx = MetalContext::new()?;
    std::fs::create_dir_all(&args.out_dir)?;
    let arms: HashSet<String> = if args.arm == "all" {
        ["r", "s", "d"].iter().map(|s| s.to_string()).collect()
    } else {
        args.arm.split(',').map(|s| s.trim().to_string()).collect()
    };

    // shared seed + traffic buffers
    let seed_host: Vec<f32> = (0..1024).map(|i| 0.5 + i as f32 * 1e-3).collect();
    let seed = ctx.buffer_from(&seed_host)?;
    let traffic_elems: usize = if args.quick { 1 << 26 } else { 1 << 29 }; // 256MB / 2GB
    let traffic = ctx.buffer_uninit(traffic_elems * 4)?;
    tp_zero(&traffic, traffic_elems * 4);
    let traffic_mask = (traffic_elems - 1) as u32;

    let cal = tp_calibrate(&ctx, &seed)?;
    println!(
        "calibration: fma {:.3} ns/iter, poll {:.3} ns/iter",
        cal.fma_ns, cal.poll_ns
    );

    let mut all_rows: Vec<serde_json::Value> = Vec::new();
    let mut r_caps: Option<TpArmROut> = None;

    if !args.dwell_extend.is_empty() {
        // focused plateau confirmation: lo variant, no traffic, W in {32,64}
        let g_tgs: usize = 10240;
        let census = ctx.buffer_uninit(8)?;
        let entry = ctx.buffer_uninit(g_tgs * 4)?;
        let out = ctx.buffer_uninit(g_tgs * 4)?;
        let dummy = ctx.buffer_uninit(4)?;
        tp_zero(&dummy, 4);
        println!("\n== arm R dwell extension (lo, no-traffic) ==");
        println!("W\tdwell_us\tmax_alive\tper_core\tsteady_p10\tgpu_ms");
        for &w in &[32usize, 64] {
            for &dw in &args.dwell_extend {
                let spin = ((dw * 1e3 / cal.fma_ns).max(1.0)) as u32;
                tp_zero(&census, 8);
                tp_zero(&entry, g_tgs * 4);
                let p = TpParams {
                    spin_iters: spin,
                    grid_tgs: g_tgs as u32,
                    ..Default::default()
                };
                let ms = tp_run_timed(&ctx, |enc| {
                    tp_dispatch(
                        &ctx,
                        enc,
                        "tp_residency_lo",
                        &p,
                        &[&seed, &census, &entry, &out, &dummy],
                        g_tgs,
                        w,
                    )
                })?;
                let max_alive = tp_read_u32(&census, 2)[1];
                let ea = tp_read_u32(&entry, g_tgs);
                let skip = (max_alive as usize).min(g_tgs.saturating_sub(1));
                let mut steady: Vec<u32> = ea[skip..].to_vec();
                steady.sort_unstable();
                let p10 = tp_percentile_u32(&steady, 0.10);
                println!(
                    "{w}\t{dw:.0}\t{max_alive}\t{:.1}\t{p10}\t{ms:.1}",
                    max_alive as f64 / TP_GPU_CORES
                );
                all_rows.push(serde_json::json!({
                    "arm": "r", "variant": "tp_residency_lo", "traffic": false,
                    "w": w, "dwell_us_nominal": dw, "spin_iters": spin,
                    "gpu_ms": ms, "max_alive": max_alive,
                    "per_core": max_alive as f64 / TP_GPU_CORES,
                    "steady_p10": p10, "dwell_extension": true,
                }));
            }
        }
    }

    if arms.contains("r") {
        let r = tp_arm_r(&ctx, &cal, &seed, &traffic, traffic_mask, args.quick)?;
        all_rows.extend(r.rows.iter().cloned());
        // pre-registered kill gate: lo / no-traffic / W=32, plateaued
        if let Some(&(max_alive, p10, plateaued)) =
            r.caps.get(&("tp_residency_lo".to_string(), false, 32))
        {
            let per_core = max_alive as f64 / TP_GPU_CORES;
            println!(
                "\narm R gate: W32/lo/no-traffic max_alive={max_alive} ({per_core:.1}/core), \
                 steady_p10={p10}, plateaued={plateaued} -> {}",
                if !plateaued {
                    "NOT PLATEAUED: extend dwell before reading the gate"
                } else if per_core < 16.0 {
                    "KILL (per-core residency < 16)"
                } else {
                    "PASS"
                }
            );
        }
        r_caps = Some(r);
    }

    let resident_cap = r_caps
        .as_ref()
        .and_then(|r| r.caps.get(&("tp_residency_lo".to_string(), false, 32)))
        .map(|&(_, p10, _)| p10)
        .or(args.grid_tgs.map(|g| g as u32))
        .unwrap_or(512);

    if arms.contains("s") {
        let rows = tp_arm_s(
            &ctx,
            &cal,
            &seed,
            &traffic,
            traffic_mask,
            resident_cap,
            args.quick,
        )?;
        all_rows.extend(rows);
    }

    if arms.contains("d") {
        let mut m_tgs_by_w = BTreeMap::new();
        for &w in &[32usize, 64] {
            let m = args.grid_tgs.or_else(|| {
                r_caps.as_ref().and_then(|r| {
                    r.caps
                        .get(&("tp_residency_lo".to_string(), false, w))
                        .map(|&(_, p10, _)| p10 as usize)
                })
            });
            if let Some(m) = m {
                m_tgs_by_w.insert(w, m.max(40));
            }
        }
        let rows = tp_arm_d(&ctx, &cal, &m_tgs_by_w, args.runs, args.quick)?;
        all_rows.extend(rows);
    }

    let summary = serde_json::json!({
        "design": "docs/bench/2026-07-05-b0-topology-probe/README.md",
        "quick": args.quick,
        "calibration": {"fma_ns": cal.fma_ns, "poll_ns": cal.poll_ns},
        "device": ctx.device.name().to_string(),
        "rows": all_rows,
    });
    let mut fname = String::from("topology-probe");
    if args.arm != "all" {
        fname.push('-');
        fname.push_str(&args.arm.replace(',', "_"));
    }
    if !args.dwell_extend.is_empty() {
        fname.push_str("-dwellext");
    }
    if args.quick {
        fname.push_str("-quick");
    }
    fname.push_str(".json");
    let path = args.out_dir.join(fname);
    std::fs::write(&path, serde_json::to_string_pretty(&summary)?)?;
    println!("\nwrote {}", path.display());
    Ok(())
}

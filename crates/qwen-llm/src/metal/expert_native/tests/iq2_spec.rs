//! Shape-specialized IQ2 native decode kernels (K = 5120 / 2304): bitwise
//! equality with the generic kernels at the DeepSeek V4.1 routed shapes and
//! at ragged channel counts, the fallback rules, and an opt-in isolated
//! timing probe (`QWEN_EXPERT_IQ2_SPEC_PROBE=1`, under the Metal lease).

use super::*;

const SPEC_TYPES: [GgmlType; 3] = [GgmlType::IQ2_XXS, GgmlType::IQ2_XS, GgmlType::IQ2_S];
const HIDDEN: usize = 5120;
const INTER: usize = 2304;

/// Expected kernel of one projection (`fused`: the fused gate/up) when the
/// shape-specialized kernels are enabled.
fn expected_kernel(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    aligned: bool,
    fused: bool,
) -> String {
    let rows = if dtype == GgmlType::IQ2_XXS {
        2
    } else {
        IQ2_SPEC_ROWS
    };
    if aligned && n_out.is_multiple_of(rows) {
        let gate_up = if fused { "gate_up_" } else { "" };
        let suffix = match dtype {
            GgmlType::IQ2_XXS => {
                return format!("kernel_expert_native_{gate_up}iq2_xxs_f32_split_k{n_in}");
            }
            GgmlType::IQ2_XS => "iq2_xs",
            _ => "iq2_s",
        };
        format!("kernel_expert_native_{gate_up}{suffix}_f32_k{n_in}_r{IQ2_SPEC_ROWS}")
    } else if fused {
        fused_gate_up_name(dtype).unwrap().to_string()
    } else {
        native_pipeline_name(dtype).unwrap()
    }
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    tensor_f32_at_offset(t)
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Gate/up (fused and two-pass) and down at K = 5120 / 2304: the specialized
/// kernels equal the generic ones bit for bit, single-token and token-batched,
/// ready and pending routes, invalid IDs. Ragged channel counts that are a
/// multiple of the channels per simdgroup use them (partial last
/// threadgroup: M - 2 for IQ2_XXS, M - 4 for all); other counts (M - 1, and
/// M - 2 for IQ2_XS / IQ2_S) and a 4-byte aligned activation fall back to the
/// generic kernels.
#[test]
fn native_iq2_spec_kernels_equal_generic_bitwise() {
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for dtype in SPEC_TYPES {
        for (k, widths) in [
            (HIDDEN, [INTER, INTER - 2, INTER - 4, INTER - 1]),
            (INTER, [HIDDEN, HIDDEN - 2, HIDDEN - 4, HIDDEN - 1]),
        ] {
            let down = k == INTER;
            for m in widths {
                let gate = bank(&ctx, &bank_bytes(dtype, k, m, 3), dtype, k, m);
                let up = bank(&ctx, &bank_bytes(dtype, k, m, 4), dtype, k, m);
                for (rows, prefix) in [(None, 0usize), (Some(2usize), 0), (None, 20)] {
                    let n = rows.unwrap_or(1);
                    let slots = IDS.len() * n;
                    let x_len = if down { k * slots } else { k * n };
                    let x: Vec<f32> = values(x_len, 7).iter().map(|v| v * 0.25).collect();
                    let shape = match (down, rows) {
                        (true, _) => vec![k as u64, slots as u64],
                        (false, Some(_)) => vec![k as u64, n as u64],
                        (false, None) => vec![k as u64],
                    };
                    let input = offset_tensor(
                        &ctx,
                        prefix,
                        bytemuck::cast_slice(&x),
                        0,
                        shape,
                        GgmlType::F32,
                    );
                    let aligned = input.offset.is_multiple_of(16);
                    let id_values: Vec<i32> = (0..n).flat_map(|_| IDS).collect();
                    let ids = if rows.is_some() {
                        offset_tensor(
                            &ctx,
                            12,
                            bytemuck::cast_slice(&id_values),
                            0,
                            vec![IDS.len() as u64, n as u64],
                            GgmlType::I32,
                        )
                    } else {
                        i32_view(&ctx, &IDS)
                    };
                    let statuses = [ROUTE_STATUS_READY, ROUTE_STATUS_PENDING][..n].to_vec();
                    let status = i32_view(&ctx, &statuses);
                    let passes: &[Option<bool>] = if down {
                        &[None]
                    } else {
                        &[Some(true), Some(false)]
                    };
                    for &fuse in passes {
                        let mut outputs = Vec::new();
                        for spec in [false, true] {
                            let out = f32_view(&ctx, &vec![f32::NAN; m * slots], &[m, slots], 0);
                            dispatch_census_begin();
                            one_shot(&ctx, |enc| match fuse {
                                None => encode_down_with(
                                    &ctx,
                                    enc,
                                    &gate,
                                    &input,
                                    &ids,
                                    &status,
                                    &out,
                                    k,
                                    m,
                                    EXPERTS,
                                    IDS.len(),
                                    rows,
                                    NativeOperands::F32,
                                    spec,
                                ),
                                Some(fuse) => encode_gate_up_with(
                                    &ctx,
                                    enc,
                                    &gate,
                                    &up,
                                    &input,
                                    &ids,
                                    &status,
                                    &out,
                                    k,
                                    m,
                                    EXPERTS,
                                    IDS.len(),
                                    10.0,
                                    rows,
                                    NativeOperands::F32,
                                    fuse,
                                    spec,
                                ),
                            })
                            .unwrap();
                            let census = dispatch_census_take();
                            let kernels: Vec<_> = census.iter().map(|r| r.kernel.clone()).collect();
                            let label = format!(
                                "{dtype:?} k={k} m={m} rows={rows:?} prefix={prefix} fuse={fuse:?}"
                            );
                            if spec {
                                let fused = fuse == Some(true);
                                let want = expected_kernel(dtype, k, m, aligned, fused);
                                let count = if fuse == Some(false) { 2 } else { 1 };
                                assert_eq!(kernels, vec![want; count], "{label}");
                            }
                            outputs.push(bits(&out));
                        }
                        let label = format!(
                            "{dtype:?} k={k} m={m} rows={rows:?} prefix={prefix} fuse={fuse:?}"
                        );
                        assert!(
                            outputs[0].iter().any(|&b| f32::from_bits(b) != 0.0),
                            "{label}"
                        );
                        assert!(
                            outputs[0].iter().all(|&b| f32::from_bits(b).is_finite()),
                            "{label}"
                        );
                        assert_eq!(outputs[0], outputs[1], "{label}");
                    }
                }
            }
        }
    }
}

/// Other K, storage types and half operands never select a specialized
/// kernel; the A/B knob is on by default.
#[test]
fn native_iq2_spec_selection_rules() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let input = |prefix| {
        offset_tensor(
            &ctx,
            prefix,
            bytemuck::cast_slice(&values(HIDDEN, 1)),
            0,
            vec![HIDDEN as u64],
            GgmlType::F32,
        )
    };
    let x = input(0);
    let unaligned = input(20);
    for dtype in SPEC_TYPES {
        assert!(native_spec_pipeline(dtype, HIDDEN, INTER, &unaligned, true).is_none());
        for fused in [false, true] {
            assert!(native_spec_pipeline(dtype, HIDDEN, INTER, &x, fused).is_some());
            assert!(native_spec_pipeline(dtype, 4096, INTER, &x, fused).is_none());
            assert!(native_spec_pipeline(dtype, HIDDEN, INTER - 1, &x, fused).is_none());
        }
    }
    for dtype in [
        GgmlType::IQ3_XXS,
        GgmlType::Q5_K,
        GgmlType::MXFP4,
        GgmlType::IQ1_S,
    ] {
        assert!(native_spec_pipeline(dtype, HIDDEN, INTER, &x, true).is_none());
        assert!(native_spec_pipeline(dtype, INTER, HIDDEN, &x, false).is_none());
    }
}

// ---- Opt-in isolated timing probe.

/// Set (any value) to run `native_iq2_spec_decode_probe`, a timing probe of
/// the shape-specialized IQ2 native decode kernels against the generic ones
/// at the V4.1 routed shapes; take the Metal lease.
const IQ2_SPEC_PROBE_ENV: &str = "QWEN_EXPERT_IQ2_SPEC_PROBE";
/// Experts per bank of `native_iq2_spec_decode_probe` (default 128, enough
/// that rotating routes stream rows from DRAM).
const IQ2_SPEC_PROBE_EXPERTS_ENV: &str = "QWEN_EXPERT_IQ2_SPEC_EXPERTS";
/// Timed rounds of `native_iq2_spec_decode_probe` (default 40).
const IQ2_SPEC_PROBE_ITERS_ENV: &str = "QWEN_EXPERT_IQ2_SPEC_ITERS";

const PROBE_TOP_K: usize = 6;
const PROBE_REPS: usize = 8;

fn probe_timed<F>(ctx: &MetalContext, encode: F) -> f64
where
    F: FnOnce(&KernelEncoder) -> Result<(), MetalError>,
{
    let cmd = ctx.queue.commandBuffer().expect("command buffer");
    let enc = KernelEncoder::begin(&cmd);
    encode(&enc).expect("encode");
    enc.end();
    cmd.commit();
    wait_completed(&cmd).expect("completed");
    (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6
}

fn probe_median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// A named native kernel at the probe shape (`fused` gate/up or one projection).
#[allow(clippy::too_many_arguments)]
fn encode_named(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    name: &str,
    channels: usize,
    fused: bool,
    banks: (&MetalTensor, &MetalTensor),
    input: &MetalTensor,
    ids: &MetalTensor,
    status: &MetalTensor,
    output: &MetalTensor,
    (n_in, n_out, experts): (usize, usize, usize),
) -> Result<(), MetalError> {
    let pso = ctx.pipeline(name)?;
    let (block, bytes) = banks.0.dtype.storage_layout().unwrap();
    let mut args = dimensions("iq2 spec probe", n_in, n_out, experts, PROBE_TOP_K)?;
    args.row_bytes = n_in as u64 / block * bytes;
    args.mode = if fused { 1 } else { 2 };
    args.clamp = 10.0;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, banks.0);
    enc.set_tensor(2, input);
    enc.set_tensor(3, ids);
    enc.set_tensor(4, status);
    enc.set_tensor(5, output);
    if fused {
        enc.set_tensor(6, banks.1);
    }
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(channels),
            height: PROBE_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Generic vs shape-specialized kernels at the V4.1 routed decode shape
/// (fused gate/up 5120 -> 2x2304, down 2304 -> 5120, top-6), 128-expert
/// banks with expert IDs advancing every dispatch (DRAM-streamed rows),
/// warmed and heated GPU, variants interleaved; median per-dispatch time in
/// 8-dispatch command buffers. Extra IQ2_XS / IQ2_S rows-per-simdgroup
/// variants are timed by name when instantiated. Nothing is asserted.
#[test]
fn native_iq2_spec_decode_probe() {
    if std::env::var_os(IQ2_SPEC_PROBE_ENV).is_none() {
        return;
    }
    let _lease = production_lease();
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let experts: usize = std::env::var(IQ2_SPEC_PROBE_EXPERTS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let iters: usize = std::env::var(IQ2_SPEC_PROBE_ITERS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let x_gate = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&values(HIDDEN, 3)),
        vec![HIDDEN as u64],
        GgmlType::F32,
    )
    .unwrap();
    let down_x: Vec<f32> = values(INTER * PROBE_TOP_K, 5)
        .iter()
        .map(|v| v * 0.5)
        .collect();
    let x_down = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&down_x),
        vec![INTER as u64, PROBE_TOP_K as u64],
        GgmlType::F32,
    )
    .unwrap();
    let status = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[ROUTE_STATUS_READY]),
        vec![1],
        GgmlType::I32,
    )
    .unwrap();
    let ids: Vec<MetalTensor> = (0..PROBE_REPS)
        .map(|_| {
            MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&[0i32; PROBE_TOP_K]),
                vec![PROBE_TOP_K as u64],
                GgmlType::I32,
            )
            .unwrap()
        })
        .collect();
    let inner = MetalTensor::zeros_f32(&ctx, vec![INTER as u64, PROBE_TOP_K as u64]).unwrap();
    let out = MetalTensor::zeros_f32(&ctx, vec![HIDDEN as u64, PROBE_TOP_K as u64]).unwrap();
    let mut counter = 0usize;
    let mut next = |t: &MetalTensor| {
        let p = t.buffer.contents().as_ptr() as *mut i32;
        for j in 0..PROBE_TOP_K {
            unsafe {
                p.add(j)
                    .write(((counter * PROBE_TOP_K + j) % experts) as i32)
            }
        }
        counter += 1;
    };
    for dtype in SPEC_TYPES {
        let tensor = |n_in: usize, n_out: usize, seed: usize| {
            MetalTensor::from_bytes(
                &ctx,
                &bank_bytes_for_experts(dtype, n_in, n_out, seed, experts),
                vec![n_in as u64, n_out as u64, experts as u64],
                dtype,
            )
            .unwrap()
        };
        let (gate, up, down) = (
            tensor(HIDDEN, INTER, 1),
            tensor(HIDDEN, INTER, 2),
            tensor(INTER, HIDDEN, 3),
        );
        // (label, fused, kernel, channels per threadgroup)
        let generic_rows = native_channels_per_threadgroup(dtype);
        let mut variants: Vec<(String, bool, String, usize)> = vec![
            (
                "generic".into(),
                true,
                fused_gate_up_name(dtype).unwrap().into(),
                generic_rows / 2,
            ),
            (
                "generic".into(),
                false,
                native_pipeline_name(dtype).unwrap(),
                generic_rows,
            ),
        ];
        for fused in [true, false] {
            let x = if fused { &x_gate } else { &x_down };
            let (n_in, n_out) = if fused {
                (HIDDEN, INTER)
            } else {
                (INTER, HIDDEN)
            };
            let (name, ch) = native_spec_pipeline(dtype, n_in, n_out, x, fused).unwrap();
            variants.push(("spec".into(), fused, name, ch));
            if dtype != GgmlType::IQ2_XXS {
                let suffix = if dtype == GgmlType::IQ2_XS {
                    "iq2_xs"
                } else {
                    "iq2_s"
                };
                for r in [1usize, 2, 4, 8]
                    .into_iter()
                    .filter(|&r| r != IQ2_SPEC_ROWS)
                {
                    let gate_up = if fused { "gate_up_" } else { "" };
                    let name = format!("kernel_expert_native_{gate_up}{suffix}_f32_k{n_in}_r{r}");
                    if ctx.pipeline(&name).is_ok() {
                        let ch = if fused { 2 * r } else { 4 * r };
                        variants.push((format!("spec_r{r}"), fused, name, ch));
                    }
                }
            }
        }
        let run = |v: &(String, bool, String, usize), enc: &KernelEncoder, ids: &MetalTensor| {
            let (_, fused, name, ch) = v;
            if *fused {
                encode_named(
                    &ctx,
                    enc,
                    name,
                    *ch,
                    true,
                    (&gate, &up),
                    &x_gate,
                    ids,
                    &status,
                    &inner,
                    (HIDDEN, INTER, experts),
                )
            } else {
                encode_named(
                    &ctx,
                    enc,
                    name,
                    *ch,
                    false,
                    (&down, &down),
                    &x_down,
                    ids,
                    &status,
                    &out,
                    (INTER, HIDDEN, experts),
                )
            }
        };
        // Warm every expert row, then heat.
        for _ in 0..experts.div_ceil(PROBE_TOP_K) + 1 {
            next(&ids[0]);
            probe_timed(&ctx, |enc| {
                run(&variants[0], enc, &ids[0])?;
                run(&variants[1], enc, &ids[0])
            });
        }
        let heat = std::time::Instant::now();
        while heat.elapsed() < std::time::Duration::from_millis(600) {
            for t in &ids {
                next(t);
            }
            probe_timed(&ctx, |enc| {
                for t in &ids {
                    run(&variants[0], enc, t)?;
                    run(&variants[1], enc, t)?;
                }
                Ok(())
            });
        }
        let n = variants.len();
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); n];
        for it in 0..iters + 3 {
            for step in 0..n {
                let v = (step + it * 3) % n;
                for t in &ids {
                    next(t);
                }
                let t = probe_timed(&ctx, |enc| {
                    for t in &ids {
                        run(&variants[v], enc, t)?;
                    }
                    Ok(())
                }) / PROBE_REPS as f64;
                if it >= 3 {
                    times[v].push(t);
                }
            }
        }
        for fused in [true, false] {
            let base = variants
                .iter()
                .position(|v| v.1 == fused && v.0 == "generic")
                .unwrap();
            let b = probe_median(&times[base]);
            for (v, variant) in variants.iter().enumerate().filter(|(_, v)| v.1 == fused) {
                let m = probe_median(&times[v]);
                eprintln!(
                    "iq2spec {dtype:?} {:<7} {:<8} {m:7.1} us  {:5.3}x  {}",
                    if fused { "gate_up" } else { "down" },
                    variant.0,
                    b / m,
                    variant.2
                );
            }
        }
    }
}

//! Maya Fast/Half router ABBA: retained forced N2048 diagnostic, or production
//! B at N2048 / optional N129 boundary (separate test). A forces generic;
//! production B only observes timestamps/substitutions, leaving selection None.
//! GLM53_GGUF and a new GLM53_ROUTER_OUT JSONL are required for each run.
use super::*;

const STEPS: usize = 3;

// u is unit roundoff, not machine epsilon. Count both a rounded multiply
// and a rounded add per term: gamma(2*K) is conservative even if Metal's
// math_mode(safe) acc += x*w contracts to FMA (only one rounding per term).
// This is an absolute forward-error bound scaled by sum |x*w|, so cancellation
// does not turn it into an empirical relative-error cutoff. The independent
// f64 products are exact for f32 inputs; allow for their f64 summation error,
// then conservatively add final F32 rounding. Upward rounding protects the
// bound's own f64 arithmetic. Actual inputs/products/partial sums must avoid
// underflow or overflow; reject an unsupported regime rather than widen it.
fn dot_accuracy(weight: &[f32], input: &[f32], gpu: f32) -> Value {
    assert!(!weight.is_empty() && weight.len() == input.len());
    let normal_or_zero = |x: f64| {
        x == 0.0
            || (x.is_finite()
                && x.abs() >= f64::from(f32::MIN_POSITIVE)
                && x.abs() <= f64::from(f32::MAX))
    };
    let (mut sequential, mut fused) = (0.0f32, 0.0f32);
    let (mut reference, mut sum_abs) = (0.0f64, 0.0f64);
    for (&w, &x) in weight.iter().zip(input) {
        assert!(normal_or_zero(f64::from(w)) && normal_or_zero(f64::from(x)));
        let product = f64::from(w) * f64::from(x);
        let rounded_product = w * x;
        assert!(normal_or_zero(product));
        assert!(normal_or_zero(
            f64::from(sequential) + f64::from(rounded_product)
        ));
        assert!(normal_or_zero(f64::from(fused) + product));
        sequential += rounded_product;
        fused = w.mul_add(x, fused);
        reference += product;
        sum_abs += product.abs();
    }
    let gamma = |operations: usize, u: f64| {
        let nu = operations as f64 * u;
        assert!(nu < 1.0);
        (nu / (1.0 - nu)).next_up()
    };
    let unit_roundoff_f32 = f64::from(f32::EPSILON) / 2.0;
    let gamma32 = gamma(2 * weight.len(), unit_roundoff_f32);
    let gamma64 = gamma(weight.len(), f64::EPSILON / 2.0);
    let sum_abs_upper = (sum_abs / (1.0 - gamma64).next_down()).next_up();
    let reference_error = (gamma64 * sum_abs_upper).next_up();
    let arithmetic_error = (gamma32 * sum_abs_upper).next_up();
    let reference_magnitude = (reference.abs() + reference_error).next_up();
    let final_rounding =
        (unit_roundoff_f32 * reference_magnitude.max(f64::from(f32::MIN_POSITIVE))).next_up();
    let bound = ((arithmetic_error + reference_error).next_up() + final_rounding).next_up();
    assert!((sum_abs_upper * (1.0 + gamma32).next_up()).next_up() <= f64::from(f32::MAX));
    assert!(
        reference.is_finite() && bound.is_finite() && sequential.is_finite() && fused.is_finite()
    );
    let error = (f64::from(gpu) - reference).abs();
    json!({"cpu":sequential,"cpu_fma":fused,"gpu":gpu,"abs_error":(f64::from(sequential)-f64::from(gpu)).abs(),
        "f64_dot":reference,"sum_abs_products_f64":sum_abs,"sum_abs_products_upper":sum_abs_upper,
        "gamma_2k_f32":gamma32,"f64_reference_error_bound":reference_error,"final_f32_rounding_bound":final_rounding,
        "gpu_abs_error_vs_f64":error,"forward_error_bound":bound,"within_bound":gpu.is_finite() && error.next_up() <= bound})
}

#[test]
fn router_dot_forward_error_contract() {
    let exact = dot_accuracy(&[1.0, -2.0], &[3.0, 4.0], -5.0);
    assert_eq!(exact["within_bound"], true);
    assert_eq!(exact["f64_dot"], -5.0);
    assert_eq!(
        dot_accuracy(&[1.0, -2.0], &[3.0, 4.0], -4.0)["within_bound"],
        false
    );
    // Absolute error may dwarf the cancelled dot while obeying the derived bound.
    let cancelled = dot_accuracy(&[1e8, 1.0, -1e8], &[1.0; 3], 0.0);
    assert_eq!(cancelled["f64_dot"], 1.0);
    assert_eq!(cancelled["within_bound"], true);
    assert!(std::panic::catch_unwind(|| dot_accuracy(&[f32::from_bits(1)], &[1.0], 0.0)).is_err());
}

fn leaf<const ROWS: usize>(ctx: &MetalContext, s: &Glm5NextSession<'_>, out: &mut std::fs::File) {
    let p = s.packed.as_ref().unwrap();
    let block = s.weights.blocks.len() - 1;
    let FfnTensors::Moe(moe) = &s.weights.blocks[block].ffn else {
        panic!("last block must be MoE")
    };
    let (h, e) = (
        s.weights.config.hidden_size as usize,
        s.weights.config.expert_count as usize,
    );
    let normed = rows_view(&p.normed, ROWS);
    let projected = rows_view(&p.router, ROWS);
    let input = read_f32(&normed).unwrap();
    let weights = read_f32(&moe.router).unwrap();
    let mut baseline = None;
    for candidate in [false, true] {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        if candidate {
            crate::metal::encode_mat_mat_f32_router_e8p32_strict(
                ctx,
                &enc,
                &moe.router,
                &normed,
                &projected,
                h,
                e,
                ROWS,
            )
            .unwrap();
        } else {
            crate::metal::encode_mat_mat_f32(
                ctx,
                &enc,
                &moe.router,
                &normed,
                &projected,
                h,
                e,
                ROWS,
            )
            .unwrap();
        }
        enc.end();
        command.commit();
        wait_completed(&command).unwrap();
        let values = read_f32(&projected).unwrap();
        assert!(values.iter().all(|x| x.is_finite()));
        let mut samples = Vec::new();
        for token in [0, 31, 32, ROWS / 2 - 1, ROWS - 1] {
            for expert in [0, 7, 8, e - 9, e - 8, e - 1] {
                let mut sample = dot_accuracy(
                    &weights[expert * h..(expert + 1) * h],
                    &input[token * h..(token + 1) * h],
                    values[token * e + expert],
                );
                sample["token"] = json!(token);
                sample["expert"] = json!(expert);
                samples.push(sample);
            }
        }
        let accuracy_passed = samples.iter().all(|s| s["within_bound"] == true);
        let difference = baseline
            .as_ref()
            .map(|a: &Vec<f32>| vector_diff(a, &values));
        emit(
            out,
            json!({"event":"last_block_leaf","block":block,"candidate":candidate,"shape":{"K":h,"E":e,"N":ROWS},
            "input_sha256":sha256(bytemuck::cast_slice(&input)),"weights_sha256":sha256(bytemuck::cast_slice(&weights)),
            "projection_sha256":sha256(bytemuck::cast_slice(&values)),"full_projection_difference":difference,
            "cpu_sequential_f32_samples":samples,"cpu_oracle_contract":"gamma(2*K,u_f32)*sum_abs_products + f64 reference summation allowance + final F32 rounding; finite normal arithmetic",
            "cpu_oracle_passed":accuracy_passed,
            "timing_eligible":false,
            "cpu_payload_bytes":(input.capacity()+weights.capacity()+values.capacity()+baseline.as_ref().map_or(0,Vec::capacity))*4,
            "note":"actual final-block input, existing GPU scratch; no leaf-times-layer-count extrapolation"}),
        );
        assert!(
            accuracy_passed,
            "router exceeds derived dot-product forward-error bound; samples flushed"
        );
        if let Some(a) = &baseline {
            assert!(
                a.iter()
                    .zip(&values)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "strict leaf changed generic projection bits"
            );
        } else {
            baseline = Some(values);
        }
    }
}

#[test]
#[ignore = "release; internal production lease; GLM53_GGUF=Maya, NEW GLM53_ROUTER_OUT"]
fn router_2048_abba() {
    packet::<2048>(false);
}

#[test]
#[ignore = "release; internal production lease; GLM53_GGUF=Maya, NEW GLM53_ROUTER_OUT"]
fn router_2048_production_abba() {
    packet::<2048>(true);
}

#[test]
#[ignore = "optional N129 boundary; release, internal lease; GLM53_GGUF and NEW GLM53_ROUTER_OUT"]
fn router_129_production_abba() {
    packet::<129>(true);
}

fn packet<const ROWS: usize>(production_b: bool) {
    assert!(!cfg!(debug_assertions), "requires release");
    let _lease = perf_lease();
    let path = std::path::PathBuf::from(std::env::var_os("GLM53_GGUF").expect("GLM53_GGUF"));
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("GLM53_ROUTER_OUT").expect("GLM53_ROUTER_OUT"))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(&ctx, &gguf, artifact.model(), ROWS + STEPS, ROWS).unwrap();
    let text = format!("[gMASK]<sop>{}", long_qualification_text());
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .take(ROWS + STEPS)
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    assert_eq!(tokens.len(), ROWS + STEPS);
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let (h, e, k, v) = (
        weights.config.hidden_size as usize,
        weights.config.expert_count as usize,
        weights.config.expert_used_count as usize,
        weights.config.vocab_size as usize,
    );
    let routers: Vec<_> = weights.blocks.iter().enumerate().filter_map(|(block,b)| {
        let FfnTensors::Moe(m) = &b.ffn else { return None };
        assert_eq!(m.router.dtype,GgmlType::F32);
        Some(json!({"block":block,"name":format!("blk.{block}.ffn_gate_inp.weight"),"dtype":"F32","shape":m.router.shape}))
    }).collect();
    assert!(
        router_e8p32_selected(PackedLineage::Fast, h, e, GgmlType::F32, ROWS, || true),
        "packet requires production-eligible geometry"
    );
    assert!(router_e8p32_supported(&ctx), "strict pipeline unavailable");
    // Peak named CPU payloads: two route snapshots, reference+current logits
    // through the continuation, or warm reference plus leaf input/weight/outputs.
    let route_bytes = routers.len() * ROWS * k * 8;
    let cpu_bytes = ((2 * route_bytes + (STEPS + 2) * v * 4)
        .max(route_bytes + v * 4 + (h * ROWS + h * e + 2 * e * ROWS) * 4)
        + tokens.capacity() * 4) as u64;
    emit(
        &mut out,
        json!({"event":"header","schema":if production_b { "glm53.router_production_abba.v1" } else { "glm53.router_2048_abba.v1" },"artifact":path,
        "shards":stamps.iter().map(|s|json!({"path":s.path,"bytes":s.size,"inode":s.inode,"mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec})).collect::<Vec<_>>(),
        "rows":ROWS,"capacity":ROWS+STEPS,"precision":"Half","lineage":"Fast","routers":routers,"tokens":tokens,
        "cpu_payload_reserve_bytes":cpu_bytes,"production_change":false,"order":["warm_A","warm_B","A1","B1","B2","A2"],
        "A":"scoped false: generic F32 router","B":if production_b { "production selection None; timestamps/substitutions captured, no forced router" } else { "scoped true: strict F32 router" },
        "B_selection_override":if production_b { None } else { Some(true) },"other_stages":"unchanged production Half MMA",
        "production_minimum_rows":ROUTER_E8P32_MIN_ROWS,"warmup":"both warm wall/GPU samples retained and excluded from measured ABBA; warm wall/GPU divergence remains visible without causal attribution",
        "timing":"ordinary prefill wall and existing completed-command GPU timestamps only; allocation, census, leaf, route/logit readback comparisons and three common TF steps excluded; no stage observer",
        "product_msl_target":crate::PRODUCT_METALLIB_TARGET,"source_sha256":{"packet":sha256(include_bytes!("large_batch.rs")),
            "router_packet":sha256(include_bytes!("../router_prefill.rs")),"packed":sha256(include_bytes!("../../packed.rs")),
            "runtime":sha256(include_bytes!("../../../glm5_next_metal.rs")),"metallib":sha256(crate::KERNELS_METALLIB)}}),
    );
    let mut reference: Option<Snapshot> = None;
    let mut continuation = Vec::<Vec<f32>>::new();
    let mut timings = Vec::new();
    let mut warm_timings = Vec::new();
    for (label, candidate) in [
        ("warm_A", false),
        ("warm_B", true),
        ("A1", false),
        ("B1", true),
        ("B2", true),
        ("A2", false),
    ] {
        let warm = label.starts_with("warm");
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            ROWS + STEPS,
            ROWS,
            cpu_bytes,
        )
        .unwrap();
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        emit(
            &mut out,
            json!({"event":"session","label":label,"ledger":session.ledger().terms(),"observed_allocation_delta":session.observed_allocation_delta()}),
        );
        assert!(!crate::metal::dispatch_census_is_active());
        if warm {
            crate::metal::dispatch_census_begin();
        }
        let selection = if candidate && production_b {
            None
        } else {
            Some(candidate)
        };
        let execute = || {
            assert_eq!(requested_override(), selection);
            let start = Instant::now();
            let result = session.prefill_packed(&ctx, &tokens[..ROWS]);
            (result, start.elapsed().as_secs_f64() * 1e3)
        };
        let ((result, wall), timing) = if candidate && production_b {
            with_production_capture(execute)
        } else {
            with_override(candidate, execute)
        };
        assert_eq!(timing.enabled, selection);
        let census = if warm {
            crate::metal::dispatch_census_take()
        } else {
            Vec::new()
        };
        emit(
            &mut out,
            json!({"event":"prefill","label":label,"measured":!warm,"wall_ms":wall,"gpu_ms":timing.gpu_ms,
            "completed_commands":timing.completed_commands,"valid_gpu_commands":timing.valid_gpu_commands,
            "substitutions":timing.substitutions,"selection_override":timing.enabled,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        let logits = result.unwrap();
        assert_eq!(
            (timing.completed_commands, timing.valid_gpu_commands),
            (1, 1)
        );
        assert_eq!(
            timing.substitutions,
            if candidate { routers.len() } else { 0 }
        );
        assert!(wall.is_finite() && wall > 0.0);
        assert_eq!(session.position(), ROWS);
        assert_eq!(logits.len(), v);
        assert!(logits.iter().all(|x| x.is_finite()));
        if warm {
            let strict: Vec<_> = census
                .iter()
                .filter(|r| r.kernel == "kernel_mat_mat_f32_f32_router_e8p32_strict")
                .collect();
            assert_eq!(strict.len(), if candidate { routers.len() } else { 0 });
            assert!(strict.iter().all(|r| (
                r.grid_width,
                r.grid_height,
                r.grid_depth,
                r.tg_threads
            ) == ((e / 8) as u64, ROWS.div_ceil(32) as u64, 1, 32)));
            let mut counts = std::collections::BTreeMap::<&str, usize>::new();
            for r in &census {
                *counts.entry(&r.kernel).or_default() += 1;
            }
            emit(
                &mut out,
                json!({"event":"warm_census","label":label,"kernel_counts":counts,
                "census_heap_bytes":census.capacity()*size_of::<crate::metal::DispatchCensusRow>()+census.iter().map(|r|r.kernel.capacity()+r.tag.as_ref().map_or(0,String::capacity)).sum::<usize>(),
                "strict_shapes":census.iter().filter(|r|r.kernel=="kernel_mat_mat_f32_f32_router_e8p32_strict")
                    .map(|r|[r.grid_width,r.grid_height,r.grid_depth,r.tg_threads]).collect::<Vec<_>>(),"timing_eligible":false}),
            );
        }
        drop(census); // Release warm observer storage before leaf copies or route snapshots.
        let current = snapshot(&session, logits);
        assert!(
            current
                .routes
                .iter()
                .all(|r| r.weights.iter().all(|w| w.is_finite())
                    && r.ids.iter().all(|&id| id >= 0 && (id as usize) < e))
        );
        let baseline = label == "warm_A" || label == "A1";
        emit(
            &mut out,
            json!({"event":"output","label":label,"logits_sha256":sha256(bytemuck::cast_slice(&current.logits)),
            "routes":current.routes.iter().map(|r|json!({"block":r.block,"ids_sha256_i32":sha256(bytemuck::cast_slice(&r.ids)),"weights_sha256_f32":sha256(bytemuck::cast_slice(&r.weights)),"slots":r.ids.len(),"weights_nonfinite":r.weights.iter().filter(|v|!v.is_finite()).count()})).collect::<Vec<_>>() }),
        );
        if baseline {
            reference = Some(current);
            continuation.clear();
        } else {
            let a = reference.as_ref().unwrap();
            emit(
                &mut out,
                json!({"event":"comparison","label":label,"reference":"warm_A or A1","comparison":compare(a,&current,k)}),
            );
            assert!(
                a.logits
                    .iter()
                    .zip(&current.logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(a.routes.iter().zip(&current.routes).all(|(a, b)| {
                a.ids == b.ids
                    && a.weights
                        .iter()
                        .zip(&b.weights)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }));
            drop(current); // Do not retain a second full route snapshot during decode.
        }
        if label == "warm_A" {
            leaf::<ROWS>(&ctx, &session, &mut out);
        }
        for (step, &token) in tokens[ROWS..].iter().enumerate() {
            let logits = session.forward(&ctx, token).unwrap();
            assert_eq!(logits.len(), v);
            assert!(logits.iter().all(|x| x.is_finite()));
            let difference = (!baseline).then(|| logit_diff(&continuation[step], &logits));
            emit(
                &mut out,
                json!({"event":"continuation","label":label,"step":step+1,"token":token,
                "logits_sha256":sha256(bytemuck::cast_slice(&logits)),"comparison":difference,"timed":false}),
            );
            if baseline {
                continuation.push(logits);
            } else {
                assert!(
                    continuation[step]
                        .iter()
                        .zip(&logits)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
        if warm {
            warm_timings.push(
                json!({"label":label,"wall_ms":wall,"gpu_ms":timing.gpu_ms,"measured":false}),
            );
        } else {
            timings.push((wall, timing.gpu_ms));
        }
    }
    let summary = |v: Vec<f64>| {
        json!({"A1":v[0],"B1":v[1],"B2":v[2],"A2":v[3],
        "mean_A_over_B":(v[0]+v[3])/(v[1]+v[2]),"paired_A_over_B":[v[0]/v[1],v[3]/v[2]],"A2_over_A1":v[3]/v[0]})
    };
    assert_eq!(gguf.revalidate_retained_shard_stamps().unwrap(), stamps);
    emit(
        &mut out,
        json!({"event":"complete","wall_ms":summary(timings.iter().map(|x|x.0).collect()),
        "gpu_ms":summary(timings.iter().map(|x|x.1).collect()),"warm_measurements_excluded":warm_timings,
        "decision":"unscored; no performance bound"}),
    );
}

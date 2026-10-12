//! Same-binary KDA prepare/history experiment. No recurrence/beta/output capture.
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 GLM53_GGUF=/path/to/Maya.gguf \
//!   GLM53_KDA_PREP_OUT=/path/to/new.jsonl /path/to/release/test-binary \
//!   glm5_next_metal::packed::kda_prepare_ab::kda_prepare_ab_packet \
//!   --ignored --exact --nocapture --test-threads=1
//! GLM53_KDA_PREP_WHOLE=1 additionally runs fresh-session whole-model ABBA.
//! GLM53_KDA_PREP_WHOLE_ONLY=1 skips capture/leaf and forces the same whole runs.
//! Events are flushed; only successful packet completion calls sync_data.
//! Incomplete packets remain diagnostic evidence, not full performance results.

use super::*;
use crate::glm5_next_metal::tests::{kl_divergence, long_qualification_text, perf_lease};
use crate::metal::{
    dispatch_census_begin, dispatch_census_take, encode_kda_prefill_prepare,
    with_kda_prepare_register_history_override,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::Write,
    rc::Rc,
    time::Instant,
};

const PREFIX: usize = 32;
const ROWS: usize = 2048;
const STEPS: usize = 3;
const READ: usize = 256 * 1024;
const NAMES: [&str; 5] = ["q", "k", "v", "gate", "conv_history"];
const ARMS: [(&str, bool); 6] = [
    ("warm_A", false),
    ("warm_B", true),
    ("A1", false),
    ("B1", true),
    ("B2", true),
    ("A2", false),
];

struct Capture {
    data: [MetalTensor; 5],
    source: (usize, u64),
    rows: usize,
    copies: Cell<usize>,
}
thread_local! { static CAPTURE: RefCell<Option<Rc<Capture>>> = const { RefCell::new(None) }; }
fn identity(t: &MetalTensor) -> (usize, u64) {
    (
        std::ptr::from_ref(&*t.buffer).cast::<()>() as usize,
        t.offset,
    )
}
fn scoped_capture<T>(c: Rc<Capture>, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<Rc<Capture>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CAPTURE.with(|s| *s.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(CAPTURE.with(|s| s.replace(Some(c))));
    body()
}

// Matches the selected layer's actual convolution-state binding (buffer+offset),
// so no block index or capture state is added to the production session.
pub(super) fn before_prepare(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    rows: usize,
    conv: &MetalTensor,
    p: &PackedScratch,
) -> Result<()> {
    let capture = CAPTURE.with(|s| {
        s.borrow()
            .as_ref()
            .filter(|c| c.source == identity(conv))
            .cloned()
    });
    if let Some(c) = capture {
        assert_eq!(rows, c.rows, "one actual packed chunk requested");
        assert_eq!(
            c.copies.replace(c.copies.get() + 1),
            0,
            "capture exactly once"
        );
        for (src, dst) in [&p.q, &p.k, &p.v, &p.raw_gate, conv]
            .into_iter()
            .zip(&c.data)
        {
            let n = dst.n_elements() as usize;
            crate::metal::encode_copy_offset_f32(ctx, enc, &flat(src, n), 0, &flat(dst, n), n)?;
        }
    }
    Ok(())
}

fn record(out: &mut File, mut value: Value) {
    value["recorded_unix_ms"] = json!(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
    // A process exit leaves flushed records available; power-loss durability
    // is requested once at successful packet end, outside all measurements.
}
fn arm<T>(candidate: bool, body: impl FnOnce() -> T) -> T {
    crate::metal::iq3_s_down_retile::with_production(|| {
        crate::metal::with_grouped_down_wide_override(false, || {
            crate::metal::with_grouped_gate_up_m32_override(false, || {
                with_kda_prepare_register_history_override(candidate, body)
            })
        })
    })
    .0
}
fn command(
    ctx: &MetalContext,
    encode: impl FnOnce(&KernelEncoder) -> std::result::Result<(), MetalError>,
) -> (std::result::Result<(), MetalError>, f64, Option<f64>) {
    let start = Instant::now();
    let cb = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cb);
    let result = encode(&enc);
    enc.end();
    let result = result.and_then(|()| {
        cb.commit();
        wait_completed(&cb)
    });
    let wall = start.elapsed().as_secs_f64() * 1e3;
    let gpu = (result.is_ok())
        .then(|| (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3)
        .filter(|v| v.is_finite() && *v > 0.0);
    (result, wall, gpu)
}
fn scan(t: &MetalTensor, reference: Option<&[f32]>) -> Value {
    if let Some(r) = reference {
        assert_eq!(r.len(), t.n_elements() as usize);
    }
    let mut hash = Sha256::new();
    let (mut max_abs, mut error2, mut norm2) = (0.0f64, 0.0f64, 0.0f64);
    for start in (0..t.n_elements() as usize).step_by(READ) {
        let n = READ.min(t.n_elements() as usize - start);
        let values = read_f32(&t.view_subrange(start as u64, vec![n as u64])).unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite prepared/captured array"
        );
        hash.update(bytemuck::cast_slice(&values));
        if let Some(r) = reference {
            for (&a, &b) in r[start..start + n].iter().zip(&values) {
                let d = f64::from(a) - f64::from(b);
                max_abs = max_abs.max(d.abs());
                error2 += d * d;
                norm2 += f64::from(a).powi(2);
            }
        }
    }
    json!({"elements":t.n_elements(),"sha256_f32_le":format!("{:x}",hash.finalize()),"finite":true,
        "vs_a1":reference.map(|_|json!({"max_abs":max_abs,"relative_l2":(error2/norm2.max(1e-30)).sqrt()}))})
}
fn save_reference(t: &MetalTensor, r: &mut [f32]) {
    for (i, chunk) in r.chunks_mut(READ).enumerate() {
        chunk.copy_from_slice(
            &read_f32(&t.view_subrange((i * READ) as u64, vec![chunk.len() as u64])).unwrap(),
        );
    }
}
fn census(candidate: bool, leaf: bool) -> Value {
    let all = dispatch_census_take();
    let prepare: Vec<_> = all
        .iter()
        .filter(|r| r.kernel.starts_with("kernel_glm53_kda_prefill_prepare"))
        .collect();
    assert!(!prepare.is_empty());
    let expected = if candidate {
        "kernel_glm53_kda_prefill_prepare_register_history"
    } else {
        "kernel_glm53_kda_prefill_prepare"
    };
    assert!(prepare.iter().all(|r| r.kernel == expected));
    if leaf {
        assert_eq!(
            all.len(),
            1,
            "prepare-only must not dispatch recurrence/output"
        );
    }
    json!(
        prepare
            .iter()
            .map(|r| json!({"kernel":r.kernel,"grid":[r.grid_width,r.grid_height,r.grid_depth]}))
            .collect::<Vec<_>>()
    )
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn leaf(
    ctx: &MetalContext,
    session: &Glm5NextSession<'_>,
    block: usize,
    c: &Capture,
    rows: usize,
    out: &mut File,
) {
    let cfg = &session.weights.config;
    let MixerTensors::Kda(kda) = &session.weights.blocks[block].mixer else {
        unreachable!()
    };
    let LayerState::Kda { conv, state } = &session.layers[block] else {
        unreachable!()
    };
    let p = session.packed.as_ref().unwrap();
    // Existing session work buffers satisfy the full KdaDecode leaf validation.
    // Unused beta/output/state bindings are neither captured nor copied.
    let work = [
        rows_view(&p.q, rows),
        rows_view(&p.k, rows),
        rows_view(&p.v, rows),
        rows_view(&p.raw_gate, rows),
        conv.clone(),
    ];
    let unused = [
        rows_view(&p.raw_beta, rows),
        rows_view(&p.output_gate, rows),
        rows_view(&p.kda_out, rows),
    ];
    let bindings = KdaDecode {
        q: &work[0],
        k: &work[1],
        v: &work[2],
        raw_gate: &work[3],
        conv_state: &work[4],
        q_conv: &kda.query_conv,
        k_conv: &kda.key_conv,
        v_conv: &kda.value_conv,
        neg_exp_a_log: &kda.neg_exp_a_log,
        dt_bias: &kda.decay_bias,
        raw_beta: &unused[0],
        output_gate: &unused[1],
        output_norm: &kda.output_norm,
        state,
        out: &unused[2],
    };
    record(
        out,
        json!({"event":"leaf_binding","block":block,"rows":rows,
        "source_capture_rows":c.rows,"source_row_offset":0,"incoming_history":scan(&c.data[4],None),
        "restore_buffers":NAMES,"restore_policy":"separate command and wait before timing"}),
    );
    let mut references: [Vec<f32>; 5] =
        std::array::from_fn(|i| vec![0.125; work[i].n_elements() as usize]);
    let mut ready = false;
    let mut times = [Vec::new(), Vec::new()];
    let mut gpus = [Vec::new(), Vec::new()];
    for (label, candidate) in ARMS {
        // Restore all four raw arrays AND the same incoming convolution history;
        // this command completes outside the measured preparation interval.
        command(ctx, |enc| {
            for (src, dst) in c.data.iter().zip(&work) {
                let n = dst.n_elements() as usize;
                crate::metal::encode_copy_offset_f32(ctx, enc, &flat(src, n), 0, &flat(dst, n), n)?;
            }
            Ok(())
        })
        .0
        .unwrap();
        let warm = label.starts_with("warm");
        if warm {
            dispatch_census_begin();
        }
        let (result, wall, gpu) = arm(candidate, || {
            command(ctx, |enc| {
                encode_kda_prefill_prepare(
                    ctx,
                    enc,
                    cfg.head_count as usize,
                    rows,
                    &bindings,
                    cfg.kda_gate_lower_bound,
                    cfg.rms_epsilon,
                )
            })
        });
        let trace = warm.then(|| census(candidate, true));
        record(
            out,
            json!({"event":"leaf_attempt","rows":rows,"label":label,"wall_ms":wall,"gpu_ms":gpu,
            "census":trace,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        result.unwrap();
        let metrics:Vec<_>=work.iter().enumerate().map(|(i,t)|json!({"name":NAMES[i],"metrics":scan(t,ready.then_some(references[i].as_slice()))})).collect();
        record(
            out,
            json!({"event":"leaf_output","rows":rows,"label":label,"arrays":metrics}),
        );
        if label == "A1" {
            for (t, r) in work.iter().zip(&mut references) {
                save_reference(t, r);
            }
            ready = true;
        }
        if !warm {
            times[usize::from(candidate)].push(wall);
            gpus[usize::from(candidate)].extend(gpu);
        }
    }
    let (a, b) = (mean(&times[0]), mean(&times[1]));
    record(
        out,
        json!({"event":"leaf_summary","rows":rows,"a_wall_ms":a,"b_wall_ms":b,"a_over_b_wall_ratio":a/b,
        "gpu_summary":gpus.iter().all(|v|v.len()==2).then(||json!({"a_ms":mean(&gpus[0]),"b_ms":mean(&gpus[1]),"a_over_b":mean(&gpus[0])/mean(&gpus[1])}))}),
    );
    eprintln!(
        "KDA prepare N{rows}: A={a:.3}ms B={b:.3}ms A/B={:.4}x",
        a / b
    );
}

fn whole(
    ctx: &MetalContext,
    w: &Glm5NextWeights,
    tokens: &[u32],
    rows: usize,
    cpu: u64,
    out: &mut File,
) {
    assert!(CAPTURE.with(|s| s.borrow().is_none()));
    let end = PREFIX + rows;
    let mut reference: Option<Vec<Vec<f32>>> = None;
    let mut times = [Vec::new(), Vec::new()];
    for (label, candidate) in ARMS {
        let warm = label.starts_with("warm");
        record(
            out,
            json!({"event":"whole_session_begin","rows":rows,"label":label,
            "warmup":warm,"capacity":end+STEPS,"workspace_rows":rows,"cpu_reserve_bytes":cpu,
            "current_gpu_bytes":ctx.current_allocated_size()}),
        );
        let start = Instant::now();
        let session =
            Glm5NextSession::with_prefill_rows_and_cpu_reserve(ctx, w, end + STEPS, rows, cpu);
        let allocation_ms = start.elapsed().as_secs_f64() * 1e3;
        record(
            out,
            json!({"event":"whole_session_allocated","rows":rows,"label":label,
            "wall_ms":allocation_ms,"measured":false,"current_gpu_bytes":ctx.current_allocated_size(),
            "error":session.as_ref().err().map(ToString::to_string)}),
        );
        let mut s = session.unwrap();
        assert_eq!(s.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(s.fast_precision(), Some(FastPrecision::Half));
        record(
            out,
            json!({"event":"whole_prefix_begin","rows":rows,"label":label,
            "position":s.position(),"prefix_rows":PREFIX,"current_gpu_bytes":ctx.current_allocated_size()}),
        );
        let start = Instant::now();
        let prefix = arm(false, || s.prefill_packed(ctx, &tokens[..PREFIX]));
        let prefix_ms = start.elapsed().as_secs_f64() * 1e3;
        record(
            out,
            json!({"event":"whole_prefix_end","rows":rows,"label":label,
            "position":s.position(),"wall_ms":prefix_ms,"measured":false,
            "current_gpu_bytes":ctx.current_allocated_size(),"error":prefix.as_ref().err().map(ToString::to_string)}),
        );
        let prefix = prefix.unwrap();
        assert!(prefix.iter().all(|v| v.is_finite()));
        drop(prefix);
        assert_eq!(s.position(), PREFIX);
        record(
            out,
            json!({"event":"whole_prefill_begin","rows":rows,"label":label,
            "position":s.position(),"warmup":warm,"measured":!warm,
            "current_gpu_bytes":ctx.current_allocated_size()}),
        );
        if warm {
            dispatch_census_begin();
        }
        let (result, ms) = arm(candidate, || {
            let start = Instant::now();
            let r = s.prefill_packed(ctx, &tokens[PREFIX..end]);
            (r, start.elapsed().as_secs_f64() * 1e3)
        });
        let trace = warm.then(|| census(candidate, false));
        record(
            out,
            json!({"event":"whole_attempt","rows":rows,"label":label,"wall_ms":ms,"warmup":warm,"measured":!warm,"position":s.position(),
            "current_gpu_bytes":ctx.current_allocated_size(),"census":trace,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        let mut logits = vec![result.unwrap()];
        for (step, &token) in tokens[end..end + STEPS].iter().enumerate() {
            record(
                out,
                json!({"event":"whole_continuation_begin","rows":rows,"label":label,
                "step":step,"token":token,"position":s.position(),"current_gpu_bytes":ctx.current_allocated_size()}),
            );
            let start = Instant::now();
            let result = s.forward(ctx, token);
            let decode_ms = start.elapsed().as_secs_f64() * 1e3;
            record(
                out,
                json!({"event":"whole_continuation_end","rows":rows,"label":label,
                "step":step,"token":token,"position":s.position(),"wall_ms":decode_ms,"measured":false,
                "current_gpu_bytes":ctx.current_allocated_size(),"error":result.as_ref().err().map(ToString::to_string)}),
            );
            logits.push(result.unwrap());
        }
        assert_eq!(s.position(), end + STEPS);
        let comparisons:Vec<_>=logits.iter().enumerate().map(|(step,row)|{
            assert_eq!(row.len(),w.config.vocab_size as usize);assert!(row.iter().all(|v|v.is_finite()));
            json!({"position":end-1+step,"sha256_f32_le":format!("{:x}",Sha256::digest(bytemuck::cast_slice(row))),
                "kl_a1_to_run":reference.as_ref().map(|a|kl_divergence(&a[step],row))})}).collect();
        record(
            out,
            json!({"event":"whole_output","rows":rows,"label":label,"steps":comparisons,"teacher_forced_tokens":&tokens[end..end+STEPS]}),
        );
        if !warm {
            times[usize::from(candidate)].push(ms);
        }
        if label == "A1" {
            reference = Some(logits);
        }
        record(
            out,
            json!({"event":"whole_session_drop_begin","rows":rows,"label":label,
            "current_gpu_bytes":ctx.current_allocated_size()}),
        );
        let start = Instant::now();
        drop(s);
        let drop_ms = start.elapsed().as_secs_f64() * 1e3;
        record(
            out,
            json!({"event":"whole_session_drop_end","rows":rows,"label":label,
            "wall_ms":drop_ms,"measured":false,"current_gpu_bytes":ctx.current_allocated_size()}),
        );
    }
    let (a, b) = (mean(&times[0]), mean(&times[1]));
    record(
        out,
        json!({"event":"whole_summary","rows":rows,"a_ms":a,"b_ms":b,"a_over_b_wall_ratio":a/b}),
    );
    eprintln!(
        "whole KDA prepare prefix{PREFIX}+N{rows}: A={a:.2}ms B={b:.2}ms A/B={:.4}x",
        a / b
    );
}

fn flag(name: &str) -> bool {
    match std::env::var(name).as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("{name} must be absent, 0 or 1"),
    }
}

#[test]
#[ignore = "Maya KDA prepare ABBA: release, GLM53_GGUF, new GLM53_KDA_PREP_OUT, no MTL_DEBUG_LAYER; optional GLM53_KDA_PREP_WHOLE=1 or GLM53_KDA_PREP_WHOLE_ONLY=1"]
fn kda_prepare_ab_packet() {
    assert!(!cfg!(debug_assertions), "release binary required");
    let _lease = perf_lease();
    let path = std::path::PathBuf::from(std::env::var_os("GLM53_GGUF").expect("Maya GGUF"));
    let whole_only = flag("GLM53_KDA_PREP_WHOLE_ONLY");
    let whole_on = flag("GLM53_KDA_PREP_WHOLE") || whole_only;
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("GLM53_KDA_PREP_OUT").expect("new JSONL path"))
        .unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let capacity = PREFIX + ROWS + STEPS;
    assert!(capacity <= artifact.model().config.context_length as usize);
    preflight_session(&ctx, &gguf, artifact.model(), capacity, ROWS).unwrap();
    let text = long_qualification_text();
    let tokens: Vec<u32> = artifact
        .tokenizer()
        .encode(&format!("[gMASK]<sop>{text}\n\n{text}"), false)
        .unwrap()
        .into_iter()
        .take(capacity)
        .map(|t| u32::try_from(t).unwrap())
        .collect();
    assert_eq!(tokens.len(), capacity);
    let w = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let block = w
        .blocks
        .iter()
        .position(|b| matches!(&b.mixer, MixerTensors::Kda(_)))
        .expect("actual KDA block");
    let MixerTensors::Kda(kda) = &w.blocks[block].mixer else {
        unreachable!()
    };
    let weight_bindings: Vec<_> = [
        ("query_conv", &kda.query_conv),
        ("key_conv", &kda.key_conv),
        ("value_conv", &kda.value_conv),
        ("neg_exp_a_log", &kda.neg_exp_a_log),
        ("decay_bias", &kda.decay_bias),
    ]
    .into_iter()
    .map(|(name, t)| {
        json!({"name":name,"shape":t.shape,
        "dtype":format!("{:?}",t.dtype),"contents":scan(t,None)})
    })
    .collect();
    let heads = w.config.head_count as usize;
    let width = w.config.kda_width() as usize;
    assert_eq!(width, heads * crate::metal::KDA_HEAD_DIM);
    let token_bytes = tokens.capacity() as u64 * 4;
    let vocab = u64::from(w.config.vocab_size);
    let shapes: [Vec<u64>; 5] = std::array::from_fn(|i| {
        if i < 4 {
            vec![width as u64, ROWS as u64]
        } else {
            vec![width as u64, 3, 3]
        }
    });
    let logical: Vec<u64> = shapes
        .iter()
        .map(|s| s.iter().product::<u64>() * 4)
        .collect();
    let cpu = logical.iter().sum::<u64>() + READ as u64 * 4 + vocab * 4 + token_bytes;
    let whole_cpu = (2 * (STEPS + 1) + 1) as u64 * vocab * 4 + 2 * vocab * 8 + token_bytes;
    record(
        &mut out,
        json!({"event":"packet","schema":"glm53.kda_prepare_ab.v2","model":path,"test_binary":std::env::current_exe().unwrap(),
        "device":ctx.describe(),"block":block,"heads":heads,"width":width,"prefix":PREFIX,"capture_rows":(!whole_only).then_some(ROWS),"leaf_rows":(!whole_only).then_some([2048,512]),
        "weight_bindings":weight_bindings,"lower_bound":w.config.kda_gate_lower_bound,"norm_eps":w.config.rms_epsilon,
        "tokens":tokens,"whole_enabled":whole_on,"whole_only":whole_only,"capture_enabled":!whole_only,"precision":"default Fast Half","arms":{"A":"production prepare","B":"test-only register history prepare"},
        "order":"warm A B then A1 B1 B2 A2; five mutable buffers restored outside every leaf timing",
        "leaf_cpu_bytes":if whole_only {0} else {cpu},"leaf_cpu_terms":(!whole_only).then(||json!({"five_reference_arrays":logical,"read_chunk":READ*4,"endpoint_logits":vocab*4,"tokens":token_bytes})),
        "whole_cpu_bytes":whole_cpu,"whole_cpu_terms":{"f32_rows":2*(STEPS+1)+1,"f64_vectors":2,"vocab":vocab,"tokens":token_bytes},
        "extra_work_gpu_bytes":0,"work_policy":"reuse five mutable session buffers; full prepare-only API validates unused session beta/output/recurrence bindings but never dispatches them",
        "logging":"flush each event; sync_data once on successful packet completion; flushed partial records survive process exit but are not power-loss durable",
        "completion_policy":"incomplete packets retained as diagnostics, no full performance claim",
        "numerics":"finite arrays and hashes, numeric differences/KL recorded without promotion bounds","performance_assertion":false}),
    );
    let ledger = Glm5NextSession::plan_memory(&ctx, &w, capacity, ROWS).unwrap();
    record(
        &mut out,
        json!({"event":"session_memory","capacity":capacity,"workspace_rows":ROWS,
        "ledger_peak_bytes":ledger.peak_bytes(),"session_buffer_bytes":ledger.session_buffer_bytes(),
        "capture_logical_bytes":if whole_only {0} else {logical.iter().sum::<u64>()},"extra_work_gpu_bytes":0}),
    );
    if !whole_only {
        record(
            &mut out,
            json!({"event":"capture_phase_begin","current_gpu_bytes":ctx.current_allocated_size()}),
        );
        let mut session =
            Glm5NextSession::with_prefill_rows_and_cpu_reserve(&ctx, &w, capacity, ROWS, cpu)
                .unwrap();
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        let logits = arm(false, || session.prefill_packed(&ctx, &tokens[..PREFIX])).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        drop(logits);
        assert_eq!(session.position(), PREFIX);
        let LayerState::Kda { conv, .. } = &session.layers[block] else {
            unreachable!()
        };
        let priced: Vec<_> = logical
            .iter()
            .map(|&n| ctx.price_shared_buffer_upper(n).unwrap().priced_upper_bytes)
            .collect();
        let capture = {
            let _allocation = ctx.begin_allocation_transaction();
            let admission = evaluate_metal_memory_admission_with_cpu_bytes(
                priced.iter().sum(),
                cpu,
                0,
                ctx.memory_signals(),
                true,
            );
            record(
                &mut out,
                json!({"event":"capture_admission","names":NAMES,"shapes":shapes,"logical_bytes":logical,"priced_gpu_bytes":priced,
                "cpu_bytes":cpu,"current_gpu_bytes":admission.signals.current_allocated_bytes,"admitted":admission.admitted,"reason":admission.reason.as_str()}),
            );
            assert!(admission.admitted);
            Rc::new(Capture {
                data: std::array::from_fn(|i| {
                    MetalTensor::zeros_f32(&ctx, shapes[i].clone()).unwrap()
                }),
                source: identity(conv),
                rows: ROWS,
                copies: Cell::new(0),
            })
        };
        let result = arm(false, || {
            scoped_capture(capture.clone(), || {
                session.prefill_packed(&ctx, &tokens[PREFIX..PREFIX + ROWS])
            })
        });
        record(
            &mut out,
            json!({"event":"capture","copies":capture.copies.get(),"position":session.position(),"error":result.as_ref().err().map(ToString::to_string)}),
        );
        let logits = result.unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        drop(logits);
        assert_eq!(capture.copies.get(), 1);
        assert_eq!(session.position(), PREFIX + ROWS);
        let original: Vec<_> = capture.data.iter().map(|t| scan(t, None)).collect();
        record(
            &mut out,
            json!({"event":"raw_capture","names":NAMES,"arrays":original}),
        );
        for rows in [2048, 512] {
            leaf(&ctx, &session, block, &capture, rows, &mut out);
        }
        for (t, hash) in capture.data.iter().zip(original) {
            assert_eq!(
                scan(t, None)["sha256_f32_le"],
                hash["sha256_f32_le"],
                "capture mutated"
            );
        }
        record(
            &mut out,
            json!({"event":"capture_drop_begin","current_gpu_bytes":ctx.current_allocated_size()}),
        );
    }
    record(
        &mut out,
        json!({"event":"capture_phase_end","skipped":whole_only,"current_gpu_bytes":ctx.current_allocated_size()}),
    );
    if whole_on {
        for rows in [2048, 512] {
            whole(&ctx, &w, &tokens, rows, whole_cpu, &mut out);
        }
    }
    record(
        &mut out,
        json!({"event":"packet_complete","whole_enabled":whole_on,
        "whole_only":whole_only,"current_gpu_bytes":ctx.current_allocated_size()}),
    );
    out.flush().unwrap();
    out.sync_data().unwrap();
}

//! Diagnostic MLA tail replay; no production-policy changes.
//!
//! env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
//!   GLM53_MLA_TAIL_OUT=/tmp/glm-mla-tail-leaf.jsonl \
//!   cargo test --release -p qwen-llm --lib \
//!   glm5_next_metal::packed::mla_tail::mla_tail_packet \
//!   -- --ignored --exact --nocapture --test-threads=1
//!
//! GLM53_MLA_TAIL_FULL=1 bypasses leaf replay: two fixed streams, eight fresh
//! widths and a fresh incumbent 4096-token prefix + timed 127-token suffix.
//! A forces old absorption, B enables the guarded tail. All other policies
//! remain production defaults. Warm A/B then B1/A1/A2/B2; no bitwise gate.
//! Leaf: five setup prefills and 60 timed leaf commands. Full: 108 target
//! prefills and 12 separately timed prefix setups if deep sessions are admitted.

use super::super::tests::{argmax, kl_divergence, long_qualification_text, perf_lease};
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{cell::Cell, fs::File, io::Write, time::Instant};

const HEADS: usize = 64;
const BLOCK: usize = 43;
const READ: usize = 256 * 1024; // One MiB readback, never a full candidate copy.
const FULL_CPU: u64 = 32 << 20;
const ARMS: [(&str, bool); 6] = [
    ("warm_A", false),
    ("warm_B", true),
    ("B1", true),
    ("A1", false),
    ("A2", false),
    ("B2", true),
];
const CODE: &str = include_str!("../../../../qwen-cli/src/serve/backend_glm5_next.rs");
const RENDER: &str = include_str!("../../../../qwen-cli/src/serve/render_glm5_next.rs");

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Timing {
    active: bool,
    completed: usize,
    valid: usize,
    gpu_ms: f64,
}
thread_local! { static TIMING: Cell<Timing> = Cell::new(Timing::default()); }

fn milliseconds(start: f64, end: f64) -> Option<f64> {
    let ms = (end - start) * 1e3;
    (start.is_finite() && start > 0.0 && end.is_finite() && ms.is_finite() && ms > 0.0)
        .then_some(ms)
}

/// Called by packed after a successful wait; inactive packets never query GPU time.
pub(super) fn record_completed_command(timestamps: impl FnOnce() -> (f64, f64)) {
    let mut t = TIMING.get();
    if !t.active {
        return;
    }
    let (start, end) = timestamps();
    t.completed += 1;
    if let Some(ms) = milliseconds(start, end) {
        t.valid += 1;
        t.gpu_ms += ms;
    }
    TIMING.set(t);
}

fn observe<R>(f: impl FnOnce() -> R) -> (R, Timing) {
    struct Restore(Timing);
    impl Drop for Restore {
        fn drop(&mut self) {
            TIMING.set(self.0);
        }
    }
    let _restore = Restore(TIMING.replace(Timing {
        active: true,
        ..Timing::default()
    }));
    let r = f();
    (r, TIMING.get())
}

#[derive(Clone, Copy)]
struct Case {
    stream: &'static str,
    prefix: usize,
    rows: usize,
    label: &'static str,
    candidate: bool,
}
fn emit(out: &mut File, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
fn record(out: &mut File, event: &str, c: Case, mut value: Value) {
    value.as_object_mut().unwrap().extend(
        json!({"event":event,"stream":c.stream,
        "prefix":c.prefix,"rows":c.rows,"label":c.label,"candidate":c.candidate})
        .as_object()
        .unwrap()
        .clone(),
    );
    emit(out, value);
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn tensor_hash(t: &MetalTensor) -> String {
    let mut h = Sha256::new();
    for start in (0..t.n_elements() as usize).step_by(READ) {
        let n = READ.min(t.n_elements() as usize - start);
        h.update(bytemuck::cast_slice(
            &read_f32(&t.view_subrange(start as u64, vec![n as u64])).unwrap(),
        ));
    }
    format!("{:x}", h.finalize())
}

#[derive(Default)]
struct Difference {
    max_abs: f64,
    error2: f64,
    norm2: f64,
}
impl Difference {
    fn add(&mut self, a: f32, b: f32) {
        let d = f64::from(a) - f64::from(b);
        self.max_abs = self.max_abs.max(d.abs());
        self.error2 += d * d;
        self.norm2 += f64::from(a).powi(2);
    }
    fn json(&self) -> Value {
        json!({"max_abs":self.max_abs,"relative_l2":(self.error2/self.norm2.max(1e-30)).sqrt()})
    }
}

fn logits_summary(out: &mut File, c: Case, step: usize, logits: &[f32]) {
    let nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
    record(
        out,
        "output",
        c,
        json!({"teacher_forced_steps":step,"position":c.prefix+c.rows+step,
        "sha256_f32_le":sha(bytemuck::cast_slice(logits)),"nonfinite":nonfinite,
        "top1":(nonfinite==0).then(||argmax(logits))}),
    );
    assert_eq!(nonfinite, 0, "nonfinite output; raw record flushed");
    assert!(!logits.is_empty());
}
fn compare_logits(
    out: &mut File,
    c: Case,
    reference_label: &str,
    reference: &[Vec<f32>],
    actual: &[Vec<f32>],
) {
    assert_eq!(reference.len(), actual.len());
    for (step, (a, b)) in reference.iter().zip(actual).enumerate() {
        assert_eq!(a.len(), b.len());
        let mut diff = Difference::default();
        for (&a, &b) in a.iter().zip(b) {
            assert!(a.is_finite() && b.is_finite());
            diff.add(a, b);
        }
        let (at, bt) = (argmax(a), argmax(b));
        let (kl_ab, kl_ba) = (kl_divergence(a, b), kl_divergence(b, a));
        assert!(kl_ab.is_finite() && kl_ba.is_finite());
        record(
            out,
            "output_comparison",
            c,
            json!({"reference_label":reference_label,"teacher_forced_steps":step,
            "difference":diff.json(),"kl_reference_actual":kl_ab,"kl_actual_reference":kl_ba,
            "reference_top1":at,"actual_top1":bt,
            "reference_choice_regret":f64::from(a[at])-f64::from(a[bt]),
            "actual_choice_regret":f64::from(b[bt])-f64::from(b[at])}),
        );
    }
}

fn expected_substitutions(rows: usize, chunk: usize, candidate: bool) -> usize {
    if !candidate {
        return 0;
    }
    (0..rows)
        .step_by(chunk)
        .filter(|&start| (rows - start).min(chunk) % 128 >= 8)
        .count()
        * 22
}

fn allocate<'w>(
    ctx: &MetalContext,
    weights: &'w Glm5NextWeights,
    capacity: usize,
    chunk: usize,
    cpu: u64,
    out: &mut File,
    c: Case,
) -> Option<Glm5NextSession<'w>> {
    let ledger = Glm5NextMemoryLedger::new(
        &weights.config,
        weights.retained_bytes,
        capacity as u64,
        chunk as u64,
        &device_price(ctx),
    )
    .unwrap();
    let bytes = ledger.peak_bytes() - weights.retained_bytes;
    let admission =
        evaluate_metal_memory_admission_with_cpu_bytes(bytes, cpu, 0, ctx.memory_signals(), true);
    record(
        out,
        "admission",
        c,
        json!({"capacity":capacity,"chunk_rows":chunk,"cpu_reserve_bytes":cpu,
        "session_gpu_upper_bytes":bytes,"admitted":admission.admitted,"reason":admission.reason.as_str()}),
    );
    if !admission.admitted {
        return None;
    }
    let start = Instant::now();
    let result =
        Glm5NextSession::with_prefill_rows_and_cpu_reserve(ctx, weights, capacity, chunk, cpu);
    record(
        out,
        "session_allocation",
        c,
        json!({"wall_ms":start.elapsed().as_secs_f64()*1e3,
        "error":result.as_ref().err().map(ToString::to_string)}),
    );
    let mut s = result.expect("ordinary session admission/allocation failed; record flushed");
    s.set_packed_lineage(PackedLineage::Fast);
    Some(s)
}

fn prefill(
    ctx: &MetalContext,
    s: &mut Glm5NextSession<'_>,
    tokens: &[u32],
    out: &mut File,
    c: Case,
    phase: &str,
) -> Vec<f32> {
    assert_eq!(s.position(), c.prefix);
    assert_eq!(tokens.len(), c.rows);
    assert_eq!(super::router_prefill::requested_override(), None);
    let chunk = s.packed.as_ref().unwrap().rows;
    let expected = expected_substitutions(tokens.len(), chunk, c.candidate);
    let (((result, wall_ms), time), substitutions) =
        with_absorb_tail_variant(Some(c.candidate), || {
            observe(|| {
                let start = Instant::now();
                let result = s.prefill_packed(ctx, tokens);
                (result, start.elapsed().as_secs_f64() * 1e3)
            })
        });
    let commands = tokens.len().div_ceil(chunk);
    let gpu = (result.is_ok()
        && time.completed == commands
        && time.valid == commands
        && time.gpu_ms.is_finite()
        && time.gpu_ms > 0.0)
        .then_some(time.gpu_ms);
    record(
        out,
        "prefill_attempt",
        c,
        json!({"phase":phase,"chunk_rows":chunk,"wall_ms":wall_ms,
        "measured":phase=="whole_target" && matches!(c.label,"B1"|"A1"|"A2"|"B2"),
        "command_gpu_ms":gpu,"command_gpu_valid":gpu.is_some(),"completed_commands":time.completed,
        "valid_commands":time.valid,"expected_commands":commands,"substitutions":substitutions,
        "expected_substitutions":expected,"error":result.as_ref().err().map(ToString::to_string)}),
    );
    let logits = result.expect("prefill failed; raw attempt flushed");
    assert_eq!(substitutions, expected, "tail substitution topology");
    assert_eq!(s.position(), c.prefix + c.rows);
    logits_summary(out, c, 0, &logits);
    assert!(gpu.is_some(), "invalid GPU timestamps; raw attempt flushed");
    logits
}

fn command(
    ctx: &MetalContext,
    encode: impl FnOnce(&KernelEncoder) -> Result<()>,
) -> (Result<()>, f64, Option<f64>) {
    let start = Instant::now();
    let cmd = ctx.queue.commandBuffer().expect("leaf command");
    let enc = KernelEncoder::begin(&cmd);
    let result = encode(&enc);
    enc.end();
    let result = result.and_then(|()| {
        cmd.commit();
        wait_completed(&cmd).map_err(Into::into)
    });
    let wall = start.elapsed().as_secs_f64() * 1e3;
    let gpu = result
        .is_ok()
        .then(|| milliseconds(cmd.GPUStartTime(), cmd.GPUEndTime()))
        .flatten();
    (result, wall, gpu)
}

fn poison(t: &MetalTensor) {
    assert_eq!(t.dtype, GgmlType::F32);
    assert!(t.is_writable() && t.offset.is_multiple_of(4));
    assert!(
        t.offset
            .checked_add(t.n_elements() * 4)
            .is_some_and(|end| end <= t.buffer.length() as u64)
    );
    // SAFETY: existing writable shared scratch, all previous commands waited,
    // checked alignment and bounds. No weight/input tensor is written here.
    unsafe {
        let ptr = t
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(t.offset as usize)
            .cast::<f32>();
        for i in 0..t.n_elements() as usize {
            ptr.add(i).write(f32::NAN);
        }
    }
}

fn leaf_output(
    out: &mut File,
    c: Case,
    direction: &str,
    t: &MetalTensor,
    reference: Option<(&str, &[f32])>,
) {
    let mut hash = Sha256::new();
    let mut diff = Difference::default();
    let mut nonfinite = 0usize;
    for start in (0..t.n_elements() as usize).step_by(READ) {
        let n = READ.min(t.n_elements() as usize - start);
        let values = read_f32(&t.view_subrange(start as u64, vec![n as u64])).unwrap();
        hash.update(bytemuck::cast_slice(&values));
        for (i, &v) in values.iter().enumerate() {
            nonfinite += usize::from(!v.is_finite());
            if let Some((_, r)) = reference {
                if v.is_finite() {
                    diff.add(r[start + i], v);
                }
            }
        }
    }
    record(
        out,
        "leaf_output",
        c,
        json!({"direction":direction,"elements":t.n_elements(),
        "sha256_f32_le":format!("{:x}",hash.finalize()),"nonfinite":nonfinite,
        "reference_label":reference.map(|(name,_)|name),"difference":reference.map(|_|diff.json())}),
    );
    assert_eq!(
        nonfinite, 0,
        "nonfinite or unwritten leaf output; raw record flushed"
    );
}
fn save(t: &MetalTensor, r: &mut [f32]) {
    assert_eq!(t.n_elements() as usize, r.len());
    for (i, chunk) in r.chunks_mut(READ).enumerate() {
        chunk.copy_from_slice(
            &read_f32(&t.view_subrange((i * READ) as u64, vec![chunk.len() as u64])).unwrap(),
        );
    }
}

fn leaf(ctx: &MetalContext, weights: &Glm5NextWeights, tokens: &[u32], out: &mut File) {
    for rows in [8usize, 32, 64, 127, 128] {
        let base = Case {
            stream: "qualification",
            prefix: 0,
            rows,
            label: "setup",
            candidate: false,
        };
        let cpu = (16 << 20) + (rows * HEADS * 512 * 4 + READ * 4) as u64;
        let mut session = allocate(ctx, weights, rows, rows, cpu, out, base)
            .expect("leaf session admission refused");
        prefill(
            ctx,
            &mut session,
            &tokens[..rows],
            out,
            base,
            "leaf_setup_not_performance_evidence",
        );
        let p = session.packed.as_ref().unwrap();
        let MixerTensors::Mla(mla) = &weights.blocks[BLOCK].mixer else {
            panic!("final MLA binding");
        };
        assert!(matches!(&weights.blocks[44].mixer, MixerTensors::Kda(_)));
        // In packed.rs these buffers are written only by encode_mla. Block44
        // is KDA, so these are block43's actual inputs after normal prefill.
        let inputs = [
            flat(&p.query, rows * HEADS * 256),
            flat(&p.output_latent, rows * HEADS * 512),
        ];
        let hashes = [tensor_hash(&inputs[0]), tensor_hash(&inputs[1])];
        for (direction, weight, input, output, k, m) in [
            (
                "key_absorb",
                &mla.key_absorb,
                &p.query,
                &p.query_latent,
                256,
                512,
            ),
            (
                "value_expand",
                &mla.value_expand,
                &p.output_latent,
                &p.heads_out,
                512,
                256,
            ),
        ] {
            let target = flat(output, rows * HEADS * m);
            record(
                out,
                "leaf_binding",
                base,
                json!({"block":BLOCK,"direction":direction,"K":k,"M":m,"groups":HEADS,
                "weight_dtype":format!("{:?}",weight.dtype),"weight_shape":weight.shape,"weight_buffer_offset":weight.offset,
                "input_token_stride":HEADS*k,"output_token_stride":HEADS*m,"input_offset":input.offset,"output_offset":output.offset,
                "input_sha256_f32_le":tensor_hash(&flat(input,rows*HEADS*k)),
                "capture_gpu_bytes":0,"cpu_reserve_bytes":cpu,"reference_bytes":rows*HEADS*m*4}),
            );
            // One reference allocated/touched before warmup. B1 is retained
            // until A1 arrives; then A1 replaces it for A2 and B2 comparisons.
            let mut reference = vec![0.125f32; rows * HEADS * m];
            let mut reference_label = None;
            for (label, candidate) in ARMS {
                let c = Case {
                    label,
                    candidate,
                    ..base
                };
                poison(&target);
                let ((result, wall, gpu), substitutions) =
                    with_absorb_tail_variant(Some(candidate), || {
                        command(ctx, |enc| {
                            absorb_rows(
                                ctx,
                                enc,
                                PackedLineage::Fast,
                                weight,
                                input,
                                output,
                                k,
                                m,
                                HEADS,
                                rows,
                            )
                        })
                    });
                let expected = usize::from(candidate && rows % 128 >= 8);
                let remainder = rows % 128;
                let dispatches = usize::from(rows >= 128)
                    + if candidate {
                        usize::from(remainder >= 8) + remainder % 8
                    } else {
                        remainder
                    };
                record(
                    out,
                    "leaf_attempt",
                    c,
                    json!({"block":BLOCK,"direction":direction,"wall_ms":wall,
                    "measured":matches!(label,"B1"|"A1"|"A2"|"B2"),
                    "command_gpu_ms":gpu,"command_gpu_valid":gpu.is_some(),"dispatches_expected":dispatches,
                    "substitutions":substitutions,"expected_substitutions":expected,
                    "error":result.as_ref().err().map(ToString::to_string)}),
                );
                result.expect("leaf failed; raw attempt flushed");
                assert_eq!(substitutions, expected);
                leaf_output(
                    out,
                    c,
                    direction,
                    &target,
                    reference_label.map(|name| (name, reference.as_slice())),
                );
                assert!(
                    gpu.is_some(),
                    "invalid leaf GPU timestamps; raw attempt flushed"
                );
                if matches!(label, "B1" | "A1") {
                    save(&target, &mut reference);
                    reference_label = Some(label);
                }
            }
            for (i, t) in inputs.iter().enumerate() {
                assert_eq!(tensor_hash(t), hashes[i], "MLA replay input mutated");
            }
        }
    }
}

fn whole(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    streams: &[(&'static str, Vec<u32>)],
    out: &mut File,
) {
    for (stream, tokens) in streams {
        for (prefix, rows) in [8usize, 32, 64, 127, 128, 129, 255, 512]
            .into_iter()
            .map(|n| (0, n))
            .chain([(4096, 127)])
        {
            let steps = if matches!(rows, 32 | 127 | 255) { 4 } else { 0 };
            let mut reference: Option<Vec<Vec<f32>>> = None;
            let mut reference_label = "B1";
            let mut complete = true;
            for (label, candidate) in ARMS {
                let c = Case {
                    stream,
                    prefix,
                    rows,
                    label,
                    candidate,
                };
                let chunk = if prefix > 0 { 512 } else { rows };
                let Some(mut session) =
                    allocate(ctx, weights, prefix + rows + steps, chunk, FULL_CPU, out, c)
                else {
                    record(
                        out,
                        "cell_skipped",
                        c,
                        json!({"reason":"normal memory admission refused","complete":false}),
                    );
                    assert!(prefix > 0, "required short cell not admitted");
                    complete = false;
                    break;
                };
                if prefix > 0 {
                    let setup = Case {
                        prefix: 0,
                        rows: prefix,
                        candidate: false,
                        ..c
                    };
                    prefill(
                        ctx,
                        &mut session,
                        &tokens[..prefix],
                        out,
                        setup,
                        "matched_incumbent_prefix_excluded_from_target",
                    );
                }
                eprintln!("GLM MLA tail {stream} prefix{prefix} N{rows} {label}");
                let mut outputs = vec![prefill(
                    ctx,
                    &mut session,
                    &tokens[prefix..prefix + rows],
                    out,
                    c,
                    "whole_target",
                )];
                for step in 1..=steps {
                    let token = tokens[prefix + rows + step - 1];
                    let result = session.forward(ctx, token);
                    record(
                        out,
                        "continuation",
                        c,
                        json!({"step":step,"token":token,"position":session.position(),
                        "error":result.as_ref().err().map(ToString::to_string)}),
                    );
                    let logits = result.expect("teacher-forced handoff failed; record flushed");
                    logits_summary(out, c, step, &logits);
                    assert_eq!(session.position(), prefix + rows + step);
                    outputs.push(logits);
                }
                if let Some(a) = reference.as_ref() {
                    compare_logits(out, c, reference_label, a, &outputs);
                }
                if matches!(label, "B1" | "A1") {
                    reference = Some(outputs);
                    reference_label = label;
                }
            }
            emit(
                out,
                json!({"event":"cell_complete","stream":stream,"prefix":prefix,"rows":rows,
                "complete":complete,"decision":"unscored_measurement_no_promotion"}),
            );
        }
    }
}

#[test]
fn mla_tail_timestamp_scope_and_expected_counts() {
    for (start, end) in [
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 2.0),
        (1.0, f64::INFINITY),
    ] {
        assert_eq!(milliseconds(start, end), None);
    }
    record_completed_command(|| panic!("inactive timestamp query"));
    let ((), t) = observe(|| {
        record_completed_command(|| (1.0, 1.25));
        assert!(std::panic::catch_unwind(|| observe(|| panic!("restore nested scope"))).is_err());
        record_completed_command(|| (0.0, 0.0));
    });
    assert_eq!((t.completed, t.valid, t.gpu_ms), (2, 1, 250.0));
    record_completed_command(|| panic!("scope leaked"));
    for (n, expected) in [
        (7, 0),
        (8, 22),
        (127, 22),
        (128, 0),
        (129, 0),
        (135, 0),
        (136, 22),
        (255, 22),
        (512, 0),
    ] {
        assert_eq!(expected_substitutions(n, 512, true), expected);
        assert_eq!(expected_substitutions(n, 512, false), 0);
    }
    assert_eq!(expected_substitutions(4096 + 127, 512, true), 22);
    assert_eq!(expected_substitutions(254, 127, true), 44);
    let mut diff = Difference::default();
    diff.add(f32::MAX, -f32::MAX);
    assert!(diff.max_abs.is_finite() && diff.error2.is_finite() && diff.norm2.is_finite());
    assert_eq!(diff.json()["relative_l2"], json!(2.0));
}

#[test]
#[ignore = "released GLM fixture; production lease; new GLM53_MLA_TAIL_OUT; optional GLM53_MLA_TAIL_FULL=1"]
fn mla_tail_packet() {
    assert!(!cfg!(debug_assertions), "packet requires --release");
    let full = match std::env::var("GLM53_MLA_TAIL_FULL").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("") | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("GLM53_MLA_TAIL_FULL must be absent, 0 or 1"),
    };
    let _lease = perf_lease();
    let mut out = File::options()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("GLM53_MLA_TAIL_OUT").expect("GLM53_MLA_TAIL_OUT"))
        .expect("new output file required");
    let ctx = MetalContext::new().unwrap();
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    preflight_session(
        &ctx,
        &gguf,
        artifact.model(),
        if full { 516 } else { 128 },
        if full { 512 } else { 128 },
    )
    .unwrap();
    let texts = [
        ("qualification", long_qualification_text()),
        (
            "code_review",
            format!(
                "[gMASK]<sop>Review this Rust inference backend for a prefill or cache-state bug. Explain the likely failure and suggest a focused fix.\n\n{CODE}\n\nRelated prompt-rendering implementation:\n\n{RENDER}"
            ),
        ),
    ];
    let streams: Vec<_> = texts
        .iter()
        .take(if full { 2 } else { 1 })
        .map(|(name, text)| {
            let limit = if full { 4227 } else { 128 };
            let tokens: Vec<u32> = artifact
                .tokenizer()
                .encode(text, false)
                .unwrap()
                .into_iter()
                .take(limit)
                .map(|id| u32::try_from(id).unwrap())
                .collect();
            assert_eq!(tokens.len(), limit, "natural stream too short");
            (*name, tokens)
        })
        .collect();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mla: Vec<usize> = weights
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| matches!(&b.mixer, MixerTensors::Mla(_)).then_some(i))
        .collect();
    assert_eq!(mla, (3..44).step_by(4).collect::<Vec<_>>());
    assert_eq!(
        (
            weights.blocks.len(),
            weights.config.head_count,
            weights.config.mla_key_head_dim,
            weights.config.mla_value_head_dim,
            weights.config.kv_lora_rank
        ),
        (45, 64, 256, 256, 512)
    );
    let stamps = gguf.revalidate_retained_shard_stamps().unwrap();
    let shards:Vec<Value>=stamps.iter().map(|s|json!({"path":s.path,"shard":s.shard_idx,"device":s.device,"inode":s.inode,"bytes":s.size,
        "mtime_sec":s.mtime_sec,"mtime_nsec":s.mtime_nsec,"ctime_sec":s.ctime_sec,"ctime_nsec":s.ctime_nsec})).collect();
    let banks:Vec<Value>=["blk.43.attn_k_b.weight","blk.43.attn_v_b.weight"].into_iter().map(|name| {
        let t=gguf.tensors.iter().find(|t|t.name==name).expect("MLA bank");assert_eq!(t.dtype,GgmlType::Q8_0);
        json!({"name":name,"shape":t.shape,"dtype":"Q8_0","shard":t.shard_idx,"offset":t.data_offset,"bytes":t.n_bytes,"sha256":sha(gguf.slice(t))})
    }).collect();
    emit(
        &mut out,
        json!({"event":"header","schema":"glm53.mla_tail.v1","mode":if full {"whole"} else {"leaf"},
        "decision":"unscored_measurement_no_promotion","device":ctx.describe(),"fixture":crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.id,
        "shards":shards,"final_mla_banks":banks,"mla_layers":mla,
        "leaf_widths":[8,32,64,127,128],"whole_widths":[8,32,64,127,128,129,255,512],
        "scheduled_work":{"leaf_setup_prefills":5,"leaf_commands_including_warmups":60,"whole_target_prefills_including_warmups":108,"whole_prefix_setups_if_admitted":12},
        "expected_substitutions":"A=0; B=22 per chunk with rows%128>=8 (leaf=1 per affected call); N128/N129/N512 are unchanged controls",
        "whole_deep":{"prefix":4096,"suffix":127,"packed_rows":512,"continuation":4,"capacity":4227,"prefix_policy":"forced incumbent in both arms; timed separately as setup"},
        "order":["warm_A","warm_B","B1","A1","A2","B2"],"arm_policy":"A Some(false), B Some(true); only MLA tail changes; production routers and IQ3 down remain enabled",
        "comparisons":"A1 versus retained B1, then A2 repeat and B2 candidate versus A1; hashes and finite f64 norms, KL/regret for logits; no bit gate",
        "leaf_method":"actual final block43 scratch after incumbent prefill; block44 KDA does not write MLA buffers; each whole absorb_rows sequence in one command; directions independent, attention not replayed; no GPU capture copies",
        "leaf_memory":"one CPU reference touched before warmup, replaced B1->A1 in place; 1 MiB readback chunks; session admission includes reference plus 16 MiB metadata allowance",
        "whole_method":"fresh admitted sessions; prefill GPU/wall only; allocation, matched prefix, diagnostics and teacher-forced steps excluded from target; no leaf work in FULL mode",
        "limits":"leaf cache-hot replay is not all-layer attribution; GPU includes stalls; wall minus GPU does not isolate CPU/wiring; deep admission refusal recorded, invalid timing fails after raw record",
        "source_binding":{"packet":sha(include_bytes!("mla_tail.rs")),"packed":sha(include_bytes!("../packed.rs")),
            "session":sha(include_bytes!("../../glm5_next_metal.rs")),"helpers":sha(include_bytes!("../tests.rs")),
            "latent":sha(include_bytes!("../../metal/latent.rs")),"mat_vec":sha(include_bytes!("../../metal/mat_vec.rs")),
            "gemm_metal":sha(include_bytes!("../../../../../kernels/mat_mat_q8_0.metal")),"gemv_metal":sha(include_bytes!("../../../../../kernels/mat_vec_q8_0.metal")),
            "expert":sha(include_bytes!("../../metal/expert.rs")),"expert_retile":sha(include_bytes!("../../metal/iq3_s_down_retile.rs")),
            "code":sha(CODE.as_bytes()),"render":sha(RENDER.as_bytes()),"metallib":sha(crate::KERNELS_METALLIB)}}),
    );
    for ((name, tokens), (_, text)) in streams.iter().zip(&texts) {
        emit(
            &mut out,
            json!({"event":"stream","name":name,"text_sha256":sha(text.as_bytes()),"token_ids":tokens,"token_ids_sha256_u32_le":sha(bytemuck::cast_slice(tokens))}),
        );
    }
    if full {
        whole(&ctx, &weights, &streams, &mut out);
    } else {
        leaf(&ctx, &weights, &streams[0].1, &mut out);
    }
    assert_eq!(stamps, gguf.revalidate_retained_shard_stamps().unwrap());
    emit(
        &mut out,
        json!({"event":"complete","decision":"unscored_measurement_no_promotion"}),
    );
}

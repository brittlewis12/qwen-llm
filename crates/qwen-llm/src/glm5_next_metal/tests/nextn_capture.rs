//! Private native target conditions for a head-only reference probe, not NextN execution.
use super::*;
use crate::glm5_next::admission::Glm5NextPreparedArtifact;
use crate::glm5_next_chat::{self as chat, Message, RenderOptions};
use crate::sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig};
use anyhow::{Context, Result as CaptureResult, ensure};
use sha2::{Digest, Sha256};
use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;

const ROWS: usize = 512;
const FILES: [&str; 4] = [
    "tokens.u32",
    "hidden.f32",
    "hidden-raw.f32",
    "target-logits.f32",
];

fn private_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

fn lengths(s: usize, t: usize, h: usize, v: usize) -> CaptureResult<[u64; 4]> {
    ensure!(s > 0 && h > 0 && v > 0, "empty prompt or model dimensions");
    let n = s.checked_add(t).context("token count overflow")? as u64;
    let bytes = |rows: u64, width: usize| {
        rows.checked_mul(width as u64)
            .and_then(|n| n.checked_mul(4))
            .context("file byte count overflow")
    };
    Ok([
        bytes(n, 1)?,
        bytes(n, h)?,
        bytes(n, h)?,
        bytes(t as u64, v)?,
    ])
}

fn carry(token: i32, vocab: usize, stops: &[i32]) -> CaptureResult<Option<u32>> {
    ensure!(
        token >= 0 && (token as usize) < vocab,
        "sample outside target vocabulary"
    );
    Ok((!stops.contains(&token)).then_some(token as u32))
}

fn write_floats(file: &mut impl Write, values: &[f32]) -> CaptureResult<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite exported features/logits"
    );
    for v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn executable_sha256() -> CaptureResult<String> {
    let mut file = File::open(std::env::current_exe()?)?;
    let mut hash = Sha256::new();
    let mut bytes = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn export_rows(
    ctx: &MetalContext,
    session: &Glm5NextSession<'_>,
    captured: &Glm5NextCapturedRows<'_>,
    normalized: &MetalTensor,
    files: &mut [BufWriter<File>],
    tokens: &[u32],
    start: usize,
    scored: bool,
) -> CaptureResult<Vec<f32>> {
    let w = session.weights;
    let (h, v, n) = (
        w.config.hidden_size as usize,
        w.config.vocab_size as usize,
        tokens.len(),
    );
    ensure!(
        n > 0 && n <= ROWS && (!scored || n == 1),
        "capture chunk bounds"
    );
    ensure!(
        captured.input_positions() == (start..start + n)
            && captured.feature_positions() == (start..start + n)
            && captured.logit_positions() == (start + n - 1..start + n)
            && captured.taps() == [44]
            && session.position() == start + n,
        "capture alignment"
    );
    let raw = captured.features().context("missing features")?;
    ensure!(
        raw.shape == [h as u64, 1, n as u64],
        "capture feature shape"
    );
    let command = ctx.queue.commandBuffer().context("command buffer")?;
    let enc = KernelEncoder::begin(&command);
    for row in 0..n {
        // Use the actual scalar head's norm kernel for each row, not a CPU norm
        // or an alternate reduction. Only H*512 extra GPU elements are retained.
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &raw.view_subrange((row * h) as u64, vec![h as u64]),
            &w.output_norm,
            &normalized.view_subrange((row * h) as u64, vec![h as u64]),
            w.config.rms_epsilon,
        )?;
    }
    enc.end();
    command.commit();
    wait_completed(&command)?;
    let last = read_f32(&normalized.view_subrange(((n - 1) * h) as u64, vec![h as u64]))?;
    let actual_head_input = read_f32(&session.s.final_normed)?;
    ensure!(
        max_abs_diff(&last, &actual_head_input) == 0.0,
        "last capture norm differs from the actual native head input"
    );
    for (row, token) in tokens.iter().enumerate() {
        files[0].write_all(&token.to_le_bytes())?;
        write_floats(
            &mut files[2],
            &read_f32(&raw.view_subrange((row * h) as u64, vec![h as u64]))?,
        )?;
        write_floats(
            &mut files[1],
            &read_f32(&normalized.view_subrange((row * h) as u64, vec![h as u64]))?,
        )?;
    }
    let logits = captured.logits().context("missing final logits")?;
    ensure!(logits.shape == [v as u64, 1], "capture logit shape");
    let logits = read_f32(logits)?;
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite target logits"
    );
    if scored {
        write_floats(&mut files[3], &logits)?;
    }
    // Preserve every completed chunk on later failure, before bank reuse.
    for file in files {
        file.flush()?;
    }
    Ok(logits)
}

#[test]
#[ignore = "private native target export: GLM53_GGUF, QWEN_NEXTN_MESSAGES, new QWEN_NEXTN_EXPORT_DIR; Metal validation and production lease"]
fn export_native_nextn_capture() -> CaptureResult<()> {
    ensure!(
        std::env::var("MTL_DEBUG_LAYER").as_deref() == Ok("1"),
        "require Metal validation"
    );
    let source =
        PathBuf::from(std::env::var_os("QWEN_NEXTN_MESSAGES").context("QWEN_NEXTN_MESSAGES")?);
    let dir =
        PathBuf::from(std::env::var_os("QWEN_NEXTN_EXPORT_DIR").context("QWEN_NEXTN_EXPORT_DIR")?);
    let model = PathBuf::from(std::env::var_os("GLM53_GGUF").context("GLM53_GGUF")?);
    let limit: usize = std::env::var("QWEN_NEXTN_SCORE_TOKENS")
        .unwrap_or_else(|_| "128".into())
        .parse()?;
    ensure!(limit > 0, "score token limit must be positive");
    let document = std::fs::read(&source)?;
    let input_sha256 = format!("{:x}", Sha256::digest(&document));
    let mut chat_input = chat::parse_document(&document)?;
    let mut missing = 0;
    // Match qwen-cli/qwen/glm5_next.rs::chat_messages, including history policy.
    for message in &mut chat_input.messages {
        if let Message::Assistant { reasoning, .. } = message {
            if reasoning.is_none() {
                *reasoning = Some(String::new());
                missing += 1;
            }
        }
    }
    let effort = chat::Effort::parse(None)?;
    let clear = chat_input.clear_thinking.unwrap_or(false);
    let prompt = chat::render_with_tools(
        &chat_input.messages,
        &chat_input.tools,
        RenderOptions::generate(effort, clear),
    )?;
    let gguf = GgufFile::open(&model)?;
    let prepared = Glm5NextPreparedArtifact::inspect(&gguf)?;
    let chat_profile = prepared.chat_profile()?;
    let stops = prepared.generation_stops()?;
    let c = prepared.config();
    ensure!(
        prepared.packed_prefill() && c.blocks.len() == 45 && c.hc_streams == 4,
        "requires packed GLM target with final completed block tap 44 and four streams"
    );
    let (h, v) = (c.hidden_size as usize, c.vocab_size as usize);
    let tokens: Vec<u32> = prepared
        .tokenizer()
        .encode(&prompt, false)?
        .into_iter()
        .map(|id| {
            ensure!(
                id >= 0 && (id as usize) < v,
                "prompt token outside vocabulary"
            );
            Ok(id as u32)
        })
        .collect::<CaptureResult<_>>()?;
    let s = tokens.len();
    lengths(s, limit, h, v)?;
    let capacity = s.checked_add(limit).context("context overflow")?.max(ROWS);
    ensure!(
        capacity <= c.context_length as usize,
        "request exceeds native context"
    );
    let binary_sha256 = executable_sha256()?;
    // Atomic mkdir/create_new: never reuse a directory or overwrite an artifact.
    DirBuilder::new().mode(0o700).create(&dir)?;
    let mut files = FILES
        .iter()
        .map(|name| private_file(&dir.join(name)).map(BufWriter::new))
        .collect::<std::io::Result<Vec<_>>>()?;
    let _lease = crate::metal::acquire_metal_benchmark_lease()?;
    let ctx = MetalContext::new()?;
    let weights = Glm5NextWeights::load(&ctx, &gguf)?;
    // Future CPU storage: sampler candidates/sort/weights + current logits,
    // small readbacks, hash/JSON/IO reserve. Prompt/tokenizer are already resident.
    let cpu_reserve = (v as u64)
        .checked_mul(64)
        .and_then(|n| n.checked_add(h as u64 * 16))
        .and_then(|n| n.checked_add(16 << 20))
        .context("CPU reserve overflow")?;
    let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
        &ctx,
        &weights,
        capacity,
        ROWS,
        cpu_reserve,
    )?;
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[44], ROWS, 1)?,
    )?;
    let specs = [BufferSpec {
        name: "normalized",
        dtype: BufferType::F32,
        shape: vec![h as u64, ROWS as u64],
        fill: 0.0,
    }];
    let normalized_bytes = memory::priced(&specs, &device_price(&ctx))?;
    let normalized = {
        let _allocation = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            normalized_bytes,
            cpu_reserve,
            memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        ensure!(
            admission.is_admitted(),
            "normalization scratch admission refused: {:?}",
            admission.reason
        );
        let mut buffers = SpecBuffers::allocate(&ctx, "nextn_capture", &specs)?;
        let normalized = buffers.take("normalized")?;
        buffers.finish()?;
        normalized
    };
    let mut logits = Vec::new();
    for (chunk, input) in tokens.chunks(ROWS).enumerate() {
        let captured = session.prefill_with_captures(&ctx, input, &mut bank)?;
        logits = export_rows(
            &ctx,
            &session,
            &captured,
            &normalized,
            &mut files,
            input,
            chunk * ROWS,
            false,
        )?;
    }
    let sampling = SamplingConfig::glm5_next(42);
    let mut sampler = Sampler::new(sampling)?;
    let (mut t, mut stop_token) = (0, None);
    while t < limit {
        let sampled = sampler.sample(&logits)?.token;
        let Some(token) = carry(sampled, v, &stops)? else {
            stop_token = Some(sampled);
            break;
        };
        let captured = session.forward_with_captures(&ctx, token, &mut bank)?;
        logits = export_rows(
            &ctx,
            &session,
            &captured,
            &normalized,
            &mut files,
            &[token],
            s + t,
            true,
        )?;
        t += 1;
    }
    let expected = lengths(s, t, h, v)?;
    for (file, bytes) in files.iter_mut().zip(expected) {
        file.flush()?;
        ensure!(
            file.get_ref().metadata()?.len() == bytes,
            "export byte length mismatch"
        );
        file.get_ref().sync_all()?;
    }
    let manifest = serde_json::json!({
        "schema":"qwen.glm-nextn-capture.v1", "status":if t > 0 {"complete"} else {"no_evaluable_rows"},
        "hidden_size":h, "vocab_size":v, "token_count":s+t, "score_start":s, "score_count":t,
        "prompt_count":s, "message_count":chat_input.messages.len(), "requested_score_tokens":limit, "byte_order":"little",
        "finish_reason":if stop_token.is_some() {"eos"} else {"score_token_limit"}, "stop_token":stop_token,
        "sampling":{"temperature":sampling.temperature,"top_k":sampling.top_k,"top_p":sampling.top_p,
            "min_p":sampling.min_p,"seed":sampling.seed,"draws":sampler.draws(),"algorithm_version":SAMPLER_ALGORITHM_VERSION},
        "native_profile":{"chat":chat_profile,"effort":effort,"clear_thinking":clear,"history_reasoning_missing":missing,
            "packed_rows":ROWS,"lineage":format!("{:?}",session.packed_lineage()),"precision":format!("{:?}",session.effective_fast_precision()),"stops":stops},
        "conditioning":"hidden[i-1] + tokens[i]; position 0 uses zero hidden; scored target logits follow consuming i, predict i+1",
        "features":"tap 44 completed four-stream mean; hidden.f32 applies native output RMSNorm and output_norm",
        "witness":"every chunk/scalar last normalized row equals the actual native final_normed head input (GPU/GPU)",
        "memory":{"session_peak_bytes":session.ledger.peak_bytes(),"bank_priced_bytes":bank.priced_bytes(),"normalization_priced_bytes":normalized_bytes,"future_cpu_reserve_bytes":cpu_reserve},
        "files":FILES.into_iter().zip(expected).map(|(name,bytes)|(name.to_owned(),serde_json::json!({"bytes":bytes}))).collect::<serde_json::Map<_,_>>(),
        "provenance":{"messages_sha256":input_sha256,"model_path":model,"model_entry_bytes":std::fs::metadata(&model)?.len(),
            "executable_sha256":binary_sha256,"package_version":env!("CARGO_PKG_VERSION"),"debug_assertions":cfg!(debug_assertions),
            "capture_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("nextn_capture.rs"))),"git_commit":null}
    });
    // No success manifest on failure; keep flushed partial private files.
    let pending = dir.join("manifest.pending.json");
    let mut manifest_file = private_file(&pending)?;
    serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
    manifest_file.sync_all()?;
    std::fs::hard_link(&pending, dir.join("manifest.json"))?;
    std::fs::remove_file(pending)?;
    Ok(())
}

#[test]
fn nextn_capture_cpu_alignment_and_stops() {
    assert_eq!(lengths(14, 3, 4, 8).unwrap(), [68, 272, 272, 96]);
    assert_eq!(lengths(14, 0, 4, 8).unwrap(), [56, 224, 224, 0]);
    assert!(lengths(usize::MAX, 1, 4, 8).is_err());
    assert!(lengths(0, 1, 4, 8).is_err());
    assert_eq!(carry(7, 8, &[7]).unwrap(), None);
    assert_eq!(carry(6, 8, &[7]).unwrap(), Some(6));
    assert!(carry(-1, 8, &[7]).is_err());
    assert!(carry(8, 8, &[7]).is_err());
    for (samples, limit, expected) in [
        (&[7, 6][..], 2, 0),
        (&[6, 7][..], 2, 1),
        (&[6, 5][..], 1, 1),
    ] {
        let mut consumed = 0;
        for &sample in samples.iter().take(limit) {
            if carry(sample, 8, &[7]).unwrap().is_none() {
                break;
            }
            consumed += 1;
        }
        assert_eq!(consumed, expected);
        assert_eq!(
            lengths(14, consumed, 4, 8).unwrap()[3],
            consumed as u64 * 8 * 4
        );
    }
    let mut bytes = Vec::new();
    write_floats(&mut bytes, &[1.0, -2.0]).unwrap();
    assert_eq!(
        bytes,
        [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()
    );
}

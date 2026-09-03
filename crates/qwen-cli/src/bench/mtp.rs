//! MTP speculative decode bench and target-state audit.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpTargetStateAudit {
    pub(crate) resume_audit_pass: bool,
    pub(crate) kv_position_equal: bool,
    pub(crate) kv_payload_exact: bool,
    pub(crate) kv_payload_max_abs: f32,
    pub(crate) kv_payload_cosine: f64,
    pub(crate) reference_final_position: Option<usize>,
    pub(crate) candidate_final_position: Option<usize>,
    pub(crate) gdn_state_max_abs: f32,
    pub(crate) gdn_conv_max_abs: f32,
    pub(crate) continuation_argmax_equal: bool,
    pub(crate) continuation_token: i32,
    pub(crate) continuation_logits_max_abs: f32,
    pub(crate) continuation_logits_cosine: f64,
}

pub(crate) fn max_abs_f32_pair(a: &MetalTensor, b: &MetalTensor) -> Result<f32> {
    anyhow::ensure!(
        a.dtype == GgmlType::F32
            && b.dtype == GgmlType::F32
            && a.shape == b.shape
            && a.n_bytes() == b.n_bytes(),
        "state tensor shape or dtype mismatch"
    );
    let mut max_abs = 0.0f32;
    unsafe {
        let pa = (a.buffer.contents().as_ptr() as *const u8).add(a.offset as usize) as *const f32;
        let pb = (b.buffer.contents().as_ptr() as *const u8).add(b.offset as usize) as *const f32;
        for i in 0..a.n_elements() as usize {
            let delta = (*pa.add(i) - *pb.add(i)).abs();
            if !delta.is_finite() {
                return Ok(f32::INFINITY);
            }
            max_abs = max_abs.max(delta);
        }
    }
    Ok(max_abs)
}

pub(crate) fn max_abs_f32_tensor_pairs(
    reference: &[MetalTensor],
    candidate: &[MetalTensor],
) -> Result<f32> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "state tensor count mismatch"
    );
    let mut max_abs = 0.0f32;
    for (a, b) in reference.iter().zip(candidate) {
        max_abs = max_abs.max(max_abs_f32_pair(a, b)?);
    }
    Ok(max_abs)
}

/// Per-layer max-abs across two tensor lists: returns (global max, layer
/// index of the max, count of layers with max-abs > 1e-6).
pub(crate) fn max_abs_f32_layers(
    reference: &[MetalTensor],
    candidate: &[MetalTensor],
) -> Result<(f32, usize, usize)> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "state tensor count mismatch"
    );
    let mut max_abs = 0.0f32;
    let mut max_layer = 0usize;
    let mut n_over = 0usize;
    for (li, (a, b)) in reference.iter().zip(candidate).enumerate() {
        let m = max_abs_f32_pair(a, b)?;
        if m > 1e-6 {
            n_over += 1;
        }
        if m > max_abs {
            max_abs = m;
            max_layer = li;
        }
    }
    Ok((max_abs, max_layer, n_over))
}

pub(crate) fn kv_bytes_metrics(
    reference: &[u8],
    candidate: &[u8],
    dtype: GgmlType,
) -> Result<(f32, f64)> {
    anyhow::ensure!(
        reference.len() == candidate.len(),
        "KV byte length mismatch"
    );
    let values: Box<dyn Iterator<Item = (f32, f32)> + '_> = match dtype {
        GgmlType::F16 => Box::new(
            reference
                .chunks_exact(2)
                .zip(candidate.chunks_exact(2))
                .map(|(a, b)| {
                    let a = half::f16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                    let b = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                    (a, b)
                }),
        ),
        GgmlType::F32 => Box::new(
            reference
                .chunks_exact(4)
                .zip(candidate.chunks_exact(4))
                .map(|(a, b)| {
                    let a = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                    let b = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (a, b)
                }),
        ),
        other => anyhow::bail!("unsupported KV audit dtype {other:?}"),
    };
    let (mut max_abs, mut dot, mut reference_norm, mut candidate_norm) =
        (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in values {
        max_abs = max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        reference_norm += (a as f64).powi(2);
        candidate_norm += (b as f64).powi(2);
    }
    let cosine = dot / (reference_norm.sqrt() * candidate_norm.sqrt() + f64::MIN_POSITIVE);
    Ok((max_abs, cosine))
}

pub(crate) fn audit_mtp_target_state(
    forward: &MetalForward<'_>,
    reference: &mut MetalSession,
    candidate: &mut MetalSession,
    expected_position: usize,
    pending_terminal_token: i32,
) -> Result<MtpTargetStateAudit> {
    let reference_position_valid = reference
        .kv_n_pos
        .iter()
        .all(|&position| position == expected_position);
    let candidate_position_valid = candidate
        .kv_n_pos
        .iter()
        .all(|&position| position == expected_position);
    let kv_position_equal = reference_position_valid
        && candidate_position_valid
        && reference.kv_n_pos == candidate.kv_n_pos;
    let reference_final_position = reference.kv_n_pos.first().copied();
    let candidate_final_position = candidate.kv_n_pos.first().copied();
    let gdn_state_max_abs = max_abs_f32_tensor_pairs(&reference.gdn_state, &candidate.gdn_state)?;
    let gdn_conv_max_abs = max_abs_f32_tensor_pairs(&reference.gdn_conv, &candidate.gdn_conv)?;
    let snapshot_identity = reference.snapshot_identity(0, 0);
    let reference_snapshot = reference
        .snapshot(snapshot_identity.clone(), vec![0; expected_position], None)
        .context("snapshot reference target state")?;
    let candidate_snapshot = candidate
        .snapshot(snapshot_identity, vec![0; expected_position], None)
        .context("snapshot candidate target state")?;
    let kv_payload_exact = reference_snapshot.kv_k_arena == candidate_snapshot.kv_k_arena
        && reference_snapshot.kv_v_arena == candidate_snapshot.kv_v_arena;
    let kv_dtype = reference
        .kv_k
        .first()
        .map(|tensor| tensor.dtype)
        .context("target state has no KV layers")?;
    anyhow::ensure!(
        candidate.kv_k.first().map(|tensor| tensor.dtype) == Some(kv_dtype),
        "candidate KV dtype mismatch"
    );
    let (kv_k_max_abs, kv_k_cosine) = kv_bytes_metrics(
        &reference_snapshot.kv_k_arena,
        &candidate_snapshot.kv_k_arena,
        kv_dtype,
    )?;
    let (kv_v_max_abs, kv_v_cosine) = kv_bytes_metrics(
        &reference_snapshot.kv_v_arena,
        &candidate_snapshot.kv_v_arena,
        kv_dtype,
    )?;
    let kv_payload_max_abs = kv_k_max_abs.max(kv_v_max_abs);
    let kv_payload_cosine = kv_k_cosine.min(kv_v_cosine);
    let reference_logits =
        forward.single_token(pending_terminal_token, expected_position as u32, reference)?;
    let candidate_logits =
        forward.single_token(pending_terminal_token, expected_position as u32, candidate)?;
    anyhow::ensure!(
        reference_logits.len() == candidate_logits.len(),
        "continuation logits length mismatch"
    );
    let mut continuation_logits_max_abs = 0.0f32;
    let (mut dot, mut reference_norm, mut candidate_norm) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in reference_logits.iter().zip(&candidate_logits) {
        continuation_logits_max_abs = continuation_logits_max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        reference_norm += (a as f64).powi(2);
        candidate_norm += (b as f64).powi(2);
    }
    let continuation_logits_cosine =
        dot / (reference_norm.sqrt() * candidate_norm.sqrt() + f64::MIN_POSITIVE);
    let continuation_argmax_equal = argmax_i32(&reference_logits) == argmax_i32(&candidate_logits);
    let continuation_token = argmax_i32(&reference_logits);
    let resume_audit_pass = kv_position_equal
        && kv_payload_cosine >= 0.99999
        && continuation_argmax_equal
        && continuation_logits_max_abs <= 5e-2
        && continuation_logits_cosine >= 0.99999;
    Ok(MtpTargetStateAudit {
        resume_audit_pass,
        kv_position_equal,
        kv_payload_exact,
        kv_payload_max_abs,
        kv_payload_cosine,
        reference_final_position,
        candidate_final_position,
        gdn_state_max_abs,
        gdn_conv_max_abs,
        continuation_argmax_equal,
        continuation_token,
        continuation_logits_max_abs,
        continuation_logits_cosine,
    })
}

pub(crate) fn merge_target_state_audit(
    aggregate: &mut MtpTargetStateAudit,
    next: MtpTargetStateAudit,
) {
    aggregate.resume_audit_pass &= next.resume_audit_pass;
    aggregate.kv_position_equal &= next.kv_position_equal;
    aggregate.kv_payload_exact &= next.kv_payload_exact;
    aggregate.kv_payload_max_abs = aggregate.kv_payload_max_abs.max(next.kv_payload_max_abs);
    aggregate.kv_payload_cosine = aggregate.kv_payload_cosine.min(next.kv_payload_cosine);
    aggregate.reference_final_position = next.reference_final_position;
    aggregate.candidate_final_position = next.candidate_final_position;
    aggregate.gdn_state_max_abs = aggregate.gdn_state_max_abs.max(next.gdn_state_max_abs);
    aggregate.gdn_conv_max_abs = aggregate.gdn_conv_max_abs.max(next.gdn_conv_max_abs);
    aggregate.continuation_argmax_equal &= next.continuation_argmax_equal;
    aggregate.continuation_token = next.continuation_token;
    aggregate.continuation_logits_max_abs = aggregate
        .continuation_logits_max_abs
        .max(next.continuation_logits_max_abs);
    aggregate.continuation_logits_cosine = aggregate
        .continuation_logits_cosine
        .min(next.continuation_logits_cosine);
}

pub(crate) const MTP_CONTINUATION_AUDIT_STEPS: usize = 16;

pub(crate) fn audit_mtp_target_state_chain(
    forward: &MetalForward<'_>,
    reference: &mut MetalSession,
    candidate: &mut MetalSession,
    expected_position: usize,
    pending_terminal_token: i32,
) -> Result<MtpTargetStateAudit> {
    let mut audit = audit_mtp_target_state(
        forward,
        reference,
        candidate,
        expected_position,
        pending_terminal_token,
    )?;
    for offset in 1..MTP_CONTINUATION_AUDIT_STEPS {
        let next_token = audit.continuation_token;
        let next = audit_mtp_target_state(
            forward,
            reference,
            candidate,
            expected_position + offset,
            next_token,
        )?;
        merge_target_state_audit(&mut audit, next);
    }
    audit.resume_audit_pass &= audit.gdn_state_max_abs <= 1e-2;
    audit.resume_audit_pass &= audit.gdn_conv_max_abs <= 1e-1;
    Ok(audit)
}

pub(crate) fn run_mtp(args: MtpArgs) -> Result<()> {
    let MtpArgs {
        model,
        prompt,
        qwen_chat,
        system,
        disable_thinking,
        spec_tokens,
        mtp_probe,
        mtp_physical_n,
        mtp_single_cb_draft,
        mtp_draft_token_embd_head,
        mtp_draft_lm_head_q4_1,
        mtp_draft_lm_head_q4_0,
        mtp_draft_lm_head_q4_affine64,
        mtp_recursive_hidden,
        mtp_base_hidden,
        mtp_history,
        mtp_rank_topk,
        mtp_state_trace,
        output,
        include_token_ids,
        tokens,
        stop_tokens,
        no_warmup,
    } = args;

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[mtp-bench] device: {}", ctx.describe());
    let spec_token_limit = 15;
    if spec_tokens == 0 || spec_tokens > spec_token_limit {
        anyhow::bail!(
            "`--spec-tokens` must be in 1..={spec_token_limit} for probe {:?}",
            mtp_probe
        );
    }
    let use_packed_verify = spec_tokens >= 2 || mtp_physical_n.is_some();
    if mtp_rank_topk.is_some() {
        if spec_tokens == 1 {
            anyhow::bail!("--mtp-rank-topk requires --spec-tokens 2 or higher");
        }
        if mtp_probe != MtpProbeMode::Normal {
            anyhow::bail!("--mtp-rank-topk is only supported with --mtp-probe normal");
        }
    }
    let draft_lm_head_override_count = usize::from(mtp_draft_token_embd_head)
        + usize::from(mtp_draft_lm_head_q4_1)
        + usize::from(mtp_draft_lm_head_q4_0)
        + usize::from(mtp_draft_lm_head_q4_affine64);
    if draft_lm_head_override_count > 1 {
        anyhow::bail!("choose at most one draft lm_head override");
    }
    let planned_verify_n = mtp_physical_n.unwrap_or(spec_tokens + 1);
    if !(spec_tokens + 1..=16).contains(&planned_verify_n) {
        anyhow::bail!(
            "--mtp-physical-n must be in {}..=16 for --spec-tokens {spec_tokens}",
            spec_tokens + 1
        );
    }
    let effective_logical_verify_n = use_packed_verify.then_some(spec_tokens + 1);
    let effective_physical_verify_n = use_packed_verify.then_some(planned_verify_n);
    let packed_base_prefill = env_flag_default_on("QWEN_MTP_PACKED_BASE_PREFILL");
    let q5_k_n2_seq = env_flag_default_on("QWEN_MATMAT_Q5_K_N2_SEQ");
    let iq2_s_n2_nc2 = env_flag_default_on("QWEN_MATMAT_IQ2_S_N2_NC2");
    let iq3_s_n2_nc2 = env_flag_default_on("QWEN_MATMAT_IQ3_S_N2_NC2");
    let skip_final_checkpoint = env_flag_default_on("QWEN_MTP_SKIP_FINAL_CKPT");
    let direct_mtp_f32_destination = env_flag_default_on("QWEN_MTP_DIRECT_F32_DEST");
    let shared_kv_q2_requested = env_flag_enabled("QWEN_MTP_ATTN_Q2_SHARED_KV");
    let build_identity = recorded_build_identity();
    let qwen_env = capture_qwen_env();

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let stops = resolve_stop_tokens(&g, stop_tokens)?;
    let m = Model::from_gguf(&g).context("parse model arch")?;
    let mtp_view = m.mtp.as_ref().ok_or_else(|| {
        anyhow!(
            "GGUF has no MTP head (\
             use brittlewis12/Qwen3.6-27B-MTP-GGUF or the 0.8B-MTP variant)"
        )
    })?;

    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load weights")?;
    let mtp_load_start = Instant::now();
    let mtp_head = MetalMtpHead::load(&ctx, &g, mtp_view).context("metal-load MTP head")?;
    let mtp_load_ms = mtp_load_start.elapsed().as_secs_f64() * 1e3;
    let mtp_moe_bank_ledger = mtp_head.attn.ffn_moe.as_ref().map(|moe| {
        (
            [moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype],
            moe.gate_exps.n_bytes() + moe.up_exps.n_bytes() + moe.down_exps.n_bytes(),
        )
    });
    let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;

    if !qwen_chat && (system.is_some() || disable_thinking) {
        anyhow::bail!("`--system` and `--disable-thinking` require `--qwen-chat`");
    }
    let rendered_prompt = if qwen_chat {
        messages::render_qwen_single_turn_prompt(
            &prompt,
            system.as_deref().filter(|system| !system.is_empty()),
            if disable_thinking {
                messages::QwenGenerationMode::NoThinking
            } else {
                messages::QwenGenerationMode::Thinking
            },
        )
    } else {
        prompt.clone()
    };
    let prompt_ids = tok
        .encode(&rendered_prompt, false)
        .context("tokenize prompt")?;
    let prompt_token_sha256 = token_ids_sha256_i32le(&prompt_ids);
    eprintln!(
        concat!(
            "[mtp-bench] model={} prompt={:?} rendered_mode={} thinking={} ",
            "spec_tokens={} probe={:?} physical_n={} single_cb_draft={} ",
            "draft_token_embd_head={} draft_lm_head_q4_1={} ",
            "draft_lm_head_q4_0={} draft_lm_head_q4_affine64={} ",
            "base_hidden={:?} recursive_hidden={:?} mtp_history={:?} ",
            "({} tokens) gen={} stop_tokens={:?}"
        ),
        model.display(),
        prompt,
        if qwen_chat { "qwen-chat" } else { "raw" },
        if qwen_chat && !disable_thinking {
            "on"
        } else if qwen_chat {
            "off"
        } else {
            "n/a"
        },
        spec_tokens,
        mtp_probe,
        effective_physical_verify_n
            .map(|n| n.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        mtp_single_cb_draft,
        mtp_draft_token_embd_head,
        mtp_draft_lm_head_q4_1,
        mtp_draft_lm_head_q4_0,
        mtp_draft_lm_head_q4_affine64,
        mtp_base_hidden,
        mtp_recursive_hidden,
        mtp_history,
        prompt_ids.len(),
        tokens,
        stops,
    );
    eprintln!(
        "[mtp-bench] execution_features: packed_base_prefill={} q5_k_n2_seq={} \
         iq2_s_n2_nc2={} iq3_s_n2_nc2={} skip_final_checkpoint={} \
         direct_mtp_f32_destination={} \
         shared_kv_q2_requested={} shared_kv_q2_min_position=16384",
        packed_base_prefill,
        q5_k_n2_seq,
        iq2_s_n2_nc2,
        iq3_s_n2_nc2,
        skip_final_checkpoint,
        direct_mtp_f32_destination,
        shared_kv_q2_requested,
    );
    eprintln!("[mtp-bench] MTP head load: {mtp_load_ms:.3} ms");
    eprintln!(
        "[mtp-bench] build_identity: status={} commit={} build_source_state={:?} runtime_source_state={:?} overrides={:?}",
        build_identity.status,
        build_identity.build_commit,
        build_identity.build_source_state,
        build_identity.runtime_source_state,
        build_identity.overrides,
    );
    if let Some((dtypes, bytes)) = mtp_moe_bank_ledger {
        eprintln!(
            "[mtp-bench] MTP MoE banks: policy={:?} gate/up/down={:?}/{:?}/{:?} bytes={bytes}",
            mtp_head.moe_bank_policy, dtypes[0], dtypes[1], dtypes[2],
        );
    }

    let mf = MetalForward::new(&ctx, &mm);
    let draft_lm_head_override = if mtp_draft_lm_head_q4_1 {
        let t = Instant::now();
        eprintln!("[mtp-bench] quantizing output.weight -> draft Q4_1 lm_head");
        let q =
            quantize_lm_head_to_q4_1(&ctx, &mm.lm_head).context("quantize draft lm_head Q4_1")?;
        eprintln!(
            "[mtp-bench] draft Q4_1 lm_head ready in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(q)
    } else if mtp_draft_lm_head_q4_0 {
        let t = Instant::now();
        eprintln!("[mtp-bench] quantizing output.weight -> draft Q4_0 lm_head");
        let q =
            quantize_lm_head_to_q4_0(&ctx, &mm.lm_head).context("quantize draft lm_head Q4_0")?;
        eprintln!(
            "[mtp-bench] draft Q4_0 lm_head ready in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(q)
    } else {
        None
    };
    let draft_affine_q4_head_override = if mtp_draft_lm_head_q4_affine64 {
        let t = Instant::now();
        eprintln!("[mtp-bench] quantizing output.weight -> draft affine Q4 gs64 lm_head");
        let q = quantize_lm_head_to_affine_q4_gs64(&ctx, &mm.lm_head)
            .context("quantize draft affine Q4 gs64 lm_head")?;
        eprintln!(
            "[mtp-bench] draft affine Q4 gs64 lm_head ready in {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Some(q)
    } else {
        None
    };
    let cap = prompt_ids.len() + tokens + 16;

    if !no_warmup {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("warmup session")?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut s)?;
    }

    // ----- MTP=off: greedy baseline -----
    let mut ref_session = MetalSession::fresh(&ctx, &mm, cap).context("ref session")?;
    let mut ref_tokens = prompt_ids.clone();
    let t_ref_total = Instant::now();
    let t_ref_prefill = Instant::now();
    // Product reference: ordinary qwen uses packed multi-token prefill, so MTP
    // must preserve this target stream rather than a retired serial-prefill
    // reduction order. The terminal audit still compares continuation state
    // against serial target transitions under numerical tolerances.
    let mut ref_layer_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16).context("ref layer scratch")?;
    let last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut ref_session,
        &mut ref_layer_scratch,
        &[],
        None,
    )?;
    let ref_prefill_ms = t_ref_prefill.elapsed().as_secs_f64() * 1e3;

    let t_ref_decode = Instant::now();
    let mut next_tok = argmax_i32(&last_logits);
    let mut pos = (prompt_ids.len() - 1) as u32;
    let mut ref_emitted = 0usize;
    for _ in 0..tokens {
        ref_tokens.push(next_tok);
        ref_emitted += 1;
        if stops.contains(&next_tok) || ref_emitted >= tokens {
            break;
        }
        pos += 1;
        let logits = mf.single_token(next_tok, pos, &mut ref_session)?;
        next_tok = argmax_i32(&logits);
    }
    let ref_decode_ms = t_ref_decode.elapsed().as_secs_f64() * 1e3;
    let ref_total_ms = t_ref_total.elapsed().as_secs_f64() * 1e3;
    let ref_decode_tps = ref_emitted as f64 / (ref_decode_ms / 1000.0);
    let ref_generated_vec: Vec<i32> = ref_tokens[prompt_ids.len()..].to_vec();

    // ----- MTP=on: speculative decode -----
    if spec_tokens == 1
        && matches!(
            mtp_probe,
            MtpProbeMode::ReplayCurrent | MtpProbeMode::BodyNoLmHead | MtpProbeMode::BridgeOnly
        )
    {
        anyhow::bail!("recorded MTP probes require --spec-tokens 2..=15");
    }

    let mut planned_state_audit: Option<MtpTargetStateAudit> = None;
    let mut normal_state_audit: Option<MtpTargetStateAudit> = None;
    let mut run_planned = |plan: PackedDraftPlan<'_>, label: &str| -> Result<DecodeOutput> {
        let audit_target_state = matches!(plan, PackedDraftPlan::Oracle(_));
        let mtp_session =
            MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
        let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
        let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
        spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
        spec.set_draft_lm_head_override(draft_lm_head_override.clone());
        spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
        spec.set_base_hidden_variant(mtp_base_hidden.into());
        spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
        spec.set_history_mode(mtp_history.into());
        if mtp_state_trace && audit_target_state {
            let mut shadow = MetalSession::fresh(&ctx, &mm, cap).context("state-trace shadow")?;
            for (i, &tid) in prompt_ids.iter().enumerate() {
                mf.single_token(tid, i as u32, &mut shadow)
                    .context("state-trace shadow prefill")?;
            }
            let mf_probe = &mf;
            spec.set_step_probe(Box::new(move |ev: &PackedStepProbe<'_>| {
                for (i, &tok) in ev.committed.iter().enumerate() {
                    mf_probe
                        .single_token(tok, ev.start_position + i as u32, &mut shadow)
                        .map_err(|e| format!("shadow advance: {e}"))?;
                }
                let (g_max, g_layer, g_over) =
                    max_abs_f32_layers(&shadow.gdn_state, &ev.session.gdn_state)
                        .map_err(|e| format!("gdn compare: {e}"))?;
                let (c_max, c_layer, c_over) =
                    max_abs_f32_layers(&shadow.gdn_conv, &ev.session.gdn_conv)
                        .map_err(|e| format!("conv compare: {e}"))?;
                let phase = match ev.phase {
                    PackedStepProbePhase::Prefill => "prefill",
                    PackedStepProbePhase::Packet => "packet",
                };
                eprintln!(
                    "[state-trace] {phase} step={} pos={} n_eff={} acc={} keep={} \
                     restore={} stop={} | gdn max={:.3e} L{} over={} | \
                     conv max={:.3e} L{} over={} | kvpos_eq={}",
                    ev.step_idx,
                    ev.start_position,
                    ev.n_eff,
                    ev.n_accepted,
                    ev.n_keep,
                    ev.restore_fired,
                    ev.stop_now,
                    g_max,
                    g_layer,
                    g_over,
                    c_max,
                    c_layer,
                    c_over,
                    shadow.kv_n_pos == ev.session.kv_n_pos,
                );
                Ok(())
            }));
        }
        let mut verify_scratch =
            MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                .context("mtp packed verify scratch")?;
        let mut layer_scratch =
            MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                .context("mtp packed layer scratch")?;
        let output = spec
            .decode_packed_n_planned(
                &prompt_ids,
                tokens,
                &stops,
                &mut spec_session,
                spec_tokens,
                &mut verify_scratch,
                &mut layer_scratch,
                plan,
            )
            .with_context(|| format!("spec decode packed-n {label}"))?;
        if audit_target_state {
            let generated = &ref_generated_vec[..ref_generated_vec.len().saturating_sub(1)];
            let pending_terminal_token = *ref_generated_vec
                .last()
                .context("oracle produced no terminal token")?;
            let mut serial_session =
                MetalSession::fresh(&ctx, &mm, cap).context("state-audit serial session")?;
            for (position, &token) in prompt_ids.iter().chain(generated).enumerate() {
                mf.single_token(token, position as u32, &mut serial_session)
                    .context("state-audit serial transition")?;
            }
            planned_state_audit = Some(audit_mtp_target_state_chain(
                &mf,
                &mut serial_session,
                &mut spec_session,
                prompt_ids.len() + generated.len(),
                pending_terminal_token,
            )?);
        }
        Ok(output)
    };

    let run_recorded_work =
        |trace: &[RecordedDraftStep], work: RecordedMtpWork, label: &str| -> Result<DecodeOutput> {
            let mtp_session =
                MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
            let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
            let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
            spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
            spec.set_draft_lm_head_override(draft_lm_head_override.clone());
            spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
            spec.set_base_hidden_variant(mtp_base_hidden.into());
            spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
            spec.set_history_mode(mtp_history.into());
            let mut verify_scratch =
                MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                    .context("mtp packed verify scratch")?;
            let mut layer_scratch =
                MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                    .context("mtp packed layer scratch")?;
            spec.decode_packed_n_recorded_mtp_work(
                &prompt_ids,
                tokens,
                &stops,
                &mut spec_session,
                spec_tokens,
                &mut verify_scratch,
                &mut layer_scratch,
                trace,
                work,
            )
            .with_context(|| format!("spec decode packed-n {label}"))
        };

    let mut mtp_rank_rows: Vec<MtpRankRow> = Vec::new();
    // Gate failures discovered mid-probe are DEFERRED until after the
    // results/JSON emission so failing-audit runs still yield harvestable
    // timing. Exit semantics are unchanged (nonzero exit, same message).
    let mut deferred_gate_failure: Option<String> = None;
    let result = match mtp_probe {
        MtpProbeMode::Normal => {
            let mtp_session =
                MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
            let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
            let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
            spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
            spec.set_draft_lm_head_override(draft_lm_head_override.clone());
            spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
            spec.set_base_hidden_variant(mtp_base_hidden.into());
            spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
            spec.set_history_mode(mtp_history.into());
            let output = if !use_packed_verify {
                spec.decode(&prompt_ids, tokens, &stops, &mut spec_session)
                    .context("spec decode")?
            } else {
                let mut verify_scratch =
                    MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                        .context("mtp packed verify scratch")?;
                let mut layer_scratch =
                    MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                        .context("mtp packed layer scratch")?;
                spec.decode_packed_n_recording(
                    &prompt_ids,
                    tokens,
                    &stops,
                    &mut spec_session,
                    spec_tokens,
                    &mut verify_scratch,
                    &mut layer_scratch,
                    None,
                    mtp_rank_topk.as_ref().map(|_| &mut mtp_rank_rows),
                    mtp_single_cb_draft,
                )
                .context("spec decode packed-n")?
            };
            if use_packed_verify {
                let generated = &ref_generated_vec[..ref_generated_vec.len().saturating_sub(1)];
                if output.tokens[prompt_ids.len()..] != ref_generated_vec {
                    // Stream divergent: a state audit against the serial
                    // token reconstruction would not be comparable. Skip
                    // it, defer the failure past the results emission.
                    deferred_gate_failure.get_or_insert_with(|| {
                        "native MTP emitted tokens differ from packed product target".to_string()
                    });
                } else {
                    let pending_terminal_token = *ref_generated_vec
                        .last()
                        .context("native MTP produced no terminal token")?;
                    let mut serial_session = MetalSession::fresh(&ctx, &mm, cap)
                        .context("state-audit serial session")?;
                    for (position, &token) in prompt_ids.iter().chain(generated).enumerate() {
                        mf.single_token(token, position as u32, &mut serial_session)
                            .context("state-audit serial transition")?;
                    }
                    let audit = audit_mtp_target_state_chain(
                        &mf,
                        &mut serial_session,
                        &mut spec_session,
                        prompt_ids.len() + generated.len(),
                        pending_terminal_token,
                    )?;
                    if !audit.resume_audit_pass {
                        deferred_gate_failure.get_or_insert_with(|| {
                            format!("native MTP terminal resume audit failed: {audit:?}")
                        });
                    }
                    normal_state_audit = Some(audit);
                }
            }
            output
        }
        MtpProbeMode::Oracle => run_planned(PackedDraftPlan::Oracle(&ref_generated_vec), "oracle")?,
        MtpProbeMode::ReplayCurrent | MtpProbeMode::BodyNoLmHead | MtpProbeMode::BridgeOnly => {
            let mtp_session =
                MetalMtpSession::fresh(&ctx, &mtp_head, &m.arch, cap).context("MTP session")?;
            let mut spec_session = MetalSession::fresh(&ctx, &mm, cap).context("spec session")?;
            let mut spec = SpeculativeDecoder::new(&mf, &mtp_head, mtp_session);
            spec.set_draft_token_embd_head(mtp_draft_token_embd_head);
            spec.set_draft_lm_head_override(draft_lm_head_override.clone());
            spec.set_draft_affine_q4_head_override(draft_affine_q4_head_override.clone());
            spec.set_base_hidden_variant(mtp_base_hidden.into());
            spec.set_recursive_hidden_variant(mtp_recursive_hidden.into());
            spec.set_history_mode(mtp_history.into());
            let mut verify_scratch =
                MetalDFlashVerifyScratch::fresh(&ctx, &mm, planned_verify_n as u32, 1)
                    .context("mtp packed verify scratch")?;
            let mut layer_scratch =
                MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, planned_verify_n as u32)
                    .context("mtp packed layer scratch")?;
            let mut draft_trace = Vec::new();
            let recorded = spec
                .decode_packed_n_recording(
                    &prompt_ids,
                    tokens,
                    &stops,
                    &mut spec_session,
                    spec_tokens,
                    &mut verify_scratch,
                    &mut layer_scratch,
                    Some(&mut draft_trace),
                    None,
                    mtp_single_cb_draft,
                )
                .context("spec decode packed-n recording")?;
            let recorded_emitted = recorded.tokens.len() - prompt_ids.len();
            let recorded_tps = recorded_emitted as f64 / (recorded.stats.wall_ms / 1000.0);
            eprintln!(
                "[mtp-bench] replay-current source: {recorded_emitted} tokens, \
                 {:.1} ms, {:.1} t/s, steps={}, trace_rows={}",
                recorded.stats.wall_ms,
                recorded_tps,
                recorded.stats.steps,
                draft_trace.len(),
            );
            match mtp_probe {
                MtpProbeMode::ReplayCurrent => {
                    run_planned(PackedDraftPlan::Recorded(&draft_trace), "replay-current")?
                }
                MtpProbeMode::BodyNoLmHead => run_recorded_work(
                    &draft_trace,
                    RecordedMtpWork::BodyNoLmHead,
                    "body-no-lm-head",
                )?,
                MtpProbeMode::BridgeOnly => {
                    run_recorded_work(&draft_trace, RecordedMtpWork::BridgeOnly, "bridge-only")?
                }
                MtpProbeMode::Normal | MtpProbeMode::Oracle => unreachable!(),
            }
        }
    };

    if let Some(rank_path) = &mtp_rank_topk {
        if let Some(dir) = rank_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut out = String::new();
        for row in &mtp_rank_rows {
            let top_tokens = row
                .top_tokens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let top_logits = row
                .top_logits
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!(
                "{{\"step\":{},\"depth\":{},\"rank\":{},\"accepted\":{},\"draft_tok\":{},\"target_tok\":{},\"target_logit\":{:.6},\"top_tokens\":[{}],\"top_logits\":[{}]}}\n",
                row.step,
                row.depth,
                row.rank,
                row.accepted,
                row.draft_tok,
                row.target_tok,
                row.target_logit,
                top_tokens,
                top_logits,
            ));
        }
        std::fs::write(rank_path, out)?;
        eprintln!(
            "[mtp-bench] MTP rank rows: {} -> {}",
            mtp_rank_rows.len(),
            rank_path.display()
        );
        eprintln!("[mtp-bench] depth\tn\tp1\tp2\tp4\tp8\tp16\tmean_rank\tmax_rank");
        let max_depth = mtp_rank_rows.iter().map(|r| r.depth).max().unwrap_or(0);
        for depth in 0..=max_depth {
            let ranks: Vec<usize> = mtp_rank_rows
                .iter()
                .filter(|r| r.depth == depth)
                .map(|r| r.rank)
                .collect();
            if ranks.is_empty() {
                continue;
            }
            let n = ranks.len() as f64;
            let pk = |k: usize| ranks.iter().filter(|&&r| r <= k).count() as f64 / n;
            let mean = ranks.iter().sum::<usize>() as f64 / n;
            let max_rank = ranks.iter().copied().max().unwrap_or(0);
            eprintln!(
                "[mtp-bench] {depth}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{}",
                ranks.len(),
                pk(1),
                pk(2),
                pk(4),
                pk(8),
                pk(16),
                mean,
                max_rank
            );
        }
    }

    let spec_emitted = result.tokens.len() - prompt_ids.len();
    let spec_total_ms = result.stats.wall_ms;
    let spec_decode_ms = result.stats.decode_ms;
    let spec_prefill_ms = result.stats.prefill_ms;
    let spec_decode_tps = spec_emitted as f64 / (spec_decode_ms / 1000.0).max(f64::MIN_POSITIVE);
    let transitions_per_verify = if result.stats.steps > 0 {
        spec_emitted.saturating_sub(1) as f64 / result.stats.steps as f64
    } else {
        0.0
    };

    // ----- Compare -----
    let ref_generated = &ref_tokens[prompt_ids.len()..];
    let spec_generated = &result.tokens[prompt_ids.len()..];
    let identical = ref_generated == spec_generated;
    let target_generated_token_sha256 = token_ids_sha256_i32le(ref_generated);
    let expected_target_transitions = ref_emitted.saturating_sub(1);
    let target_state_audit = if mtp_probe == MtpProbeMode::Oracle {
        planned_state_audit
    } else if mtp_probe == MtpProbeMode::Normal && use_packed_verify {
        normal_state_audit
    } else {
        None
    };
    if mtp_probe == MtpProbeMode::Oracle {
        let audit = target_state_audit.context("oracle target-state audit missing")?;
        // Hard harness invariants stay immediate (timing is meaningless
        // if these fire); stream/audit gate failures are deferred past
        // the results emission.
        anyhow::ensure!(
            result.stats.mtp_calls == 0,
            "oracle unexpectedly executed {} MTP calls",
            result.stats.mtp_calls
        );
        anyhow::ensure!(
            result.stats.target_transitions as usize == expected_target_transitions,
            "oracle target transitions {} != expected {}",
            result.stats.target_transitions,
            expected_target_transitions
        );
        if !identical {
            deferred_gate_failure.get_or_insert_with(|| {
                "oracle emitted tokens differ from packed product target".to_string()
            });
        } else if !audit.resume_audit_pass {
            deferred_gate_failure
                .get_or_insert_with(|| format!("oracle terminal resume audit failed: {audit:?}"));
        }
    }

    // Apples-to-apples reporting. The earlier version mixed phases —
    // comparing MTP=off decode-only t/s (excludes prefill) with
    // MTP=on overall t/s (includes prefill) made the regression look
    // worse than it was. Wall-time-vs-wall-time is the honest signal.
    let ref_decode_only_tps = ref_emitted as f64 / (ref_decode_ms / 1000.0);
    let ref_total_tps = ref_emitted as f64 / (ref_total_ms / 1000.0);
    let spec_total_tps = spec_emitted as f64 / (spec_total_ms / 1000.0);
    let _ = ref_decode_tps; // unused (replaced by ref_decode_only_tps)
    let _ = spec_decode_tps; // unused (replaced by spec_total_tps for clarity)

    eprintln!();
    eprintln!("[mtp-bench] === results ===");
    eprintln!(
        "[mtp-bench] MTP=off: {ref_emitted} tokens, prefill {ref_prefill_ms:.1} ms + \
         decode {ref_decode_ms:.1} ms = {ref_total_ms:.1} ms total"
    );
    eprintln!(
        "[mtp-bench]   t/s: decode-only {ref_decode_only_tps:.1} | total \
         {ref_total_tps:.1}"
    );
    eprintln!(
        "[mtp-bench] MTP=on : {spec_emitted} tokens, prefill {spec_prefill_ms:.1} ms + \
         decode {spec_decode_ms:.1} ms = {spec_total_ms:.1} ms total"
    );
    eprintln!("[mtp-bench]   t/s: decode-only {spec_decode_tps:.1} | total {spec_total_tps:.1}");
    eprintln!(
        "[mtp-bench]   α (acceptance rate) = {:.3}   steps={}  accepted={}",
        result.stats.acceptance_rate(),
        result.stats.steps,
        result.stats.accepted,
    );
    eprintln!(
        "[mtp-bench]   target transitions / verifier = {:.3} ({}/{})",
        transitions_per_verify,
        spec_emitted.saturating_sub(1),
        result.stats.steps,
    );
    eprintln!(
        "[mtp-bench]   base_calls={}  mtp_calls={} \
         (= prompt prefill + step-B drafts + step-E bridges)",
        result.stats.base_forward_calls, result.stats.mtp_calls,
    );
    let spec_phase_known_ms = result.stats.draft_ms
        + result.stats.verify_ms
        + result.stats.restore_ms
        + result.stats.bridge_ms;
    let spec_phase_other_ms = (spec_decode_ms - spec_phase_known_ms).max(0.0);
    eprintln!(
        "[mtp-bench]   decode phases ms: draft={:.1} verify={:.1} restore={:.1} \
         bridge={:.1} other={:.1}",
        result.stats.draft_ms,
        result.stats.verify_ms,
        result.stats.restore_ms,
        result.stats.bridge_ms,
        spec_phase_other_ms,
    );

    // Wall-time speedup: total-vs-total, the apples-to-apples ratio that
    // matches docs/H4-MTP.md §3.2's `1/(1+ε)` prediction. The earlier
    // 'decode-only-vs-total' phrasing was misleading — let people see
    // both interpretations.
    let total_speedup = ref_total_ms / spec_total_ms;
    eprintln!(
        "[mtp-bench]   speedup (total ms): {ref_total_ms:.1} / {spec_total_ms:.1} = \
         {total_speedup:.3}× (>1.0 means MTP wins)"
    );

    eprintln!(
        "[mtp-bench] equivalence: {} ({} vs {} emitted)",
        if identical {
            "PASS (identical sequences)"
        } else {
            "FAIL (sequences differ)"
        },
        spec_emitted,
        ref_emitted,
    );
    if let Some(audit) = target_state_audit {
        if identical {
            eprintln!(
                "[mtp-bench] terminal resume: {} continuation_steps={} \
                 kv_pos={:?} kv_max_abs={:.3e} kv_cos={:.10} \
                 gdn_state_max_abs={:.3e} gdn_conv_max_abs={:.3e} \
                 continuation_max_abs={:.3e} continuation_cos={:.10}",
                if audit.resume_audit_pass {
                    "PASS"
                } else {
                    "FAIL"
                },
                MTP_CONTINUATION_AUDIT_STEPS,
                audit.candidate_final_position,
                audit.kv_payload_max_abs,
                audit.kv_payload_cosine,
                audit.gdn_state_max_abs,
                audit.gdn_conv_max_abs,
                audit.continuation_logits_max_abs,
                audit.continuation_logits_cosine,
            );
        } else {
            eprintln!(
                "[mtp-bench] terminal resume: N/A (stream divergent; state \
                 audit vs serial tokens not comparable)"
            );
        }
    }
    if !identical {
        let n_show = 8usize.min(ref_generated.len()).min(spec_generated.len());
        eprintln!(
            "[mtp-bench]   ref[..{n_show}]:  {:?}",
            &ref_generated[..n_show]
        );
        eprintln!(
            "[mtp-bench]   spec[..{n_show}]: {:?}",
            &spec_generated[..n_show]
        );
    }

    if let Some(output_path) = &output {
        if let Some(dir) = output_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let steps = result.stats.steps as f64;
        let emitted_per_step = if result.stats.steps > 0 {
            serde_json::json!(spec_emitted as f64 / steps)
        } else {
            serde_json::Value::Null
        };
        let accepted_per_step = if result.stats.steps > 0 {
            serde_json::json!(result.stats.accepted as f64 / steps)
        } else {
            serde_json::Value::Null
        };
        let drafts_per_step = if result.stats.steps > 0 {
            serde_json::json!(result.stats.drafts_attempted as f64 / steps)
        } else {
            serde_json::Value::Null
        };
        let accepted_prefix_histogram = result
            .stats
            .accepted_prefix_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(accepted, count)| serde_json::json!({"accepted": accepted, "steps": count}))
            .collect::<Vec<_>>();
        let speculative_phase_ms = serde_json::json!({
            "draft": result.stats.draft_ms,
            "verify": result.stats.verify_ms,
            "restore": result.stats.restore_ms,
            "bridge": result.stats.bridge_ms,
            "other": spec_phase_other_ms,
        });
        let mtp_moe_banks =
            mtp_moe_bank_ledger.map_or(serde_json::Value::Null, |(dtypes, bytes)| {
                serde_json::json!({
                    "policy": format!("{:?}", mtp_head.moe_bank_policy),
                    "gate_dtype": format!("{:?}", dtypes[0]),
                    "up_dtype": format!("{:?}", dtypes[1]),
                    "down_dtype": format!("{:?}", dtypes[2]),
                    "bytes": bytes,
                })
            });
        let target_state = target_state_audit.map_or(serde_json::Value::Null, |audit| {
            serde_json::json!({
                "continuation_steps": MTP_CONTINUATION_AUDIT_STEPS,
                "resume_audit_pass": audit.resume_audit_pass,
                "kv_position_equal": audit.kv_position_equal,
                "kv_payload_exact": audit.kv_payload_exact,
                "kv_payload_max_abs": audit.kv_payload_max_abs,
                "kv_payload_cosine": audit.kv_payload_cosine,
                "reference_final_position": audit.reference_final_position,
                "candidate_final_position": audit.candidate_final_position,
                "gdn_state_max_abs": audit.gdn_state_max_abs,
                "gdn_conv_max_abs": audit.gdn_conv_max_abs,
                "continuation_argmax_equal": audit.continuation_argmax_equal,
                "continuation_logits_max_abs": audit.continuation_logits_max_abs,
                "continuation_logits_cosine": audit.continuation_logits_cosine,
            })
        });
        let semantics = serde_json::json!({
            "verify_mode": if use_packed_verify { "packed_n" } else { "lazy_mtp1" },
            "sampler": "greedy_argmax",
            "correction_accounting": "deferred_next_step_carry",
            "equivalence": if target_state_audit.is_some() {
                "target_greedy_sequence_and_terminal_resume_audit"
            } else {
                "target_greedy_sequence"
            },
        });
        let reference = serde_json::json!({
            "emitted": ref_emitted,
            "target_transitions": expected_target_transitions,
            "prefill_ms": ref_prefill_ms,
            "decode_ms": ref_decode_ms,
            "total_ms": ref_total_ms,
            "decode_only_tps": ref_decode_only_tps,
            "total_tps": ref_total_tps,
        });
        let planned_metrics = matches!(
            mtp_probe,
            MtpProbeMode::Oracle | MtpProbeMode::ReplayCurrent
        );
        let speculative = serde_json::json!({
            "emitted": spec_emitted,
            "prefill_ms": spec_prefill_ms,
            "decode_ms": spec_decode_ms,
            "decode_only_tps": spec_decode_tps,
            "total_ms": spec_total_ms,
            "total_tps": spec_total_tps,
            "steps": result.stats.steps,
            "accepted": result.stats.accepted,
            "drafts_attempted": result.stats.drafts_attempted,
            "acceptance_rate": result.stats.acceptance_rate(),
            "emitted_per_step": emitted_per_step,
            "target_transitions_per_verify": transitions_per_verify,
            "accepted_per_step": accepted_per_step,
            "drafts_per_step": drafts_per_step,
            "accepted_prefix_histogram": accepted_prefix_histogram,
            "base_forward_calls": result.stats.base_forward_calls,
            "mtp_calls": result.stats.mtp_calls,
            "target_transitions": planned_metrics.then_some(result.stats.target_transitions),
            "final_effective_verify_n": planned_metrics
                .then_some(result.stats.final_effective_verify_n),
            "rank_rows": mtp_rank_rows.len(),
            "phase_ms": speculative_phase_ms,
            "target_state": target_state,
        });
        let token_fixture = if include_token_ids {
            serde_json::json!({
                "schema_version": 1,
                "prompt_token_ids": prompt_ids,
                "target_generated_token_ids": ref_generated,
            })
        } else {
            serde_json::Value::Null
        };
        let row = serde_json::json!({
            "model": model.display().to_string(),
            "prompt": prompt,
            "qwen_chat": qwen_chat,
            "system": system,
            "disable_thinking": disable_thinking,
            "prompt_tokens": prompt_ids.len(),
            "prompt_token_sha256": prompt_token_sha256,
            "generated_requested": tokens,
            "stop_tokens": stops,
            "spec_tokens": spec_tokens,
            "logical_verify_n": effective_logical_verify_n,
            "physical_verify_n": effective_physical_verify_n,
            "mtp_load_ms": mtp_load_ms,
            "target_generated_token_sha256": target_generated_token_sha256,
            "execution_features": {
                "packed_base_prefill": packed_base_prefill,
                "q5_k_n2_seq": q5_k_n2_seq,
                "iq2_s_n2_nc2": iq2_s_n2_nc2,
                "iq3_s_n2_nc2": iq3_s_n2_nc2,
                "skip_final_checkpoint": skip_final_checkpoint,
                "direct_mtp_f32_destination": direct_mtp_f32_destination,
                "shared_kv_q2_requested": shared_kv_q2_requested,
                "shared_kv_q2_min_position": 16_384,
            },
            "terminal_token_target_transition_consumed": if target_state_audit.is_some() {
                Some(false)
            } else {
                None
            },
            "probe": format!("{:?}", mtp_probe),
            "single_cb_draft": mtp_single_cb_draft,
            "draft_token_embd_head": mtp_draft_token_embd_head,
            "draft_lm_head_q4_1": mtp_draft_lm_head_q4_1,
            "draft_lm_head_q4_0": mtp_draft_lm_head_q4_0,
            "draft_lm_head_q4_affine64": mtp_draft_lm_head_q4_affine64,
            "base_hidden": format!("{:?}", mtp_base_hidden),
            "recursive_hidden": format!("{:?}", mtp_recursive_hidden),
            "mtp_history": format!("{:?}", mtp_history),
            "mtp_moe_banks": mtp_moe_banks,
            "rank_topk": mtp_rank_topk.as_ref().map(|p| p.display().to_string()),
            "no_warmup": no_warmup,
            "build_identity": build_identity,
            "qwen_env": qwen_env,
            "semantics": semantics,
            "token_fixture": token_fixture,
            "reference": reference,
            "speculative": speculative,
            "speedup_total_ms": total_speedup,
            "identical": identical,
        });
        std::fs::write(output_path, serde_json::to_string(&row)? + "\n")?;
        eprintln!("[mtp-bench] wrote {}", output_path.display());
    }

    if let Some(msg) = deferred_gate_failure {
        return Err(anyhow!("{msg}"));
    }
    if !identical {
        return Err(anyhow!(
            "MTP=on and MTP=off generated different token sequences"
        ));
    }
    Ok(())
}

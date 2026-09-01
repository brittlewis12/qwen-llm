use qwen_llm::metal_forward::RMS_EPS;
use qwen_llm::runtime::{Runtime, SequenceConfig};
use qwen_llm::tensor::GgmlType;
use qwen_llm::workspace_lens::{
    AttnBlockVjpRule, DenseFfnVjpRule, GdnBlockVjpRule, GdnMixerVjpRule, LinearRole,
    WORKSPACE_LENS_IDENTITY_SCHEME, WorkspaceLensBlockKind, WorkspaceLensLinear, WorkspaceLensRule,
};
use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: workspace_lens_smoke <model.gguf> [token_id]")?;
    let token_id = args
        .next()
        .map(|value| value.to_string_lossy().parse::<i32>())
        .transpose()?
        .unwrap_or(0);
    if args.next().is_some() {
        return Err("usage: workspace_lens_smoke <model.gguf> [token_id]".into());
    }

    let runtime = Runtime::metal()?;
    let loaded = runtime.load_model(&model)?;
    let arch = loaded.arch();
    let mut sequence = loaded.create_sequence(SequenceConfig::new(8))?;
    let mut workspace_lens = loaded.workspace_lens_session(&mut sequence)?;
    let identity = workspace_lens.identity();
    let linears = workspace_lens.linears()?;
    let capture_layers = [0, arch.n_layer.saturating_sub(2), arch.n_layer - 1];
    let forward = workspace_lens.forward_token_with_dense_ffn_capture(token_id, &capture_layers)?;
    let capture_norms: Vec<f64> = forward
        .capture
        .post_block_residuals
        .chunks_exact(forward.capture.hidden_size)
        .map(|row| {
            row.iter()
                .map(|&value| f64::from(value) * f64::from(value))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let (top_token, top_logit) = forward
        .logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .ok_or("model returned empty logits")?;

    let lm_head = workspace_lens.linear_info(WorkspaceLensLinear::LmHead)?;
    let mut selected_token_ids = vec![
        0,
        arch.vocab_size
            .checked_sub(1)
            .ok_or("model vocabulary is empty")?,
        u32::try_from(top_token)?,
    ];
    selected_token_ids.dedup();
    if selected_token_ids[0] == *selected_token_ids.last().unwrap_or(&selected_token_ids[0])
        && selected_token_ids.len() > 1
    {
        selected_token_ids.pop();
    }
    let selected_readouts = workspace_lens.selected_token_readouts(&selected_token_ids)?;
    if selected_readouts.hidden_size != arch.hidden_size as usize
        || selected_readouts.lm_head_dtype != lm_head.dtype
        || selected_readouts.lm_head_shape != lm_head.shape
        || selected_readouts.output_norm_dtype != GgmlType::F32
        || selected_readouts.output_norm_shape != vec![u64::from(arch.hidden_size)]
        || selected_readouts
            .values
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("selected-token readout metadata or finiteness check failed".into());
    }
    let mut cotangent = vec![0.0f32; lm_head.shape[1]];
    cotangent[top_token] = 1.0;
    let lm_head_vjp =
        workspace_lens.frozen_linear_vjp(WorkspaceLensLinear::LmHead, &cotangent, 1)?;
    let lm_head_vjp_norm = lm_head_vjp
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    let last_capture_offset = (capture_layers.len() - 1) * forward.capture.hidden_size;
    let last_pre_ffn = &forward.capture.pre_ffn_residuals
        [last_capture_offset..last_capture_offset + forward.capture.hidden_size];
    let last_residual = &forward.capture.post_block_residuals
        [last_capture_offset..last_capture_offset + forward.capture.hidden_size];
    let selected_numerators = selected_token_ids
        .iter()
        .enumerate()
        .map(|(slot, _)| {
            selected_readouts
                .token_values(slot)
                .ok_or("selected-token readout omitted a requested row")
                .map(|readout| {
                    readout
                        .iter()
                        .zip(last_residual)
                        .map(|(&weight, &residual)| f64::from(weight) * f64::from(residual))
                        .sum::<f64>()
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let final_rms_denominator = (last_residual
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / last_residual.len() as f64
        + f64::from(RMS_EPS))
    .sqrt();
    // Row-gather and production mat-vec use different reduction orders. These
    // dtype-aware tolerances cover that roundoff and quantized deblocking error
    // while remaining tight enough to catch orientation or gamma-folding drift.
    let (selected_abs_tolerance, selected_rel_tolerance) = match lm_head.dtype {
        GgmlType::F32 => (2e-3f64, 2e-4f64),
        GgmlType::F16 | GgmlType::BF16 => (5e-3, 5e-4),
        GgmlType::Q8_0 => (1e-2, 1e-3),
        GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::IQ4_NL => (3e-2, 3e-3),
        dtype => return Err(format!("unexpected selected-token LM-head dtype {dtype:?}").into()),
    };
    for (&token_id, &numerator) in selected_token_ids.iter().zip(&selected_numerators) {
        let production_numerator =
            f64::from(forward.logits[usize::try_from(token_id)?]) * final_rms_denominator;
        let error = (numerator - production_numerator).abs();
        let tolerance = selected_abs_tolerance
            + selected_rel_tolerance * numerator.abs().max(production_numerator.abs());
        if !production_numerator.is_finite() || error > tolerance {
            return Err(format!(
                "selected-token numerator mismatch for token {token_id}: readout={numerator} production={production_numerator} error={error} tolerance={tolerance} dtype={:?}",
                lm_head.dtype
            )
            .into());
        }
    }
    let mut hidden_cotangent = vec![0.0f32; forward.capture.hidden_size];
    let hidden_coordinate = top_token % hidden_cotangent.len();
    hidden_cotangent[hidden_coordinate] = 1.0;
    let dense_ffn_r_vjp = workspace_lens.dense_ffn_vjp(
        arch.n_layer - 1,
        last_pre_ffn,
        &hidden_cotangent,
        1,
        DenseFfnVjpRule::Relp,
    )?;
    let dense_ffn_r_vjp_norm = dense_ffn_r_vjp
        .values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    let gdn_layer = (1..arch.n_layer)
        .find(|&layer| {
            workspace_lens
                .linear_info(WorkspaceLensLinear::Layer {
                    index: layer,
                    role: LinearRole::GdnQkv,
                })
                .is_ok()
        })
        .ok_or("model has no nonzero GDN layer")?;
    let gdn_forward =
        workspace_lens.forward_prompt_with_gdn_capture(&[token_id, token_id], gdn_layer)?;
    let mut gdn_cotangent = vec![0.0f32; 2 * gdn_forward.hidden_size()];
    for token in 0..2 {
        gdn_cotangent[token * gdn_forward.hidden_size()
            + (hidden_coordinate + token) % gdn_forward.hidden_size()] = 1.0;
    }
    let gdn_r_vjp =
        workspace_lens.gdn_mixer_vjp(&gdn_forward, &gdn_cotangent, GdnMixerVjpRule::Relp)?;
    let gdn_r_vjp_norm = gdn_r_vjp
        .values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    let gdn_block_r_vjp =
        workspace_lens.gdn_block_vjp(&gdn_forward, &gdn_cotangent, GdnBlockVjpRule::Relp)?;
    let gdn_block_r_vjp_norm = gdn_block_r_vjp
        .values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if !gdn_r_vjp_norm.is_finite() || !gdn_block_r_vjp_norm.is_finite() {
        return Err("GDN VJP returned a non-finite norm".into());
    }
    for (name, replay) in [("mixer", &gdn_r_vjp), ("block", &gdn_block_r_vjp.mixer)] {
        if !replay.residual_replay_max_abs_error.is_finite()
            || !replay.final_conv_state_max_abs_error.is_finite()
            || !replay.final_recurrence_state_max_abs_error.is_finite()
            || replay.residual_replay_max_abs_error > 1e-5
            || replay.final_conv_state_max_abs_error > 1e-5
            || replay.final_recurrence_state_max_abs_error > 1e-5
        {
            return Err(format!(
                "GDN {name} replay drift residual={} conv={} recurrence={}",
                replay.residual_replay_max_abs_error,
                replay.final_conv_state_max_abs_error,
                replay.final_recurrence_state_max_abs_error,
            )
            .into());
        }
    }
    drop(sequence);
    let mut attn_sequence = loaded.create_sequence(SequenceConfig::new(8))?;
    let mut attn_workspace_lens = loaded.workspace_lens_session(&mut attn_sequence)?;
    let attn_layer = (1..arch.n_layer)
        .find(|&layer| {
            attn_workspace_lens
                .linear_info(WorkspaceLensLinear::Layer {
                    index: layer,
                    role: LinearRole::AttentionQAndGate,
                })
                .is_ok()
        })
        .ok_or("model has no nonzero full-attention layer")?;
    let attn_forward = attn_workspace_lens
        .forward_prompt_with_attn_capture(&[token_id, token_id, token_id, token_id], attn_layer)?;
    let mut attn_cotangent = vec![0.0f32; 4 * attn_forward.hidden_size()];
    for token in 0..4 {
        attn_cotangent[token * attn_forward.hidden_size()
            + (hidden_coordinate + token) % attn_forward.hidden_size()] = 1.0;
    }
    let attn_block_r_vjp = attn_workspace_lens.attn_block_vjp(
        &attn_forward,
        &attn_cotangent,
        AttnBlockVjpRule::Relp,
    )?;
    let attn_block_r_vjp_norm = attn_block_r_vjp
        .values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if !attn_block_r_vjp_norm.is_finite()
        || !attn_block_r_vjp.residual_replay_max_abs_error.is_finite()
    {
        return Err("attention VJP or replay diagnostic is non-finite".into());
    }
    if attn_block_r_vjp.residual_replay_max_abs_error > 5e-3 {
        return Err(format!(
            "attention F32-oracle replay drift {} exceeds 0.005",
            attn_block_r_vjp.residual_replay_max_abs_error
        )
        .into());
    }
    if model.file_name().and_then(|name| name.to_str()) == Some("Qwen3.8-27B-Q8_0.gguf")
        && token_id == 0
        && ((attn_block_r_vjp_norm - 2.056_573_313_704_004_3).abs() > 1e-4
            || (attn_block_r_vjp.residual_replay_max_abs_error - 0.000_644_683_84).abs() > 2e-4)
    {
        return Err(format!(
            "27B attention fixture drifted: norm={} replay={}",
            attn_block_r_vjp_norm, attn_block_r_vjp.residual_replay_max_abs_error
        )
        .into());
    }
    drop(attn_sequence);
    let mut workspace_sequence = loaded.create_sequence(SequenceConfig::new(8))?;
    let mut workspace_lens_session = loaded.workspace_lens_session(&mut workspace_sequence)?;
    let workspace_forward = workspace_lens_session
        .forward_prompt_with_workspace_capture(&[token_id, token_id, token_id, token_id])?;
    let workspace_sources = [0, attn_layer.saturating_sub(2), attn_layer - 1];
    let workspace_r_vjp = workspace_lens_session.workspace_vjp(
        &workspace_forward,
        attn_layer,
        &workspace_sources,
        &attn_cotangent,
        WorkspaceLensRule::Relp,
    )?;
    let mut second_workspace_cotangent = vec![0.0f32; attn_cotangent.len()];
    for token in 0..workspace_forward.n_tokens() {
        second_workspace_cotangent[token * workspace_forward.hidden_size()
            + (hidden_coordinate + token + 17) % workspace_forward.hidden_size()] = 1.0;
    }
    let second_workspace_vjp = workspace_lens_session.workspace_vjp(
        &workspace_forward,
        attn_layer,
        &workspace_sources,
        &second_workspace_cotangent,
        WorkspaceLensRule::Relp,
    )?;
    let mut workspace_batch_cotangents = attn_cotangent.clone();
    workspace_batch_cotangents.extend_from_slice(&second_workspace_cotangent);
    let workspace_batch_vjp = workspace_lens_session.workspace_vjp_batch(
        &workspace_forward,
        attn_layer,
        &workspace_sources,
        &workspace_batch_cotangents,
        2,
        WorkspaceLensRule::Relp,
    )?;
    let mut workspace_batch_serial_max_abs_error = 0.0f32;
    for source_slot in 0..workspace_sources.len() {
        for (query_slot, serial) in [&workspace_r_vjp, &second_workspace_vjp]
            .into_iter()
            .enumerate()
        {
            let serial = serial
                .source_values(source_slot)
                .ok_or("serial workspace VJP omitted source slot")?;
            let batched = workspace_batch_vjp
                .source_query_values(source_slot, query_slot)
                .ok_or("batched workspace VJP omitted source/query slot")?;
            for (&serial, &batched) in serial.iter().zip(batched) {
                if !serial.is_finite() || !batched.is_finite() {
                    return Err(
                        "batched workspace comparison encountered a non-finite value".into(),
                    );
                }
                workspace_batch_serial_max_abs_error =
                    workspace_batch_serial_max_abs_error.max((serial - batched).abs());
            }
        }
    }
    if workspace_batch_serial_max_abs_error > 1e-6
        || workspace_batch_vjp.diagnostics != workspace_r_vjp.diagnostics
    {
        return Err(format!(
            "batched workspace VJP drifted from serial: values={workspace_batch_serial_max_abs_error} batch_diagnostics={:?} serial_diagnostics={:?}",
            workspace_batch_vjp.diagnostics, workspace_r_vjp.diagnostics
        )
        .into());
    }
    let fit_output_rows = [0, 1, 2];
    let mut fit_covectors = vec![0.0f32; fit_output_rows.len() * workspace_forward.hidden_size()];
    for (query, &row) in fit_output_rows.iter().enumerate() {
        fit_covectors[query * workspace_forward.hidden_size() + row as usize] = 1.0;
    }
    let mut workspace_fit_batch_errors = Vec::new();
    for rule in [WorkspaceLensRule::Jacobian, WorkspaceLensRule::Relp] {
        let serial = workspace_lens_session.workspace_fit_rows(
            &workspace_forward,
            attn_layer,
            &workspace_sources,
            &fit_output_rows,
            1,
            rule,
        )?;
        let batched = workspace_lens_session.workspace_fit_rows_batched(
            &workspace_forward,
            attn_layer,
            &workspace_sources,
            &fit_output_rows,
            1,
            2,
            rule,
        )?;
        let arbitrary = workspace_lens_session.workspace_fit_readouts_batched(
            &workspace_forward,
            attn_layer,
            &workspace_sources,
            &fit_covectors,
            1,
            2,
            rule,
        )?;
        if serial.target_layer != batched.target_layer
            || serial.source_layers != batched.source_layers
            || serial.output_rows != batched.output_rows
            || serial.n_tokens != batched.n_tokens
            || serial.n_valid_positions != batched.n_valid_positions
            || serial.hidden_size != batched.hidden_size
            || serial.diagnostics != batched.diagnostics
        {
            return Err(format!("{rule:?} batched workspace row metadata drifted").into());
        }
        if arbitrary.target_layer != batched.target_layer
            || arbitrary.source_layers != batched.source_layers
            || arbitrary.n_query != batched.output_rows.len()
            || arbitrary.n_tokens != batched.n_tokens
            || arbitrary.n_valid_positions != batched.n_valid_positions
            || arbitrary.hidden_size != batched.hidden_size
            || arbitrary.diagnostics != batched.diagnostics
            || arbitrary.values != batched.values
        {
            return Err(format!(
                "{rule:?} one-hot arbitrary workspace readouts drifted from basis-row fitting"
            )
            .into());
        }
        let mut max_abs_error = 0.0f32;
        for (&serial, &batched) in serial.values.iter().zip(&batched.values) {
            if !serial.is_finite() || !batched.is_finite() {
                return Err(format!("{rule:?} workspace row comparison is non-finite").into());
            }
            max_abs_error = max_abs_error.max((serial - batched).abs());
        }
        if serial.values.len() != batched.values.len() || max_abs_error > 1e-6 {
            return Err(format!(
                "{rule:?} batched workspace rows drifted from serial: values={max_abs_error} serial_len={} batch_len={}",
                serial.values.len(),
                batched.values.len()
            )
            .into());
        }
        workspace_fit_batch_errors.push(json!({
            "rule": format!("{rule:?}"),
            "rows": fit_output_rows,
            "dim_batch": 2,
            "max_abs_error": max_abs_error,
        }));
    }
    let workspace_source_zero = workspace_r_vjp
        .source_values(0)
        .ok_or("workspace VJP omitted source slot zero")?;
    let workspace_source_zero_norm = workspace_source_zero
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    let workspace_capture_max_abs_error = workspace_forward
        .post_block_residuals(attn_layer)
        .ok_or("workspace capture omitted attention target layer")?
        .iter()
        .zip(attn_forward.post_block_residuals())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let workspace_one_block_max_abs_error = workspace_r_vjp
        .source_values(2)
        .ok_or("workspace VJP omitted nearest source slot")?
        .iter()
        .zip(&attn_block_r_vjp.values)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    if !workspace_source_zero_norm.is_finite()
        || !workspace_capture_max_abs_error.is_finite()
        || !workspace_one_block_max_abs_error.is_finite()
        || workspace_capture_max_abs_error > 1e-6
        || workspace_one_block_max_abs_error > 1e-6
        || workspace_r_vjp.diagnostics.iter().any(|diagnostic| {
            !diagnostic.residual_replay_max_abs_error.is_finite()
                || diagnostic.residual_replay_max_abs_error > 5e-3
        })
    {
        return Err(format!(
            "workspace chain failed: norm={workspace_source_zero_norm} capture={workspace_capture_max_abs_error} one_block={workspace_one_block_max_abs_error} diagnostics={:?}",
            workspace_r_vjp.diagnostics
        )
        .into());
    }
    if model.file_name().and_then(|name| name.to_str()) == Some("Qwen3.8-27B-Q8_0.gguf")
        && token_id == 0
        && ((workspace_source_zero_norm - 2.595_755_330_594_522_5).abs() > 1e-4
            || workspace_r_vjp
                .diagnostics
                .iter()
                .map(|diagnostic| (diagnostic.layer, diagnostic.kind))
                .collect::<Vec<_>>()
                != [
                    (3, WorkspaceLensBlockKind::Attention),
                    (2, WorkspaceLensBlockKind::Gdn),
                    (1, WorkspaceLensBlockKind::Gdn),
                ])
    {
        return Err(format!(
            "27B mixed workspace fixture drifted: norm={} diagnostics={:?}",
            workspace_source_zero_norm, workspace_r_vjp.diagnostics
        )
        .into());
    }
    let workspace_full = if std::env::var_os("QWEN_WORKSPACE_FULL_CHAIN").is_some() {
        let target_layer = arch.n_layer - 1;
        let full = workspace_lens_session.workspace_vjp(
            &workspace_forward,
            target_layer,
            &[0],
            &attn_cotangent,
            WorkspaceLensRule::Relp,
        )?;
        let source = full
            .source_values(0)
            .ok_or("full workspace VJP omitted source layer zero")?;
        let norm = source
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        let max_replay_error = full
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.residual_replay_max_abs_error)
            .fold(0.0f32, f32::max);
        let max_replay_diagnostic = full
            .diagnostics
            .iter()
            .max_by(|left, right| {
                left.residual_replay_max_abs_error
                    .total_cmp(&right.residual_replay_max_abs_error)
            })
            .ok_or("full workspace VJP returned no diagnostics")?;
        if !norm.is_finite()
            || !max_replay_error.is_finite()
            || full.diagnostics.len() != target_layer as usize
        {
            return Err(format!(
                "full workspace chain failed: norm={norm} replay={max_replay_error} blocks={}",
                full.diagnostics.len()
            )
            .into());
        }
        if model.file_name().and_then(|name| name.to_str()) == Some("Qwen3.8-27B-Q8_0.gguf")
            && token_id == 0
            && (norm - 5.784_035_851_424_877).abs() > 1e-4
        {
            return Err(format!("27B full workspace fixture drifted: norm={norm}").into());
        }
        Some(json!({
            "target_layer": target_layer,
            "source_layer": 0,
            "tokens": full.n_tokens,
            "traversed_blocks": full.diagnostics.len(),
            "source_l2_norm": norm,
            "max_residual_replay_abs_error": max_replay_error,
            "max_replay_layer": max_replay_diagnostic.layer,
            "max_replay_kind": format!("{:?}", max_replay_diagnostic.kind),
        }))
    } else {
        None
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "qwen.workspace_lens_smoke.v5",
            "model": model,
            "identity": {
                "scheme": WORKSPACE_LENS_IDENTITY_SCHEME,
                "model_locator_id": format!("{:016x}", identity.model_locator_id),
                "tokenizer_metadata_id": format!("{:016x}", identity.tokenizer_metadata_id),
                "content_authenticated": identity.content_authenticated,
            },
            "arch": {
                "layers": arch.n_layer,
                "hidden_size": arch.hidden_size,
                "vocab_size": arch.vocab_size,
            },
            "resident_linears": {
                "count": linears.len(),
                "q8_0_count": linears.iter().filter(|linear| linear.dtype == GgmlType::Q8_0).count(),
                "lm_head_dtype": format!("{:?}", lm_head.dtype),
                "lm_head_shape": lm_head.shape,
            },
            "forward": {
                "position": forward.position,
                "token_id": forward.token_id,
                "capture_layers": forward.capture.layer_ids,
                "capture_norms": capture_norms,
                "top_token": top_token,
                "top_logit": top_logit,
            },
            "lm_head_one_hot_vjp": {
                "output_token": top_token,
                "input_width": lm_head_vjp.len(),
                "l2_norm": lm_head_vjp_norm,
            },
            "last_dense_ffn_r_vjp": {
                "layer": dense_ffn_r_vjp.layer,
                "query_count": dense_ffn_r_vjp.n_query,
                "input_width": dense_ffn_r_vjp.values.len(),
                "l2_norm": dense_ffn_r_vjp_norm,
            },
            "gdn_mixer_r_vjp": {
                "layer": gdn_r_vjp.layer,
                "tokens": gdn_r_vjp.n_tokens,
                "input_width": gdn_r_vjp.values.len(),
                "l2_norm": gdn_r_vjp_norm,
                "residual_replay_max_abs_error": gdn_r_vjp.residual_replay_max_abs_error,
                "final_conv_state_max_abs_error": gdn_r_vjp.final_conv_state_max_abs_error,
                "final_recurrence_state_max_abs_error": gdn_r_vjp.final_recurrence_state_max_abs_error,
            },
            "gdn_block_r_vjp": {
                "layer": gdn_block_r_vjp.layer,
                "tokens": gdn_block_r_vjp.n_tokens,
                "input_width": gdn_block_r_vjp.values.len(),
                "l2_norm": gdn_block_r_vjp_norm,
            },
            "attention_block_r_vjp": {
                "layer": attn_block_r_vjp.layer,
                "tokens": attn_block_r_vjp.n_tokens,
                "input_width": attn_block_r_vjp.values.len(),
                "l2_norm": attn_block_r_vjp_norm,
                "f32_oracle_residual_replay_max_abs_error": attn_block_r_vjp.residual_replay_max_abs_error,
            },
            "workspace_mixed_r_vjp": {
                "target_layer": workspace_r_vjp.target_layer,
                "source_layers": workspace_r_vjp.source_layers,
                "tokens": workspace_r_vjp.n_tokens,
                "source_zero_input_width": workspace_source_zero.len(),
                "source_zero_l2_norm": workspace_source_zero_norm,
                "capture_max_abs_error": workspace_capture_max_abs_error,
                "nearest_source_one_block_max_abs_error": workspace_one_block_max_abs_error,
                "batch_two_serial_max_abs_error": workspace_batch_serial_max_abs_error,
                "fit_batch_checks": workspace_fit_batch_errors,
                "diagnostics": workspace_r_vjp.diagnostics.iter().map(|diagnostic| json!({
                    "layer": diagnostic.layer,
                    "kind": format!("{:?}", diagnostic.kind),
                    "residual_replay_max_abs_error": diagnostic.residual_replay_max_abs_error,
                })).collect::<Vec<_>>(),
            },
            "workspace_full_r_vjp": workspace_full,
        }))?
    );
    Ok(())
}

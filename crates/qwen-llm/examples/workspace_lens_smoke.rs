use qwen_llm::research::{
    DenseFfnVjpRule, GdnMixerVjpRule, LinearRole, RESEARCH_IDENTITY_SCHEME, ResearchLinear,
};
use qwen_llm::runtime::{Runtime, SequenceConfig};
use qwen_llm::tensor::GgmlType;
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
    let mut research = loaded.research_session(&mut sequence)?;
    let identity = research.identity();
    let linears = research.linears()?;
    let capture_layers = [0, arch.n_layer.saturating_sub(2), arch.n_layer - 1];
    let forward = research.forward_token_with_dense_ffn_capture(token_id, &capture_layers)?;
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

    let lm_head = research.linear_info(ResearchLinear::LmHead)?;
    let mut cotangent = vec![0.0f32; lm_head.shape[1]];
    cotangent[top_token] = 1.0;
    let lm_head_vjp = research.frozen_linear_vjp(ResearchLinear::LmHead, &cotangent, 1)?;
    let lm_head_vjp_norm = lm_head_vjp
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    let last_capture_offset = (capture_layers.len() - 1) * forward.capture.hidden_size;
    let last_pre_ffn = &forward.capture.pre_ffn_residuals
        [last_capture_offset..last_capture_offset + forward.capture.hidden_size];
    let mut hidden_cotangent = vec![0.0f32; forward.capture.hidden_size];
    let hidden_coordinate = top_token % hidden_cotangent.len();
    hidden_cotangent[hidden_coordinate] = 1.0;
    let dense_ffn_r_vjp = research.dense_ffn_vjp(
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
            research
                .linear_info(ResearchLinear::Layer {
                    index: layer,
                    role: LinearRole::GdnQkv,
                })
                .is_ok()
        })
        .ok_or("model has no nonzero GDN layer")?;
    let gdn_forward = research.forward_prompt_with_gdn_capture(&[token_id, token_id], gdn_layer)?;
    let mut gdn_cotangent = vec![0.0f32; 2 * gdn_forward.hidden_size()];
    for token in 0..2 {
        gdn_cotangent[token * gdn_forward.hidden_size()
            + (hidden_coordinate + token) % gdn_forward.hidden_size()] = 1.0;
    }
    let gdn_r_vjp = research.gdn_mixer_vjp(&gdn_forward, &gdn_cotangent, GdnMixerVjpRule::Relp)?;
    let gdn_r_vjp_norm = gdn_r_vjp
        .values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if !gdn_r_vjp.residual_replay_max_abs_error.is_finite()
        || !gdn_r_vjp.final_conv_state_max_abs_error.is_finite()
        || !gdn_r_vjp.final_recurrence_state_max_abs_error.is_finite()
        || gdn_r_vjp.residual_replay_max_abs_error > 1e-5
        || gdn_r_vjp.final_conv_state_max_abs_error > 1e-5
        || gdn_r_vjp.final_recurrence_state_max_abs_error > 1e-5
    {
        return Err(format!(
            "GDN replay drift residual={} conv={} recurrence={}",
            gdn_r_vjp.residual_replay_max_abs_error,
            gdn_r_vjp.final_conv_state_max_abs_error,
            gdn_r_vjp.final_recurrence_state_max_abs_error,
        )
        .into());
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "qwen.workspace_lens_smoke.v2",
            "model": model,
            "identity": {
                "scheme": RESEARCH_IDENTITY_SCHEME,
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
            }
        }))?
    );
    Ok(())
}

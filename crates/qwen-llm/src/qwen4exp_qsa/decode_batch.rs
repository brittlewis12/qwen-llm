use super::*;
pub(super) mod attention;
pub(super) mod selection;
use crate::qwen4exp_metal::decode_projection::Scratch;
pub(crate) fn encode<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    weights: QwenSparseAttentionMetalWeights<'_>,
    mut members: Vec<&'a mut QwenSparseAttentionMetalWorkspace>,
    scratch: &Scratch,
) -> Result<Vec<QwenSparseAttentionMetalRead<'a>>, Qwen4ExpQsaError> {
    validate_encoder(ctx, enc)?;
    for (w, x) in members.iter().zip(inputs) {
        w.require_idle()?;
        if w.state_poisoned
            || w.geometry != weights.geometry
            || w.committed_length >= w.geometry.capacity
        {
            return invalid("native QSA member is not usable");
        }
        validate_contract(ctx, x, weights, w)?;
        preflight(ctx, weights)?;
    }
    let mut controls = Vec::with_capacity(members.len());
    for w in &mut members {
        let (p, n) = prepare_control_scalars(w)?;
        reserve_command(w, enc, n, 0)?;
        controls.push((p, n));
    }
    for (weight, field) in [
        (weights.index_query, 0),
        (weights.index_key, 1),
        (weights.query, 2),
        (weights.key, 3),
        (weights.value, 4),
    ] {
        let outputs: Vec<_> = members
            .iter()
            .map(|w| match field {
                0 => &w.index_query_raw,
                1 => &w.index_key_raw,
                2 => &w.query_gate_projection,
                3 => &w.key_raw,
                _ => &w.value,
            })
            .collect();
        scratch.project(ctx, enc, weight, inputs, &outputs)?;
    }
    let batch_attention = attention::eligible(weights.geometry, &members, &controls);
    let batch_selection = selection::eligible(ctx, weights.geometry, &controls);
    for ((w, x), &(p, n)) in members.iter_mut().zip(inputs).zip(&controls) {
        if batch_attention || batch_selection {
            encode_step_parts(ctx, enc, x, weights, w, p, n, true, false, !batch_selection)?;
        } else {
            encode_step_inner(ctx, enc, x, weights, w, p, n, true)?;
        }
    }
    if batch_selection {
        selection::encode(ctx, enc, &members, &controls)?;
    }
    if batch_attention {
        attention::encode(ctx, enc, &members, &controls)?;
    } else if batch_selection {
        for (w, &(p, n)) in members.iter().zip(&controls) {
            encode_attention_step(ctx, enc, w, p, n)?;
        }
    }
    let attention: Vec<_> = members.iter().map(|w| &w.attention).collect();
    let outputs: Vec<_> = members.iter().map(|w| &w.output).collect();
    scratch.project(ctx, enc, weights.output, &attention, &outputs)?;
    Ok(members
        .into_iter()
        .zip(controls)
        .map(|(workspace, (position, _))| QwenSparseAttentionMetalRead {
            workspace,
            position,
        })
        .collect())
}

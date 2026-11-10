use super::*;
pub(super) mod attention;
pub(super) mod norms;
pub(super) mod selection;
use crate::qwen4exp_metal::decode_projection::Scratch;
pub(crate) fn encode_into<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    positions: &[usize],
    weights: QwenSparseAttentionMetalWeights<'_>,
    mut members: Vec<&'a mut QwenSparseAttentionMetalWorkspace>,
    scratch: &Scratch,
    destinations: &[&MetalTensor],
    direct: bool,
) -> Result<(), Qwen4ExpQsaError> {
    if destinations.len() != members.len() || positions.len() != members.len() {
        return invalid("native destination/member count differs");
    }
    if direct {
        for (w, destination) in members.iter().zip(destinations) {
            validate_output_destination(ctx, w, destination)?;
        }
    }
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
    for (i, w) in members.iter_mut().enumerate() {
        // A device-owned frame already wrote this member's position-dependent
        // control scalars in the GPU input prefix. The CPU workspace's
        // committed length lags one frame behind the pipeline.
        let (p, n) = if enc.gpu_owned() {
            let p = positions[i];
            if p >= w.geometry.capacity {
                return invalid("native QSA pipeline position exceeds capacity");
            }
            (p, p + 1)
        } else {
            let (p, n) = prepare_control_scalars(w)?;
            reserve_command(w, enc, n, 0)?;
            (p, n)
        };
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
    let batch_norms = norms::eligible(ctx, weights.geometry, members.len());
    if batch_norms {
        norms::encode(ctx, enc, weights, &members, &controls)?;
    }
    let batch_attention = attention::eligible(weights.geometry, &members, &controls);
    let batch_selection = selection::eligible(ctx, weights.geometry, &controls);
    for ((w, x), &(p, n)) in members.iter_mut().zip(inputs).zip(&controls) {
        if batch_attention || batch_selection || batch_norms {
            encode_step_parts(
                ctx,
                enc,
                x,
                weights,
                w,
                p,
                n,
                true,
                false,
                !batch_selection,
                batch_norms,
            )?;
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
    let private: Vec<_> = if direct {
        Vec::new()
    } else {
        members.iter().map(|w| &w.output).collect()
    };
    scratch.project(
        ctx,
        enc,
        weights.output,
        &attention,
        if direct { destinations } else { &private },
    )?;
    drop(private);
    if direct {
        #[cfg(feature = "perf-ablations")]
        crate::metal::fn_batch_probe::observe("direct_outputs", members.len());
    } else {
        for ((workspace, (position, _)), destination) in
            members.into_iter().zip(controls).zip(destinations)
        {
            QwenSparseAttentionMetalRead {
                workspace,
                position,
            }
            .output()
            .encode_copy_to(ctx, enc, destination)?;
        }
    }
    Ok(())
}

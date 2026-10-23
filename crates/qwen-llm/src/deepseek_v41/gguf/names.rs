use super::*;

fn block(name: &str) -> Option<(usize, &str)> {
    let (layer, tail) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((layer.parse().ok()?, tail))
}
pub(super) fn bank(name: &str) -> Option<(usize, &'static str)> {
    let (layer, tail) = block(name)?;
    let part = match tail {
        "ffn_gate_exps.weight" => "w1",
        "ffn_up_exps.weight" => "w3",
        "ffn_down_exps.weight" => "w2",
        _ => return None,
    };
    Some((layer, part))
}
pub(super) fn dense(name: &str, layers: usize) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("embed.weight".into()),
        "output_norm.weight" => return Some("norm.weight".into()),
        "output.weight" => return Some("head.weight".into()),
        _ => (),
    }
    let (layer, tail) = block(name)?;
    if layer >= layers {
        return None;
    }
    let suffix = match tail {
        "attn_sinks.weight" => "attn.attn_sink",
        "attn_q_a.weight" => "attn.wq_a.weight",
        "attn_q_b.weight" => "attn.wq_b.weight",
        "attn_q_a_norm.weight" => "attn.q_norm.weight",
        "attn_kv.weight" => "attn.wkv.weight",
        "attn_kv_a_norm.weight" => "attn.kv_norm.weight",
        "attn_output_a.weight" => "attn.wo_a.weight",
        "attn_output_b.weight" => "attn.wo_b.weight",
        "attn_compressor_kv.weight" => "attn.compressor.wkv.weight",
        "attn_compressor_gate.weight" => "attn.compressor.wgate.weight",
        "attn_compressor_norm.weight" => "attn.compressor.norm.weight",
        "indexer.attn_k.weight" => "attn.indexer.wk.weight",
        "indexer.k_norm.weight" => "attn.indexer.k_norm.weight",
        "indexer.attn_q_b.weight" => "attn.indexer.wq_b.weight",
        "indexer.proj.weight" => "attn.indexer.weights_proj.weight",
        "ffn_gate_inp.weight" => "ffn.gate.weight",
        "exp_probs_b.bias" => "ffn.gate.bias",
        "ffn_gate_shexp.weight" => "ffn.shared_experts.w1.weight",
        "ffn_up_shexp.weight" => "ffn.shared_experts.w3.weight",
        "ffn_down_shexp.weight" => "ffn.shared_experts.w2.weight",
        "engram_q_norm.weight" | "engram_q.weight" => "engram.q_weight",
        "engram_k_norm.weight" | "engram_k.weight" => "engram.k_weight",
        "engram_gate.weight" => "engram.gate.weight",
        "engram_wkv.weight" | "engram_kv.weight" => "engram.wkv.weight",
        "attn_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "hc_attn_fn.weight" => "hc_attn_fn",
        "hc_attn_scale.weight" => "hc_attn_scale",
        "hc_attn_base.weight" => "hc_attn_base",
        "hc_ffn_fn.weight" => "hc_ffn_fn",
        "hc_ffn_scale.weight" => "hc_ffn_scale",
        "hc_ffn_base.weight" => "hc_ffn_base",
        _ => return None,
    };
    Some(format!("layers.{layer}.{suffix}"))
}
pub(super) fn validate_shape(t: &TensorDesc, c: &Config) -> Result<(), LoadError> {
    let Some(expected) = expected_shape(&t.name, c) else {
        return Ok(());
    };
    if !shape_matches(&t.shape, &expected) {
        return Err(invalid(format!(
            "{} shape {:?}, expected {expected:?}",
            t.name, t.shape
        )));
    }
    Ok(())
}
pub(super) fn expected_shape(name: &str, c: &Config) -> Option<Vec<u64>> {
    let h = c.hidden as u64;
    let i = c.intermediate as u64;
    let d = c.head_dim as u64;
    let q = c.query_rank as u64;
    let heads = c.heads as u64;
    let groups = c.groups as u64;
    let rank = c.output_rank as u64;
    let ih = c.index_heads as u64;
    let id = c.index_dim as u64;
    let e = c.experts as u64;
    let tail = block(name).map(|(_, tail)| tail).unwrap_or(name);
    let expected = match tail {
        "token_embd.weight" | "output.weight" => vec![h, c.vocab as u64],
        "output_norm.weight" | "attn_norm.weight" | "ffn_norm.weight" => vec![h],
        "attn_sinks.weight" => vec![heads],
        "attn_q_a.weight" => vec![h, q],
        "attn_q_b.weight" => vec![q, heads * d],
        "attn_q_a_norm.weight" => vec![q],
        "attn_kv.weight" | "attn_compressor_kv.weight" | "attn_compressor_gate.weight" => {
            vec![h, d]
        }
        "attn_kv_a_norm.weight" | "attn_compressor_norm.weight" => vec![d],
        "attn_output_a.weight" => vec![heads * d / groups, groups * rank],
        "attn_output_b.weight" => vec![groups * rank, h],
        "indexer.attn_k.weight" => vec![d, id],
        "indexer.k_norm.weight" => vec![id],
        "indexer.attn_q_b.weight" => vec![q, ih * id],
        "indexer.proj.weight" => vec![h, ih],
        "ffn_gate_inp.weight" => vec![h, e],
        "exp_probs_b.bias" => vec![e],
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" => vec![h, i],
        "ffn_down_shexp.weight" => vec![i, h],
        "engram_q_norm.weight"
        | "engram_k_norm.weight"
        | "engram_q.weight"
        | "engram_k.weight"
        | "engram_gate.weight" => vec![h, 4],
        "engram_wkv.weight" | "engram_kv.weight" => vec![6144, 5 * h],
        "hc_attn_fn.weight" | "hc_ffn_fn.weight" => vec![4 * h, 24],
        "hc_attn_scale.weight" | "hc_ffn_scale.weight" => vec![3],
        "hc_attn_base.weight" | "hc_ffn_base.weight" => vec![24],
        _ => return None,
    };
    Some(expected)
}

pub(super) fn required_dense(schedule: &Schedule, engram_layers: &[usize]) -> Vec<String> {
    let mut names = vec![
        "token_embd.weight".into(),
        "output_norm.weight".into(),
        "output.weight".into(),
    ];
    for l in 0..schedule.layers {
        let mut tails = vec![
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_sinks.weight",
            "attn_q_a.weight",
            "attn_q_b.weight",
            "attn_q_a_norm.weight",
            "attn_kv.weight",
            "attn_kv_a_norm.weight",
            "attn_output_a.weight",
            "attn_output_b.weight",
            "ffn_gate_inp.weight",
            "exp_probs_b.bias",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "hc_attn_fn.weight",
            "hc_attn_scale.weight",
            "hc_attn_base.weight",
            "hc_ffn_fn.weight",
            "hc_ffn_scale.weight",
            "hc_ffn_base.weight",
        ];
        if schedule.owns_kv(l) {
            tails.extend([
                "attn_compressor_kv.weight",
                "attn_compressor_norm.weight",
                "indexer.attn_k.weight",
                "indexer.k_norm.weight",
            ]);
            if schedule.ratio(l) == 2 {
                tails.push("attn_compressor_gate.weight");
            }
        }
        if schedule.produces_index(l) {
            tails.extend(["indexer.attn_q_b.weight", "indexer.proj.weight"]);
        }
        if engram_layers.contains(&l) {
            tails.push("engram_kv.weight");
        }
        names.extend(tails.into_iter().map(|tail| format!("blk.{l}.{tail}")));
    }
    names
}

pub(super) fn validate_inventory(
    p: &Plan,
    dense: &BTreeMap<String, &TensorDesc>,
) -> Result<(), LoadError> {
    let layers: Vec<_> = p.engram_hash.layers.iter().map(|l| l.layer).collect();
    for name in required_dense(&p.schedule, &layers) {
        let key = self::dense(&name, p.layers).unwrap();
        if !dense.contains_key(&key) {
            return Err(invalid(format!("missing dense tensor {name}")));
        }
    }
    for l in layers {
        let prefix = format!("layers.{l}.engram");
        let folded = dense.contains_key(&format!("{prefix}.gate.weight"));
        let q = dense.contains_key(&format!("{prefix}.q_weight"));
        let k = dense.contains_key(&format!("{prefix}.k_weight"));
        if !((folded && !q && !k) || (!folded && q && k)) {
            return Err(invalid(format!(
                "layer {l} needs either folded Engram gate or separate q/k"
            )));
        }
    }
    Ok(())
}

use super::*;

fn optional_uint(g: &GgufFile, short: &str) -> Result<Option<u64>, LoadError> {
    let key = format!("deepseek41.{short}");
    if g.model.metadata().get(&key).is_some() {
        Ok(Some(g.get_u64(&key).ok_or_else(|| {
            invalid(format!("{key} must be unsigned integer"))
        })?))
    } else {
        Ok(None)
    }
}
fn optional_uint_aliases(g: &GgufFile, keys: &[&str]) -> Result<Option<u64>, LoadError> {
    let mut found = None;
    for key in keys {
        if let Some(n) = optional_uint(g, key)? {
            if found.is_some_and(|old| old != n) {
                return Err(invalid(format!("conflicting aliases for deepseek41.{key}")));
            }
            found = Some(n);
        }
    }
    Ok(found)
}

fn uint(g: &GgufFile, keys: &[&str], default: Option<usize>) -> Result<usize, LoadError> {
    let mut found = None;
    for key in keys {
        let key = format!("deepseek41.{key}");
        if g.model.metadata().get(&key).is_some() {
            let n = g
                .get_u64(&key)
                .ok_or_else(|| invalid(format!("{key} must be unsigned integer")))?;
            let n = usize::try_from(n).map_err(|_| invalid("dimension overflow"))?;
            if found.is_some_and(|old| old != n) {
                return Err(invalid(format!("conflicting aliases for {key}")));
            }
            found = Some(n);
        }
    }
    let n = found
        .or(default)
        .ok_or_else(|| invalid(format!("missing deepseek41.{}", keys[0])))?;
    if n == 0 {
        return Err(invalid(format!("zero {}", keys[0])));
    }
    Ok(n)
}
fn float(g: &GgufFile, keys: &[&str], default: Option<f32>) -> Result<f32, LoadError> {
    let mut found = None;
    for key in keys {
        let key = format!("deepseek41.{key}");
        if g.model.metadata().get(&key).is_some() {
            let n = if let Some(n) = g.get_u64(&key) {
                n as f32
            } else {
                g.get_f64(&key)?
                    .ok_or_else(|| invalid(format!("non-numeric {key}")))? as f32
            };
            if !n.is_finite() || n <= 0.0 {
                return Err(invalid(format!("invalid {key}")));
            }
            if found.is_some_and(|old| old != n) {
                return Err(invalid(format!("conflicting aliases for {key}")));
            }
            found = Some(n);
        }
    }
    found
        .or(default)
        .ok_or_else(|| invalid(format!("missing deepseek41.{}", keys[0])))
}
pub(super) fn array(g: &GgufFile, keys: &[&str]) -> Result<Option<Vec<u64>>, LoadError> {
    let mut result = None;
    for key in keys {
        if let Some(values) = g.get_u64_array(&format!("deepseek41.{key}"))? {
            if result.as_ref().is_some_and(|old| old != &values) {
                return Err(invalid("conflicting array aliases"));
            }
            result = Some(values);
        }
    }
    Ok(result)
}
fn sources(
    g: &GgufFile,
    keys: &[&str],
    defaults: Option<&[usize]>,
    layers: usize,
) -> Result<Vec<u64>, LoadError> {
    let values = match array(g, keys)? {
        Some(values) => values,
        None => defaults
            .ok_or_else(|| {
                invalid(format!(
                    "missing deepseek41.{} for non-default layout",
                    keys[0]
                ))
            })?
            .iter()
            .map(|&n| n as u64)
            .collect(),
    };
    if values.iter().any(|&n| n >= layers as u64) || values.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(format!("invalid layer schedule {}", keys[0])));
    }
    Ok(values)
}
fn local_ple_offsets(mut offsets: Vec<u64>, primes: &[u64]) -> Result<Vec<u64>, LoadError> {
    let mut end = 0u64;
    for (offsets, primes) in offsets.chunks_exact_mut(24).zip(primes.chunks_exact(24)) {
        let base = end;
        for (offset, &prime) in offsets.iter_mut().zip(primes) {
            if *offset != end {
                return Err(invalid(
                    "ple.head_offsets must be contiguous global offsets starting at zero",
                ));
            }
            *offset = offset
                .checked_sub(base)
                .ok_or_else(|| invalid("PLE local offset underflow"))?;
            end = end
                .checked_add(prime)
                .ok_or_else(|| invalid("PLE global offset overflow"))?;
        }
    }
    Ok(offsets)
}
pub(super) fn parse(g: &GgufFile) -> Result<Plan, LoadError> {
    if g.architecture().as_deref() != Some("deepseek41") {
        return Err(invalid("general.architecture must be deepseek41"));
    }
    let d = Config::default();
    for (short, expected) in [
        ("scoring_func", "sqrtsoftplus"),
        ("topk_method", "noaux_tc"),
        ("hidden_act", "silu"),
        ("rope.scaling.type", "yarn"),
    ] {
        let key = format!("deepseek41.{short}");
        if g.model.metadata().get(&key).is_some() && g.get_str(&key) != Some(expected) {
            return Err(invalid(format!(
                "unsupported {key}: {:?}; implemented value is {expected}",
                g.get_str(&key)
            )));
        }
    }
    for key in [
        "deepseek41.norm_topk_prob",
        "deepseek41.expert_weights_norm",
    ] {
        if g.get_bool(key)?.is_some_and(|value| !value) {
            return Err(invalid(format!(
                "unsupported {key}=false; normalized top-k routing is implemented"
            )));
        }
    }
    // PR28696 uses GGUF ExpertGatingFuncType.SQRTSOFTPLUS = 4.
    if optional_uint(g, "expert_gating_func")?.is_some_and(|value| value != 4) {
        return Err(invalid(
            "unsupported deepseek41.expert_gating_func; SQRTSOFTPLUS is numeric value 4",
        ));
    }
    if uint(g, &["attention.head_count_kv", "num_key_value_heads"], None)? != 1 {
        return Err(invalid(
            "unsupported KV head count; attention implements one shared KV head",
        ));
    }
    engram::validate_encoding(g)?;
    if uint(g, &["hyper_connection.count", "hc_mult"], None)? != 4 {
        return Err(invalid("mHC kernels require four residual streams"));
    }
    if uint(g, &["expert_shared_count", "n_shared_experts"], None)? != 1 {
        return Err(invalid("FFN binding supports one shared expert"));
    }
    let emb = g.find("token_embd.weight");
    let hidden = uint(g, &["embedding_length", "hidden_size"], None)?;
    let intermediate = uint(
        g,
        &["expert_feed_forward_length", "moe_intermediate_size"],
        None,
    )?;
    let vocab = uint(
        g,
        &["vocab_size"],
        emb.and_then(|t| t.shape.get(1)).map(|&n| n as usize),
    )?;
    let experts = uint(g, &["expert_count", "n_routed_experts"], None)?;
    let mut config = Config {
        hidden,
        intermediate,
        vocab,
        experts,
        heads: uint(g, &["attention.head_count", "num_attention_heads"], None)?,
        head_dim: uint(g, &["attention.key_length", "head_dim"], None)?,
        rope_dim: uint(g, &["rope.dimension_count", "qk_rope_head_dim"], None)?,
        query_rank: uint(g, &["attention.q_lora_rank", "q_lora_rank"], None)?,
        output_rank: uint(g, &["attention.output_lora_rank", "o_lora_rank"], None)?,
        groups: uint(g, &["attention.output_group_count", "o_groups"], None)?,
        topk: uint(g, &["expert_used_count", "num_experts_per_tok"], None)?,
        index_heads: uint(g, &["attention.indexer.head_count", "index_n_heads"], None)?,
        index_dim: uint(g, &["attention.indexer.key_length", "index_head_dim"], None)?,
        index_topk: uint(g, &["attention.indexer.top_k", "index_topk"], None)?,
        window: uint(g, &["attention.sliding_window", "sliding_window"], None)?,
        candidate_block: uint(
            g,
            &[
                "attention.indexer.candidate_block_size",
                "candidate_block_size",
            ],
            Some(d.candidate_block),
        )?,
        candidate_topk: uint(
            g,
            &[
                "attention.indexer.candidate_top_k_blocks",
                "candidate_topk_blocks",
            ],
            Some(d.candidate_topk),
        )?,
        norm_eps: float(
            g,
            &["attention.layer_norm_rms_epsilon", "rms_norm_eps"],
            None,
        )?,
        hc_eps: float(g, &["hyper_connection.epsilon", "hc_eps"], None)?,
        hc_iterations: uint(
            g,
            &["hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters"],
            None,
        )?,
        clamp: float(g, &["swiglu_limit"], Some(d.clamp))?,
        route_scale: float(g, &["expert_weights_scale", "routed_scaling_factor"], None)?,
        gate_temperature: float(g, &["gate_temperature"], Some(d.gate_temperature))?,
        rope_theta: float(g, &["rope.freq_base", "rope_theta"], None)?,
        compress_theta: float(
            g,
            &["attention.compress_rope_freq_base", "compress_rope_theta"],
            None,
        )?,
        original_context: float(
            g,
            &[
                "rope.scaling.original_context_length",
                "rope_scaling.original_max_position_embeddings",
            ],
            None,
        )? as usize,
        rope_factor: float(g, &["rope.scaling.factor", "rope_scaling.factor"], None)?,
        beta_fast: float(
            g,
            &["rope.scaling.yarn_beta_fast", "rope_scaling.beta_fast"],
            None,
        )?,
        beta_slow: float(
            g,
            &["rope.scaling.yarn_beta_slow", "rope_scaling.beta_slow"],
            None,
        )?,
    };
    let c = &config;
    if optional_uint(g, "attention.value_length")?.is_some_and(|n| n != c.head_dim as u64) {
        return Err(invalid(
            "unsupported attention.value_length; shared KV requires key/value dimensions to match",
        ));
    }
    if c.topk > c.experts
        || c.heads % c.groups != 0
        || c.rope_dim > c.head_dim
        || c.rope_dim > c.index_dim
        || c.rope_dim % 2 != 0
    {
        return Err(invalid("inconsistent attention/expert dimensions"));
    }
    for (a, b) in [
        (c.heads, c.head_dim),
        (c.groups, c.output_rank),
        (c.index_heads, c.index_dim),
        (c.hidden, 5),
    ] {
        a.checked_mul(b)
            .ok_or_else(|| invalid("projection dimension overflow"))?;
    }
    let layers = uint(g, &["block_count", "num_hidden_layers"], None)?;
    let mut declared_clamp = g
        .model
        .metadata()
        .get("deepseek41.swiglu_limit")
        .map(|_| config.clamp);
    for key in [
        "deepseek41.swiglu_clamp_exp",
        "deepseek41.swiglu_clamp_shexp",
    ] {
        if let Some(values) = g.get_f64_array(key)? {
            if values.len() < layers {
                return Err(invalid(format!("{key} shorter than main block count")));
            }
            let value = values[0] as f32;
            if !value.is_finite()
                || value <= 0.0
                || values[..layers].iter().any(|&v| v as f32 != value)
            {
                return Err(invalid(format!(
                    "unsupported {key}; a positive uniform main-layer clamp is implemented"
                )));
            }
            if declared_clamp.is_some_and(|previous| previous != value) {
                return Err(invalid("conflicting shared/routed SwiGLU clamp metadata"));
            }
            config.clamp = value;
            declared_clamp = Some(value);
        }
    }
    if declared_clamp.is_none() {
        return Err(invalid(
            "missing deepseek41.swiglu_limit or swiglu_clamp_exp/shexp",
        ));
    }
    if layers > g.tensors.len() / 3 {
        return Err(invalid("block count exceeds available expert bank triples"));
    }
    let compression_ratios = array(g, &["attention.compress_ratios", "compress_ratios"])?
        .ok_or_else(|| invalid("missing deepseek41.attention.compress_ratios"))?;
    if compression_ratios.len() < layers {
        return Err(invalid("compression ratios shorter than main block count"));
    }
    // The named layout supplies missing ownership only, not an identity constraint.
    let fallback = Schedule::deepseek41_shared_kv_layout(layers, &compression_ratios);
    let kv_source_layers = sources(
        g,
        &["kv_source_layer_ids", "attention.kv_source_layers"],
        fallback.as_ref().map(|s| s.kv_source_layers.as_slice()),
        layers,
    )?;
    let index_source_layers = sources(
        g,
        &["index_source_layer_ids", "attention.indexer.source_layers"],
        fallback.as_ref().map(|s| s.index_source_layers.as_slice()),
        layers,
    )?;
    let candidate_source_layer = optional_uint_aliases(
        g,
        &[
            "candidate_source_layer_id",
            "attention.indexer.candidate_source_layer",
        ],
    )?
    .map(|n| usize::try_from(n).map_err(|_| invalid("candidate source overflow")))
    .transpose()?;
    let candidate_source_layer = match candidate_source_layer {
        Some(n) => Some(n),
        None if !compression_ratios[..layers].contains(&1) => None,
        None => Some(
            fallback
                .as_ref()
                .and_then(|s| s.candidate_source_layer)
                .ok_or_else(|| {
                    invalid("missing deepseek41.candidate_source_layer_id for non-default layout")
                })?,
        ),
    };
    let schedule = Schedule {
        layers,
        kv_source_layers: kv_source_layers.iter().map(|&l| l as usize).collect(),
        index_source_layers: index_source_layers.iter().map(|&l| l as usize).collect(),
        compression_ratios: compression_ratios[..layers]
            .iter()
            .map(|&n| usize::try_from(n).map_err(|_| invalid("compression ratio overflow")))
            .collect::<Result<_, _>>()?,
        candidate_source_layer,
    };
    schedule.validate()?;
    let engram_layers = sources_alias(g, layers)?;
    for (keys, expected) in [
        (
            &["engram.head_count", "engram.n_heads", "ple.heads_per_ngram"][..],
            8,
        ),
        (&["engram.max_ngram_size", "ple.ngram_size"][..], 4),
        (
            &[
                "engram.head_dim",
                "engram.key_length",
                "embedding_length_per_layer_input",
            ][..],
            256,
        ),
    ] {
        let declared = optional_uint_aliases(g, keys)?;
        if !engram_layers.is_empty() {
            let declared =
                declared.ok_or_else(|| invalid(format!("missing deepseek41.{}", keys[0])))?;
            if declared != expected {
                return Err(invalid(format!(
                    "unsupported Engram geometry: {} must be {expected}",
                    keys[0]
                )));
            }
        }
    }
    let token_map = g
        .get_i64_array("deepseek41.engram.token_map")?
        .ok_or_else(|| invalid("missing Engram token_map"))?
        .into_iter()
        .map(|n| u32::try_from(n).map_err(|_| invalid("token_map value outside u32")))
        .collect::<Result<Vec<_>, _>>()?;
    if token_map.len() != vocab {
        return Err(invalid("token_map length differs from vocabulary"));
    }
    if let Some(n) = optional_uint(g, "engram.compressed_vocab_size")? {
        if n == 0 || token_map.iter().any(|&v| u64::from(v) >= n) {
            return Err(invalid("token_map outside compressed vocabulary"));
        }
    }
    // Legacy pad_id is already compressed; recover any raw ID with that image.
    let raw_pad = if let Some(n) = optional_uint(g, "engram.pad_token_id")? {
        u32::try_from(n).map_err(|_| invalid("pad ID overflow"))?
    } else if let Some(n) = optional_uint(g, "engram.pad_id")? {
        token_map
            .iter()
            .position(|&v| u64::from(v) == n)
            .ok_or_else(|| invalid("compressed pad absent from token_map"))? as u32
    } else {
        return Err(invalid(
            "missing deepseek41.engram.pad_id or engram.pad_token_id",
        ));
    };
    if raw_pad as usize >= vocab {
        return Err(invalid("pad outside vocabulary"));
    }
    if let Some(n) = optional_uint(g, "engram.pad_id")? {
        if u64::from(token_map[raw_pad as usize]) != n {
            return Err(invalid("conflicting Engram pad aliases"));
        }
    }
    let hash_array = |keys: &[&str], width| -> Result<Option<Vec<u64>>, LoadError> {
        let values = array(g, keys)?;
        let count = engram_layers
            .len()
            .checked_mul(width)
            .ok_or_else(|| invalid("Engram hash metadata length overflow"))?;
        if values.as_ref().is_some_and(|values| values.len() != count) {
            return Err(invalid(format!("Engram hash metadata length: {}", keys[0])));
        }
        Ok(values)
    };
    let required_hash = |values: Option<Vec<u64>>, key| -> Result<Vec<u64>, LoadError> {
        match values {
            Some(values) => Ok(values),
            None if engram_layers.is_empty() => Ok(Vec::new()),
            None => Err(invalid(format!("missing deepseek41.{key}"))),
        }
    };
    let primes = required_hash(
        hash_array(&["engram.primes", "ple.head_vocab_sizes"], 24)?,
        "engram.primes or ple.head_vocab_sizes",
    )?;
    let legacy_offsets = hash_array(&["engram.offsets"], 24)?;
    let ple_offsets = hash_array(&["ple.head_offsets"], 24)?
        .map(|offsets| local_ple_offsets(offsets, &primes))
        .transpose()?;
    if let (Some(legacy), Some(ple)) = (&legacy_offsets, &ple_offsets) {
        if legacy != ple {
            return Err(invalid("conflicting Engram/PLE offset aliases"));
        }
    }
    let offsets = required_hash(
        legacy_offsets.or(ple_offsets),
        "engram.offsets or ple.head_offsets",
    )?;
    let multipliers = required_hash(
        hash_array(&["engram.multipliers", "ple.layer_multipliers"], 4)?,
        "engram.multipliers or ple.layer_multipliers",
    )?;
    let compressed_vocab_size = optional_uint(g, "engram.compressed_vocab_size")?
        .unwrap_or(u64::from(token_map.iter().copied().max().unwrap_or(0)) + 1);
    let mut engram_hash = HashConfig {
        layers: Vec::new(),
        compressed_vocab_size: u32::try_from(compressed_vocab_size)
            .map_err(|_| invalid("compressed vocabulary overflow"))?,
        max_position: uint(g, &["context_length", "max_position_embeddings"], None)? as u64,
    };
    let declared_rows = array(g, &["engram.num_embeddings", "engram.rows"])?;
    if declared_rows
        .as_ref()
        .is_some_and(|v| v.len() != engram_layers.len())
    {
        return Err(invalid("Engram rows metadata length"));
    }
    for (i, &layer) in engram_layers.iter().enumerate() {
        let primes: [u64; 24] = primes[i * 24..(i + 1) * 24].try_into().unwrap();
        let rows = primes.iter().try_fold(0u64, |end, &prime| {
            end.checked_add(prime)
                .ok_or_else(|| invalid("Engram layer rows overflow"))
        })?;
        if declared_rows.as_ref().is_some_and(|v| v[i] != rows) {
            return Err(invalid("Engram rows metadata differs from bucket total"));
        }
        let multipliers = multipliers[i * 4..(i + 1) * 4].try_into().unwrap();
        let offsets = offsets[i * 24..(i + 1) * 24].try_into().unwrap();
        engram_hash.layers.push(super::super::engram::LayerHash {
            layer: layer as usize,
            primes,
            offsets,
            multipliers,
            rows,
        });
    }
    engram_hash
        .validate()
        .map_err(|e| invalid(format!("Engram hash configuration: {e:?}")))?;
    let engram_tables = engram::resolve_tables(g, &engram_hash, layers)?;
    let engram_primes = engram_hash.layers.iter().flat_map(|l| l.primes).collect();
    let engram_offsets = engram_hash.layers.iter().flat_map(|l| l.offsets).collect();
    let engram_multipliers = engram_hash
        .layers
        .iter()
        .flat_map(|l| l.multipliers)
        .collect();
    Ok(Plan {
        config,
        schedule,
        engram_hash,
        engram_tables,
        dtype_inventory: BTreeMap::new(),
        binding_linears: BTreeMap::new(),
        bytes: ByteCounters::default(),
        token_map,
        raw_pad,
        layers,
        engram_layers,
        engram_primes,
        engram_offsets,
        engram_multipliers,
        kv_source_layers,
        index_source_layers,
        compression_ratios,
    })
}
fn sources_alias(g: &GgufFile, layers: usize) -> Result<Vec<u64>, LoadError> {
    let ids = array(g, &["engram.layers", "engram.layer_ids", "ple.layers"])?
        .ok_or_else(|| invalid("missing deepseek41.engram.layer_ids"))?;
    if ids.iter().any(|&n| n >= layers as u64) || ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("invalid Engram layers"));
    }
    Ok(ids)
}

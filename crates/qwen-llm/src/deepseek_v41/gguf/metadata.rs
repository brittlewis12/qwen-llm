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
fn float(g: &GgufFile, keys: &[&str], default: f32) -> Result<f32, LoadError> {
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
    Ok(found.unwrap_or(default))
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
    defaults: &[u64],
    layers: usize,
) -> Result<Vec<u64>, LoadError> {
    let values = array(g, keys)?.unwrap_or_else(|| {
        defaults
            .iter()
            .copied()
            .filter(|&n| n < layers as u64)
            .collect()
    });
    if values.iter().any(|&n| n >= layers as u64) || values.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(format!("invalid layer schedule {}", keys[0])));
    }
    Ok(values)
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
        ("hidden_activation", "silu"),
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
    if uint(
        g,
        &["attention.head_count_kv", "num_key_value_heads"],
        Some(1),
    )? != 1
    {
        return Err(invalid(
            "unsupported KV head count; attention implements one shared KV head",
        ));
    }
    engram::validate_encoding(g)?;
    if uint(g, &["hyper_connection.count", "hc_mult"], Some(4))? != 4 {
        return Err(invalid("mHC kernels require four residual streams"));
    }
    if uint(g, &["expert_shared_count", "n_shared_experts"], Some(1))? != 1 {
        return Err(invalid("FFN binding supports one shared expert"));
    }
    let emb = g.find("token_embd.weight");
    let bank = g.find("blk.0.ffn_gate_exps.weight");
    let hidden = uint(
        g,
        &["embedding_length", "hidden_size"],
        emb.and_then(|t| t.shape.first())
            .map(|&n| n as usize)
            .or(Some(d.hidden)),
    )?;
    let intermediate = uint(
        g,
        &["expert_feed_forward_length", "moe_intermediate_size"],
        bank.and_then(|t| t.shape.get(1))
            .map(|&n| n as usize)
            .or(Some(d.intermediate)),
    )?;
    let vocab = uint(
        g,
        &["vocab_size"],
        emb.and_then(|t| t.shape.get(1))
            .map(|&n| n as usize)
            .or(Some(d.vocab)),
    )?;
    let experts = uint(g, &["expert_count", "n_routed_experts"], None)?;
    let mut config = Config {
        hidden,
        intermediate,
        vocab,
        experts,
        heads: uint(
            g,
            &["attention.head_count", "num_attention_heads"],
            Some(d.heads),
        )?,
        head_dim: uint(g, &["attention.key_length", "head_dim"], Some(d.head_dim))?,
        rope_dim: uint(
            g,
            &["rope.dimension_count", "qk_rope_head_dim"],
            Some(d.rope_dim),
        )?,
        query_rank: uint(
            g,
            &["attention.q_lora_rank", "q_lora_rank"],
            Some(d.query_rank),
        )?,
        output_rank: uint(
            g,
            &[
                "attention.output_lora_rank",
                "attention.o_lora_rank",
                "o_lora_rank",
            ],
            Some(d.output_rank),
        )?,
        groups: uint(
            g,
            &[
                "attention.output_group_count",
                "attention.o_groups",
                "o_groups",
            ],
            Some(d.groups),
        )?,
        topk: uint(
            g,
            &["expert_used_count", "num_experts_per_tok"],
            Some(d.topk.min(experts)),
        )?,
        index_heads: uint(
            g,
            &["attention.indexer.head_count", "index_n_heads"],
            Some(d.index_heads),
        )?,
        index_dim: uint(
            g,
            &["attention.indexer.key_length", "index_head_dim"],
            Some(d.index_dim),
        )?,
        index_topk: uint(
            g,
            &["attention.indexer.top_k", "index_topk"],
            Some(d.index_topk),
        )?,
        window: uint(
            g,
            &["attention.sliding_window", "sliding_window"],
            Some(d.window),
        )?,
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
            d.norm_eps,
        )?,
        hc_eps: float(g, &["hyper_connection.epsilon", "hc_eps"], d.hc_eps)?,
        hc_iterations: uint(
            g,
            &["hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters"],
            Some(d.hc_iterations),
        )?,
        clamp: float(g, &["swiglu_limit"], d.clamp)?,
        route_scale: float(
            g,
            &["expert_weights_scale", "routed_scaling_factor"],
            d.route_scale,
        )?,
        gate_temperature: float(g, &["gate_temperature"], d.gate_temperature)?,
        rope_theta: float(g, &["rope.freq_base", "rope_theta"], d.rope_theta)?,
        compress_theta: float(
            g,
            &["attention.compress_rope_freq_base", "compress_rope_theta"],
            d.compress_theta,
        )?,
        original_context: float(
            g,
            &[
                "rope.scaling.original_context_length",
                "rope_scaling.original_max_position_embeddings",
            ],
            d.original_context as f32,
        )? as usize,
        rope_factor: float(
            g,
            &["rope.scaling.factor", "rope_scaling.factor"],
            d.rope_factor,
        )?,
        beta_fast: float(
            g,
            &["rope.scaling.yarn_beta_fast", "rope_scaling.beta_fast"],
            d.beta_fast,
        )?,
        beta_slow: float(
            g,
            &["rope.scaling.yarn_beta_slow", "rope_scaling.beta_slow"],
            d.beta_slow,
        )?,
    };
    let c = &config;
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
    let layers = uint(g, &["block_count", "num_hidden_layers"], Some(40))?;
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
    if layers > g.tensors.len() / 3 {
        return Err(invalid("block count exceeds available expert bank triples"));
    }
    let kv_source_layers = sources(
        g,
        &["attention.kv_source_layers", "kv_source_layer_ids"],
        &[2, 8, 14, 20],
        layers,
    )?;
    let index_source_layers = sources(
        g,
        &["attention.indexer.source_layers", "index_source_layer_ids"],
        &[2, 8, 14, 20, 24, 28, 32, 36],
        layers,
    )?;
    let compression_ratios = array(g, &["attention.compress_ratios", "compress_ratios"])?
        .unwrap_or_else(|| {
            (0..layers)
                .map(|l| {
                    if l < 2 {
                        0
                    } else if l < 20 {
                        2
                    } else {
                        1
                    }
                })
                .collect()
        });
    if compression_ratios.len() < layers {
        return Err(invalid("compression ratios shorter than main block count"));
    }
    let candidate_source_layer = optional_uint_aliases(
        g,
        &[
            "attention.indexer.candidate_source_layer",
            "candidate_source_layer_id",
        ],
    )?
    .map(|n| usize::try_from(n).map_err(|_| invalid("candidate source overflow")))
    .transpose()?
    .or_else(|| {
        kv_source_layers
            .iter()
            .find(|&&l| compression_ratios[l as usize] == 1)
            .map(|&l| l as usize)
    });
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
    if !engram_layers.is_empty()
        && (uint(g, &["engram.head_count", "engram.n_heads"], Some(8))? != 8
            || uint(g, &["engram.max_ngram_size"], Some(4))? != 4)
    {
        return Err(invalid(
            "Engram projection requires three n-gram orders with eight heads each",
        ));
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
        2
    };
    if raw_pad as usize >= vocab {
        return Err(invalid("pad outside vocabulary"));
    }
    if let Some(n) = optional_uint(g, "engram.pad_id")? {
        if u64::from(token_map[raw_pad as usize]) != n {
            return Err(invalid("conflicting Engram pad aliases"));
        }
    }
    let width = uint(g, &["engram.head_dim", "engram.key_length"], Some(256))?;
    if width != 256 {
        return Err(invalid("Engram row decoder requires head_dim 256"));
    }
    let primes = array(g, &["engram.primes"])?;
    let offsets = array(g, &["engram.offsets"])?;
    let multipliers = array(g, &["engram.multipliers"])?;
    for (values, width) in [(&primes, 24), (&offsets, 24), (&multipliers, 4)] {
        if values
            .as_ref()
            .is_some_and(|v| v.len() != engram_layers.len() * width)
        {
            return Err(invalid("Engram hash metadata length"));
        }
    }
    let defaults = HashConfig::default();
    let compressed_vocab_size = optional_uint(g, "engram.compressed_vocab_size")?
        .unwrap_or(u64::from(token_map.iter().copied().max().unwrap_or(0)) + 1);
    let mut engram_hash = HashConfig {
        layers: Vec::new(),
        compressed_vocab_size: u32::try_from(compressed_vocab_size)
            .map_err(|_| invalid("compressed vocabulary overflow"))?,
        max_position: optional_uint_aliases(g, &["context_length", "max_position_embeddings"])?
            .unwrap_or(super::super::engram::MAX_POSITION),
    };
    let declared_rows = array(g, &["engram.num_embeddings", "engram.rows"])?;
    if declared_rows
        .as_ref()
        .is_some_and(|v| v.len() != engram_layers.len())
    {
        return Err(invalid("Engram rows metadata length"));
    }
    for (i, &layer) in engram_layers.iter().enumerate() {
        let t = g
            .find(&format!("blk.{layer}.engram_embd.weight"))
            .ok_or_else(|| invalid("missing Engram table"))?;
        if t.shape.len() != 2 {
            return Err(invalid("Engram table rank"));
        }
        if declared_rows.as_ref().is_some_and(|v| v[i] != t.shape[1]) {
            return Err(invalid("Engram rows metadata differs from tensor"));
        }
        let default = defaults.layers.iter().find(|p| p.layer == layer as usize);
        let primes: [u64; 24] = match &primes {
            Some(values) => values[i * 24..(i + 1) * 24].try_into().unwrap(),
            None => {
                default
                    .ok_or_else(|| invalid("non-default Engram layer needs primes"))?
                    .primes
            }
        };
        let multipliers: [u64; 4] = match &multipliers {
            Some(values) => values[i * 4..(i + 1) * 4].try_into().unwrap(),
            None => {
                default
                    .ok_or_else(|| invalid("non-default Engram layer needs multipliers"))?
                    .multipliers
            }
        };
        let offsets = match &offsets {
            Some(values) => values[i * 24..(i + 1) * 24].try_into().unwrap(),
            None => {
                let mut values = [0; 24];
                let mut offset = 0;
                for (j, &prime) in primes.iter().enumerate() {
                    values[j] = offset;
                    offset = add(offset, prime)?;
                }
                values
            }
        };
        engram_hash.layers.push(super::super::engram::LayerHash {
            layer: layer as usize,
            primes,
            offsets,
            multipliers,
            rows: t.shape[1],
        });
    }
    engram_hash
        .validate()
        .map_err(|e| invalid(format!("Engram hash configuration: {e:?}")))?;
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
        dtype_inventory: BTreeMap::new(),
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
    let ids = array(g, &["engram.layers", "engram.layer_ids"])?
        .unwrap_or_else(|| [1, 14].into_iter().filter(|&l| l < layers as u64).collect());
    if ids.iter().any(|&n| n >= layers as u64) || ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("invalid Engram layers"));
    }
    Ok(ids)
}

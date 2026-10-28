//! One decode step for several independent sequences in shared command
//! buffers, encoded layer-major. Each sequence keeps its own workspace and
//! runs the dispatches of a single `run`, except where a batched kernel
//! reads a weight once for every sequence: the FFN shared expert
//! (`BATCH_SHARED_ENV`), the attention's dense projections (q_b and output_b:
//! `BATCH_ATTN_ENV`; output_a: `BATCH_OUTPUT_A_ENV`; the query-rank/KV-latent
//! pair: `BATCH_QKV_ENV`) and the head (`BATCH_HEAD_ENV`). Without such kernels batching only saves commits: on
//! M5 Ultra (GSQ, 512-token prefixes, B=1-8) per-sequence encoders in shared
//! command buffers and per-sequence command queues both measured
//! 1.00-1.03x of serial steps with unchanged GPU time, because the decode
//! kernels already occupy the whole GPU (a second queue's command waits for,
//! rather than fills, the first's).
use super::*;

/// `0` runs each sequence's FFN shared expert on its own in a batched decode
/// step (`Forward::step_batch`), bitwise equal to separate steps. Otherwise
/// batches of two or more sequences run each layer's shared expert once over
/// all sequences as columns: Q5_K weights through the skinny simdgroup-MMA
/// kernel that reads every weight once for the whole batch (F32 summation
/// order differs from separate steps), other storage column by column
/// (bitwise). M5 Ultra, GSQ, 512-token prefixes: B = 2/3/4/8 rounds 1.08/
/// 1.10/1.11/1.15x faster than serial steps (1.01x with `0`), every
/// teacher-forced argmax equal, max |logit diff| 4.6e-6 of max |logit|.
/// A/B lever.
const BATCH_SHARED_ENV: &str = "QWEN_V41_BATCH_SHARED";

#[cfg(test)]
thread_local! {
    pub(in super::super) static BATCH_SHARED_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn batch_shared_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_SHARED_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_SHARED_ENV).as_deref() != Ok("0"))
}

/// `0` runs each sequence's attention on its own in a batched decode step
/// (bitwise equal to separate steps). Otherwise batches of two or more
/// sequences whose attention runs plain decode on the fused rank-norm chain
/// (`QWEN_V41_QKV_CHAIN` not `split`) encode every sequence's attention in
/// stages around two batched projections per layer: the query up-projection
/// (q_b, query rank -> heads x head dim) and output_b (output rank ->
/// hidden), each once over all sequences as columns (Q5_K through the skinny
/// kernel, F32 rounding; other storage column by column, bitwise). M5
/// Ultra, GSQ, 512-token prefixes, with the batched shared expert: B =
/// 2/3/4/8 rounds 1.14/1.22/1.27/1.37x faster than serial steps (shared
/// expert alone 1.08/1.10/1.11/1.13x); every teacher-forced argmax equal.
/// A/B lever.
const BATCH_ATTN_ENV: &str = "QWEN_V41_BATCH_ATTN";

#[cfg(test)]
thread_local! {
    pub(in super::super) static BATCH_ATTN_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn batch_attn_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_ATTN_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_ATTN_ENV).as_deref() != Ok("0"))
}

/// `0` keeps each sequence's attention output_a (8 groups of heads x
/// head dim / 8 -> output rank) per sequence inside the staged attention
/// (`BATCH_ATTN_ENV`); otherwise one grouped dispatch projects every
/// sequence's groups (a Q5_K bank through the grouped skinny kernel, F32
/// rounding; other storage column by column, bitwise) straight into
/// output_b's input columns. M5 Ultra, GSQ, 512-token prefixes: saves
/// 1.6/5.3 ms per batched round at B = 4/8. A/B lever.
const BATCH_OUTPUT_A_ENV: &str = "QWEN_V41_BATCH_OUTPUT_A";

/// `0` keeps each sequence's query-rank/KV-latent projection pair per
/// sequence inside the staged attention (`BATCH_ATTN_ENV`); otherwise the
/// pair runs once over every sequence's attention input (a Q8_0 pair
/// through the skinny pair kernel, F32 rounding; other storage column by
/// column, bitwise) before each sequence's rank norms. M5 Ultra, GSQ,
/// 512-token prefixes: saves 0.5/2.0 ms per batched round at B = 4/8. A/B
/// lever.
const BATCH_QKV_ENV: &str = "QWEN_V41_BATCH_QKV";

/// `0` keeps each sequence's output head (hidden -> vocabulary) per
/// sequence; otherwise batches of two or more sequences project their final
/// norms once as columns (Q5_K through the skinny kernel, F32 rounding;
/// other storage column by column, bitwise). M5 Ultra, GSQ, 512-token
/// prefixes: saves 0.9/2.5 ms per batched round at B = 4/8. With every
/// batched lever on, B = 2/3/4/8 rounds run 1.16/1.30/1.38/1.55x faster
/// than serial steps. A/B lever.
const BATCH_HEAD_ENV: &str = "QWEN_V41_BATCH_HEAD";

#[cfg(test)]
thread_local! {
    pub(in super::super) static BATCH_OUTPUT_A_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    pub(in super::super) static BATCH_QKV_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    pub(in super::super) static BATCH_HEAD_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn batch_output_a_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_OUTPUT_A_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_OUTPUT_A_ENV).as_deref() != Ok("0"))
}

fn batch_qkv_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_QKV_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_QKV_ENV).as_deref() != Ok("0"))
}

/// `0` keeps each sequence's Engram projections per sequence in a batched
/// decode step; otherwise each Engram layer's projection runs once over all
/// sequences' Engram rows as columns (Q5_K through the skinny kernel, F32
/// rounding; other storage column by column, bitwise). M5 Ultra, GSQ,
/// 512-token prefixes: saves 0.5/1.1 ms per batched round at B = 4/8. A/B
/// lever.
const BATCH_ENGRAM_ENV: &str = "QWEN_V41_BATCH_ENGRAM";

/// `0` keeps each sequence's router and routed experts per sequence in a
/// batched decode step; otherwise every layer routes all sequences in one
/// dispatch and runs the fused gate/up and down once with one grid row per
/// sequence (the router itself column by column: a BF16 skinny router
/// measured no gain on M5). Every per-row kernel computes each (sequence,
/// slot) with the one-sequence arithmetic, so values are bitwise
/// per-sequence. M5 Ultra, GSQ, 512-token prefixes: saves 3.1/8.2 ms per
/// batched round at B = 4/8. A/B lever.
const BATCH_ROUTED_ENV: &str = "QWEN_V41_BATCH_ROUTED";

/// A batched decode step starts a new encoder for each sequence's own chain
/// in every per-sequence phase (HC pre and attention front, rank norms,
/// attention middle, finish and HC post, FFN HC pre, combine), so Metal's
/// resource tracking may overlap different sequences' small dependent
/// kernels; `shared` keeps each phase on one encoder (A/B lever). Same
/// dispatches, so values are unchanged. M5 Ultra, GSQ, 512-token prefixes:
/// 0.5/2.3 ms less per batched round at B = 4/8.
const BATCH_ENCODERS_ENV: &str = "QWEN_V41_BATCH_ENCODERS";

#[cfg(test)]
thread_local! {
    pub(in super::super) static BATCH_ENCODERS_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn per_sequence_encoders() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_ENCODERS_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_ENCODERS_ENV).as_deref() != Ok("shared"))
}

/// A new encoder for the next sequence's chain when `per_sequence_encoders()`.
fn sequence_boundary(stages: &mut StageEncoder<'_>) {
    if per_sequence_encoders() {
        stages.split();
    }
}

#[cfg(test)]
thread_local! {
    pub(in super::super) static BATCH_ENGRAM_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    pub(in super::super) static BATCH_ROUTED_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn batch_engram_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_ENGRAM_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_ENGRAM_ENV).as_deref() != Ok("0"))
}

fn batch_routed_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_ROUTED_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_ROUTED_ENV).as_deref() != Ok("0"))
}

fn batch_head_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = BATCH_HEAD_OVERRIDE.with(|o| o.get()) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(BATCH_HEAD_ENV).as_deref() != Ok("0"))
}

/// Column scratch of the batched attention projections, `[n, B]` each:
/// attention input, query rank and KV latent (the rank pair), query (q_b),
/// attended heads (output_a), output rank (output_a's output, output_b's
/// input) and output (output_b). About 315 KiB per sequence at GSQ
/// geometry. `qkv` and `output_a` record whether those projections are
/// batched too.
struct AttentionColumns {
    input: MetalTensor,
    rank: MetalTensor,
    latent: MetalTensor,
    query: MetalTensor,
    attended: MetalTensor,
    low: MetalTensor,
    output: MetalTensor,
    qkv: bool,
    output_a: bool,
}

/// Column scratch of the batched head: final norms and logits, `[n, B]`
/// each (about 525 KiB per sequence at GSQ geometry).
struct HeadColumns {
    normalized: MetalTensor,
    logits: MetalTensor,
}

/// Per-call scratch of a batched step, allocated (priced and admitted)
/// before any state is touched, for at least two sequences: the shared
/// expert's column scratch when `batch_shared_enabled()`, the attention's
/// when `batch_attn_enabled()`, the head's when `batch_head_enabled()`, the
/// routed experts' when `batch_routed_enabled()`, and one (rows, projected)
/// pair per Engram layer when `batch_engram_enabled()`.
pub(in super::super) struct BatchScratch {
    sequences: usize,
    shared: Option<ffn_gpu::SharedColumns>,
    attention: Option<AttentionColumns>,
    head: Option<HeadColumns>,
    routed: Option<ffn_gpu::RoutedColumns>,
    engram: Option<Vec<(MetalTensor, MetalTensor)>>,
}

impl BatchScratch {
    pub(in super::super) fn new(model: &Forward<'_>, sequences: usize) -> Result<Self> {
        let c = &model.config;
        let batched = sequences >= 2;
        let shared = if batched && batch_shared_enabled() {
            Some(ffn_gpu::SharedColumns::new(model.ctx, c, sequences)?)
        } else {
            None
        };
        let attention = if batched && batch_attn_enabled() {
            let heights = [
                c.hidden,
                c.query_rank,
                c.head_dim,
                c.heads * c.head_dim,
                c.heads * c.head_dim,
                c.groups * c.output_rank,
                c.hidden,
            ];
            let mut regions =
                columns::allocate(model.ctx, &heights, sequences, "attention column scratch")?
                    .into_iter();
            let mut next = || regions.next().expect("seven regions");
            Some(AttentionColumns {
                input: next(),
                rank: next(),
                latent: next(),
                query: next(),
                attended: next(),
                low: next(),
                output: next(),
                qkv: batch_qkv_enabled(),
                output_a: batch_output_a_enabled(),
            })
        } else {
            None
        };
        let head = if batched && batch_head_enabled() {
            let mut regions = columns::allocate(
                model.ctx,
                &[c.hidden, model.head.output],
                sequences,
                "head column scratch",
            )?
            .into_iter();
            let mut next = || regions.next().expect("two regions");
            Some(HeadColumns {
                normalized: next(),
                logits: next(),
            })
        } else {
            None
        };
        let routed = if batched && batch_routed_enabled() {
            Some(ffn_gpu::RoutedColumns::new(model.ctx, c, sequences)?)
        } else {
            None
        };
        let engram = if batched && batch_engram_enabled() {
            let heights: Vec<usize> = model
                .layers
                .iter()
                .filter_map(|l| l.engram.as_ref())
                .flat_map(|e| [e.projection.input, e.projection.output])
                .collect();
            if heights.is_empty() {
                None
            } else {
                let regions =
                    columns::allocate(model.ctx, &heights, sequences, "Engram column scratch")?;
                Some(
                    regions
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|[x, y]| (x.clone(), y.clone()))
                        .collect(),
                )
            }
        } else {
            None
        };
        Ok(Self {
            sequences,
            shared,
            attention,
            head,
            routed,
            engram,
        })
    }
}

/// One sequence of a batched step: its token, its position before the step
/// and its Engram row hashes for this token.
pub(in super::super) struct BatchInput<'r> {
    pub(in super::super) token: usize,
    pub(in super::super) position: usize,
    pub(in super::super) rows: &'r engram::LayerRows,
}

impl Workspace {
    fn encode_engram_projections(
        &self,
        model: &Forward<'_>,
        stages: &mut StageEncoder<'_>,
    ) -> Result<()> {
        for slot in &self.engrams {
            let p = &model.layers[slot.layer].engram.as_ref().unwrap().projection;
            p.encode(stages.enc(), &slot.input, &slot.projected)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn encode_layer(
        &mut self,
        model: &Forward<'_>,
        stages: &mut StageEncoder<'_>,
        id: usize,
        position: usize,
    ) -> Result<()> {
        self.encode_engram_gate(model, stages, id)?;
        self.layer_staged(model, stages, id, position, None, false)
    }

    fn encode_engram_gate(
        &self,
        model: &Forward<'_>,
        stages: &mut StageEncoder<'_>,
        id: usize,
    ) -> Result<()> {
        if let Some(slot) = self.engrams.iter().find(|slot| slot.layer == id) {
            slot.gate
                .encode(model.ctx, stages.enc(), &self.x, &slot.projected, 1)?;
        }
        Ok(())
    }

    fn encode_head(&self, model: &Forward<'_>, stages: &mut StageEncoder<'_>) -> Result<()> {
        self.encode_head_input(model, stages)?;
        model
            .head
            .encode(stages.enc(), &self.normalized, &self.logits)
            .map_err(|e| e.to_string())
    }

    /// The final HC collapse and norm: the head's input (`normalized`).
    fn encode_head_input(&self, model: &Forward<'_>, stages: &mut StageEncoder<'_>) -> Result<()> {
        let ctx = model.ctx;
        let c = &model.config;
        metal::encode_mhc4_collapse(
            ctx,
            stages.enc(),
            c.hidden,
            &self.x,
            self.ffn.next_pre(),
            &self.collapsed,
        )
        .map_err(|e| e.to_string())?;
        metal::encode_rms_norm_mul_rows_f32(
            ctx,
            stages.enc(),
            &self.collapsed,
            &self.norm,
            &self.normalized,
            1,
            c.hidden,
            c.norm_eps,
        )
        .map_err(|e| e.to_string())
    }
}

/// Layer `id` of every sequence's attention, with its dense projections run
/// once for all of them: per sequence the Engram gate, HC pre and the
/// attention checks; the rank pair over the gathered inputs (or, without
/// `qkv`, each sequence's own front); per sequence the rank norms; q_b over
/// the gathered query ranks; per sequence the attention middle (query RoPE
/// through the attended RoPE); output_a over the gathered attended heads
/// into output_b's input columns (or, without `output_a`, each sequence's
/// own output_a and a gather); output_b; per sequence the attention finish
/// (finite check, cursor advance) and HC post. With non-skinny weights each
/// sequence's values are bitwise `layer_until_ffn`'s.
fn encode_attention_columns(
    model: &Forward<'_>,
    stages: &mut StageEncoder<'_>,
    workspaces: &mut [&mut Workspace],
    inputs: &[BatchInput<'_>],
    id: usize,
    scratch: &AttentionColumns,
) -> Result<()> {
    let ctx = model.ctx;
    let w = &model.layers[id].attention;
    let cols = workspaces.len();
    let err = |e: metal::MetalError| e.to_string();
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        sequence_boundary(stages);
        ws.encode_engram_gate(model, stages, id)?;
        ws.attention_input(model, stages, id)?;
        let ws = &mut **ws;
        if scratch.qkv {
            ws.attention
                .encode_checks(stages.enc(), id, input.position, w, &ws.normalized)?;
        } else {
            ws.attention
                .encode_front(ctx, stages.enc(), id, input.position, w, &ws.normalized)?;
        }
    }
    if scratch.qkv {
        {
            let normalized: Vec<_> = workspaces.iter().map(|ws| &ws.normalized).collect();
            let io: Vec<_> = workspaces
                .iter()
                .map(|ws| ws.attention.staged_io())
                .collect();
            let ranks: Vec<_> = io.iter().map(|io| io.query_rank).collect();
            let latents: Vec<_> = io.iter().map(|io| io.latent).collect();
            metal::dispatch_profile_label("attn_qkv", Some(id));
            columns::encode_linear_pair(
                ctx,
                stages.enc(),
                (&w.query_a, &ranks),
                (&w.kv, &latents),
                &normalized,
                (&scratch.input, &scratch.rank, &scratch.latent),
            )
            .map_err(err)?;
        }
        for (ws, input) in workspaces.iter_mut().zip(inputs) {
            sequence_boundary(stages);
            ws.attention
                .encode_rank_norms(ctx, stages.enc(), id, input.position, w)?;
        }
    }
    {
        let io: Vec<_> = workspaces
            .iter()
            .map(|ws| ws.attention.staged_io())
            .collect();
        let ranks: Vec<_> = io.iter().map(|io| io.query_rank).collect();
        let queries: Vec<_> = io.iter().map(|io| io.query).collect();
        metal::dispatch_profile_label("attn_qkv", Some(id));
        columns::encode_linear(
            ctx,
            stages.enc(),
            &w.query_b,
            &ranks,
            &queries,
            (&scratch.rank, &scratch.query),
        )
        .map_err(err)?;
    }
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        sequence_boundary(stages);
        let ws = &mut **ws;
        ws.attention
            .encode_middle(ctx, stages.enc(), id, input.position, w, &ws.normalized)?;
        if !scratch.output_a {
            ws.attention
                .encode_output_a(ctx, stages.enc(), id, input.position, w)?;
        }
    }
    {
        let io: Vec<_> = workspaces
            .iter()
            .map(|ws| ws.attention.staged_io())
            .collect();
        metal::dispatch_profile_label("attn_output", Some(id));
        if scratch.output_a {
            // output_a writes output_b's input columns directly.
            let attended: Vec<_> = io.iter().map(|io| io.attended).collect();
            columns::encode_gather(ctx, stages.enc(), &attended, &scratch.attended).map_err(err)?;
            workspaces[0].attention.encode_output_a_columns(
                ctx,
                stages.enc(),
                w,
                &scratch.attended,
                &scratch.low,
                cols,
            )?;
        } else {
            let lows: Vec<_> = io.iter().map(|io| io.low).collect();
            columns::encode_gather(ctx, stages.enc(), &lows, &scratch.low).map_err(err)?;
        }
        let outputs: Vec<_> = io.iter().map(|io| io.output).collect();
        w.output_b
            .encode_columns(stages.enc(), &scratch.low, &scratch.output, cols)
            .map_err(err)?;
        columns::encode_scatter(ctx, stages.enc(), &scratch.output, &outputs).map_err(err)?;
    }
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        sequence_boundary(stages);
        let output = ws
            .attention
            .encode_finish(ctx, stages.enc(), id, input.position)?
            .clone();
        ws.attention_output(model, stages, id, input.position, &output, None)?;
    }
    Ok(())
}

/// Every sequence's Engram projections: per Engram layer, the projection
/// once over all sequences' Engram rows gathered as columns, scattered to
/// each sequence's projected rows. `scratch` holds one (rows, projected)
/// column pair per Engram layer, in layer order.
fn encode_engram_columns(
    model: &Forward<'_>,
    stages: &mut StageEncoder<'_>,
    workspaces: &[&mut Workspace],
    scratch: &[(MetalTensor, MetalTensor)],
) -> Result<()> {
    let slots = workspaces[0].engrams.len();
    op::require(
        scratch.len() == slots && workspaces.iter().all(|ws| ws.engrams.len() == slots),
        "batched Engram layers",
    )?;
    metal::dispatch_profile_label("engram_projection", None);
    for (k, (x, y)) in scratch.iter().enumerate() {
        let layer = workspaces[0].engrams[k].layer;
        let projection = &model.layers[layer]
            .engram
            .as_ref()
            .ok_or("missing Engram")?
            .projection;
        let inputs: Vec<_> = workspaces.iter().map(|ws| &ws.engrams[k].input).collect();
        let outputs: Vec<_> = workspaces
            .iter()
            .map(|ws| &ws.engrams[k].projected)
            .collect();
        columns::encode_linear(
            model.ctx,
            stages.enc(),
            projection,
            &inputs,
            &outputs,
            (x, y),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Layer `id`'s FFN for every sequence after its attention: per sequence the
/// FFN HC pre; the shared expert (batched or per sequence); the router and
/// routed experts (batched or per sequence); per sequence the combine. With
/// neither batched, each sequence runs `layer_ffn`.
fn encode_ffn_columns(
    model: &Forward<'_>,
    stages: &mut StageEncoder<'_>,
    workspaces: &mut [&mut Workspace],
    id: usize,
    scratch: &BatchScratch,
) -> Result<()> {
    let (ctx, c, l) = (model.ctx, &model.config, &model.layers[id]);
    let err = |e: metal::MetalError| e.to_string();
    if scratch.shared.is_none() && scratch.routed.is_none() {
        for ws in workspaces.iter_mut() {
            ws.layer_ffn(model, stages, id)?;
        }
        return Ok(());
    }
    for ws in workspaces.iter_mut() {
        sequence_boundary(stages);
        ws.ffn_hc_pre(model, stages, id)?;
    }
    let split = ffn_split_encoders();
    if split {
        stages.split();
    }
    match &scratch.shared {
        Some(shared) => {
            let (shared_in, shared_out): (Vec<_>, Vec<_>) =
                workspaces.iter().map(|ws| ws.ffn.shared_io()).unzip();
            ffn_gpu::encode_shared_columns(
                ctx,
                stages.enc(),
                c,
                &l.shared,
                id,
                &shared_in,
                &shared_out,
                shared,
            )
            .map_err(err)?;
        }
        None => {
            for ws in workspaces.iter() {
                ws.ffn
                    .encode_shared(ctx, stages.enc(), c, l, id)
                    .map_err(err)?;
            }
        }
    }
    let Some(routed) = &scratch.routed else {
        for ws in workspaces.iter_mut() {
            ws.layer_after_shared(model, stages, id)?;
        }
        return Ok(());
    };
    if split {
        stages.split();
    }
    let sequences: Vec<_> = workspaces.iter().map(|ws| (&ws.ffn, &ws.half)).collect();
    ffn_gpu::encode_routed_columns(ctx, stages.enc(), c, l, id, &sequences, routed).map_err(err)?;
    drop(sequences);
    if split {
        stages.split();
    }
    for ws in workspaces.iter_mut() {
        sequence_boundary(stages);
        ws.ffn_combine(model, stages, id)?;
    }
    Ok(())
}

/// Every sequence's head: per sequence the final collapse and norm, then
/// the head once over the gathered norms, scattered to each sequence's
/// logits.
fn encode_head_columns(
    model: &Forward<'_>,
    stages: &mut StageEncoder<'_>,
    workspaces: &[&mut Workspace],
    scratch: &HeadColumns,
) -> Result<()> {
    let err = |e: metal::MetalError| e.to_string();
    for ws in workspaces {
        ws.encode_head_input(model, stages)?;
    }
    let normalized: Vec<_> = workspaces.iter().map(|ws| &ws.normalized).collect();
    let logits: Vec<_> = workspaces.iter().map(|ws| &ws.logits).collect();
    metal::dispatch_profile_label("head", None);
    columns::encode_linear(
        model.ctx,
        stages.enc(),
        &model.head,
        &normalized,
        &logits,
        (&scratch.normalized, &scratch.logits),
    )
    .map_err(err)
}

/// Segments follow `submit_split` (or one command when split submission is
/// off): [layers before the first Engram layer], then the Engram projections
/// and the next layers, then the rest and the heads. Each segment is one
/// command that encodes layer-major (layer l of every sequence, then layer
/// l + 1); every sequence's Engram rows are read while the first segment
/// runs. Cancellation is honored only before the first commit; every
/// committed command is waited for before returning.
pub(in super::super) fn run_batch(
    model: &Forward<'_>,
    workspaces: &mut [&mut Workspace],
    inputs: &[BatchInput<'_>],
    scratch: &BatchScratch,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Vec<f32>>> {
    op::require(
        !workspaces.is_empty()
            && workspaces.len() == inputs.len()
            && (scratch.sequences == workspaces.len()
                || scratch.shared.is_none()
                    && scratch.attention.is_none()
                    && scratch.head.is_none()
                    && scratch.routed.is_none()
                    && scratch.engram.is_none()),
        "batched decode shape",
    )?;
    // Staged attention needs every sequence on plain fused-chain decode;
    // otherwise each sequence encodes its own attention.
    let attention = scratch.attention.is_some()
        && workspaces
            .iter()
            .all(|ws| ws.attention.staged_decode_supported());
    let ctx = model.ctx;
    let end = model.layers.len();
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        ws.commands = 0;
        ws.trace_copies = 0;
        write(
            &ws.x,
            &decode_rows(model.embedding, input.token, 1)?.repeat(4),
        )?;
    }
    let engram_layers: Vec<usize> = workspaces[0].engrams.iter().map(|s| s.layer).collect();
    let lead = engram_layers.iter().copied().min().unwrap_or(0).min(end);
    let middle = (lead + SPLIT_SUBMIT_MIDDLE_LAYERS).min(end);
    let segments = if split_submission() {
        vec![
            (0..lead, false, false),
            (lead..middle, true, false),
            (middle..end, false, true),
        ]
    } else {
        vec![(0..end, true, true)]
    };
    cancelled(cancel)?;
    let started = profile::start();
    let dims = [inputs[0].position, workspaces.len(), end, 0];
    let _wall = profile::Wall("gpu_step_batch", dims, started);
    let mut committed: Vec<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    > = Vec::new();
    // Under `profile::Scope::begin_dispatches` every dispatch of the step is
    // timed in isolation (plain encoders begun under the guard split per
    // dispatch), as in a single step.
    let dispatch_profile = if profile::dispatches() {
        Some(
            metal::dispatch_profile_begin(ctx, DECODE_DISPATCH_PROFILE_CAPACITY * workspaces.len())
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let encoded = (|| -> Result<()> {
        for (range, engram_projection, head) in segments {
            if engram_projection {
                // Earlier segments are running; read every sequence's rows.
                let _rows = profile::Wall(
                    "decode_engram_rows",
                    [0, engram_layers.len(), workspaces.len(), 0],
                    profile::start(),
                );
                for (ws, input) in workspaces.iter().zip(inputs) {
                    let values = model
                        .engram_rows
                        .decode_layers(&engram_layers, input.rows)?;
                    for (slot, values) in ws.engrams.iter().zip(values) {
                        write(&slot.input, &values)?;
                    }
                }
            }
            if range.is_empty() && !engram_projection && !head {
                continue;
            }
            let command = ctx
                .queue
                .commandBuffer()
                .ok_or("Metal command buffer allocation")?;
            let mut stages = StageEncoder::begin(&command, None);
            let segment = (|| -> Result<()> {
                if engram_projection {
                    match &scratch.engram {
                        Some(columns) => {
                            encode_engram_columns(model, &mut stages, workspaces, columns)?
                        }
                        None => {
                            for ws in workspaces.iter() {
                                ws.encode_engram_projections(model, &mut stages)?;
                            }
                        }
                    }
                }
                for id in range {
                    if scratch.shared.is_none() && scratch.routed.is_none() && !attention {
                        for (ws, input) in workspaces.iter_mut().zip(inputs) {
                            ws.encode_layer(model, &mut stages, id, input.position)?;
                        }
                        continue;
                    }
                    match scratch.attention.as_ref().filter(|_| attention) {
                        Some(columns) => encode_attention_columns(
                            model,
                            &mut stages,
                            workspaces,
                            inputs,
                            id,
                            columns,
                        )?,
                        None => {
                            for (ws, input) in workspaces.iter_mut().zip(inputs) {
                                ws.encode_engram_gate(model, &mut stages, id)?;
                                ws.layer_until_ffn(
                                    model,
                                    &mut stages,
                                    id,
                                    input.position,
                                    None,
                                    false,
                                )?;
                            }
                        }
                    }
                    encode_ffn_columns(model, &mut stages, workspaces, id, scratch)?;
                }
                if head {
                    match &scratch.head {
                        Some(columns) => {
                            encode_head_columns(model, &mut stages, workspaces, columns)?
                        }
                        None => {
                            for ws in workspaces.iter() {
                                ws.encode_head(model, &mut stages)?;
                            }
                        }
                    }
                }
                Ok(())
            })();
            stages.end();
            segment?;
            command.commit();
            committed.push(command);
        }
        Ok(())
    })();
    let encode_ms = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
    let wait = std::time::Instant::now();
    let mut waited: Result<()> = Ok(());
    for command in &committed {
        if let Err(error) = metal::wait_completed(command)
            && waited.is_ok()
        {
            waited = Err(error.to_string());
        }
    }
    for ws in workspaces.iter_mut() {
        ws.commands += committed.len();
    }
    if let (Some(encode_ms), Some(first), Some(last)) =
        (encode_ms, committed.first(), committed.last())
    {
        profile::submitted_with_driver_scheduling(
            "gpu_step_batch",
            dims,
            encode_ms,
            wait.elapsed().as_secs_f64() * 1000.0,
            encoded.is_ok() && waited.is_ok(),
            first.GPUStartTime(),
            last.GPUEndTime(),
            || (first.kernelStartTime(), last.kernelEndTime()),
        );
    }
    encoded?;
    waited?;
    if let Some(guard) = dispatch_profile {
        profile::record_dispatches(
            metal::dispatch_profile_finish(ctx, guard).map_err(|e| e.to_string())?,
        );
    }
    for (index, ws) in workspaces.iter().enumerate() {
        let status = unsafe {
            *ws.attention
                .status()
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
        };
        op::require(
            status == 0,
            &format!("sequence {index}: GPU attention status {status}"),
        )?;
        for id in 0..end {
            let status = ws.ffn.status(id);
            op::require(
                status == metal::ROUTE_STATUS_READY,
                &format!("sequence {index}: GPU FFN layer {id} route status {status}"),
            )?;
        }
    }
    workspaces
        .iter()
        .map(|ws| {
            let logits = f32_slice(&ws.logits)?;
            op::require(logits.iter().all(|v| v.is_finite()), "non-finite logits")?;
            Ok(logits.to_vec())
        })
        .collect()
}

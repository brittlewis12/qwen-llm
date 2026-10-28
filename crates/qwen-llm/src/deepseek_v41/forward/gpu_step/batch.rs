//! One decode step for several independent sequences in shared command
//! buffers, encoded layer-major. Each sequence keeps its own workspace and
//! runs the dispatches of a single `run`, except where a batched kernel
//! reads a weight once for every sequence: the FFN shared expert
//! (`BATCH_SHARED_ENV`) and the attention's q_b and output_b projections
//! (`BATCH_ATTN_ENV`). Without such kernels batching only saves commits: on
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

/// Column scratch of the batched attention projections: query rank and
/// query (q_b), output rank and output (output_b), `[n, B]` each. About
/// 185 KiB per sequence at GSQ geometry.
struct AttentionColumns {
    rank: MetalTensor,
    query: MetalTensor,
    low: MetalTensor,
    output: MetalTensor,
}

/// Per-call scratch of a batched step, allocated (priced and admitted)
/// before any state is touched, for at least two sequences: the shared
/// expert's column scratch when `batch_shared_enabled()`, the attention's
/// when `batch_attn_enabled()`.
pub(in super::super) struct BatchScratch {
    sequences: usize,
    shared: Option<ffn_gpu::SharedColumns>,
    attention: Option<AttentionColumns>,
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
                c.query_rank,
                c.heads * c.head_dim,
                c.groups * c.output_rank,
                c.hidden,
            ];
            let mut regions =
                columns::allocate(model.ctx, &heights, sequences, "attention column scratch")?
                    .into_iter();
            let mut next = || regions.next().expect("four regions");
            Some(AttentionColumns {
                rank: next(),
                query: next(),
                low: next(),
                output: next(),
            })
        } else {
            None
        };
        Ok(Self {
            sequences,
            shared,
            attention,
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
        .map_err(|e| e.to_string())?;
        model
            .head
            .encode(stages.enc(), &self.normalized, &self.logits)
            .map_err(|e| e.to_string())
    }
}

/// Layer `id` of every sequence's attention, with q_b and output_b projected
/// once for all of them: per sequence the Engram gate, HC pre and the
/// attention front (through the rank norms); q_b over the gathered query
/// ranks; per sequence the attention middle (query RoPE through output_a);
/// output_b over the gathered output ranks; per sequence the attention
/// finish (finite check, cursor advance) and HC post. With non-skinny
/// weights each sequence's values are bitwise `layer_until_ffn`'s.
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
    let err = |e: metal::MetalError| e.to_string();
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        ws.encode_engram_gate(model, stages, id)?;
        ws.attention_input(model, stages, id)?;
        let ws = &mut **ws;
        ws.attention
            .encode_front(ctx, stages.enc(), id, input.position, w, &ws.normalized)?;
    }
    let io: Vec<[&MetalTensor; 4]> = workspaces
        .iter()
        .map(|ws| ws.attention.staged_io())
        .collect();
    let slot = |k: usize| io.iter().map(|io| io[k]).collect::<Vec<_>>();
    metal::dispatch_profile_label("attn_qkv", Some(id));
    columns::encode_linear(
        ctx,
        stages.enc(),
        &w.query_b,
        &slot(0),
        &slot(1),
        (&scratch.rank, &scratch.query),
    )
    .map_err(err)?;
    drop(io);
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        let ws = &mut **ws;
        ws.attention
            .encode_middle(ctx, stages.enc(), id, input.position, w, &ws.normalized)?;
    }
    let io: Vec<[&MetalTensor; 4]> = workspaces
        .iter()
        .map(|ws| ws.attention.staged_io())
        .collect();
    let slot = |k: usize| io.iter().map(|io| io[k]).collect::<Vec<_>>();
    columns::encode_linear(
        ctx,
        stages.enc(),
        &w.output_b,
        &slot(2),
        &slot(3),
        (&scratch.low, &scratch.output),
    )
    .map_err(err)?;
    drop(io);
    for (ws, input) in workspaces.iter_mut().zip(inputs) {
        let output = ws
            .attention
            .encode_finish(ctx, stages.enc(), id, input.position)?
            .clone();
        ws.attention_output(model, stages, id, input.position, &output, None)?;
    }
    Ok(())
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
                || scratch.shared.is_none() && scratch.attention.is_none()),
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
                    for ws in workspaces.iter() {
                        ws.encode_engram_projections(model, &mut stages)?;
                    }
                }
                for id in range {
                    if scratch.shared.is_none() && !attention {
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
                    let Some(shared) = &scratch.shared else {
                        for ws in workspaces.iter_mut() {
                            ws.layer_ffn(model, &mut stages, id)?;
                        }
                        continue;
                    };
                    for ws in workspaces.iter_mut() {
                        ws.ffn_hc_pre(model, &mut stages, id)?;
                    }
                    if ffn_split_encoders() {
                        stages.split();
                    }
                    let (shared_in, shared_out): (Vec<_>, Vec<_>) =
                        workspaces.iter().map(|ws| ws.ffn.shared_io()).unzip();
                    ffn_gpu::encode_shared_columns(
                        ctx,
                        stages.enc(),
                        &model.config,
                        &model.layers[id].shared,
                        id,
                        &shared_in,
                        &shared_out,
                        shared,
                    )
                    .map_err(|e| e.to_string())?;
                    for ws in workspaces.iter_mut() {
                        ws.layer_after_shared(model, &mut stages, id)?;
                    }
                }
                if head {
                    for ws in workspaces.iter() {
                        ws.encode_head(model, &mut stages)?;
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

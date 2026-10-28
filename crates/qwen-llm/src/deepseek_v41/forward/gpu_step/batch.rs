//! One decode step for several independent sequences in shared command
//! buffers, encoded layer-major. Each sequence keeps its own workspace and
//! runs exactly the dispatches of a single `run` (bitwise-identical
//! results). This is the scaffold for batched kernels that read a weight once
//! for every sequence; by itself it only saves commits: on M5 Ultra (GSQ,
//! 512-token prefixes, B=1-8) per-sequence encoders in shared command buffers
//! and per-sequence command queues both measured 1.00-1.03x of serial steps
//! with unchanged GPU time, because the decode kernels already occupy the
//! whole GPU (a second queue's command waits for, rather than fills, the
//! first's).
use super::*;

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
        if let Some(slot) = self.engrams.iter().find(|slot| slot.layer == id) {
            slot.gate
                .encode(model.ctx, stages.enc(), &self.x, &slot.projected, 1)?;
        }
        self.layer_staged(model, stages, id, position, None, false)
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
    cancel: Option<&AtomicBool>,
) -> Result<Vec<Vec<f32>>> {
    op::require(
        !workspaces.is_empty() && workspaces.len() == inputs.len(),
        "batched decode shape",
    )?;
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
                    for (ws, input) in workspaces.iter_mut().zip(inputs) {
                        ws.encode_layer(model, &mut stages, id, input.position)?;
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

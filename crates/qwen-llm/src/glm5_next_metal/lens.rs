//! Passive lens surface: post-block residual captures for one decoded token,
//! and the deployed output tail (mean of the four streams, output norm,
//! untied head) on an arbitrary residual. A readout touches only per-token
//! scratch: no KDA, latent or indexer state moves and the position stays.
use super::*;

/// The post-block residual streams of the requested blocks at one position,
/// raw (before the head's collapse or any following norm).
#[derive(Debug)]
pub struct Glm5NextCapture {
    /// Absolute position of the decoded token.
    pub position: usize,
    /// Captured executed blocks, ascending.
    pub layers: Vec<u32>,
    /// Streams per residual (`hyper_connection.count`).
    pub streams: usize,
    pub hidden: usize,
    /// `[site][stream][hidden]`.
    pub residuals: Vec<f32>,
    /// The token's ordinary full-vocabulary logits.
    pub logits: Vec<f32>,
}

impl Glm5NextCapture {
    /// The `[stream][hidden]` residual of the `index`-th captured block.
    pub fn site(&self, index: usize) -> Option<&[f32]> {
        let width = self.streams * self.hidden;
        self.residuals.get(index * width..(index + 1) * width)
    }
}

impl Glm5NextSession<'_> {
    /// Decode one token like [`Self::forward`], also capturing the
    /// post-block residual streams (`l_out`) of `layers`, which must be
    /// ascending, unique executed blocks. A refused site list executes
    /// nothing.
    pub fn forward_with_post_block_captures(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        layers: &[u32],
    ) -> Result<Glm5NextCapture> {
        let c = &self.weights.config;
        let blocks = c.executed_block_count();
        if layers.is_empty()
            || !layers.windows(2).all(|pair| pair[0] < pair[1])
            || layers.last().is_some_and(|&last| last >= blocks)
        {
            return invalid(format!(
                "capture sites must be ascending unique blocks below {blocks}"
            ));
        }
        let (streams, hidden) = (c.hc_streams as usize, c.hidden_size as usize);
        let width = streams * hidden;
        let position = self.position;
        let mut residuals = vec![0.0f32; layers.len() * width];
        let mut filled = vec![false; layers.len()];
        let mut malformed = false;
        let logits = self.forward_observed(
            ctx,
            token,
            &[Glm5NextProbe::BlockResidual],
            &mut |probe, block, values| {
                if probe != Glm5NextProbe::BlockResidual {
                    return;
                }
                if let Ok(index) = layers.binary_search(&(block as u32)) {
                    if values.len() == width {
                        residuals[index * width..(index + 1) * width].copy_from_slice(values);
                        filled[index] = true;
                    } else {
                        malformed = true;
                    }
                }
            },
        )?;
        if malformed || !filled.iter().all(|&f| f) {
            return invalid("post-block capture is missing or malformed");
        }
        Ok(Glm5NextCapture {
            position,
            layers: layers.to_vec(),
            streams,
            hidden,
            residuals,
            logits,
        })
    }

    /// The deployed output tail on one finite `[stream][hidden]` residual:
    /// the head's fixed mean collapse, the output norm and the untied head,
    /// in the same kernels as decode, so the last block's capture reads out
    /// to the token's logits bit for bit.
    pub fn readout(&mut self, ctx: &MetalContext, residual: &[f32]) -> Result<Vec<f32>> {
        if self.poisoned {
            return invalid("session is poisoned by an earlier failed token");
        }
        let w = self.weights;
        let c = &w.config;
        let h = c.hidden_size as usize;
        if residual.len() != c.hc_width() as usize || !residual.iter().all(|v| v.is_finite()) {
            return invalid(format!(
                "readout needs {} finite values ([{}][{h}])",
                c.hc_width(),
                c.hc_streams
            ));
        }
        let s = &self.s;
        // Per-token scratch: every decoded token rewrites both residual
        // buffers from its embedding before reading them.
        write_f32(&s.residual[1], residual)?;
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let enc = KernelEncoder::begin(&command);
        encode_mhc4_collapse(ctx, &enc, h, &s.residual[1], &s.quarter, &s.final_hidden)?;
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &s.final_hidden,
            &w.output_norm,
            &s.final_normed,
            c.rms_epsilon,
        )?;
        matvec(
            ctx,
            &enc,
            &w.output,
            &s.final_normed,
            &s.logits,
            h,
            c.vocab_size as usize,
        )?;
        enc.end();
        command.commit();
        wait_completed(&command)?;
        let logits = read_f32(&s.logits)?;
        if !logits.iter().all(|v| v.is_finite()) {
            return invalid("readout produced nonfinite logits");
        }
        Ok(logits)
    }
}

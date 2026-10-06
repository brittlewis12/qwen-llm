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
    /// The `[stream][hidden]` residual of the `index`-th captured block;
    /// `None` out of range (overflow included).
    pub fn site(&self, index: usize) -> Option<&[f32]> {
        let width = self.streams.checked_mul(self.hidden)?;
        let start = index.checked_mul(width)?;
        self.residuals.get(start..start.checked_add(width)?)
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
        // Per-token scratch: a decoded token writes residual[0] from its
        // embedding and residual[1] in its first block before reading
        // either; packed prefill has its own residual buffers.
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
        // A finite residual can still leave the output norm's F32 sum of
        // squares unrepresentable (an inf statistic gives a zero scale and
        // finite zero logits); refuse it rather than read out silently.
        output_norm_statistic(&read_f32(&s.final_hidden)?).map_err(Glm5NextMetalError::Invalid)?;
        let logits = read_f32(&s.logits)?;
        if !logits.iter().all(|v| v.is_finite()) {
            return invalid("readout produced nonfinite logits");
        }
        Ok(logits)
    }
}

/// The output norm's statistic on the deployed collapse output: its sum of
/// squares, accumulated in F32 by `kernel_rms_norm_mul_f32`, must be
/// representable. Accepted when the exact (f64) sum is at most
/// `f32::MAX / (1 + n * eps)`, the accumulation-error bound for n nonnegative
/// F32 terms, so an accepted input cannot overflow; anything above is refused
/// with the statistic named. Ordinary activations are many orders of
/// magnitude below the bound; this is a domain check for external readouts,
/// not a magnitude policy.
fn output_norm_statistic(hidden: &[f32]) -> std::result::Result<f64, String> {
    let sum: f64 = hidden.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let bound = f64::from(f32::MAX) / (1.0 + hidden.len() as f64 * f64::from(f32::EPSILON));
    if sum.is_finite() && sum <= bound {
        Ok(sum)
    } else {
        Err(format!(
            "readout residual is outside the output norm's F32 domain: the collapsed stream's sum of \
             squares {sum:e} exceeds the representable bound {bound:e} for {} values",
            hidden.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The output-norm statistic check refuses residuals whose collapsed
    /// sum of squares cannot be represented in F32 (four identical 1e20
    /// streams collapse to a finite 1e20 whose square overflows) and accepts
    /// ordinary and zero inputs.
    #[test]
    fn output_norm_statistic_refuses_unrepresentable_sums_of_squares() {
        let hidden = 4096;
        assert!(output_norm_statistic(&vec![1.0e20; hidden]).is_err());
        // 4096 x (3e17)^2 = 3.7e38 > f32::MAX; 4096 x (2e17)^2 = 1.6e38 fits.
        assert!(output_norm_statistic(&vec![3.0e17; hidden]).is_err());
        assert!(output_norm_statistic(&vec![2.0e17; hidden]).is_ok());
        let refused = output_norm_statistic(&vec![1.0e20; hidden]).unwrap_err();
        assert!(refused.contains("sum of squares"), "{refused}");
        assert!(output_norm_statistic(&vec![0.0; hidden]).is_ok());
        assert!(output_norm_statistic(&vec![37.5; hidden]).is_ok());
        // Just inside and just outside the bound for a single value.
        let edge = (f64::from(f32::MAX) / (1.0 + f64::from(f32::EPSILON))).sqrt();
        assert!(output_norm_statistic(&[(edge * 0.999) as f32]).is_ok());
        assert!(output_norm_statistic(&[(edge * 1.001) as f32]).is_err());
    }

    #[test]
    fn sites_are_bounded_without_overflow() {
        let capture = Glm5NextCapture {
            position: 0,
            layers: vec![0, 3],
            streams: 4,
            hidden: 2,
            residuals: (0..16).map(|v| v as f32).collect(),
            logits: Vec::new(),
        };
        assert_eq!(
            capture.site(1).unwrap(),
            &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        );
        for index in [2, usize::MAX, 1usize << 61, usize::MAX / 8 + 1] {
            assert!(capture.site(index).is_none(), "{index}");
        }
    }
}

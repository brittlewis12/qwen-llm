//! Host-to-owned-arena translation only; the shared block graph and existing
//! forward Metal kernels implement the interventions. No fitting/backward API.

use super::lens::{LAYERS, WIDTH, allocate_rows, capture_bytes, validate_residual};
use super::*;
use crate::metal::{PostBlockIntervention, encode_post_block_intervention_f32};

/// Operations act on the current post-block residual `x`. Vectors are used as
/// supplied, WITHOUT normalization. Dot products/L2 use the kernel's F32 reduction.
#[derive(Clone, Copy, Debug)]
pub enum K2InterventionKind<'a> {
    /// `x <- x + c * v`.
    Fixed { direction: &'a [f32] },
    /// `x <- x + c * ||x||_2 * v`.
    ResidualL2Relative { direction: &'a [f32] },
    /// `x <- x - c * (x dot v) * v`. An orthogonal projection only for unit `v`.
    Projection { direction: &'a [f32] },
    /// `x <- x + c * (x dot source) * (target - source)`.
    SourceToTarget {
        source: &'a [f32],
        target: &'a [f32],
    },
}

#[derive(Clone, Copy, Debug)]
pub struct K2Intervention<'a> {
    pub post_block_layer: u32,
    /// Finite and nonzero. Use an empty operation list for an exact no-op.
    pub coefficient: f32,
    pub kind: K2InterventionKind<'a>,
}

impl K2Session<'_, '_> {
    /// Intervene only on the final appended token, after FFN residual addition
    /// and before captures/next block. At most 64 operations, in nondecreasing
    /// layer order; same-layer operations preserve caller order, never sorting.
    /// Captures may be empty; otherwise their usual sorted/unique policy applies.
    /// Empty operations follow the ordinary graph with no vector arena.
    ///
    /// Current/earlier-layer KV has already been stored; only later layers' KV
    /// can reflect a post-block operation. Earlier token rows are never edited.
    /// Final-layer interventions affect logits, not KV or future continuation.
    /// Submitted nonfinite/failure checks poison as for ordinary append; these
    /// are final/cache/capture checks, not per-operation finite instrumentation.
    pub fn append_with_interventions(
        &mut self,
        tokens: &[u32],
        post_block_layers: &[u32],
        interventions: &[K2Intervention<'_>],
    ) -> Result<K2CapturedForward> {
        if !post_block_layers.is_empty() {
            capture_bytes(post_block_layers)?;
        }
        validate(interventions)?;
        self.append_impl(
            tokens,
            post_block_layers,
            interventions,
            AppendReadout::FinalLogits,
        )
    }
}

impl<'a> K2InterventionKind<'a> {
    fn vectors(self) -> (&'a [f32], Option<&'a [f32]>) {
        match self {
            Self::Fixed { direction }
            | Self::ResidualL2Relative { direction }
            | Self::Projection { direction } => (direction, None),
            Self::SourceToTarget { source, target } => (source, Some(target)),
        }
    }
}

fn validate(operations: &[K2Intervention<'_>]) -> Result<usize> {
    if operations.len() > 64 {
        return Err(invalid("at most 64 post-block interventions are supported"));
    }
    let mut rows = 0usize;
    let mut previous_layer = 0;
    for operation in operations {
        if operation.post_block_layer >= LAYERS || operation.post_block_layer < previous_layer {
            return Err(invalid(
                "interventions require nondecreasing post-block layers in 0..36",
            ));
        }
        previous_layer = operation.post_block_layer;
        if !operation.coefficient.is_finite() || operation.coefficient == 0.0 {
            return Err(invalid(
                "intervention coefficient must be finite and nonzero",
            ));
        }
        let (first, second) = operation.kind.vectors();
        for row in std::iter::once(first).chain(second) {
            validate_residual(row)?;
            rows += 1;
        }
    }
    // The operation bound and closed enum limit this to 128 rows / 2 MiB.
    Ok(rows)
}

#[derive(Clone, Copy)]
enum Kind {
    Fixed,
    Relative,
    Projection,
    SourceToTarget,
}

struct Operation {
    layer: u32,
    coefficient: f32,
    kind: Kind,
    row: usize,
}

pub(super) struct InterventionArena {
    tensor: MetalTensor,
    operations: Vec<Operation>,
}

impl InterventionArena {
    pub(super) fn allocate(
        ctx: &MetalContext,
        operations: &[K2Intervention<'_>],
    ) -> Result<Option<Self>> {
        let rows = validate(operations)?;
        if rows == 0 {
            return Ok(None);
        }
        let tensor = allocate_rows(ctx, rows)?;
        let mut descriptors = Vec::with_capacity(operations.len());
        let mut row = 0;
        for operation in operations {
            let kind = match operation.kind {
                K2InterventionKind::Fixed { .. } => Kind::Fixed,
                K2InterventionKind::ResidualL2Relative { .. } => Kind::Relative,
                K2InterventionKind::Projection { .. } => Kind::Projection,
                K2InterventionKind::SourceToTarget { .. } => Kind::SourceToTarget,
            };
            descriptors.push(Operation {
                layer: operation.post_block_layer,
                coefficient: operation.coefficient,
                kind,
                row,
            });
            let (first, second) = operation.kind.vectors();
            for vector in std::iter::once(first).chain(second) {
                // Newly allocated checked shared F32 arena, not submitted yet.
                // Each validated vector occupies one disjoint, bounded row.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        vector.as_ptr(),
                        tensor
                            .buffer
                            .contents()
                            .as_ptr()
                            .cast::<f32>()
                            .add(row * WIDTH),
                        WIDTH,
                    );
                }
                row += 1;
            }
        }
        Ok(Some(Self {
            tensor,
            operations: descriptors,
        }))
    }

    pub(super) fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        layer: u32,
        residual: &MetalTensor,
    ) -> Result<()> {
        for operation in self.operations.iter().filter(|op| op.layer == layer) {
            let direction = self
                .tensor
                .view_subrange((operation.row * WIDTH) as u64, vec![WIDTH as u64]);
            let coefficient = operation.coefficient;
            let target;
            let op = match operation.kind {
                Kind::Fixed => PostBlockIntervention::Fixed {
                    layer,
                    direction: &direction,
                    coefficient,
                },
                Kind::Relative => PostBlockIntervention::ResidualL2Relative {
                    layer,
                    direction: &direction,
                    coefficient,
                },
                Kind::Projection => PostBlockIntervention::Projection {
                    layer,
                    direction: &direction,
                    coefficient,
                },
                Kind::SourceToTarget => {
                    target = self
                        .tensor
                        .view_subrange(((operation.row + 1) * WIDTH) as u64, vec![WIDTH as u64]);
                    PostBlockIntervention::SourceToTarget {
                        layer,
                        source: &direction,
                        target: &target,
                        coefficient,
                    }
                }
            };
            encode_post_block_intervention_f32(ctx, enc, residual, &op)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

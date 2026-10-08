//! RAM snapshots of a GLM-5.3 session's persistent state (map #15).
//!
//! A snapshot at committed length `n` holds exactly what later tokens read:
//! per KDA block the conv tails and the S state; per MLA block the F16
//! latent rows `[0, n)`, the completed pooled indexer keys
//! `[0, floor(n / pool))` and the whole pending key|gate ring (its phase is
//! `n % pool`, derived from the position). Scratch, route records, selector
//! status and logits are per token or per request and are not captured; a
//! restore needs a non-empty suffix to produce logits.
//!
//! A snapshot is bound to the loaded weights instance, the session's packed
//! lineage and [`SNAPSHOT_POLICY_VERSION`]; restoring validates every region
//! against the destination before copying anything, publishes the position
//! only after every copy, and leaves the destination poisoned if a copy
//! fails part way. Rows past the position in a recycled destination are
//! never read (attention and selection read only visible rows and completed
//! pools), so they are not cleared.

use super::*;

/// Bumped whenever the arithmetic or schedule that produced captured state
/// changes in a way that makes old snapshots non-equivalent.
pub const SNAPSHOT_POLICY_VERSION: u32 = 1;

/// One layer's captured regions.
enum LayerRegions {
    Kda {
        conv: Vec<u8>,
        state: Vec<u8>,
    },
    Mla {
        latent: Vec<u8>,
        pending: Vec<u8>,
        pooled: Vec<u8>,
    },
}

/// Persistent state of a session at a committed position.
pub struct Glm5NextSnapshot {
    position: usize,
    weights_instance: u64,
    lineage: Option<PackedLineage>,
    policy_version: u32,
    layers: Vec<LayerRegions>,
}

impl Glm5NextSnapshot {
    pub fn position(&self) -> usize {
        self.position
    }

    /// The packed lineage of the session that produced it (`None`: serial).
    pub fn lineage(&self) -> Option<PackedLineage> {
        self.lineage
    }

    /// Captured bytes (the regions; bookkeeping excluded).
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                LayerRegions::Kda { conv, state } => conv.len() + state.len(),
                LayerRegions::Mla {
                    latent,
                    pending,
                    pooled,
                } => latent.len() + pending.len() + pooled.len(),
            })
            .sum()
    }

    /// Same position, lineage and captured bytes (tests).
    #[cfg(test)]
    pub(crate) fn same_state(&self, other: &Self) -> bool {
        self.position == other.position
            && self.lineage == other.lineage
            && self.layers.len() == other.layers.len()
            && self
                .layers
                .iter()
                .zip(&other.layers)
                .all(|pair| match pair {
                    (
                        LayerRegions::Kda { conv: a, state: b },
                        LayerRegions::Kda { conv: c, state: d },
                    ) => a == c && b == d,
                    (
                        LayerRegions::Mla {
                            latent: a,
                            pending: b,
                            pooled: c,
                        },
                        LayerRegions::Mla {
                            latent: d,
                            pending: e,
                            pooled: f,
                        },
                    ) => a == d && b == e && c == f,
                    _ => false,
                })
    }

    /// Test hook: a snapshot claiming another weights instance or policy.
    #[cfg(test)]
    pub(crate) fn forge_identity(&mut self, weights_instance: u64, policy_version: u32) {
        self.weights_instance = weights_instance;
        self.policy_version = policy_version;
    }
}

/// Checked size of a snapshot at `position` for `config` (region bytes only).
pub fn snapshot_bytes(config: &Glm5NextConfig, position: u64) -> Option<u64> {
    let kda = config.block_count(MixerKind::Kda) as u64;
    let mla = config.block_count(MixerKind::Mla) as u64;
    let w = config.kda_width() as u64;
    let d = config.kda_head_dim as u64;
    let heads = config.head_count as u64;
    let id = config.indexer_head_dim as u64;
    let pool = config.indexer_pool as u64;
    let kda_bytes = w
        .checked_mul(9)?
        .checked_add(d.checked_mul(d)?.checked_mul(heads)?)?
        .checked_mul(4)?;
    let mla_bytes = (config.kv_lora_rank as u64)
        .checked_mul(position)?
        .checked_add(id.checked_mul(2)?.checked_mul(pool)?)?
        .checked_add(id.checked_mul(position / pool)?)?
        .checked_mul(2)?;
    kda.checked_mul(kda_bytes)?
        .checked_add(mla.checked_mul(mla_bytes)?)
}

/// Bytes `[0, len)` of a shared-storage tensor. Callers are synchronized
/// session code: no command writing it may be in flight.
fn read_prefix(tensor: &MetalTensor, len: usize) -> Result<Vec<u8>> {
    let end = tensor.offset.checked_add(len as u64);
    if (len as u64) > tensor.n_bytes() || end.is_none_or(|end| end > tensor.buffer.length() as u64)
    {
        return invalid(format!(
            "snapshot read of {len} bytes exceeds a session region"
        ));
    }
    // SAFETY: shared storage; range checked; no in-flight writer.
    let slice = unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize),
            len,
        )
    };
    Ok(slice.to_vec())
}

/// Whether `bytes` fit at the start of `tensor`.
fn fits(tensor: &MetalTensor, bytes: &[u8]) -> bool {
    (bytes.len() as u64) <= tensor.n_bytes()
        && tensor
            .offset
            .checked_add(bytes.len() as u64)
            .is_some_and(|end| end <= tensor.buffer.length() as u64)
}

/// Writes `bytes` at the start of `tensor` (checked by [`fits`] first).
fn write_prefix(tensor: &MetalTensor, bytes: &[u8]) {
    // SAFETY: shared storage; the caller validated the range with `fits`;
    // no command touching the session is in flight.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize),
            bytes.len(),
        );
    }
}

impl Glm5NextSession<'_> {
    /// Captures the persistent state at the committed position. Refuses a
    /// poisoned session (its state may be partly advanced).
    pub fn capture_snapshot(&self) -> Result<Glm5NextSnapshot> {
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        let c = &self.weights.config;
        let n = self.position;
        let (kv, id, pool) = (
            c.kv_lora_rank as usize,
            c.indexer_head_dim as usize,
            c.indexer_pool as usize,
        );
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                Ok(match layer {
                    LayerState::Kda { conv, state } => LayerRegions::Kda {
                        conv: read_prefix(conv, conv.n_bytes() as usize)?,
                        state: read_prefix(state, state.n_bytes() as usize)?,
                    },
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    } => LayerRegions::Mla {
                        latent: read_prefix(latent, kv * n * 2)?,
                        pending: read_prefix(pending, pending.n_bytes() as usize)?,
                        pooled: read_prefix(pooled, id * (n / pool) * 2)?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Glm5NextSnapshot {
            position: n,
            weights_instance: self.weights.instance,
            lineage: self.packed_lineage(),
            policy_version: SNAPSHOT_POLICY_VERSION,
            layers,
        })
    }

    /// Restores `snapshot` into this session: same weights instance,
    /// lineage and policy; room for at least one more token; every region
    /// within its destination. Nothing is written until all checks pass;
    /// the session is then poisoned until every region is installed, and
    /// its position becomes the snapshot's.
    pub fn restore_snapshot(&mut self, snapshot: &Glm5NextSnapshot) -> Result<()> {
        let refuse = |detail: String| Err(Glm5NextMetalError::SnapshotMismatch(detail));
        if snapshot.weights_instance != self.weights.instance {
            return refuse("captured from another weights instance".into());
        }
        if snapshot.policy_version != SNAPSHOT_POLICY_VERSION {
            return refuse(format!(
                "policy version {} is not {SNAPSHOT_POLICY_VERSION}",
                snapshot.policy_version
            ));
        }
        if snapshot.lineage != self.packed_lineage() {
            return refuse(format!(
                "lineage {:?} differs from this session's {:?}",
                snapshot.lineage,
                self.packed_lineage()
            ));
        }
        if snapshot.position >= self.capacity {
            return refuse(format!(
                "position {} leaves no room in capacity {}",
                snapshot.position, self.capacity
            ));
        }
        if snapshot.layers.len() != self.layers.len() {
            return refuse("layer count differs".into());
        }
        for (layer, regions) in self.layers.iter().zip(&snapshot.layers) {
            let ok = match (layer, regions) {
                (LayerState::Kda { conv, state }, LayerRegions::Kda { conv: c, state: s }) => {
                    c.len() as u64 == conv.n_bytes()
                        && s.len() as u64 == state.n_bytes()
                        && fits(conv, c)
                        && fits(state, s)
                }
                (
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    },
                    LayerRegions::Mla {
                        latent: l,
                        pending: p,
                        pooled: q,
                    },
                ) => {
                    p.len() as u64 == pending.n_bytes()
                        && fits(latent, l)
                        && fits(pending, p)
                        && fits(pooled, q)
                }
                _ => false,
            };
            if !ok {
                return refuse("a region does not match its destination".into());
            }
        }
        self.poisoned = true;
        for (layer, regions) in self.layers.iter().zip(&snapshot.layers) {
            match (layer, regions) {
                (LayerState::Kda { conv, state }, LayerRegions::Kda { conv: c, state: s }) => {
                    write_prefix(conv, c);
                    write_prefix(state, s);
                }
                (
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    },
                    LayerRegions::Mla {
                        latent: l,
                        pending: p,
                        pooled: q,
                    },
                ) => {
                    write_prefix(latent, l);
                    write_prefix(pending, p);
                    write_prefix(pooled, q);
                }
                _ => unreachable!("validated above"),
            }
        }
        self.position = snapshot.position;
        self.poisoned = false;
        Ok(())
    }
}

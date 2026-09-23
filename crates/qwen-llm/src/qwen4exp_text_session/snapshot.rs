//! RAM snapshots of a text session at a committed length `n`.
//!
//! A snapshot holds exactly the state the forward reads from earlier tokens:
//! each GDN layer's conv and delta state, the PLE convolution history and
//! n-gram token history, and for each QSA layer its pending index-key block,
//! `n / ratio` compressed index keys, and `n` F16 K/V rows. Hyper-residual,
//! residual, and MoE buffers are rewritten from the token embedding every
//! step, and logits are not carried: a restore always prefills at least one
//! token. Guarded top-k and HC bindings belong to the workspace and are never
//! touched.

use super::*;

pub struct Qwen4ExpTextSnapshot {
    geometry: Qwen4ExpTextSessionMetalGeometry,
    length: usize,
    history: PleHistory,
    /// Byte length of each captured region, in capture order.
    regions: Vec<usize>,
    arena: Vec<u8>,
}

impl Qwen4ExpTextSnapshot {
    /// Committed tokens the snapshot continues from.
    pub fn length(&self) -> usize {
        self.length
    }

    pub fn payload_bytes(&self) -> u64 {
        self.arena.len() as u64
    }
}

impl Qwen4ExpTextSessionMetalGeometry {
    /// Payload of a snapshot at `length` committed tokens: fixed recurrent
    /// state plus QSA rows proportional to `length`.
    pub fn snapshot_bytes(&self, length: usize) -> Result<u64, Qwen4ExpTextSessionError> {
        if length == 0 || length > self.capacity {
            return invalid(format!(
                "snapshot length {length} is outside 1..={}",
                self.capacity
            ));
        }
        // Layers zero and one are GDN with the same geometry.
        let mut bytes = 2 * self.zero_one.layer_zero().gdn().snapshot_bytes()
            + self.zero_one.ple().conv_state_bytes();
        for block in &self.post_ple {
            bytes += match block.mixer() {
                Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(gdn) => gdn.snapshot_bytes(),
                Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(qsa) => {
                    qsa.snapshot_bytes(length)
                }
            };
        }
        Ok(bytes as u64)
    }
}

impl Qwen4ExpTextSessionMetalWorkspace {
    fn snapshot_regions(
        &self,
        length: usize,
    ) -> Result<Vec<MetalTensor>, Qwen4ExpTextSessionError> {
        let mut regions = self.zero_one.persistent_state_tensors();
        for block in &self.post_ple {
            regions.extend(block.snapshot_regions(length)?);
        }
        Ok(regions)
    }

    fn require_snapshot_ready(&self) -> Result<usize, Qwen4ExpTextSessionError> {
        self.require_idle()?;
        if self.state_poisoned || self.encode_failed || self.pending_length.is_some() {
            return invalid("snapshot requires a healthy released session");
        }
        let length = self.committed_length;
        if length == 0 {
            return invalid("an empty session has nothing to snapshot");
        }
        if self.zero_one.next_position() != Some(length as u64) {
            return invalid(format!(
                "PLE history position {:?} differs from committed length {length}",
                self.zero_one.next_position()
            ));
        }
        if let Some((layer, actual)) = self
            .qsa_committed_lengths()
            .into_iter()
            .find(|(_, actual)| *actual != length)
        {
            return invalid(format!(
                "QSA layer {layer} length {actual} differs from committed length {length}"
            ));
        }
        Ok(length)
    }

    /// Payload bytes a capture at the current committed length would hold.
    pub fn snapshot_bytes(&self) -> Result<u64, Qwen4ExpTextSessionError> {
        let length = self.require_snapshot_ready()?;
        self.geometry.snapshot_bytes(length)
    }

    pub fn capture_snapshot(&self) -> Result<Qwen4ExpTextSnapshot, Qwen4ExpTextSessionError> {
        let length = self.require_snapshot_ready()?;
        let expected = self.geometry.snapshot_bytes(length)? as usize;
        let tensors = self.snapshot_regions(length)?;
        let regions = tensors
            .iter()
            .map(|tensor| tensor.n_bytes() as usize)
            .collect::<Vec<_>>();
        if regions.iter().sum::<usize>() != expected {
            return invalid(format!(
                "snapshot regions total {} bytes, geometry expects {expected}",
                regions.iter().sum::<usize>()
            ));
        }
        let mut arena = Vec::new();
        arena.try_reserve_exact(expected).map_err(|error| {
            Qwen4ExpTextSessionError::Invalid(format!(
                "snapshot arena of {expected} bytes refused: {error}"
            ))
        })?;
        for tensor in &tensors {
            // SAFETY: shared-storage buffers; the session is idle, so every
            // command that wrote these rows has completed. The view lies
            // inside its allocation (checked by `view_subrange`).
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(tensor.offset as usize),
                    tensor.n_bytes() as usize,
                )
            };
            arena.extend_from_slice(bytes);
        }
        Ok(Qwen4ExpTextSnapshot {
            geometry: self.geometry.clone(),
            length,
            history: self.zero_one.history().clone(),
            regions,
            arena,
        })
    }

    /// Reset, then install `snapshot` so the next forward runs at position
    /// `snapshot.length()`. On failure the session is left reset.
    pub fn restore_snapshot(
        &mut self,
        snapshot: &Qwen4ExpTextSnapshot,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        if snapshot.geometry != self.geometry {
            return invalid("snapshot geometry differs from this session");
        }
        self.reset()?;
        let result = self.install_snapshot(snapshot);
        if result.is_err() {
            self.reset()?;
        }
        result
    }

    fn install_snapshot(
        &mut self,
        snapshot: &Qwen4ExpTextSnapshot,
    ) -> Result<(), Qwen4ExpTextSessionError> {
        let tensors = self.snapshot_regions(snapshot.length)?;
        if tensors.len() != snapshot.regions.len()
            || tensors
                .iter()
                .zip(&snapshot.regions)
                .any(|(tensor, &bytes)| tensor.n_bytes() as usize != bytes || !tensor.is_writable())
            || snapshot.regions.iter().sum::<usize>() != snapshot.arena.len()
        {
            return invalid("snapshot layout differs from this session's state regions");
        }
        let mut offset = 0;
        for (tensor, &bytes) in tensors.iter().zip(&snapshot.regions) {
            // SAFETY: the session is idle and reset; the destination view is
            // writable, shared-storage, inside its allocation, and `bytes`
            // long, matching the source slice.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    snapshot.arena[offset..offset + bytes].as_ptr(),
                    tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(tensor.offset as usize),
                    bytes,
                );
            }
            offset += bytes;
        }
        self.zero_one.restore_history(snapshot.history.clone())?;
        for block in &mut self.post_ple {
            block.restore_mixer_length(snapshot.length)?;
        }
        self.committed_length = snapshot.length;
        self.logits_ready = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(capacity: usize) -> Qwen4ExpTextSessionMetalGeometry {
        Qwen4ExpTextSessionMetalGeometry::from_config(
            &Qwen4ExpConfig::flash_next_reference(),
            capacity,
        )
        .unwrap()
    }

    #[test]
    fn snapshot_bytes_are_fixed_state_plus_qsa_rows() {
        let geometry = geometry(8192);
        let bytes = |n| geometry.snapshot_bytes(n).unwrap();
        let qsa = geometry.qsa_layers().len() as u64;
        assert_eq!(qsa, 12);
        let ratio = geometry
            .post_ple()
            .iter()
            .find_map(|block| block.mixer().qsa())
            .unwrap()
            .compression_ratio();
        assert_eq!(ratio, 4);
        // Per QSA layer: one F16 K+V row per token, one F16 compressed index
        // key per completed block of `ratio` tokens.
        let row = (bytes(3) - bytes(2)) / qsa;
        let block = (bytes(4) - bytes(3)) / qsa - row;
        let fixed = bytes(1) - qsa * row;
        for n in [1, 3, 4, 5, 8, 4097, 8192] {
            let n = n as u64;
            assert_eq!(
                bytes(n as usize),
                fixed + qsa * (n * row + n / 4 * block),
                "n={n}"
            );
        }
        // Pinned release shapes (docs/SERVE.md): 36 GDN layers of conv +
        // delta state, PLE history, and 12 pending index-key blocks.
        assert_eq!((row, block), (2048, 256));
        assert_eq!(fixed, 118_063_104);
    }

    #[test]
    fn snapshot_bytes_reject_empty_and_over_capacity() {
        let geometry = geometry(64);
        assert!(geometry.snapshot_bytes(0).is_err());
        assert!(geometry.snapshot_bytes(65).is_err());
        assert!(geometry.snapshot_bytes(64).is_ok());
    }
}

//! CPU-only representation proof; no production snapshot or runtime ABI change.

use std::sync::Arc;

#[path = "snapshot_segments_lifecycle_pilot.rs"]
mod lifecycle;

const SLAB_ROWS: usize = 1024;
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone)]
struct Segment {
    rows: usize,
    sealed: bool,
    bytes: Arc<Vec<u8>>,
}

#[derive(Clone)]
struct Snapshot {
    layers: usize,
    row_bytes: usize,
    rows: usize,
    segments: Vec<Segment>,
}

fn size(layers: usize, rows: usize, row_bytes: usize) -> Result<usize> {
    if layers == 0 || row_bytes == 0 {
        return Err("empty layout");
    }
    layers
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(row_bytes))
        .ok_or("size overflow")
}

impl Snapshot {
    // Caller owns provenance: only its actual restored/published anchor is valid.
    fn capture_from(
        layers: &[&[u8]],
        row_bytes: usize,
        rows: usize,
        anchor: Option<&Self>,
    ) -> Result<(Self, usize)> {
        size(layers.len(), rows, row_bytes)?;
        if layers.iter().any(|s| s.len() < rows * row_bytes) {
            return Err("source shape");
        }
        if let Some(base) = anchor {
            base.validate()?;
            if base.layers != layers.len() || base.row_bytes != row_bytes {
                return Err("anchor layout");
            }
            if base.rows > rows {
                return Err("anchor ahead");
            }
            if base.rows == rows {
                return Ok((base.clone(), 0));
            }
        }
        let mut segments = Vec::new();
        let mut start = 0;
        if let Some(base) = anchor {
            for s in base.segments.iter().take_while(|s| s.sealed) {
                start += s.rows;
                segments.push(s.clone());
            }
        }
        let mut copied = 0;
        while start < rows {
            let span_rows = if anchor.is_none() {
                rows
            } else {
                (rows - start).min(SLAB_ROWS)
            };
            let mut bytes = Vec::with_capacity(size(layers.len(), span_rows, row_bytes)?);
            for layer in layers {
                bytes.extend_from_slice(&layer[start * row_bytes..(start + span_rows) * row_bytes]);
            }
            copied += bytes.len();
            segments.push(Segment {
                rows: span_rows,
                sealed: anchor.is_none() || span_rows == SLAB_ROWS,
                bytes: Arc::new(bytes),
            });
            start += span_rows;
        }
        let result = Self {
            layers: layers.len(),
            row_bytes,
            rows,
            segments,
        };
        result.validate()?;
        Ok((result, copied))
    }

    fn from_flat(layers: usize, rows: usize, row_bytes: usize, bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() != size(layers, rows, row_bytes)? {
            return Err("flat length");
        }
        // The imported root is sealed even when its length is not slab-aligned.
        let segments = if rows == 0 {
            Vec::new()
        } else {
            vec![Segment {
                rows,
                sealed: true,
                bytes: Arc::new(bytes),
            }]
        };
        Ok(Self {
            layers,
            row_bytes,
            rows,
            segments,
        })
    }

    fn validate(&self) -> Result<()> {
        size(self.layers, self.rows, self.row_bytes)?;
        let mut rows = 0usize;
        for (index, s) in self.segments.iter().enumerate() {
            if s.rows == 0 || s.bytes.len() != size(self.layers, s.rows, self.row_bytes)? {
                return Err("segment length");
            }
            if !s.sealed && (index + 1 != self.segments.len() || s.rows >= SLAB_ROWS) {
                return Err("unsealed segment");
            }
            rows = rows.checked_add(s.rows).ok_or("row overflow")?;
        }
        if rows != self.rows {
            return Err("row coverage");
        }
        Ok(())
    }

    fn copy_to(&self, dst: &mut [Vec<u8>]) -> Result<()> {
        self.validate()?;
        let used = self
            .rows
            .checked_mul(self.row_bytes)
            .ok_or("size overflow")?;
        if dst.len() != self.layers || dst.iter().any(|b| b.len() < used) {
            return Err("destination shape");
        }
        let mut start = 0;
        for s in &self.segments {
            let span = s.rows * self.row_bytes;
            for (layer, out) in dst.iter_mut().enumerate() {
                out[start..start + span]
                    .copy_from_slice(&s.bytes[layer * span..(layer + 1) * span]);
            }
            start += span;
        }
        Ok(())
    }

    fn materialize(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut layers = vec![vec![0; self.rows * self.row_bytes]; self.layers];
        self.copy_to(&mut layers)?;
        Ok(layers.concat())
    }

    fn logical_bytes(&self) -> usize {
        self.segments.iter().map(|s| s.bytes.len()).sum()
    }
}

struct State {
    layers: Vec<Vec<u8>>,
    row_bytes: usize,
    rows: usize,
    capacity: usize,
    anchor: Option<Snapshot>,
}

impl State {
    fn fresh(layers: usize, row_bytes: usize, capacity: usize) -> Result<Self> {
        size(layers, capacity, row_bytes)?;
        Ok(Self {
            layers: vec![vec![0; capacity * row_bytes]; layers],
            row_bytes,
            rows: 0,
            capacity,
            anchor: None,
        })
    }

    fn restore(base: &Snapshot, capacity: usize) -> Result<Self> {
        base.validate()?;
        if base.rows > capacity {
            return Err("restore capacity");
        }
        let mut state = Self::fresh(base.layers, base.row_bytes, capacity)?;
        base.copy_to(&mut state.layers)?;
        state.rows = base.rows;
        state.anchor = Some(base.clone());
        Ok(state)
    }

    fn append(&mut self, rows: usize, bytes: &[u8]) -> Result<()> {
        let end = self.rows.checked_add(rows).ok_or("append overflow")?;
        if end > self.capacity {
            return Err("append capacity");
        }
        if bytes.len() != size(self.layers.len(), rows, self.row_bytes)? {
            return Err("append shape");
        }
        let span = rows * self.row_bytes;
        for (layer, dst) in self.layers.iter_mut().enumerate() {
            dst[self.rows * self.row_bytes..end * self.row_bytes]
                .copy_from_slice(&bytes[layer * span..(layer + 1) * span]);
        }
        self.rows = end;
        Ok(())
    }

    fn raw_mut(&mut self) -> &mut [Vec<u8>] {
        self.anchor = None;
        &mut self.layers
    }

    fn rewind(&mut self, rows: usize) -> Result<()> {
        if rows > self.rows {
            return Err("rewind forward");
        }
        if rows != self.rows {
            self.anchor = None;
        }
        self.rows = rows;
        Ok(())
    }

    fn capture(&mut self) -> Result<(Snapshot, usize)> {
        let layers: Vec<_> = self.layers.iter().map(Vec::as_slice).collect();
        let (result, copied) =
            Snapshot::capture_from(&layers, self.row_bytes, self.rows, self.anchor.as_ref())?;
        self.anchor = Some(result.clone());
        Ok((result, copied))
    }

    fn canonical(&self) -> Vec<u8> {
        self.layers
            .iter()
            .flat_map(|b| b[..self.rows * self.row_bytes].iter().copied())
            .collect()
    }
}

fn pattern(layers: usize, rows: usize, row_bytes: usize, seed: usize) -> Vec<u8> {
    let mut x = (seed as u64) + 1;
    (0..layers * rows * row_bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

#[test]
fn segmented_snapshot_restores_and_shares_only_immutable_prefixes() {
    for root_rows in [0, 1, 1023, 1024, 1025, 32752] {
        let data = pattern(3, root_rows, 7, root_rows);
        let original = data.clone();
        let ptr = data.as_ptr();
        let root = Snapshot::from_flat(3, root_rows, 7, data).unwrap();
        if root_rows > 0 {
            assert_eq!(root.segments[0].bytes.as_ptr(), ptr);
        }
        let mut state = State::restore(&root, root_rows + 5000).unwrap();
        for (step, rows) in [0, 1, 16, 1006, 1, 1024, 1025].into_iter().enumerate() {
            let before = state.rows;
            state.append(rows, &pattern(3, rows, 7, step + 73)).unwrap();
            let (snapshot, copied) = state.capture().unwrap();
            assert_eq!(snapshot.materialize().unwrap(), state.canonical());
            assert_eq!(snapshot.logical_bytes(), state.rows * 3 * 7);
            assert!(snapshot.segments.len() <= 1 + (state.rows - root_rows).div_ceil(SLAB_ROWS));
            assert!(copied <= (rows + SLAB_ROWS - 1) * 3 * 7);
            if rows == 0 {
                assert_eq!(copied, 0);
            }
            if root_rows > 0 {
                assert!(Arc::ptr_eq(
                    &root.segments[0].bytes,
                    &snapshot.segments[0].bytes
                ));
            }
            assert_eq!(root.materialize().unwrap(), original);
            let restored = State::restore(&snapshot, snapshot.rows).unwrap();
            assert_eq!(restored.canonical(), state.canonical());
            assert!(state.rows >= before);
        }
    }
}

#[test]
fn segmented_snapshot_mutation_rewind_and_bad_inputs_fail_closed() {
    let root = Snapshot::from_flat(2, 1025, 4, pattern(2, 1025, 4, 9)).unwrap();
    let original = root.materialize().unwrap();
    assert!(State::restore(&root, 1024).is_err());
    assert!(State::fresh(usize::MAX, 8, 2).is_err());
    assert!(Snapshot::from_flat(2, 2, 4, vec![0; 3]).is_err());
    let mut state = State::restore(&root, 1030).unwrap();
    assert!(state.append(6, &pattern(2, 6, 4, 1)).is_err());
    assert!(state.append(1, &[0]).is_err());
    assert_eq!(state.canonical(), original);
    assert_eq!(state.capture().unwrap().1, 0);
    state.raw_mut()[0][0] ^= 0xff;
    let (changed, copied) = state.capture().unwrap();
    assert_eq!(copied, 1025 * 2 * 4);
    assert!(!Arc::ptr_eq(
        &root.segments[0].bytes,
        &changed.segments[0].bytes
    ));
    assert_eq!(changed.materialize().unwrap(), state.canonical());
    assert_eq!(root.materialize().unwrap(), original);
    state.rewind(1024).unwrap();
    assert_eq!(state.capture().unwrap().1, 1024 * 2 * 4);
    assert!(state.rewind(1025).is_err());
    let mut bad = root.clone();
    bad.segments[0].rows -= 1;
    assert!(State::restore(&bad, 2048).is_err());
}

#[test]
fn segmented_snapshot_forks_do_not_infer_provenance_or_retain_ancestors() {
    let original = pattern(2, 17, 3, 19);
    let root = Snapshot::from_flat(2, 17, 3, original.clone()).unwrap();
    let mut restored = State::restore(&root, 4096).unwrap();
    let mut unrelated = State::fresh(2, 3, 4096).unwrap();
    unrelated.append(17, &original).unwrap();
    let (independent, copied) = unrelated.capture().unwrap();
    assert_eq!(copied, original.len());
    assert!(!Arc::ptr_eq(
        &root.segments[0].bytes,
        &independent.segments[0].bytes
    ));

    restored.append(1, &pattern(2, 1, 3, 20)).unwrap();
    let (first, _) = restored.capture().unwrap();
    let first_bytes = first.materialize().unwrap();
    let abandoned_tail = Arc::downgrade(&first.segments[1].bytes);
    let mut fork = State::restore(&first, 4096).unwrap();
    fork.append(1024, &pattern(2, 1024, 3, 21)).unwrap();
    let (forked, copied) = fork.capture().unwrap();
    assert_eq!(copied, 1025 * 2 * 3);
    assert_eq!(first.materialize().unwrap(), first_bytes);
    assert_eq!(forked.materialize().unwrap(), fork.canonical());
    restored.append(1024, &pattern(2, 1024, 3, 22)).unwrap();
    let (sibling, _) = restored.capture().unwrap();
    assert_ne!(
        forked.materialize().unwrap(),
        sibling.materialize().unwrap()
    );
    assert!(Arc::ptr_eq(
        &forked.segments[0].bytes,
        &sibling.segments[0].bytes
    ));
    drop(first);
    assert!(
        abandoned_tail.upgrade().is_none(),
        "new snapshots retain old partial tail"
    );

    // Exposing mutable state invalidates even when the caller makes no change.
    let _ = fork.raw_mut();
    assert_eq!(fork.capture().unwrap().1, fork.rows * 2 * 3);
    fork.raw_mut()[0].clear();
    assert!(fork.capture().is_err());
    assert!(fork.anchor.is_none());
    assert_eq!(root.materialize().unwrap(), original);
}

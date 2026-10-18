# Old REAP-272E loader

This module is for LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E at
`f792019bcb369768220f01088f0f7af230786982`, not the strict 1017 contract.
The caller remains responsible for pinned-download provenance. Header geometry
does not authenticate checkpoint contents. Source files must remain immutable.

```rust,ignore
let main = SourceStore::open(root)?;
let base = Arc::new(SourceStore::open_engram(root)?);
let bytes = ResidentWeights::plan(&main, &base, &old_config)?;
let weights = ResidentWeights::load(
    &ctx, &main, base, &old_config,
    LoadLimits { resident_bytes: weight_budget },
)?;
let gate = weights.bank(0, "w1").unwrap();
let up = weights.bank(0, "w3").unwrap();
let down = weights.bank(0, "w2").unwrap();
let rows = weights.dense_rows("layers.0.attn.wq_a.weight", 0..32)?;
let embeddings = weights.engram.decode_rows(1, &hashed_row_ids)?;
```

`open` reads bounded headers of indexed completed shards, or scans safetensors
files when there is no index. `missing_shards` and `missing_tensors` allow download
progress inspection; `descriptors()` exposes available canonical names, paths and
checked byte ranges. Present but malformed/truncated shards fail. `open_engram`
opens only original-base shards 47/48, ignoring a colocated REAP index. Those
shards also supply missing small Engram projections. Main copies take precedence.
Paths must be single filenames; links escaping the canonical root are rejected.

`plan` checks every main expert pair (40 layers, 272 experts, gate/up/down), dense
scale geometry and the old Engram table geometry without allocating Metal memory.
It accounts for all available main dense weights, not MTP, vision, or table
residency. Forward still owns full operator/config compatibility checks. Missing
indexed main tensors fail preflight; a directory without an index cannot establish
the completeness of every nonexpert projection.

Each packed bank is a public `MetalTensor`, with its public Metal buffer, offset
zero, MXFP4 dtype and native `[K, M, 272]` shape. Expert bytes are exactly
204,550,963,200 (190.503 GiB). One expert's codes and scales (6,266,880 bytes) are
read at a time and repacked directly into resident 17-byte GGML blocks. Bulk
source reads use descriptor-local `F_NOCACHE` on macOS to avoid retaining a
file-cache copy beside resident weights; the separate Engram handles keep their
normal row-read cache policy. No second resident bank or expanded expert matrix
is constructed. Dense BF16/F32 and
FP8+UE8M0 tensors become resident F32 using at most 32-row tiles. The exact dense
footprint comes from `plan`; fitting the entire model in a given budget is not
assumed. Published weight buffers must not be written by CPU or GPU callers.

`ByteCounters` separates packed experts, dense F32, disk tables, skipped source
bytes, source payload bytes read during load, and peak payload staging. Staging
is a conservative bound, excluding header/index metadata, allocator overhead,
forward/KV storage and on-demand Engram output. Load prices actual Metal buffer
allocations and checks device/process headroom as well as the caller's explicit
weight budget. It never raises iogpu limits. Leave forward/KV headroom in that
budget. Allocation/data errors drop partial residency; loading is not resumable.

`bounded_read` and `decode_rows` cap individual reads/output at 32 MiB. FP8 dense
scales cover 32x32 blocks; Engram scales cover 32 columns of one row. Engram uses
positional file reads only, preserves ID order and duplicates, and does not mmap,
expand or cache entire tables. IDs are already hashed/compressed; tokenizer
mapping and hashing belong to forward. Non-finite decoded values are errors.

Tests use small split/padded shards, byte/row bounds, scale boundaries, malformed
paths/indexes, packing canaries and a tiny real Metal bank. No full checkpoint is
loaded by tests.

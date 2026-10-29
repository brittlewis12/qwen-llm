# V4.1 Attention Half-KV Reuse: Design and Budget

2026-10-29. Follow-up to the GPU attribution
(`2026-10-29-v41-ced-gpu-attribution.md`). This is a design and probe plan,
not a measured gain.

## Finding

The tensor64 prefill tile kernels convert the same KV rows to half twice per
query. `kernel_deepseek_v41_attention_score_tensor64` loads each key row with
`attention_row_key(...)`, converts it with `half4(v)` into `kblk`, and runs
QK^T. `kernel_deepseek_v41_attention_pv_tensor64` loads the value row with the
same `attention_row_key(...)` helper, converts it with the same `half4(v)`
into `vblk`, and runs PV. The key and value rows are the same latent rows.

Two additional sharing dimensions exist:

- the raw window rows of adjacent queries overlap almost completely;
- the compressed owner rows are shared by every query and every consumer
  layer that reads that owner.

So the current path reads the window/owner KV as F32 and converts it for the
score, then re-reads and re-converts the same rows for the PV.

## Proposal

Build bounded half mirrors once and let both consumers read them:

- a per-chunk history half mirror, produced while the existing history gather
  writes its F32 result (no extra read);
- a per-publication owner half mirror, produced once after the owner rows are
  published and reused by every consumer layer and chunk that reads that
  owner.

Kernel variants read the half mirror for the window and selected rows; the
selected ids index the half owner; the matmul order, scale, softmax and
normalization are unchanged. The slab values are the same `half4(f32 row)`
bits the kernels produce today, so bitwise equality is expected and must be
shown by the existing Full/CED pair and qualification.

## Budget

- Full 8K attention tiles: 1,850.0 ms of 7,453.3 ms GPU; the CED encoder
  equivalent is about 925 ms.
- Per encoder query the union of window and selected keys is about 1,024;
  F32 K/V reads are 1,024 x 512 x 4 B x 2 consumers, about 4 MiB per query.
  Over 163,840 encoder queries that is about 687 GB of F32 K/V traffic.
  Half mirrors halve those bytes and remove one of the two conversions.
- A 10% tile-kernel reduction is about 92 ms (2.1% of the 4.38 s wall); 20%
  is about 185 ms (4.2%). This is consistent with the attribution's 179 ms
  for a 10% complete attention-core cut.
- Construction is small: about 4.7 MiB of history and 8 MiB of owner per
  layer-chunk, roughly 1 ms of conversion writes over a chunk.

## Distinction From Closed Work

The historical gathered-half-slab negative paid a random gather over a
cohort of selected keys. This proposal gathers nothing: the history and
owner rows are already contiguous and the slab is a linear conversion. It is
also not the band-QK premise (no inflated FLOPs) and not the closed PV-axis
head regrouping.

## Probe and Bar

1. Microbenchmark `score_tensor64` + `pv_tensor64` at production shapes with
   F32 inputs versus half mirrors. Require at least a 10% tile-kernel
   reduction before host integration.
2. Integrate behind a toggle; run the same-binary CED A/B in two independent
   ABBA blocks on the frozen real 8K prompt. Require at least 1% wall and
   bitwise Full/CED endpoints, plus the private qualification.
3. Price the half mirrors in the CED/attention memory ledger before the A/B;
   no unaccounted peak.

## Risks

The F32 window/owner may already be resident in L2/SLC, so the DRAM saving
may be smaller than the traffic arithmetic; the probe must measure kernel
time, not bandwidth. Threadgroup/barrier overhead and extra bindings could
erase a small win. Owner/history mirror lifetime and invalidation must be
explicit, and the unit-price golden tests will change with the new scratch.

## Ownership

This touches attention kernels and `attention_gpu` scratch only. DEWM owns
the 8192-row chunk policy and the decoder unwind repair in
`gpu_step/ced/batches.rs`; that file is not part of this experiment.

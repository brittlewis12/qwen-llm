# V4.1 Prefill Chunk Scratch Reuse

2026-10-26. Source: local main `72e59e33` plus the two scratch commits
(`87e87b73` reuse, `a546b232` alternate-policy toggle). Artifact: GSQ-RCO
IQ2_XS (~144.6 GiB resident). Raw:
`metagross:~/qwen-runs/2026-10-26-v41-scratch-reuse/` (`campaign2-*`,
`warmup2.*`, `campaign2.log`; the first campaign without a warm-up and the
outsider acceptance are retained too).

## Decision

Promote the reuse. A prefill allocates one `ChunkScratch` (x/half/outputs/
incoming, `HcRows`, `ffn::Rows`, `attention::Rows`, and the Engram input/
projected tensors) for the largest chunk and reuses it for every full chunk; a
smaller tail drops the max scratch and allocates its own, so the peak stays
one set. `QWEN_V41_CHUNK_SCRATCH_REUSE=0`/`off` restores the old per-chunk
allocation; `alternate` toggles per call for in-process A/Bs. Default is reuse.

## Why

`prefill::allocate` uses `MetalTensor::zeros_dtype_unstaged`, which allocates a
fresh shared buffer and CPU-memsets the full extent. At 8K the chunk scratch is
several GiB; the second chunk allocated and zeroed a fresh set and could evict
model pages. Reusing the first chunk's scratch removes the second allocation/
memset and the eviction. The `incoming` lagged-pre buffer must be reset to
`[1,0,0,0]` at the start of every chunk; the first lifetime run caught this as
a capture-bit difference.

## Campaign A/B

Campaign lease (inherited-descriptor path, `72e59e33`), GSQ, pp8192, 120 s
cooldown, one discarded warm-up pair (current), then C N N C with 2 runs per
arm. Both arms build-identity match. Raw samples, in order:

| arm | samples (tok/s) | warm mean |
| --- | --- | ---: |
| current (C) | 758.14, 788.48 | 784.4 |
| reuse (N) | 856.37, 856.06 | 856.2 |
| reuse (N) | 781.88, 597.51 | 831.4 (781.9 warm) |
| current (C) | 247.75, 806.71 | 784.4 (806.7 warm) |

Warm comparison (excluding the two cold samples 247.75 and 597.51, retained):
current 784.44, reuse 831.44, **+5.99%**. The stable reuse arm is 856.2. The
discarded warm-up pair was 763.54/774.11. The box's warm level was below the
earlier ~850 (768-856), so compare within the campaign, not to the old number.

## Correctness

- 61 prefill tests pass; the census test now includes the
  `prefill_scratch_prepare` span.
- The self-leased M5 lifetime witness passes for 2/4/8 chunks with identical
  logits/captures and constant post-chunk device size (5,373,952 bytes);
  scratch high-water 510,528 bytes.
- The tail path allocates its own scratch after dropping the max set; the
  scalar-tail path is unchanged.

## Limits

One artifact and host session; the warm level was thermally variable; two cold
samples are retained but excluded from the warm comparison; no full-model
endpoint bitwise comparison (the toy lifetime witness validates the reuse and
the `incoming` reset, not full-model logits). The campaign lease serialized
the GPU; the page cache was shared with the other lanes on GSQ.

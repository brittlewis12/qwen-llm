# V4.1 Prefill Chunk Scratch Reuse

2026-10-26. Source: local main `72e59e33` plus the scratch commits (`87e87b73` reuse,
`a546b232` alternate-policy toggle, `87ce11b6` tail drop-first fix). Artifact: GSQ-RCO
IQ2_XS (~144.6 GiB resident). Raw:
`metagross:~/qwen-runs/2026-10-26-v41-scratch-reuse/` (`campaign2-*`,
`warmup2.*`, `campaign2.log`; the first campaign without a warm-up and the
outsider acceptance are retained too).

## Decision

Promote the reuse. A prefill allocates one `ChunkScratch` (x/half/outputs/
incoming, `HcRows`, `ffn::Rows`, `attention::Rows`, and the Engram input/
projected tensors) for the largest chunk and reuses it for every full chunk; a
smaller tail drops the max scratch before allocating its own, so the peak
stays one set. The in-binary reversed campaign gives **+3.54%** warm 8K
(reuse 863.7 vs per-chunk 834.2); the cross-build +5.99% is exploratory.
`QWEN_V41_CHUNK_SCRATCH_REUSE=0`/`off` restores the old per-chunk allocation;
`alternate` toggles per call for in-process A/Bs. Default is reuse.

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
current 784.44, reuse 831.44, **+5.99%**. This cross-build result is retained as
exploratory: low throughput alone does not establish coldness, the stable reuse
arm is two adjacent observations, and the exclusion needs a predefined
cache/I/O criterion. The discarded warm-up pair was 763.54/774.11. The box's
warm level was below the earlier ~850 (768-856), so compare within the
campaign, not to the old number.

The second campaign (campaign3) used the same binary `0a27eb05` and the
in-binary `QWEN_V41_CHUNK_SCRATCH_REUSE=0` toggle, reversed N C C N, under the
same campaign lease and a discarded warm-up. Raw samples, in order:

| arm | samples (tok/s) |
| --- | --- |
| reuse (N) | 863.66, 863.43 |
| per-chunk (C) | 834.46, 835.99 |
| per-chunk (C) | 832.98, 833.21 |
| reuse (N) | 864.69, 863.12 |

The warm-up pair was 862.58/860.16. No sample was cold; every sample is stable
and the order is reversed. Reuse 863.725 vs per-chunk 834.165, **+3.54%**; the
same binary and the in-binary toggle make this the robust warm estimate.

## Correctness

- 61 prefill tests pass; the census test now includes the
  `prefill_scratch_prepare` span.
- The self-leased M5 lifetime witness passes for 2/4/8 chunks with tails 1 and
  7, identical logits/captures/causal state, and a constant full-chunk device
  size of 5,373,952 bytes; the multi-row tail's post-chunk device is 5,029,888
  bytes, so the peak stays one set. Scratch high-water 510,528 bytes.
- The tail path now drops the max scratch before allocating the tail set; the
  earlier version constructed the replacement while the max set was still
  alive. The scalar continuation is exercised on the scalar tail; the
  multi-row tail's per-chunk reference leaves a different report/batch state,
  so that continuation is not part of this witness.

## Limits

One artifact and host session; the cross-build +5.99% is exploratory while the
in-binary reversed +3.54% is the robust warm estimate. The full-model endpoint
check passes: `qwen run --user` with a fixed prompt, 12 greedy tokens, and the
`0`/default toggle produced identical 75-byte output. The causal-state and
continuation comparison is still separate. Maya-S and then near-capacity IQ3
generality, the CED-origin guard and
the CED quick 1.9% (not comparable to the 8-10% matched-P4 cohort) are separate
follow-ups. The 8192 chunk remains a model until the production route counts
are merged. The campaign lease serialized the GPU; the page cache was shared
with the other lanes on GSQ.

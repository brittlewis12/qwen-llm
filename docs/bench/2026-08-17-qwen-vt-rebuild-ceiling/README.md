# Qwen Restored-Prefix V_T Rebuild Screen

Date: 2026-08-17

Status: **PREREGISTERED**. No candidate timing has run.

Baseline commit: `dda7a5e7e8f0cb63c4b8e29981d991ecf65dd9fe`.

## Question

On every packed-prefill invocation,
`attn_matrix_vt_valid_until` starts at zero. If a restored or otherwise resumed
Qwen prefix begins at position `P > 0`, each attention layer first scatters the
new `C` rows into canonical V and V_T, then transposes canonical V over
`0..P+C`. This reconstructs the entire transposed-V sidecar and rewrites the
current chunk a second time.

Is that exact derived-cache rebuild large enough at agentic 8K--32K prefixes to
justify a product-shaped attribution campaign? Separately, is the local no-risk
cleanup of transposing only `0..P` measurable enough to retain?

This model-free experiment measures isolated steady-state kernel-body cost. It
is not a strict bound on product savings: production interleaves each layer's
transpose with substantial compute, has different cache pressure, and still
needs some representation transfer under any cache design. Phase 0 can reject a
low-ceiling idea; it cannot authorize cache ownership or claim TTFT savings.

## Current Cost Model

For Qwen3.8 27B G6:

- 16 attention layers;
- 4 KV heads;
- head dimension 256;
- F16 canonical V and V_T; and
- 2 KiB of V per token per attention layer.

The exact read-plus-write traffic is
`L * n_kv * head_dim * rows * 2 bytes * 2 directions`. Across all layers this
is 64 KiB per row. A prefix-only B arm moves 512 MiB, 1 GiB, and 2 GiB at 8K,
16K, and 32K. A moves an additional 8 MiB for `C=128` or 64 MiB for `C=1024`.
The redundant current-chunk overlap is therefore proportional to `C`, not
`P+C`.

The current implementation is at
`crates/qwen-llm/src/metal_dflash.rs:7568`,
`crates/qwen-llm/src/metal_dflash.rs:8732`, and
`crates/qwen-llm/src/metal_dflash.rs:9225`. The fused current-chunk V_T scatter
is at `crates/qwen-llm/src/metal_dflash.rs:8758`.

## Phase 0: Model-Free Screen

Add one ignored release-test harness around the retained
`encode_attn_matrix_transpose_v_f16` kernel. It must not load a GGUF or execute
model code.

Exact geometry:

- `L=16`, `n_kv=4`, `head_dim=256`;
- restored prefixes `P={8192,16384,32768}`;
- suffix chunks `C={128,1024}`; and
- `n_pos=vt_stride=P+C`.

Allocate two independent source/destination bank sets X and Y, with one ordinary
Metal buffer per layer. Every dispatch must therefore use a distinct layer
buffer, and the two arms in a pair may not alias. Per-layer allocation avoids
assuming a buffer larger than the device's `maxBufferLength`; the harness must
record that limit and fail rather than skip if one layer does not fit.

Fully initialize and first-touch every source and destination page before
warmup. Sources use a deterministic nonzero pattern and destinations use
arm-specific sentinels. The resulting 32K working set is several GiB and uses
16 distinct source/destination pairs per arm; it must not replay one layer's
SLC-resident bytes.

Arms:

- **A / current:** call the kernel with `base_pos=0`, `n_rows=P+C`,
  `n_pos=P+C`, and `vt_stride=P+C` for all 16 layers.
- **B / overlap deletion:** call it with `base_pos=0`, `n_rows=P`,
  `n_pos=P+C`, and `vt_stride=P+C`; the common fused scatter of `P..P+C` is
  excluded from both timed arms.
- **Z / optimistic body estimate:** zero rebuild time. This is arithmetic, not
  a timed implementation or product bound; A's raw time is only the isolated
  body potentially removable.

Each arm uses one command buffer, one serial compute encoder, and 16
forward-order dispatches. The frozen warmup is `A_X, B_Y, A_Y, B_X`. The harness
then runs six fresh command-buffer pairs in frozen order with balanced physical
bank ownership:

```text
A_X B_Y, B_X A_Y, B_Y A_X, A_Y B_X, A_X B_Y, B_X A_Y
```

No pair may be retried or discarded. Wall timing starts immediately before
`commit` and ends after `waitUntilCompleted`; GPU time is the command buffer's
start/end timestamp delta. The median of six is the arithmetic mean of the
middle two sorted values. Report every raw arm, paired `A-B`, AB and BA medians,
wins, exact per-arm logical bytes, and per-arm decimal GB/s (`bytes / 10^9`). GPU
workloads are serialized.

Run one geometry per fresh test process in this frozen order:

```text
P8192/C128, P8192/C1024, P16384/C128,
P16384/C1024, P32768/C128, P32768/C1024
```

Record test-binary SHA-256, implementation commit, device name,
`maxBufferLength`, macOS version, physical memory, power source, memory pressure,
and thermal state before and after the chronology. No cell order changes,
retries, exclusions, or post-hoc cooling waits are allowed.

## Phase 0 Correctness

Before timing, a small deterministic nonzero-prefix fixture must use independent
A and B canonical/V_T destinations and establish that:

1. fused scatter writes current rows to canonical V and V_T with identical F16
   bits;
2. after a fresh scatter into each arm, full `0..P+C` transpose and prefix-only
   `0..P` transpose produce bit-identical V_T over every visible element and
   match an explicit CPU layout oracle; and
3. prefix-only transpose does not overwrite current rows or capacity padding.

The fixture must use patterned prefix/current rows, multiple KV heads and
dimensions, arm-specific sentinel padding, and `vt_stride > n_pos`. Existing
matrix-attention CPU oracles remain unchanged. Any mismatch kills and removes
the overlap-deletion candidate before timing. A retained source change also
requires an integrated nonzero-prefix matrix-attention test.

## Decision Gates

The isolated screen permits, but does not authorize, a product attribution only
if:

- median A GPU time is at least 5.0 ms at `P=32768` in both suffix cells;
- the median is nondecreasing across 8K, 16K, and 32K for each `C`;
- A's exact-byte effective rate lies between 50 and 800 GB/s in every 16K/32K
  cell; values outside that preregistered sanity range invalidate the packet
  rather than prove or kill the optimization.

The local prefix-only source change is retained only if:

- B wins at least five of six pairs at `P=32768,C=1024`, with positive median
  paired saving in both three-pair AB and BA strata;
- median paired GPU saving at that cell is at least 0.25 ms; and
- no cell's B median exceeds A by more than
  `max(0.05 ms, 0.02 * A_median)`.

Otherwise remove the prototype and bank a **KILL**. Exact byte deletion alone is
not sufficient to retain an unmeasurable branch or source complication.

## Phase 1, Only If Authorized

If the 32K screen passes, preregister a separate product A/B before changing
snapshot or cache ownership. Its control is the enabled online matrix path after
canonical prefix restore, not forced V4. It must report warm TTFT, packed suffix
wall, rebuild attribution, extra cache bytes, cache-capacity loss, and exact
continuation agreement at suffixes 128 and 1024.

A cache-owned V_T duplicate costs 1 GiB at 32K for this geometry. It is not a
no-lose optimization. Pointer-plus-position validity is also insufficient:
scratch can be reused with another session, and the same session buffers can be
overwritten by a different equal-length restore. Any persistence design needs a
session mutation epoch, typed lease, or cache-entry ownership that fails closed
on restore, rewind, and cross-session reuse.

## Safety Boundary

Phase 0 is model-free and uses ordinary Metal buffers only. It must not load a
GGUF or exercise a residency set, `requestResidency`, pre-wire, `mlock`,
cache-bypass read, uncached read, or residency-coupled pread path. PID 8770 is
user-owned and must remain untouched.

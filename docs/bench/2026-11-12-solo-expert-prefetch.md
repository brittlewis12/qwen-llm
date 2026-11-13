# Solo next-layer routed-expert prefetch oracle — 2026-11-12

M5 Ultra (80-core, 36-core CPU, 256 GiB), 32K context, B=1. V4.1 is the
four-shard `GSQ-RCO IQ2_XS` artifact; GLM is the stock `UD-IQ3_XXS` set.
`--profile iter --features perf-ablations`, one binary per family, same binary
for both arms, `nice -n 10`, canonical Metal lease. Thermal pressure was **0**
in every arm. Background `~/ghost` work was present throughout (~155 processes,
~291% CPU aggregate); it was left alone. The V4.1 campaign waited 306.7 s behind
the shared V4.1 regression-bisect lane; GLM acquired immediately.

Probe code is on branch `research/solo-expert-prefetch` at `3f3616de3`
(measured source `4b3c775ad`), base `1edb7e8d8`. Only this record and the raw
logs are landed; the hooks, kernel, and examples stay on that branch and are
not on `main`.

## Question and design

The reviewed proposal: during layer `l`'s latency-bound attention, read layer
`l+1`'s routed gate/up and down experts so layer `l+1`'s routed stage hits the
96–128 MiB / ~1.9 TB/s cache knee. Per-layer routed bytes are 58.2 MB (V4.1,
measured) and 72.7 MB (GLM, measured), inside or near that knee.

Two probes were built, both research-only:

- `examples/v41_solo_prefetch.rs`: captures the real per-layer top-k expert ids
  from a real greedy decode trajectory through `decode_bounds` (40 layers ×
  gate/up + down calls, real banks and ids), then replays that exact trajectory
  teacher-forced. The touch is recorded inside `attention_gpu::encode_core`
  after the query RoPE and index selection, binding the query and selected-id
  buffers so the causal scheduler places it in the attention core's dependency
  wave rather than wave 0.
- `examples/glm_solo_prefetch.rs`: reads the real per-block top-8 ids through
  `perf_decode_route_ids`, builds touch descriptors from the MoE banks, and
  opts the B=1 token onto the causal executor (the production GLM solo path is
  serial) with a per-block causal scope. A separate serial arm checks the
  executor preserved values.

Both arms are interleaved and order-alternated: three pairs per target, same
binary. `Same` touches the current layer's experts during its own attention
(shortest reuse distance); `Next` touches layer `l+1` (the reviewed proposal).
The touch kernel `kernels/research/solo_touch.metal` reads exactly the selected
expert rows of both gate/up banks or the down bank and writes only a private
checksum. All logits hashes are **bitwise equal** between control and touch
(V4.1 and GLM, both targets); GLM's causal control is bitwise equal to serial.

## Results

Control is the mean of the paired control arms. The first V4.1 `Same` touch
sample (22.13 ms) is excluded: it paid the lazy touch-pipeline compile; the
later two pairs are stable.

| Family / target | Control ms/token | Touch ms/token | Delta | Touch-only ms | Bitwise |
| --- | ---: | ---: | ---: | ---: | --- |
| V4.1 Same | 14.839 | 17.206 | **+2.367** | 2.683 | yes |
| V4.1 Next | 14.816 | 17.278 | **+2.461** | 2.731 | yes |
| GLM Same | 18.186 | 21.582 | **+3.396** | 4.685 | yes |
| GLM Next | 18.294 | 21.909 | **+3.615** | 4.727 | yes |

Touch-only prices every touched byte once, serially: V4.1 2.329 GB at
868/853 GB/s; GLM 3.052 GB at 651/646 GB/s. The measured token delta is
essentially the touch-only cost. The touch is **not hidden**; it is charged to
the token almost in full.

Cold then hot, one layer/block back to back (the exact selected rows):

| Family | Bytes | Cold GB/s | Hot GB/s | Hot/cold |
| --- | ---: | ---: | ---: | ---: |
| V4.1 layer 20 | 54.74 MB | 795 | 1,441 | **1.81×** |
| GLM block 3 | 71.83 MB | 628 | 729–736 | **1.16–1.17×** |

## Verdict: gate fails, close the class

The gate was **≥0.8 ms/token saved** at B=1 for either family. Both families
**lose** 2.4–3.6 ms/token. V4.1's routed read is ~3.63 ms at 896 GB/s (census);
at the measured hot 1,441 GB/s it would be ~2.26 ms, a **1.37 ms perfect-cache
bound** if the touch were free. But the touch costs 2.68 ms and less than a
tenth of it overlaps. GLM's hot gain is only 1.17×, so its perfect-cache bound
is ~0.2 ms, already below the gate even with a free touch. No variant of this
packet can pass.

## Why the Flash-Next negative repeats, and how experts differ

The 2026-11-08 Flash-Next packet found that broad known-weight packets
(35–84 MB) lost 16–44 µs/pair even at their best placement, while isolated hot
sets saved 25–27 µs; only the 13.3 MB HC packet was positive (~2.65 µs per QSA
handoff, ~0.023 ms/token). The mechanism is the same here, amplified:

1. **There is no idle DRAM during attention to hide behind.** The V4.1 causal
   census already shows the attention core and index co-scheduled *inside* the
   dense→routed waves, which are memory-bound. The touched bytes are added to
   an already-saturated wave; total token DRAM traffic is unchanged because
   the routed stage still reads the same bytes. Moving a read earlier only
   helps if the earlier window has spare bandwidth, which it does not.
2. **The touch is charged in the paired span.** With no idle window, the
   measured delta equals the touch-only read time (V4.1 2.37 vs 2.68 ms; GLM
   3.40 vs 4.69 ms), exactly the Flash-Next "loading those bytes and competing
   with current work" result.
3. **Experts differ in precision, not physics.** The oracle touches 100% of the
   exact routed bytes (no prediction, no partial packet), and V4.1's 54.7 MB
   per-layer set does reach 1.81× hot. That is a better cache story than
   Flash-Next's mixed packets. But the routed read is a mandatory read on the
   memory-bound critical path, so perfect precision does not create bandwidth.
4. **GLM is worse on both axes.** Its 72.7 MB per-block set is past the useful
   reuse region for this access pattern: only 1.16–1.17× hot, so even a fully
   hidden touch buys ~0.2 ms. Its serial solo path also needs the causal
   executor to overlap at all; the causal control here is 18.2 ms/token versus
   the published 17.62 ms serial GPU-loop standing.

The class is closed with these numbers. Stage 2 (predictor accuracy) is not
started: it would only price how to supply ids for a touch whose own cost
already exceeds any cache benefit.

## Evidence

M5 `~/qwen-runs/2026-11-12-solo-prefetch/`: `run-v41.log`, `run-glm.log`,
`build-v41.log`, `build-both.log`. Local copies of the run logs are in
`docs/bench/2026-11-12-solo-prefetch/`. Commands:

```sh
# V4.1
./target/iter/examples/v41_solo_prefetch MODEL 32768 16 3 both
# GLM
./target/iter/examples/glm_solo_prefetch MODEL 32768 16 3 both
```

Both under `QWEN_METAL_LEASE_WAIT=1 scripts/bench/with-metal-lease --task
solo-prefetch --for 15m -- nice -n 10 ...`. Prefill was 13.13 s (V4.1) and
44.22 s (GLM). The M5 bundle, worktree and build output were removed after the
run. The branch is unmerged; the probe code, kernel, and example binaries cited
above remain on `research/solo-expert-prefetch` at `3f3616de3` and are not on
`main`.

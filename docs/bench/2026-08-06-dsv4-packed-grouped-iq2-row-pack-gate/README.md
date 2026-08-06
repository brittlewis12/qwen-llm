# DeepSeek V4 Grouped-IQ2 Row-Pack Gate

Status: `HOLD - INCONCLUSIVE`. The exact multi-bin FFN-row-packing family is
closed as negative/inconclusive and removed from live source. Wall stationarity
fails the frozen packet, so this is not an economic KILL; stable GPU cells show
a consistent raw regression and provide no basis for asset work or rescue
tuning.

## Question

The exact-route V2 phase ceiling attributes 527.964 ms p95 across the 25 grouped
IQ2 gate/up/SwiGLU layers and authorizes one candidate design. The deployed
kernel maps one FFN row and up to 32 routed assignments to each SIMD group.
Exact route counts fill only 34.325% of its assignment-lane capacity.

Test one fixed row-packing family that places multiple independent FFN rows in
the inactive lanes of narrow assignment panels while preserving every dot
product's scalar reduction order. Do not sweep widths and do not touch down.

## Candidate

The sidecar planner bins each existing panel at width
`max(2, next_power_of_two(count))`:

| Width | Panels | FFN rows per SIMD group |
|---:|---:|---:|
| 2 | 614 | 16 |
| 4 | 215 | 8 |
| 8 | 207 | 4 |
| 16 | 226 | 2 |
| 32 | 486 | 1 |

Width 32 uses the deployed kernel unchanged. Four compile-time Metal
specializations preserve one 32-thread SIMD group and use at most 4 KiB TGM.
Physical row-zero lanes load each source F32 and broadcast its exact bits with
`simd_shuffle`; every lane still performs the unchanged IQ2 dequantization for
every packed row. Active lanes retain the deployed `bidx -> k0 -> j` order,
clamp/SwiGLU expression, route slot, and store layout.

Exact structural arithmetic is:

| Quantity | Deployed | Candidate |
|---|---:|---:|
| Assignment-lane capacity | 55,936 | 22,912 |
| Useful occupancy | 34.325% | 83.799% |
| Threadgroups | 3,579,904 | 1,466,368 |
| Issued gate/up lane-MAC capacity | 938,450,354,176 | 384,399,572,992 |

The nominal lane-MAC and threadgroup reduction is 59.039%. Useful MACs, scalar
dequant count, and semantic weight bytes are unchanged.

## Correctness Gate

Active release tests require bit-exact candidate output against the deployed
kernel for:

- N=1/12/31/32/33/64/128 mixed schedules;
- counts 1/2/3/4/5/8/9/16/17/31/32/33/63/64/65/96/97/127/128;
- continuation panels, row-unique weights, token-unique signed inputs, clamp,
  repeat identity, offset guards, and immutable inputs/weights/slots; and
- all 25 exact route fixtures with canonical planner SHA-256
  `58fecd77889f76bfc2e6690c80ce08d09f59de9b0b2c5e7f01fe26a9ff9a3cf3`.

All active gates pass. The frozen production-shape packet repeats full H=4096,
F=2048, clamp-10 exactness before and after timing for both cache regimes.
Baseline and candidate output digests are identical in each regime:

- disjoint:
  `bbf43456479d05e4097794dba99132df46588f25282f904917caf5ab68ad4cb2`;
- warm:
  `850ccf1105481a836ef0425e005c96257d8878a84d6719c5189a2740f98f5211`.

The packet also validates complete poison overwrite, finiteness, guards,
repeatability, immutable route/input/weight state, and 25 baseline versus 117
candidate dispatches.

## Frozen Performance Gate

Use 25 disjoint gate/up bank pairs and one all-expert warm pair. Each sample
retains 25 serial per-layer command buffers. Candidate width bins execute
serially inside each layer command.

After untimed exactness, first touch, topology validation, and one balanced
conditioning block, retain six blocks of:

```text
BD / CD / CW / BW / BW / CW / CD / BD
```

This yields 12 samples for each baseline/candidate x disjoint/warm cell. Keep
every sample; no filtering or rerun is allowed. Record summed GPU time and host
wall from immediately before first command creation through immediately after
the 25th completed wait.

Every cell must have at most 5% drift. For each regime and both GPU/wall:

```text
Q = 5 * max(baseline_range, candidate_range)
L_s = baseline_s - candidate_s - Q
threshold_s = max(158.3 ms, 0.25 * baseline_s)
```

For GPU, `Q` is also at least the V2 0.492790 ms empty bound. Median and
nearest-rank p95 must both satisfy `L_s >= threshold_s`. Invalidity or
instability is `INCONCLUSIVE`; a stable miss KILLs the family; a full pass only
authorizes one gate-only current-asset A/B.

## Result

All four GPU cells pass stationarity, but the candidate regresses every sample:

| Regime | Baseline median / p95 | Candidate median / p95 | Raw regression |
|---|---:|---:|---:|
| Disjoint GPU | 526.801 / 547.351 ms | 659.661 / 677.965 ms | 25.22% / 23.86% |
| Warm GPU | 526.850 / 549.683 ms | 658.666 / 683.148 ms | 25.02% / 24.28% |

GPU drift is 3.851979%/2.962459% for disjoint baseline/candidate and
4.265119%/3.758016% for warm baseline/candidate.

The whole packet is nevertheless invalid for an economic decision because wall
stationarity fails:

- disjoint candidate wall drift: 26.926177%;
- warm baseline wall drift: 15.339522%.

The retained final samples and the large wall ranges remain in the raw log.
Publish no contractual KILL, transferable timing result, product regression, or
promotion claim. The stable GPU arrays are raw negative observations only.

## Decision

Record `HOLD - INCONCLUSIVE`. Do not rerun, filter, tune a width, or spend a
current-asset gate. Remove the candidate kernels, planner/refactor, active tests,
and one-shot profiler from live source; their exact archived diff is the proper
home for this negative experiment.

Close exact multi-bin row packing and remove it from the active queue. Do not
generalize the result to every SIMD restructuring. Reopen only for a materially
different 25-dispatch design that preserves FFN-row parallelism.

The next gate-only candidate is dispatch-neutral SIMD weight broadcast inside
the existing one-row-per-threadgroup kernel: remove TGM traffic and both chunk
barriers while preserving current row parallelism and exact accumulation order.
It requires a new reviewed packet under the existing V2 gate-only authorization.

## Provenance

- Base revision: `ef7fabaf4adddebb7b8e54b332ba3a38bc9f0f87`.
- Device: Apple M4 Max.
- In-test/process elapsed: 82.94/83.05 seconds.
- Maximum RSS: 9,192,767,488 bytes; swaps: zero.
- Executed source diff SHA-256:
  `5745ff77e3b479101e46031b5406ee38ccbcdf9329585e0c2aabc28c65f83598`.
- Raw log SHA-256:
  `9cfba4b90af8f0ed155c27e31adf3f407a28f14274f96025f5281f924c857343`.
- Executed release test binary SHA-256:
  `cc946d5e09e1716d0d0909d09aa5020aa84be551af5259faa0751cdb3668f1c7`.
- Final retained release test binary SHA-256:
  `0cba2a97cdc83345cce88ee2ddd053a69e484e317ae7021bfdb45c9b7f7301dc`.
- Retained source diff SHA-256, empty by construction:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Checksum transcript: `checksums.log`.
- CX review: `019fd588-96b0-7033-afd1-66d7e354e523`.

Executed command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::\
profile_grouped_iq2_row_pack_production_shape \
  -- --ignored --exact --nocapture --test-threads=1
```

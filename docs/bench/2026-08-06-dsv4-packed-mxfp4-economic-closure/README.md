# DeepSeek V4 Packed MXFP4 Economic Closure

Status: `KILL` as a standalone grouped-MXFP4-down implementation under the
current packed-prefill economic contract. No new profiler, kernel, build, or
current-asset run is required.

## Question

The current asset has two routed-expert outliers:

- layer 26: `IQ3_S/IQ3_S/MXFP4`, 74 active expert buckets; and
- layer 42: `IQ3_XXS/IQ3_XXS/MXFP4`, 43 active expert buckets.

At N=128 each layer owns 768 route assignments. The ordinary fallback batches
gate and up by expert bucket, then issues one MXFP4 down matvec per assignment
and one scatter per bucket. A grouped down/scatter candidate would therefore
replace 1,536 matvec and 117 scatter dispatches while retaining gather, gate,
up, clamped SwiGLU, MXFP4 arithmetic, destination publication, and nonzero
replacement encoding.

The roadmap previously requested a direct production-shape ceiling before
writing that kernel. The accepted current-asset post-route packet already
contains a stronger supergraph ceiling: it times each complete routed stage for
both affected layers, including every operation the proposed kernel could
change and additional work it must retain.

## Existing Ceiling

The three accepted MXFP4-cohort routed-stage samples are:

```text
63.101463 ms
62.246291 ms
63.273002 ms
```

Their mean is 62.874 ms. The accepted observer has 0.275% sampled GPU drift,
no more than 0.199% absolute interpolation perturbation, complete timestamp
coverage at printed precision, and exact packed output/state evidence.

The established broad packed-prefill floor is:

```text
G = max(150 ms, 0.15 * 1054.962 ms)
  = 158.2443 ms
  -> 158.3 ms
```

Give the hypothetical candidate impossible credit for deleting the larger
63.273002 ms complete-stage sample. It reaches only:

```text
63.273002 / 158.3   = 39.97% of the economic floor
63.273002 / 1054.962 = 6.00% of the accepted anchor
```

The real kernel necessarily saves less because its target is a strict subset
of the measured stage and its replacement has nonzero cost. A down-only
model-free zero-work profiler cannot enlarge this current-asset upper bound.

## Host Caveat

The accepted 62.874 ms value is a GPU stage measurement, not a direct host
encoder measurement. To bridge the conservative gap from the largest observed
stage sample to the 158.3 ms floor, eliminated host work would need to exceed:

```text
158.3 - 63.273002 = 95.026998 ms
95.026998 / 1653  = 57.49 us per affected dispatch
```

That grants zero GPU and host cost to the grouped replacement. No current
stable telemetry establishes host encoding near this level. The caveat is a
reopen condition, not authorization to build a profiler around an unsupported
premise.

## Decision

Close standalone grouped MXFP4-down work under the current asset, device, and
economic contract. Do not implement a kernel, construct another synthetic
bank, or pay another model residency merely to rediscover a ceiling already
bounded by the complete current-asset stage.

Reopen only if one of these premises changes:

- stable direct telemetry attributes more than 95.027 ms of removable host
  encoding to these 1,653 dispatches;
- the MXFP4 work composes with a larger coherent family whose measured
  supergraph can clear 158.3 ms;
- the asset route/storage census materially changes; or
- relevant device/compiler behavior changes under a new reviewed protocol.

Move the active packed-prefill lane to the 25-layer grouped-IQ2 cohort. Its
accepted mean routed-stage envelope is 597.107 ms, so a candidate must remove
26.51% before implementation. First capture the exact current per-expert count
vectors and separately time the existing grouped gate/up/SwiGLU and
down/scatter phases at production shape. Write no replacement kernel until a
conservative removable ceiling clears 158.3 ms.

## Evidence

- Base revision: `f55a9a3e96d7e63563053c5eeee084cc444b4a14`.
- Source packet:
  `docs/bench/2026-08-05-dsv4-packed-post-route-attribution/README.md`.
- Accepted raw log:
  `docs/bench/2026-08-05-dsv4-packed-post-route-attribution/integration.log`.
- Current asset: 97.05 GiB DeepSeek V4 Flash-0731 UD-IQ3_XXS refresh.
- Device: Apple M4 Max.
- No new executable or performance sample was produced for this closure.
- CX review: `019fd588-96b0-7033-afd1-66d7e354e523`.

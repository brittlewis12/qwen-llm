# Flash Selected Value Reuse

Promote **chronological twelve-head value reuse**, not matrix-multiply PV.
On the saved native gameplay request, two brackets reduce mean prefill time
**4-6%** and request time **3-4%**, with the same generated token sequence.

## Implementation

Full selected-QSA bands contain 32 queries. Twelve query heads share each KV
head, but the incumbent value kernel gathers the same values in three groups
of four heads. The new path separates the unchanged GQA4 softmax reduction from
value accumulation, then shares each gathered F16 value across twelve named F32
scalar accumulators. Each output retains its chronological slot traversal,
multiply/add expression, denominator division and gate multiplication.

- Select only the released Q24/KV2/D256, budget-2048, ratio-four geometry and
  actual N32 selected bands with checked prefix/cache bounds and supported PSOs.
- Preserve selector IDs, causal masks, invalid-ID denominator semantics, audits,
  cache publication, F16 KV storage, and F32 probabilities/accumulators.
- Keep one denominator buffer in session-shared selected scratch: 3,072 logical
  bytes at N32, 16,384 device-priced bytes on this device. Inventory, constructor,
  admission and reconciliation agree. This is not process peak memory.
- Prepare optional pipelines with selected packed scratch, not scalar-decode
  preflight. Unsupported capability uses the incumbent; malformed bindings remain
  errors. Never fall back after stats has
  overwritten logits with masses.
- Smaller bands, tails, dense attention and decode keep their paths. DFlash2's
  short verifier does not select this N32 path. The existing
  `QWEN4EXP_QSA_GQA4_VALUE=0` rollback retains its original single-head route;
  there is no new runtime knob, cache policy or precision mode.

Product implementation: `qwen4exp_qsa::selected_value`,
`kernels/qwen4exp_qsa_value.metal`, and the existing packed-session ledger.
The product workspace, not the research TLS binding, owns scratch through command
completion or abandonment.

## Saved Gameplay Requests

M4 Max, 128 GiB, UD-Q3_K_XL, Metal API validation enabled, production lease.
The same first 14 saved messages are rendered natively with `qwen38` to 11,550
tokens. Temperature 1, top-p 0.95, top-k/min-p disabled, seed 42, 256-token cap.
Both arms retain context 11,808, packed capacity 2048, selected capability,
seven prefill commands and no scalar prefill tail.

Frozen release binaries use fresh processes and an allowlisted environment.
Each arm gets an excluded 16-token warmup, then A/B/B/A with five-second untimed
pauses. No builds or other owned GPU jobs overlap the bracket. These are
warm-weights measurements, not cold-storage or model-loading speedups.
Request time uses the common envelope and excludes model loading; process wall
is retained separately in [results.json](results.json).

| Run | Prefill s | Decode s | Request s |
| --- | ---: | ---: | ---: |
| A1 | 27.52 | 8.64 | 36.16 |
| B1 | 25.36 | 8.62 | 33.99 |
| B2 | 25.35 | 8.59 | 33.95 |
| A2 | 26.23 | 8.63 | 34.86 |
| Baseline mean | 26.87 | 8.63 | 35.51 |
| Candidate mean | 25.36 | 8.61 | 33.97 |

Final-bracket prefill latency falls **5.65%**; its measured GPU sum falls
**5.80%** (26.19 -> 24.67 s). Request latency falls **4.34%**, about 1.54 seconds.
Both paired comparisons improve. Decode is essentially unchanged; no decode
improvement is attributed to this prefill change.

An earlier bracket, before removing redundant selected-pipeline preparation
from scalar preflight, gives prefill 26.66 -> 25.57 s (-4.12%) and request
34.91 -> 33.81 s (-3.17%). Its kernels and routing are the same. Both brackets
are retained in JSON, not pooled into a single fixed speedup estimate. The final
baseline prefill range is 4.8% of its mean, so the larger final margin should not
be interpreted as an additional gain from that host cleanup.

All eight measured runs hit the cap and emit the same 256-token fingerprint.
This is a bounded sampled continuation, not completed-game quality, EOS latency or a
universal bitwise guarantee. No serving cache-hit latency claim is made.

Baseline is clean `4cc4ae59`; each frozen candidate adds this implementation and
reports dirty. Request binary/source hashes and compiler identity are in JSON.
Native development-test screens below have separate provenance; they are not
represented as clean release-request measurements.

## Native And Component Gates

The current UD parent check uses the existing raw first-14-content cost input,
not chat rendering: 8,192 prefix tokens and a 2,048-token selected suffix.
Two ordinary GPU controls take 4,993/4,910 ms; the encoder-stage profile takes
5,063 ms, about 2.3% overhead, with matching endpoint logits. Twelve coarse QSA
blocks account for 46.6%; sampled layer 7's mixer is 151.24 ms of a 195.08 ms
block. These are parent bounds, **not PV-only costs**.

The cache-hot N32 leaf fixture has twelve independent logits views per command,
excluded warmups and balanced ABBA/BAAB orders. It is not twelve distinct model
KV caches. Research GQA12 reduces softmax/value time 0.850 -> 0.633 ms at start
2051 and 0.827 -> 0.658 ms at start 8192: 20-25%. One extra dispatch and the
denominator buffer are included; allocation/reset/readback are excluded.

Native research screening gives 4,720.61 -> 4,458.21 ms (-5.56%) with control
drift, then snapshot-anchored ABBA/BAAB gives 4,897.69 -> 4,663.85 ms (-4.77%).
The **actual product** comparison gives 4,687.89 -> 4,518.92 ms (**-3.60%**),
with both balanced blocks improving. Each anchored screen excludes two warmups,
then measures four samples per arm. Snapshot admission, restoration, live-state
hashing and continuation checks are outside prefill timing. This measures the
suffix command, not prefix restoration or release request latency.

The product comparison explicitly forces old GQA4 only for A and leaves B on
the default product route. Census requires 768 product stats/value launches in
B, 768 old value launches in A, and no research shaders. All ten arms have
identical 32 held-token logit rows, zero measured KL and identical held-token
loss. The common restored prefix's active state is hash-checked before each arm.

## Why Not MMA

F32-operand gathered MMA is faster in isolation, about 42-43%, without adding a
new half-rounding point. Nevertheless, changing the PV sum order introduces
small component differences (worst observed 1.64e-7) that amplify through routing.
The native screen reaches worst KL 0.0244, top-token agreement 31/32 and mean
held-token NLL 1.104589 -> 1.121799. Its noisy ABBA GPU means differ by only
0.14%, not an established native gain. It remains research-only and unqualified;
this is not a general quality verdict on matrix attention.

Only guard-valid fixture runs enter the tables; an initial byte-typing defect
was corrected before valid measurements. The bounded cancellation fixture is
not strong general reassociation coverage. Component tolerance alone does not
grant model-level quality authority.

## Verification And Reproduction

- Actual product-metallib outputs match old GQA4 bits across 18 ordinary, tail,
  invalid-ID, metadata, masked-row, gate and bounded cancellation cases at
  starts 2051/8192 and N1/7/31/32. Masses/tails, guards and unused scratch rows
  are checked too; product dispatch selection remains N32-only.
- Scope, physical inventory, N32/tail/flag-off/unavailable routing and malformed
  denominator refusal-before-update tests pass. Seventeen selected-QSA regression
  tests pass, including reused bands, audits, faults and state/topology.
- Product cancellation after 4099 committed tokens, then retry through 10,240,
  matches cold logits and active persistent state. Snapshot replay passes a
  2047-prefix/40-suffix boundary with a full selected band plus a short tail.
- Session inventory/admission/reconciliation checks and the CLI library,
  `qwen`, and benchmark unit suites pass (119/957/90). This is targeted coverage,
  not every repository integration suite.

Compile test executables before acquiring a GPU lease. All GPU witnesses require
`MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1`, use the production lease, and run
serially. Relevant ignored library tests:

- `qwen4exp_qsa::value_mma_probe::selected_value_product_correctness`.
- `qwen4exp_qsa::tests::selected_value_product::` with `--include-ignored`.
- `qwen4exp_runtime::selected_value_probe::ordinary_selected_value_abba`, with
  `QWEN_QSA_VALUE_ARM=product QWEN_QSA_ANCHORED=1`, `QWEN4EXP_Q3_K_XL_GGUF`,
  the private raw cost input in `QWEN_QSA_TEXT`, and new `QWEN_QSA_REPORT`.
- `qwen4exp_runtime::selected_value_probe::product_selected_cancellation_retry`.

Ordinary request recipe, with fresh stats output:

```sh
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 "$QWEN" run \
  --model "$FLASH_TARGET" --messages "$PRIVATE_MESSAGES" \
  --max-tokens 256 --max-context-tokens 11808 --temp 1 --seed 42 \
  --top-p 0.95 --top-k 0 --min-p 0 --request-stats-jsonl "$NEW_STATS"
```

JSON retains successful native samples, leaf orders, build/input fingerprints
and sanitized request rows. Private inputs and raw diagnostic archives stay out
of Git. No M5, other-artifact or other-family speedup is inferred.

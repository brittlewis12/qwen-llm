# Flash-Next N=4,099 Decision Trace

Decision: **NO QSA BOUNDARY BUG**, **HOLD** the test-only F32-HC-down arm for
quant-native quality evaluation, and **KILL** the F32 down-plus-up arm.

## Question

Packed selected prefill crossed the first QSA selection boundary correctly at
component and transaction level, but a natural 4,099-token full session differed
materially from the default-safe packed-dense plus scalar-selected path. This
packet asks where that difference first appears and whether it identifies a
selected-QSA seam.

The default-safe product path is a chronological diagnostic reference, not
semantic or numerical authority. Packed matrix kernels intentionally use
different reduction arithmetic.

## Source And Workload

- Diagnostic source: `09674cd`.
- Device: Apple M4 Max with unified memory.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, on external PCIe SSD.
- Prompt: the first 4,099 released-tokenizer IDs from
  `docs/PERF-ROADMAP.md` at `26f3c14`; token 4,100 is fed teacher-forced.
- Fixture:
  `crates/qwen-llm/tests/fixtures/qwen4exp_n4099_perf_roadmap_26f3c14.tokens.u32le`.
- Fixture digest domain: `qwen4exp-n4099-decision-trace-u32le-v1\0` followed by
  all 4,100 token IDs as little-endian `u32` values.
- Fixture digest:
  `c396eaee4de9709d7cda51ff7e9f2a63fadacd200f3fc1d50c6254c5d716ac28`.

The three arms reset the same runner and persistent state:

1. default-safe: one packed dense prefix through position 2,050, then scalar
   selected overflow;
2. generic: selected packed execution through all 4,099 prompt rows; and
3. F32 down: the generic arm with exactly the two packed HC down projections in
   every layer of the selected command replaced by the existing F32-accumulating
   Q8 R2C16K64 kernel.

The F32 override matches only execution range `[2051,4099)`. The test requires
96 substitution records at `10240 -> 320`, 96 identically tagged kernel
dispatches, and no topology change outside that treatment.

## Result

### Full-vocabulary logits

| Comparison | Boundary | Argmax equal | Cosine | Relative RMS | Max abs |
|:--|:--|:--:|--:|--:|--:|
| default-safe / generic | endpoint | yes | 0.996500190375 | 8.361495e-2 | 1.324810 |
| default-safe / generic | teacher-forced token 4,100 | yes | 0.994789378303 | 1.021529e-1 | 1.293408 |
| default-safe / F32 down | endpoint | yes | 0.998269664210 | 5.885396e-2 | 1.103926 |
| default-safe / F32 down | teacher-forced token 4,100 | yes | 0.998092938204 | 6.188921e-2 | 0.712247 |

These are local arithmetic distances, not quality scores.

### One-row composition trace

The composition banks sample only target position 2,051.

- Layer-0 `LayerInput` is bit-exact.
- The first difference is layer-0 `AttentionInput`, the attention HC mixed
  output: generic relative RMS `8.330798e-5`, maximum `1.862049e-3`.
- All 336 subsequent snapshots differ; 336/337 total row-local records differ.
- F32 down reduces the first-HC relative RMS to `7.440782e-5` and maximum to
  `1.657844e-3`.

The first observed difference therefore precedes QSA and lies in the composite
packed HC read. Generic down and up both use matrix reduction arithmetic; F32
down narrows the difference, while the killed F32 down-plus-up arm nearly
reproduces the first HC row. The evidence implicates combined HC projection
arithmetic rather than uniquely assigning the discrepancy to down. There is no
M/N/K tail at released geometry, and HC has no position argument.

### QSA decision trace

The capture covers 12 QSA layers by all 2,048 selected positions, or 24,576
position-layer decisions. Duplicate rows reject before any copy dispatch; the
released test requires complete unique coverage.

| Arm | First changed top-512 set | Changed decisions | Selected-ID SHA-256 |
|:--|:--|--:|:--|
| default-safe | n/a | n/a | `16ae191919396d0d7d11d719ecb87347fe18776949945a422ebf7030b8762734` |
| generic | position 2,056, layer 31 | 16,555 / 24,576 | `d86e9aee70b395c6735459b570879f4ee14af78374bee0f9d8937388abcda9a4` |
| F32 down | position 2,060, layer 47 | 16,778 / 24,576 | `f1da2901bd5776f5079b702c35202193f619388209439b1880eea23d95a3211d` |

The first generic symmetric difference is block IDs `[302,306]`. Default-safe
ranks 306/302 at `2.990495/2.9900606`, a margin of `4.343986511e-4`; generic
ranks 302/306 at `2.9894254/2.9888113`. Layer-3 QSA input already differs at
position 2,051, so the selector is responding to upstream arithmetic rather
than creating the first discrepancy.

F32 down delays the first discrete change and improves local logit distance, but
slightly increases the total number of changed sets. That mixed result is a
`HOLD`, not a correctness fix.

## Killed Arm

A diagnostic F32 down-plus-up arm reduced the first-HC relative RMS to
`8.220074e-7` with `1.287460e-5` maximum error. It nevertheless produced
`6.806451e-2` endpoint relative RMS, worse than down-only, and
`6.077086e-2` teacher-forced-continuation relative RMS for materially more work.
The arm and all supporting code were removed.

## Validation

The final ignored released-model test passed with:

- exact 4,100-token fixture length and digest;
- endpoint and teacher-forced-continuation finite-logit plus argmax checks;
- capture-off/capture-on bitwise logits and bytewise equality for every
  enumerated persistent-state tensor in default-safe, generic, and F32-down
  arms;
- complete composition and QSA capture coverage;
- exactly 96 F32 override records;
- exactly 96 tagged `kernel_mat_mat_q8_0_f32_r2c16k64` dispatches at grid
  `(16,20,1)` by threadgroup `(128,1,1)`; and
- selected timing topology of 4,099 packed prompt rows in three commands for
  generic and F32-down arms; default-safe retains its scalar-overflow topology.

The numerical rows are observer-qualified by the capture-off/on replay. The
capture-heavy GPU intervals remain instrumented diagnostics, not performance
measurements. Focused tests also prove inactive no-op behavior, unwind-safe TLS
restoration, duplicate-before-copy rejection, and sampled-profile rejection
before reservation. Source-only adversarial review session
`01a0465b-e7eb-76b1-bc50-81b9dd68460d` returned `PASS`.

## Next Authority

Do not wait for the roughly 360 GB upstream BF16 checkpoint and do not promote
an arm for closeness to incumbent arithmetic. The primary next authority is
held-out teacher-forced NLL, followed by known-answer safety. Observed-token
top-1 and greedy replay are sentinels; llama.cpp is implementation-diverse
triangulation; component oracles and margins remain localization evidence.
Upstream BF16 rows are useful later external calibration, not the immediate
promotion gate.

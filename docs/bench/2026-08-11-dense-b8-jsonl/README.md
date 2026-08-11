# Dense B=8 JSONL product slice

Update (2026-08-11): seekable fixed cohorts now use baseline-safe prefix-aware
packing. See `docs/bench/2026-08-11-fixed-cohort-prefix-packing/README.md`.

Update (2026-08-11): the later mixed-limit qualification removes generation
limit from the compatibility key under a three-quarter utilization gate. See
`docs/bench/2026-08-11-fixed-cohort-mixed-limits/README.md`.

## Question

Can the measured fixed-width dense-Qwen backend serve real file JSONL requests
without weakening serial output semantics, sequence ownership, cancellation, or
memory admission?

## Surface

The opt-in product path is:

```bash
qwen --model MODEL --requests-jsonl REQUESTS.jsonl --batch-size 8
```

It deliberately admits only dense Qwen, file input, fixed prefill chunks, F16
KV, and temperature-zero decoding. A stable planner groups equal tokenized
prompt lengths and generation limits into complete cohorts of eight, routes
underfill through serial execution, and emits rows in original input order.
Stdin, prompt lookup, prefix caching, sampled decode, ragged active cohorts, and
per-request stats sidecars remain explicit errors rather than silent fallbacks.
`QWEN_GREEDY_GPU_ARGMAX=0` also rejects the mode so the established GPU-greedy
rollback cannot be bypassed by batching.

Prompt suffix prefill remains serial. Cohorts sharing at least 256 exact tokens
prefill one chunk-aligned common prefix and restore its transient snapshot into
seven sibling sessions; this neither inserts nor consumes a prefix-cache entry.
`QWEN_DENSE_BATCH8_PREFIX_FANOUT=0` restores eight complete serial prefills.
Decode is lockstep B=8. A lane that reaches EOS or its token limit stops changing
logically, while token zero advances its private physical state until the cohort
drains. Padding is never emitted, hashed, counted as useful work, or published
as a canonical boundary.

## Safety boundary

`DenseBatch8SequenceExecutor` now owns the runtime transition boundary. It
checks that all eight sequences belong to the loaded model, share one frontier,
and have capacity before passing their private sessions to the Metal executor.
Logical positions advance only after a successful committed command.

Committed command, argmax-readback, and NaN-selection failures poison the
executor and every participating session. Poisoned sessions reject subsequent
product decode and snapshot publication; this product path aborts the cohort
and drops them rather than attempting in-place recovery.

Before allocating seven remaining sessions, the CLI prices one real sequence
from Metal's allocation delta and evaluates the seven-session requirement
against working-set and process signals with a 2 GiB transient reserve. Prompt
text is released after tokenization so file-mode host retention is token IDs,
not duplicate source strings.

## Validation

CPU-only contract tests cover CLI/family exclusions, compatibility planning,
serialized underfill, ordered output buffering, one-token generation,
first-token EOS, transition-time EOS, N-1 target transitions, and finished-lane
padding.

One process-cold Qwen3.5 0.8B Q4_K_M product comparison used the same
eight-request varied-prompt file in batch and serial modes. All eight 12-token
output rows were byte-identical, and all eight generated-token hashes were
distinct.

For Q4_K_M, the product path completed 96 generated tokens and 88 useful target
transitions in 159.210 ms of decode wall, or 602.98 aggregate generated tok/s.
Eleven B=8 commands consumed 148.542 ms of host wall and 133.833 ms of reported
GPU time. This reproduces the extracted backend's 590.16 tok/s smoke rather than
the deliberately unpromoted F32 performance regime.

The tiny seven-token prompt priced one sequence at 30,605,312 bytes and admitted
the remaining 214,237,184 bytes with the process-budget-omitted working-set
policy. Prefix-cache capacity was zero and no cache entry was created.

`validation.json` records the exact model, request, candidate-source, and output
hashes. Acquisition was rooted at parent `322b63d`; an unrelated dirty
DeepSeek-prefill file was present but is not reachable from this dense-Qwen
execution path. No `QWEN_*` environment override was present.
An explicit `QWEN_GREEDY_GPU_ARGMAX=0` follow-up failed before model load, wrote
no stdout, and named the incompatible rollback.

## Decision

- Promote `--requests-jsonl FILE --batch-size 8` as an explicit dense-Qwen
  capability.
- Preserve serial JSONL unchanged when `--batch-size` is absent.
- Keep finished-lane physical state disposable and observable through padding
  telemetry.
- The representative 16K implementation gate is now closed by
  `../2026-08-11-dense-b8-long-context/`; its dirty-build evidence is not a
  canonical family-board cell. Retain explicit opt-in until cohort formation
  and underfill policy are separately qualified.
- Shared-prefix fanout is promoted by
  `../2026-08-11-dense-b8-prefix-fanout/`: 27B cohort prefill improves `4.812x`
  with byte-identical output against rollback and serial controls.
- Arbitrary file counts and mixed compatible shapes are promoted by
  `../2026-08-11-dense-b8-cohort-planner/`; full cohorts batch, underfill runs
  serially, and complete output remains byte-identical to serial input order.
- Let MoE and DeepSeek qualify family-specific executors behind the same cohort
  concept; do not route them through the dense implementation.

## Commands

```bash
target/release/qwen \
  --model /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-varied.jsonl \
  --batch-size 8 --tokens 12

target/release/qwen \
  --model /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-varied.jsonl \
  --tokens 12
```

# Qwen Exact-LCP Tiny-Suffix Fanout

Date: 2026-08-11

Status: product `GO` for exact token-prefix fanout when Qwen fanout already
qualifies at a fixed-chunk boundary and every exact private suffix contains at
most six tokens.

## Question

Should Qwen B=2/B=8/B=16 snapshot the exact shared token prefix rather than
rounding that prefix down to the current packed-prefill chunk boundary?

The snapshot and restore APIs accept arbitrary token positions, but changing the
boundary also changes packed-prefill segmentation. Exact matching therefore
needs an economic and numerical gate rather than unconditional longest-prefix
selection.

## Policy

The default `tiny_suffix_exact_lcp` policy selects the exact token LCP only when:

- ordinary fanout is enabled;
- the exact LCP is at least 256 tokens;
- rounding down by the fixed prefill chunk still leaves at least 256 tokens, so
  exact matching does not broaden fanout admission;
- every lane has at most six private tokens after the exact LCP, matching the
  measured singleton teacher-forcing cutoff.

Otherwise the existing chunk-aligned boundary remains selected.
`QWEN_PREFIX_FANOUT_EXACT_LCP=0` is strict aligned rollback. Setting it to `1`
force-enables broad exact-LCP selection for diagnostics; `auto` restores the
bounded default. Invalid and non-UTF-8 values fail closed.

Planner and executor capture the same policy value. Qwen pair, dense-cohort, and
MoE-cohort telemetry report the policy and selected reason. DeepSeek keeps its
separate real-chunk-boundary contract and unchanged telemetry.

## Mechanism

For chunk size 512, the primary fixture has:

```text
prompt length:          1,536 tokens
exact common prefix:   1,531 tokens
aligned prefix:         1,024 tokens
exact private suffix:       5 tokens
aligned private suffix:   512 tokens
```

Aligned execution evaluates 1,024 shared tokens and then independently prefills
512 tokens in every lane. Exact execution evaluates the additional 507 shared
tokens once, captures that frontier, and teacher-forces five singleton tokens per
lane. Scratch sizing and memory admission remain conservative and unchanged.

## Results

One release candidate served both arms on Apple M4 Max. Model processes were
serialized, all requests generated 16 greedy tokens, and stdout was hashed
byte-for-byte.

| Model/path | Aligned prefill | Exact prefill | Speedup | Process wall |
|---|---:|---:|---:|---:|
| Qwen3.5 0.8B Q4, B=8 | `1.144 s` | `0.360 s` | `3.176x` | `1.74 -> 0.86 s` |
| Qwen3.6 27B Q4, B=8 | `23.623 s` | `8.307 s` | `2.844x` | `27.60 -> 12.43 s` |
| Qwen3.6 A3B Q4, B=16 | `7.237 s` | `1.827 s` | `3.961x` | `11.54 -> 6.25 s` |
| Qwen3.6 A3B Q4, B=2 | `1.480 s` | `1.078 s` | `1.373x` | `4.42 -> 4.02 s` |

Decode is flat in every arm. The detailed decomposition is:

- dense 27B B=8: shared-prefix wall increases `4.194 -> 6.574 s`, while
  private-suffix wall falls `19.307 -> 1.603 s`;
- A3B B=16: shared-prefix wall increases `0.670 -> 0.990 s`, while suffix wall
  falls `6.479 -> 0.747 s`;
- A3B B=2: shared-prefix wall increases `0.671 -> 0.984 s`, while private wall
  falls `0.809 -> 0.094 s`;
- dense 0.8B B=8: shared-prefix wall increases `0.149 -> 0.211 s`, while suffix
  wall falls `0.952 -> 0.116 s`.

The dense B=8 arms emit stdout SHA-256
`0570a183397b3251c53bd0ba645c9f3c9ad97d4b60f01a07cc57d65a2fa827e2`.
A3B B=16 emits
`341945d99e43a00abb934f0f6c704701fab66b251df57c0f9e7524c9887e647a`,
and A3B B=2 emits
`7ed109e5894549c73e6c3a9b27115af50c1ba963d1ad7b9d7d3cec86a2974e43`.
Each hash is identical between aligned and exact arms.

## Falsifier

Exact LCP is not an unconditional win. A dense 0.8B B=8 fixture with a common
prefix only one token beyond alignment and a long private suffix moves prefill
`1.757 -> 1.800 s` and process wall `2.27 -> 2.31 s`. Both arms emit SHA-256
`e4872e79a5b701214c1bc0dbc9047d8e0a73b4d09409261fe0461955f7a5adb2`.

The production policy leaves this fixture chunk-aligned. At the same one-token
alignment delta with a six-token aligned suffix, exact selection instead moves
prefill `0.325 -> 0.303 s`, with byte-identical output. This is the reason for a
suffix-bound policy rather than a minimum alignment-delta guess.

## Exactness Scope

Arbitrary-position snapshot and restore preserve the state that was captured.
Exact-LCP and aligned executions can nevertheless use different packed matrix
shapes and reduction lineages. The authority here is the existing greedy
accelerated CLI contract: generated JSON, token hashes, stop reasons, and
terminal behavior are byte-identical through 16 tokens across dense and MoE
B=2/B=8/B=16 fixtures.

This does not claim intermediate-state bit equality, authorize sampled
concurrency, broaden fanout admission, or transfer the policy to DeepSeek.

## Decision

Promote bounded exact-LCP fanout for the six-token private-suffix pocket. Preserve
chunk alignment everywhere else and retain broad exact selection only as an
explicit diagnostic. Hierarchical multi-checkpoint fanout, ragged prefill, and
paged KV remain separate serving-lane questions.

Representative commands:

```text
QWEN_PREFIX_FANOUT_EXACT_LCP=0 qwen --model MODEL \
  --requests-jsonl FIXTURE --batch-size WIDTH --temp 0 --prefill-chunk 512

qwen --model MODEL --requests-jsonl FIXTURE \
  --batch-size WIDTH --temp 0 --prefill-chunk 512

QWEN_PREFIX_FANOUT_EXACT_LCP=1 qwen --model MODEL \
  --requests-jsonl FIXTURE --batch-size WIDTH --temp 0 --prefill-chunk 512
```

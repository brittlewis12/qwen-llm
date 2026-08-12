# Qwen Private-Suffix Singleton Replay

Date: 2026-08-11

Status: product `GO` for restored Qwen private suffixes of at most six tokens in
the existing greedy B=2, dense B=8, and qualified MoE B=16 paths.

## Question

Should a tiny private suffix after prefix restore use the large packed-prefill
organization or teacher-forced singleton transitions?

Production fanout previously sent every nonempty suffix through packed prefill.
The prefix-cache benchmark already selected singleton replay at short lengths,
and the dense B=8 fanout result identified eight five-token packed suffix calls
as its dominant residual.

## Mechanism

After a successful Qwen checkpoint restore:

- suffixes of one through six tokens call `MetalForward::single_token` once per
  forced prompt token and return the final full-logit row;
- longer suffixes and all ordinary full prompts retain packed prefill;
- every successful singleton call advances the tracked `Sequence` immediately;
- logical position, full append capacity, final-position overflow, and `u32`
  range are validated before the first Metal mutation;
- packed scratch remains allocated and conservatively admitted, so this changes
  execution policy rather than memory policy.

`QWEN_PRIVATE_SUFFIX_SINGLETON=0` restores packed suffix execution. Invalid
values fail closed. Cohort and pair telemetry report the effective cutoff plus
singleton/packed lane and token counts.

The cutoff is deliberately six, not the benchmark's older broad `64`. A dense
27B crossover screen found a win at six, a small loss at eight, and a decisive
loss at sixteen.

## Validation

Base: `64b1df4` plus this candidate. One release binary served every arm on Apple
M4 Max. Model processes were serialized and stdout was hashed byte-for-byte.
All requests generated 16 greedy tokens after prompt preparation.

### Dense Qwen3.6 27B B=8

Eight prompts share a 1,024-token prefix and carry equal-length private suffixes.

| Suffix | Packed suffix | Singleton suffix | Verdict |
|---:|---:|---:|---|
| `6` tokens | `3.325 s` | `1.899 s` | singleton `1.751x` |
| `8` tokens | `2.488 s` | `2.537 s` | packed `1.020x` |
| `16` tokens | `2.887 s` | `5.070 s` | packed `1.756x` |

At the promoted six-token cell, total cohort prefill moves
`7.642 -> 6.211 s` (`1.230x`) and a warm closing control/candidate process pair
moves `11.70 -> 10.23 s` (`1.144x`). Decode remains flat. All six-token arms
emit stdout SHA-256
`0314408a5f5510ad5d237e58df444d828cc706417acb05ed134eadbec970f24b`.

The exact eight- and sixteen-token screens emit hashes
`db7c9e5d0d664b34ef84a7a26502eae16044c573563eeb6c07f5cc16f844185c`
and `48398a0f862290c129ed2f079ea46eabdf55ba8a3f4c7ff2527af0a3a1aafb9b`.
They are retained as the reason not to generalize the cutoff by intuition.

### Qwen3.6 A3B Q4 B=16

Sixteen prompts share 1,024 tokens and each carries a five-token suffix.

- private suffix wall moves `1.968 -> 0.741 s` (`2.656x`);
- total prefill moves `2.730 -> 1.503 s` (`1.816x`);
- process wall moves `7.20 -> 5.98 s` (`1.204x`);
- decode remains flat;
- both arms emit stdout SHA-256
  `56cd97b92a3dfa4a687f7be4b88c568f5dba313cc071bbe0c9a7b44c30bfe957`.

### Qwen3.6 A3B Q4 B=2

Two prompts share 1,024 tokens and each carries a five-token suffix.

- private suffix wall moves `0.246 -> 0.093 s` (`2.645x`);
- total prefill moves `0.918 -> 0.780 s` (`1.177x`);
- process wall moves `3.95 -> 3.80 s` (`1.039x`);
- both arms emit stdout SHA-256
  `19a82058ea011cde3db7e851b45664c38735d99be4bdd6d364106f4075502026`.

## Exactness Scope

Packed mat-mat and singleton mat-vec prefill can differ numerically. Promotion
authority is therefore the existing greedy accelerated CLI contract: all tested
dense/MoE B=2/B=8/B=16 product outputs, token hashes, stop reasons, and terminal
semantics are byte-identical through 16 generated tokens. This does not claim
bit-identical packed and singleton intermediate tensors or authorize sampled
Qwen concurrency, which these paths already reject.

## Decision

Promote singleton replay only through six private tokens. It removes a measured
fixed-cost pocket using an already-established execution path, while the dense
crossover closes broader cutoffs. Keep packed execution at seven or more tokens
until a model-specific selector earns enough margin to justify added policy.

Representative commands:

```text
QWEN_PRIVATE_SUFFIX_SINGLETON=0 qwen --model MODEL \
  --requests-jsonl FIXTURE --batch-size WIDTH --temp 0 --prefill-chunk 512

qwen --model MODEL --requests-jsonl FIXTURE \
  --batch-size WIDTH --temp 0 --prefill-chunk 512
```

# Qwen Ragged Fixed Cohorts

Date: 2026-08-12
Status: experimental capability GO; default remains off

## Question

Can the existing dense B=8 and Qwen MoE B=16 decode executors consume
requests at different prompt frontiers without changing each lane's greedy
result, and does that unlock useful aggregate throughput on heterogeneous
regular-file JSONL workloads?

## Implementation

`QWEN_FIXED_COHORT_RAGGED_PROMPTS=1` removes prompt-length equality from
fixed-cohort compatibility. Each lane retains its own logical position, KV/GDN
frontier, prompt length, and terminal accounting. Immutable-weight projection
rows remain batched at fixed width; attention and position-sensitive mixer work
receive the corresponding lane position.

The first product slice remains deliberately bounded:

- greedy decode only;
- fixed B=8 dense or B=16 Qwen MoE membership;
- no refill or continuous scheduler;
- one shared maximum capacity per cohort;
- requested-transition utilization must remain at least 3/4;
- a capacity-first ordering is selected only when it preserves cohort count,
  does not increase transition slots, and strictly reduces capacity slots;
- Metal-priced sessions, prefill scratch, executor scratch, reserve, and an
  optional CPU checkpoint are admitted before executor or sequence allocation;
- denied cohorts become input-ordered serial requests rather than aborting the
  file.

Unset or `QWEN_FIXED_COHORT_RAGGED_PROMPTS=0` retains equal-length planning.
Prefix packing is reported as configured but ineffective for this experimental
ragged planner; within-cohort prefix fanout remains active.

## Exact Product Cells

All commands used the release binary at the reviewed worktree, warm model files,
`--temp 0`, fixed `--prefill-chunk 512`, and serialized GPU execution. Complete
candidate and serial JSONL files were byte-identical in every cell.

| Family / workload | Prompt tokens | Output | Serial wall | Ragged wall | Speedup |
|---|---:|---:|---:|---:|---:|
| Dense 0.8B Q8, heterogeneous short prompts | 10-137 | 8 x 64 | 2.82 s | 1.65 s | 1.709x |
| Dense 0.8B Q8, shared root + private suffixes | 1,984-2,222 | 8 x 32 | 2.64 s | 2.29 s | 1.153x |
| Qwen A3B Q4, heterogeneous short prompts | 10-139 | 16 x 64 | 14.21 s | 10.95 s | 1.298x |

The earlier implementation checkpoint, before final admission hardening, also
measured `2.23 -> 1.28 s` (`1.742x`) on dense short prompts and
`17.39 -> 11.08 s` (`1.569x`) on A3B. Those rows remain mechanism evidence;
the table above is authority for the final reviewed code.

The corrected dense shared-root cell selects a chunk-aligned 1,536-token
checkpoint from a 1,982-token exact LCP. It evaluates 4,240 private suffix
tokens, generates 256 tokens at 420.8 aggregate tok/s, and remains exact.

A3B long private suffixes remain the important negative constraint. An earlier
1,684-1,778-token, B=16, 24-token-output fixture was effectively flat
(`10.84 -> 10.78 s`, `1.006x`): 3,064 private suffix tokens erased the decode
benefit. This is why ragged admission remains explicit rather than default-on.

## Validation

- Dense and MoE complete JSONL outputs and generated-token SHA-256 values match
  their serial controls exactly.
- Dense and MoE backend unit suites pass.
- 21 fixed-cohort planner/policy tests pass, including capacity Pareto selection,
  memory-denial rewriting, and longest-lane root alignment.
- `cargo clippy -p qwen-cli --bin qwen -- -D warnings` passes.
- `cargo clippy -p qwen-llm --lib -- -D warnings` passes.
- Final adversarial review verdict: GO.

## Decision

Keep the capability as an explicit experimental opt-in. It proves that fixed
cohorts do not need equal prompt frontiers and produces useful cross-family wins
without changing model math. Do not default it on until admission prices private
suffix prefill against expected decode savings, especially for MoE workloads.
The next architectural step is refill/continuous batching over the same
per-request frontier contract, not additional model-specific fixed-width forks.

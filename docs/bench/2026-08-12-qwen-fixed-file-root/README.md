# Qwen Fixed-Cohort File Root

Date: 2026-08-12

Status: product GO for dense B8 and Qwen MoE B16.

## Question

Can fixed Qwen cohorts share one immutable file-scoped checkpoint instead of
prefilling and capturing the same long root once per cohort?

The B2 path had already established the mechanism. Fixed B8/B16 still created a
transient root independently inside every cohort, so a seekable file with four
cohorts evaluated the same 6,144-token boundary four times.

## Mechanism

A neutral `qwen_file_root` helper now owns exact token-LCP planning, fixed-chunk
alignment, policy parsing, one root prefill, final-logit retention, and immutable
checkpoint capture. B2 and fixed cohorts share that substrate without publishing
to the RAM prefix index.

Fixed execution selects participants only after memory-denied cohorts have become
serial work. Refill arenas and serial requests neither constrain nor consume the
root. The file layer prices one retained root plus the largest simultaneously
live deeper cohort checkpoint. Each cohort restores the root into its source
sequence, replays only any deeper shared bridge, captures that bridge if needed,
and restores sibling sequences at the selected boundary.

`QWEN_FIXED_COHORT_FILE_ROOT_FANOUT=0` restores cohort-local behavior. Dense B8
and Qwen MoE B16 default on behind the same rollback.

## Result

Thirty-two realistic Qwen prompts form four dense B8 cohorts. They share 6,482
tokens and select a 6,144-token file root. Prompt lengths are 6,482/6,561 tokens;
generation limits span 8-68 tokens. All runs are greedy with chunk 1,024.

| Organization | Wall | Relative |
|---|---:|---:|
| Cohort-local roots | `16.01 s` | control |
| One file-scoped root | `13.81 s` | `1.159x` |

The candidate prefills the root in `794.229 ms`, captures a `96,716,848`-byte
snapshot in `26.026 ms`, and avoids 18,432 prompt-token evaluations. Complete
JSONL is byte-identical; both outputs have SHA-256 `6b74a0226a2efaee8b264c0af0f2416eab67499bec8b3cbcc447e24dba7610d7`.

A two-cohort probe moved `8.35 -> 7.64 s` (`1.093x`), just below the standing
`1.10x` gate. The four-cohort result demonstrates the intended file-scope
amortization and clears it without changing model math or fixed decode.

### Qwen MoE B16

Sixty-four identical A3B Q4 requests form four B16 cohorts around a 1,536-token
file root. Cohort-local roots take `14.58 s`; one retained root takes `12.20 s`
(`1.195x`). Both outputs have SHA-256
`fba5ae7dc58a286da1f89b76b5dba7f878f7427513562c2b50969162ac1c516d`.
The candidate captures a `98,320,464`-byte root and avoids 4,608 prompt-token
evaluations. This clears
the same gate and qualifies the shared capability on routed execution.

## Decision

Promote file-root reuse by default inside the already explicit dense B8 and Qwen
MoE B16 modes. Keep the rollback, 1,024-token minimum, fixed-chunk boundary,
exact restore checks, process-memory admission, and separate telemetry.

Do not yet compose the root with dense refill. That is a distinct prefix-aware
refill lane, and the current bounded refill planner intentionally excludes
prefix-fanout work.

Durable fixture and machine-readable validation are in this directory.

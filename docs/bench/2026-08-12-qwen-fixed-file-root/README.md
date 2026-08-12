# Qwen Fixed-Cohort File Root

Date: 2026-08-12

Status: dense B8 product GO; Qwen MoE B16 remains explicit/default-off pending
its own exact performance cell.

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
defaults on. MoE B16 parses the same explicit gate but defaults off until a
separate asset-specific exact cell clears promotion.

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

## Decision

Promote dense B8 file-root reuse by default inside the already explicit
`--batch-size 8` mode. Keep the rollback, 1,024-token minimum, fixed-chunk
boundary, exact restore checks, process-memory admission, and separate telemetry.

Do not yet compose the root with dense refill. That is a distinct prefix-aware
refill lane, and the current bounded refill planner intentionally excludes
prefix-fanout work. Do not infer MoE promotion from the dense cell.

Durable fixture and machine-readable validation are in this directory.

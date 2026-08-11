# Prefix-Aware Fixed Cohort Packing

Date: 2026-08-11

Status: product `GO` for dense Qwen B=8 and qualified Qwen MoE B=16.

## Question

Can a seekable JSONL planner place repeated-prefix requests into the same fixed
cohort before invoking the existing snapshot fanout path?

The prior planner bucketed equal prompt lengths, sorted only by requested output
depth, and then cut exact B=8/B=16 groups. Interleaved request families could
therefore produce cohorts with no common prefix even when enough members existed
to form complete prefix-homogeneous cohorts.

## Mechanism

For each equal-prompt-length bucket:

1. Compute the unchanged generation-depth plan as the baseline.
2. Lexicographically sort exact prompt token vectors and inspect complete fixed
   width chunks.
3. Retain a prefix chunk only when the existing family fanout planner identifies
   a stable boundary of at least 256 tokens and the existing three-quarter
   transition-utilization gate passes.
4. Return every unselected row to the original generation-depth planner.
5. Replace the baseline only when the candidate forms more complete cohorts, or
   forms the same number without increasing estimated physical transitions.

The baseline comparison prevents prefix affinity from fragmenting otherwise
healthy depth cohorts. Adversarial B=8 and B=16 tests construct homogeneous
low/medium/high baseline groups plus one reusable mixed-depth prefix family; both
must preserve all three baseline cohorts and report a prefix-plan fallback.

`QWEN_FIXED_COHORT_PREFIX_PACKING=0` restores depth-only packing. Invalid values
fail closed. Prefix fanout can still be disabled by its existing family-specific
rollback, in which case no prefix cohort is selected.

This is deterministic offline packing, not lane refill, global optimal matching,
or a new snapshot representation. Exact-width lexical chunks may miss some
available groups; missed opportunities fall back to the prior plan.

## Validation

Base: `1c6a4bc` plus this candidate. All arms use one release binary on Apple M4
Max, execute model work serially, and hash stdout without normalization.

### Dense Qwen B=8

Sixteen Qwen3.5 0.8B Q4_K_M requests arrive as two interleaved families of eight.
Each family has an identical 2,048-token prompt and every request asks for 16
tokens.

| Organization | Prefix cohorts | Summed prefill | Process wall |
|---|---:|---:|---:|
| Depth/input-stable | `0` | `4.007 s` | `4.79 s` |
| Prefix-aware | `2` | `0.578 s` | `1.34 s` |

The planner changes physical groups from mixed A/B cohorts to one A and one B
cohort. Each candidate cohort prefills once and restores seven sessions. Process
wall improves `3.575x`; prefill improves `6.936x`. Both arms emit stdout SHA-256
`b0173f91fee0da43dc42a87623bcc7580022696089cba4ccf58bb307e2f5120b`.

The final default-on/rollback pair reports the same two complete cohorts and the
same `240` estimated physical transition slots in each arm.

### Qwen MoE B=16

Thirty-two Qwen3.6 35B-A3B Q4 requests arrive as two interleaved families of
sixteen. Each family has an identical 1,024-token prompt and every request asks
for 16 tokens.

| Organization | Prefix cohorts | Summed prefill | Process wall |
|---|---:|---:|---:|
| Depth/input-stable | `0` | `19.442 s` | `25.45 s` |
| Prefix-aware | `2` | `1.432 s` | `7.62 s` |

Process wall improves `3.340x`; prefill improves `13.573x`. Decode remains
effectively flat. Both arms emit stdout SHA-256
`aeb9fb17faea658abc451c3a93d6d8ee6efc5c9c2e28854da8532debfcf44ecd`.

## Decision

Promote prefix-aware packing by default for fixed B=8/B=16 files. The planner
only composes already-qualified cohort execution and snapshot fanout, while its
per-bucket baseline guard prevents loss of batch coverage or increased padding
at equal coverage.

Keep the rollback and expose planner schema 3 fields for prefix packing,
accepted prefix cohorts, baseline fallbacks, and estimated physical transitions.
Consider overlapping windows or prefix-run grouping only if real traces show
material missed affinity; they are opportunity improvements, not correctness
requirements.

Representative commands:

```text
QWEN_FIXED_COHORT_PREFIX_PACKING=0 qwen --model MODEL \
  --requests-jsonl FIXTURE --batch-size WIDTH --temp 0 --prefill-chunk 512

qwen --model MODEL --requests-jsonl FIXTURE \
  --batch-size WIDTH --temp 0 --prefill-chunk 512
```

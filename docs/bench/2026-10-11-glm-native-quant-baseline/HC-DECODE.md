# Normalized Native HC Decode

Follow-up to the generic HC baseline. Eligible single-row HC uses four dispatches
instead of six: fill unit weights, RMS-normalize residual, native split-K mix
projection, then fused controls/Sinkhorn/collapse/block RMS. Projection consumes
the normalized activation, preserving the baseline's overflow behavior rather
than multiplying raw large values and scaling afterward.

The projection supports the 17 shared native decoders. It reuses `ones` for
partial dots only after RMS has consumed it (1536 of 16384 floats at H=4096).
No extra GPU scratch or expanded weights. Unsupported geometry/decoder/device
capacity uses the existing composition. The separate fused Q8 path is unchanged.
Multirow packed HC stays batched as before; single-row packed spans are eligible
for the same new path as decode.

## Paired Maya Decode

Same release binary and one resident model. Each run uses a fresh session with
the same 512-token default Half prefill outside the timing interval, then 16
common teacher-forced decode tokens. Both arms warm first; measured order ABBA.
A forces the previous generic HC composition via a test-only scoped override;
B enables the new path. Census on the warmup confirms 90 partial and 90 finish
dispatches per B token, zero in A. Timed steps do not collect that census.

| Run | Mean milliseconds/token | Tokens/second |
| --- | ---: | ---: |
| A1 | 35.41 | 28.242 |
| B1 | 32.74 | 30.547 |
| B2 | 32.93 | 30.364 |
| A2 | 36.02 | 27.761 |

Combined A throughput was 27.999 tokens/s; B was 30.455 tokens/s: **8.8% higher
throughput**, about 8.1% less wall time. Both B means beat both A means. Worst
full-vocabulary KL against A1 across all 16 steps was 1.190e-11. This is a short
paired workload measurement, not a claim across all contexts or artifacts.
Raw results: `maya-hc-decode-ab.json` and its log.

## Validation and Limitations

- HC suite: 15 passed, one timing test ignored, with required Metal validation.
  Includes every decoder, tiny and H=4096 geometry, CPU/f64 and baseline oracles,
  normalization-first extreme F32/BF16 values, offsets, guards and allocation.
- GLM suite: 82 passed, 46 opt-in tests ignored; the paired model test was
  explicitly enabled separately.
- Attribution, Metal context, native expert, IQ1/IQ2 and generic grouped suites
  passed. All workspace targets checked successfully before the build-provenance
  fix; subsequent main/worktree library checks passed with that fix.
- Existing Q8 tests remain green. F32 means operand preservation, not bitwise
  agreement across changed reduction orders.

Validation exposed and fixed two infrastructure defects separately: dispatch
attribution followed pipeline lookup rather than binding, and a shared target
could embed another worktree's metallib. See `DIAGNOSTICS.md`. These were real,
reproduced issues, not additional model admission gates.

## Retrospective

The general baseline made the model runnable immediately. Profiling then located
a concrete small-output utilization/launch problem, and a shared-decoder fast
path removed it without changing the admission contract or adding memory. Keep
this ordering: usable baseline, measured bottleneck, optional optimization.

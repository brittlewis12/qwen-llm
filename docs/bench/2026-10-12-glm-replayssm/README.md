# KDA Replay: Traffic And Numerical Fidelity

Bounded exploration of [ReplaySSM](https://arxiv.org/abs/2610.13778) for GLM's
channelwise-decay KDA, separate from document-prefix checkpoint reuse. Production
runtime, state precision and snapshots are unchanged.

## FP32 Metal Leaf

The research kernel keeps an F32 checkpoint and L8 records of normalized keys,
per-key decay and delta corrections. It reconstructs state chronologically with
the existing multiply/FMA order, computes current outputs, and writes the full
checkpoint only on flush. Historical corrections do not require repeated dot
products. L1 is the ordinary-recurrence control. Materialization is read-only
with respect to checkpoint and records.

Seven KDA tests pass under leased Metal validation, including the existing
independent CPU contract and replay/incumbent bitwise equality across multiple
windows, every residue, nonzero state, conv tails, output guards and invalid
bindings. This is an oracle chain, not a claim that the new GPU test itself has
an independent F64 implementation.

Timing uses 34 distinct 64-head matrices with chained outputs, equal submission
structure and full L8 windows, including flushes. It is a synthetic batch-one
leaf, not a whole-model or batch-256 result. Replay adds 25.5 MiB of records to
the existing 136 MiB of recurrent matrices. Logical traffic is not measured DRAM
traffic. No production dispatch change follows from a leaf result alone.

## Precision Study

`scripts/reference/glm53/replay_precision.py` retains an independent F64 state
and runs the same prepared F32 inputs through ordinary F32, per-step F16 feedback,
shared-F16 replay, separate F16-read/F32-anchor replay, and F32 replay. L1/L8
controls separate quantization cadence from precision separation. Both F16 replay
arms recompute corrections from raw inputs at flush; reusing low-precision
corrections would contaminate the authoritative recurrence.

Three fixed synthetic streams cover ordinary forgetting, slow decay/small
updates, and extreme channelwise decay. Checkpoints include flush and nonflush
positions. Eight unit tests pass, including independent scalar-reference checks,
identical authoritative anchors despite different read precision, and a negative
control that incorrectly reuses approximate corrections.

This is CPU numerical emulation before output normalization/gating, not model KL,
task accuracy, or a Metal speed claim. NumPy reduction order differs from Metal.
The existing model already uses F32 recurrent state, even in Fast Half mode;
separate precision is not presumed more accurate than that baseline. Gains in
fidelity versus low-precision feedback remain useful even without a batch-one
speed gain, but must be reported alongside storage and flush costs.

## Initial Results

`fp32-leaf-abba.json`: all six arms have identical output, final state and
convolution hashes, with zero nonfinite outputs and valid positive GPU durations.
Measured GPU totals (16 steps across 34 states): A1 23.028, B1 23.376, B2 25.412,
A2 22.560 ms. Mean A/B is 0.9344 on GPU and 0.9121 wall: no speed promotion.
Residue timings vary substantially; do not infer a precise whole-model regression.

`precision-64k.jsonl` completes all three streams and eight configurations with
no nonfinite steps. It uses F16 round-to-nearest, not stochastic rounding or the
paper's FP8/NVFP4 formats. Full-run output relative L2 error against persistent
F64 (before output normalization/gating):

| Stream | Ordinary F32 | Ordinary F16 | Shared F16 L8 | F16 read / F32 anchor L8 |
| --- | ---: | ---: | ---: | ---: |
| Normal correlated | 1.80e-7 | 7.07e-4 | 2.50e-4 | 1.65e-4 |
| Slow, small updates | 8.93e-6 | 0.423 | 0.396 | 1.94e-4 |
| Extreme channels | 1.27e-7 | 1.64e-4 | 5.91559e-5 | 5.91568e-5 |

The slow stream deliberately exposes rounding feedback; its large improvement
is not a claim about natural model trajectories. Normal decay shows a modest
gain; strong rotating decays make shared/separate replay effectively tied.
Ordinary F32 remains more accurate in all three streams. At flushes, separate
and F32 replay have identical recorded state/output errors; short metamorphic
tests additionally verify anchor-bit independence from approximate read precision.

The 256-step smoke packet is retained separately. No result is rescored as a
model-quality pass. These results motivated the physical F16-read Metal leaf
below, measuring conversion and raw-input flush costs rather than extrapolating
CPU emulation into a speed claim. Production state stays F32.

## Physical F16 Results - 2026-10-13

The second prototype uses real F16 read buffers, not just emulated rounding.
Both low-precision policies cache F32 k/d/u plus raw post-convolution v and beta.
At flush, cached approximate u is ignored and corrections are recomputed from
the authoritative anchor. Shared F16 uses the same flush algorithm/cadence but
starts from its F16 anchor. Separate precision independently writes the F32
anchor and F16 read copy from the same F32 working state.

Eleven KDA tests pass under leased Metal validation. Tests cover physical formats,
alias restrictions, all residues, guards, multiple flushes and poisoned cached
corrections. Injecting NaNs/large values into cached u before a flush leaves the
separate authoritative anchor and flush output unchanged.

`f16-gpu-f64-1024.json` records a two-head, 1,024-step GPU study against an
independent persistent F64 oracle including convolution and output normalization.
This deliberately uses synthetic lower_bound=-0.001, not the release -5 or a
natural activation trace. Every output and materialized read state is sampled;
authoritative anchors are sampled after each 32-step batch. All pairs are finite.

| Configuration | Output RMS error | Authoritative anchor RMS error |
| --- | ---: | ---: |
| Ordinary F32 | 1.81584e-6 | 3.63860e-7 |
| F32 replay | 1.81584e-6 | 3.63860e-7 |
| F16 read / F32 anchor | 5.41567e-5 | 3.63860e-7 |
| Shared F16 | 3.11855e-4 | 7.82023e-5 |

Separate precision improves output RMS 5.758x versus shared F16 in this fixture,
but remains 29.825x above ordinary F32. Its authoritative anchor matches ordinary
F32 bits at every checked completed window. This demonstrates exclusion of local
low-precision correction feedback, not elimination of all model-level feedback:
approximate outputs can change later layers' driving inputs in a real model.

`f16-leaf-abba.json` uses the release lower bound -5 for every arm. It repeats
the same synthetic 34-state geometry, with flush-inclusive ordinary/candidate
ABBA for each policy. The first FP32 packet used fixture lower_bound=-1.25;
do not treat these different packets as a paired implementation comparison.

| Candidate | GPU B/A, lower is better | Wall B/A | Recurrent matrices + records |
| --- | ---: | ---: | ---: |
| Ordinary F32 reference | 1 | 1 | 136 MiB |
| F32 replay | 1.1878 | 1.2528 | 161.5 MiB |
| F16 read / F32 anchor | 1.0366 | 1.0443 | 238.0664 MiB |
| Shared F16 | 1.1776 | 1.1834 | 102.0664 MiB |

Separate precision's GPU pairs disagree (1.1426 and 0.9521 B/A); there is no
convincing speed win, not a precise 3.66% regression claim. Expected state/output
hash relationships hold, all outputs/anchors/read buffers are finite, and all
timestamps are valid. These are leaf timings and logical retained bytes, not
whole-model throughput or measured DRAM traffic. Shared F16 saves about 25% of
these recurrent bytes, but pays measured precision and speed costs.

## Decision

Keep ordinary F32 in production. The user's long-sequence precision hypothesis
is supported against low-precision feedback, in CPU emulation and physical Metal
execution; it is not an accuracy upgrade over the existing F32 recurrent state.
Separate precision has no demonstrated latency advantage here and retains about
102 MiB more state/records across 34 blocks. No runtime/snapshot integration or
model-quality claim is justified by these results.

Archive the complete GPU prototype in Git and remove it from active builds;
retain the small portable CPU study, its eight tests and all evidence. A future
speculative-verification experiment would need a concrete per-draft state-copy
cost and rollback requirement. That is a different hypothesis, not a reason to
keep sweeping this batch-one kernel or rerunning until a timing looks favorable.

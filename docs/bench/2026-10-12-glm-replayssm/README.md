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
model-quality pass. Next is one physical F16-read Metal leaf to measure conversion
and raw-input flush costs rather than extrapolating CPU emulation into a speed
claim. Actual-model capture is useful only if the measured precision/cost frontier
leaves a concrete model-relevance question; production state stays F32 meanwhile.

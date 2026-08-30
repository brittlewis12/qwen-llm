# Muse Transposed-F16 FF Sidecar KILL

Decision: **KILL** transposed-F16 feed-forward sidecars for the scoped R256
eight-use contract. The representation clears its 15% whole-FF performance
gate but fails every relative-L2 and cosine requirement, with no precision
rescue that fits the remaining time budget.

## Predeclared Gate

The private model-free candidate used deterministic nonzero Q8 down, gate, and
up matrices plus Q8-derived T16 primals at `n_query=512`. A 32x32 tiled kernel
dequantized and transposed each matrix into preallocated F16 sidecars: down
`[H,F]`, gate/up `[F,H]`, exactly 797,442,048 bytes total. A small oracle
verified every transposed F16 bit.

Each candidate command packed all three sidecars and then ran eight complete FF
reverses through `encode_mat_mat_f16_half_act_f32`. Each control command ran
eight incumbent Q8 FF reverses. Allocation and differential readback remained
outside both `GPUEndTime-GPUStartTime` intervals. One warmup per arm preceded
five order-alternating pairs.

Promotion required finite complete down, gate, up, and final outputs; relative
L2 at or below `5e-5`; scaled max
`max_abs(candidate-control) / max(max_abs(control), 1)` at or below `2e-4`;
cosine at or above `0.9999999`; and at least 15% median whole-FF time saving.
The invocation asserted and externally applied a 180-second limit. It loaded no
model asset.

## Result

| Arm | Samples | Median |
| --- | --- | ---: |
| F16 sidecar | `269.404875, 266.883417, 266.985375, 269.546625, 268.341750 ms` | `268.341750 ms` |
| incumbent Q8 | `323.753875, 322.294375, 323.872708, 324.075625, 322.718083 ms` | `323.753875 ms` |

Candidate speedup is `1.206498x`, a `17.116%` time saving. The 15% threshold is
`275.190794 ms`; only `6.849044 ms`, or 2.5524% of candidate time, remains for
any precision repair.

All compared values are finite, but numerical promotion fails:

| Output | Relative L2 | Scaled max | Cosine |
| --- | ---: | ---: | ---: |
| down | `6.9046998e-4` | `3.9660838e-5` | `0.999999761663` |
| gate | `2.4164549e-3` | `4.4545857e-5` | `0.999997083832` |
| up | `1.7305656e-3` | `8.2271290e-5` | `0.999998502972` |
| final | `9.9811476e-4` | `6.4104795e-4` | `0.999999501931` |

Relative-L2 misses are `13.81x`, `48.33x`, `34.61x`, and `19.96x`. Every
cosine misses its limit. Intermediate scaled max passes, while final scaled max
misses by `3.205x`.

The first `3.86 s` invocation stopped at the down assertion before printing its
already-collected timings. A reporting-only reorder moved evidence output ahead
of adjudication; one final `3.873 s` replay reproduced the down differential and
reported the complete packet. No threshold or candidate mechanism changed.

## Interpretation

The exact layout oracle proves the requested transpose and F16 rounding, not
equivalence to incumbent F32-dequantized values. Gate/up comparisons include
error propagated from down, so they are complete-chain acceptance results, not
isolated matrix attribution. The eight timed uses repeat one deterministic bank
input; that is valid for data-independent timing but is not eight-bank numerical
coverage. No model-quality claim follows from this synthetic gate.

F16 weights with F32 operands retain weight-rounding error and fall onto a
scalar local kernel. Exact transposed F32 sidecars require 1,594,884,096 bytes
and also lack an accelerated local path. A high-plus-residual F16 split adds a
second dense MMA train, far beyond the 6.849 ms rescue budget. Dense
dequantized sidecars therefore close for this exact eight-use contract without
relaxing numerical limits.

The pack kernel, Rust wrapper, layout oracle, complete FF harness, and all
private buffers were removed in full. Production remains Q8.

## Disposition

The exact released-Q8 engine is at a local optimization plateau: 27.3258 seconds
per R256 shard-prompt projects to 4.934 engine hours for 25 prompts and 26
shards. Replay is 1.165753 seconds, bank commands are 24.068764 seconds, and
all other engine work is only 2.091283 seconds, or 7.653%. Persistent-bank or
readback work cannot meet a 15% engine gate from that envelope.

Freeze the production corpus and prompt count, then use the existing resumable
26-shard workflow. Do not manufacture another optimization gate without new
capability or measured ceiling.

Adversarial KILL and corrected plateau review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.

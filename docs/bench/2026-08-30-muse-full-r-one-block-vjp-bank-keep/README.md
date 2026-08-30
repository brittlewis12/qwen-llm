# Muse Full-R One-Block VJP Bank KEEP

Decision: **KEEP** the fixed-scratch B32 full-attention block VJP as the
production motor for Muse full-R row slabs. Preserve the scalar path as the
composition oracle.

## Mechanism

One caller-owned serial Metal command applies the complete reverse block:
post-FFN norm, Q8 FFN projections, SwiGLU, residual branches, gated causal GQA,
Q/K/V/gate projections, and the pre-attention norm. Periodic VJPs reuse the
same `T` primal rows across all basis rows. Four hidden banks, three FFN banks,
three query banks, two KV banks, and two attention partial banks give the B32,
T16 release shape about 210 MiB of fixed scratch, excluding caller current and
next buffers.

The opaque prepared block binds its exact config, copied layer bindings, block
number, replay state, and Metal registry ID. The workspace is device-bound;
encoding rejects a foreign context, encoder, workspace, current tensor, or next
tensor, as well as overlapping current and next ranges. Workspace scratch must
not be reused by another command before its owning command completes.

## Correctness

The model-free composition gate compares every bank row with the incumbent
scalar one-block VJP under both J and R rules.

| Basis | Tokens | Rules | Worst relative L2 | Worst scaled max | Minimum cosine |
|--:|--:|:--|--:|--:|--:|
| 1 | 3 | J, R | 1.79e-7 | 1.49e-7 | 0.99999999999998 |
| 3 | 5 | J, R | 1.94e-7 | 2.38e-7 | 0.99999999999998 |
| 8 | 16 | J, R | 2.39e-7 | 3.21e-7 | 0.99999999999996 |

The native Q8 transpose VJP and causal-GQA VJP retain their separate release
qualification. No checkpoint or model asset was opened for this packet.

## Timing Gate

The preregistered release-shape zero-Q8 R-rule probe required B8 below 100 ms,
B32 below 300 ms, and selected B8 only if B32 cost more than five times B8.
Each arm used one warmup followed by three standalone command timings.

| Basis | Samples (ms) | Mean (ms) |
|--:|:--|--:|
| 8 | 13.886, 13.911, 13.921 | 13.906 |
| 32 | 50.804, 50.839, 50.654 | 50.766 |

The B32/B8 ratio is `3.651`, selecting B32. Synthetic model preparation took
`0.083 s`; the two focused correctness and timing tests completed in `0.46 s`
after compilation. The fixture and B32 scratch remain below 0.9 GiB resident.

At 208 B32 slabs per hidden matrix, the measured command time projects one
full-attention source step at about 10.56 GPU-seconds per prompt. Treating all
51 reverse blocks as equal gives a conservative mechanism-only projection of
about 3.74 GPU-hours for 25 prompts, before capture, reduction, checkpointing,
and the still-unimplemented sliding-block path. This replaces the scalar
nine-day compute envelope with an hours-scale motor; it is not authorization to
launch the full fit.

## Disposition

Build the first production consumer as target block 51 to source layer 50 only:
B32 R-rule row slabs, prompt-ordered F32 accumulation, replay diagnostics, and
resumable row shards. That closes the control plane without persisting a 2.64
GiB block frontier. Sliding-block inverse RoPE is the next mathematical seam
before expanding the same engine through sources 49 to 1.

Adversarial design, integration, and promotion review:
`01a05072-6951-7782-978a-2f274d90f474`.

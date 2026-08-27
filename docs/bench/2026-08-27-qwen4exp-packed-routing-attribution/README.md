# Flash-Next Packed Routing Attribution

Decision: **GO** to a default-off strict E8xP32 F32 router-projection prototype
at N=2,048. Kill top-k plus bucket fusion; park interactive projection pending
evidence from the shared exact-order candidate.

## Source And Protocol

- Source: `e86f9b1460aca2428a37b77a6588ec64fd5b2f0c`.
- Device: Apple M4 Max with unified memory.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, on the external PCIe SSD.
- Sampling: one command, 64 serial encoders, 128 timestamps, and 68 spans.
  Standard layer-5 routing expands into router projection, top-k/shared-scale
  selection, and deterministic bucket publication.
- Controls: the ordinary motor retains its exact 11-dispatch census; N=8
  monolithic/seven-encoder route metadata, scratch, output, and guards remain
  bitwise equal; endpoint logits are bitwise equal across first/warm/profile.
- Acceptance: GPU ratio `0.985-1.015`, wall ratio `0.98-1.02`, and raw coverage
  `0.995-1.005`.

## Observer Results

| Tokens | Pass | Wall (ms) | Command GPU (ms) | Outside GPU (ms) |
|---:|:---|---:|---:|---:|
| 18 | first | 4,030.849 | 254.431 | 3,776.418 |
| 18 | warm | 244.617 | 242.735 | 1.882 |
| 18 | profiled | 247.290 | 244.692 | 2.597 |
| 2,048 | first | 4,242.452 | 3,775.235 | 467.216 |
| 2,048 | warm | 3,765.955 | 3,762.647 | 3.308 |
| 2,048 | profiled | 3,759.238 | 3,754.713 | 4.525 |

- N=18: GPU ratio `1.008065`, wall ratio `1.010928`, raw coverage
  `1.000000`; output was `HELLO` and scalar decode reached EOS.
- N=2,048: GPU ratio `0.997891`, wall ratio `0.998216`, raw coverage
  `1.000000`; profiled throughput was `544.77 tok/s`.
- Routing encoder-boundary residual was `0.001 ms` at both shapes.

## Routing Components

Command shares credit the same 43 standard layers used by the parent MoE
packet.

| Component | N=18 ms/layer | N=18 share | Decision | N=2,048 ms/layer | N=2,048 share | Decision |
|:---|---:|---:|:---|---:|---:|:---|
| F32 router projection | 0.203292 | 3.572473% | PARK | 15.863542 | 18.167362% | KEEP |
| top-k/shared selection | 0.020209 | 0.355135% | KILL | 0.280416 | 0.321140% | KILL |
| bucket publication | 0.015500 | 0.272383% | KILL | 1.617875 | 1.852835% | KILL |
| routing parent | 0.239875 | 4.215350% | PARK | 17.762833 | 20.342482% | projection only |

At N=2,048, projection owns 89.31% of routing. Top-k plus bucket together are
`1.898291 ms/layer`, below the preregistered `4.367 ms/layer` component KILL
floor. Fusing them is therefore closed even though their impossible combined
deletion would exceed the separate 1% command threshold.

The current F32 projection launches one threadgroup per expert row and 32-query
tile, yielding `512 x 64 = 32,768` threadgroups per layer. The existing strict
E8xP32 kernel computes eight output rows per query lane while retaining the
same scalar K traversal for each independent accumulator. Unlike float4 or
simdgroup-matrix alternatives, it has an established bitwise-exact strategy and
does not mutate F32 router storage or arithmetic order.

## Next Falsifier

Add a default-off router-specific strict E8xP32 arm scoped to released
`H=2560`, `E=512`, N in `{18, 2048}`, F32 weights, and Apple M4 Max.

- A: current generic `kernel_mat_mat_f32_f32`.
- B: strict E8xP32 F32 router kernel.
- A: rollback repetition to bound drift.

Require bitwise router logits, all route metadata and scratch, endpoint logits,
persistent session state, HELLO output, and scalar EOS continuation. Preserve
one command and the 11-dispatch ordinary motor. Any route-ID difference kills
the candidate regardless of speed.

Against the `15.863542 ms/layer` projection baseline:

- KEEP at `<=14.277188 ms/layer` and ordinary command GPU `<=3725.021 ms`;
- KILL above `15.426947 ms/layer`, a saving below `0.436595 ms/layer`; and
- PARK between those thresholds or when the leaf win fails to produce at least
  1% ordinary command-GPU improvement.

For N=18, promote only at projection `<=0.146842 ms/layer` and ordinary command
GPU `<=240.308 ms`; these require the leaf saving to convert to at least 1% of
the `242.735 ms` warm command. Otherwise close only interactive routing.

If strict E8 fails at either shape, close routing for that shape rather than
reopening selector fusion, reassociated matrix kernels, or router dtype changes.

# Flash-Next Packed Stage Attribution

Decision: **GO** to a MoE substage attribution packet before changing a packed
kernel. Do not prototype bridge-copy, HC epilogue, or dispatch-count changes
from the coarse block timings.

## Source And Protocol

- Source parent: `f3f7e95bf8d0e5e27c7ed17958b067d73eb0c99e`, plus the
  stage-sampling implementation checkpointed with this packet.
- Device: Apple M4 Max with unified memory.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, read from the external PCIe
  SSD at `/Volumes/wdblack`.
- Sampling: M4 Max rejected dispatch-boundary counters, so one command used 58
  serial sampled encoders and 116 timestamps. Standard GDN layer 5 and QSA
  layer 7 each expanded from one coarse encoder to six flat stage encoders.
- Replay: first unprofiled, reset, warm unprofiled, reset, then profiled. The
  CLI required bitwise-equal full endpoint logits across all three passes.
- Acceptance: GPU observer ratio `0.985-1.015`, wall ratio `0.98-1.02`, and
  raw timestamp coverage `0.995-1.005`.

The N=18 prompt was the no-thinking HELLO fixture. The N=2,048 prompt was
`<|im_start|>` repeated exactly 2,048 times and admitted the maximum dense
packed chunk. Both requests used `QWEN4EXP_PACKED_PREFILL_PROFILE=1` and the
released binary.

## Observer Results

| Tokens | Pass | Wall (ms) | Command GPU (ms) | Outside GPU (ms) |
|---:|:---|---:|---:|---:|
| 18 | first | 13,750.712 | 251.658 | 13,499.054 |
| 18 | warm | 243.844 | 241.746 | 2.098 |
| 18 | profiled | 244.371 | 241.816 | 2.555 |
| 2,048 | first | 5,329.283 | 3,775.503 | 1,553.780 |
| 2,048 | warm | 3,763.985 | 3,760.795 | 3.190 |
| 2,048 | profiled | 3,761.621 | 3,757.903 | 3.718 |

- N=18: GPU ratio `1.000290`, wall ratio `1.002163`, raw coverage
  `1.000000`; output was `HELLO` and scalar decode reached EOS.
- N=2,048: GPU ratio `0.999231`, wall ratio `0.999372`, raw coverage
  `1.000000`; profiled throughput was `544.42 tok/s`.
- The five added encoder boundaries cost only `0.002 ms` in layer 5 and
  `0.002 ms` in layer 7 at N=18, then `0.003/0.003 ms` at N=2,048.

## Representative Blocks

| Tokens | Stage | GDN layer 5 (ms) | QSA layer 7 (ms) |
|---:|:---|---:|---:|
| 18 | attention HC | 0.321 | 0.324 |
| 18 | mixer | 0.953 | 0.960 |
| 18 | mixer bridge + combine | 0.761 | 0.747 |
| 18 | FFN HC | 0.326 | 0.323 |
| 18 | MoE | 1.931 | 1.985 |
| 18 | MoE bridge + combine | 0.747 | 0.795 |
| 18 | coarse block | 5.042 | 5.137 |
| 2,048 | attention HC | 2.941 | 2.936 |
| 2,048 | mixer | 28.088 | 33.054 |
| 2,048 | mixer bridge + combine | 1.468 | 1.458 |
| 2,048 | FFN HC | 2.942 | 2.940 |
| 2,048 | MoE | 39.426 | 39.472 |
| 2,048 | MoE bridge + combine | 1.461 | 1.469 |
| 2,048 | coarse block | 76.327 | 81.331 |

Using layer 5 for 34 GDN blocks and layer 7 for 12 QSA blocks attributes about
37.0% of the N=18 command and 48.3% of the N=2,048 command to common MoE work.
At N=2,048, GDN and QSA mixers account for another 25.4% and 10.6%; both HC
reads are about 7.2%, both bridge/combine stages 3.6%, bootstrap 4.5%, and the
tail 0.03%. The representative extrapolation leaves 0.46% unassigned.

At N=18, GDN/QSA mixers are about 13.4%/4.8%, HC is 12.3%, and bridge/combine
is 28.9%. The raw representative extrapolation over-assigns the exact command
by 1.44%, exposing small-N layer variance rather than hiding it with
normalization. The bridge payload is tiny and its bucket contains HC injection
work, so it earns a later split, not a destination-aware copy prototype. QSA,
bootstrap, and tail remain below the 5% interactive gate.

## Next Falsifier

Split the standard layer-5 packed MoE stage into five serial sampled encoders:

1. routing, top-k, shared gate, and route bucketing;
2. routed gate/up plus SwiGLU;
3. routed down;
4. ordered weighted reduction; and
5. shared-expert projections plus gated accumulation.

Keep one command and the existing dispatch order. Require bitwise endpoint
replay, routing/workspace equivalence, first scalar continuation, raw coverage
`0.995-1.005`, GPU observer ratio `0.985-1.015`, and wall ratio `0.98-1.02`.
Credit only the 43 standard-dtype MoE layers when extrapolating; leave all five
dtype outliers uncredited.

Prototype only when one substage is at least 10% of command GPU and a concrete
mechanism can plausibly remove at least 10% of that substage. That requires at
least `2.42 ms` at N=18 or `37.58 ms` at N=2,048 for a 1% whole-command win.

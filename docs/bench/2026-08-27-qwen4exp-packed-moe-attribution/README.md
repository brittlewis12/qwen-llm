# Flash-Next Packed MoE Attribution

Decision: **GO** to an N=2,048 routing subdivision before any routing kernel
prototype. Keep standard IQ3 gate/up as a candidate at both prompt shapes and
standard IQ4_NL down as an interactive-only candidate. Do not optimize ordered
reduction, the shared tail, encoder boundaries, or dtype outliers from this
packet.

## Source And Protocol

- Source: `0c7ec443ccfa40d6c56b956bb8f676266b25c856`.
- Device: Apple M4 Max with unified memory.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, read from the external PCIe
  SSD at `/Volumes/wdblack`.
- Sampling: one command, 62 serial encoders, 124 stage-boundary timestamps,
  and 65 spans. Only standard GDN layer 5 expands its MoE parent into routing,
  routed gate/up, routed down, ordered reduction, and shared-tail encoders.
- Replay: first unprofiled, reset, warm unprofiled, reset, then profiled. The
  CLI required bitwise-equal full endpoint logits across all three passes.
- Acceptance: GPU observer ratio `0.985-1.015`, wall ratio `0.98-1.02`, and
  raw timestamp coverage `0.995-1.005`.

The ordinary packed motor retained its exact 11-dispatch census. A focused N=8
comparison also proved the monolithic and five-encoder forms bitwise equal for
route IDs and weights, shared scales, counts and slots, every routed/shared
scratch view, final output, and guard regions.

## Observer Results

| Tokens | Pass | Wall (ms) | Command GPU (ms) | Outside GPU (ms) |
|---:|:---|---:|---:|---:|
| 18 | first | 3,162.753 | 256.907 | 2,905.846 |
| 18 | warm | 245.905 | 243.902 | 2.003 |
| 18 | profiled | 246.727 | 244.214 | 2.513 |
| 2,048 | first | 4,231.588 | 3,772.504 | 459.085 |
| 2,048 | warm | 3,766.616 | 3,763.380 | 3.236 |
| 2,048 | profiled | 3,759.329 | 3,755.511 | 3.817 |

- N=18: GPU ratio `1.001279`, wall ratio `1.003343`, raw coverage
  `1.000000`; output was `HELLO` and scalar decode reached EOS.
- N=2,048: GPU ratio `0.997909`, wall ratio `0.998065`, raw coverage
  `1.000000`; profiled throughput was `544.75 tok/s`.
- Four internal MoE encoder boundaries cost `0.001 ms` at N=18 and `0.002 ms`
  at N=2,048. The complete layer-5 boundary residual remained `0.002/0.003 ms`.

## MoE Substages

The command-share projection credits only the 43 standard IQ3_XXS gate/up plus
IQ4_NL-down layers. It gives no credit to layers 2, 4, 30, 46, and 47, whose
routed down is Q8_0; layer 2 also uses IQ4_XS gate/up.

| Substage | N=18 ms/layer | N=18 share | Decision | N=2,048 ms/layer | N=2,048 share | Decision |
|:---|---:|---:|:---|---:|---:|:---|
| routing | 0.249833 | 4.399% | KILL | 17.715500 | 20.284% | KEEP, split first |
| routed gate/up | 0.827041 | 14.562% | KEEP | 13.577458 | 15.546% | KEEP |
| routed down | 0.629292 | 11.080% | KEEP | 5.917875 | 6.776% | PARK |
| ordered reduction | 0.008209 | 0.145% | KILL | 0.454667 | 0.521% | KILL |
| shared tail | 0.204084 | 3.593% | KILL | 1.718792 | 1.968% | KILL |
| boundary residual | 0.001458 | 0.026% | KILL | 0.002208 | 0.003% | KILL |

Routing scales `70.91x` from N=18 to N=2,048, while gate/up scales `16.42x`
and down `9.40x`. This is consistent with sparse N=18 expert buckets leaving
the fixed N=16 expert kernels poorly utilized, then N=2,048 assigning about 40
routed slots per expert on average and amortizing expert weights. The packet
does not measure route-count bands or weight traffic directly. Routing still
projects every token, selects ten of 512 experts, and publishes all buckets.

The routing parent therefore does not identify one implementation mechanism.
Its router projection, deterministic top-k/shared-scale selector, and bucket
constructor have different remedies and must be measured separately. Gate/up,
by contrast, is already one isolated grouped IQ3 kernel and clears the 10%
whole-command gate at both shapes.

## Next Falsifier

Expand only layer-5 routing into three same-command encoders:

1. router projection;
2. top-k plus shared-scale selection; and
3. deterministic route bucketing.

Retain the other four MoE substages, yielding seven MoE encoders. Keep every
kernel, buffer, dispatch, and arithmetic order unchanged. Reuse the existing
bitwise scratch gate and first/warm/profile replay at N=18 and N=2,048.

For N=2,048, crediting 43 standard layers:

- KEEP a component at `>=8.734 ms/layer` only when a concrete mechanism can
  plausibly save `>=0.873 ms/layer`, or `>=37.555 ms` over the command;
- PARK `4.367-8.734 ms/layer`; and
- KILL `<4.367 ms/layer`, or any mechanism capped below `18.778 ms` command
  saving.

Prototype fused top-k plus bucket publication only if their measured
composition supports the full `37.555 ms` command gate. Removing one dispatch
is not a mechanism-level savings argument.

# V4.1 CED Encoder GPU Attribution: Attention 54%, FFN 46%

2026-10-29. Full 8K per-layer stage profile on main `508bf6e3` (the CED
encoder runs the same layer code on the same 4,096-row chunks). Artifact:
GSQ-RCO IQ2_XS, 40 layers, M5 Ultra. Raw:
`~/qwen-runs/2026-10-29-v41-ced-timeline/full-stages.*`.

## Decision

The next GPU experiment is bounded half-operand reuse for attention, with
tiles and the output projection as the primary consumers. The measured
attention budget is large enough to matter; the previous "50.3% attention at
4.834 s" figure is stale.

## Method

`QWEN_V41_PREFILL_STAGES=1` on a Full 8K prefill records per-layer GPU
intervals (`prefill_attention_hc/prepare/serial/tiles/finish_post`,
`prefill_ffn` and its stage counters). Sum the 40 layers of the two 4,096-row
command chunks. The CED encoder executes the same layer bodies for layers
0-19 on the same chunk shape, so the split transfers; the CED encoder
additionally pays publication and retention.

## Numbers

Full 8K, two chunks, GPU **7,453.3 ms**:

| group | GPU ms | share |
| --- | ---: | ---: |
| attention total | 3,991.7 | 53.6% |
| - tiles | 1,850.0 | 24.8% |
| - finish_post | 1,107.0 | 14.9% |
| - prepare | 664.4 | 8.9% |
| - hc | 208.0 | 2.8% |
| - serial | 162.3 | 2.2% |
| FFN total | 3,439.6 | 46.1% |
| - gate_up | 1,645.0 | 22.1% |
| - down | 780.2 | 10.5% |
| - shared | 606.8 | 8.1% |
| - route/router/sum/hc/mid | 439.1 | 5.9% |

Re-aggregated for encoder layers 0-19 only (the earlier packet used the
40-layer average; the halves differ little, but the per-stage split was
wrong): attention **1,967.3 ms** (hc 157.2, prepare 338.3, serial 39.7,
tiles 878.5, finish_post 553.6), FFN **1,708.5 ms**. The Full 0-19 total is
3,675.8 ms; the measured CED encoder is 3,337.7 ms (0.908x), so the CED
encoder's attention is **~1,786 ms** (tiles ~798, finish_post ~503, prepare
~307, hc ~143, serial ~36) and FFN **~1,551 ms**, plus publication/retention.

The key count is **128 window + 512 selected = at most 640 per query**, not
1,024. Useful encoder QK+PV is about **12 TFLOP** over the 20 layers, so the
attention runs about **6.7 TFLOPS** against the 129.6 TFLOPS resident-half
ceiling and the 32.1 TFLOPS band-QK probe. The pure GEMMs are not the limit;
gather, conversion, finish/projection and the scalar core dominate.

The score kernel also reloads and converts the whole query for every 32-key
panel: at 640 keys that is twenty traversals of a 128 KiB query, about
**2.5 MiB of logical query loads per query**. That is twice the K/V logical
load (about 1.25 MiB), and the half-KV probe did not remove it. A
query-resident partition is a distinct attention-dataflow premise.

## Budget

The attention operand-reuse experiment can target tiles (~925 ms), prepare
(~332 ms) and finish_post (~553 ms). A 10% complete-core reduction is
~179 ms (4.1% of the 4.38 s wall); 20% is ~358 ms (8.2%). These are upper
bounds for a full-core cut, not predictions.

## Requirements

Keep the arithmetic sequence; construct the shared operands once, charge
both consumers and invalidation, and preserve score/softmax/PV/normalize
bits. Require a material complete-core gain (at least 1% wall) before
integration, plus the existing bitwise Full/CED pair and qualification.

## Limits

Full layers proxy the CED encoder (same code, same chunk shape); the CED
encoder's publication/retention and its 20-layer selection are not in the
Full profile. Stage intervals are sampled GPU envelopes, not exclusive
resource costs, and the machine state affects absolute values. The
extrapolation is an attribution tool, not a measurement of the CED kernel
mix. Cooldown arms measure burst performance; continuous runs and real
prompts are separate observations.

# v0.133 Dense 27B llama.cpp Differential Profile

Status: attribution-only differential after `v0.132`. `llama.cpp` used
`GGML_METAL_PROFILE_OPS=1 -v`, which serializes one graph node per command buffer.
`qwen-llm` used `QWEN_PREFILL_TRACE_LAYER_PHASES=1` and detailed attention phase
tracing, which also inserts commit/wait boundaries. Do not read these as promotion
throughput numbers.

Artifacts:

- `lcpp-pp512-summary.tsv`, `lcpp-pp512.json`
- `lcpp-pp4096-summary.tsv`, `lcpp-pp4096.json`
- `qwen-pp512-summary.tsv`
- `qwen-pp4096-chunk1024-split-summary.tsv`
- `qwen-pp4096-chunk512-split-summary.tsv`

## pp4096 Read

Dense 27B with qwen matrix-G6/G8 enabled. The most useful comparison is not total
serialized time; it is the phase shape.

| Bucket | qwen ms | llama.cpp ms | Read |
| --- | ---: | ---: | --- |
| FFN gate/up/SwiGLU | `8408.19` | `8106.38` | qwen `~3.7%` slower, not a giant gap |
| FFN down/residual | `4309.56` | `4199.97` | qwen `~2.6%` slower, not a giant gap |
| GDN front projections | `3612.91` | `2997.76` | qwen `~20%` slower; highest local delta |
| GDN output projection | `1160.69` | `1134.26` | close |
| Attention KQ/KQV/softmax | `498.99` | `371.30` | qwen matrix body still slower in trace |
| Attention projection/front | `877.83` | `835.98` | close |
| Attention output/gate | `377.01` | `~399` | close |

Chunk matching did not help in the trace lane: qwen `--prefill-chunk 512` made
the serialized pp4096 summary worse than the default `1024` chunk across major
FFN/GDN buckets.

## Read

- Dense fused-Q4 SwiGLU was the wrong default bet: the split trace says qwen's
  FFN mat-mat buckets are only a few percent slower than llama.cpp under this
  attribution mode.
- The larger dense differential is now GDN/front and some attention-body residue,
  not a standalone SwiGLU fusion problem.
- The next dense kernel work should inspect the GDN front projection lowering and
  lcpp's corresponding `MUL_MAT node`/`z` shapes before touching another FFN
  fusion branch.
- This profile is still serialized attribution. Require an end-to-end candidate
  gate before defaulting anything derived from it.

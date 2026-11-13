# GLM-5.3 Flash prefill: MLX comparison

Scope: `/Users/tito/code/mlx`, `/Users/tito/code/mlx-serve`,
`/Users/tito/code/mlx-lm`, `/Users/tito/code/omlx`, plus the supplied
`/Users/tito/qwen-runs/2026-11-12-mlx-prefill-review/review.md`.

## What is comparable

The supplied review found no same-machine GLM prefill measurement from MLX or
mlx-serve. The reported 0.60x MLX prefill ratio therefore remains an
unmeasured project estimate; the implementation comparison below does not
establish a numeric gap.

MLX's Metal quantized matmuls use affine groupwise weights. The kernel loads a
packed code and reconstructs it with group scale and bias; mlx-serve's custom
gather-QMM supports affine 2/3/4/8-bit weights with group sizes 32/64/128.
The reviewed GLM-5-next pack is affine oQ4, so it does not decode this
project's IQ2_S/IQ3_S codebooks. The Opus
review estimates roughly five simple arithmetic operations per affine
weight versus about twelve for the current IQ path; treat that as a source
review estimate, not a hardware-counter measurement. Our target stays stock
UD-IQ3_XXS with IQ2_S gate/up, IQ3_S down, and Q6_K/Q8_0 dense weights.

mlx-serve's GLM plugin delegates prefill to the common MoE block, which routes
rows into expert-major order and calls gather-QMM for the routed banks. Its
separate GLM `fusedDecode` path is a two-dispatch routed-plus-shared path.
The NAX tile plan uses routed pairs per expert to choose row tile sizes. For
K≥1024 and a paired gate/up call, the policy chooses BM96 when the mean is
below 96 rows per expert, then BM128 at 96 or more; an unpaired down call
uses BM96 at those counts. At 2,048 input rows and top-8 over 288 experts,
the mean is 56.9 routed rows per expert; at 4,096 it is 113.8. This is the
full-chunk occupancy difference that matters for the proposed 128-row path.
At the 80K append widths, the
measured means are 7.1 and 28.4, with p50 counts 5 and 21 respectively, so
the append regime from the existing small-row work should not be used as a
proxy for full prefill.

The inspected mlx-serve engine default chunk is 8,192 rows; admission may pin
a lower value. oMLX's GLM-5.3 scheduler floor is 4,096 rows when its native
sparse-MLA/NAX checks pass. This is a model-specific floor, not evidence that
every GLM request runs at that width.

For attention, mlx-serve's GLM-5-next plugin uses its NAX sparse-MLA/indexer
path and a separate KDA path. MLX-LM's `dsa.py` and `deepseek_v32.py` show a
related blocked-indexer/top-k and selected-key sparse-attention design; the
16,384-token sparse-prefill threshold is for that DeepSeek path, not a claim
about GLM-5.3. oMLX's vendored GLM-5-next implementation includes gated-delta
linear attention and sparse MLA.

## Source and license notes

- MLX and MLX-LM carry MIT licenses.
- mlx-serve's own code carries MIT; its `NOTICE` lists third-party components
  that retain their separate licenses.
- oMLX carries Apache-2.0. mlx-serve's
  `src/sdk/blocks/moe/gather_qmm_nax.zig` identifies its NAX gather-QMM port
  from oMLX's `m5_gather_qmm_nax.py` and the source revision.
- No code from these repositories was copied or adapted in this work. Any
  later port must preserve the source attribution and applicable license
  notices.

## Source paths inspected

- MLX: `mlx/backend/metal/kernels/quantized.metal`,
  `quantized_nax.metal`, `quantized.h`, and `jit_kernels.cpp`.
- mlx-serve: `src/plugins/glm5_next/moe.zig`,
  `src/sdk/bases/blocks/moe.zig`, `src/sdk/blocks/moe/gather_qmm_nax.zig`,
  `src/plugins/glm5_next/dsa_nax.zig`, and `src/engines/mlx/generate.zig`.
- MLX-LM: `mlx_lm/models/dsa.py` and `mlx_lm/models/deepseek_v32.py`.
- oMLX: `omlx/scheduler.py` and its vendored
  `glm5_next` implementation under `omlx/patches/mlx_vlm_glm5_next_compat`.

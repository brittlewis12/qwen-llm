# Third-Party Notices

## DwarfStar

`kernel_deepseek_v4_lightning_indexer_scores_f16_tiled_f32` in
`kernels/deepseek_v4.metal` adapts the 8-query by 32-row tile organization from
DwarfStar's `kernel_dsv4_indexer_scores_tiled_f32`, revision
`b0309611041655f4e45671cfd9c9886aff161406`.

`kernel_glm53_kda_decode` in `kernels/glm53_kda.metal` adapts DwarfStar's
`kernel_glm53_kda_decode` (fused conv, L2 norm, per-channel decay, delta-rule
update and gated output norm) as of revision
`0aaea5a238fb41a35106a551e73c8409dfb751ac` (file last changed in
`147109a3cc49f12e4706e9fb7f81816c79e016a8`), consuming the GGUF's
`ssm_a = -exp(A_log)` instead of `A_log`. The packed prefill kernels
`kernel_glm53_kda_prefill_{prepare,recurrence,output}` in the same file adapt
DwarfStar's kernels of the same names at that revision, with the same change.

## llama.cpp

`kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mm64x32` in
`kernels/mat_mat_iq2_xs.metal` adapts the indirect 64-output by 32-route
quantized matrix topology from llama.cpp's `kernel_mul_mm_id`, revision
`6a32c29a746a2e44de463de647f9f6661eb5086b`.

`kernel_mat_mat_q8_0_fewrow_f32` in `kernels/mat_mat_q8_0_fewrow.metal`
adapts llama.cpp's few-row MMA kernel `kernel_mul_mv_mma_blk` with its Q8_0
fragment builder and `ggml_metal_op_mul_mat_mma_tiling` rule
(`ggml/src/ggml-metal/kernels/mul_mv_mma.metal`, PR #29869, revision
`a3a1c4747fdc0dcad40b3946108b89375d9a7d0e`), without batch dimensions or the
fused residual add.

The adapted source portions are provided under the following terms:

MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

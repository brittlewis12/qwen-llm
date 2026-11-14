# V4.1 attention prepare and finish/post attribution

Measurement-only capture on production file A, DeepSeek-V4.1 Flash GSQ-RCO
IQ2_XS, at 80K. One model load and one CED fill to 81,920 tokens were used;
each width had one warmup, three stage-counter samples and three
per-dispatch timing profiles. Every profiled output matched the warmup
bitwise. The selected source is `ff9dae6de4ddbd738808aca36d62ab9e1d68e50e`,
M5 `iter` binary SHA-256
`8c109045b382f2865407d95e05193d63be4a247c0e78ebca6f0fa42a42ee473d`.

Lease `v41-depth` was acquired at 02:49:07 UTC after a 413.3-second queue
wait. Thermal pressure was 0 for the 60-second settle and through completion
at 02:51:09 UTC; the measured process ran at nice 0. Preflight found no
resident model. The load snapshot recorded the expected Ghost background
jobs (including AOT workers and `program.py --jobs 8`); they were left
untouched. Full raw stage and dispatch rows, monitor, process snapshot and
stderr are in `~/qwen-runs/2026-11-14-v41-attention-post/` on M5 and the
matching local run directory.

The measured stage counters are the authoritative span times. Shares are of
the append wall in the same three stage-profile runs.

| Stage | 256 rows: median [range], wall share | 1,024 rows: median [range], wall share |
| --- | ---: | ---: |
| Append wall | 458.01 ms [457.32–463.25] | 967.72 ms [967.49–969.45] |
| `prefill_attention_prepare` | 31.61 ms [31.21–32.12], 6.9% | 74.76 ms [74.60–75.12], 7.7% |
| `prefill_attention_finish_post` | 53.55 ms [53.47–53.70], 11.7% | 116.96 ms [116.23–117.06], 12.1% |

These reproduce the earlier 80K estimates (31.02/74.84 ms prepare and
53.11/117.64 ms finish/post) within run variation. The one 80K fill took
48,349 ms.

## Per-kernel work

Times below are the median across three profiles of the sum of that kernel's
isolated dispatch durations for one append. Calls are per profile. Q5/Q8/BF16
matmul FLOPs are `2*M*K*N` from the captured shapes. The byte column is a
compulsory operand-footprint estimate: each bound weight, activation and
output is counted once per dispatch. Copy kernels use their logical source
and destination extents; RoPE counts its F32 read/write pass. These are not
measured DRAM transactions. In particular, matmul tile reuse, cache hits and
some repeated HC loads are not represented, and the timestamp profiler does
not expose hardware byte counters. “Apparent GB/s” is this footprint divided
by isolated GPU time; values above the 800 GB/s roof show why it cannot be
read as physical bandwidth. Tensor percentages use the 131.7 TFLOP/s M5 peak;
for RoPE the roof is bandwidth-limited. `—` means no useful FLOP rate for a
copy/check kernel.

| Span / kernel and work | Calls | 256-row append | 1,024-row append |
| --- | ---: | --- | --- |
| Prepare: `kernel_mat_mat_q8_0_f16act_tensor[_wide]`, Q8 query-A/KV projections | 80 | 10.01 ms; 0.673 GB; 0.188 TFLOP; 67 apparent GB/s; 18.8 TF/s (14% tensor roof) | 22.05 ms; 1.522 GB; 0.752 TFLOP; 69 apparent GB/s; 34.1 TF/s (26%) |
| Prepare: `kernel_mat_mat_q5_K_f16act_tensor[_wide]`, rank-to-query-B projection | 40 | 16.48 ms; 2.522 GB; 0.859 TFLOP; 153 apparent GB/s; 52.1 TF/s (40%) | 41.19 ms; 6.627 GB; 3.436 TFLOP; 161 apparent GB/s; 83.4 TF/s (63%) |
| Prepare: `kernel_mat_mat_bf16_f16act_tensor`, rank-fed compressor/index projections | 23 | 3.67 ms; 0.205 GB; 0.032 TFLOP; 56 apparent GB/s; 8.6 TF/s (7%) | 6.12 ms; 0.451 GB; 0.126 TFLOP; 74 apparent GB/s; 20.6 TF/s (16%) |
| Prepare: `kernel_deepseek_v41_attention_rope_rows`, query, latent and index-query RoPE | 88 / 176 | 12.19 ms; 2.793 GB; 0.140 GFLOP; 229 apparent GB/s; 11.5 GF/s (29% of 40 GF/s bandwidth roof) | 36.56 ms; 11.174 GB; 0.562 GFLOP; 306 apparent GB/s; 15.4 GF/s (38% of bandwidth roof) |
| Prepare: `kernel_scatter_offset_f32_to_f16`, half-activation staging | 120 | 1.81 ms; 2.406 GB; —; 1,330 apparent GB/s (166% of BW roof; proxy invalid) | 11.88 ms; 9.626 GB; —; 811 apparent GB/s (101%; proxy invalid) |
| Prepare: `kernel_rms_norm_mul_rows_f32`, query-rank and latent norms | 80 | 0.54 ms; 0.220 GB; ≈0.073 GFLOP; 411 apparent GB/s (51%) | 0.88 ms; 0.881 GB; ≈0.294 GFLOP; 1,002 apparent GB/s (125%; proxy invalid) |
| Prepare: `kernel_deepseek_v41_attention_history`, old ring plus new latent gather | 40 | 2.38 ms; 0.063 GB; —; 26 apparent GB/s (3%) | 13.24 ms; 0.189 GB; —; 14 apparent GB/s (2%) |
| Prepare: `kernel_deepseek_v41_attention_publish_raw`, append to raw ring | 40 | 1.03 ms; 0.021 GB; —; 20 apparent GB/s (3%) | 0.98 ms; 0.021 GB; —; 21 apparent GB/s (3%) |
| Finish/post: `kernel_mat_mat_q5_K_f16act_tensor[_wide]`, eight output-A plus output-B projections | 360 | 47.90 ms; 3.460 GB; 1.546 TFLOP; 72 apparent GB/s; 32.3 TF/s (25% tensor roof) | 86.88 ms; 7.613 GB; 6.185 TFLOP; 88 apparent GB/s; 71.2 TF/s (54%) |
| Finish/post: `kernel_deepseek_v41_group_rows`, grouped query and low-rank layouts | 80 | 3.28 ms; 3.355 GB; —; 1,023 apparent GB/s (128%; proxy invalid) | 13.13 ms; 13.422 GB; —; 1,022 apparent GB/s (128%; proxy invalid) |
| Finish/post: `kernel_scatter_offset_f32_to_f16`, output-A/B half staging | 80 | 1.35 ms; 2.517 GB; —; 1,868 apparent GB/s (234%; proxy invalid) | 8.25 ms; 10.066 GB; —; 1,221 apparent GB/s (153%; proxy invalid) |
| Finish/post: `kernel_deepseek_v4_hc_post_batch`, mHC4 residual/hyper-connection mix | 40 | 2.01 ms; 1.888 GB; 1.887 GFLOP; ≈0.94 TF/s; 940 apparent GB/s (118%; proxy invalid) | 7.93 ms; 7.553 GB; 7.550 GFLOP; ≈0.95 TF/s; 952 apparent GB/s (119%; proxy invalid) |
| Finish/post: `kernel_deepseek_v41_attention_finite`, output finite check | 40 | 0.80 ms; 0.210 GB; —; 262 apparent GB/s (33%) | 3.15 ms; 0.839 GB; —; 266 apparent GB/s (33%) |

The norm count is approximately four arithmetic operations per element; mHC4
uses nine per output element. The reported finish/post Q5 group includes
320 output-A calls and 40 output-B calls. Prepare includes the dense Q/KV,
rank-fed projections, RMS norms, RoPE and K/V history preparation; it excludes
the separately labeled position-ordered index/selection serial path. Finish
includes output-A/B, finite checking and mHC4 post.

## Attribution limits and lever ranking

The per-dispatch GPU-time sums were 47.78/55.27 ms at 256 rows and
132.62/119.39 ms at 1,024 rows (prepare/finish-post), versus stage-counter
spans 31.61/53.55 and 74.76/116.96 ms. In the same separate profiling pass,
timestamp-interval unions were 32.69/54.24 and 87.51/115.99 ms. The 1K
prepare dispatch replay therefore adds about 13 ms versus the stage sample;
the isolated per-kernel prices are not an additive wall partition. Any
round-trip or copy price from those rows is an upper-bound intermediate
price, not a demonstrated saving. Do not infer a removable gap by subtracting
these sums from the whole-buffer span.

Ranked by priced kernel time at 1K (the 256-row figures are in the table):

1. **Q5 output-A/B tensor matmuls: 86.9 ms.** They compute the eight grouped
   output-A projections and output-B projection; aggregate rate is 71 TF/s,
   54% of tensor peak. No implemented same-order tile change is shown to be
   both bitwise and stage-clearing. A wider batch tile could preserve each
   output's K accumulation order, but it needs a new threadgroup tile and a
   bitwise witness; no prototype is justified from this census alone.
2. **Query projection plus query RoPE: 41.2 ms Q5 projection and 20.9 ms of
   the 36.6 ms RoPE dispatch price.** A Q5 output epilogue could apply RoPE
   to the F32 accumulator using the same angle and ordered F32 expressions,
   eliminating the separate query read/write pass. The measured query-RoPE
   price at 256 rows is only 4.4 ms, below the 6.3 ms reduction needed for a
   1.25x prepare-span gate; this is not a two-width winner.
3. **Q8 query-A/KV projections: 22.0 ms.** They are tensor matmuls at 34
   TF/s, 26% of tensor peak. A row-tile retile that preserves the K loop may
   be bitwise, but there is no cheap, validated layout change or evidence it
   can save the 18.7 ms required for a 1.25x prepare-span gate.

The cheap exact finish-side fusion is smaller: combining the first
`group_rows` permutation with its F32-to-F16 output-A input conversion can
preserve the exact F16 activation boundary. The relevant bounded prices are
below 21.4 ms even counting the full group/cast pair, while finish/post needs
23.4 ms saved at 1K (and 10.7 ms at 256) to reach 1.25x. The actual fused
subset is smaller still. History gather is 13.2 ms at 1K, also below the
prepare-span 14.95 ms requirement even if removed entirely. No candidate
therefore has a credible >=1.25x span case at both widths, so no optimization
was prototyped and no stage gate was claimed.

## Reproduction and implementation

The bench-only `attention-post` mode and dispatch labels are in this branch's
`v41_batch_depth` example and `gpu_step.rs`; they do not edit or instrument
`query_batch.rs`'s index/selection path. Profile source: `ff9dae6de`. Command:

```text
v41_batch_depth MODEL.gguf 32768 1 1 attention-post
```

The M5 profiler records per-dispatch `gpu_ns`, GPU timestamp intervals,
tensor shapes and byte extents. It does not expose actual DRAM-byte counters;
the apparent bandwidth values above must not be used as device-bandwidth
measurements.

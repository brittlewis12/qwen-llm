# V4.1 append depth attribution

Measurement only; no runtime change. Production file A of DeepSeek-V4.1 Flash
(GSQ-RCO IQ2_XS) was loaded once from `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf`.
One progressive shrinking-CED fill reached each target depth, then the probe
measured 256- and 1,024-row appends there. Each cell had one warmup and two
profiled reps; the two reps produced bitwise-identical logits. Depths are
32,768, 81,920 and 131,072 tokens (32K, 80K and 128K).

M5 Ultra, `iter` profile, lease task `v41-depth`; thermal level was 0 for the
60-second settle and throughout the capture. No other model process was
resident at either pre-load check. The measured binary ran at niceness 0
(`ps -o nice=`). The background `~/ghost` job was present: `program.py --jobs
36`, niceness 20, about 1.9 GiB RSS; system free-memory percentage was 95%.
It was left untouched. Raw cell profiles, driver/thermal log and the load
snapshot are in `~/qwen-runs/2026-11-13-v41-depth/` on M5.
The probe source was `3579a12ef`; the pinned M5 binary SHA-256 was
`67c19b7859ac4069eabccff6c46b27e9affaf7cc19d91f8493d08c79e1b15e97`.
The lease wait was 48.4 seconds; model probe ran 00:36:04–00:37:19 UTC.

The one progressive fill took 11,758 ms to 32K, 21,165 ms from 32K to 80K,
and 25,012 ms from 80K to 128K. Append wall below is the mean of two reps;
brackets show their min–max range.

| Append wall (ms) | 32K / 256 | 32K / 1,024 | 80K / 256 | 80K / 1,024 | 128K / 256 | 128K / 1,024 |
|---|---:|---:|---:|---:|---:|---:|
| Wall | 431.38 [429.98–432.79] | 926.34 [919.74–932.94] | 472.73 [472.51–472.95] | 972.53 [972.40–972.67] | 494.46 [494.31–494.61] | 1,026.58 [1,026.21–1,026.94] |
| Submitted GPU-duration sum | 416.55 | 900.80 | 458.12 | 953.60 | 480.10 | 1,007.50 |
| Encode | 5.42 | 2.44 | 4.38 | 2.51 | 4.40 | 2.38 |
| Commit/wait | 417.44 | 893.00 | 459.06 | 936.78 | 480.98 | 993.05 |

Stage times below are means across the two profiled reps, summed across
layers. GPU stages use the profiler's sampled GPU interval; host spans use
their recorded wall duration. `—` means no valid interval was recorded for
that stage at that width. `prefill_chunk_rows` is the submitted-command GPU
duration sum, shown as a whole-chunk reference rather than an exclusive stage.

| Stage / host span (ms) | 32K / 256 | 32K / 1,024 | 80K / 256 | 80K / 1,024 | 128K / 256 | 128K / 1,024 |
|---|---:|---:|---:|---:|---:|---:|
| `engram_read_block` | 4.66 | 18.20 | 5.40 | 20.95 | 5.24 | 18.58 |
| `prefill_select` | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| `prefill_admission` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `prefill_scratch_prepare` | 0.27 | 0.33 | 0.27 | 0.34 | 0.27 | 0.35 |
| `prefill_embedding` | 1.91 | 7.37 | 1.97 | 7.15 | 1.88 | 7.46 |
| `prefill_engram_hash` | 0.04 | 0.14 | 0.04 | 0.14 | 0.04 | 0.15 |
| `prefill_engram_prepare` | 5.33 | 0.00 | 6.11 | 0.00 | 5.98 | 0.00 |
| `prefill_engram_wait` | — | 21.21 | — | 24.05 | — | 21.62 |
| `prefill_host_prepare` | 7.24 | 7.40 | 8.07 | 7.18 | 7.86 | 7.49 |
| `prefill_host_finish` | 0.18 | 0.27 | 0.16 | 0.22 | 0.17 | 0.22 |
| `prefill_scratch_release` | 0.68 | 1.34 | 0.64 | 1.15 | 0.62 | 1.14 |
| `prefill_attention_hc` | 11.83 | 25.76 | 11.86 | 25.76 | 11.89 | 25.84 |
| `prefill_attention_prepare` | 30.97 | 74.40 | 31.02 | 74.84 | 31.00 | 74.86 |
| `prefill_attention_serial` † | 26.24 | 70.52 | 54.07 | 122.81 | 86.23 | 191.43 |
| `prefill_attention_tiles` | 38.02 | 117.21 | 39.23 | 117.39 | 38.78 | 116.39 |
| `prefill_attention_finish_post` | 53.14 | 117.10 | 53.11 | 117.64 | 53.12 | 117.45 |
| `prefill_ffn` | 254.46 | 486.93 | 267.06 | 486.79 | 257.32 | 473.35 |
| `prefill_ffn_hc` | 8.21 | 12.58 | 8.22 | 12.59 | 8.23 | 12.57 |
| `prefill_ffn_router` | 4.55 | 5.21 | 4.58 | 5.23 | 4.60 | 5.25 |
| `prefill_ffn_shared` | 67.81 | 49.27 | 70.23 | 48.72 | 68.23 | 49.88 |
| `prefill_ffn_route` | 1.82 | 11.90 | 1.77 | 11.92 | 1.83 | 11.92 |
| `prefill_ffn_gate_up` | 158.80 | 302.88 | 166.90 | 302.64 | 160.58 | 293.32 |
| `prefill_ffn_mid` | 1.38 | — | 1.38 | — | 1.38 | — |
| `prefill_ffn_down` | 76.35 | 139.79 | 80.76 | 139.73 | 77.34 | 135.25 |
| `prefill_ffn_sum` | 3.29 | 13.30 | 3.29 | 13.32 | 3.29 | 13.32 |
| `prefill_chunk_rows` | 416.55 | 900.80 | 458.12 | 953.60 | 480.10 | 1,007.50 |

† `prefill_attention_serial` has valid intervals on the same eight global
selection/index layers only (layer IDs 2, 8, 14, 20, 24, 28, 32, 36) in every
cell. Its other 32 layer entries are empty-stage timestamp records, not lost
work on active layers. `prefill_ffn_mid` is likewise empty at 1,024 rows on
the grouped path. Stage intervals can overlap; the FFN envelope and its
sub-stages are not additive exclusive costs. Do not sum these rows into a wall
or Amdahl fraction; see `2026-10-28-v41-stage-counter-intervals.md`.

Net endpoint slope is `(128K - 32K) / 6`, in milliseconds per additional 16K
tokens. This is the requested per-stage depth rate; local 32K→80K and
80K→128K slopes for the top stage are shown below.

| Stage | 256 rows | 1,024 rows |
|---|---:|---:|
| `engram_read_block` | +0.097 | +0.063 |
| `prefill_select` | +0.001 | 0.000 |
| `prefill_admission` | +0.000 | 0.000 |
| `prefill_scratch_prepare` | +0.002 | +0.003 |
| `prefill_embedding` | −0.005 | +0.015 |
| `prefill_engram_hash` | +0.001 | +0.001 |
| `prefill_engram_prepare` | +0.108 | 0.000 |
| `prefill_engram_wait` | — | +0.068 |
| `prefill_host_prepare` | +0.103 | +0.015 |
| `prefill_host_finish` | −0.003 | −0.009 |
| `prefill_scratch_release` | −0.010 | −0.032 |
| `prefill_attention_hc` | +0.009 | +0.014 |
| `prefill_attention_prepare` | +0.006 | +0.075 |
| `prefill_attention_serial` | +9.999 | +20.151 |
| `prefill_attention_tiles` | +0.127 | −0.137 |
| `prefill_attention_finish_post` | −0.003 | +0.060 |
| `prefill_ffn` | +0.477 | −2.262 |
| `prefill_ffn_hc` | +0.004 | −0.001 |
| `prefill_ffn_router` | +0.008 | +0.007 |
| `prefill_ffn_shared` | +0.071 | +0.101 |
| `prefill_ffn_route` | +0.003 | +0.002 |
| `prefill_ffn_gate_up` | +0.297 | −1.593 |
| `prefill_ffn_mid` | +0.001 | — |
| `prefill_ffn_down` | +0.166 | −0.758 |
| `prefill_ffn_sum` | −0.001 | +0.004 |
| `prefill_chunk_rows` | +10.591 | +17.784 |

## Depth attribution

1. **The selection/index serial stage is the only clear depth scaler.** At
   1,024 rows it grows 70.52→122.81→191.43 ms, or +17.43 then +22.87 ms per
   16K; endpoint slope is +20.15 ms per 16K. The append wall grows 926.34→
   972.53→1,026.58 ms, endpoint +16.71 ms per 16K. At 256 rows the serial
   stage's endpoint slope is +10.00 ms per 16K and wall grows +10.51 ms per
   16K. This locates the remaining deep-index cost. Relative to the old
   report's 3.4× 128K cold per-token penalty this is much flatter, though the
   post-landing attribution is not a paired rerun of that baseline.
2. **Runner-up: shared projection, but it is flat within this probe.** Its
   1,024-row endpoint slope is +0.10 ms per 16K, from 49.27→48.72→49.88 ms;
   the direction reverses between depth steps. Attention prepare is nearly
   tied at +0.08 ms per 16K and is similarly flat. Neither is a credible
   second depth-scaling lever from two reps.

The serial stage is marked around `encode_rows_serial` in
`crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs`; its implementation in
`attention_gpu/query_batch.rs` performs the position-ordered compressor,
index and selection transitions. The shared interval brackets the shared
expert projections and SwiGLU in `crates/qwen-llm/src/deepseek_v41/forward/ffn_gpu.rs`.

The 1,024-row host/encode work remains small: host prepare is 7.18–7.49 ms,
encode is 2.38–2.51 ms, while command commit/wait rises from 893.00 to 993.05
ms. The deep serial interval's +20.15 ms/16K exceeds the net wall slope because
stage intervals overlap and other GPU spans fall slightly with depth; it is
not a removable 20 ms per 16K wall charge.

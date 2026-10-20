# Flash-Next Split Decode Attention

Candidate `abef82a1` (`perf/flashnext-split-qsa`) against its base `57558f5d`.
The released 24/2/256 QSA geometry decodes attention with the value
accumulation split across key ranges once at least 256 keys are selected.
The incumbent kernel ran one threadgroup per query head looping over every
selected key; at the 2,048-key budget that is 24 threadgroups on the whole
GPU and each value row read once per query head. The split kernel keeps the
incumbent logits and global softmax, accumulates values over up to 64 key
ranges per KV head, and merges the partials with the gate. Only the F32
summation order of the value sum changes.

The same kernels were retired on 2026-09-16 when one composed step reached a
maximum logit delta of 1.0000004e-3 against a 1e-3 bound. This packet judges
the path by next-token distribution agreement and persistent-state closeness
instead, then by throughput between binaries.

## Model-free screen (M4 Max, F64 oracle)

`qwen4exp_qsa::tests::split_decode::split_decode_attention_screen`, random
F16 caches and gates, twelve cases over 0-2,051 keys, amplitudes and
poisoned/out-of-range ids. Both routes against an F64 reference:

| keys | incumbent max-abs | split max-abs | incumbent ms | split ms |
|---:|---:|---:|---:|---:|
| 32 | 0 | 7.9e-8 | 0.050 | 0.073 |
| 33 | 8.9e-8 | 4.8e-8 | 0.051 | 0.057 |
| 128 | 8.9e-8 | 3.8e-8 | 0.132 | 0.034 |
| 2,048 | 8.6e-8 | 1.4e-8 | 0.566 | 0.128 |
| 2,051 | 8.9e-8 | 1.3e-8 | 0.490 | 0.116 |

Times are per attention leaf (logits + softmax + value + merge), averaged over
12 repeats in ABBA order; the 128-key row's control spread was 24% under GPU
contention, the 2,051-key row's was 0.01%. The split path is closer to the
oracle than the incumbent at every length and slower only below ~64 keys,
hence the 256-key threshold.

## Real-model witness (M4 Max, UD-Q3_K_XL)

`qwen4exp_runtime::tests::split_decode::product_split_decode_shared_prefix`:
the 2,179-token `natural-ssh` prefix packed, then 32 teacher-forced
continuation tokens per arm from the same checkpoint, incumbent arm forced
through the test probe, guarded top-k and HC up-mix off.

- Next-token distributions: mean KL between arms over 32 steps 1.6e-11 nats
  (max 3.3e-10 at step 23), zero arg-max flips, max logit delta 5.6e-4.
- Persistent state: F32 hyper residual and GDN states within 2e-5, F16 KV
  and index caches within 2e-3 (one half-ulp at those magnitudes); the
  committed prefix is byte-identical in both arms.
- Census: 384 of each split kernel (32 steps x 12 QSA layers), zero fused
  softmax-value dispatches in the candidate arm.
- In-process ABBA, four forwards at depth 2,179: GPU 224.2/224.7 ms incumbent
  versus 189.6/190.8 ms split (-15.3%), wall -14.9%, spread 0.2%. Measured
  while another lane's contracts run shared the GPU under the lease.

## Throughput between binaries

`qwen-bench suite --tg 128 -d 0,2048,8192,16384 --runs 3`, llama-bench
semantics (untimed fill to depth, random tokens, no logits readback), one
process per arm in A B B A order, clean builds of both commits.

### M4 Max, UD-Q3_K_XL (first pass, arm 1 contaminated by a concurrent build)

| depth | base tok/s (A1 / A4) | split tok/s (B2 / B3) | gain |
|---:|---:|---:|---:|
| 0 | 36.17 / 38.86 | 40.39 / 40.16 | n/a: split is not eligible below 256 keys; the 7% A-spread is the contamination |
| 2,048 | 26.05 / 27.54 | 37.00 / 36.16 | +36.5% |
| 8,192 | 26.19 / 27.21 | 33.09 / 35.83 | +29.0% |
| 16,384 | 25.80 / 26.85 | 34.50 / 35.79 | +33.5% |

### M5 Ultra, GSQ-RCO IQ3_S (`tg128 --pp 2048`, both arms clean; metagross)

| depth | base tok/s (A1 / A4) | split tok/s (B2 / B3) | gain | A spread |
|---:|---:|---:|---:|---:|
| 0 | 44.42 / 44.43 | 44.49 / 44.47 | +0.1% (control: split is not eligible below 256 keys) | 0.0% |
| 2,048 | 31.39 / 31.32 | 44.97 / 44.95 | **+43.4%** | 0.2% |
| 8,192 | 31.27 / 30.85 | 43.76 / 44.15 | **+41.5%** | 1.4% |
| 16,384 | 30.57 / 30.26 | 42.54 / 42.25 | **+39.4%** | 1.0% |

pp2048 at each depth is unchanged within its 6-11% control spread (prefill
does not use this kernel). Decode at 16K context now runs within 5% of
depth-0 on the M5 and within ~10% on the M4; the depth penalty was this one
kernel.

Raw rows: `~/qwen-runs/2026-10-20-flashnext/abba/arm*.json` on metagross and
`/tmp/qwen-focus-2026-10-20/abba/arm*.json` locally at the time of writing.
A clean M4 rerun is queued behind another lane's GPU window.

# GLM KDA parallel preparation and TensorOps attention probes

## KDA parallel preparation: ship with rollback

The independent-row KDA preparation candidate is bitwise against serial at the
49K + 2,048-row root cell. The isolated stage median was **31.873 ms vs
80.396 ms (2.522x)**. Its three ordinary root-wall pairs were **2,001.859 ms
vs 1,957.423 ms**, a 2.27% improvement. The 80K append campaign also passed
bitwise logits/state checks: at 256 rows, **455.161 -> 450.124 ms (1.011x)**;
at 1,024 rows, **1,156.773 -> 1,138.031 ms (1.016x)**. Append walls are not
worse, but the gains are small.

The three counted three-agent, 48K-warm, 180-second serve comparisons used the
same release binary, seed, plan, real agent sampling, and normal process
priority. Order was reference/candidate, candidate/reference,
reference/candidate. One extra `reference-1` arm was retained as a spare and
excluded from paired statistics. The p50/p95 columns are means of the three
per-agent summary quantiles; ITL is the harness gap p99.

| Arm | tok/s | append TTFT p50 / p95 | ITL p99 | censored TTFT | thermal max |
|---|---:|---:|---:|---:|---:|
| reference-0 | 85.70 | 0.840 / 5.776 s | 180.30 ms | 0 | 2 |
| candidate-0 | 88.84 | 0.814 / 3.219 s | 163.37 ms | 0 | 2 |
| reference-2 | 80.62 | 0.835 / 6.335 s | 188.91 ms | 1 | 2 |
| candidate-1 | 86.45 | 0.949 / 3.255 s | 170.12 ms | 1 | 2 |
| reference-3 | 89.43 | 0.823 / 3.236 s | 158.41 ms | 0 | 2 |
| candidate-2 | 85.03 | 0.843 / 5.743 s | 180.81 ms | 0 | 2 |

Paired candidate throughput changes were **+3.66%, +7.24%, and -4.93%**;
mean **+1.99%, 95% paired-t CI [-13.54%, +17.52%]** (n=3 arm pairs, df=2;
half-width 15.53 percentage points). The intervals are too wide to resolve a
2% change; paired p50 append TTFT change was **+36.3 ms**, 95% CI
**[-139.9, +212.6 ms]** (mean relative change +4.37%, CI [-16.71%, +25.44%]);
the interval is too wide to rule out regression. Two requests were right
censored at cutoff, one in each arm of the second pair. Paired ITL p99 change
was **-4.44 ms**, 95% CI **[-62.23, +53.35 ms]**. All six counted arms
reported compiler overlap; timing thermal pressure peaked at 2, within the
protocol limit. All six driver and server processes were verified at nice 0.

**Lead decision:** ship default-on; the bitwise-change geomean TPS ratio
**1.01861** clears the **0.98** serve floor, with
`QWEN_GLM53_KDA_PARALLEL_PREP=0` as rollback.

## TensorOps sparse MLA attention: isolated operator probe

The existing selected-latent attention microbenchmark was built in the `iter`
profile and run by itself on M5 Ultra. It used a synthetic 65,536-row F16
K=V cache, 2,051 selected slots, and eight measured repetitions after four
warmups. At the production group size of 64 queries, median GPU times were:

| Route | Median GPU time |
|---|---:|
| Split incumbent | 2.115 ms |
| F32 TensorOps | 1.122 ms |
| F16 TensorOps (production prefill route) | 1.098 ms |

The F16 TensorOps operator is **1.93x faster** than split at 64 queries. At 32
queries it is 1.35x faster (0.768 vs 1.033 ms); at 16 queries it is slower
(0.622 vs 0.502 ms), consistent with the production route's 32-query lower
bound. This is an operator-only result, not a full-chunk wall or quality gate;
no production code was changed.

Raw M5 outputs are under
`~/qwen-runs/2026-11-12-glm-prefill/kda-parallel-prep/`; the serve arms are in
`serve-180/` and the operator probe is in `tensorops-mla-probe/`.

## Landing witness

Landed tree `ad590968`, `iter` profile, one binary (sha256
`5e2a6c311c689a8165f6eb1af78473a82c4cfc602111127ee0f9d282b154b0a6`), M5 Ultra,
niceness 0, under the canonical Metal lease. The probe forces the arm per call
so the model load, prefix and host environment match; the serial arm is the
`QWEN_GLM53_KDA_PARALLEL_PREP=0` path. Three interleaved pairs:

| Cell | Pair | Rollback (`=0`) ms | Default ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 80K + 1,024 append | 1 | 1,167.904 | 1,151.963 | 1.0138 |
| 80K + 1,024 append | 2 | 1,168.187 | 1,155.360 | 1.0111 |
| 80K + 1,024 append | 3 | 1,171.454 | 1,157.543 | 1.0120 |
| 48K + 2,048 root | 1 | 2,003.321 | 1,964.355 | 1.0198 |
| 48K + 2,048 root | 2 | 2,023.076 | 1,980.852 | 1.0213 |
| 48K + 2,048 root | 3 | 2,025.690 | 1,994.152 | 1.0158 |

All runs reported bitwise identical logits and state between arms. Three
additional processes ran with `QWEN_GLM53_KDA_PARALLEL_PREP=0` in the
environment and agreed; thermal pressure stayed at or below 2. Raw:
`metagross:~/qwen-runs/2026-11-13-kda-landing/`.

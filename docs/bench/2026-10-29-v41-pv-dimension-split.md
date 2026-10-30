# Tensor64 PV Dimension Split: Replay Win, No Model Win

## Decision

Keep the production PV kernel and grid unchanged. Splitting its four D128
output chunks into four independent groups is bitwise exact and faster in the
resident replay, but does not establish an end-to-end 8K model gain. Retain the
ignored probe and experiment history; do not select it by default or rerun the
same model campaign in search of a favorable result.

This does not close attention dataflow work. Eliminating intermediate passes
and traffic is a different experiment from changing PV group granularity.

## Resident Kernel Result

Source `2d6462bd`, based on the landed Q-register cache. Apple M5 Ultra,
256 GiB, macOS 27.0.1/SDK 27.0, Rust 1.99.1. Metal compilation uses `-O3`;
the host uses the optimized test profile (opt-level 3, thin LTO, debug assertions).
Results here use GPU timestamps, not host wall time or whole-model throughput.
The test binary SHA256 is
`a39d1a92cdaa9576e31e2e89903d07588766b7e73cd5064173b1b4fd42211e58`.

The existing PV kernel computes four D128 output chunks serially in one
threadgroup. `pv_tensor64_dim1` gives each chunk its own group: four groups
per query, one accumulator each. H64/K32 tensor operations, increasing-key
MAC order, addressing, invalid-row handling and output layout remain the same.
Each group uses the same 12 KiB threadgroup scratch. Total logical P/V traffic
is unchanged; the experiment changes work granularity and parallelism, not
operand precision or tensor storage.

The resident comparison uses K640/N512/H64/D512, dense and ragged visibility,
three ABBA repeats and 32 launches per command. Both arms warm continuously
before retained samples. Poisoned, bitwise output/status checks run before
and after timing, with no large CPU readbacks between timing samples.

| Visibility | Scope | Existing GPU us | Split GPU us | Speedup |
| --- | --- | ---: | ---: | ---: |
| Dense | PV only | 708.219 | 615.593 | 1.1505x |
| Ragged | PV only | 632.605 | 506.545 | 1.2489x |
| Dense | Complete sink | 1938.234 | 1826.364 | 1.0613x |
| Ragged | Complete sink | 1660.719 | 1545.949 | 1.0742x |

The complete sink includes Q-register score, softmax, PV and normalization.
Every output bit matches in both cases and scopes: 16,777,216 output elements
per case, zero nonfinite outputs and status zero. Guards pass. An adversarial
code review finds no indexing, output-coverage, barrier or arithmetic-order
defect; it notes extra group scheduling pressure as a performance tradeoff.

The preceding `f1e0121d` probe uses four launches per command and compares all
output elements on CPU after every sample. Its PV-only medians appear about
1.5x faster but individual samples scatter widely. Those samples remain
preserved, not promoted as the PV result. The followup changes only cadence:
longer GPU batches, continuous warmup and readbacks outside the timing loop.
The kernel is unchanged.

## Actual Model Result

Experimental integration `6210b9c4` selects the split kernel on the existing
H64/D512 tensor64 path. Its `QWEN_V41_ATTN_PV_SPLIT=0` control restores the
previous kernel and one-group grid. That production switch does not land in
maintained main; the experiment commit remains in history for reproduction.

The existing 8K Full/CED numerical comparison passes on the 40-layer
DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XS artifact with IQ4_NL Engram: 257
full-vocabulary vectors per arm, 129280 elements per vector, 256 teacher-forced
steps and 56 causal regions at each position. All 257 saved observations also
equal the prior Q-cache report at `ed46c40f`, with identical artifact and selected
input identities. The release test executable SHA256 is
`15158d982ee8b5b47f500abe195b861dbeb47bf5d5bc2064306acafae86e5d06`.

Normal-release `qwen-bench suite` runs use `--pp 8192 --depth 0 --runs 2
--seed 1`, with Q-cache on in both arms. Each policy has contiguous ABBA and
BAAB blocks, two retained samples per process, eight samples per arm: 32 retained
samples overall. CED runs before Full, without a programmed cooldown. Timing
uses deterministic synthetic tokens, not the numerical witness. Native wall
time includes prefill and final logits, excluding model loading.

| Policy and block | Existing mean s | Split mean s | Throughput change |
| --- | ---: | ---: | ---: |
| CED ABBA | 4.2635 | 4.4434 | -4.05% |
| CED BAAB | 4.6823 | 4.6171 | +1.41% |
| CED, all samples | 4.4729 | 4.5303 | -1.27% |
| Full ABBA | 9.2882 | 9.3886 | -1.07% |
| Full BAAB | 9.1473 | 9.0821 | +0.72% |
| Full, all samples | 9.2177 | 9.2354 | -0.19% |

All samples stay included. Mixed block signs and substantial machine-state
variation prevent a precise universal regression claim, but there is no
demonstrated model speedup to promote. Ordinary Full uses P0 mapping and
ordinary CED uses P4; comparisons stay within each policy. Actual CED records
report `shrinking_ced+final_logits_copy`, not a Full fallback. The release
benchmark executable SHA256 is
`7f9082ba6ce75d858e72b8826bd131b750a563cb0eb9ce5836c836bb070732ba`.

The earlier two-accumulator/two-group variant remains a separate measured
negative. Neither result implies a bandwidth ceiling, closes other dataflow
mechanisms, or predicts performance of a late-context append.

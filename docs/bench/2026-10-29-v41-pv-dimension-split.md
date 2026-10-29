# Tensor64 PV Dimension Split

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
defect; it notes the extra group scheduling pressure as a performance tradeoff.

## Earlier Exploratory Samples

The preceding `f1e0121d` probe uses four launches per command and compares all
output elements on CPU after every sample. Its PV-only medians appear about
1.5x faster but individual samples scatter widely. Those samples are preserved,
not promoted as the PV result. The followup changes only measurement cadence:
longer GPU batches, continuous warmup and readbacks outside the timing loop.
The kernel is unchanged. Its original complete-sink medians are 1.0703x and
1.0866x; the longer-batch results are the primary kernel evidence here.

The earlier two-accumulator/two-group variant remains a separate measured
negative; it is not rerun or selected by this change.

## Integration Experiment

The experimental prefill path selects the one-chunk PV on the existing
H64/D512 tensor64 route. `QWEN_V41_ATTN_PV_SPLIT=0` restores the old kernel
and one-group grid for same-binary A/B; Q-register scoring stays on in both arms.
Other attention tiers and all tensor buffers/chunk policies remain unchanged.

Real-model numerical comparison and wall-time measurement follow separately.
This resident tile result alone is not an end-to-end speedup claim.

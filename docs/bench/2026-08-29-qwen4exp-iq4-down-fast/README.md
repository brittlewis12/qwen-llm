# Flash-Next Fast IQ4_NL Routed Down

Decision: **GO**. Promote selected-expert IQ4_NL row reuse by default, with
`QWEN4EXP_MOE_IQ4_DOWN_FAST=0` as rollback.

## Candidate

Checkpoint `de3d547fe260af4e8bb3b17dd7ece796667d1909` adapts the proven dense
IQ4_NL `NR0=2`, `NSG=2` body to singleton selected-expert addressing. Each
activation load serves two output rows. Expert output remains materialized and
the existing slot-ordered weighted sum remains a separate kernel.

Production-width and 512-expert odd-tail CPU-oracle gates pass at cosine `1.0`
and relative maximum error at most `7.15e-7`. The fast wrapper additionally
requires 16-byte input alignment; invalid experts and odd output tails are
explicitly covered. Source-only adversarial review passed after converging the
threadgroup barrier and assigning LUT initialization to one SIMDgroup.

The dirty model-free screen encoded 43 production-shape down dispatches per
command and measured median `3.414917 -> 1.437750 ms`, saving `1.977167 ms`.
This cleared the preimplementation leaf gate before model integration; only the
committed whole-command packet below carries promotion authority.

## Protocol

- Device: Apple M4 Max with unified memory.
- Model: UD-Q3_K_XL from `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`. The three shard SHA-256 values
  are `f2ef4328929d8b8c8930e2856eef52128dd4ce3425302f04bc3c657431cc4c49`,
  `7d230e7c9421d868b89eebaf23033af0ea1a4e046956df00fb156814fb62346e`, and
  `21d4f90f9cd7b7c3a1582667c20cb22f7b03de895b88a23bb20aaeaa44f2c199`.
- Storage: external PCIe SSD. Run 1 incurred a roughly 40.7-second prefill
  first-touch interval; this packet does not claim uniformly warm prefill.
- Source: release binary rebuilt from committed default-off checkpoint
  `de3d547`; the only worktree additions were unrelated untracked docs.
- Workload: no-thinking prompt `Write the numbers from 1 to 100, separated by
  commas.`, 28 prompt tokens, 32 generated tokens, and 31 decode transitions.
- Order: `B-C-C-B` repeated three times with five seconds between processes.
- Arms: B sets `QWEN4EXP_MOE_IQ4_DOWN_FAST=0`; C sets it to `1`. Both set
  `QWEN_MATVEC_F32_LCPP_R2=1`; no other controls differ.
- Primary endpoint: complete decode command GPU time divided by 31 transitions.
- KEEP: median saving at least `0.5 ms/transition`, positive in every balanced
  block, 31/31 GPU samples, and one generated-output digest.

## Results

| Run | Arm | GPU ms/transition | Generation wall (ms) |
|---:|:---|---:|---:|
| 1 | B | 46.6827 | 1589.7 |
| 2 | C | 45.3124 | 1483.3 |
| 3 | C | 45.3204 | 1484.1 |
| 4 | B | 46.7096 | 1526.0 |
| 5 | B | 46.7199 | 1524.9 |
| 6 | C | 45.4844 | 1489.5 |
| 7 | C | 45.3130 | 1480.2 |
| 8 | B | 46.7129 | 1523.7 |
| 9 | B | 48.3248 | 1574.3 |
| 10 | C | 47.0051 | 1530.0 |
| 11 | C | 47.0195 | 1530.1 |
| 12 | B | 48.3807 | 1571.5 |

Median command GPU moved `46.716405 -> 45.402361 ms/transition`, saving
`1.314044 ms` or `2.81%`. Balanced-block mean savings were `1.379781`,
`1.317696`, and `1.340435 ms`; all directions agree despite a shared slower
phase in block three. All runs supplied 31/31 GPU intervals.

Every stdout has SHA-256
`a43ab8b653ea3e75ed0933b22495869eccd123b3897b80b8a626cb7064d600d9`.
The candidate clears every gate. Packed multi-token MoE is unchanged, so this
packet makes no prefill claim.

Default-on and explicit rollback singleton topology gates pass, as does the
nonzero full-MoE CPU-oracle gate. Source-only adversarial review closed the two
initial public-wrapper blockers.

# Qwen MoE Ragged B16 128-Token Screen

Date: 2026-08-12
Status: automatic admission KILL at the current executor

## Question

Does extending the already exact A3B Q4 ragged B16 fixture from 64 to 128
requested output tokens amortize private prefill enough to clear the frozen
`1.10x` execution-wall gate against resident B2?

## Method

The measured release binary has SHA-256
`c18b099629007e75f54790a69f5adba9d42d01b91b19c828314c77cb8c039f6a`
and embeds commit `4238acdd182552dffeac743744848b5e62a71eab`, dirty bit `1`,
and source-state digest
`git-source-sha256-v2:0c155730c47ffb4e99a0caa0cd37c69d3e04dfe62c247b2ed51010b90b1af7b4`.
The dirty bit includes two pre-existing user-owned untracked documents; they
were not moved or modified for this packet. ABBA uses this same binary in all
four arms. This packet therefore has exact same-binary authority, not a
portable clean-build claim.
The authenticated fixture SHA-256 is
`20e90d00d0a51ec030897f50d61c49c49dac470c9d4979f978604e1792d89bf2`.

All runs use Qwen3.5 35B-A3B Q4_K_M, greedy output, fixed prefill chunk 512,
serialized GPU execution, and no prefix fanout or file root on the B2 control.
Order is B16/B2 then B2/B16. Execution wall sums model prefill/prepare and
decode telemetry, excluding model load.

## Results

| Arm | Process wall | Load | Prefill/prepare | Decode | Execution |
|---|---:|---:|---:|---:|---:|
| B16 A | 17.86 s | 2,339.6 ms | 1,626.405 ms | 13,633.639 ms | 15,260.044 ms |
| B2 A | 19.45 s | 2,522.0 ms | 1,642.115 ms | 15,035.352 ms | 16,677.467 ms |
| B2 B | 19.43 s | 2,538.5 ms | 1,648.750 ms | 14,992.068 ms | 16,640.817 ms |
| B16 B | 18.08 s | 2,516.3 ms | 1,635.920 ms | 13,668.936 ms | 15,304.856 ms |

Median execution is `16,659.142 / 15,282.450 = 1.09008x`. The two paired
ratios are `1.09288x` and `1.08729x`. All four complete JSONL outputs are
byte-identical with SHA-256
`481e5858aa5041172a9fd9a1e687a09dda6fe9311d901bf1f1196f2e7b27dfa2`,
and every process reports zero swaps.

Decode alone is `30,027.420 / 27,302.575 = 1.09980x` across both repeats. That
lands 0.02% below the product gate before private prefill is charged. A
256-token run has too little expected decision value to justify another product
packet at the
current executor.

## Decision

Keep automatic Qwen MoE ragged admission closed. Retain explicit
`QWEN_FIXED_COHORT_RAGGED_PROMPTS=1` as an exact experimental capability. Do
not spend another product packet on longer output at the current executor.
Reopen only after B16 decode materially improves relative to B2, replacement
prefill overlaps active decode, or a changed executor has an independently
measured whole-request ceiling above `1.10x`.

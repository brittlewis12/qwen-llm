# High-count grouped expert down

Archive update: the complete validated experiment remains at `950a2db3`
(also present in `d7a46ba7`). Its live helpers, capture hooks, tests and research
shader were subsequently removed to avoid permanent build/maintenance cost.
Use a separate worktree at `950a2db3` for reproduction; `restore-16k.patch`
applies there. All evidence in this directory remains retained.

Experimental M64/N64/K32 Half-operand down tile, compared with the current
production path (including IQ3_S SmallCounts at N512). No production selection
change yet. Native storage remains compressed; no persistent GPU scratch.

The focused packet captures block 4 post-SwiGLU inputs and real routes at actual
N2048 and N512 workspace widths. It compares the composed N32/N64 tile with
production, and separately measures the incumbent N32 with a captured maximum
count grid cap. That CPU-known cap is diagnostic only. Whole-model warm ABBA
uses fresh sessions, no capture/tracing during timing, and three common
teacher-forced continuation tokens.

Initial GPU tests exposed a cancellation-sensitive per-scalar relative oracle
criterion (IQ3_S slot 71: CPU -0.0040487647, absolute error 2.6822e-7).
Candidate, incumbent and capped incumbent are bit-identical. The oracle now
checks a derived F32 dot-product roundoff bound, retains the absolute check,
and measures relative RMS over all written outputs instead of a scalar near
zero. Seven targeted tests pass with Metal validation enabled.

## Outcome: Parked, Production Unchanged

All runs use the M4 Max 128 GiB, Fast Half, compressed resident weights,
ordinary internal GPU lease, and actual N2048/N512 workspaces. Base is
`fcb0c7c0` plus this experiment. No environment interventions. Ratios are
incumbent time / candidate time; greater than one favors the candidate.

| Packet | Leaf N2048 | Whole N2048 | Leaf N512 | Whole N512 |
| --- | ---: | ---: | ---: | ---: |
| `ud-abba.jsonl`, 16 KiB | 0.8718 | 1.0109 | 0.9227 | 0.9911 |
| `maya-abba.jsonl`, 16 KiB | 1.0728 | 1.0011 | 0.8455 | 0.9954 |
| `maya-8k-abba.jsonl`, 8 KiB | 0.9789 | 0.9979 | 0.8606 | 1.0044 |

Whole N2048 mean milliseconds A/B: UD 9545.050/9441.874, Maya
9388.305/9378.368, Maya 8 KiB 10674.091/10696.662. Compare within each
interleaved packet, not absolute timings across runs. Two measured samples per
arm are a focused screen, not a precise throughput estimate. UD's first
whole-model baseline range envelops both candidate samples. These results do
not justify production selection.

The retained 8 KiB shader sequentially drains two N32 result panels through one
shared-memory overlay. This tests avoidable threadgroup reservation, not
register pressure: the doubled accumulator state remains. `restore-16k.patch`
recreates the original shader/host reservation from the retained version.
At the experimental revision Rust helpers are test-only and the shader lives
in the research metallib, lazily resolved by existing pipeline lookup. Production dispatch,
admission, persistent memory and arithmetic remain unchanged.

All recorded slot-output, weighted-output and endpoint/three-step continuation
hashes match the respective A1 reference in all three packets, including
warmups. Recorded numerical differences are zero and outputs are finite.
That numerical success does not imply a performance improvement.

UD block 4 at N2048 has median count 51, maximum 262, and 233 experts above
32 slots. N64 covers 92.6% of slots and reduces decoded weight panels from
651 to 390, but increases padded token work from 20832 to 23232. Counts 65-96
account for most of that padding. At N512 only eight buckets exceed 32; the
candidate also replaces production SmallCounts with generic N32 for small
buckets. Do not attribute all of the N512 regression to N64 alone.

The capped-N32 control retains the actual bucket stride and uses a CPU-known
captured maximum solely in diagnostic replay. Results are mixed; this does not
justify introducing production CPU readback or indirect-dispatch machinery.
One bank's leaf duration is not an aggregate cost for all 42 routed blocks.

## Deep Decode Reorientation

`maya-decode-depth.json` uses the existing `decode_stage_attribution` test with
`GLM53_GGUF` pointing to Maya and `GLM53_DECODE_DEPTHS=8192,32768`.
It uses production dispatch (the experimental override is inactive), a 512-row
prefix workspace, repeated qualification text, four warmups, twelve ordinary
steps and twelve subsequent profiled steps at each depth. These are different
teacher-forced positions, not paired identical-token trials. This packet is
attribution, not a model-quality or serving-prefill throughput comparison.

| Stage, ms/token | 8192 | 32768 |
| --- | ---: | ---: |
| Ordinary wall | 34.44 | 32.85 |
| Routed experts, profiled | 10.779 | 9.709 |
| KDA, profiled | 10.049 | 9.686 |
| Sparse attention, profiled | 1.076 | 1.042 |
| Sparse selection, profiled | 0.653 | 0.922 |
| Sparse scores, profiled | 0.092 | 0.170 |

There is no observed large depth penalty in these cells. Selection and scoring
grow by about 0.35 ms; selected attention stays near 1 ms. This demotes a GLM
selected-attention rewrite behind expert gate/up prefill and KDA preparation.
It says nothing about Flash-Next's distinct selected-attention costs.

## Validation And Follow-Up

- Final research-isolated source: 55 grouped-MoE and 86 GLM tests passed with
  Metal validation; 50 GLM fixture/diagnostic tests ignored in the routine run.
- Three full model ABBA packets and one deep attribution packet ran separately.
- All-target release workspace check passed; existing unrelated unused
  `prompt_prefix_len` warning remains.
- Next: capture normalized input for a real IQ2_S gate/up replay. Test reduced
  M32/N32 state, not another wider token tile; preserve existing small buckets.
  KDA preparation history in registers is the next bounded alternative.

Reflection: mean bucket occupancy was enough to motivate an experiment, not to
predict a win. Retain actual count distributions, padding, and whole conversion;
stop after the causal resource follow-up instead of sweeping thresholds.

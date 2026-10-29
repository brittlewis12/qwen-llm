# Tensor64 Q Register Cache

## Resident Tile Result

Source `3189479f`, based on maintained `c1ba3e58`. Apple M5 Ultra, 256 GiB,
macOS 27.0.1/SDK 27.0, Rust 1.99.1. The frozen test executable SHA256 is
`ac12f2a312078f8a09f2e40edb70d8ea89f49dd00d593efdcf4b8180da695652`.

The existing Q-half score reloads its query for every 32-key panel. The new
`score_tensor64_qregister` loads each lane's 64 half4 vectors once and keeps
them live across panels. The four D128 multiply/MAC operations, key gathers,
scaling, score layout, softmax, PV and normalization are unchanged. This adds
thread-local live state, not another GPU backing buffer.

The direct comparison uses resident synthetic K640/N512/H64/D512 tiles, dense
and ragged visibility, three ABBA repeats and four launches per command. GPU
timestamps exclude input construction, allocation, pipeline compilation and
readback. Compilation holds no GPU lease; only GPU execution uses the queue.

| Visibility | Scope | Q-half GPU ms | Q-cache GPU ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| Dense | Score | 1.623 | 0.981 | 1.654x |
| Ragged | Score | 1.300 | 0.801 | 1.623x |
| Dense | Complete sink | 2.583 | 1.935 | 1.335x |
| Ragged | Complete sink | 2.156 | 1.664 | 1.296x |

All 37,715,968 active score elements and 33,554,432 output elements compare
bit-for-bit. Both cases have zero nonfinite outputs. Guards and status checks
pass. This is a resident tile result, not an end-to-end prefill speedup.

The adversarial source review finds no kernel indexing, stride, arithmetic-order
or barrier bug. It identifies a missing-store blind spot in the raw-score test:
the candidate could inherit an unwritten baseline element. The followup poisons
scores and outputs before untimed candidate validation; its new execution is
reported separately rather than attributed to this frozen receipt.

## Rejected PV Experiment

The same executable tests `pv_tensor64_dim2`: two output-dimension chunks per
group, reusing each probability panel across two accumulators. It remains exact
but loses. PV time rises from 0.764 to 0.832 ms for dense visibility and from
0.681 to 0.757 ms for ragged visibility. Complete sink time also rises, by about
3.3% and 2.6%. Keep the existing PV kernel; do not promote or rerun this variant
unchanged in search of a favorable result. Register pressure is a plausible
cause, not a measured spill diagnosis.

## Integration Experiment

The experimental branch selects Q-cache for the existing tensor64 Q-half path.
`QWEN_V41_ATTN_QREGISTER=0` restores the prior Q-half score for same-binary A/B;
disabling Q-half still restores the F32-query route. No buffer, arithmetic,
chunk-width or PV policy change is needed.

## Real Model Result

Integrated source `ed46c40f` passes the existing 8K Full/CED comparison on the
40-layer DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XS artifact with IQ4_NL Engram.
It compares 257 full-vocabulary vectors per arm, 129280 elements per vector,
256 teacher-forced steps and 56 causal-state regions at each position. Every
observation also equals the prior Q-half report at `da5d404e`, with identical
artifact and selected input identities. This is not just candidate Full versus
candidate CED: the saved prior-kernel logits and state hashes agree too.

The test binary SHA256 is
`22225477089c11d1807ef99049281b5f2c0cdc597c159b00a3b8907afbff5793`.
The strengthened, poisoned score/output check also passes on this source, with
all active scores and outputs exact and no nonfinite output lanes.

### Native Wall Time

The same `qwen-bench` executable runs ordinary `suite` with `--pp 8192
--depth 0 --runs 2 --seed 1`. Off sets `QWEN_V41_ATTN_QREGISTER=0`; on uses
the compiled default. Each policy has an off/on/on/off block and a reverse
on/off/off/on block, two retained samples per process: eight samples per arm,
32 retained samples overall. Policy order also reverses between blocks. No
debug validation is active. These are native prefill-plus-final-logit wall
times, excluding model load; the timing prompt is deterministic synthetic
input, distinct from the numerical witness. All samples remain included.

| Policy and block | Q-half mean s | Q-cache mean s | Throughput gain |
| --- | ---: | ---: | ---: |
| Full ABBA | 7.4199 | 6.8443 | +8.41% |
| Full BAAB | 7.9813 | 7.8471 | +1.71% |
| CED ABBA | 5.3294 | 5.2299 | +1.90% |
| CED BAAB | 4.3079 | 4.1278 | +4.36% |
| Full, all samples | 7.7006 | 7.3457 | **+4.83%** |
| CED, all samples | 4.8187 | 4.6788 | **+2.99%** |

The aggregate corresponds to 4.61% less Full wall time and 2.90% less CED wall
time. Throughput is 8192 divided by mean wall seconds, not the mean of rounded
rates. Ordinary Full uses P0 mapping and ordinary CED uses P4; compare off/on
within each policy, not their absolute times to a P0 paired run. Actual CED
records report `shrinking_ced+final_logits_copy`, not a Full fallback.

Machine-state drift is substantial, especially in Full: the final BAAB on
samples are 8.79/8.96 s versus 6.82/6.82 s initially. Do not present the +8.41%
block as a stable universal gain. Both orders favor Q-cache, and the CED BAAB
block is comparatively steady at 4.31 s off versus 4.13 s on. There is no
programmed cooldown within a block; the two campaigns have an approximately
160-second gap. These are one-machine/workload observations, not confidence
bounds or a physical roofline measurement.

The benchmark executable SHA256 is
`d090946bf987db283f713096a8548d82dbb07f9440bd51b82932390a8961a413`.
Encoder width stays 4096, decoder width 2048, query tiles 512, group depth two,
and pipeline/prefetch remain on. No additional tensor buffer is needed.

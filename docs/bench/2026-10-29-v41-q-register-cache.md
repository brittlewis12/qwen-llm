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

Actual Full/CED numerical comparison and end-to-end timing remain to be measured.
The tile result alone does not promote the candidate into maintained main.

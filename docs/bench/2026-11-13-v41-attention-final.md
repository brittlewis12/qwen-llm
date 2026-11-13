# V4.1 bitwise attention bundle: final served gate

**PASS under the frozen served non-regression rule.** Short index and wide PV
remain default-on, with independent `QWEN_V41_MASKED_REINDEX_SHORT=0` and
`QWEN_V41_ATTN_PV_WIDE=0` rollbacks. No arithmetic or memory-allocation change
is made in this final stage; descriptions now record qualification.

Only pair 3 B ran in this stage. Pair 1 AB and pair 2 BA, and pair 3 A,
stand unchanged. Every arm starts with the prescribed paired 4K model warmup,
both GPU fixtures, Engram purge to zero pages, fresh three-agent greedy
server at actual nice 0, and 60s continuous thermal <=1 before load and
after roots. Thermal 2 remains admissible; 3 aborts. Nothing is filtered for
load, compiler activity or disk counters. All completed arms pass controls.

## Frozen result

| Metric | Three-pair geomean | Required |
|---|---:|---:|
| root | 0.993822521x | 0.98x |
| tps | 1.004618096x | 0.98x |
| short_prefill_ratio | 1.031850574x | 1.00x |
| short_ttft_ratio | 1.007496626x | 1.00x |
| 1k_prefill_ratio | 1.015682641x | 1.00x |
| 1k_ttft_ratio | 1.017890587x | 1.00x |

The short and 1K bins each retain 6/6 and 3/3 matched/observed calls per
pair. Counts, nullable TTFTs, all roots and raw calls are retained. Ratios
for latency are main/bundle; TPS is bundle/main.

| Pair/order | Root prefill | TPS main / bundle | Short prefill / TTFT | 1K prefill / TTFT |
|---|---:|---:|---:|---:|
| 1 AB | 1.166676124x | 87.350000 / 88.316667 | 1.053300631x / 1.032869875x | 1.045456835x / 1.058348958x |
| 2 BA | 0.902509021x | 87.083333 / 88.100000 | 1.018921319x / 0.983220259x | 0.995344111x / 0.987286313x |
| 3 AB | 0.932233362x | 87.600000 / 86.833333 | 1.023664015x / 1.007011364x | 1.006919454x / 1.009325606x |

## Process reads

Bytes are maximum live process counters sampled at 0.5s, not Engram-only
reads. Warmup is separate from the measured server. Every purge reached zero.

| Pair | Arm | Server read bytes | Paired 4K warmup read bytes |
|---|---|---:|---:|
| 1 | a | 52267646976 | 159335890944 |
| 1 | b | 33156030464 | 496418816 |
| 2 | b | 34273374208 | 159334612992 |
| 2 | a | 189054726144 | 159337021440 |
| 3 | a | 188321980416 | 157310173184 |
| 3 | b | 33103966208 | 157293551616 |

## Native root remains unresolved

The loaded-host packet is **unresolved noise in both directions, not a
pass**: retained BA/AB/BA ratios 0.931764/1.710896/0.862674x, mean main
47.8270s and bundle 44.9120s, versus standing roughly 21.5s. Its mean point
ratio 1.064904x is not an isolated speed claim or an acceptance gate here.
All eight endpoints are exact; that does not resolve timing noise.

The independent quiet-host wide-PV native root was **1.012173x**,
20.755584s main / 20.505967s candidate, 3/3 positive and all eight full
signatures bitwise, iter/nice 10. The quiet append was 1.01890x.
[Original native evidence](2026-11-12-v41-prefill/README.md#wide-pv-native-wall--both-cells-pass).
Both native packets are cited without reconciling their host conditions.
The final pass/fail above is exactly the lead-authorized frozen served rule.

## Pins, composition and retained records

Both fresh arms use the archived source/profile-pinned `67b3ebaf4` program.
All nine metallibs are byte-identical to the original measured program;
original, prior rebuilt and current executable hash pairs are recorded in
[the source-pin note](2026-11-13-v41-attention-close.md#rebuild-and-bitwise-witnesses).
The legacy first pair's plan/launch files supply its older summary's missing
metadata; no fields or timings are fabricated.

The attention branch is composed on main `8b765391e`; the Engram host cache
and GLM/optional diagnostic changes are independent of the attention kernel
bodies. Cooperative experts is on a separate branch and remains default-off.
Final contracts and ENV generation are recorded in this stage's raw logs.

Raw: `metagross:~/qwen-runs/2026-11-12-v41-prefill/attention-close/`
(all six arms and independent result); final-arm controller under sibling
`attention-final/`. Gate is closed; archives may be deleted after records and
checks finish. No push or merge.

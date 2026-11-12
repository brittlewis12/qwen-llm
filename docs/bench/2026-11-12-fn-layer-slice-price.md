# Flash-Next layer-slice price and implementation

Native pricing supports a layer-slice implementation, and **60/60 paired
regressions plus ten cancel/replay cases pass**. The sustained serve gate is
inconclusive on throughput, so production remains **default off**, opt-in
`QWEN_SERVE_FN_LAYER_SLICE_MS=200`, minimum predicted append wall 500 ms.
See the [six-arm serve result](2026-11-12-fn-layer-slice-serve.md).

## Incumbent and component price

Fresh three-agent release baseline at `abc3ba1a3`, GSQ-RCO IQ3_S, 49152 warm
tokens per agent, release-default sampling and 180 s: **127.111 tok/s**, ITL
p50/p99/max **19.880/25.788/1885.943 ms**. All measured seconds were at pressure 1.
Both settling intervals required continuous 60 s at ≤1; warm-up/window allowed
2 and aborted at 3. The maximum was common to two peers during a 5479-token
append after position 50486, with TTFT 4586.9 ms. This clears the price gate.

Beyond the QSA dense-to-selected frontier, the incumbent uses 2048-row commands:
that append is 2048 + 2048 + 1383 rows. `fn_append_price` (perf-ablations)
measures those unchanged native commands, including CPU encode, completed GPU,
wait, wall and process disk-read bytes, from a 48K immutable checkpoint.

| rows | wall ms | GPU ms | CPU encode ms | commands |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 230–238 | 227–234 | 1.8–1.9 | 1 |
| 342 | 285–287 | 281–282 | 2.9–4.2 | 1 |
| 1383 | 805–810 | 797–798 | 6–11 | 1 |
| 1798 | 1018–1020 | 1010–1012 | 7–8 | 1 |
| 2048 | 1063–1068 | 1054–1055 | 8–9 | 1 |
| 5479 | 2941–2954 | 2909–2912 | 26–41 | 3 |

These first/repeated appends sampled pressure 0 and zero process disk reads.
The large-command wall is GPU-dominated, unlike the preparation/I/O tail seen
in the V4.1 live packet. This justified the native experiment.

## Exact dependency boundaries

Flash-Next has a fused PLE stage in layers 0/1, then 46 post-PLE layers with
private GDN recurrence and QSA index/KV state. The first two layers remain
indivisible. A command stages inputs and records checkpoint lineage once,
parks its existing HC hyper-residual/bridge scratch, and keeps pending length
and snapshot/decode gating across completed layer commands. The final slice
publishes only through the original validation of every child state. No extra
GPU activation allocation or per-layer stack is introduced; parked buffers
were already allocated per runner. A 2048-row token identity copy adds 8 KiB
of host storage while that command is pending.

Every submitted command drains before yielding, error disposal or unwind.
Cancelled partial state cannot be published to a checkpoint or reuse pool;
replay uses a fresh runner and a prior immutable checkpoint. Original ladder
and transcript capture points remain at validated row-command boundaries.
The original dense/selected frontier and scalar tails are preserved.

`fn_layer_slice_probe` at `a4fd2871a` passes **40 warm + 20 cold pairs**:
32/342/1383/1798/2048 rows, 1/2/4/8 cuts. Its oracle checks complete finite
logit bits, snapshot payload plus PLE-history digests, post-append decode, and
two peers' logits/state. All ten cancel/dispose/replay cases are exact.
Warm sampled pressure is 0–1; cold is 0. Mean equal-round GPU-duration ratios
are **0.99993 warm / 0.99996 cold**, individual ranges 0.98450–1.00874 and
0.98616–1.01329. Warm 2048-row worst peer gaps fall 1122/1140 → 170/170 ms;
cold falls 918 → 156 ms. These are native observations, not a live stall bound.

## Selective serve policy

The adapter uses the existing shared Driver, with a scoped QoS guard. Only
ready peers decode between slices; the appender remains exclusive until final
validation. Driver supplies one peer round per yield, suppresses duplicate
decode-percentage credit, and subsequently rejoins group feedback. Its existing
cancellation and unequal-peer-geometry fallback tests cover this handoff.

The prediction sums **225 + 0.42 × rows ms** over the append's original packed
commands. Scalar commands are excluded: they already yield per token and retain
the incumbent callback and decode credit. Slicing starts only with ready peers,
a predicted append above 500 ms, packed scratch and the probed 48-layer model.
Unprobed layer counts retain incumbent execution. A nominal 200 ms target chooses
the next tested 1/2/4/8 partition; no row boundary is invented.

Single-command 32/342-row appends stay unsliced. Live body/header cuts can pay
two floors and qualify at much smaller row counts. The coarse model has false
positives near 500 ms: a live 245-row, two-command append was predicted at
553 ms but measured 464 ms unsliced. Its slicing and extra peer service are
included in the reported TTFT distribution. No policy was retuned mid-packet.

MS=0 restores original native calls, row/frontier planning and decode credit.
Complete/partial prefill accounting is common to both arms, charges active
work and excludes peer rounds, including cancellation before the next callback.
The served maximum remains above the native bound during adjacent appends;
that limitation is documented in the serve note. No workaround was added.

## Evidence

Raw baseline `fn-price`, component `fn-append-price.jsonl`, native
`fn-layer-warm.jsonl` / `fn-layer-cold.jsonl`, and serve `fn-serve/<arm>` are under
`metagross:~/qwen-runs/2026-11-12-v41-layer-slice/protocol-v2/`. The two examples require
`perf-ablations`. The frozen adapter passes 698 serve tests and repository
contracts, with no new ignored tests. V4.1 remains separately default off after
its [inconclusive sustained packet](2026-11-12-v41-layer-slice-sustained.md).

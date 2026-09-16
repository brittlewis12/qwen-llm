# Parallel Singleton Routing: Native Research PASS

Later guarded actual-product and CLI qualification passes; current delivery and
41.7428% GPU/37.0533% executor-wall result are in `TOPK-PRODUCT.md`. Research
results and the finite-only restrictions below remain historical evidence.

## Rechart

Guarded selector delivery is now the highest-leverage next step. The native
experiment establishes a gain across the existing singleton model path; do not
gate delivery behind more IQ3/IQ4 expert profiling or another HC/attention sweep.
Product routing/defaults and the existing parallel host n<=256 limit are unchanged.
HC remains default-off experimental with its independent performance HOLD.

## Interval V2: Valid Accounting, Unstable Absolute Budget

Saved-layer2 V2 passes exactness, fixed census and interval validity in3.05s.
Unsampled per-chain GPU0.840396/0.696286ms drifts18.7559%: timing INCONCLUSIVE,
no retry. Sampled command-span normalized inclusive costs were router group
0.470525ms, routed gate/up0.150805ms, down/sum0.053844ms, shared gate/up and
down approximately0.01235ms each. Router group includes projection, selection
and shared sigmoid; do not call its65.434% envelope fraction top-k attribution.
No stable absolute/all-layer cost follows. Raw
`target/profiles/qwen4exp-moe-interval-v2-21778/` preserves all packets.

This lead plus source inspection identifies a one-thread insertion selector for
N512/K10 in the native singleton path. Existing local parallel selection can
test that mechanism without new quant downloads or changing expert arithmetic.
The pinned donor also uses parallel routing, but its full-softmax ranking and
renormalization are not equivalent to local selected-score normalization; no
donor arithmetic is copied.

## Finite Component Screen PASS

Existing parallel shader, raw research dispatch TG512/6144 shared bytes; no
product host widening.13 fixed finite fixtures pass exact IDs and bitwise GPU
weights, independent CPU order/F64 softmax, offsets/guards/poison/immutability.
Complete router and complete saved-layer2 MoE intermediates are bitwise; qualified
census differs only in selector kernel/threadgroup. All qualification precedes
timing. One warm16-chain ABBA, then measured16-chain ABBA per scope,0.40s test:

| Scope | GPU mean saved | Executor wall saved | GPU A spread | Wall A spread |
|---|---:|---:|---:|---:|
| Selector | 96.2193% | 93.6218% | 0.0450% | 0.2493% |
| Complete router | 92.7966% | 88.2597% | 0.0861% | 0.0196% |
| Complete MoE | 59.5064% | 56.3853% | 0.0798% | 1.1995% |

All mean/pair floors pass (10% GPU,5% wall), controls<=5%. MoE GPU16-chain
ABBA10.966958/4.425000/4.453292/10.958208ms; wall11.636709/5.027834/5.062291/
11.497958ms. Pre-timing census qualification, not measured-packet census. Raw
`target/profiles/qwen4exp-topk-screen-24292/` and `2026-09-15-qwen4exp-topk-screen-01.log`.

## Native Finite Qualification PASS

One packed2179-token prefix, QSA enabled and HC disabled.128 total native
forwards: ordinary32, restored captured incumbent32, candidate32, warm4-forward
ABBA and one measured4-forward ABBA. Production GPU lease, real wired-memory
gate and API validation held throughout;32.62s test.

All32 continuation full logit/hyper rows and terminal121 persistent tensors
match bitwise ordinary/captured-incumbent/candidate. All1536 router rows (48x32)
are finite; complete512-logit rows, selected IDs and weights match bitwise.
CPU descending-value/ascending-ID ordering agrees on every row. Candidate census
1536 selectors/384 split-QSA calls, no HC candidate. Timing omits capture copies
in both arms; own-arm outputs/hyper replay bitwise. Raw rows/banks and per-forward
timing evidence persist before gates in
`target/profiles/qwen4exp-native-topk-27023/`.

| Axis | A1 ms | B1 ms | B2 ms | A2 ms | Mean saved | A spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU, four forwards | 194.503416 | 116.220458 | 116.257625 | 194.191334 | 40.1901% | 0.1606% |
| Executor wall, four forwards | 216.605667 | 138.218458 | 137.859960 | 215.908416 | 36.1689% | 0.3224% |

Frozen mean and both pair floors pass, no retry. This is incremental native
executor evidence with QSA already on, not request throughput, prefill, a cold
cache result or a percentage to add to the earlier QSA gain.

## Delivery Boundary

The existing parallel selector invalidates a selected score with -infinity but
does not exclude its ID; nonfinite inputs remain unqualified. No product option
or default promotion yet. Next narrow guarded selector should classify exponent
bits before floating-point comparisons, synchronize a uniform threadgroup
decision and execute the incumbent serial algorithm on one lane if any input
is nonfinite. Keep finite normalization order and shared-router math unchanged.

Qualify finite and explicit NaN/infinity fixtures, fallback IDs/output classes,
capability/custody and actual product routing before delivery. The classification
changes the kernel, so today's performance cannot automatically qualify it.
No broad fusion, geometry expansion or extra dtype profiling prerequisite.
Read-only reviewer `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef` supports this rechart.

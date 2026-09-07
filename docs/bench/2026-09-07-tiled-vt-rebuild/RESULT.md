# Coalesced raw-bit V transpose: bank screen PASS

Research-only `619d4f03` replaces strided input gathers with a32x32 raw-u16
threadgroup transpose, padded stride33,256 threads and one barrier. It changes
copy transactions, not stored values. The existing compact-dispatch fix is the
control; no production selector or persistent storage is added.

## Oracle and bank timing

The first65536 source values are an odd-multiplier permutation of all65536 u16
patterns, explicitly checked on CPU. This covers all F16 NaNs/payloads, zeros,
infinities and subnormals without float conversion. CPU reference, incumbent and
tiled outputs agree bitwise for component counts1/38/1024, spans1/33/65/1025/
32768, bases0/7/3/31, nonzero destination views, padded strides and guard regions.
Source bytes and every untouched destination byte remain unchanged.

Then16 distinct G6 source banks serially reuse one V-transpose destination, as
in the existing single-chunk restored-prefill lifetime. This does not claim
in-place alias safety or simulate a persistent per-layer destination bank.
Both arms warm A-B-B-A, then measure A-B-B-A once. Primary GPU screen requires
>=25% aggregate and both-pair savings, <=5% control spread. Wall is secondary.

| Prefix | A1 GPU ms | B1 GPU ms | B2 GPU ms | A2 GPU ms |
| --- | ---: | ---: | ---: | ---: |
| 8840 | 7.685250 | 1.143042 | 1.121542 | 7.679708 |
| 32752 | 27.446208 | 5.058375 | 5.030917 | 27.453583 |

Means7.682479 ->1.132292 and27.449896 ->5.044646 ms save approximately85.26%/
81.62%. Both pair gates pass; control spreads are below0.1%. Corresponding
wall rows are7.835791/1.275042/1.260958/7.802833 and
27.589750/5.207792/5.164750/27.583125 ms. First-use warmup outliers remain in logs.

Actual allocation for16 sources plus one destination is307,773,440 /
1,140,293,632 B. Logical read+write counts are not measured DRAM traffic; the
reused destination can benefit from cache. Whole restored-prefill latency remains
unmeasured at this checkpoint. Fresh prefill usually builds VT through fused
scatter, so these numbers must not be projected onto fresh prompt processing.

## Closure and review

The CPU reread of the2026-08-24 N8 online-matrix closure rules out an apparent
transfer opportunity before execution: even persistent VT with rebuild cost
amortized away lost the8843-token endpoint (105.5 ->107.0 ms verification).
A cheaper transpose does not reopen that N8 work unit. The new mechanism targets
the already-selected restored-prefill rebuild instead.

Independent review confirms tile indexing, ordering and byte preservation. Its
initial objections to raw-bit coverage and a shared destination were withdrawn
after checking the actual generator and declared single-VT lifetime. A missing
Rust POD derive caused a retained compile failure; repaired compilation succeeds.
The raw-bit test runs explicitly before the timing test, both passing under an
external guardian holding the ordinary production Metal lease. No owner is
interrupted or bypassed.

All acquisition: `target/profiles/vt-tiled-rebuild/`, including build01 failure,
build02 success, separate oracle/timing logs, execution and host-counter records.
Only one GPU timing packet runs. Full-model/request authority is not established.

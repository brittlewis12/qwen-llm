# Coalesced raw-bit V transpose: bank PASS, restored-phase HOLD

The subsequent full-model phase preserves logits/state bitwise but fails its
frozen control-spread gates. No production flag, kernel promotion or endpoint
qualification follows. The bank result remains valid at its original scope.

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

## Admitted restored-prefill phase

Test-only `e3ee25e7` uses Qwen3.8-27B-Q8_0 and the same frozen Marcus fixture as
`docs/bench/2026-09-07-full-verifier-online-n16/RESULT.md`. An8840-token prefix is
primed once; a canonical snapshot, not any test-generated suffix, then extends
that prefix to32752. The CPU fixture text and prefix-token hashes are asserted.

Before any phase execution, the planned width32 was corrected to16:32K/width32
would exceed serving's existing128MiB scratch limit. Width16 is in the existing
Q8 restored-tail selector, and the real driver price is asserted before each
prime. This is a pre-data reachability correction, not a changed admission policy
or post-result gate. No width32 phase runs.

Each arm creates a fresh session outside the interval, then charges snapshot
restore, scratch planning/allocation and16-token single-chunk prefill. The pilot's
`allocation_ms` includes planning, unlike serving's allocation telemetry; compare
its total, not similarly named production subphases. A test-only scoped TLS mode
routes compact VT rebuilds through the tiled kernel after the existing host guard
wall. It resets on unwind; non-test builds cannot select it. Host dispatch counts
are0/16 for A/B, not an independent GPU timing instrument.

Frozen per-prefix order: warm A-B-B-A, measured A-B-B-A. Primary32752 requires
>=5% aggregate/both-pair total-wall savings and<=5% control spread;8840 is a<=3%
regression guard with the same spread limit. No failed timing row is rerun.

| Prefix | A1 total ms | B1 total ms | B2 total ms | A2 total ms | A spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8840 | 553.490000 | 231.057167 | 226.237417 | 222.495541 | 85.309% |
| 32752 | 462.399292 | 348.693542 | 334.033750 | 357.194167 | 25.673% |

Both spread gates fail: **HOLD / inconclusive timing**. Do not advertise the
favorable B rows or discard first A. Warmup full-state hash/readback work precedes
first measured A, whereas later measured arms have no such intervening work.
This is asymmetric conditioning, not a proven GPU-clock, cache or compression
cause. It further limits the packet; it is not authority to rescore or rescue it.

Complete F32 logits and canonical K/V/GDN/conv payload hashes agree bitwise across
all warmup arms and final measured A; measured B arms are deliberately not hashed.
All layer positions agree. This is not generated-continuation or HTTP evidence.
Actual scratch stays35,094,528 B at8840 and103,546,880 B at32752 in every arm;
driver prices35,268,456 /103,750,504 B pass the existing cap. No residency change.

The phase test passes correctness and finishes in242.21 s. Source priming takes
39.382/126.734 s, outside the compared intervals. Across the whole process,
global compression/pageout/swapout growth is zero, decompressions171 and swapins20.
These counters are not phase-local and do not explain the timing variation.
Non-test release library check passes. All phase/build/counter records and
`phase-analysis.json` remain under `target/profiles/vt-tiled-rebuild/`.

## Re-ranking checkpoint

Original and fresh independent reviewers support HOLD, no endpoint spend and no
N8 transfer. Source inspection finds a larger removable boundary: publication
copies session state into CPU `Vec<u8>` arenas, and `restore_from` copies them
back into shared Metal buffers. In the four32K measured rows, restore alone takes
113.654-142.007 ms; no stable endpoint saving follows from those observations.
Validation checks metadata, spans and token IDs, not the multi-GB state payload.

A GPU-owned RAM snapshot or a bounded live-continuation handoff would be a new
work unit, not a repair of this copy-kernel packet. Any such experiment must charge
publication, destination allocation, synchronization, lifetime and cache/admission
costs. Do not retain an unpriced CPU+GPU mirror, mutate an immutable cached prefix,
or wrap arbitrary `Vec` storage in a no-copy Metal buffer. The current safe no-copy
helper owns a page-aligned `Arc<Mmap>`; it is not a general Vec borrowing API.

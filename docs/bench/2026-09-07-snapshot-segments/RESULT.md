# Incremental snapshot publication: lifecycle PASS, production unpromoted

Test-only `3cfc7fe0` reuses immutable CPU KV from the root actually restored into
the producer. Subsequent publication plus a full restore takes 39.026% less wall
time at 8840+16 rows and 52.268% less at 32752+16. Both frozen cells pass; this is
not endpoint, first-root, fresh-prefill, decode or production-cache authority.

## Work and ownership

The model-free producer has 16 K and 16 V buffers with F16 KV geometry, 2048
bytes/token/layer, plus all 48 GDN-conv and 48 GDN-state buffers. Weight Q8 is not
KV Q8. A root is published, explicitly restored into the producer, then 16 KV
rows are appended and all recurrent state changes. No model or GPU commands run.

- A uses full host capture into uninitialized CPU arenas; B shares the actual
  immutable KV root and copies only the appended tail. Both recapture all
  156,893,184 recurrent bytes into independent arenas.
- Both retain the old root cache entry, allocate fresh uninitialized per-layer
  Metal destinations and restore the entire new logical snapshot. Publication
  allocation, destination allocation and all copies are timed. Destruction after
  the interval is excluded; this is not eviction timing.
- Initial root publication/restore is setup outside BOTH arms. At 32K it takes
  115.889/58.745 ms, separately recorded. It is not free or amortized into a claimed
  first-root gain. Each timed arm starts from the same restored root plus tail,
  not the previous arm's newly published snapshot.
- A uses the existing full-copy helpers. B's tail capture uses the CPU prototype;
  restore uses the existing offset-zero helper plus checked tail-offset memcpy.
  No CPU/GPU mirror, arbitrary Vec no-copy wrapper, GPU snapshot or wiring.

The representation moves a flat root Vec into Arc without copying payload. It
retains sealed segments and rebuilds only a partial tail plus new rows, in
1024-row slabs. Irregular imported roots are sealed. No whole ancestor snapshot
or its recurrent state is retained by the prefix anchor. Flat layer-major export
is explicit and outside timing, never a hidden RAM-cache mirror.

Three CPU tests cover roots 0/1/1023/1024/1025/32752, irregular appends, canonical
byte equality, capacity/shape/overflow rejection, pointer sharing, fork isolation,
unrelated equal-byte states, raw-access/rewind invalidation and weak-reference
release of superseded partial tails. These establish a closed-world prototype
invariant, not coverage of production mutation paths.

Real-geometry oracles compare every restored byte and explicit canonical KV
export, check independent changed recurrent arenas, mutate every producer and
destination region after publication, and restore again. Unchanged raw exposure
also forces a full recapture. Oracles finish before a separate warm A-B-B-A,
followed by measured A-B-B-A; neither block has interposed payload readbacks.

## Frozen results

Each cell requires >=20% mean AND both-pair total-wall savings, with <=5% A spread.
Both cells are required. All samples are retained, with no minimum selection,
timed rerun, context-threshold retrofit or cross-packet baseline comparison.

| Root rows | A1 ms | B1 ms | B2 ms | A2 ms | Mean saving | A spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8840 | 63.520708 | 39.126125 | 39.063667 | 64.713041 | 39.026% | 1.860% |
| 32752 | 226.436125 | 108.760375 | 109.211417 | 230.223125 | 52.268% | 1.659% |

Means are 64.116875 ->39.094896 ms and 228.329625 ->108.985896 ms. Pair savings
are 38.404/39.636% and 51.969/52.563%. Publication alone averages
31.753687 ->6.757041 ms and 99.885980 ->6.713375 ms, respectively.

32K restore-with-allocation is observed at 128.104/128.783 ms in A and
102.052/102.493 ms in B. **All restore bytes remain**; this difference is not
evidence of deleted restore work or an attributed hardware/cache effect. The
primary includes both phases rather than projecting publication-only savings.

## Byte and lifetime accounts

| Root rows | A publication bytes | B publication bytes | Full restore bytes, both |
| --- | ---: | ---: | ---: |
| 8840 | 737,280,000 | 157,941,760 | 737,280,000 |
| 32752 | 2,304,376,832 | 157,941,760 | 2,304,376,832 |

At 32K, two-entry unique CPU payload falls from 4,607,705,088 to 2,461,270,016 B:
2,146,435,072 B of root KV is shared rather than separately copied/owned. Both
entries' conservative logical accounting remains 4,607,705,088 B. At 8840 the
unique payload reduction is 579,338,240 B. Do not claim a larger logical cache
capacity: unchanged 4GiB accounting still cannot retain both 32K entries.

Live Metal allocation is identical across arms: 4,610,015,232 B at 32K, with
2,304,770,048 B of destination increment. Copy-region count rises from 128 to160
per full restore; candidate KV has two segments per arena in this packet. This
is not a many-generation fragmentation timing result. Unique CPU payload excludes
allocator slack, Arc/segment metadata, cache bookkeeping and other transients;
neither it nor driver allocation is peak RSS, physical residency or full-engine
admission. Anchors surviving cache eviction still need explicit lifetime accounting.

## Execution and disposition

One release lifecycle test passes in 9.31 s under an external guardian holding
the ordinary production Metal lease. Three CPU tests and the non-test release
library check pass. Global compression/swap/pageout growth is zero; decompressions
grow by 2 across the process. `pmset` records no thermal or performance warning.

The first lifecycle build fails on missing Metal traits and cast parentheses,
before execution; the repaired build passes. Both logs are retained. Independent
review accepts bounded phase PASS and rejects production/endpoint promotion. Its
incorrect zero-fill objection was explicitly retracted after reading
`MetalTensor::zeros_f32`, which already calls `buffer_uninit`.

Raw builds, CPU tests, sole lifecycle packet, setup/warmups, host counters,
guardian, source identity and mechanical scoring: `target/profiles/snapshot-segments/`.

Next authority is the actual Sequence restore -> owned append -> publication
boundary, not another copy or attention packet. Production currently retains no
checkpoint anchor. Source audit finds a stronger escape than the unsafe mutable
bridge: safe `Sequence::metal_session(&self)` exposes public KV tensors and retained
buffers, and safe `BlitEncoder::copy_buffer` can later write those buffers. An
alias can escape BEFORE restore and mutate KV AFTER it; merely clearing an anchor
when either accessor is called cannot safely rearm reuse on a later restore.
Independent review confirms this source counterexample, not an executed runtime
corruption test. Production eligibility needs lifetime-safe opaque access or
permanent reuse taint after raw buffer escape, including the immutable accessor.
The closed-world CPU prototype has no escaped GPU aliases and remains valid.
Durable canonical export, poison/rewind/speculative rollback, cache admission and
eviction require integration tests. No production snapshot ABI, policy, defaults,
cache keys, checkpoint eligibility or retry behavior changes in this checkpoint.

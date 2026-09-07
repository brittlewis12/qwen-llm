# Snapshot lifecycle: 32K cell PASS, two-cell promotion HOLD

Test-only `741c4ef4` compares complete snapshot publication plus one restore,
not just an already-staged GPU copy. At32K the model-free cycle drops from
244.042084 to85.585000 ms, saving64.930%; both pairs pass25% and control spread
4.333% passes5%. The8840 control spreads8.218%, so the required two-cell gate is
**HOLD**. No GPU cache implementation, default change or endpoint claim follows.

## Work unit

The synthetic completed producer has ordinary dense27 persistent-state geometry:
16 K,16 V,48 GDN-conv and48 GDN-state buffers. KV is **F16**,2048 bytes/token/layer;
weight quantization is not KV-cache quantization. Per-GDN-layer state is3,145,728 B
and conv122,880 B. Logical snapshot payloads are736,231,424 /2,303,328,256 B at
8840/32752 positions. Model weights are not loaded.

- A allocates four uninitialized CPU arenas and uses the existing
  `read_tensor_into` publication helper. It then allocates fresh per-layer Metal
  destinations and restores with `write_tensor_bytes`.
- B allocates four owned Metal snapshot arenas, encodes128 capture regions and
  waits. It then allocates the same fresh destinations, encodes128 restore regions
  and waits. No CPU mirror, no-copy wrapper, immutable-prefix alias or wiring.
- Primary wall includes snapshot allocation/publication, destination allocation,
  encoding and both GPU completion waits. Source, snapshot and destination coexist.
  Destruction/eviction is outside the interval and has no claimed speedup.
- Snapshot/source/destination ownership remains separate. The oracle mutates the
  producer after publication, then the destination after restore, and restores
  again from the unchanged snapshot. Complete payloads agree byte-for-byte.

Correctness A/B runs precede a separate warm A-B-B-A, then measured A-B-B-A.
Neither warm nor measured blocks contain payload readbacks between arms. Per-call
autorelease pools release completed command resources afterward. The previous
tiled-VT hash-conditioning failure is not rescored or reused as evidence.

## Frozen result

Each cell requires >=25% mean and both-pair total-wall savings, with <=5% control
spread; both cells are required for promotion. Gates are applied externally to
the four recorded samples, without minimum selection or reruns.

| Prefix | A1 ms | B1 ms | B2 ms | A2 ms | Control spread | Cell |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8840 | 90.783916 | 28.593917 | 32.978292 | 98.564167 | 8.218% | HOLD |
| 32752 | 238.754333 | 82.749125 | 88.420875 | 249.329834 | 4.333% | PASS |

32K pairs save65.341/64.537%. Its CPU publication takes107.178/115.897 ms and
restore-with-allocation131.576/133.432 ms. GPU publication takes44.455/50.640 ms
and restore-with-allocation38.294/37.780 ms. These are lifecycle-phase observations,
not request TTFT or decode throughput. The favorable8840 difference does not
override its failed control gate.

At32K, live Metal bytes are4,607,918,080 in A plus2,303,328,256 B of CPU snapshot
capacity, versus6,911,246,336 Metal bytes and no CPU snapshot in B. The GPU snapshot
increment is2,303,328,256 B and destination increment2,303,721,472 B. Driver limits
and headroom are printed while all owners coexist. These are buffer/capacity
accounts, not peak RSS, physical residency or a real-engine admission result.

The producer is CPU-initialized synthetic data. Metadata validation, token IDs,
logits, capture tails, activation scratch, cache lookup/policy, model execution,
serialization and durable publication are outside this model-free scope. The
oracle's second restore is correctness-only; `gpu_waits` reports the timed body.

## Execution and disposition

One release test passes its immutability/byte checks in3.29 s. The only setup
failure is a missing `BlitEncoder` import, retained as `build-01.log`; repaired
compilation succeeds before execution. One GPU packet runs under an external
guardian holding the ordinary production Metal lease, without interrupting other
owners. Global compression/decompression/swap/pageout counters do not grow.

All raw records, including warmups, allocation/headroom, build failure, execution,
host counters and mechanical scoring: `target/profiles/snapshot-transfer/`.
Independent review confirms the32K cell and overall two-cell HOLD. Do not add
a GPU cache, retrofit a context threshold, or claim that engine ownership,
abort/retry semantics and budget accounting have been qualified.

## Higher-leverage policy boundary

Source audit finds serving's default cache is4GiB (not the library's16GiB default).
At32K, prompt and completed snapshots of roughly2.3GB each cannot coexist. Prompt
capture occurs before first token; successful completed publication can LRU-evict
that prompt entry before another serialized generation uses it. Strict eligibility
checks one entry, not the pair. This is source/arithmetic evidence, not measured
traffic frequency or an unconditional reason to skip capture.

The CPU test `transient_prompt_capture_has_a_budget_and_failure_boundary` uses
small cache-index fixtures to witness three cases: completion replaces the prompt
when only one fits; elision changes retry lookup before completion; and elision
changes regeneration lookup when both fit. It does not restore model tensors or
run HTTP. These counterexamples rule out blanket prompt-capture removal and put
conditional publication elision ahead of a GPU-copy redesign as the next policy
falsifier, with explicit cancellation/failure/consumer guards.

The complete CPU prefix-cache suite passes19 tests after rebase; the non-test
release library check also passes. No production cache policy is changed.

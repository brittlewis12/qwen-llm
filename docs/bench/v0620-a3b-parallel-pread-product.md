# v0.620 A3B Direct-Pread Product Confirmation

Status: GO for explicit direct pread over explicit mmap parallel copy on the
frozen single-shard A3B asset. Default Auto remains unchanged.

## Intent

Measure the marginal process-cold product value of direct retained-descriptor
`pread` against the current topology-preserving mmap parallel-copy loader.

v0.619 is sealed with no authority after its first fresh child exposed stdout
tracing contamination. Its full-state correctness gate and all 12 loaded
children completed first, and every loaded stability and per-pair
noninferiority gate passed. Its A arm was ordinary sequential copied loading,
which is valid for post-load parity but not the current marginal cold baseline.

This is a fresh-only successor, not a v0.619 retry. It imports v0.619's sealed
correctness and loaded evidence, verifies the complete predecessor inventory,
and compares the two population methods that share the same authenticated
733-resource final storage.

## Evidence Bridge

Before preflight, require the complete v0.619 packet at
`target/profiles/v0619-a3b-parallel-pread-loader-p1` and verify:

- packet-complete SHA-256
  `daf5473344fdeec2851942e593d31b3fafd816fdf8624a86d74882ad12b815e1`;
- decision SHA-256
  `c5ac80870199ab8e99029a33f2e7fb3510ca70711a006e05480717d68714f6d1`;
- inventory SHA-256
  `c870d3a35c91d0789cde16a177f63a577bea999c1f24248ae5e61d6224360b1b`;
- every inventory member's path and digest, with no unlisted file except the
  inventory and completion seal;
- predecessor source `c7884742c425a6cb5d80b88e0c67ee05ba258892`;
- correctness passed and loaded `stable`, `performance_passes`, and `passes`
  are all true;
- status `implementation_or_contract_defect`, authority `none`, stop after
  `fresh-128`, and exact error `outer_residual_ms.left is not positive`;
- exactly 12 loaded attempts, followed by one fresh A launch and completion.

The only executable-source change allowed since the predecessor is commit
`ea8ec11a7afa2f2647544669cffc9e23cd56a72b`, whose parent must be the v0.619
source and whose sole change routes `qwen` tracing to stderr in
`crates/qwen-cli/src/main.rs`. The v0.620 packet commit must have `ea8ec11` as
its parent and may add only this contract and runner. Loader,
inference-library, dependency, compiler, and `qwen-bench` source changes
require a full successor instead of this bridge.

## Frozen Arms And Shape

- A: `QWEN_GGUF_PARALLEL_COPY=1`, forced mmap parallel copy.
- B: `QWEN_GGUF_PARALLEL_COPY=pread`, forced retained-descriptor pread.
- Both arms use the v0.602/v0.619 A3B asset, prompt, geometry, native embedding,
  storage conflicts, pressure rules, conditioning, and fixed
  `AB BA BA AB AB BA` order.
- A must emit exactly one `[metal-gguf-parallel-copied]` marker; B must emit
  exactly one `[metal-gguf-parallel-pread]` marker. Every other
  `[metal-gguf-*]` occurrence is a contract defect.
- Both markers must carry identical schema-1 geometry, schedule, options,
  inventory, plan, and timing reconciliation, followed by the exact copied
  logical ledger.
- No processes overlap. No child, pair, or packet retry is allowed.

Run only the v0.602 fresh output-128 shape: 419 prompt tokens, 128 requested and
emitted greedy tokens, 127 target transitions, prefill chunk 1024, context 1024,
disabled prefix cache, timing schema 3, and external first-byte/exit timestamps.
Require stdout SHA-256
`e18fd50a1e2add01cfed4f498bf0052b630653517057cda3620b302d1a68f198`
for every child. This both freezes generated output and rejects tracing or ANSI
contamination before the first token.

## Product Gates

For every pair record:

```text
L = runtime_and_model_load_A - runtime_and_model_load_B
Q = runtime_and_model_load_B / runtime_and_model_load_A
F = spawn_to_first_byte_A - spawn_to_first_byte_B
E = spawn_to_exit_A - spawn_to_exit_B
```

Require:

- median `L >= 112 ms`, AB median `L >= 112 ms`, BA median `L >= 112 ms`;
- median `Q <= 0.85`;
- B wins `L` in at least 5/6 pairs and at least 2/3 per order stratum;
- median `F >= 112 ms`, AB median `F >= 112 ms`, BA median `F >= 112 ms`;
- B wins first byte in at least 5/6 pairs and at least 2/3 per stratum;
- median `E >= 112 ms`, AB median `E >= 112 ms`, BA median `E >= 112 ms`;
- B wins exit wall in at least 5/6 pairs and at least 2/3 per stratum;
- maximum paired RSS and physical-footprint ratios at most `1.05`;
- exact timing schema, marker, ledger, pressure, zero block input, and zero
  fresh major faults.

Fresh prefill and model-ready TTFT component ratios are diagnostic rather than
independent one-observation gates. Complete first-byte and exit endpoints remain
authoritative.

## Decision And Authority

Any evidence-bridge, identity, source-delta, schema, output, marker, ledger,
pressure, or child-execution defect stops without authority. A valid product
gate miss kills B. Only the complete conjunction authorizes explicit
`QWEN_GGUF_PARALLEL_COPY=pread` over forced mmap parallel copy on the exact
single-shard A3B asset.

A GO does not authorize default selection, other assets, storage-cold claims,
physical-memory savings, serving, concurrent loading, or lower system CPU.

## Result

All six fixed-order output-128 pairs pass every gate. Authoritative paired
median savings are `205.122 ms` load, `205.937 ms` first byte, and `249.905 ms`
exit, with 6/6 wins and 3/3 in both order strata. Median load B/A is
`0.810124x`. Population ready improves by paired median `206.301 ms`, localizing
the endpoint movement to destination population.

Every child emits the frozen 590-byte stdout with SHA-256
`e18fd50a1e2add01cfed4f498bf0052b630653517057cda3620b302d1a68f198`.
Maximum RSS B/A is `0.504262`, while physical-footprint B/A is `0.999592`.
User CPU falls sharply, system CPU rises, and total CPU falls. All fresh major
fault, block-input, pressure, identity, marker, ledger, inventory, and
completion seals pass. The exact authority remains force-only; an Auto selector
change requires a separate policy decision and actual-Auto guard.

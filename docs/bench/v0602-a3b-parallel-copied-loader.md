# v0.602 A3B Topology-Preserving Parallel-Copied Loader Pilot

Status: preregistered; implementation and every packet remain unrun.

## Intent

Determine whether v0.599's topology-preserving four-worker materialization floor
transfers into the production A3B loader while preserving copied-storage loaded
prefill, decode, request wall, and bit-exact target state.

v0.599 keeps 733 exact-sized, offset-zero resources and reduces median materialized
ready wall from `2066.701` to `742.676 ms`, saving `1324.148 ms` in 6/6 pairs. Its
frozen arithmetic projects `2.06601x` output-1 first byte, but it does not establish
model-load transfer, loaded parity, full target state, or product latency. v0.598
shows why loaded parity comes first: one-window owned storage transfers the cold
saving but imposes an 11-12% loaded request regression.

This pilot is a warm-filesystem-cache, fresh-process and loaded-process experiment
for one exact A3B asset. It is not storage-cold, default-policy, broad-family,
alias, conversion, MTP, split-shard, serving, footprint-reduction, or asynchronous
promotion authority.

## Frozen Asset And Runtime Geometry

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- SHA-256: `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Product prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`.
- Prompt SHA-256:
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- One shard; mapped bytes `22,134,528,992`.
- Descriptor-layout digest `0x5ae645df5cf7d568`.
- Ordered inventory digest
  `f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5`.
- Retained-plan identity control
  `fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af`.
- 733 all-direct requests over `22,123,538,944` bytes.
- Page size 16,384; alignment 32; Metal maximum buffer length
  `77,309,411,328`.
- Unified-memory device and production-auto Q8_0 token embedding.
- Untied, non-MTP model with this architecture tuple:

```text
kind=Moe layers=40 hidden=2048 intermediate=0 vocab=248320
full_interval=4 q_heads=16 kv_heads=2 attn_head_dim=256
rope_theta=10000000 partial_rotary=0.25
gdn_v_heads=32 gdn_k_heads=16 gdn_head_dim=128 conv=4
experts=256 topk=8 routed_ffn=512 shared_ffn=512 mtp_layers=0
```

Runtime authenticates structure, not payload content. The runner hashes the complete
model before correctness and timing. No child hashes or rereads the payload after
load except the ignored, untimed correctness audit defined below.

## Force Policy

Add default-off environment control `QWEN_GGUF_PARALLEL_COPY=1`. Truthy values
force the candidate; falsy or absent values select the existing policy. Invalid or
non-Unicode values are hard load errors.

Resolve every storage policy before sentinel authentication or allocation. When
parallel copy is forced:

- allow explicit false `QWEN_GGUF_OWNED_ARENA` and `QWEN_GGUF_NO_COPY`;
- reject truthy owned-arena or no-copy controls;
- reject any explicit `QWEN_GGUF_NO_COPY_PREFAULT`, including false;
- require `QWEN_NATIVE_QUANT_EMBED` to be absent so selection is AutoPromoted;
- reject truthy `QWEN_MOE_ROUTER_F16`; absent or explicitly falsy is allowed;
- retain the all-direct request count, byte total, and inventory digest as the
  authoritative storage check;
- reject every unsupported asset or geometry before candidate allocation;
- never retry or fall back to copied storage after candidate allocation begins.

The conflict resolver is pure and receives a unit-tested presence/value matrix. No
CLI flag or dependency is added. When parallel copy is disabled, every existing
storage policy retains its current behavior.

## Implementation Contract

Add `DirectStorage::ForcedParallelCopied(PlannedParallelCopiedStorage)`. The storage
object is completely realized before ordinary sequential model construction begins:

1. Authenticate the frozen asset, architecture, request inventory, and geometry.
2. Recompute and require the frozen retained-plan digest before allocation. The
   plan is identity evidence only and cannot drive resources, bindings, or schedule.
3. Sort request indices by `(shard_idx, data_offset, request_index)`.
4. Validate the sorted indices are a complete permutation of `0..733`.
5. Apply the literal v0.599 cuts; do not transplant its minimax optimizer.
6. Start ready wall immediately before the first allocation.
7. Allocate 733 exact-sized resources in canonical request order with production
   copied creation options.
8. Resolve and validate all immutable source slices on the parent.
9. Prove destination identity, address disjointness, source/destination
   non-overlap, exact lengths, alignment, modes, and one writer per complete task.
10. Construct parent-owned task slices, run exactly four scoped copy workers, and
   join every started worker.
11. Drop all task/source/destination slices, then create 733 offset-zero
   `OwnedWeightReadOnly` tensors.
12. Validate physical topology, end ready wall, then format and emit one candidate
    marker outside the endpoint.

`load_direct` remains sequential. It validates the next expected request's index,
name, shard, source offset, source bytes, dtype, shape, direct kind, and resident
bytes, clones the indexed tensor, advances the cursor, and records
`SourceMaterialization::DirectCopy`.

At loader finish require cursor 733, exact tensor/resource identities, offset zero,
exact lengths/options/provenance, frozen schedule attestation, and the normal copied
logical ledger. Candidate tensors are typed read-only while ordinary copied tensors
remain writable; the inference topology and represented bytes are the parity claim,
not provenance identity. Checked typed writes must reject candidate weights.

## Frozen Schedule

The sorted schedule is frozen as:

```text
cuts          155,359,539
task_counts   155,204,180,194
worker_bytes  5532746240,5462315776,5595522304,5532954624
first_offsets 10990048,5543736288,11006052064,16601574368
last_offsets  5392741344,11004937952,16450579424,22134520800
```

Every partition is shard zero, nonempty, contiguous in sorted-task order, and owns
whole tensors only. Validate partition starts/ends, first/last source identity,
complete task union, and no duplicate request before creating mutable slices.
Alignment 32 is the required Metal binding-offset alignment; every candidate offset
is zero. This is not a CPU virtual-address alignment claim.

## Unsafe And Failure Contract

Use one narrowly scoped unsafe helper that creates an exact mutable byte slice from
an exclusive `&mut Buffer` borrow. Build task slices from `resources.iter_mut()` in
request order, then move each task exactly once into sorted order. Do not access the
resource vector until all tasks are dropped.

Before the unsafe helper:

- prove 733 unique Objective-C resource identities;
- prove every destination address range is nonempty and pairwise disjoint;
- prove each source length equals its destination length;
- prove no immutable GGUF source range overlaps any mutable destination range;
- prove Shared CPU accessibility and creation/observed resource modes;
- prove no typed view or GPU work exists.

Workers execute only an infallible task loop and `copy_from_slice`. They perform no
Objective-C operation, allocation, GGUF lookup, tensor construction, logging, or
error reporting. Parent code handles scoped spawn failures and joins every started
worker. A spawn error or panic discards partial storage, emits no ready marker, and
returns a hard load error with no fallback.

All workers join and all raw slices die before any `MetalTensor` or GPU consumer
exists. Buffer and GGUF lifetimes cover the complete scope.

## Accounting And Marker

The ordinary ledger must remain byte-for-byte identical to copied A:

```text
[metal-load-ledger] source=733/22123538944 direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 tail_fallback=0/0 converted=0/0/0 derived=0/0
```

Copied A emits no candidate marker. Candidate B emits exactly one schema-1
`[metal-gguf-parallel-copied]` marker after tensor prebuild and ready validation.
The exact one-line grammar, field order, names, decimal encoding, lowercase hashes,
and comma-separated list encoding are:

```text
[metal-gguf-parallel-copied] schema=1 resources=733 bytes=22123538944 workers=4 cuts=155,359,539 tasks=155,204,180,194 worker_bytes=5532746240,5462315776,5595522304,5532954624 first_offsets=10990048,5543736288,11006052064,16601574368 last_offsets=5392741344,11004937952,16450579424,22134520800 create=shared,default_cache,default observed=shared,default_cache,tracked page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 layout=0x5ae645df5cf7d568 inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af allocation_us=<u64> source_us=<u64> copy_us=<u64> binding_us=<u64> ready_us=<u64>
```

`source_us` covers source, task, resource, range, and schedule validation.
`copy_us` covers spawn through all joins. `binding_us` covers construction of the
prebuilt storage tensors and physical-topology validation, not later clones into
model fields. All fields are nonnegative base-10 integers except the literal
lowercase hashes and layout value. Require:

```text
abs(ready_us - (allocation_us + source_us + copy_us + binding_us)) <= 4
```

Marker formatting and emission are outside ready. Recheck the physical facts at
loader finish. Require zero candidate markers for every A load and exactly one for
every successful B materialization, including the correctness test. The marker is
attribution; fresh scoring uses `runtime_and_model_load_ms`, not ready wall.

Among recognized storage lines, A order is native-embedding policy then copied
ledger. B order is native-embedding policy, candidate marker, then copied ledger.

Production performs no post-copy 22 GB byte verification. The ignored correctness
test verifies every candidate resource against its immutable source.

## Correctness Gate

Before timing, run one ignored local-fixture test in the frozen build. Compare
production copied A with forced parallel-copied B for:

- every candidate resource byte against its authenticated GGUF source;
- exact 733-resource topology, schedule, options, offsets, provenance, and ledger;
- packed-prefill full logits and complete KV/GDN/conv state;
- identical greedy argmax;
- one forced transition;
- continuation full logits and complete KV/GDN/conv state;
- rejection by checked typed write APIs.

Every float bit and state byte must match. Any policy, byte, schedule, topology,
marker, log, or state failure is an implementation defect and stops before loaded
timing.

The test bypasses tokenization and uses this exact 12-token vector with no BOS or
special-token insertion:

```text
[7734,264,12654,709,310,12204,279,76938,8240,5199,7638,13]
```

Use session capacity 64, position zero, `PrefillScratchConfig::default()`, normal
packed prefill, full logits, and the copied prefill argmax as B's forced transition.
The vector comes from the frozen Qwen3.6 Fibonacci fixture and every ID is below the
authenticated 248,320-token vocabulary.

Refactor model construction so the test can construct one B storage object, audit
all 733 prebuilt tensors against their sources, and pass that same object into the
ordinary sequential model builder. Do not rematerialize B for full-state testing.
The complete test therefore performs one A model materialization and one B model
materialization, expects exactly two identical copied ledger lines in A-then-B
order, zero candidate markers before B, and exactly one candidate marker for B.
The B marker must precede B's ledger. Production children never perform the full
byte audit.

## Product Arms And Order

- **A - copied**: parallel copy false, owned arena false, no-copy false, no prefault
  variable, native embedding unset.
- **B - parallel copied**: parallel copy true, owned arena false, no-copy false, no
  prefault variable, native embedding unset.

Both product stages use six fresh-process pairs in this fixed order:

```text
AB
BA
BA
AB
AB
BA
```

Processes and stages never overlap. There are no child, pair, stage, or packet
retries. A valid performance loss is never retryable. A validity failure stops the
sole packet inconclusive.

For every child, execute this exact prelaunch sequence outside scored endpoints:

1. Capture host and VM state.
2. Read exactly the model file size sequentially with one 8 MiB buffer.
3. Wait exactly 30 seconds.
4. Run the frozen host sampler for at most six samples, 30 seconds apart.
5. Require AC power, no thermal/performance warning, and at least 50% parsed memory
   availability; otherwise stop inconclusive before launch.
6. Capture pre-spawn VM state and launch the child once.
7. Capture post-exit host and VM state.

Hash source, binaries, runner, contract, model, prompt, and imported helpers;
require clean matching source/build/runtime identity and normalized QWEN/Metal
controls.

## Stage 1 - Loaded Noninferiority

Run loaded parity before fresh product timing. Reuse the v0.598 shape:

```text
qwen-bench decode
prompt=current-reva-n8-interactive-qwen36.txt (419 tokens)
prefill_chunk=1024 kv_capacity=1024 full_logits=true
decode_calls=127 runs=5
```

Each process performs the existing untimed prompt-plus-one-transition warmup and
then five fresh-session repetitions. Score repetition 3-5 medians. For pair `i`:

```text
P[i] = median(prefill_A) / median(prefill_B)
D[i] = median(decode_A)  / median(decode_B)
R[i] = median(request_B) / median(request_A)
```

Every pair must satisfy:

- `P[i] >= 0.99`;
- `D[i] >= 0.99`;
- `R[i] <= 1.01`;
- each arm repetition-5/repetition-3 decode TPS is inside `[0.98,1.02]`;
- each arm repetition-3-5 decode-TPS range is at most 3% of its median;
- every pair has `RSS_B/RSS_A <= 1.05` and `footprint_B/footprint_A <= 1.05`;
- exact run count, request wall, marker, and ledger.

These are hard engineering noninferiority bounds, not a powered equivalence claim.
Complete all 12 valid loaded children before inspecting performance. Generated
output identity must be global across all 12 children. Evaluate both frozen
stability gates for every arm first. Any stability failure makes Stage 1
inconclusive. If every arm is stable, evaluate all P/D/R and memory gates; any miss
kills B before fresh timing. No additional dispersion judgment exists.

All parsed values and derived ratios must be finite with positive denominators.
Pair indices, fixed order, and AB/BA membership must match the frozen packet.
The decision artifact records P/D/R, both stability values for both arms, all six
RSS/footprint ratios and maxima, global output identity, and one Boolean per loaded
gate.

## Stage 2 - Fresh Output-128 Product Packet

Run only after Stage 1 passes. Use the v0.598 CLI shape: frozen 419-token Reva
prompt, 128 requested and emitted greedy tokens, chunk 1024, context 1024, prefix
cache disabled, and external process spawn through first byte and exit.

Require all 12 children to produce identical stdout, exactly 128 generated tokens,
127 target transitions, no early EOS, one timing row, the exact arm marker contract,
and clean identity. Require request timing schema 3 with:

```text
request_epoch=first_post_model_load request_index=0
prefix_cache_used=false prefill_chunk_requested=1024
prefill_chunk_effective=419 prompt_tokens=419 max_context_tokens=1024
ttft_endpoint=stdout_flush_complete decode_policy=greedy_argmax
stop_reason=token_limit generated_tokens=128 transition_count=127
```

Read stdout in bounded chunks. Timestamp after every nonempty read; EOF is not a
last-byte timestamp. Record:

```text
spawn_to_last_stdout_byte_ms
last_stdout_byte_to_exit_ms = spawn_to_exit_ms - spawn_to_last_stdout_byte_ms
```

Complete all 12 valid fresh children before calculating performance. Then score:

```text
F[i] = spawn_to_first_byte_A / spawn_to_first_byte_B
E[i] = spawn_to_exit_A       / spawn_to_exit_B
L[i] = runtime_and_model_load_A - runtime_and_model_load_B
Q[i] = runtime_and_model_load_B / runtime_and_model_load_A
```

Require:

- median `F >= 1.50x`, AB median `F >= 1.40x`, BA median `F >= 1.40x`;
- B wins first byte in at least 5/6 pairs and at least 2/3 in each order stratum;
- median `E >= 1.25x`, AB median `E >= 1.20x`, BA median `E >= 1.20x`;
- B wins exit wall in at least 5/6 pairs and at least 2/3 in each order stratum;
- median `L >= 750 ms`, AB median `L >= 600 ms`, BA median `L >= 600 ms`;
- median `Q <= 0.60`, AB median `Q <= 0.70`, BA median `Q <= 0.70`;
- maximum paired `RSS_B/RSS_A <= 1.05` and `footprint_B/footprint_A <= 1.05`.

Record TTFT, generation wall, request wall, outer residual, PSO wall, candidate
phase timings, final-output-to-exit wall, and complete process resources. They
cannot rescue a failed endpoint gate.

All parsed values and F/E/L/Q ratios must be finite with positive denominators.
Require exact pair and stratum membership. The decision artifact records all six
F/E/L/Q values, AB/BA arrays and medians, endpoint wins overall and by stratum, all
six RSS/footprint ratios and maxima, and one Boolean per frozen gate.

No output-1 packet runs. v0.598 measured copied first byte at `2463.65` and
`2463.67 ms` for output 1 and 128; output 128 measures the same first-byte endpoint,
prices cold total wall, and follows an independent loaded decode guard.

## Pressure And Validity

Record raw Pageouts, Compressions, compressor stored/occupied pages, Swapouts, swap
occupancy, block I/O, major faults, and all interval deltas before/after cache and
child intervals. Pageouts and Compressions are advisory because v0.600/v0.601 show
that cumulative system-wide events can rise without swap or net compressor growth.

Treat Pageouts, Compressions, and Swapouts as monotonic counters. Negative deltas
are capture regressions and invalidate the interval. Swap occupancy and compressor
stored/occupied pages are gauges; negative gauge deltas are valid.

Invalidate and stop on:

- positive swap-occupancy growth in a cache or child interval;
- any Swapouts growth;
- monotonic-counter regression, missing capture, or parse failure;
- child block input;
- fresh CLI child major faults;
- invalid AC, thermal, performance, memory, identity, command, output, marker, or
  timing state.

Loaded-bench major faults remain recorded but un-gated because v0.596 established a
repeatable executable/runtime floor without block input. No absolute swap-occupancy
zero is required; only growth is causal to the interval.

Host sampling may wait before each launch without creating a child attempt. Once a
child launches, no invalid arm or pair is retried. A pressure or host failure is
`inconclusive`, not a performance kill and not retry authority.

## Decision And Authority

Apply this exclusive precedence:

1. Schema, identity, command, marker, ledger, topology, schedule, or output mismatch:
   implementation/contract defect.
2. Host, VM capture, pressure, I/O, or child-execution invalidity: `inconclusive`.
3. Loaded instability: `inconclusive`.
4. Stable loaded performance miss: `kill`; stop before fresh timing.
5. Valid fresh performance miss: `kill`.
6. Complete conjunction of correctness, loaded, and fresh gates: `go` with
   force-only exact-A3B authority.

Implementation defects discovered by the untimed correctness gate may be repaired
under a new committed source and manifest before any product child launches. After
the first Stage 1 child launches, no repair, child, pair, stage, or packet rerun is
authorized by this contract. Every final status and gate Boolean is computed
mechanically from the complete persisted evidence permitted by the precedence
above.

A pass authorizes only explicit `QWEN_GGUF_PARALLEL_COPY=1` for the frozen A3B
asset and geometry. It does not authorize default-on selection, a CLI flag, other
assets or shards, aliases, conversions, MTP, storage-cold claims, memory savings,
persistent serving, async promotion, or cross-model breadth. Re-rank 27B/A10B
breadth separately after this result.

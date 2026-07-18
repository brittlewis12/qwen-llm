# v0.604 Dense-27B Topology-Preserving Parallel-Copied Loader Pilot

Status: preregistered; implementation and every product child remain unrun.

## Intent

Determine whether v0.603's dense topology-preserving parallel-copy floor transfers
to the production Qwen3.6-27B Q4_K_M loader while preserving copied-storage loaded
prefill, decode, request wall, first-use model-ready latency, and bit-exact target
state.

v0.603 preserves 851 independent exact-sized offset-zero resources and reduces
host materialization by a paired median `965.0405 ms`, from arm medians
`1519.5535` to `557.7565 ms`. B wins 6/6, paired B/A is `0.365706589x`, and
memory is unchanged. The floor does not establish model-load transfer, inference
correctness, loaded parity, first byte, process exit, or product CPU behavior.

This is a warm-filesystem-cache, fresh-process and loaded-process experiment for
one frozen file and one authenticated runtime geometry. It is not storage-cold,
default-policy, hostile-file, broad-family, alias, conversion, MTP, split-shard,
serving, footprint-reduction, asynchronous, or energy-efficiency authority.

## Authority Boundary

The runner authenticates the complete frozen model payload by SHA-256. Production
force mode does not hash 16.8 GB during load. It authenticates structural geometry
and the ordered storage inventory, not weight content.

Therefore:

- experiment authority is for the exact frozen file;
- runtime force authority is for the exact authenticated geometry and inventory;
- descriptor and inventory digests are drift sentinels, not cryptographic content
  fingerprints or hostile-file security boundaries.

Adding a production payload hash would erase much of the cold gain and is outside
this pilot.

## Frozen Dense Profile

Production profile ID is `dense27b-q4km-v1`. The runner authenticates:

- model: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`;
- SHA-256:
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`;
- product prompt:
  `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`;
- prompt SHA-256:
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`;
- architecture label `qwen35`, dense, untied, and no attached MTP payload;
- one shard with mapped-length vector `[16,817,244,384]`;
- descriptor-layout digest `0xd116405fd99f54d9`;
- ordered inventory digest
  `50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07`;
- 851 nonempty all-direct requests over `16,806,250,496` source and resident
  bytes;
- production-auto native quantized token embedding with dtype `Q4_K` and shape
  `[5120,248320]`;
- page size 16,384, required alignment 32, unified memory, device
  `Apple M4 Max`, and Metal maximum buffer length `77,309,411,328`.

The complete architecture tuple is:

```text
kind=dense layers=64 hidden=5120 intermediate=17408 vocab=248320
full_interval=4 q_heads=24 kv_heads=4 attn_head_dim=256
rope_theta=10000000 partial_rotary=0.25
gdn_v_heads=48 gdn_k_heads=16 gdn_head_dim=128 conv=4
experts=0 topk=0 routed_ffn=0 shared_ffn=0 mtp_layers=0
```

The literal schedule is:

```text
cuts          136,377,618
task_counts   136,241,241,233
worker_bytes  4194110464,4214375808,4204933376,4192830848
```

Every partition boundary identity is frozen as
`(request_index,name,shard,source_offset,n_bytes)`:

```text
w0 first (2,output.weight,0,10993888,1042944000)
w0 last  (135,blk.9.ssm_norm.weight,0,4205103840,512)
w1 first (136,blk.9.ssm_out.weight,0,4205104352,21626880)
w1 last  (379,blk.28.attn_qkv.weight,0,8376472160,43008000)
w2 first (378,blk.28.ffn_down.weight,0,8419480160,73113600)
w2 last  (618,blk.46.ffn_down.weight,0,12551299936,73113600)
w3 first (616,blk.46.ffn_gate.weight,0,12624413536,50135040)
w3 last  (844,blk.63.post_attention_norm.weight,0,16817223904,20480)
```

Freeze source, release binaries, runner, this contract, imported helpers, model,
prompt, OS product version/build, device string, page size, maximum buffer length,
and the sealed v0.603 profile evidence:

```text
docs/bench/v0603-dense27b-floor-describe.json
docs/bench/v0603-a3b-floor-describe.json
target/profiles/v0603-dense27b-parallel-copied-floor-p1/decision.json
target/profiles/v0603-dense27b-parallel-copied-floor-p1/packet-complete.json
target/profiles/v0603-dense27b-parallel-copied-floor-p1/artifact-inventory.sha256
```

Their exact expected SHA-256 values are:

```text
v0603-dense27b-floor-describe.json
  4cfeccc3a8110c6e2632e7886eb73c425d815f74f2becd2c8df8c6e76453f7b7
v0603-a3b-floor-describe.json
  833d63fcc628b41cff2691680de562301da6bd1812a7222c9e8b76cc6bb98180
decision.json
  a0361b55d93cee769fc8d4db44eecdf83f3d1c63bdda5ecec07fd3d2edd279ac
packet-complete.json
  bcb19bc3f8776fe16e6459abb5930e020d89f2bd8cf20e3a31a7e567959282ab
artifact-inventory.sha256
  fa87cf2788d58b2f988ce07760ad3619bad835f4abb77dedd238b3dd563e3a9e
```

The immutable runner records hashes before any correctness or product child. Any
drift terminates before scoring.

## Make The Change Easy

Mechanically refactor the A3B-only production parallel-copy path into a private
two-entry `ParallelCopyProfile` table before adding dense authority. A profile
contains:

```text
id
structural matcher
device constraint
authentication strategy
literal schedule and eight complete boundary identities
marker strategy
```

Use explicit strategy variants:

```text
DeviceConstraint::UnifiedAnyName
DeviceConstraint::ExactUnified("Apple M4 Max")

Authentication::A3bRetainedPlan
Authentication::DensePlannerFree

MarkerContract::A3bSchema1
MarkerContract::DenseSchema2
```

The A3B profile preserves the current force path exactly:

- its current structural matcher and unified-memory device contract;
- its retained-plan digest and complete planner-geometry authentication;
- its schedule, topology, ownership, ledger, and provenance checks;
- its exact schema-1 static marker grammar and five dynamic timing fields;
- its ignored marker-sensitive full-state test.

Its complete schedule boundaries are frozen from the hash-pinned independent
v0.603 A3B description:

```text
w0 first (2,output.weight,0,10990048,417177600)
w0 last  (164,blk.8.ffn_gate_exps.weight,0,5392741344,150994944)
w1 first (163,blk.8.ffn_gate_inp.weight,0,5543736288,2097152)
w1 last  (354,blk.19.attn_v.weight,0,11004937952,1114112)
w2 first (366,blk.19.ffn_down_exps.weight,0,11006052064,184549376)
w2 last  (548,blk.29.ffn_gate_exps.weight,0,16450579424,150994944)
w3 first (547,blk.29.ffn_gate_inp.weight,0,16601574368,2097152)
w3 last  (721,blk.39.post_attention_norm.weight,0,22134520800,8192)
```

Do not add an exact device-name or raw architecture-label requirement to A3B.
That would silently narrow v0.602 authority. No new A3B evidence or authority
follows from this refactor.

Keep the production table independent from `gguf_arena_floor`. The floor is a
sealed evidence tool and an independently encoded cross-check. Do not import its
table or refactor the certified command. A v1 profile is immutable; any future
geometry creates a new profile ID. Unit tests reject duplicate IDs and zero or
multiple matches.

## Force Policy And Profile Selection

Keep the existing default-off `QWEN_GGUF_PARALLEL_COPY` parser and conflict
semantics byte-for-byte:

- truthy values force exact profile selection;
- falsy or absent values preserve existing storage policy;
- invalid or non-Unicode values are hard load errors;
- truthy owned-arena or no-copy controls are rejected;
- any explicit `QWEN_GGUF_NO_COPY_PREFAULT`, including false, is rejected;
- `QWEN_NATIVE_QUANT_EMBED` must be absent;
- truthy or invalid `QWEN_MOE_ROUTER_F16` is rejected; absent or explicit false
  is allowed;
- no CLI option or new runtime policy is added.

When forced, execute this order:

1. Resolve all policy conflicts.
2. Generate the production storage-request inventory.
3. Match exactly one static profile using its complete structural contract.
4. For A3B only, invoke and authenticate the current retained planner.
5. Pass the matched profile into planner-free schedule and materialization code.

Zero or multiple matches fail before source resolution, destination allocation or
touch, `buffer.contents`, timing, or marker emission. Dense never invokes or gates
on the retained planner. Unsupported force is a hard error; no candidate failure
falls back to copied storage.

Dense matching requires exact shard-length vector, architecture label and tuple,
attachment state, embedding support/default promotion/selection/dtype/shape,
device contract, page size, maximum buffer length, descriptor digest, inventory
digest, request count, byte count, and all-direct kinds. Do not reuse the weaker
dense no-copy sentinel as parallel-copy authority.

## Generic Materialization Contract

Store the selected static profile in `PlannedParallelCopiedStorage`. Use that same
object for materialization, sequential consumption, finish validation, and marker
formatting.

The generic path must:

1. Require every request to be direct, nonzero, and have resident bytes equal to
   source bytes.
2. Prove every source endpoint fits its exact mapped shard.
3. Sort request indices by `(shard_idx,data_offset,request_index)`.
4. Prove a complete permutation and literal partition union.
5. Validate cuts, task counts, worker bytes, total bytes, and all eight complete
   boundary identities.
6. Start ready wall immediately before the first allocation.
7. Allocate exact-sized resources in canonical request order with production
   copied creation options.
8. Require every length to fit `usize` and the Metal maximum buffer length.
9. Resolve all immutable source slices on the parent.
10. Prove unique Objective-C identities, pairwise-disjoint destination address
    ranges, exact lengths/modes, and all-source-to-all-destination non-overlap.
11. Move every parent-owned task exactly once into sorted order.
12. Run exactly four scoped workers and join every started worker on every path.
13. Drop all task, source, destination, and mutable slices.
14. Construct canonical offset-zero `OwnedWeightReadOnly` tensors.
15. Validate topology and finish endpoint accounting before marker formatting.

No clone, typed view, or GPU consumer may exist before workers join and raw slices
die. A spawn failure, panic, incomplete partition, capture failure, post-copy
validation failure, or counter regression returns a hard load error, emits no
successful marker, and never retries.

`load_direct` remains sequential and records `DirectCopy`. Finish validation
requires the stored profile, schedule reattestation, complete cursor, exact
resource/tensor identity, offset zero, modes, provenance, and the ordinary copied
ledger with zero view, alias, fallback, conversion, or derived counts.

## Dense Schema-2 Marker

Preserve the A3B schema-1 grammar. Dense B emits exactly one schema-2
`[metal-gguf-parallel-copied]` line after endpoint validation. A emits none.
Schema 2 must not contain `plan` or `plan=none`.

The exact ordered static fields are:

```text
schema=2
profile=dense27b-q4km-v1
resources=851 bytes=16806250496
workers=4 cuts=136,377,618 tasks=136,241,241,233
worker_bytes=4194110464,4214375808,4204933376,4192830848
w0_first=2,output.weight,0,10993888,1042944000
w0_last=135,blk.9.ssm_norm.weight,0,4205103840,512
w1_first=136,blk.9.ssm_out.weight,0,4205104352,21626880
w1_last=379,blk.28.attn_qkv.weight,0,8376472160,43008000
w2_first=378,blk.28.ffn_down.weight,0,8419480160,73113600
w2_last=618,blk.46.ffn_down.weight,0,12551299936,73113600
w3_first=616,blk.46.ffn_gate.weight,0,12624413536,50135040
w3_last=844,blk.63.post_attention_norm.weight,0,16817223904,20480
create=shared,default_cache,default
observed=shared,default_cache,tracked
page=16384 alignment=32 max_buffer=77309411328
mapped=16817244384 layout=0xd116405fd99f54d9
inventory=50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07
```

The exact ordered dynamic fields are canonical unsigned decimals:

```text
allocation_us=<u64> source_us=<u64> copy_us=<u64>
binding_us=<u64> ready_us=<u64>
user_cpu_us=<u64> system_cpu_us=<u64> total_cpu_us=<u64>
timer_minor_faults=<u64> timer_major_faults=<u64>
instructions_delta_raw=<u64> cycles_delta_raw=<u64>
```

The emitted marker is one line beginning exactly with
`[metal-gguf-parallel-copied] schema=2`. Join every remaining static and dynamic
field above in the listed order with exactly one ASCII space, using `name=value`.
Emit no leading, trailing, duplicate, omitted, or additional field. Every integer
uses its shortest nonempty base-10 ASCII representation. Lists use commas with no
whitespace. The line ends with one newline outside authoritative ready timing.

The four phases are:

- allocation: vector setup and every exact-sized resource allocation;
- source: timed schedule reattestation, source/range/resource/ownership checks,
  and task construction;
- copy: first scoped spawn attempt through every successful join;
- binding: task/source teardown, tensor construction, and in-endpoint topology
  validation.

Add `libc.workspace` to `qwen-llm`; libc is already a workspace dependency. Use
the v0.603 capture order:

```text
getrusage_before
proc_pid_rusage_v4_before
ready_start
candidate work
ready_end
getrusage_after
proc_pid_rusage_v4_after
marker formatting and emission
```

The CPU intervals slightly bracket ready wall; no overhead is subtracted. Require
nonnegative monotonic counters, `ready_us > 0`,
`total_cpu_us == user_cpu_us + system_cpu_us`, and:

```text
abs(ready_us - (allocation_us + source_us + copy_us + binding_us)) <= 4
```

Derive CPU core-equivalents in the runner. Omit billed and serviced energy fields;
no energy claim follows.

Do not add an A-side production marker. Ordinary copied materialization is
interleaved with model construction, so making it directly comparable would alter
the baseline. Compare dense B endpoint CPU only with the frozen v0.603 B floor as
a cross-packet transfer check. Contemporary A/B CPU evidence comes from complete
process accounting and is not an endpoint-local attribution.

The runner requires exactly one anchored `/usr/bin/time -l` real/user/sys summary
and exactly one occurrence of each literal label:

```text
maximum resident set size
page reclaims
page faults
swaps
block input operations
block output operations
instructions retired
cycles elapsed
peak memory footprint
```

Parse decimal CPU seconds exactly into integer milliseconds before subtraction,
without binary-float accounting claims. Complete-process accounting includes
inference and teardown.

## Correctness Gate

Before product timing, run one ignored release test for the frozen dense file. It
compares production copied A with forced parallel-copied B and audits:

- all `16,806,250,496` candidate bytes against immutable GGUF sources;
- all 851 resource identities, lengths, modes, offsets, provenance, profile,
  schedule, and copied ledger fields;
- compute and blit checked-write rejection;
- bit-exact packed-prefill full logits and complete KV/GDN/conv state;
- identical greedy argmax;
- one forced transition;
- bit-exact continuation full logits and complete KV/GDN/conv state;
- exact A and dense-B marker ordering and grammar.

Bypass tokenization and use the same exact 12-token vector as v0.602:

```text
[7734,264,12654,709,310,12204,279,76938,8240,5199,7638,13]
```

Use capacity 64, position zero, default packed-prefill scratch, full logits, and
the copied argmax as B's forced transition. Every byte is audited and every layer,
state family, output head, and continuation is exercised; duplicating the
419-token product prompt adds cost without storage-correctness coverage.

Audit one B storage object and pass that same object to the ordinary sequential
model builder. Do not rematerialize B. The existing A3B ignored test must continue
to pass its exact schema-1 static grammar and load-line ordering.

Any policy, byte, schedule, topology, marker, log, write-guard, logit, or state
failure is an implementation defect and stops before product children.

The exact recognized production load lines are formed by concatenating each
adjacent quoted fragment with no inserted byte:

```text
POLICY =
  "[metal-load] native quantized token embedding policy: "
  "auto-promoted (Q4_K [5120, 248320])"

LEDGER =
  "[metal-load-ledger] source=851/16806250496 "
  "direct_copy=851/16806250496 direct_view=0/0 direct_alias=0/0 "
  "tail_fallback=0/0 converted=0/0/0 derived=0/0"
```

Each resulting value is one line. A requires exact order `policy,ledger`; B
requires exact order `policy,schema-2 marker,ledger`. No owned, retained, or
no-copy storage marker may appear.

## Product Arms And Immutable Order

- **A - copied**: parallel copy false, owned arena false, no-copy false, no
  prefault variable, native embedding unset.
- **B - parallel copied**: parallel copy true, owned arena false, no-copy false,
  no prefault variable, native embedding unset.

Loaded and fresh stages each use six fresh-process pairs:

```text
AB
BA
BA
AB
AB
BA
```

Processes and stages never overlap. There are no child, pair, stage, or packet
retries. A valid loss is never retryable. A validity failure stops the sole packet
inconclusive.

Before every child, outside scored endpoints:

1. Capture host and VM state.
2. Read exactly the model file size sequentially with one 8 MiB buffer.
3. Wait exactly 30 seconds.
4. Run the frozen host sampler for at most six samples, 30 seconds apart.
5. Require AC power, no thermal/performance warning, and at least 50% parsed
   memory availability.
6. Capture pre-spawn VM state and launch the child once.
7. Capture post-exit host and VM state.

Hash source, binaries, runner, contract, model, prompt, imported helpers, and
sealed v0.603 evidence. Require clean matching source/build/runtime identity and a
normalized environment. Remove every inherited `QWEN_*`, `METAL_*`, `MTL_*`, and
`RUST_LOG` variable, then add only the frozen arm environment.

Every conditioning read recomputes the complete frozen model SHA-256 while reading
exactly the model file size. Require it before launching that child. Recompute and
require the same complete model hash after the last child.

## Durable Sole-Attempt Contract

Atomically reserve the new packet directory and write plus fsync its manifest
before correctness or product work. Never reuse a preexisting artifact path.

A child attempt begins when its complete launch event is appended and fsynced
before `Popen`. The event records stage, stem, exact argv, exact arm environment,
pair/order/position, source/build/runtime identity, and timestamp. Spawn failure
consumes the sole attempt. Append and fsync exactly one completion event with the
return code or spawn failure. Raw stdout, stderr, and timing output are immutable
exclusive-create files.

Persist every parsed attempt row durably. Publish the final decision, complete
artifact SHA-256 inventory, and completion seal only after all permitted work and
validation finish. The completion seal authenticates the decision and inventory.
Without a validated completion seal, the packet is unsealed/incomplete and has no
decision authority.

## Stage 1 - Loaded Noninferiority

Run loaded parity before fresh timing:

```text
qwen-bench decode
prompt=current-reva-n8-interactive-qwen36.txt (419 tokens)
prefill_chunk=1024 kv_capacity=1024 full_logits=true
decode_calls=127 runs=5
```

Each process performs the existing untimed warmup and five fresh-session
repetitions. Score repetition 3-5 medians. For pair `i`:

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
- `RSS_B/RSS_A <= 1.05` and `footprint_B/footprint_A <= 1.05`;
- exact run count, generated output, load contract, timing identity, and ledger.

Complete all 12 children before inspecting performance. Generated output identity
must be global. Evaluate stability first; instability is inconclusive. If stable,
any P/D/R or memory miss kills B before fresh timing.

For the six B candidate markers, require endpoint total-CPU median at most
`2.45 s` and AB/BA medians each at most `2.55 s`. This permits measured v0.603
CPU transfer while rejecting material implementation drift. The frozen floor B
median is `2.2104965 s`; this comparison is cross-packet, not paired product CPU
evidence.

Record complete-process CPU, instructions, cycles, RSS, footprint, faults, and I/O
for both arms. Loaded complete-process CPU is descriptive after the endpoint cap.

## Stage 2 - Fresh Output-128 Product Packet

Run only after Stage 1 passes. Use the frozen 419-token Reva prompt, exactly 128
requested and emitted greedy tokens, chunk 1024, context 1024, prefix cache
disabled, and external spawn through first byte and exit.

Require all 12 children to produce identical stdout, 128 generated tokens, 127
target transitions, no early EOS, one timing row, exact arm load contract, and
clean identity. Timing schema 3 invariant fields are:

```text
request_epoch=first_post_model_load request_index=0
prefix_cache_used=false prefill_chunk_requested=1024
prefill_chunk_effective=419 prompt_tokens=419 max_context_tokens=1024
ttft_endpoint=stdout_flush_complete decode_policy=greedy_argmax
stop_reason=token_limit generated_tokens=128 transition_count=127
```

Timing identity means these invariant fields match; measured values differ.
Timestamp every nonempty bounded stdout read. EOF is not a last-byte timestamp.
Record first byte, last byte, exit, and final-output-to-exit wall.

After all 12 valid children complete, define:

```text
F[i] = spawn_to_first_byte_A / spawn_to_first_byte_B
E[i] = spawn_to_exit_A       / spawn_to_exit_B
L[i] = runtime_and_model_load_A - runtime_and_model_load_B
Q[i] = runtime_and_model_load_B / runtime_and_model_load_A

PF[i] = first_prefill_B / first_prefill_A
TT[i] = model_ready_ttft_B / model_ready_ttft_A
G[i]  = generation_wall_B / generation_wall_A
RQ[i] = model_ready_request_B / model_ready_request_A
CPU[i] = complete_process_cpu_B - complete_process_cpu_A
```

Bind those names to exact timing fields:

```text
first_prefill = timing["prefill_ms"]
model_ready_ttft = timing["ttft_ms"]
generation_wall = timing["generation_ms"]
model_ready_request = timing["total_request_ms"]
complete_process_cpu = /usr/bin/time user_ms + sys_ms
```

Every timing/wall input is finite and positive; every ratio denominator is
positive. For six values, median is the arithmetic mean of the third and fourth
sorted values. A three-value stratum median is its middle sorted value. Compute
medians from paired ratios or paired differences, never from ratios or differences
of arm medians. Complete-process CPU is integer milliseconds before subtraction.

Cold gates:

- median `F >= 1.25x`; AB and BA medians each `>=1.20x`;
- B wins first byte in at least 5/6 pairs and 2/3 in each stratum;
- median `E >= 1.08x`; AB and BA medians each `>=1.05x`;
- B wins exit in at least 5/6 pairs and 2/3 in each stratum;
- median `L >=750 ms`; AB and BA medians each `>=600 ms`;
- median `Q <=0.70`; AB and BA medians each `<=0.80`;
- maximum paired RSS and footprint B/A are each `<=1.05`.

Separate model-ready guards:

- overall medians `PF <=1.02` and `TT <=1.02`;
- AB and BA medians for PF and TT are each `<=1.03`;
- overall medians `G <=1.01` and `RQ <=1.01`;
- AB and BA medians for G and RQ are each `<=1.02`.

CPU guards:

- B candidate endpoint total-CPU median `<=2.45 s`;
- AB and BA B endpoint medians each `<=2.55 s`;
- median complete-process `CPU <=0.90 s`;
- AB and BA complete-process CPU-delta medians each `<=1.00 s`.

Instructions and cycles are non-gating diagnostics. CPU ratios are reported but
do not replace absolute deltas. No energy claim is available.

The prior copied output-128 envelope is about `3532.5 ms` first byte,
`8744.5 ms` exit, and `1616.0 ms` runtime/load. Full floor transfer predicts about
`1.376x`, `1.124x`, and `0.403x` B/A respectively. The gates permit roughly
200-320 ms of transfer loss and remain well beyond observed floor spread. Win
counts and strata are engineering consistency guards, not statistical
significance claims.

The 1-2% model-ready bounds are hard product guards near the packet's practical
MDE, not powered equivalence claims. No endpoint or CPU success can rescue a
model-ready miss.

## Pressure And Validity

Record raw Pageouts, Compressions, compressor stored/occupied pages, Swapouts,
swap occupancy, block I/O, major faults, and all cache/child interval deltas.
Pageouts and Compressions are advisory cumulative counters.

Stop inconclusive on:

- positive swap-occupancy growth in any cache or child interval;
- any Swapouts growth;
- positive compressor stored or occupied gauge growth;
- host/VM pressure-counter regression, missing capture, or parsing failure;
- child block input;
- fresh child major faults;
- invalid AC, thermal, performance, or memory state;
- host/VM capture failure, child spawn or nonzero-exit execution failure, or
  child-attributed I/O failure.

Negative swap-occupancy and compressor gauge deltas are valid. Loaded major faults
remain recorded but ungated because the existing harness has a repeatable
executable/runtime floor without block input. No absolute zero swap occupancy is
required.

Once a child launches, no invalid arm or pair is retried. Host sampling may wait
before launch without creating an attempt. A pressure or host failure is
`inconclusive`, not a performance kill.

## Decision And Authority

Apply this exclusive precedence:

1. Malformed or missing correctness, exact command, successful-child output,
   timing row, marker, ledger, schema, topology, schedule, identity, or CPU-
   accounting grammar: implementation/contract defect.
2. Host/VM capture failure, external pressure, spawn/nonzero-exit execution
   failure, or child-attributed I/O: inconclusive.
3. Loaded instability: inconclusive.
4. Stable loaded performance, endpoint CPU, or memory miss: kill before fresh.
5. Valid fresh cold, model-ready, CPU, or memory miss: kill.
6. Complete conjunction: GO with force-only dense structural-profile authority.

Artifact-publication failure leaves the packet unsealed/incomplete when no durable
decision and completion seal can be published. It creates no rerun authority.

Implementation defects found before the first loaded child may be repaired under
a new committed source and manifest. After the first loaded child launches, no
repair, child, pair, stage, or packet rerun is authorized by this contract.

A pass authorizes only explicit `QWEN_GGUF_PARALLEL_COPY=1` on the authenticated
`dense27b-q4km-v1` geometry and inventory. It does not authorize default selection,
payload-content identity at runtime, a new CLI flag, other dense assets, A3B
changes, aliases, conversions, MTP, split shards, storage-cold use, memory savings,
serving, concurrent loading, asynchronous promotion, or energy efficiency.

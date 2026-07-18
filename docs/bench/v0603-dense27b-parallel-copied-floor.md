# v0.603 Dense-27B Topology-Preserving Parallel-Copy Floor

Status: preregistered; the candidate path and packet remain unrun.

## Intent

Determine whether v0.602's topology-preserving parallel-copy mechanism transfers
to the exact dense Qwen3.6-27B Q4_K_M copied inventory before adding any dense
production-loader path.

v0.599 reduced the A3B materialization wall from `2066.701` to `742.676 ms` while
preserving independent offset-zero resources. v0.602 then transferred that floor
to a bit-exact force-only A3B loader at `2.06292x` process-cold first byte with
loaded parity. Neither result authorizes dense breadth. Byte scaling suggests a
dense saving near one second, but that is a prior rather than evidence.

This is a warm-filesystem-cache, fresh-process host-materialization floor. It is
not model-load, first-byte, loaded-performance, storage-cold, serving,
concurrent-loader, energy-efficiency, or product-timing evidence. A3B default
admission is deferred, not superseded.

## Frozen Dense Profile

Materializing invocations require explicit profile `dense27b-q4km-v1`. The runner
authenticates this exact asset:

- model: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`;
- SHA-256:
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`;
- architecture string `qwen35`, untied embeddings, no attached MTP payload;
- one shard with mapped length `16,817,244,384` bytes;
- descriptor-layout digest `0xd116405fd99f54d9`;
- ordered inventory digest
  `50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07`;
- 851 nonempty all-direct requests over `16,806,250,496` bytes;
- production-auto native quantized token embedding;
- page size 16,384, required alignment 32, unified memory, device
  `Apple M4 Max`, and Metal maximum buffer length `77,309,411,328`;
- retained-plan digest
  `eb367575c614faff02a45dfe9e5c55d0746ad07682df4ab895397f688f171f96`.

The retained-plan digest is sealed descriptive metadata only. It is not a profile
admission, materialization, schedule, correctness, or decision gate because the
planner drives neither arm.

The complete bound architecture tuple is:

```text
kind=dense layers=64 hidden=5120 intermediate=17408 vocab=248320
full_interval=4 q_heads=24 kv_heads=4 attn_head_dim=256
rope_theta=10000000 partial_rotary=0.25
gdn_v_heads=48 gdn_k_heads=16 gdn_head_dim=128 conv=4
experts=0 topk=0 routed_ffn=0 shared_ffn=0 mtp_layers=0
```

The profile requires the literal four-worker schedule:

```text
cuts          136,377,618
task_counts   136,241,241,233
worker_bytes  4194110464,4214375808,4204933376,4192830848
max/ideal     1.0030496234726596
max/min       1.0051385235372128
```

Every boundary identity is frozen as
`(request_index,name,shard,source_offset,n_bytes)`:

```text
worker 0 first (2,output.weight,0,10993888,1042944000)
worker 0 last  (135,blk.9.ssm_norm.weight,0,4205103840,512)
worker 1 first (136,blk.9.ssm_out.weight,0,4205104352,21626880)
worker 1 last  (379,blk.28.attn_qkv.weight,0,8376472160,43008000)
worker 2 first (378,blk.28.ffn_down.weight,0,8419480160,73113600)
worker 2 last  (618,blk.46.ffn_down.weight,0,12551299936,73113600)
worker 3 first (616,blk.46.ffn_gate.weight,0,12624413536,50135040)
worker 3 last  (844,blk.63.post_attention_norm.weight,0,16817223904,20480)
```

Freeze the source, release binary, runner, this contract, imported helpers, model,
OS product version/build, device string, page size, maximum buffer length, complete
profile, and the committed clean-source metadata evidence:

```text
docs/bench/v0603-dense27b-floor-describe.json
docs/bench/v0603-a3b-floor-describe.json
```

The runner hashes both evidence files. Raw describe output is retained, but packet
validation projects only the explicitly frozen non-planner fields above. A planner
digest or view/window/fallback field cannot participate in profile matching, drift
validation, or decision status. Any admitted-field drift terminates before scoring.

## Exact Profile Refactor

Replace the A3B-only materialization constants in `gguf-arena-floor` with an exact
two-entry floor-profile table:

- `a3b-q4km-v1` preserves v0.599's authenticated profile;
- `dense27b-q4km-v1` is the sole candidate profile in this packet.

`--profile` is mandatory for materialization. The command independently proves
that the requested profile matches exactly once. Missing, unsupported, mismatched,
or ambiguous profiles fail before payload resolution, destination allocation or
touch, `buffer.contents`, and authoritative timing.

Profile materialization accepts only `copied` and `parallel-copied`. The legacy
`arena-serial` and `arena-four` arms are rejected before planner invocation or
payload access for both profiles. Their historical packets remain valid only at
their frozen sources; v0.603 does not preserve a current arena-materialization path.

Generic `--describe` remains metadata-only and reports:

```text
materialization_supported=<bool>
matched_profile=<profile|null>
computed_schedule=<metadata-only minimax schedule>
frozen_schedule=<authenticated literal schedule|null>
```

The minimax optimizer may run only for generic description. A materializing path
constructs and validates the literal profile schedule without invoking it. An
unsupported geometry may be described but is never called authenticated.

Copied and parallel-copied materializing paths never invoke the retained-storage
planner. Generic description may invoke it and report its raw output, but planner
failure or drift cannot reject either materializing arm.

The preserved A3B profile retains:

```text
architecture=qwen35moe mapped=22134528992 layout=0x5ae645df5cf7d568
inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5
requests=733 bytes=22123538944 cuts=155,359,539
tasks=155,204,180,194
worker_bytes=5532746240,5462315776,5595522304,5532954624
```

Its complete architecture, attachment, partition-boundary, and device facts are
the clean `c05c2a1` describe result and must be encoded in the profile. This is
regression preservation only, not new A3B evidence or authority.

## Arms

- **A - production copied**: resolve requests in canonical order and call
  `MetalContext::buffer_from` once per request. Retain 851 independent exact-sized
  resources and construct 851 `(buffer, offset=0, length)` bindings.
- **B - parallel copied**: allocate the same 851 exact-sized resources in canonical
  request order, resolve and validate all immutable sources on the parent, and copy
  the four literal source-order partitions through scoped workers. Construct the
  same canonical offset-zero binding table after all workers and raw slices die.

B must not use planner windows, aliases, fallback resources, nonzero offsets,
file-backed storage, conversion, GPU work, asynchronous work, a dynamic schedule,
or a new dependency. A failure after B allocation begins never falls back to A.

## Ownership And Failure Contract

Port the v0.602 production ownership proof rather than reusing v0.599's saved raw
addresses. Before creating any mutable destination slice, require:

- every descriptor endpoint fits its mapped shard;
- nonzero exact source and destination lengths;
- non-null destination contents;
- unique Objective-C resource identities;
- pairwise-disjoint destination address ranges;
- exact source/destination length equality;
- every immutable source range is disjoint from every destination range;
- Shared/DefaultCache/Tracked observed modes;
- no resource clone, tensor view, command buffer, or GPU consumer exists;
- complete frozen schedule identity and one writer per request.

Create destination slices only from exclusive `resources.iter_mut()` borrows. Put
each task in `Option`, move it exactly once into sorted order, and require no
remainder. Do not access or clone resources until task and source vectors die.

Use `Builder::spawn_scoped`. Join every successfully started worker on all paths.
A spawn failure, invalid partition, or worker panic is an implementation error,
emits no successful result, and has no fallback or retry authority.

## Timing And Verification

The endpoint creates its resource, binding, source, destination-range, and task
vectors with capacity only after `ready_start`; no arm-local container setup is
moved outside the endpoint. `ready_wall` begins immediately before A's first source
resolution or B's first allocation. It ends after all allocation, source and safety
validation, task construction, worker spawn/copy/join, resource retention,
canonical binding construction, and in-endpoint physical validation.

The boundary is intentionally composite and asymmetric:

- A charges source resolution and 851 production `buffer_from` calls;
- B charges 851 allocations, all ownership validation, task construction,
  spawn/copy/join, and bindings;
- B's safety checks remain charged because they are required for valid unsafe code.

Use contiguous milestones from one `Instant`. B reports allocation, source, copy,
and binding microseconds and requires:

```text
abs(ready_us - (allocation_us + source_us + copy_us + binding_us)) <= 4
```

The phases are literal:

- allocation: resource-vector setup and all exact-sized resource allocations;
- source: source-vector/range/task setup plus source, resource, mode, schedule,
  ownership, and overlap proofs;
- copy: first scoped-worker spawn attempt through every successful join;
- binding: task/source teardown, canonical bindings, and in-endpoint physical
  validation.

After the endpoint, both arms verify every complete payload byte, resource identity,
length, mode, canonical binding, offset, and request association. Verification and
teardown cannot improve the score. Complete-child RSS and footprint from
`/usr/bin/time -l` include untimed verification and teardown; they are not
timer-local memory measurements.

## CPU And Energy Accounting

Capture usage snapshots in this exact order:

```text
getrusage_before
proc_v4_before
ready_start
arm work
ready_end
getrusage_after
proc_v4_after
```

The usage intervals slightly bracket authoritative `ready_wall`; usage-call overhead
is never subtracted from wall. Record nonnegative `getrusage(RUSAGE_SELF)` deltas:

```text
user_cpu_us system_cpu_us total_cpu_us
cpu_per_wall = total_cpu_us / ready_us
```

`cpu_per_wall` is aggregate process CPU core-equivalents, not utilization of one
specific core or worker.

Capture `proc_pid_rusage` v4 in the exact bracketing order above and record raw
nonnegative deltas for instructions, cycles, billed energy, and serviced energy.
Capability and monotonicity are mandatory schema conditions, but these observations
are not performance gates.

The v4 energy fields are opaque process-accounting counters. They may be zero and
do not measure joules, whole-system, DRAM, GPU, or concurrent-loader energy. No
energy-efficiency claim follows. A later loader pilot must quote and disposition
any adverse CPU or energy-accounting result rather than silently omit it.

## Immutable Process Packet

Run six fresh-process pairs in this exact order:

```text
AB
BA
BA
AB
AB
BA
```

Processes never overlap. There are no child, pair, packet, validity, or
performance-triggered retries. Prelaunch host waiting is not a child attempt. A
child attempt begins when its launch event is fsynced before `Popen`; spawn failure
therefore consumes the sole attempt and terminates inconclusive. Complete all six
valid pairs before calculating performance.

Before manifest creation, remove every inherited `QWEN_*`, `METAL_*`, `MTL_*`, and
`RUST_LOG` variable and prove none remains. Every child receives that exact
normalized environment. Materializing commands additionally require
`QWEN_NATIVE_QUANT_EMBED` to be absent; profile admission labels native embedding
`production-auto-promoted` only after proving support, default promotion, and
environment absence. The runner and endpoint do not accept a forced-embedding
Boolean as equivalent evidence.

Before every child:

1. Run at most six host samples, 30 seconds apart, before cache conditioning.
   Require AC power, no thermal or performance warning, and at least 50% parsed
   memory availability. Exhaustion stops inconclusive without a child attempt.
2. Capture VM state.
3. Read exactly the model file size sequentially through one 8 MiB buffer while
   recomputing the frozen SHA-256.
4. Wait exactly 30 seconds.
5. Run the same at-most-six, 30-second host sampler. Exhaustion stops inconclusive
   without a child attempt.
6. Capture pre-spawn VM state and validate the cache interval.
7. Recheck non-model identity and fsync the launch event.

After every child, capture host and VM state before parsing output. Persist immutable
raw stdout/stderr, exact command/environment, before/after evidence, fsynced
attempt row, completion event, and artifact hashes. Recompute the final model hash
after the last child.

Atomically reserve and fsync the packet directory plus manifest before any launch.
Append and fsync each launch/completion event during execution. After execution,
publish decision, inventory, and completion files through temporary files, ordered
renames, and directory fsync. These are ordered durable publications, not one
filesystem transaction. Only a present, validated completion seal converts the
artifact set into a decision; otherwise it is an unsealed/incomplete packet with no
authority.

## Pressure And Validity

Valid host state means AC power, no thermal or performance warning, and at least
50% parsed memory availability. Require it before cache, immediately before launch,
and immediately after child exit. For cache and child intervals:

- Pageouts, Compressions, and Swapouts are monotonic counters;
- missing capture, parse failure, or negative counter delta is invalid;
- Pageouts and Compressions growth is advisory;
- any Swapouts growth is invalid;
- positive swap-occupancy growth is invalid;
- positive compressor stored or occupied gauge growth is invalid;
- negative gauge deltas are valid.

Child block input or nonzero timer-local `ru_majflt` is invalid. Whole-process major
faults remain separately recorded and are not substituted for the timer-local gate.
Pressure invalidity is `inconclusive-pressure`, never a candidate performance kill.

## Decision

For valid pair `i`, define:

```text
d_i = A_i - B_i
q_i = B_i / A_i
D   = median(d_i)
Q   = median(q_i)
W   = count(B_i < A_i)
```

For six values, median means the arithmetic mean of the third and fourth sorted
values. `Q` is the median of paired ratios, not the ratio of arm medians. Compute
AB and BA savings and wins separately.

B passes only if:

- `D >= 750 ms`;
- `Q <= 0.70`;
- `W >= 5/6`;
- AB median saving and BA median saving are each at least `600 ms`;
- B wins at least 2/3 pairs in each order stratum;
- maximum paired complete-child `RSS_B/RSS_A <= 1.05`;
- maximum paired complete-child `footprint_B/footprint_A <= 1.05`;
- every identity, geometry, schedule, timing, CPU-counter, byte, topology, mode,
  pressure, host, and artifact-seal condition passes.

The win-count and stratum checks are engineering consistency guards, not powered
statistical significance or equivalence claims. The absolute saving gate is the
economic decision; the ratio is a drift guard. Do not project product TTFT from
this floor.

Apply exclusive status precedence:

1. Profile, geometry, schedule, schema, timing reconciliation, CPU instrumentation,
   byte, topology, binding, or mode mismatch: implementation/contract defect.
2. Host, VM capture, pressure, I/O, spawn, child-execution, or artifact-publication
   invalidity: inconclusive with no authority when a decision can be durably
   published; otherwise the packet remains unsealed/incomplete with no authority.
3. Complete valid packet with a performance miss: kill.
4. Complete conjunction of every gate: go.

No same-contract P2 follows a launched inconclusive packet. Any rerun requires a new
version, new preregistration, and an explicit changed premise. A prelaunch
implementation defect discovered before any launch event may be repaired under a
new committed source and manifest.

## Authority

A pass authorizes only implementation of one separately preregistered, force-only
`dense27b-q4km-v1` loader pilot. That pilot must independently prove exact payload
bytes, full target state, loaded 1% noninferiority, first-prefill behavior,
first-byte wall, exit wall, CPU accounting, and memory parity.

No result authorizes a dense production path, default selection, another dense
quant or size, A3B policy change, A10B, split shards, MTP, aliases, conversions,
retained storage, storage-cold claims, serving, asynchronous promotion, concurrent
loading, or energy-efficiency claims.

Adversarial design review: `cx ask` session
`019f7363-fda4-72f2-b51f-882e2526c2c5`.

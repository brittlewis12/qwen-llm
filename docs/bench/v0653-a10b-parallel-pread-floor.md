# v0.653 A10B Topology-Preserving Parallel-Pread Floor

Status: stage-0 preregistration. This revision authorizes only benchmark-local
embedding-policy/describe instrumentation and one metadata-only describe. It
authorizes no A10B floor profile or materialization child. A separately reviewed
freeze commit must bind the raw describe digest, complete profile, literal W4
schedule, and exactly one ABBA packet before payload timing.

## Intent

Decide whether the exact local split Qwen3.5 122B-A10B Q4_K_XL asset has a
large enough host-population prize to justify one later force-only product
pilot. Compare the existing sequential production-shaped copied primitive with
one allocate-first, four-worker, retained-descriptor pread bundle while
preserving independent exact-sized offset-zero Metal resources.

This is a target-file-cache-warm, fresh-process host-population floor. It is not
model load, inference, first byte, loaded performance, storage-cold behavior,
serving, concurrency, native-embedding default admission, or production policy.

## Premise Authority

For this floor only, existing evidence authorizes a benchmark-local forced
native Q8_0 token embedding:

- v0.538 proves the exact A10B asset's native embedding preserves its 128-token
  stream, gives loaded `pp512/tg128` parity of `1.00004/1.00106`, and reduces
  embedding residency from `3,051,356,160` to `810,516,480` bytes.
- v0.594 proves the split asset's native all-direct inventory can create two
  retained Metal windows, survive loader teardown, and return correct sampled
  bytes from both windows.

Those results do not admit native A10B in production. The floor never executes
Q8 lookup, logits, model state, or a full request. A pass can authorize only one
separately preregistered force-only full-state/product pilot.

## Metadata-First Freeze

Before any payload timing, extend `gguf-arena-floor` with an explicit
`--embedding-policy production-auto|force-native-if-supported` option. The
default remains `production-auto`. The forced choice must:

- call the existing structural support predicate;
- emit `bench-force-native-if-supported` rather than an automatic-policy label;
- reject any inherited `QWEN_NATIVE_QUANT_EMBED` value;
- change no runtime selector or model-loading default;
- permit metadata-only `--describe` before an A10B profile exists.

Run one metadata-only describe under the forced policy. It may parse headers,
create a Metal context, and compute schedules, but it must not resolve tensor
payload bytes, allocate model-sized resources, call `buffer.contents`, or issue
a GPU command. Preserve its raw JSON as one content-addressed record. A later
freeze commit must cite that digest and bind its exact admitted fields in a
committed profile before implementing or running either materialization arm. No
timing observation may precede that profile-freeze commit or its separate review.

Known immutable fields are:

- profile ID `a10b-q4xl-v1`;
- model first shard:
  `/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/`
  `Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf`;
- shard SHA-256 values, in order:
  `467c9bd92ea518539cf75bf5a5fbfbd35e9a0b40d766ccaa67bf120e12041df3`,
  `ecdbd42d43b0df9fa0ef9a584e09e95a43966ef03a122aba0b87a99d44d9ad98`,
  `13300e0f059e6fa21aa0fabde2a554f9deea366c0e54f268045769b214b28c97`;
- architecture `qwen35moe`, untied embeddings, no MTP, and
  `mtp_n_hidden_layers=0`;
- 48 layers, hidden size 3072, `intermediate_size=0`, vocabulary 248,320, and
  full-attention interval 4;
- Q/KV heads `32/2`, attention head dimension 256;
- RoPE theta `10,000,000`, partial rotary factor `0.25`;
- GDN V/K heads `64/16`, head dimension 128, convolution width 4;
- 256 experts, top-8, routed/shared FFN widths `1024/1024`;
- native Q8_0 embedding shape `[3072, 248320]`;
- native embedding bytes `810,516,480`; baseline F32 bytes `3,051,356,160`;
  exact removed bytes `2,240,839,680`;
- shard mapped lengths
  `[10,943,552, 49,640,779,424, 27,378,273,056]`;
- mapped-length sum `77,029,996,032` and mapped-minus-payload bytes
  `10,999,296`;
- 16,384-byte page counts `[668, 3,029,833, 1,671,038]`, totaling `4,701,539`,
  for the launch-residency proof;
- descriptor-layout digest `0x3eb290915bec2041`;
- 879 nonempty all-direct requests totaling `77,018,996,736` bytes;
- 16,384-byte host pages, 32-byte binding alignment, unified memory, device
  `Apple M4 Max`, and Metal maximum buffer length `77,309,411,328`.

The freeze commit must add the ordered request-inventory digest and the complete
literal four-worker `minimax-contiguous-v1` schedule: cuts, task counts, worker
bytes, maximum-to-ideal ratio, and first/last
`(request_index,name,shard,source_offset,n_bytes)` identity for every partition.
Cuts and integer worker bytes are schedule identity; maximum-to-ideal is derived
reporting. Materialization must be driven only by the literal schedule. Any
retained optimizer recomputation is equality-only diagnosis and cannot select or
change the materialization schedule. The later separately reviewed profile-freeze
commit must explicitly authorize replacing the current dynamic materialization
selector for this exact A10B W4 profile.

## Arms

- **A - copied**: in canonical request order, call `MetalContext::buffer_from`
  once per source. Retain 879 independent exact-sized Shared resources and bind
  every tensor at offset zero.
- **B - parallel pread**: allocate the same 879 resources in canonical request
  order, validate every destination and retained source descriptor, and pread the
  four literal contiguous source-order partitions through scoped workers. Build
  the same canonical offset-zero bindings after every worker and raw slice dies.

B is the complete bundle: allocate-first behavior, W4 partitioning, retained
descriptor pread, and exact binding validation. A pass does not attribute the
gain to pread alone. Do not add parallel mmap copy, another worker count, MTLIO,
private storage, no-copy views, aliases, conversions, or a fallback arm.

Neither arm may issue a GPU command, inference, blit, or asynchronous operation.
Creating Shared Metal resources is allowed; call the endpoint
`host-population/no-GPU-command`, not CPU-only inference.

## Timing And Verification

`ready_us` begins immediately before the first arm-local source resolution or
destination allocation. It ends only after resource creation, source and
ownership validation, population, worker joins, task/source teardown, canonical
binding construction, and topology validation.

B reports contiguous allocation, source, copy, and binding intervals from one
clock and requires their sum to reconcile with `ready_us` within 4 microseconds.
Capture timer-local `getrusage` and `proc_pid_rusage` immediately around the same
endpoint. Report user/system/total CPU, minor and major faults, block input,
swaps, instructions, cycles, and available energy counters. The later separately
reviewed profile-freeze commit must explicitly authorize adding `ru_inblock` and
`ru_nswap` capability samples and JSON deltas to the floor's timer-local `Usage`;
missing, regressing, or unparsable controlling counters are not interpreted as
zero. Unphased whole-process counters remain descriptive.

After timing and counters close, verify all `77,018,996,736` bytes against the
retained GGUF sources plus every resource identity, exact length, mode, binding,
offset, and request association. Then drop all resources and require Metal
allocation snapshots `before`, `ready`, and `after_drop`, with
`after_drop <= before`. Verification and teardown cannot improve the score and
may not enter `ready_us`, but they remain inside complete-child RSS and footprint.

No-GPU-command is a sealed-build invariant, not a runtime event counter. The
authenticated arm must be `copied` or `parallel-pread`, `blit_population` must be
absent, and those exact paths may contain no command-buffer, encoder, commit, or
wait call.

## Immutable Packet

Run four sole-attempt fresh children in exact order:

```text
A B B A
```

This creates one AB pair and one BA pair. Processes never overlap. There are no
retries, replacements, extra arms, or pooled predecessor observations. A child
attempt begins when its launch event is durably recorded before spawn.

Before every child:

1. Require AC power, no thermal/performance warning, normal memory pressure, no
   competing model process, and no positive swap-occupancy growth.
2. Emit `recommendedMaxWorkingSetSize`, `currentAllocatedSize`, and their checked
   difference. Require
   `recommendedMaxWorkingSetSize - currentAllocatedSize >= 85,608,931,328`.
   Any positive `process_limit_remaining_bytes` must also be at least
   `85,608,931,328`. Apply this before packet reservation and repeat it inside
   each child immediately before timing/allocation.
3. Capture VM, compressor, process, source pathname, and complete path-bound file
   stamps: device, inode, size, `mtime_sec`, `mtime_nsec`, `ctime_sec`, and
   `ctime_nsec`.
4. Wait at least 120 seconds after prior packet activity.
5. Read and SHA-256 all three exact path-bound shard files, require the frozen
   hashes, then require exact `mincore` page totals and complete launch residency
   for every shard.
6. Recheck path-bound stamps and launch within five seconds of the residency
   proof.

The later separately reviewed profile-freeze commit must authorize one separate
non-payload live headroom probe, or an equivalent runner-local Metal query,
immediately before packet reservation. It is distinct from the sole stage-0
describe, is not an attempt, allocates no payload, and is retained in the
aggregate seal.

Full shard residency is a launch condition, not a simultaneous-residency promise.
The kernel may reclaim consumed source-cache pages while the 77 GB destination is
populated. Requiring source plus destination to remain fully resident would
exceed the 128 GiB host envelope and is not part of validity.

Each child must expose and revalidate the complete stamp of each descriptor
retained by its `GgufFile`, and used by B for pread, immediately before timing and
again after full verification. The later separately reviewed profile-freeze
commit must explicitly authorize one narrow read-only GGUF shard-stamp accessor
and revalidation seam; parent path evidence cannot substitute for child
descriptor identity.

The child environment removes every inherited `QWEN_*`, `METAL_*`, `MTL_*`, and
`RUST_LOG` key. The embedding policy is a command argument, not an environment
override. Source, runtime, and release-build identities must be clean and equal.
The runner, contract, executable, model identities, command, environment, host
facts, attempts, output, and seals are content-addressed.

Capture controlling system samples immediately before conditioning, immediately
before spawn, and immediately after child exit. Evaluate the preconditioning and
child intervals separately. Both require system-wide `Swapouts` delta zero and
swap-occupancy, compressor-stored, and compressor-occupied growth `<= 0`.
After each child, capture host and pressure state before parsing its row. Full
verification may change later cache state, so every next child repeats complete
conditioning. Whole-process block input and CPU remain descriptive; only the
timer-local endpoint controls the materialization decision.

## Validity And Decision

Every child must prove:

- the exact frozen profile, native floor policy, source stamps, and W4 schedule;
- 879 distinct exact-sized Shared/DefaultCache/Tracked resources at offset zero;
- `77,018,996,736` timed and verified payload bytes;
- zero timer-local `ru_inblock` and `ru_nswap`;
- zero system-wide `Swapouts` growth, no positive swap-occupancy growth,
  no memory-pressure warning, and no positive net compressor stored/occupied
  growth;
- nonnegative timer-local CPU, fault, instruction, cycle, and energy counters;
- all three mandatory allocation snapshots and `after_drop <= before`;
- the sealed no-GPU-command implementation invariant;
- valid artifacts and sole-attempt lifecycle.

Pageouts, cumulative Compressions, and unphased major faults are recorded but are
not generic invalidity gates. Energy counters are descriptive and may be zero.

For each pair define `d = A.ready_us - B.ready_us` and `q = B.ready_us /
A.ready_us`. GO requires:

- both pair savings `d >= 1,500,000 us`;
- B wins both pairs;
- each paired timer-local `CPU_B/CPU_A <= 1.50`;
- each paired complete-child `RSS_B/RSS_A <= 1.05` and
  `footprint_B/footprint_A <= 1.05`;
- every CPU, RSS, and footprint denominator is positive;
- every structural, correctness, validity, and seal condition above.

Two pairs are an engineering floor and order-reversal guard, not a powered effect
estimate or confidence interval. Report both pairs and their descriptive median;
do not infer product first-byte movement from the floor.

Apply exclusive precedence:

1. Malformed, missing, regressing, or nonreconciling counters, or any profile,
   source, schedule, topology, byte, timing, or implementation mismatch:
   implementation/contract defect with no performance authority.
2. Correctly measured block input, swaps, pressure growth, failed conditioning,
   host invalidity, spawn/execution failure, or publication failure: inconclusive
   with no authority.
3. Complete valid packet missing the CPU, RSS, footprint, or 1.5-second gate:
   KILL this exact A10B copied-versus-W4-pread floor and advance to grammar.
4. Complete conjunction: GO with the narrow authority below.

Every non-GO disposition returns the active queue to grammar. A defect may
motivate a separately ranked successor preregistration, but v0.653 neither
authorizes that successor nor permits a replay or replacement attempt.

Any preflight headroom failure means this copied A10B floor is economically
infeasible on the host and returns the active queue to grammar without allocating
the payload. An in-child headroom failure after its durable launch record is
invalid, receives no retry, and returns the queue to grammar. Any repair after a
launched packet requires a new successor preregistration; v0.653 itself is never
replayed.

The sealed packet requires at least six logical content-addressed records: one
raw metadata describe, four immutable attempt bundles, and one aggregate
decision/completion seal. Each attempt bundle contains or references its launch
lifecycle, command/environment, conditioning and host samples, raw stdout,
`/usr/bin/time -l` stderr, parsed row, and content hashes. No physical-file count
is a validity gate.

## Authority

GO authorizes only one separately preregistered exact-asset, force-native,
force-only A10B full-state/product pilot. That pilot must independently prove
complete inference state, loaded noninferiority, first-prefill and first-byte
behavior, pressure, CPU, and exact selection/fallback semantics.

No outcome directly authorizes native-embedding default promotion, A10B Auto,
runtime loader code, another asset or quant, reusable/server loads, storage-cold
claims, serving, concurrency, MTLIO, no-copy storage, or energy-efficiency claims.

Adversarial design review: `cx ask` session
`019fb49b-9b59-7fe0-a969-a3496c17ee82`.

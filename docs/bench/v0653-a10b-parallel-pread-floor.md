# v0.653 A10B Topology-Preserving Parallel-Pread Floor

Status: profile freeze. The sole stage-0 metadata describe is complete. This
revision binds its content digest, the exact A10B profile, the literal W4
schedule, and the sole ABBA packet. It authorizes implementation and review of
the frozen measurement seams, but no payload observation may precede a clean
implementation commit and separate pre-run review against this contract.

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

Before any payload timing, stage 0 extended `gguf-arena-floor` with an explicit
`--embedding-policy production-auto|force-native-if-supported` option. The
default remained `production-auto`. The forced choice:

- called the existing structural support predicate;
- emitted `bench-force-native-if-supported`, not an automatic-policy label;
- rejected any inherited `QWEN_NATIVE_QUANT_EMBED` value;
- changed no runtime selector or model-loading default;
- permitted metadata-only `--describe` before an A10B profile existed.

Exactly one metadata-only describe ran under the forced policy. It parsed
headers, created a Metal context, and computed schedules, but did not resolve
tensor payload bytes, allocate model-sized resources, call `buffer.contents`,
or issue a GPU command. Its raw JSON is preserved as one content-addressed
record. This freeze cites that digest and binds its exact admitted fields in a
committed profile before either materialization arm is implemented or run. No
timing observation preceded this profile freeze or its separate review.

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

The ordered request-inventory digest and complete literal four-worker
`minimax-contiguous-v1` schedule are frozen below. Cuts and integer worker bytes
are schedule identity; maximum-to-ideal is derived reporting. Materialization
must be driven only by the literal schedule. Any retained optimizer
recomputation is equality-only diagnosis and cannot select or change the
materialization schedule. This freeze authorizes a later implementation commit
to replace the current dynamic materialization selector only for this exact
A10B W4 profile.

## Frozen Stage-0 Record And Profile

The sole authorized describe record is the tracked
`docs/bench/v0653-a10b-parallel-pread-floor.describe.json`. Its raw-file SHA-256
is
`ce1b3ccfd67a1a5b8cdaf71050dfd9547ec4ca06f29559da1b4a19473a0cdef9`.
It was produced by clean release commit
`41a12b0eed70e9d667b0fb213db931a64ebc6529`, with equal build/runtime source
state
`git-source-sha256-v2:8e7304b336116a3a555098b9e27aaec6d967aba2af609eca0a43a0a94b7f2360`.
The child reported `mode=describe`, forced benchmark-local native embedding,
no matched profile, and `materialization_supported=false`; it touched no tensor
payload and issued no GPU command.

The record directly reported the architecture tuple, attachment state, mapped
lengths, descriptor and inventory digests, request and byte totals, device
geometry, schedule, and memory signals. It freezes:

- profile `a10b-q4xl-v1`;
- inventory digest
  `b331c475123dbee3bc862a495266dee3996c5f3adabcd6fbeaff9bbabd71a4f8`;
- 879 direct requests and `77,018,996,736` logical bytes;
- cuts `[214, 435, 658]` and task counts `[214, 221, 223, 221]`;
- worker bytes
  `[19,474,295,808, 19,228,744,704, 19,231,902,720, 19,084,053,504]`;
- maximum-to-ideal `1.011402206380462` and descriptive maximum-to-minimum
  `1.0204486066819194`.

The profile ID, shard SHA-256 values, embedding dtype/shape and baseline byte
arithmetic, mapped-byte subtraction, and page counts are preregistered or
derived fields rather than direct describe outputs. They remain frozen packet
inputs and must be independently revalidated where the packet requires them;
the no-payload describe did not recompute shard hashes.

The literal source-order partition identities are:

```text
W0:
  range [0,214), tasks 214, bytes 19474295808
  first (2, "output.weight", 1, 35488, 810516480)
  last  (220, "blk.11.ffn_down_exps.weight", 1, 18920683168, 553648128)
W1:
  range [214,435), tasks 221, bytes 19228744704
  first (213, "blk.11.ffn_down_shexp.weight", 1, 19474331296, 3342336)
  last  (437, "blk.23.ffn_gate_exps.weight", 1, 38250091168, 452984832)
W2:
  range [435,658), tasks 223, bytes 19231902720
  first (436, "blk.23.ffn_gate_inp.weight", 1, 38703076000, 3145728)
  last  (657, "blk.35.ffn_up_exps.weight", 2, 7841234720, 452984832)
W3:
  range [658,879), tasks 221, bytes 19084053504
  first (650, "blk.35.ffn_up_shexp.weight", 2, 8294219552, 3342336)
  last  (867, "blk.47.post_attention_norm.weight", 2, 27378260768, 12288)
```

The describe-time raw memory signals were recommended working set
`103,079,215,104`, current allocation `475,136`, checked headroom
`103,078,739,968`, and process-limit remaining `0`. The zero process signal is
the API's omitted-limit value, not a positive finite budget. It does not replace
the packet's fresh parent and in-child headroom checks.

The committed profile makes this exact metadata/schedule recognizable, but the
current force-native materialization prohibition remains intact. The next
implementation must narrow admission to profile `a10b-q4xl-v1`, arms `copied`
and `parallel-pread`, and W4 only; no other profile or production selector gains
authority from this freeze.

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
swaps, instructions, cycles, and available energy counters. This freeze
explicitly authorizes adding `ru_inblock` and `ru_nswap` capability samples and
JSON deltas to the floor's timer-local `Usage`;
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
   Require `process_limit_remaining_bytes` to be a parsed JSON integer. Missing,
   null, non-integer, or negative values are invalid. Zero is accepted only as
   the API's omitted-limit sentinel; every positive value must be at least
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

This freeze authorizes one separate non-payload live headroom probe, or an
equivalent runner-local Metal query,
immediately before packet reservation. It is distinct from the sole stage-0
describe, is not an attempt, allocates no payload, and is retained in the
aggregate seal.

Full shard residency is a launch condition, not a simultaneous-residency promise.
The kernel may reclaim consumed source-cache pages while the 77 GB destination is
populated. Requiring source plus destination to remain fully resident would
exceed the 128 GiB host envelope and is not part of validity.

Each child must expose and revalidate the complete stamp of each descriptor
retained by its `GgufFile`, and used by B for pread, immediately before timing and
again after full verification. This freeze explicitly authorizes one narrow
read-only GGUF shard-stamp accessor and revalidation seam; parent path evidence
cannot substitute for child descriptor identity.

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

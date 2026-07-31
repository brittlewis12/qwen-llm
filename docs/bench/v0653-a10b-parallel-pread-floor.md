# v0.653 A10B Topology-Preserving Parallel-Pread Floor

Status: consumed unsealed before the first durable child launch;
`authority=none`, with no retry. Clean commit `9b982fd` and its matching release
binary passed pre-run review and the no-payload headroom probe admitted the exact
`85,608,931,328`-byte envelope. The first source-conditioning pass then failed
the frozen all-shards-full-residency predicate on shard 1 after hashing all three
shards. No attempt, arm, payload allocation, timing, correctness, decision, or
completion observation exists.

The packet binds this contract at SHA-256
`bba3ea8fbb4da3a12cffdb240abbf7c0b461afcc0f1a0ccc161ecc17ed065d07`.
A non-authoritative post-stop `mincore` diagnostic reported shard residency
`0/668`, `2,944,390/3,029,833`, and `1,671,006/1,671,038` pages in read order.
That recency gradient invalidates this exact ordered hash-then-global-residency
method on the host; it does not compare copied with W4 parallel-pread.

The frozen standalone runner is
`scripts/profile/v0653_a10b_parallel_pread_floor.py`. Its CPU-only synthetic
check is:

```sh
uv run scripts/profile/v0653_a10b_parallel_pread_floor.py --self-test
```

That mode must not inspect the model or launch `qwen-bench`.

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

The implementation recognizes this exact metadata and admits force-native
materialization only for profile `a10b-q4xl-v1`, arms `copied` and
`parallel-pread`, and W4. Production-auto A10B, forced A3B/dense, every other
A10B arm, and every non-W4 A10B request fail before payload allocation. No
runtime loader or production native-embedding selector changes.

Authenticated A10B rows require JSON output and use schema 3; existing A3B and
dense materialization rows retain their exact schema-2 projection and timing
boundary. A10B rows report the actual embedding policy and selection, in-child
memory admission and zero-sentinel semantics, retained descriptor stamps before
timing and after verification, timer-local block input and swap deltas, all
three Metal allocation snapshots, endpoint
`host-population/no-GPU-command`, and implementation seal
`gguf-arena-floor-copied-pread-v1`. The A10B parallel-pread arm is driven by the
literal frozen schedule; dynamic minimax output is equality-only diagnosis.

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

The literal runner sequence is:

1. Validate clean source/build/binary/runner/contract/describe identity without
   inspecting a model path.
2. Run the sole bounded headroom probe. Only after it succeeds, create and
   durably fsync the packet reservation and prior-activity record.
3. Capture the three model pathname stamps. Any failure after reservation but
   before one complete attempt ledger leaves the packet permanently unsealed.
4. Before each launch, wait at least 120 seconds from reservation activity or
   the prior terminal attempt. Then run bounded identity again.
5. Require AC power, no thermal/performance warning, at least 50 percent normal
   memory pressure, no competing model process, and valid VM/compressor/swap
   counters. Capture complete path-bound stamps: device, inode, size,
   `mtime_sec`, `mtime_nsec`, `ctime_sec`, and `ctime_nsec`.
6. Read and SHA-256 all three exact nofollow descriptor files, require the
   frozen hashes, then require exact `mincore` page totals and all pages resident.
7. Recheck host, VM, and pathname stamps. Durably fsync conditioning and launch
   evidence, block operator signals across the launch race, and acquire the
   authenticated child within the inclusive five-second residency boundary.
8. The child repeats the exact `85,608,931,328`-byte admission immediately before
   timing/allocation. Complete postflight and the immutable attempt ledger. A
   terminal ledger ends the packet; otherwise continue the exact ABBA plan.
9. Only after four valid ledgers or one terminal ledger, run bounded final
   identity and the final signal cutoff, replay all evidence, and publish the
   completion marker last.

The runner invokes exactly one separate `gguf-arena-floor --headroom-probe`
immediately before packet reservation. The probe requires JSON, default
policy/W4, and no profile or arm. It creates a Metal context, reports the raw
signals plus the exact A10B admission decision, and opens no GGUF, allocates no
model payload, and issues no GPU command. It is distinct from the sole stage-0
describe, is not an attempt, and is retained in the aggregate seal. Its positive
PID/PGID, timestamps, and error-free group-absence observation must fit inside
the exact bounded process interval.

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

The runner uses a frozen allowlist rather than passing the ambient environment.
It may retain only `HOME`, `PATH`, `TMPDIR`, `USER`, `LOGNAME`, `SHELL`, `LANG`,
`LC_ALL`, `LC_CTYPE`, and `LC_MESSAGES` when present. It drops and records every
other inherited name, including every `DYLD_*`, allocator, thread-count,
performance-control, `QWEN_*`, `METAL_*`, `MTL_*`, and `RUST_LOG` name. The
record contains only removed names plus the exact resulting allowlisted
environment and its digest; it does not retain unrelated inherited values. The
probe and every child receive exactly that resulting environment.

The separate headroom probe has a frozen 300-second deadline. Each timed child
has a frozen 1,800-second deadline. Output is bounded to 4 MiB per stream, and
pipe joins are bounded to two seconds. Timeout, output overflow, drain failure,
`SIGINT`, or `SIGTERM` triggers authenticated process-group `SIGKILL`, bounded
reaping, and an error-free final group-absence proof. Cleanup actions, signals,
targets, reasons, timestamps, and errors are sealed. Such measured execution or
containment failures are inconclusive; malformed lifecycle evidence is a
contract defect.

The defect parser recognizes only frozen Rust prefixes for `getrusage` and
`proc_pid_rusage` capture, time conversion, counter delta overflow or
regression, proc counter regression, duration conversion, and phase
reconciliation. Missing or unparsable controlling counters are defects.
Unrelated nonzero child exits remain inconclusive, as do correctly measured
positive timer-local block input or swap counts.

Every `build-info` identity child has a frozen 300-second deadline and uses the
same bounded, authenticated process-group containment. Per-attempt identity runs
after the 120-second cooldown and immediately before host/source conditioning.
Final identity uses the same bounded path. A timeout, malformed identity, dirty
source, or source/build/binary mismatch before a complete attempt ledger leaves
the packet unsealed. Final-identity failure also leaves it unsealed.

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

The runner seals exactly two evidence forms: four complete valid ABBA ledgers,
or a valid ABBA prefix ending in one complete terminal launched-slot ledger.
Apply exclusive precedence within those forms:

1. Malformed, missing, regressing, or nonreconciling counters, or any profile,
   source, schedule, topology, byte, timing, or implementation mismatch:
   implementation/contract defect with no performance authority.
2. Correctly measured block input, swaps, pressure growth, postlaunch host
   invalidity, or spawn/execution failure in a complete terminal ledger:
   inconclusive with no authority.
3. Complete valid packet missing the CPU, RSS, footprint, or 1.5-second gate:
   KILL this exact A10B copied-versus-W4-pread floor and advance to grammar.
4. Complete conjunction: GO with the narrow authority below.

Every non-GO disposition returns the active queue to grammar. A defect may
motivate a separately ranked successor preregistration, but v0.653 neither
authorizes that successor nor permits a replay or replacement attempt.

Any preflight headroom failure means this copied A10B floor is economically
infeasible on the host and returns the active queue to grammar without reserving
a packet or allocating the payload. After reservation, any failure before a
complete attempt ledger, during final identity or cutoff, or before the valid
completion marker leaves the packet permanently unsealed with no authority and
no retry. An in-child headroom failure after its durable launch record is a
complete terminal inconclusive attempt when its ledger can be finished. Any
repair requires a new successor preregistration; v0.653 itself is never
replayed.

The complete ABBA packet contains one raw metadata describe, four immutable
attempt bundles, and one aggregate decision/completion seal. A terminal packet
contains the exact valid prefix through its terminal attempt. Each attempt bundle
contains or references its launch lifecycle, command/environment, conditioning
and host samples, raw stdout, `/usr/bin/time -l` stderr, parsed row, and content
hashes. No physical-file count is a validity gate.

Before inventory and completion publication, the runner strictly reopens every
permitted direct packet and work member, rejects missing or orphan members,
replays raw JSON and `/usr/bin/time` parsing, cross-links lifecycle and process
identity, and recomputes attempt validity, pair scores, disposition, and
authority. It snapshots both roots before semantic validation and after durable
inventory publication and proves every preexisting member unchanged. The
inventory is direct and nonrecursive and excludes itself and the completion
record.

Reservation durably records the first prior-activity timestamp. Each later
cooldown binds to the preceding terminal-attempt activity timestamp. Semantic
replay recomputes cooldown arithmetic, raw host validity, all three ordered
source hashes, descriptor and pathname stamps, full residency, launch distance,
and the complete cross-attempt timeline. No attempt may overlap its predecessor.
The threat model is one cooperative local runner with no concurrent process
attempting a transient replace-validate-restore attack on packet files. Every
JSON and raw-stream parse uses a nofollow descriptor with before/after stamp
checks. Digest cross-links may reopen cooperative local pathnames; persistent
replacement or mutation is rejected by root snapshots and inventory.

A signal controller captures `SIGINT` and `SIGTERM`, blocks them across the
durable-launch/acquisition race, and seals exact per-attempt slices.
After final identity it blocks both signals, snapshots delivered and pending
signals, and durably writes the final cutoff. That cutoff is the explicit signal
authority boundary: any delivered or pending signal through it must belong to a
complete attempt ledger, while later blocked signals do not revise the sealed
experiment. Signals contained by a complete launched attempt make that terminal
attempt inconclusive.

Publication uses exclusive creation plus file and directory `fsync`; it is not
described as rename-atomic. Any cleanup action, including a successful forced
group cleanup, makes the attempt inconclusive. Initial group-disposition
evidence must exactly equal the mandatory error-free final absence proof. A
failed first proof followed by a second containment pass leaves the packet
unsealed rather than adding another terminal evidence form.

Publication rereads decision and inventory through stable nofollow descriptors,
exact-compares canonical content, and recomputes final membership and aggregate
hashes after every hook window. It writes `packet-complete.json` only after all
other checks and prerequisite fsyncs, then fsyncs the marker file and packet
directory. That valid final marker is the sole grant of packet authority. Marker
validity and replay, not the runner's final return code, define the commit point;
a failure before a valid marker is unsealed, while a valid marker cannot be
revoked by a later reporting or directory-fsync error. The reserved roots remain
durable no-retry evidence in either case.

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

Adversarial runner review: `cx ask` session
`019fb54e-f61b-77b0-b30a-34bd7f0c3ce1`.

Consumed artifacts: `target/profiles/v0653-a10b-parallel-pread-floor-p1/`
and `target/profiles/v0653-a10b-parallel-pread-floor-work/`. The same `cx`
session independently reviewed the pre-run identity and result interpretation.

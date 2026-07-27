# v0.643 A3B Transient Mmap-Blit Mechanism Floor

Status: independently preregistered, harness-only mechanism-floor experiment. No
v0.643 result is claimed. Authority, force authorization, and successor
authorization are all none.

## Question And Scope

On the exact host and cache-warm asset below, does diagnostic
`transient-mmap-blit` materially improve schema-2 `gguf-arena-floor` model-ready
time over `parallel-pread` W4 while preserving bounded CPU and memory? This is
mechanism-only evidence. It makes no production, Auto, loaded-endpoint,
storage-cold, Private-storage, sidecar, energy, concurrency, serving, other-host,
or other-asset claim.

The packet is independently justified by the roadmap candidate. v0.642 has
`successor_authorization=none`; this packet imports no v0.642 or dirty-smoke
row, timing, result, authority, prior, or scoring input. The historical 112 ms
is a portfolio gate only, not an MDE.

## Source Boundary And Fresh Gates

The preregistration commit must be a clean, non-merge direct child of frozen
implementation `d859345118774b934218a7eb55460ef0870da882`. It adds exactly this
file and `scripts/profile/v0643_a3b_transient_mmap_blit_floor.py`; its complete
`crates` tree must equal the implementation parent's tree. Execution requires a
clean worktree and matching clean release build/runtime identity at that commit.
No sealed runner is imported.

Before reservation, run normal and `python -O` runner self-tests, a fresh
release `qwen-bench` build, and exactly:

```text
cargo test --release -p qwen-cli --bin qwen-bench \
  gguf_arena_floor::tests -- --test-threads=1
```

The unique final Cargo summary must be exactly 13 passed, zero failed, ignored,
or measured; zero selected tests is a defect. Build/test children use the same
authenticated process-group lifecycle as attempts. Their bounded output,
pid/pgid, exact reap, zero-cleanup normal disposition, and final absence of live
group members are retained. The fresh executable is descriptor-bound and fully
hashed after building, then its bytes, complete stable stat stamp, pathname
identity, and semantic build identity are rechecked before every child and
before sealing. Preflight also hashes and authenticates the model. A preflight
failure reserves no packet. Do not use preflight for a CPU-only check because it
reads and hashes the full model.

## Frozen Asset And Treatments

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Size: 22,134,528,992 bytes; 1,350,985 pages at 16,384 bytes.
- SHA-256: `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Descriptor: `0x5ae645df5cf7d568`.
- Inventory: `f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5`.
- Profile: `a3b-q4km-v1`; 733 requests, resources, and bindings;
  22,123,538,944 logical and physical bytes.
- Topology: creation Shared/DefaultCache/default hazard and observed Shared/DefaultCache/Tracked.

Every treatment is a fresh sole-attempt process under `/usr/bin/time -l`:

```text
A = target/release/qwen-bench gguf-arena-floor --model MODEL \
    --profile a3b-q4km-v1 --arm parallel-pread --workers 4 --output json
B = target/release/qwen-bench gguf-arena-floor --model MODEL \
    --profile a3b-q4km-v1 --arm transient-mmap-blit --workers 4 --output json
```

The runner removes inherited `QWEN*`, `MTL*`, `METAL*`, `RUST_LOG`, QoS, and
chunk overrides. B's accepted output must nevertheless report `worker_count=0`
and `parallel_copy_schedule=null`.

A must report exact `minimax-contiguous-v1` schedule SHA-256
`800bf469d09879187e07a832ab573a84f19714f6d3fce8d47fce87adc5808329`,
cuts `[155,359,539]`, task counts `[155,204,180,194]`, and worker bytes
`[5532746240,5462315776,5595522304,5532954624]`. Its complete canonical
schedule, contiguous partitions, endpoints, totals, and W4 count are checked.

The describe control must authenticate transient plan digest
`fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af`.
B's exact nested blit population is:

- Order `shard-offset-request-v1`, count 733, first request 2, last request 721.
- One source window of 22,123,544,576 bytes with 13,824 gap bytes; one
  8,192-byte fallback and 8,192 CPU-staging bytes.
- 732 window copies totaling 22,123,530,752 bytes; 733 total copies totaling 22,123,538,944 bytes.
- One command buffer, encoder, commit, and wait; retained references true;
  completed status/code 4; zero errors.
- One window deallocator call, zero mismatches, and zero source buffers alive.

All three allocation snapshots are unsigned, device-wide descriptive telemetry.
They carry no ordering or equality validity condition. Command-buffer retained
references, zero live weak references, one exact deallocator callback, and zero
callback geometry mismatches are the source-lifetime proof.
The GPU timestamp triple is either entirely null or entirely finite, positive,
ordered, and consistent with its reported wall milliseconds.

## Order And Population

Run six fresh pairs in this exact order:

```text
P1 AB   P2 BA   P3 BA   P4 AB   P5 AB   P6 BA
```

This is exactly 12 sole-attempt children. There are no retries, replacements,
resume, pooling, overlap, inspection breaks, fallback treatments, performance
early stops, or runtime timeouts. The first contract defect or invalid attempt
stops population after its available evidence is persisted. A RUSAGE_SELF
regression has the exact specialized inconclusive classification below; it is
never retried or replaced. Score only after all 12 children are valid.

## Conditioning, Host, And Lifecycle

Before each child, wait an inclusive 30 seconds from the prior recorded packet
activity boundary. The reservation boundary seeds pair 1; each authenticated
attempt persists the exact boundary returned to the next cooldown. Seal-time
validation requires one monotonic, non-overlapping conditioning/launch/child/
post-exit sequence across the packet.
Require AC power, no thermal or performance warning, at least 50% free memory,
and no competing model/qwen/llama experiment before conditioning, immediately
before launch, and after exit. Capture VM state at each point.

Outside the scored endpoint, the parent sequentially reads the entire exact
22,134,528,992-byte model with one 8 MiB buffer. It then maps the full descriptor
and requires exact `mincore` residency of 1,350,985/1,350,985 pages. Capture the
residency proof timestamp immediately after success. The actual successful
`Popen` acquisition must occur within the inclusive five-second limit. Require
the same complete residency after child exit.

Host and VM validity are checked before the 22 GB read. Conditioning and
residency use declared ctypes `mmap`, `mincore`, and `munmap` signatures and
require pathname plus descriptor fstat identity before and after each operation.
Ordinary capture, `OSError`, and `RuntimeError` failures are recorded in the
conditioning or post-exit object before closing inconclusive; they do not erase
available host, VM, residency, or lifecycle evidence.

Across conditioning and child intervals, compression, swapout, and swap
occupancy changes are fatal. Counter regression is invalid. Pageouts and
compressor gauges are advisory. `/usr/bin/time -l` block input and process swaps
must be zero. Whole-process page faults are strictly parsed, persisted, and
advisory without a threshold. JSON timer-local major faults must be exactly
parsed as unsigned. A positive value is inconclusive validity, not a structural
contract defect; minor faults are descriptive.

Every child starts a new session. Immediately after `Popen`, authenticate
`getpgid(pid)==pid` and durably record acquisition before waiting. No signal mask
spans `Popen`. One packet controller records SIGINT/SIGTERM throughout execution.
An operator event, pipe/drain failure, or interruption while an authenticated
group is live sends SIGKILL exactly once to that exact pgid, then drains and
exactly reaps the leader. Acquisition-record failure receives the same exact
group cleanup before the durability error is rethrown. If pgid authentication
or exact wait ownership is lost, send no risky group signal and leave the packet
unsealed. After reap, strictly parse bounded `ps` observations until the group
is absent; never use a process-name kill, TERM grace, second kill, or kill after
uncertain ownership. Persist launch, acquired, completion, attempt, stdout,
stderr, conditioning, post-exit, and immutable signal-closure evidence.

If a nonzero child has stderr beginning exactly
`Error: getrusage counter regressed:`, classify it as RUSAGE_SELF counter
regression, persist that fact and prefix in the attempt, stop immediately,
classify the packet `inconclusive`, and never retry or replace it. A quiet,
positive nonzero exit with no host, signal, lifecycle, or other validity defect
is `implementation_or_contract_defect`. A signaled exit or a nonzero exit with
independent validity defects is `inconclusive`.

## Result Contract

Strict JSON parsing rejects duplicate keys, malformed or multiple values,
NaN/Infinity, finite-syntax overflow, and trailing content. Both arms require
the exact old schema-2 top-level key set except that only B may add
`blit_population`. This rejects candidate-only fields on A. Both require exact
build/model/profile/digests/counts/bytes/topology, all-byte correctness over all
733 entries, finite positive ready and copy throughput, finite timing, exact
millisecond/microsecond agreement, and
`total_cpu_us=user_cpu_us+system_cpu_us`.

Common output is pinned to architecture `qwen35moe`, tied embeddings false, MTP
false, shard lengths `[22134528992]`, native embedding true and supported with
selection `production-auto-promoted`, page size 16,384, alignment 32, maximum
buffer length 77,309,411,328, device `Apple M4 Max`, and unified memory true.
The architecture tuple is exactly MoE, 40 layers, hidden size 2,048,
intermediate size 0, vocabulary 248,320, full-attention interval 4, 16 query
heads, 2 KV heads, head dimension 256, rope theta 10,000,000, partial rotary
factor 0.25, GDN heads 32/16 with dimension 128 and convolution kernel 4, 256
experts with 8 used, expert and shared feed-forward lengths 512, and zero MTP
hidden layers. `proc_rusage_v4` has exactly the four unsigned instruction,
cycle, billed-energy, and serviced-energy delta counters. `cpu_per_wall` must
equal `total_cpu_us/ready_us` within frozen numeric tolerance.

A must have W4 schedule, worker count 4, no `blit_population`, no source-release
timing, and four-phase allocation/source/copy/binding reconciliation. B must
have null schedule, worker count 0, exact nested blit schema, source-release
timing, and five-phase allocation/source/copy/source-release/binding
reconciliation. Unattributed timing is finite, nonnegative, bounded by rounding,
and included in ready reconciliation tolerance. Source-resolution aliases must
agree. Phase, GPU, and allocation telemetry is reported descriptively.

Structural, source, schema, topology, timing-contract, or correctness failures
are `implementation_or_contract_defect`. Other host, VM, fault, residency,
lifecycle, signal, or incomplete-population invalidity is `inconclusive`.

## Scoring And Decision

Within pair, define:

```text
d = A.ready_us - B.ready_us
q = B.ready_us / A.ready_us
c = B.total_cpu_us / A.total_cpu_us
```

Report every pair, median d, AB median d, BA median d, median q, median c, AB
and BA median c, wins (`d>0`), maximum paired B/A RSS, maximum paired B/A peak
footprint, and descriptive phase/GPU/allocation values. Even medians are the
arithmetic mean of the two central sorted values.

`GO-mechanism-floor` requires the inclusive conjunction:

```text
median d >= 112000 us
AB median d >= 112000 us
BA median d >= 112000 us
B wins >= 5/6
B wins in AB >= 2/3
B wins in BA >= 2/3
median c <= 1.10
AB median c <= 1.10
BA median c <= 1.10
max paired RSS B/A <= 1.05
max paired peak footprint B/A <= 1.05
```

A valid miss is `mechanism-floor-miss` with closure `KILL`. A pass has closure
`GO-mechanism-floor`. Both retain `authority=none`, `force_authorized=false`,
`successor_authorization=none`, and exact-host/asset/cache-warm mechanism-only
scope. Decision precedence is preflight/no artifact, then
implementation-or-contract defect, then inconclusive invalidity or incomplete
population, then valid GO or miss.

## Sealing And Self-Tests

Fresh roots are `target/profiles/v0643-a3b-transient-mmap-blit-floor-p1` and
`target/profiles/v0643-a3b-transient-mmap-blit-floor-work`; both must be absent.
The immutable manifest binds source/build, exact asset, normalized environment,
commands/order, describe control, executable bytes, and gates. Process-lifecycle
fault tests run under the preflight process that directly owns their groups;
normal and optimized parser/validator/sealing suites then run as bounded pure
child gates. Each conditioning record retains its cooldown interval. Gate
evidence retains the complete validated lifecycle disposition. Final expected
and observed source, semantic-build, binary, and model identity plus capture
errors are persisted in the decision before the signal cutoff. The cutoff
blocks SIGINT/SIGTERM only after no child is live, attributes all recorded and
pending events through its snapshot to the pre-cutoff side, then seals with file
operations while signals remain blocked.

Complete and incomplete seals reject every symlink or nonregular member and
authenticate the exact durable attempt prefix. Attempt, closure, lifecycle,
conditioning, post, stdout, and stderr hashes and process fields must cross-link.
Decision/cutoff/log digests must cross-link. The persisted decision is reread;
its no-authority constants, analysis, status, closure, and precedence are
recomputed from authenticated attempts. `completed_attempts` is derived from the
authenticated attempt prefix, never a raw line count. The decision digest,
member inventory, and completion binding are fsynced. A complete 12-child packet
has exactly 58 inventoried members. Durability, ownership-loss, or sealing
failure may remain genuinely unsealed.

Fixture-free CPU-only self-tests use explicit exceptions and run identically
under optimization. They cover strict typed parsers, exact AB order and commands,
arm contracts and candidate-field rejection, unsigned allocation telemetry,
source-release proof failures, phase reconciliation, advisory process page
faults, timer-major/block/swap invalidity, quiet and signaled nonzero precedence,
RUSAGE regression, actual stop/no-retry control flow, inclusive scoring
boundaries, incomplete/invalid row scoring rejection, decision precedence, exact
Cargo count, preflight SIGTERM group cleanup, exact-wait ownership loss,
complete and incomplete sealing cross-links, final-identity persistence, signal
restoration, and durable exclusive-write mechanics. They do not read the model
or use the GPU.

# v0.641 A3B Direct-Pread Worker-Count Screen

Status: preregistration only. No v0.641 materialization or performance result
exists. The implementation under test is commit
`1b8dd176fe8647f276ed8f50c7714ae89dbf62ef` (`1b8dd17`).

## Question And Boundary

For one exact cache-warm Qwen3.6-35B-A3B Q4_K_M asset, which diagnostic
direct-`pread` worker count, if any, materially improves the schema-2
`gguf-arena-floor` model-ready endpoint relative to W4 without more than 10%
paired endpoint CPU inflation?

This is a bounded diagnostic screen. It changes no production or Auto code and
has no authority over production selection, defaults, QoS, chunking,
target-file-cold behavior, serving, policy, another asset, or another host. A
winner is only an exact-asset/host cache-warm diagnostic winner. A valid miss
closes only cache-warm A3B worker tuning. Neither outcome authorizes a
successor. A later PERF certification may independently authorize one cold
comparison.

Before this preregistration, default, explicit W4, and frozen W4 A3B schedules
were metadata-equal; dense default and frozen W4 were also metadata-equal.
Dense W8 was rejected by scope. This is source-free metadata evidence only. It
contains no timing and is not pooled or scored.

## Source And Build Boundary

R641 is a clean, non-merge, direct child of
`1b8dd176fe8647f276ed8f50c7714ae89dbf62ef`. It adds exactly this file and
`scripts/profile/v0641_a3b_pread_worker_screen.py`. The complete `crates` tree
must be byte-identical to the parent tree. No production, Auto, dependency, or
build file may change.

Execution requires a clean R641 worktree. Before packet reservation, the runner
runs its normal and optimized self-tests, a fresh release `qwen-bench` build,
and exactly:

```text
cargo test --release -p qwen-cli --bin qwen-bench \
  gguf_arena_floor::tests -- --test-threads=1
```

The build and test stdout/stderr are bounded and retained in the immutable
manifest. The targeted test gate requires one exact final summary with 11
passed, 0 failed, 0 ignored, and 0 measured; a successful command with zero
selected tests is a defect. Each successful build/test gate must also preserve
authenticated `pid==pgid`, exact leader reap, zero cleanup actions, and a
strictly parsed group disposition ending with no live members. Missing/false
disposition, any inspection error, or surviving member fails preflight even
when return code is zero. The manifest retains pid, pgid, complete disposition,
cleanup-signal state, cleanup-action cardinality, and the complete action list
alongside bounded gate output. The runner then requires
`qwen-bench build-info` to report matching clean build/runtime commit and source
state at R641. It opens `qwen-bench` once with `O_RDONLY|O_NOFOLLOW`, requires a
single-link regular executable that is not group/world writable, and hashes
every byte through that descriptor. Complete fstat identity/mode/owner/size/
time/flags stamps must be stable before and after hashing, the exact byte count
must equal size, and lstat/stat pathname stamps before and after must continue
to name that same inode. The resulting descriptor stamp and SHA-256 are frozen
after the fresh build. Before every child and at final sealing the runner
repeats the same single-open proof and semantic build-identity check. The same
clean source/build checks run again before sealing. Pre-reservation failure
creates no packet artifact.

## Frozen Asset And Command

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Size: 22,134,528,992 bytes; 1,350,985 pages at 16 KiB.
- SHA-256: `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Descriptor digest: `0x5ae645df5cf7d568`.
- Inventory digest: `f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5`.
- Profile: `a3b-q4km-v1`; arm: `parallel-pread`.
- Treatments: W in `{1,2,4,6,8,12}`.

Every child is a new process using exactly:

```text
/usr/bin/time -l target/release/qwen-bench gguf-arena-floor
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  --profile a3b-q4km-v1 --arm parallel-pread --workers W --output json
```

The environment removes every inherited `QWEN*`, `MTL*`, `METAL*`,
`RUST_LOG`, QoS, and chunk override. There is no allow-dirty mode. W1 is an
eligible one-spawned-worker direct-`pread` treatment, not copied serial.

## Frozen Schedules

The algorithm is exactly `minimax-contiguous-v1`. SHA-256 is over the complete
schedule object encoded as sorted-key compact JSON. The runner independently
checks each digest and the following cuts, task counts, and worker bytes; it
also checks contiguous partition structure, totals, endpoints, and worker
count.

| W | schedule SHA-256 | cuts | task counts | worker bytes |
|---|---|---|---|---|
| 1 | `c22c86524146da70a3913fb3db7daa1149a8cc6cc36eb7d2d7aa1e1f7492adfb` | `[]` | `[733]` | `[22123538944]` |
| 2 | `19165cbe11e4e023881ccfd31de196bc40240020ca68f91a41e83f32496a6b56` | `[359]` | `[359,374]` | `[10995062016,11128476928]` |
| 4 | `800bf469d09879187e07a832ab573a84f19714f6d3fce8d47fce87adc5808329` | `[155,359,539]` | `[155,204,180,194]` | `[5532746240,5462315776,5595522304,5532954624]` |
| 6 | `9d8459edc9e4920eecaf528e95037705a3c43ac522a8051410414b10489d1b87` | `[86,220,343,470,609]` | `[86,134,123,127,139,124]` | `[3581055232,3693068800,3681269504,3683499776,3720937984,3763707648]` |
| 8 | `0432332f6a5f4417852bc30b527ae2f3fd26e6d82690b5d18f9d57da3c4bb438` | `[68,155,251,357,445,542,635]` | `[68,87,96,106,88,97,93,98]` | `[2755989760,2776756480,2788430848,2672769792,2810311936,2789544960,2820862976,2708872192]` |
| 12 | `b92041f460cb6f65b3132d03a072e540b045620c0650403eb771e1149130411a` | `[26,86,153,220,287,343,410,470,537,609,666]` | `[26,60,67,67,67,56,67,60,67,72,57,67]` | `[1697137408,1883917824,1799581952,1893486848,1799581952,1881687552,1799581952,1883917824,1799581952,1921356032,1949904384,1813803264]` |

## Order And Attempts

Run the exact six-row Williams order:

```text
R1  1,2,12,4,8,6
R2  2,4,1,6,12,8
R3  4,6,2,8,1,12
R4  6,8,4,12,2,1
R5  8,12,6,1,4,2
R6  12,1,8,2,6,4
```

There are exactly 36 unique sole attempts and 36 fresh children in an
uninterrupted execution. There is no retry, replacement, resume, pooling,
inspection break, or performance early stop. Rows receive no additional
cooldown. Before each child, wait exactly the single 30-second cooldown from
prior packet activity.

The order balances each treatment once at every position and balances directed
predecessors. For candidate-versus-W4 strata, the candidate precedes W4 in:

```text
W1: R1,R5,R6    W2: R1,R2,R6    W6: R4,R5,R6
W8: R4,R5,R6    W12: R1,R5,R6
```

The other three rows form that candidate's after stratum.

## Per-Child Protocol And Validity

At all three host points require AC power, no thermal or performance warning,
at least 50% from `memory_pressure -Q`, and no competing qwen, llama, model, or
GPU experiment. Capture host and VM state before conditioning, immediately
before launch, and after exit.

The parent authenticates model device/inode/size/mtime, sequentially reads
exactly 22,134,528,992 bytes with one fixed 8 MiB buffer outside the scored
endpoint, and authenticates the descriptor again. The manifest preflight hashes
the model; conditioning need not rehash it. Immediately after conditioning,
map the whole descriptor and require `mincore` residency exactly
1,350,985/1,350,985. Immediately when that successful `mincore` call returns,
capture `residency_proved_ns`; do not reset this clock. Then recapture host/VM.
The actual successful `Popen` acquisition timestamp must be no more than five
seconds after `residency_proved_ns`. Persist both host/VM sample intervals and
timestamps, the residency timestamp, launch-attempt and acquired timestamps,
and the derived delay. After exit recapture host/VM and require the same
complete residency.

Across each conditioning and child interval, Swapouts, Compressions, and swap
occupancy must be unchanged. Counter regression is also invalid. Pageouts and
compressor stored/occupied occupancy are recorded as advisory only.
`/usr/bin/time -l` must report block input operations=0, swaps=0, and page
faults=0. The JSON `rusage.timer_major_faults` must be zero.

Each child must have an exact launch/completion lifecycle, return zero, bounded
stdout/stderr, one canonical JSON object, a unique sole-attempt identity, and
no retry. JSON parsing rejects duplicate object keys, NaN/Infinity constants,
finite-syntax exponent overflow such as `1e999`/`-1e999`, trailing content, and
multiple values while preserving finite decimals. A successful row must
validate schema 2, profile and parallel-pread arm, exact W schedule and digest,
733 requests/resources/bindings, logical and
physical bytes 22,123,538,944, Shared/DefaultCache/Tracked topology, worker
count W, and correctness over all 733 entries and 22,123,538,944 bytes. All
phase, timing, and throughput values must be finite and positive where defined;
phase timing must reconcile, including finite nonnegative
`unattributed_wall_ms` agreeing with `unattributed_us` within 0.002 ms, and JSON
`total_cpu_us=user_cpu_us+system_cpu_us`.
JSON `timing.ready_us` is the sole endpoint. JSON total CPU is the CPU gate.
Whole-process `/usr/bin/time` CPU is diagnostic because correctness follows the
ready endpoint.

Conditioning evidence is fsynced before its validity is evaluated. Expected
conditioning and pre-spawn residency failures then close inconclusive with all
available evidence. Every recorded launch has one matching completion and
attempt record, including a structured spawn failure with no PID. Expected
spawn, wait, drain/pipe, exit, and post-exit residency failures are persisted
before closing inconclusive; post-exit `mincore` failure therefore cannot erase
the host/VM or lifecycle record. Artifact write, fsync, inventory, or seal
failure is not converted to inconclusive and may remain genuinely unsealed.

There is deliberately no runtime timeout. After preflight and before packet
reservation, one packet-lifecycle controller installs non-raising SIGINT and
SIGTERM handlers. Each event receives a monotonic timestamp and sequence. The
controller remains installed through all cooldowns, conditioning, attempts,
decision construction, and durable sealing; it is never handed off per attempt.
No signal mask spans `Popen`, and caught handlers reset on exec, so children do
not inherit blocked operator signals. Checks before every cooldown,
conditioning, and launch prevent a between-attempt signal from being ignored.

Parent wait uses short polling only; authoritative `ready_us` is produced inside
the child. Immediately after successful `Popen`, the runner calls `getpgid(pid)`
and claims ownership only when `pgid==pid`, persisting both. Every child uses
`start_new_session=True`. A durable acquisition lifecycle event records pid,
pgid, and acquisition timestamp before the parent enters its wait loop. If that
artifact write fails, the now-authenticated group receives exactly one SIGKILL
before an unbounded exact leader wait/reap; the original durability failure is
then rethrown and the packet remains unsealed. If `getpgid` fails or differs
from pid, no group signal is sent: the acquired Popen leader is safely drained,
closed, and exactly reaped before `OwnershipLost` is raised. If a controller
event, interpreter interruption, or
drain/thread failure occurs while that exact owned leader is live, the runner
sends SIGKILL once to that exact pgid before leader reap, recording target,
signal, timestamps, success, and failure. It then exactly waits/reaps the leader,
drains/closes pipes, and records completion and attempt evidence. It never uses
a process-name search, TERM grace, sequential fallback drain, or `killpg` after
leader reap or uncertain ownership. Persistent controller/pipe observations
cannot issue a second cleanup signal; cleanup evidence requires cardinality one.

After leader reap, a bounded read-only `ps` observation strictly parses every
nonblank row and verifies that no live
member of the process group remains without sending another signal. Normal exit
records the same leader-reap and group-disposition evidence. Read failures close
their affected pipe; command failure, malformed `ps` rows, wait, join, read,
close, cleanup, and disposition failures are structured and cannot imply false
group disappearance. A pgid mismatch or unexpected `ChildProcessError` means exact
ownership is lost: it raises narrow `OwnershipLost`, sends no risky signal, and
remains intentionally unsealed. Artifact write/fsync/inventory/seal failures
also remain unsealed rather than being mislabeled inconclusive.

## Scoring

For every candidate W in `{1,2,6,8,12}`, pair within each Williams row against
W4:

```text
d = W4.ready_us - W.ready_us
q = W.ready_us / W4.ready_us
c = W.total_cpu_us / W4.total_cpu_us
```

Compute `D` as the six-value median of d, `D_before` as the three-value median
where the candidate precedes W4, `D_after` from the other three rows, `Q` as the
descriptive six-value median, `C` as the authoritative six-value median CPU
ratio, and wins as count `d>0`. Six-value medians are the arithmetic mean of
the sorted third and fourth values; three-value medians are the sorted middle.
Also report ratio-of-medians and maximum paired CPU ratio descriptively.

A candidate qualifies iff all inclusive gates hold:

```text
D >= 60000 us
D_before >= 60000 us
D_after >= 60000 us
wins >= 5
C <= 1.10
```

If any qualify, let `Dmax=max(D)`, retain the near-best set with
`D>=Dmax-10000 us`, and select the numerically smallest W. W1 participates
normally. If none qualify, the decision is `warm-worker-screen-miss`. A winner
is `GO-diagnostic`; a valid miss is `KILL`. Both retain `authority=none` and
`successor_authorization=none`.

## Decision And Sealing

Decision precedence is fixed:

1. A pre-reservation preflight failure creates no artifact.
2. A post-launch contract, schema, topology, or correctness defect is
   `implementation_or_contract_defect`.
3. Host, pressure, residency, fault, lifecycle, or interruption invalidity is
   `inconclusive`; interruption after launch closes the packet inconclusive.
4. Otherwise a winner is `GO-diagnostic`, and a valid miss is
   `warm-worker-screen-miss` / `KILL`. Valid slow rows remain scored.

The fresh packet roots are
`target/profiles/v0641-a3b-pread-worker-screen-p1` and
`target/profiles/v0641-a3b-pread-worker-screen-work`; neither may preexist.
Before attempts, write an immutable fsynced manifest binding source/build,
model hash and descriptor, normalized environment, host/tool dialects, frozen
orders/schedules, exact binary bytes/stamp, bounded build/test output, and
preflight gates. Preserve packet-local stdout, stderr,
conditioning, process-resource, row, attempt, order, and launch/completion
evidence. Final sealing rechecks source/build and descriptor identity, then
writes a member-complete SHA-256 inventory, a decision digest, and an fsynced
completion object binding both. No artifact is reusable or resumable.

After every attempt fsync, an attempt-signal closure captures one immutable
event slice and cross-links the authoritative packet event sequence. Its end
sequence is the last event in that slice, or the attempt start when empty; it
never rereads a live sequence. Later events are between-attempt packet events.
Any event since attempt start makes the packet
inconclusive. At the sole final authority cutoff, with no child live, the runner
first completes all final source/build/binary checks that require subprocesses.
It records the logical boundary timestamp and sequence before blocking, then
blocks SIGINT/SIGTERM and snapshots recorded and pending signals. Everything
observed through that post-block snapshot is conservatively attributed to the
pre-cutoff side. Signals arriving after the snapshot remain blocked and are
post-cutoff/outside authority. The runner writes the
authoritative signal log and cutoff record, derives the decision, and seals
using file operations only while signals remain blocked. It neither restores
handlers nor unblocks before process exit. Signals after that explicit cutoff
are outside packet authority.

The runner's fixture-free self-tests execute before preflight and normal
execution and use explicit exceptions so `python -O` is equivalent. They cover
order positions/predecessors and before/after sets, odd and even medians,
inclusive gates, 10 ms winner indifference and W1 eligibility, schedule
constants/digests and malformed JSON, `/usr/bin/time` parsing, host/VM interval
rules, strict duplicate/nonfinite JSON rejection, wall/microsecond agreement,
exact Cargo summaries including zero-test rejection, structured spawn-error
completion, successful launch/completion/attempt ordering, injected wait/drain/
join/thread-start cleanup, late-window SIGINT/SIGTERM attempt invalidation,
between-attempt and final-cutoff behavior, authenticated process-group KILL of a
leader plus TERM-ignoring descendant, pgid-mismatch and ownership-lost mocks,
real getpgid/mismatch leader reap, acquired-record durability cleanup, immutable
attempt slices, deterministic block/snapshot attribution, strict malformed-ps
rejection, post-exit residency-exception persistence, the inclusive five-second boundary,
conditioning invalidity, single-descriptor temp-file hashing and deterministic
pathname replacement rejection, exponent-overflow JSON rejection, tiny-file
sequential conditioning plus `mincore`, optimized-mode checks, and decision
precedence. Scoped self-test controllers restore handlers and masks. Tests do
not access the model or GPU.

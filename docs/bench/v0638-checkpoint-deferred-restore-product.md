# v0.638 Deferred-Restore Real-27B Product Packet

Status: runner-only preregistration. No v0.638 preflight, build, model, GPU,
scored child, or restore has run.

## Authority Bridge

This is the sole real-27B product packet authorized by sealed v0.637. The
complete v0.637 artifact is authenticated descriptor-bound and read once before
any v0.638 packet root is reserved. It has exactly 23 regular files and a
21-member inventory, with these seals:

```text
decision.json              29c4ae3c50eac56001a8ae89c226ce490f1d4617c7bd6f01d8b2dd44d304c3a0
artifact-inventory.sha256  8d218da1a4d20980256d4584e972e8ddb5f6304f5967872084fb96b117666403
packet-complete.json       6f1ed3481a729b565a05a70996d17a17f2e638819255248113df84fe65cf41f4
attempts.jsonl             18796d6137b4542c975661765474ab77b7d3fca7412d84f25dcd1160c87a8382
```

Require the exact GO decision, `authority=no-production-authority`,
`force_authorized=false`, and sole successor authorization
`preregister-one-real-27b-deferred-restore-product-packet`. Require its four
exact valid floor rows in `AB BA` order and both 250,000-us floor gates in each
pair. This consumes only the named successor authorization. It imports no
v0.637 product authority, gate result, scored row, timing, or performance
observation. Manifest and decision record:

```text
product_authority_imported=false
successor_authorization_imported=true
successor_authorization_consumed=true
gate_results_imported=0
scored_rows_imported=0
timing_observations_imported=0
performance_observations_imported=0
```

## Source And Build Identity

R638 must be a clean, direct single-parent child of exact parent
`482fa05341971755ecfcc0e1820a09be65490356`. It adds exactly this document and
`scripts/profile/v0638_checkpoint_deferred_restore_product.py`; it modifies
nothing else. The unchanged implementation commit is
`3d59d94cf314dae79b79cd05f74b4aea455cd30c` and the sealed v0.637 execution
commit is `61f852e557d42180468a9c7ce77ccf606b68d8e8`.

At execution, both `qwen` and `qwen-bench` must carry R638 as build and runtime
identity with clean matching source state. The implementation commit is not a
valid build identity. Freeze:

```text
Cargo.lock
2ea5d138ef0b5813ce29ad39c39d4a34d31d5a72fb71431f6038c5343d2e4fbe
Cargo.toml
c92ed7c228b43de3a2920cc5d2a7535660fd328cbde1444bf315545c36b162e9
crates/qwen-cli/src/main.rs
1ee1e959fe5c2616fec20fb924b9edef026fc432b635db1f46166a43a1ce0c71
crates/qwen-llm/src/checkpoint_store.rs
a5ea6b001249418febb3f876253bf5989c0603ef42cc43bc50fda7d51927f8da
scripts/profile/v0602_a3b_parallel_copied_loader.py
e5775489802dddc2d88dc9b950940417f325ebcc4ece29194550b7e430f851e3
scripts/profile/v0593_demand_paged_no_copy.py
4c1b73a2897d8bf882988407f45b59be7b19d236c06a612991491c57af0d7be4
```

The frozen sibling commits remain `c7369fd4868a6f613459fff355477f53bf4ee2f1`
for `gguf` and `fe4fb533d1ed2855b6ac5492e56c42007d410409` for
`llama-cpp-rs`, with only `.claude/settings.local.json` permitted untracked in
the latter.

Run exactly one fresh release build gate for both `qwen` and `qwen-bench`, then
the same release checkpoint-codec, checkpoint-store, and qwen CLI test gates.
Every gate is sole-attempt, no-retry, and durably launch/completion sealed.
Immediately before every scored and restore spawn, descriptor-hash
`target/release/qwen`, require stable regular-file metadata, and compare its
SHA-256 to the post-gate build identity. Bind the complete binary stamp and hash
into the launch seal. Reauthenticate it after the final child and while
rederiving GO; `qwen-bench` does not require per-child rehashing.

## Frozen Inputs

```text
model  /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf
bytes  16817244384
SHA-256 5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0

messages target/profiles/v0615-long-history/ring0-turn-2.json
SHA-256 5c2455738af1d789ce1f86cf1d2064fb8e80a242c80273273f57c25783bdd4a1

identity seed
target/profiles/v0615-long-history/cache-ring0/v1/identity/
3f6fb8c12c7fbfe881e2c43b7d98742873161ffa51a5e039773252207541734e.mid
bytes 128
SHA-256 17afb5fc1e9c3e56b7ffd2612f501d1714eefbb3c1e81d27e9501e0a18120c69
```

Before any scored child, copy the messages and identity seed into packet-local
mode-`0600` files. Deterministically create and freeze the extended restore
messages at the same time. Fsync every file and containing directory, then
publish an input seal. Scored and restore commands use only packet-local inputs.
The immutable manifest records all three expected sizes and SHA-256 values,
including the deterministic restore-message digest. Immediately before every
scored and restore spawn, reopen all three packet inputs with `O_NOFOLLOW` and
descriptor-bind regular type, mode `0600`, link count one, size, and SHA-256 to
the retained input seal.

## Product Children

Run six non-overlapping pairs, 12 fresh sole-attempt children, in exact order:

```text
AB BA BA AB AB BA
```

- A explicitly sets `QWEN_CHECKPOINT_STAGED_INTEGRITY=decode`.
- B explicitly sets `QWEN_CHECKPOINT_STAGED_INTEGRITY=deferred-restore`.

Use umask `077`. Remove every inherited `QWEN*`, `MTL*`, `METAL*`, and
`RUST_LOG` control, then add only the arm variable. Do not request timing
output. Invoke `target/release/qwen` directly with this exact argument shape:

```text
--model <frozen-model> --messages <packet-messages>
--messages-preserve-thinking --tokens 1 --temp 0.7 --top-k 200 --top-p 1.0
--min-p 0.05 --seed 42 --prefill-chunk 1024 --max-context-tokens 6516
--durable-prefix-cache <unique-store>
--durable-prefix-cache-max-mib 768
--durable-prefix-cache-max-entry-mib 768
--durable-prefix-cache-min-tokens 1024
```

Each unique store root and directory is explicitly mode `0700`. Seed only the
canonical identity file with `O_EXCL|O_NOFOLLOW`, mode `0600`, and link count
one. Before spawn, reauthenticate the seed and require no blob, lock, temporary,
symlink, nonregular, or foreign entry.

Before every scored child, and again before restore, descriptor-open the model
with `O_NOFOLLOW`, bind stable before/after metadata, and hash-read the complete
file. Cool down for 30 seconds, then require M4 Max, 128 GiB, macOS 15, AC
power, no thermal or performance warning, and at least 50% available memory.
Immediately before spawn, require the nofollow pathname to retain the same
device, inode, size, mode, link count, mtime, and ctime stamp. Model I/O failure
is inconclusive; a wrong hash or changed immutable stamp is a contract defect.
Capture VM state around conditioning and child execution. Reject capture or
monotonic-counter regression, swapout growth, swap-occupancy growth, invalid
post-exit host state, `ru_nswap != 0`, or `ru_inblock != 0`. Pageouts,
compressions, and major faults are recorded diagnostics only.

## Direct Process Boundary

Block SIGINT and SIGTERM before creating CLOEXEC stdout and stderr pipes or
starting one blocking byte-drain thread per read end. Spawn direct `qwen` with
`os.posix_spawn`, `setsid=true`, empty child signal mask, default child SIGINT
and SIGTERM dispositions, and exact DUP2/CLOSE file actions. Require runner cwd
to remain ROOT, register the exact PID/process group, and close parent write
ends. The parent installs a non-ignored no-op SIGCHLD disposition, then blocks
SIGCHLD, SIGINT, SIGTERM, and SIGALRM before pipe threads or spawn, while the
child starts with an empty mask and default relevant dispositions. ITIMER_REAL
must be inactive before acquisition and the runner never sets, cancels, or
overwrites it. After draining prerequisite pending state, the runner creates and
registers one Darwin `kqueue` for every controlled signal before `posix_spawn`.
That same already-registered queue drives the child and is closed on spawn
failure and every ordinary path. The main thread alone owns child signaling and
reaping. Kqueue timeouts derive from the active TERM/KILL deadline without
polling or process timers. The main immediately checks exact
PID `wait4(WNOHANG)` for every SIGCHLD, operator, or contamination signal and
timestamps `t_reaped_ns` immediately when that call returns the PID. Stale or
spurious SIGCHLD is harmless. On an operator signal it checks/reaps first and
sends TERM only if still live; expiration of the main-thread timed wait drives
bounded TERM-to-KILL escalation, again after an exact reap check. Pending or
observed SIGALRM is control contamination and makes the result inconclusive; it
initiates bounded cleanup but is never used as an escalation timer. The runner
checks it again while blocked immediately before reapplying the prior per-child
mask. During packet execution, SIGALRM is
reserved blocked before manifest creation and remains blocked through durable
sealing and process exit. Per-child evidence makes no atomic
snapshot-plus-restore claim. No fixed polling or sleep biases the normal
endpoint. No code sends TERM or KILL after successful reap. Never use a
name-based or pipeline kill. A
successfully reaped process always returns structured status, rusage,
timestamps, bounded buffers, drain state, and interruption state, including
negative exits and invalid drains.

For every reaped child, first durably persist raw streams and process outcome,
then capture and persist post-exit host, VM, pressure, and resource evidence
before classifying the exit. Persist `<stem>.conditioning.json` before launch
and bind its SHA-256 into the launch event. Invalid evidence is retained rather
than discarded.

Scored stdout must eventually be exactly `b"<think>\n"`, SHA-256
`9ebc01769b176bb074a065ea0974c130fc8afd12814360aaf809046160b2a999`.
Timestamp `t_stdout_complete_ns` immediately after the read that completes its
eighth byte; reject later bytes. Define:

```text
external_us = (t_reaped_ns - t_stdout_complete_ns) // 1000
```

Require one exact explicit publication line with `publish=published`,
`capture=completed`, `matched_tokens=6500`, `restored_tokens=6499`, pending
true, token-limit stop, 582,854,188 bytes, zero evictions, identity hit, and the
exact arm. `staged_integrity_us`, `publish_us`, and `post_response_us` are
canonical integers; `capture_ms` is finite and nonnegative. Require
`external_us >= post_response_us` as a non-scoring observation-quality check.
Failure of this check is observation-invalid `inconclusive`, not a parser or
implementation defect.
Require one store-empty line and one current stats line with prompt 6,499,
generated one, transitions zero, and token-limit stop. Reject checkpoint hits,
warnings, failed publication, corruption, or repair.

## Store And Equality Gates

After each child require exactly the identity file, `store.lock`, and one blob
at the canonical path. Every directory is mode `0700`, every file mode `0600`,
and every entry regular and nonsymlink. Reject temporary or foreign entries.
The blob must be link count one, 582,854,188 bytes, SHA-256
`69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65`,
with final 32-byte encoder trailer
`2833fd870009a299dbaa389ae7ed379ea69d50fc31ff250f4e349dc3056eeb67`.
Require exact global response/blob identity and pairwise streaming blob equality.
No eviction or temporary residue is allowed.

For pair `i`, independently score A minus B:

```text
P[i] = publish_us_A - publish_us_B
E[i] = external_us_A - external_us_B
```

A provisional performance pass requires `P[i] >= 250000` and
`E[i] >= 250000` for every pair, with B winning both endpoints. Medians and
order-stratum summaries are descriptive only and cannot rescue a failed pair.

## Unscored Restore

Only after all 12 valid rows provisionally pass, run exactly one unscored
restore from the chronologically first B store: pair 1, position 2. Condition it
again, use the pre-frozen extended messages and explicit B environment, change
maximum context to 8,192 and minimum cache tokens to 262,144, and otherwise
retain the command.

Require exactly one hit with `identity_cache=hit`, `hashed_bytes=0`,
`checkpoint_hit=true`, `matched=6500`, `restored=6499`, `exact=false`,
`candidates=1`, and `corrupt_removed=0`; clean exit; nonempty stdout; no
publication, warning, or corruption. No literal stdout digest is claimed. The
candidate blob path, device/inode, size, mode, link count, and SHA-256 remain
unchanged; mtime may change. No blob, temporary, or foreign entry is added.
After a valid restore, copy that candidate into the packet and fsync and hash it.

## Decision And Packet

Apply this exclusive precedence:

1. Clean-exit source, build, parser, marker, response, blob, topology, cleanup,
   or restore mismatch: `implementation_or_contract_defect`.
2. Operator signal, negative-signal exit, spawn/pipe/wait4 failure, invalid
   host, pressure, resources, or incomplete population: `inconclusive`.
3. Twelve valid rows with any P/E gate miss: `kill`; do not run restore.
4. Provisional performance pass plus valid restore: `go`.

Only GO grants authority
`force-only-exact-qwen3.6-27b-q4_k_m-deferred-restore` with
`force_authorized=true`. Default, automatic, other-model, and other-shape
authority are false, and successor authorization is `none`. Every non-GO result
has no authority. Absence of the environment control remains decode.

Use fresh roots
`target/profiles/v0638-checkpoint-deferred-restore-product-p1` and
`target/profiles/v0638-checkpoint-deferred-restore-product-work`. Never retry.
Durably seal launch/completion events, attempts, manifest, inputs, decision,
inventory, and completion.

Initialize all decision state before reservation. Reservation creates and
fsyncs the immutable `manifest.json` before any gate. Fresh post-gate build and
host identity is written separately to `execution-identity.json`; the manifest
is never rewritten. Input freezing, work-root creation, gates, children,
cleanup, final GO validation, decision publication, and descriptor-bound final
inventory all share one lifecycle envelope. Any failure after successful
reservation seals a defect or inconclusive packet, including an incomplete GO
conjunction. Only an unreaped exact PID or genuinely unsealable durability
failure may abort without a complete packet.

Final GO validation rederives rather than trusts authority booleans. It
descriptor-authenticates the current qwen binary, packet inputs, input seal,
execution identities, and retained candidate. It requires descriptor-bound
`gates.jsonl`, `rows.jsonl`, `restore.json`, and `attempts.jsonl`, with attempts
exactly equal to the 12 rows plus restore; exact launch/completion population and
order; unique stems/store roots; recomputed external endpoints; matching
conditioning, model, input, qwen, process, stream, and topology evidence; and no
pending signal or control error. It recomputes all six pair gates and descriptive
medians, then rederives the exact pair-1-position-2-B restore command, hit
telemetry, unchanged blob identity, stream semantics, and retained trailer.
Every gate, scored, and restore launch/completion schema, stage, stem, command,
and result is derived independently from protocol constants, never from recorded
row values. The validator returns a complete descriptor identity/digest map for
all packet authority inputs. Final sealing requires those exact identities to
remain unchanged; only cutoff and decision files are post-validation generated
members before inventory/completion metadata.

Independent validation also converts every persisted wait status back to its
return code, cross-links exact rusage, recomputes gate wall from process
timestamps, requires the sole-main signal-control strategy, rederives
conditioning model/host/VM identity, and independently reconstructs every
post-exit validity reason from VM and rusage evidence. Stable bytes alone cannot
make semantically altered process evidence authoritative.

Signal-release validation independently requires a canonical duplicate-free
controlled-signal snapshot, exactly rederives its operator subset and SIGALRM
membership, cross-links the process blocked state and previous mask, and
requires packet-lifecycle SIGALRM reservation. Every history record explicitly
requires schema 1, and every process requires strategy
`sole-main-kqueue-wait4`.

After cleanup and independent validation, while operator signals and SIGALRM
remain blocked, one `sigpending` snapshot is the authority cutoff. Only
controlled signal types in that snapshot are consumed and classified; operator
signals or SIGALRM contamination make non-defect status inconclusive before
authority is frozen. A timestamp taken immediately after the snapshot is
observational metadata, not a nanosecond arrival partition. Signals becoming
pending after the snapshot are explicitly outside packet authority.

Final sealing opens every inventory member with `O_NOFOLLOW`, requires regular
mode `0600` and link count one, hashes from that descriptor, and rechecks the
same full descriptor stamp and directory population before publishing the
inventory. Inventory and completion seals are themselves private, durable files.

Before source preflight or packet reservation, mandatory self-tests cover strict
telemetry parsing and adversarial negatives, inclusive 250,000 and failing
249,999 boundaries, descriptor/symlink rejection, a direct harmless
pipe/thread/wait4 child with exact output timestamp and rusage and no zombie, a
harmless exact-group interruption/reap, short test-only TERM-resistant SIGKILL
escalation, unexpected SIGALRM contamination, watchdog-bounded repeated
immediate-exit registration stress, and private store seeding. A nonprinting
core runs automatically at the start of preflight and packet execution;
`--self-test` runs the same core and then prints. Tests use explicit exceptions,
pass under `python -O`, and create no packet root. Preflight authenticates
source, build, inputs, host, and VM and may compile/run no model. Packet
execution is a separate invocation.

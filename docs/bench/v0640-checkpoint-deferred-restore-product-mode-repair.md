# v0.640 Deferred-Restore Real-27B Product Mode Repair Packet

Status: runner-only preregistration. No v0.640 preflight, build, model, GPU,
scored child, or restore has run.

v0.639 sealed a pre-model-child contract defect after all four gates because its
exact-`0755` policy rejected safe mode `0700` binaries created under umask `077`.

## Authority Bridge

The post-v0.639 PERF certification at
`fde40808ad5400c9e95466ef03cbc11ad17a8a49` independently authorizes exactly one
fresh v0.640 executable-mode repair packet, ordinal 2. v0.640 imports and
consumes that authorization. It imports or consumes no authority from v0.639;
the sealed v0.639 decision has `successor_authorization=none`. Every v0.640
result exhausts ordinal 2 and grants `successor_authorization=none`.

Authenticate the complete sealed v0.639 bridge at
`target/profiles/v0639-checkpoint-deferred-restore-product-repair-p1` without
deleting it or importing its result. Require directory mode `0700`, exactly 27
regular mode-`0600` link-count-one files, and a 25-member complete inventory:

```text
decision.json              f77a8704555e2a710815227feee31353dfafb3473acc4b2c6cd6498b8004e646
artifact-inventory.sha256  64a98e837acfe3aced637dc6f3c2b5f1c7d0bb37d1e1099b589792a281cd7a7b
packet-complete.json       8bcc59d7f844989ae70a0b685780e10ae8b59f6a64f24546216a8113e0dccb9e
gates.jsonl                d0d51ae471a64d83f0fcc7cddf13b23bbdd697558f340514d13f368ba47cdc35
launch-seal.jsonl          ce37208412648d0bf85c9901b8df0ececfde626485b332ef0e0f70460177bb7c
```

Require the exact sealed defect decision, error, and cutoff; four exact
successful gates; and eight exact gate launch/completion records with clean
process, release, and stream crosslinks. Require `build_identity=null`, empty
rows, null performance/restore/candidate, and no model-child, scored, restore,
attempt, execution/final identity, conditioning, post-exit, or retained-candidate
artifact. The v0.639 work root remains absent. Manifest model hash I/O is
acknowledged but contributes no product observation.

Record forensic authentication separately from contribution: seal
authenticated, packet reserved/completed, status defect, four gates
authenticated but zero imported, zero model children, and zero imported scored,
timing, or performance observations. The exact authority-history map is:

```text
authority_origin=fde40808ad5400c9e95466ef03cbc11ad17a8a49
authorized_packet=v0.640-executable-mode-repair
packet_ordinal=2
post_v0639_perf_authorization_imported=true
post_v0639_perf_authorization_consumed=true
v0639_successor_authorization=none
v0639_authority_imported=false
v0639_authority_consumed=false
v0639_result_imported=false
v0639_artifact_imported=false
v0639_gates_authenticated_forensically=4
v0640_authorization_exhausted_by_any_result=true
v0640_successor_authorization=none
product_authority_imported=false
gate_results_imported=0
scored_rows_imported=0
timing_observations_imported=0
performance_observations_imported=0
```

## Source And Build Identity

R640 must be a clean direct single-parent child of exact parent and authority
commit `fde40808ad5400c9e95466ef03cbc11ad17a8a49`, adding exactly this document and
`scripts/profile/v0640_checkpoint_deferred_restore_product_mode_repair.py`.
Authenticate the parent as a direct child of
`e2c3accbac7c330097970e1e672466b8a0c350fd`, modifying exactly
`docs/PERF-LOG.md` and `docs/PERF-ROADMAP.md`, with frozen committed SHA-256
`9b59930dad01eeccd6d0e80c3ee39c0872ac848f6363b9182acc69822c2bf4cb` and
`d6cee7afb87274393c066da5cdcb21694e9dc17d44c49b611c198f485b01b2ea`.
Authenticate ancestry through `e2c3acc` to `ca2ee114`, `482fa053`, `61f852e5`,
and unchanged implementation `3d59d94cf314dae79b79cd05f74b4aea455cd30c`.

At execution, both `qwen` and `qwen-bench` must carry R640 as build and runtime
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

The runner locally copies the exact canonical `git-source-sha256-v2`
`hash_section`, `hash_worktree_entry`, `tracked_source_state`, and
`source_identity` mechanics from `scripts/bench/family.py`. It independently
derives full R640 HEAD, clean and hidden-index state, and source state; expected
state never comes from `qwen-bench`.

Descriptor-open both binaries with `O_NOFOLLOW`. The canonical
`SAFE_EXECUTABLE_MODES` is exactly `{0700,0755}`. Require regular type, link
count one, and observed mode in that set; retain mode in the complete SHA-256,
device, inode, bytes, mode, link count, mtime, and ctime descriptor identity.
Record one canonical mode-policy object in the immutable manifest and post-gate
build identity. Raw policy is asymmetric and recorded exactly: `qwen` requires
the full R640 commit and source
state literals; `qwen-bench` requires source state, while contiguous full-commit
presence is irrelevant diagnostic evidence.

Run `qwen-bench build-info --output json` with complete descriptor identity
authentication immediately before and after. Require schema 2, full build and
runtime R640 commits, short `R640[:9]`, both dirty fields false, both source
states equal the independently derived state, `stamp_source=git`, null
`stamp_error`, `status=match`, and empty problems and overrides. Any semantic or
complete before/after identity mismatch fails authentication.

Run exactly one fresh release build gate for both `qwen` and `qwen-bench`, then
the same release checkpoint-codec, checkpoint-store, and qwen CLI test gates.
Every gate is sole-attempt, no-retry, and durably launch/completion sealed.
Immediately before every scored and restore spawn, authenticate the complete
current descriptor identity of `target/release/qwen` against its post-gate
identity, not only SHA-256 and link count. Bind that complete identity into the
launch seal. Reauthenticate it after the final child and while rederiving GO.
GO validation also reauthenticates complete `qwen-bench` identity and semantic
build-info and binds that evidence through final sealing; `qwen-bench` does not
require per-child rehashing.

GO freshly rederives R640 HEAD, clean and hidden-index state, canonical source
state, and a fresh exact mode-policy object independently of recorded build
identity. It compares manifest and post-gate policy before descriptor-reading both
current binaries, reapplies the asymmetric raw policy against that fresh tuple,
requires exact post-gate descriptor and raw-authentication records, and brackets
semantic build-info with complete stable `qwen-bench` identity. It independently
requires the manifest's exact authority-history map before granting authority.

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
output. Never chmod real binaries or alter Cargo gate behavior. Invoke
`target/release/qwen` directly with this exact argument shape:

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
`target/profiles/v0640-checkpoint-deferred-restore-product-mode-repair-p1` and
`target/profiles/v0640-checkpoint-deferred-restore-product-mode-repair-work`.
Both must be `lexists`-absent before self-test, preflight, and reservation, and
the v0.639 work root must remain absent. Import no v0.639 result. Never retry.
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

Before source preflight or packet reservation, mandatory self-tests cover the
asymmetric raw policy directly: accept bench source state with and without a
contiguous commit; reject qwen missing commit or state, bench missing state,
every semantic mismatch, and before/after complete identity drift including
same-byte inode replacement. They accept exact executable modes `0700` and
`0755`, prove umask `077` creates and accepts `0700`, reject `0600`, `0644`,
`0701`, `0750`, `0775`, and `0777`, and require exact fresh-copy mode-policy
equality. Mandatory self-tests also cover strict
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

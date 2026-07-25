# v0.628 Dense-27B Default-ColdOnly Target-File-Cold Composition Guard

Status: preregistered composition guard. Every outcome has `authority=none` and
`force_authorized=false`; this packet can never emit `go`.

## Sole Question And Scope

For the exact dense Qwen3.6 27B W4 file, does the actual
`LoadedModelConfig::default()` ColdOnly path compose safely with ordinary copied
loading and dense direct `pread` when each sole child invalidates the wholly
resident target file immediately before load? This is the sole successor allowed
by v0.627. It is a four-child, two-pair, fail-closed composition guard, not an
effect-size, MDE, equivalence, product, or promotion experiment.

The v0.627 performance and full-state correctness results are imported only as a
bridge. They are authenticated but never pooled, rescored, reinterpreted, or
promoted. This packet excludes the exact 419-prompt-token/output-128 cold TTFT
cell and instead uses the existing `first_byte_spike` five-token prompt and one
first token (`--tokens 0`).

## Complete v0.627 Bridge

Before reserving an artifact, authenticate
`target/profiles/v0627-dense27b-parallel-pread-fresh-repair-p1` completely:

- decision SHA-256
  `637184029bab7484c7185ff62bda13d2dbb775628471e031fe1951c25ab5f749`;
- inventory SHA-256
  `110941aaac80583f54a8122a94af15936114cef54e5c3d10f08b8c9d4c6ec9d8`;
- completion-file SHA-256
  `c9e6d1b25bc74534de88a6ae4726c5167c50b5e5e819a2ca7afae9e865b875af`;
- completion object exactly binds the decision and inventory digests;
- exactly 84 unique, safe inventory members exist and rehash, and the final
  directory consists of exactly those members plus inventory and completion,
  for 86 files total and no subdirectory, symlink, temporary, or extra member;
- the decision is `fresh_effect_pass`, source
  `26d805a911b55bdc5bf141637d9c7d8cd676f251`, `authority=none`,
  `force_authorized=false`, and successor exactly
  `preregister-and-execute-separate-default-coldonly-target-file-cold-guard-only`;
- all four CPU selectors, full-state correctness, every fresh gate, and all 12
  unique attempts passed; the exact attempts ledger, launch/completion sequence,
  conditioning, raw output, timing, post-exit evidence, final identity, model,
  prompt, golden, source, build, runtime, and binary identities remain bound.

The runner rehashes every sealed member and authenticates the complete v0.627
publication. It additionally reparses selected terminal invariants: decision and
successor scope, four passed selectors, passed correctness, all-true published
fresh gates, the exact 12-attempt/order prefix, launch/completion order, sealed
build/model/prompt/golden identity, and final model identity. It does not claim
to independently reconstruct every v0.627 metric from raw artifacts. No v0.627
authority exists to import. v0.605 supplies host/VM/process/launch/durability
mechanics, v0.626 supplies dense constants/load-marker/final-identity patterns,
and v0.624 supplies cold stdout/process/analysis ideas only; v0.628 owns
dense-specific validity.

## Source Ladder And Identity

The clean certification base is
`7fc4c6b2e08379bd45be8b9db8dc742aa5e7a04e`. Preregistration commit P must be
its direct child and add exactly this document and
`scripts/profile/v0628_dense27b_pread_cold_composition.py`. Harness
implementation H must be P's direct child and change exactly
`crates/qwen-llm/examples/first_byte_spike.rs`.

At execution require repository-root cwd, a clean worktree,
`HEAD^^=7fc4c6b2e08379bd45be8b9db8dc742aa5e7a04e`,
`7fc4c6b..HEAD^` exactly the two packet files, and `HEAD^..HEAD` exactly the
named example. Relative to dense implementation ancestor
`7fb0488a5c075d8c62c8f9802352f717ec95b085`, `Cargo.lock`, every Cargo manifest,
`.cargo`, `crates`, and `kernels` must be identical except for that example.

Release `qwen-bench build-info` build/runtime identity must be clean, matching,
and exactly H. The hashed release `qwen` must embed H and the same exact source
state. Execute the separately hashed release example binary directly under
`/usr/bin/time -l`; do not use `cargo run`. The example embeds its own source via
`include_str!`, emits that source's SHA-256, and every child must match the
manifest-bound H example source. Manifest-bind both binaries, the example,
model, default prompt literal, v0.627 seal and all members, packet source, and
both live v0.605 mechanics/common sources. Recheck every non-model identity
between gates and children, and publish a final complete model hash and file
identity report.

## Anticipated Harness-Only Change

H adds `PolicySelection::{DefaultConfig, Explicit(PrefetchPolicy)}`. Omitted
`--policy` remains explicit Off. Existing explicit `off`, `always`, and
`cold-only` behavior remains unchanged. New `--policy default` constructs
`LoadedModelConfig::default()` through a genuinely distinct path and rejects any
nonzero `--workers` or `--chunk-mib` override.

H adds the exact CPU-only example test
`tests::default_policy_selection_is_exact`. It must prove exact default-config
equality, omitted-policy explicit Off, all existing explicit policies,
default-plus-worker/chunk rejection, and all stable metadata below. v0.628 runs
this exact release example test first, singly with `--exact --nocapture
--test-threads=1`; missing, filtered, duplicate, malformed, or nonzero evidence
is `implementation_or_contract_defect`.

Stable machine evidence uses one exact equals grammar, one field per line:

```text
harness_source_sha256=<64 lowercase hex>
policy_source=default
policy_configured=cold-only threshold=0.9 workers=0 chunk_bytes=0
target_identity_before_invalidate=dev:<u64>,ino:<u64>,size:16817244384,mtime_ns:<i128>
target_identity_after_invalidate=dev:<u64>,ino:<u64>,size:16817244384,mtime_ns:<i128>
policy_observed=cold-only threshold=0.9 action=configured-policy suppressed=false
target_identity_after_load=dev:<u64>,ino:<u64>,size:16817244384,mtime_ns:<i128>
prefetch_exact=events:1,prefetched:1,skipped:0,bytes:16817244384
timing_exact=load_us:<u128>,first_byte_us:<u128>
rusage_exact=pageins:<u64>,disk_read_bytes:<u64>,disk_write_bytes:<u64>,rss_delta_bytes:<i64>,footprint_delta_bytes:<i64>
```

The configured workers/chunk values are exactly the zero sentinel fields of
`LoadedModelConfig::default()`, not resolved internal worker/chunk values. H
asserts `outcome.policy == config.prefetch_policy`, and default mode action is
`PrefetchAction::ConfiguredPolicy`, never suppressed. Stable lines derive from
typed fields and explicit matches, not Debug alone.

Capture target device, inode, size, and nanosecond mtime before invalidation,
after invalidation, and after load/post; assert all three equal. Preserve the
existing pre-arm/invalidate/post residency lines, phase walls,
`proc_pid_rusage` physical bytes, first token, and external `/usr/bin/time`
compatibility.

The runner gates exact integer microseconds and physical bytes from the stable
lines, not the rounded human summaries. The human values must reconcile within
their display precision. All stable tokens, prefetch events, native-policy
lines, storage markers, and ledgers have exact raw occurrence counts before
line parsing; mid-line, duplicate, extra, or unknown occurrences fail closed.

## Frozen Cell And Order

After the CPU test, run exactly four sole children in order `AB` then `BA`, with
no retry or replacement. Every launch has durable conditioning evidence first.
The exact command is:

```text
/usr/bin/time -l target/release/examples/first_byte_spike \
  /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf --policy default --invalidate \
  --prompt "The capital of France is" --tokens 0 --intent force-only
```

No worker or chunk flag is allowed. Arms alter only the normalized environment:

- A: `QWEN_GGUF_PARALLEL_COPY=0`;
- B: `QWEN_GGUF_PARALLEL_COPY=pread`.

No no-copy, owned, native, suppression, Auto, prefetch-policy, worker, chunk, or
other population override is admitted. Both arms use actual
`LoadedModelConfig::default()` ColdOnly.

## Per-Child Contract

Every child must prove all of the following:

- target device/inode/size/mtime equals the manifest before invalidation, after
  invalidation, and after load; page size is 16,384, derived total pages is
  exactly 1,026,444, and pre-arm residency is 1,026,444/1,026,444;
- in-child `MS_SYNC|MS_INVALIDATE` changes 1,026,444 resident pages to zero;
- source is default; configured and observed policy is ColdOnly threshold 0.9,
  configured workers/chunk are 0/0, action is `ConfiguredPolicy`, and
  suppression false;
- exactly one target prefetch warmed event and one prefetched shard, zero skipped,
  and exactly 16,817,244,384 logical bytes returned; the human line may round to
  15.66 GiB, but the exact stable line is binding;
- no suppression, already-warm skip, probe failure, shard failure, extra
  prefetch event, or extra/unknown storage marker appears;
- post residency is at least 99%; exact process-wide physical reads are in
  [13,421,772,800, 18,522,046,464] bytes (12.50-17.25 GiB); child major faults
  and swaps are zero, while cold physical reads and block input are expected and
  are not rejected merely for being cold;
- A has the exact native embedding policy and copied ledger with no GGUF storage
  marker; B has that policy, exactly one dense schema-2 direct-pread marker, the
  same ledger, and marker major faults zero;
- first token ID and piece agree across all four children. No token is
  preimported as golden; v0.627 full-state correctness is the correctness bridge.

The physical-read metric is process-wide `proc_pid_rusage`, so it cannot assign
every byte to one loader phase or distinguish all filesystem/cache activity.
Exact target identity, in-child invalidation, residency, prefetch bytes, storage
markers, and the absolute window jointly bound attribution; no stronger claim is
made. RSS and process exit remain diagnostics, while peak physical footprint is
gated.

Host identity, AC power, thermal state, memory headroom, VM deltas, swap,
compressor/pressure state, child cleanup, and interruption evidence must be
valid. Cumulative counter regressions, swap growth, invalid host state, nonzero
exit, malformed output, failed durability, or incomplete capture are
inconclusive unless they contradict the implementation contract.

## Pairwise Gates

Compute each A/B pair from exact integer timing/read evidence before any
aggregate. Both pairs, not medians, must pass:

- load `B-A <= 0 ms` (equivalently A-B saving at least zero);
- first-byte `B-A <= 0 ms`;
- physical-read `B/A <= 1.10` and each child independently in the absolute read
  window;
- complete-process CPU `B/A <= 1.10`;
- peak physical footprint `B/A <= 1.05`.

Any valid pairwise miss is `kill`. There is no effect-size or MDE claim.
Malformed or contradictory identity, policy, stable protocol, residency,
marker, ledger, or correctness evidence is an implementation/contract defect;
the frozen numeric pairwise gates remain reachable and classify only as `kill`.

## Durability, Decision, And Closure

Use an exclusive never-reused artifact. Fsync test output/metadata, raw stdout
and stderr, attempts, launch/completion, conditioning, post-exit, failure,
identity, inventory, decision, and completion evidence. A launch cannot occur
until its conditioning record is durable. First creation of every append-only
ledger is directory-fsynced. Defer SIGINT while test/model children are reaped;
do not let a child inherit an ignored disposition. Capture and durably write
post-exit host/VM state before decoding or semantic parsing. Verify exact
artifact names and hashes, reject extras, cross-bind launch, completion,
conditioning, raw, post-exit, attempt, and recomputed terminal analysis, and
seal an exact attempted prefix after early stops. Always attempt the final
non-model check, complete model rehash, and file identity report before decision
publication. An exact existing identity report is reusable after a
pre-publication verifier failure; a differing report is unsealable. Inventory
and completion are member-complete and directory-fsynced.

Statuses are only `implementation_or_contract_defect`, `inconclusive`, `kill`,
and `cold_composition_pass`, in that precedence. Every status has
`authority=none` and `force_authorized=false`. `kill` and pass require exactly
four completed sole attempts; early outcomes bind only their exact prefix.

`kill` closes dense transfer. `cold_composition_pass` authorizes only
preregistration of a separate short-period loaded-stability packet. It grants no
force, Auto/default population, product, exact-request-cold, serving,
concurrency, energy, untouched-media, or other-model authority. It never grants
`go`; every unlisted successor is `none`.

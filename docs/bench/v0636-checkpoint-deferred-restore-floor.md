# v0.636 Checkpoint Deferred-Restore Publication Floor

Status: preregistration. No v0.636 implementation, exact-size floor, model,
GPU, or timing work has run.

## Question And Scope

Can a disposable durable checkpoint publish a fsynced encoder output without
immediately rereading the complete staged inode, while retaining the unchanged
encoder/wire format, namespace durability, packet-level byte equality after
publication, and mandatory adversarial validation before every restore?

The current `decode` path rereads, hashes, parses, and materializes the complete
staged record before hard-link publication. v0.622-v0.623 replaced that semantic
decode with allocation-free full readback, but both arms still reread and hash
all `582,854,188` bytes. The replacement saved only `17.4/18.1 ms` and no product
wall. That branch is closed.

This packet changes the premise: B performs no staged-inode read. It prices only
a CPU exact-size floor. It runs no model or GPU work and cannot authorize product,
default, automatic, or even force-only CLI use. GO authorizes one separately
preregistered real-27B product packet; nothing else.

## Integrity Contract

The two explicit arms are:

- A, `decode`: the current materialized staged decode;
- B, `deferred-restore`: no staged seek, read, digest verification, semantic
  decode, or state materialization before publication.

Absent configuration remains A. Invalid, empty, non-Unicode, misspelled, or
case-variant configuration fails before model loading. B's staging descriptor is
write-only; A's is read/write.

Both arms must retain:

1. Source-snapshot identity, shape, range, capacity, and section-length
   validation before writing.
2. The exact v1 record format. `encode_snapshot` hashes each byte sequence only
   after its `write_all` succeeds, appends that BLAKE3 digest, and returns the
   exact record length and intended digest.
3. `BufWriter::flush`, staging-file `sync_all`, exact mode `0600`, regular-file
   type, exact record length, and stable descriptor device/inode checks.
4. A retained staging descriptor through publication; staging-path metadata must
   match it immediately before linking.
5. Existing no-clobber hard-link publication, eviction, staged/final inode
   equality, exact final mode/size, two links while both names exist, and final
   directory `fsync` before success.
6. Current best-effort temporary-link cleanup on both arms. Clean packet children
   require no residue and one final link. Crash-durable cleanup and hard cleanup
   errors remain default-promotion work, not an A/B difference here.
7. Full adversarial decode for existing-winner admission and every lookup. The
   decoder must still validate header, compatibility and structural ABI, bounded
   lengths, padding, payload digest, exact EOF, token/state semantics, filename
   namespace, and complete destination spans before any Metal mutation.

The exact product statement is deliberately weaker than current A:

> `deferred-restore` publication creates a durable but content-unverified cache
> candidate. It retains the unchanged canonical encoder and wire format and
> establishes only that source validation, encoder writes, flush, file
> synchronization, metadata checks, hard linking, and directory synchronization
> reported success. It does not establish before publication that staged-inode
> contents equal the intended encoder byte stream. Every admission, lookup, and
> restore still performs full adversarial record validation before Metal
> mutation. Corruption may consume budget or evict disposable entries before a
> later miss or fallback.

The store remains private same-user disposable cache state. A malicious writer
that replaces state and recomputes a valid unkeyed digest, or a process racing
parent path components, remains outside the existing threat model. Checkpoints
must never become authoritative state under this contract.

## Frozen Implementation Boundary

The base is `b6a212ab78049607fae9ba9b0ccc3535e3fa7254`. R is its clean,
single-parent child and adds exactly this document plus
`scripts/profile/v0636_checkpoint_deferred_restore_floor.py`. H must be R's clean,
single-parent child and modify exactly:

```text
crates/qwen-cli/src/main.rs
crates/qwen-llm/src/checkpoint_store.rs
```

H must not change `checkpoint_codec.rs`, the wire format, restore, identity,
eviction, directory-sync ordering, or Metal code. The runner derives and seals R
and H, authenticates both diffs, requires a clean worktree and frozen sibling
repositories, and records all source and input hashes. There is no repair commit
or retry authority.

The smallest intended seam is:

- explicit `StagedIntegrityMode::{Decode, DeferredRestore}` store configuration;
- `DurableCheckpointStore::new` remains decode-default;
- CLI parses `QWEN_CHECKPOINT_STAGED_INTEGRITY` before model loading;
- a private `StagedBlob` retains the file, path, encoder result, opening stamp,
  and integrity report through publication;
- a pure file-stamp validator covers type, length, mode, device, inode, and link
  count;
- when the research environment is explicit, success telemetry names
  `staged_integrity=decode|deferred-restore`, `staged_integrity_us`, `publish_us`,
  and `post_response_us` without calling B verified;
- explicit-mode failure telemetry reports `publish=failed`, requested mode, total
  publication-attempt duration, and either omits staged-integrity duration or
  records it as unavailable. Absent configuration retains current decode behavior
  and the current publication-line grammar.

Shared metadata checks on A are correctness hardening only. They carry no
performance or default authority.

## Required CPU Tests

Before the floor, capture and seal a release qwen build, all ordinary checkpoint
codec/store tests, and the qwen CLI test binary. Focused coverage must include:

- strict mode parsing and decode default;
- every pure file-stamp mismatch;
- byte-identical small records across modes;
- retained descriptor/staging/final hard-link identity;
- deferred same-size corruption followed by full lookup rejection and clean
  fallback or miss;
- full deferred-record restore through the existing decoder;
- mixed decode/deferred concurrent publishers converging on one decodable inode.

Existing codec writer-fault, corruption, fallback, resource-failure, concurrency,
and namespace tests remain the authority for unchanged paths. A generic
filesystem-operations abstraction and exhaustive syscall fault injection are not
part of this force-only floor; they are mandatory before any default promotion.

## Exact-Size Floor

Run four independent release-test children in exact pair order `AB BA`, once each
with no retry. Each gets a unique empty mode-`0700` private work root and invokes
only:

```text
cargo test --release -p qwen-llm \
  checkpoint_store::tests::checkpoint_deferred_restore_exact_size_floor \
  -- --ignored --exact --nocapture --test-threads=1
```

The ignored test decodes the existing v0.615 real 27B checkpoint fixture before
timing, then republishes that immutable `SessionSnapshot` under its explicit arm.
The frozen input is:

```text
path
  target/profiles/v0615-long-history/cache-ring0/v1/blobs/
  fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e/
  6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp
bytes
  582854188
SHA-256
  69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65
encoder BLAKE3
  2833fd870009a299dbaa389ae7ed379ea69d50fc31ff250f4e349dc3056eeb67
```

Its frozen v1 header describes prefix count 6,499 plus pending token 248,068,
16 attention layers, 48 GDN layers, F16 KV, and these exact section lengths:

```text
KV K              212959232
KV V              212959232
GDN convolution     5898240
GDN state          150994944
prefix tokens          25996
KV positions              128
final logits                  0
```

The identity is layout 4, KV dimension 1,024, 2,048 KV bytes/token, 786,432 GDN
state elements/layer, and 30,720 convolution elements/layer. Frozen weak model
and tokenizer IDs are `7081852628295403893` and `11867181210256840983`; state
encoding is 1. Compatibility ID is
`fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e`.
Decoder vocabulary is exactly 248,320, maximum context is 6,516, and both the
per-record and aggregate store budgets are exactly 805,306,368 bytes. The only
accepted final namespace key is:

```text
v1/blobs/fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e/
6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp
```

This freezes actual section topology and contents rather than letting H choose a
synthetic allocation shape. Fixture decode and source initialization occur before
timing. Each child then publishes one fresh record and, outside both timed regions,
hashes and fully decodes the final record.

Each child must emit one canonical marker containing arm label, exact record
bytes, encoder digest, final SHA-256, integer-microsecond walls, outcome
`published`, zero evictions, managed bytes, restored mode/prefix facts, mode
`0600`, one final link, and no temporary residue. The runner independently
rehashes and inspects the store. All four records and decoded snapshots must be
byte-identical to the frozen fixture.

Timing boundaries are exact:

- `staged_integrity_us`: after staging-file `sync_all` through completion of the
  selected decode or deferred metadata acceptance;
- `publish_us`: immediately around the complete `store.publish` call, after
  fixture decode and through current temporary cleanup;
- `full_decode_us`: post-publication verification only and non-scoring.

For each pair compute A minus B:

```text
I = staged_integrity_us_A - staged_integrity_us_B
P = publish_us_A          - publish_us_B
```

GO requires, in both AB and BA:

- `I >= 250000 us`;
- `P >= 250000 us`;
- B wins both named walls;
- every byte, format, topology, cleanup, and restore gate passes.

The 250 ms threshold is the active roadmap's practical product gate, not an MDE.
This floor intentionally excludes whole-process wall and global peak footprint:
post-timing verification equalizes process work, and v0.623 proved an earlier
high-water mark masks the removed publication allocation.

The runner seals the exact effective child environment plus separate retained,
removed-name, and overridden-value records. It durably records every build/test
and floor launch/completion. Operator SIGINT/SIGTERM terminates only the exact
new process group, makes the packet inconclusive, and is blocked through cleanup
and final packet publication. No child or gate is retried.

## Decision And Successor

Apply exclusive precedence:

1. Source, implementation, parser, marker, byte, topology, cleanup, or restore
   defect: `implementation_or_contract_defect`.
2. Interrupted launch, negative signal exit, operator SIGINT/SIGTERM, or
   incomplete child/artifact population: `inconclusive`.
3. Complete exact floor missing either pair gate: `kill`.
4. Complete conjunction: `go`.

Every result has `authority=no-production-authority` and
`force_authorized=false`. Only GO grants
`successor_authorization=preregister-one-real-27b-deferred-restore-product-packet`.
That successor must use six `AB/BA/BA/AB/AB/BA` pairs, unique seeded stores, the
real `582,854,188`-byte completed checkpoint, publication and final-response-flush
to exit endpoints, the same 250 ms gate, and an unscored restore only after the
performance decision passes. It cannot authorize default promotion. KILL removes
the branch; inconclusive or defect grants nothing. There is no `cx` authority.
The earlier 500 MB global high-water alternative is non-scoring here and in that
successor unless a phase-local footprint sampler is separately preregistered.

Adversarial design review: `cx` session
`019f9bc1-53e5-72f2-a6e8-3da71a17bc4f`.

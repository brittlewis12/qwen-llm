# v0.637 Deferred-Restore Floor Parser Repair

Status: runner-only preregistration. No v0.637 gate, floor, model, GPU, or
product work has run.

## Sealed Predecessor

v0.636 is sealed `implementation_or_contract_defect` with
`authority=no-production-authority`. Its Cargo-libtest marker was present but
the runner did not recognize the exact libtest-prefixed physical line. v0.637
imports zero v0.636 authority, gate results, scored rows, timing observations,
or performance observations. The only inherited forensic facts are that one A
child launched and no B observation exists.

The manifest and decision must state all of these fields explicitly:

```text
v0636_authority_imported=false
v0636_gate_results_imported=false
v0636_scored_rows_imported=0
v0636_performance_observations_imported=0
v0636_timing_observations_imported=0
v0636_launched_children_forensic=1
v0636_b_observations_forensic=0
```

No v0.636 timing, gate, row, or decision may contribute to v0.637. This is a
fresh sole-attempt runner packet, not a continuation or retry.

## Frozen Commits And Tree

The frozen commits are:

```text
base  b6a212ab78049607fae9ba9b0ccc3535e3fa7254
R636  2acee4244a6b3f5bd9757c36d0911bcb5114e53a
H636  3d59d94cf314dae79b79cd05f74b4aea455cd30c
```

R637 must be a clean single-parent child of H636. It adds exactly this document
and `scripts/profile/v0637_checkpoint_deferred_restore_floor_parser_repair.py`.
It modifies no existing file. R637 is the execution commit; H636 remains the
implementation commit. Both identities must appear separately in the manifest
and decision.

The runner authenticates base-to-R636 and R636-to-H636 ancestry, changed paths,
and complete binary patch hashes:

```text
R636 patch  8f6e51fb3f90b8ff810a925daaebe3fab6112b92f68a07ba2992fe0e90a0c77c
H636 patch  30babdc372cccb5dafb5d28b69c7a2db56620cf028c0e691bd5bc12d4ded78bf
```

The H636 implementation and Cargo files are frozen at these SHA-256 values:

```text
Cargo.lock
  2ea5d138ef0b5813ce29ad39c39d4a34d31d5a72fb71431f6038c5343d2e4fbe
Cargo.toml
  c92ed7c228b43de3a2920cc5d2a7535660fd328cbde1444bf315545c36b162e9
crates/qwen-cli/src/main.rs
  1ee1e959fe5c2616fec20fb924b9edef026fc432b635db1f46166a43a1ce0c71
crates/qwen-llm/src/checkpoint_store.rs
  a5ea6b001249418febb3f876253bf5989c0603ef42cc43bc50fda7d51927f8da
```

The runner requires a clean worktree and the same frozen `gguf` and
`llama-cpp-rs` siblings as v0.636. It records all source and input hashes. Cargo
files and all implementation files remain unchanged. Binary equality with a
v0.636 build is neither required nor an authority source.

## Parser Repair

The parser consumes child stdout and stderr separately as bytes. It first
requires strict UTF-8 decoding of each stream; replacement decoding is
forbidden. It then requires exactly one literal
`[checkpoint-deferred-floor]` prefix in stdout and zero in stderr.

The parser splits stdout into physical lines on byte LF and may remove one
terminal CR from each physical line. The marker line may have only one of these
leads:

```text
<empty>
test checkpoint_store::tests::checkpoint_deferred_restore_exact_size_floor ... <SPACE>
```

After removing the permitted lead, the bytes from the marker prefix through the
physical line end must full-match the existing canonical v0.636 marker grammar.
Trailing data, duplicate prefixes, a split prefix, a wrong libtest lead, a
marker in stderr, malformed field order, and invalid UTF-8 are contract defects.
There is no unconstrained combined-stream or substring search.

A mandatory parser self-test runs before source preflight and before any normal
artifact reservation. It uses explicit `ContractDefect` checks and remains
effective under Python optimized mode (`python -O`). Failure therefore creates
no packet directory. It covers the standalone marker, exact libtest-prefixed
marker, malformed and adversarial forms, and the inherited exact 250,000-us
boundary analysis.

The same pre-reservation step authenticates the complete sealed v0.636 forensic
bridge. It opens the artifact directory and every member descriptor with
`O_NOFOLLOW`, binds regular-file metadata and contents to each descriptor, reads
each of the 16 files once, and rejects population or inode drift. It authenticates
the 14-member inventory and every member from those retained bytes, then validates
the completion hash bindings and counts. It requires the exact defect,
no-authority, no-force, and no-successor decision; four gate launch/completion
pairs; one successful A launch/completion; and no B launch. It then parses the
authenticated A stdout together with the authenticated A stderr from:

```text
target/profiles/v0636-checkpoint-deferred-restore-floor-p1/
floor-p01-ab-r1-a.out
```

The bridge seals are:

```text
decision.json
  793a81c8c49fc97b1a2092602be9ae3f3ae3c5d7100f0b8fd3ebd53d3b3b4790
artifact-inventory.sha256
  19955c6eab0848c597a23c3d7c745cc6106b77477d8fa8195fe2bc5bec833de5
packet-complete.json
  6cbeb588919610e8419ad82a4a6b0c5fc62c078119f93c5f4f4f19944af1b987
launch-seal.jsonl
  bd0ff4fb91d20e61b2ebf49dc417602cf95eb9ee09c220bcce7ab486432d5e92
floor-p01-ab-r1-a.out
  ea84c12edc1a9e4bc765e90e7b24ec2cd503f8527b0fae3dadf677dd5f90591f
floor-p01-ab-r1-a.err
  2f37294743ffba841d4363980da35636ea4ae6bfa6154b9f09456aa65d635ae5
```

The authenticated bridge object is recorded in source, manifest, and decision
data with `parser_fixture_authenticated=true`. Parsing that line proves only the
parser repair. Its timing and row are not imported.

## Frozen Substantive Protocol

Except for execution identity, artifact names, zero-import bridge fields, and
the parser repair above, v0.636 is the frozen substantive protocol. The arms
remain:

- A, `decode`: materialized staged decode before publication;
- B, `deferred-restore`: no staged seek, read, digest verification, semantic
  decode, or state materialization before publication.

Both arms retain the exact encoder and v1 wire format, source validation,
post-write hashing, flush and file synchronization, mode `0600`, regular-file
and descriptor identity checks, retained descriptor, no-clobber hard-link
publication, eviction behavior, inode equality, directory synchronization,
temporary cleanup, and full adversarial decode before every admission, lookup,
or restore. Absent configuration remains A and invalid configuration fails
before model loading.

The frozen fixture remains the v0.615 real-27B checkpoint:

```text
bytes   582854188
SHA-256 69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65
BLAKE3  2833fd870009a299dbaa389ae7ed379ea69d50fc31ff250f4e349dc3056eeb67
```

Its compatibility ID, namespace, 6,499-token prefix plus pending token 248,068,
16 attention layers, 48 GDN layers, F16 KV, section lengths, weak model and
tokenizer IDs, vocabulary 248,320, maximum context 6,516, and exact
805,306,368-byte record and store budgets remain exactly those preregistered in
v0.636. Fixture decode and source initialization remain outside timing. Each
child publishes one fresh record and then independently hashes and fully
decodes it outside both scored regions.

## Fresh Gates And Floor

Run all four ordinary release gates again, once each and without retry:

1. release `qwen` build;
2. all checkpoint codec tests;
3. all checkpoint store tests;
4. the qwen CLI test binary.

Then run four entirely fresh sole-attempt CPU children in exact pair order
`AB BA`. Every child uses a unique empty mode-`0700` private work root beneath
the v0.637 work root and invokes only:

```text
cargo test --release -p qwen-llm \
  checkpoint_store::tests::checkpoint_deferred_restore_exact_size_floor \
  -- --ignored --exact --nocapture --test-threads=1
```

The exact v0.637 roots are:

```text
artifact  target/profiles/v0637-checkpoint-deferred-restore-floor-p1
work      target/profiles/v0637-checkpoint-deferred-restore-floor-work
```

They are unique and may not be reused. Environment sealing, one-launch records,
process-group isolation, SIGINT/SIGTERM handling, durable completion records,
exact cleanup, and final packet publication remain unchanged from v0.636. No
child or gate is retried.

The marker, independent topology inspection, byte equality, encoder digest,
final SHA-256, restored snapshot, exact namespace, mode, link count, zero
evictions, managed bytes, and zero temporary residue checks remain unchanged.
All four records and decoded snapshots must equal the frozen fixture.

The exact timing boundaries remain:

- `staged_integrity_us`: after staging `sync_all` through selected integrity
  completion;
- `publish_us`: around the complete `store.publish` call;
- `full_decode_us`: post-publication verification only and non-scoring.

For each pair, compute A minus B independently:

```text
I = staged_integrity_us_A - staged_integrity_us_B
P = publish_us_A          - publish_us_B
```

GO requires `I >= 250000` and `P >= 250000` microseconds in both AB and BA, B
winning both named walls, and every byte, format, topology, cleanup, and restore
gate passing. Whole-process wall and global peak footprint remain non-scoring.

## Decision And Authority

Apply the unchanged exclusive precedence:

1. Source, implementation, parser, marker, byte, topology, cleanup, or restore
   defect: `implementation_or_contract_defect`.
2. Interrupted launch, negative signal exit, operator signal, or incomplete
   child/artifact population: `inconclusive`.
3. Complete exact floor missing either pair gate: `kill`.
4. Complete conjunction: `go`.

Every result has `authority=no-production-authority` and
`force_authorized=false`. GO authorizes only
`preregister-one-real-27b-deferred-restore-product-packet`: one separately
preregistered real-27B product packet using the v0.636 successor protocol. It
does not authorize production, force use, default promotion, automatic use, or
any product behavior. KILL closes the branch; inconclusive or defect grants
nothing.

# v0.622 Allocation-Free Checkpoint Staged Validation

Status: sealed implementation/contract defect with no authority. The first A
child completed, but the runner rejected the current stats grammar before
emitting a scored row; no B child launched. v0.623 supersedes this packet.

## Intent

Determine whether replacing publication's materialized staged-file decode with
the exact encoder-digest readback removes at least `250 ms` of process wall or
`500,000,000` bytes of peak physical footprint on a real 27B checkpoint.

This changes only validation of a file written and fsynced by the same encoder
invocation. Persisted lookup, existing-winner validation, corruption repair,
record format, file and directory durability, hard-link publication, eviction,
and restore retain the full adversarial decoder.

## Evidence And Identity

Pin primitive commit `b9362e6c2c6aa3f2530ce565ae100a4f6552c99a` and
integration commit `a0aec06fea54c6be9484f3a802b29775d82166fe`. Require
that the integration changes only `checkpoint_store.rs` and qwen `main.rs`, and
that this packet commit is its direct child adding only this contract and its
runner.

Hash the clean source state, exact release binaries, 27B model, frozen v0.615
messages fixture, and frozen strong-identity seed. Require matching embedded
build/runtime identity and the M4 Max host boundary before launch.

The ignored v0.615 inputs are copied and hashed into the packet before any child
starts, then sealed in its final inventory. Freeze:

- messages SHA-256
  `5c2455738af1d789ce1f86cf1d2064fb8e80a242c80273273f57c25783bdd4a1`;
- identity-seed SHA-256
  `17afb5fc1e9c3e56b7ffd2612f501d1714eefbb3c1e81d27e9501e0a18120c69`;
- expected blob SHA-256
  `69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65`.

## Workload And Arms

Run two no-retry, non-overlapping pairs in `AB BA` order. Each child receives a
unique private store containing only the identical identity seed and no blob,
temporary file, lock file, symlink, or foreign entry.

- A explicitly sets `QWEN_CHECKPOINT_STAGED_VALIDATION=decode`.
- B explicitly sets
  `QWEN_CHECKPOINT_STAGED_VALIDATION=encoder-digest`.

Both run the same release `qwen` binary and exact Qwen3.6 27B Q4_K_M messages
request: thinking preserved, one sampled token, temperature `0.7`, seed `42`,
prefill chunk `1024`, context capacity `6516`, and a 768 MiB aggregate/per-entry
durable budget. Model pages are conditioned before every scored child; the
conditioning work is outside child timing.

## Per-Child Contract

Require:

- clean exit, valid host/VM evidence, zero swapout/swap-occupancy growth, zero
  major faults and block input, and complete `/usr/bin/time -l` resources;
- exactly one store-empty line, one stats line, and one publication line;
- `publish=published`, `capture=completed`, `6500/6499`, pending true,
  `stop_reason=token_limit`, `blob_bytes=582854188`, `evicted=0`, and
  `identity=hit`;
- exact staged-validation mode for the arm and finite nonnegative isolated
  staged-validation wall;
- `prompt_tokens=6499`, one generated token, zero target transitions, and clean
  stdout SHA-256
  `9ebc01769b176bb074a065ea0974c130fc8afd12814360aaf809046160b2a999`;
- one exact final key, mode `0600`, link count one, no temporary residue, exact
  size, and the frozen blob SHA-256.

Compare each A/B pair by streaming every blob byte before removing disposable
work roots. All four blob hashes and stdout hashes must agree globally.

## Decision

For each pair compute A minus B isolated-validation, publication, process-wall,
and peak-physical-footprint savings. A valid candidate requires B to win all
three wall endpoints in both pairs and never increase peak footprint. GO
requires the same product endpoint to clear both counterbalanced pairs:

- process-wall saving `>=250.0 ms` in both pairs; or
- peak-footprint saving `>=500,000,000` bytes in both pairs.

Report AB, BA, and medians separately. Two pairs authorize one bounded mechanism
decision, not a population distribution or noise-scale claim. Isolated
staged-validation and publication walls, RSS, CPU, faults, physical I/O, and
pressure are diagnostics; they cannot rescue a product endpoint that misses in
either pair.

Only after a passing performance decision, run one unscored process-cold restore
from the retained B blob. Require one compatible candidate, exact `6500/6499`
restore, zero corruption removal, unchanged blob bytes, successful continuation,
and no new publication.

GO authority is force-only exact allocation-free staged validation. No authority
extends to making it default, persisted-file validation changes, restore speed,
write-behind, weaker durability, energy, concurrent publication, other models,
or checkpoint compression. Any identity, fixture, command, marker, output,
resource, blob, or restore defect stops without authority; valid host pressure
is inconclusive and never becomes retry authority.

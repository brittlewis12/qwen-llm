# v0.596 A3B Retained Loaded-Recovery Attribution

Status: p2 preregistered before its timed work. P1 at source `8cdf44e` stopped
after three rejected `ABC` blocks and remains noncanonical. This tracked p2
contract and runner are part of the new frozen source commit.

## Intent

Adjudicate the v0.595 warning with no engine changes. Determine whether A3B's
retained-storage transition tax persists when the same frozen request is
repeatedly exercised in one loaded process. Deterministic execution implies the
same route, but this packet observes only each process's last repetition output
and cannot isolate route-page warming from repeated prefill, mapping, driver,
allocator, translation, or other process-global convergence.

This is a contributive loaded-model mechanism packet. It has no independent
cold-latency, policy, route-specific, persistent-server, or CPU-prefault
authority.

## Frozen Arms And Sequence

- A: copied storage.
- B: retained storage with CPU prefault disabled.
- C: retained storage with full CPU prefault through the same Metal buffer.
- Process blocks are the three-order Latin square `ABC`, `BCA`, `CAB`.
- This cyclic square balances arm position, not predecessor carryover. Fresh
  processes, complete-file conditioning, and cooldown bound but do not eliminate
  that residual.
- Every process runs one untimed packed-prefill plus one-transition warmup,
  followed by five timed fresh-session repetitions.
- Every timed repetition uses the exact A3B model, frozen Reva text, chunk 1024,
  context capacity 1024, 127 decode calls, and full-logits decode. This matches
  the context and 127 target transitions made by the v0.595 128-output CLI
  request.
- Repetition 1 measures the first full timed request after prompt plus one
  transition of untimed warmup. Repetition 2 is convergence evidence. The
  median of repetitions 3-5 is the independent process's late result.
- Complete-file cache conditioning and at least 30 seconds of cooldown precede
  every process. Processes never overlap.

All arms use the production auto-promoted Q8 embedding policy. A has no
retained-backing line. B must report one generic retained window and zero
prefault work. C must report the same window and exactly 1,350,314 prefault pages
covering 22,123,544,576 bytes; its variable prefault wall is recorded, and its
checksum must agree across C processes. All arms must reproduce their exact
source/materialization ledgers.

## Host, Identity, And Correctness

Hash the runner, this contract, imported helper scripts, model, prompt, the
executed `qwen-bench` binary, and the v0.595 manifest/decision. Require clean
matching source/build identity for `qwen-bench` and the frozen v0.595 source and
decision fingerprints. Normalize all QWEN/Metal environment controls before
assigning the arm variables.

Also authenticate p1's manifest, all nine rejected process artifacts, three
block decisions, and the four raw harness-floor controls through its pinned
25-file SHA-256 inventory. P1 observed exactly 92 process-level `ru_majflt`
events in every A/B/C process. Two `qwen-bench build-info` and two
`qwen-bench metal-counters` controls observed exactly 89 each with zero block
input. This demonstrates an executable/runtime floor but cannot localize the
three additional decode-process faults.

Before every cache read require AC power, no thermal/performance warning, and at
least 50% memory availability. Bracket cache conditioning and child execution
with pageout/swap state. Cache-interval growth terminates before child launch;
post-run host, pageout, swap, or block-input invalidation may rerun the complete
three-arm block up to three times. Parser, identity, ledger, command,
environment, token-count, or output mismatch terminates the packet.

P2's sole validity change is to record process-level page faults without using
them to accept, reject, retry, warn, or classify a row. Do not subtract the
89-fault floor, require 92, or require equality across arms. Report every raw
count plus per-block B-minus-A and C-minus-A differences. The decision endpoint
starts after model load and an untimed full-prompt prefill plus one-transition
warmup, while the process counter is unlocalized and may include the mapping
behavior under study. Zero block input, zero pageout/swap growth, complete-file
conditioning, and the cache-conditioned/non-storage-cold authority remain
unchanged.

P1 rows are descriptive failure evidence only. Do not pool them with p2, reduce
p2's three-block design, or alter any arm, threshold, request, repetition,
ordering, retry, correctness, or authority rule because of their outcome.

Parse all five per-repetition prefill and decode walls, the last-repetition MoE
GPU/CPU consistency profile, generated text, and C's prefault report. Require
419 prompt tokens, five repetitions, context capacity 1024, 127 decode calls,
full-logits mode, packed prefill chunk 1024, and identical final generated output
across all nine accepted processes. Require all three C prefault checksums to
agree. Prior v0.594 full-state evidence remains the correctness authority; this
packet does not claim intermediate-repetition token traces.

## Decision

For each Latin block calculate:

- late `B/A` and `C/A` throughput ratios from process medians of repetition 3-5
  decode walls;
- repetition-1 `B/A` and `C/A` ratios;
- each arm's repetition-1-to-late recovery and B/C recovery normalized by A;
- late prefill ratios;
- repetition-5/repetition-3 throughput and the repetition 3-5 throughput range
  divided by its median for A and B;
- repetition-1-to-late prefill recovery for B and C, normalized by A.

The packet must reproduce a material B deficit: median repetition-1 `B/A` must
be at most `0.93`. Otherwise it is mechanism-inconclusive even if later rows are
fast.

Classify **same-request loaded recovery** only if all of these hold:

- every block has late `B/A >= 0.95` and the median is at least `0.97`;
- every B process improves at least `1.03x` from repetition 1 to late, and the
  median B improvement is at least `1.05x`;
- every A process's repetition-1-to-late movement is inside `[0.97, 1.03]`;
- every A and B repetition-5/repetition-3 ratio is inside `[0.98, 1.02]`, and
  every repetition 3-5 throughput range is at most 3% of its median.

Classify **persistent retained tax** only if every block has late `B/A < 0.95`
and the median is at most `0.93`, A movement is bounded, and all late A/B rows
meet the same stability rules. Any boundary crossing, unstable row, baseline
movement, or median inside `(0.93, 0.97)` is inconclusive.

Prefill convergence is diagnostic, not a decode gate. Normalized median B
prefill recovery of at least `1.05x` is material evidence that convergence is
broader than generated-route access. No result receives route-specific
authority because every timed decode follows another full prompt prefill.

C is interpretive only:

- C recovery with persistent B implicates host mapping/PTE/residency state;
- persistent and similar B/C rules out the existing CPU-prefault operation as a
  loaded remedy, not GPU translation effects generally;
- C prefault wall is outside loaded repetitions and cannot gain product authority.

Report the last-repetition MoE total, GPU-kernel, CPU-encode, commit/wait, route,
and command-buffer fields only as consistency diagnostics. They are not
five-repetition phase aggregates.

## Authority

Same-request loaded recovery can resolve the causal interpretation of v0.595's
warning only for its already measured fingerprint, warm-cache, caller-owned
disposable-process lane through 128 outputs. It does not erase v0.595's measured
first-request regression. It may release the A10B full-state gate, not A10B
timing.

Persistent tax leaves v0.595's first-byte, exit, correctness, and footprint
measurements intact but classifies retained A3B as a startup/private-footprint
specialization with an explicit steady-inference cost. It does not itself
promote policy. Neither result authorizes arbitrary prompts/routes, longer
outputs, persistent/server use, default-on selection, or C-prefault operation.

# v0.634 Dense-27B Terminal Ramp-Controlled Loaded Stability

Status: preregistration. No v0.634 model, correctness, or timing result exists.
This is the final packet for the exact dense-27B direct-pread loaded-parity
criterion.

## Intent And Authority

Adjudicate whether explicit direct-pread population preserves loaded prefill,
decode, and complete-request performance after a fixed startup reversal block.
v0.633 completed the prior short-period packet but sealed
`inconclusive-instability`: all/ABBA/BAAB prefill B/A was
`1.013090/1.013479/1.012442`, while per-arm prefill ranges were
`9.419%/10.815%`. Quartets 3-8 had a `1.01538x` prefill median, so excluding a
fixed startup block prospectively does not favor the candidate.

This packet is a new terminal portfolio decision authorized by committed
roadmap commit `e307f273ee8d1c2cc6c8d67985e88e32876e0654`. v0.633 has
`successor_authorization=none` and contributes no observation, variance,
correctness gate, token trace, or score. Its complete packet is authenticated
for forensic continuity only.

Only a complete GO may authorize explicit
`QWEN_GGUF_PARALLEL_COPY=pread` for the exact authenticated dense Qwen3.6-27B
Q4_K_M ForceOnly profile. It cannot authorize Auto/default selection, another
model or quantization, MTP, serving, concurrency, storage-cold effect size,
energy, or a general no-copy/pread policy. Every sealed non-GO ends this exact
profile as unable to certify the frozen loaded-prefill criterion. There is no
repair, retry, selector packet, or second ramp design.

## Source Boundary

The direct parent is certification commit
`e307f273ee8d1c2cc6c8d67985e88e32876e0654`. The v0.634 commit is its clean,
single-parent child and adds exactly this document and
`scripts/profile/v0634_dense27b_pread_ramp_controlled_stability.py`. It changes
no Rust, Metal, production behavior, model asset, prompt, benchmark command, or
scored parser.

The runner authenticates the complete one-parent chain:

```text
base  cc47190654b898e2004117b6d03be1006a89d1fa
R630  d30a0f49e14aec13126d83a41553f6e7250b0f93
H630  d90aecdd4d4f9c297f98b09ba55eb479a7bf7944
R631  ecc9e70fc833ee40831b7f5891141e620a92a2cd
R632  2e71e1a8caba14ddc0fcc5ea3232354dbab86504
R633  e237917ddeaec5635fb599fd19ae1ac58e3d0c51
CERT  e307f273ee8d1c2cc6c8d67985e88e32876e0654
R634  HEAD at execution
```

R634 must contain exactly the two additions above. CERT must modify exactly
`docs/PERF-LOG.md` and `docs/PERF-ROADMAP.md`. The runner also rechecks every
historical repair diff, the frozen `bench.rs` and R630-to-H630 patch digests,
clean source state, sibling dependency state, release build/runtime identity,
macOS `15.6.1` build `24G90`, M4 Max device identity, model inode/size/mtime and
complete SHA-256, prompt identity, safe environments, and tool dialects.

Rebuild `target/release/qwen-bench` at R634 before preflight. No benchmark may
run from a binary stamped with an earlier commit.

## Complete v0.633 Forensic Bridge

Authenticate these exact files and their completion binding:

```text
decision.json             3cf6427e32056d30356176e03e18e1b37c3205aa1cf591e60c54cac04be34d8a
artifact-inventory.sha256 1faac9baebd9f453ea1dfb23149502cdc34b25697ffd6cfd4e439cf4291509f5
packet-complete.json      1892674cd1e0c5df303ecd60c24beb2f5e76590b1c55334043b010d003d186ec
```

Require exactly 137 inventory members and 139 final regular files, with every
member rehashed. Require source R633, `status=inconclusive`, sole reason
`inconclusive-instability`, `authority=none`, no force or successor authority,
32 valid children in the exact eight-quartet order, 65 launch-ledger events,
fresh correctness passage, exact shared 32-ID trace, symmetric 92 process-wide
faults, zero block input/swaps, zero B-marker major faults, failed trajectory
gates, and successful final identity.

Record only:

```text
authority_imported=false
correctness_imported_as_gate=false
performance_observations_imported=0
timed_children_imported=0
observed_valid_children=32
forensic_class=inconclusive-instability
```

The v0.631-v0.632 bridges remain transitively authenticated by the frozen repair
chain. No cx output has authority.

## Fresh Correctness And Token Protocol

Before timing, rerun the inherited CPU-only token-protocol check and exact
full-state test:

```text
target/release/qwen-bench decode --help

cargo test --release -p qwen-llm \
  metal_forward::tests::gguf_parallel_pread_dense27b_q4_is_bit_exact \
  -- --ignored --exact --nocapture --test-threads=1
```

The help output must contain the exact generated-token-trace contract. The Rust
test must freshly preserve all 851 resources and 16,806,250,496 logical bytes,
independent exact-sized offset-zero Shared/DefaultCache/Tracked storage, the
frozen four-worker schedule, checked write rejection, packed-prefill and
continuation logits, complete KV/GDN/convolution state, argmax, transition, and
the five-line copied/pread load order. Correctness-run marker faults remain
exact advisory evidence under v0.632; they are not imported from v0.633.

## Frozen Child And Arm Contract

Every child is the inherited independent serial subprocess over the exact
1,891-byte, 419-token prompt:

```text
/usr/bin/time -l target/release/qwen-bench decode
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf
  --prompt <exact prompt bytes>
  --tokens 31
  --runs 1
  --prefill-chunk 1024
  --kv-capacity 1024
  --full-logits-decode
  --generated-token-trace
```

The default untimed warmup remains enabled. P is the timed fresh packed prefill,
D is 31 full-logits transitions, and R is fresh-session allocation plus P and D.
Model load and warmup are outside P/D/R. The trace is the initial prefill argmax
plus every timed transition result: exactly 32 canonical IDs.

Arms differ only in the normalized environment:

- A: `QWEN_GGUF_PARALLEL_COPY=0`;
- B: `QWEN_GGUF_PARALLEL_COPY=pread`.

A must emit the native-embedding policy and copied ledger with no GGUF
population marker. B must emit the same policy, one exact dense schema-2 pread
marker, and the copied ledger. The B marker's phase-local `timer_major_faults`
remains hard zero. Immediately before every child, read the complete unchanged
target and prove all 1,026,444 pages resident. Require zero block input, zero
swaps, valid host/pressure state, durable output, and the inherited pressure
semantics. Whole-process `/usr/bin/time` faults remain exact symmetric advisory
integers.

## Fixed Execution Schedule

Construct the complete 40-child schedule before execution. Run it once without
pause, inspection, cooldown, retry, replacement, or conditional continuation:

```text
unscored ramp: ABBA BAAB
scored body:   ABBA BAAB ABBA BAAB ABBA BAAB ABBA BAAB
```

Ramp stems are `ramp-q01-*` through `ramp-q02-*`. Scored stems remain
`loaded-q01-*` through `loaded-q08-*`, preserving the exact v0.633 scored
population. Every attempt and launch/completion event records population,
`used_for_scoring`, global execution index, phase-local quartet index and order,
position, arm, and stem.

The first ramp A child creates the packet-wide golden trace after its completion
is durably recorded. Every later ramp and scored child must match it exactly.
Token identity, parser, load, residency, host, pressure, lifecycle, and artifact
checks occur after each child because they are validity contracts. No ramp P/D/R
ratio, median, trend, range, CPU ratio, memory ratio, simple-win count, or
performance gate is computed before all 40 children complete.

Only a hard correctness, validity, lifecycle, durability, or interruption
failure may stop the fixed schedule. Performance values never control
continuation.

Immediately before artifact reservation, one packet-wide deferred-SIGINT
handler covers reservation, manifest construction, token protocol, correctness,
conditioning, child parsing, analysis, and final identity. Token-protocol and
correctness children finish and are reaped before the next safe-boundary check.
Each timed child retains the inherited child-local deferred handler and
exact-PID reaping. A signal during conditioning is attributed to the active
child even when no launch or attempt row exists; a signal after a valid child is
checked only after its attempt and any required golden event are durable.

## Ramp Validation And Exclusion

All eight ramp children must be individually valid and must satisfy the
inherited temporal limits: conditioning-to-start at most 2 seconds, consecutive
starts at most 12 seconds, each quartet at most 45 seconds, and the full
ABBA/BAAB reversal at most 90 seconds. The first scored start must be within 12
seconds of the final ramp start, and no children may overlap.

Ramp validation contains only schedule identity, token identity, temporal
values, and temporal booleans. It must report:

```text
used_for_scoring=false
used_for_stability=false
used_for_medians=false
ramp_observations_used_for_scoring=0
ramp_observations_used_for_stability=0
ramp_observations_used_for_medians=0
```

The inherited scorer receives exactly the 32 scored rows, never a concatenated
40-row population. The 5% per-arm trajectory gate, all quartet ratios, aggregate
and ABBA/BAAB medians, 7/8 consistent-miss rule, timing diagnostics, and simple
wins therefore use only the scored body. A failed ramp temporal gate makes the
complete packet inconclusive without scoring the body.

## Scored Gates

Preserve v0.630-v0.633 scoring unchanged. For each scored quartet and metric,
form geometric arm centers and B/A ratios. GO requires:

- all/ABBA/BAAB medians `<=1.01` independently for P, D, and R;
- every quartet P/D/R `<=1.01` and complete CPU `<=1.10`;
- every quartet maximum RSS and physical-footprint ratio `<=1.05`;
- each arm's 16 scored P, D, and R observations have
  `(max-min)/median <=0.05`;
- every temporal, identity, correctness, pressure, and artifact gate passes.

A stable KILL requires at least seven of eight scored quartets over one declared
limit and all/ABBA/BAAB medians over that same limit. A complete stable mixed
result is inconclusive. Instability takes precedence over performance
classification. This remains a narrow engineering admission, not a confidence-
interval or population-level 1% proof.

## Terminal Decision And Durability

Decision precedence remains source/contract defect, invalid execution,
instability, stable GO/KILL/heterogeneous classification. Only GO sets
`authority=explicit-force-only-dense27b-direct-pread` and
`force_authorized=true`. `successor_authorization` is always `none`; even GO
does not authorize an Auto/default selector packet.

Every sealed non-GO records
`profile_disposition=closed-unable-to-certify-frozen-loaded-noninferiority`.
GO records
`profile_disposition=admitted-explicit-force-only-dense27b-direct-pread`.
A validity or harness stop closes the portfolio without converting it into a
mechanism KILL. A raw durability failure or inability to reap the exact PID is
unsealable and grants no authority; this terminal program still receives no
replacement packet.

Immediately before publication, block SIGINT and inspect both the deferred list
and pending signal set. Any signal before that boundary rewrites a non-defect
decision to sealed inconclusive. Publication, inventory, completion, and result
printing then run inside the blocked critical section. A signal arriving after
that decision boundary is delivered only after the fsynced completion exists
and does not retroactively invalidate a complete packet.

The runner durably binds every launch, completion, conditioning record, raw
stdout/stderr, post-exit state, attempt row, token golden, correctness output,
manifest, decision, final identity, and complete SHA-256 inventory. A complete
40-child packet has 169 inventoried members and 171 final regular files:

```text
9 packet-level members + 40 * 4 child members = 169
169 + inventory + completion = 171
```

The runner's CPU-only `--self-test` must pass before commit and proves the
frozen 8/32/40 schedule plus invariance of scored analysis under extreme changes
to all ramp performance values. After commit and release rebuild,
`--preflight-only` must report 8 ramp children, 32 scored children, and 40 total
without reserving the artifact directory. Execution follows only after
adversarial review of the committed packet.

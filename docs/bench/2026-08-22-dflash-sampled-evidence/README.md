# DFlash Sampled Evidence Preregistration - 2026-08-22

Status: E0/E1a/K0 evidence instrumentation is authorized in the dedicated
`dflash-sampled-evidence` worktree. No result, sampled product implementation,
serve admission, or product authority exists yet.

## Objective And Authority

Determine, separately for Qwen3.8-27B Q4_K_M and Q8_0 targets, whether exact
positive-temperature DFlash has a credible correctness and economic path. The
program has three independent evidence lanes:

- **E0 - serial-p foundations:** validate the exact target path, sampler-v1
  prepared distribution, RNG accounting, hidden capture, and state continuity.
- **E1a - online one-hot oracle:** run a deliberately slow exact generation
  oracle over live evolving contexts and compare it with a fresh plain-serial
  generation. E1a depends on E0, not K0.
- **K0 - causal sparse-q evidence:** recover and cross-check the actual DFlash2
  selector distribution over its selector-adjusted top-16 support. K0 is
  independent of E1a.

Primary `p` is always the distribution prepared by serial sampler-v1 from a
token-major target forward. Packed-verifier target rows are diagnostics only.
They are a separate, potentially different-distribution lane and cannot supply
primary p, exactness authority, or fallback-free economics.

Passing this program yields only `G1-EVIDENCE`. It may authorize one separately
bounded row-stable primitive spike under the rules below. Product sampling,
serve, generalization, tree speculation, and default admission remain deferred.

## Allowed Source Work

Source edits are allowed only for bench/test-only oracles and read-only
diagnostics needed to observe existing behavior. They may:

- add ignored real-model tests or a non-product bench harness;
- compare `single_token_with_multi_hidden` with `single_token` in independent
  sessions (`crates/qwen-llm/src/metal_forward.rs:7767` and
  `crates/qwen-llm/src/metal_forward.rs:10246`);
- expose an immutable view/copy of sampler-v1's already-prepared ordered support,
  weights/probabilities, RNG pre/post state, raw draw, and selected candidate;
- expose synchronized DFlash2 selector IDs, unary scores, adjusted scores,
  issues, predecessor choice, and production choice. The existing diagnostic
  types define the useful surface at `crates/qwen-llm/src/metal_dflash.rs:2173`.

The instrumentation must not alter the observed path, selection, synchronization,
allocation lifetime, or RNG consumption. No instrumentation result may be used
by production code.

Explicitly forbidden in this program:

- edits to main generation, serve, admission, or request policy;
- changes to sampler-v1 version, filtering order, arithmetic, tie behavior, RNG,
  errors, or draws (`crates/qwen-llm/src/sampling.rs:13` and
  `crates/qwen-llm/src/sampling.rs:196`);
- changes to packed verifier kernels, target logits, greedy selection, rollback,
  or near-tie behavior;
- sparse-q product sampling, maximal-coupling product code, GPU sampling, or a
  hidden product switch;
- model/tokenizer conversion, retraining, quant changes, or production traffic.

The current serve gate remains greedy-only (`crates/qwen-cli/src/serve/backend.rs:149`).

## Frozen Scope

### Assets And Hardware

Measure these target rows separately; do not pool or substitute them:

1. `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf`
2. `/Users/tito/models/Qwen3.8-27B-Q8_0.gguf`

Both use only:

`/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf`

The drafter has block size 8, selector top-k 16, and five SWA-2048 layers. The
real-model paths are also used by local correctness gates at
`crates/qwen-llm/src/metal_dflash.rs:26384`; the window contract is documented at
`crates/qwen-llm/src/metal_dflash.rs:82`. Each packet records file bytes/SHA-256,
GGUF metadata and target-drafter binding, tokenizer/special IDs, source commit,
binary hash, and host/OS identity. Scope is the current Apple M4 Max 128 GB host.

### Target Sampling Configurations

Both configurations use local sampler-v1 exactly as implemented. They apply to
target p and E0/E1a serial sampling, not to sparse q construction.

| ID | Mode | temperature | top_k | top_p | min_p |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | local qwen-chat | 0.7 | 200 | 1.0 | 0.05 |
| S2 | official Qwen3.8 thinking | 1.0 | 20 | 0.95 | 0.0 |

No grammar, logit bias, repetition/frequency penalty, prompt lookup, or other
processor is in scope. In particular, this packet makes no claim that official
non-thinking uses or should use a presence penalty.

## Evidence Lanes

### E0 - Serial p And Hidden-Capture Foundations

For each frozen request/config/seed, create two fresh equal-capacity target
sessions from the same prompt tokens:

- A advances only through `single_token`;
- B advances only through `single_token_with_multi_hidden`, requesting exactly
  the DFlash target layers.

At every generated position, feed the same committed token to both paths. Before
sampling, record and compare full target logits and the immutable prepared
sampler-v1 support/weights. Clone or independently reconstruct equal samplers so
both consume the same configured RNG stream; diagnostics may observe but not
advance it. Record selector/capture data from B without allowing it to select the
target token.

E0 passes a run only if A and B have, at every step:

- bit-identical target logits and complete prepared distributions;
- identical sampled token, candidate index, raw draw, draw count, RNG pre/post
  state, stop behavior, and emitted stream;
- exact active KV, GDN state, GDN conv state, logical position, and captured
  target-hidden identity at the declared boundary;
- a bit-identical next-token continuation after the measured generation, tested
  by one additional common token without counting it in economics.

Any difference is a failed E0 row. Argmax agreement, output-text agreement,
cosine, or approximate state checks are insufficient.

### E1a - Online Exact One-Hot Oracle

E1a is online and intentionally slow. It does not use packed p and does not wait
for K0. At each actual evolving context:

1. DFlash proposes its current greedy/one-hot token `y` from a live drafter
   block.
2. A live serial token-major target forward prepares p with sampler-v1.
3. The request-local sampler-v1 draws `x ~ p` exactly once.
4. Treat q as `delta_y`: accept and emit `y` when `x == y`; otherwise emit the
   correction `x`. This is the exact one-hot maximal coupling, implemented as an
   oracle decision rather than product rejection sampling.
5. Advance the target and drafter/capture state with the emitted token and
   continue from that new context, including rejection, EOS, block boundary,
   and continuation cases.

Run a fresh plain-serial reference in an independent target session with the
same prompt, config, and target-sampler seed. It must draw directly from live
sampler-v1 and must not share logits, sampler objects, KV/GDN state, capture, or
mutable storage with the oracle. With the pinned coupling, the complete token
stream, sampler draws/state, target logits/prepared distributions, active
KV/GDN/conv state, stop reason, and one-token continuation must be exact between
oracle and reference.

One-hot q and sparse q are distinct mechanisms. E1a can pass while K0 fails and
cannot authorize sparse sampling. E1b, a sparse-q online maximal-coupling oracle,
is deferred until K0 passes under a new or appended preregistered phase.

### K0 - Actual DFlash2 Causal Sparse q

K0 freezes q from the selector-adjusted top-16 scores actually used at each
causal predecessor. The production T=0 selector is a lattice walk over
`U(b) + dot(A(prev) * h_pos, B(b))`, not a full-vocabulary draft-logit sampler;
see `crates/qwen-llm/src/metal_dflash.rs:14712`. Therefore:

- q support is the valid, de-duplicated selector candidate IDs at that depth,
  at most 16 tokens; every omitted vocabulary token has q exactly zero;
- q weights are normalized only from the recorded final selector-adjusted
  scores under the pinned q temperature/policy;
- target sampler-v1 top-k, min-p, top-p, and S1/S2 temperature are never applied
  to q;
- predecessor token/choice, candidate rank, token ID, unary-score bits,
  adjustment/final-score bits, sentinel/duplicate/nonfinite issues, normalized
  weight bits, proposal draw, and selected token are part of q identity;
- q must be recomputed causally after each selected predecessor, not as
  independent per-position rows.

Before held-out K0, pin the upstream/reference repository revision, exact
selector-sampling contract, score precision, tie/order rules, q temperature,
normalization, and RNG conversion in the packet manifest. These are evidence
inputs, not a proposed local product policy, and may not be tuned from held-out
acceptance. The local trace must cross-check the pinned upstream/reference trace
on the development sentinel, then on held-out rows. Every normalized q must be
finite, nonnegative, sum to one under the pinned tolerance, reproduce from the
recorded score bits, and select the recorded proposal under replayed RNG.

K0 fails on a candidate/support/predecessor/score-bit/issue mismatch, duplicate
ambiguity, unexplained nonfinite value, normalization mismatch, RNG mismatch, or
upstream trace mismatch. It is forbidden to reinterpret q as full-vocabulary
logits, apply target filters, add epsilon mass, or repair/drop an observed row.

## Fixtures, Sessions, And Decision Discipline

Development comes first. It may use one or more sentinels to debug schema and
instrumentation. The existing code fixture is allowed for the initial smoke, but
no smoke result can promote, select a policy, or enter a confidence interval.

Before held-out acquisition, freeze prompt bytes/token hashes, request length,
four seeds, config, target row, q policy (for K0), commands, reducer, and fixture
roles for exactly these held-out requests:

1. **favorable-short:** short code/structured completion expected to sustain
   useful DFlash acceptance;
2. **canonical-real-long:** the maintained real long-context archetype, with its
   canonical prompt and request length unchanged;
3. **adversarial-low-acceptance-prose:** prose selected to exercise rejection and
   backoff economics, not to pass the favorable gate;
4. **narrative-guardrail:** representative mixed narrative behavior used to
   prevent a narrow code-only promotion.

Use frozen seeds `42`, `314159`, `271828`, and `1618033` unless a committed
fixture already has a canonical seed, in which case record that seed plus the
first three values above not equal to it. A request/seed is the clustering unit.
Tokens, proposal depths, blocks, and retries within it are correlated repeated
measurements, not independent samples. Report every cluster and aggregate only
after reducing within cluster. Select thresholds, q policy, and any row-stable
primitive shape on development only; held-out data cannot select or repair them.

Every comparison uses independent fresh sessions. Reference and candidate may
be coupled by a preregistered seed but may not share mutable model state, sampler
state, logits, capture, or snapshots. A failed candidate row is retained. A
clearly pre-observation infrastructure failure may be marked invalid with its
failure record; it is not silently replaced.

## Trace Contract

Each run writes append-only strict JSONL with these realistic record types:

- `run_start`: schema/arm, command, build and binary identity, lease setting,
  target/drafter/tokenizer hashes, config, fixture/token hash, seed, q-reference
  identity if applicable, session IDs, and requested tokens;
- `sample_decision`: position, committed-prefix hash, complete prepared target
  support/weight/probability bits, sampler/RNG pre-state, raw RNG bits, unit-draw
  bits, candidate index/token, draw count/post-state, and E1a one-hot decision;
- `proposal_block`: block/depth, carry and predecessor identity, top-16 IDs,
  unary/final score bits, selector issues and choices, normalized sparse-q
  support/weight bits and RNG bits when K0 samples q, target/drafter/capture
  state hashes, snapshot/rollback hash, and per-phase timings;
- `serial_reference_end`: stream hash, stop reason, draws, final active
  KV/GDN/conv and logical-position hashes, continuation token/logit/distribution
  hashes, and generation timings;
- `run_end`: status, row counts, stream/state/continuation comparison, timing
  totals, issue counts, stdout/stderr digests, and trace SHA-256.

Large complete p records may use immutable binary sidecars referenced by dtype,
shape, offset, bytes, and SHA-256. Sparse q must remain directly auditable from
its IDs and exact score/weight bits. JSON rejects duplicate keys and nonfinite
constants. State hashing must define active ranges and exclude unused capacity.

The first development harness need not implement a general environment census,
process supervisor, or packet publisher. Later authoritative packets may wrap
the same frozen trace schema with manifests, host checks, exclusive creation,
timeouts, and artifact authentication. Such wrappers may not change model
behavior or reinterpret prior traces.

## Metrics And G1-EVIDENCE

Correctness gates are absolute:

- **E0:** every measured step passes stream, draw, logit, prepared-distribution,
  state, hidden-capture, stop, and continuation parity.
- **E1a:** every oracle/reference request passes exact stream, sampler/RNG,
  target-logit/distribution, final state, stop, and continuation parity.
- **K0:** every sparse q is causally reproducible, normalized under the pinned
  contract, RNG-replayable, and cross-checked against the pinned upstream trace.

Economics are projected from measured request-local counts and timings. For each
request/seed cluster, charge draft generation, hidden capture, q preparation and
sampling (when sparse), exact token-major verifier p, correction, rollback/state
work, synchronization/readback, and continuation. Do not substitute current
packed p for exact p. Report projected generation speedup against its fresh
plain-serial reference and preserve the complete cost equation in the reducer.

`G1-EVIDENCE` requires all correctness gates for the lane being promoted and:

- on both named promotion archetypes, **favorable-short** and
  **canonical-real-long**, a one-sided request-cluster 95% lower confidence bound
  of at least `1.10x` projected generation speedup;
- median projected generation speedup of at least `1.15x`;
- no important held-out cluster, including adversarial prose and narrative
  guardrail, below `0.98x` unless a development-frozen abstention policy excludes
  it before charging speculative work;
- enough charged allowance for an exact verifier; a projection that wins only by
  pricing the current packed different-distribution p is a failure;
- separate passing decisions for one-hot and sparse q. One-hot evidence cannot
  promote sparse q, and sparse q must additionally pay its incremental selector
  normalization, RNG, coupling, trace-independent product complexity, and cost.

Use a one-sided 95% cluster bootstrap or a more conservative predeclared
request-cluster interval; resample request/seed clusters, never tokens. Report
all clusters, point median, lower bound, and sensitivity to adverse timing
bounds. The historical emitted-per-packet values `Q ~= 3.35` for Q4 and
`Q ~= 2.47` for Q8 are sanity floors only, not acceptance gates and not
substitutes for the full projected speedup. Repository discipline likewise
requires held-out policy evaluation and guards important rows near `0.98x`
(`docs/PERF-ROADMAP.md:2945`).

Packed-target diagnostics must be reported separately with an explicit
`different_distribution=true` label. They may estimate a future opportunity but
cannot satisfy E0/E1a, exact-verifier allowance, or G1.

## Post-G1 Primitive Authority

After G1, and only for a lane that independently passed, authorize one
development-selected row-stable primitive spike. It must retain:

- bit-identical full logits, or complete prepared sampler-v1 distributions with
  exact selected stream/draw behavior;
- exact active KV, GDN state, GDN conv state, rollback/snapshot boundary, and
  one-token continuation;
- no routine token-major fallback disguised as a fast path; fallback is an error
  or explicitly charged exceptional path;
- the G1 economics after charging the primitive and all sparse-q incremental
  work.

The spike does not authorize main/serve integration, a sampled decoder, broader
models/configs, production defaults, trees, or a second primitive. Any of those
requires a new preregistration and fresh held-out evidence.

## Metal Lease Policy

Every Metal-capable command must literally be invoked as:

```text
QWEN_METAL_LEASE_WAIT=1 <command>
```

This includes ignored GPU tests, benches, qwen/qwen-bench, and any harness that
can initialize Metal. Wait for the queue-managed lease, run one owner at a time,
and never stop, signal, reconfigure, bypass, spoof, delete, or otherwise disturb
the current owner. Lease wait is not model timing. A lease timeout is recorded
as infrastructure failure, never permission to run unleased. The local lease
precedent is `docs/bench/2026-08-20-windowed-dflash-pre-gates/README.md:85`.

## Artifacts And Failure Discipline

A development smoke may contain only:

```text
dev/manifest.json
dev/trace.jsonl
dev/stdout.txt
dev/stderr.txt
dev/result.json
```

An authoritative held-out packet adds, as needed:

```text
preregistration.sha256
manifest.json
fixtures.json
commands.jsonl
e0.jsonl
e1a.jsonl
k0.jsonl
sidecars/*
reduction.json
decision.json
failure.json
RESULT.md
```

Reserve a new packet root before acquisition. Within it, evidence is append-only:
create or append, fsync, hash, and never rewrite, truncate, delete, repair, or
reuse an artifact name. Partial traces and unfavorable rows remain evidence.
Publish one terminal `decision.json`; on failure, `failure.json` is byte-identical
or a hard-link alias. Development and authoritative roots are distinct, and
development observations never enter held-out reduction.

Failure of E0 kills exact one-hot work until a new preregistered repair premise;
it does not decide K0. K0 failure defers E1b/sparse q but does not invalidate a
completed E1a. Gate or economic failure grants no implementation authority.
Reopening requires a new append-only preregistration naming the failed premise,
the bounded change, and new frozen evidence; results are never written into this
document.

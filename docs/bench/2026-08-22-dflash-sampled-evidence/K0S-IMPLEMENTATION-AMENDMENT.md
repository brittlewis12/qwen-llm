# DFlash2 K0-S Implementation Boundary Amendment - 2026-08-23

Status: prospective append-only amendment, frozen after source reconnaissance
and before any K0-S source edit, synthetic Metal execution, or acquisition. It
narrows one over-specified diagnostic requirement in `K0S-AUTHORIZATION.md`; all
other authority limits, source paths, feature gates, caps, tests, and acquisition
prohibitions remain controlling.

This amendment authorizes no real-model execution, K0-S result, proposal RNG,
acceptance/correction, K0-L, E1b, verifier, generation, product, serve, default,
or production behavior.

## Discovery And Problem Frame

Static repository reconnaissance found that the synchronized production tail
already exposes the exact local inputs needed to define and replay sparse q:

- full drafter logits and backend top-16 IDs/unary values;
- the projected selector-hidden f32 vector consumed by score construction;
- raw predecessor A and successor B codebook rows; and
- the production greedy predecessor chain.

It also found that the selector-hidden projection may route through an Apple
`simdgroup_multiply_accumulate` MMA8 kernel whose numerical operation order,
contraction, and denormal behavior are not exposed as a portable arithmetic
contract. The relevant unchanged identities at base commit `aeebf2c` are:

| Source | SHA-256 |
| --- | --- |
| `crates/qwen-llm/src/metal.rs` | `9343fe60597cc4695df7094cd1916d204d5dc624055221dc96ec6c77a2a65002` |
| `crates/qwen-llm/src/metal_forward.rs` | `5747420a0590f6ef36ea25ae8abad39d928df58bf4ec41b7e3212916baa550f4` |
| `kernels/mat_mat_mma8.metal` | `63308c1e8f7ea363898a960852cb603ff6ff563ddc81be92af23115468b191e5` |
| `crates/qwen-llm/build.rs` | `aabca4033bdf582bab368c17659c2d851f9b36609951976111dee37ce092a3a3` |

The Metal sources are compiled with `-O3 -ffast-math`. Source spelling plus an
observed output is therefore insufficient to establish the exact staging,
rounding, contraction, reduction-tree, and flush behavior required by the
original projection bound. Inferring or tuning that contract from a later model
observation would be circular. Expanding production instrumentation would still
not turn undocumented hardware arithmetic into an independent semantic oracle.

## Correctness Boundary

For marginal correctness, speculative decoding may use any prospectively defined
causal proposal kernel. Conditional on the accepted prefix, all proposal-side
state, and all latent randomness used to construct the current proposal, the
token actually drawn at each depth must have the exact conditional law q consumed
by acceptance/correction. Acceptance/correction must also use the exact processed
target distribution and their separately specified conditional randomness. A
different selector projection can change q, acceptance, and economics, but it
does not by itself bias the target marginal when those later obligations hold.

For K0-S, define the authenticated projected selector-hidden vector at depth t
as `z_t`. The local score and raw proposal are:

```text
score_t(a, b) = unary_t(b) + dot(A(a) * z_t, B(b))
q_t(b | a) = softmax(score_t(a, b) / request_temperature)
```

The local backend computes `z_t = H(h_t)`, but K0-S begins at the synchronized
`z_t` bits. It does not claim that another backend produces the same `z_t`, score
bits, or q. This preserves the upstream sparse-q formula conditional on the
backend projection while separating proposal-distribution correctness from
learned-model/backend-fidelity evidence.

Consequently, a K0-S pass can establish only:

```text
development local conditional selector-tail conformance, treating authenticated
synchronized z_t bits and their event/depth association as inputs
```

It cannot establish selector-projection parity, cross-backend score equivalence,
proposal sampling, joint-path RNG behavior, verifier replay, coupling,
target-processor state, rollback, continuation, acceptance quality, speed, or
product readiness. Any future claim about `H(h_t)` fidelity requires a separately
named, prospectively frozen gate; K0-S outcomes cannot tune or waive that gate.

A K0-S pass is not q provenance for coupling and may not be consumed by K0-L or
E1b unless a later frozen gate binds the sampled proposal event to the same
candidate, unary, `z_t`, predecessor, temperature, support, normalization,
drafter state, and proposal-construction records and establishes the actual
conditional token law.

## Superseded Requirements

This amendment supersedes only the mandatory projection-reconstruction portions
of `K0S-AUTHORIZATION.md`:

- pre-selector-projection H vectors are no longer mandatory trace or sidecar
  material;
- raw `selector_hidden` tensor bytes are no longer mandatory sidecar material;
- independent selector-hidden projection reconstruction, its exact-dyadic error
  bound, dispatch arithmetic allowlist, and projection pass/fail tests are not
  K0-S gates; and
- a producer or reducer must not emit a projection-parity pass from this packet.

The original authority label is also superseded. Every producer and reducer
output must use exactly:

```text
development_k0s_conditional_on_authenticated_z_only_no_projection_parity_no_rng_acceptance_k0l_e1b_verifier_product_authority
```

The former broader label is rejected. No packet field, result text, or downstream
summary may abbreviate away the conditional-on-`z_t` boundary.

The selector-hidden tensor's exact GGUF descriptor, full-tensor hash, source
asset identity, observed dispatch census row, Metal source hashes, embedded
metallib hash, build identity, and host/device identity remain mandatory
provenance. They are identity facts only. An optional projection-distance field
would create an unpreregistered outcome and is forbidden rather than merely
non-gating.

## Mandatory K0-S Packet

The separate `qwen.dflash_k0s_lattice` v1 packet and reducer must still provide
and independently check:

1. complete finite full-logit rows for depths 1..7 and exact local Metal top-16
   reconstruction under descending f32 value then ascending token ID;
2. backend candidate IDs, unary f32 bits, and synchronized projected
   selector-hidden f32 bits for every active depth, each bound to the exact
   production call, run, depth/position, carry, drafter checkpoint, and
   proposal-construction/noise-input identity before its backing buffer can be
   overwritten;
3. exact GGUF descriptors/full-tensor hashes for selector-hidden, predecessor A,
   and successor B, plus independently decoded raw bytes for every referenced A/B
   row;
4. all 97 positional predecessor rows by 16 successor slots, finite local f32
   score-bit replay, strict-first-max behavior, issue order, and the production
   greedy chain;
5. prospectively supplied deterministic non-greedy slot-index traversals without
   RNG;
6. request-temperature softmax, zero mass outside support, target-filter
   independence, and disabled proposal abstention; and
7. complete trace/sidecar/source/build/executable/asset/fixture/manifest identity
   and authority labeling.

Rows at depths 2..7 remain keyed by prior backend slot, not predecessor token ID.
Duplicate predecessor IDs therefore produce distinct positional rows. Every
normal packet still has exactly 97 rows; `missing_predecessor_row` is exercised
only by malformed/synthetic traversal tests and is a semantic chain diagnosis,
while the strict reducer rejects missing rows in a purported normal packet.

A deterministic fixed chain selecting an invalid nonzero sentinel terminates on
that row's `sentinel` issue and is non-passing; it does not invent a fourth chain
event. `slot_zero_termination` remains reserved for production's no-choice
index-zero fallback when slot zero cannot furnish a valid next predecessor.

## Observation And Non-Perturbation

The existing dispatch census in `crates/qwen-llm/src/metal.rs` records kernel
name, encoder ordinal, grid, and threadgroup geometry without an edit to that
file. Feature-gated K0-S code may activate and consume that existing observer
from the authorized `metal_dflash.rs` path.

Synthetic diagnostic-off/on parity must activate dispatch census and kernel
tracing identically in both arms, compare both execution orders using fresh
equal-capacity sessions, and require exact:

- draft outputs and synchronized selector inputs;
- all inspectable drafter cache/conv/watermark state and continuation;
- dispatch-census rows, encoder ordinals, and kernel-trace counts; and
- zero RNG use by construction.

The original demand for an independently observed command-buffer commit/wait
count is superseded because no such observer exists in the authorized surface.
Instead, source tests and review must prove that the diagnostic wrapper calls the
unchanged production draft exactly once and performs only CPU reads, copies,
dequantization, hashing, and serialization after its existing completion wait.
It may encode no diagnostic GPU work and may call no additional commit or wait.

The later acquisition plan must retain the equivalent real-model off/on gate and
both execution orders. It must classify every parity arm, partial artifact, and
failure prospectively; this amendment does not execute or define an acquisition
attempt.

## Implementation And Exit Rule

The source allowlist and non-default `dflash-k0s-diagnostics` feature remain
unchanged. The v1 exact record/key order, forbidden-key set, sidecar regions, and
reducer output schema may iterate during synthetic implementation testing, but
synthetic outputs may not select a tolerance, semantic rule, or evidentiary
claim. All such outputs are development-only. The final tooling must be
committed, adversarially reviewed, and rerun on its frozen self-tests before a
separate run plan hash-freezes it. Trace self-hash remains external; no
self-referential producer claim is required.

Tooling is eligible for review only when default builds exclude the capability,
feature-enabled Rust tests pass, the strict reducer self-test passes, synthetic
Metal tests use literal `QWEN_METAL_LEASE_WAIT=1`, and no real asset has been
opened. Tooling success grants no acquisition authority or K0-S result.

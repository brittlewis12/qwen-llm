# Prospective E0 Development Clarification - 2026-08-22

Status: append-only prospective clarification written before any E0 lockstep
acquisition. It does not alter, reinterpret, or add authority to prior E1a or K0
development observations. The original preregistration remains frozen.

## Hypothesis

For a fixed target, capacity, committed token stream, position, and independent
sampler-v1 state, replacing `single_token` with
`single_token_with_multi_hidden` at exactly the DFlash target layers is
bitwise non-interfering with target logits, the complete prepared distribution,
the RNG transition, active target state, stop behavior, and one-token
continuation. Copying the captured hidden row into DFlash target context must be
byte-exact and position-exact.

E0 does not test packed verification, rollback, sparse-q correctness,
acceptance, economics, or product integration.

## Development Arm Contract

- Arm A and arm B are fresh equal-capacity target sessions with no shared
  mutable session or sampler state.
- Both arms consume the entire prompt token by token. Arm A calls only
  `single_token`; arm B calls only `single_token_with_multi_hidden` and requests
  the drafter's target layer IDs in their recorded order.
- The execution order is an explicit run input (`serial_then_capture` or
  `capture_then_serial`). Reversing it is a development sentinel, not an
  independent statistical replicate.
- Every successful positive-temperature frontier uses two samplers constructed
  independently from the same frozen configuration and seed. Diagnostics run
  before the live draws and may not advance either sampler.
- A token is committed only when full prepared-distribution, RNG, diagnostic,
  and live-sample identity pass. A mismatch terminates the row without choosing
  one arm as authoritative.

## Exact Boundaries

- Prompt and generated transitions compare full logit bits after every consumed
  token, not only hashes, argmaxes, text, cosine, or a terminal checkpoint.
- After every consumed token, each target session is snapshotted with exactly
  the consumed prefix. Equality covers snapshot identity, prefix, every active
  KV byte and position, all GDN state bytes, and all GDN conv bytes. Unused KV
  capacity and transient scratch are excluded by the existing snapshot ABI.
- Arm B's hidden destination is poisoned before each forward. Hidden identity
  means exact bytes, shape, requested layer order, and absolute position from
  the completed capture buffer through the newly appended active DFlash target
  context row. DFlash context length and position stamps must advance by exactly
  one; projection/cache watermarks are recorded but are not advanced by E0.
- The emitted terminal token remains pending and is absent from the measured
  target state. Continuation consumes that same pending token once in both arms,
  compares full logits, prepared distributions, hidden transfer, and resulting
  active state, and diagnoses the next RNG transition without advancing the
  measured samplers or charging another generated token.
- Stop precedence is EOS before token limit when both hold at the same emitted
  token, matching the existing development oracle.

## Evidence And Failure Discipline

- The development harness uses a dedicated schema. A strict independent reducer
  is required before any acquisition. E0 fields are not added to the E1a schema,
  whose `e0_status=not_measured` remains true.
- The trace and state sidecar are exclusive-create artifacts. Canonical target,
  drafter, manifest, trace, sidecar, and executable paths must be pairwise
  distinct before the trace is reserved, and a reduction accepts exactly one
  trace containing exactly one run.
- A successful acquisition requires a clean, matching build identity with no
  override. E0 owns this validation after trace reservation so a dirty or stale
  binary rejection is retained as a pre-observation bootstrap failure.
- Full logits and hidden bytes are directly auditable in the development trace.
  Target state uses independently checked active-range geometry plus per-arm
  section hashes and direct producer byte comparison. A later authoritative
  packet may replace large inline values with immutable authenticated sidecars.
- Packed or layer-major target results, if added later, use a separate arm with
  `different_distribution=true` and cannot affect E0 status.
- Any observed difference is a failed E0 row. After trace reservation,
  bootstrap and execution errors produce a retained terminal record when the
  artifact can still be synced. Path reservation failures and terminal fsync
  failures may leave no reducible row; they cannot count as evidence or justify
  a replacement run and require protocol review before retry.
- No held-out or authoritative acquisition begins until prompt bytes and token
  hashes, request lengths, both target quantizations, arm orders, seeds,
  commands, reducer hash, and fixture roles are frozen in a new packet manifest.

For the first Q4 development sentinel only,
`E0-Q4-DEV-BINDING.json` prospectively freezes target and drafter content
identities, target/drafter geometry, capture-layer order, and the target snapshot
ABI, plus the fixture ID and development-sentinel role. Those values were copied
before E0 acquisition from the already
authenticated schema-v4 development trace and its independently validated
snapshot geometry. The E0 producer and reducer must both reject any mismatch.
This development binding does not freeze a held-out fixture or grant held-out
authority.

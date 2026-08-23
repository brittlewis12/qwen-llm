# E0 Q4 v3 Reverse-Order Development Result

Status: reverse-order development sentinel passed. This establishes within-run
arm parity for one narrow fixture under `capture_then_serial`; it grants no
held-out, performance, packed-verifier, serving, or product authority.

## Decision

- **GO:** no within-run parity failure was observed under either execution order
  across the separately frozen v2 and v3 diagnostics on the same fixture.
- **NO-GO:** do not call this an independent replication, rule out order effects
  generally, authorize a sampled verifier, change serve policy, or infer
  quantization-wide/model-wide E0 closure.
- **Next gate at acquisition time:** exercise intermediate generated target
  transitions. The subsequent serial-first v4 fixture passed three such
  transitions. Reverse generated-history order and prompt diversity remain the
  next E0 gaps; authoritative K0 remains independently open.

## Frozen Run

- Source/build/runtime commit:
  `25c66df8bf1359d13b19e948c94cbd87bce63c08`.
- Build/runtime source state:
  `git-source-sha256-v2:1ec9636ef65b3b232f2a9dbee3f927c692b01a2c7c9197971233a47f086c6a72`.
- Build identity: clean `match`, both dirty flags false, no problems or
  overrides.
- Run ID: `dflash-e0-1787454177326364000-20240`.
- Fixture: `e0-q4-token-major-reverse-v3`, role `development-sentinel`.
- Arm order: `capture_then_serial`, required by binding-manifest schema v2.
- Prompt/config: UTF-8 `Hello`, one prompt token, sampler-v1, seed 0,
  temperature 0.7, top-k 200, top-p 1.0, min-p 0.05, one emitted token,
  warmup enabled.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Q4 drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Executable SHA-256:
  `b6aa8a52e6c5746cafa08ac9b063553d8b373d5fd4940a73d68113c26d96c802`.
- Binding SHA-256:
  `b22496e0fe483e2b515537d81f2682f5c80ba1ddfa348dddab2b8d9e7ef8874a`.
- Reducer SHA-256:
  `2db8c4d2df795a771be035493f3c032e945ab5234e03693b84942607f43d6eb5`.

The exact one-invocation command, identities, paths, and decision rule were
committed in `E0-Q4-V3-REVERSE-RUN-PLAN.md` before acquisition.

## Reduction

The exclusive-create reduction records exactly one input, one run, one pass,
zero development failures, zero invalid-pre-observation runs, one emitted token,
two transitions, and `observed_failure=false`:

```text
development_gate=development_lockstep_passed
development_lockstep_passed=true
runs=1
passed=1
development_failed=0
invalid_pre_observation=0
emitted=1
transitions=2
```

As in v2, the transitions are the prompt transition and pending-token
continuation; no intermediate generated target transition is present. The same
strict checks cover full logits, independently reconstructed sampler/RNG,
exact active target state and sidecar geometry, hidden transfer/context history,
terminal accounting, continuation, build/assets, causality, and authority.

## Artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Trace | 9,432,544 | `d7a23e8a0bb9ee738f25a2e949a16e5445ca60edf767fb8066fda2e3f363bf96` |
| State sidecar | 941,883,392 | `d198464ed1b46e62059ffee7e924ae279989e65202ff8dfdc3e204f304f61bd6` |
| Reduction | 828 | `898633c1e80373b8d065643743afb152e21fb3409ddfa5dc048edf5d4d048141` |

Originals remain at their frozen `/private/tmp` paths. Verified mode-`0444`
copies are in the local, non-immutable quarantine
`/Users/tito/Documents/qwen-evidence/dflash-e0/2026-08-22-v3-reverse-pass/`.

The v2 and v3 sidecars have the same SHA-256. That deterministic cross-run
identity is supportive diagnostics, not a substitute for either strict
within-run reduction and not independent evidence.

## Scope Limit

V3 deliberately repeats the already exposed v2 computational fixture and
changes only execution order as treatment. It remains one host, one Q4
target/drafter, one trivial prompt, one seed/configuration, one emitted token,
and no intermediate generated transition. Exact hidden bytes do not establish
the intended residual semantics. Packed/layer-major verification, rollback,
sparse-q/K0 correctness, acceptance, speed, memory economics, longer histories,
Q8/BF16, held-out prompts, serving, and product integration remain untested.

## Subsequent Generated-History Diagnostic

The schema-v3, four-token v4 fixture subsequently passed under
`serial_then_capture`, including three intermediate generated target transitions.
This extends depth on the same exposed prompt but does not supply reverse-order
generated-history coverage or prompt diversity. See
`E0-Q4-V4-GENERATED-RESULT.md`.

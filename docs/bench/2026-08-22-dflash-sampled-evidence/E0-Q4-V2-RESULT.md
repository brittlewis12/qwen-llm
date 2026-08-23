# E0 Q4 v2 Development Result

Status: development-only lockstep sentinel passed. This result grants no
held-out, packed-verifier, performance, serving, or product authority.

## Decision

- **GO:** retain `single_token_with_multi_hidden` as a candidate capture path for
  further sampled DFlash development. On this exact fixture, the strict reducer
  found it bitwise non-interfering with ordinary `single_token` execution.
- **NO-GO:** do not authorize a sampled product verifier, change serve from
  greedy-only, select Q4 as a default drafter, or infer model-wide E0 closure.
- **Next gate at acquisition time:** reverse arm order as the cheapest direct
  interference falsifier. The subsequently frozen v3 sentinel passed that
  narrow check; generated-transition and sustained-history E0 coverage remain
  open before authoritative sparse-q/K0 and verifier work.

## Frozen Run

- Source/build/runtime commit:
  `df5d5d754f086fbf91112746eaa0603735e9eada`.
- Build/runtime source state:
  `git-source-sha256-v2:2da689443c7e06309cea929a830c28b6f55dfdfa80a2cb96e368e3841adcde81`.
- Build identity: clean `match`, both dirty flags false, no problems or
  overrides.
- Run ID: `dflash-e0-1787452242144870000-13697`.
- Fixture: `e0-q4-token-major-smoke-v2`, role `development-sentinel`.
- Arm order: `serial_then_capture`.
- Prompt: UTF-8 `Hello`, one token; token i32le SHA-256
  `7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381`.
- Sampling: sampler-v1, seed 0, temperature 0.7, top-k 200, top-p 1.0,
  min-p 0.05, one emitted token, warmup enabled.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Q4 drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Executable SHA-256:
  `fd53d3ad456d90904b12a8ac26c9bdbc9ab0c266693a5bc9c555432777cd17b0`.
- Binding SHA-256:
  `9997f1bcbfbed4034e688c0c8f1eacbcdcd1336bb281fb501f53771fa860e066`.
- Reducer SHA-256:
  `319e0a34452fccf2d2461067dc60418e2c79c1f2c22a6bdb68ab82df56ecd09b`.

The exact v2 command and decision rule were committed in
`E0-Q4-V2-RUN-PLAN.md` before v2 acquisition. The computational fixture was not
untouched: invalid v1 had already exposed an internal producer success on the
same inputs. V2 is therefore protocol-prospective, not a fresh hypothesis test
or independent replication.

## Reduction

The exclusive-create reducer output records:

```text
development_gate=development_lockstep_passed
development_lockstep_passed=true
runs=1
passed=1
development_failed=0
invalid_pre_observation=0
emitted=1
transitions=2
observed_failure=false
```

The two transitions are the one-token prompt transition and pending-token
continuation. The requested generation length leaves no intermediate generated
target transition.

Within that narrow boundary the reducer independently validated:

- full target-logit bits and exact target snapshot bytes after each transition;
- sampler-v1 ordering/filtering/weights reconstructed from full logits,
  xoshiro256++ transition reconstructed from the frozen seed/state,
  categorical choice, draw counts, and nonadvancing continuation diagnostic;
- active KV positions and KV/GDN/conv state geometry, contiguous sidecar ranges,
  section hashes, direct arm-byte identity, and whole-sidecar coverage;
- hidden destination poison overwrite, finite values, exact source/destination
  bytes, complete active context history and positions, and DFlash watermarks;
- emitted stream, pending terminal token, stop precedence, and continuation;
- clean build, model/drafter/executable/binding identities, event causality,
  single-trace/single-run exclusivity, and development-only authority.

## Artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Trace | 9,432,483 | `0a94e8d3eb6fdec24ae0ba8ec34c09c581de26fb0802c20c41bfca16e79b7e7d` |
| State sidecar | 941,883,392 | `d198464ed1b46e62059ffee7e924ae279989e65202ff8dfdc3e204f304f61bd6` |
| Reduction | 827 | `a99fa23526f78f216e2572f70f88aa7a64802db48d0caccd7ce4e43e1dc45f06` |

Original artifacts remain at their frozen `/private/tmp` identities. Verified
mode-`0444` copies are in the local, non-immutable quarantine
`/Users/tito/Documents/qwen-evidence/dflash-e0/2026-08-22-v2-development-pass/`.

## Incident Boundary

The earlier v1 artifact is governed by `E0-INCIDENT-001.md`. Its producer text
is diagnostic only; the frozen reducer rejected malformed run metadata, so v1
has zero E0 authority and is not pooled with v2. V2 is a distinct evidence
artifact that does not erase or replace v1 in the ledger, but operationally it
is a protocol-corrected rerun of the same computational fixture. It is not an
independent replicate or an untouched prospective test.

## Scope Limit

This is one Q4 target/drafter, one one-token prompt, one seed, one sampling
configuration, one emitted token, one arm order, and one continuation. It does
not test packed or layer-major verification, rollback, sparse-q correctness,
acceptance length, speed, memory economics, longer histories, reverse execution
order, Q8/BF16 quantizations, cross-implementation K0 parity, held-out prompts,
serving, or product integration. Exact hidden transfer also does not independently
prove that the captured values have the intended target-layer residual semantics;
that remains a separate binding/correctness obligation.

## Subsequent Order Diagnostic

The separately frozen `capture_then_serial` v3 sentinel also passed its strict
reducer. Together the two reductions report no observed within-run parity
failure under either order on this same narrow, already exposed fixture. This is
not an independent replication or a general exclusion of order effects. See
`E0-Q4-V3-REVERSE-RESULT.md`.

# E0 Q4 v4 Generated-History Development Result

Status: serial-first generated-history development sentinel passed. This result
grants no held-out, reverse-order generated-history, performance,
packed-verifier, serving, or product authority.

## Decision

- **GO:** on this exact fixture, ordinary and multi-hidden target execution
  remained bitwise lockstep while three sampled tokens were subsequently
  consumed as generated target transitions.
- **NO-GO:** do not infer prompt/seed/model/quantization-wide E0 closure,
  authorize a sampled verifier, change serve policy, or treat this exposed
  fixture as independent evidence.
- **Next gate:** bind and run the same four-token treatment under
  `capture_then_serial`, then move to a distinct development prompt before
  spending on longer histories or additional seeds.

## Frozen Run

- Source/build/runtime commit:
  `30bd674d2b126522c4967c45db63791f56a18f2e`.
- Build/runtime source state:
  `git-source-sha256-v2:d4c66ae81366c60f5d12dae2bdd904563fdb5dd7124f6a2ec1058aa04baf958f`.
- Build identity: clean `match`, both dirty flags false, no problems or
  overrides.
- Run ID: `dflash-e0-1787455997509690000-23319`.
- Fixture: `e0-q4-generated-history-v4`, role `development-sentinel`.
- Prompt/config: UTF-8 `Hello`, one prompt token, four requested/emitted tokens,
  sampler-v1, seed 0, temperature 0.7, top-k 200, top-p 1.0, min-p 0.05,
  warmup enabled, `serial_then_capture`.
- Generated IDs: `[11, 353, 2688, 264]`, i32le SHA-256
  `1087620cd24c9bad460a51b627d56d9dad8e7a4b4079b0e59d3bb58080aa66da`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Q4 drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Executable SHA-256:
  `992b802f15af64b5e277b5a3603d4a1d42cd8a35fd664ac3beaea9a7cd59d10e`.
- Binding SHA-256:
  `64a09cfda9bd8944f236208f95d55c3ad9e5662750ddaabf38cd0a86b5545c6f`.
- Reducer SHA-256:
  `43b9fa9216fabcff8f573152939be01f0d1b7980000a89f500fbf83de304c059`.

Binding-manifest schema v3 prospectively fixed the full prompt and config
objects, including request length, context capacity, stops, sampler bits, seed,
warmup, semantics, and exact arm order. The command and decision rule were
committed in `E0-Q4-V4-GENERATED-RUN-PLAN.md` before acquisition.

## Reduction And Objective

The exclusive-create strict reduction reports:

```text
development_gate=development_lockstep_passed
development_lockstep_passed=true
runs=1
passed=1
development_failed=0
invalid_pre_observation=0
emitted=4
transitions=5
observed_failure=false
```

The trace contains one `prompt_step`, four `sample_frontier` rows, three
`target_transition` rows, one terminal boundary, and one continuation. Producer
terminal accounting records `target_transitions=3`, `sample_frontiers=4`, both
samplers at four draws, token-limit stop without EOS, continuation equality, and
final consumed prefix/DFlash context length five.

The sidecar is exactly `1,885,208,576` bytes, matching six serial/capture
snapshot pairs at prefix lengths `1, 2, 3, 4, 4, 5`. The reducer authenticated
every contiguous section and whole-file coverage while independently checking
full logits, sampler-v1/RNG transitions, exact cross-arm KV/GDN/conv byte parity
and geometry, hidden transfer and complete context/position history,
pending-token semantics, causality, build/assets, and development-only
authority.

## Artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Trace | 25,094,203 | `79be314c6417c3155d717742f9984f33ed1936d4a064a37008d7957bcf6df099` |
| State sidecar | 1,885,208,576 | `5e16cd524a6f21b521c3f626bc3e58f4ec2840ae2a1419f347609fb2f9abc424` |
| Reduction | 831 | `badb7d6d329b2fcf50f6f93419c7f89d004a9e9b989bfe4c4e6a20fa3c4e0cd3` |

Originals remain at their frozen `/private/tmp` paths. Verified mode-`0444`
copies are in the local, non-immutable quarantine
`/Users/tito/Documents/qwen-evidence/dflash-e0/2026-08-22-v4-generated-pass/`.

## Scope Limit

V4 deliberately reuses the exposed one-token `Hello` prompt and seed. It is one
host, one Q4 target/drafter, one sampling configuration, four emitted tokens,
and serial-first order. It demonstrates exact generated-transition parity only
inside that boundary. It does not independently prove intended hidden residual
semantics or test reverse generated-history order, sustained/long contexts,
prompt/seed diversity, Q8/BF16, sparse-q/K0, packed verification, rollback,
acceptance, economics, performance, serving, held-out behavior, or product
integration.

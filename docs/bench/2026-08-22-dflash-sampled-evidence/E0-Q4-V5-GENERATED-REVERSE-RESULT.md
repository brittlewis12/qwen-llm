# E0 Q4 v5 Generated Reverse-Order Development Result

Status: reverse-order generated-history development sentinel passed. Combined
with v4, no within-run parity failure was observed under either execution order
on the same exposed four-token fixture. This grants no general order,
replication, held-out, verifier, serving, or product authority.

## Decision

- **GO:** retain the multi-hidden capture path for prompt-diversity E0 work; the
  exact four-token trajectory passed under `capture_then_serial` as well as
  v4's `serial_then_capture`.
- **NO-GO:** do not infer general order invariance, independent replication,
  prompt/seed/quantization-wide E0, hidden semantic correctness, or product
  readiness.
- **Next gate:** a distinct development prompt at the same four-token depth.
  K0 remains independently required before sparse-q work; lane-labeled
  economics follows only after the applicable correctness breadth.

## Frozen Run

- Source/build/runtime commit:
  `2f7694147f970227a5ed2fdf047cf2ebd929b8c1`.
- Build/runtime source state:
  `git-source-sha256-v2:14f2d8afd74d1a796536ad7dabcb228d3a8fe846dfe4500b4f0d7eb95345e84e`.
- Build identity: clean `match`, both dirty flags false, no problems or
  overrides.
- Run ID: `dflash-e0-1787457456090392000-25784`.
- Fixture: `e0-q4-generated-reverse-v5`, role `development-sentinel`.
- Prompt/config: UTF-8 `Hello`, one prompt token, four requested/emitted tokens,
  sampler-v1, seed 0, temperature 0.7, top-k 200, top-p 1.0, min-p 0.05,
  warmup enabled, `capture_then_serial`.
- Generated IDs: `[11, 353, 2688, 264]`, i32le SHA-256
  `1087620cd24c9bad460a51b627d56d9dad8e7a4b4079b0e59d3bb58080aa66da`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Q4 drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Executable SHA-256:
  `735a16efa31f1c1d5e27eba3c928e2ed08e8ae41d85eb3e9c0f847d1132cbd23`.
- Binding SHA-256:
  `fe7c1e0c6550a151626a07b34bfc57241de8fb81d95c1a20c12c99077a960884`.
- Reducer SHA-256:
  `43b9fa9216fabcff8f573152939be01f0d1b7980000a89f500fbf83de304c059`.

Schema-v3 binding fixed the full prompt/config and exact reverse order. The
one-invocation command and decision rule were committed in
`E0-Q4-V5-GENERATED-REVERSE-RUN-PLAN.md` before acquisition.

## Reduction And Objective

The strict reduction reports one input, one run, one pass, four emissions, five
validated transitions, zero failures/invalid-pre-observation runs, and no
observed failure. The trace contains exactly one prompt, four sample frontiers,
three generated target transitions, one boundary, and one continuation.

Producer terminal accounting records generated IDs `[11, 353, 2688, 264]`,
three target transitions, four draws in each sampler, token-limit stop without
EOS, equal continuation, and final consumed prefix/DFlash context length five.
The sidecar exactly matches the preregistered `1,885,208,576` bytes with complete
authenticated coverage of six arm pairs at prefix lengths `1, 2, 3, 4, 4, 5`.

## Artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Trace | 25,094,659 | `e8a670d16d535c41d10d57a7fa316dd235cfe659c80b1d47c20b1250b44f101b` |
| State sidecar | 1,885,208,576 | `5e16cd524a6f21b521c3f626bc3e58f4ec2840ae2a1419f347609fb2f9abc424` |
| Reduction | 839 | `7addcc7bb8ef7a5877ddce5fe32d3e64f8cdd0dcc126d88c57dc8b51534245a5` |

Originals remain at their frozen `/private/tmp` paths. Verified mode-`0444`
copies are in the local, non-immutable quarantine
`/Users/tito/Documents/qwen-evidence/dflash-e0/2026-08-22-v5-generated-reverse-pass/`.

V4 and v5 state sidecars are byte-identical. That supports deterministic
cross-run serialized target-snapshot identity for this fixed trajectory but does
not replace either within-run strict reduction or establish independent
evidence.

## Scope Limit

V5 reuses the same exposed `Hello` prompt, seed, generated stream, Q4 assets,
host, and sampler configuration as v4. Only execution order changes as the
computational treatment. Prompt/seed diversity, sustained contexts, Q8/BF16,
independent hidden residual semantics, sparse-q/K0, packed verification,
rollback, acceptance, economics, performance, serving, held-out behavior, and
product integration remain open.

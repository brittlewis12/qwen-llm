# v0.658 Positive-Temperature Sampling Attribution Result

Status: **CONSUMED_NO_AUTHORITY**. The sole acquisition completed CPU and
model conformance plus the unprofiled product reference, then failed closed on
an incorrect runner schema expectation. No profiled child launched, no
sampling-attribution value exists, and the packet grants no candidate,
performance, falsification, or product authority.

## Identity And Host

The packet used clean matching build and runtime identity:

```text
commit: f3ce16cb08de53f3a8bd67b402ef890b6faf71a8
source: git-source-sha256-v2:b24a7a7dc63af5d87a4eadfc06592fe9a6d41af49376baf4facb6ec849410329
build/runtime status: match; clean; no overrides
```

The early readiness census found zero competitors. The full host gates before
model conformance and the reference also found zero competitors, AC power, no
thermal or performance warning, 96% parsed memory availability, and median CPU
idle of `90.15%` and `85.38%`. Static records before and after model conformance
are byte-identical.

## Completed Work

- The sampler suite passes 16 tests and the CLI suite passes 57 tests.
- The release A3B model-conformance test passes its frozen exact logits, state,
  and continuation contract.
- The ordinary reference child exits zero without timeout, process leak, or
  termination action. It emits 128 tokens, records `token_limit`, uses
  `sampled_cpu`, and writes one schema-10 timing row.

These are conformance and diagnostic facts only. The reference contributes no
performance number under the frozen contract.

## Failure Diagnosis

Reference validation stopped on:

```text
extra keys:   []
missing keys: [prefill_attention_query]
```

`prefill_attention_query` is an optional product field with
`skip_serializing_if = "Option::is_none"`. The exact 419-token fixed-chunk
fixture produced no tiled-attention query record, so omission is valid and
predates v0.657. The inherited runner incorrectly included the field in the
mandatory schema-10 key set and then required it to be an object.

This is a frozen runner-parser defect, not product schema drift, a model
failure, or evidence about the attribution candidates. The runner correctly
failed closed rather than accepting a shape it had not declared.

## Terminal Scope

The sealed terminal record states:

```text
disposition: CONSUMED_NO_AUTHORITY
failure_category: infrastructure
stage: reference-validation
profile_spawned: false
profile_result_observed: false
authority: []
```

No profile, schema-11 attribution object, W/B/C/S bound, final host gate, or VM
delta exists. The ordinary reference timing row must not be rescued as
performance evidence. v0.658 must not be rerun, repaired, or completed under
the same version or packet root.

## Next Decision

A separately preregistered parser-only v0.659 successor is justified because no
profile or candidate result was observed. It may use the missing-key observation
only to justify the parser correction. It must use a new root, import no v0.658
conformance, reference, or timing evidence, and rerun every inherited readiness,
host/VM, correctness, and acquisition gate. It may change only the tracked
predecessor identities plus the exact-fixture schema rule:

- remove `prefill_attention_query` from the mandatory reference key set;
- require it to be absent in both the reference and every profile; and
- remove the stale object validation while leaving all fixture, timing,
  equality, reduction, threshold, authority, and stopping rules unchanged.

Positive-temperature attribution remains first in the active queue because the
phase-size question is still unobserved.

## Artifacts

The 23-file sealed packet is under
`target/profiles/v0658-positive-temperature-sampling-attribution-p1/`.
`decision.json` and `failure.json` are byte-identical with SHA-256
`2cb06a3b04698cf1711524ae0114f24504cf9c2fff5d6b9cefbe2d348a289089`.
Independent result review: `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.

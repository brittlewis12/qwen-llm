# v0.655 Grammar-Row Lm-Head Charged Floor Result

Status: **GO** on the A3B primary and **GO** on the conditional dense transfer
guard. The A3B result authorizes exactly one later integrated packet for the
frozen response-shape grammar, tokenizer policy, A3B model, state-major Q6_K
bank, and sampler-v1 contract. Dense has primitive-transfer authority only.

## Scope

The floor compares:

- `A`: the complete 248,320-row production Q6_K `output.weight`; and
- `B`: a prebuilt, state-major Q6_K bank containing only each branch state's
  admissible rows.

Each scored command buffer contains exactly one unchanged production Q6_K
mat-vec. The charged wall begins before checked state lookup and includes
command creation and encoding, completion and status checks, selected-logit
readback, sampler-v1 execution, and local-to-vocabulary token mapping. A
separate full-head dispatch conditions every scored arm.

Incremental setup `B` includes manifest read, hash, parse and validation, layout
construction, all 1,468 row copies and padding, bank hashing and upload, 451
read-only views, and compact guarded output views. Model hashing, full-head
upload, Metal initialization, PSO setup, and correctness acquisition are common
fixture costs and are excluded by the frozen contract.

## Identity And Acquisition

The clean source and release binary were:

```text
commit: c0b8289630b04390b7688174faac79dffb47f4ab
source: git-source-sha256-v2:4499acefdf7a548048b2dfb58b3e0f364ce52eaec9137c488a39af5af7dcf334
build/runtime status: match; clean; no overrides
manifest SHA-256: 2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0
```

Canonical result artifacts:

| Artifact | SHA-256 |
|---|---|
| A3B JSON | `1e37f4882be856fc8043a98a00bfeaa3a7ff8a726ab0f73a99a9e98d707c689c` |
| Dense JSON | `3b57c94b97e599e0c29b9f919cfea9db55d7a8b8ec238ea72f9d5e44b86adbea` |
| Either stderr log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The machine was on AC power with no thermal or performance warning before and
after both runs. There was no competing qwen, llama, or Metal benchmark. Five
orphaned `rg --files` processes were identified and terminated before
acquisition. `mds_stores` and `mediaanalysisd` remained active while the machine
was approximately 87% CPU-idle; system daemons were not manipulated. ABBA
counterbalancing addresses temporal drift but cannot categorically remove
arm-dependent contention or unified-memory effects. This is one preregistered
acquisition, not repeat-run or confidence-interval authority.

## Correctness

Both profiles pass the complete frozen contract:

- four exact LCG hidden vectors;
- four full-head and 1,804 compact-state dispatches;
- 5,872 selected logits bit-identical to their full-head values;
- all compact outputs overwritten from a fixed NaN poison;
- intact prefix, suffix, internal-padding, and trailing-padding guards;
- unchanged source-row and complete uploaded-bank hashes;
- 27 compact-versus-full-mask sampler cases covering greedy, finite ties,
  `qwen_chat` seeds `0/1/42/u64::MAX`, finite and positive-infinity ties,
  top-k below state width, NaN at every local position, and all `-inf`; and
- exact token, error-token, candidate-index, and draw-count agreement.

Every raw conditioning and timed observation records the production kernel,
`completed` status, no Metal error, and exactly one mat-vec dispatch. The dense
command is cryptographically bound to the preceding A3B GO result and the same
clean build.

## Result

| Quantity | A3B primary | Dense guard |
|---|---:|---:|
| Charged full head `H0` | `0.992333 ms` | `2.240866 ms` |
| Compact head `H1` | `0.150366 ms` | `0.160238 ms` |
| Charged saving | `0.841967 ms` | `2.080628 ms` |
| Head wall removed | `84.8472%` | `92.8493%` |
| Frozen-denominator projection | `9.03398%` | `5.38758%` |
| Margin above 5% gate | `4.03398 pp` | `0.38758 pp` |
| Incremental setup `B` | `7.099042 ms` | `13.705458 ms` |
| Minimum 36-path net after `B` | `+8.047171 ms` | `+23.744962 ms` |

All nine width medians save time. A3B's range is
`0.816271-0.873709 ms`; dense's is `2.076792-2.085875 ms`. Width seven
retains its positive gate despite having zero canonical or trace weight.

Dense's whole-transition margin is narrower, but the AB/BA halves, acquisition
halves, leave-one-round-out reductions, and GPU-clock diagnostic remain above
the frozen 5% threshold. These are robustness diagnostics, not additional
decision gates.

## Decision

The exact branch-row organization clears every A3B gate:

- positive median paired wall saving at every width;
- at least 70% weighted charged-head removal;
- at least 5% projected whole-token saving against `T0=9.32 ms`;
- positive setup-charged net on all 36 canonical paths; and
- complete identity, bounds, correctness, sampler, guard, and Metal validity.

The conditional dense guard independently clears the same contract against
`T0=38.619 ms`. This strengthens causal transfer of the primitive to the exact
dense profile, but does not authorize dense integration.

The result does not establish integrated token or request speedup, real-hidden
performance, full-vocabulary logit equivalence, general schema or tokenizer
transfer, runtime APIs, or default behavior. Percentages remain screening
projections against frozen external denominators.

Advance to one preregistered A3B integrated packet. It must charge real grammar
state tracking, branch lookup, bank setup in TTFT, exact singleton/terminal
handling, real model hidden states, constrained sampler-v1 semantics, and total
request wall. Do not build a general grammar framework or a dense product path
under this authority.

Independent design, implementation, and result review: `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.

# v0.659 Positive-Temperature Sampling Attribution Result

Status: **GO_BOUNDED_IMPLEMENTATION** with `authority=["structural"]` and at
most one successor packet. Only the optimistic borrowed-logit plus
streaming/direct-top-k bound clears the frozen gate. Workspace, borrowed-logit,
and workspace-plus-borrowed bounds are killed in this exact cell.

## Scope

The packet attributes one exact fresh A3B request:

- 419 prompt tokens and 128 generated tokens;
- temperature `0.7`, top-k `200`, top-p `1.0`, min-p `0.05`, seed `42`;
- 128 sampler calls, 127 target transitions, and token-limit termination; and
- one ordinary reference followed by six independently fresh profiled
  processes.

This is implementation-packet authority, not a realized optimization or a
general positive-temperature sampling claim.

## Identity And Validity

```text
commit: ba2871aaff8841c3a267bcc7204a63a8ed687573
source: git-source-sha256-v2:4d848fb3cb4e4bb8c2ff72859c79386e5faa9f305887bc2b48d093e97f3a49e0
build/runtime status: match; clean; no overrides
```

Both static attestations are identical. The early readiness census reports zero
competitors. Nine subsequent full host gates--before model conformance, before
reference, before each of six profiles, and after profiles--pass on AC power
with no thermal or performance warning, at least 86.95% median CPU idle, and no
detected competitor. Swap, pageout, compression, and swapout deltas are zero;
compressor occupied/stored pages decline by 2/199 and `fatal_growth=[]`.

The decision embeds decisive child, VM, and static evidence. Intermediate host
snapshots remain separate packet files rather than cryptographic children of
`decision.json`; no stronger sealing claim is made.

## Conformance

- The sampler suite passes 16 tests and the CLI suite passes 57 tests.
- The release model test passes bit-identical ordinary/profiled logits,
  continuation logits, and KV/GDN state.
- The reference and all profiles complete with exit zero, no timeout, leaked
  process group, termination action, or failure record.
- The reference and all six profiles emit identical stdout and token digest,
  128 tokens, 127 transitions, 128 draws, `token_limit`, and `sampled_cpu`.
- Every profile records exactly 1,408 sampler and 254 transition timer spans.
  All additive inner, outer, sampler, and observer-bound reconciliations pass.
- Each call retains exactly 200 top-k candidates. The later filters retain
  1-14 candidates, total 358 over 128 calls; candidate indices are 0-4.

## Frozen Reduction

Generation wall spans `1240.127291-1263.572250 ms`. Observer subtraction spans
`0.049860-0.051522 ms` per profile.

| Bound | Adjusted range | Adjusted median | Median generation share | Clears |
|---|---:|---:|---:|---:|
| `W` workspace | `2.646814-3.088695 ms` | `2.794766 ms` | `0.222851%` | `0/6` |
| `B` borrowed logits | `6.531931-6.937473 ms` | `6.705743 ms` | `0.534794%` | `0/6` |
| `C` workspace + borrowed | `6.545602-6.952313 ms` | `6.719675 ms` | `0.535834%` | `0/6` |
| `S` structural | `79.746811-82.383260 ms` | `81.400598 ms` | `6.545897%` | `6/6` |

The authority partition is therefore exact:

- close reusable workspace alone;
- close borrowed synchronized logits while retaining current candidate work;
- close workspace plus borrowed logits without changing candidate work; and
- authorize one combined borrowed-logit plus streaming/direct-top-k design
  packet.

These closures apply only to the exact A3B fixture and current mechanism. They
do not establish universal sampler costs.

## Mechanism Learning

The full sampler wall is about `75.944 ms` per request. Full-vocabulary
candidate fill accounts for median `39.324 ms`; current selection and ordering
accounts for `35.383 ms`. Candidate-vector allocation itself is only
`0.013 ms`. Transition logits allocation plus copy contributes only the
roughly `6.7 ms` borrowed bound.

The optimistic attributed ceiling is therefore concentrated in candidate
organization, not allocation. An exact bounded top-k scan must still inspect
every logit, retain the best 200 under descending-logit/ascending-token order,
sort those 200, and run the existing f64 filters and categorical draw. Decode
transitions may consume the completed Shared F32 row in a scoped borrow; the
first prompt selection retains its current owned logits row.

The `81.401 ms` result is an optimistic removal bound, not expected saving. It
is only about `4.99%` of the profiled median complete request before charging
the replacement scan and heap. A successor therefore keeps the frozen `>=5%`
generation gate and uses an absolute complete-request saving gate rather than
requiring `>=5%` of request wall.

## Successor Boundary

The one authorized packet must combine both structural changes:

1. exact bounded top-k candidate preparation for `temperature>0` and
   `0<top_k<vocab`; and
2. scoped sampling from the synchronized Shared F32 logits row after decode
   transitions, without a full host logits allocation or copy.

It must preserve first-NaN errors, signed zero, infinities, all-negative
infinity, f32-to-f64 conversion, total-order ties, downstream filter order,
candidate index, RNG state/draw count, callbacks, stops, N-1 transition and
pending-token semantics, and complete KV/GDN/conv continuation state.

Exclude workspace-only work, GPU sampling, prompt-logit borrowing, lm-head or
command-schedule changes, sampler-v2, cache formats, prompt lookup, and broad
runtime redesign.

The charged packet should require median paired generation saving `>=5%` with
at least five of six wins, median paired model-ready request saving `>=5 ms`
with at least five of six wins, positive AB/BA strata, TTFT regression
`<=10 ms`, and nonregressive process-cold spawn-to-exit wall.

## Artifacts

The complete 61-file packet is under
`target/profiles/v0659-positive-temperature-sampling-parser-repair-p1/`.
`decision.json` SHA-256 is
`ae21f8944eae07d950127961f29b6e2814efbc6970af60994fee175655af8534`.
Independent result and candidate review: `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.

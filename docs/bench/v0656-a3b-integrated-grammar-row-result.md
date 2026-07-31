# v0.656 A3B Integrated Grammar-Row Result

Status: **KILL**. The exact compact Q6_K head removes real transition work,
but the frozen request-local bank organization slows the complete request and
regresses TTFT. The packet grants no performance or product authority.

## Scope

The packet compares one exact response-shape request on the A3B primary:

- `A` executes the resident 248,320-row production Q6_K head at every
  nonterminal state; and
- `B` prepares request-local state-major branch and singleton banks, then
  executes only the current state's admissible Q6_K rows.

Both arms share the production model body, grammar runtime, sampler-v1,
callbacks, terminal semantics, and fresh request-local sequence state. The
primary clock includes prompt and runtime parsing, B's manifest and complete
bank setup, prefill, all transitions, commit checks, teardown, and autorelease
pool drain.

## Identity And Validity

The clean source and release binary were:

```text
commit: 5b058826c2d57ccb561c09163af7a716195afdbb
source: git-source-sha256-v2:ed20011f500d519d54bbb886ad1b6d8ba3b2f71b93ca11e722c45d09449b3ad4
build/runtime status: match; clean; no overrides
```

The acquisition completed all five warmup and 12 alternating scored pairs.
The host remained on AC power with 79% parsed memory availability, no thermal
or performance warning, and no detected qwen, llama, Metal benchmark, or user
GPU competitor. Median CPU idle was `85.25%`, `83.65%`, and `86.92%` before
warmup, before scoring, and after scoring. Swap occupancy, pageouts,
compressions, swapouts, and compressor stored/occupied gauges did not grow.
The process recorded one page-in and `32,768` disk-read bytes; both were
nonfatal under the frozen validity contract.

## Conformance

All frozen correctness gates pass:

- all 451 branch and 129 singleton states are checked on four hidden vectors;
- 6,388 selected logits are bit-identical across four full and 2,320 compact
  dispatches;
- command status, output overwrite, guards, padding, source rows, uploads, and
  complete post-use bank hashes pass;
- seeds `0`, `1`, `42`, and `u64::MAX` agree at every generated position on
  selected logits, post-norm hidden state, sampler result and draw count,
  grammar edge, callback bytes, and complete model state;
- every stream emits 18 tokens, consumes 17 transitions, performs 18 sampler
  draws and callbacks, and commits token `8934` as the pending final token; and
- all 72 B tails are compact Q6_K dispatches, with zero full-head or terminal
  head dispatches.

The conformance result has `authority=none`. Both conformance and acquisition
stdout captures contain one tracing line before their unique JSON record. It
reports excluded model-load residency and is only an artifact-container
nuisance.

## Result

| Frozen quantity | Result | Gate |
|---|---:|---:|
| Median paired request saving, `A - B` | `-7.662104 ms` | `>=5.0 ms` |
| Positive request pairs | `1/12` | `>=10/12` |
| Median generation saving, `A - B` | `+14.229042 ms` | `>0 ms` |
| Median TTFT change, `B - A` | `+15.099958 ms` | `<=10.0 ms` |
| AB request-saving median | `-7.106459 ms` | diagnostic |
| BA request-saving median | `-7.901708 ms` | diagnostic |
| First-half request-saving median | `-7.662104 ms` | diagnostic |
| Second-half request-saving median | `-8.424230 ms` | diagnostic |

Every leave-one-pair-out request median is negative, from `-7.693708` to
`-7.630500 ms`.

The mechanism and integration costs separate cleanly:

| Median B-minus-A component | Movement |
|---|---:|
| Manifest validation | `+1.270875 ms` |
| Complete bank preparation | `+9.892437 ms` |
| Upload and view binding | `+4.968604 ms` |
| Prefill command | `-0.898417 ms` |
| Generation | `-14.229042 ms` |
| Commit checks | `+5.406188 ms` |
| Teardown | `+1.622104 ms` |

The compact head saves about `14.197 ms` of transition command wall across 17
transitions, or approximately `0.835 ms` per transition. The root compact tail
saves only about `0.9 ms`. Request-local manifest, bank construction, and
upload cost `16.136 ms` before first token, producing the TTFT regression.
Post-terminal authentication and cleanup add about `7.05 ms` more.

Scored pair zero is the sole positive request pair. Its A-first prefill command
is `238.555 ms`, versus roughly `159.6 ms` later, immediately after the
three-second host-validity interval. Warmup pair zero shows the same pattern.
The packet does not localize the cause; the anomaly favors B and cannot rescue
the negative median. Excluding it would move the request median only to
`-7.693708 ms`.

## Decision And Learning

Correctness, environment, and generation-saving gates pass. The request,
pair-count, and TTFT gates fail, so the frozen disposition is mechanically
`KILL`, not `INVALID` or `NO-GO/PARK`.

The result proves that exact real-hidden admissible-row execution removes the
expected production head work. It also proves that complete request-local bank
materialization is the wrong ownership boundary for this 18-token request.
Do not rerun v0.656, subtract setup, exclude the first pair, transfer the result
to dense, or build a general grammar runtime under this authority.

Only a changed premise can reopen the lane:

- reuse an authenticated bank across multiple requests under a declared
  same-grammar lifecycle;
- avoid complete materialization through a genuinely lazy or indexed head; or
- name a constrained workload with enough transitions to amortize the fixed
  costs, roughly the high twenties under constant-marginal arithmetic.

These are new experiments, not repairs to v0.656. Generic certified lm-head
screening remains an independent greedy-only lane.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Conformance stdout | `00010babdbbd29da27f81c3d9b7155bc8962c9959573595f0ff63e988add8b37` |
| Conformance stderr | `b79d981977e7ea794f89b59874fdad4a430365b529bf700ec92a16315ab2b5bc` |
| Acquisition stdout | `35391660f6842304fa0057b94c54ea8080184cfba857f7298c5863f1884d74b0` |
| Acquisition stderr | `b79d981977e7ea794f89b59874fdad4a430365b529bf700ec92a16315ab2b5bc` |

Artifacts are under
`target/profiles/v0656-a3b-integrated-grammar-row/`. Independent design,
implementation, conformance, and result review: `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.

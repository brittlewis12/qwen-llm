# Fixed Cohorts With Mixed Generation Limits

Date: 2026-08-11

Status: product `GO` for equal-prompt-length dense B=8 and qualified Qwen MoE
B=16 cohorts with independent per-request generation limits. Padding-dominated
candidate cohorts remain serial, and automatic mode may narrow them to B=2.

## Question

Must requested generation length remain part of fixed-cohort compatibility, or
can lanes share a physical decode step while retaining independent logical token
limits?

The existing executors already support early logical completion after EOS. A
finished lane supplies a synthetic token-zero transition while active peers
continue; that selected output is discarded, counted as physical padding, and
the sequence is never checkpointed or reused. A shorter token limit can use the
same mechanism if every sequence receives the cohort's maximum physical
capacity.

## Implementation

- Compatibility is tokenized prompt length. Requested generation limits remain
  ordered per-lane state, not a bucket key.
- Requests within one prompt-length bucket are stably sorted by token limit so
  complete cohorts have similar expected decode depth.
- Every lane receives the maximum derived context capacity in its cohort. This
  preserves the equal-capacity MoE contract and prices the first measured
  sequence conservatively for the remaining lanes.
- A candidate cohort batches only when requested productive transitions are at
  least three quarters of physical transition slots. Lower-utilization cohorts
  become ordinary serial work before executor or sequence allocation. Cohorts
  requesting one token per lane also remain serial because they have no physical
  batch transition to amortize executor setup.
- Cohort telemetry now reports ordered requested/generated limits, common
  capacity, per-lane productive and padding transitions, and realized physical
  utilization. Planner telemetry reports candidate, accepted, and
  economics-rejected cohorts.

This remains fixed membership over a seekable request file. It does not refill
finished lanes, combine different prompt frontiers, mask filler state, or become
continuous batching.

## Validation

Base: `024c508a27fcd62fcac83f811584f0fb8219ecd7` plus this patch. All model/GPU
work was serialized under the process Metal lease on Apple M4 Max. Prompts in
each fixture have identical tokenized lengths but distinct per-request limits.

### Moderate limits

The Qwen3.6 A3B Q4 fixture uses limits 24 through 39. Requested transition
utilization is `0.802632`.

| Mode | Process wall | Generated | Result |
|---|---:|---:|---|
| Serial | `8.85 s` | 504 | control |
| Explicit B=16 | `8.07-8.08 s` | 504 | `1.096-1.097x` whole-process screen |
| Automatic | `8.08 s` | 504 | selects B=16 |

Every process emits stdout SHA-256
`e9873db8419e9e95a13995eb6266e918d5fac42591b7a88ac2d384a1f0277668`.
The B=16 run records 488 productive and 120 padding transitions over 38
physical steps.

The exact three-quarter boundary was tested separately with eight 13-token and
eight 25-token lanes. B=16 records utilization `0.75`, preserves stdout SHA-256
`432918c0362a06bf9cee9cc0dec5d977171b5738fcaabcd1af643ee379705ed9`,
and moves process wall `6.76 -> 6.49 s` (`1.042x`). A homogeneous two-token
fixture also remains exact and moves `4.06 -> 3.98 s`; the gate therefore does
not require a large decode merely to avoid the zero-transition case.

The same 16-limit fixture on Qwen3.5 0.8B Q4 forms two B=8 cohorts with realized
utilization `0.883333` and `0.907895`. Process wall moves `2.33 -> 1.45 s`
(`1.607x`) and stdout remains byte-identical with SHA-256
`a934b0540c6aeb3775c4b3933c14273efc83f982503e57f21aab6c4577a34c7b`.

### Padding discriminator

An intentionally skewed Qwen3.6 A3B fixture uses limits
`1,2,3,4,5,6,7,8,9,12,16,20,24,28,32,40`. Its requested utilization is
`0.322115`.

- The ungated B=16 spike is exact but regresses process wall
  `6.05 -> 8.13 s` (`0.744x`), with 201 productive and 423 padding transitions.
- The three-quarter gate rejects that cohort before executor construction;
  explicit B=16 therefore executes all 16 requests serially in `6.06 s`.
- `--execution-mode auto` narrows the same file to adjacent B=2 pairs and moves
  wall to `5.61 s` (`1.078x` the serial control).
- All three organizations emit stdout SHA-256
  `d6bbeb19c95e4c9658644f4a86d24cd9208cbcb50bf6255e36d54505fe9e43ad`.

## Decision

Remove requested generation count from the compatibility key, but do not confuse
mechanical correctness with useful batching. Keep limit-sorted packing and the
measured three-quarter admission floor. This captures moderate heterogeneous
work while routing padding-dominated files to the already-faster B=2 or serial
organization.

Revisit the floor only with a cross-model sweep or dynamic lane refill that
removes filler work rather than merely tolerating it.

Representative commands were:

```text
qwen --model MODEL --requests-jsonl FIXTURE --temp 0 --prefill-chunk 2048
qwen --model MODEL --requests-jsonl FIXTURE --temp 0 --prefill-chunk 2048 \
  --batch-size 16
qwen --model MODEL --requests-jsonl FIXTURE --temp 0 --prefill-chunk 2048 \
  --execution-mode auto
```

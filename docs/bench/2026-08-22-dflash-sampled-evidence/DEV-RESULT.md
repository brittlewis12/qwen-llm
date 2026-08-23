# DFlash Sampled Development Results - 2026-08-22

Status: development diagnostics only. Post-incident Q4 E0 sentinels now pass the
strict lockstep reducer under both execution orders on one narrow computational
fixture. Four-token fixtures also pass under both orders with three intermediate
generated target transitions. K0 remains a local reference diagnostic without
cross-implementation parity, and no G1, product, serve, default-selection,
model-wide, or held-out authority exists.

## What The Development Runs Establish

The serial one-hot oracle gives DFlash2's current greedy proposal and the target
sampler-v1 one shared draw at each causal position. It accepts only when they are
the same token; otherwise the target draw becomes the correction carry. Across
the development runs below, the oracle and an independent fresh serial target
reproduced the same sampled stream, RNG states and draws, target logits, final
KV/GDN/conv state, stop boundary, and one-token continuation.

This demonstrates that positive-temperature DFlash2 is not inherently
incompatible with sampler-v1. It does not demonstrate an accelerated product:
the oracle performs serial token-major target sampling and is intentionally
slow. E1a rows remain distinct from the E0 result below and must not be
retroactively described as E0 passes.

Schema v4 now records that boundary explicitly, hashes the target, drafter, and
binary, records request/arm/cluster identities, verifies exact post-prefill draw
coverage and event causality, independently derives parity from terminal fields,
and requires an explicit local-reference q temperature. A 16-token real-model
schema-v4 Q4 smoke passed with E0 reported as `not_measured` and the gate reported
as `blocked_by_e0`.

## Q4 Versus Q8 Drafter Exploration

The Q4 drafter was observed after the preregistration froze Q8_0 as the only
drafter. These rows are therefore post-preregistration exploration and cannot be
mixed into the frozen program or used as held-out evidence. Any authoritative
Q4 comparison requires a new prospective preregistration.

Loading the released Q4_K_M asset first required narrow Q4_K selector-codebook
support. That independent compatibility change, with reference-formula and
row-boundary tests plus real-model greedy equivalence, was adversarially reviewed
and promoted to `main` as `1811c38`. It changes model compatibility rather than
sampled-decoding policy.

All rows use the same Qwen3.8-27B Q4_K_M target, seed 42, 64 generated tokens,
and exact serial sampler-v1 target stream. Values are emitted tokens per oracle
block; each row is one correlated request/seed cluster.

| Sampler | Development prompt | Q4 drafter | Q8 drafter | Direction |
| --- | --- | ---: | ---: | --- |
| S1 | code generation | 3.048 | 3.048 | tie |
| S1 | narrative guardrail | 2.286 | 2.207 | Q4 +3.6% |
| S1 | inference explanation | 3.200 | 3.200 | tie |
| S2 | code generation | 3.048 | 3.200 | Q4 -4.8% |

The sample is too small for a drafter-quality winner. It does rule out the
simple concern that Q4 necessarily destroys acceptance: Q4 tied two cells, won
one, and lost one. Proposal paths differed even where emitted/block tied.

Three alternating 256-token static-8 greedy runs on one forced-length prompt
gave steady-state draft means of `[14.4, 13.6, 16.8]` ms for Q4 and
`[15.1, 14.6, 18.2]` ms for Q8. Q4's median draft call was 14.4 ms versus
15.1 ms, about 4.6% lower, while end-to-end decode medians were 21.75 versus
22.22 tokens/s in Q8's favor. Both arms emitted the same 256 tokens in 93 outer
steps with 162 accepted drafts; target verification dominated wall time and the
third pair slowed materially, so the end-to-end difference is noise-sized.

Q4 also reduced the drafter file from 1.92 GiB to 1.06 GiB and approximately
halved observed Metal load time (about 111 ms versus 200-209 ms). The upstream
model card's GSM8K figures likewise report mean acceptance lengths of 5.28 for
BF16, 5.13 for Q8_0, and 5.39 for Q4_K_M. That external benchmark is supportive,
not a substitute for request-cluster evidence here.

Development trace SHA-256 values, as authenticated by their reductions:

| Cell | Q4 drafter | Q8 drafter |
| --- | --- | --- |
| S1 code | `92e896c7624112efaf0ebbeecae7fd6bfe17f82d58a9f873ca1dd7796da7023c` | `ba6faed673021cee930b2cb851c534354caa4c0b9515960db77aab42f57e61d1` |
| S1 narrative | `cf63273123da0128c14a93602ac03fc5dd658abc8bca732c6a1f133edcce7be7` | `6047f4feb410445edab637fcdfae13332b28a86dbf58000f0288d21340942362` |
| S1 explanation | `7db320edc6dfeacd70ceab0174fb2e853cfb078521ab6785d29a49f60a7d2336` | `ae2ba654781ff1ed6e38bb842f4a27034e18d258efa4ecc8b09c84361a0f4bd8` |
| S2 code | `8b7f9141495daa34d472e65084f668a56c72f0bc5132f436a504faec00c16556` | `129f90a0b284eb226c268b5cd7ba2a0c4a39964391645809b54502f1ee66bbd5` |

The schema-v4 Q4 smoke trace is
`37a78455f2e02a6d2a25bf8fe5277fe303689a4a0ce3eb095f266b33a2aad200`.
The older schema-v3 rows remain useful only for the exploratory comparisons in
this document; their reducer's combined `e0_e1a` label was an overclaim corrected
in schema v4.

## Q4 E0 Lockstep Development Sentinel

The first v1 acquisition was formally invalid because its run-common rows
omitted the schema-required executable path. The frozen reducer rejected it;
`E0-INCIDENT-001.md` preserves the error, hashes, quarantine, and zero-authority
disposition. It is not counted as a pass, mismatch, retry, or replicate.

After committing the schema-conformance repair and freezing a distinct v2
evidence artifact, one clean Q4 run at commit `df5d5d7` passed the independent
schema-v1 reducer. V2 reused the same computational inputs after v1 had exposed
an internal producer success, so it is not an untouched prospective test or an
independent replicate; prospectivity applies only to the corrected protocol,
identities, command, and decision rule. The fixture used the one-token prompt
`Hello`, sampler-v1 at temperature 0.7, top-k 200, top-p 1.0, min-p 0.05,
seed 0, one emitted token, and
`serial_then_capture` order. The reducer authenticated exactly one trace and one
run, reconstructed the full prepared distribution and RNG transition from full
logits, checked exact active KV/GDN/conv state bytes, verified poison-free finite
hidden transfer and complete active DFlash context history, and checked the
pending-token boundary plus a nonadvancing one-token continuation.

The compact reduction reports `development_lockstep_passed=true`, one passed
run, zero development failures, and zero invalid-pre-observation runs. Trace
SHA-256 is
`0a94e8d3eb6fdec24ae0ba8ec34c09c581de26fb0802c20c41bfca16e79b7e7d`;
state-sidecar SHA-256 is
`d198464ed1b46e62059ffee7e924ae279989e65202ff8dfdc3e204f304f61bd6`;
reduction SHA-256 is
`a99fa23526f78f216e2572f70f88aa7a64802db48d0caccd7ce4e43e1dc45f06`.
See `E0-Q4-V2-RESULT.md` for exact scope and artifact identities.

This passes only the development sentinel's one-emitted-token ordinary-decode
versus multi-hidden-capture check. It contains prompt and continuation
transitions but no intermediate generated target transition. It does not test
packed verification, rollback, sparse-q correctness, acceptance, economics,
performance, serving, longer contexts, reverse arm order, Q8/BF16 arms,
model-wide equivalence, or held-out E0 closure.

A separately frozen v3 sentinel required `capture_then_serial` through a
schema-v2 binding checked by both producer and reducer. Its strict reduction also
passed exactly one run with zero failures or invalid-pre-observation rows. The
v2 and v3 state sidecars are byte-identical, and no within-run parity failure was
observed under either execution order on this same fixture. V3 is an informed
order diagnostic, not an independent replicate, and these two rows do not rule
out order effects generally. See `E0-Q4-V3-REVERSE-RESULT.md`.

V4 added schema-v3 machine binding of the entire prompt/config object and
requested four emitted tokens under `serial_then_capture`. Its reduction passed
with four emissions, three intermediate generated target transitions, five
validated transitions total, and exact authenticated coverage of the expected
1,885,208,576-byte state sidecar. This buys generated-history depth on the same
already exposed `Hello` fixture, not prompt/seed diversity or independent
evidence. See `E0-Q4-V4-GENERATED-RESULT.md`.

V5 changed only the machine-bound computational order to
`capture_then_serial`. It also passed four emissions, three generated target
transitions, and five validated transitions, with a state sidecar byte-identical
to v4. Thus no within-run parity failure was observed under either order at this
generated-history depth on the same exposed fixture. This is paired development
diagnostics, not independent replication or general order invariance. See
`E0-Q4-V5-GENERATED-REVERSE-RESULT.md`.

## Development Recommendation

Keep Q4_K_M in the prospective drafter matrix. It is materially smaller, loads
faster, and has no observed systematic acceptance collapse, but it should not be
chosen as a default from these clusters.

The next go/no-go boundary is now:

1. Add a distinct development prompt at four-token depth before increasing
   length or seeds.
2. Independently pin and cross-check the actual causal sparse-q contract before
   E1b; the current selector softmax is local-reference-only. The one-hot lane
   does not depend on K0.
3. After the applicable correctness breadth, freeze separate Q4/Q8 arms and
   lane-labeled request/seed clusters. Sparse-q economics requires K0; one-hot
   economics still requires broader E0/E1a development exactness.
4. Authorize a row-stable exact verifier spike only after correctness gates and
   charged request-cluster economics pass. Until then, keep serve greedy-only.

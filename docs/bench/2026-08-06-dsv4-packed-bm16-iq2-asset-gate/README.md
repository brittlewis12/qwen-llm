# DeepSeek V4 BM16 IQ2 Current-Asset Gate

Status: completed; integration KILL.

## Result

The sole frozen execution completed in 40.71 seconds and rejected the reviewed
integration before its first BM16 packed chunk. The candidate supplied F32
gate/up views with shape `[2048, 768]` to the existing clamped-SwiGLU encoder,
whose contract requires the equivalent flat shape `[1572864]`.

This is a valid narrow KILL of this integration wiring. It is not a numerical
observation, a performance observation, or a KILL of the BM16 primitive: the
command failed before candidate execution. The model-free matrix result keeps
its existing authority to motivate a corrected integration candidate.

The frozen packet will not be retried. A successor must change the integration
by flattening the exact gate/up/output views, cover that production call shape
with a focused regression, and use the repository's reusable packed-prefill
measurement surface rather than another candidate-specific asset gate.

That corrected successor later passed an ordinary CLI pilot and earned a narrow
M4-Max/full-chunk/eligible-dtype default with an explicit rollback. This does
not alter the frozen gate's KILL: the integration changed before the successor
ran.

## Authority

The model-free BM16 packet passed all four co-primary cells, with conservative
paired savings of 239.87-532.58 ms across boundary/representative and
disjoint/warm regimes. This gate asks only whether that reviewed arithmetic and
dispatch substitution preserves the pinned current asset end to end.

A pass authorizes BM16 gate/up/SwiGLU wiring only for:

- the current Flash-0731 asset with content ID
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`;
- Apple M4 Max;
- IQ2_XS gate/up and IQ3_XXS down grouped layers;
- H=4096, F=2048, serial packed routing; and
- full N=128 chunks only.

N=12 tails remain on the deployed grouped scalar path. The gate does not
authorize IQ3 gate/up, down changes, another chunk size, quant, asset, or
device.

## Execution

One immutable residency passes through three fresh sessions:

1. deployed scalar baseline;
2. BM16 candidate; and
3. BM16 candidate repeat.

Every session independently executes the same frozen representative 0731 chat
request:

- first 128 tokens, digest
  `ee09a95c18d0231d195a88cc4d96c34ae27df5b3fd2fecb0c5e909807e4be8da`;
- scalar 12-token tail to position 140, full-prefix digest
  `f8e80c53bb46a0bbbd85f68ec470c75490bab089fe443afd8dd81f7435b85b74`;
- restore position 128 after reaching 140 and validate the rewind;
- restore position 140 and validate it independently; and
- singleton continuation token 35 to position 141.

The first-chunk baseline must reproduce route payload
`7454b2692359464c0d932e1c2fffe53d345e0ea969db9de67458ed992ddd539c`.

## Exact Gate

Absolute validation requires positions 128/140/restored-128/restored-140/141,
exact committed tokens, 129,280 logits, 4,096 normalized-hidden values,
expected available/unavailable observation states, complete unsampled 43-layer
profiles, finite route weights, exact bucket rows/slots, and complete singleton
routes.

Candidate and repeat must match in every checkpoint, route ID/weight, map,
schedule, snapshot digest, singleton route, invocation counter, and first/tail
dispatch row. Any repeat mismatch is a typed integration KILL.

Only after repeat determinism is established does the gate compare baseline and
candidate semantics:

- exact match continues to topology authorization;
- finite deterministic divergence records
  `HOLD_FOR_SEPARATELY_FROZEN_INTEGRATED_HIGHER_PRECISION_COMPARATIVE`;
- command, non-finite, memory/safety, or repeat failure records
  `KILL_BM16_CURRENT_ASSET_INTEGRATION`.

For exact arms, each of 25 deployed grouped-IQ2 dispatches must become exactly
two BM16 projections plus one unchanged SwiGLU. Every unrelated first-chunk
dispatch must remain identical. N=12 dispatch rows and serial encoder count
must remain exactly identical across all arms. Counters must be baseline
`[0,0,0]`, candidate `[25,25,25]` for BM16 and `[25,50,50]` for grouped IQ2.

Timings are diagnostic only; model-free performance is already settled.

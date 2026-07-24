# v0.623 Checkpoint Staged-Validation Contract Repair

Status: preregistered; v0.622 is sealed without authority and this repair is
unrun.

## Intent

Repair only the v0.622 runner's stale stats grammar and unexecutable major-fault
gate, then run the unchanged allocation-free staged-validation decision on four
fresh children.

v0.622 launched exactly one A child and no B child. Its runner rejected the
current stats line because `stop_reason=token_limit` now appears between
`transitions` and `load_ms`. The child therefore produced no scored row or blob
topology/hash evidence. Its observed decode-validation wall is reconnaissance
only and cannot enter v0.623 statistics or authorize any claim.

## Evidence Bridge

Pin v0.622 commit `b72844376905ca624f7443014fa1318751a086e9` and
require this packet commit to be its direct child adding only this contract and
wrapper runner. No Rust, v0.622 runner, v0.622 contract, workload, pair order,
threshold, or authority changes are allowed.

Before any child, verify every member of the complete sealed v0.622 packet and
these seals:

- decision: `5a1043125c6cb42cf1ab37376f6937b1bd79ccb594dbce5304e99894fe4f4f8f`;
- inventory: `da62de99ec511910f24216427b9de1ef9ff9c46ab3fcac0b399cd8908e8e1855`;
- completion: `5a0ccbfa1fa3f3322a17bce18241ba794c995a00f55ea3c182310539c9821657`.

Require the exact implementation-defect decision, one A launch/completion, no B
launch, clean exit, current stats mismatch, expected stdout, and sealed
post-exit evidence. Copy all v0.622 packet members into top-level v0.623 files so
the repaired packet's final inventory retains the complete bridge.

## Repairs

1. Require exactly one current stats line with
   `stop_reason=token_limit` between `transitions=0` and `load_ms`; preserve all
   v0.622 token and finite-timing checks.
2. Record `/usr/bin/time` major faults as advisory rather than a hard gate for
   every fresh v0.623 child. Preserve complete resource parsing, zero block
   input, host validity, swapout and swap-occupancy non-growth, pressure gates,
   model conditioning, non-overlap, and no retries.

The major-fault rule changes symmetrically before any B observation. It does not
declare v0.622's faults harmless or reuse its A result.

## Frozen Decision

Inherit v0.622 exactly:

- two fresh no-retry pairs in `AB BA` order;
- explicit `decode` versus `encoder-digest` arms in unique seeded stores;
- exact output, 582,854,188-byte blob, key, mode, identity-hit, marker, store,
  byte-comparison, pressure, and cleanup contracts;
- B wins isolated validation, publication, and process wall in both pairs and
  never increases peak footprint;
- GO only if process-wall saving is `>=250.0 ms` in both pairs or
  peak-footprint saving is `>=500,000,000` bytes in both pairs;
- one unscored candidate restore smoke only after the performance gate passes.

GO authority remains force-only exact allocation-free staged validation. No
authority extends to default promotion, persisted-file validation changes,
restore speed, weaker durability, write-behind, energy, concurrent publication,
other models, or compression. Any bridge, repair-scope, source, identity,
fixture, marker, output, blob, store, resource, pressure, or restore defect
stops without authority.

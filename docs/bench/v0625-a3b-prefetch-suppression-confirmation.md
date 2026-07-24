# v0.625 A3B Prefetch-Suppression Confirmation

Status: preregistration. No v0.625 child or product observation exists.

## Intent

Confirm the post-v0.624 implementation at the real disposable Runtime boundary.
This is a selector and marker packet, not a new performance comparison. It
imports v0.624's measured causal authority and asks only whether the edited
product path suppresses configured `ColdOnly` after authenticating the exact
A3B Auto direct-pread plan while preserving explicit `Always` and `Off`.

## Identity And Scope

Pin implementation parent `4f2117c39740f3b9187ae8a7b0e6749094b7393d`.
The packet commit may add only this contract and
`scripts/profile/v0625_a3b_prefetch_suppression_confirmation.py`.

Import and verify the complete sealed v0.624 GO. Use its exact model, host,
release `first_byte_spike`, disposable intent, five-token prompt, output-1
shape, and in-child `MS_SYNC | MS_INVALIDATE` cache reset. Processes never
overlap. No retries, substitutions, performance pooling, or post-observation
contract changes are permitted.

Before any GPU child, run and seal the two exact-filter CPU-only selector tests.
Both must report exactly one pass and zero failures. A selector-test failure is
a KILL; a spawn or malformed-test-evidence failure is inconclusive. An actual
fsync or evidence-publication durability failure aborts without publishing a
sealed decision because safe inconclusive publication cannot be guaranteed.

## Children

Run exactly three fresh children in fixed order:

1. `S`: configured `ColdOnly`;
2. `A`: explicit `Always`;
3. `O`: explicit `Off`.

All storage-policy environment variables remain absent. Every child must select
the promoted native embedding, exact Auto profile `a3b-q4km-v1`, direct pread
population, and copied-storage ledger.

The exact recognized load-line order is:

- `S`: native embedding -> Auto profile -> Runtime suppression -> direct pread
  -> ledger;
- `A` and `O`: native embedding -> Auto profile -> direct pread -> ledger.

`S` must report configured `ColdOnly` and explicit authenticated suppression;
the in-tree selector test separately requires empty shards and zero
prefetch-phase wall. `A` must prefetch the complete 20.61 GiB shard and must not
emit suppression. `O` must perform no prefetch and must not emit suppression.

## Per-Child Contract

Every child must satisfy all of the following:

- source, build, model, host, launch, completion, pressure, process-resource,
  and artifact evidence is complete and valid;
- pre-arm residency is exactly `1,350,985/1,350,985` pages;
- invalidation is exactly `1,350,985/1,350,985 -> 0/1,350,985`;
- post-arm residency is at least 99%;
- process-attributed physical reads are `20.50..=20.75 GiB`;
- the first token is exactly `id=11751 piece=" Paris"`;
- each required marker occurs exactly once and no unrecognized load-policy,
  storage, suppression, or ledger marker occurs.

Malformed or missing process/conditioning evidence, residency or invalidation
failure, physical-read-window failure, invalid host or pressure, nonzero exit,
or decodable-artifact failure stops inconclusive and outranks any simultaneous
product contradiction. In an otherwise complete zero-exit child, a
contradictory policy, suppression action, prefetch behavior, exact marker
set/order, or first token is a KILL, not a performance falsification. Unknown
extra product-marker families are complete contract contradictions. An actual
artifact durability failure aborts without a sealed decision.

## Authority

A complete GO certifies configured-`ColdOnly` suppression for the existing
authenticated disposable A3B Auto direct-pread path and confirms that explicit
`Always` and `Off` retain their configured behavior. Machine authority is
`authenticated-disposable-a3b-auto-pread-coldonly-always-off`. It imports, but
does not re-estimate, v0.624's measured first-byte benefit.

No authority extends to ForceOnly or reusable loads, JSONL, forced copy or
pread, overrides, other assets or hosts, serving, concurrency, energy,
partial-residency policy, untouched-media claims, or any new performance cell.

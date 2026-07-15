# v0.598 A3B Force-Only Owned-Arena Pilot

Status: **KILL**. The one-window owned path is exact and materially faster from
process start through first byte, but it reproduces the retained path's stable
decode tax and is strictly dominated by retained storage inside the measured
cache-warm fresh-process contract.

P1 stopped before product timing on a correctness-parser defect. P2 repaired only
that parser, authenticated P1, reran correctness, and completed every packet.

Frozen p2 source: `41ef09c251adecc77b52d89baa7c087463ea86e4`.
Canonical packet: `target/profiles/v0598-a3b-owned-arena-pilot-p2/`.

## Correctness

The full-state gate passes:

- exact source, physical-resource, fallback, and typed-view ledgers;
- `OwnedWeightReadOnly` provenance and write rejection;
- bit-exact packed-prefill logits and complete KV/GDN state;
- identical argmax, forced transition, continuation logits, and state.

## Fresh Results

All accepted rows have exact matching output, zero hard faults and block input,
and no pageout/swap growth. The first ABC block at each output length was rejected
and repeated because copied A incurred 5 and 2 hard faults respectively.

| Endpoint | A copied | B owned | C retained |
| --- | ---: | ---: | ---: |
| Output-1 first byte | `2463.65 ms` | `1322.05 ms` | `539.05 ms` |
| Output-128 first byte | `2463.67 ms` | `1332.40 ms` | `537.95 ms` |
| Output-128 exit | `3864.63 ms` | `2890.26 ms` | `1975.59 ms` |
| Output-128 transition | `108.18 t/s` | `93.50 t/s` | `93.63 t/s` |

B wins first byte 6/6 at both lengths. Median A/B speedup is `1.86694x` at
output 1 and `1.85162x` at output 128; both order strata and memory gates pass.
The floor's materialization mechanism survives: B model load is about
`765-769 ms` versus `2079-2082 ms` for A.

## Loaded Results

The loaded packet is stable and decisively fails performance gates:

| Late median | A copied | B owned | C retained |
| --- | ---: | ---: | ---: |
| Prefill | `309.05 ms` | `310.80 ms` | `310.35 ms` |
| Decode, 127 calls | `1204.60 ms` | `1390.20 ms` | `1390.25 ms` |
| Complete request | `1514.00 ms` | `1701.60 ms` | `1700.05 ms` |

B decode A/B time ratios are only `0.8655-0.8812`. B's complete loaded request
is `1.107-1.125x` slower in every block. All A/B repetition-5/repetition-3 and
late-range stability gates pass, so this is a kill rather than an inconclusive
result.

## Interpretation

The three-arm contrast changes the causal read:

- B is anonymous, fully copied, and fully touched.
- C is file-backed and demand mapped.
- B and C nevertheless have effectively identical loaded decode.
- A differs by using 733 dedicated offset-zero resources instead of one giant
  resource plus fallback and nonzero typed-view offsets.

This rules out file provenance, lazy source paging, and unpopulated destination
pages as primary causes of the steady tax. It implicates the shared
giant-resource/offset-binding class: resource size/count, offsets, residency or
hazard behavior, address translation, or cache behavior. The packet does not
separate those mechanisms.

v0.597 was still correct: it predicted the component it measured, and product
load fell by about 1.31 seconds. Its ready-to-bind endpoint could not observe the
repeated-GPU-access penalty introduced by topology coalescing.

## Boundary And Next Premise

Close one-window owned storage as a broad cold-plus-warm destination. In the
measured cache-warm fresh contract, B is dominated by C: it pays about 680 ms to
copy while inheriting C's decode tax. Do not generalize that dominance to
storage-cold requests.

The narrowest changed premise worth considering is topology-preserving parallel
materialization: allocate the same 733 offset-zero anonymous buffers as A, then
replace 733 fused `newBufferWithBytes` calls with parent allocation plus four
workers copying disjoint destination buffers. Gate loaded parity before another
fresh-process packet.

No result authorizes default-on owned storage, aliases, conversions, MTP, split
shards, other assets, asynchronous promotion, or broad loader integration.

Adversarial design, repair, and result review: `cx ask` session
`019f61c6-cb2e-7cc3-8e34-5011e456fe6d`.

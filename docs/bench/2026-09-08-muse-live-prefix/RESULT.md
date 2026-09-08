# Muse live prefix reuse: repeated-turn backend PASS

Opt-in `1cdab794` reuses the current resident sequence's exact consumed-token
prefix without allocating or copying a snapshot. The 6269-token followup takes
216.803237 s without reuse versus 3.827429 s with reuse: 98.235% less wall time,
56.645x faster, including 16 generated tokens. This does not speed fresh prefill,
scalar decode, HTTP transport, or a new CLI process.

## Mechanism and correctness

Muse retains full absolute-position K/V for every layer, including sliding
attention, and has no recurrent-state rollback requirement. The backend tracks
only consumed token IDs, matches the new prompt exactly, and rewinds the position
to at most `prompt_len - 1`. A nonempty suffix always regenerates current logits.
The final sampled token is not counted as consumed unless actually transitioned.

Admission rejection preserves the old history. History clears before state
mutation and publishes only after successful generation with a matching frontier.
Clean prefill/decode cancellation leaves the next request cold. GPU-command poison
remains fail-stop; this change does not clear poison or claim driver recovery.
There is one backend-local history, not a concurrent multi-request cache.

- Real-model core test passes exact logits and every active K/V byte through
  identical, shorter, singleton, unrelated, branched and extended histories, with
  irregular 16/128-row boundaries and a continuation. Future rewind and poisoned
  rewind/reset reject. Runtime: 31.29 s.
- Actual backend checks exact-hit emission parity, capacity rejection preserving
  history, prefill cancellation, second-emission disconnect and subsequent cold
  recovery. Long A/B arms agree on emitted bytes and all consumed IDs.
- A separate short sampled retry test (`93151c9c`) passes temperatures 1.0/0.7
  with seeds 42/99 against reset. This is fixed-seed parity, not a general
  stochastic-distribution proof. Runtime: 7.23 s.

## Frozen packet

The checked-in Current system text is rendered natively through ATEM, high effort.
The followup includes an authored short assistant reply and Mara's request to find
warmth/power. It is not the user's captured Mara session. Token SHA-256:
`bbf0248261a24f86134c70ac720ad9fdecc7894aa0e1e8ed4a303a15c4e341a2`.
Capacity 8192, temperature 0, seed 42, output limit 16.

Complete A/B oracle requests condition both paths before measured A-B-B-A. Every
timed second request follows the same first request, outside the interval. Both
arms retain history bookkeeping; A disables reuse. Timing includes tokenization,
runner creation, rewind/prefill, generation and history publication, but not
model loading, prompt rendering, priming or HTTP. No interposed K/V readbacks.

| A1 ms | B1 ms | B2 ms | A2 ms |
| ---: | ---: | ---: | ---: |
| 216806.702667 | 3827.469708 | 3827.387875 | 216799.771583 |

Both pairs save 98.235%, exceeding the frozen 50% gate. Control spread is
0.003197%, below 5%. B reuses 6230/6269 prompt tokens and computes 39. All arms
produce 16 tokens, ending with 6284 consumed tokens. Original A/B oracles take
218421.563/3830.152 ms and remain in the raw record. One packet, no rescue rerun.

Observed model/session allocations stay 29,599,907,840 /480,280,576 B. No new
K/V buffer or snapshot is retained; CPU token-history Vec storage is additional
and is not included in those Metal counts. These are not RSS measurements.

The backend test owns the ordinary production Metal lease; the library unit
test uses its external guardian. No lease bypass, wiring, or parallel inference.
Global compression/pageout/swapout growth is zero. Backend-wide decompressions
increase 12636 and swapins 3827; core increases 3/4. These are not phase-local
I/O attribution, and do not establish a host-quiet or cold-load contract.

## Disposition

Keep `QWEN_MUSE_PREFIX_REUSE=1` as an explicit resident-serve opt-in; default stays
off. No durable cache, automatic policy flip, model arithmetic, sampler, ATEM or
reasoning-tier change. Results favor this repeated-turn mechanism, but the active
priority is now fresh matrix prefill and the measured decode attention bottleneck.

Raw source identities, builds, oracles, ABBA, host counters and scoring:
`target/profiles/muse-live-prefix/`. Independent reviews validate the ownership
and measurement design; final runtime and production compilation checks are
recorded alongside the packet.

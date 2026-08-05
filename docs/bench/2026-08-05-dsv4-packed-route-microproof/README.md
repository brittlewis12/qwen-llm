# DeepSeek V4 Exact Packed Route-and-Schedule Microproof

Status: test-only topology `GO`. This authorizes an off-by-default production
route/schedule seam. Production packed prefill, grouped expert compute, default
routing, and whole-request speed claims remain `HOLD`.

## Hypothesis

The attribution packet found only 30 ms of host route/schedule work in a
140-token request, so GPU routing is not itself a TTFT win. It is still the
dependency for removing the readback boundary and giving future grouped expert
consumers deterministic GPU-owned work. Prove that dependency without changing
production prefill or building another expert backend.

## Topology

For each packed layer:

1. Learned routing launches one 256-thread group per token and copies the
   deployed singleton reduction lineage. Hash routing launches one thread per
   token and reads token IDs plus the canonical map directly.
2. One thread per expert scans tokens and original slots in ascending order. It
   writes `count[256]` and fixed-stride `slot_id[256][N]`; padding is `-1`.
3. A 256-thread validator checks route and schedule generation, producer status,
   unique in-range IDs, finite nonnegative weights, exact expert/token/slot
   order, counts, padding, and total assignments. It publishes one aggregate
   generation/status/total/completion record.
4. A separate 256-thread signature consumer requires that aggregate and hashes
   every ID bit, weight bit, count, occupied slot, and padding sentinel before
   the host may observe the result.

Every producer publishes the current nonzero generation only after its payload.
The Rust owner fails before wrap or reuse. All encoders reject concurrent Metal
passes because dispatch order is part of the authority contract.

The scheduler increments the complete count for malformed routes but writes a
slot only while `count < N`. A terminal N=128 fault routes all 768 assignments
to expert 255, publishes count 768 and `INVALID_COUNT`, and leaves 64-byte
prefix/suffix guards intact.

## Exactness

- Learned and hash routes match the CPU packed oracle for every N=1..128:
  expert IDs and schedules are exact; weights remain within the existing
  `1e-4` CPU/GPU numerical envelope.
- Packed GPU IDs, statuses, and every weight bit match the deployed singleton
  Metal kernels on all-tied corrected scores, a one-ULP cutoff, inputs directly
  around both softplus branches at -20/+20, and finite extremes.
- Repeated GPU IDs, weight bits, counts, slots, and payload signature are exact.
- The signature is independently transcribed in Rust and compared exactly for
  every N and route kind. Separate sensitivity checks mutate one ID, weight,
  count, occupied slot, or padding sentinel and change the hash.

Faults pin every aggregate and signature field for omitted token/expert
producers, omitted validator with stale aggregate, nonfinite learned logit/bias,
invalid hash token/expert/duplicate/nonfinite logit, invalid or duplicate IDs,
nonfinite weight, corrupt count/slot/padding, generation exhaustion, concurrent
encoding, and terminal bucket overflow pressure. Ordinary duplicate IDs publish
`DUPLICATE_ID`; a duplicate route that also overflows the terminal bucket
publishes the higher structural `INVALID_COUNT` precedence.

## Timing Gate

The release profiler models a 43-layer packet as three hash and forty learned
routes. Each cell receives at least one second of untimed stabilization, then
stage probes, 32 more complete warm packets, and 12/40/12 complete packet
samples. There is no timed allocation or readback. The middle p95 must remain
below 4.3 ms at N=12 and 8.6 ms at N=128; both control medians must drift no more
than 5% in GPU and wall time.

| N | Route GPU median | +schedule | +validator | +signature | Packet GPU p95 | Packet wall p95 | Max drift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.251 ms | 0.526 ms | 0.983 ms | 1.280 ms | 1.314 ms | 1.600 ms | 0.999% |
| 128 | 0.303 ms | 2.998 ms | 6.350 ms | 6.959 ms | 7.064 ms | 7.273 ms | 0.561% |

The first signature design was a deliberate KILL: one GPU thread hashed the
entire schedule per layer and drove the N=12 packet to about 10.15 ms GPU p95,
well beyond the 4.3 ms gate. The retained design partitions work deterministically
across 256 threads, then combines 256 partial hashes in fixed order.

Final raw timing log SHA-256:
`9ac99b41c0ec071e864075bc1cf4912109f89e5559f19d5ca83231b149d0ffd3`.
Reviewed source SHA-256:

- `prefill.rs`: `a6144637e083863f7649ec4e549d814acc6468e7dbdba301a7363bf2392dfd1d`
- `deepseek_v4.metal`: `ec89e56d10dbb27d08cc7bd0b5a8debec3cb412b6ea87a55b73736a82e6ef16d`

## Decision

Promote the exact route/schedule topology, not a production switch or a speedup
claim. The next seam must own admitted packed scratch and generation state, run
off by default, leave current CPU grouped experts intact, and prove integrated
current/candidate/current output and state. Grouped IQ2_XS gate/up plus IQ3_XXS
down remains the separately gated performance experiment after that seam.

CX review session `019fcf7d-e9d4-7150-b496-e70a31958e80` returned final `GO`
with no blockers.

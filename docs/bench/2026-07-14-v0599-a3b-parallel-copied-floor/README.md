# v0.599 A3B Topology-Preserving Parallel-Copy Floor

Status: **GO**. The composite parallel-copy primitive preserves the production
733-resource, offset-zero topology while retaining nearly all of v0.597's cold
materialization saving.

Frozen source: `5548ea176739a1106627d5565edb340ee0c4e2e4`.
Canonical packet:
`target/profiles/v0599-a3b-parallel-copied-floor-p1/`.

## Contract

The two arms materialize the same authenticated A3B inventory:

- A resolves and copies each tensor through production `newBufferWithBytes`.
- B allocates the same 733 exact-sized shared buffers in production request order,
  resolves all source/destination spans on the parent, and copies four contiguous
  source ranges through scoped workers.

Both arms retain 733 unique resources, 733 offset-zero bindings, and exactly
`22,123,538,944` physical bytes. B's frozen worker byte loads are
`5,532,746,240 / 5,462,315,776 / 5,595,522,304 / 5,532,954,624`.

## Validity And Correctness

- Clean source/build/runtime identity and all file/model hashes pass.
- All six fixed-order pairs pass on attempt one; no retry or selection occurs.
- Every child has valid AC/thermal/memory state, zero timer-local major faults,
  zero block input, and zero pageout/swap growth.
- Every arm passes exact bytes for all 733 resources, exact resource/binding
  identity, exact lengths, offset zero, and Shared/DefaultCache/Tracked modes.
- Candidate timing components reconcile to ready wall within floating-point noise.

## Result

| Metric | A copied | B parallel copied |
| --- | ---: | ---: |
| Median ready wall | `2066.701 ms` | `742.676 ms` |
| Median ready throughput | `10.705 GB/s` | `29.790 GB/s` |
| Median copy throughput | fused | `29.882 GB/s` |

Paired results:

- median saving: `1324.148 ms`;
- median B/A ratio: `0.358177x`;
- wins: 6/6;
- AB/BA median savings: `1319.864 / 1328.433 ms`, both 3/3 wins;
- maximum paired RSS/footprint ratios: `1.000014 / 1.000029`;
- transfer-haircut first-byte projection: `2.06601x`.

Every preregistered gate passes by a wide margin.

## Diagnosis

The control replicates across packets: v0.597 A was `2062.791 ms`, only 0.19%
from v0.599 A. The exact-topology candidate is only `40.509 ms` slower than
v0.597's one-window four-worker floor and retains about 97% of its saving.

The combined v0.597-v0.599 evidence separates the terms:

1. Four-worker composite population supplies the cold materialization saving.
2. One-window/nonzero-offset topology supplies v0.598's stable loaded decode tax.
3. Resource coalescing is not required for fast cold materialization.
4. Production copied topology can retain nearly all of the cold saving.

The packet does not isolate concurrency from batched allocation or manual copy.
It also does not prove loader/full-state correctness, loaded parity, actual model
load or first byte, storage-cold behavior, or cross-asset breadth. The `2.06601x`
first-byte figure is an arithmetic projection only.

## Authority

The result authorizes one separately preregistered, force-only A3B
parallel-copied loader pilot using this exact topology and worker schedule. Per the
frozen sequence, finish query-capped auto-prefill first. The loader pilot must
prove bit-exact full state and loaded prefill/decode/request noninferiority before
fresh product timing.

No authority follows for default-on behavior, other assets, one-window owned
storage, asynchronous promotion, memory reduction, or broad loader integration.

Adversarial design, implementation, and result certification: `cx ask` session
`019f633c-62be-7820-9dfa-256d733a58ef`.

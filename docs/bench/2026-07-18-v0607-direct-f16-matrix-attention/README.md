# v0.607 Direct-F16 Matrix Attention Falsifier

Status: **KILL**. The fixed direct-F16 body is numerically excellent but reaches
`0.309084 ms` main against a valid `0.164125/0.164209 ms` A1/A2. It is
`2.09247x` its authorization gate. No 131K row or product path is authorized.

## Admission Correction

The exploratory split score-ledger body is rejected before implementation.
Production group-8 tile4 logically reads K/V twice, but the observed wall makes
two external reads impossible:

| Context | Unique F16 K+V | V4 main | Unique-byte rate |
| ---: | ---: | ---: | ---: |
| 32K | 64 MiB | `~0.163 ms` | `~412 GB/s` |
| 131K | 256 MiB | `~0.6842 ms` | `~392 GB/s` |

Counting both sibling reads as external would require about `823/785 GB/s`.
The second read is therefore predominantly cache-served. An F32 ledger adds a
2/8 MiB allocation and at least one charged traversal. Even granting a cached
consumer, its 474 GB/s zero-compute estimates are `0.146004/0.584017 ms` at
32K/131K. The 10% and 15% gates are `~0.146513/~0.581570 ms`. There is no room
for QK, softmax, PV, stores, or the second dispatch, so the split ledger is a
design-level **KILL**.

This is an admission decision from absent charged margin, not a claim that both
ledger traversals must reach DRAM.

## Fixed Candidate

The surviving candidate tests a different measured premise from the removed
v0.577 scalar and v0.583 compressed bodies:

- exact shape: group 8, head dimension 256, F16 KV, C32, NWG256;
- grid: `[2 KV heads, 1, 256 partitions]`, 256 threads per threadgroup;
- SG0-SG3 each compute all eight Q heads against one disjoint eight-position K
  panel with direct transposed device-half `simdgroup_load` and F32 MMA;
- SG0-SG7 each own one 32-dimension V slice and accumulate all eight heads;
- SG `h` owns online-softmax state `(m_h,l_h)` for query head `h`;
- output uses the existing V4 G8 partial and H2 reducer ABI.

The body has no Q8 decode, split-plane addressing, K/V staging, device score
ledger, or scalar cross-simdgroup QK reduction. Dynamic threadgroup memory is
exactly 5,152 bytes: 4,096 bytes for row-major scaled-half Q, 1,024 bytes for
the `[8,32]` F32 score/weight tile, and 32 bytes for rescaling factors. Q cannot
be aliased because every C32 tile reuses it.

Each C32 tile has three threadgroup barriers: after QK score publication, after
softmax weight/factor publication, and after PV consumption. Matrix fragments
must end lexical lifetime before softmax/PV. Persistent per-lane state is eight
F32 output accumulators plus one owned `(m,l)` pair.

The exact 32K/131K fixtures have no partial partition. Any later product body
must mask partial score columns, avoid out-of-allocation matrix loads, and make
all eight simdgroups cross every barrier.

## Why This Point Is Not Already Measured

v0.577 distributed V ownership but paid eight scalar QK partial reductions and
threadgroup exchange. v0.583 used matrix QK, but combined it with group16 Q8
decode, split-plane addressing, 16 KiB K/V staging, serialized half-threadgroup
phases, and the resulting integer/conditional limiter signature at only
`140 GB/s`. This candidate keeps distributed ownership and matrix QK while
removing those measured pressure sources.

The prior failures remain strong negative evidence. This is one fixed
falsifier, not authorization for a matrix-attention family.

## Packet

1. Compile the fixed PSO and require `maxTotalThreadsPerThreadgroup >= 256`.
2. Use the immutable v0.575 block-3 32K tail query and F16 K/V.
3. Run the complete candidate main: direct MMA QK, online softmax, direct V
   accumulation, V4 partial stores, and unchanged H2 reduction.
4. Measure nine samples each in A1/P/A2 order for main and main-plus-reducer,
   with the existing common ramp, compute scrub, cooldown, identity, hash,
   NaN-primed partial, and independent attention-oracle checks.

No selector, cache writer, product packet, C/NWG variant, Q8 arm, limiter
capture, or rescue retune is part of the 32K decision.

## Gates

- Packet validity: clean source/build/runtime identity, invariant input/output
  hashes, complete partial writes, numerical oracle passage, and A1/A2 spreads
  no greater than `1.01x` for both main and pair.
- 32K authorization: candidate main and main-plus-reducer must each be no more
  than `0.90 * min(A1,A2)`. The v0.583 reference values are approximately
  `0.146513 ms` main and `0.18975 ms` pair.
- Any 32K charged miss closes this point. Effects below 2% are unresolved, not
  directional authorization.
- Only a complete 32K pass authorizes the identical 131K body.
- 131K promotion: main no more than `0.85 * min(A1,A2)` and full-token GPU wall
  at least 6% faster, followed by a fixed 32K repeat with no regression beyond
  1%.
- Any 131K main or full-token miss closes the point without nearby variants.

## Result

The first clean packet had valid correctness but exceeded the 1% anchor-spread
limit: main drift was `1.01933x` and pair drift was `1.01183x`. It is retained
as invalid evidence. One unchanged rerun of the same clean binary, arguments,
capture, and row order produced valid main/pair spreads of
`1.000511/1.000804x`.

| Row | Main | Main + reducer |
| --- | ---: | ---: |
| V4 A1 | `0.164125 ms` | `0.208626 ms` |
| Direct-F16 matrix | `0.309084 ms` | `0.356583 ms` |
| V4 A2 | `0.164209 ms` | `0.208458 ms` |
| Authorization gate | `0.147713 ms` | `0.187612 ms` |

The candidate is `1.88322x` slower than the faster main anchor and `1.71057x`
slower than the faster pair anchor. Relative to its gates, the misses are
`2.09247x` and `1.90064x`. This is not an MDE or run-order decision.

Correctness passes strongly:

- candidate versus V4 cosine is `0.999999999999285`, maximum absolute error is
  `4.411e-6`, and relative L2 is `1.234e-6`;
- every NaN-primed output and softmax-state slot becomes finite, with positive
  `l`;
- input and per-row output hashes remain invariant.

## Diagnosis

Removing group16 Q8 decode, split-plane addressing, 16 KiB K/V staging, and
most code-consistent integer pressure moves the v0.583 matrix body only
`0.313416 -> 0.309084 ms` (`1.01402x`). Those features were correlated with its
limiter signature but were not the dominant integration residual.

Matrix QK does remove meaningful work: v0.607 is `1.66190x` faster than the
v0.577 scalar-F16 read-once body before cross-packet anchor normalization. It
still loses badly to V4. The common losing structure is organization-level:
one 256-thread/eight-simdgroup cooperative group integrates QK, softmax, and
distributed-V PV under repeated threadgroup rendezvous. Likely contributors
include half-idle C32 QK phases, 13 barriers per partition, persistent
distributed accumulators, shuffle-heavy weight distribution, strided matrix
loads, and lost independent one-simdgroup scheduling. The packet does not
isolate one sole cause.

V4's sibling tile4 rereads are predominantly cache-served: charging both as
external bytes would require roughly `818-823 GB/s`. Read-once ownership removes
logical reads but not an equivalent external-byte floor.

C64 is not a rescue. It could activate all eight QK simdgroups and halve tile
barriers, but retains the same cooperative ownership and cannot exceed a
generous `2x` ceiling from those named changes. Even `2x` gives `0.154542 ms`,
still 4.6% slower than the gate; the body needs `2.09247x`. The split QK/PV
score-ledger body remains killed by the admission arithmetic above.

Close this fixed ownership point without C/NWG/thread-count, barrier, staging,
format, or split-ledger variants. Reopen true-long attention only for a
pre-costed body that changes ownership, scheduling, residency, or physical
bytes and has a charged ceiling inside both primitive and whole-token gates.

Artifacts:
`target/profiles/v0607-direct-f16-matrix-attention/` (`canonical.out` and
`canonical-rerun.out`).

Adversarial design review used `cx ask` session
`019f77d6-9613-7b63-b7e8-b87488a59255`. Result review used session
`019f77fc-0dae-7d11-8a33-9158769ba266`.

# v0.607 Direct-F16 Matrix Attention Falsifier

Status: **PREREGISTERED**. Run one fixed 32K integrated body only. The packet
must either authorize the identical 131K body or close this direct-F16 matrix
point without a shape, compression, staging, or thread-count rescue.

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

Adversarial design review used `cx ask` session
`019f77d6-9613-7b63-b7e8-b87488a59255`.

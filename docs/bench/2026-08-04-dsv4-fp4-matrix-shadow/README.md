# DeepSeek V4 FP4 Lightning Matrix Shadow

Date: 2026-08-04

Status: packed-semantic schedule `GO`; production/cache/snapshot `HOLD`.

## Scope

This checkpoint asks whether the official BF16-input E2M1/UE8M0 indexer
contract fits inside the previously established Metal matrix-schedule budget.
It is a test-only shadow over deterministic synthetic post-Hadamard operands.

The shadow uses separate raw-byte value and scale planes:

```text
Q values  I8-as-uchar [64, 64, query]
Q scales  I8-as-uchar [4, 64, query]
K values  I8-as-uchar [64, row]
K scales  I8-as-uchar [4, row]
```

The repo-local 68-byte row is reconstructed only in tests. These planes do not
define the eventual paged K layout, cache ABI, or snapshot format.

## Identity

- Base revision: `22bb369d081e9952f8f9fd8af9a68f684aa4de5f`.
- Candidate source, packet, and raw logs are captured by the same checkpoint
  commit.
- Hardware: MacBook Pro `Mac16,5`, Apple M4 Max, 128 GB unified memory.
- OS: macOS 15.6.1 (24G90), arm64.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)` and
  `cargo 1.97.1 (c980f4866 2026-06-30)`.
- Metal: Apple metal 32023.864 (`metalfe-32023.864`), target
  `air64-apple-darwin24.6.0`.
- Scalar fixture SHA-256:
  `0e5e2b251a960d417e7977608a363b83e072e2d90bc286cc52820b0ea7dc2b1f`.

No model weights are used. Q, K, and weights are deterministic
production-shape synthetic values.

## Schedule

The pack kernel assigns one 32-value block to each of four simdgroups. It rounds
all source values to BF16 by bits, reduces block amax, emits RNE E2M1 nibbles and
one UE8M0 code, checks decoded-F32 overflow, and publishes both planes only after
the complete row validates. A one-row pack is therefore transactional.

Packed Q is authoritative, but decoding it independently in every eight-row K
tile proved wasteful. The admitted schedule expands Q once into a transient F16
unit-value slab `[128,64,query]`, exactly 16 KiB/query. The slab contains only
the E2M1 unit values, including signed zero; Q scales remain in the packed plane
and are applied per block.

Eight scorer simdgroups cover 64 heads x eight K rows. Each of four 32-wide
blocks uses four 8-wide MMAs. K remains packed and is decoded in the row scan.
For each head/row cell, the block dot is rescaled once with
`ldexp(dot, q_code + k_code - 254)`, avoiding a false intermediate scale
overflow. Blocks accumulate in order, followed by ReLU-before-weight and a
head-0-through-63 reduction under safe Metal math. Query reuse reduces dynamic
threadgroup memory from 6,656 to 2,560 bytes.

## Frozen Gates

CX session `019fcf01-f2d0-7043-bc7c-a139ab850de3` froze the gates before
implementation:

- Exact 42 E2M1 and 13 scale primitives, five valid rows, one upper-domain pack
  rejection, and all 20 malformed packed rows.
- Rejected rows retain sentinel value/scale bytes. Status `1` means source or
  BF16 nonfinite, `2` noncanonical scale, and `3` decoded-F32 overflow.
- Exact Q unit expansion for all 16 codes, both nibble positions, signed zero,
  nonzero offsets, dispatch tails, 128 queries, and repeated runs.
- All three fixture score vectors and top-2 decisions bit-exact. General visible
  rows require scaled max error at most `2e-5` and relative RMS at most `2e-5`;
  masks, IDs, counts, visibility, ties, and statuses remain exact.
- One timed command per FP4 sample contains exactly Q pack, Q unpack, and score.
  K is prepacked. Five alternating warm calls precede 20 retained samples for
  current/FP4/current. Control-median drift must be at most 5%.
- At 16,384 rows, candidate regression is at most 0.05 ms/layer. At 65,536,
  saving is at least 0.15 ms/layer. At 262,144, saving is at least 0.80
  ms/layer, candidate median is at most 1.30 ms, and candidate p95 is below both
  current medians.
- Incremental one-row K pack median is at most 0.020 ms and p95 at most 0.030
  ms.

## Falsification History

The first scalar one-thread pack measured 0.130583 ms for one row and was
replaced rather than weakening the incremental gate. The four-simdgroup pack
retains exact bytes and reduces that operation to about 0.013 ms.

The first complete matrix schedule decoded all 64 Q heads in every eight-row K
tile. At 262,144 rows it measured current/FP4/current
`2.127 / 1.634 / 2.125` ms, only 0.492 ms saving, and failed both the 0.80 ms
saving and 1.30 ms candidate gates. That schedule is `KILL`.

Decoding packed Q once is the admitted structural work reduction. It preserves
Q QAT and all scale semantics while avoiding roughly 268 million redundant unit
decodes at terminal geometry.

## Performance

All numbers are Metal command-GPU milliseconds per layer. The FP4 arm includes
one Q pack, one Q unpack, and scoring.

| Rows | Token equivalent | Campaign | Current before | FP4 | Current after | Drift | Saving | FP4 p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 65,536 | A | 0.663 | 0.280 | 0.665 | 0.31% | 0.383 | 0.281 |
| 65,536 | 262,144 | A | 0.810 | 0.324 | 0.804 | 0.71% | 0.483 | 0.325 |
| 262,144 | 1,048,576 | A | 2.127 | 0.841 | 2.126 | 0.04% | 1.285 | 0.842 |
| 16,384 | 65,536 | B | 0.666 | 0.280 | 0.664 | 0.23% | 0.385 | 0.282 |
| 65,536 | 262,144 | B | 0.805 | 0.324 | 0.802 | 0.35% | 0.480 | 0.324 |
| 262,144 | 1,048,576 | B | 2.127 | 0.841 | 2.124 | 0.15% | 1.284 | 0.842 |

All control-median drift remains below 0.8%. Terminal geometry is stable across
campaigns at 0.841 ms candidate and 1.284-1.285 ms saving.

Incremental one-row K packing measures:

| Campaign | Median | p95 |
|---|---:|---:|
| A | 0.013375 | 0.013542 |
| B | 0.013667 | 0.013958 |

Raw samples are retained in `campaign-a.log` and `campaign-b.log` beside this
README.

Exact performance command, run twice:

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_lightning_fp4_matrix_shadow_at_far_context \
  -- --ignored --exact --nocapture
```

Correctness command:

```bash
cargo test --release -p qwen-llm --lib fp4_ -- --nocapture
```

Fourteen active FP4 tests pass; the ignored profiler is the fifteenth match.
The general packed-score differential observes zero error on its deterministic
128-query corpus, stronger than the frozen envelope. Formatting, patch
whitespace, and strict release all-target/all-feature Clippy also pass.

## Decision

The packed-semantic matrix schedule is `GO` as a test-only shadow. It cuts the
terminal score operation by about 60% and leaves roughly 1.284 ms/layer of
measured whole-token opportunity before selector and downstream work.

Production remains `HOLD`. Synthetic K is prepacked, the scorer trusts validated
status-0 Q/K planes, and neither it nor the query slab consumes a production
validity record. Existing F16 histories cannot be reconstructed under the
pre-F16 BF16-input contract. Paged K layout, snapshot v2, restart compatibility,
real-weight decision/quality evidence, and whole-token performance remain
separate promotion gates.

# DeepSeek V4 Packed Grouped Experts

Status: `GO` as the qualified Apple M4 Max default for the current
IQ2_XS/IQ2_XS/IQ3_XXS routed-expert triple. Other devices retain the existing
per-bucket path unless explicitly opted in. Rust routing remains authoritative.

## Question

The packed path already builds an exact stable expert-major schedule, but then
executes about six Metal dispatches for every active expert bucket. This packet
asks whether the 25 current-asset layers with IQ2_XS routed gate/up and IQ3_XXS
routed down can consume that schedule in two dispatches per layer while
preserving all observed output and causal-state contracts. The preregistered
promotion gate is at least 15% aggregate GPU and wall saving at N=128 before
widening to another dtype triple.

## Candidate

The host validates exact expert/token/original-slot order and complete `6N`
coverage before building compact 32-assignment tiles. A legal N<=128 schedule
has no more than 272 `{expert,start,count}` records, or 3,264 inline bytes under
Metal's 4 KiB `setBytes` limit.

Two kernels replace the eligible per-bucket chain:

1. One 32-lane simdgroup per tile and FFN output row computes IQ2_XS gate and
   up projections, preserves ordered-F32 accumulation and the deployed clamp +
   SwiGLU lineage, and writes slot-major inner values.
2. Four simdgroups per tile and 64 output rows preserve the existing IQ3_XXS
   64x32x32 SIMD-matrix lineage, synchronize before reusing 8 KiB of shared
   memory, and scatter each complete result to its unique route slot.

The 25 eligible layer invocations are observed directly. The reported 50
dispatches are the structural two-dispatch consequence, not an independent GPU
ledger. The other 18 layers retain the unchanged per-bucket compute path.

`QWEN_DSV4_PACKED_GROUPED_EXPERTS` is an isolated tri-state control:

- unset or `auto`: enable only on the qualified `Apple M4 Max`;
- true: opt in on another device only when both pipelines, SIMD width, thread
  limits, 8 KiB threadgroup memory, and the existing IQ3 MM policy qualify;
- false or malformed: retain the previous per-bucket path without changing
  the IQ3 MM policy.

The candidate's 6,291,456-byte F32 inner scratch is always admitted and
realized, including under rollback. This changes ordinary sessions from 542 to
543 allocations. At the 3,073-forward fixture, session logical/priced bytes
move from 179,129,572/183,681,024 to 185,421,028/189,972,480.

## Fixture

- Base revision: `b922c7fad73a2f60feb90933086d92a80942ea43`
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Asset size: 104,207,848,032 bytes across four pinned shards
- Prefix pattern: exact token IDs `[35, 201, 200, 34]`
- Prefix sizes: N=12, 32, and 128 from a fresh session
- Continuation: canonical snapshot restore, then exact token ID 35
- Session limit: 129 forwards for the N=128 case

The asset hashes and role-level dtype census are pinned in
`docs/DEEPSEEK-V4-STRATEGY.md` and its current census fixture.

Command shapes:

```text
cargo test --release -p qwen-llm --lib packed_grouped_

QWEN_DSV4_PACKED_GROUPED_EXPERTS=auto \
QWEN_MATMAT_IQ3_XXS_MM=1 \
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_grouped_expert_integration_packet \
  -- --ignored --exact --nocapture

QWEN_DSV4_PACKED_GROUPED_EXPERTS=1 \
QWEN_DSV4_PACKED_GROUPED_SAMPLES=5 \
QWEN_DSV4_PACKED_GROUPED_ONLY_N=128 \
QWEN_MATMAT_IQ3_XXS_MM=1 \
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_grouped_expert_integration_packet \
  -- --ignored --exact --nocapture
```

## Correctness

The active model-free differential covers N=1/12/31/32/33/64/128, hot and
sparse experts, expert IDs 0 and 255, both tile edges, deterministic nonzero
quant blocks, repeat identity, and 64-byte output guards. Slot-major inner and
final expert-output bits match the old gather/gate/up/SwiGLU/down/scatter path.

The current-asset ordinary-path integration preserves bit-exact logits,
normalized hidden output, and restored continuation logits. Causal, prefix,
compatibility, and continuation-causal digests and committed tokens also match
both forced-current controls.

| N | Logit SHA-256 | Causal digest | Continuation causal digest |
|---:|---|---|---|
| 12 | `6c5dbb88a1767fb98fd3e5eb976b310c6f97a9732f1ffabedfb2a9e82a673ed2` | `37dcd04a641cb0cd3763adc7b2267e1423b0ec6ec39cc965c30fddbf16888a9b` | `42c56a22e6887a4818cbc9bfa0a74e5593211922a6a096c84c96037ea5a49b5c` |
| 32 | `7c072707db22b5202757a8fffd3d8cf056564b62620d6b793db916258074a7bb` | `ef14d3f12c306e2a154f48fcb9d10e8b26d0b3e0be904f5a6bb09f43895da532` | `a857a160e70ffe0a9b54505dd021d7a2658da1c80e2ebfd662042c76fdf072b4` |
| 128 | `45c414408cdc688eeeb4f40166013b2ac7261e5a2c2655cdf9f51e818941f2d5` | `3de61f2054ff7e540be5beb113c5c3884875b7018de800696aa58a0393535035` | `7bc950b04b735a172ccb3c5b467488f548407b8b38f497b304d5d4bef7ae0dd8` |

The isolated rollback run reports `grouped_enabled=false`, zero grouped
invocations, zero calculated grouped dispatches, and the same N=12 output and
state identities.

## Performance

The final-source five-sample N=128 adjudication records these complete wall
arrays in milliseconds:

```text
current before: [3614.866, 3799.251, 3500.068, 3617.414, 3679.268]
candidate:      [2721.159, 2246.024, 2536.841, 2240.504, 2517.379]
current after:  [3757.453, 3667.750, 3757.012, 3729.075, 3705.767]
```

Medians are 3,617.414 / 2,517.379 / 3,729.075 ms. The candidate saves 30.409%
against the faster control median with 3.040% control drift. Every candidate
sample is below every control sample.

Wall timing is nevertheless noisy. One final-source three-sample bracket
failed the frozen gate at 12.2%; its arrays were lost because the harness still
emitted them after the assertion. A later single exactness bracket measured
only 11.526% at N=128. The bounded R5 adjudication establishes a strong median
benefit on this fixture, not a per-run 15% guarantee or a broader-device claim.

The earlier traced R3 bracket isolates the structural GPU effect:

| Metric | Current before | Candidate | Current after | Saving vs faster control |
|---|---:|---:|---:|---:|
| Post-route GPU median | 1,655 ms | 1,060 ms | 1,654 ms | 35.91% |
| Summed packed-command GPU median | 2,763 ms | 2,151 ms | 2,745 ms | 21.64% |
| Instrumented wall median | 3,711.344 ms | 3,028.644 ms | 3,576.428 ms | 15.316% |

The trace predates final policy-only qualification and rollback hardening; the
candidate kernels and arithmetic path are unchanged. Treat its GPU numbers as
attribution and its wall number as instrumented support, not the primary wall
claim.

## Decision

Enable this exact dtype triple automatically on Apple M4 Max. Preserve CPU
routing, schedule validation, the forced-current differential, capability
fallback, and the isolated rollback. Do not generalize the result to another
device or dtype from this packet.

The preregistered widening condition now clears. The next independent
falsifier should target the 16 IQ3_XXS/IQ3_XXS/IQ3_XXS layers; the two MXFP4
down layers remain outside this promotion. GPU route ownership remains KILL
under its current weight arithmetic and is not reopened by this result.

## Evidence

Final reviewed source SHA-256:

- `deepseek_v4_metal.rs`:
  `dfe9ffda9a5f4d56b17998e16e3a3912b0e78d5ac03674345c6f594dc6e58115`
- `prefill.rs`:
  `b6cce60e37382b77a2989fb6df26c5b9f62d24c251c498ab84c2b867cb1f1935`
- `metal.rs`:
  `38dea5b6f10e58ee2d70f02ad0c911a030d46f256eaa94bd89fd384f68fe1298`
- `deepseek_v4_position_zero_live.rs`:
  `1e3b1ccdc75f7a459c946fb898dc9968d8aa65a495d3ea002198d15fda4dd253`
- `mat_mat_iq3.metal`:
  `2a3db5141dc4ced26d1c2e2b88b192e2817fb514dd4f2c141bf284eddc4b8013`
- `mat_vec.metal`:
  `17abcbfcca497e4ba65b93e1abf7f75bafa0a2f87492b318015e400ccf69a0d6`

Final release diagnostic test executable SHA-256:
`0168a7a7ff3b0eb67f85e530a29aebf3f20d96365926457fb8e093a6293eba7c`.

Local raw logs and SHA-256:

- Final default exactness:
  `integration-promoted-default.log`,
  `ee29eec93e5836155aef642347ae805ba59eaea86bcd1e1016c0f719c69d0e1f`
- Failed R3 wall gate:
  `integration-promoted-n128-r3.log`,
  `594ce6279f6e56c11c271e37cc7844a1c3268b38a17e736f91e7f8660bda0a10`
- Passing R5 adjudication:
  `integration-n128-r5-adjudication.log`,
  `872151e4fc47d85274a43fcc75b7646398dcbb481d165d3594dd95b01bdd2722`
- Traced R3 attribution:
  `integration-n128-r3-trace.log`,
  `141ec48efbbad2dcafcd54bbd4d7223a5eda670f06ecf2d4b6ae637c928cae35`
- Isolated rollback:
  `integration-rollback-n12.log`,
  `337353f85807107b1f38fc0077383cea89074106d39f2cd53d7fd1f7b33602e1`

The logs live under
`target/profiles/dsv4-packed-grouped-experts/`. Strict release workspace
all-target/all-feature Clippy passes. CX design and final promotion review:
`019fcf7d-e9d4-7150-b496-e70a31958e80`.

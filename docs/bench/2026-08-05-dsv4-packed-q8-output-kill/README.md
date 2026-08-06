# DeepSeek V4 Packed Q8 Attention Output KILL

Status: `KILL` for both tested schedules. The exact mapped/token-tiled family
does not improve production-shape GPU time. The faster matrix family clears its
bounded first-chunk performance gate but fails the frozen quality and routing
gates after an exact 12-token tail and restored continuation. No experimental
kernel, session policy, allocation, or live diagnostic remains.

## Question

Packed attribution at base revision `1911ae1` assigns 377.528 ms, or 35.397%
of 128-token pre-expert GPU time, to `encode_output`. The deployed path performs
eight group pack/projection/scatter triplets for output A and one output-B
projection. This packet asks two separate questions:

1. Can one mapped or token-tiled Q8 GEMV preserve every singleton reduction bit
   while sharing weight traversal across tokens?
2. If exact traversal cannot win, can the existing Q8 SIMD matrix schedule be
   admitted for only the first full 128-token chunk, followed by exact fallback?

The numerical gate was fixed before the final run: one canonical 140-token
natural-language prefix; matrix output A and B only for position-zero N=128;
exact current output for the N=12 tail; checkpoints at positions 128 and 140;
snapshot restore and one singleton continuation; exact argmax, cosine at least
0.999, and relative RMS at most 0.05 for logits and normalized hidden vectors at
all three checkpoints; exact expert IDs and stable schedules; zero shallow CSA
selector use; at least 20% first-chunk pre-expert GPU saving, 15% aggregate GPU
and wall saving, no more than 5% exact-tail regression, and no more than 5%
within-arm drift.

## Exact Schedule

The exact candidate retained each token's deployed Q8 block order and F32
reduction lineage. T1 maps token and group in the grid; T4 and T8 retain four or
eight token accumulators while sharing decoded Q8 values. A guarded differential
covered N=1/3/4/7/8/9, all three tiles, grouped source and destination strides,
odd output width, offsets, repeated runs, and output guards. Every represented
output matched singleton GEMV.

At production output geometry and N=128, six-sample medians were:

| Schedule | GPU ms/layer | Saving vs current |
|---|---:|---:|
| Current exact | 8.912 | - |
| Mapped T1 | 8.997 | -0.955% |
| Tiled T4 | 11.408 | -28.013% |
| Tiled T8 | 11.855 | -33.022% |

The tested mappings provide no production-N=128 GPU gain. Only command-GPU time
was measured for this model-free exact family, so this is not an integrated wall
claim. Effective weight reuse in the independent-token baseline and reduced
parallelism/register pressure in T4/T8 are plausible explanations, not measured
attribution. The complete exact family was removed from the active source.

## Matrix Falsifiers

The existing Q8 matrix kernel stages Q8 values and F32 activations through F16
SIMD matrices, so it is numerical rather than bitwise. Model-free N=128 results
showed a real upper bound:

| Matrix stages | Output cosine | Output rel RMS | GPU ms/layer | Saving |
|---|---:|---:|---:|---:|
| A only | 0.999999614 | 0.000894324 | 5.633 | 36.795% |
| B only | 0.999999647 | 0.000840433 | 5.457 | 38.770% |
| A and B | 0.999999251 | 0.001230151 | 2.181 | 75.525% |

The first integrated stage split rejected general chunk admission. At N=32,
both A-only and B-only miss the 0.999/0.05 envelope, and A+B reaches only
0.997798509 cosine / 0.067231852 relative RMS. At N=128, A+B passes that broad
local envelope and saves 27.633% pre-expert GPU and 23.450% wall, authorizing
only the stricter position-zero 128+12 falsifier.

## Final 128+12 Gate

The current 97.05 GiB asset ran one untimed detailed control/candidate pair,
then timed A/B/A/B/A. The first 140 tokens of the frozen natural-language
prompt have ID SHA-256
`8dc3a3091bc13e6f8e982e25e251e3995565d26b89e70a1f7cc8da36060de2f7`.
Controls and candidates repeat their own logits, normalized hidden vectors,
causal snapshots, continuation decisions, and committed tokens bit-for-bit.
Matrix invocation ownership is exact at 43/43 after the first chunk and remains
unchanged through the exact tail and singleton continuation; every control is
0/0. Prefix and compatibility metadata remain exact.

### Performance

| Metric | Controls (ms) | Candidates (ms) | Decision value |
|---|---|---|---:|
| First-chunk pre-expert GPU | 1090.132 / 1109.609 / 1084.159 | 782.661 / 781.750 | 27.851% saving |
| Aggregate pre-expert GPU | 1312.998 / 1333.629 / 1310.401 | 1005.959 / 1004.426 | 23.291% saving |
| First-chunk wall | 3656.221 / 3767.966 / 3785.878 | 2957.347 / 2951.883 | support only |
| Aggregate wall | 4395.729 / 4510.796 / 4529.334 | 3701.990 / 3696.266 | 15.847% saving |
| Exact-tail pre-expert GPU | 222.866 / 224.020 / 226.242 | 223.299 / 222.676 | 0.055% regression |

Aggregate control/candidate GPU drift is 1.757%/0.153%; wall drift is
2.994%/0.155%. Every frozen performance condition passes.

Decision savings use the mean of the two candidate samples against the minimum
of the three controls. Drift is the symmetric range divided by its midpoint.
These deliberately conservative formulas produce 27.851382% first GPU,
23.291205% aggregate GPU, 15.847232% aggregate wall, and 0.054536% exact-tail
regression; median-to-median arithmetic is not the decision formula.

### Quality

All three argmaxes remain exact at 305 / 12,122 / 20,332, but four of six full
vectors fail. This is why argmax agreement is not an admission criterion.

| Checkpoint | Vector | Cosine | Relative RMS | Gate |
|---|---|---:|---:|---|
| 128 | logits | 0.999512353 | 0.031261980 | pass |
| 128 | normalized hidden | 0.999010728 | 0.044472216 | pass |
| 140 after exact tail | logits | 0.998832693 | 0.048418514 | fail cosine |
| 140 after exact tail | normalized hidden | 0.997495086 | 0.070735848 | fail both |
| restored continuation | logits | 0.998840549 | 0.048206680 | fail cosine |
| restored continuation | normalized hidden | 0.998319865 | 0.057944362 | fail both |

The exact tail does not repair approximation already committed into causal
state. The degradation remains visible after snapshot restoration.

### Decisions

Both arms execute zero sparse CSA selectors at this shallow fixture. Routing is
not equivalent: packed expert IDs and expert-major schedules diverge, and the
restored singleton continuation also changes selected expert IDs. Maximum packed
route-weight delta is 0.351734340; maximum continuation delta is 0.134389818.
The minimum learned rank-6/rank-7 margins are 0.000021935 in control and
0.000006676 in candidate, with a maximum margin delta of 0.272183418. Thus the
failure is not merely harmless output-vector drift: the numerical schedule
crosses consumed MoE decisions.

## Decision

Remove both candidate families. Keep the exact deployed packed output path and
do not add a hidden switch. Do not retry the same F16-staged matrix condition at
another chunk threshold under the current M4 Max/current-asset portfolio: N=32
rejects broad admission and the canonical 128+12 packet proves that a passing
first endpoint does not survive exact tail, restore, or routing evidence. This
falsifies general admission under the frozen contract; it does not claim that
every conceivable prompt-specific guarded use is impossible.

Reopen output work only for a materially different arithmetic schedule with a
credible whole-stage ceiling, such as one that preserves F32 activation
precision while retaining enough matrix parallelism. It must clear the same
integrated checkpoints before any ordinary seam. The next active optimization
target returns to attribution of the unchanged packed post-route work.

## Evidence

- Base revision: `1911ae1cc5ab3dcd3938c621d92ea3eefe53068b`.
- Device: Apple M4 Max; macOS 15.6.1 (24G90); Rust 1.97.1.
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`,
  104,207,848,032 bytes across four shards pinned by
  `crates/qwen-llm/tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_current_2026_08_04_census_v1.json`.
- Exact model-free log: `model-free-shape-sweep.log`, SHA-256
  `7e292f2d27ec5f6f96e2a11a205c2bbc5d847dec1e432d09461e9252764cc4c2`.
- Initial integrated log: `current-asset-probe.log`, SHA-256
  `b45085a512da50d60d40c2ec359b73fda9d97bfa70de5102e3615b6475c782ab`.
- Stage-split log: `current-asset-stage-split.log`, SHA-256
  `b0d6937d9a733e9c7badcaf66aa9933f83dc45e408eeeebd83d6795a088977e2`.
- Final 128+12 log: `current-asset-first-chunk-128-plus-12.log`, SHA-256
  `862236dd320e7604e71a4821efe9e1141f5c86a55aed695a8f7cc20d76b5f609`.
- The four named raw logs above are committed beside this README rather than
  referenced only from `target/`.
- The chronological noncompiled experiment source is preserved as
  `experiment-source.apply-patch.log`, SHA-256
  `5b172cba1b13aa88dc3c54874868b1d4c4422a19721d973295e87ef284227645`.
  It contains all 48 successful `apply_patch` calls from base through the logged
  test source; removal and documentation changes follow outside that bundle.
- Logged release test executable SHA-256:
  `adce7f44b42dc76e6f0f2bb85b3b73dc07fddb412b68761f395ac1d1bbdaa735`.
- Final integrated experiment source SHA-256 before removal:
  `deepseek_v4_metal.rs` =
  `277be8bc331a0fb8476e8273899cc7a73ac349cee0feeff2da2e14198db0771e`;
  `prefill.rs` =
  `1c859acf09f17ca66ae2c0b7c55cc50b6aaf3adb59c985942fc0f8d73fb16a42`;
  `diagnostics.rs` =
  `68c301060da13efa9da92952cda02bc3dd0f6266b3bf6f8ee13004428d4125f8`.
- Exact-schedule source SHA-256 before removal:
  `mat_vec_q8_0.metal` =
  `335e01f476b8aefa9c46b43f1ccdee035e9a7456a9948dd1e9888ef3376f1d6a`;
  `metal.rs` =
  `18025ce51253010ed26c202786edfd7db788b732f1c0bb1fdb365e14eb7c7d59`;
  `prefill.rs` =
  `e9239d6e578af850bcb8b24a37e8f89160f64072714b92d57ac491ee3d01e2e2`.

Log-to-source binding inside the patch transcript:

| Log | Apply patches |
|---|---:|
| `current-asset-probe.log` | 1-21 |
| `model-free-shape-sweep.log` | 1-28 |
| `current-asset-stage-split.log` | 1-28 |
| `current-asset-first-chunk-128-plus-12.log` | 1-48 |

To reconstruct a checkpoint, start from the exact base revision and process the
numbered sections in order through the bound prefix. Each section is one
OpenCode `apply_patch` input, not a shell-compatible unified diff. Replace the
archived `/Users/tito/code/qwen-llm` root with the reconstruction worktree and
submit each complete `*** Begin Patch` through `*** End Patch` body to an
`apply_patch`-compatible tool. Stop at the listed prefix before building and
running its command. Later prefixes supersede earlier experiment source; do not
apply the post-experiment cleanup when reconstructing a logged binary.

Commands:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  q8_token_tiled_gemv_preserves_grouped_singleton_bits_and_guards

cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::q8_token_tiled_attention_output_production_shape_packet \
  -- --ignored --exact --nocapture

cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_matrix_q8_output_probe \
  -- --ignored --exact --nocapture

cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_matrix_q8_first_chunk_packet \
  -- --ignored --exact --nocapture
```

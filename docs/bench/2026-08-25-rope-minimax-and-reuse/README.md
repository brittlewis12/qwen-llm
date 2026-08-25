# RoPE Minimax, Reuse, And Fusion

Status: `GO` for native Q/K RMSNorm+RoPE fusion in base-model decode and short
packed batches, with rollback. Its direct oracle is bitwise equal on the tested
M4 Max toolchain. `GO` for adaptive packed Q/K pairing as a structural
primitive. `KILL` direct minimax substitution and cross-head decode
serialization. Ordinary packed-prefill throughput remains neutral, so no broad
prefill product gain is claimed. The Metal-MTP model/head implementation is
unchanged; DFlash's packed target verifier and drafter noise-token RoPE are in
scope.

All timings are exploratory dirty-tree measurements on Apple M4 Max, macOS
15.7.9, Metal compiler 32023.864, with `-O3 -ffast-math`. The checkout already
contained unrelated untracked documents. The source and benchmark are retained
so a clean successor can repeat the packet. Product deltas below are directional
rather than canonical release numbers; this packet records summarized results,
not raw per-run artifacts.

## Question

The supplied article catalogs odd/even minimax polynomials for sine and cosine:

- <https://publik-void.github.io/sin-cos-approximations/>

Qwen RoPE had three independent sources of redundant work:

1. separate `sin` and `cos` calls for one angle;
2. identical `pow/sin/cos` evaluation for every Q/K head; and
3. separate Q norm, K norm, and RoPE dispatches with a second round trip over
   normalized Q/K.

The experiment isolates each source before changing production. The model-free
`qwen-bench rope-micro` command compares:

- incumbent head-parallel `sin` + `cos`;
- Metal's combined `sincos` intrinsic;
- packed Q/K pairing;
- one lane per `(token, rotary pair)` sharing coefficients across heads;
- article-derived degree-9 sine / degree-10 cosine absolute-minimax polynomials;
- composed Q/K norms plus RoPE; and
- a fused Q/K RMSNorm+RoPE epilogue.

## Numerical Contract

The minimax arm uses the article's radian coefficients, FMA Horner evaluation,
and split-constant Cody-Waite reduction into `[0, pi/2]`. The article's ideal
polynomial error does not include coefficient rounding, f32 phase formation,
range reduction, or Metal fast-math reassociation.

Across Qwen's real `24 Q / 4 KV / head_dim 256 / rotary_dim 64` shape and
positions `0, 1, 127, 65,535, 65,536, 262,143, 1,048,575`:

| Arm | Max absolute output delta | RMS absolute delta |
|---|---:|---:|
| head-parallel `sincos` | 0 | 0 |
| shared-head native `sincos` | 0 | 0 |
| shared-head minimax | `1.4473e-6` | `2.8751e-8` |

On this M4 Max and Metal compiler, packed Q/K pairing and shared-head native
execution are also bitwise equal to the incumbent. Across the measured packed
lengths, minimax reaches `3.2783e-7` max absolute and about `1.65e-8` RMS.

On the same toolchain, the retained fused Q/K RMSNorm+RoPE kernel is bitwise
equal to composed Q norm, K norm, and native RoPE for complete Q/K outputs at
`N=1/8/128`. Tested consecutive spans start at `0`, `65,531`, `65,536`, and
`1,048,568`, ending at position `1,048,575`. The kernel preserves the composed
normalization reduction order, normalized-value rounding point, NEOX pairing,
and native Metal `sincos` call. Bitwise equality remains compiler/hardware
sensitive rather than an API-level cross-device guarantee.

The benchmark-only minimax reduction was validated only across those recorded
positions. Its host wrappers now reject spans beyond `1,048,575`; it should not
be interpreted as a general accuracy guarantee over all `u32` positions. Fused
norm+RoPE numerical equivalence is gated by the unit test rather than reported
as an accuracy row by `rope-micro`.

Real-model gates pass:

- 0.8B F32 attention block vs CPU: cosine `1.000000`, max absolute
  `1.12e-4`;
- 27B packed prefill at `N=16` and `N=128`: final-logit cosine `1.000000`,
  hidden/KV/state gates green;
- 27B packed verifier at `N=16`: all 16 argmaxes agree and every logit-row
  cosine is `1.000000`.

## Primitive Results

Commands encode 256 in-place RoPE dispatches per GPU command and interleave
arms. Values are mean microseconds per dispatch. This amplifies
sub-10-microsecond geometry; each dispatch rotates the previous dispatch's
output rather than simulating independent model-layer dataflow.

### RoPE Alone, 24 Q / 4 KV Heads

| Workload | Baseline | Paired | Shared native | Shared minimax |
|---|---:|---:|---:|---:|
| decode N=1 | `2.106` | `2.085` (`sincos`) | `5.809` | `5.962` |
| prefill N=1 | `3.750` | `2.178` | `5.747` | `5.776` |
| prefill N=8 | `3.895` | `2.375` | `5.927` | `6.049` |
| prefill N=128 | `9.383` | `7.547` | `6.418` | `6.529` |
| prefill N=512 | `29.289` | `25.812` | `8.113` | `8.231` |
| prefill N=2048 | `104.902` | `96.424` | `32.667` | `32.562` |

The decode baseline is the incumbent paired Q/K kernel with separate `sin` and
`cos`; the packed-prefill baseline is two independent Q and K dispatches.

Cross-head serialization is decisively wrong for decode and short batches: it
starves the GPU even though it removes most transcendental evaluations. Packed
Q/K pairing wins there by deleting a dispatch while preserving head-level
parallelism. At `N>=128`, the token axis supplies enough work for cross-head
reuse; it is about `3.2-3.6x` faster than split Q/K at `N=512/2048`.

The crossover repeats in M4 Max sweeps of supported `8/2`, `16/2`, `24/4`, and
`32/2` Q/K head shapes. The current heuristic therefore pairs heads below 128
tokens and shares native coefficients at 128 or more.
`QWEN_PREFILL_ROPE_PAIRED=0` restores the original two-dispatch path. The 128-row
threshold is measured hardware policy, not a portable Metal performance
contract.

### Q/K Norm Plus RoPE, 24 Q / 4 KV Heads

| N | Norms + split | Norms + adaptive RoPE | Fused norm+RoPE |
|---:|---:|---:|---:|
| 1 | `7.444` | `6.171` | `2.785` |
| 8 | `9.821` | `8.557` | `5.959` |
| 16 | `13.507` | `12.085` | `9.873` |
| 32 | `23.293` | `21.823` | `19.492` |
| 48 | `30.135` | `28.478` | `27.392` |
| 64 | `37.434` | `35.952` | `35.479` |
| 80 | `44.630` | `43.042` | `43.461` |
| 128 | `66.816` | `63.699` | `66.891` |

Fusion wins strongly for decode and small verification/MoE chunks, then loses
to the lower-register composed kernels as row count grows. In the displayed
24/4 sweep it crosses between 64 and 80 rows; other supported head shapes were
already marginal or behind around 64. Production therefore uses the
conservative M4-derived cutoff `N<=48`. This too is a hardware heuristic, not a
portable crossover. The old pairing flags remain umbrella rollbacks; dedicated
controls are `QWEN_DECODE_QK_NORM_ROPE_FUSED=0` and
`QWEN_PREFILL_QK_NORM_ROPE_FUSED=0`. The prefill control governs batched packed
execution. F32 or mixed-dtype fallback rows call singleton attention and thus
obey the decode controls; a complete cross-path rollback sets both decode and
prefill controls to zero.

## Product Results

### Decode

Two 20-run 0.8B Q4_K `tg128` brackets give:

| Arm | Wall per 128 tokens | GPU per 128 tokens |
|---|---:|---:|
| fused | `344.50 ms` | `302.50 ms` |
| rollback | `346.95 ms` | `304.65 ms` |

That is a directional `0.71%` wall and `0.71%` GPU saving in this exploratory
dirty-tree bracket, not a release-grade estimate. The same binary on Qwen
35B-A3B gives `589.45 -> 591.75 ms` fused-to-rollback wall ordering and
`560.15 -> 562.60 ms` GPU ordering: directional `0.39%` wall and `0.44%` GPU
savings.

A cooled 27B pair is neutral: fused `2557.1 ms`, rollback `2553.4 ms` for
64 tokens. Earlier uncool runs drifted materially inside each process and have
no causal authority. The default is retained for structural dispatch deletion,
small-model/MoE wins, and no demonstrated 27B regression beyond noise; no 27B
speedup is claimed.

### Prefill And Verification

Ordinary 0.8B and A3B pp suites vary within about `+/-0.3%` except unstable tiny
rows. No general packed-prefill product credit is assigned. The adaptive path is
retained because it is bitwise equal on the tested toolchain, removes
dispatches, and has large direct primitive wins without a measured whole-request
regression.

The 27B N=16 packed-verifier wall is `188.51/190.06 ms` on two fused runs versus
`194.16 ms` on one rollback run, a directional `2-3%` verifier improvement.
This is supporting evidence, not a promotion-grade bracket.

## Decisions

1. **Retain fused native decode.** It changes Q norm + K norm + paired RoPE from
   three dispatches to one, deleting two dispatches per full-attention layer.
2. **Retain fused short packed execution.** It replaces two norms and all later
   Q/K RoPE dispatches with one locally bitwise-equal kernel for `N<=48`.
3. **Retain adaptive packed Q/K pairing.** Use head parallelism below 128 rows
   and native coefficient sharing at 128 or more.
4. **Do not promote the minimax polynomial.** It is not repeatably faster than
   native `sincos` at matched geometry and introduces nonzero phase/output error.
   The implementation remains in the model-free benchmark for reproducibility,
   not in production selection.
5. **Do not serialize decode across heads.** Reduced transcendental count is not
   a useful cost metric when it removes nearly all GPU parallelism.

## Remaining Opportunities

- A tiny immutable inverse-frequency table could remove repeated `pow`, but a
  host-generated table changes the current Metal rounding lineage. Screen it
  only inside the fused kernel with a direct exactness gate.
- A per-position coefficient slab could share sin/cos across layers. For Qwen,
  the retained fused leaf is already only a few microseconds per layer; price a
  zero-work ceiling before adding storage or a generation dispatch.
- DeepSeek V4 has greater cross-layer reuse (43 layers, two RoPE policies), but
  its prior KV-only fusion lost `0.051 ms/token`, and the optimistic packed
  RoPE/publication ceiling was only `46-66 ms` under unstable acquisition. Do
  not reopen without a changed premise or stable instrumentation.
- Consecutive-position complex recurrence avoids trig but accumulates phase and
  norm drift. It is lower priority than exact cached coefficients and needs
  periodic high-precision re-anchoring.

## Reproduction

```bash
cargo build --release -p qwen-cli --bin qwen-bench

cargo test -p qwen-llm --lib \
  qk_rms_norm_rope_fused_matches_composed_path -- --nocapture

cargo test -p qwen-llm --test dflash_correctness \
  prefill_tokens_matches_single_token_loop_27b -- --nocapture

cargo test -p qwen-llm --test dflash_correctness \
  dflash_packed_verify_layer_major_vs_token_major_27b -- --nocapture

target/release/qwen-bench --allow-dirty rope-micro \
  --tokens 1,8,16,32,48,64,80,128,256,512,2048 \
  --runs 24 --layers 256 -o text

for heads in "8 2" "16 2" "24 4" "32 2"; do
  set -- $heads
  target/release/qwen-bench --allow-dirty rope-micro \
    --q-heads "$1" --kv-heads "$2" --tokens 48,64,80,128,256,512 \
    --runs 24 --layers 256 -o json
done

target/release/qwen-bench --allow-dirty tg \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -n 128 --runs 20 -o text

QWEN_DECODE_QK_NORM_ROPE_FUSED=0 \
target/release/qwen-bench --allow-dirty tg \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -n 128 --runs 20 -o text
```

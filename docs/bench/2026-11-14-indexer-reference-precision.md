# V4.1 / GLM indexer reference-precision research

Research-only on fresh branch `research/indexer-reference-precision` from
`f8398b4b4`. No production code changed; no GPU kernel was built or run.

## Reference and current formats

| Model | Local reference code/config | Reference Q/K | Reference score path | qwen-llm today |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash | `deepseek-v41-mlx/deepseek_v41_mlx/fakequant.py:96-107`, `indexer.py:21-24,98-110`; official source/byte contract pinned in `deepseek_v4_indexer_fp4_contract.rs:156-180,239-260` | Q and K are fake-quantized to E2M1 with UE8M0 scales, 32 values/block, using BF16-rounded inputs. The score path casts Q/K to F32 before dot/reduce. The MLX mirror materializes dequantized fake-FP4 values in the input dtype (normally BF16); the repo's packed row contract is 64 value bytes + 4 scale bytes = 68 B for a 128-wide key. | F32 dot accumulation, ReLU, per-head weight, head reduction. | F32 query, F32 key cache, F32 score output; `index_heads_rows_mma` is an F32 simdgroup-MMA path. Key storage is 512 B/key. |
| GLM-5.3 Flash | `mlx-lm/mlx_lm/models/glm_moe_dsa.py:9-14,126-137`; shared `Indexer` in `deepseek_v32.py:60-108`; captured model config `mlx-serve/src/fixtures/model-configs/glm5_next-mlx-community-4bit.json` | Native model dtype, no FP4 QAT or E2M1/UE8M0 indexer path. The captured text config is BF16, 32 heads ×128, pool 4, top-k 2048, with 45 `full` indexer types. | `q @ k`, ReLU, head weighting and head reduction; the reference does not explicitly cast the index score to F32, so its result follows MLX matmul/reduction dtype semantics. | F16 pooled keys and F16 staged queries; F32 head weights and F32 score output. The current scorer is `kernel_lightning_indexer_scores_f16_matrix_h32`. Key storage is 256 B/key. |

qwen-llm source confirmation: V4.1 `index_query`, `index_weights`, and score
scratch are allocated F32 in `crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs:539-540,2385-2388`; its owner key cache is F32 at `:2352-2362`. The F32 score path and 128-key MMA tile are documented at `:1258-1277,1375-1380`. GLM's F16 query materialization is at `crates/qwen-llm/src/glm5_next_metal/packed.rs:2987-3007`; required F16 query/key and F32 head-weight/score types are checked by `crates/qwen-llm/src/metal/sparse.rs:38-51,104-115`. The GLM key pool's F16 rounding/storage is explicit in `kernels/glm53_indexer.metal:1-10,56-60`.

The GLM full-layer count and dtype are from the captured upstream-compatible
MLX config; a GGUF `attention.indexer.types` override takes precedence in a
particular qwen-llm load. The V4.1 scalar contract is a packed-semantic CPU contract; it does not make
the packed format a production cache. The V4.1 MLX mirror stores fake-FP4
values in BF16 arrays, not the 68-byte packed ABI. qwen-llm's cache and score
inputs remain F32. For GLM the local official-compatible reference code has no
FP4 indexer quantization. Matching an invented FP4 GLM path would move away
from that reference.

The V4.1 storage comparison per 128-d key is 512 B current F32 versus 68 B
packed reference (7.53× smaller); the matching query row is 16,384 B current
F32 versus 2,176 B packed E2M1/UE8M0. The dequantized MLX fake-FP4 mirror is
256 B/key and 8,192 B/query in BF16. GLM's current F16 and reference BF16
operands are both 256 B/key and 8,192 B/query. In qwen-llm, both current score
outputs are F32, 4 B per query/key pair.

## Work and byte price at production shapes

Use one pair for a query row scoring one visible key. The QK dot is
`2 × 32 heads × 128 dimensions = 8,192 FLOPs/pair`; ReLU, head weighting and
head reduction add about 35 scalar operations. The packed path changes
operand bytes and adds nibble/scale decode instructions, not the logical QK
FLOP count.

For V4.1 I used the checked schedule's eight index sources: three ratio-2
sources and five ratio-1 sources; later ratio-1 score layers reuse the layer-20
key cache (`gguf.rs:42-57,94-103`). For GLM I used the captured 45-full-layer,
pool-4 config. Pair counts are causal-visible rows, so they are exact for the
formula but can differ slightly from dispatch work if a kernel pads a
microbatch to its maximum visible pool count.

| Cell | Visible query/key pairs | QK FLOPs | V4.1 current → packed logical scan bytes* | GLM current / BF16-reference logical scan bytes* |
| --- | ---: | ---: | ---: | ---: |
| V4.1 80K + 256 | 136.53 M | 1.118 TF | 70.48 → 9.83 GB | — |
| V4.1 80K + 1,024 | 548.67 M | 4.495 TF | 283.25 → 39.52 GB | — |
| V4.1 128K + 256 | 218.32 M | 1.788 TF | 112.69 → 15.72 GB | — |
| V4.1 128K + 1,024 | 875.83 M | 7.175 TF | 452.06 → 63.08 GB | — |
| GLM 48K + 2,048 | 1,156.03 M | 9.470 TF | — | 301.34 GB (same Q/K bytes; 299.02 GB if reference scores materialize BF16) |
| GLM 80K + 256 | 236.30 M | 1.936 TF | — | 61.53 GB (same Q/K bytes; 61.06 GB if score output is BF16) |
| GLM 80K + 1,024 | 949.61 M | 7.779 TF | — | 247.28 GB (same Q/K bytes; 245.38 GB if score output is BF16) |

\* Logical scan estimate counts one key vector read per query/key pair, the
query/head-weight operands once per query, and one F32 score write per pair.
For V4.1 the key term is 512 B/pair current versus 68 B/pair packed; for GLM
it is 256 B/pair in both F16 current and BF16 reference. These are logical
operand-volume estimates, not measured DRAM bytes: keys are tiled/reused across
heads and queries, so GPU cache behavior can change bus traffic. FP4 pack/dequant ALU cost
is not included. V4.1's current per-session resident index-key banks are about
104.9 MB at 80K and 167.8 MB at 128K, versus 13.9 MB and 22.3 MB packed.
GLM's 45 F16 key banks are about 141.6 MB at 48K and 235.9 MB at 80K; a
reference BF16 cache has the same byte count.

Observed stage context is already available, but it is not a score-kernel
microbenchmark: V4.1 `prefill_attention_serial` is 54.07/122.81 ms at
80K/256 and 80K/1,024, and 86.23/191.43 ms at 128K; it also includes
position-ordered transitions and selection. The GLM census measured sparse
index-score spans of 376.71 ms at 48K+2,048, 55.18 ms at 80K+256, and
231.24 ms at 80K+1,024. Those are 18.8%, 12.2%, and 20.0% of ordinary wall.

I did not run an M5 microbenchmark. The repository has a test-only packed
FP4 score shadow (`kernels/deepseek_v4.metal` and
`deepseek_v4_metal/indexer.rs`), but it hard-codes 64 heads ×128 and its
historical M4 Max timing uses synthetic inputs. Both V4.1 and GLM here use
32×128; V4.1's incumbent is F32 rather than the shadow's F16 baseline, and
the GLM reference is not FP4. Reusing that packet would not be a valid
real-cell stage comparison; adapting the packed scorer and incumbent pair to
the target geometry is new kernel work, not a cheap census probe. The old
64-head shadow showed about 2.5× at 65K/262K rows on M4 Max, useful as a lead
but not transferable to this target.

## Quality and recommendation

For V4.1, run selection agreement first: compare current and packed-reference
top-512 IDs per query, report exact overlap/recall and score error, then
condition disagreements on the score margin at the 512th/513th boundary.
Include the candidate-block shortlist decision at layer 20. A low overlap or
flips well outside the measured score-error band closes the candidate before
teacher forcing.

If that screen is acceptable, follow the paired NI protocol in
`docs/bench/2026-11-11-glm-append-noninferiority/README.md`: same prompts and
tokens per pair, report signed dNLL, both KL directions, top-1 agreement and
route/selection changes, equal prompt weighting, prompt-cluster bootstrap,
20,000 resamples, central 90% CI and one-sided 95% worsening bound. Do not
claim absolute Exact-referenced quality where the Exact prefix did not
complete; use the protocol's paired sequential-relative comparison.

**Recommendation: build the V4.1 packed-reference scorer first, then gate
selection agreement and NI before considering a default.** It is the only
model where upstream training-time index Q/K semantics support FP4, and it
has substantial score work plus a 7.53× packed operand-size opportunity. This
is a plausibility recommendation, not a speed claim; the old 64-head M4 shadow
does not predict V4.1's 32-head M5 result. Close GLM's FP4-cache idea: current
F16 and reference BF16 have the same two-byte storage, and the reference has
no FP4 indexer path. Any GLM BF16 score-output change would need its own
selection/NI evidence and is not justified by this byte census.

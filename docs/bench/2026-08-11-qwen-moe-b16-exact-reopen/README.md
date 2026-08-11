# Exact Qwen MoE B=16 Reopen

Date: 2026-08-11

Status: `HOLD`. Correctness is green, but the corrected five-process median
misses the frozen product-spike throughput gate by `0.0272` token/s. This packet
changes benchmark and exact token-axis kernel infrastructure only; it does not
change ordinary inference or expose a new product-inference CLI mode.

## Question

The prior A3B B=8 static executor missed the independent-queue crossover after
its exact incremental repair. Wider GDN replay floors indicated that B=16 or
B=32 might amortize immutable weights enough to change the decision. The first
complete-model reopen cleared the speed line but changed generated continuation,
and was closed as a width-amplified GDN schedule failure.

That closure did not survive root-cause review. The candidate's MoE FFN tail was
serial while production uses concurrent shared/routed waves, and no projection
ablation established which schedule change caused the state drift. This packet
repairs those confounds and asks whether an exact wider organization can retain
the economic gain.

The frozen reopen requirements are:

- at least `1.15x` over serialized production execution;
- at least `1.12x` over the frozen `125.81` token/s B=8 independent-queue
  control, preserving margin above the old `1.10x` product gate;
- finite same-history values, exact generated IDs, and bit-exact logits,
  residuals, token history, active KV, GDN convolution, and recurrent state;
- charge token staging, command construction, argmax, and all model stages.

After a fresh review found that candidate wall stopped just before host argmax
readback, the final timing campaign was invalidated before any corrected run.
The corrected campaign is five serialized fresh processes from one binary and
source-state identity. Its predeclared decision rule is:

- all five runs must pass the strict exactness contract;
- median candidate throughput must be at least `140.9072` token/s (`1.12x` the
  frozen queue control);
- at least four of five runs must remain above `138.391` token/s (`1.10x` the
  queue control), so one transient host outlier cannot define or rescue the lane.

No corrected timing observation existed when this rule was written.

## Root-Cause Repair

The existing replay used matrix kernels for three Q8_0 GDN projections: qkv, z,
and out. A benchmark selector replaced each projection independently with the
existing token-axis Q8_0 GEMV, which keeps the singleton accumulation body while
sharing one dispatch across the cohort.

| Exact projection | Min residual cosine | Max abs residual | Replay GPU, ms/token |
|---|---:|---:|---:|
| qkv | 0.999999709 | 0.007381 | 6.7517 |
| z | 0.999999598 | 0.012825 | 6.5072 |
| out | 0.999999709 | 0.006846 | 6.3442 |
| qkv + z | 0.999999747 | 0.007844 | 6.9049 |
| qkv + z + out | **1.000000000** | **0.000000** | 6.9261 |

The non-exact B=16 replay was `6.7413 ms/token`; exactness therefore costs about
2.7% in the block-body organization, not the entire width gain. Exact out alone
is an attractive but invalid compromise: it reaches `172.814` aggregate
token/s, then diverges at generated step 15 while same-history logit relative RMS
reaches 15.82%.

The whole-model benchmark now calls the same route preparation and concurrent-
shared MoE FFN policy as production after each sequence-private mixer. Attention,
router state, expert scratch, recurrent state, and KV remain session-owned.

## Exact LM Head

With all GDN projections exact, the existing batched Q6_K matrix head leaves
residual state exact but perturbs logits. B=16 happens to retain generated IDs at
`157.432` token/s; B=32 changes one first-step ID. Singleton Q6_K heads restore
exact logits but reduce B=16 to `140.213` token/s.

The repair adds `kernel_mat_vec_q6_K_f32_batch`: grid Y selects a token row while
each row retains the singleton kernel's lane map, Q6 extraction, multiplication,
accumulation grouping, and `simd_sum` reduction. The host wrapper validates exact
Q6/F32 shapes, nonzero dimensions, 256-element input alignment, and u32 bounds.
A release GPU test compares B=16 against 16 singleton dispatches bit-for-bit over
three Q6 blocks and an odd seven-row output tail.

## Invalidated Preliminary Timing

Asset: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

Frontier: 1,024 tokens. Cohort: 16 independent lanes. Work: two warmup plus 64
measured autoregressive transitions, alternating serial-first and candidate-first
order. Both arms restore the same causal snapshot and generate their own greedy
feedback. Three processes used the same built source state. Each records
build commit, dirty source-state SHA-256, active `QWEN_*` variable names, and a
canonical step-major/lane-major i32-le token hash.

| Invalidated endpoint | Serialized | Exact B=16 | Ratio |
|---|---:|---:|---:|
| Median wall, ms/16 tokens | 148.1684 | 113.3543 | 1.3071x |
| Aggregate token/s | 107.985 | **141.150** | **1.3071x** |
| Median GPU, ms/16 tokens | 140.9630 | 109.2937 | 1.2898x |
| Versus frozen queue control | 125.81 tok/s | **141.150 tok/s** | **1.1219x** |

The candidate process results are `139.122`, `141.150`, and `141.281` token/s.
Two earlier code-equivalent probes reached `144.442` and `144.538`, but the
source-identical three-process median above is not authority: candidate wall
stopped immediately before host argmax readback. It cleared the frozen `1.12x`
queue line by only 0.19 percentage points, too little to waive the omission.

The exactness gate covers all 66 same-history transitions:

- all finite logits and final residuals compare bit-for-bit;
- all 1,056 selected IDs compare directly and have the same SHA-256,
  `df06b16496fd1be19ba6e5a17ce4aa2f7dd30fad064bb75ba21241a5e66af7e0`;
- final snapshots for every lane compare identity, consumed tokens, pending
  token, per-attention-layer position, active K and V bytes, every GDN
  convolution and recurrent-state byte, and final logits;
- final GDN F32 arenas are explicitly rejected if non-finite.

B=32 also passes generated continuation and rounded numerical probes. In the
matched earlier width run, B=16 reaches `144.538` and B=32 `143.462` token/s;
doubling cohort and state width therefore demonstrates no throughput gain. B=16
is the candidate product point. The hardened byte-level causal packet is
authoritative for B=16; corrected timing is decided below.

## Corrected Final Campaign

The candidate now reads argmax IDs before stopping its wall timer. Five runs use
one binary and source-state identity
`git-source-sha256-v2:6d69ee13a1d57c2a2d78889c7fbd16a83e0d4a458324d3de8048fced82bfa2f8`.

| Run | Serial tok/s | Candidate tok/s | Paired speedup | Versus queue |
|---:|---:|---:|---:|---:|
| 1 | 114.449 | 144.378 | 1.2615x | 1.147588x |
| 2 | 106.016 | 139.498 | 1.3158x | 1.108799x |
| 3 | 109.093 | 142.066 | 1.3023x | 1.129211x |
| 4 | 106.592 | 140.880 | 1.3217x | 1.119784x |
| 5 | 106.497 | 140.613 | 1.3203x | 1.117662x |

Median candidate throughput is `140.880` token/s. It misses the predeclared
`140.9072` primary gate by `0.0272` token/s. All five runs exceed the secondary
`138.391` floor, but that guard cannot rescue a failed primary gate. The median
paired speedup is `1.3158x`; ratio-of-median endpoint throughput is `1.3217x`.

Every run covers all 66 same-history transitions and reports finite, bit-exact
logits and residuals, exact final causal snapshots, no ID divergence, and the
same generated-ID SHA-256 shown above. Corrected raw records are tracked beside
this README as `corrected-r1.tsv` through `corrected-r5.tsv`.

## Validation

Passed:

```text
cargo fmt --all
cargo check -p qwen-cli --bin qwen-bench
cargo clippy -p qwen-cli --bin qwen-bench -- -D warnings
cargo clippy -p qwen-llm --lib -- -D warnings
cargo test --release -p qwen-llm \
  mat_vec_q6_k_batch_matches_singleton_bits -- --nocapture
```

The final evidence command is:

```text
QWEN_BENCH_GDN_REPLAY_EXACT=all \
QWEN_BENCH_MOE_BATCHED_HEAD=0 \
./target/release/qwen-bench --allow-dirty \
  decode-moe-gdn-repair \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --frontier-tokens 1024 --prefill-chunk 1024 \
  --warmup-steps 2 --steps 64
```

Exploratory outputs under `target/profiles/qwen-moe-b{16,32}-*` remain
disposable. The five corrected final records are tracked with this document.

## Decision

Do not build the Qwen MoE JSONL executor yet. The exact organization exposes a
real roughly 12% advantage over independent queues, but the frozen gate reserves
integration margin and fails as written. Retain the benchmark and exact
token-axis primitives so a materially better organization can be measured
without repeating the root-cause work.

If reopened, build a sibling fixed-B=16 executor rather than parameterizing the
dense B=8 executor. The product slice must preserve:

1. exact token-axis Q8_0 GDN projections and the exact token-axis Q6_K head;
2. production route and concurrent-shared FFN wave scheduling;
3. sequence-private KV, GDN, route, scratch, and pending-token state;
4. shared-prefix snapshot fanout before decode;
5. model-derived admission for 16 sessions plus bounded scratch;
6. input-order JSONL output, compatible-cohort admission, cancellation and
   frontier poisoning, and serial fallback for incomplete cohorts.

Reopen for an exact measured candidate gain that clears `1.12x` with practical
scheduler margin, or when shared executor machinery makes the marginal product
cost small enough to revisit the economics. Promotion would still require the
integrated path to clear a contemporaneous independent-queue control.

Do not promote the faster non-exact head or exact-out-only compromise without a
separately approved functional-equivalence contract. Do not infer support for
other Qwen MoE geometries until their tensor contracts and exactness are checked.

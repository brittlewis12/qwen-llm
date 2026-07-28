# v0.645 Exact GPU Greedy Product Admission

Status: preregistration. No v0.645 scored observation exists. The unscored
27B/A3B eight-token product smokes preceding this document carry no timing or
admission authority.

## Question And Authority

Can the exact GPU greedy reducer remove full-vocabulary host readback and CPU
selection from ordinary serial generation while preserving product semantics
and producing a measurable A3B benefit without regressing dense 27B?

A pass may authorize an asset-scoped follow-up that defaults on only when the
loaded model identity matches the exact 27B or A3B asset below. It does not
authorize a global default flip. `QWEN_GREEDY_GPU_ARGMAX=0` must remain the
explicit rollback. The result cannot authorize A10B, prompt lookup,
MTP/speculation, positive-temperature sampling, prefix-cache policy, another
model/quant, or a new benchmark claim.

Implementation commits:

- `252e790`: integer-only total-order reducer and opt-in product wiring;
- `a167ed7`: dense/MoE exact-token and byte-exact continuation-state gates.
- `7add76d`: bind both state gates to the exact admitted assets and emit positive
  execution markers.

The runner must execute from a clean descendant of `7add76d` whose `crates/`,
`kernels/`, and Cargo manifests have no further implementation delta. The
release `qwen` and `qwen-bench` binaries must identify that clean HEAD.

## Frozen Assets And Request

- Dense guard: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`, SHA-256
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`.
- MoE primary: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  1,891 bytes, 419 tokens on both assets, SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- Generation: temperature zero, 128 selected tokens, 127 target transitions,
  fixed chunk 1024, context capacity 1024, prompt lookup absent.
- RAM insertion, automatic prefix admission, and durable caching are disabled.

## Arms

Every child uses one clean binary and a normalized environment with all inherited
`QWEN_*` variables removed.

- A: `QWEN_GREEDY_GPU_ARGMAX` absent; full logits return to CPU selection.
- B: `QWEN_GREEDY_GPU_ARGMAX=1`; one exact GPU token/error result returns.

No other argument, environment value, model byte, prompt byte, or process policy
may differ.

## Correctness And Product Conformance

Before scored acquisition, rerun these exact model-backed tests:

```text
cargo test --release -p qwen-llm \
  metal_exact_greedy_chain_matches_full_logits -- --nocapture --test-threads=1
```

They must run serially, emit both positive execution markers, and preserve every
selected token ID, bit-identical continuation logits, and byte-identical active
KV, GDN state, and convolution state. A skipped body is a failure.

Then run one fresh single-turn A/B conformance pair per model, both before any
powered acquisition, with 16 output tokens. Require byte-identical stdout, 16
generated tokens, 15 transitions,
`token_limit`, no consumed terminal transition, no cache/prompt-lookup/sampling
surface, and labels `greedy_argmax` / `greedy_gpu_argmax`. These observations are
conformance-only and never enter timing estimates.

## Powered JSONL Acquisition

One arm-session is one fresh process and one loaded model. It executes five
identical JSONL requests:

1. request zero is an unscored in-process warmup;
2. requests one through four are scored;
3. each request must retain 419/128/127 prompt/selected/transition counts, proving
   no EOS before the final selected position; JSONL does not expose whether the
   128th token itself is EOS;
4. every output within and across paired sessions must be byte-identical;
5. cache entries/bytes/hits remain zero and sampling/prompt lookup remain absent.

Reduce each session to the median of its four scored values. The primary value is
`transition_ms / decode_transitions`. Secondary values are
`decode_ms / generated_tokens`, `model_ttft_ms`, and `prefill_ms`.

Sessions run serially. Pair order repeats `AB, BA`, giving four `ABBA` blocks for
the initial eight pairs. There are no retries or replacement children. After
eight pairs, estimate the standard deviation `s` of paired log transition ratios
and freeze the final pair count:

```text
n = ceil(((1.645 + 0.84) * s / ln(1.01))^2)
```

Include the first eight pairs, require at least eight, and cap at sixteen. Each
model independently estimates and freezes its own pair count with the same
formula. If A3B requires `n > 16`, stop the complete packet immediately and seal
inconclusive without running the dense powered guard. If dense requires
`n > 16`, seal inconclusive after its eight-pair pilot. Otherwise continue each
model's frozen alternating schedule to its own `n`. Cool down five seconds
between sessions.
Before packet creation and every charged model/GPU child, require at least 85%
system-wide free memory from `memory_pressure -Q`, no inference competitor, and
no thermal or performance warning. Archive the successful execution preflight
inside P1 immediately after creation.

For each metric and pair `i`, compute `y_i = log(A_i / B_i)`. Report
`S = exp(mean(y))` and a one-sided paired-t 95% lower bound. `S > 1` favors B.
Do not trim outliers or condition on observations after acquisition. The sample
count is driven only by transition variance; noisier secondary noninferiority
bounds can therefore conservatively leave the feature off. At a true effect of
exactly 1%, the additional observed-point-estimate gate also makes admission
probability about one half. This is accepted: the packet favors avoiding a false
default over resolving the low edge of the prior band.

## Mechanical Gates

Hard correctness/validity gates:

- clean source/build/runtime identity and exact model/prompt hashes;
- every process exits zero and every expected row/output is present once;
- all conformance and JSONL outputs/counts/cache/telemetry contracts pass;
- no generated-chain/state test failure, NaN, wrong policy, or mixed arm;
- each model's final aggregate paired prefill speed ratio lies in `[0.97, 1.03]`;
- final clean HEAD/build identity, both binary hashes, prompt/script/prereg hashes,
  and model device/inode/size/mtime identities match the opening manifest.

Performance gates:

- A3B primary transition point estimate `S >= 1.010` and lower bound `> 1.000`;
- A3B and dense transition lower bounds `>= 0.990`;
- both models' decode-per-token lower bounds `>= 0.980`;
- both models' TTFT lower bounds `>= 0.970`.

Decision:

- `ADMIT_EXACT_ASSETS` only if every hard and performance gate passes;
- `KEEP_DEFAULT_OFF_EFFECT_MISS` if A3B's upper bound is below `1.010` while
  every validity gate passes;
- `KEEP_DEFAULT_OFF_REGRESSION` for any valid noninferiority miss;
- `INCONCLUSIVE_RESOLUTION` takes precedence and stops powered acquisition if
  either model's frozen count exceeds 16, or applies if a completed interval
  still straddles the required A3B effect;
- `INVALID` for any identity, correctness, contamination, or completion failure.
  Invalid packets still write `failure.json`, `decision.json`, a complete artifact
  inventory, and `packet-complete.json`; the fixed P1 path is consumed.

No post-hoc retry, gate relaxation, pooling with May's v0.78 observations, or
A10B extrapolation is allowed.

Artifact root: `target/profiles/v0645-gpu-greedy-product-p1/`.

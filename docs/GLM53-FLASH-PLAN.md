# GLM-5.3-Flash Bring-Up Plan

Implementation map for adding GLM-5.3-Flash (`glm5-next` in llama.cpp) as a native
family. The semantic reference is llama.cpp `src/models/glm5-next.cpp` at upstream
`42d958167` (line numbers below refer to that tree; upstream through `11fe02151` has
no further `glm5-next` changes). Earlier research (2026-08-28 gap analysis,
2026-08-29 schedule) and `cx` adversarial reviews at each packet are folded in.

## Status (2026-10-05)

- P1 done: strict binder, role x dtype coverage with a phase-specific execution
  gate, memory ledger, native `glm4` tokenizer (HF and llama-tokenize parity),
  same-artifact oracle harness and the ckpt-v1 reference capture
  (`scripts/reference/glm53/ckpt-v1.json`).
- P2 done (`glm5_next_metal`, serial decode, dense range): ckpt-v1 top-1 15/15,
  worst KL 2.9e-11, block residuals <= 2e-5; KDA state, indexer pools and pending
  keys match llama.cpp at steps 3/7/14. `ModelFamily::Glm5Next` with raw-prompt
  `run` and `qwen-bench` lanes. Shared encoders lifted from DS4 (mHC, clamp,
  routing, all-slot experts) keep DS4 output identical on the real asset.
- P3 done (packed prefill; `docs/bench/2026-10-04-glm53-p3-packed-prefill/`, A-B-B-A
  against llama.cpp on the same file): pp512 207.5-209.7 vs 212.8-222.7 tok/s,
  pp1024 190-193 vs 200-203, tg128 27.7 vs 22.2-22.9. Fast lineage (half-staged
  mat-mat and grouped experts) is the default; Exact (decode kernels per row)
  equals serial decode bitwise and is the equivalence reference.
- Hardening after the P3 review: whole-request validation (a refused request
  executes nothing); session buffers built from the ledger's spec lists and priced
  with the device before admission (observed allocation 98.4% decode-only, 99.6%
  with 512 prefill rows); checked ledger arithmetic.
- Long-context qualification (qual-v1, `scripts/reference/glm53/qual-v1.json`, 591
  positions): native serial matches llama.cpp serial to KL <= 1.1e-9 until an exact
  router-score tie at position 156 that the two break differently (native: lowest
  expert id), top-1 591/591. Fast vs Exact: KL <= 9.1e-3, top-1 flips only with
  choice regret <= 0.06 (reference) / 0.176 (Fast), tighter than llama.cpp's own
  batched-vs-serial divergence on the same tokens (prompt-end KL 8.6e-3, worst
  4.2e-2). Chunk size is arithmetic-neutral (512 = 128 and 64 = 97 bitwise). Bounds
  are calibrated regression bounds, not a holdout.
- Model-scale tests hold the cross-process production lease, pass the wired-memory
  gate and run with `MTL_DEBUG_LAYER=1`; comparison helpers fail on NaN, empty or
  mismatched inputs.
- Q6_K N64 at 512 rows: no measurable pp512 change (+0.8% inside a +-5% noise
  floor). Strict router skipped: both router kernels are full F32; it only reorders
  accumulation and cannot reduce routing flips, which come from upstream activations.
- P4 done (sparse DSA selection; `docs/bench/2026-10-04-glm53-p4-sparse/`):
  shared H32 half-matrix scorer, radix4 selector, pool expansion and online
  selected attention (DS4 delegates), sparse decode and packed sparse prefill
  (64-query microbatches, sticky per-row selector status), run and bench up to
  the checkpoint context. Gates (sparse-v1, bounds frozen before observation):
  native scores equal llama.cpp's bitwise on its captured indexer inputs at
  2050-2060 in all 11 MLA blocks, selection sets equal in all 110 sparse cases;
  decode across the frontier top-1 46/46, worst KL 4.2e-5; exact packed equals
  serial bitwise across the frontier for every chunking tested. A-B-B-A vs
  llama.cpp: pp4096 178.6-182.7 vs 177.3-177.4 t/s, tg128 27.7-27.9 vs
  22.1-22.5, tg128 at depth 4096 20.7-20.8 vs 19.9-20.1.
- P4 follow-up: sparse-v2 (`scripts/reference/glm53/sparse-v2.json`) qualifies
  selection near 4096 (sets equal in all 44 cases) with replays bound to their
  manifests.
- P5 entry tasks done: one CPU preparation (`glm5_next::admission`) drives run,
  bench and capabilities with stable codes; session preflight before prefetch
  names the capacity that fits; prefetch and packed prefill cancel at read and
  chunk boundaries; phase timing separates setup from the loaded request.
- P5 chat (T1 template) done: `glm5_next_chat` renders the upstream template's
  text subset byte-for-byte (48 jinja2 fixtures from
  `scripts/reference/generate_glm53_chat_fixtures.py`, both template digests
  pinned; the GGUF template differs only on null assistant content, which it
  prints as `None`); native token ids equal HF's on every rendered case. Effort
  low/high/max (default max; others refused rather than silently Max), no
  non-thinking mode, `clear_thinking`, tools refused. Chat eligibility is the
  embedded template digest, single-token markers and released stops.
- P5 run done: `qwen run --user/--system/--messages`, reasoning to stderr and
  the answer to stdout through the shared pre-opened partition, release sampling
  (temperature 1.0, top-p 0.95, top-k/min-p off) for raw and chat unless flags
  override, chat profile and controls in `diagnostics.glm5_next` (schema 2).
- P5 serve done (`docs/SERVE.md#glm-53-flash-verified-text-chat`): verified text
  chat over one live session reused only on exact extension of its consumed
  tokens (KDA cannot rewind), fresh otherwise; prefill cancellation keeps the
  committed chunks; startup preflight, prefetch, load and warm-up before
  accepting. GPU gate: cold equals run, replayed turns reuse the whole history,
  Exact warm equals cold in output bytes and in join logits bit for bit,
  cancelled prefill resumes with the cold bytes, aborted decodes retry cold.
  Run and serve stats records are success-only, as for K2.
- Next: P5 lens sites; then P6: grouped-heads selected attention (sparse decode
  costs +4.7 ms/token at the frontier; native loses 25% from depth 0 to 4096
  against llama.cpp's 10%), long-context dense attention, a packed-prefill stage
  profiler (a fresh 22-28 token prefill costs 0.6-1.4 s), compact grouped-expert
  scheduling, and an exact top-p sampler that avoids the full-vocabulary sort
  (sampled decode 25.3 vs 27.7 tok/s greedy).

## Artifact

Local weights: unsloth `GLM-5.3-Flash-GGUF` revision `621d456e93e926e4b52f85cff5f634358c1828f9`,
`UD-IQ3_XXS` (4 shards, 112.092 GiB, 1,412 tensors), in
`/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/`. The original shard 1 says
architecture `glm5next`; `glm5-next/` links unsloth's `Shard_Rewrite` shard 1 (`glm5-next`
keys) with the original shards 2-4. Tensor bytes are identical; accept both names and
both metadata namespaces. Shard 1 holds metadata only.

Census (`.fetch/analysis/census_iq3xxs.py` on the archive):

- Trunk (blocks 0-44 plus embedding/head) 109.484 GiB: IQ2_S 59.10, IQ3_S 39.64,
  Q6_K 6.15, IQ4_XS 3.59, Q8_0 0.81, F32 0.21. All are storage-native
  (`metal_forward/residency.rs:13`); executable coverage is a separate M0 check.
- Routed experts: gate/up IQ2_S, down IQ3_S; block 11 uses IQ3_S gate/up, and blocks
  11, 12, 44 use IQ4_XS down. Block 11 also carries Q8_0 attention and shared expert.
- KDA q/k/v/o, MLA `q_a`/`wo`, shared expert, dense FFN, embedding and head: Q6_K.
  MLA `q_b`/`kv_a_mqa`/`k_b`/`v_b`, indexer, `ssm_{beta,f_a,f_b,g_a,g_b}`, `hc_*_fn`
  [16384, 24]: Q8_0. Router, norms, biases, conv, `ssm_a`, APE, `indexer.proj`: F32.
- NextN block 45: 2.608 GiB (Q2_K/Q3_K experts), a contiguous suffix of shard 4
  starting at byte 19,220,825,312. 46 stored blocks, 45 executed.
- Reference runtime: llama.cpp `build-glm5` (Metal) runs it fully resident on the M4
  Max: pp512 223.4, pp4096 174.5, tg128 22.56 tok/s, with about 7% memory free.

## Architecture (reference semantics)

Hyperparameters: hidden 4096, 64 heads, vocab 154880, RMS eps 1e-5, LayerNorm eps 1e-6,
no RoPE anywhere (`rope.dimension_count` 0).

Schedule: blocks {3, 7, ..., 43} are MLA/DSA (`head_count_kv` 1), the other 34 trunk
blocks are KDA. Blocks 0-2 have a dense SwiGLU FFN (12288); 3-44 have 288 routed experts
(top 8, 2048) plus one shared expert (2048).

mHC (4 streams, :407-543, :554-679). The embedding row is copied into 4 streams
X[4096, 4]. Each block has an attention and an FFN sub-block, each with its own
`hc_*_{fn,base,scale}`:
1. `flat` = X as [16384] (stream-major); RMSNorm without weight; `mixes = hc_fn . flat` [24].
2. `pre = sigmoid(mixes[0:4] * scale[0] + base[0:4]) + 1e-6`;
   `post = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8])`;
   `comb[dst, src] = mixes[8 + dst + 4 * src] * scale[2] + base[8 + ...]` (dst-fast).
3. Sinkhorn on comb: softmax over dst per src, + eps; divide by (sum over src + eps);
   then 19 rounds of {normalize over dst, normalize over src}, each dividing by
   (sum + eps). Fused CPU reference: ggml `ops.cpp:11371-11456`.
4. Sub-block input `h = sum_s pre[s] X[:, s]`, then RMSNorm * `attn_norm`/`ffn_norm`.
5. `X'[:, dst] = post[dst] * y + sum_src comb[dst, src] * X[:, src]`.
6. After block 44: unweighted mean of the 4 streams, `output_norm`, `output`. There is
   no learned hyper-head (unlike DS4's `hc_head`).

KDA (34 blocks, :686-762; H 64, d 128, D 8192):
- q, k, v = Q6_K projections; per-channel causal depthwise conv (k 4, separate
  `ssm_conv1d_{q,k,v}`, tap 3 multiplies the current token) then SiLU.
- q, k L2-normalized per head as `x / sqrt(sum x^2 + 1e-6)`. llama.cpp builds this as
  `rms_norm(x, eps/128) / sqrt(128)` (`models.h:14`), so a no-weight RMSNorm serves.
  The existing `kernel_l2_norm_f32` uses `max(sqrt, eps)` and is not the contract.
- `ssm_a` stores `-exp(A_log)` (:707). Per-channel log-decay
  `g = -5 * sigmoid(-ssm_a[h] * (f_b . f_a . x + dt_bias))`, in [-5, 0].
- `beta = sigmoid(ssm_beta . x)` per head.
- Per head, F32 state S[k 128, v 128]. Per token: `S <- diag(exp(g)) S` (decay along the
  key axis), `delta = beta (v - S^T k)`, `S += k delta^T`, `o = S^T q / sqrt(128)`.
  Reference: fused `ggml_gated_delta_net` (`ops.cpp:11124-11262`). The unfused path
  (`delta-net-base.cpp:335-340`, reachable when fusion is off, :435) decays the wrong
  axis; oracle harnesses must pin the fused operator.
- Output: `attn_output . (RMSNorm_head(o) * ssm_norm * sigmoid(g_b . g_a . x))`.

MLA (11 blocks, :923-1013), NoPE:
- `qr = RMSNorm(q_a . x) * q_a_norm` [1536]; `q = q_b . qr` [64 x 256].
- Latent `c = RMSNorm(kv_a_mqa . x) * kv_a_norm` [512] is the only attention cache
  entry and serves as both key and value.
- Absorbed, in storage order (`attn_k_b` [256, 512, 64], `attn_v_b` [512, 256, 64]):
  `q_lat[l, h] = sum_d Kb[d, l, h] q[d, h]`; scores `q_lat . c / sqrt(256)`; softmax
  over visible/selected positions (no sinks); `o_lat = sum p c`;
  `o[d, h] = sum_l Vb[l, d, h] o_lat[l, h]`; concat 16384; `attn_output`. Grouped matvec
  consumes this layout without repacking.

Indexer / DSA selection (per MLA block, :766-919):
- `iq = indexer.attn_q_b . qr` [32 x 128]; `ik = LayerNorm(indexer.attn_k . x)` with
  weight and bias; `ig = indexer_compressor_gate . x` [128];
  `w = indexer.proj . x * 1/sqrt(128 * 32)` [32], signed.
- llama.cpp caches `key | gate | pooled` rows at cache precision (F16 by default).
  When a pool of 4 consecutive tokens completes,
  `pooled[c] = sum_j softmax_j(ig_j[c] + ape[c, j]) ik_j[c]` (per-channel softmax over
  the pool's 4 positions), computed from the rounded cached key/gate.
- `score(pool) = sum_h relu(iq_h . pooled) * w_h`; take `min(n_pool, 512)` pools,
  ordered by descending score, then append the incomplete-pool tail (0-3 tokens,
  newest first in llama.cpp, `llama-memory-hybrid-idx.cpp:1015`). Selection width is at
  most 2051; do not size for 2048. Specify zero completed pools, ties, invalid slots
  and per-query visibility explicitly.
- Dense and sparse select identical keys through visible length 2051 (512 pools plus
  a 3-token tail). The switch is at visible length 2052. DwarfStar runs dense through
  4095 on Metal; that is a semantic divergence, not a precision choice.
- Indexer state is maintained from token zero so crossing 2052 needs no recompute.

MoE (llama-graph.cpp:2046-2174):
- `p = sigmoid(router . x)` in F32; select top 8 by `p + exp_probs_b`; weights are the
  unbiased `p` normalized by `max(sum, 6.103515625e-5)`, times 2.5. The shared expert
  is added unweighted.
- SwiGLU clamp everywhere (routed, shared, dense): `silu(min(gate, 10)) * clamp(up, -10, 10)`.

Session state: KDA S 136 MiB F32 (34 x 64 x 128 x 128) plus conv tails about 9.6 MiB,
constant in context. llama.cpp's cache is 19.25 KiB/token. A native append-only cache
needs only F16 latent rows (1 KiB/token/layer), F16 completed pooled keys (256 B per
pool) and a bounded pending key/gate tail: 11.69 KiB/token, 374 MiB at 32K. Truncating
the cache cannot rewind KDA; prompt edits need replay or a compatible checkpoint.

MTP: block 45 (`eh_proj`, `enorm`, `hnorm`, `shared_head_norm`, reusing the main head;
`index_share_for_mtp_iteration`). llama.cpp has no NextN graph (:188-193). Deferred.

## Prior Art

- llama.cpp `build-glm5` (`~/code/llama.cpp`, `glm5-next-lora-wo` = upstream `42d958167`
  plus a LoRA-only wiring fix): the executable oracle on the same artifact.
  `llama-eval-callback` and `llama-tokenize` (vocab-only, about 170 MB) are built.
- HF `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
  from the local FP8 archive (`weights-archive/glm-5.3-flash-fp8/`): tokenizer and
  template oracle. The FP8 model itself does not fit and is not a runtime oracle.
- FLA (`~/code/flash-linear-attention`, `3e52d5ab`): `fla/ops/kda/naive.py` is an
  independent recurrence reference.
- DwarfStar (`~/code/ds4`, MIT, `0aaea5a`): GLM Metal kernels, already DS4 prior art
  (`docs/THIRD-PARTY-NOTICES.md`; qwen-llm's tiled indexer scorer is attributed to
  it). Port with attribution into qwen-llm's residency/session/mHC; do not transplant
  its monolithic graph. Required adaptations:
  - `glm53_kda.metal` fused decode (:14) consumes `A_log`; this GGUF stores
    `ssm_a = -exp(A_log)`, so use `-ssm_a` (:93). Its 3-stage prefill (:162, :249,
    :289) loops serially over tokens with register-resident state; it is not chunk
    algebra.
  - Pool update (`dsv4_misc.metal:978`) reads APE as BF16 (ours is F32) and
    normalizes raw F32 keys inside pooling; llama.cpp pools the rounded cached
    key/gate. Preserve the reference rounding points.
  - Indexed decode attention (:2847) returns unless `qk_rope == 64`; GLM needs a NoPE
    specialization. Its tail expansion (:1575) is oldest-first.
  - `kernel_glm_k_b_project_q8_0` (:624) expands latent keys; it is not query
    absorption.
  - Fixes to inherit as tests: `582d809` (padded-selection decode masking), `9775e26`
    (pooled decode padding), `fff391e` (inline tool call block).

## Reuse Map

| Component | Start from | Required change |
|---|---|---|
| mHC | DS4 `hc_repeat` (`deepseek_v4.metal:4394`), `hc_controls` (:4416), `hc_collapse` (:4565), `hc_post` (:4599), batch variants | Same 4/24/20 contract; mean collapse (pre = 1/4) then `output_norm`; skip `hc_head`. |
| Router | DS4 GPU routing (`route_learned` :97, packed :278-789) and route records (`deepseek_v4_metal/moe.rs:17-43`) | Generalize together: score function (sigmoid), expert count 288, reduction groups, top-k 8, record strides, compaction descriptors and consumers. GPU routing for short prefill too (`PERF-ROADMAP.md:57`). |
| Routed experts | DS4 indexed expert encoders (`moe.rs:588-1081`, IQ2_S/IQ3_S at :1506); generic grouped kernels (`moe.metal:7145`, :7184) | Grouped fused SwiGLU is unclamped: add the DS4 clamp epilogue. Indexed decode lacks IQ4_XS: bridge to Qwen's IQ4_XS down with explicit ID/status/layout adapters. Decode scratch scales with top-k. |
| Shared expert, dense FFN | DS4 shared Q6_K path (`moe.rs:1164`), dense matvec/matmat | Clamped SwiGLU; unweighted add; block 11 is Q8_0. |
| KDA decode | DwarfStar fused decode with the `ssm_a` fix | Compare against `gdn_step_decay_f32` (`gated_delta_net.metal:137`) adapted to per-channel decay; keep the faster one that meets the contract. |
| KDA prefill | DwarfStar register-resident 3-stage path | Measure at production shapes; pursue chunk algebra only if recurrence time justifies it. |
| Norms, conv | `ssm_conv_silu_f32` (`ssm_conv.metal:39`); sigmoid-gated per-head RMSNorm (`qwen4exp.metal:382`) | L2 via no-weight RMSNorm. `ssm_conv.metal:458` is SiLU-gated and does not apply. |
| MLA attention | Grouped Q8_0 matvec (`mat_vec_q8_0.metal:304`) for absorption; DS4 cooperative/split/online shared-latent attention | Explicit NoPE/no-sink visibility contract, 512-wide latent. Not the scalar dense kernel (:1354), which recomputes the dot per output dimension. |
| Indexer | qwen-llm tiled scorer (`deepseek_v4.metal:2887`, 64-head assumptions at :2882) and parallel selector (:4244) | 64 -> 32 heads, ReLU, signed weights; preserve score-ranked order (DS4's fast selector yields cache order). Non-overlapping pool update adapted from DwarfStar. |
| Residency | retained no-copy GGUF windows (`metal/tensor.rs:65`, `metal/context.rs:937`) | Request trunk tensors only. The planner bridges gaps (`tensor.rs:248`), so gate on actual window ranges, gap bytes and priced totals. |
| Tokenizer | `k2::pretokenize` (`tokenizer/k2.rs:47-137`) and native BPE | Letter class `\p{L}` (marks are punctuation); `ignore_merges`. |

## Family Structure

Decision: separate `Glm5Next` model/session/residency types (the DS4 strategy rule:
share only primitives whose contracts match), with narrow, allocation-free encoder
extraction from DS4 and Qwen4Exp as GLM consumes them. A DS4 variant would couple two
strict bindings and attention contracts; a generic runtime first would put a DS4
refactor on the critical path without proving the abstraction.

Lift order: (1) mHC controls/collapse/post and clamped activation; (2) GPU route
records and indexed/grouped expert projections with explicit geometry, scoring policy,
clamp and caller-owned scratch; (3) shared-latent attention and indexer scoring/
selection primitives without DS4 cache scheduling; (4) recurrent helpers where
contracts match. Model binding, layer schedule, pool lifecycle, cache ownership,
collapse policy, tokenizer/template and checkpoint identity stay family-specific.
Existing shader names may stay. DS4 and Qwen4Exp tests guard every extraction.

Lanes share one preparation result and execution adapter (extend the callback seam in
`qwen-cli/src/ordinary_executor.rs:170`, not a copied generation loop). Serve overrides
the default Qwen request/render/output profile (`serve/http.rs:85`). Lens sites: raw
residual [4096, 4], collapsed sub-block input [4096] and the final mean-collapse
readout; a 4096-wide direction must say which stream(s) it targets.

## Memory

The working set is 112 GiB (`iogpu.wired_limit_mb` 114688). Trunk weights leave about
2.52 GiB for session state, scratch, logits and reserve. DS4's eager 4096-token
prefill scratch (`deepseek_v4_metal/prefill.rs:1444`) would cost about 2.25 GiB at GLM
widths (mHC buffers [16384, B], absorbed queries/latent output [64, 512, B], routed
outputs [4096, 8, B]); full logits for 4096 positions would add 2.36 GiB. Therefore:

- An allocation ledger built from the same named buffer specs the session
  allocates (state, decode scratch, optional packed scratch, route records, all
  session-resident), priced with the device before admission plus a 512 MiB
  reserve; the 16 KiB page profile is the device-free planning bound.
- Bounded prefill microbatches and scratch shared across layers.
- Last-position logits by default; full-logit captures are diagnostic only.
- Metal vs CPU budgets kept distinct for staging, captures and future snapshots
  (`metal/context.rs:454`); a snapshot copies about 146 MiB of recurrent state plus
  cache.
- Capacity comes from device admission (`metal/context.rs:409-528`), not a context cap.

Decode reads about 9.5 GiB of weights per token: 6.7 GiB dense (KDA/MLA projections,
shared expert, router, head) and 2.8 GiB for 8 routed experts. The bandwidth ceiling
is roughly 40-50 tok/s; llama.cpp reaches 22.6. Dense Q6_K/Q8_0 matvec dominates.

## Tokenizer And Template

Native first, matching the DS4 and K2 precedent (`K2-HORIZON-REVIEW-REQUEST.md:19-20`).
The pinned `llama-cpp-sys-2` (`f312d16` -> llama.cpp `8489846b`) cannot load `glm5-next`
even vocab-only, and its `glm4` handling predates `ignore_merges`. Bumping that chain
(fork branch `llama-cpp-sys-2-patches`, `llama-cpp-rs`, `llama_core`, `llm`, qwen-llm
rev) is a separate side quest, needed only for an FFI fallback or a `lcpp` bench
comparator (the bench lock b11182 also lacks `glm5-next`; start with `lcpp = false`).

- Dispatch `(glm5-next | glm5next, gpt2, glm4)` to `Glm4`: the llama3-shaped split
  (`llama-vocab.cpp:406-410`, HF Split regex) is `k2::pretokenize` with letter class
  `\p{L}`. No normalizer. Follow HF for the case fold of `'` + U+017F (long s).
- `ignore_merges`: after regex split and GPT-2 byte encoding, a whole piece found in
  the vocab is emitted directly; otherwise normal BPE (`llama-vocab.cpp:621-636`).
  About 24 single-piece vocab strings differ without it.
- BOS 154822 `[gMASK]` is identity, not insertion policy: no `add_bos_token` key, HF
  has no insertion template, and raw text gets no prefix. The chat renderer emits
  `[gMASK]<sop>`. Stops: 154820, 154827 and 154829; `GgufFile::stop_token_ids`
  (`gguf.rs:640-671`) currently omits EOM.
- Oracles: HF `tokenizers` on the local `tokenizer.json` (pinned SHA-256, frozen ID
  fixtures across English, CJK, Indic marks, code, whitespace runs, digits, specials)
  and a vocab-only subprocess differential against `build-glm5/bin/llama-tokenize`.
- Template: hand-written renderer with jinja2 byte fixtures. Freeze: only `low` and
  `high` pass through, everything else is `max`; the effort preamble is always
  emitted; generation opens `<|assistant|><think>`; historical reasoning retention
  depends on `clear_thinking` and the last-user boundary; historical assistant content
  is stripped. The GGUF-embedded template (10,648 chars) differs from the HF file
  (10,950 bytes); pin upstream and record both hashes. Tools after the text path.

## Packets

Each packet ends with a `cx` review. Estimates are focused engineer-days.

- P1 Artifact contract and executable allocation plan (3-4 d). `ModelFamily::Glm5Next`
  with a fail-closed profile; both architecture aliases; config validated against
  this release; binder classifying all 1,412 tensors with each trunk tensor bound once;
  a role x dtype x decode/prefill encoder coverage matrix that fails before residency
  on any gap (never whole-bank F32); requested-window report; compact cache sizing;
  bounded scratch formula that fits with reserve; a small same-artifact oracle capture
  harness. Gate: real inventory and allocation formula, existing dispatch unchanged.
- T1 Tokenizer and template (2-3 d, parallel, does not block token-ID execution).
- P2 Short-context native decode through the shared execution seam (7-10 d).
  Extract mHC and routing/expert encoders; fused KDA with corrected storage
  contracts; grouped Q8_0 absorption; latent attention; indexer pool maintenance from
  token zero; all 45 blocks and the real head; token-ID run and bench access. Small
  operation fixtures only for new contracts: KDA against FLA with nonzero,
  nonsymmetric initial state and nonuniform channel decay; router cases where experts
  256-287 win, ties and tiny denominators; pool/tail cases. Checkpoint: eight fixed,
  nonrepeated token IDs through all 45 layers plus several decode steps, with
  per-position full logits, selected layer/state captures and measured peak memory.
  Performance is measured from this checkpoint on.
- P3 Short packed prefill (4-6 d). Batched mHC, KDA 3-stage prefill, grouped MoE with
  GPU routing, packed latent attention. Gate: prefill-then-decode matches serial below
  visible length 2052; pp512 vs llama.cpp.
- P4 Sparse selection (5-7 d). Ordered top-512 plus tail, selected attention, a packed
  chunk straddling 2052 with per-query visibility, continued prefill. Avoid inheriting
  the Qwen4Exp `2048 + 3 + remainder` scheduling shoulder (`PERF-ROADMAP.md:56`).
  Gate: boundary fixtures at 2049-2056 vs llama.cpp; then pp4096.
  Verified semantics (llama.cpp `build_kpool_select`, `set_input_kpool`): weights =
  `indexer_proj . x / 64`; score = sum_h relu(q_h . pooled) * w_h with q rounded to
  F16 and F16 pooled keys (fused lightning indexer); a query at visible length L
  sees the first floor(L/4) pools, including one it completes; top-512 pools, each
  expanded to its 4 chronological cells, plus the tail of L % 4 cells (absent
  `kpool_select_tail` defaults to true; absent `indexer.types` means every MLA layer
  selects); attention scale 1/16, no sinks. llama.cpp fills threshold ties in atomic
  order; native breaks them by lowest pool id.
  Design (cx): binder refuses `kpool_select_tail = false` and shared indexer types;
  pool publication moves before attention (decode and packed); a 32-head half-MMA
  scorer templated from DS4's matrix scorer, DS4's radix4 selector, a pool-to-row
  expansion kernel and DS4's online selected-attention kernel (window 0, 2051 slots,
  no sink) behind family-neutral encoders in `crates/qwen-llm/src/metal/` with DS4
  delegating; decode and packed share every sparse kernel so Exact packed stays
  bitwise equal to serial; selector status is sticky per request; packed sparse
  rows run in reusable 64-query microbatches (scores [ceil(capacity/4), 64]); chunk
  offset of the first sparse row is clamp(2051 - chunk_start, 0, rows); positions
  are contiguous from zero (no prefix eviction or rebased restores). Gates:
  component fixtures (signed weights, threshold ties, every tail length, a
  distinctive excluded row, signed zero), real-input component replay from
  `indexer_q`/`indexer_weights`/`indexer_pool_k` captures (identical sets above the
  measured score-error margin; strict winners/losers and threshold counts below
  it), Exact packed 2048 + 16 teacher-forced decode steps, then crossing chunks.
  End-to-end bounds for the boundary run are frozen before it is observed.
- P5 Product lanes (4-6 d). Chat in `run`, `info` from the preparation result,
  request bench, `serve` backend (live session first), lens sites. Entry tasks from
  the 2026-10-04 review: one CPU preparation result (binding, coverage, stops and
  tokenizer admission) drives both `run` and capability projection; cooperative
  cancellation at retained-prefetch read and packed-prefill chunk boundaries, with
  allocation admission before expensive warming; `tokenizer_ms` covers encoding only
  and `total_ms` the loaded request, with setup and end-to-end time reported
  separately.
- P6 Optimization (open). Beat llama.cpp on the same artifact (tg 22.6, pp512 223,
  pp4096 174) with ABBA `qwen-bench` runs, then push toward the bandwidth ceiling.

Critical path: P1 -> P2 -> P3 -> P4 -> P5. First correct tokens at short context:
about 2.5-3 weeks.

## Validation

- Integration oracle: same-artifact llama.cpp layer outputs and full-vocabulary logits.
  No complete second CPU model; independent fixtures cover only new operation
  contracts, and existing DS4 mHC/clamp fixtures are reused.
- Capture harness under `scripts/reference/glm53/` (the `scripts/reference/k2/main.cpp`
  pattern, but `l_out` is [4096, 4], not 4096): `hc` mixes/pre/post/comb, `kda_*`,
  KDA final state and conv tails, `q_absorbed`, `kv_cmpr`, indexer tensors, router IDs
  and weights, `kqv_out`, `ffn_moe_out`, `ffn_shexp`, `l_out`, `result_output`.
  Capture selectively and stream to disk.
- Full-vocab F32 logits (605 KiB/position) with a manifest recording producer commit,
  command and artifact identity.
- llama.cpp and native cannot be resident together. Capture oracle data in one
  window, then run native under the production lease (`QWEN_METAL_LEASE_WAIT=1`),
  the wired-memory gate and `MTL_DEBUG_LAYER=1`. Leave foreign jobs alone.
- No bitwise requirements across kernels or accumulation orders. Error metrics are
  frozen early; top-1 and selection comparisons are margin-aware, and near-tie flips
  get same-input operation replay rather than relaxed global tolerances.

## Deferred

MTP (block 45 does not fit beside the IQ3_XXS trunk at 112 GiB); vision (qwen-llm has
none; a `glm5v` mmproj exists); the lcpprs chain bump; UD-Q4_K_XL and other artifacts
for the M5 Ultra; tools; snapshot implementation (state ownership is not deferred);
context qualification beyond measured lengths.

## Integration Discipline

Work proceeds on `feat/glm53-flash` in `/Users/tito/code/qwen-llm-directions`, kept
current with main. Commits are conventional with Why/What/Evidence/Risks bodies.
`cx` design and adversarial review at each packet. Shared-kernel changes must keep
DS4 and Qwen4Exp behavior and tests unchanged.

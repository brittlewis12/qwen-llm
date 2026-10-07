# GLM-5.3-Flash lane audit (2026-10-06)

A step back after the GLM-5.3-Flash bring-up, P6 and the review fixes. The
question: does the product as a whole reflect the new lane properly? That
covers product lanes, specs, docs, benches, optimization opportunities,
kernel sharing and organization. It also asks which bring-up shortcuts and
trade-offs should be revisited.

Method:
- Read-only reconnaissance at `05fca4c4` (main): code reading, git history
  and session history.
- `cargo clippy` at default level and with `-W clippy::pedantic`.
- `cargo crappy` on qwen-cli. qwen-llm is not measured: its 40-minute
  instrumented run aborts on the three failing route tests (finding B5),
  and it was not repeated.
- No GPU benchmarks.

Principles agreed for this audit:
- **Reference families are the established ones:** ordinary Qwen,
  Qwen3.8-Flash-Next, DeepSeek V4 and Muse Glimmer.
  - K2 Horizon is the least representative bring-up, so it sets no
    precedent. It is in scope for convergence like GLM.
  - GLM was modelled on K2 during bring-up, and inherited several K2
    conventions that way.
- **A "structural" justification must be checked against how an established
  family solved the same problem.** A comment or doc claim is not evidence.
  - Example: GLM's docs said "KDA recurrent state cannot rewind" to justify
    a single live session.
  - The Qwen 3.x hybrids interleave gated delta net 3:1 with full attention,
    and Flash-Next adds QSA. Both have recurrent state, and both solve reuse
    with snapshots at boundaries.
  - The 2026-10-03 plan said it correctly ("prompt edits need replay or a
    compatible checkpoint"). The qualifier was dropped in the serve commit
    `aff7e84a`.
- **Convergence direction is chosen per item:** usually toward established
  behaviour; sometimes the newer behaviour is better and should become the
  shared one.
- **Warnings are understood and ranked, not swallowed or blanket-fixed.**

## A. Divergence ledger (GLM and K2 vs established families)

Class: **S**, structural (keep and document why); **I**, inherited (converge
to established); **L**, newer behaviour better (lift to every family).

| # | Surface | Established | K2 | GLM | Class |
|---|---|---|---|---|---|
| 1 | `qwen run` chat output | literal decoded text on stdout, markers included. Qwen3.6 pre-opens `<think>` in the prompt and still prints literally | reasoning to stderr, answer to stdout (`qwen/chat_output.rs`) | same, copied in `25eb7936` | **I** |
| 2 | `qwen run` with tools | literal model text | a Responses JSON object | same | **I** |
| 3 | `qwen run` grammar and exit | a token limit exits 0; no grammar failure | a stop before the reasoning closes exits non-zero; a stderr "incomplete" line | same | **I** (follows from 1) |
| 4 | Stats line | `<fam> stats: … stop_reason= tokenizer_ms … total_ms` | own fields | own fields; prefill includes the weights' first GPU use, unlabelled | **I** |
| 5 | `--request-stats-jsonl` timing | `total` = wall minus load | sum of phases | sum of phases; own diagnostics schema | **I** |
| 6 | Sampling defaults | Qwen/DS4/Flash-Next: run = serve; Muse: the release preset in both | run ≠ serve (top_k, min_p, seed) | the release preset in both (Muse precedent) | K2 **I**; GLM **S** |
| 7 | Effort and no-thinking refusals | `CapabilityError` codes; Muse `no_thinking_unsupported` | private codes | private codes | levels **S**; codes and wording **I** |
| 8 | Drafter refusal | shared `drafter_policy` resolver, `family_no_speculation` | private, bypasses the resolver | same | **I** |
| 9 | History missing reasoning | rendered as empty reasoning everywhere | filled and reported | `qwen run --messages` uses the template's inline `</think>` split, while GLM serve fills empty reasoning: **CLI and serve disagree** | **I** (bug) |
| 10 | `qwen info --json` | `{reasoning, input, template}` | adds `execution` | adds `execution` | **I** (schema); text `qwen info` is wrong (B3) |
| 11 | Serve body decoding and parsing | serde decoding; shared parser; string `input` = one user message | own parser; unknown fields 400; string `input` = **raw completion** | lossless decoding, duplicate keys 400, shared parser | K2 parsing **I**; lossless decoding **L** |
| 12 | Serve limits and warnings | shared `fixed_session_limits` | private `limits()`, duplicate warning | shared | **I** (K2) |
| 13 | Serve startup | no preflight, prefetch or warm-up | none | device preflight, prefetch, 2-token warm-up | preflight **L**; prefetch for no-copy families **L** (DS4 has it opt-in); warm-up: decide per product |
| 14 | Prefix reuse | snapshots at boundaries (Qwen `SessionSnapshot`: K/V prefix + GDN conv/state; Flash-Next: GDN state, QSA pending block, compressed index keys, K/V rows) | one live session, rewinding by longest common prefix | one live session, exact extension only | **I**: the missing snapshot codec is an implementation gap, not a model property |
| 15 | Memory refusal status | 503 `memory_admission_denied` | session allocation → 500; tool block unpriced | tool block admitted (503); session allocation → 500 | session 503 **L** (DS4/Flash-Next also 500); K2 tool pricing **I** |
| 16 | Grammar failure in serve | Qwen/DS4: an unparsed tool block becomes visible text; Muse: strict 500 | strict 500 | strict 500 | policy decision across families |
| 17 | Serve log lines | `serve phases` includes `tokenize_ms` | missing; stale "raw only" labels | missing | **I** |
| 18 | Bench adapters | DS4 logs a skipped prefetch; `muse-request` uses the chat renderer | `k2-request` raw, greedy | suite rows only; silently skips the prefetch run does | **I** |

## B. Correctness and house-rule findings (fix now)

1. **Snapshot-policy flags silently ignored** on GLM, K2 and Muse:
   `--snapshot-idle-ttl-secs`, `--snapshot-max-age-secs` and
   `--snapshot-half-life-secs`. They are read only by the
   Qwen/DS4/Flash-Next backends (`serve/mod.rs:613, 668, 738`), and
   `check_warmth_flags` never checks them.
2. **Muse `qwen run --drafter` is silently ignored.** The lane is
   dispatched before the shared resolver (`main.rs:303`), contradicting
   `drafter_policy.rs:5-9`.
3. **Text `qwen info` prints Qwen facts** (a "GDN / full-attn" block count,
   `qwen35.*` metadata) for GLM, K2, Muse and Flash-Next (`main.rs:1170-1235`).
   `--json` is correct.
4. **GLM `qwen run --messages` and GLM serve render the same history
   differently** (ledger row 9).
5. **`c640a113` few-row Q8_0 MMA: three failing route tests on main**
   (`qwen4exp_metal` packed HC, `qwen4exp_gdn` packed motor, `qwen4exp_moe`
   packed common MoE).
   - The routing reaches every default-dispatch caller for Q8_0 at 2..=8
     rows, N=8 included. The session that made it knew it reached
     Flash-Next, K2 and the lens readouts, and noted "not separately
     re-qualified".
   - It ran only the two new unit tests, so these route expectations broke
     unnoticed.
   - Three ignored tests that use the release weights (PLE, QSA, MoE) carry
     the same stale expectations.
   - Flash-Next's preflight pipeline lists (`projection_kernel_names`, the
     QSA list) omit the few-row pipelines, so these are built lazily in the
     middle of a forward pass.
   - **The kernel is structurally closer to the serial path:** exact Q8 in
     half, F32 activations and accumulation. The fix is to derive route
     expectations from the dispatch table (including the device
     thread-limit downgrade), add the pipelines to preflight, and
     re-qualify on the GPU. The MoE test's tolerances (3e-8 / 1.5e-7) are
     the risk; investigate before touching them.
   - GLM and DS4 are not affected: their dispatch policy disables few-row.
6. **Session memory refusals return 500.** GLM, K2, DS4 and Flash-Next
   session allocation failures are `server_error`; only transport and tool
   admission use 503. Clients cannot tell pressure from a fault.
7. **K2's tool block is not priced against memory.** GLM's admission
   (`1c4e103a`) was never applied to K2.
8. **Wrong claims in code and docs:**
   - "cannot rewind" as a justification: `SERVE.md:897`,
     `GLM53-FLASH-PLAN.md:75`, `backend_glm5_next.rs:2`,
     `decode_loop.rs:109`, and the #13 packet.
   - `qwen/glm5_next.rs:7` says "tools not implemented".
   - The admission envelope still says K2 is raw-only.
   - The `serve limits` label reads `raw_input_string_only` when chat is
     verified.
   - The GLM plan lists tools as deferred.
9. **ENV.md has never covered GLM or K2.** The generator's prefix filter
   (`scripts/reference/env_knobs.py:18-19`) excludes `GLM*` and `K2_*`; the
   file has not been regenerated since 2026-09-07. 160 of 362 source-line
   references are stale, two entries no longer exist in code, and about 88
   variables are missing.

## C. Documentation gaps (GLM missing where peers appear)

- **README:** no GLM in the model list, examples or serving paragraph. The
  serving paragraph is also wrong about Flash-Next and K2.
- **`SERVE.md` family lists:**
  - "Implemented families" (:434-455), CPU allowance per family (:501),
    `max_output_tokens` defaults (:936), release identity (:1163), missing
    reasoning (:1217), the `upstream` refusal list (:1249).
  - The synopsis lacks `--idle-residency-secs`.
  - ":926 GLM's text chat" predates tools.
- **`CLI-UX.md`:** no GLM section. Effort and `--no-thinking` lists omit
  GLM, and `:263` still lists split presentation as deferred.
- **`serve-opencode.md`:** no GLM; no note on the reasoning-replay fix
  (#13); the DS4 "no tools" line is stale.
- **`H5-DFLASH.md`:**
  - It has no family-support statement.
  - Speculation for a recurrent family needs per-token state checkpoints,
    as the Qwen GDN rollback does, not a claim that rewinding is impossible.
  - `H4-MTP.md` does not record GLM's NextN block 45 (deferred).
- **`BENCH.md`:**
  - No family recipes and no GLM entry.
  - The `glm53-flash` registry row (`models-families.toml:75`,
    `lcpp = false`) and its out-of-lock oracle comparator are unexplained.
  - No family sweep includes GLM.
- **`PERF-TOOLS.md`:** no GLM profiling recipe (stage profiler, dispatch
  screens). `gpu_limiter_capture.py` has no GLM alias, although roadmap #2
  prescribes a GLM expert limiter profile.
- **PERF-ROADMAP:** the baseline snapshot (:2750) has no GLM row, and its
  K2 entry is superseded. Row #11's cost cell still describes the
  pre-landing state.
- **`GLM53-FLASH-PLAN.md`:**
  - Status omits #12.
  - Tools still listed as deferred; the anchor link to `SERVE.md` is broken.
  - The P6 "Next" list is stale.
  - llama.cpp reference numbers are mixed (22.6 vs the placement-matched
    23.7).
  - `p2-baseline` packet orphaned.

## D. Code organization and kernel sharing

- **No GLM-private kernels.** Every GLM GPU path is a shared `metal/`
  encoder. All host waits use `wait_completed`.
- **Layering inversion:** `metal/kda.rs` and `metal/indexer_pool.rs` import
  `crate::glm5_next::oracle`, a production `pub mod` used only by tests.
  Move the reference math under `metal/`, or gate it to tests.
- **Family names on shared code:** GLM calls
  `encode_ds4_shared_swiglu_q6_k_f32`; the kernels behind the neutral
  encoders are `glm53_kda.metal` and `glm53_indexer.metal`.
- **GLM-internal duplication:**
  - The 5-encoder sparse pipeline appears in both decode and packed
    prefill.
  - The output head is encoded three times.
  - Command-buffer creation is repeated.
- **Promote GLM's checked host buffer read/write helpers to `metal/`.**
  About 20 unchecked copies elsewhere (DS4, K2, Muse) cast Metal contents
  pointers directly. This is the real content behind clippy's 79 non-test
  `cast_ptr_alignment` hits; alignment holds by construction today.
- **CPU-side duplication shared with K2:**
  - tool stream, partition and run-lane ending;
  - tool byte budget, computed in three places with different fallbacks
    (`usize::MAX` vs `0`);
  - sampling defaults;
  - metadata readers;
  - admission shape;
  - prefetch policy;
  - the "what fits" advice, written twice.

  Converging the ledger removes most of this.
- **Prefill rows decided by four different rules** (run, serve, bench,
  lens). The prefetch threshold 0.98 is typed twice; DS4 names its
  constant. Hyper-connection streams are hard-coded as 4 in shapes.
- **Error typing:**
  - `Glm5NextMetalError` flattens `MfError` into `Invalid(String)`.
  - Poisoning and cancellation are strings.
  - Memory refusals cannot be distinguished, which is what B6 needs.
- **Tests:**
  - `glm5_next_metal/tests.rs` is 2,995 lines with 18 of 20 tests ignored.
    K2 splits its tests into about 8 files.
  - CPU-reachable GLM logic with no CPU test: poisoning after a failed
    command (the `fail_next_completion_check` seam exists), sparse
    visibility arithmetic, admission, the prefix-reuse state machine.
  - `kl_divergence`, `argmax`, `max_abs_diff` and `assert_finite` are
    duplicated across families; they should live in one logits
    test-support module.
- **Bring-up shortcuts:** no TODO/FIXME left. There are `expect`s in packed
  paths (`expect("packed scratch")`; pass `&PackedScratch` instead) and ten
  `too_many_arguments` allows without justification.

## E. Lints, understood

**Default level: 302 warnings in qwen-llm and qwen-cli.**

| Lint | Count | Reading | Action |
|---|---|---|---|
| `chunks_exact_to_as_chunks` | 141 | New lint in this toolchain; constant-size chunking → `as_chunks::<N>()` (typed arrays). Mostly oracles and tests | Mechanical modernization; do it file-by-file with the files it touches anyway, to avoid colliding with concurrent work |
| `needless_borrow` | 39 | Style | Fix with `--fix` |
| `needless_range_loop` | 15 | The May 2026 policy requires an allow with a justification; these 15 are new and unannotated | Review each: iterate, or annotate |
| `result_large_err` | 13 | Lens HTTP `ApiError` is large | Box the payload |
| `drop_non_drop` | 2 | `drop()` on values without Drop (`full_lens/tests.rs:702`, `qwen4exp_runtime.rs:4293`); no-ops | Remove; confirm intent |
| `single_range_in_vec_init` | 8 | Test tables of one range; intended | Accept, or write `vec![(0..n)]` |
| `large_enum_variant` | 6 | Lens and control enums | Box the large variant where it travels by value |
| The rest | ~78 | Small style items | Fix with the files touched |

**Pedantic: 16,666 hits, 471 in the GLM lane.**
- **Style lints, not adopted workspace-wide:** casts (`cast_possible_truncation`
  3,715, `cast_precision_loss` 1,923, `cast_lossless` 1,167), `doc_markdown`
  1,355, `missing_errors_doc` 1,154, `items_after_statements`,
  `too_many_lines`, `must_use_candidate`. Volume far exceeds value, and a
  sweep would collide with concurrent work.
- **Worth acting on, narrowly:**
  - `cast_possible_truncation`/`cast_possible_wrap` on **untrusted
    metadata and sizes** (GGUF header values → `u32`/`i32`/`usize`, token
    ids): use `try_from` there. Positions bounded by capacity are fine.
  - `cast_lossless` in the GLM lane (91): `u64::from` is free clarity.
  - `ignore_without_reason` (122): every ignored test should say why.
    Model-scale GLM tests already do.
  - `cast_ptr_alignment`: see section D (shared checked helpers).
- **Checked and not bugs:** `manual_midpoint` (29 non-test sites, all float
  medians).

## F. Change risk (`cargo crappy`, qwen-cli only)

- **2,930 functions flagged.** The top 40 are lane, bench and lens drivers
  at 0% unit coverage with cyclomatic complexity 39-137. Examples:
  - `bench/mtp::run_mtp` (CC 102)
  - `qwen/deepseek_v4::run_deepseek_v4_single_turn` (CC 137)
  - `qwen/dflash::generate_dflash` (CC 125)
  - `serve/backend::generate` (CC 84)
- **GLM: 51 flagged.** One is high: `qwen::glm5_next::run` (CC 54). Then
  `plain_logit_lens::glm5_next::Prepared::execute` (CC 22) and the serve
  backend's `generate` (CC 15). The renderer, tool output and admission
  helpers are well covered.
- **Reading:** risk concentrates in monolithic lane drivers that only
  GPU-scale runs exercise. The remedy is structural, not more tests around
  the monoliths. A shared lane skeleton (admission, rendering, output,
  stats) would make each part CPU-testable, which is the same work as
  converging ledger rows 1-8.
- **qwen-llm: not measured** (see Method).

## G. Open product items (existing roadmap rows, re-read)

- **#12, Fast lineage:** a known failing qualification; the default-Fast
  investigation stays open.
- **#13:**
  - The client is fixed.
  - The serve-side cost on GLM comes from ledger row 14, not from a model
    limit.
  - The fix is a snapshot codec, as the established families have.
    Flash-Next's snapshot (GDN state, QSA pending block, compressed index
    keys, K/V rows) maps almost one to one onto GLM's KDA conv and state,
    indexer pending ring, pooled keys and MLA latent rows.
  - Apply PERF-LOG's Qwen lesson on choosing the capture point.
- **#14:** output memory across families (packet 2026-10-06).
- **DFlash/MTP for GLM:** feasible in principle with per-token KDA
  checkpoints, as the GDN rollback does. NextN block 45 remains deferred.
  The spec should state family support honestly.

## H. Batch plan

Each batch is reviewed, verified and integrated separately; the roadmap
and PERF-LOG are updated as it lands.

1. **`qwen run` convergence for GLM and K2:**
   - literal output (ledger rows 1-3);
   - the stats line and the request-stats total in the established format,
     with cold start labelled, not warmed (rows 4-5);
   - refusals through shared admission and the drafter resolver (rows 7-8);
   - the history rendering fix (row 9);
   - text `qwen info` per family (B3);
   - the Muse drafter refusal (B2).
2. **`c640a113` re-qualification:** route expectations derived from the
   dispatch table, Flash-Next preflight, GPU runs of the three failing
   tests with few-row on and off, the release-weight tests where the files
   exist, and the full qwen-llm suite (B5).
3. **Serve house rules:**
   - snapshot-policy flags refused on families without snapshots (B1);
   - memory refusals → 503 across families (B6, row 15);
   - K2 tool pricing (B7);
   - K2 sampling run = serve (row 6);
   - log lines (rows 12, 17).
4. **Docs truth pass:** C, B8, ENV.md generator and regeneration (B9), the
   GLM plan refresh, and PERF-ROADMAP's baseline snapshot.
5. **Organization:**
   - the layering inversion;
   - checked host helpers promoted;
   - the shared logits test-support module;
   - the GLM tests split;
   - the sparse pipeline and output head de-duplicated;
   - error typing;
   - default-level clippy cleared in the files touched;
   - `ignore_without_reason`;
   - checked casts on untrusted metadata.
6. **Roadmap rows, not batches:**
   - GLM snapshot codec (replaces the live-session shortcut; enables
     branching reuse and durable snapshots);
   - serve policy decisions (K2 raw string input, strict vs lenient grammar
     failure, prefetch and warm-up across no-copy families, lifting lossless
     decoding);
   - a shared lane skeleton;
   - a GLM request bench and llama.cpp comparator documentation;
   - the DFlash family-support statement.

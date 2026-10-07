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
- **A "structural" justification must be checked against how an
  established family solved the same problem, and against recorded intent.**
  A comment or doc claim is not evidence.
  - Example: GLM's docs said "KDA recurrent state cannot rewind" to justify
    a single live session.
  - The Qwen 3.x hybrids interleave gated delta net 3:1 with full attention,
    and Flash-Next adds QSA. Both have recurrent state, and both solve reuse
    with snapshots at boundaries.
  - The 2026-10-03 plan said it correctly ("prompt edits need replay or a
    compatible checkpoint"). The qualifier was dropped in the serve commit
    `aff7e84a`.
  - The product intent (owner, 2026-09-23, session history): every family
    reuses prefixes across multi-turn requests, by transcript-boundary
    snapshots, or by truncating live KV to the longest common prefix for
    attention-only families, and later persists snapshots durably.
- **Convergence direction is chosen per item.** Sharing a mechanism is
  distinct from enabling a policy everywhere. Established-family drift
  (section B2) is never a convergence target.
- **Warnings are understood and ranked, not swallowed or blanket-fixed.**

Review: cx session 01a10cc. The first draft (`019e0e3f`) was corrected
after an adversarial review and recall checks; the corrections are folded
in below.

## A. Divergence ledger (GLM and K2 vs established families)

Class:
- **S**, structural: forced by architecture or artifact; keep and
  document.
- **P**, product policy: an intentional contract; keep, or change only by a
  named decision with migration.
- **I**, inherited: a bring-up convention with no reason; converge.
- **M**, share the mechanism: lift the implementation, keep per-family
  policy.

| # | Surface | Established | K2 | GLM | Class |
|---|---|---|---|---|---|
| 1 | `qwen run` chat output | literal decoded text on stdout, markers included (Qwen3.6 pre-opens `<think>` and prints literally) | reasoning to stderr, answer to stdout (`qwen/chat_output.rs`) | same, copied in `25eb7936` | **I**: converge to literal (definition in H) |
| 2 | `qwen run` with tools | literal model text | a Responses JSON object, a documented replay interface (`K2-HORIZON-REVIEW.md:1826`, `scripts/reference/k2/check_tools_cli.py`) | same | **P**: keep structured output, but explicit (an output-format option) instead of implied by declaring tools |
| 3 | `qwen run` exit status | a token limit succeeds | a token limit inside reasoning succeeds; an invalid stop before the close fails (`partition_k2.rs:223` tests the distinction) | same | **P**: keep the distinction; with literal output, `reasoning_closed` stays a diagnostic |
| 4 | Stats line | `<fam> stats: … stop_reason= tokenizer_ms … total_ms` | own fields | own fields; prefill includes the weights' first GPU use, unlabelled | **I**; cold start is labelled, never warmed |
| 5 | `--request-stats-jsonl` timing | `total` = wall minus load | sum of phases | sum of phases | **I** |
| 6 | Sampling defaults | Qwen/DS4/Flash-Next: run = serve; Muse: the release preset in both | run uses CLI defaults; HTTP uses its raw defaults (recorded as intentional) | the release preset in both (Muse precedent) | GLM **P** (keep); K2 **P**, changed only by a named decision; **M**: share default resolution with per-family values and explicit-zero overrides |
| 7 | Effort and no-thinking | release-supported levels; `CapabilityError` codes | private codes | private codes | levels **S**; refusal codes **I** (shared codes do not imply identical levels) |
| 8 | Drafter refusal | shared `drafter_policy` resolver, `family_no_speculation` | private, bypasses the resolver | same | **I** |
| 9 | History missing reasoning | rendered as empty reasoning | filled and reported | `qwen run --messages` uses the template's inline `</think>` split, while serve fills empty reasoning: **CLI and serve disagree** | **I** (bug) |
| 10 | `qwen info --json` | `{reasoning, input, template}` | adds `execution` | adds `execution` | **M**: standardize an additive capability schema (implementation, artifact, device, qualification); do not delete `execution` |
| 11 | Serve body decoding | serde decoding | **lossless** (`request_profile.rs:47`; GLM took it from K2); own parser; unknown fields 400; string `input` = raw completion | lossless; shared parser; string `input` = user message | lossless decoding **M**, after checking every renderer's serde assumptions end to end; K2's raw-string contract **P** |
| 12 | Serve limits and warnings | shared `fixed_session_limits` | private `limits()`, duplicate warning | shared | **I** (K2) |
| 13 | Serve startup | admission before execution everywhere; Flash-Next has explicit preflights | none beyond admission | device preflight, prefetch, 2-token warm-up | split four ways: admission **M**; pipeline compilation **M**; prefetch machinery **M** (policy per family: a blanket prefetch can evict its own working set); serve warm-up **P**, needing readiness and cost accounting. No warm-up in `qwen run` |
| 14 | Prefix reuse | transcript-boundary snapshots (Qwen, Flash-Next); DS4 prompt-end and completed only | live session, longest-common-prefix rewind: **intended** for attention-only KV (2026-09-23) | live session, exact extension only | K2 **P** (keep); GLM **I**: below the stated intent (needs snapshots; contract in G) |
| 15 | Memory refusal status | Qwen request admission 503 `memory_admission_denied` | session allocation → 500; tool block unpriced | tool block admitted (503); session allocation → 500 | **M**: typed pressure refusals mapped to 503, after error typing; never map invalid geometry or GPU faults to 503 |
| 16 | Grammar failure in serve | Qwen/DS4 lenient (an unparsed tool block becomes text); Muse strict 500 | strict | strict | **P**: decide per case (budget truncation, malformed generated call, invalid client input, engine fault); never publish partial calls; leniency is not automatically the target |
| 17 | Serve log lines | `serve phases` includes `tokenize_ms` | missing; stale "raw only" labels | missing | **I** |
| 18 | Bench adapters | DS4 logs a skipped prefetch; `muse-request` uses the chat renderer | `k2-request` raw, greedy | suite rows only; silently skips the prefetch run does | raw greedy controls **P** (keep); **I**: add chat/request benches and label renderer, sampling, placement, warm-up and lineage |
| 19 | Pinned release geometry | per-family release profiles | pinned | 45 executed blocks, stored NextN, 4 streams, sparse geometry | **S**: centralize validated geometry; do not relax admission to remove constants |

## A2. Capability and qualification matrix (summary)

Columns: Qwen dense (D), Qwen MoE (M), Flash-Next (FN), DS4, Muse (Mu), K2,
GLM (G). "qual" names the evidence; "impl" without "qual" means none was
found.

| Lane | D | M | FN | DS4 | Mu | K2 | G |
|---|---|---|---|---|---|---|---|
| Serve reuse | transcript-boundary + prompt-end + completed; qual | same | transcript-boundary + completed; engine qual only (no serve test) | prompt-end + completed, **no transcript boundary** | live LCP; qual | live LCP; qual | live exact extension; Fast warm reuse **fails** #12 |
| RAM snapshot codec / durable | yes / yes (default) | yes / yes | yes / refused | yes / yes | – / refused | – / refused | – / refused |
| Snapshot ttl/max-age/half-life | honoured | honoured | honoured | honoured | **ignored** | **ignored** | **ignored** |
| `qwen run --drafter` | DFlash/DFlash2 impl + qual | accepted, serial, **unqualified** | refused (shared) | refused (shared) | **silently ignored** | private refusal | private refusal |
| `serve --drafter` | impl | refused | refused | refused | refused | private | private |
| MTP | bench only | bench only | – | – | – | – | – (NextN block 45 deferred) |
| Plain logit lens | impl | impl | refused | refused | impl | impl | impl, content identity **not computed** |
| Fitted/imported lens, interventions | impl | impl | plans/`lens run` | refused | impl | transport only | refused |
| Native `/lens` jobs | Qwen3.6/3.8 house style | same | unsupported | unsupported | unsupported | unsupported | unsupported |
| Template pinning | name fields only; oracles in tests | tokenizer gate | metadata profile | digests | digests + content id | digests | digests |
| CLI concurrency/batch | greedy concurrency; batch 8/16 | same | refused | concurrency (residency set off) | refused | refused | refused |
| Prefill cancel | chunk ticks | same | ticks | ticks | history dropped | discarded | resumes committed prefix; bitwise |
| Idle residency | refused (copied weights) | same | yes | yes | yes | yes | yes, plus warm-up |
| Tool block priced | no | no | no | no | no | no | yes |
| Session refusal status | 503 | 503 | 500 | 500 | 500 | 500 | 500 |
| `max_output_tokens` default | 65,536 | 65,536 | required flag | 65,536 | required | required | required |
| Text `qwen info` | right | **blank** (`qwen35moe.*` keys) | **wrong** | own branch | **wrong** | **wrong** | **wrong** |
| Request bench | – | – | – | – | `muse-request` | `k2-request` (raw) | none |
| llama.cpp comparator | yes | yes | yes | yes | yes | no | no (out-of-lock oracle pin) |

## B. Correctness and house-rule findings

### B1. Fix now

1. **Snapshot-policy flags silently ignored** on GLM, K2 and Muse:
   `--snapshot-idle-ttl-secs`, `--snapshot-max-age-secs` and
   `--snapshot-half-life-secs` (`check_warmth_flags` never reads them,
   `serve/mod.rs:322-353`). DS4 silently ignores
   `--durable-idle-publish-secs`. The snapshot-budget warning fires even
   for `--snapshot-cache-mib 0`.
2. **Muse `qwen run --drafter` is silently ignored.** `7e624311` dispatched
   Muse before the shared resolver and undid `2ebc80c6`'s pre-load refusal;
   `drafter_policy.rs:5-9` and `qwen info --json` still claim it is
   refused.
3. **Text `qwen info` is wrong** for GLM, K2, Muse and Flash-Next (Qwen
   "GDN / full-attn" facts and `qwen35.*` keys), and blank for Qwen MoE
   (`qwen35moe.*` keys) (`main.rs:1170-1235`). `--json` is right.
4. **GLM `qwen run --messages` and GLM serve render the same history
   differently** (ledger row 9).
5. **`c640a113` few-row Q8_0 MMA left three failing route tests on main**
   (`qwen4exp_metal` packed HC, `qwen4exp_gdn` packed motor,
   `qwen4exp_moe` packed common MoE).
   - The routing reaches every default-dispatch caller for Q8_0 at 2..=8
     rows (N=8 included): Flash-Next, K2 and the lens readouts. The session
     that made it recorded them as "not separately re-qualified", and ran
     only the two new unit tests.
   - Three ignored tests that use the release weights (PLE, QSA, MoE)
     carry the same expectations.
   - Flash-Next's preflight lists omit the few-row pipelines, so these are
     built lazily mid-forward.
   - GLM and DS4 opt out through their dispatch policy.
   - **Plan (reviewed):** contain first. Flash-Next opts out until it is
     qualified, as GLM and DS4 do, which keeps the demonstrated Qwen and
     DFlash gain.
   - Route expectations stay **independent of the selector**: written out
     per row count (1, 2-8, 9) with the device thread-limit fallback, so a
     routing regression stays visible. A table-driven check is used only
     for preflight coverage.
   - **Qualification bar** before any family is promoted:
     - shape and tail numerical checks;
     - Flash-Next HC, GDN and common-MoE tests, plus the release PLE, QSA
       and MoE tests;
     - continuation and reuse checks, and lens checks;
     - a representative performance A/B.

     Existing tolerance failures are investigated, never loosened
     opportunistically. "Structurally closer to serial" is not a
     qualification.
   - K2 and the lens callers join the same qualification ledger.

### B2. Established-family drift (never a convergence target)

- DS4 has no transcript-boundary capture: a turn resent unchanged cannot
  hit.
- Qwen MoE accepts `--drafter` in `qwen run` without qualification.
- Qwen identifies its release by name fields only; Flash-Next has only a
  tokenizer gate.
- Native lens jobs use a metadata identity, not a content hash; Qwen3.5
  has no native profile.
- Flash-Next serve has no in-crate serve test.
- Qwen/DS4 lenient tool-block handling (ledger row 16).
- Stale serve help: `cli.rs:42` says "K2: raw only" and omits GLM and
  Flash-Next.

### B3. Errors and admission

1. **Session memory refusals return 500** on GLM, K2, DS4 and Flash-Next.
   Error typing comes first: map only typed pressure refusals to 503.
2. **K2's tool block is not priced against memory**; neither are Qwen's,
   DS4's or Muse's (#14).

### B4. Wrong claims in code and docs

- "cannot rewind" as a justification: `SERVE.md:897`,
  `GLM53-FLASH-PLAN.md:75`, `backend_glm5_next.rs:2`, `decode_loop.rs:109`,
  and the #13 packet.
- `qwen/glm5_next.rs:7` says "tools not implemented".
- The admission envelope still calls K2 raw-only, and the `serve limits`
  label reads `raw_input_string_only` when chat is verified.
- The GLM plan lists tools as deferred.

### B5. ENV.md

- ENV.md has never covered GLM or K2: the generator's prefix filter
  (`scripts/reference/env_knobs.py:18-19`) excludes `GLM*` and `K2_*`.
- It has not been regenerated since 2026-09-07: 160 of 362 source-line
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
  paths (`expect("packed scratch")`; pass `&PackedScratch` instead). The
  ten `too_many_arguments` allows are covered by the workspace's FFI-shaped
  encoder rationale.
- **Sharing is not qualification.** A neutral encoder name does not
  establish cross-family qualification. De-duplicate only while preserving
  dispatch order, scratch lifetimes and arithmetic policy.

## E. Lints, understood

**Default level: 302 warnings in qwen-llm and qwen-cli.**

| Lint | Count | Reading | Action |
|---|---|---|---|
| `chunks_exact_to_as_chunks` | 141 | New lint in this toolchain; constant-size chunking → `as_chunks::<N>()` (typed arrays). Mostly oracles and tests | Mechanical modernization; do it file-by-file with the files it touches anyway, to avoid colliding with concurrent work |
| `needless_borrow` | 39 | Style | Fix with `--fix` |
| `needless_range_loop` | 15 | The May 2026 policy requires an allow with a justification; these 15 are new and unannotated | Review each: iterate, or annotate |
| `result_large_err` | 13 | Lens HTTP `ApiError` is large | Box only on measured or contractual need, not for the warning count |
| `drop_non_drop` | 2 | Not no-ops in intent: they end a writer borrow and a closure's captured borrow (`full_lens/tests.rs:702`, `qwen4exp_runtime.rs:4293`) | Replace with a lexical scope if touched |
| `single_range_in_vec_init` | 8 | Test tables of one range; intended | Accept, or write `vec![(0..n)]` |
| `large_enum_variant` | 6 | Lens and control enums | Same rule: only on measured need |
| The rest | ~78 | Small style items | Fix with the files touched |

**Pedantic: 16,666 hits, 471 in the GLM lane.**
- **Style lints, not adopted workspace-wide:** casts (`cast_possible_truncation`
  3,715, `cast_precision_loss` 1,923, `cast_lossless` 1,167), `doc_markdown`
  1,355, `missing_errors_doc` 1,154, `items_after_statements`,
  `too_many_lines`, `must_use_candidate`. Volume far exceeds value, and a
  sweep would collide with concurrent work.
- **Worth acting on, narrowly:**
  - `cast_possible_truncation`/`cast_possible_wrap` in **boundary
    modules** (GGUF metadata, header sizes, token ids, wire input): use
    `try_from`, and validate arithmetic before converting. Casts are not
    all style; elsewhere they are reviewed with the code they sit in.
  - `cast_lossless` in the GLM lane (91): `u64::from` is free clarity.
  - `ignore_without_reason` (122): every ignored test should say why.
    Model-scale GLM tests already do. **Adopt `ignore_without_reason =
    "warn"` in `[workspace.lints]`** once the backlog is cleared.
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
  GPU-scale runs exercise.
  - Complexity scores alone do not justify a restructure.
  - Converging ledger rows 1-9 moves admission, output and stats into
    shared, CPU-testable code anyway.
  - A broader "shared lane skeleton" waits for stage attribution and
    request benches.
- **qwen-llm: not measured** (see Method).

## G. Open product items (existing roadmap rows, re-read)

- **#12, Fast lineage:** a known failing qualification; the default-Fast
  investigation stays open.
- **#13:** two problems, two fixes.
  - Client reasoning replay: fixed in opencode. Conditioning was the
    problem, and snapshots cannot restore omitted conditioning.
  - Branching reuse: GLM needs snapshots (ledger row 14), the way the
    established hybrids reuse.
- **GLM snapshot state contract** (design packet before any code). Flash
  Next is a design reference, not a drop-in mapping: it also has PLE
  history state and a different pending-buffer representation.
  - **At committed length `n`, capture:**
    - each KDA block's conv tail and recurrent state;
    - the MLA latent prefix (`n` F16 rows per block);
    - the complete pooled keys through `floor(n / pool)`;
    - the pending key-and-gate ring with its phase.

    A non-pool-aligned capture keeps the F16 rounding and the incomplete
    pool exactly; reconstructing from pooled keys is not enough.
  - **Specify:**
    - committed vs emitted tokens (a pending token);
    - who owns the logits;
    - arithmetic lineage (Exact vs Fast; #12);
    - identity;
    - capture only after a healthy, completed command;
    - restore failure;
    - memory admission.
  - **Capture points:** the transcript boundary (PERF-LOG 2026-09-23: the
    generation header's start), plus prompt end and completed as the Qwen
    backend does.
  - **Tests:** every pool residue, the sparse frontier, and branch, edit and
    cancel continuations. Restore followed by continuation must be bitwise
    against a cold run in the Exact lineage.
  - RAM checkpoints do not provide durable snapshots; durability is a
    separate step.
- **#14:** output memory across families (packet 2026-10-06).
- **DFlash/MTP for GLM:** feasible in principle with per-token KDA
  checkpoints, as the GDN rollback does. NextN block 45 remains deferred.
  The spec should state family support, separating flags accepted,
  weights available and qualified execution.

## H. Batch plan (reviewed)

Practice:
- Small reviewed commits in this worktree (`feat/glm53-p6`), never in the
  shared main worktree.
- Integrate only after the overlap check against the main worktree's
  uncommitted files.
- No whole-tree formatting or `clippy --fix`.
- Docs are updated with each batch, not in a late cleanup.
- GPU work waits for the lease; another session shares the GPU.
- Recorded contracts (recall, docs, scripts) are checked before any
  behaviour changes.

0. **This packet:** corrected classifications, matrix, plans.
1. **CPU-facing correctness:** the ignored flags (B1.1), the Muse drafter
   refusal (B1.2), text `qwen info` per family (B1.3), and GLM history
   rendering (B1.4). Golden prompt and token fixtures, omitted vs explicit
   values, and refusal tests.
2. **`qwen run` presentation and stats** (ledger rows 1-5, 7-8):
   - **Literal output:** the decoded generated bytes in order, including
     generated reasoning and tool markers, with no synthesized prompt
     opener and the lane's stop handling unchanged. A token limit inside
     reasoning still succeeds; an invalid stop before the close still
     fails; `reasoning_closed` stays a diagnostic.
   - **Structured output:** a Responses JSON object for families with
     verified tools, requested by an explicit option rather than implied
     by declaring tools. `check_chat_cli.py`, `check_tools_cli.py` and the
     docs migrate in the same change.
   - **Stats:** established format; cold start labelled.
   - **Refusals:** shared codes and the shared drafter resolver.
   - **Verification:** scheduled GPU checks compare generated token
     fingerprints and both tool round trips.
3. **Errors and admission:** typed errors; pressure refusals → 503 (B3);
   tool pricing beyond GLM (#14). Then, separately and only by named
   decision, decoding and default-policy changes (ledger rows 6, 11).
   JSON/SSE fixtures, malformed/duplicate/sentinel bodies, memory peaks,
   and abort, control-plane and residency checks.
4. **`c640a113` containment and qualification** under the GPU lease (B1.5);
   promote only families that pass. Shared checked host helpers and small
   organization items (section D) follow.
5. **GLM RAM snapshots:** a design and qualification packet first (section
   G); durable persistence after. Prefetch and warm-up policy changes need
   their own cold, warm and pressure measurements.
6. **Docs truth pass**, alongside each batch: section C, B4, the ENV.md
   generator and regeneration (B5), the GLM plan refresh, and
   PERF-ROADMAP's baseline snapshot. Historical PERF-LOG measurements stay
   intact; current baselines are added with date, lineage, placement and
   debug-layer status.

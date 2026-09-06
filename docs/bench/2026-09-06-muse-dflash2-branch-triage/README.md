# Muse DFlash2 branch triage (feat/muse-glimmer-text)

Read-only review of the unmerged branch (6 commits, tip `864d66ec`, merge-base `af816bb8`) against main `75e30353`, produced 2026-09-06 with cx ask (default model). Decision: defer the port; do not merge wholesale.

## Verdict

- The branch's Muse verifier is serial: T generated tokens still cost T-1 target forwards. It can demonstrate correctness and overhead, not speedup. No Muse speedup packet exists on the branch; the only speedup numbers there are inherited Qwen3.8 results.
- Recoverable: ~2,344 lines of runtime (drafter geometry/capture semantics in loader.rs, layer-input capture bank, target-neutral draft core extraction, explicit noncausal attention, admission, runner, greedy CLI loop and counters) plus ~318 lines of cheap tests.
- Leave behind: ~631 lines of theater (whole-drafter hashing on load, pinned release SHA/size/census gates, source-text contract tests, duplicate validators) and ~214 lines of unrelated prompt/tokenizer changes.
- Semantic conflicts with main: resident sessions are borrowed, not consumed; packed prefill must carry capture; declared-context admission replaces the branch's 7,168 cap; metal.rs is split; ModelFamily::MuseGlimmer; pinned ATEM prompt bytes.
- Estimate: 3-5 days for a serial-verification port with measurements; 1-2 weeks more for a packed Muse verifier (all-row decisions, prefix commit/rollback, numerical qualification), which is where any speedup would come from.

## Full report

```
**Recover the drafter adapter, layer-input capture, numerical fixes, and serial correctness loop. Do not cherry-pick the branch wholesale.** It contains a working Muse proposal path, but its serial verifier does not amortize target computation, and the branch provides no recorded Muse speedup measurement.

I inspected main at `75e30353`, branch tip `864d66ec`, and merge-base `af816bb8`. The branch is **6 commits ahead / 117 behind**, rather than approximately 75 behind. No files were modified or tests executed. Main had concurrent uncommitted edits; the comparison below concerns committed main.

**1. Commit inventory and classification**

The requested log and three-dot diff report **19 files, 3,621 insertions, 539 deletions**. The `metal_dflash.rs` figure of 1,559 is total changed lines: **1,180 additions and 379 deletions**, not 1,559 new lines.

All six commits are dated 2026-08-29:

| Commit | Classification and disposition |
|---|---|
| `0d3b12c9` — close prompt and tokenizer identity gaps | **C:** retained-FD/stamp machinery in `gguf.rs`, `tokenizer.rs`, residency commentary. **D:** prompt/date changes, model-info routing, tokenizer call-site changes in `main.rs` and a session test. These are separable from DFlash; leave this commit behind. |
| `3d6ead93` — bind strict dflash2 release contract | **A:** generic target geometry/capture semantics in `loader.rs`; drafter epsilon/scale/cap execution; shared softcap kernel; CPU-oracle rejection of unsupported DFlash2; Qwen accessor plumbing in `bench.rs`; module export. **B:** numerical transform test. **C:** most of new `muse_glimmer_dflash.rs`: pinned release identity, exact census, authentication, and metadata-only tests. Recover selected hunks. |
| `c8c3965b` — capture dflash layer inputs | **A:** capture bank, pre-normalization copies, forward/prefill/runtime entry points. **B:** small layout/range tests. Real-model capture comparisons are useful but expensive. **D:** retained-tokenizer changes in `muse_lens_fit.rs` and `muse_lens_run.rs`. Recover capture hunks against main’s newer execution paths. |
| `de4e70cc` — run greedy dflash2 decoding | **A:** target-neutral drafting, explicit noncausal attention, session publication, memory admission, Muse runner, CLI loop and counters. **B:** attention oracle, invalid dtype and zero-group tests, greedy CLI policy. **C:** source-text “contract” assertions, duplicate validation, authenticated CLI binding. This contains most of the useful work. |
| `fb741bb2` — preserve intervention-aware capture | **A:** pass explicit empty intervention slices through capture calls. Its removal of the old `execute_token` wrapper must **not** be copied onto main, where scalar prefill still calls that wrapper. |
| `864d66ec` — publish dflash2 invocation | Supporting documentation. Recover the command and accurate limitations after implementation; remove “authenticated/matched release” framing and update stale claims about main’s capabilities. |

**2. File and hunk ledger**

Counts below approximate **added lines**, including replacement lines—not final implementation size. A includes runtime safety and necessary integration glue. B means small behavioral tests. **B† identifies real-model tests that do not meet the requested “cheap” criterion.** D includes separate prompt/tokenizer work and supporting documentation; it does not mean every such change is bad.

All links in this ledger point to the branch worktree.

| Changed file | Approximate allocation | Hunk classification |
|---|---:|---|
| [README.md:3–83](/Users/tito/code/qwen-llm-muse-glimmer/README.md:3) | D 23 | Family listing and invocation documentation. Useful supporting material, not runtime. |
| [qwen-cli/bench.rs:13573–13702](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/bench.rs:13573) | A 3 | Adapt Qwen callers to `decoder.base()` / `head()`. |
| [qwen-cli/main.rs](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/main.rs:4267) | A 270; B 36; C 5; D 82 | **A:** imports, drafter admission at 3432–3435, loading/runner selection at 4267–4440, counters at 4470–4485, loop at 9765–9911. **B:** 12552–12585 greedy policy. **C:** authenticated binding at 4274–4278. **D:** prompt preparation 2912–2962, retained tokenizer, model-info changes 12335–12452. |
| [muse_lens_fit.rs:48](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/muse_lens_fit.rs:48) | D 1 | Retained-tokenizer signature migration. |
| [muse_lens_run.rs:115](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/muse_lens_run.rs:115) | D 1 | Same migration. |
| [forward.rs:24–670](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/forward.rs:24) | A 14 | Reject unsupported convolution/selector CPU oracle at 393–397; use artifact RMS epsilon at seven normalization sites, 441–670. Useful shared-path correctness, although not the Muse GPU execution path. |
| [gguf.rs:262–269](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/gguf.rs:262) | C 8 | Clone retained primary file for the tokenizer continuity machinery. |
| [lib.rs:54](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/lib.rs:54) | A 1 | Module export; retain only if retaining a small Muse adapter module. |
| [loader.rs:342–418](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/loader.rs:342) | A 103 | Target geometry and capture enum; generic loader 843–864; metadata keys and epsilon/scale/cap parsing 884–995; geometry/capture-shift adjustments 1026–1224; resulting config/head fields 1319–1332 and fixture updates. |
| [metal.rs:2695–2718](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/metal.rs:2695) | A 161; B 168 | **A:** backing-range helper and explicit noncausal attention wrappers at 21228–22018. **B:** existing oracle plumbing and numerical noncausal test at 47358–47518. |
| [metal_dflash.rs:4323–4745](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/metal_dflash.rs:4323) | A 1,073; B 49; C 58 | **A:** approximately 139 lines pricing, 284 resident validation, 35 session/config changes, 129 direct publication, 160 target adapter/accessors, 315 extraction/rewiring, plus imports. Main ranges: 4997–5023, 5598–5724, 8219–8398, 16401–17717. **B:** 20687–20734. **C:** source-text contract updates around 18971, 20570–20600, and new test 20649–20685. |
| [muse_glimmer_dflash.rs:1–552](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_dflash.rs:1) | A ≈90; C ≈462 | **A:** generic head binding 114–122, compatibility intent 100–112, ordinary Metal loading/admission 166–203, necessary holder/error plumbing. **C:** release constants, hashes, exact sizes, stamps, hard-coded config/census, authentication and metadata-only tests. Rewrite this file small rather than retaining its structure. |
| [muse_glimmer_metal.rs:109–160](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_metal.rs:109) | A 11; B 7 | Generalize scale/softcap to allow cap zero; numerical test additions at 312–356. |
| [muse_glimmer_prompt.rs:96–572](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_prompt.rs:96) | D 106 | Optional date field, runtime clock helper 134–165, profile-specific system handling 200–239, validation and byte tests 494–572. Separate prompt behavior, not a DFlash dependency. |
| [muse_glimmer_residency.rs:6–10](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_residency.rs:6) | C 10 | Authentication commentary and message wording at 141–143, 391–423. The substantive target hashing machinery already existed. |
| [muse_glimmer_runtime.rs:175–245](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_runtime.rs:175) | A 304; B† 62 | Runner creation/admission; capture wrappers 283–297, 329–343, 435–453; DFlash runner/state publication 461–628. Real-model first-candidate test 654–708. |
| [muse_glimmer_text_session.rs:63–188](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_text_session.rs:63) | A 284; B 58; B† 52; D 1 | **A:** capture bank; alias checks 697–722; capture forwarding/prefill 808–922; validation 1080–1116; actual copy 1232–1243; sink 1539; intervention composition. **B:** 1952–2009. **B†:** changed live test 2037–2123. **D:** tokenizer migration 2216. |
| [tokenizer.rs:76–144](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/tokenizer.rs:76) | A 25; C 88 | **A:** architecture-independent vocabulary comparison helper and factoring. **C:** retained `FILE`, stamp checks, global load lock, ownership scaffolding at 235, 316, 386–435, 664–722. |
| [kernels/muse_glimmer.metal:38–52](/Users/tito/code/qwen-llm-muse-glimmer/kernels/muse_glimmer.metal:38) | A 5 | Shared scale/softcap kernel; cap zero means scale only. |

Approximate totals: **A 2,344; B 318; C 631; D 214; expensive real-model qualification 114**.

Several distinctions matter:

- **`DFlashTargetContract` is not theater.** Its three geometry fields feed actual tensor binding. Likewise, capture semantics, supported dtypes, allocation bounds, and convolution dimensions protect real execution.
- The 284-line resident validator is repetitive but checks actual GPU storage. Keep one concise validation boundary. The branch calls `validate_for_draft_target` both in [runner creation:194](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_runtime.rs:194) and [draft-decoder construction:8316](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/metal_dflash.rs:8316); that duplicate call adds no qualification.
- **The release wrapper has a production consumer.** It is inaccurate to call the entire module dead. However, its `artifact_profile()` and `contents_authenticated()` getters are only consumed by its tests, while exact file sizes, release hashes and tensor census unnecessarily gate production loading.
- The branch’s layout preserves **sorted, unique** layer selection; it does not support arbitrary caller ordering. The constructor delegates to that restriction, and the copy uses binary search. [Capture constructor:86](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_text_session.rs:86), [restriction test:1859–1866](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_text_session.rs:1859).

**Authentication finding:** production calls `from_authenticated_release`, whose implementation hashes the complete mapped drafter: **2,959,518,912 bytes Q8 or 5,558,092,992 bytes BF16**. The target runtime instead calls non-hashing `for_release`; the full-target hash function is inherited, not introduced by these commits. [CLI:4274–4278](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/main.rs:4274), [drafter identities:28–44 and hash:431–447](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_dflash.rs:28), [target load:68](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_runtime.rs:68).

**3. What changed underneath it on main**

| Main development | Semantic conflict and porting consequence |
|---|---|
| **Reusable resident sessions**, including `79dcff3b` resident serve | Main stores a session directly and runners borrow it. The branch uses `Option<Session>` and `.take()`, consuming it permanently. Port the DFlash runner as a borrower; retaining branch ownership would break repeated resident requests. [Main runtime:50–57, 179–200](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_runtime.rs:179). |
| **Packed prefill**, `6d489465`, `231d6944`, `d8bdafc2` | Main partitions prefill into packed chunks and scalar tails. Branch capture replaces the old scalar loop. Applying it mechanically either loses main’s prefill acceleration or misses captures from packed chunks. Add capture to packed execution, or explicitly use scalar capture only for the final drafter window during the initial reference landing. [Main session:1613–1660](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_text_session.rs:1613). |
| **Declared context support**, `6e8dbb7b` | Branch retains a 7,168-position reference limit; main admits the declared model context. Do not reinstate that old general limit. Give the initial DFlash path its own tested bound if necessary. [Branch session:40](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_text_session.rs:40), [main session:149–153](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_text_session.rs:149). |
| **Batched full-trace readouts**, `4b59800e`, top-32 extension `9bf94cf9` | Main has reusable transport/readout workspaces and additional output-tail consumers. These process independent residual rows; they are **not a speculative verifier**. Preserve these paths and attach layer-input capture to the existing token graph. A global softcap rename must update both ordinary inference and batched readout. [Main runtime:255–304](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_runtime.rs:255), [session readout:1388–1430](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_text_session.rs:1388). |
| **Metal module split**, `9bdb0a54` | `metal.rs` no longer exists on main. Move attention changes into `metal/dflash.rs`, the tensor helper into `metal/tensor.rs`, and tests into the test module. Preserve old public wrapper behavior; Muse’s explicit noncausal setting must not change Qwen’s default. [Main attention:115–123](/Users/tito/code/qwen-llm/crates/qwen-llm/src/metal/dflash.rs:115). |
| **CLI/bench splits**, `96243673`, `d98cee80`; shared option validation `04702d55` | Land Muse CLI changes in `qwen/muse_glimmer.rs`, Qwen accessor adjustments in `bench/dflash.rs`, and tests in current test modules. Main explicitly rejects Muse `--drafter` through the shared unsupported-option table; remove only that rejection for the supported Muse lane. [Main Muse CLI:81–101](/Users/tito/code/qwen-llm/crates/qwen-cli/src/qwen/muse_glimmer.rs:81). |
| **`ModelFamily::MuseGlimmer`**, `4c275380` | Use the current family dispatch. Do not restore branch-era architecture-string special routing for inspection/loading. [Main family:4–36](/Users/tito/code/qwen-llm/crates/qwen-llm/src/model_family.rs:4). |
| **Annotated/structured ATEM requests**, `2af3b46d`, `2e532605`; reasoning/sampling fix `9b87804e` | Main pins default rendered bytes, supports explicit request dates, and makes selected reasoning authoritative over existing system directives. Branch changes date to runtime-dependent and preserves/appends Unsloth system directives. Those are observable prompt/token differences, not conflict-resolution details. Leave branch prompt behavior out of this landing. [Main pinned bytes:1207–1227](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_prompt.rs:1207), [request rendering:157–188](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_request.rs:157), [system normalization:381–400](/Users/tito/code/qwen-llm/crates/qwen-llm/src/muse_glimmer_prompt.rs:381). |

A favorable finding: **main’s `metal_dflash.rs`, `loader.rs`, and `tokenizer.rs` are unchanged from the merge-base.** Most difficult adaptation is in Muse runtime/session integration and moved files, not competing changes to the drafting algorithm.

**4. Verify/accept comparison**

**The drafter is an extraction of existing Qwen machinery; the Muse accept loop is a serial variant of the same greedy algorithm.** It is not a new speculative acceptance method.

The branch extracts the existing draft forward and selector walk behind a small target surface: context, dimensions, embedding matrix, output matrix and noncausal mode. Qwen’s `DFlashDecoder::draft_block` delegates to that same core. Muse supplies different capture semantics and numerical settings. [Branch adapter:8219–8360](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/metal_dflash.rs:8219), [shared Qwen wrapper:17681–17717](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/metal_dflash.rs:17681).

| Behavior | Branch Muse | Main Qwen |
|---|---|---|
| Proposal anchor | Carry token; draft slot zero discarded | Same |
| Verification | Forward carry, compare next draft, forward only accepted tokens | Packed forward of `[carry, drafts…]` |
| Rejection | Target argmax becomes next carry | Same greedy rule |
| Full acceptance | Target bonus becomes next carry | Same |
| State recovery | No speculative target rollback needed: rejected tokens never enter target state | Restore accepted-prefix GDN/convolution state and KV positions; optional exact replay |
| Sampling | Greedy only | Greedy and sampled sparse-proposal coupling |
| Capture | Direct block inputs, committed rows only | Block outputs from verification, publish accepted prefix |
| Performance policy | Always drafts while generation remains | Adaptive serial fallback/backoff, reprobes, exactness fallback |

Evidence: [Muse loop:9812–9895](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/main.rs:9812), [Qwen packed verify/accept:695–819](/Users/tito/code/qwen-llm/crates/qwen-cli/src/qwen/dflash.rs:695), [commit/restore:942–1010](/Users/tito/code/qwen-llm/crates/qwen-cli/src/qwen/dflash.rs:942), [Qwen state restoration:15549–15583](/Users/tito/code/qwen-llm/crates/qwen-llm/src/metal_dflash.rs:15549).

For **T generated tokens**, Muse still performs **T−1 target decode forwards**, exactly as ordinary serial generation, plus drafter execution and capture. Acceptance improves proposal statistics but does not eliminate target passes. Any performance benefit requires a cheaper multi-token target verifier or another measured execution improvement.

A shared accept loop would need:

- One carry/accepted-prefix/correction-or-bonus state machine, with identical EOS, token-limit and callback semantics.
- A target adapter returning verification decisions and supporting `commit_prefix`; serial verification can implement rollback as a no-op, packed verification cannot.
- Target-owned capture publication and position restoration. Qwen’s GDN checkpoints cannot be reused as Muse state.
- Common counters for proposals, physically verified rows, accepted candidates, committed transitions and phase time.
- Sampling and Qwen’s backoff/exactness policy kept outside the initial shared greedy core.

Main’s packed prefill is useful groundwork, but it does not already expose all-row verification results, speculative prefix commit, or rollback. Batched lens readout alone supplies none of those.

**5. Measurements and qualification**

**No committed Muse DFlash2 speedup packet was found.** None of these six commits changes `docs/bench` or `docs/PERF-LOG.md`, and those locations contain no Muse-specific DFlash measurement.

The commit messages report:

- `0d3b12c9`: “token 24, cosine 0.999999999, relative RMS 5.229603e-5, and maximum absolute error 8.702278e-4.”
- `3d6ead93`: “2.96 GB Q8 release with 81 tensors, 32 F32 support tensors, 49 Q8_0 matrices.”
- `de4e70cc`: “accepts the first live DFlash2 candidate” and serial/upstream token agreement.

These are numerical correctness or artifact-binding claims, not throughput results. The committed live test checks just the first draft candidate against one target transition. [Runtime test:654–708](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-llm/src/muse_glimmer_runtime.rs:654). It does not qualify the complete rejection/full-accept/EOS loop.

There are speedup numbers **inherited from Qwen** on this branch:

> “Serial decode 16.59 tps vs drafter 28.54 tps (1.72x).”

That packet explicitly identifies **Qwen3.8-27B-Q8_0**, a 2,232-token prompt and 128 generated tokens. [Branch benchmark packet:135–146](/Users/tito/code/qwen-llm-muse-glimmer/docs/bench/2026-08-20-windowed-dflash-pre-gates/README.md:135).

The inherited performance log also reports Qwen code speedups of **2.80× / 1.23× / 1.96×** at 8K/32K/64K, versus prose **1.10× / 0.79× / 0.88×**. These cannot qualify Muse. [Branch PERF-LOG:1027–1035](/Users/tito/code/qwen-llm-muse-glimmer/docs/PERF-LOG.md:1027).

The Muse CLI does provide useful counters, but two measurement issues need attention:

- `accepted/scored` measures acceptance among candidates reached before rejection, not accepted/all-drafted. Preserve all three counts.
- Drafter authentication happens before the `load_ms` timer starts. Its full-file read is included in total request time but hidden from reported load time. [CLI:4274–4278, 4342–4354, 4470–4485](/Users/tito/code/qwen-llm-muse-glimmer/crates/qwen-cli/src/main.rs:4342).

**6. Minimal landing recommendation**

Use a **hunk port**, in this order:

1. **From `3d6ead93`:** generic loader geometry, input/output capture enum, artifact RMS epsilon, scale/softcap, and shared numerical kernel. Retain a lightweight target/drafter vocabulary comparison. Replace the 552-line strict release module with a small loader adapter using actual geometry and supported tensor formats.
2. **From `c8c3965b`, incorporating `fb741bb2`:** direct layer-input capture and small layout tests. Attach it before attention normalization in main’s existing graph. Preserve main’s borrowed session ownership and intervention handling.
3. **From `de4e70cc`:** shared draft-core extraction, explicit Muse noncausal attention, direct capture publication, allocation admission, runner, greedy CLI loop and counters. Collapse duplicate validation. Place code in current modules.
4. **From `864d66ec`:** the invocation and a clear statement that the first landing uses serial verification.
5. Add the missing small **behavioral** accept-loop tests: rejection at different positions, full accept plus bonus, EOS and token limit inside a block, and the T−1 committed-transition invariant. Keep real-model capture/first-candidate tests opt-in.

For an initially measurable landing, record a compact same-model, same-tokenized-prompt comparison against current main’s serial path: token equality, prefill time, generation wall time, draft time, target time, draft calls and accepted/drafted/scored counts. Include code and prose, a multi-block continuation, and a prompt crossing the 2,048-token capture window. Freeze the rendered prompt/date; compare actual token IDs. Expect this first serial-verification landing to demonstrate correctness and overhead, not speedup.

**Effort estimate:** approximately **3–5 focused engineer-days** for the selective scalar-verification port, current-main integration and meaningful measurements. A packed Muse verifier with all-row decisions, capture, prefix commit/rollback and numerical qualification is an additional **roughly 1–2 weeks**. These are source-review estimates, not results from a trial rebase.

Explicitly leave behind:

- Whole-file model hashing on normal load.
- Pinned release SHA, revision, exact file-size restrictions and uniform quantization census.
- Authentication/profile bookkeeping without a runtime decision that needs it.
- Repeated validators at adjacent construction layers.
- Source-string tests asserting implementation names or visibility.
- Metadata-only “release qualifies” tests presented as execution evidence.
- Retained-FD tokenizer locking/continuity work and unrelated lens call-site migrations.
- Runtime-date and template-policy changes from `0d3b12c9`.
- Branch session consumption, old context limits, and replacement of main’s packed prefill.
- Any claim that a verified first candidate or successful serial loop establishes speculative speedup.
```

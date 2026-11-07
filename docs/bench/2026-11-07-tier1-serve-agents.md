# Tier-1 serve agent standing, 2026-11-07

Measured current main `dabb97021e65f192a15057b3f226281cbe1ad81f` on M5 Ultra,
using an unmodified source build in an owned M5 worktree. No engine, client
application or model changes. API client scripts live only in the evidence directory.

One resident model at a time, no drafter. Each serial `/v1/responses` session
has four real Rust-source `read_file` call/result turns of about 16K tokens,
then three 1–2K append turns. Actual server reasoning/message items are replayed
without trimming. A fork truncates back to build turn 3 and edits its last user
question, retaining its source/tool history. This is a synthetic agent-shaped
API workload, not an invocation of OpenCode itself.

Four buildup responses request 64 tokens; appends, fork and return controls
request 200. All reached their requested token limits. Greedy, seed 1; V4.1
uses default chat, GLM/Flash-Next low reasoning. Fixed context limit 98,304;
default cache budgets, lineage, precision and idle-residency settings.

TTFT is client time from POST to the first nonempty primary text/reasoning
(or tool-argument) delta; heartbeat/comments and duplicate reasoning aliases do
not count. Decode rate is the server's generated-token/decode-wall statistic.
Flash-Next's three 200-token appends produced reasoning only: these TTFTs are
not first-visible-answer latency. GLM first visible answer followed its first
reasoning delta; V4.1 chat emitted visible text immediately.

Every accepted turn and native control had thermal pressure 0 before and after.
Client requests have 25-second cooling gaps followed by three zero readings
five seconds apart. Build lock, nice 10 and a canonical campaign Metal lease
with wait enabled covered builds/campaigns. Build, all three sessions, seven
native controls and cleanup fit about 45 minutes of M5 wall time. All servers
stopped/reaped; the M5 scratch worktree and cloned target were removed.

## Per-turn measurements

`B1..B4` build the history; `A1..A3` append at approximately 64–70K. The prompt
length is processed + reused. `Return*` returns from the fork to the main
branch while omitting historical reasoning. It changes two conditions, so its
penalty is not an isolated estimate of client reasoning omission.

### V4.1 GSQ-RCO IQ2_XS + IQ4_NL Engram

| Turn | Processed | Reused | TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| B1 | 16,460 | 0 | 14.666 | 75.54 |
| B2 | 16,095 | 16,523 | 14.368 | 75.22 |
| B3 | 16,094 | 32,681 | 15.202 | 74.60 |
| B4 | 16,094 | 48,838 | 15.936 | 73.44 |
| A1 | 1,118 | 64,995 | 2.395 | 73.57 |
| A2 | 1,629 | 66,312 | 3.234 | 73.44 |
| A3 | 1,886 | 68,140 | 3.273 | 73.10 |
| Fork | 16,158 | 32,616 | 15.116 | 74.05 |

### GLM-5.3-Flash UD-IQ3_XXS

| Turn | Processed | Reused | TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| B1 | 16,322 | 0 | 17.240 | 57.13 |
| B2 | 16,060 | 16,385 | 16.895 | 56.79 |
| B3 | 16,060 | 32,508 | 17.344 | 56.16 |
| B4 | 16,060 | 48,631 | 17.771 | 55.65 |
| A1 | 1,084 | 64,754 | 2.194 | 55.20 |
| A2 | 1,596 | 66,037 | 3.018 | 55.14 |
| A3 | 1,852 | 67,832 | 3.449 | 55.05 |
| Fork | 33,994 | 14,573 | 34.339 | 55.66 |
| Return* | 54,923 | 14,573 | 55.953 | 54.88 |

### Flash-Next GSQ-RCO IQ3_S

| Turn | Processed | Reused | TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| B1 | 16,493 | 0 | 9.526 | 69.08 |
| B2 | 16,093 | 16,556 | 9.982 | 68.64 |
| B3 | 16,094 | 32,712 | 9.121 | 67.62 |
| B4 | 16,092 | 48,869 | 9.223 | 66.79 |
| A1 | 1,117 | 65,024 | 1.353 | 66.18 |
| A2 | 1,630 | 66,340 | 1.699 | 66.05 |
| A3 | 1,886 | 68,169 | 1.868 | 65.93 |
| Fork | 16,093 | 32,712 | 9.037 | 67.07 |
| Return* | 52,905 | 16,488 | 28.856 | 65.98 |

## Native and PERF-LOG bridge

PERF-LOG has no exact 64–70K standing for all three artifacts. Fresh same-build,
same-artifact native controls therefore use A2's **exact processed rows/reused
depth** for pp, and A2's **exact prompt depth** for tg200. One retained sample,
`--no-warmup --runs 1 --seed 1`; each depth is filled afresh, with allocation,
loading and prefix fill outside timing. Native inputs are known synthetic IDs,
not the served code/generated tokens. These are descriptive comparisons, not
isolated measurements of network overhead or a statistically qualified rollout.
All native build/runtime identities match clean `dabb9702`.

| Model | PP rows / depth | Native PP s | Serve A2 TTFT s | Excess s | Native TG200 tok/s | Serve tok/s | Serve/native decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 GSQ-RCO IQ2_XS + IQ4_NL Engram | 1629 / 66312 | 1.522 | 3.234 | +1.712 | 71.68 | 73.44 | +2.45% |
| GLM-5.3-Flash UD-IQ3_XXS | 1596 / 66037 | 2.927 | 3.018 | +0.091 | 56.42 | 55.14 | -2.27% |
| Flash-Next GSQ-RCO IQ3_S | 1630 / 66340 | 1.252 | 1.699 | +0.447 | 68.99 | 66.05 | -4.27% |

Existing 32K PERF-LOG decode references are V4.1 **74.81–74.94**, GLM
**57.65**, Flash-Next **71.10 tok/s**; these have shorter depth and tg64,
so they are context, not exact serve penalties. The fresh controls establish
that most recent decode gains carry through serve. Existing V4.1 1300/32K
native append is **1.146 s**; GLM 1024/32K **1.090 s**; Flash-Next 512/32K
**0.364 s**. Larger served suffixes at doubled depth cannot be divided by those
numbers and called overhead.

An additional GLM shape control at the same **66037** depth reads **2048**
rows in **2.090 s / 979.69 tok/s**, versus **1596** rows in **2.927 s / 545.20
tok/s**. The larger aligned append is faster. Inputs/row counts differ; this
is not an isolated switch A/B, but it agrees with the explicit shape gates below.

## Fast paths and gaps

- **V4.1:** warm requests use live state; fork uses RAM snapshot. Each append
  makes two native Full calls: body **1116/1627/1884** rows plus **2** header
  rows. Actual batch counters: width max/mean **1**, **199** plain transitions,
  **0** batched transitions and **0** loaded-prefill calls. Full row ceiling
  **4096**. Exact shrinking CED and its implicit P4 loading profile do not
  apply to serve. The body geometry is eligible for the recent long publication,
  dense-tail and Engram overlap paths; masked reindex is deliberately off below
  128K. Native traces show three overlap submissions/body. Aggregate valid driver
  intervals are **0.993/1.435/1.342 s** per append, with GPU **1.244/1.639/1.812 s**.
  These intervals overlap wait/GPU timing and are not an additive wall budget.
  The tiny two-row header alone costs **119–127 ms GPU**, plus variable scheduling.
  See [Full-only state reuse](../../crates/qwen-cli/src/serve/backend_v41/reuse.rs#L105),
  [cuts](../../crates/qwen-cli/src/serve/backend_v41/execution.rs#L286), and
  [publication gate](../../crates/qwen-llm/src/deepseek_v41/forward/attention_gpu/query_batch.rs#L893).
- **GLM:** Fast/Half with 2048-row workspace and live reuse. Every deep append
  is a single **1084/1596/1852-row** partial command. Those row counts miss the
  qualified IQ tensor gate/up/panels, tensor sparse QK/PV, KDA value4 and wide-Q6
  row gates, although full 2048-row buildup chunks can use them. The engine
  still uses other generic optimized routes; this does not mean tensor operations
  are globally disabled. See [actual command rows](../../crates/qwen-llm/src/glm5_next_metal/packed.rs#L1853),
  [expert gate](../../crates/qwen-llm/src/glm5_next_metal/tensor_gate_up.rs#L51),
  [attention](../../crates/qwen-llm/src/glm5_next_metal/prefill_attention.rs#L20),
  [wide Q6](../../crates/qwen-llm/src/glm5_next_metal/tensor_prefill.rs#L106).
  Only the first turn published a deep checkpoint, at **14573**; all six
  subsequent live-extension turns have empty checkpoint-capture lists. Fork
  restores that first checkpoint and publishes **47341**. The return/omission
  control cannot use 47341 after its earlier reasoning changes and restores 14573.
  This is provenance-limited, not an exhausted cache: the warm cache holds two
  entries / **482.6 MB**. See [live provenance](../../crates/qwen-cli/src/serve/backend_glm5_next.rs#L1149)
  and [publication restriction](../../crates/qwen-cli/src/serve/backend_glm5_next.rs#L1236).
- **Flash-Next:** ordinary optimized runner, RAM snapshot restore on every warm
  turn; no retained live runner. A2 restore **37.6 ms**, captures **98.7 ms**,
  prefill **1535.1 ms**. The workspace resets after the request. Decode uses
  the same runtime as run/bench, including its M5 GDN/FFN fusion selection.
  The fork falls back to the preceding 32K boundary and rereads 16K.
  See [restore](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L470),
  [captures](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L551), and
  [reset](../../crates/qwen-cli/src/serve/backend_qwen4exp.rs#L618).

Per-kernel route counters are not exposed by ordinary serve for these backends.
No source instrumentation was added. Kernel eligibility statements are source
inferences from measured geometry/settings; they are not measured dispatch counts.
V4.1 native-call/scheduler counters and all prefix/snapshot hits above are measured.

## Ranked follow-up work

Savings below are engineering targets, not measured patch gains.

1. **Usable checkpoints before editable messages / at existing chunk boundaries.**
   GLM needs valid Fast live-trajectory provenance, not removal of its canonical
   guard. Target roughly **30 s/fork GLM**, **8–12 s V4.1**, **6–8 s Flash-Next**
   here by limiting replay to a few K rows. Keep a budgeted ladder, not every
   large state indefinitely. Roughly **1–2 weeks across backends**; V4.1's
   existing per-call committed states make its isolated part smaller.
2. **GLM arbitrary-suffix fast paths.** Qualify the existing tensor/KDA/Q6
   routes for real 1–2K partial commands, or a consistent chunk policy. The
   shape contrast supports a target **0.5–1.2 s per warm turn**, roughly
   **3–7 days** including distribution and restore-provenance checks. New
   Fast execution splits must be treated as a numerical-policy change.
3. **V4.1 mapped-weight/idle scheduling.** Independently qualify existing
   `QWEN_SERVE_V41_WEIGHT_WINDOW_MIB=4096` and keep-alive controls. Target
   **0.8–1.4 s per idle warm turn** from the observed scheduling budget;
   **1–2 days** for a controlled current-artifact comparison. The present
   native/serve comparison changes mapping, idle regime and input content,
   so it does not isolate either knob.
4. **Exact CED integration in serve.** Existing 4K run/bench evidence saves
   **0.65 s**; project **3–6 s** for this 16K cold opener, not a measured
   16K result. Roughly **1–2 weeks** for whole-request/cut planning, admission
   and exact-CED-origin prefix compatibility. Current CED does not accelerate
   these warm suffix calls; suffix support is additional work. Bounded replay
   remains rejected and is not a proposed shortcut.
5. **Flash-Next live continuation / tiny-header path.** Avoid unnecessary
   reset/restore/capture traffic on a strict extension and qualify a cheap
   few-token header path. Target **0.1–0.3 s/turn**, **2–5 days**. The measured
   copy/capture times bound the copy-only opportunity; they do not explain all
   of the 0.447 s native/serve TTFT difference.

Full reasoning replay is a prerequisite: the combined return/omission controls
reread **54,923 / 52,905** tokens and cost **55.95 / 28.86 s** for GLM/Flash-Next.
This does not establish the behavior of Britt's installed OpenCode build, and
GLM's single live-session branch switch is a confound. Do not attribute that
entire penalty solely to dropping reasoning.

## Evidence

`$HOME/qwen-runs/2026-11-07-serve-agents/` on both hosts contains source/corpus
hashes, client scripts, each request/response/event stream, per-turn server logs,
all native JSON/build identities, thermal observations and executable checksums.
`summary.json` contains all 26 serve turns; `comparisons.json` the exact-depth
controls. SIGINT exit 130 is the expected graceful server shutdown after each
completed campaign. No app bundle, system toolchain install, code patch or model
artifact mutation was involved. The retained local worktree is documentation-only.

# Real OpenCode agent sessions against qwen serve, 2026-11-08

M5 source: clean **`a41e20b0c9b7ddceec023017705a0c80499c15b0`**, main at the
start of this task. Existing installed laptop OpenCode: dispatcher `c4616404`,
legacy/run implementation **`0c8dfd95`**; both executable hashes match its
installation manifest. This is the real installed `opencode run`, not a substitute
API workload. One session/model does the same pin-guard coding task, with a
continuation and effort change. The models choose different reads, outputs and
repairs; their total times are not engine speed ratios.

## Isolation and exceptions

**The initial pilot violated the no-touch boundary by auto-discovering port
3456.** Two initial invocations made discovery GETs (health, path, providers),
rejected the overlay-only model and fell back to an in-process server. They did
not restart or signal Britt's server or write his config. This is still a
constraint exception: GET handling can initialize workspace state, so an
unchanged PID is not proof of a wholly untouched server. That attempt is
retained/excluded in `fn-discovery-attempt/`, session
`ses_e440f67efffeMNrkKGuLe0iWAV`. All subsequent runs explicitly set
**`OPENCODE_DISABLE_SERVER_DISCOVERY=1`**; the source guard is
[run.ts:1091](/Users/tito/code/opencode/packages/opencode/src/cli/cmd/run.ts:1091).
Both discovery attempts failed before session creation or prompt dispatch on 3456. Its PID **33878**, start/command,
and the config's SHA-256 **and mtime** remain unchanged. Config hash:
`e862b7cbdbbae9a22ab3a1cdbbdfb2f00160a3558c2225a35b4402554d2e310b`.

The corrected pilot finished in **68.6 s**, below its five-minute ceiling, with
`entrypoint=run`, `autoAccept=false`, the working tunnel, and no lock errors.
Only one owned OpenCode instance runs at once, using cx's exact argument shape:
`opencode run --model ... --format json --prompt-file - --run-meta-file PATH
--dir DIR [--session ID] --effort LEVEL --no-auto-accept`.

Every model entry and permission override is in per-process
`OPENCODE_CONFIG_CONTENT`: provider `qwen-serve-m5`, bundled
`@ai-sdk/open-responses`, the exact `/v1/models` ID, tool/reasoning capabilities,
effort choices and context/output limits **98304 / 16384**. Title/summary/
compaction models also resolve to that same endpoint; LSP is disabled to avoid
downloads/daemons. No model/server/client source is changed.

Each task runs in a separate `/private/tmp` local clone with no hard-linked Git
objects. Its overlay allows in-clone reads/edits and CPU build/test commands,
denies external directories, delegation, web tools, installs, destructive commands
and pushes. Permission asks are denied by non-interactive run. All four scratch
clones are removed; **all sessions remain in the shared database**.

qwen serve binds **127.0.0.1:18737 on M5**. The laptop uses an explicit loopback
`ssh -N -L 127.0.0.1:18737:127.0.0.1:18737` passenger over the ControlMaster.
Each model server holds the build lock/nice 10 and a canonical Metal lease with
wait enabled. The forward specification is canceled after each campaign; the
shared Master is never exited. One model resident at a time. One-second thermal
monitoring covers server lifetimes; **all samples are zero**, including excluded
attempts. A separate pre-existing M5 AOT worker used roughly 5–6 CPU cores during
much of the work; it was left alone and later exited. These are shared-host
observations, not quiet-room A/B timings.

Two further **study setup** effects are kept visible, not attributed to Britt:

- The first excluded coding call exhausted the initial **4096** output ceiling
  while still reasoning. Subsequent runs use 16384; this is a study resource
  limit, not evidence that Britt's normal output limit needs changing.
- The corrected FN coding turn reached an operator-set **12-step cap**. The
  client then appended an assistant-only max-steps reminder, which the tier-1
  API rejected with HTTP 400. The cap was removed; wall-clock guards remain.
  Overly narrow shell rules also denied harmless command forms. Later overlays
  allow those forms, preserving external/destructive denials. The same FN session
  was resumed after an explicit owned-server restart; its first continuation
  cold prefill is an operator restart cost. Do not count it as a cache failure.
  Source of the cap tail: [prompt.ts:1125](/Users/tito/code/opencode/packages/opencode/src/session/prompt.ts:1125).

## Task and timing method

The task investigates `SnapshotPolicyPinGuard` and serve callers, adds behavioral
regressions for duplicate/pre-existing/missing pin ownership, runs the standalone
Rust CPU test module, then adds/reviews unwind and dynamic-pin coverage. Real
compiler failures and repairs occurred. Each final tool receipt reports
**15 passed / 0 failed**. Scratch diffs and tool outputs are retained, not shipped.

The installed fork's `--format json` emits a **terminal record**, not a per-step
JSON stream ([run.ts:882](/Users/tito/code/opencode/packages/opencode/src/cli/cmd/run.ts:882),
[terminal:1039](/Users/tito/code/opencode/packages/opencode/src/cli/cmd/run.ts:1039)).
We join that record/run-meta input ID with read-only SQLite queries restricted
to our session IDs. No extra OpenCode instance, export command, or 3456 API is
used for observation. Persisted message/part creation, update and tool timestamps
supply the internal timeline; a read-only observer also retains row versions.

The end of text handling **overwrites `text.time.start`**
([processor.ts:676](/Users/tito/code/opencode/packages/opencode/src/session/processor.ts:676)).
Using that final field would mistake finished decoding for first-token delay.
`client_first_part_ms_approx` instead uses the original part creation timestamp
(or earlier preserved start), minus assistant-message creation. It approximates
the first SDK-exposed nonempty text/reasoning/tool part, not a packet-captured
first delta. Empty parts are excluded. Auxiliary title calls have no stored
assistant part and therefore no client TTFT.

Server logs are timestamped as read. Per-call prompt/generated/cache counts join
**119 primary calls exactly** to DB messages; three small initial unjoined calls
are classified as title helpers from ordering/client source. The CSV retains
message IDs and every native request. One additional setup-rejected request has
no native timings. Server pre-decode phases combine the logged restore/prefill/
copy/boundary fields once: GLM prefill already includes captures; FN subtracts
capture from its prefill field, and its restore field includes departure copy.
V4.1's separate ladder-copy time is added back. Driver and GPU diagnostic spans
are never added together.

For wall allocation we union server phase intervals and tool intervals, preserving
their overlap. `Tool-only` excludes overlap with model work. `Other` is the
remaining launch-to-terminal-JSON time: client startup, DB/Git/transport work,
server work outside emitted phases, and measurement uncertainty. It is **not a
pure client-overhead measurement**. Terminal file mtime avoids the controller's
half-second exit-polling delay. Cross-host clock calibration bounds the offset
at roughly tens of milliseconds; the median midpoint estimate is applied. Round
tables to 0.1 s. No confidence interval or controlled effort-quality comparison.

## Whole-session invocation totals

Includes the FN pilot and failed setup-bounded coding turn; excludes the first
discovery attempt and pauses between invocations. FN's explicit restart/setup
pause is about **220 s** in addition to its active invocation time.

| Model | Native calls | Max prompt | Generated | Active wall s | Server pre-decode s | Decode s | Tool-only s | Other s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 | 58 | 81,109 | 25,645 | 501.7 | 112.7 | 354.8 | 14.4 | 19.9 |
| GLM | 34 | 37,469 | 13,370 | 321.8 | 70.2 | 240.1 | 5.8 | 5.5 |
| FN | 30 | 78,926 | 34,321 | 631.7 | 117.9 | 504.2 | 4.7 | 5.0 |

## Per-turn wall allocation

`Server` is the union of logged model-phase intervals, including sampling/stream
callbacks; it is not GPU-only time. Small post-decode capture/retention costs
explain rounding differences from pre-decode + decode. Tool spans can overlap
server work; the table uses only their non-overlapping portion.

### V4.1 GSQ-RCO IQ2_XS + IQ4_NL Engram

| Turn / effort | Result | Calls | Max prompt | Wall s | Server s (pre / decode) | Tool-only s | Other s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| work / none | success | 26 | 55,790 | 228.8 | 215.3 (46.8 / 168.5) | 6.3 | 7.2 |
| continue / none | success | 29 | 75,394 | 163.1 | 145.9 (19.7 / 126.2) | 7.2 | 10.0 |
| effort-change / high | success | 3 | 81,109 | 109.7 | 106.2 (46.1 / 60.1) | 0.9 | 2.6 |

### GLM-5.3 UD-IQ3_XXS

| Turn / effort | Result | Calls | Max prompt | Wall s | Server s (pre / decode) | Tool-only s | Other s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| work / low | success | 21 | 28,690 | 190.0 | 184.2 (28.8 / 155.3) | 3.1 | 2.7 |
| continue / low | success | 8 | 33,279 | 51.3 | 48.2 (6.5 / 41.8) | 1.6 | 1.4 |
| effort-change / high | success | 5 | 37,469 | 80.5 | 78.0 (35.0 / 43.0) | 1.1 | 1.4 |

### Flash-Next GSQ-RCO IQ3_S

| Turn / effort | Result | Calls | Max prompt | Wall s | Server s (pre / decode) | Tool-only s | Other s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot / low | success | 5 | 31,625 | 68.3 | 66.5 (20.4 / 46.1) | 0.8 | 0.9 |
| work [setup cap] / low | error | 11 | 59,966 | 286.3 | 283.9 (14.2 / 269.8) | 0.9 | 1.4 |
| continue [restart] / low | success | 9 | 70,319 | 136.7 | 133.4 (40.5 / 93.0) | 1.7 | 1.6 |
| effort-change / medium | success | 5 | 78,926 | 140.4 | 138.1 (42.8 / 95.3) | 1.2 | 1.0 |

The complete [per-native-call table](2026-11-08-real-opencode-agent-calls.csv)
contains new/reused tokens, reconstructed client TTFT, logged pre-decode and
decode time/rate, generation count, restore origin, CED/prefill mode, call count,
snapshot-copy time and tool spans. Auxiliary/title rows are explicit. The
setup HTTP-400 tail rejection is documented above and absent from native rows.

## What the real traffic shows

- **V4.1 reuse:** `{'none': 3, 'live_session': 54, 'snapshot_ram': 1}`. Cold entries include the initial auxiliary title and main root, effort changes, and the explicit FN restart; they are not all unexpected misses.
- **GLM reuse:** `{'none': 3, 'live': 31}`. Cold entries include the initial auxiliary title and main root, effort changes, and the explicit FN restart; they are not all unexpected misses.
- **FN reuse:** `{'cold': 4, 'live_session': 25, 'snapshot_ram': 1}`. Cold entries include the initial auxiliary title and main root, effort changes, and the explicit FN restart; they are not all unexpected misses.

- **No blanket historical-reasoning loss appears.** All 31 GLM reused requests
  extend live state. FN has 25 live and one snapshot hit; V4.1 has 54 live and
  one snapshot hit. Most ordinary requests process a short tool result, not the
  whole transcript. The two snapshot fallbacks replay **5266 FN rows / 3.39 s**
  and **9076 V4.1 rows / 6.22 s**, from pre-reply prompt coverage after large
  outputs. We did not capture raw HTTP request bodies/first differing token, so
  the precise whitespace/tool-canonicalization cause is not assigned. These are
  pre-reply prompt-state hits; no interior-ladder restore was needed here.
- **Effort switches really invalidate early-prefix reuse.** Low→medium FN
  processes **71389 new / 0 reused**, **38.75 s** server pre-decode; low→high GLM
  **33717 / 0**, **31.95 s**; none→high V4.1 **75998 / 0**, **43.28 s**. Changed
  leading effort instructions explain eligibility loss. These are different
  review prompts, not a paired test of reasoning quality or total effort cost.
- **V4.1 live CED already works on main.** Its **20650-row live extension**
  uses `shrinking_ced+full` in **12.14 s** pre-decode; the 9076-row replay also
  uses CED, as does the effort-change cold root. Short eligible appends fold
  headers into one call; other short calls retain two. Do not re-propose live
  CED. All **58** V4.1 batch records have width **1** and zero loaded-prefill
  calls, as expected for this serial client workload. See [default control](../../crates/qwen-cli/src/serve/backend_v41.rs#L61).
- **Parallel FN copies already close the old standing's copy gap.** Logs show
  bounded **8-worker** copies around **30–34 ms** at 63–70K in the continuation;
  total logged prompt capture across the 30 requests is only **1.01 s**. Ordinary
  appends still make two native calls. The prior 178 ms copy standing is not a
  remaining cost estimate on this source.
- **Generation volume dominates.** V4.1's largest call decodes **9062 tokens
  in 125.16 s**. Its stored visible text is **31051 characters**; edit arguments
  are only ~4.4K characters, so this is predominantly verbose visible planning,
  not evidence for a copied-argument speedup. FN's largest call generates
  **5164 / 74.66 s**; GLM's **3822 / 68.48 s**. CPU test execution is small;
  reasoning about and repairing compiler failures costs far more than running
  the tests. Throughput changes cannot be inferred from the unequal task totals.

## Ranked next work

Savings are workload projections, not measured patch wins; they are not additive.

1. **Decode throughput with prefix-preserving serving state.** The server-side
   target is native decode work/cache-compatible speculation, not generic HTTP
   tuning. A **1.1x** decode rate would save approximately **32 s V4.1 / 22 s
   GLM / 46 s FN** over these active sessions (roughly **4–25 s per user turn**, depending on output volume). Current native GLM/FN drafting
   serve paths cold-prefill and skip cache reuse; enabling that path unchanged
   can erase its decode savings at long depth ([serve contract](../SERVE.md)). Roughly **1–2 weeks/backend**
   for a viable prototype, transactional state/rollback and numerical/performance
   qualification; no speedup is promised here. Britt's alternative is output
   and reasoning policy: **1000 fewer generated tokens saves roughly 14–18 s**,
   but task success must be checked rather than assumed.
2. **Effort changes without paying a new prompt-head fill.** Observed exposure
   is **32–43 s per switch**. A family-qualified append-only effort instruction
   could retain prefixes, but changes the prompt contract (**about 1–2 weeks**
   including quality/tool checks). Alternatively, budgeted idle precomputation
   of an exact alternate-effort trajectory could move that cost off TTFT
   (**3–7 days**); it spends additional GPU time/RAM rather than eliminating
   compute. Britt can keep effort stable or start the conversation at its intended
   effort immediately; that is a client/orchestration choice.
3. **Measure and narrow the two post-reply replay gaps.** Upper observed
   exposure is **6.22 s V4.1 / 3.39 s FN per affected call**, only two hits here.
   Instrument the first token divergence, then consider a budgeted committed
   checkpoint near a large reply's channel/tool boundary or exact raw-output
   provenance. **1–3 days diagnosis, 3–7 days** for a justified capture policy.
   Restore must retain exact token keys and actual state; never alias a changed
   transcript. GLM shows no such replay in this session. Existing ladder/cold/live
   CED and copy improvements are already active.
4. **Auxiliary work and observability.** The three initial title calls cost
   roughly **6–8 s each**. Britt could supply a deterministic `--title` from cx
   or configure a cheaper title policy (**half-day to one day**, optional).
   Server request IDs and first-streamed-content timing would make joins direct
   (**about a day**, no direct latency claim). Preserving the client's original
   text start and offering progress JSON are Britt's options, not edits made
   here. Current unassigned time is small; V4.1's **~20 s/session** is an upper
   envelope containing unlogged server tokenization as well as client/transport.

The strict-profile shell denials, the initial 4096 ceiling and the artificial
12-step assistant tail are setup effects, not evidence that Britt's normal
configuration needs those fixes. The assistant-tail rejection is a real API
compatibility case when finite agent step limits are used; supporting typed
assistant prefills on tier-1 serve, or changing the reminder's role client-side,
needs a deliberate contract decision rather than a measurement-task patch.

## Retained session IDs

No session was deleted:

- `ses_e440f67efffeMNrkKGuLe0iWAV` — excluded discovery/setup pilot; `entrypoint=run`.
- `ses_e440c2950ffeLSHcSSlnm7Kobq` — FN pilot/coding/recovery/continuation/effort; `entrypoint=run`.
- `ses_e43fbb2d0ffe7P1SO1C5Jkd3y7` — GLM coding/continuation/effort; `entrypoint=run`.
- `ses_e43f553a0ffeeT1QQ9hUPl3x60` — V4.1 coding/continuation/effort; `entrypoint=run`.

## Evidence and cleanup

Raw records on metagross and the local mirror:
`$HOME/qwen-runs/2026-11-08-real-opencode-agents/`: actual overlays/argv/prompts,
run-meta/terminal JSON, own-session DB snapshots and row observation log, timed
qwen logs/thermal traces, CSV source `calls.json`/`turns.json`, scratch diffs,
compiler/test receipts, clock calibration, source/build/installed-binary identity,
setup exceptions, `sessions.jsonl`, validation and cleanup receipts, SHA-256
manifest. Model outputs and operator failures are retained for review.

Documentation/path/link/data checks pass. All owned model servers, client runs,
observers and forwarding passengers have exited. Both port-18737 listeners are
closed; M5 source/target and all local coding clones are removed. Local docs
worktree is removed after commit; its docs-only branch remains, without merge.
Britt's config is unchanged and his server was never restarted or signaled;
the initial discovery GET exception above prevents claiming it was wholly
untouched. His server and all four session records are left in place.

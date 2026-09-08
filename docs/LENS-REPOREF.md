# RepoRef: From Construal To Evidence-Seeking Action

Research audit, 2026-09-08. Paper, released data, construction prompts and static
harness inspection only. No new model inference, GPU work, benchmark runs,
upstream execution or dependency installation. Seven conversations were selected
and read; external candidate artifacts and frozen response fixtures are not yet
assembled. This is a proposed research arm, not an experimental result.

## Scientific Purchase

RepoRef is a useful expansion of the J/R program, not a replacement for the
[open-ended construal casebook](LENS-CONSTRUAL-CORPUS.md). Its contribution is an
externally checkable intended referent that an agent must discover through tools:

```
working interpretation -> query / tool choice -> evidence exposure
    -> candidate comparison -> commitment or reformulation -> next query
```

Interpretation changes which evidence becomes available. A mistaken author,
topic or temporal constraint can produce an apparently confirming search history
by excluding useful evidence. Conversely, successful search can repair the
initial interpretation. This gives us observable consequences for a working
construal beyond verbal acknowledgment of alternatives.

The strongest new question is whether J/R readouts forecast consequential
search choices and evidence uptake beyond what the transcript, query and tool
results already predict. It is not whether lens tokens literally name a hidden
"search" or "verification" operation. A forecast is not a causal mechanism;
steering efficacy would require a separate intervention and appropriate controls.

RepoRef deliberately retains references judged to have one recoverable target.
That supports research on temporary uncertainty, evidence binding and commitment.
It does not supply a representative collection of genuinely unresolved meanings,
negotiated goals or multiple legitimate intended referents. Keep both arms.

## Sources And Scope

- Fuchs, Katz and Goldberg, *You Know What I Mean: A Benchmark for Agentic
  Conversational Reference Grounding*, [arXiv 2608.29834v1](https://arxiv.org/html/2608.29834v1).
- [Released repository](https://github.com/karenShaked/RepoRef-Benchmark/tree/83ff5a67fd005ef4f4f48148d5a0351980c392af),
  pinned at `83ff5a67fd005ef4f4f48148d5a0351980c392af`, dated 2026-09-07.
- Local release counts and byte manifest:
  `target/reporef-review/data-audit/REPORT.md` and `manifest.json` in the isolated
  `qwen-llm-reporef-review` worktree. Raw release material and audit scratch are
  ignored, not redistributed with this memo.

The benchmark starts from real multi-party Gitter conversations containing
explicit issue, PR or commit links. Segmentation, link masking, natural rewrites
and model judgments produce retained indirect-reference tasks. Segments can
include messages after the marked reference: this is retrospective grounding,
not necessarily what was inferable at the moment of utterance. Agent-side local
indexing is prohibited in the original task; offline indexing during dataset
construction is a different operation.

The original link establishes the original referent. It does not prove that the
rewritten conversation uniquely identifies that referent among all possible
artifacts. A finite distractor pool and pairwise ambiguity judgments constrain
this problem without eliminating it.

## What The Reported Results Establish

The main set has 400 segments, 92 repositories and 23 communities. Gemini 3 Flash
scores 268/400 (67%) at the nominal ten-call budget. The budget sweep uses a
different, 280-case subset: 189/280 (67.50%) at ten and 207/280 (73.93%) at sixteen.
Do not turn the full-set 67% and subset 73.93% into a same-set improvement.
Cross-model results also mix model, provider settings and harness differences.

Appendix H attributes about 70-92% of failures to the target not being surfaced.
Conditional accuracy after surfacing is about 87-91% for most reported models,
but GPT-5 mini is 80.9%. "Surfaced" includes appearing among search items, not
just direct artifact inspection. It does not establish recognition, sufficient
evidence, use in the next action, or a causal benefit from injecting the target.
Conditioning on surfacing selects on both case difficulty and search policy.
Failure to surface can itself begin with misunderstanding the conversation.

Maintain distinct observations:

1. The artifact exists in the external environment.
2. A tool returns it.
3. Its identifying/evidential bytes enter the actual model context.
4. The model can recognize its relevance on an independent probe branch.
5. That evidence affects the next query, inspection or answer.
6. The final commitment identifies and supports the intended referent.

The metadata verifier is useful as a refutation template: inferred author,
date, repository, type or state constraints can reject candidates. Passing
those checks does not certify correctness. Historical versus current state
must be explicit. Likewise, the paper's "information gain" counts previously
unseen artifacts, not measured uncertainty reduction. Excess calls relative to
the smallest successful observed path are not a proven intrinsic inefficiency.

Some appendix quantities need clarification before reuse: Grok's reported
step-ten discovery and overall accuracy do not reconcile under an obvious shared
definition, and Appendix I differs from the main and budget-sweep tables. Use
each table's own denominator and definition, not a combined reconstructed metric.

## Released Material: Useful And Incomplete

The release contains 400 main and 1,448 extended records, gold identities,
diagnostic metadata, redirect/duplicate mappings, construction prompt documents,
tools and baseline/scoring code. The extended set is also retained construction
output, not the rejected ambiguous population.

It does not contain evaluated model trajectories, per-example predictions,
ordered distractor pools, rejected cases, original/brutal-masked conversations,
per-row rewrite inputs and judgments, or historical artifact snapshots and a
warmed tool cache. Trace schemas in code are not released traces. We can mine
task material and prompt ideas now; we cannot reanalyze the reported agents'
internal search trajectories from this release alone.

Three ingestion controls are essential:

- **Opaque agent IDs.** Every main instance ID encodes the gold artifact number
  or an acceptable SHA prefix. The shipped prompt builder does not forward that
  ID or gold; naive serialization of whole records would. Whitelist prompt fields
  and keep IDs, gold, aliases and diagnostic annotations evaluator-only. Requested
  entity type is an explicit benchmark input, distinct from accidental leakage.
- **Grouped partitions.** Main and extended have no shared instance IDs but share
  57 target groups and 2,047 message IDs under the audit's documented grouping.
  Partition connected components joining shared messages and target/alias groups,
  not rows. Repository/community holdouts can test stronger generalization. This
  is release overlap, not evidence about a model's pretraining contamination.
- **Check visible identity cues.** The main scan found no direct gold/alias URL,
  `#gold` or seven-plus-character gold SHA hits under its explicit rules. Three
  extended rows contain alias-equivalent links or a short gold SHA. Neither result
  proves the absence of every possible shortcut; inspect selected cases manually.

The natural masker explicitly favors natural chat over minimal editing, can
receive artifact author/date metadata, and can use relative authorship/time cues.
It can rewrite multiple link-bearing messages. The release does not let us
determine which individual cues were original versus introduced. Connection
strength is judged on brutal-masked text before this rewrite. The reported
100-case comparison (77% natural versus 75% placeholder, McNemar p=.83) is not an
equivalence test. Separate manual review of 100 rewrites is scoped evidence, not
a guarantee for every row. Retained-case/model-legibility selection remains a
boundary on generalization, not a reason to discard the dataset.

Data is CC BY-NC-SA 4.0; code has MIT-style terms. License path names lag the
repository restructure. Keep raw conversations private here; respect attribution,
noncommercial and share-alike conditions for redistributed data derivatives, and
clarify path coverage before broader redistribution. Public access is not an
unrestricted commercial license.

## Prompts To Adapt, Not Adopt As Ground Truth

The segmenter, connection-strength, natural-mask and pairwise-ambiguity prompts
are useful design artifacts. Extract their explicit evidence fields and their
allowance for equally valid candidates; do not import their judgments as latent
state labels. An independent case audit should record:

- The marked reference's scope and which speaker owns each relevant claim.
- Candidate constraints, the exact supporting message, and whether each constraint
  is explicit, inferred or uncertain.
- Evidence against each candidate, including current/historical-state ambiguity.
- Acceptable distinct referents separately from aliases or duplicate identities.
- A still-plausible alternative and what observation would discriminate it.

Audit annotations stay outside primary agent histories. Do not ask an agent to
recognize the target and then treat its primed continuation as unprompted uptake.

## Three Prioritized Pilots

### A. Supplied Evidence To Actual Use

Begin with a small audited candidate packet, holding candidate identity constant
while replacing decisive evidence with matched nondecisive material. Optional
URL-only and missing-target conditions separate identity exposure from support.
Counterbalance rank and presentation; include irrelevant-span/context-removal
controls to detect direct shortcuts. Record the next inspection, query or answer,
not merely candidate mentions. Test explicit recognition on independent forks.

This estimates a conditional effect of supplied evidence. It does not measure
natural retrieval competence. Readouts at pre-result, post-result and pre-action
positions can ask whether uptake is visible before the next action, with plain
logit, text/history-only and appropriate null baselines. Small discovery samples
yield case evidence, not a validated general predictor.

### B. Search Revision Versus Repetition

Freeze a failed or empty-search checkpoint. Compare a justified alternative
query strategy with repetition under the same executed-tool budget, using actual
captured responses. Forecast which branch helps before revealing its response.
Then test whether lens features add anything beyond query text and history.

Changing query and result together is a joint intervention. Do not manufacture
incoherent query/result crossings merely to get a clean factorial design. An
unsupported request in a small fixture tree is not an actual empty search result.
Only after held-out forecasts work should a closed-loop policy compete against
text-only and random policies with tools, tokens and lens compute accounted for.

### C. Order, Provenance And Revision As Separate Contrasts

Present the same evidence in AB/BA order with source labels attached to their
content. Separately introduce a contradiction to test whether the appropriate
claim changes without overwriting unrelated commitments. Include legitimate
non-revision and genuinely underdetermined cases. Source credibility and
historical state are evidence variables, not nuisance labels to shuffle blindly.

This connects reference grounding to accumulated epistemic status: an agent's
prior guess must not become independent evidence merely because it was repeated.
Several possibilities may remain usable without forcing either premature choice
or compulsory clarification. Multiple valid search routes are not errors.

Teacher-forced previews of proposed queries could later test whether shadow
readouts detect bad filters before tool execution. This is a lower-priority
read-only action study, not a demonstrated safety gate or a "Wait"-token alarm.

## Seven-Case Starter Panel

Selected by visible mechanisms before inspecting model outcomes or lens features.
These descriptions paraphrase released conversations, not verified remote targets.
Original answer-bearing IDs remain in the private evaluator audit.

| Case | Research opportunity | Main caution |
| --- | --- | --- |
| JTAppleCalendar, 2018-01-11 | New report, author/time binding across an intervening topic | Adjacent cell-spacing discussion need not describe the report |
| JHipster, 2015-05-22 | Apparent architecture failure revised to browser filtering | Discourse correction is not yet a verified target-content comparison |
| Cucumber, 2016-05-11 | Cross-speaker correction of Java/version diagnosis | Duplicate identity is not plural intended meaning |
| ImageJ, 2015-09-17 | Newly opened feature PR among several visible PRs | Nearest URL and most recent artifact are shortcuts to test |
| Ancient Beast, 2016-03-05 | Sparse social/technical description of an older PR | Particularly needs an independent underdetermination audit |
| ImageJ, 2015-09-22 | Cross-repository analogy between JavaScript patches | Preserve speaker, repository, type and temporal distinctions |
| JSPM, 2015-01-07 | Earlier fix followed by doubt and later fixes | Later successful-looking commit need not be the marked referent |

Next artifact: a versioned small research packet, not a general benchmark runner.
It needs opaque agent inputs, evaluator-only identities and evidence constraints,
audited gold/close competitors, timestamped source captures, separately labeled
experimental variants, and one Pilot A contrast plus a bounded Pilot B branch.
Freeze pagination/truncation and state limitations. Retain cases that fail the
uniqueness audit rather than quietly replacing them with cleaner successes.

Seal a held-out packet separately from adaptive exploration. Record what each
contrast could falsify before running it; report nulls and alternative explanations.
There is no bias-free selection procedure, but there can be inspectable selection,
competing predictions and protection against post-hoc success chasing.

## Harness Boundaries Before Any Run

Static findings apply to the pinned release, not necessarily the environment that
produced the paper. Do not attribute published failures to them without run logs.

- The restructured baseline resolves default data paths beneath the wrong parent.
  The active loop is `agent/architectures.run_loop`, not the older loop module.
- Reported tool counts include submission and rejected over-budget requests. The
  active loop caps executed retrieval dispatches separately; a mean above ten is
  not by itself an executed-budget violation. Separate requests, executions,
  rejections, submissions and underlying HTTP calls.
- Some advertised pagination/order parameters are dropped during dispatch. The
  loader does not populate a historical cutoff, and a cutoff would not reconstruct
  old bodies/comments anyway. Cache misses query the live world.
- Normalized snippets can coexist with nested raw response data. Trace conversion
  loses parallel-call detail and per-step thinking. Capture the actual chain from
  HTTP response through serialized tool content to rendered bytes/token IDs and
  the forwarded prefix; do not define visibility from a cache alone.
- Submission is not restricted to previously inspected candidates. URL extraction,
  aliases, duplicate identities, issue/PR equivalence and commit-prefix acceptance
  make scoring broader than strict URL equality. Preserve the published identity
  contract when reproducing it; label any stricter diagnostic separately.
- Cost/time limits are not all hard bounds; HTTP timeout/retry behavior needs
  attention before unattended execution. Read-only GitHub wrappers do not make the
  entire dependency/runtime environment certified safe.

These findings argue for a small explicit capture/replay protocol first, not an
upstream repair project or a stock leaderboard run.

## Native Lens Readiness

The [data-only transport contract](LENS-DATA-CONTRACT.md) now accepts the new
Qwen3.8 J/R pair through `read-full`, `trace-full` and projected token banks.
Both data-v1 payloads are converted and CPU-verified, with identity at layer 62.
Import is no longer pending. Exact deployment binding and live numerical smoke
remain unqualified in that record; BF16-to-GGUF transfer remains a caveat.

Before this packet's model run, qualify bounded J/R scalar, packed and passive
projected-bank paths plus layer-62 parity against the plain logit lens under an
exclusive Metal lease. Freeze the actual qualified binary, model and payload
identities; do not reuse an older reader that predates the contract.

Use native OpenResponses rendering for tool histories (`--open-responses`, not
the narrower `--messages` input), then exact token-ID replay where the scalar
reader needs it. Preserve authored spans, rendered bytes, token IDs and forwarded
prefixes. Use sparse positions across dense depth, not full-vocabulary retention
at every token. Raise explicit input limits rather than silently truncating.
Use consistent arithmetic for paired contrasts; known scalar/packed differences
can dominate small effects. Treat early-layer R as a primary readout in its own
right, not contingent on agreement with J.

Matched Qwen3.6/3.8 work should compare within-model effects on the same authored
histories using each model's native formatting/modes. It is not raw cross-model
coordinate alignment or a causal claim about training differences.

## Decision

Proceed with RepoRef as an externally grounded evidence-to-action arm beside the
open-ended casebook. Build and audit the small packet before spending inference
or developing a controller. The research synthesis received an independent
adversarial review of its statistical, plurality and native-readiness claims;
that review did not independently rerun every release count. No new experimental
effect, globally unique candidate set or completed fixture packet is claimed here.

# GLM-5.3-Flash Fast shared-prefix split: quality against Exact (map #12/#15, 2026-10-08)

Question: serve's Fast snapshots (`fast_shared_split_v1`) read a prompt as
`[0, s)` then `[s, N)`, each in 512-row chunks from its own start, with `s`
the end of the shared instructions-and-tools prefix. Does that new Fast
schedule meet the quality limits Fast at 512 rows met
(`../2026-10-08-glm53-fast-quality/`), so it can become serve's default?

Answer: **INCONCLUSIVE** under the preregistered rule, so Fast snapshots stay
opt-in (`QWEN_GLM_FAST_SNAPSHOTS=1`); Exact snapshots are unaffected. Every
gated limit passed except one: the short-tail arm's overall NLL upper bound,
0.0054 against 0.005 (observed +0.0011 nats/token). The tool screen passed
(12/12 greedy, 1.00 sampled for every arm) and the agent-shaped cases
passed their frozen first-call check in every arm (8/8). The tail arm's
interval against unsplit Fast includes zero (+0.0007, central 90%
[-0.0025, +0.0040]); the cohort establishes neither qualification nor
unacceptable degradation.

## Preregistration

`scripts/reference/glm53/quality-split-v1.json` (cuts, cases, generated at
`2f4a6553` on the CPU) and `scripts/reference/glm53/quality_split_analysis.py`
(limits, gates, decision rule; quality-v1's analysis imported unchanged)
were committed (`83757f5f`) before any GPU run, amended after the cx 01a10cc
preregistration review (`976071ed`: validation, invalid-evidence refusals,
checkpointing; limits unchanged) and once more while the run was in progress
and before any result was viewed (`3b63a8a1`: KL roundoff tolerance). Runs
at `d65a09a2` with Metal API validation on (`run.log` not committed;
`report.json` and `verdict.json` are).

- Arms on quality-v1's 38 text items (64 teacher-forced tokens each):
  `exact_512` (a full rerun), `fast_512` (the qualified control),
  `fast_split_grid` (cut at the deployed agent prefix's in-chunk offset:
  the opencode instructions and 12 tools are 11,104 tokens, offset 352, so
  cuts 352 / 1,888 / 3,936), `fast_split_tail` (cut at L - 17 - (document
  index mod 4): a user turn and header after the cut), and on the six long
  items `fast_split_frontier` (cuts 2050-2053 across the sparse frontier).
- Gates: grid and tail each meet every quality-v1 limit against Exact
  (overall NLL one-sided 95% upper bound <= 0.005; each stratum <= 0.010;
  top-1 loss <= 1 point); frontier the long-stratum NLL and top-1 limits;
  the tool screen with the split at each task's real shared-prefix end (255
  tokens). Decision: default only on QUALIFIES and PASS; INCONCLUSIVE keeps
  the opt-in and licenses neither a failure diagnosis nor wider limits.

## Results (`verdict.json`)

| Split arm vs Exact | Statistic | Observed | Central 90% | Limit | Verdict |
|---|---|---:|---:|---:|---|
| grid | overall NLL | -0.0005 | [-0.0055, +0.0041] | <= 0.005 | PASS |
| grid | overall top-1 | +0.0021 | [-0.0029, +0.0070] | >= -0.01 | PASS |
| grid | prose / code / long NLL | -0.0013 / +0.0026 / -0.0063 | upper +0.0075 / +0.0078 / +0.0030 | <= 0.010 | PASS |
| tail | **overall NLL** | **+0.0011** | **[-0.0034, +0.0054]** | <= 0.005 | **INCONCLUSIVE** |
| tail | overall top-1 | +0.0012 | [-0.0012, +0.0037] | >= -0.01 | PASS |
| tail | prose / code / long NLL | -0.0001 / +0.0044 / -0.0047 | upper +0.0093 / +0.0080 / +0.0013 | <= 0.010 | PASS |
| frontier | long NLL | -0.0028 | [-0.0132, +0.0043] | <= 0.010 | PASS |
| frontier | long top-1 | +0.0026 | [-0.0078, +0.0130] | >= -0.01 | PASS |

Reported, not gated:
- Split vs unsplit Fast 512 (the schedule change alone): grid -0.0009
  [-0.0037, +0.0019], tail +0.0007 [-0.0025, +0.0040], frontier (long)
  -0.0005 [-0.0106, +0.0095]. No split arm reproduced unsplit Fast bitwise
  on any text item (logit fingerprints), so the split does change Fast's
  arithmetic; the split-minus-unsplit mean-NLL intervals include zero,
  which establishes neither equivalence nor the absence of other quality
  effects. The tail arm's code-stratum difference from Exact is positive
  (+0.0044, central 90% [+0.0006, +0.0080]), though within its 0.010
  margin.
- Controls: this run's `exact_512` and `fast_512` equal the committed
  quality-v1 report exactly on all 38 items (mean NLL, top-1 and per-token
  NLL): no change detected in these recorded metrics after the sparse IQ3_S
  down retile and the rebases (not a general logit-identity claim; this
  report's logit fingerprints serve later ones).
- Worst-position KL from Exact per item: Fast 512 max 0.200 (median 0.054),
  grid 0.323 (0.045), tail 0.128 (0.043), frontier 0.094 (0.047).
- Agent-shaped cases (opencode prefix, 11,104 tokens; read, grep, bash,
  glob; greedy and one sample): 8/8 pass the frozen first-call check (tool
  name and argument substring; not tool execution) in every arm; Exact
  restored at
  the cut equalled an unsplit Exact prefill bitwise; prompt-end KL from
  Exact up to 0.27 for both Fast and the split (identical prompt-end logits
  on three of four tasks).
- Prefill time, means including session creation, under Metal validation:
  Exact 18.5 / 58.9 / 136.7 s at 512 / 2,048 / 4,352 tokens (the 512 mean
  includes one 64.2 s observation; median 14.8 s); Fast 2.5 / 10.4 / 23.9 s;
  split arms 3.1-3.3 / 11.2-11.3 / 24.3-25.1 s.

## Reading

The tail arm's overall interval crosses the non-inferiority margin: this
cohort establishes neither qualification nor unacceptable degradation.
(Unsplit Fast's own upper bound was 0.0046 against the same 0.005.) The
tail arm models one deployment-relevant geometry, a short final segment
after the cut; shared prefixes also precede long user turns and
accumulated histories, and raw-text tails model segment length, not
user-turn or header content. Under the preregistered rule nothing changes:
Fast snapshots stay opt-in. Deciding them now would mean widening the limit
or choosing arms after the fact, which the rule does not allow.

Next (cx 01a10cc): a fresh confirmatory cohort, `quality-split-v2`, with
v1 used only for planning: frozen eligibility, repository revision,
hash-ordered selection, exclusions (generated artifacts, near duplicates)
and strata proportions; the same grid, tail and frontier gates against a
full Exact rerun, `fast_512` as a control, the same limits and bootstrap;
the sample size fixed in advance with no adding documents until PASS. From
v1's document-cluster variance the overall tail-NLL standard error is
about 0.0028; for one gate, about 70 fresh documents (about 120 items) give
80% power and about 95-100 give 90% if the true increase is +0.001, but
the joint gate's power must be simulated before N is frozen. Default
promotion also needs the release time-to-first-token screen. Accepting the
unresolved uncertainty because of the ~61 s replay cost would be a policy
exception, not a statistical qualification.

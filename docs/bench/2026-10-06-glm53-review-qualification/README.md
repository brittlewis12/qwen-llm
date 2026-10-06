# GLM-5.3 review fixes: live qualification (2026-10-06)

Live checks for the review-fix batch on `feat/glm53-p6` (cx session
01a10cc).
- **Binaries:** frozen at `dfef7dc1`. Commits after it are docs, scripts and
  tests only.
- **Hardware:** M4 Max, GLM-5.3-Flash UD-IQ3_XXS.

## Results

| Check | Result | Artifact |
|---|---|---|
| `glm5_next_metal::tests` gates, `MTL_DEBUG_LAYER=1` | **14/14 pass**: llama.cpp oracle, Fast chunking qualification, Exact packed vs serial across the frontier (bitwise), sparse selection replays, lens readout refusal and acceptance | `gates.txt` (the reported 13,594 s includes about 3 h 30 min waiting for the GPU lease) |
| Tool loop, live serve, JSON and SSE | **pass**: the call is published; the replayed loop reuses 218 of 236 tokens (prefill 400 ms vs 1.5 s cold) | `tool-loop.json` |
| Idle-residency poll | **pass**, every verdict (table below) | `residency-poll.json` |
| #12 Fast reuse | **fails its frozen gate** (below) | `fast-reuse.log` |

Idle-residency finishes, from the server's debug lines:

| Traffic | Compute encoders begun | Window |
|---|---|---|
| model-list polling | none | unchanged |
| refused request (400) | 0 | unchanged |
| completed request | 9 | renewed once |
| client abort after the first event | 2 | renewed once |

The first residency run was inconclusive. Serve is serial, so it answers
503 while an aborted generation unwinds, and the script treated that as
fatal. Fixed in `638a3327`; the rerun above passes.

## #12: Fast reuse across reuse boundaries

**Setup.**
- Teacher-forced over the join plus 32 positions, against one cold Fast
  prefill.
- Frozen bounds: per position, KL in both directions ≤ 2e-2 and both
  regrets ≤ 0.2.
- Diagnostic only, with nothing asserted: each Fast run is also compared
  with the same cold run in the Exact lineage.

| Case | Warm vs cold Fast | Cold Fast vs Exact | Warm Fast vs Exact |
|---|---|---|---|
| Ordinary: Fast 700, 40 serial, Fast suffix to 1300 | worst KL 2.13e-2 at position 12; position 24 flips (regrets 0.008 / 0.397) | worst 2.17e-2; flip at 24 (Exact-side regret 0.24) | worst 6.6e-3; no flips |
| Tool loop across the frontier: consumed 2004 < 2052 < next 2370 | worst 5.6e-2 at positions 1–2; top-1 33/33 | worst 7.9e-2; flip at position 2 (regrets 0.54 / 0.48) | worst 0.111; flip at position 2 (regrets 0.54 / 0.13) |
| Cancel at the second chunk boundary, then resume | **bitwise** | — | — |

**Recorded outcome** (cx's wording):

> Review fixes qualified for integration. Fast reuse exceeds the frozen
> warm/cold tolerance on held-out prompts; both cold and warm Fast also
> exceed the existing Exact-reference policy. Cancellation/resume on an
> unchanged schedule is bitwise. No branch-induced numerical regression
> identified; general Fast-lineage qualification remains open.

**Notes.**
- The test stays on the branch as a **known failing qualification**. Its
  assertions and bounds are unchanged; it is ignored and labelled.
- The Fast policy's bounds (`LOGIT_KL` 2e-2, `TOP1_REGRET` 0.2) were
  calibrated on one prompt as about 2× its worst. The frontier gate's own
  prompt shows Fast vs Exact ≤ 2.1e-5. So Fast's error is strongly
  prompt-dependent, and these two prompts act as holdouts that exceed the
  policy.
- **Possible confound:** the continuation is unrelated text (positions
  3000–3031), teacher-forced after a chat prompt or after a jump from
  position 1300. Every arm saw identical tokens, so it does not invalidate
  the failure.

**Next** (design review):
- **Four arms:** Exact cold, Exact warm, Fast cold and Fast warm, on
  identical histories. Exact warm and cold must be bitwise equal across the
  ordinary and tool boundaries.
- Compare both Fast arms against Exact under the unchanged policy, and keep
  warm/cold as a separate measure of schedule sensitivity.
- **Natural continuations:** decode them greedily with Exact, freeze them
  before evaluating Fast, and keep the unrelated-token case as a stress
  test. Inspect position 0 separately.
- A small frozen holdout of natural chat and tool continuations, below and
  across the frontier, with several chunk schedules.
- Release-build time to first token, to price the Exact vs Fast trade-off.
- No bound change is approved. If representative trajectories still
  violate the policy: fix Fast, or make Exact the qualified default with
  Fast opt-in.

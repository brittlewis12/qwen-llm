# V4.1 intake + exact tokenization reuse: served qualification

Stage began 2026-11-14 12:36:17 UTC, stop/report by 14:36:17 UTC. Rebased `perf/serve-ttft-instrumentation` onto main `e70da3f8a`. Only generated ENV documentation conflicted and was regenerated. Shared tracing method signatures, stages and snapshots remain stable for the GLM hook lane.

## Frozen protocol before the first arm

One release binary, candidate defaults on for `QWEN_SERVE_ASYNC_PREPARE` and `QWEN_SERVE_TOKEN_PREFIX_CACHE`; each has its independent `=0` rollback. All baseline arms explicitly set both to 0; candidate arms exercise the new defaults without those overrides. No shader, numerical kernel or checkpoint change.

Run order is **A0, A1, B1, B2, A2, A3, B3**. A0/A1 are the two baseline controls. A1 is also preregistered as pair1's baseline, with no selection based on its speed; pairs are A1/B1 (AB), B2/A2 (BA), A3/B3 (AB). This avoids one redundant baseline arm while preserving the requested interleaving. Every arm is its own waiting <=15-minute lease, actual nice0, inherited FD, Engram purge to 0 resident pages before loading, no other resident model, thermal <=1 continuously60s before load/window, allow2/abort3. Build nice10; release serve and test-profile tests. File-A artifact/corpus/plan remain unchanged: greedy, seed20261112, 3 agents, 49,152-token sequential roots / 32-token root replies, 15s ramp, 120s window, context196608, V41_MAX_ACTIVE4 / SAMPLE_WORKERS3 / PREFILL_FAIR1.

Symmetric warmup before every arm: identical CPU path tests (including the cache and blocked-pump paths even for baseline), released-tokenizer full/incremental witness, then the same three roots and three-agent ramp. No new GPU pipelines exist; metallib identity is recorded. The cache's per-conversation serving state warms only where the feature is enabled, as intended.

Control: compare every common response's text/reasoning/argument bytes; completed pairs require equality, partial pairs require common-prefix equality, no-output cases are unobserved. Report response mismatch rates plus first divergence per conversation, matched-input hashes/cuts, and decode-round cohort histories. If A0/A1 have no genuine observed divergence while the prior candidate did, stop for attribution and do not promote. A positive A/A control supports the existing batch-numerics hypothesis only where request inputs/cuts match and cohort histories differ; it is not proof of every possible candidate difference.

Frozen six-metric gate, preserving `v41_prefill_serve_compare.py`: geometric mean over three pair ratios, candidate/reference TPS >=0.98, reference/candidate root **native prefill** >=0.98; short and ~1K native prefill and client TTFT >=1.00. Match labels and exact new rows, both completed/offered inside the window, requested append150–2000, no CED; short native rows128–511, ~1K512–2000. Retain all excluded/unmatched/censored counts. Root client TTFT is reported separately, including all three roots per arm and host load, without substituting it for native prefill or changing the frozen rule. Outside-native p50/p95 is reported for every arm using all eligible observed TTFTs.

## Diagnostics added without changing the shared API

The existing timing knob now additionally emits full generation round records on round completion, after releasing the history mutex. Request/round IDs, ordered participants, times and shape are recorded, with no token content. The prior snapshots still stop at first visible text. Opt-in input witness lines carry exact rendered-prompt/token-ID SHA256, byte boundaries, resolved token cuts, and the initial prefill-logit SHA256. These do not change arithmetic; every new arm uses the same probes. Raw request/response payloads are retained on M5 for attribution rather than being discarded after a failed or cancelled turn. No tracing API signatures changed.

## Results

**Promoted by explicit lead decision after the frozen rule failed.** The original short/~1K native-prefill threshold was 1.00; measured geomeans 0.998399/0.999226 failed it. The lead ruled that threshold a spec error for an intake/tokenization change that does not change prefill compute. Native prefill is a non-regression check at >=0.98 for this change; both cells pass. Promotion rests on that non-regression plus short/~1K client TTFT 1.096472/1.050986, outside-native p50 roughly 100→45 ms, TPS 1.006389 and root native prefill 1.020042. This is a recorded lead decision, not a retroactive pass under the frozen rule. All original measurements, thresholds and the FAIL JSON remain unchanged.

Both `QWEN_SERVE_ASYNC_PREPARE` and `QWEN_SERVE_TOKEN_PREFIX_CACHE` are default-on again, each with its own `=0` rollback. Lifecycle tracing remains opt-in through `QWEN_SERVE_TTFT_TIMING=1` and is included in this branch.

The candidate divergence rate (1/39, 2/39, 2/39) matches the A/A control range (2/39), with no identical-prompt token-ID or cut mismatch. The lead accepted this as batch-composition variance that predates the change. The traces establish baseline variation and differing batch histories; they do not identify a particular numerical kernel. No cross-run served output-bitwise claim is made.

### Baseline divergence control

A0/A1 are both rollback0 on release binary SHA256 `66b15ced6d8832b8b782b085a79ec09e9747a51a1374ae35038d53198538ea2b`, source `3482f60cb` after the rebase. Full traces have 4,278/4,258 completed round records and 40 input/cut records each. Their shared cases include **39 jointly visible responses**: **2/39 (5.13%)** differ, including **1/37 completed pairs** and one nonmatching partial prefix. One no-visible-output case is unobserved. The prior candidate/reference comparison had **1/39 (2.56%)** observed-prefix differences, zero among37 completed pairs. These are small-sample response rates, not independent-trial significance estimates.

The first A/A divergence is completed **`a2-t1`, byte22**, with identical prompt bytes, token IDs, token count, byte boundaries and resolved cuts. Initial prefill-logit hashes already differ. It has 74 three-member /21 two-member decode rounds in A0 versus75/20 in A1; its first cohort difference is transition24 (round1242 versus1245), after its initial-logit difference. The earlier context is more informative:

- `a2-warm`: identical initial logits and31 identical single-member rounds.
- `a2-t0`: identical input/cuts and initial logits, identical1,064-token visible output, but cohort differences from the first transition. A0 widths3/2/1 count950/102/11 versus957/92/14 in A1.
- `a2-t1`: both restore `live_session` at50,834 tokens, prompt51,944; identical current input/cuts but different initial logits and delivered text.
- `a2-t2`: prior differing assistant text now changes the prompt hash; it is a downstream cascade, not a second demonstrated independent numeric event.

Among38 jointly visible calls with matched input/cut witnesses, one differs. This directly demonstrates baseline numeric variation without either feature. Different batch histories are consistent with the existing quantized-column rounding caveats; these records do not isolate a particular kernel. [Complete per-call hashes, cohort examples and rates](2026-11-14-v41-intake-qualification/control/baseline-comparison.json). The control meets the preregistered condition to proceed; it does not excuse a future candidate token/cut mismatch at identical prompt text.

### Frozen gate result (historical, unchanged)

**FAIL** under the frozen six-metric rule. At the initial cutoff both switches were returned to default-off without promotion. The subsequent lead decision above restores default-on; it does not change this frozen result.

Ratios above 1 favor the candidate; each latency ratio is baseline/candidate. TPS is candidate/baseline. No arms or cells were dropped.

| Metric | Pair1 AB | Pair2 BA | Pair3 AB | Geomean | Threshold | Result |
|---|---:|---:|---:|---:|---:|---|
| Aggregate TPS | 1.012866 | 1.019753 | 0.986849 | 1.006389 | 0.98 | PASS |
| Root native prefill | 1.036409 | 1.023182 | 1.000852 | 1.020042 | 0.98 | PASS |
| Short native prefill | 1.004573 | 1.002851 | 0.987859 | 0.998399 | 1.00 | FAIL |
| Short client TTFT | 1.098230 | 1.100713 | 1.090500 | 1.096472 | 1.00 | PASS |
| ~1K native prefill | 0.998643 | 1.000307 | 0.998731 | 0.999226 | 1.00 | FAIL |
| ~1K client TTFT | 1.049950 | 1.055007 | 1.048014 | 1.050986 | 1.00 | PASS |

Root client TTFT ratios (descriptive, separate from native prefill): pair1 1.036828, pair2 1.023804, pair3 1.001348.

Each pair has pair1: 9 short / 7 ~1K, pair2: 9 short / 7 ~1K, pair3: 9 short / 7 ~1K matched calls. Detailed exclusions and row counts remain in the pair JSON.

### Outside-native time and process I/O

All eligible observed TTFTs, milliseconds, p50/p95; these distributions include calls outside the stricter frozen matched bins and censored calls with a visible first token. Disk bytes are the maximum sampled live process counter, covering startup, roots, ramp and window. Window read bytes are the counter delta between the first and last samples inside the 120s window; exact sampled bounds are in the summary. Compiler presence is retained as a covariate, not an exclusion.

| Arm | TPS | Short outside | ~1K outside | Root outside | Process disk-read bytes | Window sampled read bytes | Max thermal | Compiler in window |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | 90.392 | 107.901/1093.753 | 100.612/104.988 | 92.743/116.827 | 189581676544 | 2691858432 | 2 | False |
| A1 | 89.383 | 103.077/1110.298 | 103.233/112.712 | 87.124/103.407 | 189811970048 | 2826551296 | 2 | True |
| B1 | 90.533 | 44.289/908.695 | 40.172/56.629 | 70.890/88.851 | 189883318272 | 2928050176 | 0 | True |
| B2 | 90.775 | 42.722/678.714 | 41.360/54.532 | 70.909/86.770 | 189403471872 | 2444345344 | 2 | True |
| A2 | 89.017 | 106.707/1176.004 | 99.104/119.625 | 88.001/111.195 | 189743656960 | 2829828096 | 2 | True |
| A3 | 89.350 | 106.177/1266.314 | 104.037/126.518 | 82.726/109.142 | 190175219712 | 2903867392 | 2 | True |
| B3 | 88.175 | 51.292/1003.581 | 46.475/63.838 | 73.639/89.730 | 190445670400 | 3462529024 | 2 | True |

### Every root and its host load

Native prefill and client TTFT are separate seconds. Load is the sampled one-minute host load during that root, min/p50/max. All arms purged the Engram companion to zero resident pages and ran the actual model process at nice0.

| Arm | Root | Native prefill s | Client TTFT s | Host load min/p50/max |
|---|---|---:|---:|---:|
| A0 | a0-warm | 30.750404 | 30.867231 | 51.52/53.12/55.49 |
| A0 | a1-warm | 28.883603 | 28.976346 | 54.81/55.34/55.68 |
| A0 | a2-warm | 26.667962 | 26.751757 | 52.90/53.78/54.27 |
| A1 | a0-warm | 29.927144 | 30.030551 | 16.15/16.33/16.74 |
| A1 | a1-warm | 27.387337 | 27.474461 | 16.20/16.59/16.98 |
| A1 | a2-warm | 25.210708 | 25.288997 | 15.62/16.55/17.38 |
| B1 | a0-warm | 28.796818 | 28.885669 | 15.66/16.96/17.69 |
| B1 | a1-warm | 26.078253 | 26.145612 | 14.47/15.02/15.66 |
| B1 | a2-warm | 24.751020 | 24.821911 | 13.33/13.84/14.20 |
| B2 | a0-warm | 32.024941 | 32.111711 | 15.57/17.22/18.48 |
| B2 | a1-warm | 26.805299 | 26.874113 | 17.01/18.55/18.85 |
| B2 | a2-warm | 25.185476 | 25.256385 | 16.73/16.97/17.66 |
| A2 | a0-warm | 30.237667 | 30.348862 | 28.21/29.20/29.45 |
| A2 | a1-warm | 29.065052 | 29.153053 | 29.45/30.68/32.21 |
| A2 | a2-warm | 26.660642 | 26.745631 | 32.21/33.82/33.99 |
| A3 | a0-warm | 29.179402 | 29.288545 | 16.84/18.10/18.58 |
| A3 | a1-warm | 25.889837 | 25.972563 | 16.86/17.37/17.79 |
| A3 | a2-warm | 24.872183 | 24.954299 | 15.83/16.35/16.86 |
| B3 | a0-warm | 28.555262 | 28.644992 | 15.46/16.06/16.35 |
| B3 | a1-warm | 26.292142 | 26.365781 | 16.35/17.52/18.33 |
| B3 | a2-warm | 25.025950 | 25.096650 | 18.36/19.29/19.40 |

### Candidate output control

| Comparison | Observed responses differing | Completed responses differing | Identical-prompt ID/cut mismatches |
|---|---:|---:|---:|
| A0/A1 baseline control | 2/39 (5.128%) | 1/37 | 0 IDs; 0 cuts |
| Pair1 | 1/39 (2.564%) | 0/37 | 0 IDs; 0 cuts |
| Pair2 | 2/39 (5.128%) | 1/36 | 0 IDs; 0 cuts |
| Pair3 | 2/39 (5.128%) | 1/36 | 0 IDs; 0 cuts |
| B1/B2 candidate repeat | 0/40 (0.000%) | 0/37 | 0 IDs; 0 cuts |

First divergence in each pair (later turns may cascade from it):

| Pair | First call | First differing byte by channel | Input/cuts identical | Initial logits identical | Current decode cohorts identical |
|---|---|---|---|---|---|
| 1 | a2-t2 | {'response.output_text.delta': 866} | True | False | False |
| 2 | a2-t1 | {'response.output_text.delta': 58} | True | False | False |
| 3 | a2-t1 | {'response.output_text.delta': 58} | True | False | False |

The A/A control already diverges, and candidate comparisons show no prompt-identical token/cut mismatch. In the completed control and all three pairs, all sequential roots have identical initial-logit hashes and output bytes. Candidate first differences already have differing initial logits; their current and preceding decode cohorts differ, so a later cohort difference alone is not claimed to explain an earlier logit difference. Full-generation round IDs and cohort examples are retained in each output comparison. Output bytes are therefore not asserted identical across served runs; the CPU tokenizer witness proves the narrower ID-identical reuse contract.

### Provenance and checks

Measured runtime source `3482f60cb`, release profile; the flags were temporarily returned to default-off after the failed gate, then restored by lead decision. All Rust and shader source, including the defaults, is byte-identical to measured source `3482f60cb`; preparation, tokenization, tracing and numerical implementations are unchanged. [Pin manifest](2026-11-14-v41-intake-qualification/pin-manifest.json) records source/tree and all executable hashes. All three metallibs match the preceding intake experiment byte-for-byte. Every arm passed six symmetric CPU warmup tests and the released-tokenizer witness (57 cases, 54 incremental hits, 699 boundary checks, full/prefix IDs identical). No GPU/numerical/checkpoint edits were made.

Raw data: `metagross:~/qwen-runs/2026-11-12-v41-prefill/intake-qualify/`. The [frozen reduction](2026-11-14-v41-intake-qualification/results/summary.json), per-call component ledgers, pair exclusions, arm commands and response comparison digests are retained here without transcript text.

Protocol deviation: one read-only `AGENTS.md` file search accidentally exceeded the named repository boundary; it timed out after 10s and was not repeated. No measurement or source was changed by it.

The [recorded arm driver](2026-11-14-v41-intake-qualification/run_qualify_arm.py) fixes the warmup, purge, FD inheritance, niceness and thermal sequence. Its stage paths and start cutoff are historical, not a general-purpose runner.

Cleanup after the closed FAIL gate: the [exact-PID audit](2026-11-14-v41-intake-qualification/results/cleanup-m5.json) found none of 35 registered arm/warmup/harness/dispatch PIDs alive and no live process referencing task paths. No kills were needed. The M5 worktree, Git bundles, build output, temporary build directory and closed-gate pins (including the preceding candidate pins) were removed; raw evidence remains.

CPU contracts passed on measured source `3482f60cb` and the initial default-off record `9eb9bbc1e`. The promoted default-on tree is checked again before reporting.

# M5 artifact choice: results and decisions

Run root: `metagross:~/qwen-runs/2026-11-11-artifact-choice/`. The original
first packet used the same clean `qwen-bench` iter binary for all five
candidates from `a2a1965fe` (`build_dirty=0`, source identity matched). No
model file was changed.

## Correction — 2026-11-11: wide cohort and held-packet rescore

This correction supersedes the first-packet decisions below. The original
four-prompt results remain as historical evidence. The current wide and held
scores used `build/iter/qwen-bench` from the clean M5 worktree at
`bba889ccf`; the successful current-build suite record also identifies
`bba889ccf`. No model files were moved or deleted.

### Cohorts and analysis

The wide cohort is `cohort-v3/`: 11 natural sources (10 code prompts across
six repositories and four languages, plus one prose control), 49,152 prefix
tokens and 512 teacher-forced targets per prompt (5,632 positions per model).
The V4.1 tokenizer hashes match across A/B/C; the GLM hashes match across D/E.
Every paired comparison below also verifies prompt IDs, target IDs and
per-prompt token hashes.

The first cohort in `quality/` is separate: four prompts (three code and one
prose), 49,152 prefix tokens and 1,024 targets per prompt (4,096 positions).
The two cohorts are not pooled. The October held packet is a third, separate
instrument: six frozen documents, 512 prefix tokens and 64 targets each.

Wide quality arm thermal start→end: V4.1 A 0→2, C 2→1, B 1→1; GLM D 0→2,
E 2→1. Held rescoring A/B/C was 0→0 for every arm. These are packet boundary
readings.

All pairwise means below are equal-prompt candidate-minus-reference dNLL in
nats/token; negative means the candidate assigns higher probability to the
same targets. Intervals are central 90% prompt-cluster bootstrap intervals
(20,000 resamples, seed 1,196,182,867). The p-values are exact two-sided
sign tests over prompt means. Top-1 agreement is pooled over target positions.

### Wide V4.1 results

| Pair | All prompts: mean [90% CI], sign-test p | Code only: mean [90% CI], sign-test p | Top-1 agreement |
| --- | --- | --- | ---: |
| B−A | −0.02163 [−0.03104, −0.01286], p=0.0117 | −0.01793 [−0.02575, −0.01035], p=0.0215 | 95.19% |
| C−A | −0.01560 [−0.02590, −0.00726], p=0.0654 | −0.01030 [−0.01501, −0.00521], p=0.1094 | 95.03% |
| B−C | −0.00603 [−0.01260, +0.00064], p=0.5488 | −0.00763 [−0.01426, −0.00065], p=0.3438 | 96.16% |

B and C both improve on A in mean NLL; B separates from A in both the
bootstrap and prompt sign tests. C's bootstrap intervals exclude zero against
A, while its sign tests do not. B−C is unresolved overall. The code-only
bootstrap favors B, but the code-only sign test does not show a consistent
prompt-level difference.

### First cohort results, reported separately

| Pair | All prompts: mean [90% CI], sign-test p | Code only: mean [90% CI], sign-test p | Top-1 agreement |
| --- | --- | --- | ---: |
| B−A | −0.02001 [−0.04013, −0.00460], p=0.1250 | −0.00886 [−0.01462, −0.00310], p=0.2500 | 94.95% |
| C−A | −0.01789 [−0.03392, −0.00356], p=0.1250 | −0.00879 [−0.01595, −0.00163], p=0.2500 | 94.87% |
| B−C | −0.00212 [−0.00621, +0.00204], p=0.6250 | −0.00007 [−0.00293, +0.00278], p=1.0000 | 95.95% |
| E−D | −0.00615 [−0.01151, +0.00011], p=1.0000 | −0.00466 [−0.00946, +0.00014], p=1.0000 | 95.39% |

### October 22 held-packet reconciliation

The current scorer re-read the six frozen source files. All six source hashes
match the held manifest, and all 18 candidate/prompt
`scored_token_ids_sha256_i32le` values match
`held-oct22-expected-ids.json`. A/B/C scored identical 576-token
prefix-plus-target sequences.

On this exact short-context packet, Maya-S C is worse than UD-IQ3_XXS B by
**+0.06320 nats/token** (90% CI **[+0.01989, +0.10657]**); code-only is
**+0.03661** (**[+0.00515, +0.06807]**). Top-1 agreement is **76.82%**.
The sign tests are not significant (all prompts p=0.6875; code p=1.0000),
reflecting the small and mixed set of document directions. The October report's
**+0.079659** Maya-S−IQ3 mean therefore reproduces in direction and remains
large on the held texts, though the current mean is 0.01646 lower.

That held result does not generalize to the wide deep cohort: there B−C is
−0.00603 overall, with an interval crossing zero. The instruments differ in
prefix depth (512 versus 49,152), targets per prompt (64 versus 512), and
corpus mix (three code/three prose held documents versus ten code/one prose
wide prompts). The earlier score also used the October build
`3e63384c`, while the rescore uses the current build. Since exact target IDs
match, tokenization or target alignment does not explain the held result.
The wide result answers a different long-context, code-heavy question.

### GLM wide result

| Pair | All prompts: mean [90% CI], sign-test p | Code only: mean [90% CI], sign-test p | Top-1 agreement |
| --- | --- | --- | ---: |
| E−D | −0.00217 [−0.01278, +0.00845], p=0.5488 | −0.00624 [−0.01563, +0.00281], p=0.3438 | 93.87% |

E trends lower in NLL, but neither interval nor sign test distinguishes it
from D. The wide run scored all 11 prompts and 5,632 targets per candidate,
with the tokenizer fingerprint check enabled.

### Production speed and serving

The current-build `qwen-bench suite --tg 64 --depth 32768 --runs 3` rerun for
A is **14.63–15.23 ms/token** (14.84 mean). The old 30.24 ms/token A row did
not reproduce and remains an unexplained outlier.

The production `v41_batch_depth` causal schedule at 32K, from three
interleaved pairs in `speed-v2/`, gives:

| Width | A, current GSQ-RCO | C, Maya-S | Difference |
| ---: | ---: | ---: | --- |
| 1 | 14.052 ms/token | 16.358 ms/token | A is 2.305 ms/token faster |
| 3 | 25.837 ms/round | 36.113 ms/round | A is 10.276 ms/round faster |

B's production-path decode and serve measurements are in "B speed and serve"
below. In the earlier `a2a1965fe` solo suite, D was 17.71 ms/token and E 18.22
ms/token at depth 32K; there is no comparable current-build GLM width-3
result.

For the three-agent 48K serve, A2 completed at **84.006 tok/s** (15,121
window tokens, 180 seconds, thermal max 1). A1 aborted at thermal level 2.
C1 was rejected at preflight because thermal pressure was already level 2;
C2 and C3-retry aborted at level 2. C serving is **not measured** because no
attempt passed the thermal gate. B serving is measured below.

### Route differences

All four `routes-v2/` packets exited successfully. Width-1 profile overflow
was zero. A uses IQ2_S/IQ2_XS/IQ2_XXS expert kernels, including IQ2_XS routed
gate/up and down work. C has no IQ2_XS route; it uses IQ2_S/IQ2_XXS gate/up
and IQ3_S/IQ3_XXS down kernels. The attention/control sequence and route/HC
control kernels are common. A's shared, attention output and Engram projection
work uses Q5_K kernels; C uses Q8_0 for those rows. At width 3 the direct
capture counts 2,186
dispatches for A and 2,672 for C. No fallback is reported in the route
records; the width-3 capture provides dispatch counts but no separate fallback
counter.

### Current recommendation and exact directories

No V4.1 candidate dominates. B has the strongest quality-first result, with a
small code-only mean edge over C (**0.00763 nats/token**) whose prompt sign
test is not significant. The production A/C path and the only accepted serve
run favor A for throughput: A is 2.305 ms/token faster at width 1 and 10.276
ms/round faster at width 3, and A2 delivered 84.006 tok/s. C improves NLL
against A, but is slower on that production path and has no accepted serve
measurement. **Britt's choice is whether that quality evidence justifies
giving up the measured A/C throughput and serving evidence.** The V4.1
directories are therefore conditional:

- Keep A: B and C are deletable:
  `~/models/unsloth-DeepSeek-V4.1-Flash-1017/UD-IQ3_XXS/` and
  `~/models/DeepSeek-V4.1-Flash-1017-Maya-S/Maya-S/`.
- Keep B: A and C are deletable:
  `~/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/` (including its IQ4_NL
  Engram companion) and
  `~/models/DeepSeek-V4.1-Flash-1017-Maya-S/Maya-S/`.
- Keep C: A and B are deletable:
  `~/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/` (including its IQ4_NL
  Engram companion) and
  `~/models/unsloth-DeepSeek-V4.1-Flash-1017/UD-IQ3_XXS/`.

On-disk sizes from the identity records: A is 155.23 GB backbone plus a
110.60 GB IQ4_NL Engram; B is 394.74 GB; C is 339.48 GB. No one of these
tradeoffs was hidden in the NLL result.

For GLM, **keep D and delete E**. Their wide NLL intervals overlap and the
prompt sign tests are inconclusive; the older solo suite gives D a 0.51
ms/token speed edge. E is 4.36 GB smaller on disk (116.00 versus 120.37 GB),
but the wide cohort does not establish a quality gain. The exact deletable
directory is `~/models/glm-5.3-flash-maya-m/`; keep
`~/models/glm-stock-iq3-download/UD-IQ3_XXS/`.

Raw wide and held analyses are retained under
`metagross:~/qwen-runs/2026-11-11-artifact-choice/analysis/`; model files
remain untouched.

## Original first-packet decisions (superseded by the correction above)

### DeepSeek V4.1 Flash 1017: keep C, delete A and B

Keep **C, Maya-S IQ2_XXS**. Delete the whole current **A GSQ-RCO IQ2_XS**
directory, including its companion IQ4_NL Engram, and the whole **B Unsloth
UD-IQ3_XXS** directory.

B and C are indistinguishable on this four-prompt NLL instrument: B−C mean
signed dNLL is **−0.00212 nats/token**, with equal-prompt central 90% CI
**[−0.00621, +0.00204]**. The code-only mean is **−0.000073**, CI
**[−0.00293, +0.00278]**. Top-1 agreement is **95.95%**. Both load, stream
Engram from disk and pass the three-agent 48K fit. C is **3.33× faster** than
B at solo 32K decode: **17.11 vs 57.07 ms/token**.

Both B and C score lower NLL than A in the paired cohort, with intervals that
exclude zero. B−A is **−0.02001** (90% CI **[−0.04013, −0.00460]**) and C−A
is **−0.01789** (90% CI **[−0.03392, −0.00356]**); their code-only CIs also
exclude zero. A remains fast at **30.24 ms/token**, but it is not the best
quality artifact the machine can fit under this deeper measurement.

### GLM-5.3 Flash: keep D, delete E

Keep **D, stock UD-IQ3_XXS**. Delete the whole **E Maya-M IQ2_S** directory.

E's mean NLL is lower by **0.00615 nats/token**, but the all-prompt 90% CI
**[−0.01151, +0.00011]** includes zero. The code-only mean is **−0.00466**,
CI **[−0.00946, +0.00014]**. Top-1 agreement is **95.39%**. Both fit three
48K sessions; neither has a separate Engram table. D is faster at **17.71
ms/token** versus E at **18.22 ms/token**, so retain D under Britt's
quality-first rule when this instrument cannot distinguish them.

## Tokenizer and model identity

The tokenizer fingerprint hashes the complete tokenizer model/pretokenizer,
vocabulary, token types, merges, and special-token settings. Architecture is
reported separately because D/E use the equivalent `glm5next` and `glm5-next`
architecture spellings.

| Candidate | Artifact | Tokenizer SHA-256 | Files on disk | Retained weights | Disk Engram |
| --- | --- | --- | ---: | ---: | ---: |
| A | V4.1 GSQ-RCO IQ2_XS + companion IQ4_NL | `9bf88317177f…24ed9fe5` | 155.23 GB backbone + 110.60 GB companion | 155.22 GB | 110.60 GB |
| B | V4.1 Unsloth UD-IQ3_XXS | same as A | 394.74 GB | 233.45 GB | 161.28 GB |
| C | V4.1 Maya-S IQ2_XXS | same as A | 339.48 GB | 178.19 GB | 161.28 GB |
| D | GLM stock UD-IQ3_XXS | `e1f9c598a76c…818a5de88` | 120.37 GB | 117.56 GB | none |
| E | GLM Maya-M IQ2_S | same as D | 116.00 GB | 112.40 GB | none |

Every candidate passed header/tokenizer inspection. All quality results therefore
use the same token IDs within each family. The initial fingerprint included
the GGUF architecture string and flagged D/E; inspection showed that was only
the architecture alias. The final tokenizer hash excludes that spelling and
retains all tokenizer fields. Full hashes and shard paths are in
`metagross:~/qwen-runs/2026-11-11-artifact-choice/identity/`.

## Quality method and paired results

Each candidate scored the same four complete natural-source prompts: three
code prompts and one prose control. For each prompt, 49,152 source tokens were
prefilled and the next 1,024 real tokens were teacher-forced. No prompt or
target cycling was used. Per-position NLL and top-1 IDs, prompt hashes and
token-ID hashes are retained in the raw quality JSON files.

The analysis bootstraps prompt clusters with 20,000 resamples (seed
1,196,182,867), reporting an equal-prompt mean and central 90% interval for
all prompts and the three code prompts. Negative candidate-minus-reference
dNLL means the left artifact assigns higher probability to the same target.

| Pair | All-prompt mean dNLL (90% CI) | Code-only mean dNLL (90% CI) | Pooled top-1 agreement |
| --- | ---: | ---: | ---: |
| B − A | −0.02001 [−0.04013, −0.00460] | −0.00886 [−0.01462, −0.00310] | 94.95% |
| C − A | −0.01789 [−0.03392, −0.00356] | −0.00879 [−0.01595, −0.00163] | 94.87% |
| B − C | −0.00212 [−0.00621, +0.00204] | −0.00007 [−0.00293, +0.00278] | 95.95% |
| E − D | −0.00615 [−0.01151, +0.00011] | −0.00466 [−0.00946, +0.00014] | 95.39% |

The NLL intervals separate A from both higher-quality V4.1 alternatives. They
do not separate B from C or D from E. The four prompt clusters (three code,
one prose) limit population-level precision even though each model scored
4,096 targets. The per-position analysis is retained; it does not treat tokens
as independent bootstrap units.

## Solo decode speed

`qwen-bench suite --tg 64 --depth 32768 --runs 3 --seed 1`, default family
loader/prefill and GPU decode path, three retained repetitions. B=1. Timings
include the ordinary full-logits readback path.

| Candidate | Tokens/s | ms/token | Sample tokens/s | Thermal before → after |
| --- | ---: | ---: | --- | --- |
| A | 33.07 | 30.24 | 34.75 / 30.36 / 34.09 | 0 → 0 |
| B | 17.52 | 57.07 | 17.37 / 18.11 / 17.08 | 0 → 0 |
| C | 58.43 | 17.11 | 58.46 / 58.45 / 58.38 | 0 → 1 |
| D | 56.48 | 17.71 | 56.63 / 56.45 / 56.34 | 0 → 0 |
| E | 54.89 | 18.22 | 54.84 / 54.64 / 55.19 | 0 → 1 |

The optional B=3 decode timing was not run: the maintained ordinary suite has
only a solo decode row. The separate fit packet allocates and prefills three
real sessions, but does not claim concurrent B=3 decode throughput.

## Three-agent 48K fit

The fit probe loaded each artifact, allocated three live 49,152-position
sessions, then prefetched each with a distinct real 48K code prompt while
keeping all sessions alive. It recorded the runtime's per-session 4,096-row
V4.1 CED or GLM 2,048-row packed prefill geometry and actual post-prefill
Metal allocation signals. All five admitted all three sessions and completed
all three prefills.

| Candidate | 3 sessions and 3 prefills | Retained weights | Engram reader | Metal bytes after sessions | Host headroom after | Thermal |
| --- | --- | ---: | --- | ---: | ---: | --- |
| A | pass | 155.22 GB | disk, 110.60 GB | 170.41 GB | 57.08 GB | 0 → 0 |
| B | pass | 233.45 GB | disk, 161.28 GB | 234.46 GB | 40.42 GB | 0 → 0 |
| C | pass | 178.19 GB | disk, 161.28 GB | 179.20 GB | 69.04 GB | 0 → 0 |
| D | pass | 117.56 GB | none | 127.25 GB | 60.29 GB | 0 → 2 |
| E | pass | 112.40 GB | none | 122.50 GB | 108.35 GB | 2 → 2 |

M5 reports 274.88 GB physical memory. All five fits completed without changing
memory-admission policy. B has the narrowest observed headroom but passed the
three-session allocation and all three prefills. No other large model process
was running when the packets began. Thermal pressure was captured at packet
boundaries, not sampled continuously. Swap usage began at 146.62 MB before the
V4.1 series and was 150.50 MB at the end of the GLM series; the maximum observed
host free-memory reading during B's quality run was 9% before it recovered.

## Earlier evidence and conflicts

The earlier V4.1 witness (GSQ 2.1421, Q2 2.1539, IQ3 2.1543 over 384 targets)
called those artifacts indistinguishable. This longer code-at-depth cohort
separates both B and C from A while still finding B and C indistinguishable.

The request said Maya-S had not been compared. `docs/bench/2026-10-22-maya-s-eval.md`
already compares Maya-S with UD-IQ3_XXS on 384 held targets and reports Maya-S
+0.079659 NLL. The new cohort is a deeper same-token comparison and is not
represented as the first comparison.

The request also said GLM D/E had not been compared for quality. PERF-LOG's
2026-10-17 M5 blind eight-case engineering-answer pilot compared these same
artifacts and found no clear quality winner. It was not teacher-forced NLL; the
new run supplies that missing metric.

The M5 snapshot reported 1.8 TiB available under `metagross:~/qwen-runs`, rather than
the stated ~28 GiB. Probe outputs stayed under the requested run root. The
external Aven annotation for V41-HFAB could not be posted because Aven is not
available as a connected tool/plugin in this session; the result is recorded
here and in PERF-LOG.

## B speed and serve — 2026-11-12 (M5 UTC)

This section closes the gap named in the correction above. B is
`~/models/unsloth-DeepSeek-V4.1-Flash-1017/UD-IQ3_XXS/`. All GPU work ran under
the canonical Metal lease; no model file was changed. The branch was rebased
onto main `f6475cb2e` first; the M5 worktree used commit `3c3249b0`.

### Production-path decode

The `v41_batch_depth` example was rebuilt from the rebased tree with
`--profile iter --features perf-ablations` and placed at the path the existing
`speed-v2/` runner expects. The decode command is the same reference mode as
`speed-v2/`: `MODEL 32768 WIDTH 32 bounds`. Three interleaved A/B pairs per
width were run (`A B A B A B`); A was re-measured in every pair. Each arm
contributes two Reference passes per run, six per arm.

| Width | A mean | B mean | B − A | B / A |
| ---: | ---: | ---: | ---: | ---: |
| 1 (ms/token) | 13.897 (SD 0.056, n=6) | 17.580 (SD 0.029, n=6) | +3.683 | +26.5% |
| 3 (ms/round) | 25.603 (SD 0.108, n=6) | 39.621 (SD 0.072, n=6) | +14.018 | +54.8% |

The width-1 pair means are A 13.854 / 13.899 / 13.939 and B 17.569 / 17.611 /
17.560. The width-3 pair means are A 25.522 / 25.689 / 25.597 and B 39.658 /
39.638 / 39.566. B is slower in all six pairs. At width 3, B's 39.621
ms/round is 13.207 ms/token-equivalent versus A's 8.534. Every accepted arm
sampled thermal level 1, never 2; one A pair waited 30 s for the thermal
settle. The re-measured A is consistent with the earlier session: 13.897 vs
14.052 ms/token and 25.603 vs 25.837 ms/round. The first attempt at this
campaign aborted all six arms at ambient level 2; the `campaign-status.jsonl`
begin/end rows retain those aborts, and the later pair runs reused the IDs.

### Three-agent 48K serve

Same settings as `serve-v2/`: `plan.json` seed 20261109, real agent sampling,
180 s window, ramp 15, 48,000 warm tokens, context 196,608, warm-cool 45,
cool 30, thermal gate 1. The only scheduling addition was a 240 s stable
thermal-0 settle inside the lease before each arm; the harness flags and gate
were unchanged. Raw runs and per-agent ITL are in `serve-v2/` and
`serve-v2/itl-*.json`; the consolidated record is `b-measurement-summary.json`.

| Arm | Pair / order | tok/s | Thermal max | prefill s | partial_prefill_s | ITL p50 s | ITL p99 s | Per-agent p99 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A `Ab2` | 1, second | 85.106 | 0 | 32.43 | 0.00 | 0.02764 | 0.05507 | 0.0573 / 0.0547 / 0.0549 |
| A `Ab3` | 2, first | 81.856 | 1 | 34.10 | 0.00 | 0.02813 | 0.05577 | 0.0709 / 0.0543 / 0.0558 |
| B `Bb1r` | 1, first | 48.806 | 0 | 41.21 | 0.00 | 0.04157 | 0.16194 | 0.1630 / 0.3738 / 0.1570 |
| B `Bb2` | 2, second | 50.517 | 1 | 39.80 | 0.00 | 0.04225 | 0.15997 | 0.1580 / 0.1650 / 0.1580 |

Means: A **83.481 tok/s**, B **49.661 tok/s**; aggregate ITL p99 **0.0554 s**
for A and **0.1610 s** for B. Partial prefill is zero for all four arms; the
round-budget prefill estimate is 32.4–34.1 s for A and 39.8–41.2 s for B. B's
warm-up is longer: 202–220 s versus 169–173 s.

Thermal history: in serve, the A1 attempt was a path collision (below), A
`Ab1` aborted at level 2 at the end of its window, A `Ab1r` aborted at level 2
during warm-up cooling, and B `Bb1` aborted at level 2 at the end of its
window. With the 240 s settle, `Bb1r`, `Ab2`, `Ab3`, and `Bb2` all completed.
B tripped the gate once, so the two-trip stop rule was not reached. The
external load is not under this lease: `shadps4` processes from the ghost
infrastructure ran at roughly 800–1160% CPU throughout and are the likely
reason the same runs tripped without the settle. This is a shared-machine
conflict, not a B property.

### Memory and output

B ran at the memory edge. During the two accepted B windows:

| Run | Free pages min | Inactive min | Compressor max | Swap max |
| --- | ---: | ---: | ---: | ---: |
| B `Bb1r` | 0.78 GB | 7.77 GB | 10.53 GB | 180.5 MB |
| B `Bb2` | 0.12 GB | 8.20 GB | 14.34 GB | 182.4 MB |

A's corresponding free-page minima were 10.94 and 5.24 GB, with compressor
maxima near 1.1 GB; swap max was 180.9 MB. Swap stayed within about 30 MB of
the pre-campaign baseline (146–150 MB), so there was no swap growth, but B's
free pages and compressor headroom are clearly narrower. These are live-window
readings, not the earlier post-prefill fit headroom of 40.42 GB.

B's outputs are coherent. Sampled non-warm responses contain ordinary English
code-review prose, zero U+FFFD replacement characters, 0.84–0.97 alphabetic
fraction, and no repeated 8-word shingle in the longest response (4,581
characters). Example from `Bb2/a2-t14`: "The reader is serving the same fixed
V4.1 header on every path... I will not call it again; the report continues
against the supplied corpus."

### Record collision

A runner ID collided with the earlier lane: an aborted attempt used `A1` and
overwrote the wrapper-level `A1.before`, `A1.after`, `A1.exit` and driver logs.
The `serve-v2/A1/` run directory is intact (`failed.json` records
`thermal_abort: true`, thermal max 2, no summary), and the overwritten wrapper
files are retained with a `.collision-20261112T0325Z` suffix. The collision
attempt did not load a model. Subsequent IDs use `Ab*`/`Bb*`.

### Combined size/speed table

| Candidate | Quality vs A (nats/token) | Width-1 ms/token | Width-3 ms/round | Serve tok/s | ITL p99 | Resident weights |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A GSQ-RCO IQ2_XS | reference | 13.897 | 25.603 | 85.106 / 81.856 (mean 83.481) | 0.0554 s | 155.22 GB |
| B Unsloth UD-IQ3_XXS | −0.02163 [−0.03104, −0.01286], p=0.0117 | 17.580 | 39.621 | 48.806 / 50.517 (mean 49.661) | 0.1610 s | 233.45 GB |
| C Maya-S IQ2_XXS | −0.01560 [−0.02590, −0.00726], p=0.0654 | 16.358 | 36.113 | not measured (thermal gate) | not measured | 178.19 GB |

C's speed row is from the earlier A/C session; the prior A row there was
14.052 ms/token and 25.837 ms/round, so C is presented against that session.
Resident weights are the retained backbone from the identity records; on-disk
totals including the Engram companion are A 265.83 GB, B 394.74 GB, and C
339.48 GB (Engram: A 110.60 GB, B and C 161.28 GB).

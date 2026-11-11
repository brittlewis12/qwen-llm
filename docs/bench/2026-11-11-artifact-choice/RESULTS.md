# M5 artifact choice: results and decisions

Run root: `metagross:~/qwen-runs/2026-11-11-artifact-choice/`. Build and all
five candidates used the same clean `qwen-bench` iter binary from
`a2a1965fe` (`build_dirty=0`, source identity matched). No model file was
changed.

## Decisions

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

The M5 snapshot reported 1.8 TiB available under `~/qwen-runs`, rather than
the stated ~28 GiB. Probe outputs stayed under the requested run root. The
external Aven annotation for V41-HFAB could not be posted because Aven is not
available as a connected tool/plugin in this session; the result is recorded
here and in PERF-LOG.

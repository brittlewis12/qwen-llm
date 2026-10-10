# Flash frontier frozen quality packet

## Acquired results — 2026-10-09

**UD: PASS. GSQ: INCONCLUSIVE. Keep the cross-artifact promotion parked.**
These are the independently recomputed frozen-policy verdicts, not test-exit
statuses. Both acquisitions completed without runtime errors. Owner-reported
elapsed acquisition times were 301 seconds for GSQ and 401 seconds for UD;
neither duration is an A/B performance measurement. No additional GPU run was
performed for this analysis. Protocol, inputs and scoring remain unchanged.

Raw acquisitions: [GSQ](gsq-quality.jsonl), [UD](ud-quality.jsonl).
The unchanged [independent analyzer](analyze.py) produced
[quality-analysis.json](quality-analysis.json), checking frozen fixture and
artifact bindings, acquisition order, summaries, target IDs, per-token NLL
records, and independent retrieval parsing.

Positive delta means candidate B is worse. Each artifact has eight documents
with 64 source continuation targets each; pooling below is within an artifact
only. The limit is the predeclared ln(1.01) = 0.009950330853168092 nats/token.

| Artifact | Mean NLL A | Mean NLL B | Pooled B−A | Descriptive paired-document 95% interval | NLL gate | Overall |
| --- | ---: | ---: | ---: | --- | --- | --- |
| GSQ | 0.396859141566 | 0.394448679483 | -0.002410462083 | [-0.005952029724, +0.000629456969] | PASS | INCONCLUSIVE |
| UD | 0.374206460843 | 0.374206460843 | +0.000000000000 | [+0.000000000000, +0.000000000000] | PASS | PASS |

Intervals use the frozen 20,000 document-bootstrap resamples, seed 20261009,
with eight resampling units. They are descriptive, not an extra gate or a
statistical non-inferiority claim. GSQ’s interval crosses zero; this screen
does not establish a quality improvement. UD’s zero recorded NLL deltas do not
establish bitwise identity of complete logits or general model equivalence.

| Document | GSQ NLL A | GSQ NLL B | GSQ B−A | UD NLL A | UD NLL B | UD B−A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bash | 0.226652733 | 0.229155914 | +0.002503181 | 0.130497682 | 0.130497682 | +0.000000000 |
| csh | 0.770522901 | 0.760241300 | -0.010281602 | 0.746203143 | 0.746203143 | +0.000000000 |
| curl | 0.001731332 | 0.001714999 | -0.000016333 | 0.003217364 | 0.003217364 | +0.000000000 |
| find | 0.020695017 | 0.018713997 | -0.001981020 | 0.055727555 | 0.055727555 | +0.000000000 |
| launchctl | 1.103290403 | 1.092704859 | -0.010585544 | 1.044350978 | 1.044350978 | +0.000000000 |
| security | 0.003769354 | 0.003584859 | -0.000184495 | 0.003404096 | 0.003404096 | +0.000000000 |
| tar | 0.076089474 | 0.077592434 | +0.001502960 | 0.079663853 | 0.079663853 | +0.000000000 |
| tcpdump | 0.972121917 | 0.971881073 | -0.000240844 | 0.930587017 | 0.930587017 | +0.000000000 |

Real-target top1 accuracy is unchanged within each artifact: GSQ A/B both
467/512 (91.2109375%); UD A/B both 465/512 (90.8203125%). Exact document
NLLs and document accuracy are retained in the analysis JSON.

### Known answers and exact generated text

| Artifact | Case | A parsed / correct | B parsed / correct | Frozen-policy status | Generated IDs per arm | Final position |
| --- | --- | --- | --- | --- | ---: | ---: |
| GSQ | retrieval_before | no / no | no / no | INCONCLUSIVE_INVALID_FIXTURE | 21 | 4117 |
| GSQ | retrieval_after | no / no | no / no | INCONCLUSIVE_INVALID_FIXTURE | 19 | 4115 |
| UD | retrieval_before | yes / yes | yes / yes | PASS | 24 | 4120 |
| UD | retrieval_after | yes / yes | yes / yes | PASS | 22 | 4118 |

All eight greedy trajectories stop on EOS before the 64-token budget.
For each artifact/case, A and B generated identical bytes and token IDs. The
following blocks reproduce the exact generated text (no added marker):

GSQ, retrieval_before, both A and B:

```text
{"code":"cobalt-meadow-7301","record":"J-42"}
```

GSQ, retrieval_after, both A and B:

```text
{"code":"silver-orbit-8264","record":"M-63"}
```

UD, retrieval_before, both A and B:

```text
FINAL_JSON: {"code":"cobalt-meadow-7301","record":"J-42"}
```

UD, retrieval_after, both A and B:

```text
FINAL_JSON: {"code":"silver-orbit-8264","record":"M-63"}
```

GSQ contains the expected field values but omits the required `FINAL_JSON:`
marker. Under the unchanged first-marker parser, neither A nor B parses.
The predeclared incumbent-parse-failure rule therefore makes each GSQ case
`INCONCLUSIVE_INVALID_FIXTURE`; it is neither a retrieval pass nor evidence of
a candidate-specific factual regression. No post-hoc marker relaxation or
content-only rescore has been used. UD satisfies the exact output contract in
both cases and both arms.

### Finiteness, positions and acquisition validity

Per artifact, all 20 fresh prefills follow the recorded A ranges
`[0,2048), [2048,2051), [2051,4096)` or B ranges
`[0,2048), [2048,4096)`, with three/two commands respectively and position 4096.
All 1,024 scored rows per artifact report 248,320 finite logits and finite NLL;
every target ID matches the frozen source continuation. Positions before
feeding are exactly 4096 through 4159, and all 16 natural trajectories end at
4160. Paired natural position/QSA/PLE records match. Every recorded QSA length
also equals its recorded position. Retrieval terminal positions equal 4096
plus the generated-ID counts above, and paired retrieval terminal states match.
Raw generated bytes decode exactly to the recorded text. Runtime completion
also entails the harness’s finite unscored-terminal-row checks. Full logit
arrays are not retained, so this analysis validates the recorded finite flags
and recomputes aggregates; it does not independently recalculate logsumexp
from full logits. Fixture/protocol hashes match and both artifact revalidations
report unchanged shards.

### Decision and next qualification

UD alone meets this frozen quality policy and is eligible for promotion
consideration after a clean performance repeat. Before any such consideration,
repeat ordinary A/B full-call measurements on the intended UD scope with
production arithmetic and routers, paired ABBA ordering, reset outside timing,
valid complete GPU timestamps and wall times, and no timed observers/census.
Retain every attempt and review paired drift; the quality packet supplies no
new speed claim.

For a common UD/GSQ production change, keep the candidate parked: GSQ has not
passed the required known-answer component. Its NLL passes and raw answers
are encouraging, so this is an unresolved output-contract qualification, not
a demonstrated numerical quality regression. Any follow-up contract experiment
must be separately versioned and declared before new acquisition, preserving
all current inputs, policy, raw attempts and verdicts. Do not relabel these
GSQ results as PASS. No artifact-specific production allowlist is proposed by
this report.

### Result bindings

| File | SHA256 |
| --- | --- |
| [gsq-quality.jsonl](gsq-quality.jsonl) | `717848cd7d8d1e9c6e007e248c3e86a203eeeaa90d3abd51da24d877071ba334` |
| [ud-quality.jsonl](ud-quality.jsonl) | `30f5857a89b722c69131e950ba10db6bf56575bc14e3aa79d5997e2671c95dbf` |
| [quality-analysis.json](quality-analysis.json) | `0199660681b6edb73dc73cb1fe7ec2d4c349d8760e9bf6631803cbb1afb37d8d` |
| [analyze.py](analyze.py) | `80c55e896796ca6f018d916b318cb332c4d1353cf341e08d4eb4df7c1c6ac4ad` |

Reproduce the independent analysis without overwriting the retained result:

```sh
python3 docs/bench/2026-10-09-flash-frontier-quality/analyze.py \
  docs/bench/2026-10-09-flash-frontier-quality/gsq-quality.jsonl \
  docs/bench/2026-10-09-flash-frontier-quality/ud-quality.jsonl
```

## Fixture freeze and original invocation

Fixtures frozen CPU-only on2026-10-09, before any GPU acquisition. No model
output was used to choose inputs. See PROTOCOL.md for the predeclared policy.

`fixtures.json` SHA256:
`d592ccda9273cdc03b26bca284e8bb48e4aa8db404a3b140c2a51b37f009e73e`

This is superseding pre-acquisition freeze revision 2. No GPU run preceded
this repair. Review found that Z-99 duplicated the correct answer; it now uses
the distinct wrong code violet-drift-9027. Freeze/check and the Rust fixture
validation require the correct code exactly once, solely in its authoritative
fact. The eight manuals and their source continuation IDs are unchanged.
Metrics, thresholds and acquisition order are unchanged.

The original 17 files are preserved byte-for-byte in
[archive/preacquisition-v1/](archive/preacquisition-v1/), with
[an archive hash inventory](archive/preacquisition-v1-sha256.json).
Original fixture SHA256: `85ff5333762cbaae20049f696504ffd3879930967457a7ffa849b784b0459de0`.
Treat the archive as immutable historical evidence; run only the current
producer/harness. Archived instructions intentionally remain unchanged.
GPU raw outputs and analysis below use persistent repository paths; /tmp is
used only for the disposable uv package cache.

`produce.py --check` verified all hashes, released tokenizer identity and full
rendered-source tokenization on both UD and GSQ with the existing native CPU-only
qwen-tok. Eight complete rendered manuals have55883/50317/59781/6559/5853/
10649/10088/18800 tokens in frozen order. Natural fixtures retain first4160 IDs;
prompt4096 plus64 source continuation targets, no repetition. Complete rendered text is
retained alongside source/render hashes. Targets are from the original manuals,
not generated by either inference arm.

The two fresh no-thinking retrieval prompts are exactly4096 IDs. Their
independent ratified fact spans are[1033,1074) and[3083,3122). CPU sizing changes
only irrelevant synthetic filler; complete facts, question and template suffix
are preserved. Both actual GGUF templates render byte-identical prompts;
actual templates and message inputs are retained. Expected answers are J-42 /
cobalt-meadow-7301 and M-63 / silver-orbit-8264.

## CPU reproduction, no build or GPU

```sh
UV_CACHE_DIR=/tmp/flash-frontier-quality-uv uv run \
  docs/bench/2026-10-09-flash-frontier-quality/produce.py --check
python3 docs/bench/2026-10-09-flash-frontier-quality/analyze.py --self-test
```

Producer dependency: pinned jinja2==3.1.4 via uv; existing CPU-only qwen-tok
(default /Users/tito/code/qwen-llm/target/release/qwen-tok), local GGUF headers,
/usr/bin/mandoc and /usr/bin/col. --ud/--gsq/--qwen-tok override local locators.
The manifest binds the binary/tool/source hashes used in this freeze. --freeze
is for a new empty acquisition directory and refuses an existing freeze marker;
do not delete/rewrite frozen inputs in response to model output.

Rust CPU test (added, not executed during implementation):
`qwen4exp_runtime::tests::prefill_map::frontier_schedule::frontier_quality::frontier_quality_metric_scorer_and_boundaries`.
Checks stable f64 NLL, tie-breaking, invalid rows, independent JSON parsing,
duplicate required keys, frozen hashes and prompt/target/fact boundaries.

## Owner GPU invocation (not run during implementation)

Unset ambient QWEN_* and QWEN4EXP_* arithmetic/diagnostic overrides; only the
normal QWEN_METAL_LEASE_WAIT is allowed. No capture, census or profiling.

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  FLASH_FRONTIER_QUALITY_ARTIFACT=gsq \
  FLASH_PREFILL_MODEL=/Volumes/wdblack/weights-archive/qwen3.8-flash-next-gsq-rco-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  FLASH_PREFILL_OUT=docs/bench/2026-10-09-flash-frontier-quality/gsq.jsonl \
  cargo test --release -p qwen-llm --lib \
  qwen4exp_runtime::tests::prefill_map::frontier_schedule::frontier_quality::native_frontier_quality \
  -- --ignored --exact --nocapture --test-threads=1
```

Run separately with `FLASH_FRONTIER_QUALITY_ARTIFACT=ud`,
`FLASH_PREFILL_OUT=docs/bench/2026-10-09-flash-frontier-quality/ud.jsonl` and:
`FLASH_PREFILL_MODEL=/Volumes/wdblack/weights-archive/qwen3.8-flash-next/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf`.

The packet requires normal production lease/wired-memory and native combined
load admission, selected scratch, capacity4160. The existing diagnostic CPU
gate conservatively prices one checkpoint plus96MiB although this packet
allocates no checkpoint. Fresh arms reset and zero persistent storage. No
full-vocabulary history or model-sized CPU banks. Runtime revalidates frozen
text/token arrays using its current native tokenizer before model load.

Per artifact:20 fresh4096 prefills,1024 teacher-forced transitions and at most
256 greedy transitions. Both artifacts:40 prefills,2048+up to512 transitions.
Historical resident4K timings imply roughly5.5–6min for prefills alone;
reserve15–20min including decode/loading, not a measured runtime or a timeout.
NLL rows and terminal rows must all be finite. No bitwise or KL gate.

```sh
python3 docs/bench/2026-10-09-flash-frontier-quality/analyze.py \
  docs/bench/2026-10-09-flash-frontier-quality/gsq.jsonl \
  docs/bench/2026-10-09-flash-frontier-quality/ud.jsonl \
  --out docs/bench/2026-10-09-flash-frontier-quality/analysis.json
```

Analyzer validates completion, acquisition order, every target/NLL/position,
causal metadata across arms, independent answer parsing and summary arithmetic.
It reports each artifact separately, with eight document deltas, actual-target
accuracy and a document-level bootstrap interval (descriptive, not a new gate).
Incumbent parsed-but-wrong output never excuses wrong candidate output. An
incumbent parse failure makes that task inconclusive as in the historical
policy. Another valid task failure or NLL failure still fails the artifact.
Incomplete/invalid acquisition cannot pass. Preserve every attempt.

Passing means this technical-text and controlled no-thinking screen passes,
not general language/chat/agent equivalence or statistical non-inferiority.
Performance qualification and owner promotion review are separate. Production
behavior is unchanged; this packet does not enable the candidate by default.

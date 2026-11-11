# V4.1 / GLM artifact choice on M5 Ultra

Status: measurement packet for DeepSeek V4.1 Flash 1017 and GLM-5.3 Flash
third-party GGUF choices. No model files are modified. The `qwen-bench
artifact-choice` subcommand fingerprints tokenizer metadata, emits per-position
teacher-forced NLL/top-1 observations, and tries three live 48K sessions with
the selected production prefill workspace. `qwen-bench suite` supplies the
ordinary solo decode timing.

## Frozen cohort

`scripts/bench/prepare_artifact_choice.py` assembles four non-cycled prompts:

- the frozen three-agent Rust code corpus from the 2026-11-07 serving run;
- `kernels/moe.metal` followed by `kernels/mat_vec.metal`;
- `scripts/profile/dflash_k0s.py`;
- `docs/PERF-LOG.md` as the prose control.

The scorer tokenizes each entire source with the artifact's native GGUF
tokenizer, starts from the first 49,152 tokens, then records 1,024 real next
tokens. It fingerprints architecture, tokenizer model/pretokenizer, complete
tokens/types/merges and special-token settings before any comparison. All
quality invocations require that preverified identity hash. Candidate minus
reference dNLL is analyzed as paired per-position NLL differences. Confidence
intervals bootstrap prompt clusters (20,000 resamples, central 90% and
one-sided 95% upper bound); the code-only interval is also reported. Top-1
agreement is the percentage of target positions with the same greedy token.

There are four prompt clusters, three code and one prose, so token count does
not remove prompt-level uncertainty. The packet reports that limit instead of
treating thousands of target positions as independent samples.

## Run paths

Build only `qwen-bench`, with Cargo output rooted in
`~/qwen-runs/2026-11-11-artifact-choice/target`. The cohort text files, identity
reports, raw scores, speed rows and fit reports are all stored below that run
directory. Each model's GPU packet is held through
`scripts/bench/with-metal-lease` for at most 14 minutes. Build outside the GPU
lease.

Speed is the production suite at B=1, depth 32,768, 64 generated tokens and
three retained repetitions. B=3 is optional and is omitted if the serial
three-agent memory proof or lease budget is tight.

Fit uses the production model loader and a 49,152-token capacity. It attempts
three live per-agent KV states and the default prefill workspace, recording
per-agent admission geometry, estimated bytes, retained weight bytes, actual
Metal allocation signals, Engram backing and thermal-pressure readings. An
admission or load error is retained as a failed candidate result.

## Prior evidence to reconcile

The request noted that Maya-S had no earlier comparison. Existing
`docs/bench/2026-10-22-maya-s-eval.md` compares the V4.1 1017 Maya-S six-shard
artifact against Unsloth UD-IQ3_XXS on 384 held targets. It reports Maya-S
+0.079659 NLL, lower top-1, and explicitly warns the six-document cohort is
descriptive only. The new 4,096-target deep cohort extends that evidence and
is not represented as the first comparison.

The request also said GLM D vs E had not been compared for quality. PERF-LOG's
2026-10-17 three-way GLM pilot and
`docs/bench/2026-10-16-glm-quality-comparison/` compare this same Maya-M IQ2_S
artifact with stock UD-IQ3_XXS using eight generated engineering responses,
two blind reviews and exact-fact counts. The scores were Maya 23/23, IQ3 23/24
and 22/24 reviewer-score agreement; it found no clear quality winner. That is
real-response qualitative evidence, not teacher-forced NLL. The new deep cohort
adds the requested paired numerical comparison.

The original 2026-10-25 V4.1 witness also reports GSQ 2.1421, Q2 2.1539 and
IQ3 2.1543 on only 384 targets and calls them indistinguishable. It does not
measure the new 1017 UD-IQ3_XXS candidate in a deeper code-at-depth cohort.

## Analysis and decision

Run `scripts/bench/analyze_artifact_choice.py` on the raw quality JSON files.
This directory's `RESULTS.md` is filled in after the M5 packets complete and
names the recommendation and deletable artifact trees.

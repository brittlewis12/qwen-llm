# V4.1 GGUF Operation

The [requirement](DEEPSEEK-V41-STRATEGY.md) is ordinary GGUF family support,
independent of whether a particular artifact fits the local machine.

## Runtime

`general.architecture = "deepseek41"` selects the shared `run`, `serve`, `info`,
and benchmark paths. Geometry, ownership schedules and Engram hashes come from
metadata. Base and pruned expert counts are not an allowlist. No adjacent HF
configuration, safetensors loader, converted-weight cache or conversion command
exists in the engine.

```sh
qwen info -m /path/to/model.gguf --json
qwen run -m /path/to/model.gguf --user 'Say hello.' --max-tokens 32
qwen serve -m /path/to/model.gguf --max-context-tokens 4096 --max-tokens 512
```

The chosen convention is llama.cpp PR 28696: standard GGML expert/dense types
and Engram tables `[256, rows]` decoded through the shared GGML codec. MXFP4
experts stay packed. Engram reads only requested rows from disk and is excluded
from resident Metal weight banks. Its dtype is a codec capability, not a quant
quality policy; the primary converted artifact uses Q8_0, but other supported
GGML types work without new Engram kernels.
Known metadata spellings are aliases. Unsupported consumed encodings are named
in errors; unused vision/MTP tensors and vision router biases are not bound.
The converter revision is provenance, never an accepted-file restriction.

Serving uses the official recipe renderer/parser, the common startup/request
spine and typed HTTP 503 `memory_admission_denied` responses. CLI statistics
use the existing v2 schema; benchmarks use the existing family harness. Each
request reuses the longest available exact causal prefix from live state or RAM
snapshots. Cold prefill advances in 256-row layer-major chunks without computing
unused prompt logits. Dense projections and supported routed experts use shared
NativeF32 matrix kernels, as does attention's value reduction at 256 or more
attended key/value rows; cache updates remain causally ordered. CED, speculation
and Lens are not implemented for V4.1.

## Verification

The repair passes 93 V4.1 library tests, including required-Metal/API-validation runs,
including nonzero GGUF load-to-forward tests at 272/320/384 experts, quantized
projection tests, ownership transitions and bounded Engram reads. Shared
CLI/server/benchmark fixtures cover dispatch, request preparation, cancellation,
typed refusals and statistics. Full committed-tree contracts and release
workspace compilation pass at `ee232eac`.

Actual Rust `gguf::plan()` succeeds against all ten mrfakename 1017 Q8_0 shard
headers: 1,046 tensors, 384 experts, 120 MXFP4 banks. The resident payload is
278.508 GiB and disk Engram is 194.555 GiB. The architecture is supported even
though that resident artifact exceeds a 256 GiB host. Header inspection reads
no model payload and does not initialize Metal. The repository revision used
for this check is `1df54e7d1b1e442051ad393ce871870a929f243e`.

The same all-shard Q2_K check now passes through the shared Engram codec:
186.260 GiB resident and 60.083 GiB disk Engram, with 384 experts. This establishes
planning compatibility, not its numerical quality or measured local fit.
Header checks used about 5.8 MB per artifact, not full downloads.

## First End-to-End Comparison

The existing offline HF CPU oracle remains the first conversion-plus-execution
signal. The chosen converter requantizes dense FP8 and Engram to Q8_0, so its
errors must be characterized, not tested against the old lossless-path bound.
The independent GGUF-reading CPU oracle measures execution error alone.
NativeF32 uses lower-expert-ID ties and ID-sorted normalization;
this is a local arithmetic policy, not official `torch.topk` tie parity.

Resolve the converted artifact and retained CPU reference by name through
`QWEN_MODEL_SEARCH_PATH`, then explicitly run only the short characterization:

```sh
uv run scripts/model-witnesses.py --kind characterization hf_conversion_seven_token
```

The model fixture is `DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf`; the
reference marker is `deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu.json` beside
its `run.json` and traces. The command reports complete finite comparisons, not
tight numerical acceptance. Original tight-bound negative controls remain unit
tests. The long teacher-forced attention comparison is also explicit
characterization, not an automatic 10K run or a full-model continuation oracle.

The converted 272E now executes through normal `run` and `serve`. The seven-token
HF comparison covers 40 layers and 567 tensors: 7/7 top-1 agreement, worst logit
error 0.178663 and worst KL 5.3512e-5. Against independently decoded GGUF weights,
the same GPU trace has worst logit error 1.2398e-5 and KL 1.7810e-12, again with
7/7 top-1 agreement. Worst hidden normalized error is 2.411e-6; hidden elementwise
differences remain characterization, not a new acceptance claim.

The tight execution witness retains the original logit bound 1e-4 and KL bound
1e-10. Its distinct CPU marker is
`deepseek-v4.1-reap-272e-gguf-native-f32-7t-40l-cpu.json`, beside `run.json`,
`tokens.json` and the independent GGUF reference traces:

```sh
uv run scripts/model-witnesses.py independent_gguf_seven_token_execution_acceptance
```

## Speed Measurements

On the M5 Ultra, the initial shared-harness baseline measures pp128 at 22.823
tokens/s and tg32 at 21.635 tokens/s: depth zero, seed one, three warmed
repetitions, API validation off. That baseline uses single-token prefill with
per-token logits copies, not batched prefill. JSON, SSE and two-turn serving
checks pass. Three warmed short HTTPS SSE
requests have first-content-delta latency 856-876 ms (median 860 ms), with the
client on the serving host and API validation on. Header receipt is not counted
as TTFT. These short probes establish neither long-context speed nor quality.

The subsequent grouped-MXFP4 implementation measures pp1024 at 40.475 tokens/s
against 22.653 for the preceding 32-row dense-only batching path; tg32 is 24.363
versus 24.356. These are single warmed shared-harness repetitions, depth zero,
seed one, with API and shader validation off. Batching output-A attention
projections then measures pp1024 at 43.616 tokens/s (tg32 21.468 in that run).
Selection/projection cleanup reaches 48.686, and SIMD score reduction measures
50.116 +/- 0.224 tokens/s over three repetitions. Parallel softmax at `a707908b`
then measures **76.550 +/- 0.048 tokens/s pp1024** and **26.861 +/- 0.011 tg32**,
also over three repetitions. These cold-KV prefill rates use resident, warmed
weights. At that checkpoint, a 1,024-token prompt takes 13.377 seconds.

The shared F32 attention-value matrix path at `94907364` measures pp1024 at
111.810 +/- 0.043 tokens/s over three repetitions (9.158 seconds), and tg32 at
26.730 +/- 0.011. A separate single pp2048 repetition measures 101.836 tokens/s
(20.111 seconds). These are improvements, not the practical-latency target.

Parallel index heads at `b29d6991` measure **118.729 +/- 0.010 tokens/s pp1024**
(8.625 seconds) and **26.746 +/- 0.021 tg32**, with three repetitions. A single
pp2048 repetition measures **114.636 tokens/s** (17.865 seconds). No validation
layers or diagnostic ablations are enabled. The owner's roughly 500 tokens/s
figure is a headroom heuristic, not an acceptance target; current device-local
attribution should decide the next optimizations. Evidence locations and the reuse
review are recorded in [the review packet](bench/2026-10-19-v41-review.md).
Three-repetition checks at `733d629f` measure 118.664, 114.549 and 108.079
tokens/s at pp1024, pp2048 and pp10000 respectively, with resident weights and
fresh KV. The 10K prompt takes 92.525 seconds on average.

Exact prefix reuse is separately useful: a private HTTPS smoke request reuses
492 of 494 prompt tokens in 0.21 seconds, and a follow-up reuses 495 tokens in
0.73 seconds. Those are whole short-request timings, not cold-prefill throughput.
The realistic agent screen exposed a missing system/tool-prefix checkpoint.
`fbd5770c` fixes it through the official renderer and shared SnapshotCache:
changed-task requests reuse 11,453 tokens and evaluate only 27 new tokens,
reaching first text in 0.588 seconds over three screens, versus a 109-second
baseline miss. Cold first text for 11,479 tokens still takes 113.6-124.0 seconds.
The packet retains the baseline tool-call error, bounded tool-step limits and
a 20-second prefill outlier; this is not a blanket agent-latency success claim.

The grouped path passes the independent seven-token GGUF CPU comparison and the
65-token serial/batched, capacity-growth and snapshot-continuation comparisons.
The matrix path also passes continuation through position 257 with the same
distribution limits (maximum logit error 2.360e-5, KL 5.568e-14).
The index-head path extends that internal serial/batched comparison through
position 1,025, with maximum logit error 1.955e-5, KL 2.169e-12 and matching top-1.
Index scores remain within the existing F32 tolerances, with exact selected-ID
comparisons across tested head geometries; they are not claimed bitwise equal to
scalar scores.
This is not independent full-model correctness through 1,025 tokens. Independent
GGUF full-model comparison still covers seven tokens. The registered 2K/10K
replay is scalar, teacher-forced HF conversion characterization; its completion
does not establish batched GGUF correctness or candidate-pruning coverage.
The separate independent GGUF attention replay at `733d629f` compares 32 scalar
and 56 batched checkpoints at 2K/10K and the real pruning boundary. All selected
IDs and candidate sets match; worst batched attention-output error is 2.933e-5
against the unchanged 1e-4 bound. This is conditional on captured normalized
inputs, not independent long full-model logits; see the review packet for scope.
The latest 65-token comparison has maximum logit error 2.737e-4 and KL 6.855e-10;
same-arithmetic snapshot/growth continuation remains bitwise equal. SIMD and
grouped reductions are F32, not bitwise scalar reduction order. Existing
independent-CPU and grouped-distribution bounds are unchanged.
The repeated seven-token oracle prompt is a correctness input, not a throughput
proxy for the random-token shared benchmark or real agent prompts.
A separate 1K stage sample attributes 9.84 of 13.13 GPU seconds to attention
and 3.29 seconds to FFN, with 0.15 seconds of CPU encoding and no measured disk
reads. That sample predates the final value/index optimizations and is not a
current-tree stage attribution.

## Conversion Status

The [output-only converter patch](../scripts/convert_v41_gguf.md) reuses PR 28696's
existing lazy expert and chunked tensor support. It removes both the global
writer spool and intermediate Engram tables without changing output bytes.
Eight tiny-fixture tests pass against the unmodified pinned converter. The old
272E metadata-only preflight predicts 423,681,448,000 bytes of final output and
zero temporary payload staging. The single conversion completes in about
43 minutes; normal serving and both short numerical comparisons complete.
The obsolete runtime cache has been removed; all 48 source shards are retained.

Further work targets cold prefill through the shared benchmark harness.
Historical safetensors timings and pilot results remain
[archived](archive/DEEPSEEK-V41-BRINGUP-2026-10-18.md), not evidence
for the repaired GGUF path.

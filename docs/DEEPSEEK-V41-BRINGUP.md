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
request starts fresh. Prefill is still token-at-a-time; prefix reuse, snapshots,
CED, speculation and Lens are not implemented for V4.1.

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

## Speed Baseline

On the M5 Ultra, the shared harness measures pp128 at 22.823 tokens/s and tg32
at 21.635 tokens/s: depth zero, seed one, three warmed repetitions, API validation
off. This is still single-token prefill with per-token logits copies, not batched
prefill. JSON, SSE and two-turn serving checks pass. Three warmed short HTTPS SSE
requests have first-content-delta latency 856-876 ms (median 860 ms), with the
client on the serving host and API validation on. Header receipt is not counted
as TTFT. These short probes establish neither long-context speed nor quality.

## Conversion Status

The [output-only converter patch](../scripts/convert_v41_gguf.md) reuses PR 28696's
existing lazy expert and chunked tensor support. It removes both the global
writer spool and intermediate Engram tables without changing output bytes.
Eight tiny-fixture tests pass against the unmodified pinned converter. The old
272E metadata-only preflight predicts 423,681,448,000 bytes of final output and
zero temporary payload staging. The single conversion completes in about
43 minutes; normal serving and both short numerical comparisons complete.
The obsolete runtime cache has been removed; all 48 source shards are retained.

After first-light correctness, improve exact batched prefill and prefix reuse
through the shared benchmark harness. Historical safetensors timings and pilot
results remain [archived](archive/DEEPSEEK-V41-BRINGUP-2026-10-18.md), not evidence
for the repaired GGUF path.

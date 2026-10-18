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
and Q8_0 Engram tables `[256, rows]`. MXFP4 experts stay packed. Engram reads
only requested rows from disk and is excluded from resident Metal weight banks.
Known metadata spellings are aliases. Unsupported consumed encodings are named
in errors; unused vision/MTP tensors and vision router biases are not bound.
The converter revision is provenance, never an accepted-file restriction.

Serving uses the official recipe renderer/parser, the common startup/request
spine and typed HTTP 503 `memory_admission_denied` responses. CLI statistics
use the existing v2 schema; benchmarks use the existing family harness. Each
request starts fresh. Prefill is still token-at-a-time; prefix reuse, snapshots,
CED, speculation and Lens are not implemented for V4.1.

## Verification

The repair passes 90 V4.1 library tests with required Metal and API validation,
including nonzero GGUF load-to-forward tests at 272/320/384 experts, quantized
projection tests, ownership transitions and bounded Engram reads. Shared
CLI/server/benchmark fixtures cover dispatch, request preparation, cancellation,
typed refusals and statistics. These are not a production-model execution result.

Actual Rust `gguf::plan()` succeeds against all ten mrfakename 1017 Q8_0 shard
headers: 1,046 tensors, 384 experts, 120 MXFP4 banks. The resident payload is
278.508 GiB and disk Engram is 194.555 GiB. The architecture is supported even
though that resident artifact exceeds a 256 GiB host. Header inspection reads
no model payload and does not initialize Metal. The repository revision used
for this check is `1df54e7d1b1e442051ad393ce871870a929f243e`.

The same all-shard check of Q2_K names its actual incompatibility: its Engram
tables are Q2_K, not Q8_0. That artifact is outside the current Engram scope.
Header checks used about 5.8 MB per artifact, not full downloads.

## First End-to-End Comparison

The existing offline HF CPU oracle remains the first conversion-plus-execution
signal. The chosen converter requantizes dense FP8 and Engram to Q8_0, so its
errors must be characterized, not tested against the old lossless-path bound.
An independent GGUF-reading CPU oracle will subsequently measure execution
error alone. NativeF32 uses lower-expert-ID ties and ID-sorted normalization;
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

## Conversion Status

No new conversion or production-model run has been performed for this repair.
Source inspection pins PR 28696 at
`2076412598e99127ea2554467b2b7530856895d3`. It supports the old 272E configuration
and losslessly repacks MXFP4 experts, but its safe-RAM temporary-file mode stages
complete Q8 Engram files plus a writer spool. Predicted final payload is
423,675,613,728 bytes; peak additional disk occupancy is about 983.71 GiB,
including temporary staging. This is not a direct-to-final-file conversion.
Resolve that write-cost decision before running it. The converter's `--dry-run`
still prepares tensors and is not a safe metadata-only preflight.

After first-light correctness, improve exact batched prefill and prefix reuse
through the shared benchmark harness. Historical safetensors timings and pilot
results remain [archived](archive/DEEPSEEK-V41-BRINGUP-2026-10-18.md), not evidence
for the repaired GGUF path.

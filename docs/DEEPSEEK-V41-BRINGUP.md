# V4.1 GGUF Operation

The [requirement](DEEPSEEK-V41-STRATEGY.md) is ordinary GGUF family support,
independent of whether a particular artifact fits the local machine.
Measurements live in the [performance log](PERF-LOG.md), with scope and evidence
in the [bring-up review](bench/2026-10-19-v41-review.md) and
[follow-up measurements](bench/2026-10-19-v41-leverage.md).

## Runtime

`general.architecture = "deepseek41"` selects shared `run`, `serve`, `info` and
benchmark paths. Geometry, KV ownership schedules and Engram hashes come from
metadata; base and pruned expert counts are not an allowlist. No adjacent HF
configuration, safetensors loader, converted-weight cache or conversion command
exists in the engine.

The convention is llama.cpp PR 28696: standard GGML expert/dense types and Engram
tables `[256, rows]` decoded through the shared GGML codec. MXFP4 experts stay
packed. Engram reads only requested rows from disk, outside resident Metal weight
banks; its dtype is a codec capability, not a quantization-quality policy.
Known metadata spellings are aliases; unsupported consumed encodings produce
named errors. Unused vision/MTP tensors and vision router biases are not bound.
Converter revision is provenance, not an accepted-file restriction.

Serving uses the official recipe renderer/parser, common startup/request spine,
typed HTTP 503 `memory_admission_denied` responses and automatic longest exact
causal-prefix reuse from live state or RAM snapshots. CLI statistics use the v2
schema and benchmarks use the shared family harness. Cold prefill uses 256-row
layer-major chunks without unused prompt logits. Dense/shared projections and
supported routed experts use shared NativeF32 operators. At 256 or more attended
rows, the 64-head attention value GEMM uses the shared scalar-F32 implementation,
not MMA. Cache updates remain causal.
IQ2_XXS routed experts lack grouped F32 gate/up and down support and fall back
per token; dense/shared work remains batched. This is not measured attribution
of the artifact's speed gap. Vision execution, CED, speculation and Lens are
not implemented.

The V4.1 family profile keeps `idle_residency_eligible = false`: shared GPU
idle-keepalive integration and lifecycle/admission qualification are not
established, not an architectural inability or a measured performance verdict.
The backend's existing `idle()` hook sweeps the snapshot cache; it does not
keep GPU weights wired through the shared residency mechanism.

## Readiness And Evidence Limits

- REAP-272 GGUF executes through normal `run` and `serve`, including JSON, SSE
  and two-turn checks. The 1017 384-expert IQ2_XXS/Q8 artifact has a native
  load/forward smoke and measured throughput, not independent numerical or
  task-quality qualification.
- Independent full-model GGUF acceptance covers seven tokens, 40 layers and
  567 tensors: 7/7 top-1, worst logit error 1.2398e-5 and KL 1.7810e-12, within
  unchanged bounds 1e-4 and 1e-10. Worst hidden normalized error is 2.411e-6;
  hidden elementwise differences remain characterization.
- The separate HF conversion comparison has 7/7 top-1, worst logit error
  0.178663 and KL 5.3512e-5. Dense FP8 and Engram requantization to Q8_0 makes
  this characterization, not lossless-path acceptance. NativeF32 uses
  lower-expert-ID ties and ID-sorted normalization, not official `torch.topk`
  tie parity.
- At `733d629f`, independent GGUF attention replay covers 32 scalar and 56
  batched checkpoints at 2K/10K and pruning counts 16,384-16,386. Selected IDs
  and candidate sets match; worst batched output error is 2.933e-5 against 1e-4.
  This is conditional on captured normalized inputs, not long full-model logits.
  Internal serial/batched parity through position 1,025 is a different witness.
- Short prefix smoke (492/494 reused) and real agent shared-prefix reuse
  (11,453 reused, about 0.59 s to first text) are distinct from cold prefill.
  The retained 20-second suffix outlier remains unreproduced and unexplained.
- Fresh GGUF quality screening has not run. The retired safetensors pilot is
  not a current REAP GGUF score.

The existing `scripts/model-witnesses.py` registry separates
`hf_conversion_seven_token` characterization from
`independent_gguf_seven_token_execution_acceptance`. Fixtures resolve by name
through `QWEN_MODEL_SEARCH_PATH`; the HF and GGUF CPU markers are respectively
`deepseek-v4.1-reap-272e-native-f32-7t-40l-cpu.json` and
`deepseek-v4.1-reap-272e-gguf-native-f32-7t-40l-cpu.json`. Long scalar HF replay
does not establish batched correctness or pruning coverage.

## Artifact Compatibility And Conversion

Rust `gguf::plan()` at `1df54e7d` accepts all ten mrfakename 1017 Q8_0 shard
headers: 1,046 tensors, 384 experts, 120 MXFP4 banks; 278.508 GiB resident and
194.555 GiB disk Engram. That resident payload exceeds a 256 GiB host. The
all-shard Q2_K check also passes: 186.260 GiB resident, 60.083 GiB disk Engram,
384 experts. These are planning checks, not quality or measured local fit;
each reads about 5.8 MB of headers, no model payload, without initializing Metal.
Fixtures also cover nonzero load-to-forward at 272/320/384 experts, quantized
projections, ownership transitions, bounded Engram reads and shared contracts.

The [output-only converter patch](../scripts/convert_v41_gguf.md) reuses the
pinned converter's lazy experts/chunked tensors, removing writer spool and
intermediate Engram tables without changing output bytes. Eight tiny-fixture
tests pass. The old 272E preflight predicts 423,681,448,000 output bytes and zero
temporary payload staging; conversion completes in about 43 minutes. The
obsolete runtime cache is removed and all 48 source shards are retained.
[Archived safetensors results](archive/DEEPSEEK-V41-BRINGUP-2026-10-18.md) are not
evidence for the repaired GGUF path.

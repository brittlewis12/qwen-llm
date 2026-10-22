# GLM-5.3-Flash NextN Reference Probe

Bench-only tooling, not native NextN integration. Run only after the native
two-row cost gate and normal memory/lease admission. There is no sampler,
target generation, native executor, memory-policy override, or artifact-name/hash
qualification gate. Raw and post-norm runs are independent fresh processes.

## Extract On CPU

`extract.py` copies exactly the 29 block-45 tensors plus `token_embd.weight`,
`output_norm.weight`, and `output.weight`. No trunk tensor is copied or synthesized.
It retains metadata/tokenizer types, removes every `split.*` key, and canonicalizes
the legacy `glm5next` namespace to `glm5-next`. Block numbering remains 45, with
46 stored blocks and one NextN layer. Missing tensors, unexpected block-45
tensors (including separate embedding/head overrides), conflicting metadata,
missing siblings, and existing output files/symlinks are refused.

The copy preserves storage bytes, quantization types, and dimensions. Every
selected tensor is SHA-256 checked after writing. A private staging file is
published without replacement only after verification and a source-stamp recheck.
The result is mode 0600; source stamps, header hashes and tensor hashes live in
`qwen.nextn_probe.provenance`, also printed as JSON. Preserve the supplied first
shard's symlink filename: sibling discovery deliberately does not resolve it.

The IQ3 artifact's known inventory implies about 3.58 GiB including the two Q6_K
fallback matrices (not merely its 2.61 GiB NextN block). Actual
`payload_bytes` and inventory, not these estimates or filenames, are authoritative.

From the qwen-llm root, using a llama.cpp checkout with gguf-py dependencies:

```sh
P=scripts/reference/glm53/nextn_probe
L=/path/to/llama.cpp
PYTHONPATH="$L/gguf-py" uv run --offline --no-project \
  --python "$L/.venv/bin/python" python -B "$P/extract.py" \
  --source "$SOURCE_FIRST_SHARD" --output "$NEW_PRIVATE_HEAD_GGUF"
```

The output's parent must exist; output itself must not. No full-trunk payload is
read for copying. `GGUFReader` maps source shards read-only and parses their
headers; only the selected tensor payloads are hashed/copied.

The PEP 723 dependency also supports `uv run --script "$P/extract.py"`.

## Build Only This Harness

```sh
cmake -S "$P" -B "$P/build" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_DIR="$L" -DLLAMA_BUILD_DIR="$L/build"
cmake --build "$P/build" -j 2
```

This links existing `libllama`, `libggml`, and `libggml-base`; it never configures
or builds llama.cpp. It uses that checkout's nlohmann header. The report records
checkout revision/dirty state, build-info commit, configure-time library/header
hashes, runtime library location/version, and probe source hashes. These are
provenance, not identity allowlists. Reconfigure if the library changes; the
Freeze the binary and dependencies for paired measurements.

Inspected checkout: `d3f2715f6a31f2d212599c532d36b31641f21275`. Existing build-info:
`900645073`. No diff between those revisions in `include/llama.h`,
`src/llama-ext.h`, `src/llama-batch.cpp`, `src/llama-model.cpp`, or the GLM graph.
The extended batch/state APIs resolve in the existing library.

## Capture And Run

The native exporter renders the supplied messages with native GLM defaults,
prefills in 512-row chunks, and samples a fresh seed-42 continuation. It checks
every exported final normalized row against the actual target head input.

```sh
export GLM53_GGUF="$SOURCE_FIRST_SHARD"
export QWEN_NEXTN_MESSAGES="$PRIVATE_MESSAGES_JSON"
export QWEN_NEXTN_EXPORT_DIR="$NEW_CAPTURE_DIRECTORY"
export QWEN_NEXTN_SCORE_TOKENS=128
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 cargo test -p qwen-llm --lib \
  glm5_next_metal::tests::nextn_capture::export_native_nextn_capture \
  -- --exact --ignored --nocapture --test-threads=1
```

`manifest.json` must contain:

```json
{"schema":"qwen.glm-nextn-capture.v1","status":"complete","hidden_size":4096,"vocab_size":154880,
 "token_count":11169,"score_start":11105,"score_count":64}
```

Additional provenance fields are preserved in the report. `N = S + T`, `T > 0`.
An immediate target EOS produces `no_evaluable_rows`, not an acceptance result.
All binary files are little-endian, with no headers:

| File | Elements | Meaning |
| --- | ---: | --- |
| `tokens.u32` | N | Consumed target token IDs |
| `hidden.f32` | N * 4096 | Post-output-norm target hidden rows |
| `hidden-raw.f32` | N * 4096 | Pre-output-norm completed stream means |
| `target-logits.f32` | T * 154880 | Target logits after consuming positions S..N-1 |

Head position 0 receives zero hidden and token 0. Position i > 0 receives
hidden[i-1] and token[i]. Its logits predict token i+1 and align with target row
i-S. The last captured hidden row is therefore not consumed by this probe.
All prefix positions are processed, never reset between scored rows. Inputs are
length-checked before loading weights; selected hidden rows and returned logits
are finite-checked. Target logits are size-checked, not evaluated or sampled.

```sh
scripts/bench/with-metal-lease "$P/build/glm53_nextn_probe" \
  --model "$NEW_PRIVATE_HEAD_GGUF" --input-dir "$CAPTURE" \
  --output-dir "$NEW_POST_RESULT" --hidden-kind post-norm --history-batch 512
scripts/bench/with-metal-lease "$P/build/glm53_nextn_probe" \
  --model "$NEW_PRIVATE_HEAD_GGUF" --input-dir "$CAPTURE" \
  --output-dir "$NEW_RAW_RESULT" --hidden-kind raw --history-batch 512
```

Use `MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1` for measured reference runs too.
The lease provides exclusion, not memory admission: verify head payload plus
context, graph scratch, and CPU storage fit before launching. Do not bypass
engine memory limits.

Output directories must not exist; they are created mode 0700. Each contains
`draft-logits.f32` (T * vocab) and `report.json`. Only `status: complete` is usable;
exceptions leave failure evidence, and abrupt termination can leave `running`
plus partial logits. No existing output is overwritten.

The harness requests full GPU layer offload and otherwise keeps upstream default
loading/memory policy. It creates one MTP context, not a target context. The
header is inspected before weight loading to reject trunk tensors. The caller owns
live memory admission, environment isolation and the production Metal lease.

History uses up to 512 rows per batch with every output-logit flag false and
NextN extraction disabled. The inspected graph crops to zero output rows before
FFN/head work; it still performs attention/cache work. Scoring is scalar, with
one full-vocabulary head row per step. `--history-batch 1` is the chunk witness:
compare its scored logits with 512 on the same capture before relying on batching.

Timing reports startup/context separately, complete history-init wall, and each
scored process/synchronize/logit-readback wall. Input preparation and output-file
IO are excluded from score-step timing. No body/head profiler is added:
`body_only_ms` is null. This measures the reference engine, not native draft cost.

## CPU Checks And Limits

```sh
PYTHONPATH="$L/gguf-py" uv run --offline --no-project \
  --python "$L/.venv/bin/python" python -B "$P/test_extract.py"
ctest --test-dir "$P/build" --output-on-failure
```

Python tests use tiny generated GGUFs, including Q2_K/Q3_K/Q8_0/BF16 storage,
split metadata, rewritten-shard symlinks, collisions and missing weights. The
C++ test does not link llama/ggml: it checks manifest extents, zero initialization,
previous-row alignment at chunk and scoring boundaries, and malformed inputs.

Real head-only extraction/loading and full-history initialization are measured
in [the short-packet packet](../../../../docs/bench/2026-10-22-native-short-packets/README.md).
Scalar versus 512-row history agrees on all 128 scored top tokens, with worst
full-softmax KL 2.36e-6; it is not bit-identical. Eight extraction tests and the
standalone C++ alignment test pass. Missing upstream tensors must remain an
explicit error, never zero placeholders.

## Native Distribution Analysis

```sh
export QWEN_NEXTN_CAPTURE_DIR="$NEW_CAPTURE_DIRECTORY"
export QWEN_NEXTN_PROBE_DIR="$NEW_POST_RESULT"
export QWEN_NEXTN_ANALYSIS_OUT="$NEW_PRIVATE_ANALYSIS_JSON"
cargo test -p qwen-llm --lib \
  glm5_next_metal::tests::nextn_analysis::analyze_nextn_capture \
  -- --exact --ignored --nocapture --test-threads=1
```

The CPU analyzer uses `Sampler::diagnose`, including the capture's temperature,
top-k, top-p, min-p and tie rules. It reports greedy agreement, full-softmax KL,
and filtered `sum(min(P,Q))`, not agreement with a held transcript token.
Optional `QWEN_NEXTN_SERIAL_MS` and `QWEN_NEXTN_VERIFY2_MS` provide a hypothetical
one-draft budget; both must be supplied together. Reference head timings are not
native draft timings. One-step overlap on target-on-policy states does not
establish recursive acceptance, gameplay quality, or end-to-end speedup.

The source-backed default is post-norm. Raw is a discrepancy control, not an
equivalent conditioning contract. All artifacts and analysis outputs are
private and create-new; an immediate target EOS has no evaluable rows.

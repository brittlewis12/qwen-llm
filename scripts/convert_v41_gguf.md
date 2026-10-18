# Output-only V4.1 Conversion

This is a small patch to llama.cpp PR 28696 at
`2076412598e99127ea2554467b2b7530856895d3`, not another model format or an engine
loader. It reuses the converter's lazy MXFP4 helper and `LazyChunkedTensor` to
write Engram rows directly into the final GGUF. Quantization, metadata and the
GGUF writer are unchanged. No global spool or Engram staging files are created.

Remove the local patch when the selected upstream converter includes equivalent
bounded expert/Engram streaming without payload staging. Verify that with source
review and the tiny-fixture tests below; no extra full-model conversion is needed
just to retire the patch.

## Prepare And Run

Use an owned checkout of the pinned converter and a Python environment with its
dependencies. The runner never downloads weights, installs dependencies or
modifies the checkout. Existing CPU-oracle environments can be reused.

```sh
SCRIPTS=/path/to/qwen-llm/scripts
SOURCE=/path/to/owned/llama.cpp
MODEL=/path/to/DeepSeek-V4.1-Flash-REAP-272E
OUTPUT=/path/to/DeepSeek-V4.1-Flash-REAP-272E-MXFP4-Q8Engram.gguf
PYTHON=/path/to/environment/bin/python

uv run --offline --no-project --python "$PYTHON" python "$SCRIPTS/convert_v41_gguf.py" \
  --llama-cpp "$SOURCE" --check-source
git -C "$SOURCE" apply --check "$SCRIPTS/convert_v41_gguf.patch"
git -C "$SOURCE" apply "$SCRIPTS/convert_v41_gguf.patch"
uv run --offline --no-project --python "$PYTHON" python "$SCRIPTS/convert_v41_gguf.py" \
  --llama-cpp "$SOURCE" --check-source
uv run --offline --no-project --python "$PYTHON" python "$SCRIPTS/convert_v41_gguf.py" \
  --llama-cpp "$SOURCE" --model "$MODEL" --outfile "$OUTPUT" --dry-run
```

The source checks should report `canonical`, then `patched`; do not apply twice.
They audit converter provenance only, never restrict the engine's accepted files.
The output parent must exist and be outside the source model directory.

Replace `--dry-run` with `--convert` to perform the one-time write. The default
is a true metadata-only preflight: payload mappings and output creation are
blocked while shapes, scales, tokenizer/hash metadata and output size are checked.
Do not substitute the stock converter's late `--dry-run`, which prepares payloads.

## Resources And Failures

The old 272E preflight predicts 423,681,448,000 bytes (394.58 GiB) of final output
and zero temporary payload staging. MXFP4 experts are repacked losslessly;
Engram and other FP8 matrices become Q8_0. This is not lossless FP8 conversion.
Source BF16 matrices and F32 vectors retain their canonical output types.

Experts materialize one projection at a time (about 1.59 GiB for 272E). Engram
defaults to 65,536-row chunks, with 64 MiB per F32 scratch array. Active allocation
is estimated at 8-16 GiB, not a measured RSS ceiling; the largest dense tensors,
allocator retention and file mappings also contribute. `--chunk-rows` can reduce
Engram scratch without changing output bytes.

One private `OUTPUT.incomplete/OUTPUT` inode receives the data. Completion flushes
and fsyncs it, checks its size, and publishes through a no-overwrite hard link;
no second payload copy is made. Failure preserves the partial file and retries
refuse it. There is no resume or automatic cleanup. Do not load incomplete files.
Source weights are never deleted by the runner.

## Tests

```sh
uv run --offline --no-project --python "$PYTHON" python "$SCRIPTS/test_convert_v41_gguf.py" \
  --canonical /path/to/unmodified/pinned/source --patched "$SOURCE" -v
```

Eight tests exercise tiny real safetensors, complete canonical/patched GGUF byte
equality, split scales, row tails, bounded chunk lifetimes, no staging artifacts,
metadata-only preparation, no overwrite and interrupted output. The old 272E
tokenizer/hash preflight also passes without payload reads. Neither result is a
production-model numerical comparison; run the existing HF characterization
after conversion, then the independent GGUF execution oracle when available.

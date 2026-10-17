# Old REAP-272E bounded CPU oracle

Run from the repository root; `uv` manages PyTorch and all Python dependencies.
No CUDA, Triton, TileLang, system installation, or checkpoint conversion is used.

```sh
uv run scripts/reference/deepseek_v41/cpu/oracle.py \
  --model-dir /path/to/old-reap \
  --token-ids 0,1234 --dump-dir /tmp/v41-native \
  --mode NativeF32 --projection-rows 128 --max-seq-len 32 --generate 2 --threads 8
```

Add `--engram-dir /path/to/original-base` when the REAP directory does not contain
Engram. That directory can contain just base shards 47/48 and its safetensors
index. Main-directory entries take precedence; the fallback directory can supply
only Engram embedding weights/scales, never missing REAP backbone weights.
Missing model tensors/scales fail;
Engram is never silently disabled. `tokenizer.json` must be in the main directory.
Use the original tokenizer, not a substitute: compressed vocabulary size is checked.
`old_config.json` is a provenance copy, not an automatic fallback for a missing config.

The CLI prints greedy generated token IDs and saves `tokens.json`. Every consumed
token produces `pos-NNNNN-layer-NN.pt` (post-block HC residual and next pre-mix) and
`pos-NNNNN-logits.pt` (full vocabulary). `run.json` records arithmetic mode, source
hashes, correction, and bounds. Choose a fresh dump directory to avoid stale files.

## Arithmetic

- `NativeF32`: decoded F32 weights, F32 activations/accumulation/output. No simulated
  activation or KV/index-cache quantization; this is the runtime baseline, not a
  claim of upstream BF16/FP8 equivalence.
- `ReferenceFp8`: E4M3 activation quantization, block 32, upward power-of-two UE8M0
  scales using the pinned FP32 reciprocal multiply and exponent/mantissa bit
  round-up (not `ceil(log2(x))`); F32 GEMM accumulation and BF16
  quantized-projection output. BF16 dense
  projections retain BF16 output; compressor pooling projections and logits retain
  F32. Index keys/queries use E2M1 block32/UE8M0 and compressed KV uses E2M1
  block16/E4M3. Sparse attention follows 64-entry online softmax, including BF16
  probability rounding and a denominator-only attention sink. HC follows the
  pinned Sinkhorn iteration order and epsilon placement.

These are explicit numerical modes, not selectable tolerance relaxations. CPU
BLAS reductions, transcendental functions and GPU tensor-core reductions need not
be bit-identical. No real GPU-reference comparison or tolerance certification has
been performed. Tests never substitute NativeF32 for ReferenceFp8.

## Source and correction

`old_model.py` is unmodified source downloaded from
`LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E` at
`f792019bcb369768220f01088f0f7af230786982`:
SHA256 `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
The runner checks that hash before adapting imports, lazy projections and the
grouped output projection. The owner fix is explicit in `source_module()`:
publish `shared_attn.index_k = self.k_cache` whenever `owns_k`, even on an
incomplete compression pair. Each model gets its own module globals and caches.

The old grouped output projection intentionally does NOT gain 1017's additional
activation quantization. Its reference weight decode does round to BF16, matching
the old converter. Existing pinned `fixtures/inference/engram.py` is reused
read-only with SHA256 checking. Its hash
`11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` also matches
the original base at `314113f8a7dadb388215396ba5fbee2c610a04d6` (verified by fetch).
Upstream source license is in `../fixtures/LICENSE`.

Local checkpoint contents are NOT cryptographically authenticated against HF
revisions. The recorded revisions are expected provenance, not verification.
Supply the pinned main checkpoint and original base Engram shards, not 1017 weights.

## Bounds and limitations

- Safetensors remain mmap-backed and packed. Projection decode is bounded to
  `projection_rows * input_width` F32 values plus temporary unpack/scale buffers;
  row count must be a multiple of 32. Selected experts alone execute. No dense
  full expert, vocabulary embedding, output head, or Engram table is allocated.
- Small non-projection parameters (HC matrices, routers, norms) are resident;
  this is hundreds of MiB at full shape, plus bounded caches and activations.
  Mapped file pages are managed by the OS: this is NOT a hard RSS/page-cache cap.
  All K columns of a row chunk are decoded; there is no K-axis tiling.
- Batch one, text-only, greedy IDs, no vision/MTP/speculative generation. The
  upstream text graph, routing, Engram, RoPE, compression and sharing are used.
  Prompt chunks are deliberately executed token-by-token. `Oracle.feed()` can
  continue a sequence across arbitrary caller chunks; no parallel chunk prefill.
- Sequence state is in-process only. No restart/resume serialization, batching,
  or concurrent construction/thread-safe execution is promised. Separate models
  may be interleaved without sharing attention state.
- Full real-checkpoint generation is pending weight availability. Disk traffic
  is substantial and CPU generation is expected to be slow. No remote/GPU job
  was launched. Synthetic generated IDs are not evidence of language quality.

## Verification and synthetic generation

```sh
uv run --with torch --with safetensors --with numpy --with sympy --with tokenizers \
  python -m unittest discover -s scripts/reference/deepseek_v41/cpu -p 'test_*.py' -v

# The checkpoint target must not already exist.
uv run --with torch --with safetensors --with numpy --with sympy --with tokenizers \
  python scripts/reference/deepseek_v41/cpu/test_cpu.py --make-checkpoint /tmp/v41-tiny
uv run scripts/reference/deepseek_v41/cpu/oracle.py \
  --model-dir /tmp/v41-tiny --token-ids 1,2,3 --dump-dir /tmp/v41-tiny-out \
  --mode ReferenceFp8 --projection-rows 32 --max-seq-len 16 --generate 3 --threads 1
```

Tests cover FP8 ties and zero scales, FP4 nibble order and ties, both GEMMs,
attention sink/all-invalid rows, Sinkhorn, row-only Engram decoding, and a full
five-layer synthetic packed checkpoint with two Engram layers, routed experts,
multiple KV owners/consumers, completed/incomplete pairs, ring wrap and generation.
Continuation and interleaved-instance logits match exactly across projection row
bounds 32 and 64 in both modes. No existing fixtures are modified.

Review regressions also check FP32 `nextafter` on both sides of FP8/FP4 scale
boundaries, missing/wrong scales, unbound backbone tensors, Engram-only fallback,
and NativeF32 dense outputs that would change under BF16 rounding. The separate
`test_review.py` runs `../operator_fixture.py`'s generator read-only (never its
writer) with corrected and original Indexer methods over all 40 layers and
positions 0..6. It checks owner identity on completed/incomplete pairs and proves
selection/output divergence, not merely a different cache pointer. The first
divergence is position 4/layer 8; maximum attention error is `0.510894492`.

## Real-weight readiness without payload downloads

```sh
uv run scripts/reference/deepseek_v41/cpu/readiness.py \
  --cache-dir /tmp/v41-cpu-readiness-f792019
# Repeat without network, using the same pinned metadata cache:
uv run scripts/reference/deepseek_v41/cpu/readiness.py \
  --cache-dir /tmp/v41-cpu-readiness-f792019 --offline
```

This fetches bounded JSON config/index/tokenizer documents and strict HTTP Range
safetensors headers. A server ignoring Range is rejected before reading its body.
Engram headers come from original-base shards 47/48. It constructs the entire
real text graph on `meta`, checks every required tensor's name, physical shape,
dtype, scale shape/dtype and payload-offset bounds, and uses the same
`validate_bindings()` as the real runner. Unknown backbone tensors fail; vision,
MTP, and the 40 VL routing biases are deliberately unused by text-only generation.
The exact runtime tokenizer adapter is also checked against the pinned Engram
compressed vocabulary and sample continuation hash ranges.

Verified: 40 layers, 272 experts per layer, dimension 5120; 66,498 bound tensors
in each mode, 44 required shards, 7,389,072 header bytes, zero tensor payload bytes
downloaded. Real tokenizer vocabulary 129,280 compresses to the required 99,092.
NativeF32 resident non-projection parameters total 382,454,720 bytes, excluding
caches, decoded scratch, Python metadata, and OS-managed mapped pages. Full-model
metadata checks and all 14 tests pass with uv-managed PyTorch 2.14.1 on CPU.

This is descriptor/tokenizer readiness, NOT a verification of downloaded tensor
payloads, real generation, or GPU-reference numerical parity. `readiness.json`
in the cache directory explicitly records those limits. No weight download is
started by the readiness command.

## Compare CPU and runtime traces

```sh
uv run scripts/reference/deepseek_v41/cpu/compare.py \
  --reference-dir /tmp/v41-native --runtime-dir /tmp/v41-runtime

uv run --with torch --with numpy python -m unittest discover \
  -s scripts/reference/deepseek_v41/cpu -p 'test_compare.py' -v
```

The command writes one JSON summary to stdout: first divergent position/layer and
tensor (`hidden`, `pre`, or `logits`), worst maximum absolute error, RMSE,
`normrelative = ||runtime-reference||_2 / ||reference||_2`, and each position's
logits KL(reference || runtime) and top-1 IDs/agreement. Statistics use F64;
logits KL uses temperature-one softmax and natural logs. Zero reference norm
produces zero if both tensors are zero, otherwise the JSON string `"inf"`.
Tensors are processed in position/layer order, hidden then next-pre-mix, followed
by logits; only one layer dictionary and the active tensors need be resident.

Without tolerance arguments, this is exact-difference characterization only.
Optional `--atol` and `--rtol` select the reporting threshold
`abs(runtime-reference) > atol + rtol*abs(reference)`; an omitted component is
zero. No tolerance is certified against real weights, and neither defaults nor
explicit thresholds constitute an agreement gate. Valid comparisons return 0
even when tensors differ. Invalid traces return 2 with a JSON error on stderr.

Reference inputs are existing `pos-NNNNN-layer-NN.pt` dictionaries with `hidden`
and `pre_mix`, plus `pos-NNNNN-logits.pt`. Singleton batch/token axes are removed.
Runtime inputs are `.hidden.f32`, `.pre.f32`, and `-logits.f32` files interpreted
as little-endian F32. Exact byte counts, consistent shapes, finite values, both
file inventories, contiguous positions/layers, and matching known `run.json`
modes are required. Runtime manifest shapes, byte counts, layer count, encoding,
completion status and per-position tokens are checked when present.

CPU `tokens.json` prompt/generated IDs and runtime `positions[].token_id` or
prompt/generated IDs are compared as consumed inputs. The last sampled token
has no forward trace and is excluded. Missing IDs are explicitly reported as
an unavailable/partial token check, not silently counted as agreement. Without
metadata declaring a full layer/position count, a tail missing from both traces
cannot be inferred from raw files alone. Raw files have no shape header: their
logical dimensions come from the reference, checked against runtime manifest
shapes when available. Same-size layout errors appear as numerical differences.

Fixtures cover matching traces, differing hidden/pre/logits, missing files,
truncation, shapes, nonfinite values, modes, token IDs/counts, explicit reporting
tolerances, zero reference norms, and the CLI's current success/incomplete
manifest schema. This comparison tool has not yet been run on real-weight traces.

# Data-only Linear Lens Contract

Implemented producer/consumer contract. A new fit is data, not a
new entry in the runtime's published-asset registry. Legacy pinned PyTorch
imports remain separate compatibility adapters; none of this format executes
pickle, Python hooks, model code, or manifest-supplied operations.

An artifact directory contains `lens.json` and `transport.f16le`. Schema:

```json
{
  "schema": "llm.lens.linear_transport",
  "schema_version": 1,
  "status": "complete",
  "transport": {
    "operator": "post_block_linear",
    "method": "arbitrary producer method label",
    "source_layers": [0, 2],
    "target_layer": 2,
    "orientation": "target_source",
    "bias": "none",
    "output": "deployed_native",
    "identity_layers": [2]
  },
  "model": {
    "architecture": "qwen35",
    "n_layers": 3,
    "hidden_size": 4,
    "vocab_size": 32,
    "source_checkpoint": {"id": "producer/model", "revision": "revision"},
    "source_tokenizer": {"id": "producer/model", "revision": "revision"}
  },
  "payload": {
    "path": "transport.f16le",
    "dtype": "f16_le",
    "shape": [2, 4, 4],
    "byte_length": 64,
    "sha256": "64 lowercase hex characters",
    "matrix_sha256": ["64 lowercase hex characters", "64 lowercase hex characters"]
  },
  "provenance": {},
  "qualification": {}
}
```

The example hashes are placeholders, not a valid artifact. Structural objects
reject unknown fields and duplicate JSON keys. `provenance` and `qualification`
are bounded arbitrary JSON objects, retained as producer claims only. Method
labels are bounded nonempty strings and never dispatch arithmetic or grant
scientific validation. Optional checkpoint/tokenizer descriptors are provenance,
not a claim that an arbitrary GGUF is their equivalent.

Optional `model.exact_binding` contains `gguf_content_blake3` and
`tokenizer_metadata_id`: the native ordered-GGUF content identity and 16-character
hex tokenizer metadata identity, not a raw-file SHA renamed as BLAKE3. If present,
both must match the opened deployment, even with a transfer override. Exact
binding hashes the retained GGUF bytes; downloader declarations and cached roots
cannot authorize it. This binds deployment bytes, not scientific fit quality or
HF-to-GGUF equivalence. Otherwise
consumers require explicit unvalidated-transfer acknowledgement and report
source/deployment equivalence as unverified. No separate certificate authority
or new per-fit permissions registry is needed.

The operator is `z = T_layer h_post_block`, followed by the actual deployment's
native output normalization/projection/scaling/softcap. It does not execute the
remaining blocks. `target_layer` names the fitted coordinate reference. Source
layers are unique, in-range, and define payload order; consumers may request
them in another order. `identity_layers` is an optional empty-by-default subset
whose exact matrices are checked, not an assumption inferred from target equality.
The payload is contiguous little-endian F16, with shape `[sources, H, H]`.

Validate checked dimensions and resource limits, exact length, every finite
value, whole-payload SHA256, each matrix SHA256, and any declared identities.
Retain the opened payload handle; reverify each matrix before passing its bytes
to native execution so an in-place change after initial verification cannot
silently change consumed coefficients. Do not allow alternate payload paths,
symlinks, executable deserialization, or a claimed qualification to bypass checks.
Producer publication must fail without overwriting existing artifacts and expose
a complete manifest only after the payload is durable.

Runtime support follows existing native capture/head capabilities, not checkpoint
names or fit recipes. Same-size matrices alone do not establish model or tokenizer
equivalence. Actual unsupported geometry, output operations, or execution modes
need explicit capability errors, not another artifact whitelist. Existing v1
published-format outputs and pinned import validation retain their contracts.

Producer export streams matrices or bounded row chunks directly from live
JacobianLens tensors, checking for overflow after F16 conversion. It uses
standard-library SHA256 and does not add a fitting dependency. A separate legacy
`jlens.transport` decoder may interpret a strictly bounded ZIP/pickle grammar
as data descriptors without resolving globals, calling reducers or unpickling;
unsupported encodings fail closed. Future exports need no legacy conversion.

## Usage And Capability

The producer's `JacobianLens.export_data(directory, model=...)` writes this
directory directly from live tensors. The standalone
`jlens/data_artifact.py convert-pt` handles supported legacy archives without
importing Torch or executing pickle. See the producer README for its bounded
grammar, model descriptor, and exclusive-publication contract.

```sh
qwen-lens verify-full --full-lens /path/to/lens-data
qwen-lens read-full --model /path/to/model.gguf \
  --full-lens /path/to/lens-data --prompt 'The capital of France is' \
  --layers 7,46,62 --top-k 8 --allow-unvalidated-transfer
```

Choose layers actually present in the artifact. `verify-full` is CPU-only and
verifies the entire payload, including unselected layers; it does not bind a
deployment or validate producer qualification. `read-full`, `trace-full`, and
plan-selected token banks accept these directories without an import step or a
new published-profile entry. Plans may use `kind: "linear_transport"`, with
the existing `path`, `token_ids`, and `allow_unvalidated_transfer` fields.

Ordinary Qwen supports scalar dense/MoE readouts, dense packed traces/cohorts,
and the existing projected-bank run/cohort/sweep paths. Muse supports scalar
readout, single-input packed traces and projected-bank runs through its existing
native runtime; this change does not broaden the runtime's model-geometry support.
Packed ordinary-MoE capture and Muse cohort tracing remain unavailable. Flash-Next
does not gain a full-transport runtime from this format change.

Geometry, binding and bank selections are checked before Metal/model loading.
Muse tokenizer loading authenticates and pins the same retained GGUF shards,
rather than trusting another open of the original pathname. Trace/run outputs
and inspection/comparison retain producer contracts separately from runtime
binding evidence. Unbound deployment locators are nonempty file-stamp identities,
explicitly not authenticated weight-content roots.

## Verification Record (2026-09-08)

- Both new Qwen3.8 n25/T128/skip4/target62 J/R archives converted using only the
  restricted decoder, in 77.89s and 84.83s respectively. Output directories are
  `j/data-v1` and `r/data-v1` alongside the original files under
  `/Users/tito/models/qwen38-27b-lenses/qwen3.8-27b/pile10k25-t128-skip4-target62-v1/`.
- Each payload is 3,303,014,400 bytes, shape `[63,5120,5120]`, with exact declared
  identity at layer62. Both pass the native CPU verifier. Neither has an exact
  deployment binding; the original BF16-to-GGUF transfer caveat remains.
- Payload SHA256 J: `a7ec94fb82c71e8d17507d3c637bfc4245152fdbf5fefa89f08ddbc46eaed886`.
- Payload SHA256 R: `a4c408dae91a3e82ea6b08d3bfab488e5e93e2ff7115b3de49b005943e9bdbaa`.
- Producer: 42 CPU tests pass with executable deserializers guarded against use.
  Consumer: 326 CLI tests plus 19 targeted runtime tests pass with
  `GGML_METAL_DEVICES=0`; nine opt-in CLI tests are ignored and the explicitly
  Metal-initializing operation-order test is skipped. All CLI binaries check and
  the release `qwen-lens` builds. Independent producer fixtures cover arbitrary
  method labels and different geometry, without adding profile entries.
- Adversarial re-review closes metadata alias amplification, quadratic pickle
  MARK scans, declaration-only exact binding, tokenizer pathname races, late
  selection validation, empty deployment locators and dropped comparison evidence.
- Live numerical smoke remains pending: the bounded Q8 `read-full` attempt could
  not acquire another session's Metal lease. No owner was interrupted. This is
  not evidence of live numerical equivalence, new-pair steering efficacy, or
  general cross-model qualification.

Private verification logs and frozen executable live under
`target/lens-data-contract/verification/` in the implementation worktree. Binary
SHA256: `12e69ef2c6e6856da2399fc7d47663578f4f18cb46531b6a46549ccbf15df9d1`.
Follow-up: [comparative archive exploration](LENS-COMPARATIVE-PHYSIOLOGY.md)
now exercises the new 3.8 pair through live scalar/packed readout, cohorts,
inspection and passive projected-bank execution. All twelve fitted/plain
layer62 full-vocabulary comparisons in that panel are byte-identical. The earlier
lease-blocked attempt remains historical evidence, not the current capability
limit. This does not qualify generic Muse execution, interventions, or scientific
transfer, and scalar/packed rounding differences remain distinct from fit quality.

## Opt-in full-vocabulary trace summaries

Native Qwen `trace-full --distribution-summaries` (including `--requests-jsonl`)
adds `distribution_summary` to every returned cell. Without the flag, this field
is absent and existing output semantics are unchanged. Muse rejects the flag
before Metal initialization. This does not add live generation capture: packed
token-ID replay remains a reconstruction, not bit-exact serial decode evidence.

The summary contains `vocab_size`, `entropy_nats`, `logsumexp`, `top_k_mass`,
`score_max`, `score_mean`, and `score_variance_population`. These are F64 CPU
reductions of the full deployed F32 logit row, at temperature one, with no sampler
filters. Variance uses population normalization. Entropy is computed with
max-shifted weights, not by subtracting two large unshifted logit quantities.
F64 logsumexp itself can still lose its small normalization increment at extreme
offsets. These lens probabilities are not calibrated generation probabilities.

The existing GPU path materializes the full vocabulary, selects 16 entries,
masks those entries, then selects 16 more without a final mask. The summary
restores all first-pass entries from their saved F32 IDs/values in one bounded
host row and rejects any non-finite vocabulary tail. No new GPU kernel or head
evaluation is used. The inspector validates optional fields, vocabulary and
bounds, and rejects partially summarized trace documents; it never fills old
artifacts with inferred statistics.

Inspection validation version 2 permits only the immediately adjacent F64
values around the selected F32 maximum when comparing `score_max`. A retained
live trace reproduced a one-F64-ULP drift in serde_json's default decimal parser:
`15.563257217407227` became bits `402f206340000001`, while the selected F32 score
converted exactly to `402f206340000000`. All 8,384 rejected cells in that
35,658-cell shard differed only by one ULP in this check. The 17-line score-only
`distribution_roundtrip_fixture.json` records the first failure (layer 0,
position 5), without prompt text or token identities. Two-ULP changes and
neighboring F32 maxima remain rejected. No score calculation, kernel, stored
statistic, or artifact is changed or clamped. `inspect ... summary` reports
`distribution_summary_validation_version: 2` for summarized traces, so retained
version-1-rejected captures can be revalidated without regeneration. Unsummarized
legacy inspection output omits this field.

For C cells and V vocabulary entries, added CPU work is O(CV), including
exponentials and several scans, with 4CV bytes of logical shared-buffer readback
and a 4V-byte host row. This may be expensive. Each summary has 52 bytes of
numeric payload before layout/JSON overhead; admission reserves an additional
512 JSON bytes per cell. Existing document/cohort caps remain enforced. GPU
readout timing does not include CPU reduction; total trace execution wall time
does. Collectors may explicitly choose a representative subset of complete
traces for summaries while retaining frozen-reader top-k coverage elsewhere;
they must record that subset, not claim that every collected cell has statistics.

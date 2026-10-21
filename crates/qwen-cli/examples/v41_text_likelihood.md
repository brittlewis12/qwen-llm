# PRIVATE v4.1 text likelihood witness

This is a bounded raw-completion measurement, not a production CLI or task
acceptance test. It uses public production forward APIs without capture hooks,
features, additional GPU buffers, forced routes, or changed numerical contracts.
The currentNative/groupedNativeF32 choice comes from the source branch, not a
runtime option. Cherry-pick this additive helper onto each branch and build each
from its own clean committed tree with the same toolchain/SDK and recipe.

## Interface

```sh
cargo build --offline --locked --release -p qwen-cli --example v41_text_likelihood
target/release/examples/v41_text_likelihood prepare MODEL CORPUS.json FRESH_PACKET.json
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 \
  target/release/examples/v41_text_likelihood score MODEL PACKET.json FRESH_OUTPUTDIR

# CPU-only unit tests: no model files, MetalContext, lease, or GPU execution.
cargo test --offline --locked -p qwen-cli --example v41_text_likelihood
cargo fmt --all -- --check
```

No optional arguments. Paths with spaces must be quoted. Keep evidence outside
the source tree. Parents of fresh paths must exist. Never reuse or delete failed
outputs. `prepare` is CPU-only (GGUF mapping and embedded native tokenization).
`score` is a GPU/model job: run only on the separately approved host/model, never
on M4 or external models. Implementing/testing this helper does not authorize
such a run. It requires a clean matching release build, no diagnostic features,
Metal validation, and the normal production process lease. SIGINT/SIGTERM use the
existing cancellation flag and unwind for shutdown; a hard kill cannot finalize
failure reports.

## Corpus and Frozen Packet

The parent prepares exactly six documents, three `prose` and three `code`, from
the pre-existing GLM quality cohort's retained **raw Git source**, not its token
IDs or chat prompts. No source retrieval, cohort construction, or remote job is
performed by this helper. Corpus JSON is an object with one `documents` array;
every element has exactly these five string fields:

```json
{
  "documents": [
    {
      "name": "unique-document-name",
      "stratum": "prose",
      "source_commit": "0123456789abcdef0123456789abcdef01234567",
      "path": "original/repository/path.md",
      "text": "Full retained raw source text goes here, including original whitespace."
    }
  ]
}
```

The illustrated entry must be replaced with real retained provenance and raw
text, and the array must contain all six documents. Names and commit/path pairs
must be unique; commits must be full 40- or 64-hex IDs. Provenance is declared,
not remotely authenticated. Unknown fields are rejected.

Each whole text is encoded **once** with `NativeTokenizer::from_gguf` and
`encode(text, false)`: no added BOS/EOS, GLM IDs, chat template, trimming, or
helper normalization. Literal special-token spellings are still recognized by
the native tokenizer. Documents with fewer than 576 tokens fail the entire
preparation; none are silently skipped or substituted. The packet retains all
source text/metadata, whole-text token counts, explicit `prefix_ids` (first 512)
and `target_ids` (next 64), tokenizer metadata/policy, preparation build identity,
and canonical model/shard paths and sizes. Only the declared suffix after token
575 is omitted. Scores do not re-tokenize or authenticate the frozen IDs. Use
the exact same packet bytes for both branches. Expansion requires a deliberate
new protocol revision, not implicit sampling.

## Evidence

For every document: a fresh `new_state_with_gpu_step(576)`, one production
`prefill` of 512 IDs (chunk 256), then target 0 scored using prefill logits.
Consume targets 0 through 62 to score targets 1 through 63; target 63 is never
consumed. No sampling, EOS stopping, or cross-document state reuse.

- `manifest.json`: build/runtime identity, compiled Metal compiler/SDK stamps,
  environment, layout, and limitations. Runtime `rustc -vV` is explicitly not a
  build attestation; keep the build invocation/toolchain record with evidence.
- `packet.json`: exact input bytes, plus `model.json`, `geometry.json`, and
  `device.json` for the actual run. Model/shard paths and sizes must match the
  packet; no model hashes or authenticity claims.
- `NNN.logits.f32le`: one headerless binary file per document, ordered as
  `[64, 129280]`, target-index major then raw vocabulary ID, little-endian F32.
  Exactly 33,095,680 bytes per completed document, suitable for later paired
  TV/KL. Do not treat a partial/nonfinite failed file as completed evidence.
- `NNN.tokens.jsonl`: one JSON object per completed target, including zero-based
  target index and absolute position, ID/logit, F64 natural-log NLL, 1-based rank,
  top-1 ID and correctness. Ties prefer lower token IDs for both rank and top-1.
- `NNN.summary.json` and `result.json`: document means and equal-document corpus
  means for NLL, rank, and top-1 accuracy. All targets have equal count, but they
  are not independent statistical trials. No token-level significance claims.

F64 max-shifted log-sum-exp computes NLL as `(max - target) + ln(sum(exp(x-max)))`,
including extreme finite F32 values. Invalid lengths, token bounds, nonfinite
logits, or execution failures fail the run. Changes in NLL, top-1, rank, or routes
are **not** synthetic label failures; there is no task acceptance threshold or
baseline-exactness requirement. Compare paired document means and retained
distributions later; this helper does not implement the paired analysis.

Files are create-new only, and score directories are exclusively created.
`manifest.json` records a started run and `result.json` records completion or
failure. Missing/failed results are not successful evidence. Preparation failure
leaves a failure manifest at the reserved packet path, with a different schema
that `score` rejects. I/O failure or hard termination may leave partial files;
retain them and choose new paths for retries.

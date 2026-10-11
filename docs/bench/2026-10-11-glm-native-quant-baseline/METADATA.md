# Equivalent GLM Storage Conventions

Follow-up to `0c25c9d1`. The executed graph remains the supported 45-block GLM
trunk; this does not introduce support for a different model architecture.

## Normalization

Reference semantics are the local llama.cpp `glm5-next.cpp`, `llama-hparams.h`,
`llama-model.cpp`, and metadata-loader getters.

- Missing NextN count means zero. A declared unused NextN block may omit some
  or all of its tensors. Present recognized tensors still receive name, shape,
  storage-layout and byte-range validation and never enter execution residency.
- Missing shared expert width is expert width times shared expert count.
- Missing shared clamp inherits the routed clamp; a scalar clamp broadcasts.
- Missing routing function or explicit NONE (0) resolves to sigmoid, as in the
  reference loader. Other explicit routing functions remain unsupported.
- Indexer `true` and integer `1` both mean full per-layer indexer execution;
  false/zero do not silently become full execution.
- Convolution `[4,1,width,1]` and `[4,1,width]` use the same bytes. Residency
  supplies the canonical rank-three view required by existing KDA encoders.

Malformed explicit values are not replaced by defaults. Block-count arithmetic
still requires exactly 45 executed blocks and at most one declared NextN block.

## Results

- 24 binding/metadata tests passed; two opt-in artifact tests were separate.
- The pinned saved GSQ3 header passed actual Rust metadata parsing and descriptor
  binding: 1,383 trunk tensors, no NextN, and 102 singleton-tailed convolutions.
  Header SHA-256: `856fb2662ccea169fd6c2d6fde1bd908a11c12cb54999f9badc12b32b053dadd`.
- Full targeted GLM suite: 80 passed, 44 existing opt-in tests ignored.
- Native Maya prefill/decode integration passed again with unchanged reported
  KL values and session allocation accounting.
- Independent source review found no concrete correctness issues.

The GSQ3 result establishes descriptor compatibility only: its full weights were
not downloaded, loaded, quality-tested, or timed in this checkpoint. Existing
native primitive tests cover the underlying formats and compositions. No new
artifact-specific execution gate was added.

## Retrospective

Storage equivalence is a loader responsibility, not a reason to add a second
execution path. Normalize before the existing kernel checks, preserving those
checks and avoiding broad reshaping or invented architectural defaults.

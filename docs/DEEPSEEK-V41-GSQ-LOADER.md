# DeepSeek V4.1 STANDARD GSQ-RCO loader bridge

This is serialization/source compatibility, not a new GSQ implementation or an
artifact promotion. It reuses the existing native expert capability gate,
resident weight ownership/admission, GGUF split validation, HashConfig
normalization, DiskEngram gather, and standard quantization codecs. No IQ1
adapter, tokenizer identity map, hash parameter fallback, or new GPU path is
introduced.

The CPU contract was checked against the saved headers and bounded controls for
`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` at revision
`cb02422c900df489cea5833e8a92171224059b3c`. Headers/prefixes do not verify full
weight hashes. Neither the model card nor this bridge establishes performance,
quality, local capacity, or full-context execution.

## Source ownership

- The backbone remains its validated four-shard, 1008-tensor logical GGUF group.
  The auxiliary file is not a fifth shard or two additional backbone tensors.
- `deepseek41.engram.file` selects only the declared safe relative path. Resolve
  it beside the logical first-shard path, without canonicalizing the backbone
  symlink into an HF blob directory. No network, filename aliases, or model scan
  is used. Relative path checks are lexical, not a symlink sandbox.
- The companion must be a standalone `deepseek41-engram` container containing
  only the declared per-layer lookup tables with matching bucket geometry. Its
  architecture remains non-runnable. A declaration together with inline tables
  is rejected rather than selecting one silently.
- Both containers must declare the same nonempty `deepseek41.engram.source`.
  This is an opaque declared identity match, not a hash allowlist or
  cryptographic authentication.
- A plan owns the auxiliary container and table descriptors; the caller retains
  the borrowed backbone. DiskEngram owns both containers. Each descriptor's
  shard index is local to its selected container; auxiliary rows read the
  retained auxiliary FD, not backbone shard zero. Plan/DiskEngram source-group
  reports preserve separate stamp lists.
- Both groups' retained descriptor stamps are checked around planning and
  before/after resident loading. The disk reader also checks around row gathers.
  Path replacement never redirects retained-FD reads. Mutation invalidates the
  stamp contract; stamps do not replace full artifact verification. Sources must
  remain immutable, as with the existing shared resident mappings.

The companion has two IQ4_NL `[256, rows]` tables. Their 144-byte packed row
stride follows the existing codec block traits. Only selected rows are decoded
on CPU, preserving shuffled/repeated IDs and checking bounds and finite output.
The 110,595,290,400 table payload bytes are counted as disk-only bytes and dtype
inventory, never as resident weights. The generic GGUF reader retains a lazy
read-only virtual file mapping; this is not table materialization or GPU import.
Resident storage windows still use only backbone resident descriptors.

## Control serialization and CPU accounting

The existing metadata-inline Maya/PLE path keeps its semantics. Tensor controls
normalize into that same path, including layer-local offsets, PLE global-offset
normalization, metadata/tensor conflicts, and HashConfig validation. No controls
are synthesized. A zero `hash_layer_count` does not disable the declared Engram
layers. `engram_vocab_size` is not the compressed vocabulary size.

Tensor controls require exact signed little-endian reads, with negative values
rejected before conversion to the unsigned address domain:

| Tensor | GGML type | Logical shape | Bytes |
| --- | --- | --- | --- |
| `engram_token_map.weight` | I32 | `[129280]` | 517120 |
| `engram_primes.weight` | I64 | `[24, 2]` | 384 |
| `engram_offsets.weight` | I64 | `[24, 2]` | 384 |
| `engram_multipliers.weight` | I64 | `[4, 2]` | 64 |

Planning is therefore NOT zero-payload metadata-only for this serialization:
each call reads 517,952 control bytes, plus ordinary GGUF headers. It never reads
table payloads. Each control is capped at 16 MiB, checked against its exact
dtype/shape/storage extent, and allocated with fallible reservation. All byte
arithmetic is checked. The shared parser drops trailing unit axes; control
geometry uses the same unit-axis convention as the existing loader.

`Plan.controls` reports payload bytes, conservative cold normalization bytes,
retained token/hash bytes, and source bookkeeping separately. For this package,
the normalization bound is 3,105,632 bytes: serialized bytes plus 20 bytes per
integer, deliberately summing non-overlapping phases to cover raw staging,
normalized u64 copies, optional inline conflict copies, and final u32 mapping.
The retained token/hash bound is 519,648 bytes. Source bookkeeping includes
table/stamp descriptors, path/name/shape storage, transient stamp copies, and
small allocation slack. It excludes the already-established GGUF parser/header
storage and virtual mappings; these are not a total process-RSS prediction.

Resident loading first applies the existing CPU/Metal admission function to a
conservative six-times-serialized cold control bound before normalization. Its
resident admission adds retained control/source bookkeeping to the existing CPU
domain alongside private mapping shadows, not to GPU residency or a CPU weight
shadow. Standalone CPU contract planning remains hardware-independent and uses
the bounded, fallible allocations rather than imposing a host-fit gate. Existing
request-owned CPU row staging and forward token-map ownership remain unchanged.

The additional underscore Engram geometry/pad/vocabulary aliases, attention
candidate declarations, and `expert_scoring_func` are consumed and checked for
conflicts. Equal current defaults are not a reason to ignore declarations.
The exact GSQ grouped output-A shape `[group_input, rank, groups]` is accepted
with the same contiguous row order as the existing flattened runtime view;
other axis permutations are not treated as equivalent.

## Verification and integration

Parent checkpoint, 2026-10-24: all five requested IQ2_XS/auxiliary files match
their pinned LFS lengths and full SHA256s, totaling 265821115072 bytes. The
private verification receipt SHA256 is
`4261f9175dec537eafcfce0c6b7e6e37ff63ba0447cb179130eaa8c9fa20af0f`.
Separately, frozen source `b2486f97` passes all 71 GGUF fixtures on M5 under
Metal API validation, including eight GPU loader fixtures. Frozen binary SHA256:
`4a54e659ad16cc0065b37ee24c0857e04b3bf5a4753090d071d9f906970375af`.
The sparse CPU replay is ignored in that run and qualified separately. These
synthetic fixtures do not load the verified full artifact or establish its
quality, performance, context capacity or Full/CED equivalence.

Full-weight checkpoint, 2026-10-25: separate unbatched CED composition
`05edb66ff4bf5baf25e022faef8eac57c526dad6` now passes actual IQ2_XS pairs at
128/2048/8192 prompt rows. Each compares all 257 complete 129280-logit vectors
and all 56 causal regions per position, with unchanged frozen input, one import,
one live KV state, no fallback and 21/21/25 completed CED commands. V2 reports
retain both source groups, planned/loaded FD equality and unchanged endpoint
identities; six corrupt source/control report variants are rejected. This is
within-GSQ equivalence, not a comparison with Maya logits.

The actual resident shared views total 155220049920 bytes (144.560 GiB), with
zero private weight shadow and 110595290400 table payload bytes disk-only.
Observed/accepted incremental Metal bytes are 214777856/237092864 at 128,
2983034880/3005284352 at 2048 and 5718163456/5740412928 at 8192. These are sampled
increments and admitted bounds, not continuous process/driver peaks. Frozen core
SHA256 is `d26af009441a86f4bbc152ffb89db38d67215f232d56755a6a382fc753acff6e`.
Ordered validation-on timings are diagnostic only. No quality, native-speed,
full-1M-context or CED-default claim follows, and the CED executor remains on its
separate branch rather than being shipped by the main loader bridge.

CPU fixtures cover inline/tensor equivalence, large exact integers, malformed
controls, offsets and aliases, four backbone shards plus a separate companion,
source/architecture/path/geometry failures, ambiguity/duplicates, retained FDs,
stamp changes, HF-style symlinks, IQ4_NL rows, and disk-only storage exclusion.
Existing Maya/PLE, mixed quant, ownership and mapping CPU regressions remain
applicable. No GPU test is required to establish these serialization contracts.
Focused optimized checks pass: 63 GGUF CPU tests, the separate saved-evidence
replay, three Flash-Next MoE compatibility tests, four projection preflight
tests, 33 codec-related tests, and five hash/address tests. GPU tests are
explicitly excluded. Workspace all-target check and Clippy complete with
existing warnings, including the shared large LoadError type. Changed Rust
files pass formatting; workspace-wide formatting still reports unrelated
pre-existing formatting in `deepseek_v41/forward.rs`, which is left untouched.

The ignored `cpu_only_sparse_header_control_replay` test accepts
`V41_GSQ_AUDIT_DIR`. It creates temporary sparse files using saved authentic
headers and only the exact saved control payloads, checks the control digest,
and exercises CPU contracts for both variants. Sparse holes are NOT weights;
the replay never calls resident loading, table gathers, or GPU APIs. IQ2_XS
binds all 120 main-layer expert banks at this CPU contract level. IQ2_XXS
metadata/control normalization succeeds while its IQ1 expert banks still fail
the unchanged native bank gate.

Parent integration must include `gguf/companion.rs` and `gguf/controls.rs` in any
frozen CDFwd/source inventory when joining this branch, plus
`gguf/tests/gsq_loader.rs` for test-source inventories. Do not modify frozen
controls to qualify this loader. Preserve the backbone snapshot path when
opening HF symlinks so the declared companion can be resolved.

Full qualification remains separate: complete and verify all four requested
IQ2_XS shards and the companion against pinned LFS sizes/full hashes; use the
intended M5 with its native compiler/device tier and normal lease/admission;
check actual resident windows, ownership, zero CPU weight shadow, and disk row
source identity; run bounded nonzero load/prefill/decode checks and numerical
comparisons against a qualified reference before any performance/quality or
context-capacity claim. The 2.00 package still requires a separate IQ1 expert
adapter before execution; the requested 2.25 package does not need IQ1 support.

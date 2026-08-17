# Structural Thinness Falsifiers

Status: three representation-level hypotheses were measured before any sparse,
factor-state, or block-certificate kernel was written. Broad Ridge FFN support
deletion, the tested Lightning norm cone/fixed-block family, page/run loading,
and a universal rank-24 GDN state are KILL. The reusable feature-gated and
bench-only observers remain; no failed execution kernel was added.

These are dirty-tree mechanism probes at source commit `1ac9446`; they carry no
throughput-promotion authority. Raw captures live under `target/profiles` and
are intentionally not retained in Git.

Model attribution reuses previously authenticated packet identities rather than
rescanning 12.6-104.2 GB of weights inside an observer. Capture manifests bind
the exact model path/byte length and hash every produced tensor; `results.json`
also records the local manifest/analysis hashes. Those hashes make the local
evidence auditable but do not substitute for the prior model-content identity.

## Ridge FFN Inner Occupancy

The bench-only capture records the exact down-projection input after SwiGLU for
all 64 dense layers. Two independent prompts each capture eight generated-token
transitions, giving 512 layer/token vectors and 34,816 physical 256-wide blocks
per trace.

| Metric | Short explanation trace | Code-review trace |
|---|---:|---:|
| Exact-zero scalar fraction | `1.12e-7` (one `-0`) | `0` |
| Scalar `abs(inner) <= 1e-2` | 24.60% | 26.42% |
| Complete block `max(abs(inner)) <= 1e-2` | 0.0460% | 0.00862% |
| Complete block `max(abs(inner)) <= 1e-1` | 9.12% | 9.90% |
| Median blocks removable at 0.1% activation-energy budget | 0/68 | 0/68 |
| P95 blocks removable at 1% activation-energy budget | 4/68 | 4/68 |
| Median blocks carrying 90% activation energy | 55/68 | 56/68 |

Scalar near-zero occupancy does not survive the physical block layout. Even an
uncharged 1% activation-energy approximation usually removes only one of 68
blocks, projecting roughly 1.5% of the 7.395 ms Ridge down projection before
selection, index, irregular traversal, or quality costs. This is below a useful
whole-token ceiling and far below the preregistered 20-35% occupancy signal.

Decision: do not stream Ridge down-weight norms or write a Ridge sparse-down
kernel. This trace makes no claim about A3B or DSV4 activation distributions;
measure those independently only when their phase ceiling justifies it.

## DSV4 Lightning Cone And Locality

The diagnostics-only temporal transcript records operation-wise upward-rounded
F64 norms for each 64x128 index query and every visible 128-wide F16 index key.
It evaluates the real-arithmetic signed-weight bound

```text
score(row) <= ||k_row|| * sum[h where w_h > 0] w_h * ||q_h||
```

against observed deployed scores and prices refinement for contiguous blocks.
This is an empirical envelope, not an admissible runtime certificate: the
fast-math F32 scorer's complete accumulation error is not enclosed. A formally
safe allowance can only loosen the envelope, so it cannot rescue a result that
already retains every row. The real-text request uses the first 34,000
characters of `docs/PERF-ROADMAP.md`, rendered through official DeepSeek chat:
8,412 prompt tokens followed by nine captured transitions. It covers 189 CSA
decisions and 397,635 visible rows. `QWEN_DSV4_PREFETCH=off` and
`QWEN_DSV4_RESIDENCY_SET=0` were explicit.

- The empirical bound has zero observed violations over all 397,635 rows, but
  retains 100% of rows for block sizes 16/32/64/128/256. Of 189 rank-512
  cutoffs, 126 are nonpositive, making any nonnegative norm-only upper bound
  unable to prune immediately.
- Even a perfect bound that knows which fixed blocks contain selected rows would
  retain 62.31%/78.28%/91.69%/98.67% of visible rows at block sizes 8/16/32/64.
  The current competitive target is approximately 20-25% charged work.
- Selected IDs form a median 205 exact runs (p95 262), with median run length
  2.50. Eight-row page loading amplifies selected payload by 2.55x median and
  2.96x p95.
- Merging only one-row gaps lowers the median descriptor count to 153 but still
  loads 1.105x selected payload. The current direct attention already reads only
  512 KiB/layer and charges only one 32-bit ID per selected row; descriptor-only
  work has no credible phase ceiling.

Decision: KILL the positive-weight norm cone, fixed contiguous block envelopes,
page loading, and run-descriptor attention under this premise. This does not
claim that every possible row-specific certificate is invalid; a successor must
first demonstrate below 20-25% complete charged work without relying on a
nonnegative bound when the cutoff is negative.

## Ridge GDN Alpha And State Rank

The alpha observer captures all 48x48 per-token decay values for 512 source-text
tokens. Full F32 states for all 48 value heads are retained at positions 64 and
512 for GDN indices 0/23/47 (absolute layers 0/30/62), followed by host F64 SVD.
The authoritative capture forces the ordinary serial decode topology because
the first acquisition showed that the default concurrent path bypassed the
serial observer; that all-zero observer output is rejected rather than treated
as model evidence. All six position/layer state tensors are bit-identical
between the default concurrent acquisition and the repaired serial observer, so
the SVD result is not a scheduling artifact.

- Across 1,179,648 alpha values: zero exact zeros, eight exact ones, minimum
  `2.31e-20`, median `0.982703`, and p95 `0.999927`.
- Fractions at or below `1e-6/1e-4/1e-2/0.1/0.5/0.9` are
  `0.0687/0.1958/0.8091/2.6649/12.1874/30.9730%`. Median decay half-life is
  39.7 tokens, but p95 is 9,493 tokens. Reverse division by alpha is therefore
  badly conditioned on a nontrivial tail even though no exact alpha reset occurs.
- At position 512, rank 24 has median/p95 relative Frobenius residual
  `1.42%/20.28%`; rank 32 has `0.93%/15.09%`; rank 64 still has
  `0.19%/4.97%`.
- Reaching 1% residual needs median rank 31 and p95 rank 100. Reaching 0.1%
  residual needs median rank 77 and p95 rank 120. A two-factor rank-100 state is
  larger than the dense 128x128 matrix.
- Rank is strongly layer/head dependent: at position 512, rank-24 median
  residual is 0.43% at layer 0, 6.92% at layer 30, and 4.52% at layer 62.

Decision: KILL a universal fixed-rank-24 GDN representation and do not build the
factor update/recompression kernel. The state has a dominant singular direction
but a consequential, highly nonuniform tail. Also do not call update-log rollback
exact: compact logs can remove checkpoint writes, but FP32 reversal is
non-injective and the observed alpha tail makes repeated inverse rollback
ill-conditioned. Existing N2 checkpoint timing remains below a compelling
product ceiling.

## Reproduction

```sh
uv run scripts/profile/dsv4_selected_locality.py \
  crates/qwen-llm/tests/fixtures/deepseek_v4_position3070_native_decisions.json

FFN_CENSUS_MODEL=/Users/tito/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf \
FFN_CENSUS_OUT=target/profiles/ffn-inner-census-ridge \
FFN_CENSUS_TOKENS=8 \
cargo run --release -p qwen-llm --example ffn_inner_census_capture
uv run scripts/profile/ffn_inner_census.py \
  target/profiles/ffn-inner-census-ridge/manifest.json

QWEN_DECODE_DENSE_CONCURRENT_GDN=0 \
GDN_CENSUS_MODEL=/Users/tito/models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf \
GDN_CENSUS_TEXT_FILE=docs/PERF-ROADMAP.md \
GDN_CENSUS_OUT=target/profiles/gdn-state-census-ridge-serial \
cargo run --release -p qwen-llm --example gdn_state_census_capture
uv run scripts/profile/gdn_state_census.py \
  target/profiles/gdn-state-census-ridge-serial/manifest.json
```

The DSV4 acquisition command is intentionally not promoted as a performance
command: diagnostics perform host readback. Its complete output is
`target/profiles/dsv4-lightning-cone-8k-temporal.json`.

No run used `MTLResidencySet`, `requestResidency`, pre-wiring, `mlock`,
cache-bypass reads, or the residency-coupled A10B path. GPU/model workloads were
serialized. The user-owned OvisOCR llama server remained alive and untouched.

# DFlash2 K0-S Semantic Tooling Authorization - 2026-08-23

Status: prospective source-work authorization, frozen before K0-S code changes
or acquisition. It authorizes only bench/test/read-only diagnostic tooling for a
development K0-S probability/causal semantics packet. It grants no K0-S result,
cross-backend score equivalence, K0-L, proposal RNG, coupling, E1b, verifier,
held-out/reserve, product, serve, default, or production authority.

## Objective And Corrected Contract

Determine whether the local DFlash2 selector implements the upstream sparse-q
semantic object on its own backend:

```text
score_t(a, b) = unary_t(b) + dot(A(a) * H(h_t), B(b))
q_t(b | a) = softmax(score_t(a, b) / request_temperature)
```

K0-S covers the backend-selected candidate set, causal predecessor chain,
score construction, request-temperature normalization, and issue behavior. Raw q
has zero mass outside the selected candidate token IDs. Target top-k/top-p/min-p,
grammar, penalties, and other target processors never apply to q.

Candidate slot order and top-k ties are backend-specific replay inputs. K0-S
does not claim one portable slot order or exact score/probability bits across
Metal, MLX, vLLM/Triton, and llama.cpp/ggml arithmetic. It also does not select a
portable RNG: MLX categorical, vLLM candidate-keyed Gumbel, and llama.cpp
`std::mt19937` are distinct realizations.

Proposal abstention is disabled in K0-S. Proposal `p_min`/`n_min` are null and
must remain distinct from target sampler `min_p`. No duplicate repair, epsilon
mass, score clipping, nonfinite drop, or independent proposal temperature is
allowed.

## Pinned Semantic References

- Released asset `incoai/Qwen3.8-27B-DFlash2-GGUF`, revision
  `6cb5872e2cee6b4e780a8414922350be8e42d65c`, file
  `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, local path
  `/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
  `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`.
- Author-maintained MLX reference `z-lab/dflash` commit
  `07ebd93db9f472af339b644bb70221ad8428328a`,
  `dflash/model_mlx.py` SHA-256
  `2f8598eaca4cb814e63ea69e791c1bdf55ba82280bfd299f9141906356b7cb87`.
- Merged vLLM reference commit
  `b389ac29465b33f9e9c534df221ea3c129e9793f`:
  `vllm/model_executor/models/qwen3_dflash2.py` SHA-256
  `c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f`;
  `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` SHA-256
  `1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836`.
- Same-GGUF llama.cpp PR 27342 head
  `1deefcca395743049c3820ab8f9b15043f3e9446`:
  `src/models/dflash.cpp` SHA-256
  `3b31b1a6b888ec37013a4d6275fdaf1fdeb6a9e3ca25b933ab136acdbcd58c49`;
  `common/speculative.cpp` SHA-256
  `e14da24022c4c16d2de7cb582257020a02893c5bbcc6bb2b052cf1e8a8525138`.

These sources establish the semantic formula and request-temperature sparse q.
They are not an acquired cross-backend real-model fixture. A local K0-S pass
therefore remains development semantic conformance and cannot be relabeled
authoritative cross-implementation parity.

The commit, path, and hash identities above are reviewed constants embedded in
the authorized tooling. This phase neither downloads semantic-reference files
nor accepts a producer-supplied local copy as a trust root. A later run plan may
vendor and authenticate reference copies only under separate prospective
authority.

## Current Local Baseline

At clean commit `8b99602`, the relevant source identities are:

| Source | SHA-256 |
| --- | --- |
| `crates/qwen-llm/src/metal_dflash.rs` | `315237d2c62d43784e73c6d45d10d4f7b9f885010b9d41aaf8a8073c5c35ed45` |
| `crates/qwen-cli/src/bench.rs` | `bfee954fe3579aefc04622d842eaed7790a9f110c0d1332267b68bf330ad242d` |
| `crates/qwen-llm/Cargo.toml` | `780c4ce96f79b923fccf66bdd1e79bf0b3398557a31a39de75018ea1b780a914` |
| `crates/qwen-cli/Cargo.toml` | `39e1f8363a09872c0bd8455fd962e31aaddff23298756d655a173cb0d1e9b1c3` |
| `crates/qwen-llm/src/metal.rs` | `9343fe60597cc4695df7094cd1916d204d5dc624055221dc96ec6c77a2a65002` |
| `crates/qwen-llm/src/metal_forward.rs` | `5747420a0590f6ef36ea25ae8abad39d928df58bf4ec41b7e3212916baa550f4` |
| `kernels/dflash2.metal` | `3a36d935f449511c649839a86123e554b3c00213ff17bdd91ebbc8b2eb06b66f` |

The existing diagnostic synchronizes and reads backend top-k IDs, unary logits,
selector-hidden rows, and CPU codebooks, then exactly replays only the production
greedy predecessor chain. The current reducer computes a separately configured
local-reference softmax and rejects all issues; it is correctly labeled
`local_reference_only`, not K0-S.

## Authorized Source Surface

Allowed edits are exactly:

- `crates/qwen-llm/Cargo.toml` and `crates/qwen-cli/Cargo.toml`: declare one
  non-default `dflash-k0s-diagnostics` feature, propagated from CLI to library;
- `crates/qwen-llm/src/metal_dflash.rs`, including inline unit tests: add
  feature-gated immutable full-lattice/raw-provenance diagnostic records and pure
  CPU helpers over already synchronized selector buffers;
- `crates/qwen-cli/src/bench.rs`, including inline parser/call-site tests: add
  only the feature-gated module declaration, hidden subcommand, and dispatch;
- one new feature-gated bench module `crates/qwen-cli/src/dflash_k0s.rs`, including
  inline tests, with schema `qwen.dflash_k0s_lattice` v1;
- one new pure-offline reducer `scripts/profile/dflash_k0s.py`; its built-in
  self-test contains every reducer test.

No generation, serve, sampler, kernel, model-loader, other script, other source,
Cargo lock, or other documentation file is authorized. The default feature set
must not compile the full-lattice types/method/call site. A source test must prove
that no product-generation or serve file refers to them.

The K0-S module/trace is separate from `dflash_sampled_oracle` and E0. It contains
no target p distribution/logits, target or proposal RNG, sample decision,
acceptance/correction, generated stream, or generation event. The reducer rejects
those fields and concepts anywhere in the schema. The hidden entrypoint advances
only the fixed prompt/carry needed to expose one selector block; it is not a
sampled generation loop.

Instrumentation must not alter candidate generation, GPU synchronization,
production greedy selection, codebook storage, drafter state, target sampling,
RNG consumption, acceptance, generation, or serve behavior. Product code may not
read raw lattice records or reducer output. Existing greedy replay remains
mandatory.

No new model asset, conversion, quantization, package dependency, upstream clone,
network execution, or external backend build is authorized by this source-work
phase. This phase permits CPU tests and synthetic Metal tests that open no model
asset; every such Metal invocation literally sets `QWEN_METAL_LEASE_WAIT=1`.
Any command/test that opens a target or drafter asset, executes a real-model
forward, or records real-model selector values is acquisition and remains
unauthorized until a separate committed run plan.

## Full-Lattice Diagnostic Contract

For block size `N=8`, selector top-K `K=16`, rank `R=256`, hidden size `H=5120`,
and vocabulary `V=248320`, exactly `N-1=7` selector depths are active. Depth 1
has one carry predecessor row. Depths 2..7 each have K predecessor rows in the
prior depth's backend slot order. The score lattice therefore has exactly
`1 + (N-2)K = 97` rows of K scores.

The diagnostic exposes read-only inputs sufficient for non-circular local replay:

- complete draft-logit f32 bits for depths 1..7, then backend-ordered K candidate
  IDs and unary f32 bits. A normal row requires every full draft logit to be
  finite. For a finite row, the reducer independently reconstructs all 16 slots
  in descending ordinary f32 numeric value, then ascending vocabulary token ID
  for equal values, and exactly verifies each ID and unary bit pattern against
  the full logits. Positive and negative zero compare equal, so lower token ID
  resolves their order and every tie crossing the K boundary. This freezes only
  the authenticated local Metal rule in `kernel_topk16_f32` and grants no
  cross-backend ordering claim;
- pre-selector-projection normalized hidden H-vectors and locally projected
  selector-hidden R-vector f32 bits for depths 1..7;
- authenticated GGUF tensor names, descriptor shapes/dtypes/layout/offsets and
  full-tensor SHA-256 for `selector_hidden`, predecessor A, and successor B;
- raw `selector_hidden` tensor bytes plus raw quantized bytes/descriptors for
  every referenced A/B row, including carry, vocabulary-row IDs, and proof that
  A/B row domains equal the target/drafter vocabulary contract;
- the local 97-by-16 score lattice, production greedy predecessor/token chain,
  exact issues, and chain termination;
- target/drafter/executable/source/build/host identities and the pinned semantic
  source hashes.

Raw tensor material goes in one authenticated exclusive-create binary sidecar;
JSON contains only bounded scalar records and sidecar dtype/shape/offset/bytes/
SHA references. The reducer independently decodes supported raw GGUF types,
computes an independent bounded selector-hidden reference, reconstructs A/B
rows, and replays every finite edge and score. Local
dequantized values alone cannot serve as their own reference. Tensor name,
orientation, row mapping, descriptor, asset aggregate, and sidecar coverage must
all agree; an A/B swap or consistently wrong local codec is a failure.

Selector-hidden projection comparison is tolerance-based, not bitwise. The trace
binds the exact Metal dispatch, compiled-kernel identity, reduction geometry,
input/weight staging, product and accumulator formats, output conversion,
rounding mode, and denormal handling. The reducer maps source/build hashes and
dispatch identity to an embedded allowlisted contract; producer metadata cannot
define that contract. An unknown or mismatched dispatch, precision, operation
count, rounding mode, or denormal mode is invalid.

Reference and bound arithmetic are frozen. The reducer independently decodes
each finite logical operand, represents every f32/f16/bf16 value as an exact
signed dyadic integer and exponent, and uses integer exponent alignment to form
the exact unstaged dot `S` and `sum_abs_products`; it does not use iteration-order
f64 addition. Format-derived exponent/bit-length caps and the fixed H bound are
checked before integer shifting or allocation. It then independently applies the
contract's round-to-nearest-even staging (and any documented flush) to both
operands and forms the exact staged products and sum `S_stage`. Thus the exact
term `E_stage=abs(S_stage-S)` includes input/weight cross terms. If products round
separately, the reducer exactly rounds each product and sets `E_product` to the
sum of its exact absolute product rounding errors; an authenticated fused
contract instead sets `E_product=0`.

For accumulator unit roundoff `u_a`, the frozen conservative maximum is
`m=2H=10240` rounded accumulation operations per output; the allowlisted graph
must prove it does not exceed that count. The reducer requires `m*u_a < 1` and
uses `gamma_m=(m*u_a)/(1-m*u_a)`. Its accumulation term is
`E_accum=gamma_m*sum_abs_rounded_products +
(m*delta_a)/(1-m*u_a)`. For gradual underflow, `delta_a` is half the accumulator
format's minimum positive subnormal; for flush-to-zero it is that format's
minimum positive normal. Product and output underflow are likewise either
exactly modeled from the allowlisted operation graph or charged once per possible
rounding/flush; an undocumented denormal mode or operation cannot pass. The
gamma term alone is never an underflow bound.

The final comparison is performed on exact dyadic rationals and outward-rounded
only for human-readable reporting. `E_output` is zero for an unchanged f32
accumulator-to-f32 write; otherwise it is the exact supremum conversion/flush
error over the dyadic interval formed by the exact sum of rounded products plus
or minus `E_accum`. Crossing output-format overflow is invalid. The total bound
is `E_stage + E_product + E_accum + E_output` plus two f32 ULPs. Here one ULP is
`2^-149` at zero or a subnormal reference and otherwise
`2^(floor(log2(abs(S)))-23)`, including at the largest finite binade. A nonfinite
or f32-out-of-range `S` is invalid. No observed value or post-run constant may
tune any term. Authenticated local projected f32 bits become the exact input to
subsequent local score replay only after every component passes this bound.

Tooling hard caps are one block, 7 depths, 97 lattice rows, 16 slots, rank 256,
hidden 5120, vocabulary 248320, 16 JSONL records, 64 MiB trace, 64 MiB sidecar,
128 MiB combined trace-plus-sidecar payload, and JSON nesting depth 16. Pinned
external target/drafter files are separately size-bound by the run manifest and
hashed with fixed-size streaming buffers; their multi-GiB bytes do not count
against the payload cap or enter memory as one allocation. Any geometry or byte
count over its cap fails before allocation. A later run plan may tighten but not
expand these limits without a new authorization.

Any nonfinite full draft logit makes top-K support and q undefined and fails the
row; its raw f32 bits remain evidence. At each lattice row, slots are scanned in
ascending backend order. Duplicate valid slots remain present and are scored;
invalid successor IDs retain a null score. Within a slot, `duplicate_id` is
emitted first when applicable, followed by either `sentinel` for an invalid ID or
`nonfinite_score` for a valid computed nonfinite score; a slot cannot emit both
of the latter issues. `no_valid_choice` is emitted after every slot issue and is
the final row issue.

Every local score retains its raw f32 bits, but exact arithmetic-bit comparison
is required only when all operands and the result are finite. For NaN, the raw
local payload bits are preserved as evidence while the reducer compares only
classification plus issue kind/order; it never claims to reconstruct a NaN
payload through arithmetic. Strict `score > best_score` governs local choice:
NaN and negative infinity cannot win, positive infinity can win, and equal
values, including signed zeros, retain the first backend slot.

Chain events are named exactly `invalid_carry`, `missing_predecessor_row`, and
`slot_zero_termination`. They respectively cover an out-of-domain initial carry,
an externally traversed valid token without its required next-depth lattice row,
and a `no_valid_choice` index-zero fallback that cannot furnish a valid next
predecessor. Each records the offending depth/token/slot as applicable and
terminates the affected traversal before an invalid dequantization; none may
escape as an unclassified exception. q is undefined on any issue row. Any issue
makes a normal K0-S row non-passing; no slot or later failure is repaired.

The full lattice permits offline traversal of preregistered non-greedy slot
chains as well as the production greedy chain. The producer never samples q and
never changes the production predecessor from those chains.

## Independent Reducer

The reducer is read-only for every input and exclusive-create for its single
output. It neither appends to nor overwrites any path. Requirements:

- duplicate-key/nonfinite-JSON rejection, exact schema keys/version, one input
  trace/run, bounded records/bytes/nesting, pairwise distinct canonical paths,
  and complete authenticated sidecar coverage with no overlap, gap, alias,
  integer-wrapped range, noncanonical offset, or unreferenced trailing byte;
- independently recomputed SHA-256/bytes for the reducer itself, executable,
  source files, target/drafter assets, trace, sidecar, fixture, and command
  manifest; expected values come from the committed run plan, and
  producer-supplied identities are claims, not trust roots. Each GGUF is hashed
  and descriptor/row-parsed through the same already-open regular-file handle;
- exact comparison of declared semantic-reference identities against the
  reducer's embedded reviewed constants, with no accepted local reference path;
- explicit rejection of any target-p logits/distribution, sample/RNG,
  acceptance/correction, generated-stream, or generation-event field;
- full-logit top-K membership and ID-to-unary verification before selector-score
  replay;
- independent raw-tensor descriptor/orientation/row-domain/Q4_K/Q8_0/F32/F16/
  BF16 decoding and the frozen bounded selector-hidden projection comparison;
- exact f32-bit reconstruction, only for finite operands and finite results, of
  local gate multiplication, rank-ordered dot accumulation, unary addition,
  local first-max replay, and the finite score lattice;
- request temperature taken only from the target request's f32 bits; zero,
  negative, nonfinite, or separately supplied proposal temperature is invalid;
- for each finite valid row, `m=max(scores)`,
  `w_i=math.exp((float64(score_i)-float64(m))/float64(temperature))`,
  `W=math.fsum(w)`, and `q_i=w_i/W`; require finite nonnegative weights, `W>0`,
  exact token mapping, zero mass outside support, and
  `abs(math.fsum(q)-1) <= float.fromhex("0x1p-48")`;
- causal traversal of the authenticated production-greedy chain and
  prospectively supplied fixed slot-index chains without RNG; slot permutation
  is test-only metamorphic coverage that moves IDs/unary/scores together and may
  claim choice invariance only without a maximum-score tie;
- exact issue kind/order parity, nonfinite classification parity, preserved raw
  local nonfinite bits without NaN-payload replay, and no silent
  duplicate/sentinel/nonfinite repair;
- explicit `proposal_abstention={enabled:false,p_min:null,n_min:null}` and target
  filter-independence checks on identical authenticated selector inputs while
  varying only ignored target-policy metadata;
- authority label `development_k0s_semantic_only_no_k0l_e1b_product_authority`.

Python arithmetic must explicitly round every local f32 multiply/add step when
checking Rust score bits; native Python f64 accumulation is insufficient. The
softmax is a semantic reference over already authenticated f32 scores, not a
claim of backend-portable probability bits.

Exact finite f32 score replay is scoped to the authenticated executable, host,
Rust profile, and scalar operation graph. The frozen finite-arithmetic contract
requires no contraction/FMA, round-to-nearest-even, subnormal preservation, and
signed-zero comparison. It grants no requirement or authority to reproduce raw
NaN payloads through arithmetic. A synthetic compile/run test must prove the
reducer matches that graph before acquisition; another compiler/host receives no
bitwise authority.

## Non-Perturbation Gate

Before acquisition, fresh equal-capacity sessions run diagnostic-off and
diagnostic-on synthetic fixtures in both execution orders. They must compare
exact draft tokens, selector inputs, complete drafter KV/conv state, target state
when a synthetic target harness is involved, command-buffer/synchronization
counts through production draft completion, zero RNG draws/state changes, issue
ordering, and one-block continuation. Diagnostic-only CPU dequantization,
serialization, and hashing occur after the production draft synchronization and
may not feed any session or production buffer. Any difference is a tooling
failure.

The separate acquisition plan must also freeze an equivalent diagnostic-off/on
fresh-session parity check on the exact real-model input before its evidence row,
including both execution orders and all observable production outputs/state and
synchronization counts available to this backend. Failure stops acquisition. If
the post-synchronization extraction architecture cannot support that check, a
separate prospective justification and authorization must replace this rule
before any real-model run; the synthetic result alone cannot waive it.

## Required Tests

Rust and reducer tests must cover:

- normal multi-depth lattice and exact production greedy replay;
- a fixed non-greedy predecessor chain that changes later score rows;
- exact top-K membership and ID-to-unary mapping against complete logits,
  including signed zeros, a tie crossing the K boundary under descending-value/
  ascending-token-ID order, and a nonfinite full-logit row that cannot define
  support;
- backend slot permutations and equal-score first-slot ties under the scoped
  metamorphic rule;
- duplicate IDs, negative/out-of-range sentinels, NaN, positive/negative
  infinity, and no-valid-choice propagation;
- invalid carry, valid successor with missing predecessor row, slot-zero fallback
  failure, deterministic issue ordering, and explicit chain termination;
- extra/missing rows, depth numbering, exact 97-by-16 geometry, malformed
  dimensions, overflow, absent codebook rows, score-bit mutation, and rejected
  trailing schema data;
- temperatures 0.7 and 1.0 plus rejected zero/nonfinite/mismatched temperature;
- extreme finite score spreads causing exp underflow and normalization-tolerance
  boundary cases; duplicate-slot token mass is undefined/non-passing, never
  silently aggregated;
- target top-k/top-p/min-p metadata changing over identical selector inputs while
  q remains unchanged;
- proposal `p_min`/`n_min` separation and rejected RNG/acceptance fields;
- A/B tensor swaps, row/domain/orientation/dequant mutations, source/embedded-
  reference/asset/fixture/trace/sidecar mutation, input/size/nesting caps, and
  exclusive output paths;
- diagnostic-off/on fresh-session parity in both orders and compile/link plus
  source-scan proof that the default feature set lacks the full-lattice
  capability.

Existing Q4_K codebook row-equivalence and malformed-geometry tests remain
required. Unit/synthetic tests run before any real-model run. All normal tests,
clippy, and the strict reducer self-test must pass.

## Stage Exit

This authorization terminates after reviewed tooling is committed. A separate
prospective run plan must freeze the exact development prompt/tokens, target and
drafter assets, temperature/config, externally traversed chains, clean build,
command, paths, artifact limits, reducer identity, and pass/fail rule before one
real-model acquisition.

Tooling success grants no K0-S result. A later local semantic pass still does not
establish cross-backend score/probability identity, sparse-q acceptance,
sampler-v1 marginal correctness, verifier exactness, economics, or product
readiness. K0-L/E1b and serve/product remain closed.

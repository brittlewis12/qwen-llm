# V4.1 CED Prefill Design

Date: 2026-10-23. Status: researched design and executable CPU dependency
witness, **not an implemented native CED executor or a performance result**.
The ordinary full-forward path remains the reference and default.

## Recommendation

Implement a cold, opt-in shrinking dependency-preserving executor directly.
Keep rectangular reconstruction as a diagnostic control, not the optimization's
endpoint. Add trained bounded-128 replay through the same publication,
initialization, and state-commit primitives as the next numerical-policy arm.
Do not combine the first executor with encoder replay, prefix reuse,
speculation, or intermediate checkpoints.

This separates two questions: whether the engine constructs the required state
correctly, and whether the trained approximation is acceptable for a deployment.
Bounded replay has the larger work reduction, but it does not remove the hard
publication, scheduling, and transaction problems. The dependency-preserving
arm gives those problems an independently checkable reference.

Do not turn either policy into a claim of bitwise or model-wide quality
equivalence. Different matrix shapes, grouped-expert counts, and scalar versus
batched dispatch can change reductions and routing even when dependencies match.

## Current Native Boundary

At the reviewed `91226d04` baseline, `forward/gpu_step.rs` executes every layer
for each supplied prompt chunk; `forward/prefill.rs` bounds submissions at 2048
rows. Global owner sharing is implemented, optimized CED skipping is not.

For the default 40-layer schedule, encoder layers are 0-19 and decoder layers
20-39. Owners 2/8/14 compress pairs; owner 20 publishes ratio-one main KV and
index keys. Later decoder layers reuse those globals, with reindex producers
24/28/32/36. Each layer still has its own sliding-window KV.

The boundary for absolute row `p` is the pair:

```text
E[p]: post-FFN-19 residual, four F32 streams of width 5120
P[p]: FFN-19 outgoing pre-mix, four F32 coefficients
U20[p] = attention_norm20(collapse(E[p], P[p]))
```

The newly computed attention-20 controls are not `P[p]`. Neither a stream mean
nor a normalized single-vector representation is sufficient for later residual
merging. The final head also uses the final FFN's carried mix.

Every prompt row needs owner-20 publication from `U20`: compressor projection
and normalization; index-key projection from the normalized **pre-RoPE** latent;
key normalization/RoPE and main-latent RoPE; both cache writes. Publication does
not need decoder queries, attention outputs, output projections, or MoE.
Current `prepare_rows` combines several of these operations, and `select=false`
does not suppress publication. Extract a real publication operation rather
than interpreting that boolean as replay mode.

Local sources: `crates/qwen-llm/src/deepseek_v41/forward/gpu_step.rs`,
`crates/qwen-llm/src/deepseek_v41/forward/prefill.rs`,
`crates/qwen-llm/src/deepseek_v41/forward/attention_gpu.rs`, and
`crates/qwen-llm/src/deepseek_v41/operators/mhc.rs`.

## Dependency Argument And Execution Ranges

Hold encoder-derived global KV and index keys fixed. A decoder query reads
global **encoder** representations, not historical hidden states of preceding
decoder layers. Candidate selection and reindexing add same-query, vertical
control dependencies; their row identities must travel with the hidden state.
Current routing has no competitive cross-token expert capacity/drop rule.

The only lateral decoder-hidden-state edge is local SWA. Its window of 128
includes the current position, so each decoder block adds at most 127 earlier
inputs. For `D` decoder layers and window `W`, the boundary suffix required for
the final logits and every layer's last `W` raw KV inputs is:

```text
B = 1 + D * (W - 1)
default before layer 20: D=20, W=128 -> B=2541
after a COMPLETE layer 20: D=19     -> B=2414
```

These are different cut points. Publishing layer-20 global KV is not completing
layer 20. Neither bound changes the SWA window itself.

For decoder index `j=0..D-1`, prompt endpoint `N`, use:

```text
a[j] = max(0, N - 1 - (D - 1 - j) * (W - 1))
s[j] = max(0, a[j] - (W - 1))

KV-only initialization: [s[j], a[j])
complete block outputs: [a[j], N)
```

The preceding block supplies precisely the next block's required inputs.
Layer 39 needs one complete output and at most 127 KV-only initialization rows;
layer 20 needs at most 2414 outputs plus 127 initialization rows. Every retained
complete output is protected, unlike a rectangular replay's invalid warmup band.
For long prompts the complete decoder work is 24,150 layer-output rows plus
2540 KV-only rows, not `2541 * 20 = 50,820` complete rows.

| Prompt Rows | Full Decoder Outputs | Shrinking Outputs | Bounded-128 Outputs |
| ---: | ---: | ---: | ---: |
| 512 | 10240 | 8955 | 2560 |
| 2048 | 40960 | 23433 | 2560 |
| 8192 | 163840 | 24150 | 2560 |

All policies additionally execute `20*N` encoder outputs and owner publication.
At 2K, shrinking removes about 21% of total complete layer-output rows and
bounded replay about 47%. These are work proxies, not wall-speed predictions:
KV-only work, publication, global selection scans, smaller-batch efficiency,
staging, and scheduling remain. CED does not make the whole model O(1) in context.

The proof is conditional on a structurally compatible schedule and row-local
operators. Valid GGUF schedules can introduce later decoder owners. Such a
layout must fall back before mutation unless separately proved; family support
must not become an artifact/revision/device whitelist. Decoder-side Engram and
extra hidden-state readers are initially unsupported features, not silently
ignored inputs.

## Publication, Visibility, And State

Use five narrowly scoped operations, not a new generic graph scheduler:

1. Execute encoder rows over an absolute range.
2. Publish decoder globals from the boundary pair.
3. Initialize local KV over an absolute range.
4. Execute decoder output rows over an absolute range.
5. Validate and commit the completed prefill.

Replay must explicitly read already-published owner-20 state without appending
or overwriting it. Normal scalar decode must resume ordinary publication at `N`.
Even after physical publication reaches `N`, query `p` may see only `p+1`
ratio-one global rows; ratio-two owners expose completed pairs only. The forced
newest candidate block follows `p`, not the physically last published row.

Selected IDs, candidate masks, local-history validity, and tensor views must
all use the same absolute query-row origin as output ranges shrink. Clear or
rebuild derived query layouts when views change. Current history gathering
assumes an ordinary committed ring; it cannot treat zeroed, absent, or final-N
ring slots as valid earlier history. Supply exact KV initialization or an
explicit truncated floor, according to policy.

The current attention cursor describes one whole-stack traversal; public
`State.position()` is the committed Engram address position. Neither can stand
for encoder progress at `N` and decoder reconstruction at an earlier row.
Use private provisional phases with separate encoder, publication, and decoder
frontiers. Only a completed `Ready` state permits decode or snapshot publication.
Advance Engram/address history once, during encoding, never during replay.

Initially retain synchronous command completion. Cancellation stops further
submissions, drains submitted commands, and poisons/discards the working state.
It does not require rollback copies of immutable existing snapshots. Failure
after mutation must not restart full forward on that same partially changed
state or advertise partial work as a cache hit.

Plan over the logical prompt, not separately over each 2048-row internal chunk.
Continue using bounded submissions for large decoder suffixes. Do not rebuild
the decoder after every encoder chunk and accidentally erase the saving.

## Memory And Initial Eligibility

A current F32 boundary pair costs `4*5120*4 + 4*4 = 81936` bytes per row:
about 199 MiB for 2541 rows, or 10 MiB for 128 rows. Retain a bounded suffix,
not the whole prompt. Price the simultaneous tail, existing scratch, command
lifetimes, CPU inputs, and reserve through normal admission. Insufficient
exact-mode memory is not permission to select approximate replay or shrink
context silently; full-forward fallback needs its own successful admission.

Existing candidate scratch scales as query rows times `ceil(capacity/8)` I32
entries. Bounded submissions do not remove the obligation to retain owner-20
query controls between layer-major passes. The simplest first executor retains
those controls over the largest complete-output suffix (at most 2414 rows),
with explicit absolute origins and tile views. Price that storage, including
any overlapping ordinary scratch: at one-million-token capacity, the current
candidate representation alone is about 1.2 GiB over that suffix. Compact block
IDs or deterministic recomputation from an immutable encoder boundary are
separate measured alternatives, not free assumptions. The engine's current
owner/local caches are F32; the paper's 890-byte packed-global-cache figure is
not this engine's memory ledger.

First executable scope: cold text CLI/benchmark requests, known total prompt
length, final logits, final state, and ordinary scalar continuation. Exclude
nonzero starting positions, prefix reuse, encoder replay, speculative readers,
prompt logprobs, intermediate checkpoints/captures, later decoder globals, and
unsupported decoder Engram. Fall back before mutation for unsupported cases.

Keep `FullForward`, dependency-preserving CED, and bounded replay distinguishable
in execution policy and telemetry. Bounded replay deliberately truncates SWA
at every decoder layer, including 20; its restored state remains approximate
even if snapshot restoration itself is exact. Initially keep those decoder rings
out of prefix reuse. Future namespaces need policy/version, replay bounds,
arithmetic/schedule identity, and approximate-lineage provenance.

## Executable Prior Art

Inspected snapshots, not claims that every backend or configuration is covered:

| Source | Pin / Status | Transferable Result And Limit |
| --- | --- | --- |
| [V4.1 report](https://arxiv.org/html/2609.19969v1) | v1, sections 2.2 and 3.2.2 | Encoder-derived decoder globals and trained approximate SWA128 replay; distinct encoder replay contract. |
| [YOCO](https://arxiv.org/abs/2405.05254) | 2024 paper | Publish shared KV before skipping cross-decoder work. Its decoder lacks V4.1's per-layer SWA obligation. |
| [ds4 shrinking suffix](https://github.com/antirez/ds4/blob/bd66c402070042bf0a79ad6ece8242de4c93680c/ds4.c#L40320) | `bd66c402` | Closest Metal precedent: full publication, per-layer 127-row seeding, shrinking outputs, provisional invalid state; own threshold/storage/arithmetic are not our defaults. |
| [Official DeepInfer compaction](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/yoco.py) | `9bd9cf24` | Preserve all lazy mHC and query-aligned selection state; rebuild derived layouts. Mask/compact parity is same-policy parity, not bounded/full equivalence. |
| [vLLM replay](https://github.com/vllm-project/vllm/blob/19c0e170d257dc654b500ecccb5e6d3f153366bd/vllm/models/deepseek_v41/nvidia/model.py) | `19c0e170`; [58132 merged](https://github.com/vllm-project/vllm/pull/58132) | Current path publishes at 20 then replays 20-39. Wrapper tests do not prove full-model quality; graph/config eligibility matters. |
| [vLLM late exact halo](https://github.com/vllm-project/vllm/blob/dec8e010e05e00dd920aa941e740f8e99176be3b/vllm/models/deepseek_v41/decoder_tail.py) | `dec8e010`; [57281 open](https://github.com/vllm-project/vllm/pull/57281) | Full layer 20 then rectangular 19-layer halo, 2414 aligned to 2560. Different cut, and alignment is not a Metal requirement. |
| [oMLX CED](https://github.com/jundot/omlx/blob/e3bcfb55e35a1ee343b39535e903509d8f64d58d/omlx/patches/deepseek_v41/language.py) | `e3bcfb55`; [3607 merged](https://github.com/jundot/omlx/pull/3607) | Projection-only full-row midpoint, tail decoder; explicit scheduler/config gating and changed logits. Current chunk-local gating is not arbitrary whole-request equivalence. |
| [ATOM replay](https://github.com/ROCm/ATOM/blob/a1ebc608b4d2f0d1f9779854b4f1265e198c4268/atom/models/deepseek_v41/bounded_replay.py) | `a1ebc608` | Final-tail scheduler/backoff, write-from/cached-until bounds, tail KV with last-row final-layer queries. Do not import its distributed machinery or compound encoder replay default. |
| [PipeNetwork model](https://github.com/PipeNetwork/deepseek-v41-mlx/blob/a9567ae7196e596355b6fdaa7de7af4a2c92ba91/deepseek_v41_mlx/model.py) | `a9567ae7` | Full layer loop with shared globals; not decoder skipping in this inspected path. |

Important evidence distinctions:

- [ds4's cited state test](https://github.com/antirez/ds4/blob/bd66c402070042bf0a79ad6ece8242de4c93680c/tests/test_deepseek41_prefill.c#L193)
  toggles decoder deferral, not decoder-suffix execution. Both arms can shrink.
  Its ring/logit/snapshot checks support deferral invariants, not automatically
  equivalence to unpruned full forward. Its carry storage can retain the full
  input, so do not inherit its memory claim as bounded-tail staging.
- [DeepInfer's equivalence driver](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/yoco/yoco_equiv.py)
  compares mask and compact at the same replay window. It validates compaction,
  not the quality cost of truncating decoder history. Its hit cap of `L-134`
  includes speculative backtracking and is not our constant.
- oMLX's small-model tests explicitly allow changed final logits. Its current
  chunk-boundary controls do not establish the older PR's broad one-pass/chunked
  equality claim. Vendor task/performance reports were not rerun and do not
  qualify our quantized GGUF execution.

## Implementation And Acceptance Sequence

1. Freeze the accepted full-forward artifact/arithmetic/input reference. Keep
   the latest production full-forward path as a separate comparison so kernel
   drift does not disappear inside the CED baseline.
2. Implement the five operations and explicit provisional state. Use the
   shrinking plan directly; retain rectangular full/cone replay for diagnosis.
   Running complete layer 20 first is a useful fault-isolation control, not a
   necessary product detour.
3. Prove publication and seeded KV against controlled arithmetic before
   measuring speed. `with_f32_linears` controls dense projections only; a
   NativeF32 witness must also account for attention and FFN operands/dispatch.
   Use an untimed census to prove the actual path.
4. Compare full versus shrinking execution for boundary residual/mix, owner
   KV and keys, selected/candidate rows, valid local-ring contents in **every**
   decoder layer, final logits, and at least 256 teacher-forced scalar steps.
   Repeat under production dispatch; routing changes require investigation,
   not automatic tolerance relaxation.
5. Exercise odd/even pooling, 127/128/129, 2047/2048/2049, 2540/2541/2542,
   and pruning around 16384. Include snapshot restore/growth, cancellation in
   every phase, failed-status publication, and exclusion of non-ready states.
6. Demonstrate native cold 8K benefit with actual executed encoder/output/
   KV-only/publication row counters, full request/first-decode timing, and
   priced peak memory. Then evaluate bounded128 with an independently defined
   truncated policy plus reasoning, retrieval, tool, and repeated-request
   quality tests. Automatic crossover/default policy follows evidence.

The independent GGUF oracle's seven-token REAP case and conditional long
attention replays do not validate this full-model construction. The pinned
1017 source in `scripts/reference/deepseek_v41/sources.json` remains full forward;
preserve its corrected owner-index publication and regression controls. Do not
weaken the frozen `ced=false` pilot contract to make a new adapter appear covered.

## CPU Witness

```sh
uv run --script scripts/reference/deepseek_v41/ced_dependency.py
```

Eight check groups pass: ownership eligibility, small/short/window-one profiles,
rectangular validity bands, adversarial `B-1`, bounded128 and wrong incoming
mix, poisoned future-global visibility, released/late-boundary rings and scalar
continuation, and the shrinking executor with omitted-seed/misaligned-candidate
negatives and row counts. The wrong-candidate control preserves every final
ring yet changes the final output, demonstrating why both must be checked.
Encoder-derived globals are held fixed when omitting decoder history.

This is an asymmetric exact-integer dependency model with query-specific
selection control, not actual mHC or a real checkpoint. It establishes neither
native arithmetic parity, model quality, nor wall-time speed. No optimized CED
execution has been enabled by this design.

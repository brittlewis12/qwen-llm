# Bounded GLM KDA Half Input Candidate

Base: committed `093ea5dd`, branch `perf/glm-kda-half-stage` in the isolated
`qwen-llm-glm-kda-half-stage` worktree. This is one default-off attempt, not
a qualification, performance result, or default promotion. No GPU runs are
part of implementation. The V4 donor's measured conversion is not evidence
of a GLM gain.

## Boundary

`QWEN_GLM53_KDA_HALF_INPUTS=1` is process-cached and opt-in. It copies the
normalized F32 input to F16 once per KDA block only when at least one of
q/k/v, decay-A, gate-A, or beta uses the existing eligible dense TensorOps
route. The actual Q4 checkpoint has six Q8 consumers per KDA block: input
4096, outputs 8192/8192/8192/128/64/128, across 34 KDA blocks.

The resolved Fast/Half mode, actual packet N >= 1024, native dtype geometry,
M5 tensor tier, global rollback, original F32-input pipeline capacity, F16
twin capacity, serial encoder, physical extents, alignment, device identity,
writability and binding disjointness all remain mandatory. Selection freezes
Q8 wide versus Q6 narrow before conversion; dispatch uses that exact twin.
Missing eligible pipelines and bad bindings are errors before copy encoding.
Unsupported capability declines the complete staging attempt before mutation;
all original F32-input routes remain available. With no eligible consumers,
there is no copy. Strict, F32, short-packet and non-M5 paths read original F32.

Both rank expansions retain their original inputs and dispatch, as do KDA
output projection, all other projections, FFN/experts, MLA and decode.
There is no new shader, expanded weight bank, precision policy or V4 helper
dependency. Q8 MLA tail batching remains parked and untouched. Qualified
Q8-wide and Q5-down defaults stay on.

## Ownership And Admission

On the routed integration branch this is the shared
`packed_scratch.normalized_half` F16 buffer, with shape
`[hidden, admitted_prefill_rows]`: 8 MiB at 1024 and 16 MiB at 2048 for
hidden=4096. It has no tile padding; half-input twins mask ragged rows.
The whole backing is checked, not merely the current packet prefix.

The ledger and allocator consume the same option-resolved buffer-spec list.
Live `preflight_session` (full retained model and capacity advice) and
`Glm5NextSession::plan_memory` resolve the tensor tier with the context.
Session construction carries the ledger's frozen options into allocation
after admission, including caller-owned CPU reserve. Device pricing charges
the entire physical allocation separately before allocation. Options do not
depend on the mutable pre-first-token precision or lineage: selecting strict
or F32 later leaves the already-admitted scratch owned but unused.

Device-free ledger constructors and capacity search explicitly assume a
potentially eligible M5 when the opt-in is set. This intentionally overprices
unknown/non-M5 devices under the Apple 16 KiB planning profile rather than
underpricing an M5 session. Context-aware non-M5 and default-off live plans
add no buffer. Capacities below 1024 and configurations without KDA add none.

## A/B Definition

CURRENT means qualified Q8-wide **on** and Q5-down **on**, not the historical
off/off CURRENT in `quality/tensor_m5_transfers.rs`. That file and the old
stock packet are unchanged.

Any future same-weight model bitwise/plain A/B must launch with global
`QWEN_GLM53_KDA_HALF_INPUTS=1`, perform normal CPU preflight and fresh session
construction outside any staging override, and scope only prefill with
`KdaHalfInputsScope::new(false/true)`. Force Q8-wide and Q5-down on in both
arms. Both arms must physically own and price the full buffer, including
16 MiB at R2048 even when baseline does not use it. Do not create/admit the
baseline inside a false override. Census runs stay outside plain timing;
normal strict pinning, stamps and fresh private reports remain required.
Native process flag 0 versus 1 separately measures actual allocation cost.
The upstream test-only review follow-up adds the bounded model smoke described
below; its production code and frozen native `4ace096b` binary are unchanged.
The routed integration branch separately rebuilds both canonical release
artifacts from its final clean commit; see
`GLM-ROUTED-TENSOR-GATE-UP-CANDIDATE.md` for that build protocol.

## Witnesses And Limits

CPU contracts cover selection at 1024/2048 and ragged/short boundaries,
scope nesting/thread isolation/unwind restoration, unique owned buffer
names, full physical extents, exact priced deltas, default-off/non-tensor
footprints, and conservative device-free capacity advice. They acquire no
GPU lease.

Two ignored GPU witnesses live under `tensor_prefill::tests::kda_half_inputs`:

- `m5_kda_half_converter_offsets_guards_and_edge_rounding`: validation on,
  offset/guard/immutability checks, finite edge half rounding, small Q6-narrow
  bitwise/F64 witness, and pre-write decline/binding failures.
- `m5_kda_half_complete_eight_op_fanout_bitwise_and_full_copy_cost`: actual
  six-input Q8 fanout shapes plus two unchanged synthetic Q8 rank expansions,
  R1024/R2048 and one ragged R1025. Launch with global staging flag 1. Both
  timing arms share the same 16 MiB half backing and force qualified baseline
  flags on. Full copy, preflight and eight operations are inside GPU/wall
  timing; census/readback/poisoning/logging are outside continuous ABBA.

These require an actual M5, a release test binary and an explicit production
GPU lease. Validation uses `MTL_DEBUG_LAYER=1`; diagnostic timing may use 0.
The synthetic fanout deliberately reuses weight recipes and GPU outputs;
each of six projections and both expansions receives a separate bounded
bitwise comparison against the current F32-input TensorOps path. Existing
compact tensor fixtures provide F64 sample oracles, never a whole-weight F32
oracle. Admission caps prospective fixture physical buffers plus 160 MiB CPU
allowance plus the existing 512 MiB reserve at 1 GiB. Outputs are streamed
between two reusable GPU targets; mismatch reporting is bounded too.
The review follow-up fixes its aligned Q8 weight prefixes to use `view_bytes`,
not the non-quantized-only `view_subrange` API.

The separate ignored
`quality::tensor_prefill::kda_half_inputs::m5_kda_half_actual_q4_three_document_bitwise_smoke`
reuses the parent frozen documents, session/admission, observer and report
helpers. It requires global staging flag 1, the pinned six-shard Q4 artifact,
and a fresh private `GLM53_KDA_HALF_SMOKE_REPORT`. Both qualified controls are
forced ON throughout; only staging TLS differs during prefill, never during
preflight/session construction or continuation. Every session has capacity
4416, workspace 2048 and the same ledger-owned 16 MiB half backing. In the
routed integration binary, run this KDA-only smoke with global routed gate/up
flag 0; its local TLS override also forces routed gate/up execution OFF. The
shared `normalized_half` buffer/options retain the same KDA-only ownership and
census semantics. The warm 2K census requires 34 added conversions, 204 half-input consumers, unchanged
39 Q5-down calls and otherwise identical dispatch counts, backed by a named
weight/shape audit. Prefixes are 2048/2048/4352, each followed by all 64 common
teacher-forced forward calls. Both complete 65-row trajectories (prefill plus
continuation) are CPU-reserved and compared full-vocabulary, finite and
bitwise after sessions finish. No tools, 22-document expansion, state capture,
timing or performance verdict is added. Reports contain hashes/metadata only,
not raw token IDs, logits, decoded output or private paths. Shard stamps are
checked unchanged after completion.

The only edit to the old quality parent is a new module declaration; removing
that declaration restores its exact prior bytes. The historical transfer
test, stock fixtures and old packet bodies/semantics remain unchanged.

The witnesses are compile-checked, not executed here. Bitwise equivalence,
half edge behavior on hardware, model continuation and any performance
benefit remain unverified. Aliased synthetic outputs are not a model-level
performance claim; the candidate remains default-off regardless of leaf
timing. No GPU, remote operation, merge, deployment or promotion is included.

## CPU Verification

Offline dev and release `cargo check --workspace --tests` pass. All ten
`cpu_kda_half` contracts (the original seven plus three smoke contracts) pass
both with the environment unset and with global staging flag 1; the nine
existing dense tensor CPU policy tests also pass.
Formatting and diff checks pass. No GPU lease is acquired by these checks.
Three additional device-free smoke contracts cover protocol/capture reserves,
exact census deltas and finite full-vocabulary bit comparisons.

The broader `glm5_next::tests::` run is not green: 21 pass, two artifact
witnesses are ignored, and three unchanged IQ1_M binder/coverage expectations
fail (`binder_refuses_structural_drift`,
`binding_rejects_unimplemented_codecs_not_independently_typed_experts`, and
`coverage_matrix_names_remaining_adaptations`). Those tests, their binder,
coverage and admission implementations are byte-identical to the base; this
lane does not repair or weaken them. The ledger regressions in that run pass.

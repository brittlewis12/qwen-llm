# Admitted Request and CED Composition

This composition joins the reviewed request lifecycle, execution capability,
staging ownership, mapping admission and maintained acceptance sources onto
the provisional CED runtime. It does not transfer their historical device
qualification to the combined tree. Fresh M5/model/serving/timing qualification
is parent-owned. The mask-publication experiment is not included.

## One Runtime Geometry

`PrefillGeometry { start, tokens, selected_rows }` is the logical input component.
It retains `chunk_count`, `max_chunk_rows` and absolute half-open `chunk_ranges`.
`RequestPlan::prefill` and `State::last_prefill_report` expose that component.
The execution `PrefillReport` retains `execution`, `fallback`, `counters`,
`memory` and allocation observations, and adds `geometry`, serialized as
`input_geometry`. CED decoder rows, seeds, batches and completion counts remain
separate; input geometry does not assert that Full decoder work ran.

CLI diagnostics use a version-2 typed wrapper. Full `prefill_geometry` retains
the request lane's actual-chunk/range meaning. CED geometry explicitly uses
`input_encoder_chunks`, `input_encoder_chunk_ranges` and
`scope=logical_input_encoder_not_decoder_work`. Run diagnostics retain capacity
and actual execution/fallback. The outer bench schema and interleaved v2
schedule/protocol meaning remain unchanged; these report fields are additive.

Cold metadata and loaded runtime planning use the same state, scratch, CED and
CPU pricing helpers. `ColdPrefillPlan` is the resident/binding/load envelope;
`RequestPlan` is its identity-bound runtime realization, not an independent
resident-size planning stack. A typical paired caller uses:

```rust,ignore
let cold = plan_cold_prefill(ctx, gguf, prompt_rows, capacity,
    &[PrefillExecution::FullForward, PrefillExecution::ShrinkingCed], caller_cpu)?;
let (selected, load_assessment) = cold.select_load(ctx.memory_signals())?;
// Import and bind once, behind the fresh load gate.
let request = forward.plan_request_from_cold(&selected)?;
let mut state = forward.new_state_from_plan(&request)?;
let (logits, report) = forward.prefill_with_execution(
    &mut state, tokens, cancel, PrefillExecution::ShrinkingCed)?;
```

`select_load` returns the selected plan explicitly and clears alternatives.
`assess_load` only assesses its own fixed geometry; it never silently substitutes
another plan. `plan_request_from_cold` checks geometry consistency and reprices
the selected runtime against loaded metadata/tier. The resulting plan retains
same-Forward Arc identity, exact capacity, selected width and CPU reserve.
Allocation re-admits it with fresh signals, without selecting a different width.
`caller_cpu_reserve()` remains the caller's explicit reserve; a separate
`binding_cpu_reserve()` exposes the conservatively recharged cold binding subset.

4096 is a preference where appropriate, not a capacity theorem or a CED
profitability threshold. A typed pressure refusal permits a 2048 candidate
before import/allocation/mutation. No smaller chunk or smaller context is
substituted. CED-only callers may admit a separate Full candidate; paired
benchmarks require supported CED and reject execution fallback. Every actual
execution retains its normal fresh inner gates.

CED plans and arithmetic lineage use the admitted state's selected width,
including the genuine scalar-tail distinction at 2049/4097. Full and CED must
be freshly qualified at the SAME selected width. There is no bitwise claim
between 2048 and 4096. Once a request mutates KV, subsequent prefill calls cannot
reselect/narrow its width. A new compatible live request may select before its
first mutation through `ensure_capacity_for_request`.

## Accounting and Eligibility

Resident Metal views, private CPU shadows, binding controls, state/runtime
buffers and CPU caller reserves remain separate domains. A full cold peak must
fit capacity; fresh load availability admits only the load phase. After binding,
fresh runtime admission applies before every new state. Neither capacity nor
planning is a memory reservation. Shared GPU views do not bind on CPU available
memory. Device/physical/process/pending/private-shadow constraints remain active.

GSQ bounded control normalization, retained normalized controls and separate
backbone/auxiliary source bookkeeping remain charged. Loader summary bookkeeping
is one shared descriptor/window category:
`max(loader_summary, caller_descriptor_window)`, never their sum. The included
subset is explicit as `retained_summary_cpu_bytes` and
`descriptor_window_cpu_bytes`; disjoint control/source/private costs are added
separately. Direct loader entries recheck their local bound before import.
Mapping target stays `None`; product loading is quiet. Bench/current-pair may
emit the existing structured resource inventory line.

Descriptive GGUF planning remains broader than backend execution. Intrinsic
execution is validated before preparing imported mappings or pricing runtime
scratch, including Sinkhorn/top-k and grouped output-A fast-path boundaries.
Canonical grouped 3D serialization, native IQ1/GSC support and wider-group
general fallback are retained. No arithmetic is reimplemented by composition.

## Lifecycle Boundaries

Serving retires incompatible idle live ownership before output admission,
prefers compatible live state, then a usable RAM snapshot, then cold allocation.
Live growth refusal releases old ownership before the next source is attempted.
Ready/Provisional/Poisoned transaction semantics remain. CED-origin live state
and snapshots remain ineligible for serving reuse and checkpoint retention.

Full stays default. CED is an explicit whole-logical-prompt cold operation;
continuation is ordinary scalar execution. Unsupported requests fall back before
mutation only after Full admission. No approximate mode, automatic threshold,
retry/variance/free-floor performance gate or serving CED-origin qualification
is introduced. Two-layer batches, pending/completed separation and logical
fault precedence remain intact.

The capture APIs remain Full qualification, not CED execution. The maintained
suffix ring witness keeps `[127, 513]` and its selected MMA/tensor dispatch census.
Historical reference receipts/source lists are unchanged. Current compiled-source
inventories retain batching and soft-window inventories and append current
planning/contract/lifecycle/helper paths. Exact CPU acceptance discovers all
maintained names and ignore states, then executes only explicitly classified
full CPU names; GPU/model/timing/external-fixture witnesses stay excluded.

The legacy candidate-versus-frozen-reference harness now also uses the common
cold/request plan. It checks admitted geometry against the immutable production
reference width before import, rather than checking the resident preference and
then executing another width. Its candidate report becomes
`v41-ced-qualification-v2`; the original archive schema and frozen source digest
list are unchanged. Restoring causal state clears stale prefill telemetry while
retaining the newly admitted request geometry and the snapshot's causal origin.

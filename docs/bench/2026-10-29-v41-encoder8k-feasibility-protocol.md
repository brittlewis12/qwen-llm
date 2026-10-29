# V41 fixed-8192 encoder feasibility protocol

## Hypothesis and scope

At source baseline `4a54d49e`, compare one explicitly selected 8192-row encoder
chunk with the current two 4096-row chunks for an 8192-token CED prompt. Encoder
submission/completion overlap and boundary handling already exist; this is not a
new overlap mechanism, an expected 2x gain, or a kernel/math/scheduling change.

This first unit contains CPU-only cost, plan, geometry, lineage, and guard tests.
It does not authorize 8192-row execution. Production `MAX_CHUNK_ROWS=4096`,
`DEFAULT_CHUNK_ROWS=2048`, resident-size preference, and standard fallback are
unchanged. No environment selector is added. Test builds also default to 4096.
Only `prefill::with_full_chunk_ceiling_for_test(8192, ...)` enables the wider
planning/geometry ceiling on its calling thread. It validates the fixed ceiling,
nests, restores on panic, and permits a nested 4096 scope. Execution guards and
decoder prefix/replay backing limits deliberately remain unchanged.

## CPU evidence required before execution

Independent read-only adversarial review of the source commit precedes actual
Rust CPU qualification. The new CPU tests must then pass without creating a
Metal context, importing weights, reading an artifact, or executing a kernel:

- Scope rejection, nesting, panic restoration, thread isolation, and default
  behavior outside the scope; existing unsupported-shape/context/overflow gates.
- Exact `CedPlan` work counters, retention, absolute row views, scalar-tail
  identity, and decoder coverage. At N8192 only the encoder chunk width changes;
  decoder chunks and batched replay units remain at most 2048 rows.
- Original-query arithmetic lineage at every row, including compacted
  singletons, partial tails, and nondivisor query tiles. The original 4K and 8K
  chunk boundaries can imply different arithmetic routes: lineage checks do not
  establish numerical equality across chunk widths.
- Full scratch, CPU upload/hash staging, CED incremental costs, and the shared
  Full/CED runtime maximum, using the full request KV capacity (8448 and 16384
  in the pricing fixture), never the chunk length as substitute capacity.
- Explicit half/tensor scratch pricing off/on and explicit existing CED estimate
  pipeline argument false/true. At N8192 the 4K pipeline prices a second live
  chunk and upload staging; one 8K chunk has no second chunk to overlap.
- Per-allocation conservative pricing, monotonicity in capacity and assumed
  heap price, maximum-buffer refusal, checked overflow, and synthetic CPU/Metal
  capacity versus current-availability refusals. No fallback may silently turn
  the fixed 8192 hypothesis into a smaller encoder chunk: exercise
  `planning::select` success and pressure refusal with fixed 8192, supplied
  synthetic signals, and exactly one price callback in either case.
- Direct A/B equality of retained, seed, status, batch selection/diagnostic/CPU,
  encoding reserve, command reserve, and retained-control terms. Derive encoder
  layer counts from the actual `CedPlan`; source inputs and CPU tests, not a
  layer-count narrative, are the authority.

## Pricing interpretation and kill gate

The fixture uses default `Config`, `Schedule`, and `HashConfig`, with assumed
Engram projection geometry. These are model-shaped source inputs, not artifact
authentication or a real-header cold plan. Its assumed heap quote is logical
bytes, alignment 256, and host page 16384, with 1024 extra per nonempty allocation.
The existing pure `price_shared_buffer_upper` performs the actual checked rounding
calculation; these assumed inputs are not a measured device upper bound. A larger
per-allocation allowance must not reduce the estimated demand. Pricing tests
require the existing default 512-row query tile rather than silently repricing an
ambient tile override. At N8192, both chunk widths have 16 such logical query
tiles; the full 8192-by-capacity score matrix is not materialized.
The prior observed 11.202 GiB pipeline increment is not an actual 8K admission
quantity and is not substituted for this still-unexecuted synthetic ledger.

The synthetic maximum-resident boundary is conditional on stated synthetic
device/physical budgets and zero gap/shadow inputs. It is not a model size limit
or load approval. Real cold-load, binding, caller, maximum-buffer, and total
admission quantities must come from the actual header and real plan later.
Do not add an old short-prompt load packet to this runtime fixture and call the
sum an actual 8K cold peak. Admission is not an allocation reservation.

Kill before GPU if reviewed CPU pricing, geometry, lineage, or rejection tests
fail, if conservative real-plan admission is unavailable, or if 8K execution
requires broader unreviewed runtime changes. This unit deliberately leaves
execution guards closed. Later reviewed source changes and checks under the
existing delegation must address these encoder-only guard sites explicitly:
`prefill::run_captured` (and the test-only hybrid reference),
`attention_gpu::ced::valid_chunk` (shared with publication/range validation), and
`attention_gpu::ced::Cursor::new`. Do not globally widen the shared guard: keep
decoder replay/prefix backing at 2048 and the query-tile bound unchanged. Carry
fixed width through cold plan, request plan, state, and admitted plan rather than
overwriting a 4K admission. That propagation is not delivered by this CPU unit.
Completion, cancellation, Engram, decode, and shader code are not changed here.
Actual mixed projection/expert route populations still require header-based
pricing and later validation. Plan/query-lineage geometry is not a proof of all
projection/FFN routes or numerical bits; no uniform expert-dtype assumption is
made here.

No Rust typecheck/CPU result, device admission, numerical proof, model import,
GPU timing, or performance measurement is claimed. Precise fixture cost outputs
are available only after the reviewed Rust CPU run. Existing correctness/F32
tolerances remain unchanged.

## Proposed later gates (not frozen)

Before any timing, freeze the correctness and performance protocol after review.
The concrete performance proposal is >=2% inclusive CED prefill wall improvement
in each of two counterbalanced blocks; both must pass, without filtering,
retries, pooled-block acceptance, or a shape/selector sweep. That is about
87.6 ms at the historical 4.3819 s reference, not a new measurement or prediction.

Correctness first: strict full-vocabulary equality at 257 positions and causal
state across all causal positions through 256 teacher-forced steps, compared
against both current Full and current 4K CED controls. Include work counters,
geometry, status, address history, cancellation/drain, admission, and postflight
checks before timing. A pass against 4K CED alone is not a Full-equivalence
claim. These are proposed later gates, not current CPU-test or GPU/model results.

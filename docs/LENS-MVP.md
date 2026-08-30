# Lens MVP

North star: make real J-lens, R-lens, and template-lens readouts plus exact,
ordered post-block interventions usable from a simple local CLI soon enough to
preserve experiment time.

This is an implementation project. Scientific workflows, experiment design,
UI, generic plugin systems, and production hardening are out of scope.

## Working Capability

- Imported Qwen3.6 matched full J/R and Qwen3.8 full J transports support packed
  layer x position top-k, selected transported vectors, and selected-token live
  directions.
- Native selected-token J/R artifacts support live readout and intervention.
- Ordinary dense Qwen native J/R fitting supports the published T128 sequence
  length for row shards and selected-token artifacts.
- The released Qwen3.6 phrase/template asset supports real cosine readout and
  directions by row or exact label.
- Fixed add, residual-L2-relative add, projection ablation, and directed
  source-to-target displacement share exact layer, prefill, and decode scopes.
- Multiple matching operations execute in plan-file order.
- Raw prompts, literal token IDs, and strict message JSON control rendering.
- Ordinary dense and MoE inference paths run interventions. Muse selected/full
  transport mechanics exist; Flash-Next exposes only raw native hyper control.

## Qualification

- Published Qwen3.8 J readout: `boot` resolves to Italy at L31/L47 and the final
  prompt position resolves to `Euro` at L62 on the deployed Q8 model.
- Published Qwen3.8 J intervention: a no-thinking one-word animal prompt changes
  from `Elephant` to `lightning strike` when the selected `lightning` direction
  is applied over L24-L58 at residual-L2 coefficient `0.1`.
- One-layer steering raises the selected J score without changing behavior;
  the broader intervention changes behavior, distinguishing the two schedules.
- The released introspection prompt reproduces its `ele...` baseline, but a
  user-turn-only coefficient `0.1` does not flip it. Published strength
  sensitivity is not yet claimed as replicated.
- The pinned Qwen3.6 pair imports without pickle execution; both target-62
  anchors are verified as exact F16 identity matrices during import.
- On `The athlete Michael Jordan plays the sport of`, R ranks ` basketball`
  second at the Jordan position on L20 while J omits it from top-8; their L62
  readouts coincide. This is a bounded released-asset parity check.
- A one-site R `basketball` intervention raises its live selected-token score
  from `2.3286` to `6.7954` and records exactly one requested application. The
  baseline already emits `basketball`, so no behavioral sensitivity is claimed.
- The active `qwen-lens` unit suite passes with 97 tests and two model-bound
  tests intentionally ignored.
- Real T128 Qwen3.8 Q8 J and R row fits pass across a hybrid L58-to-L62
  traversal. The R proof takes `7.84s` forward plus `1.67s` VJP at B1.

## Honest Boundaries

- Muse fitting remains capped at T16 by its separate fixed-scratch bank. T16 is
  an implementation smoke lane, not parity with published T128 fitting.
- Published Qwen transports were fitted against BF16 model execution and stored
  as F16. GGUF transfer is explicit: Qwen3.8 late-layer Q8 agreement is strong;
  Qwen3.6 Q4 has bounded local J/R qualification, not broad equivalence.
- `source_to_target` is a directed displacement, not the paper's two-coordinate
  pseudoinverse swap.
- No complete, method-matched local R asset currently exists for Qwen3.8 or
  Muse. Qwen3.6 now has the released T128/skip-4/target-62 matched J/R pair plus
  its separate real template asset.
- Muse full-J/R assembly and application code does not make a T16 fit a
  method-comparable scientific asset.
- Native Qwen3.8 full-R fitting is correct at T128 but not currently economical:
  a measured B8/all-source prompt takes `167.65s` of VJP, projecting the matched
  25-prompt, 5,120-row asset to roughly 31 days.

## Active Gate

The shortest honest full-R path is complete through the released Qwen3.6 pair.
Checkpoint and reintegrate this lane before choosing the next gap. Native
Qwen3.8 full fitting still requires the independent block-operator optimization;
do not launch a full native fit at current throughput.

## Deferred

REST, UI, corpus-scale batching, canonical pseudoinverse coordinate swaps,
Flash-Next lens fitting, Muse T128, and native full-Qwen block-operator
optimization wait until the selected full-R lane requires them.

Keep this file concise and update it in place. It is a capability and scope
ledger, not a work log.

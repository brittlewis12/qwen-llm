# Lens MVP

North star: make real J-lens, R-lens, and template-lens readouts plus exact,
ordered post-block interventions usable from a simple local CLI soon enough to
preserve experiment time.

This is an implementation project. Scientific workflows, experiment design,
UI, generic plugin systems, and production hardening are out of scope.

## Working Capability

- Imported Qwen3.8 full J transports support packed layer x position top-k,
  selected transported vectors, and selected-token live directions.
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
- The active `qwen-lens` unit suite passes with 96 tests and two model-bound
  tests intentionally ignored.
- Real T128 Qwen3.8 Q8 J and R row fits pass across a hybrid L58-to-L62
  traversal. The R proof takes `7.84s` forward plus `1.67s` VJP at B1.

## Honest Boundaries

- Muse fitting remains capped at T16 by its separate fixed-scratch bank. T16 is
  an implementation smoke lane, not parity with published T128 fitting.
- The published Qwen3.8 transport was fitted on BF16. Late-layer Q8 transfer is
  strong in existing comparisons; early-layer transfer remains unresolved.
- `source_to_target` is a directed displacement, not the paper's two-coordinate
  pseudoinverse swap.
- No complete, method-matched local R asset currently exists for Qwen3.8 or
  Muse. The real Qwen3.6 template asset is not an R-lens substitute.
- Muse full-J/R assembly and application code does not make a T16 fit a
  method-comparable scientific asset.
- Native Qwen3.8 full-R fitting is correct at T128 but not currently economical:
  a measured B8/all-source prompt takes `167.65s` of VJP, projecting the matched
  25-prompt, 5,120-row asset to roughly 31 days.

## Active Gate

Choose the shortest honest full-R path. The released Qwen3.6 matched J/R pair
has the required T128/skip-4/target-62 recipe and a matching local model; native
Qwen3.8 full fitting requires a non-incremental block-operator optimization.
Do not launch a full native fit at current throughput.

## Deferred

REST, UI, corpus-scale batching, canonical pseudoinverse coordinate swaps,
Flash-Next lens fitting, Muse T128, and native full-Qwen block-operator
optimization wait until the selected full-R lane requires them.

Keep this file concise and update it in place. It is a capability and scope
ledger, not a work log.

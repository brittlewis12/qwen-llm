# Lens MVP

North star: make real J-lens, R-lens, and template-lens readouts plus exact,
ordered post-block interventions usable from a simple local CLI soon enough to
preserve experiment time.

This is an implementation project. Scientific workflows, experiment design,
UI, generic plugin systems, and production hardening are out of scope.

## Working Capability

- Native plain logit-lens readouts and opt-in lossless full-vocabulary exports
  are available through `read-full`; see [observer controls](LENS-OBSERVER-CONTROLS.md)
  for semantics, examples, and bounded scalar/packed qualification.
- Imported Qwen3.6 matched full J/R and Qwen3.8 full J transports support packed
  layer x position top-k, selected transported vectors, and selected-token live
  directions.
- The pinned eyes-ml Muse Glimmer J transport and independently fitted
  first-25-Pile R transport import without pickle execution, support
  selected-position and layer x position full-vocabulary traces, and supply
  plan-selected live directions and interventions under an explicit
  BF16-to-GGUF transfer gate. Muse traces use the existing offline inspector.
- Packed traces can be persisted as versioned, replaceable result artifacts and
  inspected offline by summary, token/anchor map, aggregate frequency, semantic
  or numeric position, exact token-ID trajectory, and strict paired comparison
  without loading a model.
- Lens run, sweep, and trace workflows share one input contract. `--user` plus
  optional `--system`, or strict `--messages`, use the same release Qwen or Muse
  renderer as normal model runs; model-valid reasoning modes remain explicit.
  Omitted Qwen3.8 mode now resolves to the normal run default, `xhigh`.
- Ordinary Qwen also accepts bounded `--open-responses` request JSON through the
  exact parser, capability gates, and prompt renderer shared with `qwen serve`.
  Developer/system instructions, server-normalized reasoning history, tool
  calls, and tool results remain represented without translating through the
  strict message subset. Qwen3.8 history retains its server preclosure rules.
- Qwen and Muse message traces retain renderer-authored byte spans and exact
  token spans where BPE boundaries permit them; structural selector markers
  remain exact. Open Responses annotates compiled roles, instruction source,
  reasoning, tool-call/result channels, and per-call identity; Muse ATEM adds
  recipients, synthetic system metadata, and EOM/EOT records. Trace and run
  stdout remains JSON by default unless `--output` selects a compact summary;
  either format can be requested explicitly.
- Plan v2 can bind prefill operations and live readouts to exact authored spans
  by kind, message/call index, role, channel, label, occurrence, and start/end
  token edge. Bindings fail on ambiguity or inexact BPE boundaries and resolve
  to the existing numeric execution contract for Qwen, Muse, and Flash-Next.
- Native selected-token J/R artifacts support live readout and intervention.
- Ordinary dense Qwen native J/R fitting supports the published T128 sequence
  length for row shards and selected-token artifacts.
- The released Qwen3.6 phrase/template asset supports real cosine readout and
  directions by row or exact label.
- Fixed add, residual-L2-relative add, projection ablation, canonical
  coordinate swap, and directed source-to-target displacement share exact
  layer, prefill, and decode scopes.
- Multiple matching operations execute in plan-file order.
- Ordinary Qwen coefficient sweeps reuse one resident model and prepared lens,
  while every serial arm gets a fresh sequence and same-seed sampler. Ordered
  duplicate and zero controls are preserved in immutable hashed child artifacts.
- Model-free sweep inspection verifies complete bundle topology, child hashes,
  effective-plan isolation, and shared run context, then groups controls and
  reports exact reference-arm readout differences.
- Raw `--prompt` (`--raw-prompt` alias) bypasses chat templating, while literal
  `--token-ids` bypasses both rendering and tokenization. Both accept deliberate
  malformed or simulated forged structure without normalizing it; semantic span
  selectors correctly fail because these paths make no authored-span claim.
- Run artifacts use schema v4 to bind the authored plan and canonical digest,
  numeric resolved plan, exact semantic position bindings, input source,
  automatic-special-token policy, renderer/mode, role/channel spans, and prompt
  IDs. Sweep v2 embeds and hashes the source plan and treats bindings plus
  rendering metadata as shared arm context.
- Ordinary dense and MoE inference paths run interventions. Muse supports native
  selected and published full-transport directions; Flash-Next exposes only raw
  native hyper control.

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
- A coefficient-1 coordinate swap on Qwen3.6 R reverses the local `basketball`
  versus `Jordan` selected-token ranking at exactly the requested L20/position-3
  site. The unchanged output is recorded without a behavioral claim.
- The active `qwen-lens` unit suite passes 214 tests with four model-bound
  qualification tests intentionally ignored by default.
- Real T128 Qwen3.8 Q8 J and R row fits pass across a hybrid L58-to-L62
  traversal. The R proof takes `7.84s` forward plus `1.67s` VJP at B1.
- The exact 4.52 GB eyes-ml Muse source imports to 51 pinned F16 matrices. On the
  deployed Q8 model, `The capital of France is` ranks ` Paris` first at L50; a
  one-site published direction run records one application and emits ` Paris`.
  This is implementation qualification, not a transfer-equivalence claim.
- A real 6-position x 3-layer Muse trace self-validates as `qwen.lens.trace` v3,
  includes an L50/P5 transported vector, and passes summary, positions,
  position, token-trajectory, and exact-comparison inspector paths.
- The real Muse Q8 tokenizer reconstructs an ATEM system/user prompt exactly;
  BOS, start, message, EOT, and generated-assistant markers all align to exact
  nonempty token ranges.
- The private published Muse R source independently verifies at SHA-256
  `3675c137...068c`, imports as 51 finite F16 matrices with payload BLAKE3
  `31f9cbad...acf8`, and verifies source/target block 50 as exact positive-zero
  F16 identity. Its opaque pickle digest binds the embedded released recipe,
  model/tokenizer revisions, corpus hashes, and estimator/arithmetic contracts.
- On the deployed Muse Q8 GGUF, a 66-position ATEM R trace over L0/L25/L50
  self-validates with two 6,656-coordinate vectors. At the exact final user
  boundary, L25 ranks `_request` first and L50 ranks `Interpret` first. A
  separate literal-token trace accepts a deliberately non-rendered three-token
  sequence and records `renderer=literal_token_ids`.
- A semantic `message_content(role=user,last,end)` R intervention at exactly
  L25/P62 records one application and raises the selected `_request` numerator
  from `14.124642` to `23.320082`. All other captured prefill/decode sites and
  the three generated tokens remain unchanged, so this qualifies projection,
  position binding, and execution without claiming behavioral sensitivity.
- The real Qwen3.8 Q8 tokenizer reconstructs a tool-loop Open Responses prompt
  exactly; role, thinking, tool-call/result, and generated structural markers
  align to exact nonempty token ranges.
- A real Qwen3.8 Q8 published-J `[0,0.1,0]` resident sweep produces byte-identical
  zero controls with no operation applications. The active L31 arm records five
  applications and raises the selected `lightning` score at every prompt site.
- `inspect-sweep` verifies that real bundle with 10,886 total child bytes, one
  byte-identical zero-coefficient group, one generated-output group, and five
  changed active-arm readouts without opening the model or lens.

## Honest Boundaries

- Native Metal Muse fitting remains capped at T16 by its separate fixed-scratch
  bank. T16 is an implementation smoke lane, not parity with published T128
  fitting.
- Published Qwen transports were fitted against BF16 model execution and stored
  as F16. GGUF transfer is explicit: Qwen3.8 late-layer Q8 agreement is strong;
  Qwen3.6 Q4 has bounded local J/R qualification, not broad equivalence.
- `coordinate_swap` is the paper-equivalent two-coordinate exchange;
  `source_to_target` remains a separate directed displacement for compatibility.
- No complete method-matched R asset currently exists for Qwen3.8. Qwen3.6 has
  the released T128/skip-4/target-62 matched J/R pair plus its separate real
  template asset. Muse has a complete first-25-Pile/T128/skip-4/target-50
  matched J/R pair with the same pinned model/dependency stack and corpus
  protocol on one fitter lineage.
- Muse full-J/R assembly and application code does not make a T16 fit a
  method-comparable scientific asset.
- Strict Lens messages match the normal run lane's system/user/assistant subset;
  Open Responses supplies the server's genuine developer/tool grammar for
  ordinary Qwen. The current server intentionally compiles developer and system
  input to one model-visible system role, while artifact labels retain the input
  source. Raw text and literal IDs remain the path for malformed or forged
  structure and make no genuine-channel claim.
- The published eyes-ml Muse J remains a separate 900-Wikitext, target-51
  artifact and is not interchangeable with the matched pair. All published
  Muse profiles were fitted on text-only BF16 execution; Q8 use and image-token
  positions remain explicitly unvalidated transfers. The importer is
  profile-owned and never executes pickle.
- Native Metal Muse fitting remains a small-shape correctness lane. Full T128
  matched production uses the pinned CUDA fitter; the runtime consumes its
  normalized artifacts without coupling inference to that fitting stack.
- Native Qwen3.8 full-R fitting is correct at T128 but not currently economical:
  a measured B8/all-source prompt takes `167.65s` of VJP, projecting the matched
  25-prompt, 5,120-row asset to roughly 31 days.
- The independent Qwen3.8 block-native screen lowers the idealized linear floor
  to `21.46-22.50h`, but GDN traffic alone reaches `23.92-25.19h` before other
  reverse work. It does not provide a credible one-day full-fit path.

## Active Gate

The shortest honest full-R and intervention-parity paths are complete through
the released Qwen3.6 pair, the matched Muse pair, and canonical coordinate swap.
No additional native full-fitting lane remains credible for this deadline. The
current output lane prioritizes artifact-first offline inspection and comparison
over speculative fitting or a resident service.

## Deferred

REST, corpus-scale batching, Flash-Next lens fitting, native Metal Muse T128,
and native full-Qwen block-operator optimization remain post-deadline unless
requirements or available mechanisms materially change. A bounded Neuronpedia
extraction spike found its current J-lens components coupled to application
providers and string/probability data contracts; browser work waits for a
passive, token-ID/score-aware component seam rather than importing auth/database
product infrastructure.

Keep this file concise and update it in place. It is a capability and scope
ledger, not a work log.

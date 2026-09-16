# Settled-Default Parent Ledger: Qualified Observation

One frozen run, five same-token native forwards after one2179-token packed prefix:
ordinary/profiled warmup, ordinary-before/profiled/ordinary-after. Actual defaults
guarded top-k+HC on, incumbent QSA, no split scratch. Production lease/real wired
gate/API validation. Harness PASS7.95s and explicit QUALIFIED_PARENT_OBSERVATION;
these are separate facts because an inconclusive harness would still complete.

All measured full logits/hyper and terminal state agree bitwise. Dispatch sequence
and shapes match, ignoring intended encoder identities; guard48/HC97 and incumbent
QSA12 pertoken, no split kernels. Both profiled96-counter packets are monotonic.
All raw observations, census and timestamps persist before qualification gates.

| Axis | Before ms | Profiled ms | After ms | Control drift | Observer delta |
|---|---:|---:|---:|---:|---:|
| GPU |50.842875|50.735917|51.177375|0.65575%|-0.53756%|
| Executor wall |56.658958|56.785625|56.891042|0.40878%|+0.01871%|

Frozen control<=5%, GPU observer within5%, wall observer within10% all pass.

| Inclusive normalized parent | Count | ms |
|---|---:|---:|
| Complete QSA-containing blocks |12|25.707522|
| Complete GDN-containing blocks |34|20.138984|
| Bootstrap layers0/1 |1|3.795122|
| Final HC/logits |1|1.069374|
| Encoder-boundary remainder |-|0.024915|

These are command-normalized encoder spans, not exclusive leaf costs or a speedup.
The QSA-containing aggregate is largest, but contains HC, all index/QKV/output
work, attention, full routed/shared MoE and residual combines.

## Front-Loading And Scope

Layer3 QSA6.516745ms, layer7=2.429290, layer11=2.280373; the nine later QSA blocks
cluster around1.59-1.63ms. Early GDN blocks are also inflated (layer2=2.301207,
layer4=1.846832), while late GDN blocks are roughly0.44-0.48ms. The warm profile
also front-loads. It is not legitimate to multiply the first QSA cost by12.

Possible causes include command-start frequency/cache behavior, ordered residency
and first kernel-family use; this packet does not identify the cause. CPU encode
is5.413875ms for the sampled command, outside a GPU-only child attribution. This
single position2179 is a four-token index pool-publication boundary (length2180,
545 visible blocks,2048 active IDs), not a context/decode-phase sweep.

## Source Rechart And cx Challenge

`qwen4exp_post_ple_block.rs:1500` shows attention HC read, mixer, output copy/combine,
FFN HC read, full MoE and output copy/combine. `qwen4exp_qsa.rs:4201` shows index
projections/norm/pool publication, score/radix selection/expand IDs, QKV projections,
norm/KV publication, incumbent logits/softmax-value and output projection. The
selector is already cooperative radix4, not another one-thread expert selector.

Fresh follow-up on cx Sol session01a0ab8f-956e-7603-9044-994b9188816c confirms observer
integrity and recommends one matched common-late child ledger, not direct coding.
Choose layer39 (parent1.619040ms, neither bootstrap nor tail), capture native input
AND QSA persistent/publication state, then prove isolated complete-block output and
mutated state bitwise against native before attribution. Use the actual guarded/HC
configuration on any separate replay workspace; do not inherit flags by assumption.

Bounded next groups: attention HC; index preparation through expand IDs; QKV/norm/KV
publication; incumbent logits+softmax/value; output projection/copy/combine; FFN HC;
guarded router/shared gate; expert bodies/accumulation/final combine. Include host
encode spans alongside GPU intervals. Freeze one warmup and ordinary/profiled/
ordinary packet with existing control/observer gates, persist counters before
validation, no timing retry or candidate. Untimed source/dtype/census comparison
with layer3 can explain structural differences without pretending isolation
reproduces command-start behavior.

Weight recoverable work by occurrence: QSA-specific12, common HC/MoE up to48,
rare dtype only its cohort. Require a credible roughly1-1.5ms whole-forward ceiling
before an exact-work candidate. If costs remain distributed, re-rank another
structural lane. If attention dominates, report the bottleneck without reopening
online/split/grouped-head/PV variants under a new label. Tiny activation-copy or
projection fusion is not justified solely because it is visibly exact.

Raw `target/profiles/qwen4exp-default-parent-ledger-24985/`; run log
`target/profiles/2026-09-16-default-parent-ledger.log`. No model, production math,
server state, or remote branch changed for this observation.

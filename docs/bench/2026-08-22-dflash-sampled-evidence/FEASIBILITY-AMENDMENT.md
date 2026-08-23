# DFlash Sampled Feasibility Sequencing Amendment - 2026-08-23

Status: prospective append-only amendment, frozen before L0 fixture preparation,
new calibration/F0 measurement, K0-S acquisition, or verifier implementation. It
grants authority only to prepare and seal L0 and, after a valid L0 seal, execute
the frozen Q4 F0 reduction. It does not authorize V0, V0.5, V1, held-out or
reserve acquisition, K0-S/K0-L execution, E1b, integrated sampled decoding, Q8
feasibility work, serve, product, defaults, or production traffic.

This amendment grants no model, kernel, Metal, sampler, verifier, proposal,
generation, serve, or F0 measurement-harness edit authority. It does authorize
one pure-offline L0 validator/power simulator and its tests under
`scripts/profile/`; that code may read proposed or sealed L0 metadata and
calibration artifacts but may not import model runtime code, initialize Metal,
or execute a model. Any other source or observation-producing instrumentation
change needs a separate prospective authorization committed before the edit.

## Precedence And Scope

This amendment derives authority from the original preregistration and preserves
all restrictions except the explicitly superseded provisions enumerated below.
`NEXT-PHASE-DECISION.md` is a hashed recommendation/problem frame, not an
independent authority grant. This amendment is based on clean source commit
`dee1a00ad392e81114b6efdf2e718a9471a4e9d1` and the following controlling
documents:

| Document | SHA-256 |
| --- | --- |
| `docs/bench/2026-08-22-dflash-sampled-evidence/README.md` | `e90bbf94c1bfc4c3887d855a1fcc4ddd524fb7ca4aafe1431467ddddb983b8f0` |
| `docs/bench/2026-08-22-dflash-sampled-evidence/NEXT-PHASE-DECISION.md` | `2b5f8d5abdf30efcdf938e96dfde6ae348fc6b102531f0711b912532a6c536e2` |
| `docs/bench/2026-08-22-dflash-sampled-evidence/DEV-RESULT.md` | `8e3b91c4a924f2404419e7b0cfb5447eef5fda17a9298e4eb82ca2d4a92d60d8` |
| `docs/bench/2026-08-22-dflash-sampled-evidence/E0-CLARIFICATION.md` | `79b7deedfdae432d50e90e2e94a342c55cb84b5714f9ff9b2f28a0cc592fccbb` |

Prior evidence classifications, correctness requirements, failure records, and
product prohibitions remain controlling. This amendment explicitly supersedes:
(1) the original sequence that put verifier feasibility after G1, (2) the
original Q8-only drafter freeze solely for the fixed Q4-target/Q4-drafter F0
scope, and (3) fixture-custody/L0-seal mechanics with the stricter rules below.
It does not reinterpret prior evidence or weaken correctness, economic,
held-out, product, or serving restrictions.

The prospective sequence is:

```text
L0 seal
  -> F0-Q4
  -> at most one separately authorized F0-selected Q4 target-only V0
  -> separately frozen V0.5-Q4 causal-packet gate
  -> separately frozen V1-Q4 full-verifier gate
  -> separately frozen authoritative Q4 lane acquisition
```

This amendment grants no K0-S acquisition authority. K0-S research and planning
may proceed independently, but execution requires a separate append-only
authorization committed before acquisition that freezes references, fixtures,
commands, reducer, resource budget, attempt IDs, and decision rule. K0-L, E1b,
Q8 F0, and all product/serve work remain closed. Passing one stage never
implicitly authorizes the next.

Reopening a failed or inconclusive stage requires a new append-only amendment,
a named new premise, and fresh calibration clusters that were sealed before the
new execution. Fresh requires a never-executed prompt-bytes/token-hash and seed
pair for every cluster; changing only source/build, attempt ID, timing episode,
or another tuple field is not fresh. All prior calibration and F0 observations
become development-only and are excluded from the new decision; they may not be
pooled with fresh clusters.
The exposure-family/near-duplicate rules also apply to a reopened calibration
frame; punctuation, whitespace, truncation, wrapper, or semantic paraphrase of a
failed frame is not a fresh premise.
Failed outcomes may motivate the premise but cannot alter this amendment's
matrix, fixture roles, candidate set/ranges, thresholds, reducer, or reserved
retries. No artifact or role may be overwritten.

## Prospective Fixture Custody

Archetype names, partition sizes, and historical seed identities may be public.
Exact held-out/reserve bytes and tokens remain unavailable to F0 selection,
V0/V0.5/V1 implementation/repair, and their reviewers. The defense is
prospective identity, independent custody, fixed roles, a complete exposure
ledger, and no selection or repair from held-out/reserve outcomes.

- L0 commits custodian-authenticated opaque fixture identities, byte/token
  hashes/counts, newline policy, tokenizer identity, request length, seed list,
  sampler/config, target/drafter arm, role, and partition. Plaintext
  held-out/reserve fixtures are neither committed nor disclosed before their
  separately authorized acquisition.
- CPU-only tokenization/hashing by the fixture custodian is allowed and recorded.
  The custodian may not perform target, drafter, verifier, proposal, acceptance,
  generation, or performance execution on held-out/reserve fixtures.
- Disclosure or plaintext access by a selector, implementer, or repair reviewer
  contaminates the affected fixture family and requires replacement before seal.
- Any exact or normalized-byte collision with a prior fixture becomes
  `development`, `calibration`, or `contaminated`; it cannot remain held-out or
  reserve. Unresolved provenance is not clean provenance.
- Whitespace, punctuation, template, and wrapper variants of exposed prompts
  and any semantically derived/near-duplicate family are forbidden from held-out
  and reserve, not merely disclosed.
- Opaque held-out/reserve custody paths are absent from every F0/V0/V0.5/V1
  command allowlist.

Disclosure or unauthorized plaintext access at or after sealing invalidates the
affected L0 scope and stops F0 and every later stage. Replacement requires a new
append-only amendment, new fixture identity, and new L0 seal; it is never an
in-place repair.

Roles are disjoint:

| Role | Selects/estimates F0 | Promotes | Stops | Repairs policy |
| --- | ---: | ---: | ---: | ---: |
| development/calibration | yes | no | yes | before seal only |
| held-out | no | yes | yes | no |
| reserve | no | confirm only | yes | no |
| contaminated | no | no | no | no |

Role disjointness applies to exposure families as well as exact files. Reserve
cannot rescue held-out failure, select abstention/drafter/primitive,
change thresholds, or trigger a second optimization candidate. Held-out cannot
change the reserve reducer. Before authoritative acquisition, L0 must document
the interval estimator's operating characteristics and freeze enough
request/seed clusters to support the declared confidence claim; the historical
four seeds are not automatically sufficient.

## Q4-First Portfolio Constraint

The first authority scope is fixed as Q4 target, Q4 drafter, one-hot proposal,
both S1 and S2, and all three context bands: short `1..512`, mid `513..2048`, and
long `>=2049` prompt tokens. All six cells are required. F0 ranks primitives
within this fixed scope; it does not select an easier drafter, sampler, context,
or target after observing data.

The finite L0-CAL sampling frame is also fixed before acquisition and is derived
mechanically from exposed calibration sources rather than selector-authored
prompts:

- short: exact UTF-8 `Write a Python function that parses a GGUF file header.`,
  55 bytes, SHA-256
  `9216f4c46490cd0b16da5995631357ee380c77598c66473995df932366a03b90`;
- mid adversarial-control: first 1024 Qwen3.8 tokenizer tokens of
  `docs/bench/tokenizer-prompts/current-marcus-long-strip.txt` (106,495 source
  bytes, SHA-256
  `3542e4d2c04ed95145042092035dc41ab193f61362fb13539ffa56bfebc375b4`),
  decoded once with no special token and committed as literal bytes;
- mid narrative-guardrail: first 1024 Qwen3.8 tokenizer tokens of
  `docs/bench/tokenizer-prompts/current-reva-short-qwen36-preserve.txt` (37,128
  source bytes, SHA-256
  `7bc9c088be89b8049be4371a35de2f3b3d97e7557b79e2f208bde5a0735cec16`),
  decoded once with no special token and committed as literal bytes;
- long: exact preserve-rendered
  `/Users/tito/code/llm/game/v02_reva.json` under the maintained messages/render
  policy (143,038 source bytes, SHA-256
  `2dac190c9ff86451e07318d7384a124e575d77515f1ef8903befe1c6cd703fb1`),
  using Qwen3.8 target-family rendering with `--messages-preserve-thinking`, an
  appended generation prompt, and no post-render normalization; rendering is
  bound to `crates/qwen-cli/src/messages.rs` SHA-256
  `cdaf1cd919af5dea5d160d82296a73a8eb16e4f0d7fbd125b272639452122432`
  and `crates/qwen-cli/src/bench.rs` SHA-256
  `bfee954fe3579aefc04622d842eaed7790a9f110c0d1332267b68bf330ad242d`,
  then committed as a calibration identity and byte artifact;
- seeds for every fixture: `[42, 314159, 271828, 1618033]`;
- every listed fixture/seed runs under both S1 and S2 in its assigned band;
- requested generation is 128 tokens for short/mid and 64 for long;
- exact prompt/token identity, rendering policy, and execution order are frozen
  by L0-CAL and every planned cluster is mandatory.

This yields four clusters in each short/long cell and eight in each mid cell.
F0 makes a decision only about this finite sealed calibration frame and claims no
population estimate or exchangeability beyond it. Newline/template variants,
unlisted prompts, replacement prompts, and post-result subsampling are forbidden.

Q8 target and the Q8 drafter require separate future scope. Q4 success grants no
Q8 authority and Q4 failure does not kill Q8. The Q8 drafter may be inventoried
but cannot open or rescue the first scope.

Adding the released Q4 drafter is a new prospective scope choice, not a
reinterpretation of the original Q8-only drafter freeze. Acceptance, timing,
memory, and decisions may not be pooled across drafter arms. A target-only
primitive can be shared conceptually, but its affordable budget is the minimum
over the six required cells. Exploratory inventory cannot open or rescue them.

## L0 Preparation Boundary

L0 preparation before a calibration authorization is non-acquisitional. It may
perform offline repository/static evidence inventory, contamination
classification, custodian CPU tokenization/hashing, deterministic power
simulation under a precommitted plan, manifest construction, and pure-offline
validation. It may not execute target/drafter/Metal/timing/thermal/memory work.

Any new target, drafter, Metal, timing, thermal, or memory observation requires a
separately committed `L0-CAL` plan that freezes commands, source/build/binary and
asset identities, calibration fixtures, attempt/replacement IDs, limits,
reducers, and decision rules before acquisition. Before any L0-CAL acquisition,
that plan must also freeze/hash the exact required matrices, confidence
procedures, complete F0 candidate records/ranges/provenance, materiality and VOI
rules, floor method, thresholds, exhaustive data-eligibility registry, and
command allowlists. None may first appear or change after the first observation.

The plan and its first observed result may not appear in the same commit. Offline
power-plan and power-result likewise appear in separate pre-calibration commits;
the power result must pass before L0-CAL. Calibration results and the final L0
seal are committed only in later commits and are append-only.

## L0 Seal

L0 must commit literal fixture files and machine-readable records under `l0/`:

```text
L0-MANIFEST.json
L0-SEAL.json
contamination.jsonl
data-registry.json
matrices.json
estimators.json
power-plan.json
power-result.json
f0-candidates.json
fixtures/calibration/*
fixtures/heldout-identities.json
fixtures/reserve-identities.json
commands.jsonl
builds.json
validators.json
```

`L0-MANIFEST.json` records every subordinate artifact's canonical path, bytes,
and SHA-256. It also binds the commit containing this amendment, amendment blob,
every controlling document, authorized source-tree commit, build command and
binary, model/drafter/tokenizer identities, every executable command,
reducer/validator, and schema version. `L0-SEAL.json` is written last and
authenticates the manifest without self-hashing. The final seal is committed
after all manifested artifacts; an uncommitted or dirty execution surface grants
no F0 authority.

Repository paths must be regular, non-symlink, and pairwise distinct. Custody
receipts authenticate opaque external held-out/reserve identities without
exposing plaintext. JSON rejects duplicate keys and nonfinite numbers. No sealed
artifact may contain placeholders, mutable `latest` paths, unclassified cells,
unhashed inputs, or authority not named here.

The contamination ledger distinguishes prompt-byte exposure, target execution,
drafter execution, policy selection, observed outputs, observed timing, and
unknown/unrecoverable fixture families. At minimum it covers `Hello`, `def`,
`Write code`, sampled development code/narrative/explanation cells, forced-length
timing prompts, maintained code/narrative fixtures, long-rollout fixtures, and
unrecoverable narrative-tail/long-context families.

The seal freezes numeric process/Metal/transient/scratch/artifact/storage limits,
paired timing rules, thermal admission, invalidity rules, exact required cells,
commands/allowlists, exactly one deterministic invalid-pre-observation
replacement ID per planned cell, and the power result. A seal with a missing
fixture, unresolved held-out/reserve exposure-family collision, failed power
rule, missing required cell, or unverifiable hash is invalid and grants no F0
authority.

## F0 Estimator And Costs

F0 pass/fail uses every complete cluster acquired by the sealed L0-CAL plan and
no historical development cluster. Before acquisition, `data-registry.json`
must exhaustively classify every known historical row matching the fixed matrix;
those rows may inform precommitted priors/sensitivity only and cannot enter the
primary bound. A frozen sensitivity grid is diagnostic only and cannot pass F0
or authorize V0. Held-out and reserve rows are forbidden. Each complete
request/seed is one cluster; tokens, blocks, depths, retries, and timing
repetitions within it are correlated.

For every cluster in each required Q4 matrix cell, compute:

```text
b_i = Q_i * T_serial_i / 1.15 - T_non_verifier_i
M_i = b_i - V_floor_i / 0.80
```

`Q`, `T_serial`, and `T_non_verifier` use one common per-packet basis with explicit
milliseconds and emitted-token units. The reducer evaluates both expressions
within each complete cluster. Combining separately favorable marginal estimates
is forbidden.
`T_non_verifier` charges draft generation, hidden capture, q work when
applicable, sampling/correction, checkpoints/state publication, rollback,
continuation, allocation/buffer initialization, copies, command boundaries,
synchronization/readback, and measured memory traffic. Current packed
different-distribution p is only a labeled performance ceiling and cannot be
substituted for exact verifier cost.

`V_floor_i` is a deterministic precommitted physical/dispatch lower bound for a
complete V1 verifier at that exact fixture/context. It is derived before L0-CAL
from immutable asset geometry, exact prompt length, advertised hardware maxima,
and a hashed formula; calibration results cannot change it. F0 claims no
population confidence or exchangeability. It passes the budget gate only when
every mandatory finite-frame cluster has both `b_i > 0` and `M_i > 0`. Report
the minima and every cluster; no median, bootstrap, outlier deletion, or
pseudo-replication can rescue a failing row.

The floor formula enumerates every unavoidable subcomponent's operations, bytes,
state traffic, and dispatch dependencies at the exact packet/context shape. For
each subcomponent it takes the maximum of compute, bandwidth, and mandatory
dispatch floors; it sums serial dependency groups and takes the maximum only for
groups proven concurrently schedulable. It rejects missing geometry, unknown
composition, overlap/double counting, or an unvalidated hardware maximum. The
pure-offline validator independently recomputes the DAG and value from sealed
inputs; any discrepancy makes F0 invalid. Because this is an optimistic physical
lower bound, passing it is only an entry screen and never a speedup projection.

Hardware maxima follow a frozen authority hierarchy: exact detected SoC identity,
then that SoC's Apple-published maximum, then the largest value across all named
vendor specifications only if Apple publishes none. Third-party estimates and
observed benchmark bandwidth/compute cannot replace a vendor maximum in the
floor. All discovered conflicting specifications are registered; the validator
applies the hierarchy mechanically rather than selecting a favorable source.

For reporting only:

```text
B_1.15_cell = min_i(b_i)
M_cell = min_i(M_i)
```

F0 decomposes full-verifier cost/risk across quantized linears/output projection,
attention, GDN/conv recurrence, state/checkpoints, synchronization, and sampler
readback. The complete candidate universe is fixed as:

```text
row-stable-quantized-linear-reuse
causal-attention-frontier
recurrent-gdn-conv-state-publication
state-checkpoint-publication
output-projection-row-bank
synchronization-sampler-readback
```

No candidate may be omitted. Every candidate freezes represented/unsupported
components,
physical floor, memory risk, `p_fail` range, downstream engineering days avoided,
effort-days range, and provenance before new F0 measurements.

For each candidate:

```text
voi_low  = p_fail.low  * cost_avoided_days.low  / effort_days.high
voi_base = p_fail.base * cost_avoided_days.base / effort_days.base
voi_high = p_fail.high * cost_avoided_days.high / effort_days.low
```

L0 validates for every candidate:

```text
0 <= p_fail.low <= p_fail.base <= p_fail.high <= 1
0 <= cost_avoided_days.low <= cost_avoided_days.base <= cost_avoided_days.high
0 < effort_days.low <= effort_days.base <= effort_days.high
```

Missing, nonfinite, provenance-unsupported, or structurally ineligible values
make F0 invalid; they cannot remove a candidate. A candidate is economically
material only under a numeric cost-share/decision-impact rule frozen in L0. It is
selected only if its `voi_low` is strictly above every other candidate's
`voi_high`. Otherwise F0 is `inconclusive` and authorizes no V0. Lexical order,
easiest implementation, output-head preference, point-score ties, and
post-result judgment cannot break a tie.

VOI ranges may be narrowed only by a predeclared mapping from at least two
independent immutable measured/static sources. Expert judgment alone receives
the same maximally uninformative defaults for every candidate: `p_fail=[0,1]`,
`cost_avoided_days=[0,portfolio_cap]`, and
`effort_days=[global_min_effort,portfolio_cap]`. L0-CAL freezes positive
`global_min_effort`, `portfolio_cap`, source-independence rules, units, and range
mapping before acquisition. Candidate-specific unsupported narrowing or changing
range width to manufacture dominance makes F0 invalid.

Before L0-CAL, an exhaustive source-eligibility registry must classify every
known candidate-relevant measurement/static estimate found by the frozen search
procedure. Narrowed endpoints use the conservative envelope across all eligible
independent sources--lowest supported low and highest supported high--rather than
a favorable subset. Omitting an eligible adverse source, double-counting related
sources as independent, or changing eligibility after observation invalidates
F0.

The search procedure is repository-wide over tracked source/docs/manifests plus
every retained raw-artifact root named by those files or the contamination/data
registries. It freezes component/candidate aliases and numeric-field patterns
before execution, records every match and unreadable/missing named root, and
permits no manual match exclusion. A missing source widens to the uninformative
default; it never narrows a range.

The L0 coverage taxonomy is granular and fixed: embedding/input norm, Q/K/V
projections, causal attention/softmax, attention output projection, GDN input
projections, GDN recurrence, GDN output projection, FFN gate/up/down, final norm,
output head, target-hidden capture, KV checkpoint, GDN/conv checkpoint, partial
rollback, allocation/initialization, command/synchronization, full-logit
publication/readback, and sampler plumbing. The coverage map must assign every
subcomponent to at least one candidate's `represented` set or to a named
pre-existing exact mechanism with a frozen cost. An `unsupported` assignment
never satisfies coverage. `Represented` requires the proposed V0 to execute the
subcomponent's actual target dtype, shape, packet width, arithmetic/state order,
and cost premise with material inputs/outputs in its exactness oracle; conceptual
similarity is insufficient. A pre-existing exact mechanism must bind source,
tests, exactness evidence, covered shapes/regimes, and cost provenance by hash.
Overlap is allowed and recorded; a missing, unsupported, unauthenticated, or
overly broad unexpanded assignment invalidates F0.

V0 entry additionally requires every required selected-target cell to satisfy:

```text
B_1.15_cell > 0
M_cell > 0
```

The deterministic `V_floor_i` covers the complete full-V1 physical/dispatch
floor, not merely the selected component's floor. `M_cell > 0` leaves 20% of
each cluster's affordable verifier budget unused relative to that floor. Every
one of the six required cells must pass. F0 failure kills only the fixed
Q4-target/Q4-drafter scope; it does not decide Q8. A result based only on
historical point estimates is `inconclusive`, not a pass.

## Timing, Thermal, And Memory Discipline

- Complete paired units use identical build/assets/input/state/shape and a
  sealed balanced AB/BA order. Pair identity is preserved in reduction.
- Each calibration cluster contains exactly eight admitted timed pairs split
  across two fresh-process timing episodes, four pairs per episode and four AB/
  four BA overall in a sealed counterbalanced order. Both episodes must pass the
  same pre-start admission before either result is viewed. `T_serial_i` is the
  minimum serial wall time and `T_non_verifier_i` is the maximum fully charged
  non-verifier wall time across all eight pairs; this adverse-direction estimator
  is fixed before acquisition. No within-cluster median, episode replacement, or
  selective repetition is allowed.
- Each arm gets one unmeasured post-load/allocation warmup. Per-request
  allocation and initialization remain charged.
- Wall time is primary for economics; GPU time is diagnostic. Lease wait and a
  predeclared cooldown are excluded, while command submission, waits, and
  readback inside an arm are included.
- Pre-start thermal/power admission is frozen from calibration. A miss is an
  invalid pre-observation attempt and is retained. Drift/throttling after a
  material observation is an adverse result, never a selective retry.
- Every Metal-capable command literally uses `QWEN_METAL_LEASE_WAIT=1`. No owner
  is signaled, bypassed, reconfigured, or disturbed.
- OOM, paging, unified-memory or wired-residency pressure, scratch/RSS limit
  breach, timeout after admitted execution begins, mismatch, crash after admitted
  execution begins, and partial output are retained adverse results whether or
  not logits are produced. Whole-model `MTLResidencySet` remains forbidden.

An attempt is invalid only when execution of the measured arm has not begun and
no feasibility-relevant resource demand or material logits/state/output/timing
has been observed: for example an asset/build/hash mismatch, dirty-source
violation, output collision, lease acquisition failure, pre-start thermal miss,
schema/bootstrap rejection, or inability to reserve mandatory evidence storage.
Each planned cell has exactly one reserved replacement ID, consumed only after
such an invalid attempt in deterministic order. Incidents are append-only; no ad
hoc replacement or selective stopping is allowed.

## K0-S Boundary

K0-S planning may freeze a proposed comparison of upstream selector support,
causal predecessor
chain, score formula/comparator, request-temperature normalization, and issue
behavior without proposal RNG or acceptance. It must keep target filters out of
raw q and represent backend slot order/ties/duplicates as backend-specific replay
inputs. Proposal `p_min`/`n_min` abstention is disabled and distinct from target
`min_p`.

K0-S may use only prospectively frozen deterministic predecessor traces or
backend-authenticated replay inputs. It requires its own committed authorization,
fixture/reference identities, comparator, commands, reducer, resources, and
decision rule before acquisition. It may not generate a proposal draw, select a
local RNG policy, consume held-out/reserve fixtures, or consume the V0 budget.
This amendment authorizes neither K0-S acquisition nor K0-L. K0-L's local RNG,
acceptance/correction RNG, abstention, and sampled replay remain unauthorized.

## Terminal Authority

A valid L0 seal grants only Q4 F0. F0 may terminate as `pass`, `failed`,
`inconclusive`, or `invalid_pre_observation`; only `pass` with one robustly
dominant candidate can support a new V0 run plan. Even then, V0 remains closed
until that plan is separately committed and adversarially reviewed.

No stage in this amendment establishes integrated decoder correctness. Any later
product phase must separately compose and differentially verify exact target p,
proposal q, coupling/RNG domains, target/drafter state, rollback, stop, and
continuation. Serve and product generation remain greedy-only.

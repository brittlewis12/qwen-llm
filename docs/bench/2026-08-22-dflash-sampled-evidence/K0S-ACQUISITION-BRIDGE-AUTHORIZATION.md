# DFlash2 K0-S Acquisition Bridge Authorization - 2026-08-23

Status: prospective source-work authorization, frozen after K0-S tooling commit
`430e5fbfea2b0ae3ab620957e22a003efba7cb4c` and before bridge code changes,
real-asset inventory, real-model parity, or K0-S acquisition. It authorizes only
feature-gated inventory/preparation and integrated parity tooling. It grants no
asset-open, model execution, K0-S result, projection parity, proposal RNG,
acceptance/correction, K0-L, E1b, verifier, held-out/reserve, product, serve,
default, or production authority.

The controlling K0-S authorization and both prior amendments remain in force
except where this document explicitly extends the source surface and packet
preparation/parity mechanics. Passing source tests never authorizes inventory or
acquisition; each requires a later separately committed prospective plan.

## Problem And Chosen Architecture

The reviewed K0-S producer requires a complete static manifest before opening an
asset or initializing Metal. Exact tensor offsets/hashes, target/drafter identity,
mask-token metadata, executable/metallib identity, and device identity cannot all
be obtained from repository text. Filling them from the same model run whose
evidence they govern would be circular.

The producer also currently executes one diagnostic-on block. The mandatory
real-model non-perturbation gate requires fresh sessions in both execution orders.
A separate parity run would test a different event and require at least a fifth
draft before the retained evidence row.

The selected architecture is therefore:

```text
prospectively authorized read-only inventory
  -> deterministic inventory-to-manifest seal in a separate control checkout
  -> separately authorized four-arm acquisition
       off-A -> on-A -> on-B -> off-B
  -> retain preregistered on-A capture only if every parity comparison passes
  -> offline reduction
```

This uses the minimum four real first-block arms needed to test both orders.
`on-A` is selected before observation and doubles as the evidence capture; there
is no fifth model run and no choice between successful captures after results.

## Baseline And Source Surface

At clean source commit `430e5fbfea2b0ae3ab620957e22a003efba7cb4c`,
the only authorized edits are:

| Source | Baseline SHA-256 | Authorized purpose |
| --- | --- | --- |
| `crates/qwen-llm/src/metal_dflash.rs` | `108782baf6825755f99d505a6cf76f4f57de9c0689790b120d0dacdc91c77034` | expose feature-gated observed-draft and read-only extraction stages |
| `crates/qwen-cli/src/dflash_k0s.rs` | `7dd35bc62fd9894a17e0c13e8197a49858bcd84b3e0fd09d7f9df7f4b633125d` | add inventory mode and integrated four-arm parity |
| `crates/qwen-cli/src/bench.rs` | `bdda153a02d23482df477ee9eb54c47bedaa9ef36a2413c54f8be4c8c565b761` | add only a hidden feature-gated inventory subcommand and tests |
| `scripts/profile/dflash_k0s.py` | `8d11a4871e47adacfe2a537098998adf72d2adfc5f6811612a109c59131c9967` | validate inventory/parity and deterministically prepare static artifacts |

No Cargo manifest/lock, loader, model, tokenizer, Metal runtime, kernel, sampler,
generation, serve, other script, other source, or other documentation edit is
authorized by this source phase. No dependency or build-script change is allowed.
Every new API and call site remains under the existing non-default
`dflash-k0s-diagnostics` feature.

## Read-Only Inventory Tooling

Add hidden command `qwen-bench dflash-k0s-inventory`. The command may be executed
only by a later committed inventory plan. Source-phase tests use synthetic files
and no real asset.

The inventory command accepts exact target/drafter paths, frozen prompt bytes,
frozen carry token, one exclusive output, and an authenticated inventory-spec
path/hash. Before any asset open it validates canonical paths, source/build
identity, output exclusivity/caps, and literal `QWEN_METAL_LEASE_WAIT=1` if a
device-identity probe is requested.

Inventory may perform only:

- same-file-descriptor fixed-buffer SHA-256 and byte counts for the two GGUFs;
- bounded GGUF metadata/tensor-table parsing and exact hashes of
  `selector_hidden.weight`, `selector_predecessor.weight`, and
  `selector_successor.weight`;
- target vocabulary/tokenizer identity and tokenization of the already frozen
  prompt, plus drafter geometry/mask-token metadata and deterministic noise-input
  hash;
- prebuilt executable, reducer, exact source, scalar-fixture, command template,
  and embedded-metallib identities; and
- a Metal device name/registry/family-capability query with no command buffer,
  pipeline, allocation census, dispatch, model load, or forward.

Inventory must not construct `MetalModel`, `MetalDFlashHead`, or a model session;
encode/commit/wait a command buffer; observe logits, top-K values, `z_t`, score,
draft token, chain, state, timing, memory-performance outcome, or K0-S pass/fail;
invoke Cargo, a compiler, linker, build script, or Metal compiler; or write a
static manifest whose choices were not already in the inventory spec. A later
inventory plan may separately authorize one exact offline build command in a
fresh exclusive build root with enumerated paths and byte caps. Inventory only
authenticates its resulting files.

The exclusive-create artifact schema is `qwen.dflash_k0s_inventory` v1. It binds
the spec hash, source/build checkout, all observed identity facts, exact parser
caps, command/environment, and authority
`development_k0s_inventory_only_no_model_forward_or_semantic_authority`. It is
invalid on an unexpected asset, tensor, tokenizer, geometry, host, build, or
source identity; no replacement is implicit.

## Deterministic Preparation And Seal

Extend the existing pure-offline reducer with an explicit preparation mode. It
consumes only a committed inventory artifact and committed preparation spec. It
must not initialize Metal, import runtime code, open a model beyond independently
rehashing inventory-bound files, or infer any semantic outcome.

The preparation spec freezes before inventory:

- run/attempt IDs, Q4 target/drafter paths and expected file hashes;
- literal prompt bytes and known token expectation, carry token, request
  temperature, two distinct ignored target-policy variants, and fixed slot-index
  chains;
- source/build execution worktree X, control checkout Y, all input/output paths,
  caps, environment allowlist, and command ordering;
- selector-dispatch structural predicate, arm order, selected evidence arm,
  parity comparisons, reducer command, and failure/partial-artifact rules.

Preparation deterministically emits, with exclusive creation:

1. the six-case compiled scalar fixture;
2. the command manifest containing literal `${MANIFEST_SHA256}` exactly once;
3. the static K0-S manifest; and
4. a seal authenticating the inventory, spec, three emitted files, frozen
   transformation-code identity, execution worktree-X commit, and the
   pre-preparation control-Y input commit/tree.

The seal must not contain or claim the hash of a commit that contains the seal.
The later run plan authenticates the descendant control-Y commit containing the
prepared artifacts and seal.

The static manifest contains only pre-run facts. It cannot contain expected draft
tokens, logits, candidate values, `z_t`, complete dispatch census, kernel counts,
state/event/capture hashes, score lattice, production chain, sidecar hash, trace
hash, or parity outcome.

To avoid source/executable self-reference, build and acquisition run from an
immutable clean worktree X at the bridge-tooling commit. Inventory, preparation,
manifest, command, fixture, seal, and the later run plan are committed from
separate control checkout Y. Advancing Y never changes X. All manifest source and
executable paths point to X, and runtime build/source validation runs with X as
the working directory.

## Structural Dispatch Contract

The two ignored target-policy variants are reducer-only transformations over one
captured selector input. They authorize no extra target/drafter forward, arm,
dispatch, or acquisition observation.

The pre-run manifest must not predict a dynamic encoder ordinal or complete
dispatch census. Replace the static full-row expectation with a structural
selector-dispatch predicate containing only source-derived fields:

- exact tag and kernel name;
- weight/input/output dtype and fixed N/H/R geometry;
- grid and threadgroup geometry; and
- exact Metal source/metallib/build identities and allowed dispatch environment.

Each observed arm still records its complete ordered census, encoder ordinals,
concurrency flags, and kernel counters. The producer and reducer require exactly
one tagged selector dispatch satisfying the predicate and exact census/counter
equality across all four arms. Any unexpected dispatch fails; it cannot update
the predicate in place.

## Staged Library Contract

Expose feature-gated immutable observation and extraction APIs while keeping
existing `draft_block_with_k0s_diagnostic` as their exact composition. The
observation is a private-field, move-only token bound to one decoder session and
event; extraction consumes it exactly once and rejects stale, mismatched, or
reused observations:

```text
draft_block_with_k0s_observation
  = observer guard + exactly one unchanged production draft + synchronized
    selector-input identities + complete DFlash state identity

extract_k0s_observation
  = read-only post-sync extraction from that observation and unchanged session
```

The observation owns draft tokens, complete dispatch rows/counters, exact
full-logit/top-K/unary/`z_t` counts and little-endian hashes, and diagnostic-state
identity. Extraction may run only once against the same session/event and must
recheck the observation binding before returning a capture. Neither API samples,
generates a stream, changes target/drafter state, adds GPU work, or exposes a
default/product call site.

## Integrated Real-Model Parity Contract

The later acquisition command creates four fresh, equal-capacity target and
drafter sessions from one loaded target/head in this exact order:

```text
off-A, on-A, on-B, off-B
```

The loaded target/head must be immutable after construction, with no
session-independent semantic cache or lazy model work. No unrecorded warm-up,
probe forward, draft, or dispatch is allowed. Process-global observer/counter
baselines are captured and restored per arm. If source review and tests cannot
establish semantic immutability/reset, each order pair must execute in a fresh
process under a newly frozen plan.

Every arm independently performs the same fixed prompt prefill and target-hidden
append. Each off arm invokes the unchanged production first-block entrypoint
exactly once and invokes neither `draft_block_with_k0s_observation` nor
`extract_k0s_observation`. Existing dispatch census/kernel tracing and a common
read-only parity-summary helper may run identically after completion in all arms;
that helper performs no K0-S extraction. Each on arm invokes the exact composed
observation-plus-extraction path once. `on-A` is the only eligible evidence
capture. The live `on-B` session/capture object is discarded only after its
complete canonical content projection has been serialized into authenticated
bounded artifact material for independent reduction.

Each on arm has distinct arm/session/event provenance and a distinct
domain-separated event-envelope digest. Complete event-envelope hashes must not
be equal. Parity instead requires equality of a schema-defined canonical capture
content projection that excludes only an exhaustive fixed allowlist of
arm/session/event provenance fields. Every excluded field is separately
validated. Both complete canonical projections remain available through
reduction, and the reducer independently replays both before any evidentiary
artifact may be discarded.

After the first block, every arm performs the same deterministic continuation:

1. process the frozen carry through target `single_token_with_multi_hidden`;
2. append its exact hidden column to DFlash context;
3. use the separately preregistered literal continuation-draft carry token from
   the preparation spec; it cannot depend on any observed logit, token, state,
   timing, or parity result; and
4. run one observed DFlash continuation block with extraction disabled.

Target argmax may be recorded as an equality-only digest, but it cannot select a
later input or acquisition action.

Parity requires exact equality within each order and across orders for:

- prompt and continuation target logits/hidden identities;
- first and continuation draft tokens;
- full-logit/top-K/unary/`z_t` identities and counts;
- complete target `SessionSnapshot` contents and DFlash diagnostic-state hashes;
- complete ordered dispatch census, encoder ordinals, kernel counters, and
  observer baseline restoration;
- continuation token/output and final target/DFlash state; and
- `on-A`/`on-B` canonical content-projection SHA-256 and independently replayed
  content, while requiring their separately validated event-envelope digests to
  remain distinct.

No sampler, proposal, acceptance, correction, or target-sampling RNG API may be
reachable. The manifest enumerates every tracked request/session RNG domain and
the exact precomputed noise-input bytes/hash; every tracked state/draw counter
must remain bitwise unchanged. The packet claims zero use only for those
enumerated domains, not universal process randomness. Any arm error, extraction
error, observer leak, mismatch, unexpected dispatch, custody drift, or incomplete
continuation fails the gate and prohibits K0-S evidence publication.

## Packet And Failure Artifacts

Successful evidence remains exactly 16 `qwen.dflash_k0s_lattice` v1 JSONL records
and one bounded sidecar. Add a required `diagnostic_nonperturbation_parity` object
to the run payload; do not add a seventeenth success record. It contains the
fixed order, four bounded arm summaries, comparison matrix, selected `on-A`
identity, enumerated zero-RNG-domain attestation, and `status=passed`. The reducer
independently checks all equalities before lattice reduction.

The run object and sidecar also retain `on-B`'s complete canonical projection:
full draft-logit rows, candidate IDs/unary bits, `z_t` bits, draft/chain tokens,
state and dispatch identities, and exact references needed to reconstruct all
97-by-16 rows from the independently opened A/B tensors. It excludes only the
fixed arm/session/event provenance fields named by the projection contract. The
reducer independently verifies `on-B` support, scores, issues, chain, q, and
projection digest, then compares it with independently reduced `on-A`. No
producer-supplied equality boolean or digest alone can pass parity. This added
material remains within the existing 64 MiB sidecar and 128 MiB combined caps.

This pass means only equality of enumerated production observables under
diagnostic off/on execution. It establishes no target-p correctness, target-model
correctness, selector-projection parity, cross-backend parity, or authority beyond
K0-S conditional on authenticated synchronized `z_t`. Every producer/reducer
output retains exactly:

```text
development_k0s_conditional_on_authenticated_z_only_no_projection_parity_no_rng_acceptance_k0l_e1b_verifier_product_authority
```

For a handled parity failure, write and sync exactly one `parity_failure` record
to the already reserved trace and leave the sidecar empty. The record retains
every completed arm summary, failed arm/stage, first mismatch, observer cleanup
state, and available identities. An abrupt termination may instead leave a
reserved authenticated partial artifact and is terminal. Success records and
sidecar bytes remain buffered until all four arms and parity checks pass.

Semantic observation begins with the first successful return of any real-model
logits, hidden state, draft token, selector value, or model-state snapshot from
any arm. A predeclared invalid-preobservation replacement may be consumed only
when failure occurs before any target/drafter forward, dispatch, model output,
model state, or feasibility-relevant observation. Once any arm begins model
execution, every parity, extraction, continuation, observer, custody,
serialization, or abrupt failure is a terminal observed attempt under that plan.
Preserve it without retry or substitution. Any later execution requires incident
review and a new prospective plan with a distinct attempt identity; it cannot
replace, erase, or rescue the failed attempt.

No output is overwritten, deleted, renamed over, repaired, reused, or silently
retried. Trace and sidecar remain at most 64 MiB each and 128 MiB combined. Arm
summaries contain digests/counts, not repeated full logits or state arenas.

## Required Source Tests

Before any inventory plan, all existing gates remain required, plus:

- synthetic observation then extraction equals the existing composed wrapper;
- extraction rejects stale/session-mismatched/already-consumed observations and
  performs no GPU/state/RNG work;
- pure four-arm mock parity passes both order comparisons and deterministically
  selects `on-A`; every field mutation and arm/extraction/continuation failure
  produces one retained failure record and no sidecar payload;
- inventory parser/hash/tensor/tokenizer/spec/source/device logic over synthetic
  bounded files, proving no model/runtime construction or dispatch call;
- deterministic preparation byte-for-byte repeatability, placeholder-cycle
  freedom, static/dynamic field separation, worktree-X/control-Y binding, and
  exclusive output behavior;
- reducer rejects absent/failed/mutated parity, wrong selected arm, unequal
  `on-B`, missing/aliased/mutated `on-B` projection material, unexpected dispatch
  predicate, inventory/seal mutation, and every forbidden
  RNG/acceptance/projection field; and
- default-feature compile/link/source absence and unchanged product/serve paths.

The committed synthetic Metal parity command remains mandatory and every
Metal-capable test command literally sets `QWEN_METAL_LEASE_WAIT=1`.

## Stage Exit

This authorization terminates after bridge tooling is committed, adversarially
reviewed, and all source/synthetic gates pass. It does not authorize opening the
Q4 assets. A later inventory plan must freeze the exact preparation spec,
commands, output, attempt/replacement rule, and inventory decision before any
asset open. After inventory and deterministic seal are committed, another
prospective acquisition plan must freeze the four-arm command and reducer before
any real-model execution.

No bridge, inventory, parity, or K0-S outcome opens K0-L, E1b, verifier,
economics, serving, or product work.

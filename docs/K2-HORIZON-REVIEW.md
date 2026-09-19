# K2 Horizon adversarial review

Date: 2026-09-18. Base: `4d8716ab`. No engine changes, builds, model loads, GPU
execution, downloads of weights, or server operations occurred in this review.

## Reproduction

Initial command in the K2 worktree:

```sh
cx ask -m luna -e high --sandbox read-only --text \
  --prompt-file docs/K2-HORIZON-REVIEW-REQUEST.md
```

Session: `01a0b4e8-1183-7780-98fd-4a2038bf2504`. The initial call hit the CLI
tool's 120-second limit; a read-only resume with a longer timeout completed the
review. The latest response and complete review conversation remain available
through:

```sh
cx show-last --all 01a0b4e8-1183-7780-98fd-4a2038bf2504
cx transcript 01a0b4e8-1183-7780-98fd-4a2038bf2504 --assistant-only --include-commentary
```

Initial verdict: **revise before implementation**. No user decisions were
identified as blocking. The following is a disposition summary, not a verbatim
transcript or a claim of reviewer approval for implementation correctness.

## Findings and disposition

| Finding | Disposition |
| --- | --- |
| Ordinary loader/runtime encode Qwen hybrid assumptions | Accepted; explicitly use a family-local K2 binder/runtime, sharing primitives only |
| Optimized attention is not already qualified for K2 | Accepted; split M2a short-context correctness from M2b bounded-scratch long-context/packed attention |
| Existing full-lens imports bind Qwen geometry or fitting schemas | Accepted; specify forward-only, dynamically validated imported assets, no local fit dependency |
| Ordinary packed eligibility is Qwen-specific | Accepted; K2 packed implementation/qualification is explicit, not a flag enabling the Qwen path |
| RoPE bases and coordinates cannot inherit final/Qwen defaults | Already required; strengthen nonzero-base packed fixtures and stage coverage from M0 |
| Cache/admission/snapshot identity needs explicit layout and precision | Accepted; separate declared context, execution limit, request capacity, and physical allocation |
| Generic serving inherits Qwen ChatML | Accepted and extended: override both rendering and default output/tool protocol |
| Native BPE reuse is not tokenizer conformance | Already required; HF-independent fixtures and checkpoint-specific identities remain mandatory |
| Packed/serial bitwise equality is not generally appropriate | Accepted; declare numerical contracts and calibrated tolerances, not a global loose tolerance |
| MoVA and advanced KV can expand scope prematurely | Accepted; dense product utility is the MoVA gate; advanced KV remains separate measured research |

Local spot checks confirmed ordinary attention's head-dim256 restriction,
Muse's optimized packed Q32/KV2/H128 restriction, fit-token admission in
`lens_run/lenses.rs`, and Qwen defaults in `serve/http.rs`. Head-dim128 code
does exist in Muse: the gap is K2's complete geometry/layout/qualification, not
the absence of every reusable 128-wide algorithm.

The review's suggested first packet spanned profile, full CPU reference,
tokenizer, lens import, and family registration. That is too broad for the first
implementation checkpoint. Its suggested HTTP token-ID coverage also exceeds
the present serving requirement. Neither is adopted as an immediate obligation.

## First bounded implementation packet

Deliver only a stage-aware model-free K2 7B profile/binder foundation. Do not
make K2 selectable as runnable, touch Metal execution, update FFI dependencies,
change request schemas, add fitting, or begin MoVA.

Proposed touched files:

- `crates/qwen-llm/src/k2_horizon.rs`: validated 7B geometry and positional
  metadata, structural GGUF tensor-role binding, checked capacity estimates,
  and clear separation of model validity from execution capability.
- `crates/qwen-llm/src/lib.rs`: expose the foundation module only.
- Small synthetic fixtures/tests following the repository's existing GGUF test
  construction conventions. Pin small external config/reference artifacts if
  included; do not call hand-authored data independent checkpoint evidence.
- `docs/K2-HORIZON-PLAN.md`: record actual completed scope and outstanding gates.

Targeted CPU-only acceptance cases:

1. Pretrain/midtrain/final positional profiles preserve their own context/theta.
2. Raw-valid metadata does not require a chat template or final release name.
3. Incompatible geometry, unsupported optional features, missing/duplicate tensor
   roles, bad shapes/storage alignment, and arithmetic overflow fail explicitly.
4. Untied embedding/output roles and final grouped norm are structurally bound.
5. F16 KV is exactly 147456 bytes/token; proposed Q8_0 sizing is 78336 bytes/token
   with each K/V row independently block-aligned. Neither calculation claims an
   implemented Q8 backend or includes unstated allocator reserve.
6. Checkpoint-declared context does not turn into a runnable-capacity claim.
7. Existing family detection remains unchanged; no accidental Qwen fallback.

The next bounded packets are the native tokenizer conformance implementation
and independent scalar math fixtures, which can proceed independently once
their source pins are established. Lens manifest design starts early, but its
implementation does not need to share the first profile patch.

Live numerical and performance evidence remains a later coordinated step.
No model-free test can establish final/earlier checkpoint output quality or
full-context throughput.

## Closure review

The reviewer reread the revised plan and this disposition, returning:
**proceed with the first packet**. No blocking user decisions remain.

Three nonblocking clarifications are accepted:

- The first packet is a subset of M0, not completion of its CPU-math fixtures.
  RoPE numerical and grouped-normalization fixtures stay in the next packet.
- The binder is an explicit library API/test target only. Runtime family
  discovery and model selection remain unavailable until safe dispatch exists.
- Profile types must distinguish validated configuration, execution support,
  and missing required facts. Optional provenance can remain unknown; required
  execution metadata cannot default silently. Unknown is not interchangeable
  with known-but-unsupported.

This is approval to begin a bounded foundation patch, not correctness or
performance approval for an implemented K2 runtime.

## Packet 1 pre-commit review

The same session inspected `k2_horizon.rs`, its tests, and the module export.
Verdict: **proceed; commit-ready**, no blockers. Eight CPU-only tests passed.
After the download completed and its SHA256 was verified, the explicit ignored
CPU/header-only test also passed against the real Q8 GGUF. No model inference
or GPU execution occurred.

Nonblocking follow-ups: add a direct adversarial synthetic test of the public
backed-range path; split tensor error categories if callers need structured
classification; do not describe this packet as completing M0's scalar math.
The existing GGUF reader already rejects out-of-file and overlapping payloads,
and the new public binder rechecks bound ranges with `try_slice`.

## Packet 2 pre-commit review

Native tokenizer and independent HF fixture generation were reviewed in the
same session. Initial and closure verdicts: **commit-ready**, no blockers.
Follow-ups were addressed before commit: strict paired-separator metadata,
cumulative normalized-input byte accounting, and explicit metadata-only download
documentation. The HF single-sequence postprocessor inserts BOS but no trailing
EOS; the GGUF paired SEP flag must not change that behavior.

Validation: five regular K2 tests (including 256 Unicode fuzz cases), two
explicit ignored CPU-only conformance tests (both stage vocabularies and the
real Q8 tokenizer), and three existing Qwen/JoyAI splitter regressions pass.
Expected IDs come from pinned `tokenizers==0.22.2` with independently hashed HF
tokenizer artifacts, not from the new Rust encoder. No model forward or GPU
execution is involved. Lens span/identity integration is not claimed.

## Packet 3 design and pre-commit review

The same session recommended a composed tiny CPU reference over isolated math
helpers. The design review required an explicit prefill/continuation contract,
rounded cache reads, all-or-nothing commits, and pinned readout conventions.
The implementation review returned **proceed; no commit blocker** after reading
the Rust reference/tests, independent NumPy generator, and generated fixtures.
No GPU, model loads, or server operations occurred during review or validation.

The review confirmed output-major matrices, split-half RoPE, contiguous GQA,
causal masking/current-token inclusion, and full-append rollback. Two nonblocking
notes were addressed: documentation calls corrupted NumPy equations fixture
distinguishability controls rather than Rust mutation tests; tests explicitly
assert all six expected positional/storage cases. An additional end-to-end F16
cache-overflow test checks rollback after earlier staged rows/layers succeeded.

Seven targeted CPU tests pass. This packet supplies synthetic equation evidence,
not checkpoint parity, an executable GGUF path, GPU ABI qualification, actual
F16 cache memory savings, or runtime admission.

## Packet 4 design and pre-commit review

The reviewer endorsed a host-only bridge using existing primitives rather than
adding a grouped kernel prematurely: four direct-gamma RMS slices, fixed full
NEOX, and canonical F16 cache spans for the generic short-attention candidate.
The implementation review returned **proceed; no blocker**. Six CPU-only tests
pass, including exact scratch-formula comparison to the existing helper; no
Metal device or GPU execution is involved.

Nonblocking integration requirements remain explicit: caller-supplied committed
prefixes do not establish residency or freshness; raw logical ranges do not
validate physical dtype/aliasing/base allocation; arithmetic bounded by fixed
7B geometry and capacity must be revisited before any geometry generalization.
The source ceiling of 7168 is deliberately not described as a numerically
qualified backend limit. Structural theta admission remains positive, while the
candidate paired-RoPE wrapper requires theta greater than one.

## Packet 5 pre-commit review

Primitive wiring review returned **no blocker for this unqualified wiring
packet**. Three host-only descriptor tests pass; both explicitly ignored GPU
probes compile but remain unexecuted. No GPU device is created by the host tests.
One compile-only issue (missing ObjC queue trait import in the ignored test) was
fixed before the successful build.

Nonblocking feedback: partial encoding requires whole-command-buffer discard;
serial encoders do not prove dependencies across commands or session identity;
unqualified public helpers must not be mistaken for model support. The ignored
attention probe now covers zero/nonzero arena base offsets and outer guards.
Source inspection confirms the selected norm/RoPE/store/attention variants use
scalar float/half loads, matching the current element-alignment checks; future
vectorized kernels must revalidate alignment. Numerical thresholds and descriptor
behavior on a live device remain a separately authorized validation gate.

## Authorized synthetic GPU follow-up

After the user explicitly selected the two synthetic probes, inspection found
that unit-test contexts isolate their leases and bypass the wired-memory gate.
Both probes now explicitly retain `acquire_metal_benchmark_lease()` from before
context creation until all GPU resources drop. No foreign process was stopped.

Executed serially, with Metal API validation enabled:

```sh
MTL_DEBUG_LAYER=1 cargo test -p qwen-llm --lib \
  k2_horizon_metal::tests::gpu_grouped_norm_and_full_neox_match_scalar_formulas \
  -- --ignored --exact --nocapture --test-threads=1
MTL_DEBUG_LAYER=1 cargo test -p qwen-llm --lib \
  k2_horizon_metal::tests::gpu_f16_store_and_gqa4_attention_exclude_poisoned_future_rows \
  -- --ignored --exact --nocapture --test-threads=1
```

Both pass (0.10 s and 0.07 s harness elapsed); no validation errors. This is
primitive wiring evidence at the tested inputs only, not full-model numerical
qualification or a performance claim. Broader GPU work remains unauthorized.

## Packet 6 design and pre-commit review

The design jam identified two important boundaries: matrix dtype support is not
embedding gather support, and finite checks require concrete synchronous command
ownership/readback. Both are explicit in the family-local runtime. It borrows the
exact model/context, owns native read-only backings, uses one reusable logits row,
and poisons abandoned submitted appends via a CPU-tested RAII ledger.

Initial implementation verdict: **revise**. Two findings were fixed before
commit: source stamps are now rechecked after session allocation, and every
session buffer is explicitly validated as bounded, writable, zero-offset shared
storage before any CPU upload/readback can occur. Additional host tests exercise
those bounds. A single-session permit makes the queue policy explicit and is
released only when the session drops; loaded models cannot be shared across
threads through a Sync interface. Allocation reconciliation is documented as an
upper-bound check, not exact physical residency evidence.

Validation: eleven regular CPU tests and the explicit actual-Q8 header/residency
planning test pass. The full-checkpoint GPU smoke test compiles but remains
ignored and unexecuted; the prior two-probe authorization does not cover it.

Closure review: **no remaining commit blocker**. The reviewer confirmed both
fixes, the single-session permit lifetime, conservative ledger transitions, and
the explicit qualification/identity/accounting limitations. Packet is commit-ready.

## Packet 7 independent checkpoint oracle

The user superseded the earlier synthetic-only permission boundary with explicit
direction to continue using the lease as intended and not ask per-test questions.
The full-model smoke test passes. An independent IFM-fork comparison passes 42
full-vocabulary rows, including a 32-token continuation, with all top-1 IDs equal
and max error 0.002706051. Detailed scope and reproduction are in the plan and
`scripts/reference/k2/README.md`; there is no HF or general-context claim.

Initial review requested three fail-closed evidence fixes: include untracked
files in source cleanliness, bind executable identity to the wrapper/CMake source
digests, and machine-check actual runtime backend/cache/context log records. All
three are implemented. The final provenance-complete 42-row rerun passes with
unchanged numerical thresholds and the pinned full-artifact SHA256.

Two harness-only attempts are retained transparently: a debug-build whole-file
SHA256 preparation exceeded the 300-second tool timeout (no surviving test
process remained); hashing now uses the native system SHA256 tool. A subsequent
attempt rejected a non-UTF-8 byte in the reference vocabulary log dump before
native execution. Original log bytes are now preserved while ASCII runtime
markers are parsed via lossy text decoding. No numerical gate was loosened.
The reference harness initially used a removed `use_mmap` field; compilation was
fixed to use the pinned API's `load_mode = LLAMA_LOAD_MODE_MMAP`.

Closure verdict: **no remaining commit blocker**. The reviewer confirmed source
cleanliness, bound wrapper identity, runtime-record checks, lease lifetime,
provenance, and the unchanged scoped regression gates. Ready to commit.

## Packet 8 raw CLI design

The design review required strict family-local early dispatch, no Qwen stats or
template inheritance, explicit 32-forward budgeting, and a clear BOS owner.
Implemented native special insertion without heuristic deduplication; a modern
raw-only `--no-special-tokens` option provides the explicit serialized-input path.
Explicit token-budget presence is tracked through clap rather than inferred from
its default value. An existing normalization test was updated to compare like
explicit budgets instead of assuming token presence was untracked.

Registration exposed a real integration hazard: the serving gate admitted every
recognized family via `is_some()`. It now lists only implemented backends and
tests K2 refusal. New enum arms reject K2 speculation, concurrency, template
fallback, and unimplemented fitting/lens paths rather than defaulting to Qwen.
Five K2 host tests and eighteen related normalization/drafter/serve regressions
pass, all CLI binaries typecheck, and the actual raw Q8 generation smoke passes
under the production lease with Metal API validation. No shared server operation.

Pre-commit review found remaining family arms in the queue-overlap diagnostic;
those now reject K2 before Metal creation and have exhaustive unsupported arms.
The generic stop reader could accept additional EOT IDs, so K2 now validates its
resolved stop set as exactly EOS 1. A host test rejects extra/invalid sets. The
explicit-BOS/no-special-tokens CLI rerun matches both input count and generated
token fingerprint of the automatic-BOS run, without changing tokenizer behavior.

Closure verdict: **no remaining commit blocker** for the scoped research raw-run
lane. The reviewer confirmed the diagnostic guards, exact EOS policy, early
dispatch, BOS ownership, budgeting, stats identity, and unimplemented-lane gates.

## Packet 9 forward-only capture/readout core

The design jam endorsed reusing the block graph and final readout, with explicit
post-FFN site labels, final-position-only captures, finite checks before commit,
optional priced storage, and a distinct one-command/zero-advance transaction.
The public capture API rejects empty sites; ordinary append uses the private
no-capture path. The arena is per-call rather than retained lazily, avoiding
stale/unselected rows entirely. No caller-owned GPU storage is accepted.

Implementation review found no immediate blocker. Two hardening suggestions were
applied: explicit allocated-byte equality and another source freshness check
after capture allocation. GPU testing now passes under the production lease and
API validation, including all/sparse sites, zero/final readout, exact cache bytes,
split-prefill and plain continuation bitwise equivalence. The independent 42-row
checkpoint oracle rerun has unchanged metrics. Forty-one regular K2 CPU tests
pass; CLI binaries typecheck. A stale packet-8 family registration assertion was
fixed when the broader suite exposed it.

The reviewer also suggested actual post-submit fault/source-mutation injection.
No source file mutation or artificial Metal fault was performed on the shared
checkpoint. Host ledger tests cover abandoned submitted work before/after checks
and zero advance; this is not evidence of live device-fault recovery. Empty-site
public captures are intentionally invalid, not an alternate ordinary-append API.

Closure verdict: **no remaining concrete commit blocker; commit ready**. The
reviewer verified the byte/freshness hardening, zero-advance transaction, explicit
empty-site policy, and scoped evidence. Imported assets, interventions, fitting,
CLI wiring, and broader qualification remain outside this packet.

## Packet 10 ordered forward interventions

Design feedback emphasized exact kernel signs/formulas, final-token-only scope,
current-layer KV already being committed to the staged arena, stable same-site
order, and private priced host-to-GPU vectors rather than public Metal buffers.
All four existing operations are reused without normalization or a second graph.
Nondecreasing layer order is validated, never silently sorted. Capture occurs
after all same-site operations. Empty operations allocate no vector arena.

The implementation review found **no concrete blocker; commit-ready within
scope**. It verified vector bounds/lifetimes/row separation, command ordering,
no-op behavior, and conservative transaction semantics. Diagnostic wording in
the shared checked allocator was clarified to cover all forward hooks. A suggested
extra host descriptor-row pairing test is nonblocking; the live SourceToTarget
equation check exercises actual source/target arena translation.

The leased/API-validated Q8 intervention probe passes all four independent scalar
equations, noncommuting same-site order, capture timing, no-op equality, final-site
KV/continuation identity, early-site causal KV changes, split reproducibility,
retryable host rejection, and intentional finite-input GPU overflow poisoning.
Forty-two regular K2 tests pass and CLI binaries typecheck. No claim of actual
device-fault injection or intermediate per-operation nonfinite detection; no
imports, checkpoint transfer identity, fitting, or CLI intervention support yet.

## Packet 11 native plain-lens CLI

The design review endorsed a native adapter for existing plain-lens JSON/bundle
infrastructure, not a separate output format. It required early unsupported-mode
gates, rejecting the meaningless transfer override, explicit complete-input versus
executed-prefix counts, requested versus runtime site order, and reuse of the
existing tokenizer metadata identity with architecture/implementation named.
Source review confirmed all tokenizer.* inputs, including paired-separator keys,
are covered by that lightweight identity; it is not cryptographic authentication.

Pre-commit review found a real cross-family compatibility defect: adding count
fields to every input object changes the existing `input_blake3` contract without
versioning. Counts now apply only to the new K2 output, and a regression verifies
exact unchanged Qwen/Muse input metadata and digests. Central family dispatch now
uses `ModelFamily::detect`; explicit requested/runtime site orders are recorded.
A capabilities test was updated from the previous all-lens-unsupported expectation
to the precise partial plain-lens lane.

Eight plain-lens host tests and five K2 run/capability tests pass. Actual Q8 CLI
smokes pass under production leases/API validation, including default 36-site
output, serialized-BOS/no-special-token equality, literal IDs and preserved suffix,
unsorted requested output order, raw vectors, binary bundle row/top-1 agreement,
and pre-Metal rejection of fitted mode, transfer override, and excessive position.
The opt-in uv script records commands, stdout/stderr, and bundle evidence; it uses
serial children holding their own production leases, never a conflicting outer
lease. No additional model download or shared-server operation.

Research found `linear_transport::VerifiedTransport` already supplies the strict
data-only dynamic-geometry reader needed next. The reviewer agreed to reuse it
instead of inventing a parallel asset schema; native observations remain separate.

Closure verdict: **commit ready; no remaining correctness blocker**. The reviewer
verified K2-only digest changes and the CLI checker assertions. Its suggested
rerun ergonomics change is intentionally not overwrite behavior: the checker
requires a new evidence directory, now explicit in its help, preserving prior
results and immutable bundle publication.

## Packet 12 typed raw linear readout

The design reviewer rejected an untyped public byte-slice seam in favor of one
validated owned `K2LinearF16`: exact 4096x4096 finite F16, row-major target/source,
no orientation toggle, unchanged bytes, no asset trust policy. Implemented that
type and one shared zero-advance readout executor, reusing existing session scratch
and the exact final norm/head. The matrix is copied into checked/admitted private
GPU storage; source freshness and both residual/logit finite checks gate commit.

Implementation review found **no concrete source blocker**, conditional on the
prior capture/plain-readout GPU regression. Both the new linear probe and that
regression now pass under production leases/API validation. Evidence covers exact
identity, nonsymmetric orientation, cache/prefix/continuation isolation, retryable
host errors, and numerical overflow poisoning. Forty-three regular K2 CPU tests
pass and CLI binaries typecheck. The type remains non-Clone; the upload destination
is privately writable, not caller-owned storage. No imported-asset binding or CLI
transport capability is claimed by this packet.

## Packet 13 imported data-only readout CLI

The design review required removing the former fitted-readout rejection while
retaining all other K2 unsupported-mode gates, early K2 dispatch with no fallback,
pre-payload profile checks, mandatory exact binding even with overrides, and
algorithm-correct digest provenance. Source rereading confirmed the shared reader
already computes payload BLAKE3 separately from its SHA256 integrity fields; both
are recorded without changing or relabelling an algorithm.

The optional expected-profile API preserves existing reader behavior and checks
K2 geometry/target before payload access. Imported execution reuses the same K2
prepared input/capture path, rehashes one selected matrix at a time, and delegates
to the typed linear readout. Actual retained checkpoint bytes are hashed before
Metal. Unbound transfer requires an explicit recorded override; exact mismatch
cannot be bypassed. Producer claims and verification observations remain distinct.

Pre-commit source verdict: **no concrete commit blocker**, conditional on the
synthetic CLI checker. An initial 300-second debug-host run timed out after its
first successful cases; process inspection found no surviving children. Without
changing numerical gates or GPU code, the rerun optimized only host BLAKE3/SHA256
dependencies and completed all cases. Identity asset full-logit bytes match plain
readout; explicit transfer, source ordering, nonsymmetric equations, non-overridable
binding errors, wrong target, and payload digest rejection pass. The shared plain
CLI regression subsequently passes all its cases too. Existing CPU mutation tests
exercise selected rehashing; the expected-profile test deletes the payload and
still observes the earlier profile error. Nine plain-lens tests, ten transport-
filter tests, five K2 run/capability tests, and all CLI binary checks pass.

No real fitted asset, local fitting, CLI intervention plan, or research-quality
transport qualification is included. Asset verification means data integrity and
the recorded binding/transfer policy, not validation of producer scientific claims.

Closure verdict: **commit ready; no concrete blocker remains**. The reviewer
confirmed both completed checker results, host-only optimization scope, timeout
transparency, and the absence of fitting/checkpoint-quality/serving claims.

## Packet 14 bounded request benchmark

The design review endorsed a dedicated `k2-request` command and small accounting
loop, with exact `P+T-1` capacity, native raw input, EOS-only policy, fresh sessions,
and no reuse of Qwen/Muse model or rendering assumptions. Separate clocks and
actual transition counts prevent the first sampled token being counted as a
decode forward. Zero-transition rates are null; inconsistent outcomes suppress
aggregate rates, not evidence. Bench build-identity policy remains unchanged.

Pre-commit review requested removal of a hard-coded final-Q8 qualification label
and explicit head dtype/geometry/capacity metadata. Those fixes are implemented:
the runtime scope is unqualified short-context guarded request-wall measurement,
with actual checkpoint geometry/precision recorded independently. No kernel-only,
llama-bench, steady-state, or performance claim is made even after warmup.

Six host tests pass (including mock EOS/budget/cancellation accounting) and the
five K2 run/capability tests pass. An explicit clap conflict fixed a `requires`
interaction that otherwise admitted no-special-token flags with literal IDs.
The actual-Q8 instrumented checker passes warmup/timed token equality, raw-run
fingerprint, serialized BOS, literal IDs, zero-forward null rates, exact committed
prefix/phase accounting, and invalid request/family rejection before Metal.

The first checker attempt used the raw i32le digest as if it were the request-
stats fingerprint. Inspection established the latter's domain/count prefix; the
checker now computes both correctly. No runtime outputs, algorithm, or numerical
threshold changed. Successful evidence is `target/profiles/k2-request-bench-check-v2`.

Closure verdict: **no remaining concrete commit blocker; commit-ready**. The
reviewer confirmed the corrected qualification scope, head/capacity metadata,
completed leased checker, and hash-contract distinction.

## Packet 15 raw serving

The design review favored a family-owned parse hook over global raw-input modes.
K2 now retains explicit raw input and BOS policy, rejects unsupported field presence
including nulls, normalizes accepted sampling controls, and owns both prompt and
output protocols. A RawText partition uses only incremental UTF-8 assembly, never
Qwen/Muse marker grammar. Non-K2 default parsing remains unchanged except explicit
rejection of the K2-only extension namespace.

Two design suggestions were deliberately refined: the request output limit may
use the explicitly configured startup default, and listener binding stays after
K2 CPU admission but before weight loading to fail cheaply on busy addresses.
Prefill uses genuine per-token appends/ticks, not fictional cancellation points
around a single multi-token call. Existing HTTP JSON duplicate-key behavior is
documented; strict field whitelisting is not claimed to change that decoder.

Source review found **no concrete blocker before GPU validation**. Borrowed model
lifetimes remain on the accept-loop thread with no unsafe self-reference/leak or
Send bound. Exact capacity and EOS contracts precede session allocation, and every
abort discards fresh per-request KV. The actual-Q8 test now passes production-lease/
API-validated direct backend and ephemeral loopback JSON/SSE checks, raw-run parity,
serialized/automatic BOS, optional stats, and next-request identity after aborts
before prefill, during prefill, and at early/later generation boundaries. The CLI
test links a non-test library, so its context takes the production lease itself;
no conflicting outer lease was taken.

Shared HTTP/parser/rendering/startup/CLI regressions pass. Additional tests cover
raw marker/UTF-8 chunking, EOS-first/late delegation to the canonical generator,
explicit startup limits, and actual-header refusal before socket/Metal creation.
No shared server was restarted/stopped, and no separate long-running server was
started. No claim of sustained-service, chat/tool, snapshot, or long-context support.

Closure verdict: **no remaining concrete commit blocker; commit-ready**. The
reviewer confirmed the family boundary, listener/lease ordering, explicit raw/BOS
policy, completed JSON/SSE and abort checks, and correctly limited documentation.

## Packet 16 serving documentation consolidation

The user questioned the separate K2 serving manual. The raw family contract still
needs an explicit boundary from Qwen/DeepSeek/Muse chat, but not a separate manual.
It now lives in `SERVE.md#k2-horizon-raw-profile`, with test reproduction in the
reference README and development evidence here/in the plan. The old generic wire
section anchor is preserved; that section explicitly excludes the K2 raw profile.

Read-only adversarial verdict: **no documentation commit blocker**. Contract,
limitations, lifetime rules, BOS controls, evidence scope, and reproduction were
retained; stale links were removed. No runtime behavior changed.

## Packet 17 longer-context qualification diagnostics

The 256-token extension intentionally retains the 42-row regression bounds. It
streams full-vocabulary oracle rows, retains 12 native split checkpoints, checks
whole/split/singleton equality and capacity+1 refusal, and records all failures.
CPU protocol tests cover coordinates, exact extents, finite values, trailing data,
and unread rows. Production lease, real wired gate, pinned Q8 identity, serial
reference children, and Metal API validation remain mandatory for executed probes.

Results, not a promotion:

- `target/profiles/k2-oracle-256-79492-1789774227874737000`: **234/768 rows fail**
  existing numeric bounds. All 768 top-1 IDs agree; all native append partitions
  agree bitwise. Maximum logit errors: ledger 0.03625, code 0.08722, Unicode 0.13724.
- `target/profiles/k2-layer-diagnostic-80387-1789775028263169000`: traced IFM logits
  equal untraced logits bitwise. First-block residual errors are already nonzero
  (about 7.6e-6, 1.14e-5, 1.24e-4); they grow through later blocks. This does not
  isolate attention, RoPE, normalization, projections, or F16 rounding as a cause.
- `target/profiles/k2-oracle-256-81900-1789775133807348000`: temporary reciprocal-
  power RoPE algebra fails 185/768 rows and worsens worst-case ledger/code error.
  The experiment was reverted; no production shader change is retained.
- `target/profiles/k2-oracle-256-83475-1789775397971942000`: identical ledger IDs at
  bases 0/37/8191 fail 252/768 rows, with all top-1 IDs equal and native split controls
  bitwise. First max-error failures occur at lengths 60/60/58, respectively. Earlier
  cross-corpus onset differences cannot be attributed solely to absolute position.
- `target/profiles/k2-oracle-83411-1789775337538090000`: the original 42-row test
  passes after reverting the arithmetic experiment and rebuilding the wrapper.

Adversarial review found no harness-safety blocker and requested capture hashes;
the diagnostic now records those separately alongside its full binary identity.
One earlier review claim about a zero-byte EOF read was rejected: Rust `[0u8]`
contains one byte. Its explicit `[0u8; 1]` spelling changes no behavior.

The next discriminating experiment is same-input attention replay against the IFM
intermediates and an independent F64 computation, not more blind algebra changes.
No public cap promotion or tolerance change: run/bench/lens/serve remain at 32.
High-position probes are not long-history evidence. Diagnostic infrastructure
passing is explicitly not successful 256-token qualification.

## Packet 18 same-input attention replay

The reference callback now captures layers 0 and 20's last post-RoPE query,
every visible post-RoPE key/projected value row, and last pre-output-projection
attention output. The independent replay deliberately bypasses native projections
and RoPE: both the native kernel and an F64 host calculation receive the same IFM
query and F16-rounded captured K/V. Captures bind exact shapes, extents, tokens,
positions, layer order, finite values, and SHA256. Native replay prices/reconciles
all buffers, validates CPU layout, and completes synchronously under the parent
production lease before host reads. Two CPU tests cover parser corruption and
independent softmax stability/GQA mapping.

Evidence `target/profiles/k2-layer-diagnostic-84458-1789776080546622000`:

- All ordinary/reference-traced logits still agree bitwise for the three prefixes.
- Across six operations, max native-vs-F64 error is 4.18e-6, IFM-vs-F64 7.75e-7,
  and native-vs-IFM 4.89e-6. F64 uses the graph's F32 scale and rounds final output
  to F32; these are sampled arithmetic diagnostics, not full-model tolerances.
- Layer 0 cached K/V bit differences are 17/14 of 61440 ledger elements, 28/6 of
  34816 code elements, and 971/5 of 27648 Unicode elements. Layer 20 differs in
  thousands of elements. Small incoming numerical differences and F16 rounding
  are consistent with amplification, but a unique root cause is not established.

Adversarial review endorsed that narrow conclusion and requested explicit full
tensor shapes, now enforced (Q `[128,32,1,1]`, K/V `[128,8,1,1]`, IFM kqv
`[128,1,32,1]`). Its offset concern was already covered: `validate_cpu_layout`
rejects every nonzero offset before the unsafe read. The comment now makes this
precondition explicit. No production kernel, public cap, or numeric gate changes.

Closure verdict: **no remaining concrete checkpoint blocker; commit-ready**.
The exact-shape callback diagnostic and original 42-row regression both pass with
the final rebuilt wrapper; all six CPU oracle tests pass. No claim is made that
these checks resolve the failed 256-token qualification.

## Packet 19 bounded-scratch H128/GQA4 attention candidate

Design review favored a separate small K2 online-softmax shader over refactoring
Muse's proven G16 path. The K2 entry assigns one 32-lane SIMDgroup per query head,
maps `kv_head = query_head / 4`, widens F16 K/V to F32, and retains fixed-size state
without a context-sized score buffer. Every visible cached position participates;
there is no eviction, truncation, or KV-format change. The runtime continues to
select the existing materialized primitive.

The checked wrapper reuses extracted attention-view validation and additionally
requires float4/half4 offset alignment. Shape, dtype, serial ordering, aliasing,
write access, physical extents, and token visibility retain their prior contracts.
The existing typed plan excludes zero history and bounds extent at its source
ceiling 7168; a suggested new 256 primitive cap was not adopted. Testing position
257 is not a public-cap promotion or qualification through 7168.

All four host-view tests and three K2 GPU primitive tests pass. The new synthetic
probe covers lengths 1/32/33/128/256/257, all 32 heads, nonuniform KV-head values,
nonzero arena offset, NaN guards/future/other-layer storage, cache bit immutability,
and flat/sharp-score variants. Predeclared max-error bounds remain 2e-5 against
both independent F64 and materialized controls; observed maxima are below 1.75e-6
and 1.67e-6 respectively. Allocations are priced/reconciled; the candidate adds no
score scratch allocation. Production lease, real wired gate, and API validation
are used throughout.

`target/profiles/k2-layer-diagnostic-87341-1789776927937272000` replays both kernels
on the six captured IFM operations. Candidate max error versus F64 and IFM is
below 1.67e-6. Native full-model execution still uses the old path; the original
42-row checkpoint regression passes after the shared host-validation extraction.
No speed, packed-prefill, Q8 KV, whole-model parity, or long-context claim is made.

Adversarial source verdict: **no concrete blocker; safe to commit as an unqualified
K2 primitive**. The reviewer confirmed recurrence, visibility, alignment, alias
checks, scratch accounting, unchanged runtime/Muse paths, and scoped evidence.

### Whole-model online experiment (not promoted)

A subsequent temporary import substitution routed the unchanged serial graph
through the candidate. New manifests bind native runtime/primitive source hashes
as well as metallib hashes so host-only dispatch experiments are distinguishable.
The original 42-row oracle passes, but
`target/profiles/k2-oracle-256-89612-1789777299496754000` fails **215/768 rows** under
the unchanged extended gates. All top-1 IDs and native append-partition controls
still agree. Max errors are 0.03633/0.09371/0.12305 for ledger/code/Unicode.
Mixed gains versus the materialized control do not justify promoting the candidate.
The temporary dispatch change was reverted exactly; only the evidence-hash additions
remain. Resolving whole-model numerical qualification still precedes public-cap
promotion. This also confirms that better sampled primitive accuracy alone does
not resolve the closed-loop F16-cache/model divergence.

## Packet 20 cache/backend precision control

The external IFM wrapper adds an explicitly diagnostic `--f32-kv` mode, mutually
exclusive with capture mode. Default execution remains F16/flash-off. Its identity
advertises both modes; K2REF001 cache bits and exact runtime logs independently
bind actual F16/F32 K/V, 256 allocated cells, and 36/72 MiB respectively. Mode
mismatches fail closed. No native cache, dispatch, or public capability changes.

The three-way diagnostic streams two reference rows plus one native row, retaining
no full logit sequence in memory. All reference children complete serially before
native model/context creation under the production lease and real wired gate.
Disk was checked (33 GiB free) before writing roughly 1.5 GiB of evidence.
Stable F64 log-softmax metrics report KL(reference || actual), half-L1 total
variation, centered RMSE, and mean logit shift alongside raw error metrics.
Tests cover cache-mode binding, runtime-log drift, distribution shift invariance,
KL direction, hand-computed probabilities, sharp logits, and nonfinite rejection.

Evidence: `target/profiles/k2-cache-precision-92281-1789784511128729000`.
All three pairings retain exact top-1 agreement on every one of 768 teacher-forced
rows. Native-F16 versus IFM-F16 max total variation is 0.00039643 and max KL is
6.84e-6. IFM-F16 versus IFM-F32 reaches 0.00158392 and 1.131e-5 respectively;
its worst raw-logit difference is 0.44402, larger than native/IFM F16's 0.13725.
Native/IFM F32 probability differences are comparable to the reference's own
cache-mode sensitivity. This does not establish a unique rounding cause: changing
cache dtype also changes backend kernels/reductions, and neither mode has full-
precision weights or serves as an HF ground truth.

Read-only review found no implementation/safety blocker. It endorsed probability
impact as an additional diagnostic, not justification to quietly relax old gates.
The original 42-row strict regression and callback-neutrality/replay diagnostics
pass with the rebuilt wrapper; nine CPU oracle tests pass. Historical extended
failure and the public 32-token cap remain unchanged. A proposed wider acceptance
envelope is a post-hoc engineering hypothesis that still requires frozen,
nonrepeating holdout fixtures and separately recorded continuation/lens evidence.

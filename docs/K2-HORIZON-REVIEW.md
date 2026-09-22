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

## Packet 21 candidate holdout freeze (before target execution)

`scripts/reference/k2/HOLDOUT-POLICY.md` defines a candidate empirical guarded-256
experiment, not independently derived tolerances or automatic promotion. Four new
assistant-authored synthetic sources cover prose, code, multilingual editing, and
ambiguous instructions. Selection occurred after calibration but without target
forward feedback; this is not a randomly sampled or independently audited corpus.
The first 256 native IDs are tested at bases 0/37/8191. Per-row probability/raw
gates, 15 split boundaries, four residual sites, and four 241+15 argmax trajectories
are fixed before execution. EOS is data in those mathematical trajectories;
serving's EOS-stop contract remains separately tested.

Freeze review required stronger tokenizer/artifact binding and exact semantic
policy assertions. The preflight now hashes the actual single GGUF shard between
stamp checks, verifies tokenizer metadata identity, exact full token counts
(349/440/349/371), and SHA256 of each 256-ID prefix. CPU tests assert every gate
object, continuation setting, site list, and boundary, plus the entire policy hash.
Only CPU hashing/tokenization has run at this checkpoint; no holdout forward pass.

Frozen JSON SHA256: `65a517ad27cab60a7a38989ce8cf3499902940fb94f82c26d83dcca86f5f6889`.
Closure verdict: **freeze commit ready; no remaining blocker**. Failed fixtures
must not be replaced and thresholds must not be tuned after observing results.
Original strict 42-row regression, historical extended failures, and public cap 32
remain unchanged. Holdout execution and any surface promotion are separate reviews.

## Packet 22 first frozen holdout execution (failed)

The reviewed runner executes the unchanged v1 freeze at `990bb56d`. The reference
wrapper adds an exclusive `--greedy-15` mode with at most 256 total rows and a
coordinate/prefix-bound token sidecar. Highest-ID ties and EOS-inclusive fixed
mathematical trajectories match the frozen policy; native per-row top-1 agreement
is required. Twelve ordinary, four traced, and four trajectory reference children
finish before native residency. All capture-enabled reference logits are checked
bitwise against ordinary execution. The pre-execution review found no blocker.

Evidence: `target/profiles/k2-holdout-v1-96305-1789786294833100000`.
The first execution **fails v1**: three failing records among 4096 row checks and
16 capture-site checks. They are two distinct top-1 disagreements, with one also
appearing in a trajectory's supplied prefix. All numerical/capture gates pass;
all twelve singleton/split/whole controls are bitwise; all 60 generated trajectory
steps agree exactly. Capacity+1 remains retryable. API validation and production
lease/wired-memory gate were active; disk was checked at 30 GiB before about 5 GiB
of reference evidence. Elapsed test time was 730 seconds, not a performance result.

- `field_notes`, base 8191, length 223: native selects reference's second-ranked
  ID; the reference logit gap is 0.000043869 and probability gap 0.000018228.
- `multilingual_editor`, base 8191, length 174: reference second-ranked choice,
  logit gap 0.000116349 and probability gap 0.000001890. The duplicated record is
  before the generated tail, not a second free-generation divergence.
- Overall maximum KL is 1.768e-6, TV 0.00076142, raw error 0.10353, RMSE 0.02044,
  centered RMSE 0.01437; minimum logit cosine is 0.99998707.
- Captured residual relative L2 is at most 0.00023726; minimum cosine 0.999999977.

The read-only analyzer verifies the retained reference row before reporting these
gaps; native maximum/candidate logits were not retained, so it cannot establish
the reciprocal native-side ranking margin. The original strict 42-row regression
passes with the new wrapper, and twelve CPU oracle tests pass.

Adversarial post-result review confirms this is a genuine policy failure, not a
harness failure. No threshold, fixture, runtime, or public-cap change follows.
A possible v2 ranking-indeterminate rule needs new frozen fixtures, explicit
two-sided regret evidence, and unchanged exact generated-tail requirements. It
cannot reinterpret v1 as passed. Public capacity remains 32.

## Packet 23 v2 ranking-indeterminate freeze

V2 keeps all v1 numerical/capture controls but explicitly bounds two-sided logit
regret below 0.001 on non-generative rows. This is a within-distribution probability
ratio bound, not an absolute probability-error bound. Generated predictor lengths
241-256 remain exact, including the terminal prediction. The original strict
42-row test and failed v1 evidence remain unchanged.

Four fresh synthetic sources and their artifact/tokenizer/count/token-ID bindings
are frozen at JSON SHA256
`44bd53a7dcf72ae6afb8e1f9921bf461df3f5cf9cc14c0c7705f46188418a1fe`.
Counts are 352/451/375/384 before taking the fixed 256-ID prefixes. CPU tests bind
the complete policy and check reciprocal witnesses, strict decimal/F32 boundaries,
signed-zero/highest-ID ties, and invalid/contradictory data. The actual artifact
hash/tokenizer preflight passes; no v2 target forward has run at this checkpoint.

Read-only adversarial verdict: **freeze-ready; no concrete blocker**. The future
runner must record every accepted indeterminate witness and enforce exactness on
all 16 trajectory predictor rows. Temporary Cargo package-level test opt-level 1
is used only to shorten host metric work; no Cargo configuration or Metal kernel
changes are introduced. Execution and public-cap decisions remain separate.

## Packet 24 first v2 execution (passed, before surface promotion)

The shared runner takes an explicit frozen policy/hash/commit and classifier. V1
still uses its exact-ID classifier. V2 records reciprocal live-vector witnesses,
exact mismatch/accepted-indeterminate flags, and trajectory exactness on every row.
Predictor lengths 241-256, including the terminal row, use the exact requirement.
Pre-execution adversarial review confirmed the wiring before any v2 target forward.

First-run evidence: `target/profiles/k2-holdout-v2-12878-1789830419367474000`.
All 4096 rows, 16 residual sites, 12 bitwise partition controls, and 64 exact
trajectory predictor checks pass. There are **zero top-1 mismatches and zero
indeterminate allowances used**. Maximum raw error is 0.09023, RMSE 0.01234,
centered RMSE 0.008886, KL 1.47e-6, and TV 0.00078526. Minimum logit cosine is
0.99999031. Capture relative L2 is at most 0.002564, with cosine above 0.99999699.

The original strict 42-row GPU regression and fifteen CPU oracle tests also pass.
The native metallib SHA256 and runtime/primitive source hashes match v1 exactly.
Temporary package-level host opt-level 1 is recorded along with the native test
binary hash; it changes no Metal code or persistent Cargo configuration. Execution
used the production lease, real wired gate, and API validation. Free disk was
23 GiB before the approximately 5 GiB run. Elapsed 552 seconds is not speed evidence.

Post-result adversarial verdict: **sound checkpoint; sufficient to begin separate
guarded surface-promotion checks**. Scope is the pinned final Q8-weight artifact
with F16 KV, not F16 weights or all intermediate checkpoints. Public capacity is
still 32 here. V1 remains failed. Run/bench/plain+imported lens/JSON+SSE exact-boundary
checks and a central application-budget constant are required before promotion.

## Packet 25 guarded application ceiling

Run, request bench, plain/imported lens, and raw serving share the application
constant `GUARDED_APPLICATION_FORWARD_CEILING = 256`. Explicit generation budgets,
startup output defaults, and snapshot budget zero remain required. The source-plan
limit 7168 and declared model context remain separate. No kernel, runtime math,
KV layout, chat/tool policy, or checkpoint whitelist changes accompany promotion.

Pre-execution adversarial source review found no blocker. Boundary evidence lives
in `target/profiles/k2-guarded-256-surfaces-v1`: run/bench agree on bytes and token
fingerprints for 256 prompt tokens plus one sample and 255 prompt tokens plus two
samples. Plain and exactly bound identity-transport lens execute position 255 with
bitwise full-logit agreement. Both reject position 256 before Metal. Wrong exact
binding rejects even with override; unbound assets require explicit transfer.
Run/bench reject 257-forward or capacity requests and run requires explicit `-n`.

The companion ignored serving probe passes direct/JSON/SSE parity against those
benchmark rows, startup-default behavior, rejection before session allocation,
and fresh state after an abort at prefill tick 250. Its first execution exposed a
test assumption, not a production defect: streamed errors occur after HTTP 200,
as `response.failed`, whereas nonstream errors return HTTP 400. The assertion now
checks that established contract and absence of output text; transport behavior
was not changed. The original short serving/BOS/abort probe and actual-header
startup-refusal probe also pass. All GPU checks use API validation and the normal
production lease/real wired-memory gate; no foreign process or server is touched.

CPU surface tests pass (12 run/serve, 6 benchmark, 5 lens). A legacy regression
checker initially refused a stale benchmark build/source identity after a test
edit, before Metal. That guard is retained; rebuilds, not overrides, resolve it.
After rebuilding, all three legacy CLI checkers pass: request accounting,
all-36-site plain lens, and imported identity/nonsymmetric transport. Evidence is
in `target/profiles/k2-{bench,plain,imported}-after-256-v2`. No guard was bypassed.
Final read-only adversarial verdict: **commit-ready; no correctness or evidence
blocker**. The boundary-check script is included with its reproduction docs.
Scope remains the pinned final Q8_0-weight artifact with F16 KV on M4 Max. Failed
strict-extension/v1 history is retained. This is not F16-weight, all-checkpoint,
full-context, packed-prefill, compact-KV, or performance qualification.

## Packet 26 experimental online runtime integration

An immutable private backend field selects attention at the existing dense token
graph call site. Public loading explicitly selects materialized attention; the
online enum variant exists only under `cfg(test)`. No application flag, environment
switch, public-cap change, shader change, or cache-layout change is introduced.
Both encoders retain the existing physical-view/alias/causal checks and the same
session admission, transaction, source-stamp, and finite-value checks.

The adversarial design jam rejected duplicating already passing primitive tests
or rerunning multi-GiB IFM jobs. Instead, a native-only integration harness reuses
the four previously seen v2 fixtures at bases 0/37/8191/0, pinned artifact/tokenizer
checks, unchanged numerical/ranking gates, live ranking witnesses, and checked row
reader. Materialized controls run first and are retained with token/coordinate/
dtype/extent bindings and SHA256 provenance. The model and all sessions are dropped
before loading the online candidate. This is native-backend regression evidence,
not another holdout or independent-oracle qualification.

Pre-execution adversarial review found no blocker. The first execution passes:
`target/profiles/k2-online-integration-22255-1789833678489281000` records all 2048
rows, 16 residual sites, four bitwise singleton/split/whole controls, and 64 exact
trajectory predictor rows. There are zero failed records, zero top-1 mismatches,
and zero indeterminate allowances. Trajectories are materialized-derived, fixed
241+15 mathematical argmax sequences including EOS; terminal predictions remain
exact. Both backends reject appending position 257 at capacity 256 without prefix
advance or poisoning. Materialized and online weight/session lifetimes do not
overlap. API validation and the production lease/real wired gate are active.

The existing poisoned-future/guard synthetic primitive probe also passes again
through 257 positions, with maximum F64 error below 1.75e-6. Thirty-three CPU
runtime tests pass and the production CLI typechecks. The 267-second instrumented
native integration run is not a speed measurement. Disk was checked at 16 GiB
before writing approximately 2 GiB of native controls. Historical strict/v1
failures and v2 independent results are untouched. Independent-reference replay
with the online backend remains the next promotion gate, ahead of packed prefill
and compact KV.

Final read-only adversarial verdict: **commit-ready on correctness and evidence
scope**; the new native-comparison module is included in the checkpoint. The full
K2-filtered CPU suite passes 58 tests (19 opt-in GPU/artifact tests remain ignored
in that command). Retained independent-reference replay must validate provenance,
mode, hashes, coordinates, and capture noninterference before any promotion.

## Packet 27 online replay against retained independent reference

The unchanged holdout evaluator is factored from reference generation so it can
consume existing outputs. Fresh v1/v2 tests still explicitly select materialized
attention. A separate opt-in test selects online attention after validating the
first successful v2 reference directory, without executing a reference child.
The first adversarial review required source-pinned root digests rather than only
self-described hashes. Before online execution, the retained metadata was pinned:

- Manifest SHA256: `191387561440961677bbd64fad302627a1f5dc428fafe79eeec3294f53d1e3af`.
- References SHA256: `bbc0e26e0d18e0058cf4b6fef797f45cbb00555916ce582c572860099588b8ee`.

The loader also checks complete frozen policy/input equality, model/tokenizer
binding, reference source/wrapper/CMake identity and binary digest, F16 cache and
256-cell mode, M4 Max offload markers, every logit/capture/token payload digest,
and full finite coordinate/token/dimension/EOF protocols before native Metal setup.
All four capture traces are rechecked bitwise against ordinary reference logits;
greedy sidecars must match the supplied prefix and exact argmax continuation.
Original log bytes were not digest-bound by their producer: their mode/device
markers are validated, not represented as authenticated logs. Root digests bind
retained local evidence, not a remote cryptographic attestation.

Pre-execution closure found no remaining blocker. First replay passes in
`target/profiles/k2-online-retained-v2-23569-1789834548394017000`: 4096 rows, 16 sites,
12 bitwise partition controls, and 64 exact trajectory predictors. There are zero
failed records, top-1 mismatches, or indeterminate allowances. Maximum raw error
is 0.068885, RMSE 0.010222, centered RMSE 0.008722, KL 1.390e-6, and TV 0.00075459;
minimum logit cosine is 0.99999427. Maximum capture relative L2 is 0.0008490 and
minimum capture cosine 0.99999963. Capacity+1 remains retryable.

Eighteen CPU oracle tests pass, including a known-scalar RMSE check. No numerical
gate or arithmetic changed. The instrumented 433-second run uses the production
lease, real wired-memory gate, and API validation; it is not speed evidence. Only
small manifests/metrics are newly written, not another multi-GiB reference corpus.
The original materialized result and native-to-native comparison remain separate
evidence. This is independent-reference regression on seen fixtures, not a new
holdout. Application default/capacity remain materialized/256 pending separate
promotion and surface checks; historical strict/v1 failures are untouched.

Final read-only adversarial verdict: **commit-ready; no concrete blocker**. The
next default-only promotion needs the strict short regression plus forward/lens/
intervention and run/bench/JSON/SSE boundary checks; no new numerical policy.

## Packet 28 guarded online default

The K2 default now selects the already tested register-only online encoder.
Materialized attention remains an explicit test-only control in the native-paired
comparison, original frozen v1/v2 evaluators, and cache-precision diagnostic. No
shader, cache format, capacity, allocator admission, transaction, physical-view,
or legacy-family dispatch changes accompany promotion. The compiled metallib hash
remains `0f449fd916a7181dfbe46aabbf9e218025a40a581f21df20ce085a2608bb4462`.
The source-plan ceiling remains 7168 and the application guard remains 256.

Following adversarial promotion-gate review, actual default-path checks pass:

- Strict original 42-row oracle: `target/profiles/k2-oracle-24817-1789835255834896000`;
  all top-1 IDs match, maximum error 0.002500, RMSE 0.0006220, minimum cosine
  0.999999963. Its manifest explicitly records the online default.
- Native forward/split-prefill, captures and readout continuation, ordered
  interventions with causal KV preservation, and imported transport orientation,
  identity, isolation, and poison probes pass under the new default.
- CLI boundary evidence: `target/profiles/k2-online-default-256-surfaces-v1`;
  run/bench/plain+imported lens execute 256, reject 257, preserve exact binding and
  explicit transfer, and agree on output bytes/fingerprints/full identity logits.
- The same directory's serving result records direct/JSON/SSE boundary parity,
  startup-default output behavior, 257 refusal before session allocation, and fresh
  state after late prefill abort. The short BOS/abort serving regression also passes.
- Legacy CLI checkers pass in `target/profiles/k2-online-default-{bench,plain,imported}-v1`.
  CPU checks pass 61 K2 library, 12 run/serve, 6 benchmark, and 5 lens tests; CLI
  binaries build and unsupported-mode guards remain intact.

All GPU execution is serial under the production lease/real wired gate with API
validation. Ephemeral loopback probes do not alter shared servers. Qualification
remains pinned final Q8_0 weights/F16 KV on M4 Max, not every compatible checkpoint.
Online attention removes context-sized score scratch but does **not** reduce the
144 KiB/token stored F16 KV. Packed prefill, compact KV, longer-context qualification,
remaining lens CLI operations, and checkpoint-specific chat/tools remain separate.

Final read-only adversarial verdict: **commit-ready; no concrete blocker**. Default
versus historical-control separation and actual default-path surface evidence
support the stated scope. No repeated primitive or IFM qualification run is needed.

## Packet 29 shared exact-arithmetic packed baseline

The design jam chose existing token-axis Q8 `lcpp` GEMV rather than half-staged
GEMM. A single `encode_tokens` graph now serves singleton and test-only layer-major
chunks up to 32. The K2 wrapper checks projection geometry, Q8 byte extent, dtype,
alignment, activation access and aliasing. Checked row views serve grouped norm,
absolute RoPE, store-then-attention causal ordering, captures and interventions.
Only the last appended row receives interventions/captures/head evaluation.

Eleven temporary activation buffers are admitted/reconciled once before the first
submission and reused for the append: 237572 logical bytes/row, at most 7602304
bytes, excluding the existing shared KV arena and one logits row. The ledger now
counts physical commands independently from token advancement. All IDs are checked
before upload; every newly stored KV row and final residual row is checked after
each chunk. The final prefix publishes once. Unsubmitted failure is retryable;
any abandoned submitted work poisons without partial publication. This preserves
the original boundary-finiteness contract, not per-intermediate instrumentation.

Pre-execution adversarial review found no blocker. The initial synthetic probe
failed before dispatch because its Q8 test fixture incorrectly used the scalar
element-view API. Changing that fixture to the existing block-aligned byte-view
API fixed it; no runtime arithmetic or tolerance changed. All four K2 projection
shapes then pass bitwise at 1/2/31/32 rows with offsets, guards, future NaNs, and
immutable inputs/weights. The refactored singleton graph also passes the strict
42-row oracle: `target/profiles/k2-oracle-36755-1789852781269082000`.

Full-model evidence: `target/profiles/k2-packed-q8-37041-1789852837298819000`.
All 88 checkpoints across four seen v2 corpora, bases 0/37/8191/0, and three append
partitions pass bitwise for logits, captures, and complete KV bytes including
poisoned future rows. Readout preserves prefix/cache. Ordered last-token operations
and subsequent multi-token continuation match serial execution exactly. Invalid
IDs in a later chunk and 257 tokens at capacity 256 reject without cache mutation;
an actual nonfinite final intervention after three submitted chunks poisons while
leaving committed length zero, and a fresh session remains usable.

Sixty-five K2 CPU tests pass; the production CLI typechecks. All GPU execution uses
API validation and the production lease/real wired gate. Test-package host opt-level
1 and SHA2 opt-level 3 accelerate host checks only. The 137-second instrumented run
is not speed evidence. No shader, public capacity, KV layout, or other family changes.
Public prefill and serving cancellation remain serial; packed selection is private
and test-only pending independent-reference and application-surface promotion.

Final read-only adversarial verdict: **commit-ready; no code or evidence blocker**.
The new packed implementation and test modules are included. Explicit packed
retained-reference replay is the next gate; fresh historical evaluators stay serial.

## Packet 30 packed replay with retained independent controls

The shared evaluator now takes an explicit attention/prefill pair. Fresh v1/v2
remain `(Materialized, Serial)` and the existing online replay remains
`(Online, Serial)`. Only the new opt-in test selects `(Online, BatchQ8)` for its
split/whole controls. The pinned root digests, payload/header/token/finite checks,
reference identity and trace-noninterference validation are unchanged.

Pre-execution adversarial review approved this narrow reuse and its explicit
topology labeling. First replay passes in
`target/profiles/k2-BatchQ8-retained-v2-37737-1789853432693300000`: 192 endpoint
comparisons in twelve bitwise partition controls, covering 72 multi-token appends
plus singleton boundaries. All 4096 independent-reference baseline rows, 16 sites,
and 64 exact trajectory predictors pass with zero top-1 differences or ranking
allowances. Those rows are singleton execution, not 4096 packed outputs; packed
endpoints match the qualified baseline bitwise. Packet 29 independently supplies
full-cache, packed-capture, intervention, and fail-stop checks.

The original independent baseline metrics remain exactly unchanged. Eighteen CPU
oracle tests pass. The production lease, real wired gate, and API validation cover
the 336-second instrumented run, not a speed measurement. No reference executable
is run and no multi-GiB outputs are regenerated. Public prefill remains serial;
automatic Q8/lcpp selection and application records/checks remain a separate gate.

Final read-only adversarial verdict: **commit-ready; no concrete blocker**. Automatic
selection must require all 252 block projections plus lcpp compatibility, record
the selected topology, and retain explicit serial historical/serving behavior.

## Packet 31 automatic guarded packed prefill and surface records

Public loading resolves packed mode only when all 252 block projections are Q8_0
and the compatible lcpp singleton kernel is enabled. Other admitted projection
formats and disabled lcpp select serial execution; malformed checkpoint inventories
still fail binding rather than gaining a fallback. The private control loader
still selects serial prefill, preserving historical evaluator arithmetic. CPU
negative controls cover each eligibility condition. No new environment switch,
shader, KV representation, context guard, or other-family execution change.

Run, bench, and plain/imported lens record typed prefill facts: mode, actual maximum
chunk, physical commands, and logical temporary activation bytes. Bench adds
`method.prefill_execution` without changing the existing `method.prefill` string
type. K2's optional run diagnostics are omitted from DeepSeek records; its only
source change sets that optional field to `None`, and shared envelope tests pass.
Lens processes bounded chunks and checks shutdown between them, so its cancellation
granularity is now at most 32 positions. Serving retains actual per-token appends
and cancellation boundaries, not synthetic ticks around a packed call.

Default-path native validation exposed an existing private-state invariant: the
session residual scratch must retain the final forward row. Packed output/captures
were correct but that persistent scratch remained unchanged. Rather than weakening
the old test, the final packed command now mirrors its last residual into the
persistent buffer with the existing checked F32 copy encoder. Scratch lives through
completion, and encoding/command/finite failures retain the same transaction rules.
Targeted adversarial review approved this ordered state-mirroring fix. The original
test then passes unchanged, as do forward/split, all four intervention formulas,
ordered operations/causal KV, imported transport, isolation, and poison probes.

Final surface evidence: `target/profiles/k2-packed-default-256-surfaces-v2` passes
256-token prefill and a transition reaching 256, output/fingerprint parity,
plain/imported full-logit identity, strict binding/explicit transfer, and 257
refusals. Records report 32-row chunks, eight commands, and 7602304 temporary bytes.
Actual two-token bench/lens runs with lcpp disabled report serial/one-row chunks,
two commands, and zero packed scratch; these are fallback checks, not numerical
qualification of the alternative matvec. JSON/SSE serving matches the packed
run/bench results while preserving startup defaults, late-abort freshness and
pre-session overbudget refusal. Short serving/BOS/abort and CPU startup refusals
also pass. All three legacy CLI scripts pass in
`target/profiles/k2-packed-default-{bench,plain,imported}-v2`.

CPU checks pass 66 library, 13 run/serve, six benchmark, five lens, and 18 shared
request-stats tests. GPU checks use the production lease, real wired gate and API
validation, with no shared server or foreign process operations. This promotes a
guarded Q8/F16-KV/M4 Max packed baseline, not half-staged GEMM, faster throughput,
long-context qualification, or lower KV bytes/token. Stored KV remains 144 KiB/token;
compact KV is the next storage milestone.

Final read-only adversarial verdict: **commit-ready; no concrete blocker**. The
selection, retained-state fix, additive records, fallback, and actual application
checks support promotion without repeating independent qualification.

## Packet 32 compact cache layout, kernels and private runtime

The adversarial design jam retained symmetric Q8 as the first storage baseline and
required byte-safe loads, immutable cache-kind plans and explicit invalid-value
handling. `COMPACT-KV-POLICY.md` records the rules before GPU execution. Each block
stores an LE F16 scale and 32 signed bytes without padding. The 1088-byte K or V
row gives 78336 bytes/token across all layers, 46.875% below F16's 147456, excluding
allocator overhead, weights, and other scratch. There is no eviction or history
expansion. The new K2-only shader does not modify any legacy quantizer or attention
kernel; query/output alignment is distinct from unaligned byte payload access.

The quantizer checks input exponent bits before reductions/casts, uses original
F32 scale for codes, and stores rounded F16 scale. Invalid input or scale overflow
writes a NaN-scale/zero-payload sentinel. Scale underflow and signed-zero inputs
produce canonical zero blocks. An initial CPU test bound incorrectly assumed the
F32 scale itself could not underflow; its zero-block bound was corrected to original
maximum magnitude before GPU execution, without changing encoding. Row validation
rejects nonfinite/negative scales, -128 codes and noncanonical zero payloads.

Typed plans derive all storage-specific byte ranges and preserve existing context
guards. Private Q8 loading prices and allocates an I8 arena, not a hidden F16 cache.
Session creation and append preflight check the complete physical cache contract
before encoder construction. Packed scratch shares the arena through byte views.
Every newly stored row is checked before publication; submitted failure retains
the existing poison semantics. Materialized attention cannot select Q8. Public
loading and all application records remain F16, with no new runtime switch.

Pre-execution reviews approved the primitive and runtime gates. Both pass on M4 Max
with the production lease, wired-memory gate and API validation. Curated quantizer
bytes match the independent CPU policy exactly. Bounded general quantization error,
guards, future/layer exclusion and source/cache immutability pass. Inline attention
through 257 positions has maximum F64 error 2.312e-6, below the predeclared 2e-5 gate.
The full-model four-position smoke first executes serial Q8, then verifies packed
Q8 bitwise captures, complete cache and continuation; it also checks readout and
ordered interventions, pre-graph mismatch rejection, capacity refusal, a genuine
NaN-scale sentinel from overflow, prefix-zero poisoning, and fresh-session recovery.

Seventy CPU K2 tests and CLI typechecking pass. The unchanged F16 strict 42-row test
passes in `target/profiles/k2-oracle-68764-1789921797087492000`, with the same metrics
as before; existing F16 forward/capture/readout controls also pass. These results
establish storage/kernel/runtime invariants, not Q8 full-model quality or speed.
The next step is the separately reported 256-token F16-versus-Q8 diagnostic under
unchanged v2 metrics, never automatic default promotion or post-result retuning.

Final read-only adversarial review found **no concrete blocker** in storage/runtime
wiring and approved the bounded follow-up diagnostic. Its F16 control is native,
not an independent implementation; that distinction remains explicit in reports.

## Packet 33 compact-cache quality diagnostic (failed envelope, no promotion)

Pre-execution adversarial review found no blocker in the bounded native F16/Q8
comparison. The run completed under the production lease, actual wired-memory
gate, and Metal API validation on the pinned final Q8-weight checkpoint. Controls
and candidate use separate model lifetimes; this is a native self-comparison on
previously seen v2 fixtures, not an independent implementation or new holdout.
Artifacts: `target/profiles/k2-compact-diagnostic-70232-1789922651144287000`.

The 1088 compared rows comprise four 256-token teacher-forced cases and four
16-predictor F16-derived mathematical trajectories (visible lengths 241..256,
EOS-inclusive). Sixteen post-block captures cover layers 0/11/23/35. Frozen v2
numerical, ranking and capture gates are reused without edits. The test's success
means hard invariants passed; its summary explicitly says the quality envelope
**failed**: 808 rows (762 teacher-forced, 46 trajectory) and 8 capture sites fail.
Seven teacher-forced top-1 choices disagree, all beyond the two-sided regret
allowance. All 64 trajectory predictors agree exactly, which does not erase their
46 numerical failures or establish arbitrary free-generation equivalence.

Worst row metrics: absolute error 6.586494, RMSE 1.138811, cosine 0.9633473,
centered RMSE 0.818272, TV 0.0562574 and reference-to-candidate KL 0.00687541.
Capture relative L2 reaches 0.0581351 and cosine falls to 0.9985846. Gate failure
counts overlap: TV 770, KL 461, centered RMSE 64, RMSE 46, top-1 7, absolute error
13, cosine 12. This is measurable quantization drift, not a quality or speed win.

All within-Q8 serial/split/whole logit, capture and complete-cache bitwise checks
pass, with poisoned future storage, identity imported readout and capacity refusal.
The actual logical arena at capacity 256 shrinks from 37748736 to 20054016 bytes;
Metal's allocator-reported `allocatedSize` matches those values in this run.
Observed session allocation deltas are separately 38993920 and 21299200 bytes.
These are storage/allocation observations, not proof of total physical residency.

The read-only inspector checks pinned identities, selected corpus/base coordinates,
input/control/capture digests, protocol headers/extents, metric coverage, finiteness,
summary counts and allocation sizes. It reports recorded metrics, not recomputed
candidate logits. Adversarial review caught missing standalone provenance checks
for model/tokenizer/compact-policy identity and selected bases; these were added
with CPU negative controls before checkpoint. The `k2` CPU filter passes 76 tests
(including incidental name matches); 31 opt-in tests remain ignored.

F16 stays the application default. No quantizer change, tolerance retuning, extra
checkpoint download, independent-oracle rerun, or public option follows from this
diagnostic. The design jam recommends closing this failed baseline and next wiring
bounded CLI forward interventions to the existing graph, advancing M3 without an
open-ended compression detour. Compact quality and long-context qualification
remain explicitly unfinished.

Final read-only adversarial closure: **commit-ready; no remaining concrete blocker**.
The inspector's three CPU tests pass, and the retained evidence passes its read-only
integrity/count audit while retaining the failed quality verdict.

## Packet 34 main readiness and principled capacity admission

The user requested removal of arbitrary forward/position/context/response limits,
warning-free readiness and integration into main, with checkpoint-specific templating
high priority but nonblocking. The out-of-band review's eight-warning count, private
packed default and uncommitted compact state were stale: all-target checking was
already warning-free, packed prefill was already selected automatically, and the
failed compact diagnostic was committed. Its positive conclusions about explicit
serve-family admission, early family request parsing and literal RawText output
remain applicable. Dispatch-spine, serial-driver and family-profile consolidation
are intentionally left to the concurrent maintenance lane. Exact imported lens
artifact/deployment identities are already implemented; snapshots remain disabled.

Application run/bench/plain+imported lens/serve no longer enforce 256 forwards or
response tokens. Cache planning no longer inherits the materialized kernel's 7168
score limit. The initial profile-specific 524288 metadata ceiling and hardcoded
8 GiB buffer cap are removed: positive u32 checkpoint context, checked absolute
extents, configured request budgets, actual device buffer size and memory admission
are authoritative. Backend-local materialized guards remain before encoding or
loading. Q8 KV is not promoted. A normal `load` entry point supersedes the old name,
with a nondeprecated compatibility delegate for existing callers. The raw startup
research warning is gone; numerical evidence remains in docs and metadata. Run
accepts its normal token default; serve retains explicit memory capacity and response
default. Run/bench prefill now checks cancellation between bounded chunks.

CPU tests cover 257/7169, stage contexts, declared-context end, context+1, overflow
and u32 metadata limits, without allocating hypothetical caches. An initial expanded
primitive run FAILED the unchanged 2e-5 F64 bound at 7168 rows: error 2.469399e-5.
Rather than relax the gate, an adversarial numerical jam reviewed a stable merge
of fixed 256-row online summaries. The original shader remains selected through
256; a separate longer-history entry merges independent local/global summaries in
registers with max-shifted exponentials. Loop ends and byte offsets cannot wrap.
This keeps all history, constant scratch and no full-history dequantization.

The revised synthetic corpus passes unchanged bounds through 8192 retained rows,
including flat/sharp distributions, separated block maxima, boundary spikes, all
heads and immutable poisoned future/guard storage. Worst F64 error is 8.096181e-6
(the 512-row adversarial case); ordinary 7168 error falls to 1.018474e-6. This is
improved observed behavior, not proof that accumulation order was the only source
of the original discrepancy. Materialized comparisons stop at 7168; larger calls
are explicitly rejected. No kernel outside K2 dispatch changes behavior.

The new actual-checkpoint runtime test passes serial/split/whole bitwise logits,
complete cache, captures, final readout and continuation at lengths 257/1024 and
bases 0/523263, with capacity 1025. An initial harness call incorrectly requested
an empty capture list through the nonempty-only capture API; it was corrected to
the existing empty-intervention path, without changing runtime semantics. Device
price-only inspection reports 77309411328 cache bytes at full declared context;
the test never allocates that cache or infers residency from its price. CPU device-
buffer refusals leave GPU allocation unchanged. This is not a 512K-history test.

Strict independent 42-row regression passes unchanged with the same prior metrics:
`target/profiles/k2-oracle-767-1789926412803888000`. Retained independent v2 replay
passes all 4096 singleton rows, 16 sites, 64 predictors and 192 partition checks:
`target/profiles/k2-BatchQ8-retained-v2-1378-1789926541377783000`. These short tests
do not independently qualify the new long-history arithmetic.

Surface evidence in `target/profiles/k2-context-1024-surfaces-v1` passes run/bench
fingerprint and byte parity, a 257-token generated response budget, 1024 forwards,
plain/imported full-logit identity, exact binding/explicit transfer, serial fallback
and pre-Metal request/context refusals. The ephemeral HTTP test also passes direct,
JSON and SSE parity against those rows, defaults, request-capacity+1 rejection and
fresh state after prefill cancellation (`serve-context-result.json`). All GPU tests
use the production lease, real wired gate and API validation; no shared service
was stopped or restarted. Current CPU counts are 77 engine and 24 CLI tests under
the K2 filters, and all-target checking emits no warnings.

Main integration will retain unrelated user work. The user explicitly approved
replacing the colliding untracked historical K2 plan with this evolved plan.
Remaining product work is checkpoint-appropriate templating and CLI intervention
plans; broader quality/latency and compact-cache research stay separately scoped.

Pre-commit review found no runtime blocker, but caught stale CLI/help and benchmark
README cap text, now corrected. Its repeated serve-conversion observation was stale:
the K2 call already used `u32::try_from`, verified in the diff rather than blindly
carried forward. Conditional commit/merge approval requires only those text fixes
and preservation of unrelated main work.

Final local accounting audit also tightened benchmark reporting: its reported
retained-window plan now uses the actual device buffer limit after a CPU-only
structural preflight, and the method labels explicitly record that cancellable
chunk appends compute intermediate head rows (only the final prompt row is sampled).
This is not described as unchanged prefill throughput or a speed improvement.

Final adversarial closure: **no remaining commit blocker; commit and authorized
main integration ready**. The final benchmark smoke passes in
`target/profiles/k2-main-ready-bench-v1`, including warmup/repeat identity, raw-run
fingerprints, native BOS, literal IDs, phase/prefix accounting and host-only refusal.
All-target checking remains warning-free.

## Packet 35 verified final-checkpoint no-tools CLI templating

After main integration, the user asked to continue. The templating design jam
recommended family-local rendering and a separate later output/HTTP packet, with
raw checkpoints unaffected. Eligibility binds the structural dense profile,
posttrained tokenizer metadata, embedded-template digest, and verified retained
checkpoint content `719ae3a7c9386c25db2c33b50be15d715f883aa5a762495d5a65776660179e99`.
This first chat profile is not a whitelist for raw execution. Compatible earlier
checkpoints and unregistered repacks still run raw but do not inherit chat claims.

Pinned IFM revision `2c9659a84c4eea6f9f60462221fe762c8c84d75c` supplies the no-tools
contract. Upstream template SHA256 is
`a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590`; the actual GGUF
embedded template is separately bound as
`f6e3cd6dbf0f95016fff531f41f921dee541025733c14580a881cf5a5f9fa750`, not described as
byte-identical. Native rendering follows the pinned upstream oracle, not embedded
Jinja. Source revision association is the publisher's conversion declaration,
not independent quantization fidelity proof. Only metadata was downloaded.

Independent Jinja 3.1.6/HF tokenizers 0.22.2 fixtures contain 21 cases: 17 valid
renderings and four upstream errors. All valid UTF-8 bytes and native token-ID
sequences agree, including system/Unicode, every assistant thinking alias, empty
thinking, priority conflicts and literal protocol markers. Initial source-reading
and reviewer advice incorrectly omitted the assistant-history newline: Jinja
actually preserves the newline before its generation block. The first CPU fixture
test caught this; native rendering was corrected without changing oracle bytes.

The reviewer initially advised retaining EOS-only chat until explicit evidence
existed. The pinned upstream `generation_config.json` provides that evidence:
EOS IDs `[1,250019]`, SHA256
`2da7d47641f4509da4ae47711e31d8b5f0f3f801ee08d87e7e9f07f814bdc4a3`. Native tokenization
identifies 250019 as `<|ifm|im_end|>`. CPU controls prove either stop is counted but
not emitted or forwarded. Raw completion and HTTP serving retain EOS 1 only.
The generation-config hash is reference provenance, not a verified GGUF field.

`qwen run --user`/`--system`/`--messages` bind the profile before input file/stdin
acquisition and Metal setup. The tokenizer owns automatic BOS; authored specials
are preserved, not deduplicated. High/medium/low select the released thinking
opener, default high. There is no guessed non-thinking transition. String-only
system/user/assistant turns, one leading system and final user are the supported
subset. Tools/developer/multimodal/unknown or duplicate fields/null thinking are
rejected rather than silently coerced/dropped. Assistant history requires an
explicit thinking string and preserves upstream alias priority.

Output is literal generated text, not separated reasoning/final-answer items;
tools remain unavailable, and HTTP/bench/lens stay raw. Capability projection and
request stats name this scope, profile, effort, stop set, BOS owner and input token
digest. `qwen info --json` now performs read-only full retained-byte verification
when identifying this profile; no identity cache or downloader claim substitutes.

Pre-execution adversarial review found no blocker. The leased/API-validated CLI
smoke passed in `target/profiles/k2-chat-cli-v1`: all three efforts, assistant
history, system/Unicode, exact token hashes and byte/fingerprint parity with
serialized raw controls, plus pre-GPU refusal of unsupported requests. These
eight-token probes are wiring evidence, not answer-quality or tool qualification.
No maintenance-lane refactor or shared-server operation was performed.

Final adversarial verdict: **commit-ready, no concrete blocker**. The final
metadata-inclusive smoke also passes in `target/profiles/k2-chat-cli-v2`. Eighty
engine and 25 CLI tests pass under K2 filters, plus the opt-in CPU native-token/
artifact-identity check. All-target checking remains warning-free. The next packet
is K2-specific incremental reasoning partitioning and reuse of this renderer for
HTTP chat; tool support remains a separate, unadvertised capability.

## Packet 36 verified HTTP chat and shared reasoning partition

The family-local HTTP parser now accepts array input only with an owned opaque
chat capability verified once at startup. String input retains the raw contract.
Unknown artifacts and verification errors remain raw-only with a diagnostic.
The strict original-wire allowlist runs before common control parsing; a neutral
placeholder deliberately avoids Qwen's transcript-specific marker normalization.
Only no-tools system/user/assistant history is admitted, with explicit preceding
reasoning (including empty) for assistants. Replayed generic reasoning uses IFM's
base/high history alias; selected effort applies to the new generation suffix.
Completed emitted items replay, including empty summary/annotations metadata;
incomplete history and unsupported metadata reject rather than disappear.

CLI and HTTP share a bounded incremental UTF-8 partitioner. It starts in reasoning,
optionally removes one matching opener at byte zero, and switches to visible text
at the first matching effort-specific close. Later/wrong-effort/Qwen/tool strings
remain literal. Empty reasoning emits an item. Token limits remain incomplete;
EOS/im_end before close fail; abort discards ambiguous pending bytes without a
synthetic completion. Chat stops remain `[1,250019]`, raw `[1]`. CLI reasoning goes
to stderr and answers to stdout; unfinished reasoning emits an explicit mandatory
diagnostic and `reasoning_closed:false` in optional stats, with normal budget exit.

Source audit: vLLM `k2_horizon_reasoning_parser.py` at
`1f76efaa2195485b92cb04215aba6fb8f5fe523d` confirms the three effort tags, preopened
streaming reasoning and optional leading opener. This no-tools subset intentionally
does not inherit its tool-marker fallback or batch no-close-as-visible behavior.
No complete vLLM equivalence or duplicate-JSON-key rejection is claimed.

Adversarial review before execution found two issues: public profile fields could
be forged by an internal caller, and CLI incomplete reasoning lacked a mandatory
user-visible diagnostic. Both were corrected: profile fields are private, serving
owns an opaque capability with a test-only mock constructor, and incomplete CLI
reasoning is explicitly diagnosed. Follow-up review approved the leased smoke.

The first actual test correctly exposed a bad test assumption: low-effort chat
reached im_end within eight samples, whereas raw deliberately emitted that marker.
The comparison now uses chat's emitted prefix, excluding its counted terminal ID;
production stop behavior was not changed. The next run passed behavioral checks
but failed writing a relative evidence path from Cargo's crate working directory.
The absolute-path rerun passes and retains
`target/profiles/k2-chat-http-v3.json`: all three efforts, eight JSON/SSE responses,
high/medium incomplete reasoning, low-effort completed answers, raw-prefix controls,
abort isolation and session reacquisition. No foreign server was started/stopped.

`target/profiles/k2-chat-cli-v3` passes all-effort/history/Unicode/input-token oracle
controls, unchanged short raw token fingerprints, stderr/stdout partitioning,
incomplete diagnostics, refusal before GPU, and completed CLI/HTTP answer/token
count parity. Both paths use production-exclusive leases, the real wired-memory
gate, and Metal API validation. These are wiring and termination checks, not new
answer-quality, tool, full-context or performance claims. F16 KV remains unchanged.

The obsolete chronological plan is replaced by a concise current-scope/remaining
work map; this review record retains the historical failures and decisions.

Closure adversarial verdict: **commit-ready, no concrete correctness or merge
blocker**. Final checks pass: 74 engine tests under `k2_`, 19 qwen K2 tests,
127 serving CPU regressions, warning-free all-target checking, and clean diff
whitespace. Ignored GPU/research suites were not silently counted as executed.

## Packet 37 remove unpriced tensor-sized host staging

The out-of-band allocation finding is confirmed: K2 session and packed scratch
called `zeros_dtype`, allocating a host zero vector before copying into Metal,
while admission priced only the eventual Metal buffers. This establishes unpriced
staging and copy work, not a reproduced OOM or exact doubling of physical memory.

K2 session, packed scratch and lens-owned buffers now use a crate-private
`zeros_dtype_unstaged`: checked shape/device/CPU extent, one fallible owned Shared
allocation, full direct byte-zero initialization before publication. The existing
public helper and other families are unchanged. Outer memory admission, allocation
transactions, price reconciliation, cache layout and session publication ordering
are preserved. There is no tensor-sized CPU staging allocation to price. Quantized
byte-zero semantics match the old helper; no new decoded-numeric-zero claim.

Design and pre-execution adversarial review approved this narrow approach over a
global allocator rewrite. The optional null-pointer suggestion was not adopted:
the Metal binding returns `NonNull<c_void>` from `contents()`, already expressing
that FFI precondition. Extent/storage/ownership checks remain explicit.

Both synthetic ignored probes pass with production-exclusive leases, the actual
wired-memory gate and API validation. Small I8/I32/F16/F32/Q8 byte checks verify
full zeroing, independent allocations, writable ownership and invalid extents.
An injected partial multi-buffer allocation failure releases prior buffers and
allows a clean retry. The fresh-process 32768-capacity stress probe allocates
4,831,838,208 logical KV bytes, with first/middle/last page checks and full teardown.
Its admitted Metal upper price is 4,833,132,544 bytes; observed delta is
4,833,083,392. Initialization takes 162.719 ms. Darwin process `ru_maxrss` rises
from 49,463,296 to 4,883,021,824 bytes. Evidence:
`target/profiles/k2-unstaged-allocation-32k-v1.json`.

RSS is a cumulative process high-water diagnostic, not exact allocation/physical
residency attribution. This is a synthetic allocation check without model weights
or full-context inference; there is no paired speedup claim. No giant unsafe old
staging control was run merely to restate the source-grounded defect.

Closure adversarial review found no commit blocker. All-target compilation is
warning-free; twelve runtime host tests pass, in addition to the two explicitly
executed allocation probes.

## Packet 38 separate prefill advancement from output readout

The discarded-readout finding is confirmed. The runtime already skips heads on
intermediate tokens within an append, but frontends split prompts for cancellation;
every append previously evaluated/scanned/downloaded logits that most chunks threw
away. `K2Session::advance` now runs the same causal graph and append transaction
without a head, logits finite scan, or logits download. The internal explicit
readout policy is separate from last-chunk residual mirroring and hook placement.
Residual and newly written KV finite checks, source checks, transaction publication,
and poison semantics remain. Empty advancement rejects before any work, and
no-readout mode cannot be combined with captures/interventions.

CLI, actual singleton HTTP, request benchmark, and lens prefill use advancement
for nonfinal chunks and ordinary append/capture for the final chunk. Existing
shutdown/checkpoint/tick locations and generation-transition appends are unchanged.
Benchmark method metadata names final-prompt-only readout. Per-append packed scratch
still exists; lazy retention is deliberately deferred because it changes admitted
storage lifetimes, independently of this head-scheduling correction.

Design/pre-execution adversarial review approved the bounded change. Leased,
API-validated actual-Q8 tests compare the old append-per-chunk control with the
new policy at singleton, 32-row and 17-row boundaries. All final logits, three
capture sites, full cache bytes, and genuine fixed-token continuation agree
bitwise. Test-only counters at actual head encoding and host logits copying prove
exactly one of each for optimized prompt execution; no command-count inference.
Stale NaN logits are ignored during advance; invalid/empty inputs remain retryable.
An injected nonfinite residual after command submission poisons without publishing
a prefix; the entire session drops cleanly and a fresh session is available.

A separate paired diagnostic uses the same loaded artifact and 128 synthetic
tokens, fresh sessions, warmups, then three alternating-order pairs for each policy.
Every final logit row agrees exactly. Singleton head/download counts fall 128->1;
packed counts fall 4->1. Warm singleton prefill wall ranges 5062-5095 ms for the old
control versus 4690-4711 ms for advance (roughly 7% lower). Packed ranges 2758-2761 ms
versus 2734-2745 ms (under 1%). These include debug/API-validation and per-append
scratch costs, exclude session allocation/model loading, and are not end-to-end
CLI/HTTP throughput claims. Retained evidence:
`target/profiles/k2-readout-schedule-pairs-v1.json`.

Actual HTTP chat JSON/SSE and raw/BOS/abort-isolation regressions pass after the
change (`target/profiles/k2-chat-http-readout-v1.json`). The 1024-capacity surface
checker passes raw run/benchmark fingerprints, 257-output budgeting, plain/imported
lens readouts, packed/serial controls and pre-GPU refusals in
`target/profiles/k2-readout-1024-surfaces-v1`. This is application consistency and
readout-scheduling evidence, not new independent >256-history qualification.

Closure adversarial verdict: **commit-ready, no concrete blocker**. Final
all-target checking is warning-free and all 127 serving CPU regressions pass.

## Packet 39 cooperative exhaustive-identity cancellation

The verification responsiveness finding is confirmed. The retained-file scan used
1 MiB chunks but did not observe CLI shutdown while hashing. A new cancellable
entry point checks before source work, before each bounded read, after each shard
and before publishing the content report. The existing uncancellable API wraps it
with a never-cancel callback; content-root domains, source-stamp validation and
the no-cache/no-downloader verification policy are unchanged.

K2 profile verification propagates cancellation and checks before profile publication.
CLI run, JSON info and server preparation use the shutdown checkpoint. Info checks
again before emitting JSON; serving checks again before handling an unverified
profile, so a signal cannot silently start a raw-only fallback. Text inspection
still does not perform exhaustive chat verification. Help and comments now describe
CPU retained-byte inspection accurately. Cancellation is cooperative, not an
interruptible-filesystem-read guarantee.

Adversarial review found no blocker before the CPU checks. Fifteen identity tests
pass, including synthetic-reader cancellation before the first read, between
chunks, and after the final partial read. Subsequent reads stop and no root returns;
successful chunked hashing matches the whole-input root. Three chat CPU tests and
all-target checking pass without warnings. The opt-in actual-artifact CPU test
also passes: an injected mid-scan cancellation returns no verified profile, followed
by unchanged content identity and all native template token-ID fixtures. No GPU
execution or new weights are needed for this packet.

Owned-process SIGINT checks also pass for JSON info, chat run (stdin input), and
serve startup with an ephemeral address: each exits 130 with no JSON/stdout, no
Metal initialization under `MTL_DEBUG_LAYER=1`, and no raw-only fallback diagnostic.
Only the exact spawned child receives the signal. Observed teardown after signal
is 48-95 ms in this smoke, not a cancellation-latency guarantee; the synthetic
reader test is the deterministic proof of between-chunk cancellation.

Closure adversarial verdict: **commit-ready, no concrete blocker**. Ten CLI parser/
help tests, nineteen K2 frontend tests, final all-target compilation and diff
whitespace checks pass. No compiler warnings remain.

## Packet 40 artifact-derived capability contracts

The unconditional execution-capability finding is confirmed. K2 JSON info used to
advertise the family run implementation independently of rejected configuration,
tensor/storage admission, tokenizer, or generation stop policy. It now reports
private family implementation facts separately from shared artifact CPU preparation
and request/device admission. Per-lane status is `conditional` after artifact
admission or `unsupported` on rejection; `request_device` is explicitly
`not_evaluated`. No device allocation or numerical qualification is implied.

Library `K2ArtifactLayout` and `K2PreparedArtifact` perform the same config/full
inventory/range/native-embedding/retained-layout/tokenizer checks consumed by CLI,
serve, benchmark and lens. Layout and tokenizer stages remain separable for honest
benchmark setup timing. The retained one-position plan is only a CPU artifact
layout probe; execution still replans the requested capacity on the actual device
and live memory signals. Stable codes distinguish configuration, tensor inventory,
embedding storage, retained storage, tokenizer and generation-stop failures.

Generation stops are a separate check used by run/serve/bench. Lens does not sample,
so an otherwise tokenizer-valid extra EOS set does not disable its core lane.
Failed core or generation admission prevents any chat-profile verification attempt.
The final artifact still needs the same opaque, full-byte verified chat proof;
`execution.serve.chat` now reflects that artifact-specific decision as a boolean.
CLI CPU admission precedes prompt-file/stdin acquisition and Metal initialization.
No cross-family driver/profile refactor is introduced.

Model-free CPU fixtures use real GGUF headers and sparse, nonoverlapping tensor
extents. They write no payload, perform no payload hashing or GPU execution, and
are not trained models. Tests cover compatible 8192-context raw preparation,
generation-only extra stops, bad geometry, missing/extra tensors, norm storage,
unsupported Q5_K embedding gather, malformed tokenizer, out-of-range descriptors
and overlapping retained ranges. Pure projection tests cover rejected/conditional
lanes and the absence of device or numerical admission claims.

Design and pre-execution adversarial reviews found no blockers. Actual pinned-Q8
info/chat, request benchmark and plain-lens checks pass in
`target/profiles/k2-admission-chat-v1`, `k2-admission-bench-v1`, and
`k2-admission-lens-v1`; they retain template/token IDs, all 36 lens sites, raw/chat
controls, benchmark accounting and host-only refusal evidence. Actual borrowed
HTTP JSON/SSE/stop/abort checks also pass (`target/profiles/k2-admission-http-v1.json`).
GPU children use production-exclusive leases, the wired-memory gate and API
validation. These are preparation/surface regressions, not new quality or speed
claims. Compiler checking remains warning-free.

Closure adversarial review found no concrete commit blocker; final K2 CPU tests,
all-target compilation and whitespace checks pass.

## Packet 41 separate loaded-request timing from setup and waiting

The telemetry mismatch is confirmed: K2's old request timer included artifact
verification, tokenizer construction and input waits, subtracting only the later
Metal/model/session interval. New namespaced timing phases distinguish artifact
layout, tokenizer construction, profile verification, acquisition, rendering,
request preparation, encoding, model/context loading, session setup and resident
execution. Encoding wraps only `tokenizer.encode`; prompt token fingerprinting is
not mislabeled as tokenization.

Because encoding must precede capacity-shaped allocation, K2 core `timing_ms.total`
is the explicit reconstructed sum of encoding, request preparation and resident
execution. Its policy string states that it is not a continuous interval. The
continuous `end_to_end_lane_ms` preserves all waiting/verification/setup from K2
lane entry through generator return, with its exclusions named: initial dispatcher
GGUF opening, final formatting and stats serialization. Unclassified host overhead
is an explicit residual, not silently assigned to inference. Durations use monotonic
clocks and checked addition/subtraction; overlapping/overflowed accounting rejects.
Other families' telemetry and generation behavior are unchanged.

Design review approved the segmented/continuous distinction. Pre-execution review
caught potential `load_ms` semantic drift: the first implementation also added
artifact/tokenizer setup there. It now retains model/context plus session meaning;
the other setup costs remain in their named phases. Injected-duration tests prove
that waiting, verification, rendering and setup affect their own fields and lane
wall without contaminating the loaded-request total or encoding. They also cover
raw zero phases, explicit residual, overlaps, overflow and the preserved load scope.

The leased/API-validated actual CLI checker passes in
`target/profiles/k2-timing-chat-v1`: all effort levels, history/Unicode, raw controls,
completed answer/HTTP parity, unchanged token fingerprints, complete timing phase
inventory, finite nonnegative values, core/diagnostic total agreement and full
phase-plus-residual reconciliation. These are attribution/behavior checks, not
new latency or throughput qualification. All-target checking is warning-free.

Closure adversarial verdict: **commit-ready, no concrete blocker**. All twenty-two
K2 frontend CPU tests and final all-target/whitespace checks pass.

## Packet 42: practical final Q4 generation

The user reprioritized final dense 7B quantizations, native tools/reasoning, native
context and applicable existing optimizations. Intermediate-shape campaigns and
lens overflow design are not blockers; the latter remains parked, not canonical.

The standard publisher Q4_K_M (revision `a5094087a5a55c2de80264c11504d8ca95a022ff`)
is downloaded to `~/models`, exact size 5,592,217,984 bytes and SHA256
`eb89c15a0ae9712be2ee462bf43802de14200f20f93b73da6eb68c2ebdd28e4e` verified.
Native retained-content BLAKE3 is
`a53e7e9ab4a9c2e82596b9526d837668fe4fd4145de085f9d9beafa3c0dfe3b1`.
CPU preparation passes with tokenizer `51ebd8140ea2abd9`, the same embedded-template
hash as the final Q8, Q4_K embedding, Q6_K head and mixed Q4_K/Q6_K block weights.
No runtime storage change was needed: K2 already uses the shared quantized kernels.

Before observation, adversarial review accepted a bounded same-artifact screen
instead of a broad fixture campaign or Q4-versus-Q8 equality. Exclusive limits are
max logit error .005, RMSE .001 and reciprocal top-choice regret .001; near ties
are not automatically implementation failures. These are inherited short-screen
engineering targets, not necessary universal quality laws. Frozen Q8 evidence is
unchanged. The screen owns the real production lease/wired gate, refreshes wired
and available-memory checks before serial oracle children, then loads the native
model under its actual memory admission. Full native/reference rows and verdict
are retained, not inferred from a diagnostic process exit.

Evidence: `target/profiles/k2-q4-cpu-preflight.log` and
`target/profiles/k2-q4-screen-98087-1790096166549546000`. All 42 reused short rows
pass, all top choices agree (zero near-tie allowances used), maximum absolute
logit error .00302315 and maximum RMSE .000687908. Four selected embedding rows
(IDs 0, 42, 250019, 250623) match CPU dequantization exactly. This is independent
implementation agreement on correlated short inputs, not a new unseen holdout,
quantization-fidelity, long-history, packed-Q4 or lens qualification claim.

Actual CLI raw smoke, API validation/production lease enabled: six prompt tokens,
16 generated tokens, output starts ` Paris. The capital of Germany is Berlin.`;
stats retained at `target/profiles/k2-q4-raw-smoke.jsonl`. This is a product wiring
check, not comparative performance or answer-quality qualification. Q4 verified
chat identity is deliberately a separate semantic-profile step.

## Packet 43: verified final Q4 chat

The standard Q4_K_M retained-content identity from packet 42 now authorizes the
same final-release no-tools renderer as Q8. Tokenizer, embedded-template and
structural model-config checks plus cancellable exhaustive retained-byte hashing
remain unchanged. The generation-config hash denotes the pinned upstream reference,
not a separate generation-config file authenticated inside the GGUF; product
preparation enforces raw EOS 1 and the renderer uses reference chat stops 1/250019.
Each artifact
reports its own content identity; filename, quantization label, header similarity
or a successful numerical screen alone cannot authorize chat. Unknown compatible
artifacts remain raw-capable. This adds no model-quality or conversion-fidelity claim.

Model-free negative controls independently reject tokenizer/template/content drift
for both identities. The existing CPU artifact fixture test now compares the
returned verified identity with the bytes actually hashed, not a Q8-only expected
value. It passes on Q4, including cancellation and all valid upstream token fixtures.

Actual Q4 CLI evidence: `target/profiles/k2-q4-chat-cli-v1`. Existing checker passes
all three efforts, history/Unicode, input/token identity, rendered/raw controls,
completed low-effort answer, timing and explicit unsupported-control refusals.
Actual Q4 HTTP evidence: `target/profiles/k2-q4-chat-http-v1.json`; existing test
passes JSON/SSE, raw-prefix equivalence, all efforts, completed/incomplete outputs,
abort and session reacquisition. GPU work uses API validation and production lease
ownership; HTTP uses only ephemeral owned loopback sockets. Tools remain next,
not silently interpreted by the no-tools output partitioner.

`target/profiles/k2-q4-chat-paired-v1` also passes completed CLI/HTTP equality.
All-target checking is warning-free; all 26 K2 frontend CPU tests pass. Adversarial
review found no behavioral blocker and requested the reference-generation-config
wording clarification above rather than inventing a new artifact requirement.

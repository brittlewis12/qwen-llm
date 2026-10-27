# Exact CED native range discriminator

Checkpoint based on `0c5b4589eb10feea4fc79cc19f14bea8033eeb70`
(qualified exact baseline plus range execution). The initial checkpoint used
source checks only; follow-up verification is recorded separately. No GPU,
model import, SSH, remote execution, test-profile build or LTO run in this lane.

## Selectors and launch contract

Ignored actual-model selector:

```text
deepseek_v41::forward::ced::qualification::current_pair::range_pilot::range_host_full_model_native_abba
```

New CPU-only selectors (not executed here):

```text
deepseek_v41::forward::ced::qualification::current_pair::range_pilot::frozen_prefix_uses_two_true_next_ids_only
deepseek_v41::forward::ced::qualification::current_pair::range_pilot::preferred_encoder_geometry_uses_actual_chunk_ceiling
deepseek_v41::forward::attention_gpu::tests::query_batch::range_execution::tensor64_parity_geometry_and_shape_budget
```

Use the parent's canonical release lib-test executable with the actual-model
selector and `--exact --ignored --nocapture --test-threads=1`. The parent MUST
enforce a <=60-second whole-process deadline for N2048, or <=90 seconds for an
explicit private N8192 follow-up, including setup and teardown.
Cooperative internal checks cannot interrupt a blocking import, GPU call, hash
or report sync. No automatic retry or reuse of a timed-out state is authorized.

The binary requires aarch64 macOS, little endian, assertions disabled, no
diagnostic features, actual-device NAX/TensorOps availability and the production
tensor route. It never gates on a device name. It acquires its own production
lease fail-fast: `QWEN_METAL_LEASE_WAIT` absent/0, no inherited lease FD or other
runtime QWEN controls. Fixed build metadata, if supplied, must match the binary.
MTL, DYLD, Metal validation/wrapper, Instruments/DT, LLVM/CPU profiler and malloc
diagnostic controls are refused; diagnostic observers must be idle. The parent
must also launch without an attached profiler.

Supply absolute private paths outside this checkout through `V41_CED_GGUF`,
`V41_CED_TOKENS`, and `V41_CED_CURRENT_PAIR_REPORT`. Reports and the automatically
named `.logits.f32le` sidecar must be new files. Artifact identification uses
canonical metadata/geometry, descriptor inventory and retained source FD stamps,
not a filename or artifact byte-count production selector. The parent supplies
the externally identified GSQ artifact; there is no fresh global model content
hash/authenticity claim.

`V41_CED_RANGE_ROWS` is a private fixture choice: absent/2048 selects N2048;
only an explicit 8192 selects N8192. The parent first launches ONLY N2048. There
is no automatic escalation, new production policy or production allowlist.
Reports record the raw control, selected input ranges and planned work counts.

## Frozen experiment

Order is warm Full reference, warm A=old-host exact ShrinkingCED, warm B=range
exact ShrinkingCED, then primary A B B A. All seven selected-length prefills and two real
teacher-forced scalar steps per arm are retained: 14,336 prompt tokens, 14 steps,
21 complete vocabulary vectors (N8192: 57,344 prompt tokens, same 14 steps and
21 vectors). The first N frozen prompt IDs and next two
IDs from prompt concatenated with continuation are selected once; no
retokenization, sampling, feedback, filtering or retries.

One resident GGUF import/binding, one live KV state at a time, capacity N+2,
common default None storage mapping. Preferred encoder width is derived through
the production `prefill_rows_for_resident(planned.bytes.resident_bytes())`:
normally 4096 for GSQ at <=240 GiB resident, matching the qualified baseline.
N2048 then has one actual encoder chunk despite selected width 4096; N8192
normally has two 4096-row chunks. Counts use `N.div_ceil(preferred)`.
Preferred width, selected width and actual chunk count are synced before the
geometry gate. The private experimental control refuses pressure-demoted widths;
it does not pin a width, weaken normal admission or change production planning.
The strict cold plan requests BOTH Full and CED with RequiredFullCed admission,
not a reused Full-only plan. Runtime admission refreshes before each new state;
production allocation and execution gates remain active. Only the existing
test-scoped `query_batch::with_old_host_loop` changes A's host scheduling.

Every CED vector must equal the corresponding Full vector bit-for-bit over the
entire finite vocabulary. Full and old-CED references are retained separately.
Every subsequent CED arm must also match old CED's actual causal observation at
all three positions: 56 region hashes, address history, cursor and readiness,
with per-position exact CED origin and production completion/status validation.
Full/CED causal-state differences are reported diagnostically, not treated as
an inactive-state equivalence requirement. Full-vector bit mismatch metrics
(count, max absolute error, RMSE, first difference) are written before refusal.
No existing Full/CED precision gate is relaxed.

## Budgets and evidence

Retained reference heap is `6 * vocab * 4` bytes (3,102,720 bytes at vocab
129280), plus a charged 64 MiB CPU reserve for current logits, input/final input
parse, state hashes/reports, artifact descriptor staging and raw-file staging.
Preparse admission is 96 MiB. The streamed raw file is `21 * vocab * 4` bytes
(10,859,520 at vocab 129280), not 21 retained heap vectors. Strict cold/runtime
ledgers also charge production request, max(Full,CED), loader controls, binding
CPU/Metal, private shadows and retained resources. Gates are not reservations.

Synced partial reports precede arms, follow prefill/TF API returns, observations
and arms, and retain failure/elapsed time. Raw vectors are synced per position;
owned outputs are sealed read-only on ordinary completion/failure. Hard process
termination can only leave the last synced partial evidence. Timings include
all warmups and primary samples, favorable or unfavorable, without a speed gate.
Prefill clocks include API completion/status/logits but exclude state allocation,
TF, external hashing, report/raw I/O and teardown. Production API-internal
allocation/hashing remains included, not subtracted. Encoded/completed command counters,
publication/selection coverage, no-fallback checks, memory signals/ledgers and
realized retained resources accompany results. `performance_qualified` stays false.

Source identity checks compiled bytes before/after for the fixture, common
current-pair/provenance/input/metrics/report helpers, range/host fixture and its
unchanged weight/seed helpers, admission/planning and maintained production
inventories, plus binary/metallib identity. This is fixed-source evidence, not
a global model authenticity assertion.

Publisher eligibility repair remains out of scope: the inherited unconditional
CDF publication rollback caveat still applies; N8192 is an explicit private
follow-up, not newly qualified by adding the selector. No bounded-window/PV-axis/compact-grid implementation,
Engram, overlap, decoder batch-width or FFN change is introduced. All CPU-test,
Metal, native HOST and actual-model execution remain for the parent. Earlier
M4/5c1b HOST results are not relabeled as release or M5 data.

Initial source checks: rustfmt and whitespace checks; 14 offline model-free V4.1
source-contract checks; acceptance inventory schema and 15 corruption-control
self-tests; identity-gate registry and generated environment inventory checks.
These do not compile/type-check the new Rust tests or validate hardware behavior.
Independent arithmetic of the H64/N128 production shape formulas gives
37,801,992 bytes for the two retained CPU snapshots and 191,618,064 bytes for
conservatively priced buffers (64 KiB surcharge per allocation, including seed
overlap). Adding 128 MiB CPU and 64 MiB Metal reserves gives 392,944,656 bytes,
below the declared 512 MiB ceiling. This is not execution of the Rust helper,
actual-device allocator pricing or measured memory fit; those checks remain
mandatory in the parent's CPU/Metal runs.

## Preferred-geometry follow-up

Parent review identified that the initial fixed-2048 assertion did not match
the normal planner preference for the supplied GSQ resident size. The fixture
now derives that preference from the actual plan and refuses a pressure-demoted
selected width only after syncing the preferred/selected/actual geometry. The
production planner and strict joint admission are unchanged; CedPlan continues
to use the actual selected width. The new CPU test covers selected width larger
than N, N8192 counts and ceiling division at a nonmultiple.

One permitted local `cargo check --offline --release -p qwen-llm --tests` passes
under fail-fast `scripts/bench/with-metal-lease`, using the parent's specified
`/Users/tito/.cache/qwen-llm/cargo-target`. Controller elapsed time is 26.49s,
exit 0, no timeout, signals or remaining owned processes. Raw output and
controller metadata remain private outside the checkout. This type-checks the
new Rust tests; it does not build/link a test executable, run tests, exercise
Metal/model execution or establish native performance. Local warnings include
the unsupported Metal 4.1 compiler probe and existing deprecated/dead/unused
items; no diagnostic reports a new fixture type error. Final rustfmt, whitespace,
acceptance schema/self-tests and environment-reference checks also pass.

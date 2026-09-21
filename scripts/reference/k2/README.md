# K2 checkpoint oracle

Current capacity policy: native applications use checkpoint-declared context and
real device/memory admission, not the historical 256-forward or 7168-position
planner gates. The numerical evidence below remains scoped to its measured lengths;
removing a product cap does not convert short-context tests into full-context proof.
The materialized test backend alone retains its 7168-position score-scratch limit.
F16 remains the application cache. CLI and HTTP no-tools chat are bound to the
verified final artifact and pinned IFM renderer. Raw mode deliberately
does not guess a chat/tool contract for unknown checkpoints.

## Owned allocation check

K2 session/scratch/lens allocations directly zero their owned Shared Metal bytes;
they do not stage a tensor-sized host vector. An opt-in synthetic stress probe
measures allocation wall time, Metal delta and Darwin process RSS high-water:

```sh
MTL_DEBUG_LAYER=1 K2_ALLOCATION_EVIDENCE="$PWD/target/profiles/k2-allocation-new.json" \
cargo test -p qwen-llm --lib \
  k2_horizon_runtime::allocation_tests::gpu_k2_session_allocation_high_water_probe \
  -- --ignored --exact --nocapture --test-threads=1
```

Run alone in a fresh test process with adequate system memory. Its default 32768
capacity allocates 4.5 GiB logical KV (no weights); normal production lease and
memory gates apply. `K2_ALLOCATION_CAPACITY` can lower the probe size. Results do
not qualify model history length or establish a paired performance improvement.

## Prefill readout scheduling

Nonfinal frontend prefill chunks use `K2Session::advance`, preserving residual/KV
finite checks and transaction semantics while skipping the output head and host
logits download. Final prompt chunks and generation transitions still use append.
HTTP retains singleton cancellation points; CLI/bench/lens retain packed chunk
boundaries. Packed scratch remains per-append, not a retained session workspace.

```sh
K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" MTL_DEBUG_LAYER=1 \
cargo test -p qwen-llm --lib \
  k2_horizon_runtime::readout_tests::gpu_k2_advance_skips_readouts_preserves_state_and_poisoning \
  -- --ignored --exact --nocapture --test-threads=1
K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" MTL_DEBUG_LAYER=1 \
K2_READOUT_EVIDENCE="$PWD/target/profiles/k2-readout-pairs-new.json" \
cargo test -p qwen-llm --lib \
  k2_horizon_runtime::readout_tests::gpu_k2_readout_schedule_paired_wall_probe \
  -- --ignored --exact --nocapture --test-threads=1
```

The paired probe counts real head encodes and host downloads separately, requires
bitwise final-logit agreement, and alternates old/new order after warmup. Its
loaded-model synthetic prefill timings exclude setup and session allocation; they
are diagnostic, not end-to-end product performance or independent quality evidence.

## No-tools template checks

The CPU oracle downloads only immutable template/tokenizer/generation metadata,
never weights or remote model code. Jinja renders exact bytes, with output-preserving
generation blocks; the pinned HF tokenizer supplies IDs. Rust removes only the
template-owned leading BOS and enables native BOS insertion. Tests deliberately
include authored marker tokens, Unicode and assistant thinking aliases.

```sh
uv run scripts/reference/generate_k2_chat_fixtures.py
K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
cargo --config 'profile.test.package.qwen-llm.opt-level=1' \
  --config 'profile.test.package.blake3.opt-level=3' test -p qwen-llm --lib \
  k2_horizon_chat::tests::cpu_k2_chat_artifact_identity_and_native_tokens \
  -- --ignored --exact --nocapture --test-threads=1
cargo --config 'profile.dev.package.blake3.opt-level=3' \
  --config 'profile.dev.package.sha2.opt-level=3' build -p qwen-cli --bin qwen
uv run scripts/reference/k2/check_chat_cli.py --binary target/debug/qwen \
  --model "$HOME/models/K2-Horizon-7B-Q8_0.gguf" --output target/profiles/k2-chat-new-run
```

The final script is opt-in GPU execution: serial CLI children take their own
production leases and wired-memory gates with API validation. It checks all three
efforts, assistant history, system/Unicode input, user/messages equivalence,
rendered-input token hashes, unchanged short token fingerprints versus serialized
raw controls, reasoning stderr/final stdout partitioning, incomplete diagnostics,
profile records, a completed answer, and pre-GPU capability refusals. This does not
establish answer quality or tools. See `docs/CLI-UX.md`.

The HTTP check uses the production library lease/memory gate and owned ephemeral
loopback sockets, never a shared server. Pass an absolute evidence path because
Cargo tests run from the crate directory:

```sh
K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" MTL_DEBUG_LAYER=1 \
K2_CHAT_HTTP_EVIDENCE="$PWD/target/profiles/k2-chat-http-new.json" \
cargo --config 'profile.test.package.blake3.opt-level=3' \
  --config 'profile.test.package.sha2.opt-level=3' test -p qwen-cli --bin qwen \
  serve::backend_k2::tests::gpu_verified_k2_chat_http_matches_raw_and_releases_sessions \
  -- --ignored --exact --nocapture --test-threads=1
```

It covers all effort levels, JSON/SSE partition agreement, incomplete reasoning,
completed answers, stop-aware raw-prefix controls, aborts and session reacquisition.
Add `--http-evidence target/profiles/k2-chat-http-new.json` to `check_chat_cli.py`
to compare completed HTTP/CLI text and token counts. These remain wiring checks,
not sustained-service, full-context or cross-checkpoint qualification.

This standalone test harness links the IFM llama.cpp fork at
`42adf019f76013dac873b5b43950d54d5ab27216`. It is not a production dependency,
tokenizer implementation, converter, or HF BF16 oracle. CMake rejects a different
or dirty/untracked reference source tree. Executable identity also binds SHA256
digests of this wrapper and its CMake source. The main engine and its FFI dependency
are unchanged.

Build outside the source checkout:

```sh
cmake -S scripts/reference/k2 -B target/profiles/k2-oracle-build \
  -DLLAMA_CPP_DIR=/absolute/path/to/pinned/IFM/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build target/profiles/k2-oracle-build --target k2_checkpoint_oracle -j 6
```

Run ONLY through the Rust test, which holds the production-exclusive Metal lease
and checks the real wired-memory gate before starting any reference child. The
standalone executable does not acquire the repository lease on its own.

```sh
MTL_DEBUG_LAYER=1 \
K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_LLAMA_ORACLE="$PWD/target/profiles/k2-oracle-build/bin/k2_checkpoint_oracle" \
cargo test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::gpu_checkpoint_logits_match_independent_ifm_fork \
  -- --ignored --exact --nocapture --test-threads=1
```

The parent validates reference build identity and the whole Q8 artifact SHA256,
runs reference children serially before creating native GPU resources, and keeps
logs, full-vocabulary F32 reference rows, a provenance manifest, and native error
metrics under a unique ignored `target/profiles/k2-oracle-*` directory. The binary
protocol binds exact token IDs and absolute positions; there is no hidden BOS,
template, generation, or sampling step. Inputs are four raw IDs, a six-token text
prompt at base 37, and the first 32 tokens of a code fixture at base 128.

Both backends retain F16 K/V. The oracle disables flash attention and RoPE scaling,
reads theta from the model, and uses singleton decoding. Its requested capacity
32 is rounded internally to 256 cache slots; only 1..32 rows are visible in the
original test. Its native capacity is 32. Model geometry/context and request/cache capacity remain
separate. The oracle requests all layers on Metal; retained logs establish actual
offload and device behavior, rather than treating the request as evidence. Exact
normalized log markers are machine-checked for M4 Max, all 37 offloaded layers,
all 36 cache layers on MTL0, F16 K/V, singleton batch sizes, disabled flash
attention, effective capacity 256, theta 10000000, and frequency scale 1.
Raw vocabulary dumps may contain non-UTF-8 bytes; the original logs are retained
and only their ASCII runtime markers are matched after lossy decoding.

The first ten-row measurement used exploratory bounds (max error 0.2, RMSE 0.02,
cosine above 0.99999, matching top-1) and observed max error below 0.00086. Before
extending to 42 rows, bounds were tightened to max error 0.005, RMSE 0.001, cosine
above 0.9999999, and identical top-1. These are scoped regression bounds for this
Q8/F16 corpus, not bitwise, full-context, arbitrary-checkpoint, or HF parity claims.
The 42-row corpus includes the initial screen, so it is not a statistical holdout.
Text inputs use native token IDs: independent text tokenization is established
by the earlier HF fixture suite, not by this token-ID-only oracle.

## Extended qualification and diagnostics

The wrapper accepts up to 256 teacher-forced IDs. With the same environment and
test flags above, the following exact tests exercise additional evidence:

- `k2_horizon_runtime::oracle_tests::gpu_256_token_corpora_match_independent_ifm_fork_and_native_splits`:
  256-token text/code/Unicode prefixes at bases 0/37/8191, unchanged strict gates,
  singleton/split/whole-append bitwise controls, and capacity+1 rejection.
- `k2_horizon_runtime::oracle_tests::gpu_identical_tokens_at_three_bases_diagnostic`:
  identical ledger IDs at those three bases, to separate content from position.
  Records out-of-gate rows without claiming qualification.
- `k2_horizon_runtime::oracle_tests::gpu_first_divergence_layer_diagnostic`:
  all 36 post-block residuals at the first failing corpus rows (lengths 60/34/27).
  Reference `--capture-last` writes a coordinate-bound `.layers` sidecar; the test
  requires ordinary and traced reference logits to agree bitwise at every step.
  This test validates tracing infrastructure, not model parity.

The layer diagnostic also captures post-RoPE Q, all visible post-RoPE K/projected
V rows, and the last attention output at layers 0 and 20. A bounded `.attention`
sidecar binds geometry, layer, full token IDs, base, and exact finite payload.
It replays the native attention kernel on these IFM inputs after F16 K/V rounding,
compares both outputs with independent F64 reductions (F32 graph scale, final
output rounded to F32), and reports actual native-cache differences separately.
Capture hashes and ordinary/traced equality are recorded in `captures.json`.
These six sampled operations do not establish full-model or long-history parity.

The extended reader streams one reference row and retains only 12 native split
checkpoints. Each 256-token experiment writes about 735 MiB of reference logits;
inspect available disk space before running repeated experiments. An absolute
base of 8191 is **not** 8K retained history. New manifests include the compiled
native metallib hash; the callback wrapper remains outside the IFM checkout.

Read-only analysis (no GPU or model load):

```sh
uv run scripts/reference/k2/inspect_oracle_metrics.py target/profiles/k2-oracle-256-RUN
```

Historical strict result: the original 42-row regression passes, but the 256-token extension
fails 234/768 rows under the unchanged bounds despite all top-1 IDs agreeing.
That failed extension did not promote the application limit. The separately frozen
v2 experiment and surface checks later support a guarded **256**-forward budget;
the old failures and strict bounds remain intact. No production arithmetic change
or full-context qualification is implied.

## Bounded-scratch attention

The separate K2 H128/GQA4 online primitive retains all F16 K/V and uses constant-
size working state instead of materializing one score per visible position. It
is now selected by the guarded runtime after independent-reference and surface
checks. It does not change KV bytes/token or public caps.
The materialized test backend retains its 7168-position score-scratch limit; the
online plan uses declared context. Synthetic coverage now reaches 8192:

```sh
MTL_DEBUG_LAYER=1 cargo test -p qwen-llm --lib \
  k2_horizon_metal::tests::gpu_online_attention_matches_f64_and_materialized_with_future_poison \
  -- --ignored --exact --nocapture --test-threads=1
```

The test acquires the production lease/real wired gate, prices its buffers, and
checks all heads, GQA mapping, nonzero offsets, poisoned future/guard rows, cache
immutability, flat/sharp scores, and independent F64/materialized controls with
predeclared 2e-5 absolute limits. The layer diagnostic also replays this candidate
on captured IFM inputs and records separate `online_vs_*` metrics. No claim of
whole-model parity, speed improvement, or compact KV storage follows from these
primitive checks.

A temporary full-model online-dispatch experiment passes the original 42 rows
but fails 215/768 extended rows under unchanged gates. It was reverted, not
promoted. New manifests also record native runtime/primitive source SHA256 values
to distinguish host-only dispatch experiments sharing the same metallib.

A later native-only integration regression introduced an immutable, test-only
backend selector. After default promotion, materialized attention remains an
explicit test-only control and public loading selects online attention. The test uses
the four previously seen v2 corpora at their selected bases, not a new holdout, and
applies the unchanged v2 gates prospectively to online versus materialized results:

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_HOST_BUILD_NOTE="temporary qwen-llm test-package opt-level=1; Metal unchanged" \
cargo --config 'profile.test.package.qwen-llm.opt-level=1' test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::holdout::online::gpu_online_runtime_matches_materialized_256_controls \
  -- --ignored --exact --nocapture --test-threads=1
```

Allow about 2 GiB for retained native controls. The production lease and real wired
gate cover both serial model lifetimes; the materialized model is dropped before
online loading. Baselines reuse the checked row protocol but are explicitly labeled
native, with SHA256/token/coordinate/policy bindings. The test checks 2048 rows, 16
capture sites, four online singleton/split/whole controls, capacity+1 refusals, and
64 exact predictors on materialized-derived EOS-inclusive trajectories. Its first
run passes with no top-1 disagreement or indeterminate allowance. This is not
independent-oracle qualification or performance evidence; default promotion uses
the separate gates described below.

The next gate reuses the original independent v2 reference, without executing its
binary or producing another multi-GiB logit dump. The loader pins the original
manifest/references SHA256 in source before replay, validates every payload hash
and full coordinate/token/finite-row protocol, and rechecks trace noninterference.
Original logs have exact mode/device marker checks, not digest authentication.

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_LLAMA_ORACLE="$PWD/target/profiles/k2-oracle-build/bin/k2_checkpoint_oracle" \
K2_RETAINED_V2="$PWD/target/profiles/k2-holdout-v2-12878-1789830419367474000" \
K2_HOST_BUILD_NOTE="temporary qwen-llm test-package opt-level=1; Metal unchanged" \
cargo --config 'profile.test.package.qwen-llm.opt-level=1' test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::holdout::retained::gpu_online_against_retained_independent_v2_reference \
  -- --ignored --exact --nocapture --test-threads=1
uv run scripts/reference/k2/inspect_holdout.py target/profiles/k2-online-retained-v2-RUN
```

`K2_LLAMA_ORACLE` is hashed only. This replay is a regression on previously observed
fixtures against an independent implementation, not a new statistical holdout. It
passes all 4096 rows, 16 sites, 12 partition controls, and 64 exact predictors with
zero top-1 differences or indeterminate allowances. Default/surface promotion is
a separate checkpoint, not automatic on replay success. New evidence is
small metric/manifest output; the original reference directory must remain intact.

The guarded online default subsequently passes the original strict 42-row oracle,
native forward/capture/readout/intervention/transport checks, and all run/bench/
plain+imported lens/JSON+SSE boundary checks. Fresh v1/v2 holdout execution and the
cache-precision diagnostic explicitly retain their historical materialized control.
Other model families, shaders, the 256 application guard, and F16 KV allocation are
unchanged. These are correctness gates, not speed or long-context claims.

## Exact-arithmetic packed Q8 baseline

The packed prefill mode schedules up to 32 rows layer-major through the
same block graph as singleton execution. It reuses the token-axis Q8 GEMV kernel's
singleton `lcpp` arithmetic, not a half-staged GEMM or Qwen routing heuristic.
Grouped norm, RoPE, F16 store and causal online attention remain rowwise; only the
final appended row receives captures/interventions/readout. All IDs are validated
before upload, all new KV and final residual rows are checked after each command,
and the whole append publishes its prefix once. Submitted failure poisons it.

```sh
MTL_DEBUG_LAYER=1 cargo --config 'profile.test.package.qwen-llm.opt-level=1' \
  test -p qwen-llm --lib \
  k2_horizon_metal::tests::gpu_q8_batch_projection_matches_singleton_bits_and_guards \
  -- --ignored --exact --nocapture --test-threads=1
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  cargo --config 'profile.test.package.qwen-llm.opt-level=1' \
  --config 'profile.test.package.sha2.opt-level=3' test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::holdout::packed::gpu_packed_q8_matches_serial_captures_cache_partitions_and_interventions \
  -- --ignored --exact --nocapture --test-threads=1
```

Both probes acquire the production lease and real wired gate. The synthetic test
checks all four projection shapes at 1/2/31/32 rows, offsets/guards and immutable
inputs/weights. The full-model test reuses pinned, previously seen v2 inputs, not
a new holdout. It passes 88 bitwise checkpoints including the full poisoned-future
cache, readout isolation, ordered interventions, continuation and an actual late
nonfinite failure. Temporary scratch is 237572 bytes/row (7602304 bytes at 32),
separately priced/admitted before submission, with no replicated KV or logits slab.
The extra SHA optimization only accelerates host cache digests. These tests make no
speed or KV-savings claim. After the separate default gate, run/bench/lens select
packed mode only for all 252 Q8 block projections with lcpp enabled, otherwise
serial execution. Singleton decode and serving's per-token cancellation stay serial.
Lens shutdown checks occur between chunks (up to 32 positions), not every token.
Run stats (`diagnostics.k2_horizon.prefill`), bench (`method.prefill_execution`), and
lens (`deployed_model.prefill`) report mode, actual maximum chunk size, command count,
and logical temporary activation bytes separately from persistent session storage.

With the same `K2_GGUF`, `K2_LLAMA_ORACLE` (hash only), `K2_RETAINED_V2`, and lease
environment as the retained replay above, run
`k2_horizon_runtime::oracle_tests::holdout::retained::gpu_packed_against_retained_independent_v2_reference`
to select packed mode for split/whole controls. The first replay passes 192 endpoint
comparisons across twelve partition controls, using 72 multi-token appends plus
singleton boundaries. Its 4096 independent-reference rows, 16 capture sites, and
64 exact trajectory predictors remain singleton baseline checks, not 4096 packed
outputs. No oracle child or new large reference dump is produced. Fresh historical
v1/v2 tests remain explicitly materialized/serial; the online-only replay stays serial.

## Experimental compact cache

`COMPACT-KV-POLICY.md` specifies the K2-only 34-byte block format, invalid/underflow
rules, and predeclared primitive gates. The storage-aware plan uses raw I8 byte
arenas for Q8, not weight-tensor views. Byte-addressed attention dequantizes only
four values per lane into registers; no retained history is evicted or expanded.
The private runtime actually allocates 78336 bytes/token, versus 147456 for F16.
Public loading remains F16; compact KV has no CLI/environment selector.

```sh
MTL_DEBUG_LAYER=1 cargo --config 'profile.test.package.qwen-llm.opt-level=1' \
  test -p qwen-llm --lib \
  k2_horizon_metal::compact::tests::gpu_q8_store_bytes_and_inline_attention_match_independent_controls \
  -- --ignored --exact --nocapture --test-threads=1
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  cargo --config 'profile.test.package.qwen-llm.opt-level=1' test -p qwen-llm --lib \
  k2_horizon_runtime::compact_tests::gpu_compact_cache_preserves_transactions_causality_and_lens \
  -- --ignored --exact --nocapture --test-threads=1
```

Both probes acquire the production lease and real wired gate. The primitive checks
exact curated quantizer bytes, bounded general quantization error, all-head F64
attention through 257 positions, offsets/guards/poisoned future, and immutable
inputs/cache. Runtime checks cover actual bytes, pre-encoder layout rejection,
singleton-before-packed execution, exact within-Q8 captures/cache/continuation,
readout isolation, ordered interventions, capacity refusal, and genuine nonfinite
input producing a visible sentinel and poisoned unpublished append. These checks
do not establish full-model quality equivalence or speed. F16 strict regression
and forward/lens controls remain separate and pass.

## Cache/backend precision control

`--f32-kv` is a reference-only diagnostic, mutually exclusive with `--capture-last`.
Default reference and native execution remain F16. The binary header and exact
runtime markers must agree on actual cache precision; F32 reference logs must show
72 MiB of F32 K/V in 256 cells. This is not a full-precision-weight/HF oracle, and
cache dtype may also change backend kernels/reduction topology.

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_LLAMA_ORACLE="$PWD/target/profiles/k2-oracle-build/bin/k2_checkpoint_oracle" \
cargo test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::cache_precision::gpu_f16_native_and_ifm_against_f32_cache_control \
  -- --ignored --exact --nocapture --test-threads=1
uv run scripts/reference/k2/inspect_cache_precision.py target/profiles/k2-cache-precision-RUN
```

Check free disk space first: two reference modes across three 256-token corpora
write about 1.5 GiB. Children are serial under the parent's production lease and
finish before native GPU residency. Probability metrics use F64 stable log-softmax:
KL(reference || actual), TV = half the L1 probability distance, and RMSE after
centering logit differences. Tiny negative KL from floating-point summation is
not silently clamped. The analyzer rejects incomplete/mismatched corpus rows.
These metrics characterize sensitivity; no acceptance thresholds or public caps
change based on this diagnostic alone.

## Frozen guarded-context holdout

`HOLDOUT-POLICY.md` and hash-bound `holdout-256-v1.json` were reviewed and committed
before target execution. This is a synthetic engineering holdout, not a random
scientific benchmark. It binds checkpoint/tokenizer/token IDs, per-row gates,
capture sites, append boundaries, and reference argmax trajectories. The fixed
`--greedy-15` reference mode is mutually exclusive with other modes and never
exceeds 256 rows. Its EOS-inclusive trajectory is not a serving stop-policy test.

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_LLAMA_ORACLE="$PWD/target/profiles/k2-oracle-build/bin/k2_checkpoint_oracle" \
cargo test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::holdout::runner::gpu_frozen_guarded_256_holdout \
  -- --ignored --exact --nocapture --test-threads=1
uv run scripts/reference/k2/inspect_holdout.py target/profiles/k2-holdout-v1-RUN
```

Allow roughly 5 GiB of disk evidence and more than ten minutes for the instrumented
debug-host run. The first v1 result is **failed**, with two distinct teacher-forced
top-1 disagreements (one duplicated in a supplied trajectory prefix). Numerical,
capture, split/whole, and all generated-tail checks passed. The analyzer reports
reference-side ranking gaps without guessing unretained native logits. Neither
fixtures nor gates may be retuned after observing this result. V1 did not promote
the public cap; the later v2/surface evidence is separately recorded.

V2 is separately frozen in `HOLDOUT-POLICY-V2.md` / `holdout-256-v2.json`, not a
reinterpretation of v1. It records reciprocal live ranking witnesses and keeps
all trajectory predictor lengths 241-256 exact. The first run passes all 4096
rows and 16 sites without using any indeterminate-ranking allowance. Scope is
the pinned Q8-weight checkpoint with F16 KV, not F16-weight qualification.

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_HOST_BUILD_NOTE="temporary qwen-llm test-package opt-level=1; Metal unchanged" \
K2_LLAMA_ORACLE="$PWD/target/profiles/k2-oracle-build/bin/k2_checkpoint_oracle" \
cargo --config 'profile.test.package.qwen-llm.opt-level=1' test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::holdout::v2::gpu_frozen_guarded_256_v2_holdout \
  -- --ignored --exact --nocapture --test-threads=1
uv run scripts/reference/k2/inspect_holdout.py target/profiles/k2-holdout-v2-RUN
```

The host-only optimization is optional and does not alter Metal kernels. The
manifest binds the native test binary as well as source/kernel identities. Passing
the holdout does not itself promote public limits; surface checks remain separate.

## Request capacity and long-history checks

This separate check exercises 1024-forward requests and 257-token response budgets
through run, request bench, and plain/imported lens. Rebuild after any tracked
source changes: `--allow-dirty` does not bypass benchmark binary/source identity.

```sh
cargo --config 'profile.dev.package.blake3.opt-level=3' \
  --config 'profile.dev.package.sha2.opt-level=3' build -p qwen-cli \
  --bin qwen --bin qwen-bench --bin qwen-lens
uv run scripts/reference/k2/check_guarded_capacity.py \
  --qwen target/debug/qwen --bench target/debug/qwen-bench \
  --lens target/debug/qwen-lens --model "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --capacity 1024 --output target/profiles/k2-context-1024-new-run
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  K2_BOUNDARY_EVIDENCE="$PWD/target/profiles/k2-context-1024-new-run" \
  cargo test -p qwen-cli --bin qwen \
  serve::backend_k2::tests::gpu_context_json_sse_match_run_bench_and_reject_capacity_plus_one \
  -- --ignored --exact --nocapture --test-threads=1
```

The first command chain checks the selected capacity, generation transitions,
output fingerprints, full-logit identity transport, strict asset binding, explicit
transfer, invalid positions, and requested-capacity/context refusals. The serving follow-up reads
its benchmark evidence, checks direct/JSON/SSE parity and explicit startup defaults,
and verifies fresh state after a late prefill abort. Nonstream overbudget requests
return HTTP 400; SSE sends `response.failed` after HTTP 200 with no generated text.
Every GPU child/probe owns its production lease and real wired-memory gate with API
validation; do not acquire an outer lease. Evidence is pinned final Q8_0 weights
with F16 KV on M4 Max, not a speed, other-checkpoint, or long-context qualification.

The synthetic online attention test listed earlier now covers 7168/7169/8192
visible rows, flat/sharp scores, and separated block maxima against F64 using
unchanged 2e-5 bounds. Histories above 256 use 256-row online summaries merged in
registers; the shorter kernel stays unchanged. No history is dropped or expanded.
Runtime self-controls at 257/1024 rows and high absolute positions are separate:

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
cargo --config 'profile.test.package.qwen-llm.opt-level=1' \
  --config 'profile.test.package.sha2.opt-level=3' test -p qwen-llm --lib \
  k2_horizon_runtime::context_tests::gpu_checkpoint_context_boundaries_and_memory_admission \
  -- --ignored --exact --nocapture --test-threads=1
```

This test prices (but never allocates) the full declared-context cache. It checks
1024-token singleton/split/whole cache/capture/readout/continuation equality at
base zero and near the declared end. High absolute position is not long retained
history, and self-consistency is not independent full-context numerical evidence.

## Native lens CLI checks

The opt-in uv scripts run native CLI children serially. Unlike the standalone
llama.cpp wrapper, each CLI child acquires its own production lease and real
memory gate; do not wrap these scripts in an outer lease. Both scripts force
Metal API validation, retain stdout/stderr/commands, and require new evidence
directories. A busy lease is a failure, never permission to stop another owner.

```sh
cargo --config 'profile.dev.package.blake3.opt-level=3' \
  --config 'profile.dev.package.sha2.opt-level=3' build -p qwen-cli --bin qwen-lens
uv run scripts/reference/k2/check_plain_lens_cli.py \
  --binary target/debug/qwen-lens --model "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --output target/profiles/k2-plain-lens-new-run
uv run scripts/reference/k2/check_imported_lens_cli.py \
  --binary target/debug/qwen-lens --model "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --output target/profiles/k2-imported-lens-new-run
```

Host digest optimization avoids long debug hashing of the full retained checkpoint
on each imported request; it does not change Metal kernels. The plain check uses
the final tokenizer text fixture, all 36 sites, serialized BOS, literal IDs with
an explicitly unexecuted suffix, requested site order, vectors and binary bundles.
The imported check creates only synthetic identity/nonsymmetric F16 matrices under
the existing data-only schema; this is neither checkpoint conversion nor fitting.
It compares complete identity-transport logits bitwise, checks orientation,
exact binding versus explicit unvalidated transfer, and pre-Metal refusals. Assets
target final post-block layer 35 and are hashed/bound before native execution.
No claim of fit quality, cross-checkpoint scientific equivalence, or long context.

## Compact-cache diagnostic

The private Q8 cache experiment uses the frozen storage contract in
`COMPACT-KV-POLICY.md`. Compare it with native F16 online attention (not an
independent implementation) on four previously seen v2 cases:

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
K2_HOST_BUILD_NOTE="temporary qwen-llm opt-level=1 and sha2 opt-level=3; Metal unchanged" \
cargo --config 'profile.test.package.qwen-llm.opt-level=1' \
  --config 'profile.test.package.sha2.opt-level=3' test -p qwen-llm --lib \
  k2_horizon_runtime::oracle_tests::holdout::compact::gpu_compact_cache_256_quality_and_packed_controls_diagnostic \
  -- --ignored --exact --nocapture --test-threads=1
uv run scripts/reference/k2/inspect_compact.py target/profiles/k2-compact-diagnostic-RUN
uv run scripts/reference/k2/test_inspect_compact.py
```

Allow about 1.1 GiB of evidence disk space. The parent owns the production lease
and real wired gate. F16/Q8 model lifetimes do not overlap. Reports include actual
logical arena bytes, Metal allocator-reported `allocatedSize`, and session
allocation deltas separately; none proves total physical residency. Hard failures
cover storage/transactions, finiteness, Q8 serial/packed/cache/capture equality,
and synthetic identity transport. Quality failures are reported, not test panics:
**read `summary.json`; a successful test is not quality qualification.**

The first diagnostic fails unchanged v2 gates on 808/1088 rows and 8/16 captures.
Seven teacher-forced top-1 choices differ; 64/64 F16-derived trajectory predictors
agree, with 46 of those rows still failing numerical gates. Measured KV allocation
falls 46.875%, but Q8 remains private and F16 stays default. See review packet 33
for evidence and extrema. The inspector verifies provenance/coverage/counts and
reports recorded metrics; it does not reconstruct absent candidate logits.

## Request benchmark accounting

```sh
cargo build -p qwen-cli --bin qwen-bench
uv run scripts/reference/k2/check_request_bench.py \
  --binary target/debug/qwen-bench --model "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --output target/profiles/k2-request-bench-new-run
```

This is an API-validated correctness smoke of the benchmark's accounting, not a
speed measurement. The dedicated `k2-request` lane uses raw native input, greedy
sampling and exact EOS 1, within configured capacity and checkpoint context. It reports host-wall request phases
with guard/readback overhead, not GPU-command timings or llama-bench pp/tg results.
The checker records its explicit dirty-build override, tests warmup/repeat identity,
raw-run fingerprint parity, BOS/literal input, zero-transition rates, timing/prefix
counts, and pre-Metal errors. Each child owns its normal production lease.

## Raw serving correctness

The canonical user contract lives in `docs/SERVE.md#k2-horizon-raw-profile`, not a
separate serving manual. This opt-in probe uses direct borrowed-backend calls and
ephemeral loopback JSON/SSE connections, without starting a long-running server:

```sh
MTL_DEBUG_LAYER=1 K2_GGUF="$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  cargo test -p qwen-cli --bin qwen \
  serve::backend_k2::tests::gpu_borrowed_backend_matches_raw_run_and_discards_aborted_requests \
  -- --ignored --exact --nocapture --test-threads=1
```

It checks raw-run parity, native BOS controls, optional stats, budget refusal, and
fresh-request behavior after prefill/generation aborts. The CLI test links the
library without `cfg(test)`, so its `MetalContext::new` already takes the production
lease and real wired-memory gate. Do not acquire a second/outer lease. This differs
from the isolated-context behavior of the library's own unit tests. Evidence is
short-context final-Q8 correctness, not sustained-service or performance evidence.

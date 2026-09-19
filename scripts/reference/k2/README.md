# K2 checkpoint oracle

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

Current result: the original 42-row regression passes, but the 256-token extension
fails 234/768 rows under the unchanged bounds despite all top-1 IDs agreeing.
The public application limit remains **32**, not 256. See the development review
for controlled experiments and unresolved numerical questions; no relaxed gate,
production arithmetic change, or longer-context qualification is included.

## Bounded-scratch attention candidate

The separate K2 H128/GQA4 online primitive retains all F16 K/V and uses constant-
size working state instead of materializing one score per visible position. It
is not selected by the runtime and does not change KV bytes/token or public caps.
The typed plan retains its existing 7168-position source ceiling; that ceiling
is not numerical qualification. Synthetic coverage currently reaches 257:

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
fixtures nor gates may be retuned after observing this result. Public cap stays 32.

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

## Request benchmark accounting

```sh
cargo build -p qwen-cli --bin qwen-bench
uv run scripts/reference/k2/check_request_bench.py \
  --binary target/debug/qwen-bench --model "$HOME/models/K2-Horizon-7B-Q8_0.gguf" \
  --output target/profiles/k2-request-bench-new-run
```

This is an API-validated correctness smoke of the benchmark's accounting, not a
speed measurement. The dedicated `k2-request` lane uses raw native input, greedy
sampling, exact EOS 1, and at most 32 forwards. It reports host-wall request phases
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

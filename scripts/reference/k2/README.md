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
32 is rounded internally to 256 cache slots; only 1..32 rows are ever visible.
Native capacity is 32. Model geometry/context and request/cache capacity remain
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

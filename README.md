# qwen-llm

An experimental native Metal inference, serving, benchmarking, and
workspace-lens toolkit for local GGUF models on Apple Silicon. The engine is
written in Rust and includes custom Metal kernels for dense, mixture-of-experts,
and hybrid recurrent/attention architectures.

## Status

This is experimental research software at version `0.0.1`. Support and evidence
are specific to model identity, quantization, execution topology, and measured
benchmark cell. The project distinguishes bitwise, numerical, greedy-semantic,
distributional, and approximate results; it does not claim blanket output
equivalence or universal performance superiority over `llama.cpp`.

Current model paths include:

- Qwen3.5 and Qwen3.6 dense and MoE GGUFs (`qwen35` / `qwen35moe`), including
  metadata-validated Qwen3.8 27B profiles.
- The exact released Qwen3.8 Flash-Next `qwen4exp` profile.
- DeepSeek V4 Flash-0731 `deepseek4` profiles.
- The released Muse Glimmer 30B profile with its supported chat template and
  uniform Q8_0 or BF16 matrix storage.
- K2 Horizon dense 7B (`k2-horizon`): native raw execution, forward-only lens,
  and verified final-artifact native CLI/HTTP chat and tools (Q8_0/Q4_K_M).
  Rendering is native Rust, without dynamic template dependencies. Numerical evidence is scoped
  to the documented final Q8_0/F16-KV corpus, not every admitted checkpoint/context.

Model-family and request capabilities are intentionally strict. Unsupported
architectures, templates, and execution combinations fail closed rather than
silently selecting a fallback contract.

K2 `qwen info --json` separates family implementation from artifact CPU admission;
admitted lanes are `conditional` until actual request/device/memory checks pass.
Verified chat inspection hashes retained checkpoint bytes on the CPU, with
cooperative cancellation. See [current K2 scope](docs/K2-HORIZON-PLAN.md).

## Build

Requires Apple Silicon macOS, Rust 1.88 or newer, and the Xcode command-line
tools for `xcrun metal` and `xcrun metallib`. Cargo fetches the pinned public
`gguf-rs` and `llama-cpp-sys-2` sources declared in `Cargo.toml`.

```sh
cargo build --release -p qwen-cli --bin qwen
./target/release/qwen --info
```

## Inference

Use `run` for an ordinary model-templated request:

```sh
./target/release/qwen run -m MODEL --user "Explain this"
./target/release/qwen run -m MODEL --system "Be concise" --user "Explain this"
./target/release/qwen run -m Qwen3.8-27B.gguf --reasoning-effort low --user "Explain this"
./target/release/qwen run -m Qwen3.8-Flash-Next.gguf --no-thinking --user "Explain this"
./target/release/qwen run -m Muse-Glimmer-30B-Q8_0.gguf --reasoning-effort xhigh --user "Explain this"
./target/release/qwen run -m DeepSeek-V4-Flash.gguf --reasoning-effort high --user "Explain this"
```

Structured messages accept either a strict bare array or
`{ "messages": [...], "tools": [...] }` with OpenAI-shaped tool definitions,
assistant `tool_calls`, `tool` results, and `reasoning_content` on pinned Qwen
templates. Muse accepts the shared ATEM wrapper with tools, reasoning history,
and tool results. `-` reads one complete JSON document from
stdin:

```sh
./target/release/qwen run -m MODEL --messages messages.json
./target/release/qwen run -m MODEL --messages -
```

Raw model input remains explicit:

```sh
./target/release/qwen run -m MODEL --raw-prompt '<exact model input>'
```

Muse Q8_0 on unified Apple M4 Max automatically uses optimized generation math
in the CLI, serving, and library runtime. Decode uses partitioned
attention when a layer sees at least 1024 KV positions, throughout the admitted
model context. Shorter visible ranges keep the existing path. The session admits
an additional 528 KiB scratch buffer.
Prefill, sampling policy, reasoning, and stop handling are unchanged. This is
tolerance-qualified arithmetic, not bitwise or seed-for-seed sampled equivalence;
set `QWEN_MUSE_SPLIT_DECODE=0` to roll back CLI decode to original math.

The same qualified lane defaults to matrix-based
packed prefill with tiled F32 attention for full 128-token chunks and row-parallel
online attention for smaller packed remainders, throughout the admitted model
context (131072 tokens for the released profile). There are no benchmark-length
cutoffs; context/capacity, geometry and buffer validation remain enforced. It adds
no session buffers and composes with split decode. The tiled kernel
reuses KV across query heads/rows. The smallest chunk loses the shape screen;
remainders conservatively keep online attention. The measured 32K late-chunk wall saving
is 17.95%, not a complete fresh-prompt speedup claim.
Logs report `packed_matrix_online`
instead of `packed_exact`. Scalar-tail kernels are unchanged, but consume the
numerically changed matrix-prefilled KV; this is not bitwise or sampled-output
equivalence. `QWEN_MUSE_MATRIX_PREFILL=0` rolls back CLI prefill.

Serving has independent rollback variables `QWEN_SERVE_MUSE_SPLIT_DECODE=0` and
`QWEN_SERVE_MUSE_MATRIX_PREFILL=0`. Unset or `1` permits qualified math, not force-on
for unsupported hardware; other values fail. BF16 and other devices retain original
math. Logs report effective selection. Lens/fit tools and the reference
`muse-request` benchmark explicitly retain original math to preserve their contracts.
Model-context invariants have independent attention checks through131072 tokens;
whole-model numerical evidence reaches32K plus512 teacher-forced transitions, not
whole-model131K equivalence. See `docs/bench/2026-09-17-muse-defaults/PROTOCOL.md`.

Qwen3.8 execution is text-only; image/projector execution is unavailable.
Qwen3.8 Flash-Next accepts only the exact released architecture contract and a
serial single-turn generation lane. Multi-token prompts request packed prefill
by default and log any fallback to scalar admission. JSONL batching, serving,
prefix or durable caches, and DFlash are not supported for Flash-Next.

Flash-Next automatically uses qualified singleton optimizations on Apple M4 Max:
parallel N512/K10 expert selection with serial fallback for nonfinite inputs, and
Q8 HC up-plus-mix. Both CLI and library defaults enable these routes after pipeline
capability checks; other devices retain incumbent execution. Neither adds GPU
scratch or changes packed kernels. Top-k qualification is bitwise; HC is numerical.

`QWEN4EXP_GUARDED_TOPK=0` and `QWEN4EXP_HC_UP_MIX=0` independently roll back these
defaults. Unset or `=1` allows qualified execution, not force-on for unsupported
hardware; other values fail. HC's final guarded-router/incumbent-QSA comparison
saves6.56% GPU time and5.73% executor wall time. These are bounded continuation
measurements, not a general request-throughput guarantee or additive gain claim.

The split-QSA production option is removed after a failed compatibility guardrail
and one unsuccessful repair timebox. Its environment variable has no effect;
production uses incumbent QSA without split scratch. Failed attempts and earlier
measurements remain documented in `docs/bench/2026-09-16-flash-defaults/RESULT.md`.

The existing strict-order packed router now also defaults on at exact1024 rows,
alongside512,527,2048, within its qualified M4 Max geometry. The natural1024-token
packet saves14.77% prefill GPU time with bitwise endpoint/state agreement; no shader
or new switch is added. `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0` remains the rollback.
See `docs/bench/2026-09-16-flash-defaults/ROUTER-1024-RESULT.md` for scope and evidence.

Run `qwen -h` for the common interface or `qwen --help` for the expanded
research and diagnostics surface.

## Serving

`qwen serve -m MODEL` starts a resident, loopback-only Open Responses subset
with `POST /v1/responses` and `GET /v1/models`. It supports Qwen3.5/3.6-family
models, validated Qwen3.8 27B profiles, DeepSeek V4, and Muse Glimmer, but not
Flash-Next.

Serving is single-flight: concurrent connections receive `503` with
`Retry-After: 1`. Qwen and DeepSeek use bounded in-memory snapshot caches; Muse
and K2 reuse the longest common token prefix of their live session by default
(`QWEN_MUSE_PREFIX_REUSE=0` / `QWEN_K2_PREFIX_REUSE=0` disable).
Qwen and DeepSeek V4 also keep warm prefixes across restarts in a durable disk
tier (default `~/.cache/qwen-llm/serve-checkpoints`, `--durable-snapshot-dir off`
disables; the first start on a model hashes its GGUF in the background). See
`docs/SERVE.md` for the write policy and flags.
Muse Q8_0 on unified Apple M4 Max defaults to matrix/tiled prefill and
split-position decode with admitted528 KiB scratch. Separate rollback controls
`QWEN_SERVE_MUSE_MATRIX_PREFILL=0` and `QWEN_SERVE_MUSE_SPLIT_DECODE=0` disable each.
Unset/`1` permits qualified execution; invalid values fail startup. The CLI
math switches do not implicitly affect serving. Reuse matches tokens exactly,
but optimized arithmetic is tolerance-qualified, not bitwise or sampled-exact.
The exact protocol and capability matrix live in [`docs/SERVE.md`](docs/SERVE.md).

## Workspace Lens

The `qwen-lens` binary provides live J/R workspace-lens readouts and ordered
post-block interventions. Its commands cover singleton runs, resident ordinary-
Qwen cohorts, coefficient sweeps, J/R fitting and import, full-vocabulary
traces, and model-free artifact inspection and comparison. Cohorts retain model
residency but do not share KV, sampler, or scratch state between requests.

See [`docs/LENS-RUN.md`](docs/LENS-RUN.md) for the run contract and examples.

## Benchmarking

Build and run the benchmark CLI with an explicit subcommand:

```sh
cargo build --release -p qwen-cli --bin qwen-bench
./target/release/qwen-bench suite \
  -m /path/to/model.gguf \
  --pp 512 --tg 128 --runs 3 -o json
```

Benchmark methodology is documented in [`docs/BENCH.md`](docs/BENCH.md).
Current priorities and measured outcomes live in
[`docs/PERF-ROADMAP.md`](docs/PERF-ROADMAP.md),
[`docs/PERF-LOG.md`](docs/PERF-LOG.md), and versioned evidence packets under
[`docs/bench/`](docs/bench/). [`docs/PLAN.md`](docs/PLAN.md) is the historical
architecture plan.

## Layout

```text
crates/qwen-llm/  engine library
crates/qwen-cli/  qwen, qwen-bench, qwen-tok, qwen-census,
                  qwen-grammar-oracle, and qwen-lens
kernels/          Metal compute shaders compiled into the embedded product metallib;
                  kernels/research/ holds bench-only probes in a second metallib
                  loaded lazily on first use
docs/             runtime contracts, research notes, and evidence packets
```

The generated [environment knob reference](docs/ENV.md) lists every engine
environment variable, its default behavior, source location, and module family.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option. Adapted third-party components are
documented in [`docs/THIRD-PARTY-NOTICES.md`](docs/THIRD-PARTY-NOTICES.md).

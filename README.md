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

Model-family and request capabilities are intentionally strict. Unsupported
architectures, templates, and execution combinations fail closed rather than
silently selecting a fallback contract.

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

Muse Q8_0 on unified Apple M4 Max has an opt-in decode path:
`QWEN_MUSE_SPLIT_DECODE=1 ./target/release/qwen run ...`. It uses partitioned
attention for generated-token positions 1024 through 32783, retaining the existing
path outside that range. The session admits an additional 528 KiB scratch buffer.
Prefill, sampling policy, reasoning, and stop handling are unchanged. This is
tolerance-qualified arithmetic, not bitwise or seed-for-seed sampled equivalence;
omit the variable or set it to `0` for the original math. This CLI opt-in does not
enable split decode in serving or lens workflows.

`QWEN_MUSE_MATRIX_PREFILL=1` opts Muse Q8_0 on the same device into matrix-based
packed prefill with row-parallel online attention for chunks ending at or before
absolute position 32768. Larger session reservations are allowed; chunks crossing
or beyond the qualified boundary use the original kernels. It adds no session
buffers and composes with the split-decode opt-in.
Logs report `packed_matrix_online`
instead of `packed_exact`. Scalar-tail kernels are unchanged, but consume the
numerically changed matrix-prefilled KV; this is not bitwise or sampled-output
equivalence. Both options remain off by default.

Qwen3.8 execution is text-only; image/projector execution is unavailable.
Qwen3.8 Flash-Next accepts only the exact released architecture contract and a
serial single-turn generation lane. Multi-token prompts request packed prefill
by default and log any fallback to scalar admission. JSONL batching, serving,
prefix or durable caches, and DFlash are not supported for Flash-Next.

Run `qwen -h` for the common interface or `qwen --help` for the expanded
research and diagnostics surface.

## Serving

`qwen serve -m MODEL` starts a resident, loopback-only Open Responses subset
with `POST /v1/responses` and `GET /v1/models`. It supports Qwen3.5/3.6-family
models, validated Qwen3.8 27B profiles, DeepSeek V4, and Muse Glimmer, but not
Flash-Next.

Serving is single-flight: concurrent connections receive `503` with
`Retry-After: 1`. Qwen and DeepSeek use bounded in-memory snapshot caches; Muse
currently reports no cache reuse. Cross-restart durable warmth is not wired.
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

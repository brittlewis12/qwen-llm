# qwen-llm

A local inference engine for Apple Silicon Macs, written in Rust with its own
Metal kernels. It runs GGUF models from several families and includes a
command-line runner, an HTTP server for local clients, interpretability
("lens") tools and benchmarks.

The project started with Qwen models, which is where the name and the `qwen`
commands come from. It now runs other families too.

It is experimental (version 0.0.1): supported models, flags and output formats
change often. Most tuning has been done on an M4 Max, and some faster kernels
are currently enabled only on that chip; other Apple Silicon Macs run the
general versions.

## Supported models

| Family | GGUF architecture |
|---|---|
| Qwen3.5, Qwen3.6 and Qwen3.8, dense and mixture-of-experts | `qwen35`, `qwen35moe` |
| Qwen3.8 Flash-Next | `qwen4exp` |
| DeepSeek V4 Flash-0731 | `deepseek4` |
| GLM-5.3-Flash | `glm5-next` |
| Muse Glimmer 30B | `muse-glimmer` |
| K2 Horizon 7B | `k2-horizon` |

Input is text only. What a given file supports depends on its release,
metadata and tensor layout, not just its architecture name. A request the
file can't support is refused with an error that says why.

To see what the engine makes of a file without loading it onto the GPU:

```sh
qwen info -m MODEL.gguf          # summary
qwen info -m MODEL.gguf --json   # reasoning levels, input forms, tools, sampling
```

## Build

Requirements: an Apple Silicon Mac, Rust 1.88 or newer, and the Xcode
command-line tools (for `xcrun metal` and `xcrun metallib`).

```sh
cargo build --release -p qwen-cli
```

This puts `qwen`, `qwen-bench`, `qwen-lens` and a few smaller tools in
`target/release/`. The examples below assume they are on your `PATH`.

The Metal kernels are compiled during the build and embedded in the binaries.
llama.cpp is linked as a library for tokenization and CPU-side dequantization;
models do not run through it.

## Running a model

```sh
qwen run -m MODEL.gguf --user "Explain this"
qwen run -m MODEL.gguf --system "Be concise" --user "Explain this" -n 1024
qwen run -m MODEL.gguf --reasoning-effort low --user "Explain this"
qwen run -m MODEL.gguf --no-thinking --user "Explain this"
cat question.txt | qwen run -m MODEL.gguf --user -
```

- The prompt is rendered with the model's chat template. Templates are
  implemented in Rust; there is no Jinja runtime.
- `--reasoning-effort` takes the levels the release defines, such as
  `none`, `low`, `medium` and `xhigh` on Qwen3.8. `--no-thinking` selects the
  release's non-thinking mode where it has one.
- `qwen run` generates at most 64 tokens unless you pass `-n`
  (`--max-tokens`).
- Sampling uses per-release presets, which `qwen info --json` reports.
  `--temp 0` gives greedy decoding. A sampled run without `--seed` picks a
  seed and prints it.
- stdout gets the generated text, including any reasoning and tool-call
  markup. Diagnostics and a stats line go to stderr. For K2 Horizon and
  GLM-5.3-Flash, `--format responses` prints an Open Responses JSON object
  instead.

Conversations and tool use take messages as JSON, from a file or stdin:

```sh
qwen run -m MODEL.gguf --messages conversation.json
qwen run -m MODEL.gguf --messages -
```

The JSON is an array of chat messages, or `{ "messages": [...], "tools": [...] }`
with OpenAI-style tool definitions. The exact message and tool-call fields
vary by family. `qwen info --json` shows which input forms a given file
accepts.

`--raw-prompt '<text>'` sends exact model input with no template.

Dense Qwen models (`qwen35`) can use a DFlash drafter for speculative decoding
with `--drafter DRAFTER.gguf`, in both `qwen run` and `qwen serve`. MoE Qwen
models (`qwen35moe`) accept one in `qwen run` only.

`qwen -h` shows the common options. `qwen --help` adds JSONL batch processing
and research options.

## Serving

```sh
qwen serve -m Qwen3.8-27B.gguf
qwen serve -m GLM-5.3-Flash.gguf --max-context-tokens 32768 --max-tokens 4096
```

`qwen serve` keeps one model loaded and serves a subset of the Open Responses
API on `127.0.0.1:8737`: `POST /v1/responses`, with streaming, reasoning and
function tools, and `GET /v1/models`. It listens on loopback only and has no
authentication. Clients that speak Open Responses can use it directly;
[docs/serve-opencode.md](docs/serve-opencode.md) shows the OpenCode setup.

- Requests run one at a time. A request that arrives while another is running
  gets `503` with `Retry-After: 1` straight away.
- The API is stateless: clients send the whole conversation with each
  request. The server keeps state from earlier requests and reuses the
  longest part that matches, so a follow-up turn mostly processes only what
  is new.
- Qwen and DeepSeek V4 also save that state to disk (by default under
  `~/.cache/qwen-llm/serve-checkpoints`) and can restore it after a restart.
- Ordinary Qwen models size each request as it comes. The other families
  allocate a fixed context at startup: DeepSeek V4 needs
  `--max-context-tokens`, and Flash-Next, Muse Glimmer, K2 Horizon and
  GLM-5.3-Flash need both `--max-context-tokens` and `--max-tokens`.

[docs/SERVE.md](docs/SERVE.md) covers the API subset, caching and flags in
detail.

## Lens

`qwen-lens` is a set of interpretability tools. It reads out what a model's
intermediate layers predict, using a plain logit lens or J- and R-lenses
(fitted locally or imported). It can also apply interventions between blocks
during a run, sweep an intervention's strength and compare saved traces. See
[docs/LENS-RUN.md](docs/LENS-RUN.md).

`qwen serve` can also host a browser workbench for lens jobs:

```sh
(cd web && bun install --frozen-lockfile && bun run build)
qwen serve -m MODEL.gguf --lens-data-dir LENS_JOBS_DIR --web-root web/dist
```

Running new jobs from the workbench currently needs a dense or MoE Qwen3.6 or
Qwen3.8 model (not Flash-Next) with its standard chat template. See
[docs/LENS-WEB.md](docs/LENS-WEB.md).

## Benchmarks

```sh
qwen-bench suite -m MODEL.gguf --pp 512 --tg 128 --runs 3 -o json
```

`qwen-bench pp` and `qwen-bench tg` use the same test definitions as
`llama-bench`'s prompt-processing and generation tests, so the two can be
compared on the same model file. `suite` runs many shapes against one model
load. `qwen-bench --help` lists the rest, most of which are profiling and
research probes.

- [docs/BENCH.md](docs/BENCH.md): methodology
- [docs/PERF-LOG.md](docs/PERF-LOG.md): measurements over time
- [docs/PERF-ROADMAP.md](docs/PERF-ROADMAP.md): open performance work
- [docs/bench/](docs/bench/): raw data and write-ups for individual
  measurements

## Repository layout

```text
crates/qwen-llm/   engine library: GGUF loading, tokenizers, model families, Metal dispatch
crates/qwen-cli/   qwen, qwen-bench and qwen-lens, plus qwen-tok, qwen-census
                   and qwen-grammar-oracle
kernels/           Metal shaders, embedded in the binaries at build time
kernels/research/  benchmark-only shaders, loaded on first use
web/               the lens workbench (Bun and React)
scripts/           reference implementations and fixture generators used by
                   tests, plus benchmark and serving checks
docs/              behaviour references, design plans and measurement records
```

Other documents:

- [docs/CLI-UX.md](docs/CLI-UX.md): how the `qwen` command line was designed
  and tested
- [docs/ENV.md](docs/ENV.md): environment variables
- Per-family bring-up plans, such as
  [docs/GLM53-FLASH-PLAN.md](docs/GLM53-FLASH-PLAN.md) and
  [docs/K2-HORIZON-PLAN.md](docs/K2-HORIZON-PLAN.md)
- [docs/PLAN.md](docs/PLAN.md): the original architecture plan, kept for
  history

## Development

`cargo test` builds with optimizations, because the `test` profile inherits
`release`: numerical tolerances and timing-sensitive tests depend on it. Lint
configuration is in `[workspace.lints]` in `Cargo.toml` and in `clippy.toml`.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option. Adapted third-party components are
listed in [`docs/THIRD-PARTY-NOTICES.md`](docs/THIRD-PARTY-NOTICES.md).

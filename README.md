# qwen-llm

A local inference engine for Apple Silicon Macs, written in Rust with its own
Metal kernels. It runs GGUF models from several families, from the command
line or as a local HTTP server for clients such as coding agents.

Because the engine executes every layer itself, it is also an instrument. It
can project a model's intermediate activations into readable form, apply
controlled interventions at chosen sites during a run, and save the results
with the model and settings that produced them. Speed makes those experiments
cheap to repeat, and owning the forward pass is what exposes the sites to
measure and change.

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

A model that loads and runs has not thereby been shown to match a reference
implementation or to be fast. Agreement and speed depend on the exact file,
quantization, execution mode and hardware; the measurements under
[docs/bench/](docs/bench/) say which combinations were checked and how.

To see what the engine makes of a file without loading it onto the GPU:

```sh
qwen info -m MODEL.gguf          # summary
qwen info -m MODEL.gguf --json   # reasoning levels, input forms, tools, sampling
```

### Memory

Weights stay resident in unified memory, and the GPU can only use as much
memory as macOS lets it wire (`iogpu.wired_limit_mb`). A model needs roughly
its file size plus room for context state, which grows with context length:
Qwen3.8-27B's attention cache is about 8 GiB at 133K tokens. The file sizes in
the performance table below are a guide. The 112 GiB GLM-5.3-Flash UD-IQ3_XXS
file has been run on a 128 GB Mac with the wired limit raised to 118 GiB.

## Build

Requirements: an Apple Silicon Mac, Rust 1.88 or newer, and Xcode with its
Metal toolchain, so that `xcrun metal` and `xcrun metallib` work. Since Xcode
26 the toolchain is a separate download:
`xcodebuild -downloadComponent MetalToolchain`.

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
qwen run -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -n 4096 --user "Why is the sky blue?"
```

Thinking models reason before they answer, so give them an output budget:
without `-n` (`--max-tokens`), `qwen run` stops after 64 tokens. Other common
forms:

```sh
qwen run -m MODEL.gguf -n 4096 --system "Be concise" --user "Explain this"
qwen run -m MODEL.gguf -n 4096 --reasoning-effort low --user "Explain this"
qwen run -m MODEL.gguf -n 1024 --no-thinking --user "Explain this"
cat question.txt | qwen run -m MODEL.gguf -n 4096 --user -
```

- The prompt is rendered with the model's chat template. Templates are
  implemented in Rust; there is no Jinja runtime.
- `--reasoning-effort` takes the levels the release defines, such as
  `none`, `low`, `medium` and `xhigh` on Qwen3.8. `--no-thinking` selects the
  release's non-thinking mode where it has one.
- Sampling uses per-release presets, which `qwen info -m MODEL --json` reports.
  `--temp 0` gives greedy decoding. A sampled run without `--seed` picks a
  seed and prints it.
- stdout gets the generated text, including any reasoning and tool-call
  markup. Diagnostics and a stats line go to stderr. For K2 Horizon and
  GLM-5.3-Flash, `--format responses` prints an Open Responses JSON object
  instead.

Conversations and tool use take messages as JSON, from a file or stdin:

```sh
qwen run -m MODEL.gguf -n 4096 --messages conversation.json
qwen run -m MODEL.gguf -n 4096 --messages -
```

The JSON is an array of chat messages, or `{ "messages": [...], "tools": [...] }`
with OpenAI-style tool definitions. The exact message and tool-call fields
vary by family. `qwen info -m MODEL --json` shows which input forms a given file
accepts.

`--raw-prompt '<text>'` sends exact model input with no template.

`qwen -h` shows the common options. `qwen --help` adds JSONL batch processing
and research options.

## Serving

```sh
qwen serve -m Qwen3.8-27B-Q4_K_M.gguf
qwen serve -m GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf --max-context-tokens 32768 --max-tokens 4096
```

`qwen serve` keeps one model loaded and serves a subset of the Open Responses
API on `127.0.0.1:8737`: `POST /v1/responses`, with streaming, reasoning and
function tools, and `GET /v1/models`. It listens on loopback only and has no
authentication. Clients that speak Open Responses can use it directly;
[docs/serve-opencode.md](docs/serve-opencode.md) shows the OpenCode setup.

- Requests run one at a time. A request that arrives while another is running
  gets `503` with `Retry-After: 1` straight away.
- The API is stateless: clients send the whole conversation with each
  request. The server keeps state from earlier requests and reuses the part
  that is compatible with the new one. How much it can reuse depends on the
  family and execution mode; the table below summarizes it.

[docs/SERVE.md](docs/SERVE.md) covers the API subset, caching and flags in
detail.

## What differs by family

<!-- BEGIN GENERATED FAMILY PROFILE TABLE -->
| Family | `qwen serve` sizing flags | Snapshots | DFlash drafter |
|---|---|---|---|
| Qwen (qwen35) | none (request-shaped) | RAM and disk | `run`, `serve` |
| Qwen MoE (qwen35moe) | none (request-shaped) | RAM and disk | `run` |
| Qwen3.8-Flash-Next | `--max-context-tokens`, `--max-tokens` | RAM | no |
| DeepSeek V4 | `--max-context-tokens` | RAM and disk | no |
| Muse Glimmer | `--max-context-tokens`, `--max-tokens` | none | no |
| K2 Horizon | `--max-context-tokens`, `--max-tokens` | none | no |
| GLM-5.3-Flash | `--max-context-tokens`, `--max-tokens` | RAM | no |
<!-- END GENERATED FAMILY PROFILE TABLE -->

- **Snapshots.** A snapshot saves model state at a conversation boundary; a
  later request restores the longest matching prefix. The disk tier writes
  under `~/.cache/qwen-llm/serve-checkpoints` by default.
- **Live-session reuse.** Muse Glimmer and K2 Horizon rewind their resident
  session to the longest common prefix. GLM-5.3-Flash continues its session
  when the new prompt extends the previous one exactly; otherwise it restores
  a compatible snapshot when one is cached, or prefills a fresh session.
  With its default Fast prefill, GLM snapshots only at
  the end of the shared instructions-and-tools prefix; with Exact prefill
  (`x_qwen.prefill_lineage: "exact"`) it also snapshots at the start of the
  generation header.
- **DFlash drafter.** `--drafter DRAFTER.gguf` enables speculative decoding.

## Lens

`qwen-lens` projects intermediate activations into token-space readouts, using
a plain logit lens or J- and R-lens transports (fitted locally or imported).
With it you can:

- read a run's projections across layers and positions;
- intervene during a run with a lens row or your own direction vector, at
  block outputs or, on dense Qwen and GLM-5.3-Flash, at each residual writer
  (embedding, attention or GDN mixer, FFN) before its output is added;
- sweep an intervention's strength over one model load and compare the arms;
- reopen and compare saved results later: a result records the plan it ran
  and content hashes of the lenses and directions it used.

For example, projecting one direction out of GLM-5.3-Flash's residual writers
gave the same refuse, hedge or comply label as llama.cpp running a published
rank-1 weight edit, at all 152 prompt and dose pairs tested, from a single
vector instead of edited weights
([docs/bench/2026-10-08-glm53-directions/](docs/bench/2026-10-08-glm53-directions/README.md)).
See [docs/LENS-RUN.md](docs/LENS-RUN.md) for what each family supports.

`qwen serve` can also host a browser workbench for lens jobs:

```sh
(cd web && bun install --frozen-lockfile && bun run build)
qwen serve -m MODEL.gguf --lens-data-dir LENS_JOBS_DIR --web-root web/dist
```

Running new jobs from the workbench currently needs a dense or MoE Qwen3.6 or
Qwen3.8 model (not Flash-Next) with its standard chat template. See
[docs/LENS-WEB.md](docs/LENS-WEB.md).

## Performance

A snapshot, not a current guarantee: qwen-llm `70ec9a9b` against llama.cpp
b11182 on 2026-09-25, M4 Max 128 GB, same file for both. Tokens per second
for qwen-llm, with the ratio to the better of llama.cpp's `-ub 512` and
`-ub 2048` runs.

| Model, file size | pp512 | pp4096 | tg128 | tg128 at 8K |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Q4_K_M, 15.9 GiB | 246 (1.01x) | 233 (1.07x) | 25.6 (1.05x) | 23.2 (1.07x) |
| Qwen3.6-35B-A3B UD-Q4_K_M, 20.6 GiB | 1535 (0.97x) | 1793 (1.05x) | 109.9 (1.19x) | 98.8 (1.16x) |
| Qwen3.5-122B-A10B UD-Q4_K_XL, 71.7 GiB | 491 (0.96x) | 555 (1.04x) | 46.0 (1.11x) | 42.6 (1.12x) |
| Qwen3.8-Flash-Next UD-Q3_K_XL, 83.8 GiB | 554 (0.86x) | 473 (0.77x) | 36.4 (0.88x) | 26.8 (0.75x) |
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS, 97.1 GiB | 141 (0.48x) | 254 (0.83x) | 28.7 (1.00x) | 22.4 (0.85x) |
| Muse-Glimmer-30B Q8_0, 27.6 GiB | 228 (0.86x) | 198 (0.85x) | 16.5 (0.97x) | 15.7 (0.99x) |

These are synthetic-token tests with llama-bench's semantics, not request
latency, and several rows have changed since. Setup, caveats and raw data:
[docs/bench/2026-09-25-1759-families-family/](docs/bench/2026-09-25-1759-families-family/FINDINGS.md).
GLM-5.3-Flash was measured separately on 2026-10-04: prefill at 93-98% of
llama.cpp, decode 21-25% ahead
([docs/bench/2026-10-04-glm53-p3-packed-prefill/](docs/bench/2026-10-04-glm53-p3-packed-prefill/README.md)).

To measure yourself:

```sh
qwen-bench suite -m MODEL.gguf --pp 512 --tg 128 --runs 3 -o json
```

The default `qwen-bench pp` and `tg` tests follow `llama-bench`'s timing
semantics, so they can be compared with llama-bench on the same file when
context depth, KV cache type and batch settings match. `suite` runs many
shapes against one model load. `qwen-bench --help` lists the rest, most of
which are profiling and research probes.

- [docs/BENCH.md](docs/BENCH.md): methodology
- [docs/PERF-LOG.md](docs/PERF-LOG.md): measurements over time, including
  approaches that were tried and closed
- [docs/PERF-ROADMAP.md](docs/PERF-ROADMAP.md): open performance work

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

- [docs/README.md](docs/README.md): index of the documentation and measurement packets
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

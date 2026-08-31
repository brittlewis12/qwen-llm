# qwen-llm

A from-scratch Apple Silicon inference engine for **Qwen 3.5 / 3.6 / 3.8 hybrid Gated
DeltaNet** models and **DeepSeek V4 Flash-0731**. Single goal: maximum tok/sec
for prompt processing and token generation, single-stream and batched.

Architecture decisions live in [`docs/PLAN.md`](docs/PLAN.md).

## Status

v0 — active bring-up. Numerical-oracle target: byte-for-byte logits match vs
`llama-cli` on `~/models/Qwen3.5-0.8B.F32.gguf`. Throughput target: beat the
clean-box `llama.cpp` baseline on `Qwen3.6-27B-Q4_K_M.gguf` on M4 Max; the
maintained benchmark table lives in `docs/PLAN.md`, and adaptive DFlash notes
live in `docs/H5-DFLASH.md`. Native DeepSeek V4 status and evidence live in
`docs/DEEPSEEK-V4-STRATEGY.md`.

DeepSeek V4 automatically uses its exact multi-group sparse selector on an
Apple M4 Max once a singleton reaches the qualified far-context band. Packed
and ineligible singleton positions retain radix4. To force the rollback path:

```sh
--deepseek-v4-multigroup-selector=off
```

`qualified-experimental` remains available as an explicit diagnostics policy;
the CLI rejects unsupported devices and unreachable request geometry before
model residency.

## Layout

```
crates/qwen-llm     — engine library
crates/qwen-cli     — `qwen` inference CLI + `qwen-bench` throughput tool
kernels/            — Metal compute shaders (compiled to embedded .metallib)
docs/PLAN.md        — architectural decisions
```

## Build

Requires the Xcode command-line tools (for `xcrun metal` / `xcrun metallib`).

```sh
cargo build --release -p qwen-cli --bin qwen
./target/release/qwen --info

cargo build --release -p qwen-cli --bin qwen-bench
./target/release/qwen-bench -m ~/models/Qwen3.6-27B-Q4_K_M.gguf
```

## Inference

Use `run` for an ordinary model-templated request:

```sh
qwen run -m MODEL --user "Explain this"
qwen run -m MODEL --system "Be concise" --user "Explain this"
qwen run -m Qwen3.8-27B.gguf --reasoning-effort low --user "Explain this"
qwen run -m Qwen3.8-Flash-Next-Q3_K_XL.gguf --no-thinking --user "Explain this"
qwen run -m Muse-Glimmer-30B-Q8_0.gguf --reasoning-effort xhigh --user "Explain this"
qwen run -m MODEL --user -
```

Structured messages accept a strict bare array or `{ "messages": [...] }`;
`-` reads one complete JSON document from stdin:

```sh
qwen run -m MODEL --messages messages.json
qwen run -m MODEL --messages -
```

Raw model input remains explicit. On supported detected model families,
`--raw-prompt` has the same tokenization and output semantics as the legacy
`-p/--prompt` interface:

```sh
qwen run -m MODEL --raw-prompt '<exact model input>'
```

On validated Qwen3.8 27B and Flash-Next surfaces,
`--reasoning-effort low|medium|xhigh` selects the exact upstream ordinary-chat
transition; omission remains xhigh and `high` fails closed. Muse Glimmer accepts
`low|medium|high|xhigh` as exact `Reasoning strength` values and defaults to
`high`. Muse uses its released sampling preset (`temperature=1`, `top_p=.95`,
`top_k=64`, `min_p=0`) unless a sampling field is explicitly overridden. Raw
prompts bypass these reasoning controls. On
validated Qwen3.6 35B A3B and Qwen3.8 surfaces, `--no-thinking` uses the
model-family non-thinking transition. DeepSeek ordinary chat is already
non-thinking, so that option is idempotent there. These controls do not suppress
CLI diagnostics. Existing flat invocations and resident
`--requests-jsonl FILE|-` remain supported; `qwen -h` shows the common path and
`qwen --help` shows expanded documented legacy/research options. Legacy flags
cannot be combined with `qwen run`.

Flash-Next currently uses request-shaped QSA caches and a correctness-first
token-at-a-time prompt path for the released `UD-Q3_K_XL` profile. Its CLI lane
is serial single-turn text generation; batching, serve, prefix caches, and
packed prefill remain explicit follow-ups rather than silent fallbacks.

Qwen3.8 support is text-only. Modern messages do not accept image content,
developer or tool roles, structured tool calls/results, response-format
objects, or structured reasoning history. Vision/projector execution and those
protocol surfaces are not implemented.

## Reference quarry

- `~/code/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` — kernel source
  (gated_delta_net, ssm_conv, l2_norm, mul_mv_q4_K, mul_mm_q4_K, set_rows,
  rope, rms_norm).
- `~/code/vllm-metal/vllm_metal/metal/kernels_v2/` — varlen / paged-KV
  argument layout reference.
- `~/code/llama-cpp-rs/llama-cpp-sys-2` — `ggml_get_type_traits().to_float`
  codec seam, used for tooling and re-pack-on-load (not the GPU hot path).

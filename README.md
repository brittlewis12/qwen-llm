# qwen-llm

A from-scratch Apple Silicon inference engine for **Qwen 3.5 / 3.6 hybrid Gated
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

DeepSeek V4 keeps radix4 sparse selection by default. On an Apple M4 Max, a
request budget that can reach at least 196,608 compressed rows may explicitly
select the qualified far-context policy with:

```sh
--deepseek-v4-multigroup-selector=qualified-experimental
```

The CLI rejects unsupported devices and unreachable request geometry before
model residency. Packed and ineligible singleton positions continue to use
radix4; this option is not a default-on or wider-device claim.

## Layout

```
crates/qwen-llm     — engine library
crates/qwen-cli     — `qwen` interactive CLI + `qwen-bench` throughput tool
kernels/            — Metal compute shaders (compiled to embedded .metallib)
docs/PLAN.md        — architectural decisions
```

## Build

Requires the Xcode command-line tools (for `xcrun metal` / `xcrun metallib`).

```sh
cargo build --release -p qwen-cli
./target/release/qwen --info
./target/release/qwen-bench -m ~/models/Qwen3.6-27B-Q4_K_M.gguf
```

## Reference quarry

- `~/code/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` — kernel source
  (gated_delta_net, ssm_conv, l2_norm, mul_mv_q4_K, mul_mm_q4_K, set_rows,
  rope, rms_norm).
- `~/code/vllm-metal/vllm_metal/metal/kernels_v2/` — varlen / paged-KV
  argument layout reference.
- `~/code/llama-cpp-rs/llama-cpp-sys-2` — `ggml_get_type_traits().to_float`
  codec seam, used for tooling and re-pack-on-load (not the GPU hot path).

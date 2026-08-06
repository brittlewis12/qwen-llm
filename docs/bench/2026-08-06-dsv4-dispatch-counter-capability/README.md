# DeepSeek V4 Dispatch-Counter Capability

Status: frozen target capability screen. No model or GGUF access is permitted.

## Question

The held four-encoder pre-attention packet exposed a material unowned encoder
transition. Static review found a potentially better observer: retain one
compute encoder and place shared timestamp samples at dispatch boundaries.

Before any current-asset execution, ask whether Apple M4 Max and the current
Metal stack legally support `MTLCounterSamplingPoint::AtDispatchBoundary`.
The answer is binary. Unsupported capability closes this observer; it is not a
timing HOLD and does not authorize a fallback through stage-boundary buffers.

## Probe

The target-aware release test:

1. constructs a Metal context and requires device name `Apple M4 Max`;
2. requests a five-entry dispatch-boundary timestamp buffer;
3. if supported, places five samples around four groups of eight scale
   dispatches inside exactly one compute encoder;
4. requires 1 encoder, 32 dispatches, strict adjacent timestamps, an error-free
   command, and exact output; and
5. prints an execution marker only after all checks pass.

The target device may not silently skip the capability request. The earlier
green development probe did so and is explicitly not evidence.

## Fixture

- Base revision: `f04219b`
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Model access: forbidden
- Ignored/current-asset tests: forbidden

## Source Freeze

- Release test executable:
  `c8cd4e4b3b6ae5e4086877980f3b56c41049ef97a686cce1054114d29444c590`
- `deepseek_v4_metal.rs`:
  `c16f1b707b4ecbdbe07529e85226d1a915389bc5f6122870d52684646750e0c8`
- `deepseek_v4_metal/prefill.rs`:
  `0efa55a6003fea7562ca1987e28d9b689f2c17e8882d49bb1cf3a60028ab5a29`
- Rejected-observer diff:
  `2cf5ad71c36c3da64ae34de46da99f130d38321b329718a0d1cc3012444d3199`
- Capability source slice:
  `6daf0eb3cf5fa755e9f48121a0d75d17d3452e3a16efb2b7ac216c0bad98c9a7`

Focused shared-boundary resolver tests and strict release diagnostics Clippy
pass. CX session `019fd685-acb3-7890-83c9-192cbea48c6e` gives static GO to the
observer design, then requires this target-aware capability proof before any
model run.

## Frozen Command

```bash
set -o pipefail && cargo test --release \
  -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::\
dispatch_boundary_samples_are_strict_inside_one_encoder \
  -- --exact --nocapture --test-threads=1 \
  2>&1 | tee \
  docs/bench/2026-08-06-dsv4-dispatch-counter-capability/capability.log
```

If buffer creation returns unsupported, retain the exact error, classify the
same-encoder protocol as capability-falsified, remove the rejected observer,
and reopen only after device/OS/Metal capability drift or a genuinely new legal
intra-pass timing primitive.

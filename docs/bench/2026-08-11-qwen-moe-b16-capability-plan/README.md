# Qwen MoE B=16 Capability Plan

Date: 2026-08-11

Status: `GO` for capability-driven execution planning and the previously
qualified Q4 A3B composition. Qwen3.5 A3B IQ4_XS and MTP-tagged Q4 execute
exactly through mixed plans, but this packet does not make B=16 the preferred
width for every accepted composition. A10B remains unmeasured and `HOLD`.

## Question

Does fixed Qwen MoE B=16 need an asset-by-asset allowlist, or can the executor
select exact accelerated stages from tensor and architecture contracts while
falling back to ordinary per-lane production work elsewhere?

## Plan

The executor now builds one immutable plan before allocating its persistent
arena. Each block independently selects:

- token-axis Q8_0 GDN qkv/z/out when the singleton-compatible LCPP arithmetic is
  active, otherwise the complete production mixer per lane;
- packed Q4_K routed gate/up when its shape contract matches, otherwise the
  complete production routed FFN tail per lane;
- token-axis Q6_K or Q8_0 language-model heads, otherwise production heads per
  lane.

Attention, route/top-k, shared experts, routed down/final waves, KV, GDN causal
state, and pending-token ownership remain sequence-private. A fallback is not a
new arithmetic body: it calls the same production encoder used by serial decode.

Admission validates the Qwen MoE architecture ABI, supported GQA group, exact
integral rotary width, tensor inventory, shapes, quant block alignment, kernel
integer widths, storage dtypes, diagnostic graph overrides, and command-queue
residency. A plan with no accelerated stage is rejected as an efficiency policy.
Telemetry schema 3 reports accelerated and per-lane block counts plus head mode.

## Results

All GPU work was serialized under the process Metal lease. The fixture contains
16 identical 22-token prompts and requests 32 greedy output tokens, producing 31
B=16 transitions. Full stdout includes ordinary JSON rows in input order.

| Asset | Plan | Serial | B=16 | Aggregate | Exact |
|---|---|---:|---:|---:|---|
| Qwen3.6 A3B Q4_K_M | 30 Q8 GDN, 40 packed Q4 gate/up, Q6 head | 12.17 s | 7.33 s | 158.269 tok/s | yes |
| Qwen3.5 A3B IQ4_XS | 30 Q8 GDN, 40 per-lane IQ3 gate/up, Q6 head | 8.77 s | 7.28 s | 145.810 tok/s | yes |

The Q4 canary retains the previously authorized composition and clears its
`143.4234` aggregate floor. Complete serial and B=16 output share SHA-256
`65854b2c9e83d3255693a087ea8bc67c1f5af16629983baf30a38af43f1c2edb`.
This is a regression canary; the committed full-state authority remains the
unchanged Q4 packet in
`docs/bench/2026-08-11-qwen-moe-b16-packed-gateup-go/README.md`.

The IQ4 mixed plan improves process wall `1.205x` over serial and remains exact
for all 512 emitted tokens, with SHA-256
`a5c265ac381e4b04aa83607a4a1d89d59bddc4ce082bc9e248e04f27d9c6028b`.
An independent-queue B=2 control is slightly better at `7.17 s` and roughly
`149-153` aggregate token/s per pair. Therefore IQ4 establishes correct fallback
composition, not a B=16 recommendation; use B=2 until a wider IQ3 gate/up stage
creates margin.

A separate one-transition Qwen3.5 IQ4 smoke is exact and reports the same plan.
An MTP-tagged Qwen3.6 Q4 asset also executes the base 40-layer plan exactly, with
serial and B=16 stdout SHA-256
`4ee06052cd14e36b24fcdc72f8bf815d1925ce5e584ccaa5a6b6693d05f5c60e`.
Its short `7.93 -> 4.46 s` wall observation is not a performance endpoint.

## Decision

Keep support capability-driven rather than matching model filenames. Exact
kernel availability determines what can batch; measured composition economics
determine which width should be preferred. Q4 A3B remains B=16 `GO`. IQ4 and MTP
are executable but retain narrower recommendation scope. A10B's expected Q8 GDN,
mixed Q4/Q5 routed weights, and Q8 head now have a valid plan, but its 77 GiB
asset remains `HOLD` until a serialized short probe clears exactness, memory, and
B=8/B=2 economics.

The next serving step is not another model allowlist. It is a scheduler that can
select among serial, independent B=2, and ready B=16 work from plan telemetry and
queue occupancy, followed by dynamic B=16 cohort refill.

## Validation

```text
cargo fmt --all
cargo test -p qwen-llm --lib moe_batch16
cargo test -p qwen-cli --bin qwen fixed_cohort_jsonl
cargo check --workspace
cargo clippy -p qwen-llm --lib -- -D warnings
cargo clippy -p qwen-cli --all-targets -- -D warnings
cargo build --release -p qwen-cli --bin qwen
git diff --check

./target/release/qwen --model MODEL \
  --requests-jsonl FIXTURE.jsonl --temp 0 --prefill-chunk 2048

./target/release/qwen --model MODEL \
  --requests-jsonl FIXTURE.jsonl --temp 0 --prefill-chunk 2048 \
  --batch-size 16
```

# Automatic JSONL Execution Width

Date: 2026-08-11

Status: product `GO` for an opt-in, fail-closed selector over existing exact
executors. Default CLI behavior remains serial.

## Contract

`qwen --requests-jsonl FILE --execution-mode auto` prepares a regular request
file once, inspects executable capabilities without allocating sequence state,
performs conservative memory admission, emits one schema-v1 selection record,
then chooses exactly one existing backend:

- dense Qwen fixed B=8 when a complete compatible cohort is ready;
- Qwen MoE fixed B=16 for the measured non-MTP A3B Q4 composition;
- independent resident B=2 for the measured A3B IQ3-routed composition, for an
  underfilled qualified Q4 file, or as a memory-safe narrower fallback;
- DeepSeek V4 independent B=2 when two-session admission succeeds and the
  command-queue-scoped residency set is off;
- serial execution otherwise.

The MoE preference key is architecture geometry plus immutable capability-plan
telemetry, including exact routed dtype class; it does not inspect a model
filename. MTP and unmeasured/A10B compositions remain serial even when B=16 is
mechanically executable. Capability and measured width preference remain
separate decisions.

Otherwise-supported Qwen prompt lookup, legacy request sidecars, request traces,
explicit cache policy, sampled requests, explicit GPU-argmax rollback, stdin,
and unqualified compositions select serial rather than silently losing a
requested feature. Options already unsupported by Qwen or DeepSeek JSONL remain
fail-closed before selection. An accelerated selection disables the still-empty
RAM prefix cache before request execution; fixed and B=2 prefix fanout remain
active.

Explicit `--batch-size 8`, `--batch-size 16`, and `--concurrency 2` keep their
existing fail-closed behavior and conflict with `--execution-mode`.

## Memory Admission

The selector prices the maximum request capacity, maximum prompt scratch, every
candidate sequence, persistent executor scratch, a small allocation-rounding
allowance, and the existing 2 GiB dynamic reserve before selecting Qwen B=2,
B=8, or B=16. A denied wide candidate may narrow to admitted B=2; denial of both
selects serial before any mutable sequence or executor arena is allocated.

DeepSeek evaluates the existing complete residency-plus-session memory plan for
two sessions before consuming the load plan. Denial selects the ordinary
one-session load instead of failing a convenience mode after residency.

## Product Smokes

All GPU work was serialized under the process Metal lease. The release binary
was built from base `a6d19b7ad5d5b8f46b55448b02a7557cdad9c87c` plus this patch.

| Model/profile | Selected | Observation | Stdout SHA-256 |
|---|---|---|---|
| Qwen3.6 A3B Q4, 16 x 32 | B=16 | `154.188` aggregate token/s; exact prior canary | `65854b2c9e83d3255693a087ea8bc67c1f5af16629983baf30a38af43f1c2edb` |
| Qwen3.5 A3B IQ4_XS, 16 x 32 | B=2 | pair throughput `147.2-152.4` token/s; exact prior canary | `a5c265ac381e4b04aa83607a4a1d89d59bddc4ce082bc9e248e04f27d9c6028b` |
| Qwen3.5 0.8B Q4, 16 x 32 | two B=8 cohorts | `608.3/621.4` aggregate token/s | `ae5481107b139960f6a8c2b504f98b0f5364ea1589a090a97616ecba63da3682` |
| DeepSeek K160, two short requests | B=2 | `37.225` aggregate token/s | `08f953db2ed754811c4d8065cfe1c1f17ecbe6c76ad69235cb35cac9f3d82322` |
| Qwen3.6 A3B MTP Q4, 16 x 32 | serial | deliberately outside measured preference | `65854b2c9e83d3255693a087ea8bc67c1f5af16629983baf30a38af43f1c2edb` |

The Q4 and IQ4 hashes match their committed explicit-width canaries. The MTP
smoke confirms that executable B=16 capability does not silently broaden
automatic preference scope.

## Validation

```text
cargo fmt --all
cargo test -p qwen-cli --bin qwen execution_selector
cargo test -p qwen-cli --bin qwen fixed_cohort_jsonl
cargo test -p qwen-cli --bin qwen concurrent_jsonl
cargo test -p qwen-llm --lib moe_batch16
cargo check --workspace
cargo clippy -p qwen-llm --lib -- -D warnings
cargo clippy -p qwen-cli --all-targets -- -D warnings
cargo build --release -p qwen-cli --bin qwen
git diff --check
```

This is a bounded selector over fixed executors, not continuous batching. It
does not benchmark choices online, pad incomplete cohorts, combine B=16 with B=2
remainders, or retry after committed execution begins.

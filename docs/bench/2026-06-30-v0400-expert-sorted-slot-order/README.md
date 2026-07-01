# v0.400 Expert-Sorted Slot-Order Kill-Test

Goal: test whether A10B's weak multi-slot MoE batching is primarily an expert
locality problem. The new `moe-batch-sweep --slot-order` option can replay the
same captured routes in the production slot order or in a perf-only
expert-sorted order.

Important caveat: `expert-sorted-perf-only` is not correctness-preserving with
the current kernels. It sorts packed route slots by expert id while the kernels
still derive token rows from `slot / topk`. This makes it a generous locality
upper-bound kill-test: if this does not help, an exact sorted implementation
would also need to overcome additional gather/scatter/regrouping cost.

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B `ctx32` smoke with `--slot-order exact,expert-sorted-perf-only`
- A3B/A10B independent-file `ctx512` sweeps over 8 game prompt files
- `cx ask` adversarial review, session `019f1bb9-5bef-73d1-b5ef-5e51e7164c78`

Artifacts:

- `target/profiles/v0400-a3b-moe-batch-slot-order-smoke.out`
- `target/profiles/v0400-a3b-independent8-ctx512-slot-order.out`
- `target/profiles/v0400-a10b-independent8-ctx512-slot-order.out`

## Independent-File Results

Eight prompt slots, `ctx512`, `iters=5`, `warmup=2`, same loaded process.

### A10B Q4_XL

| Slots | Exact ms/tok | Expert-sorted ms/tok | Read |
| ---: | ---: | ---: | --- |
| 1 | `5.7233` | `5.7208` | flat |
| 2 | `4.7254` | `4.7231` | flat |
| 4 | `4.8401` | `4.9445` | slower |
| 8 | `5.1352` | `5.2233` | slower |

### A3B Q4

| Slots | Exact ms/tok | Expert-sorted ms/tok | Read |
| ---: | ---: | ---: | --- |
| 1 | `1.8244` | `1.8451` | slower |
| 2 | `1.3985` | `1.3857` | tiny win |
| 4 | `1.2968` | `1.2921` | tiny win |
| 8 | `1.2211` | `1.2146` | tiny win |

## Decision

A10B expert-sorted rescue is killed for this workload. The locality-only upper
bound is flat at `b1/b2` and slower at `b4/b8`, so an exact sorted path would
need to pay additional correctness overhead despite no observed locality win.
Keep A10B capped at `b2` or disabled above `b2` for batching.

A3B exact slot order remains the strong batching signal. Expert sorting is at
most a sub-1% local tweak and should not distract from the next decisive gate:
an A3B targeted end-to-end decode prototype that measures whether the captured
MoE projection win survives scheduler, KV, attention, sampling, and packing
overheads.

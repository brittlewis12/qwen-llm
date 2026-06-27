# v0.341 MoE FFN Pipeline2 Negative

Status: tested and removed a dirty MoE decode FFN pipeline proof. The proof split
top-k routed experts into two groups, overlapped down for group A with gate/up for
group B, ran shared down in the same wave, then accumulated group B before the
final residual. The goal was to test scheduling/work-granularity overlap without
rewriting the Q5 down kernel.

Validation:

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `QWEN_DECODE_MOE_FFN_PIPELINE2=1 cargo test -p qwen-llm metal_single_token_concurrent_gdn_moe_matches_serial_a3b -- --nocapture`
- sequential A3B/A10B `tg128` smoke A/B, no parallel GPU workloads

Correctness: A3B serial-vs-pipeline smoke matched argmax through four prompt
steps plus one follow-up, with `max|delta| <= 0.0002` and `cos=1.000000`.

Performance:

| Model | Default | Pipeline2 | Ratio |
| --- | ---: | ---: | ---: |
| A3B `tg128` | `103.90 t/s` | `100.64 t/s` | `0.969x` |
| A10B `tg128` | `45.01 t/s` | `44.09 t/s` | `0.980x` |

Interpretation: the naive two-stage expert pipeline is correctness-safe but loses
at short decode before spending long-context GPU time. The extra waves, scratch
aliasing, and second accumulation pass outweigh any overlap. Do not revive this
split without a counter signal showing real concurrent overlap and a design that
does not add the extra final accumulation pass.

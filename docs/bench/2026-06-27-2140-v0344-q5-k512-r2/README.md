# v0.344 Q5 K512 Routed-Down R2

Status: added and defaulted a Q5_K routed-down kernel for `f_exp=512` that uses
both half-row lane groups in each simdgroup by computing two output rows per
simdgroup. Rollback is `QWEN_DECODE_MOE_Q5_DOWN_K512_R2=0`.

## Mechanism

The v0.343 geometry inversion showed the small A3B Q5 down shape is the culprit:

| Synthetic shape | Active weight GB | GPU ms | Weight GB/s |
| --- | ---: | ---: | ---: |
| A3B-like `K512/H2048` | `0.2134` | `1.2100` | `176.4` |
| A3B bytes as `K1024/H1024` | `0.2134` | `0.8302` | `257.0` |
| A10B-like `K1024/H3072` | `0.8132` | `2.6282` | `309.4` |
| A10B bytes as `K512/H6144` | `0.8132` | `3.5972` | `226.1` |

The default Q5 kernel maps `ix=0..3` across K blocks. With `K=512`, only two
Q5_K blocks exist, so half the lanes are idle. The R2 kernel remaps `ix=0/1` to
one output row and `ix=2/3` to a second output row.

## Validation

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- `cargo test -p qwen-llm metal_single_token_concurrent_gdn_moe_matches_serial_a3b -- --nocapture`
- `qwen-bench moe-down-micro --k512-r2 --check-k512-r2` on A3B
- sequential A3B default-vs-rollback `ctx-sweep`, no parallel GPU workloads

Correctness checks:

| Gate | Result |
| --- | --- |
| Micro default vs R2 | `max_abs=0.000000`, `cos=1.000000000` |
| Decode serial vs concurrent | `max|delta| <= 0.0001`, `cos=1.000000`, argmax matched |

## Primitive Results

Real A3B Q5 routed-down, 37 Q5 layers:

| Kernel | Tokens | GPU ms | Weight GB/s |
| --- | ---: | ---: | ---: |
| rollback/default kernel | 1 | `1.1274` | `189.3` |
| R2 | 1 | `0.9147` | `233.3` |
| rollback/default kernel | 16 | `14.4364` | `236.5` |
| R2 | 16 | `8.6434` | `395.0` |

## Decode Results

Default-vs-rollback A3B `ctx-sweep --fresh-per-checkpoint --window 4`, two
alternating blocks:

| Context | Rollback | Default R2 | Read |
| --- | ---: | ---: | --- |
| `ctx128` | `106.6 / 104.7 t/s` | `108.2 / 109.8 t/s` | positive |
| `ctx1024` | `100.7 / 100.6 t/s` | `103.7 / 104.0 t/s` | positive |
| `ctx8192` | `93.8 / 94.0 t/s` | `98.0 / 97.7 t/s` | `+4.0-4.5%` |

Interpretation: this clears the long-decode gate for A3B and is shape-guarded to
`f_exp=512`, so A10B's `f_exp=1024` path is unchanged. Keep the rollback env and
continue treating broader Q5 work as work-unit/dataflow work, not scalar load
retuning.

## Artifacts

- `target/profiles/v0344-a3b-moe-down-real-default-t1-repeat.tsv`
- `target/profiles/v0344-a3b-moe-down-real-k512-r2-t1-repeat.tsv`
- `target/profiles/v0344-a3b-moe-down-real-k512-r2-tokens.tsv`
- `target/profiles/v0344-a3b-k512-r2-default-rollback.json`

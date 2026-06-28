# v0.346 Q8 K512 R4 Negative

Status: tested and removed a Q8_0 `n_in=512` mat-vec sidecar that widened the
small-K shared-down work unit to four output rows per threadgroup.

Hypothesis:

- The default Q8_0 lcpp-style mat-vec uses four simdgroups over K stripes.
- For `n_in=512` (`16` Q8 blocks), half the K-stripe simdgroups are idle.
- MoE shared down uses this small-K shape, so a no-shmem row-widened kernel could
  recover the shared-down phase without hurting large-K GDN projections.

Validation:

- `cargo fmt`
- `cargo build --release --bin qwen-bench`
- dirty sidecar correctness: `cargo test -p qwen-llm mat_vec_q8_0_k512_matches_cpu -- --nocapture`
  reported `max|delta|=2.98e-8`
- sequential A3B/A10B phase A/B and A3B/A10B ctx-sweep A/B; no parallel GPU runs

Key results:

| Row | Default sidecar | Rollback | Read |
| --- | ---: | ---: | --- |
| A3B shared down phase | `0.27 ms` | `0.56 ms` | named phase improved |
| A10B shared down phase | `0.82 ms` | `1.36 ms` | named phase improved |
| A3B `tg128` repeat 1 | `108.31 t/s` | `109.05 t/s` | regressed |
| A3B `tg128` repeat 2 | `108.56 t/s` | `108.85 t/s` | regressed |
| A3B `ctx8192` sweep | `98.0 / 98.2 t/s` | `96.4 / 98.2 t/s` | noise/mixed |
| A10B `ctx8192` sweep | `41.8 / 41.7 t/s` | `41.6 / 41.7 t/s` | neutral |

Interpretation: the mechanism was real but not production-useful. The sidecar
made a named phase faster, yet the stable `tg128` gate regressed. Do not default
or revive the Q8 K512 R4 sidecar without counter evidence explaining the lost
wall time.

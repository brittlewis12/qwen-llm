# v0.401 MoE Batch Upper-Bound Gate

Goal: decide whether A3B's strong captured MoE batching microbench is large
enough to justify an end-to-end multi-slot decode prototype. The new
`scripts/profile/moe_batch_upper_bound.py` combines a production phase profile
with a `moe-batch-sweep` output and estimates the ideal routed-FFN savings after
accounting for unbatched down-projection fallback layers.

Validation:

- `uv run scripts/profile/moe_batch_upper_bound.py` on A3B `ctx512`
- `uv run scripts/profile/moe_batch_upper_bound.py` on A3B `ctx2048`
- `cx ask` upper-bound review, session `019f1bc4-b08e-7030-8a44-367e056a197e`

Artifacts:

- `target/profiles/v0401-a3b-ctx512-phase-moe-split.out`
- `target/profiles/v0401-a3b-ctx2048-phase-moe-split.out`
- `target/profiles/v0401-a3b-ctx512-moe-batch-upper-bound.tsv`
- `target/profiles/v0401-a3b-ctx2048-moe-batch-upper-bound.tsv`

## A3B Upper Bound

The A3B phase profile at `ctx512` has `phase_sum=9.32 ms`, with MoE routed
gate/up at `1.20 ms` and routed down at `0.86 ms`. The independent-file exact
micro row at `b8` is `1.2211 ms/token` for Q4 gate/up plus Q5 down. A3B has
three Q6 down fallback layers, estimated as `0.0645 ms/token` from the down
phase share.

| Ctx | Slots | Projected routed ms | Saved ms | Saved phase | Ideal speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | `1.4630` | `0.5970` | `6.41%` | `1.068x` |
| 512 | 4 | `1.3613` | `0.6987` | `7.50%` | `1.081x` |
| 512 | 8 | `1.2856` | `0.7744` | `8.31%` | `1.091x` |
| 2048 | 2 | `1.4630` | `0.5970` | `6.32%` | `1.067x` |
| 2048 | 4 | `1.3613` | `0.6987` | `7.39%` | `1.080x` |
| 2048 | 8 | `1.2856` | `0.7744` | `8.19%` | `1.089x` |

## Decision

The A3B MoE batching micro win is real, but its ideal end-to-end decode ceiling
is only about `8-9%` before scheduler, pack/scatter, command-buffer, KV,
attention, shared-FFN, and sampling overheads. This demotes a full multi-slot
scheduler prototype as the immediate top branch.

Keep batching alive as a later system feature, but require a broader batchable
scope than routed gate/up+down alone before spending architecture effort. The
next decode work should pivot back to larger single-token byte-reduction or
fusion targets unless a future upper-bound row shows `>=12-15%` credible
end-to-end savings.

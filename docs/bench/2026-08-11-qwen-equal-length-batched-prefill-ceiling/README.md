# Qwen Equal-Length Batched-Prefill Projection Ceiling

Date: 2026-08-11

Status: `KILL` before stateful implementation.

## Question

Does flattening eight equal-length dense-Qwen prompt chunks into one larger row
slab reduce enough immutable projection work to justify building a multi-session
packed-prefill executor with lane-private attention and GDN state?

Current fixed-cohort execution prefills unrelated prompts serially. A complete
batched implementation would need sequence-major activation slabs, lane-private
RoPE/KV scatter/causal attention, and eight independent GDN recurrences. Before
building that surface, the dominant shared projections must show a useful
cross-sequence row-merging gain.

## Probe

Use the existing `decode-proj-batch` primitive against
`Qwen3.6-27B-Q4_K_M.gguf` on Apple M4 Max. Compare eight `N=512` mat-mat
executions with one `N=4096` execution for the complete production projection
inventory:

- 48 GDN QKV/Z and 48 GDN output projections;
- 16 attention Q/K/V and 16 attention output projections;
- 64 dense FFN gate/up and 64 dense FFN down projections.

The LM head is excluded from the charged comparison because production prefill
would evaluate eight final prompt rows, not all 4,096 activation rows. Each cell
uses one completed GPU sample after a warm model load. This is a ceiling probe,
not a product timing packet.

Representative commands:

```text
qwen-bench --allow-dirty decode-proj-batch \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --tokens 512 --iters 1 --warmup 0

qwen-bench --allow-dirty decode-proj-batch \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --tokens 4096 --iters 1 --warmup 0
```

The release benchmark binary was rebuilt at `f800de4`. `--allow-dirty` was
required only because two unrelated user documentation files remained untracked;
no engine or benchmark source differed from `f800de4` during these completed
samples.

## Results

| Projection family | 8 x N=512 | N=4096 | Speedup |
|---|---:|---:|---:|
| GDN QKV/Z | `2,588.455 ms` | `2,583.874 ms` | `1.002x` |
| GDN output | `1,045.654 ms` | `1,133.909 ms` | `0.922x` |
| Attention QKV | `766.170 ms` | `856.308 ms` | `0.895x` |
| Attention output | `316.465 ms` | `353.177 ms` | `0.896x` |
| Dense FFN gate/up | `7,288.660 ms` | `7,292.248 ms` | `1.000x` |
| Dense FFN down | `3,691.636 ms` | `3,552.735 ms` | `1.039x` |
| **Charged projection sum** | **`15,697.040 ms`** | **`15,772.250 ms`** | **`0.995x`** |

Including the benchmark's required input-pack/output-scatter organization moves
`16,106.983 -> 16,313.519 ms`, or `0.987x`.

The benchmark aggregate that includes a full `N=4096` vocabulary head is not a
production-prefill comparison and is deliberately excluded. Even without layout
cost, the six charged projection families are flat. Several attention and output
projections regress at the larger row slab; only FFN down improves materially.

## Interpretation

Packed prefill already gives each 512-token sequence large matrix dimensions.
Increasing the query row count to 4,096 does not create the decode-style
one-weight-pass gain: matrix kernels still tile the row dimension and repeatedly
stage the same weight blocks. It mainly changes dispatch count and scheduling,
which are already amortized at `N=512`.

This misses the preregistered `1.10x` projection-ceiling gate before charging any
of the unavoidable costs:

- eight lane-private GDN recurrent scans;
- eight lane-private causal-attention bodies and KV scatters;
- custom multi-geometry scratch and activation packing;
- frontier publication, failure poisoning, and product integration.

A full batched-prefill executor therefore cannot reasonably clear the `1.20x`
complete-prefill gate through this mechanism.

## Decision

Kill equal-length cross-sequence row merging as the next implementation lane.
Do not build the custom dense B=8 packed-prefill executor or its MoE B=16
extension from this premise.

This does not kill:

- prefix fanout, which removes duplicate token evaluation rather than merely
  reshaping it;
- decode batching, where multiple singleton rows genuinely amortize a weight
  pass;
- a future kernel that explicitly reuses weight tiles across sequence groups;
- continuous scheduling whose value comes from occupancy and queue service,
  rather than prompt mat-mat row merging.

Reopen only with a new GPU mechanism whose microproof moves dominant projection
work by at least `1.10x`, not with additional host orchestration around the same
mat-mat kernels.

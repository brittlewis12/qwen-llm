# GLM sorted expert-major prefill: rejected at the isolated-stage gate

Status: closed. The default-off candidate reused V4.1's sorted expert-major
chain and added IQ3_S to its existing generic dequant route for GLM's down
bank. The code is removed; the M5 differential and timing records remain.

## Gate result

The routed-expert stage was profiled three times per arm in interleaved order.
Both arms restored the same grouped 49,152-token prefix snapshot and processed
the same 2,048 real tokens with production routes. M5 Ultra, `iter`, pressure 0:

| Arm | Routed-expert stage samples | Median |
| --- | --- | ---: |
| Grouped incumbent | 721.45, 720.61, 723.53 ms | 721.45 ms |
| Sorted expert-major | 705.56, 705.13, 706.27 ms | 705.56 ms |

The stage speedup was **1.023x**, below the required 1.25x. Stage-profile wall
medians were 2,014.36 ms grouped and 1,982.03 ms sorted, but the ordinary-wall
paired gate was not reached. The candidate also did not reach model-level
quality or serve gates.

The selected geometry matches the census occupancy: 56.9 mean routed rows per
expert at 2,048 rows (p50 41, p90 125). The prefix and input rows were held
constant across arms. Background `~/ghost` compiler work was visible in the
pre-load snapshot; load average was 15.83/13.61/16.56 before and 16.58/14.52/16.65
after. The pre-load snapshot had no resident model. Pressure remained 0.

The scratch plan reused the admitted normalized-half buffer and the existing
F32 inner/slot slabs, so the candidate added no session allocation. The IQ3_S
extension passed the existing M5 sorted-kernel differential in the test
profile with `MTL_DEBUG_LAYER=1`: sorted inner activations and down outputs
matched the rectangular path bitwise across IQ formats, partial/tail counts,
and empty expert runs. This is a kernel-level exactness result, not a
full-model quality result.

**Decision: remove the sorted candidate.** It failed the isolated-stage gate;
no ordinary-wall A/B or serve run was made. The next census lever is sparse
MLA scoring and attention: 376.71 ms and 396.36 ms respectively at the 49K +
2,048 cell, with modeled tensor utilization below 1%.

## Source and license

The implementation reused the existing sorted path in
`crates/qwen-llm/src/metal/moe_grouped_generic/sorted_tensor.rs`. Its kernel
header, `kernels/tensor/v41_sorted_experts.metalh`, retains the MIT notice and
attribution to mlx-serve v26.11.2. The GLM change added an IQ3_S instantiation
using the repository's existing generic tile decoder; no new third-party code
was copied. The V4.1 selector remains restricted to its previous dtype set.

## Evidence

- Same-prefix stage comparison and device metadata:
  [`sorted-expert-major/candidate.jsonl`](2026-11-13-glm-prefill/sorted-expert-major/candidate.jsonl).
- IQ3_S differential output, thermal samples, process snapshots, load averages,
  corpus hash, and exit status:
  [`sorted-expert-major/`](2026-11-13-glm-prefill/sorted-expert-major/).
- Baseline stage census:
  [GLM prefill baseline](2026-11-13-glm-prefill-baseline.md).
- M5 run directory: `~/qwen-runs/2026-11-12-glm-prefill/sorted-expert-stage/`.

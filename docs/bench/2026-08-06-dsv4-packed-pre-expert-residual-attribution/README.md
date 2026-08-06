# DeepSeek V4 Packed Pre-Expert Residual Attribution

Status: `BeforeAttentionBody` is `GO` for decomposition only;
`AfterAttentionOutput` is `KILL` as a standalone current-asset N=128 target.
No new GPU work is executed, and no kernel or savings claim is authorized.

## Question

The accepted four-pass packed-attention packet recorded four complete physical
stage intervals but formally adjudicated only attention body and attention
output. After the grouped-IQ2 scalar lane closes, recover the two residual
intervals from that already accepted evidence before paying for another run:

1. `BeforeAttentionBody`: command ingress through chronological publication.
2. `AfterAttentionOutput`: attention hyper post through router projection.

The two intervals are independent decisions. Do not add them, combine them with
post-route work, or treat either containing interval as removable cost.

## Evidence Reuse

The original packet already accepted:

- exact packed logits, normalized hidden output, restored continuation logits,
  causal identities, committed tokens, dispatch order, and complete geometry;
- three ordinary controls and two sampled runs inside 2.019% and 2.447% drift;
- at most 0.882% sampled-topology perturbation;
- aggregate timestamp coverage within rounding error and transition ambiguity
  below 0.006%; and
- physical timings for all four stages in every one of 43 layers.

The accepted log is copied into this packet unchanged with SHA-256
`3fd96776530bd90903513aa790488cf314a81ce5a1ccfe96158b9d18a589278b`.
The original packet's source and executable identities remain:

- `deepseek_v4_metal.rs`:
  `bc39d52d7cf62782d1baec90f74d81e3b3dbacb85180aa213c7b408b85b5e92d`;
- `deepseek_v4_metal/prefill.rs`:
  `7987bb878af8963b93ba6826f7425914407654f419ea31127a1f3afc689083d7`;
- release diagnostics executable:
  `f623c458299c7a633d3af7d5b3496490eb5627cb90d51012a4fef113a0a1ce47`.

This retrospective changes no observer, sample, filter, or formula. It only
applies the packet's accepted adjudication method to stage indices zero and
three, which were emitted but not named decision targets.

## Method

For each sample, multiply the observed stage share by its interpolated ordinary
control. Use the conventional median across the two normalized values. Define:

```text
common_uncertainty = max(transition_uncertainty,
                         abs(sampled_topology_perturbation))
stage_uncertainty = max(common_uncertainty,
                        abs(stage_share_0 - stage_share_1))
lower_share = mean(stage_shares) - stage_uncertainty
lower_ms = median(normalized_ms)
           - stage_uncertainty * ordinary_gpu_median_ms
```

The common uncertainty is 0.881965%; the ordinary-control median is
1,067.138708 ms. Require:

- lower share at least 15%;
- lower time at least the current 158.3 ms floor, stricter than the original
  packet's 150 ms floor; and
- mean CSA and HCA stage shares each at least 10%.

`analyze.py` checks the accepted log hash, parses only the retained records,
recomputes every value, and fails if either frozen result changes. Its complete
output is `analysis.json`.

## Results

| Stage | Normalized samples | Median | Charged uncertainty | Lower time / share | CSA / HCA mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| Before attention | 481.678 / 499.786 ms | 490.732 ms | 1.233408% | 477.570 ms / 44.7712% | 47.6961% / 43.4194% | GO, decompose only |
| After output | 66.433 / 65.557 ms | 65.995 ms | 0.881965% | 56.583 ms / 5.3057% | 5.9380% / 6.5423% | KILL standalone |

The before-attention repeat-share delta dominates its uncertainty. CSA/HCA
repeat deltas are 0.2092 and 2.5553 percentage points, both inside the original
three-point cohort rule. The after-output repeat-share delta is only 0.1445
points; its CSA/HCA repeat deltas are 0.0287 and 0.2962 points.

## Scope

At the captured revision, `BeforeAttentionBody` contains raw-ring preservation,
layer-zero embedding/repeat setup, attention mHC pre, attention normalization
and Q/KV preparation, compressor projections, and the chronological row loop.
The separately measured chronological loop remains closed as a standalone
target. A 477.570 ms containing-phase lower bound authorizes decomposition, not
any one fusion or kernel.

`AfterAttentionOutput` contains attention mHC post, FFN mHC pre, normalization,
and the router projection. Its lower time, aggregate share, CSA share, and HCA
share each miss independently. Close it only as a standalone current-asset,
synthetic-initial-prefix N=128 target; a separately measured cross-boundary
composition may still reopen relevant work.

## Decision

Do not execute another broad packed-prefill profiler. Decompose only
`BeforeAttentionBody`, beginning with an exact operation, dispatch, intermediate,
and materialization census. Before implementation, give impossible zero-work
credit to the exact candidate-touched subset and require it to retain at least
158.3 ms after uncertainty.

Stop if the leading residual decomposes entirely into the already-closed
chronological interval and sub-threshold components. Do not reopen grouped-IQ2,
Q8 output, all-IQ3 widening, GPU route weights, or HCA tiling through this result.

CX review and independent recomputation:
`019fd685-acb3-7890-83c9-192cbea48c6e`.

## Reproduction

```bash
uv run \
  docs/bench/2026-08-06-dsv4-packed-pre-expert-residual-attribution/analyze.py
```

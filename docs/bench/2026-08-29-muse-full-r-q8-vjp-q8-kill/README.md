# Muse Full-R Q8 Multi-Query VJP KILL

Decision: **KILL** the query-serial eight-query Q8 activation-VJP kernel. The
experiment is removed completely; the singleton kernel remains unchanged.

## Hypothesis

The existing frozen Q8 activation VJP assigns each query an independent grid
slice and scans every quantized weight block once per query. The candidate kept
eight independent accumulators in one SIMDgroup so one weight load could serve
eight queries while preserving each query's output-row accumulation order.

The preregistered gate required raw-bit equality, positive savings in both
balanced comparisons, at least 4x on each dominant Muse FFN shape, and at least
5.5x geometric-mean speedup across Q=128 and Q=512.

## Correctness

Model-free release tests passed raw-F32-bit equality against the incumbent for
Q in `{1, 4, 5, 8, 9, 128, 512}`. The Q=9 case also preserved offset-backed
input/output guards and reduced dispatch depth from nine query slices to two.
The complete focused correctness run took 0.11 seconds after compilation.

## Falsifier

The first production-shape arm used Muse gate/up geometry
`[n_in=6656,n_out=19968]`, Q=128, synthetic resident Q8 weights, and one dispatch
per fresh command after one warmup per implementation.

| Arm | GPU time (ms) |
|:--|--:|
| B1 incumbent | 94.218125 |
| C1 query-tile 8 | 111.482375 |
| C2 query-tile 8 | 111.068750 |
| B2 incumbent | 94.412500 |

Mean GPU time regressed `94.315313 -> 111.275563 ms`, or 17.99%; reported
speedup was `0.848x`. Both balanced comparisons regressed. This fails the first
mandatory gate, so Q=512 and the down-projection shape did not run. The complete
timing test terminated after 0.67 seconds.

The result falsifies serializing eight queries inside one SIMDgroup as the
weight-reuse mechanism. It does not establish whether register pressure,
broadcast cost, lost query parallelism, or cache reuse in the incumbent is the
dominant cause. Do not adaptively retile this kernel. The next source screen
must retain query parallelism, such as a matrix path or an explicitly 2D tiled
transpose design.

Adversarial design and disposition: `01a05072-6951-7782-978a-2f274d90f474`.

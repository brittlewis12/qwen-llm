# N16 long-context online attention: primitive PASS

Test-only `4a8ef407`, built on `d06c3e31`, uses existing kernels and changes no
production selector, scratch constructor, policy or numerical contract.

## Mechanism and scope

Current dense G6 physical-N16 verification attends each causal row separately.
The experimental work unit shares queries through existing two-pass online matrix
attention, rebuilding one packet-local V-transpose first. This is neither the
closed one-pass flash design nor the N2-8 non-online direct-V verifier path.

Geometry is 24 query heads, 4 KV heads, head256, 16 queries and final KV extent32768
(base32752). The real dense27 model has 16 full-attention layers, not 27. This
model-free screen measures only one attention body, not a full layer or verifier.
Q/K/V have deterministic nonperiodic values with variance approximately one.
Both arms omit RoPE, scatter, projections, GDN, checkpoints and output heads.

A uses current V4 NWG128/C32 with one encoder per row and causal extent
`base + row + 1`. B uses one encoder for full-prefix transpose, online KQ and
normalized KQV. GPU duration is primary; encode-through-completion wall secondary.
No transpose cost is amortized away or excluded.

## Frozen packet and result

Warmup A-B-B-A once, measured A-B-B-A with eight invocations per arm. Frozen
screen: >=20% aggregate and both-pair GPU savings, <=5% control spread,
incremental actual VT/scores/ml allocation <=128 MiB, every row cosine >=0.9999
and maximum absolute error <0.005. Extent32769 is correctness-only, afterward.

| Arm | GPU ms/body | Encode-through-completion ms/body |
| --- | ---: | ---: |
| A1 | 6.237693 | 6.269172 |
| B1 | 3.606734 | 3.627781 |
| B2 | 3.607823 | 3.628927 |
| A2 | 6.263266 | 6.288578 |

GPU means 6.250479 -> 3.607279 ms, saving **42.288%**. Both pairs pass;
control spread is 0.409%. The one-layer screen passes, not an endpoint gate.
The retained instrument emits raw timings rather than compiling the experiment's
performance decision; the original packet's decision record remains in its log.
Subsequent single-order diagnostics are transpose1.721953, KQ0.528401 and
KQV1.341734 ms. They are attribution only, not separately selected best times.

Incremental VT/scores/ml logical and actual allocation both equal 93,847,552 B
at32768. At32769 they are 93,853,440 / 93,896,704 B. These are not total verifier
scratch, session memory, peak process footprint or full-request admission claims.

All16 row outputs are finite. Worst row cosine is 0.999999918401 at32768 and
0.999999918636 at32769; maximum absolute errors are 0.000016246 and 0.000015032.
The timed packet's final output matches its initial numerical check. No bitwise,
full-model state, greedy acceptance or sampled-distribution claim follows.

## Execution and review

One release test passes. An external guardian acquires and holds the ordinary
production Metal lease around the unit test's per-PID lease. It waits for existing
owners without interruption; no whole-model allocation or lease bypass occurs.
Global compression/decompression, swapin/out and pageout counters do not grow.

All attempts remain in `target/profiles/verifier-online-n16/`: `build.log` fails
because this package has no `metal` feature; `build-02.log` fails on a missing
command-status import. `build-03.log` succeeds; `build-04.log` rebuilds the reviewed
status/error checks after rebase. Only one GPU packet runs: `screen.log`,
`execution.json`, guardian `run.py` and pre/postflight counter files.

Independent Luna review checks causal indexing, synchronization, allocation scope
and protocol before execution, then confirms the bounded PASS. It does not confer
production authority. Full verifier cost, numerical state, partial restore and
exact-replay economics remain unmeasured; multiplying this layer delta by16 is
not a substitute. All existing verifier and single-chunk-VT safety gates remain.

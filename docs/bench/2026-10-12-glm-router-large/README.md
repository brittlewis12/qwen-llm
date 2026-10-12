# Strict router at the actual large workspace

The current N2048 Half profile uses 42 generic F32 router projections because
production admits the strict E8P32 kernel at exactly N128/N512. Its primitive
supports any positive query count, positive K, and E divisible by eight.

Test the existing kernel at K4096/E288/N2048 before changing selection. Preserve
all other production paths, including Half latent MMA. Warm A/B followed by
ordinary ABBA, routes/weights and join/three-step output comparison; replay the
final-block normalized input for a full bitwise projection comparison and
independent sampled F64 dot-product forward-error check.

CPU error-bound test passes. The initial unleased primitive run timed out while
other GPU validation was active; its subsequent explicitly leased run passes
in0.23 seconds, including the actual E288/N2048 case. Do not treat the timeout
as a numerical result. The benchmark packet takes its own production lease.

## Results And Selection

All packets use current production Half latent MMA and change only router
projection selection. Actual N2048/E288 primitive bit checks pass, as do odd
K/E shapes (K7/E8 and K257/E24 at N129). Model leaf projections, route IDs and
weights, and full logits through three common continuation steps match bits.
The sampled F64 dot check uses a derived gamma(2*K) forward-error bound, not
an empirically adjusted quality threshold.

| Packet | Mean wall A/B | Mean GPU A/B | Paired wall A/B |
| --- | ---: | ---: | --- |
| N2048 forced strict | 1.03364 | 1.03346 | 1.02923, 1.03797 |
| N2048 production, unforced B | 1.02117 | 1.02138 | 1.01214, 1.03018 |
| N129 production, unforced B | 1.00772 | 1.00791 | 1.00511, 1.01035 |

Raw files retain every attempt, including the N2048 warm A wall of about82.5
seconds versus7.9 seconds GPU. Warmups, allocation, leaf checks and continuation
are outside measured ABBA. These are warmed prefill gains, not cold TTFT gains;
do not hide the warm-only gap or attribute it uniquely to a CPU mechanism.

Production now uses the strict kernel for Fast lineage, F32 router weights,
rows>=128, positive u32 K/E/rows and E divisible by8. It no longer whitelists
H4096/E288 or exactly N128/N512. Pipeline execution-width/threadgroup-capacity
checks and generic fallback remain. Small batches retain generic based on the
historical N32 regression;128 is a conservative performance boundary, not a
claim that127 is architecturally invalid. Exact and other storage dtypes retain
their existing paths. No new persistent buffers, production knob or quant gate.

Production B leaves selection unforced while the test-only observer records
timestamps/substitutions. Warm census confirms42 strict routers. CPU tests
cover neighboring widths, independent valid shapes, invalid geometry, capability
declines, nested/unwind capture and unchanged operand-precision policy.

Validation before main refresh:90 GLM tests and the strict primitive test pass
under a real external lease plus Metal validation; all-target release check
passes. The model packets use their own internal lease and no Metal validation.
Serving session reuse from `77d9fc56` and optional-scratch recovery from
`65b09776` are incorporated before integration, without changing their files.

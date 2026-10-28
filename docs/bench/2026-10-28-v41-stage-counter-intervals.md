# FFN Counter Intervals Are Not Exclusive Stage Costs

A fixed 4K FFN experiment completed its production capture, then stopped during
the first sampled incumbent replay because its diagnostic required all 16
timestamp values to follow stage-index order. No candidate, warm A/B comparison,
or primary ABBA block ran. Replay bitwise verification was also not reached.
That failed packet is neither a kernel correctness result nor a performance
verdict, and its original failure status remains unchanged.

The source assigns eight start/end pairs to HC, router, shared expert,
route/bucket, gate/up, mid, down and sum/post. Their semantic labels must remain
attached to those pairs. Encoding order alone does not establish globally
nonoverlapping timestamp envelopes: independent shared-expert and routing work
can overlap. A source-derived data dependency also does not, without calibrated
boundary semantics, make counter envelopes a direct resource-fence observation.

## Immutable Observation

The triggering diagnostic used M5 Ultra, SDK 27 and a native release build.
Its eight individual intervals are positive and within the measured command
envelope. The shared and route intervals have the same start; the shorter route
interval lies within the shared interval. Mid and down also have a small
dependent-boundary overlap whose interpretation remains unresolved.

| Diagnostic Quantity | Milliseconds |
| --- | ---: |
| Command counter hull | 223.638750 |
| Sum of eight stage intervals | 64.427500 |
| Union of those intervals | 60.422500 |
| Overlap excess, sum minus union | 4.005000 |
| Time covered by at least two intervals | 4.005000 |
| Uncovered time inside the hull | 163.216250 |
| Shared/route overlap | 4.004125 |
| Mid/down boundary overlap | 0.000875 |

The two overlap quantities happen to agree here because there is no triple
coverage. They differ in general. Subtracting the stage sum from the hull would
give 159.211250 ms, which is not the uncovered interval length.

Neither the union nor its complement proves shader-active time, occupancy,
idleness, scheduling overhead or the cost removable by a kernel change.
Stage-sum fractions are not exclusive time shares or inputs to an Amdahl model.

## CPU Analysis Contract

`scripts/reference/deepseek_v41/stage_intervals.py` analyzes named sample pairs
without sorting or coercing the supplied records. It validates plain positive
u64 values, sentinel rejection, pair orientation, complete unique sample-index
mapping and absolute command-envelope containment. Its fixed 5,000 ns allowance
applies only to the cross-API counter/command envelope comparison.

The analyzer uses exact integer half-open interval arithmetic on a separate
copy to compute union, multiply covered time, overlap excess and uncovered
intervals. It retains signed adjacent-encoding gaps and source-derived dependency
observations. The -875 ns mid-to-down observation remains visible; the cross-API
allowance is not silently repurposed to excuse it or establish a race.

The regression fixture contains normalized offsets starting at 1 ns, not
original host-clock values or execution-session metadata. Tests cover the real
interval geometry, independent overlaps, explicit mappings, triple coverage,
touching boundaries, absolute-envelope errors and large integer arithmetic.
These CPU checks do not perform GPU work or revise any benchmark acceptance.

## Future Measurements

Keep checked GPU completion, correctness verification and diagnostic
interpretation as separate recorded outcomes. Run and preserve host-side bitwise
verification before fallible diagnostic analysis can mask it, with both outside
timed intervals. Preserve the actual selected counter-set layout and sample-index
mapping as evidence rather than assuming every set containing a timestamp has
the same resolved representation.

A corrected diagnostic does not turn the original packet into a pass. Any
follow-up uses its own frozen source and bounded protocol, while retaining the
failed packet and keeping workload, correctness and comparative performance
gates unchanged.

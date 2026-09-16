# Stage Interval Instrumentation Diagnostic

Separate model-free protocol, frozen after MoE observation01 fails its global
timestamp monotonicity gate. Observation01 remains failed, with no stage budget;
its raw counters were unfortunately lost because persistence followed assertions.
Fix persistence before gates without rerunning that packet or relaxing its gate.

Apple documents serial dispatch within a compute pass, not globally disjoint
counter intervals across encoders. Overlap is plausible, not yet diagnosed.
References:
- https://developer.apple.com/documentation/metal/mtldispatchtype/serial
- https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers

One model-free command,96 sampled serial encoders, each copying65536 finite F32
values between two ping-pong buffers. Read-after-write dependencies match ordinary
tracked resource handling; no added fences or waits to force disjoint samples.
Production lease/real wired gate before Metal, API validation on. No model load.

Save status and GPU endpoints first, then all192 counters or resolution error
before any counter gates. Compare both final buffers bitwise to deterministic
input. Classify invalid/sentinel samples, nonpositive individual spans, adjacent
cross-encoder overlaps, and reversed start ordering separately. Reject invalid
individual spans. One run, no retries and no throughput/attribution claim.

Do not use summed overlapping intervals as exclusive costs or call command time
minus that sum unattributed time. This diagnostic informs a later explicitly
versioned MoE instrument design; it cannot retroactively qualify observation01.

## Follow-up: Saved Layer2 Interval Classification

The model-free control passes with zero overlaps; it does not reproduce the
original anomaly. Freeze one further diagnostic using preserved observation01
layer2 input/output/IDs/top-k, each SHA-pinned in the harness. Load existing weights
but execute no prefix and no native forward. One16-chain sampled command with
the same six MoE stages; no warm/bracket/throughput measurement, normalization,
candidate kernel, or timing comparison. This investigates the interval failure,
not a retry of the failed performance protocol.

Persist status and raw counters before gates; verify native output/routes bitwise
and input unchanged. Classify sentinel/nonpositive-pair errors independently from
adjacent overlaps and reversed starts. Report indices of overlaps for source
mapping. One attempt only. Observation01 remains timing-invalid either way.

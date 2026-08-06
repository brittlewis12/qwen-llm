# Adjudication

Decision: `INCONCLUSIVE_HOLD`.

The sole frozen execution failed during the warm-control identity preflight,
before the sampled warm arm or timed `A/B/A/B/A` campaign. It therefore contains
no timing evidence and is neither an optimization authorization nor a lane
kill.

The expected committed-token SHA-256 literal in the frozen source had only 63
characters:

```text
7c7d71da5dfb70e9d6f0555d949f7476acd5a68173331a228537c9e46dc728e
```

The observed value was the independently retained canonical 64-character
identity:

```text
7c7d71da5df3b70e9d6f0555d949f7476acd5a68173331a228537c9e46dc728e
```

That canonical value predates this experiment in
`../2026-08-05-dsv4-packed-attention-attribution/README.md`. The failure is a
clerical fixture defect, not a value inferred from this execution.

The packet remains immutable and receives no retry. CX adjudicated that one
separately frozen corrected packet is principled because this packet exposed no
timing result or candidate direction. The corrected packet must retain every
boundary, campaign, gate, and classifier while changing only the canonical
digest and a compile-time fixed-width hex preflight.

CX session: `019fd685-acb3-7890-83c9-192cbea48c6e`.

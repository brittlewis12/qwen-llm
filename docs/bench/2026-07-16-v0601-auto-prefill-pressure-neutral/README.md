# v0.601 Pressure-Neutral Auto-Prefill Confirmation

Status: **inconclusive; no authority**.

Frozen source: `bfa3902bdf2a9debb8f1b94443b1d1bacbef1fab`.
Artifact: `target/profiles/v0601-auto-prefill-pressure-neutral-p1/`.

## Outcome

The sole packet completes the full A3B block, then stops before A10B's first child
on its preregistered cache-interval compression predicate. Completion identity
passes. No retry or successor packet is authorized.

All eight A3B children and four `AB/BA/BA/AB` pairs are valid:

- exact output bytes and runtime identity across every row;
- exact A3B schema-5 candidate topology `2048/1024`, `60/120` calls, and overlay;
- complete 41-allocation plan and reconciled memory admission;
- valid AC/thermal/memory and VM signals, zero block input;
- pair speedups `1.052868/1.052415/1.052678/1.053172x`;
- median `1.052773x`, AB/BA medians `1.053020/1.052547x`, and 4/4 wins;
- median A/B TTFT `7457.786/7082.920 ms`, a paired median saving of `373.859 ms`.

The A3B subset mechanically clears every profile speed gate. It remains
non-authoritative because the frozen contract evaluates performance only after all
16 children complete.

A10B cache conditioning reads 77,029,996,032 bytes and waits 120.010 seconds. The
interval records:

- Pageouts `+985`;
- Compressions `+76`;
- Swapouts growth `0`;
- swap-occupancy growth `0`;
- memory availability `96%` at both endpoints;
- valid AC and thermal state.

The compression count represents 76 system-wide events, not 1.1875 MiB of net
compressor growth or activity caused by the asset. Pages stored in the compressor
fall by 235 and occupied compressor pages fall by 130 during the interval. The
preregistered zero-event predicate nevertheless requires immediate invalidation.

## Implication

v0.567, v0.568, and the complete v0.601 A3B subset make the A3B performance belief
high, but governance remains incomplete. Neither profile receives default-on
authority. The exact memory-admitted implementation remains available via
`--prefill-chunk auto`.

The confirmation chain ends here. Future cold protocols must distinguish
storage-cold, process-cold/cache-warm, and model-ready objectives rather than use
one cumulative system-wide event veto across asset sizes and conditioning windows.
The active queue returns to the topology-preserving parallel-copied A3B loader
pilot authorized by v0.599.

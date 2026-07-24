# v0.621 A3B Direct-Pread Auto Admission

Status: preregistered; selector implementation is present and the packet is
unrun.

## Intent

Authorize the certified direct-pread population method for the existing
authenticated disposable A3B Auto policy without changing explicit overrides,
fallback, reusable loads, or prefetch policy.

The selector commit changes only the Auto population choice for the exact A3B
profile. Explicit true remains forced mmap parallel copy, explicit `pread`
remains forced pread, explicit false remains ordinary copied rollback, and any
storage/representation override suppresses Auto. Reusable Runtime and JSONL
loads remain outside Auto admission.

## Evidence And Identity

Import and verify the complete sealed v0.620 packet, including all inventory
members, decision, completion seal, output hash, force-only GO, and its v0.619
evidence bridge. Pin selector commit
`532f7a5db8aaa73fa1ad8f8cd1fde17ae8191413`; require that it changes only
`crates/qwen-llm/src/metal_forward.rs` from parent `545c264`, and require this
packet commit to be its direct child with only this contract and runner added.

Hash current source, runner, contracts, common helper, model, prompt, `qwen`,
`qwen-bench`, and `first_byte_spike`. Require clean matching build/runtime
identity, embedded current identity in `qwen`, the hashed example binary, and
observed Auto-policy/pread markers from the example.

## Stage 1: Actual Auto Product Confirmation

Use the v0.620 six-pair output-128 shape and fixed
`AB BA BA AB AB BA` order.

- A sets only `QWEN_GGUF_PARALLEL_COPY=1` and must emit the forced mmap marker.
- B leaves `QWEN_GGUF_PARALLEL_COPY` and every Auto-suppressing variable absent.
  It must emit native embedding policy, exact Auto policy
  `profile=a3b-q4km-v1`, the pread marker, then the copied ledger.
- Any other `[metal-gguf-*]` occurrence is a contract defect.
- Every child must emit the frozen 590-byte stdout with SHA-256
  `e18fd50a1e2add01cfed4f498bf0052b630653517057cda3620b302d1a68f198`.

Retain all v0.620 L/Q/F/E, order-stratum, win, memory, pressure, fault, timing,
and output gates unchanged. Fresh prefill and model-ready TTFT remain recorded
diagnostics.

## Stage 2: Storage-Cold Composition Guard

Run only after Stage 1 passes. Use the release `first_byte_spike` example with
disposable intent, the frozen short prompt, output-1, and in-child targeted
`MS_SYNC | MS_INVALIDATE`. Processes never overlap.

Run two counterbalanced pairs:

```text
MP  M=forced mmap + ColdOnly, P=actual Auto pread + ColdOnly
PM  P=actual Auto pread + ColdOnly, M=forced mmap + ColdOnly
```

Every child must require:

- invalidation ends at zero of exactly 1,350,985 pages;
- total physical reads are at least 16.50 GiB, a rounded 80% proxy;
- post-arm residency is at least 99%;
- M emits forced mmap with no Auto policy; P emits exact Auto policy plus pread;
- ColdOnly returns exactly 20.61 GiB from one full shard with none skipped;
- identical first token, valid host/pressure state, and complete process
  metrics.

Default-admission gates for P versus M, in both pairs:

- `load_P - load_M <= 112 ms`;
- `first_byte_P - first_byte_M <= 112 ms`;
- physical-read, total-CPU, and physical-footprint P/M ratios each `<=1.10`,
  `<=1.10`, and `<=1.05` respectively.

This is target-file buffer-cache cold with certified physical I/O, not an
untouched-media start: packet conditioning reads the model before the child
invalidates its pages. Whether Auto pread should suppress the preceding
ColdOnly pass remains a separate follow-up and cannot rescue this admission.

## Decision And Authority

Any bridge, identity, selector, marker, output, validity, or Stage-1 defect
stops without authority. A Stage-1 performance miss kills admission. A valid
M/P cold miss also kills admission. Only the complete conjunction authorizes
the existing disposable single-turn A3B Auto selector to use direct pread.

Malformed command, timing, marker, or resource evidence is a contract defect.
Valid host/pressure confounds are inconclusive and never become performance
losses or retry authority.

No authority extends to reusable/JSONL loads, other assets, concurrent loading,
energy, physical-memory savings, or prefetch suppression.

# V4.1 PV delivery superseded by current all-head attention

## Disposition: superseded exact delivery

Source assessment baseline: `7c16da89`, which already contains `e0230244`
(all-head tensor attention). Historical candidate: `50cbafad`, based on
`90ee3393`. The exact old H64/D512 axis64 kernel is not a contract-preserving
replacement for the current default PV. This closure changes documentation only;
no kernel, host selector, test, inventory, identity gate, or environment reference
is changed. No new runtime knob or device/artifact admission policy is introduced.
Current CED, Engram, decode, serve, preparation, publication, and lineage code
remain untouched. The old full-model harness is not imported.

## Source contract and resources

The current `attention_gpu/query_batch.rs` tensor64 branch selects
`score_tensor64` and `pv_tensor64` for H64/D512 when tensor arithmetic and the
existing tensor-tier selectors permit it. `QWEN_V41_SINK_TENSOR64=0` restores
the incumbent 16-head score/PV route. Other head counts and non-tensor tiers keep
their existing routes. Logical arithmetic lineage, not just physical tile width,
controls eligibility, including compacted singletons.

| PV form | Threadgroups/query | Tile H x D x K | Explicit half operands/TG | Invalidity gate |
| --- | --- | --- | --- | --- |
| Current `pv_tensor64` | 1 | 64 x 128 x 32, four D slices serially | 12 KiB | Head 0 suppresses all 64 heads |
| Historical `pv_tensor_axis64` | 4 | 64 x 128 x 16, disjoint D slices | 6 KiB | Heads 0/16/32/48 each suppress 16 heads |

Both use half operands and F32 accumulators, with 8192 logical accumulator
elements per active tile and four SIMD groups per threadgroup. Explicit operand
storage is not a measured compiled resource footprint or occupancy result.
Current PV already shares each V slice across all heads; the old removal of
three redundant V constructions is not a current-source traffic benefit.
Only a different dimension-parallel schedule/resource tradeoff remains a
hypothesis, with no promised gain or measured current PV share.

The current PV reads the head-0 denominator sentinel before gathering V. If it
is `-1`, all unnormalized outputs are zero without V gathers. Otherwise, it
accumulates all heads; shared `normalize_rows` independently zeros each head
whose own denominator is `-1` and checks output finiteness. A sentinel at head
16 alone therefore does not suppress heads 17 through 31. The historical kernel
does suppress those neighbors, including explicit post-store zeroing to prevent
NaN V leaking through zero P. Conversely, head 0 alone does not suppress all
heads in the historical kernel. These are observable differences, not permission
to change current invalidity semantics. K16 versus K32 also changes the matmul
reduction partition; equal operand precision does not prove equal output bits.

## Proof obligations before any new candidate

Independent read-only review precedes CPU/GPU qualification. A future proposal
must first establish the current single-gate contract explicitly; silently
porting the old four-gate behavior is rejected. Any revised kernel is a new
candidate, not the exact historically qualified kernel.

- Keep current `score_tensor64`, softmax, normalization, buffers, preparation,
  publication, and lineage identical in both arms. Compare current `pv_tensor64`
  with dimension-parallel PV only, not with the old `pv_tensor` baseline.
- Candidate selection must be test-only opt-in, default off even on an isolated
  branch, with nested thread-local restoration. Actual CPU tests must cover
  eligibility, nested scopes, panic unwinding, thread isolation, and source
  contracts; fallback selectors and noneligible geometry must stay unchanged.
- Exact current-baseline bit comparison has priority: compare score slab bits,
  pre/post-normalization output bits, status, input immutability, and canaries.
  Include all-valid/all-invalid rows; independent sentinels at heads 0/16/32/48
  and non-first heads; NaN V introduced after softmax; nonfinite query/sink;
  invalid IDs with no unsafe V gather; raw-ring wrap; and K16/K32 panel tails.
- Adapt small H16/H64, singleton, and CED lineage regressions to current APIs.
  Historical tests encode the old four-gate contract and cannot be copied as
  current expected results. Existing tolerance-based tensor64/tensor32 witnesses
  are not an exact current-PV bit proof.
- Preserve the independent strict half-versus-F32 accuracy gate and its tolerance.
  A separate baseline-relative gate must not relabel an absolute failure as a
  pass. Do not port the old Full-N512/2048 model helpers or their stale resource,
  cancellation, or lease assumptions.

## Evidence limits

The historical warmed 2K model comparison reported a 1.93% baseline-relative
improvement and equal model vector/state bits against its historical incumbent.
The independent strict half-versus-F32 accuracy bound failed at output element
739; the complete set of 14 failing indices and stage/output bits was identical
in old and new arms. This is not an absolute-accuracy pass and is not
current-baseline equivalence or performance evidence.

No fresh Rust typecheck, Rust CPU tests, GPU/context execution, model validation,
or performance measurement is claimed here. Exact current-source equivalence
and current 8K CED model qualification/performance remain unproved. No new GPU
work is launched by this closure. A different candidate needs fresh source-bound
review and qualification under the current execution protocol; the historical
packet cannot qualify it.

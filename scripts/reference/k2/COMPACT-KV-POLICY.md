# K2 compact KV v1 experiment

This is an experimental cache representation, not a default change or model
quality qualification. F16 remains the application cache. No eviction, truncation,
rolling history, or full-history float expansion is allowed.

## Storage and arithmetic

- Canonical arena order: layer, K-or-V, position, KV head, channel block.
- Each 32-channel block is two little-endian F16 scale bytes and 32 signed i8
  codes, with no padding. Four blocks per 128-channel head, eight KV heads.
- Each K or V row is 1088 bytes. Both rows across 36 layers cost 78336 bytes/token,
  versus 147456 for F16: 46.875% less logical KV. Allocator/page/reserve overhead,
  model weights and other scratch are separate.
- Quantization uses original finite F32 `d = max(abs(x)) / 127` and `1/d`, with
  ties-away-from-zero rounding and clamping to [-127,127]. The stored scale is
  rounded to F16; decoding uses that stored scale, not the original F32 scale.
- Any nonfinite input or scale overflow emits quiet-NaN scale bits `0x7e00` and
  zero payload. Bit-pattern checks are used in Metal to survive fast math.
- Zero blocks and F16 scale underflow (including F32 division underflow) produce
  canonical positive-zero scale and all-zero payload without reciprocal/casting.
- The CPU commit check rejects nonfinite/negative scales, -128 codes, and nonzero
  payload with zero scale. Invalid blocks must never become committed history.
- Attention reads byte-addressed codes/scales, reconstructing only four values
  per lane in registers. The 34-byte blocks do not support aligned vector payload
  loads. Query/output vectors are separately checked for float4 alignment.

## Primitive gates, declared before GPU execution

Curated CPU/GPU byte equality covers positive/negative ties, signed zero, F16 and
F32 scale underflow, a finite maximum F16 scale, NaN/Inf, and scale overflow.
For bounded general finite values with nonzero stored scale, decoded error must
not exceed `0.5*d + 127*abs(d-d16) + 1e-4*d`. The final term covers F32 reciprocal/
rounding arithmetic; this is not a full-model tolerance. Canonical-zero blocks
instead have error bounded by their original maximum magnitude, even if F32 `d`
itself underflows to zero.

Inline attention on actual stored bytes must match independently decoded F64
softmax/reduction to absolute error below 2e-5 for the bounded synthetic corpus:
visible lengths 1/32/33/128/256/257, flat/ordinary/sharp queries, all 32 heads,
nonzero arena/query/output offsets, poisoned future/layer/guard storage, and
immutable source/cache controls. Storage, dtype, extent, access, alias, or cache
kind mismatch must fail closed. Source and application context guards do not grow.

## Runtime and quality gates

Private research loading must price and actually allocate the compact arena,
share the same block graph and transaction machinery, and check every newly
stored row. Submitted failure poisons the append without partial prefix commit.
Prove singleton/packed self-consistency, causal future exclusion, capture/readout/
intervention behavior, and capacity refusal before any application promotion.

Full-model F16 versus Q8 drift may be measured on the already observed, pinned v2
corpora. Report all numerical metrics and live ranking witnesses, including
unchanged v2 gate failures and exact trajectory disagreements. This is a diagnostic
of additional quantization error, not a new holdout or automatic pass/promotion.
Do not relax those gates after observing results. Layout/accounting, sentinel,
nonfinite, transaction, or F16-control failures are hard blockers regardless of
the distribution metrics. Defaults, speed claims and broader qualification remain
separate reviewed decisions.

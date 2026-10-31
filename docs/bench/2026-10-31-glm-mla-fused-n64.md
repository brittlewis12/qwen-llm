# GLM Fused MLA N64 Runtime Gate

## Result And Scope

The fixed absorbed-D512 fused attention kernel passes bounded M5 Ultra N64
collection, authored offline checking and independent native/arithmetic replay.
All **9,469,952 active outputs** pass their frozen numerical classes. Exact
seeded-PV, off-TF32 and tiny-F32 controls also pass. This qualifies these submitted
fixtures, not performance, model quality, universal arithmetic, N2048 admission
or a production default.

Experimental source is `a00af8c42283360eea973c5e8330cb99869bba3d` on
`probe/glm-mla-fused-512`, under `scripts/bench/glm-mla-fused-512/`.
Production code remains unchanged. The earlier
[current attention-only budget](2026-10-31-glm-current-attention-budget.md)
funds the experiment; its approximately 21.8 ms control is not a fused speedup.
The [separate mixed-operand gate](2026-10-31-glm-mla-mixed-operand-gate.md)
does not substitute for this new coupled-attention and seeded-PV evidence.

## Frozen Cases

Q/output are F32 `[512,64,64]`, C is one shared F16 `[512,64]` cache, and sinks
are F32 `[64]`. Position is zero, scale is 1/16 and visibility is inclusive causal.
The immutable 32-byte Case ABI is distinct from the old 16-byte API probe.
Each fused launch uses grid `[2,64,1]`, 256 threads and eight actual SIMD groups.

| Capture | Purpose |
| --- | --- |
| General | Finite tagged Q/C and finite NoSink=-1e30 |
| Positive sink | Sink +8 above visible score maxima |
| Tight precision | Off-TF32-grid Q and tiny F32 probabilities |
| Ordered cancellation | Declared four-part F32 association, not real-dot accuracy |
| Query tail | 33 active rows; rows 33..63 stay bitwise poisoned |
| Support | Four seeded PV destinations and seven scalar controls |

There are six commands, thirteen dispatches and six guarded raw frames of
18,483,232 bytes each. The support command contains one PV and seven scalar
dispatches, with immutable/disjoint argument and output subviews.

## Independent Arithmetic

The authored checker and a separate standard-library audit both pass. The
independent audit does not call the authored checker or its Reference class.
It checks actual input bytes, reconstructs scores using a fully verified Q-affine
decomposition and computes dense C-weighted F64 outputs. The authored oracle's
different rank-two-C factorization is limited to the verified fixed fixture.
Neither is a general model or hardware-reduction emulator.

| Numerical Class | Maximum Absolute Error | Maximum Error / Tolerance |
| --- | ---: | ---: |
| General | 1.6736970e-7 | 0.000836849 |
| Positive sink | 1.0201265e-7 | 0.000510063 |
| Query tail | 1.6223406e-7 | 0.000811170 |
| Tight precision | 4.0252092e-8 | 0.368005 |
| Ordered cancellation | 1.7882179e-8 | 0.155011 |

General/sink/tail use the unchanged absolute 2e-4 rule. Tight families use
`max(1e-8, 8 ULP32)`. Absolute-error and tolerance-fraction maxima may occur at
different coordinates. Cancellation agrees with the declared ordered score
**1/16**, not the exact-real dot score 1/8. At this deliberately ill-conditioned
input, deviation from the real-dot output reaches approximately 0.01558944;
that is not counted as general F64-dot accuracy.

All 16,384 physical PV records pass actual-coordinate exact-once coverage over
four destinations and eight SIMD groups. Each destination has eight exact
`0x3fc00800` seeded results and eight exact `0x33000000` tiny-P results.
All 1,792 scalar records pass separate multiply/add and the versioned exp rule;
normal-result exp error peaks at 1.158384 ULP, below four ULP.

## Pre-GPU Underflow Amendment

Before any kernel execution, independent review identifies an unnecessary
gradual-underflow requirement: the original mandatory exp(-100) check could
reject MSL-permitted F32 flush-to-zero behavior. Apple's
[MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 8.1 and 8.5, permits this behavior; precise variants do not remove that
permission. The N64-only v2 amendment is frozen before hardware results exist.
The original policy, shader, compiler flags and historical packet stay unchanged.

Eligibility uses the rounded F32 reference exponent. Zero/subnormal references
permit either signed zero or the unchanged nonzero four-ULP criterion. References
rounding to normal `0x00800000` still reject zero. Normal accuracy, exact
`2^-25` F32 bits, full-output tolerances, masked-P +0 and all safety checks remain
unchanged. `2^-25` is normal F32, not an F32 subnormal.

Actual exp(-100), finite NoSink and exp(-103) each produce 256 positive-zero
words. Raw bits and histograms are retained. These are compatible observations,
not causal proof of hardware FTZ or general IEEE compliance. The old
`eta=2^-149` analytic envelope remains conditional, not FTZ-inclusive; arbitrary
coupled operand/result flushing needs separate error propagation. This fixture's
non-NoSink full-attention exponentials have normal F32 results.

## Collection And Resources

The successful DEBUG1 process exits in one second and releases the normal
fail-fast self-lease naturally. An external 90-second timeout and 20-second
TERM-to-KILL grace bound the attempt. Offline math runs only after exit and
the parent's observed lease release. No model, server, installed binary, PATH,
SDK selection or service changes occur.

An earlier attempt refuses before the ledger, PSO/buffer creation or dispatch:
read-only staged input mode 0400 conflicts with the collector's required private
0600 mode. Correcting only those task-owned permissions and using a fresh report
produces the successful capture. Source, binary, policy and input bytes do not
change. The refusal is retained separately, not pooled or called a math failure.

- All 163,840 fused status records have zero errors and complete unique actual
  tid/SG/lane membership. The checked scratch ABI is an eligibility condition,
  not a general Metal guarantee.
- All 1,015,808 inactive output values retain poison; 393,216 guard bytes and
  50,737,344 immutable input/padding bytes match.
- All three actual PSOs are SIMD32/max256, with static LDS 30272/0/0 and dynamic
  zero within the 32768-byte device limit.
- Device-priced B is 18,546,688 bytes; buffer growth is 18,497,536 and final
  metadata/buffer/command growth is 19,316,736, below Metal quote 119,209,984.
- CPU quote 265,687,488 includes both libraries' copies/maps, checker workspace,
  managed-file-cache reserve and separate framework policy. These are not formal
  peaks; there is no backing credit or duplicated shared reserve.

The unchanged 128 MiB cap applies to owned raw backing, not total process demand.
Native owners retain buffers, assets, libraries, PSOs and context through checked
completion and error/unwind drains. Independent native replay accepts physical
views, aliases, prices, all raw hashes and terminal receipts. The supervisor's
process/lease observations are trusted assertions, not remote attestation.

Configured SDK 26.2/Metal 32023.864 builds the shader as MSL 4.0 without fast
math. The collector is direct opt1/noLTO against the pinned cached production
rlib, not a canonical application build. All 160 cached inputs remain unchanged;
native CPU/CLI controls and 61 Python CPU tests pass before collection.

## Frozen Evidence

| Evidence | SHA256 |
| --- | --- |
| Collector | `79613997bb3e820aff47a82594548fe8c241470efc305b94bcb28c7ce341c200` |
| Final input manifest | `8255cdffdd6a3343af12cefc138d94b2054f39eafa7d067d7e7e12c8fca5f1cf` |
| N64 v2 policy | `49869f9ee079032c4bf6ae8ad18872d22948f3cd6d49f0843e51f80c20296def` |
| Complete GPU report | `ad0cfbc70acb070595ac8d6a16b390b4ec2e5e7b9d68db1c86ccfd54cf5ad218` |
| Authored offline result | `b33ccaa7f0b04b98fd7b329d11ede3acc7429b4d4ed1f64038f0c58c97767fd0` |
| Independent native replay | `af1ed68fb474e9e97b84b0027069dc4165776ebb81d36266137f990b7f135426` |
| Independent arithmetic replay | `b7525fea4679fcfd29a23c6940335a02bdecf7c4258164682b1183424e16e127` |

The report binds both frozen libraries and all six raw frame hashes. Accepted
end-to-end results and zero internal proof errors do not independently expose
hidden operand maps or QK partials; direct new QK-stage capture remains unfinished.
No universal arithmetic, spill/occupancy or roofline claim follows.

Next preparation is a separately admitted N2048 correctness gate, then matched
CURRENT/candidate attention cost in the same balanced packet. Whole-MLA
continuation still requires both >=30% and >=8 ms saved per block, followed by
ordinary-model quality and release qualification. N64 acceptance alone authorizes
none of those GPU stages or production promotion.

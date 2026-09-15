# Bound The Remaining Packed Signatures Before More Model Runs

IQ4_NL down is repaired at `82a21f9d`, paired proof at `958adb2f`. Native03 still
aborts before prefix completion. This is not HC numerical/performance evidence.
Instead of one full-prefix attempt per kernel, inventory the actual artifact and
source selectors, then test only remaining reachable signatures in isolation.

CPU-only metadata inventory (no Metal or tensor-payload reads), PASS0.04s:

-94 IQ3_XXS gate/up tensors (47 layers), already repaired.
-2 IQ4_XS gate/up tensors, both layer2, not yet repaired.
-43 IQ4_NL down tensors, already repaired for M64xN32.
-5 Q8_0 down tensors at layers2,4,30,46,47, not yet repaired.

All banks have512 experts. Gate/up K2560/M640, down K640/M2560. Source selectors
in `qwen4exp_moe.rs` use exactly these four dtype/kernel paths. The IQ4_NL
M128xN16 option is restricted to512/527-token chunks; it is not selected for this
packet's2048/131 extents. Do not edit that unreachable variant speculatively.

## Frozen Additional Scope

Original empty-route512-expert tests for IQ4_XS gate/up and Q8 down, separate exact
processes with production lease, real wired check, API validation. Gate/up uses
M64xN16,16384 shared bytes, nb01 in256-element blocks; down uses M64xN32,8192
shared bytes, nb01 in bytes (34 bytes per32-element block). Preserve full current grids/args except
unread placeholder extents, zero counts, guarded outputs and immutable inputs.

Only reproduced failures earn the same isolated uint-builtin/local-ushort paired
variant,511 control,512 widened case, and populated bitwise comparisons. Use
two experts,33 tokens,70 outputs, permuted66 slots, counts33/9, both tile tails;
gate K256 and down K64, valid synthetic native quant blocks. Inputs for gate/up
are token-indexed via slot/top-k; down inputs are indexed directly by slot.
Capture original/widened full outputs before assertions. No timing sweep.

After the bounded signature set qualifies, retire temporary paired shaders and
resume the unchanged native HC packet once. This covers the known reachable
packed-MoE set, not a guarantee about every other prefix kernel. No broader
builtin widening, HC gate changes, validation bypass, or new weight downloads.
Read-only strategy review: `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.

Both original signatures reproduce the512-expert API assertion. Before any
populated Q8 test, source review corrected the initial empty Q8 probe's unused
nb01 from20 blocks to680 bytes. That original abort is retained as a signature
reproducer, not exact-argument evidence; rerun the original512 case with correct
args before accepting the paired packet. Zero counts never accessed weights.

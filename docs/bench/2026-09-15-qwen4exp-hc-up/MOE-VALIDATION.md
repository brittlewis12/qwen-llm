# Native HC Hold: Existing Packed MoE Validation Boundary

Outcome: paired reproducer preserved at `60e644ae`; isolated width repair at
`95a71437`. Narrow511 passes, narrow512 reproduces, wide512 passes, populated
outputs match bitwise. Temporary shader/trace removed; retained product
regressions pass. Native attempt02 still asserts before prefix completion; no
remaining kernel is identified. Park per stop rule. See `RESULT.md` for current
authority and next actions; the protocol below is retained as written before runs.

Native attempt01 aborts under Metal API validation during the unchanged packed
prefix, before HC candidate execution or native timing. A temporary test-only
census trace localizes the abort to
`kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16`, grid128x10x512,
threadgroup128x1x1, dynamic threadgroup memory16384 bytes. Assertion:
`component 0: 65536 must be <= 65535 for tiitg [[thread_index_in_threadgroup]]`.
The source local index should be0..127. The relationship of65536 to grid sizes
is suggestive, not an established explanation of validator/compiler internals.

Logs: `target/profiles/2026-09-15-qwen4exp-native-hc-up-01.log` and
`2026-09-15-qwen4exp-native-hc-validation-trace.log`. Neither supplies native
HC numerical or performance authority. LLDB launch timed out without reaching
the test; subsequent process check found no debugger/test process remaining.

## Bounded Reproducer Protocol

Before any further model run, three separate exact ignored tests, each with
production lease and wired gate plus Metal API validation: unchanged narrow
index at depth511 and512, then one widened builtin at depth512. All expert
counts zero, preserving the full failing grid, bindings, and shared memory.
The uniform return precedes weight/input/ID accesses. Valid small placeholders
suffice for those unread buffers; guarded outputs and all inputs stay unchanged.
Expected-abort cases must run in separate processes, never a whole ignored suite.

One additional small nonempty test compares both signatures bitwise: two expert
banks of valid synthetic IQ3 blocks, hidden256/FFN64,17tokens, topk2, counts17/9,
permuted output slots and N16 tail, poisoned unused slots and offset guards.

The temporary paired shader instantiations differ only in annotated builtin
width (`ushort` versus `uint`) and immediately narrow to the same local ushort
used by the original body. No arithmetic/grid change. This diagnostic variant
must be retired after the outcome; do not promote a width change solely on an
empty-route check. If the assertion just moves elsewhere, park native HC and
retain the evidence rather than preemptively editing every builtin.

No validation bypass, timing sweep, or model replay before this packet completes.
Read-only preflight: `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.

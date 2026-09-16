# Exact N1024 Packed Router Qualification

Baselineb57f6be2. A new policy-coverage question, not a closed kernel/observer retry:
the unchanged strict-order E8P32 router is default on only at512,527,2048 packed
rows. A natural standalone1024-token prompt uses generic F32 routing at all48
layers despite the same released geometry. Shader query bounds are safe at1024;
shape safety alone does not establish equality or performance.

Fresh cx Luna01a0ac41-e142-7f72-a725-b980fbf06e2a supports this finite exact-width
qualification. Correct its initial planner example:3072 selected tokens produce
2048+3+1021, not2048+1024. Use one ordinary1024-token prefix instead of padding a
synthetic3075-token prompt. Do not widen an interval, change chunk planning, alter
the shader, add a flag, or promote before all gates pass.

## Frozen Packet

All Metal work requires production lease, real wired-memory gate, API validation,
serial tests; no server restart, special quant or remote push.

1. CPU policy/planner checks: exact1024 added to existing M4 Max/H2560/E512/F32
   scope only; current rollback stays. Direct1024 planner is one packed command.
2. One model-free N1024 differential using the established deterministic router
   inputs: full logits, IDs/weights, shared scale, route counts and slots bitwise;
   generic/strict grids and remaining route kernels exactly witnessed.
3. One loaded native model, current guarded/HC defaults, incumbent QSA, no split
   scratch. SHA-pinned natural-SSH fixture874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97:
   first1024 prompt tokens, next4 teacher-forced continuations. Decode/re-encode
   roundtrip before execution. No new semantic corpus claim.
4. Two fixed correctness/warm passes A then B, restored from one empty checkpoint.
   A disables existing packed strict router; B enables it. Capture full vocabulary,
   hyper,121persistent tensors, QSA lengths and PLE history at prefill and EACH
   continuation step. All bitwise/exact. Correctness census permits only48
   generic-to-strict substitutions; singleton continuation routing is unchanged.
5. One measured prefill ABBA, restoring the same empty checkpoint before each.
   No census, timestamp sampling, adaptive warmup or retry in timing arms. Preserve
   each prefill endpoint/state and ordinary timing before gates, and require replay
   equality. One1024-row command, same allocation plan and all other policies.

Performance KEEP requires>=1% mean aggregate prefill GPU saving, positive saving
in BOTH balanced pairs, control spread<=5% on GPU AND executor wall, and candidate
mean executor wall<=baseline mean (zero regression tolerance; mean-only wall rule).
All GPU samples must be valid. Any failure/HOLD/INCONCLUSIVE ends this attempt;
remove1024 from production scope rather than leave a candidate switch.

After a KEEP only: ordinary CLI strict UNSET versus rollback0 on the roundtripped
1024-token raw prompt, same deterministic output; delivery check, not another
performance bracket. Existingknown-answer full prompt need not be synthesized to
hit this width. EarlierN527 gain supports plausibility, not the N1024 decision.

## Pre-Execution Harness Corrections

CPU scope/planner and N1024 component differential pass. First native invocation
stops BEFORE prefill/forward at the<1GiB snapshot-budget assertion: retaining all
five baseline states is too large (log2026-09-16-router-n1024-native.log,0.13s).
This is no numerical or timing observation. Preserve the failed resource check;
do not raise its limit. Version the harness to retain only the prefill baseline
state and compare continuation states against persisted per-tensor baseline bytes,
one tensor at a time. Full per-step bitwise coverage and timing gates are unchanged;
the conservative simultaneous-state bound becomes four snapshots plus logit rows.
An earlier Option<Vec<census>> inference compile error was fixed before execution.

# Token K32 GateUp: Exact Layout, Pack-Inclusive Negative

The token K32-major layout passes its GPU bit proof but loses the fixed
actual-route GateUp comparison. Including a fresh pack on every candidate
call, GPU latency increases by 0.6773% / 0.4749% and inclusive host latency
increases by 0.7461% / 2.0281%. Both blocks miss the predeclared 10% GPU
reduction and no-host-regression gate. Close this implementation without
another timing run or a production-selector change.

## Source And Workload

- M5 Ultra, 256 GiB, native TensorOps tier; release build and SDK 27.0.
- Completed research source `1378c2ceeae0ee88326ab62ec7264b2f9f1e63f2`, tree
  `db2e8efdc5a036fa586aa95384213ac67e261490`.
- Frozen executable SHA-256:
  `e015377411aa9eb000b12757312c5b4d03f7b52746b51ca091aacf6b9041a272`.
  Build, freeze, all 397 classified CPU tests and the exact CPU-only header
  fixture pass without a GPU lease. Discovery finds 2,967 library tests.
- Existing DeepSeek V4.1 Flash GSQ-RCO IQ2_XS mixed-quant artifact with its
  IQ4_NL Engram companion. Original layer-8 IQ2_XXS Gate/Up banks, hidden
  5,120, intermediate 2,304, 384 experts, top-6 and actual clamp 10.
- Fresh completed production FullForward capture, start 0 and first 1,024
  frozen prompt IDs, `Ordinary` arithmetic. Runtime scratch is 1,024 while
  the stock selected planner width stays 4,096. Actual population is 6,144
  unique routed slots, including 121 zero-count experts.

Selected ID SHA-256 over little-endian u32 words is
`628c172eb1fc489e4393393e3eec1c039b4a475fcf89ef281f7c37098b9f97df`.
These are the retained inputs identified in
`docs/bench/2026-10-28-v41-vector4-ffn-result.md`, not a new public corpus.

This is a captured-operand GateUp experiment, not a whole-FFN replay: saved
normalized half input, counts and slot IDs are intentionally its operands.
It does not claim to rerun routing during the timed calls. Source includes
maintained history through `956d0ed0`; later decode and serving improvements
are separate, not part of this executable.

## Layout And Bit Proof

The raw permutation is
`packed[k_tile][token][lane] = original[token][32*k_tile + lane]`.
One 10 MiB F16 scratch holds `[32,1024,160]`. Both Gate and Up use it with the
original token-to-slot relation, IQ2_XXS decode, ascending 160 K32 accumulations,
F32 accumulator, clamp/SILU expressions and slot-major output addresses.

The same new executable first runs the debug-layer correctness fixture, with
no model variables. All seven raw permutations and six fixed matrix cohorts
pass: balanced counts, 0/1/31/32/33 boundaries, six-expert skew, invalid counts,
invalid IDs and a restricted count range. There are 22 priced allocations;
cumulative priced storage plus the declared CPU reserve stays within the
4 GiB fixture cap, with ordinary admission and the 6 GiB device cap retained.

Proof compares raw Gate, final GateUp and fresh-pack pair bits, all live-word
finiteness, inactive poison, offsets, guards, unchanged input and controls.
Distinct qNaN poisons expose missing Gate stores; a separate finite-`f32::MAX`
Up witness exposes common omitted in-place stores. Raw all-bit permutation
cases, including NaNs, remain separate from finite matrix operands.
This proof is correctness evidence, not performance timing.

## Real-Route Comparison

A executes unchanged narrow Gate0 then UpClamped2. B executes a FRESH pack,
Gate0 and UpClamped2 on every call, sharing that one newly populated pack
between the consumers. Both arms use the same physical working output and
original borrowed banks. No expert replication, output scatter, vector4 MID,
paired Down or compact-path composition occurs.

An untimed incumbent raw-Gate oracle occupies already charged Replay13;
captured production GateUp is the immutable final oracle. Every timed output
matches the full final reference. Separate raw-Gate/final proofs and Up-store
witnesses pass. Full control bindings, captured references, untouched Replay
regions, guards/slack, source, artifact and input checks pass, and borrowed-bank
hashes match before and after execution. All 6,144 logical output slots are
active; undefined bucket tails are checked for immutability, not invented as
valid slots or required to contain A5.

Warm A/B census requires exactly two versus three ordered dispatches, including
B's pack. Eight uncensused primary calls run `A B B A`, `A B B A`. All 32
commands complete serially: 10 computes, seven correctness proofs and 15
targeted resets. Reset and full verification occur outside compute timers but
inside the native budgets. Primary commands have no counters, census or debug
layer. GPU duration includes B's pack; inclusive host duration covers command
creation through checked completion. The fixed 5 us cross-API containment
allowance never relaxes comparative gates or seriality.

| Block | A GPU Mean, ms | B GPU Mean, ms | GPU Reduction | A Host Mean, ms | B Host Mean, ms | Host Regression |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.671604 | 13.764208 | -0.677347% | 194.205229 | 195.654146 | 0.746075% |
| 2 | 13.708813 | 13.773917 | -0.474907% | 196.477125 | 200.461855 | 2.028088% |

Every block independently requires `B_GPU <= 0.90*A_GPU` and `B_host <= A_host`.
Both fail both conditions. Native and independent public reduction agree on
`continuation_passed=false`, `performance_qualified=false`,
`decision=close_implementation`. Native execution completes in 84.607010125 s;
all 40/45/45/20 s phase caps and the 150 s native bound pass.

## Preserved Incomplete Attempt And Wrapper Failure

The first combined source, `23cb3013`, passes its same-binary raw GPU proof,
but its native packet stops during the seventh of eight primary calls: full
verification exceeds the original 25 s primary-phase budget. It has no eighth
call, completed postflight or reduction. No partial block is selected, pooled
or interpreted as a performance result.

The bounded followup changes only K32 verification headroom: phase caps become
40/45/45/20 s, native total 150 s, native-only parent metadata 170 s. The
prospective composite uses a 250 s pre-write decision cutoff and requires
150 s remaining before native entry; proof cancellation/decision stays 60/80 s.
The new policy is fixed before followup hardware execution. All 97 kernel
files and five protected source files, plus the route compute/full-verification
body, remain byte-identical. Workload, checks, timing windows, sample order,
resource limits and the 10% gate do not change. A new build, CPU/header
qualification and mandatory same-new-binary proof precede the followup.

The followup native child exits successfully with all checks and postflight
complete, but the original outer helper rejects valid empty-capture libtest
success framing: this native fixture writes its JSON report without printing a
captured stdout section. A separate copied framing adapter validates that exact
success shape and the complete packet offline; it does not relax the proof's
required cohort output, alter data or rerun GPU work. All 45 offline controls
pass, including missing/wrong/duplicate framing rejection; the original proof
parser and SHA-bound public economic reader stay unchanged.

Original outer acceptance remains failed and absent. Its final 250 s pre-write
verification decision is not established. Final durable receipt I/O is excluded
from that declared decision cutoff, not claimed completed by it. A separate
later unleased CPU postflight matches the pre-GPU reviewed binding; it is not
relabeled as the missing original outer postflight. Both GPU children exit
without signals, with the canonical descriptor retained through drain and
signal handlers restored after context teardown.

Completed native packet SHA-256:
`6af2e317e9b7d32af5659641b8807bb40596befa1a54a7b3c71a8bd8445a35c0`.
The original incomplete packet, completed native negative, raw samples,
failures, attempt latches and isolated CPU analysis remain retained separately.

## Limits

The pack adds one dispatch and 20 MiB of logical read/write traffic each B
call; it does not remove the consumers' logical requests or establish measured
DRAM traffic. The capture/Replay/pack ledger is 1,452,500,480 bytes before
device slack and pipeline envelopes, with the original 12 GiB extra cap and
normal resident-model Metal-plus-CPU admission. No giant bank/oracle clone or
new quant artifact is introduced.

The result does not identify cache, TLB, register or residency behavior, nor
qualify another shape, Full/CED parity, logits, likelihood, sustained throughput
or a production default. In particular, exact layout equivalence is not a
speedup. Do not turn this miss into a window/layout sweep. The frozen protocols
are `docs/bench/2026-10-28-v41-k32-real-route-protocol.md` and
`docs/bench/2026-10-28-v41-k32-route-budget-followup.md` at source `1378c2ce`.

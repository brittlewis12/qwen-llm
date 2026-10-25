# V4.1 Execution Leverage: Native Controls And Driver Attribution

## Scope

Apple M5 Ultra, 256 GiB physical memory, operator-selected 240 GiB Metal working
set. The Maya-S six-shard artifact uses 165.950 GiB of shared GPU views, no
private weight shadow, and disk-backed Engram tables. No smaller artifact,
percentage reserve, purge, explicit residency or pinning is used.

The native controls freeze qualified score32 source
`efea49080046acc3a19da06dc1b38dd498560771`, before main's 4096-row Full chunk
and wide64 gate/up landings. Frozen benchmark SHA256:
`371c17ef59be867ec8f7622bd5bb7373fb62651e38f321de9a609231065c9263`.
These are controls for later integration, not measurements of decoder batching.

## Native Controls

Each length uses one resident import, native v2 observation, warmup AB, pilot
ABBA and retained BAAB/ABBA blocks, giving four retained calls per policy and 14
chronological calls total. Timing launch removes known validation/injection/
capture variables only from the child environment. Effective external injection
is not independently attested. All raw samples remain; dispersion is descriptive,
not an arbitrary rejection rule. The observer has a bounded deadline, stops on
observed critical pressure and owns only its launched child.

Retained native seconds, in chronological order within each policy:

| Prompt | Full | CED | Median Full / CED |
| --- | --- | --- | --- |
| 2048 | 3.519960, 2.237465, 8.099559, 2.858367 | 3.090901, 3.375681, 3.783339, 4.098704 | 3.189164 / 3.579510 |
| 8192 | 13.888737, 9.934380, 11.018717, 11.014149 | 7.349264, 7.675195, 8.172907, 8.316854 | 11.016433 / 7.924051 |

The 2K cohort does not establish a win. At 8K, all six CED pilot/main calls
(7.349-9.507 s) are faster than all six Full calls (9.934-13.889 s). The retained
median CED latency is 28.1% lower in this cohort. This does not establish a
universal stable latency, default policy promotion or the incremental benefit of
scratch reuse, fusion, current 4K geometry, wide64 or batching.

All 14 endpoint heads per length match the archived same-input reference:

```text
2048: 178706f5248735a6e8443987c3e7cb513afa481d499b24db70e8778e0a2ba97d
8192: a17fa5d7e5a1b67a3648cd19c4223b429667ea10fdbce361a0687da2b1e67090
```

CED has no fallback, completes 21/27 commands and remains within its accepted
sampled allocation bound. All sampled host pressures are normal. Conservative
CPU availability reaches 46.41/45.79 GiB; system wired plus compressor reaches
177.88/178.25 GiB. These are sparse whole-host samples, not continuous process
peaks, an absence-of-swap proof or a reservation of remaining capacity.

## Driver Attribution

Diagnostic source `deea1b323e588616ef2bc661c6489c16ad0dd422` adds Metal's
completed command-buffer `kernelStartTime`/`kernelEndTime` interval and includes
tensor-library cold pipeline lookups in the existing miss accounting. Frozen
benchmark SHA256:
`669488eba2f44510b5d22e3521c0d9a2ca33db1f5404fa83e09daab488d2fd78`.
The actual M5 cold/hit/missing-function pipeline regression also passes.

Only positive finite ordered timestamp pairs are accepted; missing/invalid
pairs remain explicit. This driver interval can include scheduling/waiting. It
is neither shader-active duration nor CPU busy time. GPU envelopes and inclusive
CPU encode/wait/preparation spans are distinct; overlapping spans cannot be added
into a native-time decomposition. No Rust `Instant` is subtracted from a Metal
clock.

The separate validation-on, instrumented 14-case 2K run observes:

- Warm Full GPU envelopes about 1.944 s; driver intervals 0.030-1.139 s.
- Warm CED GPU envelopes 1.616-1.684 s; summed driver intervals 0.812-1.571 s.
- Warm process read bytes/page-ins and ordinary plus tensor pipeline misses zero.
- Cold Full native 24.875 s, GPU 1.943 s, driver 15.356 s, inclusive wait
  17.301 s and Engram preparation 7.369 s; 132.795 GiB process reads, 134437
  page-ins and 42 pipeline misses. Cold reads and compilation are not silently
  attributed to the steady-state scheduling hypothesis.

CED roles are coalesced across all dimension variants, not overwritten by a
last-writer dictionary: one encoder command, three standalone decoder-row
commands and 17 seed/output commands. Standalone decoder-row driver intervals
sum to about 0.452-0.526 s, while their GPU envelopes sum to about 0.16 s.
Seed/output GPU sums are about 0.48-0.55 s; driver sums vary about 0.018-0.856 s.
Encoder driver intervals are normally about 0.030 s, with one about 1.093 s.
Inclusive CPU preparation/allocation spans overlap and are not another additive
latency budget.

These facts motivate bounded cross-layer retirement. They do not prove that
large backing resources cause driver delay or that the delay scales with
command count. Full binds the same weights and itself has variable scheduling
intervals. Explicit whole-model residency remains closed.

The diagnostic profile operation remains 136 bytes with 4096-entry capacity.
The measured source's diagnostic charge is 2,598,377 bytes within the existing
16 MiB reserve; the 64 MiB caller bound is unchanged. Adding attribution does
not justify relaxing memory admission or treating sampled allocation as a
formal transient driver bound.

## Wide Activation Correction

Main `e402dcb1` selected F16-only wide kernels before validating activation dtype.
The public F32 path could therefore reinterpret F32 bits as half. Direct probes
also admitted non-half storage. Fix `12d098eb` validates the public activation
type before selection, confines wide execution to F16 and rejects unsupported
direct-probe inputs before dispatch. No shader, threshold, F16 arithmetic or
rollback change is made.

Actual M5 validation-on red/green source uses identical nonzero K32/M64,
4096-token fixtures with sparse active slots, an empty expert, offset guards and
dispatch census. The genuine unfixed source fails two of three tests: maximum
F32 error 0.099202536 and invalid types dispatching/writing outputs. Fixed source
passes all three: F32 remains narrow and F16 remains wide, both with zero
reference error; invalid types encode nothing and leave counts/IDs/output
unchanged. Complete explicit prices are at most 8,192,544 bytes, admitted before
fixture allocation and below the 128 MiB fixture cap.

Maintained tools rebuild at `12d098eb` with clean matching build/runtime identity;
all three command help checks pass. This does not certify future integration
heads or replace full-model qualification of the concurrent geometry changes.

## Current 4K Control And Two-Layer Candidate

Current integrated control `be874b3f39ad44c8cf7cfb9c11ddad988e4fca2e`
includes main's selected 4096-row Full/encoder path, wide64 F16 gate/up,
the dtype correction and attribution. Decoder rows/seeds remain bounded at
2048. A separate M5 fixture initially stops on an incorrect census expectation:
the narrow gate intentionally uses a Down-role pipeline name. The test-only
repair asserts the complete ordered typed gate/up/down sequence, not a weaker
nonzero count. All six fresh M5 tensor/dtype/geometry tests then pass.

Two-layer source `9874c68b1602f8ff6058179a982c54775c22b638` adds bounded
transactional decoder groups. It passes 33 focused CPU checks and 18 self-leased
M5 GPU fixtures under validation, including automatic oversized-group incumbent
selection, strict arithmetic/control/state parity, sticky status, GPU canaries,
fault precedence, cancellation/unwind and release while State remains poisoned.
Independent read-only source review finds no concrete P1/P2; that review does
not replace execution or measure driver heaps.

Both sources pass fresh full-model current Full/CED pairs at all five lengths.
Each length compares all 257 complete vocabulary vectors and all 56 causal
regions at every position, with one resident import and at most one KV state:

| Prompt Rows | Control Commands | Two-Layer Commands |
| ---: | ---: | ---: |
| 128 | 21 | 11 |
| 2048 | 21 | 11 |
| 2049 | 24 | 11 |
| 4097 | 45 | 12 |
| 8192 | 25 | 12 |

Every candidate head hash and complete state record matches its control; all
per-layer work counters are unchanged, completion is checked, fallback is null
and sampled allocations stay within accepted bounds. Control 128/2K/8K also
preserves the measured old-reference bridge. The 2049/4097 checks extend current
Full/CED transition coverage, not an unmeasured pre-4K reference. In particular,
2049 is packed under selected 4K geometry; 4097 has a genuine scalar tail.

The two-layer command allowance remains explicitly unmeasured: 1 MiB per actual
physical unit per CPU/Metal domain, capped at 16 MiB/domain beyond the incumbent
allowance. This is not a formal driver/CPU transient bound. Larger-window
over-budget requests retain bounded incumbent CED commands before batch pricing
or encoding, rather than rejecting context or restarting Full after mutation.

Fresh production-timing-requested native cohorts retain four calls per policy.
Seconds below are rounded to six decimals; raw nanoseconds and all chronological
samples remain in the private reports:

| Source / Prompt | Full | CED | Median Full / CED |
| --- | --- | --- | --- |
| Control / 2048 | 12.431765, 2.237038, 3.006496, 2.700650 | 3.042474, 4.909086, 3.459496, 8.904997 | 2.853573 / 4.184291 |
| Control / 8192 | 18.883852, 9.500112, 29.603241, 10.561673 | 20.833771, 18.744491, 7.456756, 10.233436 | 14.722762 / 14.488964 |
| Two Layers / 2048 | 3.132919, 2.251518, 19.672501, 2.692496 | 3.372027, 19.311431, 2.743518, 8.568895 | 2.912708 / 5.970461 |
| Two Layers / 8192 | 23.951903, 9.526876, 11.380274, 12.466781 | 18.749036, 13.450416, 13.541780, 33.457407 | 11.923527 / 16.145408 |

Neither new cohort establishes a stable native win. The earlier frozen-source
clean 8K separation is not reproduced. The two-layer 2K Full pilot/main spread
exceeds 10x and CED exceeds 7x; the 8K spreads are about 2.5x. A noisy cross-build
median change is not causal evidence that geometry or batching causes the
regression, but command-count reduction alone clearly cannot justify promotion.

All 14 endpoint heads per cohort match the archived synthetic-input reference;
actual CED command counts and sampled allocation bounds pass independent checks.
Across these four native cohorts, 387 sampled health observations are normal,
CPU availability reaches 35.23 GiB and system wired plus compressor reaches
180.84 GiB. These remain sparse host observations, not a swap or peak-footprint
proof. Native synthetic routes are not a measured text-route distribution.

Four-layer source `4672dc8172aa9e7047bf204f44302d846b262415` changes only
the production grouping default; all other production logic and the 16-unit
budget remain. It passes 49 CPU and four bounded M4 GPU fixtures. Default 4K
encoder plans complete 6/6/6/7/7 commands at the five lengths, with at most
15 MiB/domain additional allowance. M5/full-model/native qualification is still
pending. Keep all frozen sources unchanged while comparing it and localizing
remaining stalls.

## Evidence And Next Decision

Private execution archive labels, with no machine-specific path required:

```text
qualified-base-v26-native-summary.json
maya-base-n2048-v26-r1.{json,stderr,events.jsonl,controller.json}
maya-base-n8192-v26-r1.{json,stderr,events.jsonl,controller.json}
maya-attribution-n2048-v26-r1.{json,stderr,events.jsonl,controller.json}
attribution-n2048-v26-role-summary-v2.json
attribution-m5-pso-v26.*
wide-type-fix/m5-{red,green}/
control-{types,wide,ffn,edges}-v27.*
maya-control-n{128,2048,2049,4097,8192}-v27-r1.*
maya-control-native-n{2048,8192}-v27-r1.*
batch2-{freshness,plan,preflight,groups,types,wide,ffn,prefix,ledgers,nonfinite,edges}-v27.*
maya-batch2-n{128,2048,2049,4097,8192}-v27-r1.*
maya-batch2-native-n{2048,8192}-v27-r1.*
```

The v2 role summary supersedes only the erroneous last-writer `role_totals` in
the first private diagnostic summary; original case totals and raw evidence
remain unchanged. Full-model input SHA256 remains
`aaf5840de38220f6ed4579ffcf83dc828a28e7d71f8e1f2517077ee77020eca2`.

Next: qualify the four-layer candidate and use driver/host attribution to locate
the remaining stalls. Actual text-route population capture is a separate
diagnostic follow-on; synthetic count bands cannot establish its headroom.
Do not promote batching from command counts, old controls or instrumented timing.
The independent
[matched-F16 register family closes negative](2026-10-24-v41-register-operands-32x32-f16.md),
with reproducible standalone source and preserved raw evidence, not a product
kernel landing.

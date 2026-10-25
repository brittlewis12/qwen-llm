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
```

The v2 role summary supersedes only the erroneous last-writer `role_totals` in
the first private diagnostic summary; original case totals and raw evidence
remain unchanged. Full-model input SHA256 remains
`aaf5840de38220f6ed4579ffcf83dc828a28e7d71f8e1f2517077ee77020eca2`.

Next: qualify current 4K/wide integration, then the joined transactional batch
candidate with exact heads, causal regions, fault/lifetime/status/freshness
checks and native controls on the same source/configuration. Do not promote a
batching speedup from old controls or instrumented timing. The independent
[matched-F16 register family closes negative](2026-10-24-v41-register-operands-32x32-f16.md),
with reproducible standalone source and preserved raw evidence, not a product
kernel landing.

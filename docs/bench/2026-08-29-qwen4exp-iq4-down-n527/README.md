# Flash-Next Packed IQ4_NL Retile At N=527

Decision: **KEEP** the existing M128xN16 routed-down kernel at exact N=527.
The qualified token set is now `{512, 527}`; every other width keeps the
M64xN32 kernel.

## Route Screen

A temporary test-only capture observed only the final N=527 command in the
frozen `2048+3+527` known-answer prefill. It ran one release-mode prefill from
the internal SSD, with no generation, model hashing, broad suite, replay arm,
or profiling pass. The capture and its test were removed after acquisition.

Across the 43 released IQ4_NL-down layers, the command has `226,610` routes,
`S16=23,864`, `S32=18,253`, and
`R=S16/(2*S32)=0.653700762`. Individual layer ratios span
`0.626068376..0.691891892`, so every layer and the aggregate clear the existing
`0.75` retile entry gate. The complete three-command prefill used
`5,054.111583 ms` of GPU time; that value is descriptive, not an A/B endpoint.

## Component Falsifier

The model-free release falsifier used production `H=2560`, `R=640`, `E=512`,
N=527, and top-k 10 geometry. Each arm encoded 43 routed-down dispatches in one
command after a one-dispatch warmup. The `R~0.70` control is conservative
relative to the captured suffix; `R=0.75` exercises the admission boundary.

| Routes | B1 M64xN32 (ms) | C1 M128xN16 (ms) | C2 M128xN16 (ms) | B2 M64xN32 (ms) | Mean B -> C | Saving |
|:--|--:|--:|--:|--:|--:|--:|
| `R=0.699728261` | 153.065167 | 127.094625 | 127.079250 | 152.651208 | 152.858188 -> 127.086938 | 16.86% |
| `R=0.750000000` | 152.612166 | 133.931250 | 133.793708 | 152.572500 | 152.592333 -> 133.862479 | 12.27% |

Both balanced pairs are positive in both controls. The conservative packet
saves `25.771250 ms` across 43 dispatches and clears the preregistered 10% leaf
gate. The boundary control also remains positive. The complete falsifier ran in
1.25 seconds; its temporary timing body was removed after recording these rows.

No whole-model timing packet followed. The N=512 promotion predicts only about
25-30 ms against the selected workload's roughly 5.1-second aggregate prefill,
while the prior whole-command bracket varied by 33-50 ms within each arm. Such
a packet would not resolve this component effect.

## Correctness And Scope

- The retained differential now covers exact N=527, top-k 10, all ten expert
  counts at 527, the seventeenth M64xN32 column, and the thirty-third M128xN16
  column. Candidate and incumbent outputs match as raw F32 bits; inputs remain
  unchanged, every output is written, and guards remain intact.
- Scope tests admit only exact N in `{512, 527}` on Apple M4 Max with IQ4_NL,
  `H=2560`, `R=640`, `E=512`, and top-k 10. Preflight follows the same scope.
- Roll back both qualified widths with
  `QWEN4EXP_MOE_IQ4_DOWN_M128_N16=0`.

Adversarial leverage and falsifier design:
`01a0500f-6313-7563-b258-b1664fc1320b`.

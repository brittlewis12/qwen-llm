# Flash-Next Paired IQ4_NL Decode KILL

Decision: **KILL** the retile-private paired decoder after the first candidate
row. The source-screened rewrite reduced the representative routed-down leaf,
but missed both the preregistered leaf ceiling and whole-command gate.

## Candidate

The promoted M128xN16 kernel called the same IQ4_NL helper twice, once for each
nibble plane. Optimized AIR retained one shared scale load but emitted two
independent 16-iteration byte-load loops. The candidate replaced those calls
with one private helper that loaded eight aligned `ushort` words and emitted
both nibble planes from each packed word.

The optimized candidate AIR retained one scale load and the same F32 table
multiply followed by half conversion. Its four-row loop executes two 16-bit
loads per row, reducing IR-visible payload load bytes from 32 to 16 per block.
This cleared the compiler screen. The Metal-only source diff retained the same
kernel entry point, threadgroup allocation, MMA body, and output path.

## Correctness

- The focused M128xN16 differential reported raw-F32-bit equality across the
  established matrix before timing. Its raw transcript was not preserved, so
  the packet does not use that run as durable correctness authority.
- Both CLI arms completed the built-in first/warm/profile deterministic replay
  within the arm and produced the same generated-output SHA-256. This is an
  equivalence check, not an external semantic oracle or a cross-arm logits row.
- Both 128-sample observers were accepted with raw timestamp coverage 1.0.

## Protocol

- Base commit: `be2d79a3a1dd8699e88e241139c3988e2f5668ac` plus candidate
  diff SHA-256
  `108463fef41458810632a8aaa21860a8659986b576ae76cf10327e03916ca53a`.
- The same base commit preregistered the B1 and command ceilings in
  `docs/PERF-ROADMAP.md` before candidate code. That roadmap blob is
  `10a168740e439405dca3235635961e2ecf7f7d3a`.
- Device: Apple M4 Max. Model: internal-SSD UD-Q3_K_XL from
  `unsloth/Qwen3.8-Flash-Next-GGUF` revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
- Workload: the tracked natural 512-token prompt, no special tokens, one
  generated token, exact 512-forward capacity, one packed command, and no
  scalar tail.
- Order: separate-process A1 baseline then B1 candidate with five seconds
  between processes. Both explicitly selected the promoted M128xN16 path.
- Futility rule: stop immediately when B1 routed-down time exceeds
  `2.520169 ms/layer`. A survivor also had to beat `957.463279 ms` warm and
  `958.767129 ms` profiled command GPU.

## Results

| Arm | Down leaf (ms/layer) | Warm GPU (ms) | Profiled GPU (ms) |
|:---|---:|---:|---:|
| A1 promoted decoder | 2.800708 | 969.474125 | 970.360875 |
| B1 paired decoder | 2.686791 | 962.015000 | 963.504167 |

The candidate saves `0.113917 ms/layer`, or 4.07%, but remains
`0.166622 ms/layer` above the fixed leaf ceiling. Warm/profiled command GPU
saves only 0.77%/0.71% and misses the fixed command ceilings by
`4.551721/4.737038 ms`. B2 and A2 cannot repair the worst-case B1 leaf gate and
did not run, so these rows are directional rather than a completed effect
estimate.

The helper was removed. A clean rebuild reproduces the promoted metallib and
MoE AIR SHA-256 values
`d65856754594b3a0b9ee708e3aa0e4a2f6101f5c85ec9c30aa7617cf58a7dace` and
`79c594e12bf0220b6531dd785c6dfa52d8451b7b4439aaf765220e2d76e98b81`.
`cleanup.json` records those hashes, a clean `kernels/moe.metal`, and absence of
the temporary baseline executable.
Raw logs and compiler artifacts remain under
`target/profiles/qwen4exp-iq4-paired-decode-screen-20260829/`.

Adversarial evidence review: `01a04f52-ae71-7740-9714-01a6fc467bcd` (PASS).

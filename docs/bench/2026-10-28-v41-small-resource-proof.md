# Two-SIMD Small-Resource GateUp: Target Exactness Passes

The dedicated 16-slot x 32-output x K32 IQ2_XXS prototype passes its standalone
M5 GPU bit proof. Actual pipeline facts validate the intended two-SIMD,
64-thread, 3 KiB dynamic-shared path. This clears a correctness gate only:
no operator economics, model workload or production-selector change is qualified.

## Frozen Source

- M5 Ultra, native TensorOps tier, SDK 27.0 and release build.
- Research source `8ebb8c6303887795d46399e21404d3ba0ec75dd8`, tree
  `df8859fe2c99232ad3ac7ef2c24445b5c21a2ca3`.
- Frozen library executable SHA-256:
  `b388f94af331b2d69528c06ddfabc388b51ecc85c78564f09944275e405314dd`.
- All 403 classified CPU tests pass, including six new rejection/partition/
  mapping/census/closure controls and both actual cooperative-signal subprocess
  tests. Discovery finds 2,979 library tests. Build, offline Metal compilation,
  freeze and CPU qualification use no GPU lease.

The original full Rust build fails on a temporary-array lifetime in the embedded
source inventory. A one-function constant-storage fix precedes the accepted
build; the failed build and its evidence stay preserved. All three compiled
metallibs are byte-identical across that repair. It changes no kernel, proof
check, layout, resource cap or numerical expression.

## Candidate And Actual Resource Facts

Counts 1..16 use the dedicated kernel. Counts 17..1024 use the unchanged
incumbent narrow kernels. The test-only pair emits SmallGate, LargeGate,
SmallUp, LargeUp, with all four PSOs and resource facts validated before writes.
The count classes are disjoint; zero/invalid counts and invalid IDs retain the
incumbent uniform-return and no-store rules. There is no K32 packing or output
scatter, and production selection stays unchanged.

| Class | Slot x Output x K Tile | Launch Threads | Actual SIMD Width | PSO Max Threads | Static Shared, bytes | Dynamic Shared, bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Small | 16 x 32 x 32 | 64 | 32 | 64 | 0 | 3072 |
| Incumbent Large | 32 x 64 x 32 | 128 | 32 | 1024 | 0 | 8192 |

These facts hold for both Gate and Up. A maximum is not the launch size.
Small grid is `[1,72,384]`; large grid is `[32,36,384]`. The smaller descriptor
and execution scope do not establish register usage, occupancy or speed.

Both paths retain original token-major F16 input, expert-major slot IDs with
stride 1024, `b_div=6`, slot-major F32 output and in-place Up. Canonical F32
IQ2_XXS decode narrows to half before 160 ascending K32 F32 accumulations;
SILU/clamp expressions and guarded output addresses remain unchanged.

## Executed Proof

The same frozen executable runs the exact ignored standalone selector with the
Metal debug layer enabled in that child only. Shape is K5120/I2304/E384,
top-6, 1,024 tokens. Synthetic inputs are finite; streamed IQ2_XXS banks have
varied finite nonzero block scales. The fixture uses clamp 7, not model clamp 10.
It imports no model, GGUF, prompt IDs or header artifact.

All eight fixed cohorts pass: balanced16; 0/1/15/16/17/31/32/33 boundaries;
six-expert skew; invalid counts; invalid IDs; ranges 16..17, 31..32 and 0-only.
Five checked submissions per cohort establish exact raw Gate, final Up and
whole-pair bits, complete ordered A2/B4 census, finite active words, distinct
inactive qNaN poison, and separate finite-`f32::MAX` Up-store coverage. Physical
guards/slack, original input, full controls and borrowed banks remain unchanged.
Zero-only work must preserve poison; matching common no-ops cannot qualify.

Seven guarded allocations have cumulative priced storage of 2,528,231,424 bytes,
including the 64 MiB pipeline envelope. Adding the 128 MiB CPU reserve stays
within the 4 GiB cumulative cap; ordinary allocation admission and the 6 GiB
device cap also pass. No giant CPU bank/oracle clone is introduced.

The source fixture completes in 35.47 s. The outer pre-write verification
decision completes in 38.9383 s, within its 80 s cutoff, excluding queue wait
and final durable receipt I/O. These are correctness execution times, not kernel
latencies or performance measurements. The child exits normally without signals;
the canonical descriptor remains retained through checked completion and drain.

Proof stdout SHA-256:
`c6a6b0b1697b69ee523daa9382891bc7101b35b706d08b609f5e2d9a55fe5113`.
The accepted proof and raw records remain retained with their source, CPU,
configuration, executable, completed-review and resource bindings.

## Limits And Next Gate

This is a real target proof for the fixed synthetic cases, not every floating
input, device/compiler, population or clamp. Model clamp 10 still needs its own
raw/final checks. No claim follows about issued arithmetic, DRAM, registers,
occupancy, causal state, logits, likelihood, sustained throughput or a default.

The resource specialization is distinct from the closed four-SIMD width-only
adaptive leaf; a lower logical accumulator average alone is not new. The failed
token-K32 layout is not reopened or composed into it. Both closed negatives stay
closed.

The next separate discriminator reuses one fresh ordinary L8/start0/1024 capture
with actual model clamp and original banks. It compares incumbent A2 against B4
on the same output, warm AB then two uncensused ABBA blocks. Each must show at
least 10% combined command GPU reduction with no host regression; a miss closes
the implementation. That source, acquisition, same-new-binary proof and timing
packet are not qualified by this standalone result.

Frozen protocol: `docs/bench/2026-10-28-v41-small-resource-protocol.md` at
source `8ebb8c63`. Closed layout result:
`docs/bench/2026-10-28-v41-token-k32-result.md`.

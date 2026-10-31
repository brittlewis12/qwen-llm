# GLM Fixed Fused D512: Correct But Too Slow

## Decision

Close this fixed full-F32/Half D512 API/dataflow implementation. Matched M5
attention-only GPU medians are approximately **64 ms versus 21.8 ms CURRENT**,
or about **2.93x the latency**, in both paired blocks with stable controls.
There is no promotion, native whole-MLA adapter, model-quality campaign, repeat,
tile/barrier/offset sweep or precision-flag rescue for this design.

This is not a universal negative about fused attention, a causal ISA/spill
diagnosis, or a measured application/whole-MLA regression. The ordinary runtime
remains unchanged. Genuinely different formulations need a new leverage and
numerical-contract review, not another parameter setting for this packet.

Source is `9bfa7a212209d497dabaabdf3715208f9955c98d` on the experimental
`probe/glm-mla-fused-512` branch. Only portable evidence is maintained on main.
The [N64 runtime gate](2026-10-31-glm-mla-fused-n64.md) passes before this separate
N2048 admission, validation and cost gate.

## Accepted N2048 Correctness

The fixed position-zero request uses F32 Q/output `[512,64,2048]`, one shared
F16 C `[512,2048]`, finite NoSink=-1e30, scale 1/16 and inclusive causal visibility.
Inputs are byte-identical to the
[qualified CURRENT attention control](2026-10-31-glm-current-attention-budget.md).
CURRENT calls public `encode_latent_attention`; no private baseline bindings
are reconstructed. The candidate uses the unchanged frozen CAB library and
Case32 mode1, grid `[64,64,1]`, 256 threads. All proof/status work is charged.

A separate DEBUG1 process completes exactly CURRENT then candidate in four
seconds and exits successfully. Authored checking and independent raw replay pass:

- All 67,108,864 candidate output values are finite.
- All 1,048,576 status records across 4096 groups have zero errors and complete,
  valid actual tid/SG/lane membership under the checked scratch ABI.
- All 3072 guard bytes and 270,532,896 immutable Case/input bytes match.
- All 990 F64 samples pass unchanged absolute tolerance 2e-4; maximum error is
  2.6963301647e-8, or 0.000134817 of the tolerance.
- Samples cover all eleven planned rows, five heads and all sixteen output tiles
  across four partitions. The original 330-reference subdigest remains exact.

The independent audit parses the actual raw and recomputes expectations from
represented F32 Q, actual Half C and sink bytes. It does not call the authored
checker. Its expectations agree with the frozen table within 2.7755576e-17.
Only 990 outputs receive mathematical comparisons, not all output elements.

Candidate and CURRENT are not bitwise identical: 733/990 sampled words differ,
with maximum inter-arm difference 2.9802322e-8. Both pass their frozen F64 rule.
CURRENT's complete output hash matches its previously accepted full raw.
The actual PSOs remain SIMD32/max256: CURRENT static0/dynamic16384 and candidate
static30272/dynamic0, within the 32768-byte limit.

## Matched Cost Result

Only after accepted actual validation and independent replay, a fresh parent
assertion authorizes one separate DEBUG0 run of the SAME frozen executable.
Two untimed prechecks precede ABBA; every arm sub-block has eight excluded
warmups and sixteen retained samples. A final untimed candidate call supplies
the single final raw after ABBA ends in CURRENT.

| Segment | Retained GPU Median ms | Mean ms | Range ms |
| --- | ---: | ---: | ---: |
| A0 CURRENT | 21.807146 | 22.273383 | 21.745833 - 28.830708 |
| B1 Candidate | 64.002250 | 64.408695 | 62.746125 - 70.865125 |
| B2 Candidate | 63.902146 | 64.319216 | 63.145833 - 70.746083 |
| A3 CURRENT | 21.795229 | 22.227440 | 21.764000 - 28.282750 |

Paired candidate increases are **42.195104 / 42.106917 ms**, ratios
2.934921 / 2.931933, or +193.49% / +193.19% latency. Interblock median drift is
-0.054646% CURRENT and -0.156407% candidate. Each segment's first retained
sample is its maximum and stays included. No filtering, retries or clock/thermal
attribution occurs. No ISA, instruction, register or spill measurement exists.

Independent analysis verifies all 96 raw intervals, positive serial timestamps,
units, command order and retention flags. There are 99 commands, 64 retained
samples, 32 predefined excluded warmups and eleven full checkpoints. GPU
start/end times are read after checked completion and cover one complete
attention dispatch per command. Standalone opt1 host wall is separate and
noncanonical. No CPU hashing/readback/reporting occurs inside sample loops.

All eleven checkpoints retain accepted arm output hashes. They check the last
output, not every intermediate command. The complete final 555,748,640-byte
candidate raw is byte-identical to accepted validation, allowing reuse of the
independent correctness evidence. The process exits naturally in sixteen seconds
under an actual external 90-second limit and 20-second TERM-to-KILL grace.
Normal fail-fast self-leasing, DEBUG0 and no wired bypass remain in force;
post-exit lease inspection finds no openers. Offline CPU replay and analysis are
unleased.

## Resource And Interpretation Limits

The separately reviewed N2048 quote is B555843584 + Metal policy536870912 +
CPU explicit676989702 + CPU framework67108864 + shared reserve33554432 =
1,870,367,494 bytes, below the new 2 GiB cap. It includes one final raw-file
cache reserve, not two output clones. Validation buffer growth555810816 and
pre-call total556367872 fit their quotes; timing pre-call growth is556285952.
Policies are not formal physical peaks;
there is no incumbent credit, domain borrowing or duplicated shared reserve.

The collector links the exact cached production rlib using configured rustc
opt1/noLTO; all 160 cached inputs stay unchanged. This is a GPU-API attention
discriminator, not a canonical application benchmark. Both formulations satisfy
the bounded numerical checks but have different reduction/exp implementation
costs. The result does not isolate any one source of overhead or hardware peak.

The earlier 376.43 ms combined eleven-block MLA parent is not subtracted or
multiplied to invent application costs. The whole-MLA continuation target remains
both >=30% and >=8 ms saved per block; this negative attention screen does not
fund that next tranche. No weights, model, serving endpoint, installed binary,
PATH or SDK selection changes during this experiment.

## Frozen Evidence

| Evidence | SHA256 |
| --- | --- |
| Final host | `7a0b184e7a46dc543fa998335e2f135fbd50b1be64735407a322e1f55bf6c94e` |
| Candidate library | `cab33997b4866ce1e000ebce6895cf3551961e771d05c2c4d49258c50c70dba3` |
| 990-reference table | `53185ce3c4435037464d9fb660ed198beffc9e06916dd0fa5285c6e82c21aaf5` |
| Validation report | `0980a416697aea311acf7c6c3e530c90cb44c6769072a6dab751d58bb5bd7cc2` |
| Independent validation math | `5f2bf3c707dcbf73023551a915b53306adf5f52fc85f763eac11da60f05059cd` |
| Matched cost report | `73f3d1674ddae6ab1713da75b21032859cd202ab1b4d0098e8b35cce74ee1cc1` |
| Independent cost analysis | `9a198a81f9f8fe811b96a8913aaddc08eb4fb0e25044dc3f8657053bc058ba48` |
| Candidate raw, identical in both modes | `82a5c036abc1399c55aa6f9b3a4fb88d239b83db54e891f47f5fb208d1c0f8ad` |

Prototype source, successful arithmetic evidence and the negative cost are all
retained. None is discarded or relabeled to justify another run of this family.

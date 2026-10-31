# Bounded GLM M5 Routed Tensor Gate/Up Default

## Decision And Status

The parent authorizes C, `QWEN_GLM53_TENSOR_GATE_UP`, default ON within the
existing bounded M5 Fast/Half policy. The qualified configuration keeps
Q8-wide and Q5-down ON and KDA half staging (A) OFF. This is the same kind of
bounded precision-policy judgment described in
[the maintained M5 policy](2026-10-30-glm-m5-q5-down.md), not a formal global
noninferiority or equivalence result. In particular, long-stratum uncertainty
is accepted explicitly; the descriptive intervals are not a retroactive pass
of the old Exact/Fast preregistered procedure.

This source/default-only promotion is prepared on the isolated routed branch.
The parent owns independent review, maintained-main integration and a clean
final release handoff. No new GPU run, release/LTO/native build, installation
or service action accompanies this source change. Existing qualified binaries
predate the default flip and selected C explicitly.

A remains default OFF. Its standalone evidence below does not qualify A+C.
Neither the parked decode-fusion D nor the FFN encoder split is promoted;
their maintained-main additions are documentation only.

## Frozen Evidence And Workload

- Qualified source: `eea1d369e710e834372df462b8c47d1f3be1ea50`.
- Shared release library-test binary SHA256:
  `a514850e87208d07d18603e1cb6fd33778b3b76cc584da0d23c203e91e3ae136`.
- Frozen quality-v1 fixture SHA256:
  `7f9ac936b08ee0f4e47fbbb42dbfd46c3055f058b4f36fa5d832fbefcad43057`.
- Actual six-shard Q4 pin SHA256:
  `3db086f2521dccb556a3afa4cc456f55a33c34ceab67a6f96d16eee78be2b74b`.
- Quality report SHA256:
  `da5d41c5df38958508bf3d6a01cbfce451d11a516121bdac9ac65f33eb7553e1`.
- Quality analysis artifact SHA256:
  `3baa4b8c0b7e33c29f33fd6e79820eabaa804251e60b2a5e16433650989c95cd`.
- C heated latency report SHA256:
  `9e35df41321d6c27c03d97cf1cb5d22e310ba64ee30f02a6853976b240a12536`.
- Authoritative strict scorer source SHA256:
  `44ee0c82e3fdbb212be30ac88b0663d3a3e929b77b7c2b6251cd9f5942bb946f`.

Hardware is actual Apple M5 tensor-tier hardware, enforced by the tests. The
parent's maintained [M5 cohort hardware record](2026-10-30-glm-m5-transfers/README.md)
identifies M5 Ultra, 80 GPU cores, 256 GiB, macOS 27.0.1, SDK 26.2 and Metal
compiler 32023.864. The new reports do not independently record a frequency
trace or remeasure that full topology; no cross-device universal rate is claimed.

The actual-Q4 inventory pins all 42 main routed expert triples: 41 Q4_K
gate/up pairs and the Q5_K pair at block 11, with 39 Q5_K down banks and
Q6_K down at 11/12/44. All 24 serialized Q8 MLA k/v banks, including MTP 45,
and all 306 Q8 KDA projection matrices are checked before weight loading.
Rank expansions and KDA output are Q8, not the Q6 synthetic-fixture defaults.
Primary metadata SHA, shard names/sizes and retained stamps are checked;
weight payload hashes are not reverified. These are actual named-role pins,
not an assertion that a recorded table digest alone pins every tensor role.

Normal CPU-aware full-model/session preflight precedes weights. Each arm owns
the same admitted 16 MiB `normalized_half` activation buffer at R2048. It is
shared scratch, not an expanded weight bank or a Half down-input buffer.
The full admitted charge remains even for inactive TLS or shorter actual
views. C always converts the fresh FFN-normalized producer; it never reuses
old KDA contents. A's independent execution flag is not enabled by ownership.

## Extended Quality

The validation-ON run completes in 345.48 s with one weight load, 60 sessions,
22 documents, 1,408 targets per arm and 2,816 finite full-vocabulary rows.
There are 8 prose, 8 code and 6 long documents, each with 64 teacher-forced
targets. CURRENT means qualified Q8-wide/Q5-down Fast/Half with C OFF, not
the historical Exact reference. C uses the same controls with C ON; A is OFF
in both arms, and ordinary decode is outside prefill scopes. Stamps remain
unchanged and observers are clear.

| Metric | CURRENT | C |
| --- | ---: | ---: |
| Mean NLL | 1.445751627193185 | 1.4481441681213159 |
| Mean KL(CURRENT \|\| arm) | 0 | 0.003209616080377 |
| Maximum KL(CURRENT \|\| arm) | 0 | 0.070720390913 |
| Top-1 correct | 980/1408 | 982/1408 |
| Top-1 agreement with CURRENT | 1408/1408 | 1377/1408 |

Mean NLL delta is +0.002392540928131, approximately +0.2395% relative
perplexity. Thirty-one top-1 predictions change; net correctness increases
by two. This is not bitwise or numerical equivalence to generic gate/up.

The existing paired-document bootstrap is reused descriptively: 20,000
resamples, seed 20261008, NumPy 2.5.4, central 90% intervals. It compares
CURRENT Fast/Half to C Fast/Half, not Exact to Fast.

| Stratum | Mean NLL delta | Descriptive central 90% interval |
| --- | ---: | --- |
| Overall, 22 documents | +0.0023925409 | [-0.0027069224, +0.0071478978] |
| Prose, 8 documents | +0.0000504289 | [-0.0115507504, +0.0103064367] |
| Code, 8 documents | +0.0004203329 | [-0.0041686588, +0.0050389829] |
| Long, 6 documents | +0.0081449676 | [-0.0005322327, +0.0173199514] |

Overall top-1 accuracy delta is +0.0014204545, with descriptive central 90%
interval [-0.0014204545, +0.0049715909]. The overall NLL interval and especially
the long-document upper bound remain material uncertainty, not an NI pass.
The parent accepts this bounded change under the existing Fast/Half policy.

All eight tool sequences per arm are strict-correct, independently rescored
with the authoritative Python scorer. All complete sampled-ID sequences,
decoded UTF-8 bytes and serialized calls match byte-for-byte. These are
T09-T12 with greedy and seed 11, cap 256. Four of eight sequences per arm
(T09/T10 at 1455 rows) are controls with no eligible C gate/up chunk. Thus
8/8 is a narrow registered result, not a general agent guarantee.

## Primary Performance Evidence

The DEBUG-OFF one-load run completes in 106.87 s: 40 fresh sessions, eight
excluded warms and 32 measured samples. Every shape warms both arms, then
runs CURRENT/C/C/CURRENT followed by C/CURRENT/CURRENT/C. Capacity is 4416,
workspace R2048, A OFF and qualified core/wide/Q5 ON throughout. Each timer
covers ordinary prefill including full wait/readback, excluding fresh admitted
session allocation/drop. Checks, hashes and JSON follow each eight-sample
block; there is no profiler, census, oracle work or diagnostic capture between
samples. Both arms own equal admitted Half storage and equal ledgers.

| Prompt | CURRENT mean ms | C mean ms | Latency change | ABBA / BAAB C-over-CURRENT | CURRENT block drift |
| --- | ---: | ---: | ---: | --- | ---: |
| 512 control | 1272.470417 | 1272.796323 | +0.025612% | 1.000220 / 1.000292 | +0.027313% |
| 1024 | 1490.772354 | 1047.987917 | -29.701680% | 0.704022 / 0.701945 | +0.091141% |
| 2048 | 2707.382094 | 2002.000948 | -26.053993% | 0.739881 / 0.739040 | +0.134187% |
| 4096 | 6518.268615 | 4995.776365 | -23.357311% | 0.761315 / 0.771035 | +10.932701% |

The stable headline is about 29.7% lower latency at 1K and 26.1% at 2K on
this cohort. The 512 control is bit-exact and effectively flat. At 4K, both
arms slow between blocks: CURRENT +10.9327%, C +12.3491%. Both block ratios
still favor C, but the pooled 23.36% decrease is a qualified run result, not
a universally precise 4K effect. There is no frequency trace that resolves
that nonstationarity.

All 32 measured samples follow, in milliseconds rounded to three decimals;
the report hash identifies full precision. No sample or drifting control is
dropped. Columns are the continuous ABBA+BAAB order described above.

| Prompt | A0 | C1 | C2 | A3 | C4 | A5 | A6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1272.524 | 1273.907 | 1271.247 | 1272.070 | 1272.696 | 1273.988 | 1271.300 | 1273.336 |
| 1024 | 1489.692 | 1049.473 | 1048.644 | 1490.495 | 1046.742 | 1490.729 | 1492.174 | 1047.093 |
| 2048 | 2706.235 | 2001.209 | 2002.385 | 2704.898 | 2001.879 | 2708.728 | 2709.667 | 2002.531 |
| 4096 | 6014.154 | 4655.503 | 4754.992 | 6346.696 | 5011.050 | 6740.564 | 6971.661 | 5561.560 |

The earlier canonical native 6991 A/G/G/A conversion corroborates roughly
20-25% prefill improvement, but its noisy controls prevent treating it as
the primary point estimate. Its unchanged native binary SHA256 is
`03a3742ac5e82dc07d196b5185c72c74316b4fb784e5e2e9e362585d9977eee8`.
Earlier validation-ON profile spans are diagnostic and are not recoverable
ordinary wall-time savings. No new native build is made for this decision.

## A Is Standalone And Still Off

A's separate DEBUG-OFF one-load EEA run completes in 115.47 s with 32 measured
samples, eight warms, 40 sessions and the same 16 MiB ledger. It keeps C OFF,
KDA staging requested globally, and core/wide/Q5 ON. Report SHA256:
`4543343f46c3219d7a98a0f3d8b927fc36ce515e77d2bdd1e225d8b86219ca06`.

Stable 1K means are 1490.202604 -> 1468.896938 ms (-1.4297%); ABBA/BAAB
improvements are 1.4507%/1.4087%, with CURRENT block drift -0.0056%.
Stable 2K means are 2706.686490 -> 2667.610448 ms (-1.4437%); ABBA/BAAB
improvements are 1.4206%/1.4668%, with drift +0.05826%. The short control
is flat (block changes +0.0271%/-0.0302%). At 4K, the pooled -1.655% has
CURRENT block drift +6.0005% and block improvements 2.245%/1.098%; it is
not a precise stable headline. Prefer the standalone approximately 1.44%
stable 1K/2K result.

The parent's A07 standalone numerical run passes 195 paired full-vocabulary
row comparisons (390 arm rows), all bits equal, with no state captures.
That standalone proof and the reusable Half buffer do not establish A+C
composition. A therefore stays default OFF. After C's default changes, every
A-only replay must explicitly set `QWEN_GLM53_TENSOR_GATE_UP=0`; the owned A
tests enforce it. A+C is neither this default decision nor a qualified mode.

## Default, Rollback And Invariants

`QWEN_GLM53_TENSOR_GATE_UP` is default ON and latches on first read through
the existing process-cached flag helper. `0`, `false`, `FALSE`, `no` and `NO`
disable it; unset/unrecognized values resolve to the declared ON default.
Set policy before process startup/admission: changing an environment variable
after first read does not alter an existing process. This source change does
not restart services. Test-only TLS controls remain scoped and unwind-safe;
they do not discount admitted physical ownership.

- C=0 restores generic gate/up with the current down policy, independently of
  Q8-wide and Q5-down. It restores qualified CURRENT, not an older all-generic
  baseline unless those other policies are also rolled back.
- Eligibility remains M5 tensor tier, Fast/Half, resolved routed Fast,
  H4096/F2048/E288/top8, equal actual Q4_K or Q5_K pairs and actual 1024 or
  2048 rows. A 4096 prompt uses two eligible 2048-row chunks.
- Exact, F32, other dtype/geometry and short-row dispatch fallthrough, M4 live
  behavior, global tensor rollback and the current Q5-down rollback are
  unchanged. Q5-down OFF declines C for Q5-down compositions; it does not
  disable eligible C gate/up in the three Q6-down blocks.
- Converter, both gate/up, effective down, bucket and weighted sum still
  preflight before mutation. Missing eligible pipelines/bad bindings error;
  unsupported capacity declines the whole candidate to the old composition.
- Operands remain Half with F32 sums. Only the already-qualified tensor
  reduction order differs from generic; the epilogue and unweighted F32 inner
  -> current down -> post-down routing weights are unchanged.
- The one admitted activation buffer is priced in full, including inactive
  execution views; device-free planning stays conservative. No new buffer
  specification, expanded weights, shader, codec, precision mode or decode
  route is added by this default flip.

Owned replay protocols keep their explicit environments: C tests require C=1
and A=0; A-only tests require C=0 and A=1. Their math, sampling, measured blocks
and fixtures are unchanged. See [exact tests and replay controls](../GLM-ROUTED-QUALIFICATION.md).

Historical `tensor_m5_transfers` CURRENT/WIDE/Q5 replays must explicitly set
both `QWEN_GLM53_TENSOR_GATE_UP=0` and `QWEN_GLM53_KDA_HALF_INPUTS=0` after C's
default becomes ON. Those old scopes do not know either newer flag, and Q5
rollback alone leaves the three Q6-down C compositions eligible. The old Rust
packet bodies and fixtures remain byte-unchanged; this note preserves their
original isolation rather than requalifying or relabeling old measurements.

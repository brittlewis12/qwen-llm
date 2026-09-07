# Fresh serving: Q8 endpoint PASS / opt-in KEEP, automatic HOLD

## Disposition and commits

- `537215bc` tests default-off fresh packed serving on Q4/Q8. The independently
  frozen Q8 A-B-B-A packet passes all target and guard gates. Q4 fails its
  512-token guard and remains unsupported, including when forced.
- `f5b77787` narrows to Qwen38/Q8 dense27 and provisionally enables the default.
  A post-narrowing correctness check preserves every response and selection but
  exposes a large first-request timing outlier amid substantial global compression.
- `c1133d1b` conservatively returns the feature to explicit opt-in. The Q8
  balanced PASS remains valid; it is not re-scored or relabeled a failure. The
  unpaired observation is risk evidence, not replacement timing authority.
  Automatic rollout waits for attribution under an explicit host-state contract.
- Final use: `QWEN_SERVE_FRESH_PACKED=1`. Unset, `0`, malformed and non-Unicode
  values disable it. This switch does not change the existing restored-Q8 policy.

No repeat qualification, retrospective gate change, or warm-only gate rewrite.
No cold-load, CLI, sampled-distribution or speculative benefit is claimed.

## Implementation and safety

Fresh selection requires Qwen38 template, Q8_0 LM head, dense 64-layer / hidden5120
/ 24 query heads / 4 KV heads / head256 geometry, no loaded drafter, greedy current
request, no cache lookup hit, and 19-48 prompt tokens. Cached/exact requests cannot
enter the fresh branch. The prior Q8 restored 7-32 policy is unchanged.

Existing explicit single-chunk prefill uses full prompt key extent and query width.
The complete driver-priced scratch plan must fit 128 MiB. Candidate denial
re-admits the baseline; allocation failure retains the original serial allocation.
Every fresh candidate's original baseline needs zero packed scratch, so this does
not reuse a smaller candidate admission for a larger fallback. The source retains
the same lookup and request-state contract; no consumed-alias machinery returns.

CPU tests cover profile/range boundaries, cached/sampled/drafter exclusion, flag
defaults/rollback, and the existing mocked admission/allocation fallback. Those
fault tests are not real Metal OOM tests. The earlier release oracle checks logits
and every KV/GDN state/conv layer numerically, with 3/EOS,64,64 equal greedy tokens
per model. Actual scratch in that oracle is 11,698,176 / 19,382,272 / 29,048,832
bytes at 19/32/48. Serial uses zero packed scratch: this is a bounded memory trade.

## Frozen measurement contract

Same release binary `537215bc` for both arms, A env=0 / B env=1. Four disposable
processes per model, A1/B1/B2/A2; all GPU execution serial. Fixtures are generated
CPU-only using the real renderer/tokenizer and match the state-oracle prompt IDs.

Eight sequential requests: fresh19/color, exact19, fresh32/code128, fresh48/prose128,
guard18/color, guard49/code128, sampled32/color and guard512/color. First-token
time ends at the first real SSE string delta; request wall ends at `[DONE]`.
Model readiness precedes the timer. Fresh19 is first inference in its process;
32/48 are model-warm cache misses. These are not process-cold measurements.

Each model qualifies independently. Fresh19 must save >=40% TTFT and >=30% wall
in both orders. Both code32 and prose48 must save >=40% TTFT and >=10% wall in
both orders. Each primary control spread must be <=5%. Guard18/49/512 and sampled32
wall median regressions must be <=3%, with <=5% control spreads. Exact-hit timing
is reported without percentage authority. All request inputs, response text/hash,
full usage/cache counts, status and terminal reason must match within each model.

## Q8: all gates pass

Median times in ms; positive savings means faster.

| Row | TTFT A -> B | TTFT savings | Wall A -> B | Wall savings |
| --- | --- | ---: | --- | ---: |
| Fresh19 / 3 | 1263.159 -> 333.724 | 73.580% | 1391.780 -> 464.215 | 66.646% |
| Exact19 / 3 | 9.590 -> 9.949 | no authority | 138.807 -> 138.526 | no authority |
| Fresh32 / 128 code | 1810.316 -> 224.946 | 87.574% | 9327.613 -> 7747.548 | 16.940% |
| Fresh48 / 128 prose | 2697.724 -> 310.878 | 88.476% | 10238.716 -> 7872.038 | 23.115% |
| Guard18 / 2 | 1021.686 -> 1021.531 | 0.015% | 1090.953 -> 1090.827 | 0.012% |
| Guard49 / 128 | 311.775 -> 311.209 | 0.182% | 7871.042 -> 7913.916 | -0.545% |
| Sampled32 / 2 | 1813.677 -> 1814.297 | -0.034% | 1883.481 -> 1884.261 | -0.041% |
| Guard512 / 3 | 2073.808 -> 2084.588 | -0.520% | 2207.346 -> 2218.650 | -0.512% |

| Primary | TTFT paired savings AB / BA | Wall paired savings AB / BA | TTFT / wall control spread |
| --- | --- | --- | --- |
| Fresh19 | 74.060% / 73.096% | 67.220% / 66.067% | 0.963% / 0.784% |
| Fresh32 code | 87.549% / 87.600% | 16.630% / 17.247% | 0.652% / 0.723% |
| Fresh48 prose | 88.414% / 88.538% | 22.960% / 23.270% | 0.055% / 0.129% |

Guard wall control spreads are at most 1.782%; worst median wall regression is
0.545%. Exact-hit TTFT moves +0.359 ms; this is disclosed, not a percentage claim.

## Q4: guard FAIL / unsupported

| Row | TTFT A -> B ms | Wall A -> B ms | Wall savings | Wall control spread |
| --- | --- | --- | ---: | ---: |
| Fresh19 / 3 | 814.025 -> 323.084 | 903.477 -> 413.079 | 54.279% | 0.903% |
| Exact19 / 3 | 10.439 -> 10.205 | 101.249 -> 99.568 | no authority | 0.680% |
| Fresh32 / 128 code | 1198.782 -> 243.020 | 6295.149 -> 5326.122 | 15.393% | 0.241% |
| Fresh48 / 128 prose | 1803.321 -> 362.514 | 6929.466 -> 5589.939 | 19.331% | 2.456% |
| Guard18 / 2 | 692.735 -> 708.327 | 743.721 -> 759.508 | -2.123% | 4.473% |
| Guard49 / 128 | 349.976 -> 375.994 | 5521.383 -> 5609.463 | -1.595% | 3.589% |
| Sampled32 / 2 | 1229.850 -> 1253.253 | 1281.592 -> 1304.872 | -1.817% | 2.394% |
| Guard512 / 3 | 2260.531 -> 2464.293 | 2358.958 -> 2559.053 | **-8.482% FAIL** | 3.801% |

All primary target gates pass, but the independently frozen guard fails. Its
paired wall regressions are 13.562%/3.589%. The guard does not select fresh packed
execution; its regression is not causally localized. Do not excuse it as noise,
retune the policy or use the Q8 result to promote Q4.

## Correctness, selection and post-narrowing check

All 64 paired responses match. Each B selects exactly fresh19/32/48, with actual
single-chunk query rows and matrix key capacity equal to prompt length. The seven
non-exact requests per process report zero cached tokens; exact19 reports19.
No candidate allocation failure or fallback occurs. Logs prove allocated plan
selection, not merely an eligibility decision.

The `f5b77787` check adds 16 matching responses, Q4 selecting none and Q8 selecting
19/32/48 with the environment unset under its provisional default. These checks
are not pooled into A-B-B-A timing. Current `c1133d1b` restores explicit opt-in;
its absent/0/invalid-off and 1-on semantics are checked by pure tests. The earlier
Q8 selected execution is the same route reached with explicit 1 today.

Packed state is numerical, not bitwise versus serial. The sampled guard is a
fresh cache miss, not a sampled continuation oracle. Future sampled requests can
inherit checkpoints created by greedy packed requests; no distributional
equivalence is established by these greedy and forced-color witnesses.

## First-request outlier and host-state limits

Post-narrowing Q8 first19 records TTFT 1386.080 ms, prefill 1348.7 ms and wall 1514.773
ms, despite correct packed19 selection. Balanced B1/B2 prefill was 293.0/298.0 ms.
Later packed32/48 TTFT is 233.122/326.968 ms. Model load is 4391.0 ms versus roughly
3580-3684 ms in the balanced packet. Do not infer an intrinsic first-use bug,
PSO effect, or percentage regression from an unpaired observation.

The CPU audit finds no thermal/performance warnings, but substantially different
whole-process global VM activity:

| Q8 run | Compression events | Decompression events | Pageouts |
| --- | ---: | ---: | ---: |
| A1 | 0 | 20 | 556 |
| B1 | 0 | 3 | 1 |
| B2 | 0 | 4 | 480 |
| A2 | 0 | 1 | 256 |
| Post-narrowing | 660122 | 610262 | 665 |

These counters span the entire process and host, not the first prefill or Qwen
alone. They show host interference risk, not its cause or timing. The packet has
no retroactive zero-pageout gate; nonzero pageouts are disclosed. Warm file
residency and absence of thermal warnings do not establish equivalent VM state.

Independent review initially favored default promotion. Challenging that advice
with the outlier and then the VM counters led to explicit-opt-in rollout instead:
preserve valid positive evidence without silently imposing unpriced first-request
risk. Any future attribution needs phase-local observations and a declared host
contract, not a repeated benchmark to erase this observation. No generic PSO or
residency project is justified by these data.

## Artifacts and validation

Raw packet: `target/profiles/fresh-serving-http/`, including frozen `PROTOCOL.md`,
CPU fixture logs/JSON, all 8 balanced processes, both summaries, `score.py`, both
post-narrowing processes, `check_narrowed.py` and CLI test logs. Prior state oracle:
`docs/bench/2026-09-06-fresh-short-packed/RESULT.md`.

All ten owned servers exit cooperatively with 130. No remote push. The final CLI
suite has 376 passed, 13 ignored; targeted policy/rollback, formatting and diff
checks pass. Mocked
fallback tests remain explicitly distinct from real allocation-pressure evidence.

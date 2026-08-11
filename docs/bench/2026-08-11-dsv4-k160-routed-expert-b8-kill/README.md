# DeepSeek K160 routed-expert B=8 floor

## Question

Can fixed-cohort execution reuse routed-expert weights across eight DeepSeek V4
K160 requests strongly enough to justify a family-specific static batch
executor, before paying for eight full sessions, route scheduling, or product
integration?

This is the cheapest favorable falsifier. Every row deliberately selects the
same six experts. Real requests can only have equal or lower route overlap.

## Contract

- Asset: exact 1,328-tensor K160 REAP Q3_K/Q4_K cohort,
  `89,920,886,108` source bytes.
- Device: Apple M4 Max.
- Layer: 20, representative of the uniform 43-layer expert layout.
- Rows: eight deterministic, distinct F32 hidden vectors.
- Routes: experts `[0,1,2,3,4,5]` for every row, with distinct positive weights
  normalized to the model's `1.5` routed scale.
- Control: eight production fast all-slot Q3_K/Q4_K bodies plus eight scalar
  weighted sums.
- Candidate: six expert-major N=8 gate/up mat-mats, exact clamped SwiGLU, six
  N=8 down mat-mats, row scatter into token-major slots, and one packed weighted
  sum.
- Excluded symmetrically: router, RMS norm, architectural shared expert,
  attention, mHC, residual, LM head, and session state.
- Primary clock: `MTLCommandBuffer` GPU timestamps.
- Acquisition: two warmup pairs followed by twelve counterbalanced AB/BA pairs.
- Whole-model residency set: explicitly disabled.
- Source identity:
  `git-source-sha256-v2:c051559e258dac6b347f38b6839fe00de7ce118a0dca3fae578a6602c9b0cf24`.
  The dirty identity is sufficient to kill a candidate that misses its local
  gate by a wide margin; it carries no promotion authority.

Invocation:

```text
QWEN_DSV4_RESIDENCY_SET=0 ./target/release/qwen-bench --allow-dirty \
  dsv4-routed-expert-b8-floor \
  --model /Users/tito/models/deepseek-v4-flash-0731-reap-k160/DeepSeek-V4-Flash-0731-REAP-K160-Q3_K_Q4_K-00001-of-00004.gguf \
  --warmups 2 --samples 12
```

## Result

The probe loaded residency in `1,959.51 ms`, constructed no model session, and
added `3,211,264` Metal allocation bytes (`3,282,720` logical scratch bytes).
Memory returned normally after process exit; wired pages remained at the host
baseline.

| Arm | Dispatches | Median GPU | Median wall | Relative |
| --- | ---: | ---: | ---: | ---: |
| eight fast all-slot rows | 40 | `1.976000 ms` | `2.118583 ms` | `1.0000x` |
| six common experts at N=8 | 31 | `1.862979 ms` | `2.005375 ms` | `1.0607x` |

The candidate saved `0.113021 ms/layer` across the complete eight-row cohort.
Its twelve GPU samples ranged from `1.829167` to `1.874625 ms`; control samples
ranged from `1.955875` to `1.985083 ms`.

Numerical comparison over 32,768 routed-output elements:

- finite: yes
- global cosine: `0.9999999183`
- minimum per-row cosine: `0.9999999105`
- relative RMS error: `4.0486e-4`
- maximum absolute error: `2.7992e-8`
- bit-identical elements: `2`

The functional difference is expected from N=8 mat-mat reduction/staging versus
the singleton all-slot kernels. It is numerically healthy, but the performance
result is far below the preregistered economic threshold.

## Whole-token economics

The maintained K160 attribution places routed experts at approximately
`10.51 ms` of a `46.402 ms` token. The measured local speedup removes only
`5.72%` of that stage:

```text
routed saving = 10.51 * (1 - 1 / 1.0606667) = 0.60 ms/token
whole speedup  = 46.402 / (46.402 - 0.60)     = 1.013x
```

This is an optimistic ceiling because the fixture grants perfect six-expert
overlap and charges no route grouping or cohort scheduling. It cannot approach
the existing `1.37x` K160 B=2 independent-queue fallback.

## Decision

**KILL common-route Q3_K/Q4_K mat-mat as the basis for a K160 static batch
executor.** Delete the diagnostics spike and retain only this evidence.

This does not claim that every DeepSeek batching strategy is impossible. Reopen
only for one of these materially different premises:

1. A new expert kernel clears at least 30% on this charged routed-stage floor.
2. Attention, mHC, or another stage demonstrates a separate large whole-token
   reuse ceiling and a complete-cell candidate beats queue overlap.
3. A shared cross-family executor lowers implementation cost enough that a
   measured low-single-digit increment becomes product-worthy.

# Flash-Next shape-complete appends, 2026-11-09

**Default-on candidate; not merged.** M5 Ultra, production GSQ-RCO IQ3_S,
base `fe0df5796c69d7076ec83f9d011d48996ac59a99`, measured/tested source
`76cce044f6c9b39e3b9b2b883fe169cfd129327c`. This implements the first swing
from [the decomposition](2026-11-09-append-decomposition.md), including
whole-model qualification rather than inferring a gain from isolated tiles.

## Native result

Same 65,536-token native snapshot, real Rust continuation, restored outside
timing. ABBA + BAAB, four retained samples per policy/shape; warm route census
outside timing. Full endpoint logits/status handling is inside timing. All
campaigns require the canonical Metal lease, nice 10, thermal zero and a
one-second stop-on-rise monitor; timing never acquires the build lock.

| Real append rows | Rollback ms | Default ms | Saved ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 400 | 580.12 | **313.35** | **266.78** | **1.851x** |
| 512 control | 387.15 | 360.74 | 26.40 | 1.073x |
| 2,000 | 1,522.94 | **1,126.16** | **396.78** | **1.352x** |
| 2,048 control | 1,064.23 | 1,063.10 | 1.13 | 1.001x |

The 400-row path now beats the former 512-row path and the 330–390-ms target;
2,000 rows meets the 1,050–1,150-ms target. The first route-completion screen
was approximately 580→349 / 1,522→1,230 ms; HC reduction and qualified Q5
completion supplied the remaining gain. The final table uses one current
source and paired controls, not those development screens.

At 400 rows the previous optimistic compute/byte roof remains about 39.1 ms,
now **12.5%** of native wall; at 2,000 the 195.6-ms roof is **17.4%**. These
are useful-work lower bounds, not measured recoverable gaps. No new exclusive
post-change kernel decomposition was taken. Do not subtract the earlier
profile's category times from the new total.

## What ships

- Admit bounded 128..511 rows to the existing native-quant tensor dense,
  gate/up and parallel-bucketing routes; complete IQ4_NL/Q2_0 tensor down
  through 2,047 rows. Preserve the incumbent staged Half operand values and
  F32 accumulation. Existing 2,048-row qualification stays intact.
- Mask real HC and narrow-F32 tails without adding logical model timesteps.
  BF16 HC uses four K partitions, balanced F32 sums and 32-query tiles only
  for newly admitted rows. The existing aligned kernel remains the control.
  Complete the strict router/down widths and the F32 coalesced injection/
  48-wide GDN projections. Missing pipelines retain the qualified fallback.
- Admit qualified Q5 narrow tensor shapes. N12,288 failed the F64 peak gate;
  aligned M2,048 was bitwise equal but about 7.6 ms slower in a paired screen.
  Both keep their incumbents. No new persistent buffers or snapshot format;
  decode batches below 128 rows and non-tensor hardware retain their routes.

`QWEN_FN_APPEND_SHAPES=0` restores all prior row/alignment gates. Independent
`QWEN_FN_APPEND_TENSOR_DOWN=0` and `QWEN_FN_APPEND_HC_SPLIT=0` restore those
components. All three are default-on; prior subordinate controls still apply.
Run, bench and serve share these engine routes.

**Expert-panel probe: stop.** Although 79.6% of touched expert buckets at
400 rows fit 16 slots, separate/fused smaller panels measured 314.97/314.75 ms
at 400 and 1,129.11/1,135.78 ms at 2,000 in fresh whole-model screens. Separate
panels matched all output bits in four native formats; the fused model's
prefix and every endpoint matched the separate policy. Padding reduction did
not yield a useful whole-append gain. The aligned K160/M64 two-part HC probe
failed staged-F64 peak error at M513; it received no text or timing promotion.
Their runtime paths/knobs are removed; replay patches remain under
`scripts/research/fn-append-shapes/`. Further progress needs a new measured
attention/dataflow or expert-weight reuse hypothesis, rather than more panel
padding arithmetic. No additional TTFT reduction is claimed from the roof.

## Distribution and F64 qualification

Frozen four Rust/four manual continuations; 128 true-next-token targets per
document/cell, including endpoint. Baseline, candidate and F32 GDN control
prefixes execute independently, including irregular internal final chunks.
The baseline repeat restores the baseline snapshot; each append executes
separately. Full-vocabulary teacher forcing; no generated feedback. All
baseline repeats match bits. Greedy agreement is descriptive, not a gate.

Prespecified signed ΔNLL upper-95 bound from a document/circular-32-target-block
bootstrap must fit the arithmetic-control envelope, with mean KL ≤0.01.
16/64-block sensitivity must agree. The envelope is the 95th percentile
absolute per-document control mean. Seed 20261109, 10,000 resamples. It is an
empirical reduction-order control, not a claim of statistical equivalence.

| Prefix / append | Signed ΔNLL nats/token | Mean KL | Upper95, block32 | Control envelope |
| --- | ---: | ---: | ---: | ---: |
| 8,192 / 400 | +0.000018 | 0.004854 | 0.006912 | 0.025042 |
| 8,192 / 2,000 | −0.000332 | 0.003933 | 0.004856 | 0.014015 |
| 8,693 / 400 | −0.004414 | 0.005285 | 0.000746 | 0.024806 |
| 8,693 / 2,000 | +0.001255 | 0.002503 | 0.005906 | 0.020030 |
| 65,536 / 400 | −0.006227 | 0.007808 | 0.004071 | 0.022880 |
| 65,536 / 2,000 | +0.002120 | 0.003829 | 0.007246 | 0.007918 |

All six primary cohorts pass, including all three block lengths. The largest
positive mean is +0.00212, within the earlier +0.005 precedent; a negative
ΔNLL is an improvement and is not treated as an absolute-loss violation.

**Code-only check:** the initial four-code-document slice at 64K/2K averaged
**+0.00603**, versus −0.00179 for manuals; its separate confidence bound did
not pass. Do not hide that slice behind the pooled result. One additional,
disjoint code window was frozen before measurement (offset 2,304 instead of
128, 256 targets/document, four documents; no repeated sampling until pass).
It yields **−0.00216**, KL **0.00323**, upper95 **0.00491** against control
**0.00847**, passing all block sensitivities. Equal weight per 2K window gives
**+0.00194** across the two code windows; this is descriptive, not a new
post-hoc confidence test. Both windows and each document remain in
[quality.json](2026-11-09-fn-append-shapes/quality.json).

Actual-operand F64 gates compare both SSE and peak with no additive waiver:
F32 values for F32 operators, identical incumbent-staged BF16/F16 values for
HC/quantized operators. New routed gate/up/down, HC, Q5 and narrow-F32 cases
all satisfy no-worse error. The original-F32 conversion floor is also reported
separately; it is not used to excuse a worse reduction. Poison-offset guards
and actual dispatch census cover rows 128,129,399,400,401,511,512,513,1009,
2000,2047,2048 as applicable. Unqualified Q5 and aligned fallbacks are checked.

## Steady serving

Pending final two arms. Three distinct 48K warm agents, seeded real FN call
distribution, 15-s ramp and 180-s common active window, natural EOS, concurrent
default. Both policies retain the same main batching/GPU-feedback defaults.
Warm-up and cancellation/drain are excluded; report censored waits and stream
gaps, rather than imposing exact completed-token work.

## Evidence and reproduction

Compact [native samples](2026-11-09-fn-append-shapes/native.csv),
[quality summaries](2026-11-09-fn-append-shapes/quality.json),
[test counts](2026-11-09-fn-append-shapes/tests.json) and launch receipts are
committed beside this packet. Raw numeric JSONL/error/thermal records are in
`$RUNS/2026-11-09-fn-append-shapes`; regenerated corpus/source/target/bundles
are removed. Evidence identity is the source commit plus recorded command.

```sh
# At 76cce044; build under lockf -k BUILD_LOCK, nice -n 10, mise exec --.
cargo test --offline --locked --release -p qwen-llm --lib --no-run
cargo test --offline --locked --release -p qwen-cli --no-run
cargo build --offline --locked --release -p qwen-cli --bin qwen --bin qwen-tok
python3 scripts/research/fn-append-shapes/prepare_inputs.py --out "$RUNS/inputs"
# LIB_TEST is Cargo's reported qwen_llm executable, not a guessed cache path.
FLASH_PREFILL_MODEL="$MODEL" FLASH_PREFILL_OUT="$RUNS/native.jsonl" \
FLASH_APPEND_PLAN="$RUNS/inputs/timing-65536.json" \
python3 scripts/research/fn-append-shapes/run.py --out "$RUNS/native-driver" -- \
  "$LIB_TEST" qwen4exp_runtime::tests::append_shapes::native_append_shape_packet \
  --ignored --exact --nocapture --test-threads=1
# Replace timing plan by each quality plan, then analyze_quality.py FILE --out JSON.
```

Full release library **2,471 passed / 779 ignored**; every CLI test target
**1,697 passed / 72 ignored**. Model-backed ignored Q5/routed F64 tests also
pass under Metal validation. Quality-analysis unit tests **4 passed**;
steady-harness unit tests **8 passed**. Contracts pass at `76cce044`; the
final documentation tree is checked again before handoff.

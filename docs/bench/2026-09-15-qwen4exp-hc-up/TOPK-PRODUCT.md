# Guarded Singleton Selector: Product Opt-In PASS

Historical delivery policy. September16 promotes qualified M4 Max defaults and
retires split QSA: `../2026-09-16-flash-defaults/RESULT.md`. Measurements below retain
their original QSA-on/HC-off composition; do not relabel them as a new joint gain.

Default-off `QWEN4EXP_GUARDED_TOPK=1`, independently selectable from QSA and HC.
Unset/0 retains incumbent routing; malformed/non-Unicode values fail. Library
configuration is typed `Qwen4ExpDecodeOptions.guarded_topk`, not a global environment
switch. CLI retains that option through packed admission, scalar retry and scalar
loading. Existing generic parallel selector host n<=256 remains unchanged.

## Scope And Ownership

Only N512/K10 singleton selection uses the new kernel. Eligible scalar prefill and
decode use it; packed bodies and packed N1 singleton reuse retain incumbent math.
Other geometry falls back. SIMD width32,512 threads and6144 dynamic shared bytes
plus static memory are preflighted before model admission. The checked host validates
serial encoder/device, dtype/shape/range/alignment, writable outputs and disjointness.

No GPU buffer or memory-plan change. Forty-eight private MoE workspace booleans
default false. Root configuration first validates root, every child and nested MoE
owner/device, then mutates bindings in a second pass. Reset and checkpoint restore
preserve configured options. No public mid-flight toggle or test override in the
actual-product packet.

## Nonfinite Compatibility

The kernel classifies exponent bits with integer loads, then makes a synchronized
uniform decision. Finite input uses parallel selection and the incumbent ordered
selected-score softmax; nonfinite input runs the serial algorithm on one lane.
Shared-router sigmoid arithmetic is unchanged.

Compatibility01 FAIL is retained: explicit shader n==512/k==10 checks plus copied
serial body produced finite weights for all-negative-infinity, unlike incumbent
NaNs. This demonstrates a compatibility failure, not a proven compiler cause.
One versioned source change removes those equality checks while keeping the
incumbent dynamic k bounds. The checked host still enforces N512/K10/TG512.

Compatibility02 PASS53 cases in0.22s under API validation:13 finite cases, all
negative infinity/positive infinity/NaN, seven finite candidates with remaining
negative infinity, and positive/negative infinity/quiet-NaN/signaling-NaN values
at early/cutoff/half-boundary/late indices. Exact IDs and non-NaN output bits;
NaN placement agrees, with zero observed payload differences. Payload identity
is not the promised general comparison policy. Distinct finite poison values,
explicit overwrite checks, offset buffers, guards and input immutability pass.
Finite cases also pass independent CPU ordering and F64 softmax checks.

Raw failure `target/profiles/qwen4exp-topk-compat-87044/`; passing packet
`target/profiles/qwen4exp-topk-compat-90515/`. No indirect-dispatch implementation
was needed. Do not substitute better-defined mathematical NaN semantics for the
incumbent behavior, or claim exhaustive nonfinite equivalence from these fixtures.

## Actual Product Qualification

Production GPU lease, real wired-memory gate and Metal API validation throughout.
Strict parser/build checks pass. Model-free custody PASS0.08s: independent sessions,
default off, idempotence, unchanged allocation/plan, root/pending/poison/late outer
child refusal without partial mutation, reset/checkpoint preservation.

Product01 PASS39.69s,212 native forwards, one2179-token prefix:
- Empty scalar4 A/restored-A/B, QSA off/HC off.
- Split-QSA-on32 A/restored-A/B, HC off.
- Split-QSA-off32 A/B, HC off.
- HC-on4 A/B with split-QSA on.
- Warm four-forward ABBA and one measured four-forward ABBA, QSA on/HC off.

All full logit/hyper rows and each composition's terminal121-state snapshot match
bitwise. Existing census witnesses guarded48/token, expected QSA and HC composition
routes, and zero guarded dispatches in packed prefix. Raw observations persist before
numerical gates: `target/profiles/qwen4exp-product-topk-94868/`.

| Axis /four forwards | A1 ms | B1 ms | B2 ms | A2 ms | Mean saved | A spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU | 186.734749 | 108.798041 | 108.814875 | 186.803625 | 41.7428% | 0.0369% |
| Executor wall | 209.369958 | 131.798001 | 131.857375 | 209.485209 | 37.0533% | 0.0550% |

Both mean and pairwise10% GPU/5% wall floors pass, controls<=5%; no timing retry.
This is incremental continuation executor evidence with QSA already enabled, not
request throughput, prefill, cold-cache performance or an additive QSA percentage.

## CLI Delivery PASS

Rebuilt production CLI, guarded top-k=1, split-QSA=1, HC=0, API validation enabled:
2578 prompt tokens,23 output tokens,22 transitions, EOS, `status: ok` and exact
`FINAL_JSON: {"code":"amber-lattice-2049","record":"K-17"}`.

Artifacts `target/profiles/2026-09-16-qwen4exp-product-topk-cli.*`. Reported decode
29.23 token/s and prefill411.85 token/s are unpaired observations, not performance
comparison authority. The CLI logs all three option states and rollback flags.

## Limits And Next Leverage

Default-off experimental delivery only. Geometry fallback is directly checked as
an eligibility predicate; packed N1 and inner-owner checks are source-reviewed, not
separately exercised by the product packet. Negative host tests exercise alias,
dtype and misalignment rejection; no cross-device hardware proof is claimed.

Re-establish the remaining complete-MoE budget with guarded routing ON, QSA ON,
HC OFF before choosing the next expert-body mechanism. The old serial-router
budget is obsolete. Reuse saved inputs where sufficient; add IQ3/IQ4 dtype evidence
only where it can change that decision. Keep HC/attention/RMS body sweeps parked;
HC's independent performance promotion remains HOLD. No new quants downloaded.
Server remains stopped; nothing pushed remotely.

Follow-up: `MOE-GUARDED-BUDGET.md` records the completed three-dtype observation.
Common IQ3 paths show distributed remaining costs; layer2 timing is INCONCLUSIVE.
The next rechart is a bounded whole-forward parent ledger, not a speculative
expert rewrite or an HC timing retry.

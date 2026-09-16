# HC Product: Numerical PASS, Performance HOLD

`Qwen4ExpDecodeOptions.hc_up_mix` and strict CLI `QWEN4EXP_HC_UP_MIX=0/1`
provide a default-off experimental route. Unset/0 retain incumbent; malformed
values fail. Eligible singleton Q8 up /four branches /hidden2560 /rank320 only.
Independent of split-QSA. No additional GPU allocation or memory-plan change;
unsupported geometry/dtype falls back. Packed bodies and singleton packed tails
retain incumbent math. Atomic all-child binding, owner checks and preflight
precede mutation; reset/checkpoint restore preserve session configuration.

## Qualification

Production lease, real wired-memory check and API validation throughout. Strict
parser/build, guarded routing and binding-custody tests pass. Actual-product02
passes4 empty-state scalar steps plus32 continuation steps each with split-QSA
on and off; all121 persistent tensors pass unchanged gates. All argmax values
agree; old F16 prefixes immutable, unused suffixes equal. Raw rows retained in
`target/profiles/qwen4exp-product-hc-93173/`. No research TLS override.

| Packet | Maximum logit absolute error | Maximum relative RMS | Minimum cosine |
|---|---:|---:|---:|
| Empty scalar4 | 5.732179e-4 | 3.635205e-5 | 0.999999999340 |
| Split-on32 | 7.624626e-4 | 3.909643e-5 | 0.999999999240 |
| Split-off32 | 6.787777e-4 | 3.961146e-5 | 0.999999999217 |

Candidate census97 HC reads/token; packed2179 has zero candidate HC dispatches.
Attempt01 correctly failed that packed census after passing empty scalar checks:
packed final tails reused the singleton route. Explicit encode-time tail policy
repairs this without mutating live configuration or weakening gates.

## Frozen Performance Decision: HOLD

Split-QSA on in both arms, fixed four-forward ABBA:

| Axis | A1 ms | B1 ms | B2 ms | A2 ms | Mean saved |
|---|---:|---:|---:|---:|---:|
| GPU | 193.388334 | 184.080709 | 184.665500 | 195.212374 | 5.1092% |
| Executor wall | 215.195000 | 205.675959 | 206.282376 | 216.934250 | 4.6678% |

GPU first pair saves4.814%, below the frozen5% floor, despite passing mean and
second pair. A spreads0.9388% GPU/0.8050% wall; wall mean/both pairs clear3%.
Do not promote performance, recommend enabling for speed, substitute the earlier
research PASS, change floors, or rerun timing for a preferred outcome. Rust test
success indicates numerical/contract assertions, not performance promotion.

## Actual CLI PASS After Isolated Repair

Attempt01 aborts before generation, empty stdout/stats. Its2578 tokens schedule
2048/3/527, reaching the previously excluded M128 IQ4_NL down variant. Isolated
511/512 boundary and original populated capture qualify the minimal builtin-width
repair; see `M128-VALIDATION.md`.

CLI02 with both options and API validation completes with `status: ok`,2578 prompt
tokens,23 generated tokens, EOS and exact answer:

`FINAL_JSON: {"code":"amber-lattice-2049","record":"K-17"}`

Artifacts `target/profiles/2026-09-15-qwen4exp-hc-product-cli-02.*`. Unpaired delivery
observation only: prefill5378.1ms/479.35tps, reported decode19.82tps; not a speedup
or cold-cache claim. Correctness/CLI qualification does not lift performance HOLD.

## Leverage Map

Update: `TOPK-RESULT.md` now establishes a separate native bitwise parallel-routing
gain. Guarded selector compatibility/delivery outranks further MoE dtype profiling;
HC's own promotion HOLD remains unchanged. The following records the prior queue.

1. Complete-MoE budget remains next: QSA on/HC off, layers2/4/5 covering artifact
   dtype combinations, whole router-to-accumulation path. Do not optimize a leaf
   before its parent cost is established.
2. Initial observation passes native capture/state checks but fails timestamp
   validity, so no stage ranking yet; see `MOE-RESULT.md`.
3. Keep HC, attention-body and RMS sweeps parked. No special quants downloaded.
   Server remains stopped; no remote push.

# HC Product Delivery Protocol

Frozen before actual-product GPU tests. Parent `32ebcf16`. Default-off typed
`Qwen4ExpDecodeOptions.hc_up_mix`, strict CLI `QWEN4EXP_HC_UP_MIX=0/1` retained
through packed/scalar retry. Independently selectable from split-QSA, contingent
on both compositions passing below. No global library environment switch.

Move the unchanged shader body into the product metallib. Scope remains singleton
Q8 up projection /four branches /hidden2560 /rank320; other shapes/dtypes fall
back after ordinary validation. Packed HC kernels unchanged; eligible scalar
prefill also uses this singleton option. Existing normalized/low/raw/mixed scratch
remains allocated: no new GPU buffer or memory-plan adjustment. One configured
bool per49 private HC scratches,97 calls/token. Packed-chunk singleton output
tails explicitly retain incumbent math, as do packed HC bodies. Preflight pipeline capability
before admission/graph encoding; validate root, every child, nested layer-zero
and final scratch ownership/device before any binding changes. Reset and test
checkpoint restoration preserve configuration independently of persistent state.

## Checks

- CPU strict-parser tests, default off and malformed/non-Unicode rejection.
- Model-free guarded tests: no added allocation/unchanged pricing, independent
  sessions, idempotence, pending root and late-child atomic refusal, reset,
  empty-checkpoint logits readiness, actual Q8 dispatch/F32 fallback and alias
  rejection before graph encoding. Unsupported geometry eligibility is checked
  directly. Do not claim unavailable cross-device hardware testing.
- Production lease plus real wired-memory check before Metal; API validation on,
  exact tests serial. One existing artifact load, no new weights.
- Extend test-only checkpoint to preserve logits readiness and genuine empty
  PLE history. Four empty-state scalar-prefix forwards A/restored-A/B with QSA
  off, actual product HC route. Drop these snapshots before the long packet.
- One current packed2179 SSH prefix; census confirms no HC singleton candidate
  dispatch in it.32 continuation forwards A/restored-A/B with QSA on, then32
  A/B with QSA off from the same checkpoint. No TLS research override. Save full
  raw logit matrices before numerical/census gates. Witness97 HC calls/token and
  the appropriate split versus incumbent QSA route in each candidate arm.
- Unchanged native gates: identical argmax, cosine>0.99999999, relative RMS<1e-4,
  maxabs<1e-3 for every full row; hyper/F32 state RMS<=3e-4/maxabs<=.01; new F16
  rows RMS<=1e-3/maxabs<=.03125, old prefixes immutable/unused suffixes equal.
  Parameterize state ranges by prefix position, including the empty-state packet.
  Drop long-arm state snapshots once checked to keep the CPU proof budget bounded.
- Only after all numerical gates: QSA-on fixed warm four-forward ABBA, then one
  measured four-forward ABBA; own-arm replay bitwise. GPU mean and both pairs
  >=5% saved, executor wall>=3%, A spread<=5%. No timing claims for QSA-off arm.
  Instability INCONCLUSIVE, budget miss HOLD; no threshold or timing-retry fishing.
- Actual production CLI known-answer request, configured options visible, with
  request stats. Unpaired delivery check, not another speedup/cold-cache claim.

Proof budget:204 native forwards, one packed prefix. No broad quality/default-on
claim. After this delivery decision, continue to bounded complete-MoE observation
before another body topology. Read-only preflight reviewer:
`01a0a13a-f2f8-7673-ace3-4fbfd25a3aef`.

Implementation attempt01 passes the empty-state scalar packet, then correctly
fails the frozen no-HC-in-packed-prefix census: the packed path reuses singleton
final HC once per chunk. Fix that routing leak with an explicit encode-time tail
policy, not a flag mutation or changed gate. No native timing from attempt01.
Source scheduler review also corrects the earlier documentation shorthand:
the2179 prefix spans2048/3/128 tokens across the dense-end2051 boundary, not
2048/131. Neither schedule uses512/527, so the earlier M128xN16 exclusion remains
valid. Preserve the failure log and unchanged numerical/performance thresholds.

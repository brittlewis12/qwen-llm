# Default Policy Disposition

Guarded singleton top-k and HC up-plus-mix now default ON in CLI and library on
exact Apple M4 Max with supported pipelines. `=0` independently rolls each back;
unset/`=1` allows qualified execution, never forces unsupported hardware. Packed
kernels remain unchanged. Neither improvement adds a GPU allocation.

Production split QSA is RETIRED, not left behind an off-by-default switch. The CLI
flag/API field, product scratch planning/allocation/bindings and execution route
are absent from non-test builds. Its shader lives in the research metallib; retained
host/binding experiments are `cfg(test)`. This does not remove packed selected QSA.

## Why The Decision Changed

Default-off was delivery staging, not a principled permanent home for qualified
improvements. Fresh read-only cx reviews challenged both hidden wins and premature
deletion. Sol recommended a finite HC timebox on the materially faster router
baseline and a dissimilar QSA guardrail; Luna emphasized exact device/capability
scope, logical rather than rounded physical horizon, and explicit reference arms.
We rejected separate slow library defaults and force-unqualified semantics.

The guardrail mattered. Original online split QSA fails known-answer first-row
maxabs0.0012910366 against the frozen<0.001 gate. Argmax, cosine and relative RMS
pass; these do not override the failed maximum. One source-led repair preserves
incumbent QK/global-softmax order and splits only PV. All12 primitive/F64 cases
pass, but its intended composition fails HC-off/on at natural step23, maxabs
0.0010004044. Isolated repaired-QSA comparisons were not reached: its isolated
model accuracy is UNKNOWN, not demonstrated inaccurate.

No threshold relaxation, split-size sweep or repair performance run followed.
Retirement gives up the old online-QSA16% measured opportunity to honor the frozen
compatibility contract and finite timebox. It is not a claim that parallel attention
cannot work. Implementation snapshot724af78b and outcome f13ff275 retain the attempt;
`PROTOCOL.md` preserves exact budgets, review disagreements and failures.

## Settled Qualification

All GPU work: production-exclusive lease, real wired-memory gate, serial tests,
Metal API validation. No special weights downloaded; existing UD-Q3_K_XL corpus.

- Final default closure144 forwards PASS18.53s: untouched constructor versus
  explicit all-on32 and top-k-off32 bitwise full logits/hyper/terminal121states;
  HC-off32 numerical; dissimilar known-answer8 HC-off greedy then same-input HC-on8
  numerical. Route census checks every arm; packed prefix excludes singleton
  candidate kernels. Default plan contains no split scratch.
- Final HC204-forward product packet PASS14.47s: empty4, natural32 restored controls,
  numerical/state/census gates, one warm and one measured four-forward ABBA.
  Guarded top-k on and incumbent QSA throughout; no HC source tuning.
- CPU independent default/capability/rollback matrix and strict CLI parsing pass.
  Unsupported-device behavior is pure-policy/source coverage, not cross-device
  hardware proof. Exact M4 Max support does not assert all Apple GPU profiles.
- Real CLI with all three optimization variables explicitly UNSET returns exact
  `FINAL_JSON: {"code":"amber-lattice-2049","record":"K-17"}`;
  statusok,2578prompt,23output,22transitions,EOS,guarded/HC effective=true.
  No QSA option is parsed or logged. Release build is warning-free.

| HC /four forwards | A1 ms | B1 ms | B2 ms | A2 ms | Mean saved | Control spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU |154.443500|144.441334|144.219417|154.466584|6.5551%|0.01495%|
| Executor wall |176.582374|166.795667|166.821334|177.321624|5.7323%|0.41777%|

Frozen5% GPU/3% wall floors pass for mean AND both pairs; controls<=5%. This is an
incremental HC continuation result, not joint top-k+HC, prefill or request speedup.
The earlier top-k41.7428% GPU result held now-retired QSA constant; do not add its
percentage to HC or relabel it as this baseline's measured joint gain. CLI24.15
reported decode TPS is an unpaired observation, not performance authority.

HC's original serial-router HOLD stays HOLD. The intermediate guarded+online-QSA
timebox passed8.4653% GPU/7.7876% wall, but only the final QSA-off bracket qualifies
the current composition. The baseline changes are explicit, finite and recorded.

## Evidence And Limits

Raw logs under `target/profiles/`:
- `2026-09-16-hc-guarded-disposition.log`: intermediate composition PASS.
- `2026-09-16-defaults-closure.log`: original QSA guardrail FAIL. Dissimilar raw rows
  were not saved due assertion-before-persistence; this limitation is retained.
- `2026-09-16-qsa-repair-primitives.log`:12 primitive cases PASS0.21s.
- `2026-09-16-defaults-closure-repair.log`: composed repair FAIL11.24s; full raw32-row
  arms/state saved in `qwen4exp-defaults-15813/` before numerical gates.
- `2026-09-16-defaults-final-closure.log`: compile-only mutability error; no GPU run.
- `2026-09-16-defaults-final-closure-02.log`: final144-forward closure PASS; raw
  logits/state and known-answer continuation in `qwen4exp-defaults-18612/`.
- `2026-09-16-hc-final-disposition.log`: final204-forward HC PASS; raw directory
  `qwen4exp-product-hc-guardedtrue-splitfalse-19838/`. Its historical `split-on-*`
  filenames name the harness primary arm; actual split=false in every arm.
- `2026-09-16-defaults-final-cli.*`: actual production known-answer delivery.

Historical experiments should be interpreted with their source revisions, not
rerun on changed research kernels and treated as equivalent measurements. This
bounded greedy/numerical evidence is not universal quality or sampled equivalence.

## Next Leverage

Final cx source audit finds no production QSA leak or runtime blocker. Its main
correction is keeping README and active roadmap aligned with the actual defaults.
Next: one bounded whole-forward parent ledger on guarded+HC ON/incumbent QSA.
Count complete GDN-containing/QSA-containing blocks, bootstrap and tail, then inspect
the largest parent's source. Prior guarded+split-QSA/HC-off costs cannot rank current
parents. No HC retuning, QSA revival, broad expert retile or new quant prerequisite.
Server remains stopped; no remote push.

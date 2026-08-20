# P0 baseline — results (Qwen 3.8)

## Corrections after adversarial review (cx k3 session `ses_fe958fc5affe`)

Three interpretations in the original draft below are wrong. Corrections here
supersede; the original text is retained for the audit trail.

1. **The v0.556 anchor was an oracle probe, not a native-MTP measurement.**
   Verified in-tree at `target/profiles/v0555-n8-oracle-final/27b-run1.json`:
   `probe=Oracle`, `mtp_calls=0`, `acceptance_rate=1.0`, `emitted_per_step=8.0`,
   `phase_ms.draft=0.0`. The 2.60× decode-only was priced with a *perfect*
   proposer to establish the verifier denominator (`PERF-LOG.md:8876-8881`:
   "one N8 packet costs about 3.047 serial transitions"). There is no
   native-MTP α to regress from. The nearest actual native-MTP narrative
   anchor is v0.510 at `PERF-LOG.md:10431-10435` (α=0.429 on a 28-token prompt
   with `single_cb_draft=true`, not p02-comparable).
   Cheap follow-up if needed: rerun p02 D7 with `--mtp-probe oracle` on HEAD.
   Prediction: ~16 packets, ~8 tok/pkt, ~2.5× dec× reproduces since verify
   is unchanged.

2. **The p01 prompt-construction fix has a named attribution in the perflog.**
   `PERF-LOG.md:210-215` (2026-08-15 "Qwen3.8 Packed D1 Reframe", commit
   `026de74` "repair packed D1 verification"): "Packed base prefill captures
   every final residual row in one target pass, then executes only the
   shifted MTP KV bridge calls required by official post/post semantics …
   speculative prefill moves `992.0 → 369.9 ms`; decode is unchanged; total
   moves from `0.868×` rollback to `1.061×` candidate." The arithmetic
   matches: v0.587's 80.5s / 1926 tok = 41.8 ms/tok ≈ serial per-token cost
   (38.6 ms) — the old path ran the prompt through serial target forwards
   one token at a time to capture post-norm hidden. HEAD's spec_prefill
   17.49s ≈ packed prefill 17.15s + ~340 ms bridge. My original "no perflog
   entry announces this" claim was a keyword-search miss.

3. **The 3.8 vs 3.6-MTP wall delta cannot be bytes.** `PERF-LOG.md:257-259`:
   "The 64-layer text backbone has exactly the same 16,806,250,496 base
   source bytes and 851 base requests as Qwen3.6-27B. Qwen3.8 adds one
   detached 289,527,808-byte MTP head." The +270 MB in the load ledger *is*
   the MTP head. Serial reference never reads it, so it cannot explain a
   6-8% serial regression. Live hypotheses: (a) **thermal / process order
   drift** — `PERF-LOG.md:216-219` records a 19.95s → 25.95s baseline swing
   in consecutive processes on this machine class, larger than the entire
   cross-model delta I measured; my 3.6 and 3.8 sweeps ran back-to-back,
   confounding order; (b) identity-keyed code paths (`a4f7564` embedding
   retention, `1ac9446` prompt contract). Falsification: interleave
   models A/B/A/B in one session before any bisect. Until that survives,
   treat every 5-20% cross-model number in the compare table as unmeasured.

Additional reframes:

- The "89 ms unexplained" verify-decomposition frame was wrong. Serial decode
  is already ~39 ms/tok on p01 — within ~10% of the 35.5 ms weight-pass
  floor. v0.556's `8*C1/C8=2.626 → 3.047 serial-transitions per N8 packet`
  → 3.047 × 39 ms ≈ 119 ms ≈ observed 111 ms (3.6) / 125 ms (3.8). Verify
  cost structure has not changed since 2026-07-10. The real quantity to
  attribute is the **marginal per-position term**: (124.6 − 39)/7 ≈
  **12.2 ms per additional position** at N=8. Decisive cheap experiment: a
  verify N-sweep (N=1/2/4/8 on the same packet stream) splits fixed vs
  linear terms empirically.

- p01's `resume_audit_pass` failure on both models is a real ship risk. Greedy
  output is byte-exact but internal KV drift is unbounded over long sessions.
  Either fix `bench.rs:10652-10656` threshold justification or bound the
  drift before calling p01 D7 "shippable."

## Corrected program ranking (supersedes the tail of the original)

Adversarial review's ranked next actions:

1. **PLD probe on p01 + p02** (`qwen-bench pld`, v0.558 gates at
   `PERF-LOG.md:8846-8849`). Cheapest possible experiment. `qwen-bench pld`
   is in-tree with `MATCH=8, DRAFT=7` aligning with physical-N8. p01 is a
   code-refactor prompt — output copies prompt chunks with edits, the
   exact-copy/periodic regime where v0.558 measured high ceilings. PLD has
   ~zero draft cost, so even moderate acceptance beats MTP's 22.5 ms/pkt
   draft tax. Prediction: PLD ≥ MTP on p01, abstains on p02.
2. **P3 as N-sweep + lm_head/draft instrumentation**, not a ctx sweep. The
   quantity to explain is the 12.2 ms/position marginal term. Two suspects
   to instrument: (a) whether lm_head is batched over N=8 or per-position
   (output.weight Q6_K ≈ 640 MB / 474 GB/s ≈ 1.4 ms/pass; per-position
   lm_head alone ≈ 10 ms/pkt); (b) GDN scan per-position serialization
   (read the chunked-GDN KILL at commit `aa0c478` first).
3. **Interleaved A/B/A/B cross-model measurement** for the 3.8 wall delta
   (thermal control). Until this survives, the 5-20% delta is unmeasured.
4. **One prefill measurement vs MLX** at pp512/pp2048. If MLX has a 2×+ gap,
   the DeepSeek packed-prefill program (grouped dense attention, Q8 Q-B,
   online attention, BM16/32, 4096-token packed prefill; commits `6095d9a`
   through `e35f12d` era) is not ported to dense Qwen and would outrank P4
   entirely. p01 wall is 59-63% prefill; MTP adds nothing there.
5. **N=8 verifier kernel shape audit for Q4_K_M.** Precedent:
   `PERF-LOG.md:163-166` Ridge N2 32-column-tile fix moved 5240.6 → 2255.4
   ms (2.324×). Nobody has screened Q4_K N=8 packet kernels for the same
   class of shape cliff.
6. **P4 (one-launch T-scan) only if P3 attributes the 12.2 ms/position term
   to dispatch overhead.** No blind commit.

Also on the table but not immediate:

- **Calibrated sidecar draft head** (`PERF-LOG.md:8265-8270` MTPLX
  affine-INT4 g32 clue). Highest ceiling on α (attacks acceptance itself,
  not cost). Highest cost (training pipeline).
- **Boundary-fused chain / memoized QK-norm** (Layr-Labs leaderboard-style):
  benefits serial and spec equally → moves absolute wall + MLX gap but not
  tot×. Weight by whether the goal is "beat MLX absolute" (yes) or "prove
  spec ratio" (cancels).

---

## Original results (as-published; superseded above where corrected)



Model: `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf` SHA
`7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`.
HEAD: `98aba93` (source-state
`git-source-sha256-v2:a138b3ac585c06c6192453010c17b80ffceeeb8ab1a213483b8f1c0f5a49da52`).
Preregistration: `README.md`. Runner: `run.sh` (polite lease, resumable). 3
fresh processes per cell, warmup on, sampler = greedy argmax,
`mtp_history=committed`, `hidden=post/post`.

## Per-cell medians (n=3)

| Cell | α | e/s | pre× | dec× | tot× | v/pkt (ms) | chain/pkt (ms) | ref wall (s) | spec wall (s) | resume | min kv_cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| p01-D1 | 0.925 | 1.92 | 1.035 | 1.059 | 0.988 | 71.76 | 75.52 | 29.372 | 29.447 | ❌ | 0.999984915 |
| p01-D3 | 0.743 | 3.24 | 1.033 | 0.894 | 0.944 | 137.67 | 148.43 | 29.269 | 31.044 | ❌ | 0.999984911 |
| p01-D7 | 0.499 | 4.49 | 1.033 | 1.246 | **1.053** | 124.61 | 148.89 | 29.306 | **27.834** | ❌ | 0.999984905 |
| p02-D1 | 0.671 | 1.68 | 1.033 | 0.929 | 0.946 | 67.46 | 71.25 | 8.998 | 9.513 | ✅ | 0.999999871 |
| p02-D3 | 0.339 | 2.03 | 1.031 | 0.555 | 0.684 | 133.27 | 143.85 | 8.991 | 13.148 | ✅ | 0.999999869 |
| p02-D7 | 0.165 | 2.17 | 1.033 | 0.605 | 0.725 | 117.85 | 140.80 | 8.994 | 12.398 | ✅ | 0.999999862 |

**Only positive-tot× cell in the pool is p01 D7 at 1.053×.** Best-of-D per prompt:
p01 → D7 (1.053×), p02 → D1 (0.946×, still net-negative).

## Correctness roll-up

Preregistered gate (`identical=true` ∧ `continuation_argmax_equal=true`) passes
for all 18 invocations.

`resume_audit_pass` sub-gate:
- p01 all D configs FAIL: min `kv_payload_cosine` = 0.99998491 (vs strict
  threshold 0.99999 at `bench.rs:10652-10656`). Greedy stream byte-exact for
  all 256 tokens. Q8_0 `eh_proj` did **not** fix p01's audit failure — the
  drift comes from something else (deep-recursion state propagation, likely).
- p02 all D configs PASS: min `kv_cos` ~ 0.999999871. Consistent with the 3.6
  observation that the audit fails on long-code / deep-recursion path only.

## Cross-model delta (3.8 vs 3.6-MTP)

Same anchors, same paired A/B, same HEAD, same runner. 3.6-MTP results in
`../2026-08-18-qwen36-mtp-program/baseline.md`.

| Cell | α (3.6/3.8) | e/s (3.6/3.8) | dec× (3.6/3.8) | tot× (3.6/3.8) | spec wall s (3.6/3.8) | 3.8 wall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p01-D1 | 0.896 / 0.925 | 1.90 / 1.92 | 1.041 / 1.059 | 0.976 / 0.988 | 28.06 / 29.45 | +5.0% |
| p01-D3 | 0.761 / 0.743 | 3.28 / 3.24 | 0.955 / 0.894 | 0.965 / 0.944 | 28.65 / 31.04 | +8.4% |
| p01-D7 | 0.477 / 0.499 | 4.34 / 4.49 | 1.262 / 1.246 | 1.064 / 1.053 | 25.59 / 27.83 | +8.8% |
| p02-D1 | 0.764 / 0.671 | 1.78 / 1.68 | 0.995 / 0.929 | 0.985 / 0.946 |  8.65 /  9.51 | +10.0% |
| p02-D3 | 0.409 / 0.339 | 2.25 / 2.03 | 0.644 / 0.555 | 0.752 / 0.684 | 11.32 / 13.15 | +16.1% |
| p02-D7 | 0.213 / 0.165 | 2.51 / 2.17 | 0.745 / 0.605 | 0.827 / 0.725 | 10.29 / 12.40 | +20.4% |

Observations:

1. **3.8 is uniformly 5-20% slower than 3.6-MTP** at the wall on the same
   anchors, same paired mechanism. Reference (serial) decode is also 6-8%
   slower on 3.8 (ref wall p01: 27.20 → 29.31 s; p02: 8.51 → 9.00 s), so ~half
   the wall-delta comes from the serial side. 3.8's byte count on load is
   16.81 GB (270 MB more than 3.6-MTP's 16.54 GB) — some extra work per
   forward pass, not purely MTP-side.

2. **α on p01 is slightly HIGHER on 3.8** (D1: +3.2%, D7: +4.6%). The Q8_0
   `eh_proj` gives marginally better draft candidates on code content.

3. **α on p02 is uniformly LOWER on 3.8** (D1: −12%, D3: −17%, D7: −23%).
   The narrative anchor gets *worse* drafts on 3.8 despite the precision bump.
   Suggests draft quality is not bandwidth-of-eh_proj bound on narrative.

4. **The p02 α "regression" is not a bug in the current code** — it's the
   observed steady-state behavior of native MTP on narrative content across
   both current model releases. v0.556's implied α ≈ 1.0-effective on this
   prompt looks like an anomaly / measurement artifact / earlier-code state,
   not a target reproducible with Qwen team's shipped MTP models. That
   conclusion changes the P0 program plan (see below).

## Delta vs perflog anchors (context, not comparable)

The v0.587 and v0.556 anchors are **3.6-MTP** on old commits. Direct compare of
either model to those anchors is measuring cross-model AND cross-commit at
once. The apples-to-apples compare is 3.6-MTP HEAD (`../2026-08-18-qwen36-...`)
vs the perflog anchors, done in the sibling dir. Bringing forward the key
findings:

- **v0.587's 89s spec wall / 0.310× tot× on p01 is not reproduced** on 3.6-MTP
  HEAD (25.6 s / 1.064×) or 3.8 HEAD (27.8 s / 1.053×). The prompt-construction
  disaster it recorded was fixed sometime between b818986 (2026-07-13) and
  98aba93 (2026-08-17).
- **v0.556's 2.60× dec× on p02 is not reproduced** on either model at HEAD
  (3.6-MTP: 0.745×; 3.8: 0.605×). See point 4 above — this is a real behavior
  gap, but the direction of causality is that v0.556 was anomalous, not that
  HEAD regressed.

## Chain-inclusive per-packet decomposition (p01 D7, the productive cell)

Median across 3 reps:

```
verify   : 124.6 ms/pkt (83.7% of chain-inclusive)
draft    :  22.5 ms/pkt (15.1%)
bridge   :   1.4 ms/pkt ( 1.0%)
restore  :   0.4 ms/pkt ( 0.3%)
other    :   0.0 ms/pkt
─────────
chain    : 148.9 ms/pkt (3.8, vs 135.9 on 3.6-MTP)
```

Weight-pass floor at 474 GB/s measured M4 Max stream: 16.81 GB / 474 GB/s =
**35.5 ms/pkt**. Observed verify 124.6 ms/pkt → **~89 ms unexplained
per-packet** on 3.8 (vs ~77 ms on 3.6-MTP). This is the P4 kill-criterion
target.

## Preregistered gates — 3.8 (informational, formalize on next milestone)

3.8 baseline has never been measured before. Gate proposals below are anchored
on today's numbers; formalization moves to a follow-up preregistration doc
when P1/P2/P4 candidates are picked.

Proposed post-`X` targets (X = next candidate optimization, to be named):

| Anchor | Metric | Baseline | Proposed target |
| --- | --- | ---: | ---: |
| p01 | charged-total D7 | 1.053× | ≥ 1.15× |
| p01 | absolute wall D7 | 27.83 s | ≤ 24 s |
| p01 | verify ms/pkt | 124.6 | ≤ 100 |
| p02 | best-of-D tot× | 0.946× (D1) | ≥ 1.05× |
| all cells | correctness gate | 18/18 PASS | must remain 18/18 |
| serial reference | median regression | — | ≤ 3% |

Anti-guardrail (MTLResidencySet exclusion per `PERF-LOG.md:322-350`) stands.

## Program correction

Given cross-model baseline:

1. **The p02 α "regression" is real behavior, not a bug.** Both 3.6-MTP and 3.8
   at HEAD show low narrative α. v0.556's 2.60× was anomalous. This closes
   the "bisect the α regression" line of investigation I proposed in the
   3.6-MTP baseline.md. Save the cycles.

2. **p01 D7 is the shippable cell** on both models — the only positive one.
   3.8's 1.053× and 27.83s wall define the actual product baseline. The
   old-session-context targets (1.11×, 1.21×, 1.37×) can be transplanted only
   after we understand whether 3.8's ~9% wall regression vs 3.6-MTP is
   itself worth attacking as a pre-P1 baseline fix.

3. **P3 round-cost decomposition on 3.8** is the next real deliverable. The
   89 ms/pkt unexplained (124.6 measured − 35.5 weight-pass floor) needs to
   be attributed: dispatch overhead per D7 position, GDN scan launches, T-scan
   specialization on N=8, lm_head share. Anchored on 3.8, not 3.6-MTP.

4. **P2 (adaptive depth) is now a defensible workstream** — with the α story
   settled as "narrative is inherently low-α", the marginal-utility rule will
   correctly choose `depth=0` on p02, avoiding the negative-tot× cells today.
   For p01 D7 it stays at 7. The tuning target is the middle-α regime we
   don't have anchors for yet.

## Artifacts

- Preregistration: `README.md`
- Runner: `run.sh` (polite lease + resumable)
- Result JSONs: `baseline/pXX-DK-rN.json` (18 files)
- Stderr logs: `baseline/pXX-DK-rN.out` (18 files)
- Model: `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf`
- Anchor prompts: same as `../2026-08-18-qwen36-mtp-program/prompts/`
- Cross-model compare: `../2026-08-18-qwen36-mtp-program/baseline.md`

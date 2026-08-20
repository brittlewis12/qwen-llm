# P0 baseline — results

## Corrections after adversarial review (cx k3 session `ses_fe958fc5affe`)

Two interpretations below are materially wrong. Corrections here supersede;
the original text is retained for the audit trail.

1. **v0.556 was an oracle probe.** `target/profiles/v0555-n8-oracle-final/`
   `27b-run1.json`: `probe=Oracle mtp_calls=0 α=1.0 e/s=8.0 draft_ms=0.0`.
   The 2.60× dec× was priced with a *perfect* proposer to establish the N=8
   verifier denominator, not native MTP. My "p02 α regression bisect" line
   of investigation is closed for the correct reason (there was never a
   native-MTP p02 α to regress from), not the wrong one (a bug was
   introduced). The nearest actual native-MTP narrative anchor is v0.510
   (`PERF-LOG.md:10431-10435`) — not p02-comparable.

2. **The p01 D7 prompt-construction fix has a named attribution.**
   `PERF-LOG.md:210-215`, commit `026de74` "repair packed D1 verification"
   (2026-08-15): "Packed base prefill captures every final residual row in
   one target pass … speculative prefill moves `992.0 → 369.9 ms`; decode is
   unchanged; total moves from `0.868×` rollback to `1.061×` candidate."
   The arithmetic matches: v0.587's 80.5s / 1926 tok = 41.8 ms/tok ≈ serial
   per-token cost — the old path forwarded the prompt serially to capture
   post-norm hidden. The "mysterious fix" framing below is a keyword-search
   miss on my part.

Also worth flagging: `resume_audit_pass` failing on p01 for both models
(`kv_cos ≈ 0.99998` vs strict 0.99999 threshold at `bench.rs:10652-10656`)
is a real ship-side risk. Greedy stream byte-exact, but internal KV drift is
unbounded over long sessions. Either justify the threshold or bound the drift.

---

## Original results (as-published; superseded above where corrected)



Model: `/Users/tito/models/Qwen3.6-27B-MTP-Q4_K_M.gguf` SHA
`8d8bb840c0d6422f05b72126ec22bb8f06ec4a465c3537a564ffe74ac1feb7be`.
HEAD: `98aba93` (source-state
`git-source-sha256-v2:a138b3ac585c06c6192453010c17b80ffceeeb8ab1a213483b8f1c0f5a49da52`).
Preregistration: `README.md`. Runner: `run.sh`. 3 fresh processes per cell,
warmup on, sampler = greedy argmax, `mtp_history=committed`, `hidden=post/post`.

## Per-cell medians (n=3 across fresh processes)

| Cell | ptok | gen | α | e/s | pre× | dec× | tot× | v/pkt (ms) | chain/pkt (ms) | ref wall (s) | spec wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p01-D1 | 1926 | 256 | 0.896 | 1.90 | 1.063 | 1.041 | 0.976 | 69.83 | 73.49 | 27.443 | 28.057 |
| p01-D3 | 1926 | 256 | 0.761 | 3.28 | 1.030 | 0.955 | 0.965 | 129.70 | 140.41 | 27.663 | 28.646 |
| p01-D7 | 1926 | 256 | 0.477 | 4.34 | **1.027** | **1.262** | **1.064** | 112.20 | 135.88 | 27.197 | **25.591** |
| p02-D1 |  418 | 128 | 0.764 | 1.78 | 1.028 | 0.995 | 0.985 | 64.76 | 68.43 | 8.514 | 8.645 |
| p02-D3 |  418 | 128 | 0.409 | 2.25 | 1.027 | 0.644 | 0.752 | 123.11 | 133.50 | 8.513 | 11.324 |
| p02-D7 |  418 | 128 | 0.213 | 2.51 | 1.028 | 0.745 | 0.827 | 106.25 | 128.96 | 8.514 | 10.294 |

Rep-to-rep noise is negligible: p01 D7 spec wall across 3 reps is 25.462/25.591/25.727
s (σ ≈ 0.13 s, 0.5%); p02 all cells σ ≤ 6 ms.

## Correctness roll-up

| Cell | `identical` | `continuation_argmax_equal` | `resume_audit_pass` | min `kv_payload_cosine` |
| --- | :---: | :---: | :---: | ---: |
| p01-D1 | ✅ | ✅ | ❌ | 0.999989032 |
| p01-D3 | ✅ | ✅ | ❌ | 0.999989032 |
| p01-D7 | ✅ | ✅ | ❌ | 0.999989029 |
| p02-D1 | ✅ | ✅ | ✅ | 0.999999898 |
| p02-D3 | ✅ | ✅ | ✅ | 0.999999898 |
| p02-D7 | ✅ | ✅ | ✅ | 0.999999890 |

**Preregistered correctness gate PASSES for all 18 invocations** (`identical=true`
∧ `continuation_argmax_equal=true` — greedy-semantic + numerical-state
equivalence per v0.587's `PERF-LOG.md:8237-8238` policy).

**`resume_audit_pass` fails on p01** (all three D configs, all three reps). Root
cause: `bench.rs:10652-10656` requires `kv_payload_cosine ≥ 0.99999`; we
measured 0.99998903 (misses the 5th decimal). This is one order of magnitude
looser than v0.587's D7 measurement (`0.99999999`). The greedy stream is
byte-exact across all 256 emitted tokens; the divergence is internal
KV-numerical drift only.

## Delta vs perflog anchors — the story

### p01 vs v0.587 anchor (`PERF-LOG.md:8258-8264`)

| Metric | v0.587 (2026-07-13, b818986) | HEAD (2026-08-18, 98aba93) | Δ |
| --- | ---: | ---: | ---: |
| ref decode (s) | 10.41 | 10.05 | −3.5% |
| ref total wall (s) | 27.55 | 27.20 | −1.3% |
| spec decode D7 (s) | 8.41 | 7.96 | −5.3% |
| spec prefill D7 (s) | 80.57 | 17.49 | **−78.3%** |
| spec total wall D7 (s) | 88.98 | **25.59** | **−71.2%** |
| pre overhead (spec/ref) | 4.70× | **1.027×** | fix |
| tot× (charged) | 0.310× | **1.064×** | +0.754 abs |
| dec× | 1.240× | 1.262× | +0.022 abs |
| α (aggregate) | 0.477 | 0.477 | identical |
| packets | 59 | 59 | identical |
| accepted/drafts | 197/413 | 197/413 | identical |
| e/s | 4.322 | 4.339 | +0.4% |
| verify ms/pkt | 116.5 | 111.3 | −4.5% |

The 80s MTP prompt-construction disaster from v0.587 is **gone on current HEAD**.
Verify semantics are identical (α, packets, accepted, e/s all match to 3 s.f.).
The change is entirely upstream of decode — MTP-side prefill/prompt propagation.

I could not locate a single perflog entry between v0.587 and HEAD that
announces this fix under an obvious keyword (`MTP prompt construction`, `lazy
MTP prefill`, `batched flush`, etc. — none match). The change must have landed
as a side-effect of another line of work between 2026-07-13 and 2026-08-17.

### p02 vs v0.556 anchor (`PERF-LOG.md:8867-8877`)

| Metric | v0.556 (2026-07-10) | HEAD (2026-08-18) | Δ |
| --- | ---: | ---: | ---: |
| ref decode (ms) | 4904 | 4898 | −0.1% (identical) |
| spec decode D7 (ms) | 1884 | **6577** | **+3.49×** (regression) |
| dec× D7 | 2.602 | **0.745** | regression |
| verify ms/pkt | 117.6 | 106.3 | −9.6% (improved) |
| packets (implied v0.556 / measured HEAD) | ~16 | 51 | 3.2× more |
| e/s (implied / measured) | ~8 (near-perfect) | 2.51 | −69% |
| α (implied / measured) | ~1.0-effective | 0.213 | regression |
| tot× D7 | (~1.9× implied) | 0.827× | regression |

**p02 D7 α collapsed from ~1.0-effective to 0.213** — a real acceptance-rate
regression. The greedy stream is still exact (`identical=true`); the difference
is in *which drafts get generated* to compare against that stream.

Verify per-packet actually *improved* by 10% (117.6 → 106.3 ms). Root cause is
purely the draft side — either draft-head quality or draft-position hidden
propagation changed between v0.556 (2026-07-10) and HEAD.

Sanity: v0.556's implied α ≈ 1.0-effective (needed to hit 2.60× dec ratio with
~8 tok/packet emit) is anomalous for narrative content. It may reflect a
prompt-specific completion path that was uniquely predictable under the
then-current draft head. HEAD's α=0.213 is more consistent with typical
narrative α — but the regression is nonetheless real vs the anchor.

## Preregistered gate check

Preregistered post-P1 targets (from `README.md`):

| Anchor | Metric | Target | HEAD | Status |
| --- | --- | ---: | ---: | :---: |
| p01 | charged-total D7 | ≥ 1.11× | 1.064× | **miss by 4.3%** |
| p01 | absolute wall total D7 | ≤ 20 s | 25.59 s | **miss by 28%** |
| p02 | decode-only D7 | ≥ 1.60× | 0.745× | **miss by 53%** |
| p02 | absolute wall spec-decode D7 | ≤ 1.75 s | 6.58 s | **miss by 275%** |

Correctness gate: **PASS** (preregistered definition).
`resume_audit_pass` sub-gate: informational only; noted p01 miss.

## Reading

The session's premise — "P1 = lazy MTP prefill batched flush is the primary
lever" — is stale. The p01 prefill fix already landed. Current HEAD is **at**
the post-P1 target on p01 charged-total to within 4% (1.064 vs 1.11), and **at**
the P1 wall target to within 28% (25.6s vs 20s). The remaining gap on p01 is
entirely on the decode side: verify per-packet 111 ms × 59 packets = 6.57 s
dominates the 8.0 s spec decode.

**The story is now not P1. It is:**

1. **p02 acceptance-rate regression is a real bug**, not a design target. v0.556
   hit α≈1.0-effective on this prompt; HEAD gets 0.213 with the same greedy
   stream. Root cause must be found before any adaptive-depth (P2) work — P2
   optimizes over a broken denominator.
2. **p02 negative-tot× across every D config** means MTP is a net loss on
   narrative on current HEAD. Best cfg is D1 at tot× 0.985 (still 1.5% slower
   than serial). P2's marginal-utility rule would correctly select `depth=0`
   (i.e. no drafting) on p02 today; the observation isn't a P2 win, it's a
   drafting-quality failure.
3. **p01 D7 is the only positive cell** in the pool. Getting p01 D7 from
   1.064× to ≥1.37× (post-P1+P2+P4 target) requires attacking the 111 ms/pkt
   verify cost. Chain-inclusive breakdown at D7: verify 6568 ms + draft 1272 ms
   + bridge 84 ms + restore 40 ms = 7964 ms total spec decode. Verify is 82.5%
   of decode; draft is 16%.

## Chain-inclusive per-packet decomposition (p01 D7)

Median across 3 reps:

```
verify   : 111.3 ms/pkt (82.5% of decode)
draft    :  21.6 ms/pkt (16.0%)
bridge   :   1.4 ms/pkt ( 1.0%)
restore  :   0.7 ms/pkt ( 0.5%)
other    :   0.0 ms/pkt
─────────
chain    : 135.0 ms/pkt (of which 3.05 serial-equiv per v0.556 economics; here
                          e/s 4.34 → 135/4.34 = 31.1 ms per emitted token
                          vs serial 39.3 ms/tok on ref → dec× 1.26 checks out)
```

Weight-pass floor at 474 GB/s measured M4 Max stream (`PERF-LOG.md:17473-17476`):
16.74 GB / 474 GB/s = **35.3 ms** per full model pass. Verify at N=8 does one
weight pass per packet plus per-position work. Observed verify 111.3 ms/pkt =
35.3 ms weight-pass floor + **~76 ms** unexplained per-packet overhead. This is
the P4 target.

## Program correction — proposed next actions

Given the above, the session-context sequencing is stale. Proposed reorder:

1. **Investigate p02 acceptance-rate regression (blocking).** Bisect between
   v0.556 (2026-07-10) and HEAD (2026-08-18), or diff the draft-head/draft-KV
   propagation paths. Until this is understood, P2 tuning is invalid on p02.
   Cheap bisect targets: v0.587 (2026-07-13 known-broken on prefill, but α
   should still be v0.556-era on p02 if the prefill fix is orthogonal),
   v0.640 (2026-07-26 deferred-restore), v0.660 (2026-07-31 sampling GO).
2. **Publish `round-cost-decomposition.md` (P3).** Now that p01 D7 baseline is
   frozen at 111 ms verify/pkt, the ctx sweep and 76 ms/pkt unexplained
   decomposition become the primary product. Adaptive-depth math (P2) is data
   from P3.
3. **Defer P1 as an active workstream.** It's largely done. If P3 shows a
   remaining prefill wedge worth chasing, reopen; otherwise close.
4. **P4 (batched GDN front + one-launch T-scan) gates on P3's 76 ms
   decomposition.** If the 76 ms is dominated by kernel dispatch overhead per
   position (7 positions in a D7 packet), one-launch T-scan attacks it; if it's
   dominated by intra-position kernel time already, P4 is a no-op.

## Artifacts

- Preregistration: `README.md`
- Runner: `run.sh`
- Result JSONs: `baseline/pXX-DK-rN.json` (18 files)
- Stderr logs: `baseline/pXX-DK-rN.out` (18 files)
- Model: `/Users/tito/models/Qwen3.6-27B-MTP-Q4_K_M.gguf`
- Anchor prompt files: `prompts/p01-refactor-code.txt` (1926 tok, extracted from
  `target/profiles/v0587-mtp-history/code-full-a.out`);
  `prompts/p02-reva-narrative.txt` (418 tok, copied from
  `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`).

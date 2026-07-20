# W0: A3B packed-verify flag matrix (exploratory nested ablation, re-registered)

Status: PREREGISTERED v3 (v1 frozen -> REDESIGNED on adversarial review;
v2 -> v3 after the same reviewer found C2===C3 in code: the legacy mixer
branch calls `encode_moe_ffn_after_mixer_by_index` which does serial
route unconditionally (metal_forward.rs:6340-6354, verified), so a
"serial mixer + batched route" cell does not exist without new code.
All before any run; cx session of record
019f7fb5-9d3b-7ee1-808b-0107793d702a). No W0 run preceded this version.
Date: 2026-07-20
Worktree: ~/code/qwen-llm-wy @ branch wy/w1-chunked-gdn
Program: W1 (chunked/WY GDN), packet zero.

## What this packet can and cannot answer (scope, post-review)

W0 is an EXPLORATORY flag matrix, not a causal localization. It answers
exactly one program decision:

  Decision A: does a no-new-kernel, fixture-sufficient mitigation for the
  A3B packed-verify state-contract failure exist among the shipped
  serialization flags?

It does NOT establish mechanism (route flip vs continuous amplification),
does NOT establish product robustness, and produces NO timing evidence.
Mechanism causality (Decision B) is explicitly deferred to a W0b packet
(actual-packet route traces: pre-router hidden delta, 8th-9th expert
margins, route-set equality, first-divergence packet; and if needed a
serial-route injection oracle). All verdict language below is
"fixture-sufficient intervention", never "localized cause".

## Background

v0.556 killed the A3B physical-N8 packed verifier in the numerical/exact
lane (code fixture: KV cosine 0.999699 vs >= 0.99999 at 128 tokens; chat
fixture: stream divergence before the audit). Code audit (task
ses_0821dc07fffeWNEB0PeIdc76Xu) established the packed-vs-serial kernel
differences on A3B:

- Router logits: serial `kernel_mat_vec_f32` vs packed
  `kernel_mat_mat_f32_f32` — different fp32 reduction order feeding the
  discrete top-8-of-256 selection (selection kernel itself identical).
- Mixer projections (GDN qkv/z/out, attn q/k/v/o): batched mma8
  (half-staged dequant) vs serial mat-vec.
- lm_head: batched vs serial — affects verify-time argmax (hence accept/
  commit decisions and stream equality) but not recurrent/KV state.
- Identical in both paths: GDN recurrence tail (bit-exact given same
  inputs), top-k selection kernel, expert FFN application functions,
  RMS norms, RoPE/KV scatter/attention core, embedding gather.

Dense-27B passes the contract under the same batched scheme. Post-review
this is SUPPORTING CONTEXT ONLY: dense differs in width, depth, head
ratio, quant geometry, and learned decay spectra, so it does not bound
A3B's continuous-amplification behavior. H-route (near-tie expert flips
amplified through 30 recurrent layers) is the working hypothesis, NOT a
conclusion this packet can reach.

Known structural consequence used in predictions: under C3 (serial mixer
+ serial route), every arithmetic op feeding recurrent state and KV is
per-token identical to serial decode; the only remaining packed-vs-serial
arithmetic difference is the batched lm_head (argmax/stream risk only).

## Flags under test (read ONLY in metal_dflash.rs packed-verify encoder —
verified by grep; the serial reference side never consults them; OnceLock
caching requires one fresh process per run, which the protocol guarantees)

- C-ROUTE: `QWEN_MTP_MOE_VERIFY_BATCHED_ROUTE=0` — per-token serial route
  prep (`encode_moe_route_prepare_by_index`) inside packed verify.
- C-MIXER: `QWEN_MTP_MOE_VERIFY_BATCHED_MIXER=0` — legacy fully-serial
  per-token mixer inside packed verify.
- Contingent (layered on C3 only): `QWEN_MTP_MOE_VERIFY_ROW_VIEWS=0`,
  `QWEN_MTP_MOE_VERIFY_CONCURRENT_FFN=0`, `QWEN_MATMAT_SMALLN_TABLE=0`.

## Fixtures

Binary: `qwen-bench mtp` (release, this worktree). Model:
`/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
Common: `--spec-tokens 7 --mtp-probe oracle --mtp-physical-n 8` (stops
resolve from GGUF = [248046]; history Committed).

Historical (v0.556 comparability set — one greedy trajectory + prefix):
- F-code16: `-p "Write a Python function to compute the Fibonacci
  sequence iteratively." --tokens 16`. v0.556 FAIL: kv 0.999983264292694,
  cont-cos 0.9989122142976149, gdn 0.04716444, conv 0.38506365.
- F-code128: same, `--tokens 128`. v0.556 FAIL: kv 0.9996988637773118,
  cont-cos 0.9991970091305232, logit-max 0.4379127, gdn 0.06411895,
  conv 0.33678818.
- F-chat128: Reva prompt (v0.556 exact text), `--qwen-chat
  --disable-thinking --tokens 128`. v0.556 FAIL earlier: "oracle emitted
  tokens differ from serial target" (stream mode; audit unreached; the
  batched lm_head alone could cause this mode — see failure taxonomy).

Prospective generalization fixture (frozen NOW, before any C1-C3
observation; C0 behavior unknown at freeze time):
- F-gen32: `-p "Write a Rust function that reverses the words in a
  sentence while preserving whitespace runs." --tokens 32` (raw mode).

F-code16 and F-code128 share one trajectory; the independent trajectories
are {code, chat, gen} = three. Claims are bounded accordingly.

## Failure taxonomy (recorded per run; "nonzero exit" is not an outcome)

- MODE-STREAM: harness aborts "oracle emitted tokens differ from serial
  target" (target-argmax parity failure; under C-SER this classifies as
  HEAD-OR-ORCHESTRATION — head parity and packed bookkeeping are both
  unexonerated).
- MODE-AUDIT: exit via "oracle terminal resume audit failed" — record the
  full audit struct (kv cosine/max-abs, gdn/conv deltas, continuation
  metrics).
- MODE-PROCESS: any other failure (infra; rerun once, else invalid run).
- PASS-STRONG: audit passes with kv_payload_cosine >= 0.999999 AND
  continuation_logits_cosine >= 0.999999 (10x margin over the gate).
- PASS-MARGINAL: audit passes but under the STRONG margins. Threshold
  luck is presumed; a PASS-MARGINAL config cannot support Decision A
  without W0b mechanism evidence. (Guards the 0.99998-vs-0.99999
  marginality of F-code16.)
- Additionally recorded on every run: kv_payload_max_abs. Investigation
  threshold (not a gate): any PASS with kv_payload_max_abs > 0.05 is
  flagged for W0b (global-cosine dilution guard; the cosine pools the
  identical prompt prefix and all layers, so max-abs is the honest
  side-channel).

## Config ladder and run plan (frozen; NESTED ablation, not factorial)

| Config | Env | Meaning |
|---|---|---|
| C0 | (defaults) | batched mixer + batched route (baseline) |
| C1 | BATCHED_ROUTE=0 | batched mixer + serial per-token route |
| C-SER | BATCHED_MIXER=0 | legacy fully-serial branch: per-token mixer AND route AND FFN (route flag ignored there — C2===C3 collapse) |

The cells are nested (C0 superset-batched > C1 > C-SER), so outcomes
support ladder statements ("serialization up to X is fixture-sufficient"),
never component isolation between mixer and route beyond what nesting
gives.

Symmetric repetition: EVERY config runs twice on F-code16 (failures need
flake guards as much as passes). Pass/fail or audit-value disagreement
between a config's two runs => verdict NONDETERMINISTIC for the packet;
immediately rerun that config with CONCURRENT_FFN=0 x2; if stable,
concurrency hazard is implicated (finding, packet closes
NONDETERMINISTIC-CONCURRENCY); if still unstable, close NONDETERMINISTIC-
UNKNOWN. No majority voting in either case.

Promotion: every config whose F-code16 result is PASS (either grade) runs
TWICE each on F-code128, F-chat128, F-gen32 (any run used for Decision A
gets the same x2 guard — reviewer demand, accepted). C0 also runs once on
F-gen32 (prospective baseline) and one fresh confirmation run each on
F-code128/F-chat128 against the v0.556 record.

Contingency (only if C-SER shows MODE-AUDIT failure on F-code16): layer
contingent flags cumulatively on C-SER (+ROW_VIEWS=0, then
+CONCURRENT_FFN=0, then +SMALLN_TABLE=0), x2 each on F-code16. If C-SER
fails at stream level: classify HEAD-OR-ORCHESTRATION (serial per-token
arithmetic does NOT exonerate packed checkpoint/commit bookkeeping or the
batched lm_head); NOT grounds for the state-side contingency ladder;
record and close that arm.

Runtime discipline: exclusive GPU (Britt, 2026-07-20); fresh process per
run; sequential runs; `pmset -g therm` before/after the batch (context
only); wall time context-only; artifacts `<config>-<fixture>-runN.{out,err}`
in this directory plus a manifest.

## Manifest (recorded once, before runs)

Commit + dirty state of this worktree; sha256 of the GGUF; macOS +
hardware identifiers; the complete QWEN_* environment for each config
(explicit whitelist — no other QWEN_* set); rendered prompt token counts
as printed by the harness; stop-token IDs. STALE-DEFECT claims (C0
passing) additionally require manifest-verified fresh failures... i.e. a
STALE-DEFECT verdict requires C0 x2 PASS on F-code16 AND fresh C0 runs on
F-code128 + F-chat128 also passing, with the manifest checked against the
v0.556 record for explicable drift (model file, commit distance).

## Preregistered outcome table (nested ladder, fixture-sufficiency language)

On F-code16 (after x2 agreement):
- C0 PASS => STALE-DEFECT path (see manifest rule); no other claim.
- C1 PASS => route-side serialization is fixture-sufficient under the
  batched mixer. (Mechanism open: serial-router inputs remain
  packed-perturbed; W0b decides mechanism.)
- C1 FAIL, C-SER PASS => serialization beyond route is required;
  mixer-vs-interaction NOT decomposable in W0 (no batched-route/serial-
  mixer cell exists). Decision A = YES via C-SER (cost question open).
- C-SER FAIL (MODE-AUDIT) after contingency ladder => NOT-FIXTURE-
  SUFFICIENT: shipped flags cannot reproduce serial state on this
  fixture; the identical-kernel inventory or the orchestration
  (checkpoint blits, scratch reuse) is suspect; escalate to W0b traces.
  Decision A = NO (for now).
- C-SER FAIL (HEAD-OR-ORCHESTRATION) => per-token serial arithmetic did
  not restore stream equality; batched lm_head and/or packed verify
  bookkeeping implicated; Decision A = OPEN pending a head-parity probe
  (new code, out of W0 scope).
- Nonmonotonicity (C1 PASS while C-SER FAIL, mode-matched) =>
  INVALID-INTERVENTION finding: flags do not compose as nested;
  do not interpret further; escalate.

Decision A = YES additionally requires the passing config to be
PASS-STRONG on F-code16 AND pass (any grade) on F-code128, F-chat128,
and F-gen32. A config passing code fixtures but failing chat/gen is
recorded as PROMPT-DEPENDENT (near-tie density varies by trajectory);
Decision A = PARTIAL, W0b required.

## What no W0 outcome may claim (frozen limits)

- No causal mechanism claim (route flip vs continuous amplification) —
  W0b territory.
- No product-fix claim: "serial router projection is a fixture-sufficient
  mitigation under the current packed mixer; product robustness remains
  unestablished" is the STRONGEST permitted phrasing for the best case.
  Promotion to a product fix requires (per review): a broader frozen
  suite with route-set stability evidence, or serial-equivalent router
  inputs, or exact route replay — plus a verifier-economics packet.
- No W1 priority claim: W1a (derivation) and W1b (CPU numerical-contract
  oracle) proceed regardless — they are cheap and feed any chunked
  future including prefill. W1c (GPU kernel) investment is BLOCKED
  pending a post-W0 verifier-economics packet (phase shares of the
  corrected verifier, packed-N8 GDN ceiling, end-to-end projection) with
  its own predeclared threshold. (Review demand 16, accepted.)

## Known limitations deliberately carried (declared, not hidden)

- Terminal + 16-step-chain audit only; no per-packet boundary audits
  (W0b). A terminal pass does not prove every packet preserved state.
- Global KV cosine dilutes suffix errors on long fixtures; max-abs
  side-channel partially compensates; suffix-only/per-layer metrics are
  W0b.
- F-chat's information under C0/C1/C2 is stream-mode only (audit
  unreached); its state signal requires a config that first fixes the
  stream.
- No head-parity flag exists; MODE-STREAM outcomes on C3 cannot be
  further decomposed in W0.

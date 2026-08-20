# P0 MTP-program baseline (SUPERSEDED — see erratum)

Frozen: 2026-08-18. Fresh baseline on the two frozen perflog anchors before any
P1/P2/P4 change. Gates and pool composition are declared here first; results
land in `baseline.md`.

## ERRATUM (2026-08-18, post-run)

The paragraph below claims "There is no `Qwen3.8-27B-MTP-*` GGUF." **That is
wrong.** `Qwen3.8-27B-Q4_K_M.gguf` has MTP baked into the base weights:
`qwen35.nextn_predict_layers=1`, same `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`
tensor set as `Qwen3.6-27B-MTP-Q4_K_M`, with `eh_proj` promoted from Q4_K to
Q8_0 (~5.9× precision on the MTP encoder-hidden projection). Qwen 3.8 ships
MTP in the base GGUF rather than a separate `-MTP-` variant.

Consequence: this dir's baseline is on the wrong model for the shippable
program. The 3.6-MTP numbers are still useful as a direct compare against the
v0.587 / v0.556 anchors (which are also 3.6-MTP), but they are **not** the P0
gate anchor for shipping. The current P0 lives at
`docs/bench/2026-08-18-qwen38-mtp-program/`.

The rest of this document is preserved as-was for the audit trail.

## Model + provenance corrections

The MTP-aware GGUF on disk is `Qwen3.6-27B-MTP-Q4_K_M`. There is no
`Qwen3.8-27B-MTP-*` GGUF. Every native-MTP anchor in `docs/PERF-LOG.md`
(v0.587, v0.556, v0.474, v0.511) runs on 3.6-MTP; the 2026-08-17 wide-sweep
kernel-gap work runs on 3.8 non-MTP. **P0 measures native MTP on 3.6-MTP.**

The perflog entry cited elsewhere as "v0.555" is actually **v0.556**
(`PERF-LOG.md:8858`). The 418-token / 2.602× / 117.6 ms figures belong to v0.556.

Weight-pass floor for verify decomposition uses measured M4 Max stream
bandwidth **474 GB/s** (`PERF-LOG.md:17473-17476`), not the 546 GB/s spec-sheet
figure. At 16.74 GB weights, one full weight pass ≈ **35.3 ms**. Verify-only
measured 117.6 ms/packet → unexplained overhead ≈ **82 ms**, not 88 ms. P4's
kill criterion updates accordingly.

## Machine + build

- Host: M4 Max 40c, 128 GB, AC power.
- HEAD: `98aba93` (untracked bench dirs and two untracked docs; none touch
  MTP/decode source).
- Binary: `target/release/qwen-bench`.
- Model: `/Users/tito/models/Qwen3.6-27B-MTP-Q4_K_M.gguf`
  SHA-256 `8d8bb840c0d6422f05b72126ec22bb8f06ec4a465c3537a564ffe74ac1feb7be`.

## Serial reference

`qwen-bench mtp` emits an in-process paired A/B per invocation: `reference`
block (MTP=off greedy) + `speculative` block (MTP=on requested config), same
prompt, same process. This is the paired denominator/numerator. No external
`llama-bench` reference; same-model, same-graph, same-process, per-prompt.

## Pool

Two frozen perflog anchors. Extractable, reproducible, and independently
covered in the perflog — no synthetic ceremony:

- `prompts/p01-refactor-code.txt` — v0.587 anchor. "Return a complete
  behavior-preserving refactor..." + full body of `test_proposal_economics.py`
  at SHA `7a7adad7...`. 1926 tokens, gen 256, code (high α expected).
- `prompts/p02-reva-narrative.txt` — v0.556 anchor. Frozen Qwen chat prompt
  from `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`.
  418 tokens, gen 128, narrative/chat (medium α).

If P0 shows the anchors alone are insufficient signal (e.g. bimodal α, no
gradient), we expand from game rollouts in `~/code/llm/game/` — real production
workload.

## Configs

Three MTP configs per prompt, all with harness defaults (`--mtp-recursive-hidden
post-norm --mtp-base-hidden post-norm --mtp-history committed` — v0.587-canonical
and match the harness defaults, so passed implicitly):

| Row | `--spec-tokens` | `--mtp-physical-n` | Meaning |
| --- | ---: | ---: | --- |
| D1 | 1 | 2 | Packed D1/N2 lazy-verify |
| D3 | 3 | 4 | Packed D3/N4 recursive |
| D7 | 7 | 8 | Packed D7/N8 recursive (v0.587-canonical) |

Sampler: greedy argmax. Warmup: enabled (matches v0.587). Reps: **3 fresh
processes** per (prompt, config). Serial `reference` re-measured every process.

## Terminology (defined here; not in PERF-LOG.md under these names)

- **charged-total ratio**: `reference.total_ms / speculative.total_ms`
  (includes prefill on both sides).
- **decode-only ratio**: `reference.decode_ms / speculative.decode_ms`
  (excludes prefill).
- **chain-inclusive per-packet cost**: `sum(speculative.phase_ms.*) /
  speculative.steps`. Distinct from `verify_ms/steps` which excludes
  draft/restore/bridge.
- **equivalence PASS**: `identical=true` ∧
  `target_state.continuation_argmax_equal=true` (v0.556, v0.587 usage).

## Preregistered gates

### Correctness (hard, per-invocation)

`equivalence=PASS` on every (prompt, config, rep). Any single failure across
the 2 × 3 × 3 = 18 invocations blocks the milestone. No greedy-window tolerance.

### Performance — post-P1 (lazy MTP prefill, batched flush)

Fixed anchor targets:

| Anchor | Metric | Baseline (v0.587/v0.556 or fresh P0) | Target post-P1 |
| --- | --- | ---: | ---: |
| p01 | charged-total (D7) | ~0.31× | **≥ 1.11×** |
| p01 | absolute wall total (D7) | ~89 s | **≤ 20 s** |
| p02 | decode-only (D7) | ~2.60× chain-free | **≥ 1.60×** |
| p02 | absolute wall spec-decode (D7) | ~1.88 s | **≤ 1.75 s** |

### Performance — post-P1+P2+P4

| Anchor | Metric | Target |
| --- | --- | ---: |
| p01 | charged-total (D7) | **≥ 1.37×** |
| p02 | decode-only (D7) | **≥ 1.90×** |

### Serial-leg regression guard

Every milestone re-runs P0 and requires `reference.total_ms` median regression
**≤ 3%** vs baseline P0 across the pool. Paired ratios are meaningless if the
serial denominator drifted.

### Anti-guardrail: MTLResidencySet exclusion

No milestone enables whole-model MTLResidencySet or wired-residency mechanism,
per `PERF-LOG.md:322-350` closure. Any candidate implicating it exits P0/P3 and
enters a separate reopen review before measurement.

## Deliverables

- `README.md` — this preregistration.
- `prompts/pXX-*.txt` — pool.
- `run.sh` — deterministic invocation script.
- `baseline/pXX-DK-rN.{json,out}` — 18 result JSONs + stderr logs.
- `baseline.md` — post-run summary: per-prompt paired ratios, α, chain-inclusive
  per-packet cost, correctness roll-up, delta vs v0.587/v0.556 anchors.

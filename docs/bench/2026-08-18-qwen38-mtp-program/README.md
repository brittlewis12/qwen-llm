# P0 MTP-program baseline — Qwen 3.8

Frozen: 2026-08-18. Fresh baseline on Qwen 3.8-27B-Q4_K_M with MTP native.
Two frozen anchor prompts, same pool as `docs/bench/2026-08-18-qwen36-mtp-program/`
so cross-model deltas are directly readable.

## Why this dir exists (and 3.6-MTP sibling is superseded)

Earlier session context asserted "no `Qwen3.8-27B-MTP-*` GGUF exists" and P0
was landed on `Qwen3.6-27B-MTP-Q4_K_M`. That assertion was wrong:
`Qwen3.8-27B-Q4_K_M.gguf` has MTP baked into the base weights, verified via
`gguf-dump`:

- `general.architecture = 'qwen35'` (same loader path as 3.6)
- `qwen35.nextn_predict_layers = 1` (MTP layer count, same as 3.6-MTP)
- MTP tensors at `blk.64.nextn.{eh_proj, enorm, hnorm, shared_head_norm}`
- **Difference vs 3.6-MTP**: `eh_proj.weight` is **Q8_0** on 3.8 (52 MB @
  `10240×5120`) vs **Q4_K** on 3.6-MTP (same shape) — ~5.9× precision bump on
  the MTP encoder-hidden projection

Qwen 3.8 shipped in the last week; the release rolled MTP into the base GGUF
rather than a separate `-MTP-` variant. **This dir is the authoritative P0 for
the shippable program.** The 3.6-MTP dir is preserved because its numbers are
the direct compare against v0.587/v0.556 perflog anchors (also 3.6-MTP).

## Machine + build

- Host: M4 Max 40c, 128 GB, AC power.
- HEAD: `98aba93` (source-state
  `git-source-sha256-v2:a138b3ac585c06c6192453010c17b80ffceeeb8ab1a213483b8f1c0f5a49da52`).
  Untracked bench dirs + two untracked docs (`H6-VISION.md`, `dsv4-paper.md`);
  none touch MTP/decode source.
- Binary: `target/release/qwen-bench` (built at session start).
- **Model**: `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf`
  SHA-256 `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`,
  17.11 GB (16.81 GB in-memory MTP-inclusive load per `[metal-load-ledger]`).

## Serial reference

`qwen-bench mtp` emits an in-process paired A/B per invocation: `reference`
block (MTP=off greedy) + `speculative` block (MTP=on requested config), same
prompt, same process. This is the paired denominator/numerator. No external
`llama-bench` reference; same-model, same-graph, same-process, per-prompt.

## Pool

Same two frozen anchors as the 3.6-MTP sibling dir. Files copied byte-identical
from `../2026-08-18-qwen36-mtp-program/prompts/` so cross-model compare is
apples-to-apples:

- `prompts/p01-refactor-code.txt` — extracted from v0.587's frozen
  `target/profiles/v0587-mtp-history/code-full-a.out`. 1926 tokens, gen 256,
  code refactor (high-α workload).
- `prompts/p02-reva-narrative.txt` — v0.556's frozen
  `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`. 418
  tokens, gen 128, narrative/chat.

**Caveat on p02 anchor semantics**: the fixture filename encodes `qwen36`, and
v0.556 measured it on 3.6-MTP. Using it on 3.8 measures 3.8's behavior on the
same input, not a reproduction of v0.556's measurement. That's the intent.

## Configs

Three MTP configs per prompt, all with harness defaults (`--mtp-recursive-hidden
post-norm --mtp-base-hidden post-norm --mtp-history committed`):

| Row | `--spec-tokens` | `--mtp-physical-n` | Meaning |
| --- | ---: | ---: | --- |
| D1 | 1 | 2 | Packed D1/N2 lazy-verify |
| D3 | 3 | 4 | Packed D3/N4 recursive |
| D7 | 7 | 8 | Packed D7/N8 recursive |

Sampler: greedy argmax. Warmup: enabled. Reps: **3 fresh processes** per
(prompt, config). Serial `reference` re-measured every process.

## Terminology (matches the 3.6-MTP dir; not in PERF-LOG.md under these names)

- **charged-total ratio**: `reference.total_ms / speculative.total_ms`
  (includes prefill on both sides).
- **decode-only ratio**: `reference.decode_ms / speculative.decode_ms`.
- **chain-inclusive per-packet cost**: `sum(speculative.phase_ms.*) /
  speculative.steps`.
- **equivalence PASS**: `identical=true` ∧
  `target_state.continuation_argmax_equal=true`.

## Preregistered gates

### Correctness (hard, per-invocation)

`equivalence=PASS` on every (prompt, config, rep). Any single failure across
the 2 × 3 × 3 = 18 invocations blocks the milestone. `resume_audit_pass` is
observed separately but is not the primary gate — that call was made in the
3.6-MTP dir and applies here too. Given 3.8's Q8_0 `eh_proj` (v.s. 3.6's Q4_K),
the strict `resume_audit_pass` is expected to be met more often on 3.8.

### Performance — informational only until data lands

3.8-MTP has never been measured before. There is no perflog anchor to gate
against. Post-baseline gates (P1/P2/P4 targets) will be preregistered in a
follow-up doc **after** baseline lands, so gates are anchored on real 3.8
numbers instead of transplanted 3.6-MTP ones.

### Serial-leg regression guard (informational for P0; hard from P0.1 onward)

Every milestone re-runs P0 and requires `reference.total_ms` median regression
**≤ 3%** vs baseline P0 across the pool. Paired ratios are meaningless if the
serial denominator drifted.

### Anti-guardrail: MTLResidencySet exclusion

No milestone in this program enables whole-model MTLResidencySet or
wired-residency mechanism, per `PERF-LOG.md:322-350` closure (98.7 GiB wired
after SIGKILL, machine reboot required). Any candidate implicating it exits
this program and enters a separate reopen review before measurement.

## Deliverables

- `README.md` — this preregistration.
- `run.sh` — deterministic invocation script (sed-derived from the 3.6-MTP
  sibling with only the model path changed; verified equivalent otherwise).
- `prompts/pXX-*.txt` — pool (same bytes as 3.6-MTP sibling).
- `baseline/pXX-DK-rN.{json,out}` — 18 result JSONs + stderr logs.
- `baseline.md` — post-run summary with cross-model delta table
  (3.8 vs 3.6-MTP on the same anchors, HEAD `98aba93` for both).

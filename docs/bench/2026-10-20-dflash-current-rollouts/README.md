# DFlash2 on The Current Rollouts

Retention note (October 22): the `target/` directory was cleared after these
measurements. The aggregates and protocols below remain, but the referenced
local scripts, manifests, outputs, and raw logs are no longer available.

## Historical Replay Controls

Binary: clean `0f6f4436`, M4 Max, ordinary Metal lease, API validation enabled.
Replay the saved first six messages, ending on a real user turn, without
changing the saves or generating separate histories between arms:

- `~/code/llm/game/qwen38-q4-vs-q8-terse/q4.json`
- `~/code/llm/game/qwen38-q4-vs-q8-chaos/q4.json`

This reproduces the historical flat `--messages` path: generic ChatML with
explicit `--messages-preserve-thinking`, not the modern native `run` renderer.
Sampling is the original temperature 0.7, top-k 200, top-p 1, min-p 0.05, seed
42. Each arm generates 256 tokens; every run reaches the cap. These slices may
be entirely reasoning and establish neither completed-turn quality nor EOS
latency. Sampled serial/spec outputs need not be equal.

Each cell runs serial/spec/spec/serial in separate processes. Decode rates are
means of two observations. Request ratios use mean request time, including
prefill but excluding loading; retained manifests also record process wall.

| Target / Drafter | Prefix | Tokens | Serial tok/s | Spec tok/s | Decode ratio | Request ratio |
|---|---|---:|---:|---:|---:|---:|
| 3.6 A3B UD-Q4_K_M / BF16 | terse | 7,724 | 76.19 | 51.74 | 0.679x | 0.842x |
| 3.6 A3B UD-Q4_K_M / BF16 | chaos | 10,826 | 72.92 | 54.33 | 0.745x | 0.890x |
| 3.8 27B Q4_K_M / Q8_0 | terse | 7,724 | 19.38 | 18.95 | 0.978x | 0.937x |
| 3.8 27B Q4_K_M / Q8_0 | chaos | 10,826 | 19.41 | 19.83 | 1.022x | 0.985x |

A3B acceptance is 38/182 drafted tokens on terse and 10/105 on chaos; both
back off to serial. The terse dense arm accepts 99/364 and also backs off.
The earlier short-code witness is not representative of this sampled gameplay.
Keep speculation opt-in; there is no measured end-to-end benefit here.

These are exploratory timings, not isolated-machine performance qualification:
other development builds ran during parts of the sweep. The repeated large A3B
slowdown is actionable; the small dense decode differences are not a speedup
claim. API-validation timing is not production timing.

Local reproduction and raw records:
`target/dflash-current-2026-10-20.NnVbGx/measure.py`, `runs.jsonl`, per-arm logs,
request stats and generated output, and `legacy-summary.json`. No gameplay
content or model weights are included in this packet.

## Current Native Replay

A separate ABBA replay uses the first 14 messages of
`~/code/llm/game/claude_flash_next_gsq.json`, ending at its last saved user turn.
Only the messages are copied into a private request fixture; the original save
is unchanged. The modern native House renderers use the saved temperature 1.0
and seed 42, with release defaults for other fields (top-k 20, top-p 0.95,
min-p 0). History is preserved by native policy; missing historical reasoning
is not invented. Generation is again capped at 256 tokens.

| Target / Drafter | Native renderer | Prompt tokens | Serial tok/s | Spec tok/s | Decode ratio | Request ratio |
|---|---|---:|---:|---:|---:|---:|
| 3.6 A3B UD-Q4_K_M / BF16 | qwen36 | 11,512 | 72.52 | 55.74 | 0.769x | 0.899x |
| 3.8 27B Q4_K_M / Q8_0 | qwen38 | 11,550 | 21.06 | 19.67 | 0.934x | 0.999x |

A3B accepts 20/105 across 15 speculative steps, then spends 220 steps off;
dense accepts 51/238 across 34 speculative steps, with 170 off. Both report
acceptance-triggered backoff, no numerical fallback, and identical counters
between repeats. The dense cold request is prefill-dominated, masking its
decode slowdown. This native replay confirms the negative result rather than
rescuing the older generic-renderer numbers.

The replay waited for compiler quiet before starting; no development builds or
downloads were launched by this lane during it. Other work was not globally
suspended. These remain two-observation, API-validation-on measurements, not a
precise production-performance estimate. Local script: `native.py`; manifests:
`native-runs-v2.jsonl` and `native-summary.json`. The initial wrapper-with-meta
attempt was rejected before inference; only the corrected messages-only runs
enter the table.

## Resident Restored Replay

The existing `scripts/serve/request_probe.py` runs one cold request and two
identical restored requests in separate serial/spec server processes. Disk
snapshots are disabled, RAM cache is 4096 MiB, and
`QWEN_DFLASH_PREFIX_REPLAY=0` prevents replayed outputs from impersonating new
draft work. Each request explicitly preserves history, uses temperature 1.0 /
seed 42, and caps output at 256. The script cooperatively stops only its owned
server PID after each arm.

| Target | Restored prompt tokens | Serial wall | Spec wall | Serial first delta | Spec first delta | Wall ratio |
|---|---:|---:|---:|---:|---:|---:|
| 3.6 A3B | 11,507 | 3.738 s | 5.172 s | 105.6 ms | 137.1 ms | 0.723x |
| 3.8 27B Q4 | 11,545 | 12.903 s | 13.728 s | 295.2 ms | 396.4 ms | 0.940x |

These are means of two warm requests per arm, not separate order-balanced
process repetitions. A3B runs 14 speculative/verify/restore calls, accepts
19/98 drafts and spends 222 steps off. Dense runs 34 calls, accepts 51/238 and
spends 170 steps off. Both report acceptance backoff, zero numerical fallbacks,
and the same counters on both restored requests. Thus the negative result is
not caused by missing restored captures or silent admission fallback.

Local reproduction: `warm.py`, `resident-requests.json`, `resident-summary.json`,
and `serve-{a3b,dense27}-{serial,spec}/` under the retained result directory.
The probe now defaults disk snapshots to `off`; explicit paths remain available.

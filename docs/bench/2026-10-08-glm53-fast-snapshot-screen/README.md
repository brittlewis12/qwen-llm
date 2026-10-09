# GLM-5.3-Flash serve with Fast snapshots on: agent-shaped release screen (map #13/#15, 2026-10-08)

Question: with Fast snapshots on by default (`b4c1e79f`), what does an
agent-shaped conversation cost per request, against the same build with
them off (`QWEN_GLM_FAST_SNAPSHOTS=0`)?

Answer: a branch, a new session with the same instructions and tools, and a
return to an earlier session each restore the 11,104-token shared prefix in
5-11 ms and reach their first streamed delta in 0.5-1.7 s, against 57.7-59.6
s with snapshots off. A cold first request pays one 28 ms capture
(0.27 GiB, inside the 4.26 GiB auto budget). Continued steps extend the live
session either way.

## Method

`scripts/reference/glm53/serve_agent_reuse_screen.py` (no opencode process,
no external server) against a `qwen serve` it launched from
`/tmp/qwen_release_b4c1e79f` (release build, no API validation,
`--max-context-tokens 32768`, prefill rows 512, Fast lineage, effort low),
sending `scripts/reference/glm53/opencode-shape-v1.json` (the opencode
client's instructions and 12 tool schemas) with canned tool results. The
"on" run went first, then "off", sequentially, warm weights. `report.json`
per run; `phases.log` holds serve's `serve limits` and `serve phases:` lines.

## Results

| Request | On: reused / prefilled | On: first delta | Off: reused / prefilled | Off: first delta |
|---|---:|---:|---:|---:|
| A1 cold | 0 / 11,129 | 60.8 s | 0 / 11,129 | 58.7 s |
| A2.. continued | 11,153 / 99 | 1.18 s | 11,164 / 43 (and two more steps) | 0.56-0.99 s |
| B branch (edited first turn) | 11,104 / 24 (snapshot) | **0.57 s** | 0 / 11,128 | 57.9 s |
| C1 new session, same prefix | 11,104 / 19 (snapshot) | **0.50 s** | 0 / 11,123 | 57.7 s |
| A' return to session A | 11,104 / 229 (snapshot) | **1.73 s** | 0 / 11,469 | 59.6 s |

- Restores took 10.5, 4.8 and 6.0 ms; boundary verification 0.9-2.0 ms of
  CPU per request; the one capture 27.9 ms.
- The two runs generated different text from the first request on: with
  snapshots on, Fast reads the prompt in two segments (cut at 11,104), so
  its logits differ from the unsplit read (the schedule sensitivity the
  policy accepts), and the conversations diverged (session A took one tool
  step with snapshots on, three with them off). Request rows are therefore
  comparable by kind, not token for token.
- The cold request's 2.1 s difference is unresolved: one on/off pair (on
  ran first) cannot separate the split's schedule overhead from order
  effects and run-to-run variability.

Scope: one scripted conversation in one release build; a cost screen, not
a quality measurement (quality: `../2026-10-08-glm53-fast-split-quality/`).

# Serve idle residency on DeepSeek V4 (2026-10-05)

Leverage map 2026-10-05 #1. This checks a second, dissimilar no-copy family
after the generalization in `344da7e7`. DeepSeek-V4-Flash-0731 UD-IQ3_XXS,
M4 Max 128 GB, binary `dc2d5ada`, `--max-context-tokens 4096`, 4 output
tokens.

```sh
uv run scripts/reference/serve_pause_screen.py --qwen target/release/qwen \
  --model <DS4 shard 1> --out <dir> --max-context-tokens 4096
```

Each arm starts its own server and sends fresh ~40-token prompts (no
prefix or snapshot reuse):
- a warm-up request;
- a back-to-back request;
- requests after 10 s and 30 s pauses.

Wired memory is sampled with `vm_stat` before and after each request.
`report.json` has every row; `serve-lines.txt` has the servers' `serve
phases:` lines.

| Request | off: wired before (GiB) | off: prefill (ms) | default 60 s: wired before (GiB) | default: prefill (ms) |
|---|---:|---:|---:|---:|
| warm (cold weights) | 5.15 | 12,246 | 4.91 | 17,103 |
| back to back | 104.55 | 1,244 | 105.33 | 1,302 |
| after 10 s | **4.91** | **2,032** | **102.12** | **1,285** |
| after 30 s | **4.92** | **1,994** | **102.12** | **1,212** |

- With the keep-alive off, DS4 drops to idle wiring within 10 s of a
  request, like GLM. A paused request then pays ~0.75–0.79 s to re-wire
  ~100 GiB.
- With the default window, the weights stay wired through 10 s and 30 s
  pauses, and paused requests prefill as fast as back-to-back ones.
- Both servers stopped on SIGINT (exit 130 by design). Wired memory then
  fell to the host's other work (36.99 GiB, held by another process's
  Qwen3.8-27B run).
- The ~1.2 s prefill for 40 tokens is DS4's short-prompt cost (leverage
  map #9), not placement.

Decision: the default 60 s window holds for every no-copy family. GLM was
measured in the placement screen and DS4 here; the mechanism and the kill
check are family-neutral.

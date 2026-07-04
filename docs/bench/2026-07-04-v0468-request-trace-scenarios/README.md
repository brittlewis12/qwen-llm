# v0.468 Request Trace Scenarios

Goal: make replay request-economics inputs reproducible without pretending that
modeled arrivals are empirical workload traces.

## What Changed

- `scripts/profile/request_trace_from_game.py` converts game transcripts into
  `replay_economics.py --request-trace` TSV files.
- The trace header records provenance: arrival source, content source,
  completion source, token-count source, and `is_empirical_arrival=false`.
- The builder supports char-estimated counts for fast scenario packets and
  optional exact counts through `qwen-bench tok`.

## Commands

```sh
uv run python -m py_compile scripts/profile/request_trace_from_game.py

uv run scripts/profile/request_trace_from_game.py \
  --input /Users/tito/code/llm/game/chaos.json \
  --arrival-model burst --max-requests 4 \
  --model /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --output target/profiles/v0468-game-trace-builder-exact-smoke.tsv

uv run scripts/profile/request_trace_from_game.py \
  --input '/Users/tito/code/llm/game/*.json' \
  --arrival-model fixed-gap --fixed-gap-ms 500 \
  --output target/profiles/v0468-game-scenario-fixed500.tsv

uv run scripts/profile/request_trace_from_game.py \
  --input '/Users/tito/code/llm/game/*.json' \
  --arrival-model burst \
  --output target/profiles/v0468-game-scenario-burst.tsv
```

Then run the scenario traces through the v0.466 S1/S2/S4/S8 replay rows:

```sh
uv run scripts/profile/replay_economics.py \
  --real-margin target/profiles/v0466-a3b-real-economics-slotcounts-c8192.tsv \
  --request-trace target/profiles/v0468-game-scenario-fixed500.tsv \
  --interpolate-slots \
  > target/profiles/v0468-game-fixed500-c8192-econ.tsv
```

## Scenario Results

The game corpus packet contains 34 JSON inputs, 163 assistant completions, and
about 218k char-estimated completion tokens.

| Scenario | Context | Save | p95 Delta | Replayed Tokens |
| --- | ---: | ---: | ---: | ---: |
| fixed-gap 500 ms | `8192` | `8.13%` | `-31.06%` | `90.82%` |
| fixed-gap 500 ms | `16384` | `5.02%` | `-20.85%` | `90.75%` |
| burst | `8192` | `8.76%` | `-9.01%` | `97.43%` |
| burst | `16384` | `5.42%` | `-5.57%` | `97.43%` |

## Decision

This is useful scenario/stress coverage, not a replay promotion gate. The content
and completion lengths come from real transcripts, but the arrivals are modeled
and the full packet uses char-estimated tokens. Replay remains blocked on
empirical arrival traces from actual usage or a real workload source.

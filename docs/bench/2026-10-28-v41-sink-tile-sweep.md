# V4.1 8K CED Sink-Tile Width Sweep: 512 Is Optimal

2026-10-28. Source: main `65324572`. 8K CED, 40 layers, exact host ranges.
Raw: `~/qwen-runs/2026-10-28-v41-sink-sweep/`.

## Decision

Keep the 512-row query tile. Narrowing the tile loses 14.6% at 256 and 17.7%
at 128. The query-tile width is not the attention lever; the span-wide window
partition is the next experiment, and only if a complete-core probe shows a
material reduction.

## Protocol

`QWEN_V41_PREFILL_EXECUTION=shrinking_ced`, `QWEN_V41_SINK_TILE` override,
`--pp 8192 --depth 0 --runs 2 --seed 1`, reversed order `128 256 512 512 256
128`, 180 s between arms, 120 s pre-cooldown. Same binary; ordinary suite.

## Results

| tile | samples (tok/s) | mean |
| ---: | --- | ---: |
| 128 | 1,349.07 / 1,344.78 | 1,346.93 |
| 256 | 1,405.12 / 1,390.20 | 1,397.66 |
| 512 | 1,633.62 / 1,638.11 | **1,635.87** |

Both directions agree; the 512 arms are stable to 0.3%. The narrower arms have
larger spread but the gap is far outside it.

## Limits

Ordinary-suite CED throughput, not the Full/CED pair gate; no endpoint
comparison here. One prompt, one artifact, one session. The 512 default is
already the maximum query tile; the result does not speak to the key/window
partition or selected-key gather.

## Next

Complete-core span-wide window probe: compare the current per-query 32-key
panels against a shared window-band partition for a 512-query tile, including
band construction, masked softmax, selected gather and PV. Require at least
20% less complete attention-core time before integrating.

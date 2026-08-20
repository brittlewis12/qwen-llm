# F02 — N=321 padding cliff exists in isolation, small in production

Date: 2026-08-18. HEAD `307007d` (mid-session drift; measurement, no code
change). Model: `Qwen3.8-27B-Q4_K_M.gguf`.

## Hypothesis (k3 gut-check callout)

Q4_K matmat kernel N-tile is 32 (`metal.rs:7236`). Any N not a multiple of 32
pays proportional waste from the last tile. Test near-boundary sizes to
confirm the cliff and estimate production impact.

## Measurement

`qwen-bench pp --n-prompt N --runs 5`, single fresh process per N.

Isolated small prompt:
| N | t/s | ms/tok | vs N=320 |
| ---: | ---: | ---: | ---: |
| 320 | 240.88 | 4.151 | — |
| 321 | 215.28 | 4.645 | **−10.6%** |
| 384 | 233.73 | 4.278 | −3.0% |
| 448 | 227.13 | 4.403 | −5.7% |
| 512 | 231.25 | 4.324 | −4.0% |

pp321 arithmetic: `div_ceil(321/32) = 11` tiles, `11*32 = 352` rows, `352-321
= 31` padding rows → waste ratio 31/352 = **8.8%**. Measured throughput drop
10.6% is consistent with pure padding + noise.

p01-anchor-length regime (multi-chunk with chunker at `p_max`):
| N | t/s |
| ---: | ---: |
| 1920 (aligned) | 221.47 |
| 1926 (not mult) | 223.21 |

**No cliff at 1926 vs 1920.** Both share chunks 1-3 at N=p_max=512 (aligned);
chunk 4 differs by 6 tokens (384 aligned vs 390 misaligned), waste bounded by
26/416 = 6.3% of that last chunk × chunk-4/prompt = 390/1926 = 20% share
of total → 1.3% of total prefill wall.

## Conclusion

Cliff exists in the kernel (measured −10.6% at pp321 in isolation) but
production impact is bounded by the last-chunk share × padding fraction on
that chunk. For p01 anchor: ~1.3% of total prefill wall. Not a real lever.

## Action

- **No source change needed.** Cliff is bounded structurally by production
  chunker sizing.
- One-line polish available if desired: chunker at
  `metal_forward.rs:7653 chunk_p = (total_n - chunk_base).min(p_max)` could
  round the last chunk up to a multiple of 32 (or 64) with pad-then-truncate
  semantics. Save ~1-2% on residual-chunk prompts. Not high-priority.
- F02 closes the "N-alignment cliff is the MLX gap" hypothesis. Move on.

## Retraction of prior framing

Earlier commentary that framed the cliff as a "real actionable lever" was
overreach based on the isolated pp321 measurement. When re-measured at
p01-anchor scale (pp1920/pp1926), the effect vanishes into noise.

# Serve terminal-Off setup elision: HOLD

Baseline `2a839537`, candidate `0dd6b6c2`, separate release builds; removed at
`61559330`. Apple M4 Max, Qwen3.8-27B Q8_0 and DFlash2 Q8_0, ordinary loading.
No filesystem-cache-cold claim. All four natural-prompt processes ran serially
in A-B-B-A order and exited cooperatively. Raw requests, event timestamps,
server logs, host observations, protocol, and summary remain at
`target/profiles/request-elision/` in the request-work-elision worktree.

The candidate validates the hard context ceiling before request work and skips
drafter seeding/verifier allocation above it, retaining the capture-fed serial
path and cache tails. The production hard stop remains unchanged.

## Correctness

- Forced `QWEN_DFLASH_OFF_CTX=0` short baseline/candidate pilots preserve greedy
  and seeded temperature-0.7 output, exact hits, followups, and one-output limits.
- Default-policy natural input is 24,194 tokens, from the first 85,000 characters
  of the baseline roadmap plus the frozen summary instruction. All four arms
  have identical output hashes and usage/cache counts on all six requests.
- Long followup restores 24,195 of 24,216 tokens, including a pending-token
  boundary. Short EOS continuation restores 21 of 41 and changes orange to blue.
  Short in-boundary requests remain on DFlash; only long requests change path.
- Ten focused CPU backend tests pass. No full-state bitwise or broad sampled
  speculation claim is made from HTTP output equality.

## Endpoints

Milliseconds below are medians of two processes per arm. No outliers removed.

| Request | A first content | B first content | A complete wall | B complete wall |
| --- | ---: | ---: | ---: | ---: |
| Long fresh / 64 outputs | 118783.289 | 115116.239 | 123050.149 | 119345.117 |
| Long exact hit / 64 outputs | 167.933 | 157.962 | 4438.758 | 4391.713 |
| Long exact hit / 1 output | 160.354 | 151.582 | 276.127 | 266.064 |
| Long followup / 2 outputs | 1695.135 | 1666.499 | 1881.769 | 1851.379 |
| Short EOS seed | 1183.013 | 1180.709 | 1349.727 | 1340.157 |
| Short EOS followup | 1302.448 | 1309.292 | 1372.613 | 1378.599 |

Warm 64-output wall saves only 1.06%, below the 2% screen. Exact-hit TTFT
controls spread 33.28%; one-output wall controls spread 15.27%. Fresh controls
spread 7.39% in wall and cannot support attribution of the apparent 3.01% saving.
Long followup nominally clears the alternative 20 ms TTFT screen with 28.636 ms
saved and 1.15% control spread, but its reverse-order pair does not reproduce the
gain (B2 1689.004 vs A2 1685.443 ms). This is not a formal all-cell KILL: one
screen clears, but the evidence is too narrow to call a robust product promotion.

Disposition: remove the prototype and retain HOLD evidence rather than widen
this campaign now. Resource/setup removal is structurally real; neither its
memory benefit nor a stable endpoint gain was certified. Revisit only if setup
or verifier lifetime is material in a named deployment cell. Advance to deleting
the serial-prefill heads, a larger and independently attributable work unit.

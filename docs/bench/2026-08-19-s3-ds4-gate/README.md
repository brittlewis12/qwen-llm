# S3 live gate: DeepSeek V4 serve

Executed 2026-08-19. Server: `qwen serve` @ serve/s1 (release), model
`DeepSeek-V4-Flash-0731-UD-IQ3_XXS` (97 GB, four shards),
`--max-context-tokens 8192 --max-tokens 512`. Cells and pass criteria
were proposed by the k3 adversarial review (see "Review closure"), which
ranked them by falsification power; cells 1–2 ran first because they
would catch every critical finding the review had just made.

## Results

| # | cell | criterion | result |
| --- | --- | --- | --- |
| 1 | Warm snapshot hit rate | every continuation restores; zero capture failures | **PASS** — turn 2 `matched=14/24`, turn 3 `25/36`; restores 37 ms; 37.8 s → ~0.85 s wall; 0 capture failures |
| 2 | Byte identity vs CLI | serve output byte-identical to `qwen` DS4 single-turn, incl. non-ASCII | **PASS** — ASCII, CJK (`数学は面白い`), emoji (`🎉→✓`) all identical; zero U+FFFD |
| 3 | Reasoning round-trip | verbatim reasoning echo keeps checkpoint reuse | **PASS** — `matched=107/118` (91 %), restore 56 ms |
| 4 | Headless partition | reasoning item separate; no `</think>` in visible text | **PASS** — `['reasoning','message']`, both `completed`, no leakage |
| 5 | Fail-closed | tools / effort=medium / over-budget / store:true rejected, residency healthy after | **PASS** — all four rejected with correct `param`; server served normally afterwards |

Cells 6–9 (LRU eviction, cancellation/heartbeat timing, TTFT at 8k,
memory envelope) are deferred; they are the conditional-pass class and
need dedicated runs.

## What each cell proved about the review findings

- **Cell 1 validates the R1.1 fix.** Before it, DS4 sessions carried no
  bound model-content identity, so every `capture_causal_snapshot` and
  `restore_causal_snapshot` hard-failed into a `warn` and the snapshot
  LRU could never fill — the warm path was 100 % dead while looking
  healthy. Non-zero `matched_tokens` on every continuation is the direct
  falsifier, and it passes.
- **Cell 2 validates the R1.6 fix.** Per-token `decode_piece` is
  `from_utf8_lossy`, so multibyte characters split across tokens became
  U+FFFD — corrupting output *and* breaking checkpoint reuse (clients
  echo the corrupted text, which no longer re-encodes to the captured
  tokens). Byte identity with the CLI on CJK and emoji is the falsifier.
- **Cell 4 validates S3-1.** DS4 thinking tiers pre-open `<think>` in the
  prompt, so generation is headless; the pre-existing splitter would have
  classified an entire reasoning block as visible answer text.
- **Cell 5 validates R1.7 and R1.5.** Tool definitions now fail closed
  instead of being silently dropped, and rejections do not poison the
  residency slot.

## Process finding (harness, not product)

A boot attempt used the *main checkout's* binary (shell cwd drift), which
lacks DS4 serve and exited in the first second. The readiness loop probed
only the socket, never process liveness, and printed its success message
unconditionally — so it polled a dead process for ~33 minutes before the
first request failed. Fixed by `scripts/serve/boot.sh`, which checks
`kill -0` each iteration, fails fast with the tail of the server log, and
resolves the binary by absolute path.

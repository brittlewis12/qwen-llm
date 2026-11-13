# Short-index bounded qualification, 2026-11-13

Stage starts 15:47:46 UTC; stop and report before 17:47 UTC. Only short
masked-index qualification is active. MID is already landed. The old lane
history is preserved in `backup/v41-prefill-before-bounded-stage`; its net
changes were rebased onto rewritten main `56a1e8fe5`. Rebase conflicts in
MID declarations, generated ENV and the shared bench note were resolved
without dropping the short candidate or main evidence. Main MID excerpt
is retained as `2026-11-12-v41-prefill/MID-LANDED.md`.

The cooperative expert prototype is deferred and removed from this branch.
The previous cache-priming controller is superseded. Historical nice10
measurements and unpurged serve results remain records, not new acceptance.

## Corrected protocol

- Build with nice10; measurements and serve at nice0, verified with `ps -o nice=`.
- Same `iter` binary for native A/B, release for serve.
- Before EVERY serve arm, target-invalidate only the IQ4_NL Engram using
  the residency study's `msync(MS_SYNC|MS_INVALIDATE)` helper, and require
  zero resident pages afterward. No priming process.
- Three interleaved pairs, orders AB / BA / AB, same release binary, plan,
  cap32 and three 48K agents. Use greedy sampling in both arms to reduce
  stochastic history/I/O variation, as in the residency study. Keep every
  matched observation and root. Fixed active4, sample-workers3, fair1.
- Record live process disk-read counters, native per-call I/O, load,
  compiler activity and exact model niceness for every arm.
- Settle <=1 for 60 continuous seconds per load, allow2, abort3.
- Each lease <=15min; Python children inherit the lease via `pass_fds`.

Earlier completed boundaries:128 rows 1.01678x (2/3 positive, all8bitwise),
511 rows1.04877x (3/3 positive, all8bitwise), nice10/pressure0. New nice0
controls include150/256 target cells,127/512/1024 inactive cells and two
root shapes:49152 and49509. The latter matches the first served root and
can exercise a357-row CED tail; do not call every served root inactive.

Serve promotion requires >=3 eligible equal-row short append matches in
each pair, improved mean and median prefill and client TTFT across the
three pairs, and no material root or aggregate latency regression. Report
each order separately and all disk bytes; a failed gate removes the short
candidate. Optional wide-PV served-root recheck only if this finishes within
~90 GPU minutes; it is evidence only and cannot promote in this stage.

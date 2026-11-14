# V4.1 conditional yield: replacement BA pair

Initial preparation started 2026-11-14 19:12:35 UTC. The lead stopped the
RAM-before-queue attempt and reset this stage at 19:38:30 UTC; stop/report by
21:08:30 UTC.
The lead authorized repeating only the invalid BA pair with the retained
release pins, then combining it with valid standing pairs 1 and 3. The
original B2 and failed A2 remain recorded; no slow valid pair is discarded.

## Frozen protocol

No performance threshold, call matching, percentile, warmup, purge, sampling,
profile or measurement niceness changes from the [standing protocol](2026-11-14-v41-conditional-yield.md).
New arm names are B2r then A2r, each its own waiting lease of at most 15 minutes.
The same archived qwen/qwen-tok/test binaries are used; runtime e41e037e2.
No M5 rebuild. Valid pairs 1 and 3 stand as measured. Native root/append
bitwise and empty-command-topology witnesses stand, with byte-identical
serve/witness metallibs. Branch is already based on main 1522261bc.

## Host-memory preflight

The lead requires recording free/available memory before each arm and waiting
when admission would fail. Availability uses the existing server formula:
`(free_pages + max(inactive_pages - anonymous_pages, 0)) * page_bytes`.
Speculative and active file-cache pages receive no extra credit.

The initial RAM-before-queue wait starved the lane: active lease holders could
keep occupying RAM before this lane joined the FIFO. The lead stopped that
attempt and killed its exact waiter PIDs 15561/15563 and SSH dispatch. No
model was loaded, and no measured arm was attempted under that ordering.

**Corrected order: join the Metal FIFO first.** Check RAM only after acquiring
the lease. Require 196 GiB (210,453,397,504 bytes) of conservative availability,
sustained for 15 seconds. This preload estimate rounds up the standing run's
observed device/shadow charge (171,370,217,472 bytes), maximum process footprint
(32,601,915,016), failed transport reserve (2,046,812,160), plus 4 GiB of margin.
Accounting can overlap, so this intentionally overestimates incremental demand.

If insufficient, wait for up to five minutes inside the hold, recording each
sample. If still insufficient, or if too little hold time remains to complete
an arm, release before model load and immediately requeue. Deferred preparation
is retained separately. Holds remain below 15 minutes; no memory polling delays
or retries are added to the measured request window. Ghost processes remain
untouched. Record memory during each arm to expose later pressure changes.

First-arm latest start 20:25 UTC; second-arm latest start 20:44 UTC leaves
cleanup/check/report time. An incomplete packet retains all pins.

## CPU readiness

`QWEN_SERVE_V41_PREFILL_YIELD=1 cargo test -p qwen-cli` passed all 12
targets: 1,748 passed, 0 failed, 73 pre-existing ignored. The legacy scheduler
fixture now explicitly chooses rollback settings, independent of the process
default. Production remains default-off until the full gate passes.

## Result

Pending. No promotion before the unchanged three-pair gate completes.

# V4.1 conditional yield: replacement BA pair

Stage starts 2026-11-14 19:12:35 UTC; stop/report by 20:42:35 UTC.
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

Before entering the queue, and again under the lease before load, require
196 GiB (210,453,397,504 bytes) of conservative availability, sustained for
15 seconds. This conservative preload bound rounds up the previous run's
observed device/shadow charge (171,370,217,472 bytes), maximum process footprint
(32,601,915,016), failed request's transport reserve (2,046,812,160), plus 4 GiB
of margin. Overlap between these charges may overestimate incremental demand;
the guard intentionally waits rather than gambling on that overlap.

Low RAM outside a lease waits until the bounded stage cutoff. If RAM drops
while queued, the leased preparation waits within its remaining 15-minute
budget and can defer before loading a model; preparation evidence is retained.
No request pacing, retry or delay is inserted into the measured window.
Memory is sampled during each arm for diagnosis; future Ghost bursts cannot
be guaranteed by a preflight. Ghost processes are never modified.

First-arm latest start 20:00 UTC; second-arm latest start 20:18 UTC leaves
cleanup/check/report time. An incomplete packet retains all pins.

## CPU readiness

`QWEN_SERVE_V41_PREFILL_YIELD=1 cargo test -p qwen-cli` passed all 12
targets: 1,748 passed, 0 failed, 73 pre-existing ignored. The legacy scheduler
fixture now explicitly chooses rollback settings, independent of the process
default. Production remains default-off until the full gate passes.

## Result

Pending. No promotion before the unchanged three-pair gate completes.

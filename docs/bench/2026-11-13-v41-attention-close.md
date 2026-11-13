# V4.1 attention gate: source/profile pin and final measurements

Start21:01:00 UTC; hard report<=23:01, no extension. Measurement cutoff22:49
reserves cleanup/records/contracts. The lead clarified that source67b3ebaf4
plus the same iter/release/test profiles and byte-identical metallibs is the
program pin; executable SHA identity is not required. Record original,
previous-rebuild and current executable hashes. No Rust/kernel/probe changes
are made for this measurement, and pair1AB stands exactly as recorded.

## Artifact lifecycle

Rebuild source67b at the original M5 worktree/target paths. Require all9
main/research/tensor metallibs across profiles to match the old hashes.
Copy each executable into attention-close/pins outside target, verify its
SHA and make it read-only. Both arms use those same archived executables.
Keep pins while any gate remains open across stages; cleanup removes bulk
build output, worktree and bundles, not open-gate pins. Delete pins only after
the gate closes. The manifest keeps build and archived paths and all hashes.

## Unchanged measurement protocol

Use the prepared finish-stage protocol: purge Engram0 before EVERY fresh
single native48K process; warmupAB and retainedBA/AB/BA; nice0 verified by ps,
60s continuously<=1 before each load, allow2/abort3. Record per-run load,
Engram prepare wall and completed-submit encode time, process reads and
endpoint digest. Existing full-state/following-step witnesses from the same
source/profile/metallibs stand; new endpoints must match the prior golden.
The ratio of retained mean A/B wall times must be>=0.98; also report the
geometric pair ratio. One<=15m hold if it fits, otherwise checkpoint at a
pair boundary and use a lease per remaining pair. Do not repeat completed runs.

If native passes, finish fresh pair2BA and pair3AB, one ARM per<=15m lease,
with symmetric paired4K model warmup, both GPU witnesses, Engram purge0,
fresh server, greedy/cap32,3x48K agents,60s windows/15s ramp, active4,
sample-workers3/fair1 and the original thermal rules. Each arm has its own
860s internal budget inside885s/15m. Warmup and measured process reads remain
separate. Refuse arm overwrites; require identical pair metadata and plan.
Both arms differ only by SHORT=0/PV_WIDE=0 versus SHORT=1/PV_WIDE=1.
MASKED_REINDEX and MID remain1. Pair1's old executable hash is legitimate
under the explicitly approved source/profile+metallib pin.

Across complete AB/BA/AB: geometric means TPS>=0.98, root prefill>=0.98,
short128..511 and1K-class512..2000 prefill AND client TTFT each>=1.00, with
at least one matched/observed call per bin per pair. All equal-new-row
completed offered calls, roots, counts/null TTFT, compiler/load/thermal and
I/O evidence retained. Native pass plus served pass => ready to land.
Purged root failure drops wide PV; short remains default-on only if its own
150/256 cells hold. Prepared fallback uses PV=0/SHORT=1 and scoped MID
false/true at80K/sub512 rows, exactly main versus short eligibility, with
full paired signatures. No wider sub-bisection or next-lever implementation.
If the queue prevents completion within the stage, report; do not extend.

The independent Engram cache is already composed on local mainc2935fa36,
with contracts passing2e8efa847. Measurement remains on9877/67b as authorized.

## Planning only: lead-forwarded waste audit

For the next stage, cooperative experts shares priority with the forwarded
perf/waste-audit findings (not acted on or remeasured here): score-to-softmax
round trip~34/17ms at2048@48K/1024@80K; scatter_offset_f32_to_f16 in indexer
and expert input plus v41_sorted_pack~71/34ms GPU (4.2% of chunk GPU); three
apparently duplicated attention_index_heads_rows_mma append dispatches~7.5ms,
0.68%, same inputs and no intervening write. Wide PV already covers PV to
normalization. Audit reference: perf/waste-audit:docs/bench/2026-11-13-waste-audit.md.

# Private Coarse Decode Observation, Schema V2

This private checkpoint descends from `27115a64` on production base
`1388793d`. It is not canonical throughput or a production change. The parent
owns independent review and metagross scheduling; no worker/model qualification
is implied by local fixture tests.

## Observation Versus Partition

`observation_accepted` admits the complete restored five-window diagnostic:
model warmup, instrumented warmup, control, profile, control. The existing gates
remain: all 32 finite full-vocabulary rows bitwise equal, exact positions and
stage coverage (including absence), 32 commands, zero trace copies, valid
timestamps, present monotonic I/O counters, zero measured physical reads/pageins,
and profile wall no greater than 1.10 times the mean control wall.

Overlapping encoder timestamp envelopes do not reject an otherwise valid
observation. Per-stage timings are **inclusive encoder-envelope durations** and
are explicitly nonadditive. `costs.inclusive_stage_stats_nonadditive` reports
their count, sum, minimum and maximum. Raw per-layer/per-position intervals and
their individual durations remain available. Unvalidated raw statistics are
not an observation verdict.

`exclusive` is a separate statement about the sampled envelopes: false if any
positive-length overlap exists, true if the sampled intervals are pairwise
disjoint (touching endpoints are allowed). Window-level exclusivity is the
conjunction over its 32 positions; the depth verdict reports the measured
profile's value. Null means a valid profile summary was unavailable. This is
not a claim that the intervals partition the entire command, and is never an
acceptance requirement in the runner.

## Exact Sampled-Interval Geometry

For each position, `sampled_envelopes` reports integer nanoseconds. Sorted
endpoint events, grouped at equal timestamps, determine interval coverage;
encoded adjacency is not used. Nested and identical intervals are included.

- `first_sample_ns`, `last_sample_ns`: minimum start and maximum end.
- `sum_inclusive_durations_ns`: sum of all individual end-minus-start durations.
- `interval_union_ns`: time covered by at least one sampled envelope.
- `sampled_envelope_ns`: maximum end minus minimum start.
- `overlap_multiplicity_ns`: sum of durations minus union; triple coverage counts
  twice in this quantity.
- `covered_by_two_or_more_ns`: time covered by at least two envelopes, counted
  once regardless of multiplicity.
- `uncovered_sampled_envelope_ns`: sampled envelope minus union, i.e. gaps between
  sampled intervals, not uncovered command time.

Invalid endpoints, missing/duplicate/wrong-position samples, operation-duration
mismatches and arithmetic overflow still reject the observation. There is no
clipping, normalization, tolerance or substitute timing.

`command_gpu_ms`, CPU encode, commit/wait, command wall and full-window wall
remain separate measurements. V2 removes the old command-duration-minus-stage-
sum residual. Apple counter timestamps are Mach-aligned nanoseconds and command
GPU timestamps use the same timebase in seconds, as established during review;
no clock defect was found. Nevertheless, this report does not retain absolute
command GPU start/end endpoints, so it cannot witness command containment or
claim command coverage/uncovered time. Even disjoint sampled intervals do not
establish those claims from a command duration alone.

## Local M4 Fixture Finding

The prior M4 Max synthetic fixture observed one overlapping encoder-envelope
pair per decoded token with Engram enabled, at fixture depths 0, 3 and 6.
Without Engram it observed none. One recorded initial-Engram-projection end was
`935995051793708` ns, while the following layer-0 attention-mHC-pre envelope
began at `935995051774083` ns: a 19,625 ns overlap. All 32 full-vocabulary outputs
remained bitwise equal across restored control/profile windows, with one command
per token and zero trace copies.

The V2 local rerun reproduced 32 nonexclusive sampled positions in every
Engram-enabled TG32 fixture window at all three fixture depths, and none without
Engram. Bitwise and valid inclusive-geometry checks passed. The 28 checks comprise
12 family/protocol CPU tests, five profiler/geometry CPU tests, eight GPU-step
fixtures and three snapshot fixtures. They used offline locked Cargo, the normal
target directory, one test thread, `MTL_DEBUG_LAYER=1`,
`QWEN_REQUIRE_METAL_TESTS=1` and `QWEN_METAL_LEASE_WAIT=1`.

This is a timestamp-envelope observation, not proof of concurrent kernel
execution, an execution hazard, a clock defect, a launch bottleneck or a speedup.
V2 retains it as a valid inclusive timing observation with `exclusive=false`,
subject to all the other observation gates. No GPU encoding, barriers, bindings,
arithmetic, copies, sample locations, command counts or control-path encoders
change in this successor. Real-model I/O/overhead/bitwise qualification still
belongs to the pending reviewed worker run.

## Runner

`/tmp/v41-complete-rows-decode-attribution.sh` pins the successor and requires
schema `v41-private-restored-decode-coarse-v2` on all diagnostic records. It
requires `observation_accepted=true` at all three depths and completion, records
`exclusive_by_depth` separately, and admits false exclusivity. Other protocol,
build-identity, lease, fresh-evidence and clean-detached-abef restoration guards
remain. The `27115a64` runner is retained separately. Neither runner is executed
as part of this source checkpoint.

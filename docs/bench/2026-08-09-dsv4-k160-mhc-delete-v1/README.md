# DeepSeek V4 K160 mHC Representation-Deletion Ceiling

Status: implementation in progress. The packet was frozen at `0b4c62d` and its
passive timing clarified at `9b0307d` before model or timing observations. The
engine oracle and campaign substrate now exist. An independent arithmetic-parent
endpoint was acquired before experimental timing; no C/A/P/Z acquisition or
result exists.

Passive-timing clarification: implementation review exposed that retaining all
completed command objects until request end would change their ordinary
lifetime. Before any model execution or timing observation, this packet instead
freezes immediate scalar reads after the wait that proves each command complete.
Those common observer reads remain charged inside every A/P/Z wall sample.

## Independent Parent Endpoint

The arithmetic-parent endpoint was acquired from a clean detached `bbfcca8`
release build on the frozen M4 Max/K160/token cell at `2026-08-09T14:56:01Z`.
The campaign accepts only the exact committed artifact at
`parent-endpoint.json`; it does not trust a merely self-consistent replacement.

- artifact SHA-256:
  `61fbbeec26a9cf026f88102674e0dd20054c1d91eb0c9bf4d0c8b813a37081bb`;
- endpoint SHA-256:
  `4f0c8ccd41021df7fde0518a953e81a7e18c25b7b04288c8c95ad37787cee67f`;
- parent binary SHA-256:
  `00fd8591a8715a1018df0aa3313eaa8e614cb66c005782d2817052f3461ebb71`;
- acquisition harness SHA-256:
  `d155175d04b6448125be37aea54530f12dccd7d078a609abff51ec79c8df0d4a`;
- parent metallib SHA-256:
  `3f7d2bc9a1bf2e0caf69e1f45189de2b28bc1f0f2c36b020e940e1549b623a13`.

This bridge is a guard chosen by this packet: it checks that diagnostics
instrumentation preserves the ordinary parent endpoint. It is not an inherent
requirement of mHC deletion, and it grants no performance authority by itself.

Before experimental timing, implementation review also made two existing
validity requirements fail-closed: AC power mode must be exactly `2` both
immediately before and after the timed loop, and the complete immutable
acquisition is atomically published as an authority-free `HOLD` before the
reducer runs. A reducer or final-publication failure therefore cannot strand
the 36 observations only in process memory.

## Pre-Acquisition Hold

A clean `c5a082a` campaign launch at `2026-08-09T15:39:46Z` stopped after
residency load and before C0 with `fresh session realized allocation census
differs from the frozen plan`. No C/A/P/Z arm executed and no experimental
timing was observed.

The census exposed a model-geometry bug rather than environmental noise: the
session memory inventory hard-coded 256 router-logit rows while K160 constructs
160. The successor derives that row count from the authenticated model config
and adds a model-free 160-versus-256 inventory proof. The stopped launch remains
an authority-free implementation `HOLD`, not a retryable observation.

A second clean `0978bf7` launch at `2026-08-09T15:46:41Z` also stopped before
C0. Static reconciliation then established that live construction and the plan
both contain 635 Shared buffers and 4,357,310,628 logical bytes with matching
per-size multiplicities; they differ only in semantic emission order because
Rust constructs compressor and diagnostics locals before the final session
initializer. Constructor order is not part of the frozen allocation contract.

The next successor therefore compares exact `(requested bytes, realized length,
storage mode)` multisets against the named plan, while retaining the actual
ordered census and requiring it to match bit-for-bit across all 36 sessions.
Count, byte, storage, capacity, oracle-residency, and cross-arm order drift still
fail closed. The second launch likewise observed no C/A/P/Z execution or timing.
Named plan entries remain pricing labels rather than runtime buffer identities;
the normalized scratch's semantic presence is established separately by source,
site/dispatch ledgers, and bitwise endpoint correctness.

## Question

Can deleting the F32 `[16384,N]` normalized mHC slab and its only consumer
clear 2% of a loaded K160 `N=2,048` packed-prefill request?

The current path runs two mHC-pre packets in each of 43 layers. Each packet
materializes the four-stream residual as normalized F32, projects
`16384 -> 24`, then computes controls, Sinkhorn, and collapse. The normalized
slab is read only by that projection. Controls, collapse, both mHC posts,
persistent residuals, and the downstream `[4096,N]` normalized buffers are not
part of this deletion premise.

The direct half-matrix candidate retained the slab, saved about 51.9 ms, and
changed final logits. It is closed. This packet prices representation deletion
before another producer is designed.

## Frozen Cell

- Device: Apple M4 Max; emit registry ID, OS, Metal, and power mode.
- Model: the 1,328-tensor, 89,920,886,108-byte, E=160 K160 cohort. Emit the
  complete model-content ID.
- Input: exactly 2,048 tokens repeating `[35, 201, 200, 34]`, start position
  zero, logits enabled, continuation token 35.
- Build: one clean committed release binary with `dsv4-diagnostics`; emit the
  arithmetic parent, experimental source, worktree, and binary identities.
- Policy: emit every resolved attention, compressor, Q-A/KV, router, route,
  expert, and arithmetic policy. Arms are selected in process, not by changing
  process-global environment variables.
- Ordinary topology remains active. Stage counter sampling, trace-layer mode,
  and any observer that disables shared-expert overlap are forbidden.
- The instrumented `A` endpoint must match one independently acquired
  `bbfcca8` parent endpoint before timing receives authority.

The oracle contains `43 * 2 * 24 * 2048` F32 values: 16,908,288 bytes.

## Arms

1. `C` — capture. Run the current RMSNorm and exact Q8 projection, but write
   each `[24,N]` result directly into its layer/site oracle slice. Existing
   controls consume that slice. Capture is untimed.
2. `A` — current control. Run the committed path unchanged; the oracle is
   allocated but unread.
3. `Z` — impossible zero-work ceiling. Skip all 86 RMSNorms and all 86 mHC
   function projections. Existing controls consume the sealed exact oracle;
   controls, Sinkhorn, collapse, posts, routing, experts, and head remain
   unchanged.
4. `P` — producer charge. Run current RMSNorm/projection into dead ordinary
   scratch, but make controls consume the oracle. `P - Z` isolates the complete
   producer under the same oracle consumer; `A - Z` measures direct zero-work
   substitution, and `P - A` records replay locality, dependency, and layout
   effects.

`Z` is trace-specific and deliberately impossible. It can KILL this premise;
it cannot authorize a fused implementation.

Do not build the required-work chargeback or a no-slab producer unless the
packet reaches `IMPOSSIBLE CEILING CLEAR`. A later chargeback must preserve the
incumbent RMS reduction, rounded per-element F32 normalization, Q8
dequantization, K traversal, partial sums, and reduction order. Post-dot
scaling is not exact evidence.

## Capture Contract

Capture twice from independently fresh position-zero sessions into distinct
persistent Shared Metal buffers. Require:

- an ordered ledger of exactly
  `(layer 0 attention, layer 0 FFN, ..., layer 42 attention, layer 42 FFN)`,
  with one non-overlapping write per site;
- bitwise-identical C0/C1 oracle payloads and manifest digests;
- bitwise-identical C0/C1 endpoint and continuation evidence;
- frozen model, token, start-position, geometry, epsilon, and policy identity.

The manifest binds model, source, binary, tokens, start position, N, layer/site
order, per-site offsets and lengths, epsilon bits, dtype/shape/storage, resolved
policies, and complete payload SHA-256. Designate one buffer canonical and seal
it before warm-up. No writable capture API remains reachable from A/P/Z.
Missing, duplicate, reordered, overlapping, or wrong-width sites fail before
timing.

Oracle replay is valid by induction: exact mixes produce exact controls and
collapse; unchanged downstream execution therefore reaches the next site with
the same residual, through both sites and all 43 layers.

## Structural Preflight

Use untimed executions to seal the complete command, encoder, and dispatch
structure. Tag the two mHC sites explicitly in a host ledger; kernel names are
not sufficient because RMSNorm and Q8 kernels have other callers.

- `A`, `C`, and `P` each contain exactly 43 attention and 43 FFN mHC RMSNorm
  dispatches plus 43 attention and 43 FFN mHC function projections.
- `P` matches `A` for every dispatch row and command/encoder boundary.
- `Z` matches `A` after deleting only those 172 tagged rows.
- Per-layer command counts, queue identity, merged-route policy,
  shared-overlap policy, session capacity, and allocation-plan identity match.

At each site, RMSNorm, projection, and controls must remain in the same
non-concurrent encoder. `P` controls read only the canonical oracle while its
producer writes only ordinary dead scratch; `Z` controls bind the identical
oracle buffer, offset, and shape. A concurrent encoder, separate command,
fence/event, reordered dispatch, or topology-dependent timing collector makes
the campaign HOLD.

Dispatch census, order hashing, oracle hashing, and correctness copies are
forbidden inside timed runs.

## Passive GPU Timing

Preallocate the complete scalar interval ledger before timing. After each
ordinary command's existing wait, read status, error,
`GPUStartTime`/`GPUEndTime`, and append without allocation. A shared-overlap
command remains reachable through its incumbent `CommittedPackedCommand` until
the later expert command on the same queue completes; read its scalars then
without extending that ordinary lifetime or adding another wait. These reads
stay inside the common wall interval and may not be subtracted.

The collector must not acquire command ownership, extend any command object's
ordinary lifetime, create a stage recorder, split an encoder, set trace mode, or
participate in any topology decision. An untimed on/off proof must show
identical command and encoder structure. Every command must report completed
status, no error, finite positive duration, and the exact frozen per-layer
submission order.

Record pre-expert, expert, and shared-overlap command intervals separately.
Report raw summed duration and the union of all intervals. Define
`outside_gpu_ms = wall_ms - union_gpu_busy_ms`; raw sums may double-count
overlap and carry no wall attribution authority.

## Acquisition

Use one persistent immutable oracle and residency for every arm. Each arm gets
a new `DeepSeekV4Session` at position zero, with no prior forward or restore,
and is destroyed after evidence collection. Session construction remains
outside timing, but record its wall and allocation identity. The normalized
scratch allocation remains present and untouched in `Z`; this packet prices
loaded execution, not allocation or session construction.

First run a separate untimed mirrored A/P/Z correctness campaign with fresh
sessions. It owns all snapshot exports and restored continuations. Then warm
every pipeline and resource with one fixed untimed `A P Z` sequence. Hash the
canonical oracle immediately after warm-up and once after all 36 timed runs.
Between those hashes, timed P/Z consumer reads are the only permitted oracle
accesses; no host or auxiliary operation may import, hash, traverse, or touch
it.

Timed execution compiles no pipeline and performs no arm-specific or persistent
buffer allocation. Ordinary command-buffer and encoder creation remains part
of every arm. All 36 sessions must have exactly equal allocation-plan digests,
requested and committed bytes, storage modes, capacities, and oracle residency;
allocator addresses are diagnostic only.

Acquire six disjoint, position-balanced sextets in two complete blocks:

```text
block 1:
A P Z Z P A | P Z A A Z P | Z A P P A Z

block 2:
Z A P P A Z | P Z A A Z P | A P Z Z P A
```

No observation is shared across sextets. There is no early stop, retry,
replacement, filtering, outlier exclusion, or reordered run. Any failed arm
returns `HOLD` with no authority.

Timing begins immediately before token staging and packed execution. It includes
the common passive scalar reads and ends after the final command evidence and
normal loaded completion. Immediately afterward, copy the fixed-size endpoint
evidence in one frozen order without another GPU command, drain the queue, and
destroy the session. No snapshot export, continuation, oracle hash, or other GPU
correctness work occurs between timed observations.

For sextet `i`, define means over its two observations:

```text
A_i = mean(A wall)
P_i = mean(P wall)
Z_i = mean(Z wall)
q_i = (P_i - Z_i) / A_i
s_i = (A_i - Z_i) / A_i
r_i = (P_i - A_i) / A_i
kill_i = max(q_i, s_i)
clear_i = min(q_i, s_i)
```

`q_i` is the measured producer charge under the oracle handoff, not a
mathematical upper bound over every future fused producer. `s_i` is the direct
zero-work substitution effect and `r_i = q_i - s_i` is the
replay/dependency/layout effect. The candidate-favorable `kill_i` prevents a
KILL from depending on the less favorable handoff; `clear_i` prevents a CLEAR
from depending on the more favorable handoff.

For GPU intervals, define `q_gpu_i = (P_union_i - Z_union_i) / A_wall_i` and
`s_gpu_i = (A_union_i - Z_union_i) / A_wall_i`, then derive `kill_gpu_i` and
`clear_gpu_i` with the same max/min rules. All wall and union durations must be
finite and positive.

Apply the following bound independently to each six-value wall and union-GPU
decision series:

```text
mean = sum(x_i) / 6
sd = sqrt(sum((x_i - mean)^2) / 5)
se = sd / sqrt(6)
lower95 = mean - 2.015048 * se
upper95 = mean + 2.015048 * se
```

These are preregistered Student-t decision bounds. Their 95% one-sided coverage
interpretation assumes independent, approximately Gaussian sextet effects. No
normality test, retry, or alternative interval may be selected from the six
observed values. Zero variance in a decision series is HOLD.

Stationarity is the symmetric relative difference
`2 * abs(median(block1) - median(block2)) / (median(block1) + median(block2))`
for raw A, P, and Z wall samples separately; each must be at most 5%.
Medians and denominators must be finite and positive.

## Gates

Evaluate validity first. Any structural, allocation, policy, timestamp,
correctness, oracle, stationarity, parent-identity, or evidence-record failure
returns `HOLD` with no authority. Otherwise:

- `KILL`: wall and union-GPU `upper95 < 0.02` for `kill_i` and `kill_gpu_i`,
  and each block median for both series is below 0.02.
- `IMPOSSIBLE CEILING CLEAR`: wall `lower95 >= 0.02` for `clear_i`, each block
  median wall `clear_i >= 0.02`, union-GPU `lower95 > 0` for `clear_gpu_i`, and
  each block median union-GPU `clear_gpu_i > 0`.
- Otherwise: `HOLD`. Do not implement or promote a producer.

Before gate evaluation, serialize all 36 arm labels, ordinal positions, wall
values, command intervals, allocation identities, and endpoint digests into one
immutable evidence record. Hash it, then analyze it without relabeling,
omission, or replacement.

`IMPOSSIBLE CEILING CLEAR` authorizes only a separately preregistered exact
required-work chargeback. It authorizes no producer, fusion, expected speedup,
or product claim.

## Correctness And Authority

Every C/A/Z/P endpoint in the untimed mirrored campaign must match bitwise for
final logits, final normalized hidden, literal committed tokens and final
position, prefix digest, compatibility digest, causal digest, snapshot
observation state, and one restored continuation's logits, hidden, tokens, and
causal digest. Each timed run must match the corresponding fixed-size endpoint
digest without additional GPU work. Oracle payload and manifest must match the
pre-acquisition seal after all 36 timed runs.

Require zero pipeline-cache misses, no arm-specific fallback, and
`outside_gpu_ms >= -0.25 ms`.

A KILL closes only zero-producer substitution for the two
normalization/projection pairs under the frozen serial command placement and
the candidate-favorable A/P envelope, at fixed session allocation, while
controls, collapse, posts, routing, command ownership, and adjacent consumers
remain unchanged in this K160/M4/N=2,048 cell. It does not price removing the
128 MiB allocation, session construction, fusion with controls/collapse,
shared residual reads, adjacent normalization, or independently bounded command
deletion. Reopen requires one of those changed premises, different
hardware/shape, or an explicit numerical quality contract—not another skinny
matrix schedule.

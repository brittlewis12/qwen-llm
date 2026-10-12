# Existing-chunk document checkpoint

Promoted to default Fast serving after the bounded
experiment archived at `42b3cdad`; default, unforced validation passes. Retain one
snapshot at the last existing full packed chunk strictly before the prompt
endpoint. The original chunk schedule, operand precision, final-head selection
and shared-prefix snapshot policy stay
unchanged. The checkpoint has a separate origin namespace including the verified
shared cut, workspace width and precision. Partial chunks or decoded history
cannot regain publication provenance merely by reaching an aligned position.
The read-only postcommit observer copies state after a successful existing chunk;
it adds no execution split or extra final-head calculation.

All Fast precision settings use this behavior under the existing prefix-reuse,
Fast-snapshot and cache-budget controls. There is no new environment knob. Exact
and serial execution are unchanged. A compatible checkpoint requires the same
verified shared cut, rows, precision and token prefix; live reuse wins ties.
Optional capture uses existing admission and protects shared/restored entries.
Denial skips capture without shrinking workspace, changing chunks or failing the
request. This preserves equality with cold execution on the same shared-split
schedule, not an unsplit Fast schedule.

For the frozen real-source packet, the shared cut is 51 and workspace is 2048;
the expected checkpoint is 41011, leaving 409 original prompt tokens. The branch
changes only the final question and verifies the matching checkpoint token prefix.
One model load, fresh backend per A/B arm, ordinary seed and branch requests;
greedy output, full prefill logits and termination must agree. This is backend
timing, not HTTP first-answer latency. Optional admission fallback is recorded;
a successful test alone does not establish that the checkpoint was used.

Before timing, six CPU policy/packet tests pass. The actual-model observer oracle
passes under Metal validation: no-op/capturing observer and restored execution
match ordinary logits and all persistent state across full/ragged chunks and
decode. It also checks cancellation, rejected input, failed GPU status and the
unchanged serial fallback. These checks preceded the initial private A/B below;
default integration validation is recorded separately below.

## Initial Ordinary A/B

[maya-branch-ab.jsonl](maya-branch-ab.jsonl) records the initial private candidate,
with checkpoint admission and selection confirmed; it is not a new timing run
of the production default.
The seed prompt has 41,420 tokens; the edited branch has 41,373. Both arms use
workspace 2048 and identical first-request chunks. Initial and branch prefill
logits, greedy output bytes and termination match exactly between arms.

| Branch metric | Shared-prefix incumbent | Existing-chunk candidate |
| --- | ---: | ---: |
| Reused tokens | 51 | 41,011 |
| Fresh tokens | 41,322 | 362 |
| Prefill ms | 217,658.66 | 2,522.78 |
| First decoded piece ms | 217,727.52 | 2,555.85 |
| Backend wall ms | 240,292.96 | 22,665.94 |
| Restore ms | 53.58 | 25.41 |
| Cache bytes, all entries | 153,264,332 | 796,901,784 |

Candidate capture costs 76.221 ms on the seed and adds 643,637,452 cache-accounted
bytes under the unchanged automatic budget. This is not an RSS measurement.
Session allocation and precision are unchanged.
This is one ordered A/B, not a stable universal 10.6x throughput claim. The
large avoided prefix computation, verified selection and exact outputs support
the mechanism despite timing drift. Startup/load and rendering are excluded.

Both 1024-token seed outputs are truncated equally and are not task-quality
passes. Both branch outputs end normally after 576 tokens and are byte-identical;
they correctly distinguish discarded live decode state from retained snapshots
in the pinned source. This does not erase the earlier sampled answer errors or
establish broader quality.

## Default promotion validation

The reviewed production change enables checkpoints without a test override
(`chunk_checkpoint_override=None`). Owner-reported completed checks:

- CLI suite: 807 passed, 45 ignored in 29.32 s; all-target check passed.
- `gpu_chunk_checkpoint_cancel_resume` under `MTL_DEBUG_LAYER=1`: passed in
  53.60 s. Shared cut 13, cancellation committed at 2061, target 4109, prompt
  4112. Live cancellation/resume publishes the target; a later restore prefills
  three tokens. Full logits, greedy bytes and termination match the incumbent;
  no pins remain held.
- Separate injected `deny_captures` case: test workspace cap 8, suffix 65.
  Denial preserves width, logits, greedy bytes and termination. This is a
  deterministic denial check, not an actual memory-pressure experiment.
- Primitive observer/state-oracle validation passed in 139 s, covering unchanged
  ordinary execution and restored state. No broader task-quality claim follows.

## Telemetry and scope

The execution schedule remains `snapshot_schedule=fast_shared_split_v1` with its
original `cuts`. A deep hit reports `reuse_source=snapshot` and
`restore_origin=fast_chunk_checkpoint_v1`; shared hits report
`restore_origin=fast_shared_split_v1`. Production `serve phases:` adds
`checkpoint_target` (0 if ineligible), `checkpoint_captures` and
`checkpoint_capture_ms`. The private JSON packet calls these `chunk_target`,
`chunk_captures` and `chunk_capture_ms`. `snapshot_capture_ms` includes both
shared and chunk capture time; do not add checkpoint time to it again.

This completes the bounded document-branch lever. The
[earlier 41K live-reuse run](../2026-10-12-glm-maya-repository-context/README.md)
and its factual answer errors remain historical evidence. No additional timing
or quality campaign is queued by this result.

# Existing-chunk document checkpoint

Bounded test-private candidate: retain one snapshot at the last existing full
packed chunk strictly before the prompt endpoint. The original chunk schedule,
operand precision, final-head selection and shared-prefix snapshot policy stay
unchanged. The checkpoint has a separate origin namespace including the verified
shared cut, workspace width and precision. Partial chunks or decoded history
cannot regain publication provenance merely by reaching an aligned position.

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
unchanged serial fallback. Production checkpoint selection remains disabled
pending the bounded experiment and cancellation integration check.

## Initial Ordinary A/B

`maya-branch-ab.jsonl` completes with checkpoint admission and selection confirmed.
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

Candidate capture costs 76.22 ms on the seed and adds 643,637,452 bytes under the
unchanged automatic cache budget. It does not change the session allocation.
This is one ordered A/B, not a stable universal 10.6x throughput claim. The
large avoided prefix computation, verified selection and exact outputs support
the mechanism despite timing drift. Startup/load and rendering are excluded.

Both 1024-token seed outputs are truncated equally and are not task-quality
passes. Both branch outputs end normally after 576 tokens and are byte-identical;
they correctly distinguish discarded live decode state from retained snapshots
in the pinned source. This does not erase the earlier sampled answer errors or
establish broader quality. Next is the focused cancellation/resume integration
check and ordinary-default promotion, without another broad timing campaign.

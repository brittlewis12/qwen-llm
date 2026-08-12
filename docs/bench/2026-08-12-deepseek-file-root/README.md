# DeepSeek File-Scoped Root Fanout

Date: 2026-08-12

Status: product GO for exact multi-pair DeepSeek V4 B2 root reuse.

## Question

Can one immutable causal snapshot replace repeated pair-local prefix evaluation
when several independently planned DeepSeek B2 pairs share the same stable
boundary?

The eight-request fixture is interleaved as `A0,B0,C0,D0,A1,B1,C1,D1`.
Affinity scheduling forms four pairs. All requests share 6,258 tokens, every
pair selects the same 6,144-token restorable boundary, and each request adds a
small pair/private suffix before generating eight greedy tokens. This charges
cross-pair reuse without changing pair selection or decode organization.

## Command

Run the two arms serially with the same binary. The rollback arm adds
`QWEN_CONCURRENCY_FILE_ROOT_FANOUT=0`; the candidate leaves it absent.

```bash
/usr/bin/time -l env \
  QWEN_DSV4_RESIDENCY_SET=0 \
  QWEN_DSV4_PREFILL_CHUNK_TOKENS=4096 \
  QWEN_CONCURRENCY_PREFIX_FANOUT=1 \
  target/release/qwen \
  --model /Users/tito/models/deepseek-v4-flash-0731-reap-k160/DeepSeek-V4-Flash-0731-REAP-K160-Q3_K_Q4_K-00001-of-00004.gguf \
  --requests-jsonl requests.jsonl \
  --concurrency 2
```

## Result

| Organization | Wall | Model prompt evaluations | Pair wall sum |
|---|---:|---:|---:|
| Pair-local rollback | `125.99 s` | `25,544` | `124.584 s` |
| One file root | `67.17 s` | `7,112` | `33.070 s` |

The same-binary candidate improves process wall by `1.876x` and summed pair
wall by `3.767x`. It evaluates the 6,144-token root once, captures one
60,137,472-byte causal snapshot in `385.251 ms`, restores it into both lanes of
each pair, and evaluates only 968 private-suffix tokens across the pairs. This
avoids 18,432 prompt-token evaluations.

Every candidate pair reports two file-root restores, zero pair snapshot bytes,
and `selected_file_root`. Total restore wall is about `1.545 s`; pair-local
rollback instead performs four root prefills, four captures, and four restores.
Concurrent generation remains in the same range (`3.106 s` candidate versus
`3.154 s` rollback), isolating the gain to exact prefix reuse.

Complete input-ordered stdout is byte-identical between arms with SHA-256
`33c34db7d52c3ce49b7a6e12b45bc7784bb18eadc6c628b69bbadf015ce0afb7`.
Both processes report zero swaps. The candidate prices two live Metal sessions
at 8,780,218,368 bytes and CPU snapshot/logit/restore state at 66,307,104 bytes
before allocation; denial retains pair-local execution. Run order did not favor
the candidate: it spent `3.147 s` warming two shards, while the later rollback
found all four shards warm.

## Decision

Enable file-scoped DeepSeek root fanout by default when at least two planned
pairs select one exact restorable boundary of at least 1,024 tokens. Odd serial
tails neither constrain nor consume the root. V1 deliberately does not bridge
from a shallower global root to deeper pair-local checkpoints: boundary drift
falls back to the prior pair-local path rather than growing a hierarchy.

Keep `QWEN_CONCURRENCY_FILE_ROOT_FANOUT=0` as strict rollback. The root remains
request-file-local and immutable; it is not published into a cache index.
Telemetry schema 1 reports planning, admission, actual uses, avoided work, and
snapshot geometry. Pair telemetry schema 4 distinguishes file-root restores
from pair-local capture/restore.

The measurement binary reports base commit `ef186cb` with a dirty worktree
because it contains this candidate plus two unrelated user-owned untracked
documents. `validation.json` binds the measured source and transient binary
hashes. The complete fixture, exact outputs, machine-readable measurements, and
compact raw process telemetry are retained here.

Fixture SHA-256:
`c2690bcd3a47e374a5328ebab52291490fba6d85582fccfad7b9e1e792b01e83`.

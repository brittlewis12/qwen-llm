# Default greedy completion replay, 2026-11-08

Enable `QWEN_DFLASH_PREFIX_REPLAY` by default; `=0` restores the old path.
Ordinary Qwen serving with an explicitly loaded DFlash drafter only. Remove
sampled consensus, multi-seed history and proposal-RNG skipping. Both cache-key
construction and the decoder reject sampled replay. Keep target verification,
existing margin fallback, context limits and backoff.

The current conversation checkpoint ladder serves GLM, V4.1 and Flash-Next;
it does not replace this ordinary-Qwen decode optimization. Existing ordinary
Qwen snapshots reuse prefill state. Completion replay supplies verified draft
tokens and saves drafter work on repeated exact prompts.

A single retained completion has at most 4096 tokens. Admission reserves
**53,248 bytes** for retained history, request clone, replacement and overhead,
before cloning. Check the cheapest viable request plan without prefix eviction;
on pressure, release/disable replay first and admit the normal request. A later
refusal also retries once without replay before allocation. No new knob or
multi-tenant authority is introduced; backend/model scope, exact prompt hash,
greedy sampler version and target verification remain authoritative.

## Initial charged serve packet

Measured source `1656c1fab9e791d815f18096ae2d61ca33aec3d2`, based on main
`7d90e34a`; [driver](run-replay.py), [per-request receipts](replay-initial.json).
Subsequent hardening adds the pre-eviction admission check described above;
normal admitted numerical execution is unchanged. The shipping-tree confirmation
is recorded in the task note and final verification receipts.

M4 Max; real Qwen3.8-27B Q4_K_M + DFlash2 Q8_0, `/v1/responses`, greedy seed 42,
64 output tokens, no thinking, 4096 context, 1 GiB ordinary snapshot budget,
current checkpoint/precision defaults. Four fresh resident servers in **off,
on, on, off** order. The on arms leave the environment variable unset.
Each sends A,A,A,B,A,A, then an untimed sampled diagnostic. A requests a Python
merge function; B requests binary search. Thus first-request miss, warm hits,
replacement and return miss are charged. Each arm emits 384 tokens over six
requests. Every response is incomplete at the requested token ceiling.

| Arm | Six-request wall totals, s | Mean, s |
| --- | --- | ---: |
| Rollback | 7.236325, 7.148049 | 7.192187 |
| Default | 6.650800, 6.634816 | 6.642808 |

**1.083x request throughput** (7.6% wall reduction), including tokenization,
request and DFlash admission/allocation, ordinary snapshot restore/prefill,
verification, completion capture/insertion, transport and JSON parsing. This
is a bounded screen, not a confidence interval or universal speedup claim.
Process startup/model loading is common resident setup and excluded; its raw
costs are retained separately (first cold rollback 6.14 s, later ~2.55 s).
No startup saving is credited.

All 24 greedy response contents match by request index. On every engaged hit
(indices 1,2,5), replay accepts 56/56 in eight packets with no mismatch and
zero drafter calls; rollback makes nine drafter calls. Hit walls are about
0.943 versus 1.116 s. B and the first A after B miss the single completion
cache, while A's existing 28-token transcript snapshot remains reusable. All
four sampled diagnostics report zero replay.

Both locks were held throughout each campaign: shared build lock and canonical
Metal lease, nice 10. Every monitoring observation is thermal pressure 0 with
no rustc. Binary SHA256 is in the receipt. This packet includes the first
request; no training completion or cache-admission cost is excluded.

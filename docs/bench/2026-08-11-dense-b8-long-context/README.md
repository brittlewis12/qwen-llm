# Dense B=8 long-context continuation gate

## Question

Does the fixed-cohort dense-Qwen executor retain useful aggregate throughput
and bounded state drift after a representative 16K causal frontier, rather
than only at the short frontiers used during backend bring-up?

## Method

The probe constructs one canonical 16,384-token state with packed prefill at a
1,024-token chunk width. It snapshots only active persistent state, restores
that state into eight serialized sessions and eight fixed-B=8 sessions, and
then runs one untimed plus four measured teacher-forced transitions.

The restored prefix is shared across lanes. Continuation tokens differ by lane
and position. This isolates the long-attention and large-session behavior that
the promotion gate needs; it is not a claim about eight unrelated 16K prompt
prefills.

Before timing, the probe:

- prices one capacity-sized session from an observed Metal allocation;
- prices the exact packed-prefill scratch plan with Metal alignment;
- admits the remaining 15 sessions with a 2 GiB reserve;
- verifies the first restore byte-for-byte against the source for all active
  F16 K/V rows and complete GDN recurrent and convolution state; and
- applies a 500 ms high-occupancy fixed-B=8 ramp.

Arm order alternates by position. Every transition requires exact per-lane
argmax agreement and bounded full-logit and residual drift. The final audit
checks complete GDN state and only the newly written continuation K/V rows, so
the common 16K prefix cannot dilute a continuation error.

The exact parameter packet below emits `evidence_scope=restored_16k_checked`.
Other restored-frontier settings are diagnostic, and `--no-check` explicitly
removes the checked scope. This is a harness-local evidence label, not an
override of the benchmark framework's build-provenance policy.

The binary and checkout source states matched at acquisition under
`git-source-sha256-v2:4ac4dbf279fed9b29b45dc71931c5e3bad314a4f85d27626e26caf6bea77ca4e`.
The run used `--allow-dirty` because this candidate and unrelated DeepSeek WIP
were uncommitted; it is therefore non-canonical under the global family-board
policy. The exact candidate source, complete build packet, model digest, empty
`QWEN_*` override inventory, and raw output are retained in `validation.json`.
The result closes this implementation gate, not a clean family-board cell.

## Result

### Qwen3.6 27B Q4_K_M

| execution | aggregate tok/s | median wall for 8 | relative |
|---|---:|---:|---:|
| production serialized | 22.393 | 357.250 ms | 1.000x |
| fixed static B=8 | **47.925** | **166.929 ms** | **2.141x** |

Median summed serial GPU time was 353.537 ms versus 164.071 ms for the
fixed-cohort command, a 2.155x GPU-time speedup. The host and GPU ratios
therefore agree; the result is not explained by host-side overlap.

The 16K result retains about 84% of the short-frontier 2.564x backend gain.
That reduction is expected: long-context attention adds private, per-sequence
K/V work that cannot share the model's weight stream. The remaining 2.141x is
still a large product-relevant crossover.

All 40 warmup and measured argmax decisions agree. Worst observed evidence is:

- logits cosine `0.999999965`, relative RMS `0.000269747`, max abs `0.005765`;
- residual cosine `0.999999972`, relative RMS `0.000292126`, max abs
  `0.065063`;
- recurrent-state cosine `0.999999940`, max abs `0.000956`;
- convolution cosine `0.999999923`, max abs `0.004951`; and
- continuation-KV cosine `0.999999881`, max abs `0.031250`.

The packed prefill took 74.476 s. Capturing a 1,230,700,672-byte snapshot,
allocating the 16 destination sessions, and restoring all of them took
1.010 s of setup wall. One capacity-sized session allocated 1,127,137,280
bytes; the exact scratch upper bound was 2,026,295,336 bytes; the final
post-setup Metal allocation delta was 19,134,431,232 bytes.

These setup numbers are not decode throughput and are not a proposed serving
policy. They expose the next product lever: shared-prefix prefill and snapshot
fan-out can avoid paying eight independent 16K prefills when requests share a
prefix, while unrelated prompts still need a separately qualified batching or
prefill policy.

## Decision

- Close the representative long-context implementation gate for explicit
  fixed-cohort dense B=8 decode. The backend remains useful at 16K with all
  enforced state gates green; retain the dirty-build provenance caveat above.
- Keep the capability boundary unchanged: dense Qwen, equal frontiers, F16 KV,
  greedy decode, and complete cohorts of eight.
- Do not infer support for ragged continuous batching, heterogeneous
  long-history correctness, or batched prefill from this result.
- Prioritize common-prefix prefill/snapshot fan-out and an underfill fallback
  before automatic cohort formation.
- Let Qwen MoE and DeepSeek qualify family-specific executors behind the same
  scheduler contract rather than inheriting dense assumptions.

## Command

```bash
target/release/qwen-bench --allow-dirty \
  decode-dense-whole-batch \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --frontier-tokens 16384 \
  --prefill-chunk 1024 \
  --warmup-steps 1 \
  --steps 4 \
  --ramp-ms 500 \
  --seed 1
```

## Artifacts

- `qwen36-27b-frontier16k.txt`: complete ten-line probe output.
- `validation.json`: command contract, source/evidence hashes, allocation
  accounting, and parsed result.

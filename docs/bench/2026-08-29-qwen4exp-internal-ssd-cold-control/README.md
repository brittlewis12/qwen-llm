# Flash-Next Internal SSD Cold Control

Decision: **GO** for internal placement. Run one charged full-shard prefetch
falsifier before closing storage source work.

## Asset Substitution

The internal data volume initially had only `116 GiB` free, while the Flash-Next
quant occupied `84 GiB`. The substitution therefore preserved data rather than
filling the volume blindly:

1. `/Users/tito/models/deepseek-v4-flash-0731-reap-k216` was copied to
   `/Volumes/wdblack/weights-archive/deepseek-v4-flash-0731-reap-k216`.
2. A full `rsync --checksum --dry-run --itemize-changes` comparison returned no
   differences; only then was the internal source removed.
3. The three Flash-Next shards were copied from the external archive to
   `/Users/tito/models/Qwen3.8-Flash-Next-UD-Q3_K_XL`.
4. A second full checksum comparison returned no differences. Both trees report
   `87,877,304 KiB`, and the internal volume retains `160 GiB` free.

## Protocol

- Source: `f719640`; the release CLI was rebuilt from a clean tracked tree.
- Device: Apple M4 Max with unified memory.
- Model: the checksum-identical UD-Q3_K_XL release, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
- Workload: the tracked natural N=512 prompt, one generated token, promoted
  strict router, first/warm/profile execution in each process.
- Order: external A1, internal B, external A2 with five seconds between arms.
- Cache state: before every arm, a temporary helper called the repository's
  targeted `invalidate_file_cache` on all three shards. It required every
  post-invalidation `mincore` report to contain zero resident pages. Its source
  was removed before the measured CLI was rebuilt; no machine-wide purge ran.
- KEEP: internal cold first-pass wall improves by at least 20% and 5 seconds
  versus the external bracket, with warm observer acceptance, raw coverage, and
  one output digest.

## Residency Proof

| Arm | Shard 1 after | Shard 2 after | Shard 3 after |
|:---|---:|---:|---:|
| A1 external | 0 / 669 | 0 / 3,050,736 | 0 / 2,440,928 |
| B internal | 0 / 669 | 0 / 3,050,736 | 0 / 2,440,928 |
| A2 external | 0 / 669 | 0 / 3,050,736 | 0 / 2,440,928 |

The before-state varied as expected after copying and prior arms; the
load-bearing fact is complete targeted eviction immediately before each process.

## Results

| Arm | First wall (ms) | First GPU (ms) | First outside GPU (ms) | Warm GPU (ms) | Profiled GPU (ms) |
|:---|---:|---:|---:|---:|---:|
| A1 external | 43,331.779 | 1,004.641 | 42,327.138 | 994.940 | 995.821 |
| B internal | 23,224.459 | 1,004.127 | 22,220.332 | 992.592 | 994.109 |
| A2 external | 44,406.617 | 1,005.442 | 43,401.175 | 991.907 | 993.471 |

External first wall averages `43,869.198 ms`; internal saves `20,644.739 ms`,
or 47.06%. It wins by 46.40% against A1 and 47.70% against A2. The external
range is only 2.45% of the bracket mean.

Outside-GPU time explains `20,643.825 ms` of the saving. First-pass GPU differs
by 0.09%, while warm/profiled GPU differs by 0.08%/0.05%. The placement result
therefore carries no warm-kernel claim. All observer ratios and raw timestamp
coverage passed, and every stdout has SHA-256
`e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9`.

## Next Falsifier

Keep the internal asset. Bracket internal demand paging, existing full-shard
prefetch, then internal demand paging, with targeted eviction before every arm.
Charge prefetch wall inside process-cold first-pass wall. KEEP only with at least
20% and 5 seconds of improvement plus unchanged warm acceptance and output;
otherwise close storage work without adding a bespoke range warmer.

Raw logs remain under
`target/profiles/qwen4exp-storage-internal-n512-20260829/`. Their hashes are
recorded in `results.json`.

Adversarial leverage review: `01a04de7-d66e-73f1-b3de-14ba60527c89`.

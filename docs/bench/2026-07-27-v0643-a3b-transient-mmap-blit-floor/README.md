# v0.643 A3B Transient Mmap-Blit Mechanism Floor

## Verdict

Sealed `inconclusive` after the exact P1 `A -> B` prefix. B reports one
timer-local major fault, so the runner persists the attempt, stops without a
retry or replacement, and performs no scoring. Authority, force authorization,
and successor authorization are all none.

This packet does not establish a mechanism pass or miss. Its two structurally
valid rows are bounded descriptive evidence only.

## Scope

- Exact cache-warm A3B Q4_K_M asset on the local M4 Max.
- A: admitted `parallel-pread` W4 into 733 independent offset-zero Shared
  destinations.
- B: one transient mmap-backed source window plus one tail source, 733 GPU blits,
  and the same 733 independent Shared destinations.
- Six frozen pairs were planned in `AB BA BA AB AB BA`; only P1 A and B ran.
- No production, Auto, loaded, storage-cold, Private-storage, sidecar, energy,
  serving, other-host, or other-asset authority exists.

## Integrity

- Source `e69df14` is a clean direct child of implementation `d859345`; it adds
  only the preregistration and runner and preserves the complete `crates` tree.
- Normal and optimized pure self-tests, ten direct lifecycle fault tests, the
  fresh release build, and the exact 13-test Cargo gate pass.
- Both model children exit zero, authenticate `pid=pgid`, are exactly reaped,
  and leave no process-group member. No signal, cleanup, retry, P2 conditioning,
  or inspection break occurs.
- AC power, thermal/performance state, memory, competing-process, compression,
  swapout, swap-occupancy, block-input, process-swap, and pageout gates are clean.
  All 1,350,985 model pages are resident before and after both children.
- All 733 resources and all 22,123,538,944 payload bytes verify in both arms.
- B proves source lifetime through retained command-buffer references, one exact
  window deallocator call, zero geometry mismatches, and zero live weak sources.
- The seal inventories 18 evidence members. Decision, inventory, completion,
  attempts, lifecycle, signal, artifact, and source/build identity bindings
  verify.

## Bounded Prefix

| Metric | A: W4 pread | B: mmap blit | B/A or A-B |
| --- | ---: | ---: | ---: |
| Ready wall | 530.984 ms | 542.067 ms | 1.020873x |
| Copy phase | 528.360 ms | 538.971 ms | +10.611 ms |
| Timer-local CPU | 2,098.103 ms | 451.027 ms | 0.214969x |
| Timer major faults | 0 | 1 | invalid B |
| Process page faults | 89 | 90 | descriptive |
| Maximum RSS | 44,314,394,624 B | 44,318,965,760 B | 1.000103x |
| Peak footprint | 22,189,997,640 B | 22,367,517,144 B | 1.008000x |

The single-pair ready delta is `A-B=-11,083 us`: B is descriptively 2.087%
slower while using 78.503% less timer-local CPU. This is not a median, order
effect, win count, reliability estimate, or mechanism decision.

B reports `90.627 ms` of GPU execution inside a `538.971 ms` copy phase. The
remaining `448.344 ms` includes host encoding, driver/submission and queueing,
synchronization, completion checks, and any coherence or residency handling.
The packet does not isolate those terms. The similar `451.027 ms` whole-ready
CPU total is suggestive but overlaps GPU work and is not a valid decomposition.

The one timer-local major fault is real for this attempt but unattributed. It
does not show that transient mmap sources intrinsically fault: complete mincore
residency and fatal VM gates remain clean, and the timer provides no phase,
address, or backing-object attribution.

## Leverage Consequence

Close the current transient-source Shared-destination mechanism-floor premise.
The observed B row misses the 112 ms portfolio gate directionally, but its fault
prevents a scored miss. Do not reopen by retrying this packet or by relaxing the
fault contract.

A materially different, independently justified diagnostic premise would need
both explicit major-fault attribution and a command-encoding/queue/GPU interval
decomposition. v0.643 does not authorize that work. Private destinations,
sidecar images, write-combined storage, and production policy remain separate.

## Artifacts

- Packet: `target/profiles/v0643-a3b-transient-mmap-blit-floor-p1/`
- Work: `target/profiles/v0643-a3b-transient-mmap-blit-floor-work/`
- Source: `e69df1484fd83e86960d898533bd16146fd24e4c`
- Implementation: `d859345118774b934218a7eb55460ef0870da882`
- Decision: `43c248613fe8da88e3323207542c908967a6e4bec061484a4842508088b5b659`
- Inventory: `fa86d92330ef44489fe4bfd5a723909c3b8d4edd054cfb9833380a1f8c5fa70f`
- Completion: `68e88f4032f150d0ef498c2cb080cf308318f8a8b6da153ae60d596e21b9c4b`
- Attempts: `ba3db094b791666d5fd8d9d92990be7cb6ec440a1deee99380be0f7ec1cd01bb`
- Lifecycle: `5ba3fc38369a6a88fc94748807c15053e1256ad449a8c4d194719fb6b713a8b5`
- Adversarial certification: `cx` session
  `019fa4ab-3694-71d2-b975-3f3173454ce0`.

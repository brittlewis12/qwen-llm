# V4.1 Retained-Window Partition Experiment

## Scope And Frozen Sources

Apple M5 Ultra, 256 GiB physical memory, operator-selected 240 GiB Metal working
set. Keep the same Maya-S six-shard artifact, batch-two CED, 16-unit grouping
budget, 4096-row Full/encoder chunk and synthetic seed-1 native input. Do not
substitute the separately qualified GSQ artifact into this comparison.

The implementation continues from qualified batch-two source
`9874c68b1602f8ff6058179a982c54775c22b638`. The preserved partial source was
copied into a fresh owned checkout, leaving the original checkout and frozen
references unchanged. Final comparison sources:

- P0, disabled soft target: `b70384066bf2bfe50fae86c513ea927afa23bde5`.
- P4, 4 GiB soft target: `7398cb765178ba3ca0ccc022e12d9faef766497b`.

Their complete difference is one constant in
`crates/qwen-llm/src/deepseek_v41/gguf/resident_storage.rs`:
`None` versus `Some(4_294_967_296)`. Arithmetic, kernels, work rows, command
grouping and admission are unchanged. The soft target limits coalescing
extension, not singleton eligibility or Metal's hardware limit. Oversized
tensors remain direct views; contained padding may reuse an existing window.
Other families retain the generic disabled-target wrapper.

The partition also changes mmap provenance, not just Metal resource lengths.
Any eventual gain would qualify this mapping strategy, not establish that
buffer length alone causes scheduling stalls. No residency set, pinning,
pre-wire, purge, forced eviction or new fit cap is introduced. Existing
4096-byte/window and 1024-byte/descriptor CPU accounting covers the temporary
bookkeeping; there is no new per-4-GiB price or duplicated mmap charge.

Frozen M5 executable SHA256s:

| Variant | Core Tests | Benchmark |
| --- | --- | --- |
| P0 | `52ad2d6acc844e0016f5c5a13d6c86b953e2a55f5c7c6f134d5f6f56b1b906d3` | `63779e87e6421311b9dae3c7a56891f8aa09c6718f65b9c49e4ec1c9a0e731c3` |
| P4 | `4a1feeddd3b95d65787db70150b3e00a9316b076e18793d7dcf9a154dd916fc4` | `98481b4caa27aef176732e739b493e1ee83eb51488cdee19e0ed25a15f677ce3` |

Both are frozen before build-lease release. Benchmark build/runtime identities
are clean and matching, without overrides. Metal compiler is 32023.921.6, SDK
27.0; the product tensor library still targets `metal4.0` at macOS 26.0.

## Actual Mapping Geometry

Metadata audit and actual one-time live-resource inventories agree. The live
inventory queries each unique retained Metal object's length once, normalizes
its ordinal and counts all 1002 tensor views without cloning scratch resources
or adding a per-command profiler.

| Quantity | P0 | P4 |
| --- | ---: | ---: |
| Unique retained objects | 44 | 85 |
| Largest object, bytes | 47881928704 | 4235247616 |
| Sum of unique lengths, bytes | 178187976704 | 178188648448 |
| Original-shared windows | 40 | 81 |
| Shared-EOF windows | 4 | 4 |
| Raw-file union, bytes | 178187370496 | 178187370496 |
| Repeated page span, bytes | 606208 | 1277952 |
| Private shadow, bytes | 0 | 0 |

The largest object falls from 44.594 GiB to 3.945 GiB; extra exposed span is
671744 bytes (656 KiB). This is neither a physical-residency measurement nor a
traffic/fit guarantee. Resident payload is 178187243968 bytes; Engram payload
161284798500 bytes remains disk-backed and excluded from resident windows.

The largest retained tensor is IQ3_S `blk.0.ffn_down_exps.weight`, shape
`[2304,5120,384]`, payload 1946419200 bytes, page-rounded singleton 1946435584
bytes. There are zero above-4-GiB singletons in this artifact; that observation
does not remove general oversized-singleton eligibility.

Planner layout SHA256s:

```text
P0 15a5757967c6dfb672e0447a8e306269d1bd04b2d01029d910eb78c67a4fa3f0
P4 9374a78088e90d5cef7e85816ddaf57862d39a3adf7574557d9b005decf812db
```

Realized tensor-to-resource binding SHA256s:

```text
P0 c497729d8179b89d907386413e95eb748ec2db886b99924696ce1880e1b25813
P4 95b8179b243f532226d43d95d34118e1aa3f353fe01714e07b12139bc0919ce8
```

## Correctness Gates And Fixture Repair

Each final variant passes 74 focused CPU checks. Both active source inventories
contain all 16 experiment sources and reject each individual stale-body control;
the historical 13-entry receipt remains unchanged. Independent source review
finds no concrete P1/P2.

Actual validation-on M5 execution passes two bounded GPU witnesses per variant:
overlapping mapping/deallocator independence and first-containing resource
association with an owning subview that survives dropping both weight/GGUF
owners. The latter performs a real post-drop GPU blit of nonzero data into a
guarded output and checks all bytes, not only retained pointer metadata.

The earlier v29 owner fixture correctly failed the production bounds guard:
it requested eight F32 values from a 16-byte bank. The v30 correction changes
only that test file: offset four bytes, three F32 values, independent expected
`[2.5,3.75,5.0]`, two guards on each side and a CPU negative for the old invalid
request. Nonzero fixture data is written before opening/stamping GGUF. Frozen
v29 sources and failure receipts remain intact; no production guard is relaxed.

Both final sources pass actual Full/CED at 2048 prompt rows plus 256
teacher-forced steps. All 257 complete 129280-logit vectors and all 56 causal
regions per position match bitwise. Each pair uses one import, at most one KV
state, no fallback and 11 completed CED commands, with unchanged work counters.
P0 also matches the old qualified batch-two report; P4 matches P0, for all 257
heads and complete state records. Sampled incremental Metal allocation is
2983034880 bytes, within the accepted 3022651392-byte bound for both variants.

Full-model report SHA256s:

```text
P0 ffb69ee4530bd73607bd064116d93111c9b9149ef76216d4f346e9143bd0085f
P4 d15a1ea6d60ec2ad339f74ee3f7c4c377a69ece9018313d1fa4e2a6a1ee9e4da
```

These ordered validation-on correctness timings are not performance evidence.
The old bridge checker counted matches without requiring them; the new
fail-closed checker asserts 257/257 heads, complete states and unchanged work.
Healthy and corrupted head/state/work controls verify that correction. Local
offline rechecking reproduces both model bridges and both layout digests.

## Instrumented P0-P4-P4-P0 Diagnosis

Each process imports once and runs 14 chronological Full/CED cases:
`ABABBABAABABBA`, with warmup AB, pilot ABBA and eight retained BAAB/ABBA cases.
All 56 endpoints match the archived 2K synthetic-input head; CED completes 11
commands without fallback and stays inside its sampled allocation bound. All
timestamp pairs are valid; Full records one diagnostic submit and CED 11.

These are explicitly instrumented, noncanonical diagnostics requesting
production timing, not validation-on runs. Known debug/injection/capture flags
are removed only at child launch; effective external injection is unattested.
All raw samples remain, including cold/pilot cases and stalls. There are no
percentage-free-floor or variance rejection gates, retries or cache purges.

Retained seconds, chronological within each policy:

| Process Order | Full | CED | Median Full / CED |
| --- | --- | --- | --- |
| P0 first | 3.642446, 2.238566, 12.722566, 2.861878 | 3.057172, 3.020055, 3.264812, 12.075593 | 3.252162 / 3.160992 |
| P4 first | 3.631827, 2.239180, 3.617333, 3.476736 | 3.720616, 3.438803, 3.605230, 4.464019 | 3.547034 / 3.662923 |
| P4 second | 3.634096, 2.255514, 3.616851, 3.477030 | 3.564323, 3.620776, 3.609617, 4.561604 | 3.546940 / 3.615197 |
| P0 second | 9.672397, 2.245287, 24.231628, 19.368454 | 3.268162, 3.341667, 3.232655, 7.734265 | 14.520425 / 3.304915 |

Across pilot/main calls, P0 Full GPU envelopes stay at 1.941-1.946 s while
driver intervals span 0.027-22.006 s; P0 CED is 1.621-1.654 s GPU and
0.721-10.065 s driver. P4 Full is 1.943-1.945 s GPU and 0.027-1.415 s
driver; P4 CED is 1.698-1.730 s GPU and 1.240-2.481 s driver. All warm
process read/page-in deltas and pipeline misses are zero.

Thus both P4 processes avoid the long P0 stalls in these cohorts, but typical
calls do not improve and CED GPU envelopes are slightly larger. This is a
reason to run matched uninstrumented confirmation, not a speedup claim.
The diagnostic matrix receipt SHA256 is
`94e08297809676759167cbf5a1ec1be99f643f767389e4b06ea6fbd9c0c1ea29`.

Independent evidence review found two offline acceptance holes: missing exact
joins between stderr native-wall values and report samples, and failure to
require a naturally completed controller disposition. The separate v31
recheck closes both without changing or rerunning the frozen observations.
All four cohorts reproduce their v30 checks, match every native-wall join and
event protocol/regime, and report `reason=completed`. One healthy and six
corrupt-copy controls respectively pass and fail, including altered native
wall, protocol, regime, timeout reason, endpoint hash and resource length.

## Uninstrumented 2K Confirmation

Repeat the same P0-P4-P4-P0 process order without diagnostic profiling. The
native v2 protocol requests canonical production timing; effective external
validation remains unattested. All 56 endpoint hashes, chronological schedules,
no-fallback checks, 11-command accounting, sampled bounds, clean identities and
qualified resource inventories pass. Every controller completes naturally;
all sampled pressures are normal. No diagnostic events appear in native stderr.

Retained seconds, chronological within each policy:

| Process Order | Full | CED | Median Full / CED |
| --- | --- | --- | --- |
| P0 first | 9.572018, 2.262159, 9.475646, 16.740290 | 5.527857, 11.034354, 19.845433, 30.209282 | 9.523832 / 15.439894 |
| P4 first | 3.529913, 2.237501, 3.638264, 3.482191 | 3.608012, 3.687281, 3.668068, 4.450876 | 3.506052 / 3.677674 |
| P4 second | 3.648015, 2.242523, 3.642387, 3.480384 | 3.468653, 3.659355, 3.617482, 4.564018 | 3.561386 / 3.638418 |
| P0 second | 21.646523, 2.241493, 18.478038, 10.762239 | 24.621247, 9.430623, 12.981856, 17.574091 | 14.620138 / 15.277973 |

Across the eight retained calls per layout/policy, Full/CED medians are
10.167128/15.277973 s for P0 versus 3.506052/3.663711 s for P4. All eight
retained P4 CED calls are faster than all eight retained P0 CED calls. Preserve
the pilot counterexample: a P0 CED pilot is 2.807655 s, faster than any P4 CED
call. This is a repeated tail/stall improvement in these 2K cohorts, not a
claim that smaller mappings improve every call or eliminate temporal confounding.
P4 CED still does not beat Full at 2K.

The native matrix receipt SHA256 is
`b98f89beddea67acedf43db591e24c87c51a6415327f65c636d07a4e3187b9c0`.
Frozen sources and binaries remain unchanged for the subsequent 8K
correctness/native followup. No main mapping or CED default is promoted from
these 2K results.

## Interpretation Boundaries

- Driver scheduling intervals include waiting/scheduling, not measured CPU busy
  or shader-active time. GPU, driver, inclusive wait and host wall overlap and
  must not be added into a decomposition.
- Operation roles aggregate across every dimension variant, not a last-writer
  dictionary. Process resident size/physical footprint are current samples,
  not peaks; sparse host health cannot bound transient driver costs or swap.
- Zero CPU process reads/page-ins does not prove absence of GPU faults or DRAM
  traffic. Native synthetic routes are not a real-text workload distribution.
- This qualifies the final sources at 2K only. It does not extend the new
  mapping strategy to other lengths, artifacts, full context or quality tests.
- Main does not ship the experimental CED executor or these mapping changes.
  The frozen source and private evidence remain available without a push.

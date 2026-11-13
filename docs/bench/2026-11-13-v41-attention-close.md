# V4.1 attention gate: source/profile pin and final measurements

**Completion:** [the frozen served gate passes](2026-11-13-v41-attention-final.md). Only the final B arm ran in the closing stage; prior observations stand.

**Lead correction, 2026-11-13 23:00 UTC:** the loaded-host native packet is
**unresolved noise in both directions, not a pass**. Its mean point ratio is
1.064904x, but BA/AB/BA are 0.931764/1.710896/0.862674x and the candidate
root is 44.912s versus the standing roughly 21.5s. The independent quiet-host
wide-PV native packet was 20.755584s / 20.505967s = **1.012173x**, 3/3
positive pairs, with eight full signatures bitwise (iter, nice 10, recorded
in [the original packet](2026-11-12-v41-prefill/README.md#wide-pv-native-wall--both-cells-pass)).
These observations supersede every earlier native-pass interpretation below.
Raw timings and historical threshold outputs remain intact. The lead directs
completion of only pair 3 B and the unchanged frozen serve rule for pass/fail.


Start 21:01:00 UTC; hard report by 23:01, no extension. Measurement cutoff 22:49
reserves cleanup/records/contracts. The lead clarified that source `67b3ebaf4`
plus the same iter/release/test profiles and byte-identical metallibs is the
program pin; executable SHA identity is not required. Record original,
previous-rebuild and current executable hashes. No Rust/kernel/probe changes
are made for this measurement, and pair 1 AB stands exactly as recorded.

## Artifact lifecycle

Rebuild source `67b` at the original M5 worktree/target paths. Require all nine
main/research/tensor metallibs across profiles to match the old hashes.
Copy each executable into attention-close/pins outside target, verify its
SHA and make it read-only. Both arms use those same archived executables.
Keep pins while any gate remains open across stages; cleanup removes bulk
build output, worktree and bundles, not open-gate pins. Delete pins only after
the gate closes. The manifest keeps build and archived paths and all hashes.
The metallibs are embedded with `include_bytes!`; the archived executables
retain their GPU code when the original target directory is removed.

## Unchanged measurement protocol

Use the prepared finish-stage protocol: purge Engram to zero resident pages before EVERY fresh
single native 48K process; warmup AB and retained BA/AB/BA; nice 0 verified by ps,
60s continuously<=1 before each load, allow2/abort3. Record per-run load,
Engram prepare wall and completed-submit encode time, process reads and
endpoint digest. Existing full-state/following-step witnesses from the same
source/profile/metallibs stand; new endpoints must match the prior golden.
The ratio of retained mean A/B wall times must be>=0.98; also report the
geometric pair ratio. One<=15m hold if it fits, otherwise checkpoint at a
pair boundary and use a lease per remaining pair. Do not repeat completed runs.

If native passes, finish fresh pair 2 BA and pair 3 AB, one ARM per<=15m lease,
with symmetric paired 4K model warmup, both GPU witnesses, Engram purge to zero resident pages,
fresh server, greedy/root reply cap 32, three 48K agents, 60s windows/15s ramp, active 4,
sample-workers 3/fair 1 and the original thermal rules. Each arm has its own
860s internal budget inside 885s/15m. Warmup and measured process reads remain
separate. Refuse arm overwrites; require identical pair metadata and plan.
Both arms differ only by SHORT=0/PV_WIDE=0 versus SHORT=1/PV_WIDE=1.
MASKED_REINDEX and MID remain1. Pair 1's old executable hash is legitimate
under the explicitly approved source/profile+metallib pin.

Across complete AB/BA/AB: geometric means TPS >=0.98, root prefill >=0.98,
short 128..511 and 1K-class 512..2000 prefill AND client TTFT each>=1.00, with
at least one matched/observed call per bin per pair. All equal-new-row
completed offered calls, roots, counts/null TTFT, compiler/load/thermal and
I/O evidence retained. Native pass plus served pass => ready to land.
Purged root failure drops wide PV; short remains default-on only if its own
150/256 cells hold. Prepared fallback uses PV=0/SHORT=1 and scoped MID
false/true at 80K/sub-512 rows, exactly main versus short eligibility, with
full paired signatures. No wider sub-bisection or next-lever implementation.
If the queue prevents completion within the stage, report; do not extend.

The independent Engram cache is already composed on local main `c2935fa36`,
with contracts passing `2e8efa847`. Measurement remains on `9877/67b` as authorized.

## Planning only: lead-forwarded waste audit

For the next stage, cooperative experts shares priority with the forwarded
perf/waste-audit findings (not acted on or remeasured here): score-to-softmax
round trip~34/17ms at 2048@48K/1024@80K; scatter_offset_f32_to_f16 in indexer
and expert input plus v41_sorted_pack ~71/34ms GPU (4.2% of chunk GPU); three
apparently duplicated attention_index_heads_rows_mma append dispatches~7.5ms,
0.68%, same inputs and no intervening write. Wide PV already covers PV to
normalization. Audit reference: perf/waste-audit:docs/bench/2026-11-13-waste-audit.md.

## Rebuild and bitwise witnesses

Source: `67b3ebaf44ceebdd1b98c914f3730e12da810de0`. Builds use the original iter, release and test
profiles at the original worktree and target paths. Cargo commands run through
`nice -n 10`; the immediate launcher samples are before `nice` exec, so their
recorded zero does not establish the Cargo process niceness. Both GPU fixture
processes were independently checked at nice 0. Both passed with required Metal
and `MTL_DEBUG_LAYER=1`, including all 15 wide-PV exactness markers.

The following SHA-256 values record every executable generation; the prior
rebuild was stopped on a literal-hash check before the lead clarified the pin.

| Artifact | Original | Previous rebuild | Current archived |
|---|---|---|---|
| iter | `19d9adf78d3a53f7f8d209386a1c06e0d4da10162524cdcba123cabfc1facaef` | `1fa66184f9a13bde213e09ee2f459e4268a3bb1649790132a3a89b05ee9f7d87` | `a896361f30248b7e6ef0f6848c85cb5e90abef6bd2b056ef61e454ee803388b6` |
| release | `5c538677b90c074f36c40ec44b9616942e71ed392ff3368154dbb72a153d575f` | `78131445f024043993933239282ca69adafb105a332eafbb8a1982a49db84d4a` | `fce29b316a826b22ddd5649e84b020bdbf40e81f9b8137145617c0b79859797c` |
| test | `a83fa4a6639aeb0d8579f1cdc8413e22ef8d9a77152aec61fb45d8e850ddfc10` | `b253930bc372e38790453ac06b861983234c51d695c7f36928d1b41cbe7f14c6` | `e86a0ed37aed6f82e6d88378b9ca10995568a3a4b28c901bfd9d9871aa5cced6` |
| tokenizer | `eb94be08eb27fb5dfb0feaa423c83f958323970686e946b9f99e752fd7fc000e` | `eb94be08eb27fb5dfb0feaa423c83f958323970686e946b9f99e752fd7fc000e` | `eb94be08eb27fb5dfb0feaa423c83f958323970686e946b9f99e752fd7fc000e` |

All three profiles have the same metallib hashes, identical to the original:

| Metallib | SHA-256 |
|---|---|
| kernels_tensor.metallib | `43ae770545fa60f1f51d2bb3220a93c0aaf3bfc13d382a99ea2345920eaf184c` |
| kernels.metallib | `4c006656e021a51266a79cfcea38a97c36fac9b5c08b5ad5ccd55e6e8bc15e15` |
| kernels_research.metallib | `be63c8ae9332529da45c21328333191834bd767917f5f9bc55d94eb3702b557f` |

The archived binaries are under
`metagross:~/qwen-runs/2026-11-12-v41-prefill/attention-close/pins/`.
The full manifest is `build-functional-done.json` in that run directory.
A missing harness-bundle path stopped the first controller before any model
load. The transferred source bundle was linked to that path and scheduling
restarted without repeating any measurements.

## Composed branch checks

Fresh `scripts/check-contracts.sh --full` first failed in the unchanged main
probe `examples/glm_q6_probe.rs`: its unconditional call to
`diagnostic_q6k_r1` requires `perf-ablations`, but Cargo had no corresponding
example declaration. This was raised on SRV-2K0Y (RYKTR82QETD596DD) and fixed
separately in `e64d364d5` by declaring the existing feature requirement. No test
was ignored and no measured attention code changed. The full contract retry
passed for commit `e64d364d5`, tree
`968e09349aa1f234d6c9aac38a05f5ba751c4e46`, including the generated ENV check,
CPU inventory, workspace test-profile compilation, source-cache tests and
release/README checks. The first failure remains in the raw contract log.

The targeted iter-profile check of `glm_q6_probe` with `perf-ablations` also
passed, so the feature declaration preserves compilation of the actual probe.

## Purged native root result

The frozen native gate **passes**: mean main 47.8270s / bundle 44.9120s =
**1.064904x**, floor 0.98x. Geometric mean of pair ratios: 1.112053x.
Retained BA/AB/BA pairs are 0.931764x / 1.710896x / 0.862674x; one of three favors
the bundle. No observation is filtered. All eight endpoint hashes match the
previous golden (`a4c13784ef52ffb732b3620e94e4d215b33926c29dd84be2a6204b17aea746ab`).
An independent reduction checks raw SHA-256 values, the eight identical
endpoints, actual nice 0, max thermal <=2, zero-page purges, and the final
continuous 60s settle at <=1 before every load. Prior full-state/following-step
paired witnesses remain those of the identical pinned GPU program.

| Pair/order | Arm | Wall s | Engram prep s | Completed-submit encode s | Load (1m) before → after | Sampled process read bytes | Max thermal |
|---|---|---:|---:|---:|---|---:|---:|
| warmup AB | a | 54.439069 | 5.945468 | 6.594666 | 23.91 → 24.56 | 166988447744 | 2 |
| warmup AB | b | 53.829119 | 6.352373 | 6.821413 | 24.69 → 31.72 | 168350732288 | 0 |
| 1 BA | b | 43.201008 | 7.286036 | 8.744070 | 45.13 → 55.28 | 13104295936 | 0 |
| 1 BA | a | 40.253152 | 6.726028 | 7.865286 | 55.93 → 55.19 | 13103611904 | 0 |
| 2 AB | a | 48.939206 | 5.641880 | 5.698442 | 29.18 → 28.69 | 171385593856 | 2 |
| 2 AB | b | 28.604424 | 4.307222 | 4.706500 | 16.26 → 14.49 | 13103640576 | 0 |
| 3 BA | b | 62.930679 | 8.050408 | 7.884928 | 20.67 → 23.61 | 172307513344 | 2 |
| 3 BA | a | 54.288647 | 6.604858 | 7.037885 | 26.60 → 29.91 | 166279540736 | 0 |

Full 1/5/15-minute load vectors and half-second run traces remain in each
`purged-root-p*-*-result.json` / `-thermal.json` under the run directory.
Per-run disk reads are maximum live process counters sampled every 0.5s, not
Engram-only bytes. Reads and load vary sharply even after uniform Engram purge;
the 6.49% native aggregate is a gate result, not isolated attention attribution.
There is no quality relaxation or adjusted timing threshold.

## Serve scheduling checkpoint

The first served launch stopped before warmup/model loading because the private
controller opened the arm's own `.log` as its outer lease log. The public
harness's no-overwrite guard correctly rejected it. The failed launch log is
preserved as `finish-serve-pair2-b-bootstrap.log`; no model/warmup directory or
arm summary existed. The private scheduler now uses `.lease.log`, preserves
completed native/arm checkpoints on restart, and resumed with native unchanged.
This changes scheduling only; no benchmark threshold, warmup, measurement
binary, order or sampling rule changed.

<!-- SERVE_RESULTS_START -->
## Served observations

Latency ratios are main/bundle; TPS columns are delivered tokens per second.
All completed paired observations are retained. The short bin uses actual
128..511 rows; the 1K bin uses 512..2000 rows. Counts are matched calls / pairs
with observed client TTFT. Null TTFT is retained in raw calls, never imputed.

| Pair/order | Root prefill | TPS main / bundle | Short prefill / TTFT | Short counts | 1K prefill / TTFT | 1K counts |
|---|---:|---:|---:|---:|---:|---:|
| 1 AB | 1.166676x | 87.350000 / 88.316667 | 1.053301x / 1.032870x | 6 / 6 | 1.045457x / 1.058349x | 3 / 3 |
| 2 BA | 0.902509x | 87.083333 / 88.100000 | 1.018921x / 0.983220x | 6 / 6 | 0.995344x / 0.987286x | 3 / 3 |
| 3 AB | 0.932233x | 87.600000 / 86.833333 | 1.023664x / 1.007011x | 6 / 6 | 1.006919x / 1.009326x | 3 / 3 |

Process counters are maximum live values sampled every 0.5s. Warmup and
measured-server reads are separate; bytes are total process reads, not an
Engram-only counter. Every listed arm has a zero-page purge and actual nice 0.

| Pair | Arm | Server read bytes | 4K warmup read bytes | Root client TTFT s | Load (1m) before → after | Max thermal (server / warmup) | Compiler observed / overlapped |
|---|---|---:|---:|---|---|---|---|
| 1 | a | 52267646976 | 159335890944 | 33.048588, 32.029116, 29.378886 | 21.26 → 15.86 | 1 / 0 | True / True |
| 1 | b | 33156030464 | 496418816 | 28.110331, 27.191419, 25.680483 | 14.48 → 12.44 | 0 / 1 | True / True |
| 2 | b | 34273374208 | 159334612992 | 30.966365, 29.407303, 40.897552 | 19.99 → 14.89 | 0 / 1 | False / False |
| 2 | a | 189054726144 | 159337021440 | 37.820759, 26.780003, 26.810891 | 12.17 → 17.10 | 0 / 0 | True / False |
| 3 | a | 188321980416 | 157310173184 | 31.039243, 25.165384, 25.605822 | 13.11 → 13.83 | 0 / 0 | True / True |
| 3 | b | 33103966208 | 157293551616 | 30.253172, 31.344943, 26.141006 | 18.37 → 20.88 | 0 / 0 | True / True |

Independent three-pair result: **PASS**. Loaded native mean point 1.064903853x (unresolved noise, not a pass).

| Metric | Geomean ratio | Required |
|---|---:|---:|
| root | 0.993822521x | 0.98x |
| tps | 1.004618096x | 0.98x |
| short_prefill_ratio | 1.031850574x | 1.00x |
| short_ttft_ratio | 1.007496626x | 1.00x |
| 1k_prefill_ratio | 1.015682641x | 1.00x |
| 1k_ttft_ratio | 1.017890587x | 1.00x |

<!-- SERVE_RESULTS_END -->

ENV was regenerated in this stage: 828 knobs, no polarity conflicts. Only
the source-revision header changed. The other pinned native cells and their
full-state/following-step bitwise witnesses stand as recorded in
[the bundle packet](2026-11-13-v41-attention-bundle.md); this stage replaces
its unpurged native root timing with the per-run-purged packet above.

## Bounded-stage close: gate remains open

At 22:39:23 UTC the final B arm was still queued behind Ghost stream work,
with 576.8s remaining before the declared 22:49 measurement cutoff and a
600s minimum arm admission budget. Exact controller PID 81921 was interrupted;
it stopped its exact queued child 63148. Neither B warmup nor model loading
had begun: no B process record, arm directory or measured log existed. Both
PIDs were subsequently absent. No completed arm is discarded or repeated.

The result is **incomplete, not ready to land**: native passes, pairs 1 AB
and 2 BA are complete, and pair 3 A is complete. The frozen three-pair served
rule cannot be evaluated yet. No candidate is promoted or withdrawn on this
incomplete packet. Both separate default-on candidate flags and their `=0`
rollbacks remain staged on the branch.

The independent partial audit checks every completed arm's binary pin, plan,
greedy fields, root reply cap, actual nice 0, zero-page purge, paired 4K full
signatures and both GPU fixtures, including all 15 wide-PV exact markers. All
five arms pass controls. The older pair 1 summary lacks newer metadata keys;
its original plan and launch files provide the same strict 60s window, seed,
plan hash and cap checks. No missing value is invented and no metric gate is
relaxed. `independent-partial-result.json` reports `passed: null`.

After measurement stopped, the branch was rebased onto main `8b765391e`
(GLM KDA and waste-audit landings). The only conflicts were two generated
ENV revision headers; regeneration resolves both. Final ENV: 830 knobs, no
polarity conflicts. There is no production attention source conflict; the
upstream V4.1 GPU-step changes add optional causal diagnostics. Pinned model
measurements and archived executables remain the approved 67b program.

The earlier full contracts passed `e64d364d5`; the five-arm checkpoint also
passed contracts at `302d24186`, tree
`55d5d279d44c4080801fe81c6e8635aa464dd511`. The final composed-tree contract
result is preserved in `attention-close-composed-contracts.log` and its
result JSON in the run directory. The separate probe feature fix is retained
through the rebase (now `761a48c6d`). No new ignored tests, pushes or merges.

M5 cleanup removed the private worktree, bulk target under attention-bundle,
and both transferred bundles. `cleanup.json` records no remaining lane
processes. The four executables under `attention-close/pins/` and manifests
are deliberately retained for this open gate; all hashes were rechecked
after target/worktree removal (`retained-pin-check.json`). Raw records remain
on M5 and have local evidence copies. The sole local transfer bundle is
deleted; the shared local MBX target is retained.

Next bounded action: restore the Python harness from `cf6652fd1`, run only
`finish-serve-pair3` B (`--order ab --only-arm b`) with the archived binaries,
one <=15m lease, inherited FD, nice 0, symmetric warmups and zero-page purge.
The exact argv and required environment are in `pending-serve-arm.json`. Use
a distinct or append-only outer lease log. Native and all five completed
served arms stand. Then apply the unchanged full three-pair rule. Cooperative
experts and the forwarded waste candidates remain planning only.

# Native dense Q8 CLI: resource confirmed, latency HOLD

## Disposition

The ordinary-CLI packet misses its frozen latency floors. No default native-Q8
allowlist expansion, kernel change or repeat follows. The separately proven
**3,734,732,800-byte Metal allocation reduction** remains valid and reproduces in
all eight CLI children. The existing `QWEN_NATIVE_QUANT_EMBED=1` force path remains
available; this packet does not qualify automatic dense Q8 selection.

The large fixed-order constructor difference in the resource oracle does not
translate into a comparable process endpoint gain. Resident-byte removal and
constructor-only observations must not substitute for end-to-end measurement.

## Contract and provenance

Runtime binary `6d59201a`, same in both arms. Test-only resource/bitwise oracle
`6e36f048`: `docs/bench/2026-09-07-native-q8-embedding/RESULT.md`.
Installed Qwen3.8-27B-Q8_0, raw complete code ChatML prompt matching the oracle,
25 tokens, no tokenizer-added special tokens, greedy, no drafter/MTP mode.
Bound MTP weights are present but not executed. Fresh-serving packed is explicitly
off and is not part of this ordinary-CLI experiment.

Two separately reported output-limit cells, 1 and 128, each A1/B1/B2/A2. A forces
native embedding `0`, B forces `1`. Parent timing measures first nonempty stdout bytes
and process exit. Child request, allocation and PSO telemetry support attribution
but are not substituted for parent endpoints. All processes are disposable and
all GPU work is serial.

File-cache-warm contract: every child reports source residency 100.0% and skips
prefetch as already warm. Before each child require two-second compression/swap
counter stability and no thermal/performance warnings. Any whole-child global
compression or swap growth invalidates timing and stops the packet. Pageouts,
pageins and decompressions are recorded, not retrospective zero-count gates.

All eight children pass these host/file checks. Compression, swapin, swapout and
pageout growth are all zero. Decompressions range 0-16 pages. These host-wide
counters do not establish perfectly equivalent scheduling or load conditions.

Frozen gates: first stdout >=15% faster in both orders for both cells; full process
>=10% faster for output1 and >=5% for output128; endpoint control spreads <=5%.
Output128 loaded generation median regression <=3%, control spread <=5%.
Outputs, token hashes, usage and terminal reasons must agree; actual model-ready
allocation must remove >=99% of the independently proven one-tensor reduction.

## All parent endpoint rows

Times in ms. A/B use the same stdout pipe observer and private artifact files.

| Output limit | Arm | First stdout | Process exit |
| --- | --- | ---: | ---: |
| 1 | A1 | 3798.104 | 4122.313 |
| 1 | B1 | 3511.785 | 3823.003 |
| 1 | B2 | 3529.012 | 3836.931 |
| 1 | A2 | 3735.631 | 4045.463 |
| 128 | A1 | 3779.286 | 11629.612 |
| 128 | B1 | 3819.155 | 11635.245 |
| 128 | B2 | 3816.781 | 11633.372 |
| 128 | A2 | 4010.517 | 11878.044 |

| Cell / endpoint | Median A -> B ms | Observed saving | Paired savings AB / BA | Control spread | Gate |
| --- | --- | ---: | --- | ---: | --- |
| 1 / first stdout | 3766.867 -> 3520.399 | 6.543% | 7.538% / 5.531% | 1.672% | misses 15% |
| 1 / process exit | 4083.888 -> 3829.967 | 6.218% | 7.261% / 5.155% | 1.900% | misses 10% |
| 128 / first stdout | 3894.902 -> 3817.968 | directional only | -1.055% / 4.831% | 6.118% | spread/floor fail |
| 128 / process exit | 11753.828 -> 11634.308 | 1.017% | -0.048% / 2.060% | 2.136% | misses 5% |

The longer cell also fails first-output stability. No warm-only reclassification,
averaging across cells, additional conditioning or gate relaxation follows.

## Resource and correctness results

Every A child reports model-ready Metal allocation 32,321,585,152 bytes; every B
reports 28,586,852,352: exact reduction 3,734,732,800 bytes in both orders and cells.
The actual conversion ledger changes from one input tensor, 1,350,860,800 source /
5,085,593,600 converted bytes, to zero conversions. The remaining model-copy path
and output head are unchanged. This is not a peak-RSS or physical-residency claim.

Output bytes, generated-token hashes, prompt/output counts, transition counts,
stop reasons, build commit and source-state identity agree within each cell.
Both cells actually emit their full requested output counts. Existing native
selection logs confirm rollback-disabled versus forced on the same Q8 matrix.
The preceding state oracle supplies bitwise full-logit/KV/GDN evidence through 64
tokens; CLI output equality extends to 128 tokens, not 128-token full-state equality.

## Child phase accounting, not endpoint substitutes

| Output | Phase | Median A -> B ms |
| --- | --- | --- |
| 1 | Runtime + model load | 3376.733 -> 3161.009 |
| 1 | Prefill | 325.698 -> 296.363 |
| 1 | Generation | 1.088 -> 1.392 |
| 1 | Complete request | 350.228 -> 318.784 |
| 128 | Runtime + model load | 3511.010 -> 3450.944 |
| 128 | Prefill | 328.619 -> 307.284 |
| 128 | Generation | 7536.467 -> 7507.976 |
| 128 | Complete request | 7887.330 -> 7838.210 |

Output128 generation improves an observed 0.378%, with 0.201% control spread,
passing the warm guard. Output1 generation has no target transition and is too
small/noisy for percentage authority. Request timing encloses its phases; do not
sum request wall with prefill/generation, or PSO costs with enclosing phases.

Existing PSO telemetry records 18 prefill misses, about 1.05-1.19 ms, and 0 or 6
generation misses. Total miss wall is about 1.05-1.68 ms. This cell does not justify
a generic PSO optimization project or explain the earlier serving outlier.

The current cold short-code shape is now priced: roughly 3.4-3.5 s in runtime/load,
0.33 s prefill, and 7.54 s generation at 128 outputs. Native embedding saves bytes
but leaves most of those costs intact. The remaining load interval is not yet
split into enough subphases to authorize broader copy scheduling or layout work.

## Review, attempts and artifacts

Independent review accepts resource PASS / latency HOLD and rejects projecting
constructor-only timing. Before any parallel-copy proposal, source/phase accounting
must expose a large enough serial allocation/copy interval; the byte ledger alone
does not establish its time. No MTP, drafter, sampled, tied-model, storage-cold or
server-startup qualification is inferred.

An initial static prompt file lost trailing blank lines. The CPU byte guard catches
this before any child launches. The original file remains; `check_input.py` writes
the canonical oracle bytes to a new file. No measured request or gate changes.
All eight owned children exit normally with 0; no cancellation or forced kill.

Raw: `target/profiles/native-q8-cli/` contains frozen `PROTOCOL.md`, both prompt
files, `check_input.py`, `probe.py`, `score.py`, every child directory and
`summary.json`. Each child retains command/environment, preflight and post-host
snapshots, stdout/stderr, child timing and parent result. The scorer preserves
all phase/PSO/allocation rows before reporting failed latency gates.

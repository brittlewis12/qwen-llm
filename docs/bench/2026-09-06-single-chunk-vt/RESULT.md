# Single-chunk VT and Qwen3.8/Q8 restored tails: KEEP

## Surviving scope

- `2ae6f191` adds an explicit dense single-chunk scratch plan with one
  transposed-V layer slot. Default constructors and multi-chunk VT retention
  are unchanged. The plan, constructor, and runtime entrypoint share the mode.
- `f0d1d67b` experiments with bounded restored packed prefill. Final
  `f140c206` qualifies serving selection to the validated Qwen3.8 template,
  Q8_0 LM head and dense-27 geometry, no drafter, current greedy request,
  non-exact restored boundary, and **7-32 consumed suffix tokens**.
- Query width is the actual suffix; key extent still covers the full prompt.
  Price must fit 128 MiB. Unsupported/oversized plans retain the old path;
  optional admission denial or allocation failure falls back to the original
  serial constructor. The prepared lookup pins the consumed boundary throughout.
- Fresh/exact hits, other profiles, <=6/>32 suffixes and direct sampled/drafter
  selection are unchanged. Q4 serving selection is HOLD, not promoted from its
  favorable aggregate. No serve scratch-lifetime change or new tuning switch.

## Storage and correctness proof

The preceding screen failed because 16 retained VT layers dominated query-32
scratch. For a single block, each layer's VT consumers finish before the next
layer overwrites a shared slot. Scatter, prefix transpose and all query-tile
consumers use serial tracked compute encoders. Diagnostic phase flushes commit
and wait. This uses existing GPU ordering, not new per-layer CPU waits. All
three VT views use slot zero only in the new mode. Per-call, per-layer validity
starts at zero; the whole restored V prefix is rebuilt for each layer.

Single-chunk execution rejects empty/oversized/uncovered spans, hidden or
attention capture and custom tails before mutation. Packed verification rejects
the new mode. Plan construction rejects non-dense, zero-width, non-matrix and
speculative use. Ordinary plans preserve their allocation inventories.

Release Qwen3.6 Q4 and Qwen3.8 Q8 use the same 8,840-token fixture, restored
8,808 consumed / 8,809 matched including pending, followed by 32 forwards:

| Quantity | Full VT | Single VT |
| --- | ---: | ---: |
| Actual Metal scratch bytes | 323,256,320 | 51,691,520 |
| Priced upper bound | 323,907,240 | 52,342,440 |
| VT slots | 16 | 1 |

This removes 271,564,800 actual scratch bytes, meeting the unchanged 128 MiB
gate. **The served baseline used no matrix scratch for these serial tails:**
the new serving path spends a bounded workspace to reduce TTFT. Do not portray
the isolated full-VT saving as a memory reduction versus serial serving.

Full-VT, single-VT, and a second invocation reusing NaN-poisoned VT storage have
bitwise-identical packed logits and complete KV/GDN snapshots on both targets.
Invalid empty/33-row invocations leave persistent state unchanged. Against
serial no-tail prefill, final-logit cosine is Q4 `0.9999991016`, Q8
`0.9999992798`; worst per-layer new-KV/GDN/conv cosine is `0.9999987589` /
`0.9999989203`. Restored KV prefix bytes are unchanged, and the next 64 greedy
tokens agree per target. Phase screens are 4.840x / 7.728x including candidate
construction, **not** end-to-end speedup claims.

Storage sharing is bitwise relative to existing packed prefill. Packed versus
serial is numerical, with greedy continuation witnesses, not globally bitwise.
Sampled requests are not selected, but can inherit a numerical checkpoint from
an earlier greedy request. **No distributional-equivalence claim is made.**

## End-to-end packet

Baseline `2ae6f191`, broad candidate `f0d1d67b`; release A-B-B-A with two
disposable server processes per arm for each target. Ten sequential real SSE
requests per process: natural 8,810-token seed, exact hit, short blue reply,
compact 128-output code/prose turns, forced sampled-green witness, longer
code/prose suffixes, packed suffix guard, and fresh short guard. First string
delta and wall through `[DONE]` include restore/allocation/prefill/publication;
model loading precedes the endpoint clock. No process-cold or OS-cold claim.

All **80** corresponding response texts/hashes, usage/cache counts, completion
statuses and reasons agree per target. Candidate logs select exactly widths
21/27/27 on Q8 and 21/27 on the broad Q4 candidate; no fresh/exact/sampled row
is directly selected. Final profile validation adds 20 equal responses:
Q8 retains 21/27/27; Q4 selects none. Final timings are not pooled into A-B-B-A.

### Qwen3.8 Q8: promoted endpoints

Median milliseconds; percentages are reductions, not multipliers.

| Request | TTFT A / B | Full wall A / B | TTFT saved | Wall saved | Wall control spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blue, 2 output | 1302.024 / 320.824 | 1406.235 / 419.354 | 75.360% | 70.179% | 1.993% |
| Compact code, 128 | 1655.930 / 326.112 | 9560.182 / 8150.475 | 80.306% | 14.746% | 0.766% |
| Compact prose, 128 | 1659.369 / 349.062 | 9518.522 / 8187.976 | 78.964% | 13.978% | 0.906% |

Both orders clear the frozen gates: blue wall saves 70.633% / 69.716% and TTFT
75.644% / 75.072%; code wall saves 15.225% / 14.263%, prose 14.176% / 13.779%.
The selected rows' TTFT control spreads are also below 2%. This is reduced
prefill work cost inside restored requests, not faster serial decode.

Every remaining Q8 row is reported here. Positive changes are regressions;
these unselected rows confer no causal speedup authority:

| Request | TTFT change | Wall change | Wall control spread |
| --- | ---: | ---: | ---: |
| Fresh long seed | -0.709% | -0.706% | 2.633% |
| Exact long hit | +0.123% | -2.483% | 4.493% |
| Sampled green | -1.540% | -1.930% | 3.961% |
| Longer code suffix | -0.759% | -0.374% | 0.498% |
| Longer prose suffix | -0.012% | -1.042% | 0.788% |
| Packed suffix guard | -2.256% | -2.694% | 0.560% |
| Fresh short guard | -1.058% | -1.272% | 2.269% |

All unchanged full-wall medians satisfy the 3% guard. Exact-hit TTFT has
13.533% control spread and no stable effect authority. The sampled color turn
is a small regression witness, not broad sampling validation.

### Qwen3.6 Q4: broader selection held

Positive changes are regressions. These are broad-candidate observations;
final production Q4 selection is disabled.

| Request | TTFT change | Wall change | Wall control spread |
| --- | ---: | ---: | ---: |
| Fresh long seed | -1.108% | -1.103% | 3.534% |
| Exact long hit | +4.873% | -0.687% | 5.022% |
| Blue, 2 output | -63.108% | -56.728% | 3.035% |
| Compact code, 128 | -69.986% | -10.710% | 0.308% |
| Compact prose, 128 | +1.069% | +2.215% | 0.871% |
| Sampled green | +2.161% | +2.895% | 2.660% |
| Longer code suffix | +1.124% | +2.235% | 1.388% |
| Longer prose suffix | +0.038% | +0.513% | 2.092% |
| Packed suffix guard | +1.841% | +2.305% | 0.129% |
| Fresh short guard | -1.718% | -2.034% | 0.636% |

Compact-code wall saves 13.295% in the first order but only 8.133% in reverse,
missing the required 10% in **both** orders. No threshold relaxation, extra
timing attempt, or aggregate-based override. Q4 remains a dissimilar
storage/numerical witness, not a promoted serving lane.

Q4 compact prose restores the preceding prompt boundary at 8,860 rather than
the completed boundary at 8,988, requiring 153 forwards. It therefore does not
exercise the new selector, unlike Q8's 27-row prose suffix. The cause of the
different reuse boundary is unresolved; do not label it template trimming or
an allocator effect without further evidence.

## Validation, review and attempt accounting

- Final `f140c206` passes 374 CLI tests (nine ignored). CPU plan tests establish
  that only VT layer storage changes; existing default allocation residuals
  remain intact. Q4/Q8 release Metal oracles run separately.
- Pure injected faults cover optional admission denial with unchanged serial
  re-admission, both-denied and signal-error cases, failed candidate allocation
  returning the serial constructor's untouched result, no retry after success,
  and propagated serial failure. This is mocked control-flow coverage, **not**
  a real driver OOM or GPU recovery experiment. No host OOM is provoked.
- Independent Luna audit/review/disposition jams use cx session
  `01a07893-0d8c-7c12-a567-07f5c8c1331e`. They challenge synchronization,
  allocation pricing, pending positions, fallback ownership, sampled provenance,
  and model-scope narrowing. GPU ordering, not mandatory CPU waits, is the
  corrected synchronization argument.
- A pre-existing socket-mode test races immediate nonblocking accept after
  connect. `2e2279df` adds bounded readiness waiting in that test only. The
  failure and repaired run are retained in session logs; production HTTP is
  unchanged. A metric-only scorer syntax error launches no extra benchmark;
  it is repaired without altering data or gates.
- No GPU timing is parallelized, no lease or template gate is bypassed, and no
  whole-model wiring is used. All owned servers unwind cooperatively via SIGINT.

Raw audit/protocols, plan/state oracle logs, all endpoint manifests/requests/SSE,
host reports, metric-only summaries and final checks remain in the dedicated
worktree under `target/profiles/single-chunk-vt/`. Acquisition uses the tracked
`scripts/serve/request_probe.py`; no new public benchmark framework is added.

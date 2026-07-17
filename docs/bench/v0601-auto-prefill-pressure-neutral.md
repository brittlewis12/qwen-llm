# v0.601 Pressure-Neutral Auto-Prefill Confirmation

Status: closed inconclusive with no authority. The completed A3B subset is retained
as informative evidence only; A10B stopped before its first child.

## Intent

Run the one final default-adjudication packet for the exact query-capped
auto-prefill implementation committed by v0.600. Change only the invalid VM
confound predicate that prevented both v0.600 packets from launching a child.

v0.600 P2 observed global Pageouts increase by 354 pages during A3B cache
conditioning while memory remained 96% available and swap occupancy, cumulative
Swapouts, and cumulative Compressions were unchanged. Pageouts therefore remains
an advisory field, not an anonymous-pressure-specific veto.

This is not a v0.600 retry. Pool no v0.600 observation, change no performance
cell or gate, and authorize no successor confirmation.

## Frozen Product Cells

- A3B: Qwen3.6 35B A3B file type 15, no MTP, outer/query `2048/1024`.
- A10B: Qwen3.5 122B A10B file type 15, no MTP, outer/query `4096/1024`.
- Prompt: tracked 11,287-token Mei strip, 51,876 bytes, SHA-256
  `ca924d7a3613ef8a6aa02fdcbfc56f2a7bffb5ec7aa2ddbd11e7efa3be7ad3f6`.
- A3B model SHA-256:
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- A10B shard SHA-256 values in order:
  `467c9bd92ea518539cf75bf5a5fbfbd35e9a0b40d766ccaa67bf120e12041df3`,
  `ecdbd42d43b0df9fa0ef9a584e09e95a43966ef03a122aba0b87a99d44d9ad98`,
  `13300e0f059e6fa21aa0fabde2a554f9deea366c0e54f268045769b214b28c97`.
- One redirected output token, no special tokens, prefix-cache limit 16 GiB,
  effective sequence capacity 11,304.
- A is explicit numeric 1024 and must emit schema 3. B is admitted `auto` and
  must emit schema 5.
- Common qwen argv remains `-m MODEL --prompt-file PROMPT --tokens 1
  --no-special-tokens --prefix-cache-max-mib 16384 --request-timings ROW`, followed
  by `--prefill-chunk 1024` for A or `--prefill-chunk auto` for B.
- Pass no max-context override, cache-prefix flag, prompt lookup, JSONL serving,
  warm follow-up, or performance environment control.

## Frozen Product Validation

Retain the v0.600 runner's exact checks:

- clean source/build/runtime identity and hashes for the runner, contract, prompt,
  binaries, imported helpers, and every model shard;
- Apple M4 Max, 128 GiB unified memory, macOS 15.x;
- complete child-environment digest after removing every `QWEN_*`, `METAL_*`, and
  `MTL_*` key plus exact `RUST_LOG`;
- exact output bytes, runtime identity, one timing row, zero target transitions,
  and ordered timing milestones;
- exact A3B/A10B allowlist, plan geometry, 41-row logical allocation inventory,
  eager/deferred classification, Metal-price validity, checked-u64 totals,
  sequence reserve, memory signals, and admission arithmetic;
- A3B outer/query and calls `2048/1024`, `60/120`, with overlay
  `393,052,160/393,052,160/235,405,312/235,405,312`;
- A10B outer/query and calls `4096/1024`, `36/144`, with overlay
  `807,403,520/786,104,320/807,403,520/786,104,320`;
- hard failure after admitted candidate allocation begins; no baseline retry.

## Pressure-Neutral VM Predicate

Before every child, read every model shard completely with one 8 MiB buffer, then
wait exactly 30 seconds A3B or 120 seconds A10B. Preserve the existing host-state
sampler and its frozen retry behavior: up to six samples, 30 seconds apart, until
AC power, no thermal/performance warning, and parsed memory availability `>=50%`.

Capture `vm_stat` and `sysctl -n vm.swapusage` immediately before the cache read,
immediately before spawn, and immediately after child exit. Every endpoint records
raw text plus parsed:

- Pageouts;
- Compressions;
- Swapouts;
- swap occupancy bytes.

For both cache and child intervals, compute all four signed deltas. Pageouts and
its delta are evidence only and never enter validity.

An interval is invalid if any of these holds:

- swap occupancy delta is positive;
- cumulative Swapouts delta is positive or negative;
- cumulative Compressions delta is positive or negative;
- either endpoint or any required field cannot be captured or parsed.

Negative swap-occupancy delta is valid. Do not gate compressor occupancy,
reactivations, decompressions, purge activity, or file-backed-page changes. They
remain available in raw `vm_stat` only.

Child validity additionally requires zero `/usr/bin/time -l` block input and a
valid post-exit host sample. Persist stable reasons such as
`cache_compressions_growth`, `cache_swapouts_counter_regressed`, and corresponding
`child_*` forms. Persist every raw endpoint and all deltas even when zero.

This predicate establishes only that no sampled system-wide anonymous compression
or swap activity occurred. It does not claim causal attribution or prove the
absence of transient/file-cache pressure.

## Fixed Packet And Gates

- Run profiles in order A3B then A10B.
- For each profile run exactly four pairs in order `AB/BA/BA/AB`.
- Add no VM-predicate, child, arm, pair, profile, or packet retry. Add no nonzero VM
  allowance, rate threshold, rescue cell, width, asset, prompt, or environment.
- The frozen host sampler is the only retry mechanism.
- Any validity failure terminates the sole packet immediately as inconclusive with
  no authority.
- Evaluate performance only if all 16 children and eight pairs complete.

For pair `i`, speedup is `A_i.ttft_ms / B_i.ttft_ms`. The even-size median is the
arithmetic midpoint of the two middle sorted values.

- A3B requires median `>=1.04x`, both AB/BA medians `>=1.03x`, and 4/4 wins.
- A10B requires median `>=1.15x`, both AB/BA medians `>=1.10x`, and 4/4 wins.

Report each profile independently as go or kill. A passing profile authorizes only
a separate profile-specific default decision on Apple M4 Max, 128 GiB, macOS 15.x.
The packet itself does not change defaults or authorize cache-bearing JSONL, 32K,
sibling assets, server concurrency, or numeric overrides.

After the resulting go, mixed, kill, or inconclusive decision, do not run another
auto-prefill confirmation. Return to the force-ranked optimization queue.

## Result

Source `bfa3902bdf2a9debb8f1b94443b1d1bacbef1fab` completes all eight
A3B children and four fixed pairs with valid identity, host/VM state, zero block
input, exact output/runtime identity, and exact schema/topology/admission records.
The completed subset mechanically clears every A3B speed gate:

- pair speedups `1.052868/1.052415/1.052678/1.053172x`;
- median `1.052773x` and paired median saving `373.859 ms`;
- AB/BA medians `1.053020/1.052547x`;
- 4/4 wins;
- median A/B TTFT `7457.786/7082.920 ms`.

These are post hoc subset calculations, not a formal v0.601 profile evaluation or
authority. The contract permits performance evaluation only after all 16 children.
The one-token output equality does not claim exact logits or complete-state parity.

A10B pair 1 arm A never launches. Its 77.03 GB cache-conditioning interval records
Pageouts `+985`, Compressions `+76`, and zero growth in swap occupancy and
Swapouts, with 96% memory availability and valid AC/thermal state. The 76 events
equal 1.1875 MiB of page-sized compression work, not net compressor growth or
causal attribution: compressor storage and occupied pages both decrease over the
interval.

The packet therefore stops inconclusive with completion identity verified and no
authority. Per preregistration, no successor confirmation follows. Both auto
profiles remain exact and opt-in; the roadmap returns to cold loader work.

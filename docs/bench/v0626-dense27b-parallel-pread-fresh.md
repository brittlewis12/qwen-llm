# v0.626 Dense-27B Direct-Pread Fresh Transfer

Status: preregistered oracle-ladder prerequisite; no v0.626 implementation or
product observation exists. Authority is `none` for every outcome.

## Question And Scope

For one exact dense Qwen3.6 27B W4 asset whose target file is already wholly in
the unified file-buffer cache, does explicit four-worker direct `pread` transfer
improve fresh-process load and end-to-end output-128 latency over explicit
ordinary copied loading without harming model-ready work, CPU, or memory?

This packet contains one exact full-state correctness gate followed by six
fresh-process pairs. It has no target-file-cold stage, no repeatedly loaded
stage, and no selector, default, or force authority. It makes no storage-cold,
serving, concurrent-load, energy, or other-model claim. The thresholds are
economic and engineering gates, not powered MDE or equivalence claims.

A valid performance miss closes this transfer and authorizes no successor. A
complete `fresh_effect_pass` authorizes only preregistration and execution of a
separate default-ColdOnly target-file-cold guard, followed by an independently
designed loaded-stability packet. Only the conjunction of all three separately
sealed passes could later authorize force use; v0.626 itself never does.

## Imported Evidence

The runner imports only the sealed v0.603 **performance floor** inventory. It
must authenticate and parse every inventory member, not merely the top-level
seals:

- decision SHA-256 `a0361b55d93cee769fc8d4db44eecdf83f3d1c63bdda5ecec07fd3d2edd279ac`;
- inventory SHA-256 `fa87cf2788d58b2f988ce07760ad3619bad835f4abb77dedd238b3dd563e3a9e`;
- completion SHA-256 `bcb19bc3f8776fe16e6459abb5930e020d89f2bd8cf20e3a31a7e567959282ab`;
- `schema=1`, `status=go`, source
  `e6e964ffad896a480ac280de8ff07634cdd344bd`, authority
  `implement-force-only-dense27b-loader-pilot`, six completed blocks, six
  wins, and paired median saving exactly `965.0405000000001 ms`.

`scripts/profile/v0605_dense27b_parallel_copied_loader.py` is a mechanics
library only. v0.626 must not authenticate, import, pool, rescore, reinterpret,
or derive authority from any v0.604/v0.605 decision, attempt, closure, or
successor authorization.

The exact fresh-request stdout SHA-256 is frozen before implementation as
`c94cb4d2661b181e91ab5db8c65ff252fcd97064073ff332d60715052b784cd0`.
The same hash was independently reproduced by all four ordinary-copied and
demand-paged arms of the exact model/prompt/output-128 v0.593 cell. This imports
only a correctness golden, not v0.593 performance evidence or authority. Every
successful v0.626 child must match it; within-packet A/B agreement alone is not
sufficient. Because v0.593 predates completion seals, v0.626 retrospectively
pins and rehashes its manifest, attempt ledger, protocol sources, four timing
rows, and four output files, then verifies their clean build identity, exact
request shape, validity, timing-row bindings, two-arm balance, and common hash.

## Source And Runtime Boundary

The packet preregistration commit must be the direct child of clean main
`129379f1c1918e30ab4725856fe9699a7f54d0ed`. The later implementation HEAD must
be the direct child of that packet commit. At execution the runner requires:

- `HEAD^^ == 129379f1c1918e30ab4725856fe9699a7f54d0ed`;
- `129379f..HEAD^` changes exactly this document and
  `scripts/profile/v0626_dense27b_parallel_pread_fresh.py`;
- `HEAD^..HEAD` changes exactly `crates/qwen-llm/src/metal_forward.rs`;
- a clean worktree, the repository root as working directory, and release
  `qwen` and `qwen-bench` build/runtime identities that independently equal
  implementation HEAD with clean and matching source states.

`qwen-bench build-info` supplies the structured build/runtime proof. The runner
must additionally find implementation HEAD and that exact build-source-state
stamp embedded in the hashed `qwen` executable before any packet artifact is
reserved; later schema-3 rows are a second check, not the identity root.

No other production, dependency, build-script, command, tokenizer, loader, or
Metal source change is admitted. The manifest hashes the two packet files, the
v0.605 mechanics library and its common helper as mechanics, both binaries,
the exact prompt/model, and all sealed v0.603 members.

## Frozen Cell

- Host: local Apple M4 Max with 128 GiB unified memory; exact device, macOS
  product/build, and memory identity are captured and rechecked.
- Model: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`, SHA-256
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`.
- Prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  SHA-256 `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`,
  exactly 1,891 bytes and 419 prompt tokens.
- Request: 128 emitted greedy tokens, 127 transitions, prefill chunk and maximum
  context 1024, full logits, prefix cache disabled, first post-load request.
- Pair order: `AB/BA/BA/AB/AB/BA`; no retry, replacement, parallel benchmark,
  serving, concurrent load, or inspection before all 12 sole attempts finish.
- Arm A: normalized controls plus `QWEN_GGUF_PARALLEL_COPY=0`.
- Arm B: the same normalized controls plus
  `QWEN_GGUF_PARALLEL_COPY=pread`.

No redundant no-copy or owned-arena environment override is admitted. Their
defaults follow from the normalized environment and Arm A's explicit parallel
disable; Arm B changes only the population selector.

Both arms retain configured default ColdOnly. Before every child, the runner
fully reads and SHA-256 hashes the exact target file, waits for valid host state,
then proves every target page resident immediately before launch. Each child
must emit exactly one ordinary ColdOnly `prefetch: skipped (already warm)` event
for that target at 100.0% residency. Neither arm may emit a
`[runtime-prefetch]` suppression marker or perform target physical I/O.
The runner records and matches the target path's device, inode, size, and mtime
across manifest creation, every post-conditioning residency proof, and final
identity validation.

## Anticipated Implementation Contract

Arm A emits native embedding policy followed by the ordinary copied ledger.
Arm B emits native embedding policy, then exactly one
`[metal-gguf-parallel-pread] schema=2 profile=dense27b-q4km-v1` marker, then the
same ordinary copied ledger. The pread marker has exactly the dense v0.605
schema-2 static and dynamic fields, field order, resource/byte accounting,
four-worker cuts/tasks/bytes, topology endpoints, allocation modes, page and
alignment values, mapped layout and inventory values, timing reconciliation,
CPU reconciliation, and no `plan=` field. Only the marker kind changes from
`parallel-copied` to `parallel-pread`. No other storage marker is permitted.

The following four anticipated CPU selector tests are frozen and must each run
in release mode, singly with `--exact`, and be sealed before GPU correctness:

```text
metal_forward::tests::gguf_parallel_pread_capability_table_is_exact
metal_forward::tests::gguf_parallel_pread_dense_marker_table_is_exact
metal_forward::tests::gguf_parallel_pread_dense_auto_remains_none
metal_forward::tests::gguf_parallel_pread_dense_advice_preserves_configured_policy
```

They respectively freeze direct-pread capability, dense marker selection,
dense Auto population remaining `None`, and forced dense pread preserving the
configured prefetch policy. Missing, filtered, malformed, nonzero, or multiply
executed selector evidence is an `implementation_or_contract_defect`.

## Full-State Correctness

After all CPU selectors pass, run this exact ignored release test on one thread:

```text
metal_forward::tests::gguf_parallel_pread_dense27b_q4_is_bit_exact
```

The test must compare ordinary copied A with direct-pread B and prove all source
bytes, 851 resources, exact offsets and lengths, independent Shared / Default
Cache / Tracked resources, frozen four-worker schedule and topology, checked
compute/blit write rejection, bit-exact packed-prefill logits, complete KV/GDN/
convolution state, prefill argmax, one forced transition, continuation logits
and continuation state. It must also prove the exact five-line A/B policy,
marker, and ledger ordering and all schema/accounting constraints above.
Malformed output or metadata, a non-exact test selection, or any missing
full-state assertion is an `implementation_or_contract_defect`.
Any unknown `[metal-gguf-*]` marker is also a defect: total occurrence count is
exactly zero for A and exactly one for B in correctness, complete children, and
any parseable prefix from a failed child. The exact selector tests separately
freeze the marker-choice table without claiming A/B child output.

## Fresh Execution And Gates

Each fresh child uses release `qwen`, one schema-3 timing row, exact stdout and
load identities, the frozen request, and independently captured launch,
completion, post-exit, timing, raw output, resource, host, pressure, and
conditioning evidence. Use paired ratios or differences before medians.
Malformed complete marker or timing evidence from a nonzero child remains an
implementation/contract defect; only absent or valid incomplete-prefix evidence
may fall through to execution invalidity.

All of these gates must pass:

- runtime/model-load saving A-B: median at least `750 ms`, AB and BA medians
  each at least `600 ms`;
- runtime/model-load B/A: median at most `0.70`, AB and BA medians each at most
  `0.80`;
- spawn-to-first-byte A/B: median at least `1.25`, AB and BA medians each at
  least `1.20`, with at least 5/6 total and 2/3 per-stratum B wins;
- spawn-to-exit A/B: median at least `1.08`, AB and BA medians each at least
  `1.05`, with the same win counts;
- first-prefill and model-ready TTFT B/A: medians at most `1.02`, AB and BA
  medians each at most `1.03`;
- generation and model-ready request B/A: medians at most `1.01`, AB and BA
  medians each at most `1.02`;
- B candidate-endpoint total CPU: median at most `2.45 s`, AB and BA medians
  each at most `2.55 s`;
- complete-process CPU B-A: median at most `0.90 s`, AB and BA medians each at
  most `1.00 s`;
- maximum paired RSS B/A and footprint B/A each at most `1.05`.

No load, CPU, memory, or end-to-end win can erase a model-ready miss.

## Validity And Pressure

Capture raw Pageouts, Compressions, compressor stored/occupied pages, Swapouts,
swap occupancy, block I/O, major faults, and cache and child interval deltas.
Cumulative Pageouts, Compressions, and Swapouts regressions are invalid. Any
positive Swapouts or swap growth, child block input/disk reads, child or marker
major faults, invalid host/AC/thermal/performance/memory state, spawn failure,
nonzero exit, malformed capture, or operator interruption is inconclusive.
Target physical I/O, block input, and major faults must each be zero.
On macOS `/usr/bin/time -l`, `page reclaims` records minor faults and `page
faults` records major faults; the row preserves the raw name and adds an
explicit `major_page_faults` alias before applying the zero gate.

Pageouts and Compressions growth and compressor stored/occupied gauges are
recorded and advisory; the compressor gauges are not cumulative activity
counters. This semantics reflects deliberately cache-warm children and does not
create a target-file-cold stage.

Correctness, source/build/runtime identity, schema, output, marker, ledger,
accounting, and selector defects are `implementation_or_contract_defect`.
Host, pressure, execution, interruption, and durability invalidity is
`inconclusive`. Preserve deferred interruption, complete child cleanup, no
retry/nonparallel semantics, exclusive artifact creation, fsync, final model
rehash, and final inventory/completion durability.
Final identity collection must always publish a fsynced report, including
observed values or capture errors. A mismatch is a sealed
`implementation_or_contract_defect`; only an exact match is admissible for
`kill`, `fresh_effect_pass`, or `inconclusive`.

## Decision And Closure

Decision precedence is: implementation/contract defect; inconclusive validity;
valid fresh performance miss; complete fresh pass. A miss is `kill`, closes the
transfer, and permits no successor. A complete pass is exactly
`status=fresh_effect_pass`, `authority=none`, and authorizes only the separately
preregistered default-ColdOnly target-file-cold guard described above. The
runner must never emit `status=go`.
The only valid statuses are `implementation_or_contract_defect`,
`inconclusive`, `kill`, and `fresh_effect_pass`; every one has
`authority=none`, `force_authorized=false`, and fail-closed successor scope.

The final seal must bind the manifest, all CPU selector output and metadata,
correctness output and metadata, exact launch/completion records, a fsynced
conditioning/residency artifact for every launch, the exact attempted-row
prefix, post-exit evidence, timing and raw artifacts, final model rehash,
decision, complete member-verified artifact inventory, and completion seal.
`kill` and `fresh_effect_pass` require all 12 unique sole-attempt rows; an early
terminal status seals only its exact attempted prefix. An incomplete or
malformed seal has no authority.

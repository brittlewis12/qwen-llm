# v0.630 Dense-27B Direct-Pread Short-Period Loaded Stability

Status: preregistration. No v0.630 implementation or timing result exists.

## Intent And Authority

Decide the sole successor authorized by v0.629: whether direct `pread`
population preserves loaded prefill, decode, and complete-request performance
for the exact dense Qwen3.6-27B Q4_K_M ForceOnly profile when A/B observations
alternate well below the drift period that invalidated v0.605.

This is a new loaded-stability packet. It imports v0.629 only as successor
authority. It imports no v0.605, v0.627, v0.628, or v0.629 performance value,
variance estimate, golden output, or pass/fail gate for scoring. A complete pass
may authorize explicit `QWEN_GGUF_PARALLEL_COPY=pread` for this exact ForceOnly
profile. It cannot authorize absent-environment Auto/default selection,
disposable admission, another model or quantization, MTP, serving, concurrent
loading, storage-cold performance, or an energy claim.

## Source Boundary

The production-and-roadmap base is
`cc47190654b898e2004117b6d03be1006a89d1fa`. The preregistration commit R is a
clean direct child and adds exactly this document and
`scripts/profile/v0630_dense27b_pread_loaded_stability.py`. The implementation
commit H is a clean direct child of R and changes exactly
`crates/qwen-cli/src/bench.rs` to add opt-in exact generated-token-trace reporting
to `qwen-bench decode`. It may not change loading, inference, timing, warmup,
model, tokenizer, Metal, or kernel behavior.

R and H must each have exactly one parent. The R diff is exactly two added files;
the H diff is exactly one modified file. The complete H `bench.rs` SHA-256 and
the binary R-to-H patch SHA-256 are frozen in the runner before R is committed;
path-only matching is insufficient. The tracked-clean sibling dependencies are
also frozen at `gguf=c7369fd4868a6f613459fff355477f53bf4ee2f1` and
`llama-cpp-rs=fe4fb533d1ed2855b6ac5492e56c42007d410409`. GGUF permits no
untracked file; llama-cpp-rs permits exactly the non-build local file
`.claude/settings.local.json`. Any other tracked or untracked state fails.

```text
H bench.rs SHA-256     5efea85fc599f8efe1b5ce2fd7e930a779065555df84b81d6de08370021e05ba
R..H patch SHA-256     bf6ad710294af218b802ddb6631f8381c639eca419d332fd26c87cb96f5bc14e
```

At execution, `HEAD=H`, `HEAD^=R`, and `HEAD^^` is the base above. The runner
checks both exact diffs, clean source state, build/runtime identity H, sibling
dependency identity, macOS build, device, model inode/size/mtime/SHA-256,
prompt bytes/SHA-256, packet sources, release binary, and all contract inputs.
Once a manifest exists, the final identity report recomputes every non-model
digest and the complete model digest before a terminal decision can be
published, and writes failure-shaped evidence if verification fails. A failure
before manifest creation seals only preflight evidence and makes no final-
identity claim.

The exact host software identity is macOS `15.6.1` build `24G90`. Benchmark and
Cargo children receive separate complete allowlisted, nonsecret environments;
DYLD, allocator, wrapper, arbitrary thread-count, Python, and unrelated inherited
variables are absent. The manifest records both exact maps and the names removed.

The v0.629 decision, inventory, and completion files are authenticated at these
SHA-256 values:

```text
decision.json             949f8066c3ab7261a2ee7916ca4d3e8ce92b56e6a6cf4396b55474020be21aa9
artifact-inventory.sha256 6a94d45e1b8ccb170ac77226a65033d26e6441fbe415298828288dc09a2c4e5c
packet-complete.json      11c2a86d958ee4d935a87edfa37e335616232a9e63ed2c8b713c7b08c39a8ba5
```

The runner verifies their complete binding and terminal fields, including the
sole successor `preregister-separate-short-period-loaded-stability-packet-only`.
They establish authority only.

## Correctness Gate

Before timing, rerun exactly:

```text
cargo test --release -p qwen-llm \
  metal_forward::tests::gguf_parallel_pread_dense27b_q4_is_bit_exact \
  -- --ignored --exact --nocapture --test-threads=1
```

Require zero exit, the unique exact Cargo-test result, and the complete expected
five-line A/B load protocol. The test must freshly establish all 851 resources
and 16,806,250,496 logical bytes, exact sizes and offset-zero topology,
Shared/DefaultCache/Tracked storage, the frozen four-worker schedule, marker and
ledger, checked write rejection, bit-exact packed-prefill logits, prefill
argmax, complete KV/GDN/convolution state, one forced transition, and bit-exact
continuation logits/state. A recognition, assertion, marker, or accounting
failure is an implementation or contract defect, not a performance result.

The runner executes a CPU-only `qwen-bench decode --help` protocol check before
correctness, requiring the opt-in flag and exact description once. H's complete
source and patch digests prove that omission leaves the old text protocol and
the new opt-in path emits one compact canonical JSON integer array; every timed
child then exercises and parses that exact line.

## Loaded Cell

The exact model is `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`. The exact prompt
is `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`, 1,891
bytes and 419 model tokens. Each sole child is an independent serial subprocess:

```text
/usr/bin/time -l target/release/qwen-bench decode
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf
  --prompt <exact prompt bytes>
  --tokens 31
  --runs 1
  --prefill-chunk 1024
  --kv-capacity 1024
  --full-logits-decode
  --generated-token-trace
```

The default warmup remains enabled. It performs one untimed full packed prefill
and one untimed full-logits transition. The timed fresh session then performs
one full packed prefill and exactly 31 full-logits transitions. Model loading and
the warmup are outside the scored P/D/R measurements. `P` is timed packed
prefill wall, `D` is the complete 31-transition decode wall, and `R` is the
fresh-session request wall including session/scratch allocation, P, and D.
The opt-in trace contains the initial prefill argmax followed by the result of
every timed transition: exactly 32 IDs. It therefore binds the output of the
31st scored transition rather than only the tokens consumed by those calls.

Arms differ only in the normalized environment:

- A: `QWEN_GGUF_PARALLEL_COPY=0`;
- B: `QWEN_GGUF_PARALLEL_COPY=pread`.

A emits exactly the native-embedding policy and ordinary copied ledger, with no
`[metal-gguf-*]` marker. B emits that policy, exactly one dense schema-2 direct-
pread marker, then the same copied ledger. Any unknown or extra load/storage
line is a contract defect. B marker `timer_major_faults` must be zero.

Immediately before each child, the runner reads the complete target file, proves
the unchanged inode and 100% `mincore` residency, captures host/VM state, then
durably records conditioning before launch. Each child must have zero block-input
operations and zero `/usr/bin/time` page faults. No child overlaps another.
There is no deliberate cooldown, retry, replacement, inspection break, or
performance-loss rerun. AC power, no thermal/performance warning, adequate
memory, no swap growth, and durable raw output are hard validity requirements.
Pageouts and compressions are recorded as advisory evidence under the repaired
pressure semantics below.

Pageouts and Compressions are cumulative counters: only regression is malformed,
and positive deltas are advisory. Swapouts regression is malformed and positive
Swapouts or swap-used-byte growth is invalid. Swap-used-byte decreases and both
directions of the compressor stored/occupied gauges are valid advisory evidence.
Before execution, the runner also authenticates the exact `/usr/bin/time -l`,
`vm_stat`, `pmset`, `memory_pressure`, and swap-usage parser dialects.

## Short-Period Order

Run eight four-child temporal blocks, with exact order:

```text
ABBA BAAB ABBA BAAB ABBA BAAB ABBA BAAB
```

There are 32 sole children, 16 observations per arm, and four quartets per order
stratum. In either quartet the two arm centers are both position 2.5. At the
expected six-to-eight seconds per child, nearest cross-arm observations are
about six-to-eight seconds apart, each center-matched quartet spans about
24-32 seconds, and one ABBA/BAAB reversal period spans about 48-64 seconds.
Short-period execution is a hard validity property. The runner brackets `Popen`
and conservatively uses its completion timestamp as child start:
conditioning-to-child-start must be at most 2 seconds, consecutive child starts
at most 12 seconds, each quartet span at most 45 seconds, and each eight-child
ABBA/BAAB reversal span at most 90 seconds. Any miss is inconclusive before
performance classification.

The symmetric quartet removes log-linear multiplicative time drift under roughly
regular spacing.
Alternating ABBA and BAAB reverses endpoint-versus-center placement. It cannot
remove abrupt machine-state changes, nonlinear within-quartet drift, or
arm-dependent carryover; hard host/pressure validity and complete quartet
trajectories remain necessary.

## Scoring And Gates

For each positive time metric M in each quartet, use geometric arm means:

```text
GM_A = sqrt(A1 * A2)
GM_B = sqrt(B1 * B2)
P = GM_B(prefill) / GM_A(prefill)
D = GM_B(decode) / GM_A(decode)
R = GM_B(request) / GM_A(request)
```

Require all nine loaded noninferiority gates:

- median of all eight P scores `<= 1.01`;
- median of four ABBA P scores `<= 1.01`;
- median of four BAAB P scores `<= 1.01`;
- the same aggregate and stratum gates for D at `<= 1.01`;
- the same aggregate and stratum gates for R at `<= 1.01`.

GO additionally requires every quartet to satisfy P/D/R `<= 1.01`, complete CPU
`<= 1.10`, and both memory ratios `<= 1.05`. Simple wins against 1.0 remain
diagnostic. A mixed set of quartet passes and misses is inconclusive, not a kill.
A stable kill requires at least seven of eight quartets over one declared limit
and the aggregate, ABBA, and BAAB medians for that same metric over the limit.

For complete-process CPU time, form the same quartet B/A ratios and require the
aggregate, ABBA, and BAAB medians each `<= 1.10`. For peak RSS and physical
footprint, require in every quartet
`max(B1,B2) / max(A1,A2) <= 1.05`. These protect untimed loading cost and memory
without importing either into P/D/R.

Before any performance classification, each arm's 16-observation P, D, and R
series must have `(max-min)/median <= 0.05`. Larger movement is inconclusive.
Report every value, range, median, first-to-last ratio, child wall and midpoint,
conditioning delay, child-start separation, quartet span, and reversal span.

Every child must report exactly 32 canonical trace IDs in `[0,248320)`. All 32
exact arrays must
equal a golden created by the first valid A child and frozen in the launch
ledger before any B result is parsed. Also require identical decoded-text debug
lines, prompt token count, model metadata, command shape, one timed repetition,
and one request-wall observation. A decoded-text hash alone is not token
identity.

This is a narrow engineering admission packet, not a confidence-interval or
population-level 1% noninferiority claim. Thirty-one transitions average the D
cell internally; eight center-matched quartets provide 16 P/D/R observations per
arm while keeping the treatment alternation short.

## Decision And Closure

Precedence is exact:

1. Source/build, correctness, token identity, parser/schema, marker/ledger,
   command, accounting, artifact, or durability defect seals
   `implementation_or_contract_defect` with no authority.
2. Invalid host/pressure state, lost target residency, physical input, child
   failure, interruption, incomplete quartet, or capture failure seals
   `inconclusive` with no authority.
3. A complete stable packet with a directionally consistent miss as defined
   above seals `kill` and closes direct pread for this dense ForceOnly profile.
4. A complete stable but heterogeneous miss seals `inconclusive`.
5. Only the complete valid conjunction seals `go` and
   `authority=explicit-force-only-dense27b-direct-pread`.

After artifact reservation, all classifiable outcomes seal the sole-attempt
prefix with a member-complete SHA-256
inventory, launch/completion and conditioning records, attempts digest, final
identity report, decision, and fsynced completion binding. `go` authorizes no
automatic selector. A separate policy packet would still be required before an
absent environment may choose dense direct pread.

A raw-output/metadata durability failure or inability to reap the exact launched
PID is genuinely unsealable and produces no packet completion or authority.
Every successfully recorded launch otherwise has a matching completion and
attempt row, including spawn, recovered wait, and failed-child evidence.

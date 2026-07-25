# v0.627 Dense-27B Direct-Pread Fresh Harness Repair

Status: preregistered repair packet. Every outcome has `authority=none` and
`force_authorized=false`.

## Repair Boundary

v0.626 is immutable forensic evidence, not authority. It sealed at
`target/profiles/v0626-dense27b-parallel-pread-fresh-p1` with status
`implementation_or_contract_defect` after all four CPU selectors passed and the
full-state Rust correctness test passed every assertion. It launched zero fresh
children. Its result is not promoted, rescored, or treated as a passing gate.

The sole repaired premise is the Python harness recognition of the correctness
test result. The frozen v0.626 runner demanded contiguous text of the form
`test <exact-name> ... ok`. The raw transcript instead contains exactly one
anchored `test <exact-name> ... ` start, then the expected interleaved load and
panic-capture lines, then one standalone `ok`, followed by exactly one
`test result: ok. 1 passed; 0 failed;` summary. The sealed evidence therefore
demonstrates a harness-recognition defect and no Rust assertion failure; v0.626
did not separately seal Cargo's return code.

v0.627 changes no production source and does not change the scientific question,
cell, arms, ordering, conditioning, validity rules, pressure semantics, gates,
status precedence, or closure registered by v0.626. It reruns all four exact CPU
selectors and the exact full-state correctness test; no v0.626 result is reused
as a gate. Only after those fresh gates pass does it run the original 12 sole
fresh children in `AB/BA/BA/AB/AB/BA` order, with no retries.

## Frozen v0.626 Evidence

Before reserving a v0.627 artifact, the runner must authenticate all of this:

- decision SHA-256
  `d2fe54d9b7a7c81adae9d85611f9a95e12a51547539d3e576993184f64471f5a`;
- inventory SHA-256
  `c596fa38affa3bcca3717b2878b7b2a121b41cf5b04b9e1d779c59ca2a689b87`;
- completion-file SHA-256
  `8563e7929c07925ffc1fa2846776dd6d78cc3c7645dfc4558c9ee99cda780d03`;
- completion object exactly binds those decision and inventory digests;
- inventory has exactly nine unique safe members, each present and rehashed;
- the directory has exactly those nine members plus inventory and completion;
- no attempt ledger, launch seal, conditioning, fresh output, timing, post-exit,
  spawn-failure, or prelaunch-failure artifact exists;
- decision terminal fields are `authority=none`, `force_authorized=false`,
  `successor_authorization=none`, `stopped_after=correctness`, `attempts_sha256`
  null, `correctness` null, `stages={}`, and source
  `7fb0488a5c075d8c62c8f9802352f717ec95b085`;
- all four selectors are passed, uniquely named by the frozen exact selector
  list, and bound to strict contiguous raw test transcripts;
- the inventory-bound correctness transcript has the exact interleaved harness
  form described above, exactly five expected load lines in A/B order, exactly
  one direct-pread marker, and a valid frozen marker schema.

The v0.626 document and runner are imported only as frozen protocol mechanics
and forensic evidence. The entire seal and all nine inventory members are
included in and rehashed by the v0.627 build manifest.

## Source And Runtime Boundary

The v0.627 preregistration commit R must be the direct child of clean production
implementation commit `7fb0488a5c075d8c62c8f9802352f717ec95b085`. R adds exactly
this document and
`scripts/profile/v0627_dense27b_parallel_pread_fresh_repair.py`.

At execution, the runner requires the repository root as working directory, a
clean worktree, `HEAD=R`, `HEAD^=7fb0488a5c075d8c62c8f9802352f717ec95b085`,
and an exact two-file diff. `Cargo.lock`, every `Cargo.toml`, `.cargo`, `crates`,
and `kernels` must equal production commit `7fb0488`. This follows from and is
also checked independently of the exact packet diff.

Release `qwen-bench build-info` must report clean matching build and runtime
identities stamped R. The hashed `qwen` binary must embed R and the same exact
build-source-state stamp. Both binary identities, model path/hash/size/inode
identity, prompt, golden output, and all inherited protocol dependencies are
manifest-bound and rechecked through the final identity report.

The exclusive artifact directory is
`target/profiles/v0627-dense27b-parallel-pread-fresh-repair-p1`. Existing paths
are never reused. No v0.626 artifact is copied into it.

## Repaired Recognition

CPU selector recognition remains strict: zero return code, exactly one
contiguous anchored `test <exact-name> ... ok` line, and exactly one anchored
successful one-test summary. Generic or unanchored `ok` is never sufficient.

Correctness recognition requires all of the following:

- zero return code;
- exactly one anchored `test <exact-name> ... ` start;
- exactly one anchored, complete `test result: ok. 1 passed; 0 failed; ...`
  one-test summary;
- either one contiguous start-plus-`ok` result, or exactly one standalone `ok`
  after the start and before the summary;
- no second standalone or generic unanchored `ok` as a substitute.

After harness recognition, the inherited correctness parser must still count
and validate the exact five policy/ledger/marker lines, A/B order, marker schema,
resource accounting, and full-state contract. The repair broadens no marker,
load, selector, command, or assertion contract.

## Frozen Execution And Gates

The runner executes the same four selector names and
`metal_forward::tests::gguf_parallel_pread_dense27b_q4_is_bit_exact` with the
same release, exact-selection, one-thread commands. It then uses the unchanged
v0.626 mechanics for the exact model, prompt, 128-token greedy request, stdout
golden, arm environments, default-ColdOnly behavior, target residency proof,
conditioning-before-launch records, launch/completion ledger, and 12 children.

All v0.626 fresh-effect gates remain exact: runtime/model-load savings and
ratios; first-byte and exit ratios and win counts; first-prefill and model-ready
TTFT noninferiority; generation and model-ready request noninferiority; candidate
and complete-process CPU bounds; and RSS and footprint bounds. Paired values are
formed before medians. No model-ready miss can be erased by another win.

The v0.626 cache-warm page-fault and pressure semantics remain exact. Each child
is conditioned and proven wholly target-file resident before launch. Target
physical I/O, block input, child major faults, and marker major faults must be
zero. Host, thermal, AC, memory, VM, compressor, swap, interruption, malformed
capture, durability, and failed-child rules are unchanged. There are no loaded
or target-file-cold stages and no retries or replacement children.

The manifest hashes this document and runner, the frozen v0.626 document and
runner, v0.605 and common helpers, binaries, model, prompt, every inherited
v0.603 and v0.593 dependency, and the complete authenticated v0.626 seal and
members. Non-model identity is rechecked between stages; complete model and
inode identity is checked before children and in the final report.

## Authority And Closure

Decision precedence remains: `implementation_or_contract_defect`, then
`inconclusive`, then valid `kill`, then complete `fresh_effect_pass`. Every
status has `authority=none` and `force_authorized=false`; the runner must never
emit `go`.

A valid miss is `kill`, closes this transfer, and authorizes no successor. A
complete `fresh_effect_pass` authorizes only preregistration and execution of a
separate default-ColdOnly target-file-cold guard. It does not authorize force,
default selection, or production change. Any early terminal result seals only
the sole-attempt prefix. `kill` and `fresh_effect_pass` require all 12 attempts.
The final identity report, member-complete inventory, decision, and completion
seal retain the v0.626 durability and fail-closed rules.

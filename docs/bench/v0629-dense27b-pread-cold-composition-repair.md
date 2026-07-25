# v0.629 Dense-27B Cold-Composition Page-Fault Repair

Status: preregistered harness-only repair. No v0.629 model or GPU observation
exists. Every outcome has `authority=none` and `force_authorized=false`.

## Sole Repair

v0.628 correctly sealed `inconclusive` after its first A child because the
frozen runner treated nonzero process-wide `/usr/bin/time` `page faults` as an
invalid major-fault condition. That child recorded exactly 638 page faults and
638 `proc_pid_rusage` pageins while executing the deliberately target-file-cold
cell. Load-phase pageins 613 plus prefill pageins 25 exactly reconcile to 638.
The same child physically read 16,818,487,296 bytes after proving the model file
went from 1,026,444/1,026,444 resident pages to zero.

The zero-page-fault rule was structurally misplaced: it came from cache-warm
fresh-process validity, while v0.628 deliberately requires target-file-cold
physical I/O. It is not evidence of a composition failure. Every other observed
A1 contract passed, and no B child launched.

v0.629 changes exactly one premise before any B observation: process-wide child
`page_faults` remains exactly recorded in `/usr/bin/time`, attempt, post-exit,
and decision evidence, but is advisory rather than a hard validity failure for
both arms. It adds no threshold and does not require equality with
`proc_pid_rusage` pageins. It retains:

- child swaps exactly zero;
- B marker `timer_major_faults=0`;
- every target identity, invalidation, residency, default-policy, prefetch,
  storage, ledger, token, physical-read, timing, CPU, footprint, host/VM,
  durability, pairwise, status, and authority rule from v0.628.

No crate, example, runtime, loader, model, prompt, command, environment, order,
or gate changes. No v0.628 performance value is pooled, compared, rescored, or
used to derive a threshold. All four children are new sole attempts.

## Complete v0.628 Forensic Import

Before reserving an artifact, authenticate
`target/profiles/v0628-dense27b-pread-cold-composition-p1`:

- decision SHA-256
  `c0c2447016e88585161605897bf8cc80777cd382ecee5efb284daabd16dd18a5`;
- inventory SHA-256
  `f8e63bd70d2f4ac6b34f6cb64e2c83752819e99370875d95dcf92880c3e44fb7`;
- completion SHA-256
  `b14714cae32e08d24a385d3d2e289765bb2c2e791fa052e6f1c5f7022c3b2680`;
- exactly 12 inventory members and 14 final regular files, with no symlink,
  directory, temporary, or extra member;
- terminal status `inconclusive`, source
  `113350615c33e0a17773050befcb5fbbff7f65ec`, `authority=none`,
  `force_authorized=false`, successor `none`, failed child A1, and sole reason
  `child_major_faults_nonzero`;
- one exact A1 attempt and one launch/completion pair, zero B observations;
- the complete v0.628 verifier passes under its original process-validity rule;
- A1's stable protocol, identity, invalidation, default ColdOnly action,
  prefetch, copied ledger, physical-read window, token, host/pressure, raw,
  post-exit, final identity, and seal all remain bound.

The import is forensic only and has `authority_imported=false`.

## Source And Build Boundary

The clean repair base is
`113350615c33e0a17773050befcb5fbbff7f65ec`. Execution commit P must be its
direct child and add exactly this document and
`scripts/profile/v0629_dense27b_pread_cold_composition_repair.py`. Relative to
dense implementation `7fb0488a5c075d8c62c8f9802352f717ec95b085`, the protected
Cargo/crates/kernels tree must still differ only in
`crates/qwen-llm/examples/first_byte_spike.rs`.

Release `qwen` and `qwen-bench` must be rebuilt at P; their
build/runtime/source-state identity must be clean and exactly P. The executable
example and every linked crate input are unchanged by this docs/runner-only
repair, so require the exact v0.628 authenticated example-binary SHA-256
`a9e4e6d25aaa91738eb0b302ff227d58e5d923b8a7afcfc9715853dd144aaa89`
rather than claiming a new embedded P identity it cannot expose. Rehash all live
packet/mechanics/common sources, both CLI binaries, the example source and exact
binary, model, complete v0.627 bridge, and complete v0.628 forensic import
between gates and children. The executed example must continue to emit the
manifest-bound self-source SHA-256.

## Frozen Execution

Run the exact v0.628 CPU test and then four sole children, order `AB` then `BA`,
with no retry or replacement. Commands, environment, model, prompt, token count,
force-only intent, default ColdOnly config, in-child invalidation, parsers, and
all numeric gates are byte-for-byte inherited from v0.628.

The only executable repair is:

```text
original process validity:
  page_faults != 0 -> child_major_faults_nonzero

v0.629 process validity:
  page_faults -> recorded advisory only
```

Any process-resource parse failure remains invalid. Child swaps remain hard
zero. B's phase-local marker major-fault field remains hard zero. Attempts and
post-exit evidence retain the observed process-wide page-fault integer.

## Decision And Closure

Statuses remain only `implementation_or_contract_defect`, `inconclusive`,
`kill`, and `cold_composition_pass`, with the same precedence and pairwise gates
as v0.628. All outcomes have `authority=none` and `force_authorized=false`.

`kill` closes dense transfer. `cold_composition_pass` authorizes only
preregistration of the same separate short-period loaded-stability packet.
Nothing authorizes force, Auto/default population, product, exact-request-cold,
serving, concurrency, energy, untouched-media, or another model. v0.629 imports
no authority from v0.628 and grants no `go`.

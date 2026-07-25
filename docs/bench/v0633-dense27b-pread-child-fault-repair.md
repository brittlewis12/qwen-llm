# v0.633 Dense-27B Timed-Child Fault Attribution Repair

Status: preregistered harness-only validity repair. No v0.633 correctness or
timing result exists; authority is none.

## Sealed v0.632 Classification

v0.632 is immutable `inconclusive` evidence. Fresh correctness passed with the
complete full-state contract and a zero candidate marker fault delta. Its first
timed A child then completed valid inference but the inherited runner rejected
92 `/usr/bin/time` process-wide page faults before any B child launched.

The exact A evidence is:

- complete 16,817,244,384-byte target read immediately before launch;
- unchanged inode/size/mtime identity and 1,026,444/1,026,444 mincore pages;
- zero block-input operations and zero process swaps;
- zero swapout/swap-used/compressor growth, valid AC/thermal/memory state;
- exact arm load ledger, P/D/R output, and 32-ID initial-plus-final trace;
- 92 process-wide page faults and advisory global Pageouts `+130`.

The wrapper counter spans startup, model load, untimed warmup, timed request, and
teardown. It cannot identify a file, address, phase, target I/O, or P/D/R impact.
Zero block input plus exact immediately-preceding target residency is the direct
available guard against storage I/O. The 92 faults remain evidence but are not
an attribution oracle.

v0.632 remains inconclusive. Its refined forensic class is
`timed-child-process-wide-page-fault-gate-overbroad`. Its incomplete A timing is
not rescored or imported.

## Complete v0.632 Forensic Bridge

Authenticate the complete v0.632 seal:

```text
decision.json             561399219b2ccb066d584932b5ca1a72ca42f56757fc7fa9d6f68b2a8ae5d353
artifact-inventory.sha256 e3ce48c108578401ebf0e1950b683aa14a7273394160a4183da8f0c2623e01eb
packet-complete.json      535f82c9c2d032b29e944cee4b133b1058d3741b71bdbb8b659bf7f7ae178366
```

The completion binds the first two hashes. Require exactly 13 inventory members
and 15 final regular files, with every member rehashed. Require source
`2e71e1a8caba14ddc0fcc5ea3232354dbab86504`, terminal `inconclusive`, sole reason
`child_major_faults`, failed child `loaded-q01-abba-r1-a`, stopped after loaded,
null stage, no authority or successor, and zero imported performance.

Require fresh correctness passage and its exact transcript; exactly one A
attempt and one launch/completion pair; no golden event and no B observation;
the exact A token trace, return 0, page faults 92, block input 0, swaps 0, sole
invalidity reason, complete conditioning, Pageouts `+130`, flat hard pressure
fields, and successful final identity.

Record:

```text
authority_imported=false
correctness_imported_as_gate=false
performance_observations_imported=0
timed_children_imported=0
observed_a_children=1
observed_b_children=0
forensic_class=timed-child-process-wide-page-fault-gate-overbroad
```

v0.629 remains the direct scientific successor authority. v0.627, v0.631, and
v0.632 are forensic only. No cx output has authority.

## Sole Repair

For every fresh timed A and B child:

```text
/usr/bin/time page_faults = canonical recorded advisory integer
```

There is no threshold and no A/B equality requirement. Preserve the integer in
raw stderr, post-exit evidence, attempt row, and terminal artifacts.

All independent hard gates remain:

- process resource parsing succeeds;
- `block_input_operations == 0` and process swaps equal zero;
- complete target read, exact file identity, and 100% mincore residency;
- valid host, AC, thermal, memory, VM capture, swapouts, and swap-used bytes;
- timed B direct-pread marker `timer_major_faults == 0`;
- exact load/ledger, token trace, temporal, trajectory, P/D/R, CPU, memory,
  heterogeneity, consistency, durability, and final identity.

A has no phase marker; adding one would change the measured program without
making process-wide faults target-aware. A stronger claim of zero physical I/O
throughout each child requires a separate symmetric target-attributed packet.
It is not the claim here.

The v0.633 runner uses an explicit child implementation identical to the frozen
v0.630 function except that it records `page_faults_advisory` rather than adding
`child_major_faults` to validity reasons. Timed B marker parsing still invokes
the original hard-zero marker parser before process resources are considered.

## Source And Fresh Execution

The exact one-parent lineage is:

```text
base  cc47190654b898e2004117b6d03be1006a89d1fa
R630  d30a0f49e14aec13126d83a41553f6e7250b0f93
H630  d90aecdd4d4f9c297f98b09ba55eb479a7bf7944
R631  ecc9e70fc833ee40831b7f5891141e620a92a2cd
R632  2e71e1a8caba14ddc0fcc5ea3232354dbab86504
R633  HEAD at execution
```

R633 adds exactly this document and
`scripts/profile/v0633_dense27b_pread_child_fault_repair.py`. No existing file
changes. Rebuild `qwen-bench` at R633. Recheck every inherited commit/diff,
binary/source/dependency/OS/model/prompt identity, v0.629 authority, and complete
v0.632 bridge.

Execute entirely fresh evidence in inherited order: token-trace protocol check,
one full-state correctness run, then all 32 new timed children. Do not reuse the
v0.632 A trace, golden, P/D/R values, correctness as a gate, or any partial
artifact. No retry, replacement, cooldown, or inspection break.

## Authority

Decision precedence and authority remain unchanged. Complete stable GO may
authorize only explicit dense-27B ForceOnly direct pread. Stable consistent miss
KILLs. Unstable or heterogeneous evidence is inconclusive. Auto/default
selection remains separate. This repair grants no authority before all fresh
gates and all 32 children complete.

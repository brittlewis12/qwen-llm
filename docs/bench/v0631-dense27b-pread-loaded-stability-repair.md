# v0.631 Dense-27B Loaded-Stability Preflight Repair

Status: preregistered harness-only repair. No v0.631 timing result exists.

## Defect Boundary

The v0.630 `--preflight-only` invocation stopped before artifact reservation and
before any model load or GPU work. Its runner called the nonexistent command
`qwen-bench metal-info`; Clap exited 2 and Python raised `CalledProcessError`.
The current binary exposes `metal-counters`, whose first line contains the same
device description. The remainder of v0.630 preflight passes when only that
command is corrected, including source/build/dependency identity, complete model
SHA-256, prompt identity, v0.629 authority, safe environments, host tools, and
all frozen cell fields.

There is no v0.630 packet artifact, performance observation, correctness result,
or authority to import. This repair changes no scientific premise, arm, command,
order, validity rule, gate, status precedence, or authority. It changes only the
CPU-light device-description preflight probe:

```text
v0.630: qwen-bench metal-info       # nonexistent; exit 2
v0.631: qwen-bench metal-counters   # authenticate exact device first line
```

The repair requires the current Metal-counter protocol: exact first line
`device\tApple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes`,
one sampling line, `counter_sets\t1`, one timestamp-set line, and one timestamp-
counter line. It then presents the already-frozen v0.630 device string to the
unchanged manifest validator. No timing endpoint uses this command.

## Source Boundary

The frozen lineage is exact:

```text
base  cc47190654b898e2004117b6d03be1006a89d1fa
R630  d30a0f49e14aec13126d83a41553f6e7250b0f93
H630  d90aecdd4d4f9c297f98b09ba55eb479a7bf7944
R631  HEAD at execution
```

Each commit has exactly one parent. Base-to-R630 adds only the v0.630 document
and runner; R630-to-H630 modifies only `crates/qwen-cli/src/bench.rs`; H630-to-
R631 adds only this document and
`scripts/profile/v0631_dense27b_pread_loaded_stability_repair.py`. The repair
runner independently rechecks the frozen v0.630 complete `bench.rs` and patch
digests, clean worktree, R631 build/runtime identity, sibling commits and exact
untracked allowlist, OS build, model, prompt, binary, and every inherited input.

No production, loader, inference, Metal, tokenizer, model, or kernel source may
change in R631. The release `qwen-bench` binary is rebuilt at R631 so its source
identity matches the repair commit even though executable Rust content is
unchanged from H630.

## Execution And Authority

After the repaired device check, execute the complete v0.630 protocol verbatim:

- the CPU-only generated-token-trace help check;
- the fresh dense full-state bit-exact correctness gate;
- 32 independent cache-warm children in eight exact `ABBA/BAAB` quartets;
- 31 timed transitions and one 32-ID initial-plus-final trace per child;
- exact load, token, pressure, temporal, trajectory, P/D/R, CPU, memory,
  heterogeneity, consistency, artifact, and final-identity rules;
- no retry, replacement, cooldown, inspection break, or imported performance.

The v0.630 decision precedence and closure remain exact. A complete GO may
authorize only explicit dense-27B ForceOnly direct pread. KILL requires the same
stable directionally consistent miss. Unstable or heterogeneous evidence is
inconclusive. Auto/default selection remains a separate successor. This repair
itself grants no authority before the full packet completes.

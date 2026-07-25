# v0.632 Dense-27B Correctness-Fault Gate Repair

Status: preregistered harness-only correctness-parser repair. No v0.632
correctness or timing result exists; authority is none.

## Sealed v0.631 Classification

v0.631 is immutable forensic evidence with no authority. It sealed
`implementation_or_contract_defect` before a timed child because the Python
correctness parser required the direct-pread marker's process-wide
`timer_major_faults` delta to equal zero. The exact value was 2.

The underlying ignored release test returned zero and reported `1 passed; 0
failed`. It completed every byte, topology, read-only, packed-prefill-logit,
argmax, KV/GDN/convolution-state, forced-transition, and continuation assertion.
Rust treats marker major faults as telemetry and validates their canonical
schema; only Python promoted zero into a correctness requirement.

`timer_major_faults` is a `getrusage(RUSAGE_SELF)` delta. It is process-wide,
includes all threads and mappings, and has no target-file or destination
attribution. Before candidate B, manifest creation had SHA-read the complete
16,817,244,384-byte GGUF; ordinary copied A had separately read all 851 expected
resource payloads totaling 16,806,250,496 bytes in the same Cargo process. The
host was quiet with 96% available memory. The two faults remain evidence, but
they cannot invalidate the passed state contract.

v0.631 remains labeled `implementation_or_contract_defect`; the refined class is
`correctness-marker-major-fault-gate-overbroad`, not a loader defect and not a
passing correctness import.

## Complete v0.631 Forensic Bridge

In `--preflight-only`, the runner authenticates v0.631 without reservation. In a
full run, it first reserves the artifact and empty ledgers, then authenticates
the complete v0.631 seal during preflight before manifest publication,
correctness, or any child:

```text
decision.json             64730255135525cf289e6f36b1af81c33dd7f59b5575a950ca9945caaa02147e
artifact-inventory.sha256 27c07cbc68c6ff33aacc437f92ea44c97ffa33035f441f077eec3b430c579f7a
packet-complete.json      e32ffba74b4d38dbc94dc3bfbacdec14a9d80da07f0a6d6f9a48f38be9bc0482
```

The completion object must bind the first two hashes. The inventory has exactly
eight members and the final directory exactly ten regular files. The runner
rehashes every member and requires, among other fields:

```text
correctness.out    9fbdb5296d2f31658d0aae6c81d0f2c72764118b046dc61dca65af7c95c57ab0
manifest.json      e1d7f7263bd066741f1eb9b228197b4f7b3ddeced9fc299ce002a230521d0d64
final-identity     15a3908e8350e6e8e379b131b50d7af93660a0e8bd3714465414fe9b6aa3e94b
attempts.jsonl     e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
launch-seal.jsonl  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Require the exact terminal status, reason, source, empty attempts and launch
ledgers, null correctness/stage, no child, no authority, no imported performance,
successful final identity, exact five-line A/B transcript, marker major-fault
value 2, and unique successful one-test summary.

Record the bridge as:

```text
authority_imported=false
correctness_imported_as_gate=false
performance_observations_imported=0
timed_children_imported=0
forensic_class=correctness-marker-major-fault-gate-overbroad
```

v0.629 remains the direct scientific successor authority under its already-
frozen seals. v0.627 is corroborating evidence only. No prior performance or
correctness result scores v0.632.

## Sole Repair

The only protocol change is correctness-marker parsing:

```text
correctness: timer_major_faults is canonical unsigned advisory telemetry
timed B:     timer_major_faults == 0 remains a hard requirement
```

The repair parser first captures the exact canonical integer, substitutes zero
only into a temporary line passed through the complete frozen v0.630 marker
parser, restores the real value in the returned object, and records it as
`correctness_marker_major_faults_advisory`. Thus every other prefix, field,
order, count, canonical-number, timing, CPU, marker, and ledger check remains
identical. The shared parser is restored before any timed child.

Run one fresh full-state correctness test. Do not reuse v0.631 as a gate, add a
warm/mincore premise, retry, cooldown, or replacement run.

## Source And Execution Boundary

The exact one-parent lineage is:

```text
base  cc47190654b898e2004117b6d03be1006a89d1fa
R630  d30a0f49e14aec13126d83a41553f6e7250b0f93
H630  d90aecdd4d4f9c297f98b09ba55eb479a7bf7944
R631  ecc9e70fc833ee40831b7f5891141e620a92a2cd
R632  HEAD at execution
```

R632 adds exactly this document and
`scripts/profile/v0632_dense27b_pread_correctness_fault_repair.py`. No existing
file changes. Rebuild `qwen-bench` at R632 for exact source identity. The runner
rechecks every inherited commit/diff, frozen H source/patch digest, safe
environment, sibling/OS/model/prompt/binary identity, v0.629 authority, and the
complete v0.631 bridge.

Execute the inherited order exactly: the CPU-only token-trace protocol check,
then the fresh correctness gate, then 32 cache-warm children in eight
`ABBA/BAAB` quartets. All temporal, trajectory, pressure, token, P/D/R, CPU,
memory, heterogeneity, consistency, durability, and final-identity rules remain
intact.

Any timed B marker major-fault value above zero remains a hard contract defect.
Every timed child also retains zero `/usr/bin/time` page faults, zero block input,
full target read conditioning, 100% mincore residency, and no swap growth.

## Authority

Decision precedence and authority are unchanged. A complete stable GO may
authorize only explicit dense-27B ForceOnly direct pread. A stable consistent
miss KILLs. Unstable or heterogeneous evidence is inconclusive. Auto/default
selection remains separate. This repair grants no authority before all fresh
gates and all 32 timed children complete.

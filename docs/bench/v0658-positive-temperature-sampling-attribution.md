# v0.658 Positive-Temperature Sampling Attribution Successor

Status: preregistration. No v0.658 runner, packet root, model-backed
observation, product reference, profiled child, or timing result exists.

## Intent And Authority

Answer the positive-temperature CPU-attribution question that v0.657 did not
observe. This is a new independently preregistered packet, not a rerun, repair,
or result rescue. It imports no v0.657 correctness, timing, phase-size, gate, or
candidate result.

v0.657 stopped at its first model-facing host gate after detecting a foreign
process. Its terminal record has `profile_spawned=false`,
`profile_result_observed=false`, `authority=[]`, and disposition
`CONSUMED_NO_AUTHORITY`. Freeze the predecessor identities:

```text
v0.657 preregistration
51aaaf5c4ddad604d539b90609d6c8b2d98201e18f22ea65cc465961108b9a1d

v0.657 runner
f9e6f1db659f2431c380901dba3ebbdfdd44b831d166bd7502a1e5f041ca6ae7

v0.657 decision.json and failure.json
d0ddf95125a5f444b8c69911deec20aaf01da35ab6bd473623b73dacb1702020
```

The tracked v0.657 preregistration and runner are normative inputs and the
v0.658 runner must hash-authenticate both fail-closed. The ignored `target/`
decision and failure records are historical justification only. They are not
runtime inputs, need not exist in a disposable clean worktree, and must not be
read or imported by the v0.658 runner.

The complete normative contract in
`docs/bench/v0657-positive-temperature-sampling-attribution.md`, authenticated
by the hash above, is incorporated by reference except for the explicit v0.658
deltas below. That includes its intent and authority, exact fixture,
instrumentation, schema-11 request record, correctness gates, one-reference and
six-profile acquisition, host/VM validity, W/B/C/S reductions, decision rules,
failure dispositions, and stopping rules.

v0.658 may independently authorize at most one bounded implementation packet
under those inherited rules. It cannot promote a product optimization, import
greedy evidence, change sampler-v1, or claim another workload or model.

## Frozen Deltas

The successor changes only packet identity, adds fail-closed authentication of
the two tracked predecessor inputs, and adds one early readiness census while
retaining every inherited host gate:

```text
runner
scripts/profile/v0658_positive_temperature_sampling_attribution.py

packet root
target/profiles/v0658-positive-temperature-sampling-attribution-p1

decision schema
qwen-v0658-sampling-attribution/v1

failure schema
qwen-v0658-sampling-attribution-failure/v1
```

Immediately after creating and fsyncing the new packet root, and before static
hashing, VM capture, or CPU conformance, run one readiness census using the
inherited `process_census` implementation. Durably publish
`host-readiness.json` containing:

- label `before-static-and-cpu-conformance`;
- the complete redacted process census and its SHA-256 digest; and
- the exact detected-competitor records.

The readiness census performs no CPU-idle, memory-pressure, power, or thermal
sampling. It passes only when the inherited competitor set is empty. Argument
strings remain represented only by byte length and SHA-256; executable names
and argv0 identities remain plaintext. A readiness failure is a pre-profile
environment failure: consume v0.658 with no authority and launch no test or
model child.

After readiness passes, execute the inherited acquisition unchanged. In
particular:

- rerun both CPU conformance commands; import no v0.657 test result;
- retain the complete three-sample host gate immediately before model
  conformance, the product reference, every profiled child, and after profiles;
- run model conformance before the reference;
- run exactly one reference and six profiled fresh processes; and
- preserve every inherited identity, equality, reconciliation, VM, threshold,
  authority-partition, and no-rerun rule.

The early census reduces wasted setup when a known competitor already exists.
It does not attest GPU idleness, weaken later gates, authorize dropping a child,
or condition a result after observation. The explicit operator GPU attestation
remains mandatory for acquisition.

## Source And Build

Commit this preregistration before the v0.658 runner exists. The implementation
commit may add only the new runner; it must not modify Rust, Metal, the v0.657
runner, or either preregistration. Derive the v0.658 runner from the frozen
v0.657 bytes, changing only:

- script, preregistration, packet-root, and schema identities;
- tracked predecessor-preregistration and runner authentication; and
- the readiness-census function and its placement before static work.

Build fresh release `qwen` and `qwen-bench` binaries from one clean committed
v0.658 implementation state. Require matching clean build/runtime/source
identity with no problems or overrides. Re-run static checks and adversarial
review before acquisition. No old binary, packet child, or model-backed result
may be imported.

## Acquisition And Decision

Run the inherited acquisition exactly once under the new root after the manual
operator check and runner check-only phase are green. The sole authorized
command is:

```text
uv run scripts/profile/v0658_positive_temperature_sampling_attribution.py \
  --phase acquire --attest-no-other-user-gpu-workload
```

The inherited reference, six-child, W/B/C/S, authority, and failure rules are
the complete decision procedure. A readiness or later pre-profile environment
failure consumes v0.658 with no authority. No child may be repaired, replaced,
trimmed, or rerun. Record the terminal result even when no profile launches.

## Implementation Order

1. Commit this preregistration by itself.
2. Add the derived runner and its fail-closed readiness self-tests.
3. Run non-GPU static and CPU checks and obtain adversarial review.
4. Build and authenticate clean release binaries in a disposable clean
   worktree.
5. Confirm the host is quiet, then execute the sole acquisition once.
6. Reduce mechanically and update the log and leverage map without importing
   any unobserved v0.657 claim.

# v0.657 Positive-Temperature Sampling Attribution Result

Status: **CONSUMED_NO_AUTHORITY**. The packet stopped at its first
model-facing host gate, before model conformance, the product reference, or any
profiled child. It contains no sampling-attribution measurement and grants no
candidate, performance, model-backed correctness, or product authority.

## Scope And Identity

The packet was acquired from a disposable clean worktree with matching clean
build and runtime identity:

```text
commit: 79943b305706e18365a023a0a1cafd77fca93f11
source: git-source-sha256-v2:0e21f96661a8ceced7e3044f46c7059d3a09a7da9473fb4e0fccdf3e33a44898
build/runtime status: match; clean; no overrides
```

Static validation authenticated the frozen model file, prompt bytes, release
binaries, source-bound expected prompt-token digest and command configuration,
and sanitized child environment. It did not execute the product command or
observe runtime prompt tokenization. All runtime model and token fields were
unobserved.

## Pre-Acquisition Security Amendment

The first check-only implementation serialized inherited environment values in
plaintext to its output. That was discovered before a packet directory,
model-backed observation, or GPU command existed. The preregistration and
runner were amended and committed before acquisition:

- parent and child environments retain only variable names, byte lengths, and
  SHA-256 commitments;
- packet children receive a frozen non-secret allowlist rather than the full
  inherited environment;
- helper output and process-census argument strings retain only lengths and
  digests, while frozen helper argv and executable identities remain plaintext;
  and
- runner self-tests reject plaintext retention.

The sealed packet contains no plaintext environment values.

## Completed Work

The packet completed its static gates and both CPU conformance commands:

| Command | Result | Packet PID |
|---|---:|---:|
| `qwen-llm` sampling tests | 16 passed | `70478` |
| `qwen-cli` binary tests | 57 passed | `75856` |

Both commands exited zero, timed out zero times, leaked no process group, and
required no termination action. These results validate the CPU instrumentation
and CLI wiring only; they do not substitute for the frozen model-backed test.

## Stop

Before model conformance, the host gate detected PID `68030`, an extant
`cargo test -p qwen-llm --lib` process outside both packet-owned process groups.
The runner therefore stopped under the frozen pre-profile environment rule.
It did not kill, wait for, or otherwise mutate the foreign process.

The terminal records seal:

```text
disposition: CONSUMED_NO_AUTHORITY
failure_category: environment
stage: host-model-conformance
profile_spawned: false
profile_result_observed: false
authority: []
```

The following work did not run:

- model-backed state and logits conformance;
- the unprofiled 128-token product reference;
- any of the six profiled children;
- any W, B, C, or S removable-work reduction; or
- a before/after VM-counter delta.

Consequently there is no speed, phase-size, model-backed or product
correctness, or falsification inference to make about positive-temperature
sampling. In particular, this result neither authorizes nor kills reusable
workspaces, borrowed logits, a combined implementation, or streaming/direct
top-k selection.

## Decision And Learning

v0.657 is consumed and must not be rerun, repaired, or completed under the same
version or packet root. Its lack of observation leaves positive-temperature
sampling attribution first in the active queue.

Any successor must be independently preregistered under a new version and
packet root after the competing process is gone. It should run a cheap initial
competitor census before the expensive CPU conformance commands, while
retaining the full host gate immediately before model conformance and every
timed child. That changes operational ordering only; it does not weaken or
rescue any v0.657 decision rule.

## Artifacts

The sealed packet is under
`target/profiles/v0657-positive-temperature-sampling-attribution-p1/`.
`decision.json` and `failure.json` are byte-identical and every recorded
inventory size and digest verifies. Independent design, implementation,
security, and result review: `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.

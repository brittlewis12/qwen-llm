# v0.659 Positive-Temperature Sampling Parser Repair

Status: preregistration. No v0.659 runner, packet root, build, test, model,
reference, profiled child, or timing result exists.

## Intent And Authority

Correct one deterministic pre-candidate parser defect from v0.658, then answer
the still-unobserved positive-temperature sampling-attribution question under a
new packet root. This is a separately preregistered successor, not a rerun or
repair of the sealed v0.658 root and not a rescue of its reference timing row.

v0.658 completed conformance and its ordinary reference, then stopped at the
first reference-validation assertion. No profile launched and no attribution or
candidate value was observed. The only v0.658 observation v0.659 may use is:

```text
reference top-level keys
extra=[]
missing=[prefill_attention_query]
```

That observation justifies only the parser correction below. v0.659 imports no
v0.658 conformance, generated-output, reference, timing, performance, VM, gate,
or authority evidence.

Freeze the predecessor identities:

```text
v0.658 preregistration
341dda990f6856c7b2e9bb619eaa87ec9d701de1c0b9291abcaf684cbbc57cad

v0.658 runner
f3b4e4ef08065b80182683219d9103cd00887b51bd3c3bc0522a95a5a1f40485

v0.658 decision.json and failure.json
2cb06a3b04698cf1711524ae0114f24504cf9c2fff5d6b9cefbe2d348a289089
```

The tracked v0.658 preregistration and runner are normative inputs and the
v0.659 runner must hash-authenticate both fail-closed. The ignored terminal
records are historical justification only. They need not exist in a clean
worktree and the v0.659 runner must not read or import the v0.658 packet root.

The complete normative v0.658 contract, including its incorporation of v0.657,
is inherited except for the explicit deltas below. v0.659 may independently
authorize at most one bounded implementation under the inherited W/B/C/S rules.
It cannot promote product code, import greedy evidence, alter sampler-v1, or
claim another fixture, workload, or model.

## Frozen Deltas

Change packet identity:

```text
runner
scripts/profile/v0659_positive_temperature_sampling_parser_repair.py

packet root
target/profiles/v0659-positive-temperature-sampling-parser-repair-p1

decision schema
qwen-v0659-sampling-attribution/v1

failure schema
qwen-v0659-sampling-attribution-failure/v1
```

For this exact 419-token fixed-chunk fixture,
`prefill_attention_query: Option<_>` is `None` and the product omits the key.
The v0.659 parser must:

1. remove `prefill_attention_query` from `REFERENCE_KEYS`;
2. remove it from `FROZEN_REQUEST_FIELDS`;
3. replace the stale reference object/key/value checks with an explicit
   `"prefill_attention_query" not in row` requirement; and
4. explicitly require the field absent in every profiled row before validating
   the schema-11 attribution object.

No `null`, object, or extra-key form is accepted. The strict reference key set
and `set(profile) == set(reference) | {"sampling_attribution"}` remain in force,
so the absence rule is fail-closed in both arms.

Add a runner-contract self-test proving that the field is excluded from both
frozen key collections. Do not synthesize, default, or strip the field from a
parsed product row.

All v0.658 readiness, host/VM, environment-redaction, process ownership,
identity, CPU/model correctness, product-reference, six-profile equality,
schema-11 attribution, reconciliation, W/B/C/S reduction, threshold, authority,
failure-disposition, and no-rerun rules remain unchanged.

## Source And Build

Commit this preregistration before the v0.659 runner exists. The implementation
commit may add only the new runner; it must not modify Rust, Metal, any
predecessor runner, or any preregistration. Derive the runner from the exact
v0.658 bytes and change only:

- script, preregistration, packet-root, and schema identities;
- tracked v0.658 preregistration and runner authentication; and
- the four parser corrections and their self-test named above.

Build fresh release `qwen` and `qwen-bench` binaries from one clean committed
v0.659 implementation state. Require matching clean build/runtime/source
identity with no problems or overrides. Run every inherited static, readiness,
host/VM, CPU, model, reference, and profile gate afresh. Import no old binary,
child, output digest, timing row, or model result.

## Acquisition And Decision

After non-GPU checks and adversarial review, confirm the host is quiet and run
the inherited sole acquisition exactly once:

```text
uv run scripts/profile/v0659_positive_temperature_sampling_parser_repair.py \
  --phase acquire --attest-no-other-user-gpu-workload
```

The inherited reference, six-child, adjusted-bound, five-of-six, authority, and
failure rules are the complete decision procedure. No child may be repaired,
replaced, trimmed, or rerun. Record the terminal result even if no profile
launches.

## Implementation Order

1. Commit this preregistration by itself.
2. Add the derived runner and exact absence self-test.
3. Run non-GPU syntax, contract, and static review.
4. Build and authenticate fresh release binaries in a clean worktree.
5. Confirm readiness and execute the sole acquisition once.
6. Reduce mechanically and update the log and leverage map without importing
   any v0.658 result beyond the parser-correction premise.

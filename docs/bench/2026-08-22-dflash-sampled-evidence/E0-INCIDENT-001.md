# E0 Incident 001 - Malformed v1 Producer Artifact

Status: closed as an invalid development instrumentation acquisition after
protocol review. This record was written after the incident and before any v2
acquisition. It grants no E0 authority.

## Timeline And Disposition

- Acquisition source was clean commit
  `3e65dee7fec699cf93dad7234d7b2db699422be8`. Bootstrap began at
  `2026-08-23T02:04:13Z`; measured execution began at
  `2026-08-23T02:15:45Z` under run ID
  `dflash-e0-1787450653168115000-10798`.
- The producer exited zero and printed that its internal comparisons passed.
  That message is diagnostic only and has no evidentiary weight.
- The frozen strict reducer exited with status `2` without creating its requested output:
  `run dflash-e0-1787450653168115000-10798.paths keys mismatch:
  missing=['executable'], extra=[]`.
- The trace contains eight rows: `bootstrap_start`, `bootstrap_end`,
  `run_start`, `prompt_step`, `sample_frontier`, `terminal_boundary`,
  `continuation`, and `run_end`. Both bootstrap rows contain
  `paths.executable`; all six run-common rows omit it.
- Classification is `invalid development instrumentation acquisition`.
  It is neither an E0 pass, an E0 mismatch, a valid reduction, nor an
  independent replicate. It will never be edited, completed, permissively
  reduced, pooled, or counted.

## Frozen Identities

- Trace: `9,432,048` bytes,
  SHA-256 `837f5b51d55d28591d75849f7d2a04299fe52f516d0b86e2e454e70b2f60cc7e`.
- State sidecar: `941,883,392` bytes,
  SHA-256 `d198464ed1b46e62059ffee7e924ae279989e65202ff8dfdc3e204f304f61bd6`.
- Executable: `48,083,448` bytes,
  SHA-256 `7b5654cd6378c08823b792b9fc1b78cd6ab9e82ba4061fc1ce948183d759a67a`.
- Binding manifest: SHA-256
  `2f083305357ef49409998d6f8bbddee030395f929c28337f722e590e0420bcf8`.
- Reducer: SHA-256
  `38ff5725f73df765eec1b94a78c1839b7e62083ba2e01ef433c9085df0f4e861`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Prompt was UTF-8 `Hello`, SHA-256
  `185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969`,
  with one token and i32le token SHA-256
  `7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381`.
- Configuration was one requested token, sampler-v1, seed `0`, temperature
  bits `0x3f333333`, top-k `200`, top-p bits `0x3f800000`, min-p bits
  `0x3d4ccccd`, resolved stop token `[248046]`, warmup enabled, and
  `serial_then_capture` arm order.

The original artifacts remain at their recorded `/private/tmp` identities and
were also copied byte-for-byte as mode-`0444` files to the local quarantine
`/Users/tito/Documents/qwen-evidence/dflash-e0/2026-08-22-invalid-v1/`.
Hashes were verified after copying. This protects against accidental writes but
is not an immutable or adversary-resistant archive.

## Root Cause And Repair

The schema-v1 reducer contract already required the complete six-field paths
object on every row. The producer independently assembled bootstrap and run
paths, then omitted `executable` from the latter. This was a malformed
schema-v1 producer instance, not a schema change and not a model observation
that can be adjudicated without the frozen reducer.

The repair constructs one immutable paths value and clones it into bootstrap
and run common metadata. Producer and reducer regressions cover complete path
identity, bootstrap/run drift, alias rejection, and input-trace linkage. Schema
versions remain unchanged because the implementation now conforms to the
already-frozen contract.

Any subsequent v2 acquisition is a distinct, prospectively frozen,
post-incident development sentinel. It is not a retry, replacement,
continuation, or replicate of v1. Another malformed or nonreducible artifact
stops acquisition and requires a new incident review.

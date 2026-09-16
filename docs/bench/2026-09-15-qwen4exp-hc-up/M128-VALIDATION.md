# CLI M128 IQ4_NL Validation

## Frozen Isolation Protocol

The actual product CLI attempt has empty stdout/stats and a Metal API validation
abort, not a completed answer. Its2578 tokens schedule2048/3/527, selecting the
M4 Max IQ4_NL down M128xN16 variant at527. This variant was excluded from the
earlier2179-token packet, whose actual chunks are2048/3/128 (not2048/131).

Before changing product math or retrying the model: serial production benchmark
lease and real wired-memory check, `MTL_DEBUG_LAYER=1` throughout. Isolate original
M128 empty-route grid33x20x512, TG128, shared9216, K640/nb01=20 blocks. One expected
512 failure and511 control. Capture populated checked-host original output with
K64/M130/N17, two experts/counts17 and9, permuted34 slots (26 active,8 unused),
M and N tails. Save raw output before comparisons; own replay bitwise, active
finite/nonzero, unused poison retained, all guards and inputs immutable.

Only after reproduction: widen this kernel's annotated thread index to uint and
immediately cast to the original local ushort; preserve all body math and grid.
Run512 and511 controls plus identical populated fixture with original SHA256
assertion. Sequential original/repaired captures avoid shipping a duplicate
shader. Retain both raw outputs and logs. No broad builtin widening, no API
validation disable, no HC performance rerun or threshold change. Retry real CLI
once after isolated proof, explicitly inspect process status, answer and stats.

## Result: Repair And CLI PASS

Original511 PASS0.04s;512 reproduces exact tiitg65536 API assertion/SIGABRT.
Original populated checked-host capture PASS0.03s,3380 finite nonzero outputs,
26 active/eight untouched slots, all guards/input bytes intact. Raw original:
`target/profiles/qwen4exp-m128-index-95262/`, SHA256
`a81d1e5299d92066cbbd46aed742ea17e6c220c2f2510dd80c151ec4ff22dd10`.

Only the annotated builtin changes to uint plus immediate ushort local cast.
Rebuilt511/512/populated tests PASS0.12s, identical original-output SHA; repaired
raw `target/profiles/qwen4exp-m128-index-98633/`. Logs
`2026-09-15-m128-{narrow-511,narrow-512,narrow-populated,repaired-regression}.log`
under target/profiles. Read-only reviewer found no source-level blocker.

Rebuilt CLI02 completes with both HC/QSA flags, API validation, expected answer
and successful request stats. See `PRODUCT.md`. No HC timing rerun.

# Muse qualified defaults

## Decision and scope

Remove the generation selection-policy gap, not change kernels or requalify timing.
Runtime, run, and serve allow matrix/tiled-online prefill and split decode by
default only for Muse Q8_0 on unified Apple M4 Max. Existing independent variables
become strict rollback controls: unset/1 allows qualified selection,0 disables.
Unsupported lanes fall back rather than failing simply because defaults are on.
Diagnostics must use resolved options, not requested permissions.

Lens/fit consumers and the existing exact-labelled muse-request benchmark select
the named reference loader. Their arithmetic and serialized measurement contracts
must not silently change. The library ordinary load() follows generation defaults.

No benchmark-length cliff is reintroduced. Existing context/capacity/shape/PSO
checks remain; split minimum1024 visible positions and528KiB admission are unchanged.
Independent attention oracles reach131072; full-model evidence is8K/32K plus512
teacher-forced transitions. No whole-model131K equivalence or sampled exactness claim.

Authority remains the math/long-context/serve-math results from September8-9:
controlled6229 split8.663->15.938 forwards/s,32K3.955->14.722; original failures and
holds remain unchanged. This patch makes existing qualified math reachable by default.

## Frozen checks

1. CPU resolver truth table: both artifacts, qualified and other device names,
   unified/nonunified, all four independent option combinations. Unsupported
   requests resolve reference; supported requests preserve each rollback.
2. Shared CLI/serve strict parser: unset/1 allow,0 disables; malformed values fail.
   Run and serve keep independent variable names. Build all affected binaries.
3. One ignored native runtime default-delivery test, production benchmark lease
   and real wired gate before Metal, MTL_DEBUG_LAYER=1, serial. Explicit-on then
   load() on the existing6229-token Current Reva rendered fixture,16 actual
   generated transitions:17 full-vocabulary rows/greedy IDs, complete kernel-name
   sequence and session memory plan must match bitwise. Actual tiled/split dispatch
   presence and capacity rejection without position advance are required.
   Named-reference then explicit0/0 use its31-token prefix plus16 transitions:
   same equality, neither optimized kernel nor split scratch permitted.
   This is routing/delivery identity, not new numerical or timing qualification;
   it does not repeat the prior all-active-KV numerical oracles.
4. Native CLI unflagged versus explicit1/1 on the existing rendered fixture,
   high/temp0/seed42,17 outputs, if GPU ownership is available. Require effective
   true/true, identical stdout/token fingerprint/finish. No timing gate or speed claim.
5. Existing serving lifecycle/HTTP qualification is unchanged; CPU wiring checks
   and source review cover policy plumbing. Do not claim a new HTTP packet.

Never stop another owner's GPU process without approval. A busy production lease
blocks execution, not an excuse to use a test-only lease or skip the wired gate.
No timing retries or broadened devices. Resolve code/protocol mistakes explicitly;
retain any actual failing attempt. Separate implementation and outcome records.

## Adversarial review

cx Luna session01a0b18d-c63c-7571-8627-d2c8d1bae95c identified shared lens consumers,
requested/effective telemetry, unsupported-device fallback and the exact-labelled
benchmark as necessary boundaries. We retain full admitted-context kernel policy
on invariant/oracle evidence, explicitly not an end-to-end131K claim. Default
delivery compares against already-qualified explicit-on math rather than repeating
a costly all-original numerical qualification without a kernel change.

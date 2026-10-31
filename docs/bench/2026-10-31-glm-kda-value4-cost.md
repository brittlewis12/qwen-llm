# GLM KDA Value4: Material Captured-Input Cost Pass

## Decision And Scope

The fixed value4 design passes its predeclared resident device-cost screen:
**61.24% and60.84% lower median GPU time** than CURRENT in the two ordered
blocks. Every timed command includes the same4MiB incoming-state restoration.
Independent raw-byte and timing recomputation accepts the result.

This funds an ordinary-path conversion gate, not promotion. It is NOT measured
whole-KDA/application latency, general carry or model quality. Resident captured
inputs and the restoring blit have different cache conditions from ordinary
producer execution. Do not multiply this saving by34 sites or use a coarse
nonexclusive stage span as an application prediction.

## Frozen Case And Baseline

Source is `b69feffddeab8ed78af65fee20dc64b82691ddae`, following the accepted
[actual-prepared witness](2026-10-31-glm-kda-actual-prepared.md). The only input
is that unchanged block0, prefix3/N2048 snapshot and its accepted provenance.
There is no GGUF open, model load, session, preparation or recapture in this run.

Both PSOs use owned, verified external library bytes: the actual capture's full
PRIMARY `fc086f1a...`, not cached `2c2b...`, and unchanged value4 `e7380f68...`.
Both libraries, PSOs, input/proof mappings, context and eleven buffers survive
checked completion/error/unwind. CURRENT and candidate arithmetic, Args and
128-thread grids64x32/64x8 are unchanged. No shader or alternate precision flag
is introduced by the timing host.

Inputs0..6 remain immutable. Equal guarded4MiB-state/64MiB-output pairs use
buffers9/10 and8/7. Ordinary output/state references stay borrowed from the
original snapshot; there is no duplicate prepared asset or GPU reference bank.
Different whole-file A/B hashes are expected from index-colored guards.

## Protocol And Result

Two excluded full-correctness commands precede timing. Block0 has four warm ABBA
commands and eight retained ABBA-ABBA commands; block1 uses BAAB and BAAB-BAAB.
Total:26 commands/restores/recurrence dispatches, eight excluded warm and16
retained intervals, eight retained per arm. All24 timing intervals are retained,
including first-retained outliers; none is filtered.

Each continuous12-command block runs inside its real admitted transaction, with
at most one command in flight. No poisoning, hashes, readback, validation or
report I/O occurs between those commands. GPU timestamps are read after checked
completion and cover the entire restore-plus-recurrence command. Host wall is
separate, not an application/host-performance result.

| Block | CURRENT Median ms | Value4 Median ms | B/A | Reduction |
| --- | ---: | ---: | ---: | ---: |
| ABBA | 6.033750 | 2.338625 | 0.387590636 | 61.2409% |
| BAAB | 5.991750 | 2.346187 | 0.391569657 | 60.8430% |

All four retained-quartet ratios pass:0.400586673,0.388298460,0.380682855 and
0.391590875. CURRENT median drift is-0.696084%; value4 is+0.323374%. The frozen
rule requires both block ratios and all quartet ratios<=0.70, plus absolute
CURRENT median drift<=5%. This clears the material screen, without an automatic
repeat, sweep, clock/thermal explanation or application-performance claim.

Full comparisons before timing and after each block pass. Independent final
replay checks every17,825,792 output/state words per arm against the captured
ordinary reference: zero bit differences and zero nonfinite values. Zero-sign
comparison stays strict, though these payloads contain no zeros. Snapshot/raw/
binding hashes, all106,496 guard bytes and all three checkpoint input/output
hashes pass. Checkpoints do not audit every intermediate GPU run separately.

## Resources And Execution

The fresh diagnostic subtotal is1,154,768,912 bytes below2GiB: priced buffers
415,940,608; original snapshot plus final raw-file cache487,170,064; controls
16MiB; executable/library pages128MiB; Metal/private policy96MiB. Separate CPU
framework policy128MiB remains charged. Actual buffer allocated-size sum is
415,932,432 bytes. There is no model W/S/R512MiB or incumbent/scratch credit.
Opaque policies are not measured/formal peak bounds.

Canonical executable size73,997,384 bytes satisfies the unchanged96MiB cap.
Ordinary opt3/fat-LTO/codegen1/default-feature compilation uses rustc1.99.1,
SDK26.2 and Metal32023.864. All372 exact library CPU tests, strict603 scoped
classifications/107 ignored entries and eight Python controls pass. Source,
library and878 protected-file identities remain unchanged through the gate.

The M5 Ultra DEBUG0 native test passes in6.52 seconds, seven seconds by UTC
supervision, under the external90-second timeout plus20-second termination
grace. Canonical WAIT0 self-leasing ends naturally. Offline acceptance runs
unleased after exit; supervisor/lease facts are trusted parent observations,
not remote attestation. No server, installed binary, PATH or default changes.

## Frozen Evidence

| Evidence | SHA256 |
| --- | --- |
| Canonical libtest | `f28ad2a3a2c6d60835544d77e945b003f9982c9b5b77c67a61903a4c058207c7` |
| Build/CPU receipt | `92027ed96ddf285ad1d5534c7af15d3c19efddde44235efa599bc21d2e4fc00c` |
| Frozen PRIMARY | `fc086f1a69bc63b1b427d9133c40e0c8dd341aaae17ca9ec09bfff064a5919c4` |
| Candidate library | `e7380f68c8a94f4ff5d463c37b3f55aff6ff4956bacad32de50ffc799a62fc0c` |
| Actual report | `363086bf61beef37123488c94c535641576058c874f6e53607acb8cfca0d39a1` |
| Authored offline acceptance | `c7727e9c6d2ddc6ea51c27a6560d5df0330cd6ea40c8be4cda9a2e2b82dd5fa0` |
| Independent actual replay | `739f1628b4d997eb4fe06aa6aafcd922b33a7f8bccde92923fc24f93f0d90239` |

Next is a test-only ordinary PRIMARY integration and all34-site/nonempty-state
model comparison, including scalar continuation across the sparse frontier.
That new translation unit needs fresh machine/artifact proof. Any bit mismatch
stops conversion; no tolerance waiver follows from this cost result. Matched
ordinary-call performance and broader carry/quality/default decisions remain
separate gates.

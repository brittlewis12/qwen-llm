# Muse FFN source audit and CPU decision tooling

Source `77289eb9`; no GPU dispatch, server action, model download or new production
path. Pending Muse default/alignment GPU delivery remains unresolved and comes
before future optimization screens. Shared-server access still requires coordination.

## What changed in the leverage map

Optimized32K generated FFN remains44.200ms/64.96%, attention12.43%; these are prior
stage attributions, not new timing. Source inspection finds an already-existing
fused Q8 gate/up/SwiGLU primitive compatible in source geometry with Muse's
H6656/F19968 default lcpp GEMV. It deserves an independent bounded reuse screen,
not a new shader or a prerequisite successful sparsity experiment.

| Source quantity | Per layer unless stated | Meaning |
| --- | ---: | --- |
| Each Q8 matrix | 141213696B | Unique serialized34B/32-element block payload |
| All52 FFN layers | 22029336576B | Gate+up+down payload, unchanged by fusion |
| Dispatches removed | 104 pertoken | Gate/up/product3->1; down/norm/residual unchanged |
| Intermediate operations avoided | 319488B |5F->1F F32 reads/writes |
| Repeated x loads avoided | 265814016B | Source-issued loads acrossF/2 threadgroups |
| Unique x footprint | 26624B | Same vector reused, cache service unknown |
| Dynamic threadgroup scratch | 256->512B | Same128threads; accumulator pressure also increases |

None of these is measured DRAM traffic or a latency prediction. The initial cx
review understated intermediate savings as3F and counted x only once. Reading
the actual lcpp loop establishes4F savings and per-two-row threadgroup x reuse;
the reviewer confirmed both corrections. This prevents both overclaiming bandwidth
and incorrectly dismissing fusion solely from the tiny unique activation footprint.

## CPU-only implementation and checks

The existing `ffn_inner_census.py` gains schema2 gate/up/inner streams and Q8block32
scoring while preserving schema1 operation/defaultblock256. No capture producer or
new execution flag was added. Schema2 requires complete declared-layer coverage,
matching finite tensor shapes/lengths/hashes and retained producer identity fields.
Model/source/prefix identities are assertions, not authenticated by the scorer.

Reports distinguish the down-only post-product oracle (at most1/3 of FFN payload)
from the unproven post-gate up+down hypothesis (at most2/3). Raw gate versus F64
SiLU proxy, observed product energy, physical-block occupancy, zero-energy
denominators and perlayer results stay explicit. Product-energy budgets are not
down-output or model-quality certificates. Selection/repacking costs remain unpriced.

Eleven CPU tests PASS in0.123s:

- Exact Muse source ledger, invalid geometry and Q8 block accounting.
- Small gate/large up counterexample; scattered zeros fail fullblock occupancy.
- Post-product masks cannot claim saved gate/up work; negative raw gate is distinct
  from its small SiLU proxy; zero-energy treatment is explicit.
- Nonfinite/mismatched streams, malformed metadata, incomplete layers and ambiguous
  threshold keys reject. File hashes cannot override shape/byte mismatch.
- Synthetic schema1/schema2 CLI runs, including custom thresholds without zero.

First9-test packet also passed0.129s; review added coverage/semantics tests, not a
relaxed failing gate. Logs: `target/profiles/2026-09-17-muse-ffn-cpu-{01,02}.log`.
Tests use synthetic fixtures only; there are **no new Muse activation observations**.

## Adversarial review and next decision

cx Luna `01a0b1dd-70c4-74b0-8dbf-e387726bbe14` reviewed source, accounting, scorer
and protocol. Complete-layer coverage, proxy/raw distinction, provenance scope
and zero-vector denominators were strengthened. Final read-only review found no
material CPU/protocol blocker. No additional observer framework is justified now.

The future protocol freezes an existing-primitive bitwise screen and>=10% isolated
GPU savings in every selected cell with<=5%controls/no wall regression. This earns
only an all-layer native qualification, not production promotion. A separate
fixedblock/down-only sparsity census needs>=30%down blocks removable at1e-3 inner
energy (10%all-FFN serialized payload), including both token halves, merely to earn
further cost/output-error study. See `PROTOCOL.md`; neither experiment ran here.

## Native packet prepared, not executed

Source `77bce02a` implements the ignored test
`muse_glimmer_text_session::tests::muse_ffn_capture_and_fusion_screen` and its
test-only forward capture hooks. No production branch, shader, session buffer,
selector, or execution switch changes. All resource/capture fields disappear from
non-test compilation; the existing fused primitive is called only by the screen.

The packet restores the pinned32640-row state, extends128 rows once, and captures
eight teacher-forced generated transitions across all52 layers. It is prepared to
compare ordinary/observed full logits and activeKV bitwise at every step with
immutable prefix, then export finite poisoned gate/up/product streams plus normalized
input sidecar. Model authentication uses the existing pinned full-shard loader;
prefix provenance is explicitly fixed historical replay, not authenticated fresh
history. Scorer output now includes first/second sample-half oracle distributions.

Six fixed primitive cells (layers0/25/51, samples0/7) compare both baseline and fused
outputs against captured deployed products. Preflight checks actual lcpp/fused
kernel names. Timing has fixed64 repetitions, warmAB then oneABBA percell, separate
measured output buffers, post-bracket exact/guard/input-immutability checks, explicit
all-cell HOLD or advance-to-native verdict. No arbitrary repeat choice or retiming.

Validation performed without GPU:

- Rust CPU contract test PASS0.00s; ignored native packet compiles. Checks include
  416-slot layer/sample boundaries, invalid/overflow positions, checked host byte ranges,
 110755840-byte observer accounting, bothpair/control/wall gates and invalid timing.
- All-target CLI check PASS without warnings; cargo fmt and diff checks PASS.
- Eleven Python scorer tests PASS0.155s, including schema1/schema2 CLI and new
  sample-half denominator assertions. These are synthetic fixtures, not Muse data.
- Initial compile/CPU packet also passed. Final raw logs are
  `target/profiles/2026-09-17-muse-ffn-packet-{cpu-02,check-02,python-01}.log`.

cx Luna `01a0b201-e0d8-7cf3-8b4c-3c89993d3b94` reviewed design and implementation.
Checked write bounds, poison offsets, pre-Metal artifact collision handling and
loader-backed shard identity were strengthened. We also fixed Cargo package-CWD
versus worktree artifact resolution. The review's Path comparison objection was
incorrect: fixture.path() returns&str, confirmed by source and successful compile.
Its initial requirement for native candidate replay belongs to later all-model
qualification, not this isolated screen. Different source commits alone do not
invalidate a hash-pinned common-state experiment; the prefix scope is explicit.
Final review finds no remaining serious compile-only blocker.

**No native packet ran, no capture was produced, and no fusion speedup or numerical
equivalence is established.** First coordinate GPU access, close the existing Muse
default/alignment delivery gate, then execute this frozen packet once. No server
process or production executable was touched by this preparation.

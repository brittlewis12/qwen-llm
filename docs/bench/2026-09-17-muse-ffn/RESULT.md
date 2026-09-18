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

# v0.597 A3B Anonymous Owned-Arena Floor

Status: **GO for four-worker arena C only**. Serial arena B is rejected. The
result authorizes one force-only A3B loader pilot.

Frozen source: `1b7a897075ea3fddb3c5b0232007467a5d7b6655`.
Canonical packet: `target/profiles/v0597-a3b-owned-arena-floor-p1/`.

## Result

All six prospective `ABC/BCA/CAB/CBA/ACB/BAC` blocks passed on attempt one.
Every arm reproduced the frozen model, descriptor, inventory, planner, resource,
binding, byte, and correctness contracts. No timer-local major faults, block
input, pageout, or swap growth occurred.

| Arm | Median ready | Median copy | Rate | Paired ratio |
| --- | ---: | ---: | ---: | ---: |
| A: 733 copied buffers | `2062.791 ms` | fused | `10.725 GB/s` | `1.000x` |
| B: serial owned arena | `3584.763 ms` | `3584.218 ms` | `6.172 GB/s` | `1.73227x` |
| C: four-worker arena | `702.167 ms` | `701.612 ms` | `31.532 GB/s` | `0.33942x` |

C saves a paired median `1365.044 ms`, wins all six blocks, and ranges only
`0.33440-0.34319x` of A. Applied to v0.595's frozen `2463.13 ms` copied
first-byte row, the arithmetic projection is `2.24311x`.

B loses a paired median `1515.518 ms` and wins zero blocks. Reducing resource
count without parallelizing first-touch and copy is not useful.

## Mechanism

B and C allocate the same one-window plus one-fallback geometry. Their median
allocation walls are only `0.559/0.545 ms`; nearly the entire difference is copy
wall. B reaches only `6.172 GB/s` with one host copy. C uses four page-aligned,
disjoint scoped workers and reaches `31.532 GB/s`.

The result is not deferred allocation:

- all arms incur about 2.70 million timer-local minor faults;
- every C destination byte, planner gap, fallback, and binding is verified;
- C workers join before the authoritative endpoint;
- maximum RSS is about 44.31 GB for every arm;
- peak footprint is about 22.19 GB for every arm;
- Metal allocation returns to the exact 475,136-byte baseline after each arm.

The floor does not fully attribute C's superlinear improvement over B. Parallel
destination first-touch/page zeroing and copy implementation efficiency are
credible contributors. The pilot must reproduce the complete first-byte gain;
the floor does not convert this attribution into product authority.

## Authority

Implement one default-off A3B pilot with exactly:

- production-auto native-Q8 direct inventory;
- one anonymous shared planner window plus one 8,192-byte fallback;
- complete-window copy, including gaps, over the measured four page-aligned
  worker ranges;
- serial fallback copy;
- planner-relative typed views retained for model lifetime.

The pilot must compare copied, owned, and file-backed retained storage in one
build. It must independently clear bit-exact full state, `>=1.20x` fresh first
byte, `>=0.99x` late decode and warm prefill, and no loaded output-128 request
regression.

No authority follows for default-on selection, serial arena B, aliases,
conversions, MTP, split shards, other models, asynchronous promotion, memory
savings, storage-cold behavior, or broad loader integration.

Adversarial design and result review: `cx ask` session
`019f61c6-cb2e-7cc3-8e34-5011e456fe6d`.

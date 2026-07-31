# v0.655 Grammar-Row Lm-Head Charged Floor

Status: preregistration. No v0.655 manifest emitter, row bank, Metal view seam,
benchmark command, timing observation, or decision exists. The only new
observation preserved with this preregistration is the CPU-only A3B topology
transfer produced by the unmodified v0.654 implementation at `1983711`.

## Question And Authority

Can one fully charged, 32-byte-aligned, state-major Q6_K row bank replace the
full-vocabulary head cheaply enough to justify one later A3B integrated grammar
packet?

A GO may authorize only one later A3B integrated packet for:

- the exact response-shape finite language and tokenizer policy from v0.654;
- the exact A3B model profile frozen below;
- the unchanged production Q6_K mat-vec;
- exact constrained selected logits and sampler-v1 semantics; and
- a prebuilt state-major branch bank with the frozen layout.

It grants no grammar-runtime API, product path, default policy, real-hidden
result, full-logit equivalence, other schema/tokenizer/model transfer, or product
speedup claim. Dense is a conditional guardrail and has separate transfer
authority. A dense performance miss cannot revoke an already-cleared A3B
screen. Any shared correctness, identity, bounds, byte-accounting, or Metal
validity failure kills the organization for both profiles.

Implementation may use only `GgufFile + Model` descriptor binding and may upload
only `output.weight`, the compact bank, and benchmark scratch. `MetalModel`, full-
model loading, production decode wiring, and product grammar code are forbidden.

## Frozen Topology

The A3B transfer command, run with no model forward or GPU command, was:

```sh
target/release/qwen-grammar-oracle \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --grammar docs/bench/v0654-grammar-response-shape.json \
  --traces docs/bench/v0654-grammar-response-shape-traces.json \
  --serial-transition-ms 9.32 \
  --fixed-n8-packet-ms 117.633 \
  --output docs/bench/v0655-a3b-grammar-topology.json
```

The source commit was `1983711911ced1e44fdc8f29ff54bc924fdfc68b`; the
release binary SHA-256 was
`ce959ad5fceaf3c77eac16844379106c65160502a00fc69ea957c9b74bfdbb23`.
The complete output SHA-256 is
`98c5c51296f4580b3dd6d7d4122c3d6dfd175ae36a90801f728fbc514b57c4fe`.

A3B exactly matches dense v0.654 on:

- grammar-piece-policy SHA-256
  `58748dfe811c1710c732239ae59053f2888bf32c1c3ee4f8ec10f0ac089fdd60`;
- finite-language SHA-256
  `94dc537ecb3d9209cfbc8e28b502916b20ee7b685393b6cf85ab610c08be0faf`;
- state SHA-256
  `afa64a9a91677309fffbc54833b73d8881d7c4745127ae5ec10eebc87d70be40`;
- canonical-path SHA-256
  `b72213294b55231dd4eaa12a05276be7dfe4e5ce994ff7a9451c258876333401`;
- 451 branch states, 1,468 branch incidences, 222 unique branch rows, and
  observed widths `{2,3,4,5,6,7,11,12,17}`.

This identity transfers exact branch prefixes and token IDs. It does not
transfer performance.

## Exact Model Profiles

Primary A3B:

- path `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`;
- file size `22,134,528,992`;
- model SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`;
- MoE, 40 layers, hidden 2,048, vocabulary 248,320, untied head;
- `output.weight` Q6_K `[2048,248320]`, `417,177,600` bytes;
- row payload `(2048 / 256) * 210 = 1,680` bytes; and
- screening whole-token denominator `T0 = 9.32 ms`.

Conditional dense guard:

- path `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`;
- file size `16,817,244,384`;
- model SHA-256
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`;
- dense, 64 layers, hidden 5,120, vocabulary 248,320, untied head;
- `output.weight` Q6_K `[5120,248320]`, `1,042,944,000` bytes;
- row payload `(5120 / 256) * 210 = 4,200` bytes; and
- screening transition denominator `T0 = 38.619 ms`.

The command must authenticate these exact profiles before source-row access or
Metal allocation. Model hashing and normal full-head upload establish the
fixture and are not incremental grammar cost. Both remain reported.

## Manifest Contract

The first implementation stage may only add optional deterministic branch-
manifest emission to `qwen-grammar-oracle`. The emitted JSON must bind:

- schema, grammar, piece-policy, finite-language, state, and canonical-path
  digests;
- all 451 lexicographically ordered branch prefixes;
- strictly ascending, unique, in-range token IDs for every state;
- exact row-count histogram, 1,468 incidences, and 222 unique branch rows;
- canonical state/width paths for all 36 equally weighted 18-token strings;
- the 20-record/360-incidence descriptive trace and separate source strata; and
- deterministic state and token-set digests.

The manifest is generated CPU-only, reviewed, and committed with its SHA-256
before benchmark implementation. The timed command parses only that frozen
manifest and performs no grammar analysis.

## Physical Bank

For state width `w` and profile row payload `R`, freeze:

```text
payload(w) = w * R
span(w)    = align_up(payload(w), 32)
offset(i)  = sum(span(previous states))
bank_bytes = sum(span(all states))
```

Trailing padding after the final state is included. Exact totals are:

| Profile | Payload | Padding | Bank span |
|---|---:|---:|---:|
| A3B | 2,466,240 | 2,720 | 2,468,960 |
| Dense | 6,165,600 | 5,408 | 6,171,008 |

Copy every complete source row in state order and ascending token-ID order.
Poison all internal padding and hash the complete physical bank before upload
and after correctness dispatches. Prefix/suffix output guards are separate and
do not count as bank bytes.

Do not make caller-selected `owned_weight_view` public. Add only a narrow
public or doc-hidden Q6_K row-bank view constructor that hardcodes 32-byte
offset alignment, validates `[n_in,n_out]`, checked Q6_K byte extent and buffer
bounds, and returns read-only weight provenance. No kernel change is allowed.

## Incremental Setup Cost

`B`, the incremental bank cost, begins before the first manifest read and ends
only after every view and compact output is usable. No prior manifest read is
allowed. It includes:

- file read, SHA-256, parse, and all manifest validation;
- state metadata, token maps, offsets, and checked arithmetic;
- all 1,468 source-row copies and internal padding;
- complete bank hash;
- Metal allocation and upload;
- per-state Q6_K read-only view binding; and
- compact output plus guard scratch.

Report each phase and total `B`. Full-head upload, model/tensor identity hashing,
PSO compilation, correctness readback, and warmup are reported but excluded
from `B` because they establish or test the common fixture.

## Correctness Before Timing

Use exactly four deterministic nonzero F32 hidden vectors. For seeds
`{1,2,3,4}`, initialize `s=seed`; for each element update
`s = s*6364136223846793005 + 1442695040888963407` with wrapping u64 arithmetic,
then set `x_i = ((((s >> 40) & 0xffff) as i32) - 32768) as f32 / 32768.0`.
Seed 1 is the sole timed vector. Dispatch the full head, then every branch-state
view. Require every compact result bit-identical to its corresponding original-
vocabulary full-head value. Poison compact outputs before correctness dispatch
and require complete overwrite. Require intact output guards, internal bank
padding, source-row hashes, bank hashes, command status, and Metal errors.

Sampler-v1 correctness uses strictly ascending original token IDs and freezes:

- greedy finite distinct logits and finite ties;
- `qwen_chat` seeds `0`, `1`, `42`, and `u64::MAX`;
- positive finite ties and positive-infinity ties;
- top-k smaller than the admissible set;
- NaN at every local position, remapped to the original token ID; and
- all admissible logits `-inf`, which must fail closed before sampling.

Compare compact selection with a full-vocabulary `-inf` grammar mask for every
nondegenerate case. Require the same original token, candidate index, sampler
draw count, and error identity. Equal decoded pieces remain distinct token IDs.
Singletons and terminals are outside the bank; canonical paths receive no
singleton credit. After all correctness dispatches, hash the complete uploaded
Shared Metal bank buffer, including internal and trailing padding, and require
equality with the pre-upload CPU-bank hash.

## Conditioning And Timing

The product body is absent from this primitive floor, so every scored arm first
runs one unscored full-head streaming dispatch and waits outside the timer. This
common predecessor must complete successfully and conservatively evicts the
small bank before both full and restricted observations.

Each timed command buffer contains exactly one unchanged production Q6_K
mat-vec dispatch. The conditioning dispatch uses its own unscored command
buffer. No poison, fill, second mat-vec, or other GPU dispatch may enter a timed
command buffer.

The primary charged-head wall starts before a checked vector-index state lookup
and ends after:

```text
state lookup
-> command-buffer creation and encoding
-> commit, wait, status, and error check
-> selected-logit readback or gather
-> Sampler::sample
-> local or error token mapping
```

Report lookup, create+encode, commit-wait, GPU timestamps, readback/gather,
sampler+mapping, and total wall separately. The timed sampler is exactly sampler
version 1 `qwen_chat(0)` at draw zero; construct identical sampler clones before
each arm's timer. Other configurations are correctness-only.

Measure all widths `{2,3,4,5,6,7,11,12,17}`. The timed state for each width is
the lexicographically first manifest branch state of that width. Freeze
`A = full head` and `B = compact state view`.

Warmup is exactly five complete arm-by-width rounds. Rotate width order each
round and alternate `AB`, `BA`; every unscored arm still receives the common
conditioning dispatch. Then acquire the full-control screen as exactly 12
rounds. Rotate width order each round; for every width run one conditioning
command buffer followed by one timed A command buffer. Reduce the per-width
screen by median A wall, then weight those nine medians by canonical incidence
to obtain screening `H0`.

If screening continues, run exactly 12 paired rounds per width. Alternate
adjacent pair direction `AB`, `BA`, so each two rounds form ABBA, and rotate
width order each round. Before each scored A or B, run the common conditioning
dispatch. Preserve raw samples, arm order, predecessor, and conditioning status.

For each width report arm medians and median paired wall saving. Every width is
important and must have positive median paired saving; individual noisy pairs
need not all win.

## Weighting And Economics

The primary weights are the 648 exact branch incidences on the 36 equally
weighted canonical 18-token paths:

```text
q_w = canonical incidences of width w / 648
H0  = sum(q_w * adjacent full-control median at w)
H1  = sum(q_w * compact median at w)
```

Report the uniform 451-state topology weighting and frozen 20-record/360-
incidence trace weighting as diagnostics only. For every canonical path compute:

```text
Net(path) = sum over its 18 widths [H0(w) - H1(w)] - B
```

Also report uniform-path mean net. Do not replace minimum path net with
`18 * (H0 - H1) - B`.

Screening projections are:

```text
head removal       = 1 - H1/H0
whole-token saving = (H0 - H1) / T0
```

These use frozen external `T0` denominators and are not integrated product
measurements.

## Stopping Rule And Gates

Run order is fixed:

1. A3B identity, setup, and untimed correctness.
2. Recorded A3B full-control-only screening.
3. If weighted `H0/T0 <= 5%`, stop before scored candidate timing as
   mathematically incapable of clearing the whole-token gate.
4. Run A3B paired acquisition and recompute final paired `H0/T0`.
5. Run dense only if A3B clears every GO gate.

A3B GO-to-integrated-packet requires all of:

- every identity, layout, source-row, bank, view, correctness, sampler, guard,
  command, and accounting check passes;
- positive median paired wall saving at all nine widths;
- weighted charged head removal `>=70%`;
- `(H0 - H1) / 9.32 >= 5%`;
- every one of the 36 B-charged canonical path nets is positive; and
- no host or Metal validity failure.

Equivalently, A3B must satisfy
`H1 <= min(0.30 * H0, H0 - 0.466 ms)`.

After A3B GO, dense runs the same correctness and performance contract with
`T0=38.619 ms`, equivalently
`H1 <= min(0.30 * H0, H0 - 1.93095 ms)`. A dense performance miss closes only
dense transfer. A dense shared correctness or validity failure kills the common
organization and revokes A3B authority.

Any A3B non-GO ends v0.655 without a dense run or an optional diagnostic. A
different causal diagnostic requires a new preregistration. No result may be
rescued by changing widths, weights, setup accounting, conditioning, sampler,
or denominators after observation.

## Validation Before GPU Work

- CPU tests must cover manifest order, digests, counts, canonical weights,
  malformed states, Q6_K geometry, checked row ranges, duplicate incidences,
  padding, exact profile totals, token mapping, ties, NaN, infinities, and all-
  negative-infinity rejection.
- The benchmark command must preserve qwen-bench build-identity enforcement,
  archive relevant `QWEN_*` values, emit one JSON document with raw samples, and
  reject noncanonical warmup/sample counts.
- Timed GPU work remains serialized under the normal quiet-box contract.

Independent design review: `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.

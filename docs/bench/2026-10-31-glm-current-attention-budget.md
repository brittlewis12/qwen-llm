# Current GLM MLA Attention-Only Budget

## Decision And Scope

The unchanged current attention dispatch costs about **21.8 ms** at the fixed
position-zero 2048-row shape on M5 Ultra. This leaves enough room to fund one
fixed fused tensor-attention source/CPU gate. It is not a candidate speedup,
whole-MLA measurement, exclusive affected fraction, or model qualification.

The [current coarse profile](2026-10-31-glm-current-post-c-2k-budget.md) measures
376.432251 ms across eleven complete MLA blocks. That parent includes projections,
the grouped Q8 absorption/expansion pair, cache and index publication, and attention.
Do not subtract this synthetic leaf measurement from that parent to invent other
leaf costs, or multiply one dispatch's potential saving into an application gain.

The continuation target remains **both at least 30% and at least 8 ms saved per
complete MLA block**. At the parent's average 34.22 ms/block, 30% is about
10.27 ms. That is planning context, not a measured block-specific denominator or
proof that a fused candidate can save it. Candidate overhead and ordinary-model
conversion still have to be measured.

## Frozen Current Path

Experimental source is `f2add63d04c38594efed62348070ee110a07daed`, under
`scripts/bench/glm-mla-current-attention/` on its measurement branch. No production
shader, encoder, selector, precision policy or default changes for this packet.
The host calls only public `encode_latent_attention` through the ordinary serial
encoder:

```text
Q/output: F32 [512, 64, 2048], dimension-fastest
cache: F16 [512, 2048], one shared key/value head
sinks: F32 [64], public finite LATENT_NO_SINK = -1e30
start = 0; rows = 2048; scale = 1/16; inclusive causal visibility
kernel_deepseek_v4_grouped_online_dense_sink_attention_f16
grid [2048, 8, 1]; threads [256, 1, 1]
```

This kernel belongs to the primary MSL 3.1 library, built with SDK 26.2,
Metal 32023.864, `-O3 -ffast-math`, deployment target 14.0. It is not the optional
MSL 4.0 TensorOps library. The finite sink must not be replaced with negative
infinity. Encoder/kernel bodies and whole source files are byte-identical across
the cached build lineage, cache-admission release, coarse-profile source and
measurement base. Full-span hashing verifies the embedded primary metallib.

The standalone host links the exact cached production rlib using configured
`rustc`, opt1 and no LTO; cached inputs are hash-checked before and after. This is
a GPU device-cost fixture, **not** a canonical application/host timing artifact
or a fresh production rebuild. It does not load model weights or a tokenizer.

## Correctness Before Timing

The separate DEBUG1 process completes one checked dispatch and exits successfully
in four seconds. Independent CPU replay verifies all four physical binding hashes,
2048 guard bytes, every regenerated Q/cache/sink input byte, and finiteness of all
67,108,864 output floats.

The fixed recipe keeps Q in F32 and rounds cache entries to actual Half before
reference evaluation. Independent F64 causal attention uses raw Q/cache bytes at
eleven query rows, five heads and six dimensions: 330 selected outputs. Maximum
absolute error is **2.6963301647e-8**, below the unchanged current attention-test
tolerance of **2e-4**. The independent reference SHA256 is
`f7d7a033d036b71a867c12925e3556faf43ce5be188102b56b138649ed51a430`.

Only those 330 outputs receive mathematical reference checks. Full-output
finiteness/hash is not mathematical proof for unsampled outputs; a singleton
validation does not establish repeatability or model quality. Source/CPU controls
also check actual-Half rounding, causal/tail/layout behavior, poison and guard
refusal, resource/lifetime rules, and the fixed timing sequencer without a device.

## Fixed DEBUG0 Timing

After independent validation acceptance, a separate process runs the same frozen
binary with DEBUG0. One untimed validation dispatch precedes two blocks, each
with eight predefined excluded warmups and sixteen retained samples. There are
49 commands/dispatches, five full checkpoints and 48 preserved raw intervals.
Each command contains exactly one unchanged attention dispatch.

| GPU ms, Retained Samples | Block 0 | Block 1 |
| --- | ---: | ---: |
| Count | 16 | 16 |
| Median | 21.786396 | 21.784458 |
| Mean | 22.272971 | 22.286622 |
| Minimum | 21.764208 | 21.750875 |
| Maximum | 28.947125 | 29.241750 |

Both maxima are the first retained samples and remain included. There is no
outlier removal, retry, or clock/thermal attribution. Independent timing review
accepts all interval/order/retention checks and raw-dump equality. Block median
drift is -0.00889%; within-block half-median drift is -0.04110% and +0.03586%.
Separate standalone host-wall
medians are 22.015271 and 22.001896 ms; they include admission, command factory,
encoding, checked wait and pool work and are not canonical application latency.

GPU start/end timestamps are read only after checked completion. Full hashes,
guards and reference checks run outside uninterrupted sample loops. The checked
last output after each warmup/block matches the initial baseline; these checkpoints
do not audit every intermediate dispatch. Final guarded raw bytes are identical
to the DEBUG1 capture, with output SHA256
`4e6193749686588f000b0a69bc435f32fc59135d594281ad6025e0d20e9651d7`.

Timing completes successfully in nine seconds. Each process uses the normal
fail-fast canonical self-lease with WAIT0, an actual external 90-second timeout
and 20-second TERM-to-KILL grace, without a wired/lease bypass. Both end naturally;
post-exit lock inspection finds no openers. CPU work runs unleased. No server,
installed binary, PATH, SDK selection or service state changes.

## Actual Resource Gate

The actual PSO has SIMD width 32, maximum 256 threads, zero static LDS and
16384 bytes dynamic LDS, within the device's 32768-byte limit. Four distinct
owned Shared buffers have 256-byte front/back guards. Dtype, shape, writability,
device, physical addresses/ranges, disjointness and actual prices are checked
before mapped initialization. Inputs and output stay owned through checked
completion and submitted error/unwind drains; no precommit command is waited.

```text
guarded buffer bytes = 538970368
actual device-priced B = 539033600
P_Metal policy = 100663296
C_CPU explicit = 547358976
  includes dump-file-cache policy 538970368, not a full user-space clone
R shared reserve = 33554432, passed once
B + P + C + R = 1220610304 < separate 2147483648-byte fixture cap
Q separate CPU framework policy = 134217728
```

This is a new shape-sized ledger, not an increase to the earlier N64 128 MiB cap.
Policies are not formal driver/allocator peaks. There is no incumbent/backing
credit, full CPU initializer, full-output clone, or CPU/Metal-domain borrowing.
The complete guarded dump streams in bounded chunks after terminal completion.

## Evidence And Next Gate

| Frozen Evidence | SHA256 |
| --- | --- |
| Standalone executable | `04009c726df753d83a1f859c0838cbc3c4d4ad29690fd44c5c0600658e414ed1` |
| Cached production rlib | `4d30e61480b395f00b5ec3432eaade2bf8aecad477b5a914dc5ca27fcf6e6fa4` |
| Embedded primary metallib | `2c2b3f58cfc836e19b4a4e636bad64a8194bf9c96aedc3e5f55dd0bfbe5f06bc` |
| DEBUG1 report | `c4b870115bd7df160dbccc1641de3d64bc91abec5d27d0a4c85334d5715fc32c` |
| Independent validation replay | `f3e780941c63df779448f489e745edc5e8d42035b6c3f6dad42ac8a3693b724f` |
| DEBUG0 report | `c7d27625da6493d12636c3ac17c7c2da36fa64bbb8f5eadafc36390b5b6e33b4` |
| Independent timing analysis | `6f80c37ac728e8e73b0639d9f30175e01caeb1d172457c0e59b164f94e13b40b` |
| Guarded raw dump, identical in both modes | `182b909eba58ed16d285017afb3a8286240a00b7023f9eaf329d59453dc7a0cb` |

The [mixed-operand gate](2026-10-31-glm-mla-mixed-operand-gate.md) already verifies
bounded F32/Half QK and independently supplied F32 P/Half V arithmetic and
role-specific physical ownership. It does not qualify fused exponentiation,
denominator/rescaling, seeded PV accumulation, full attention or performance.
Those are new proof obligations for the fixed fused design. Source/CPU/compile
review comes first; no GPU stage or model integration follows automatically.

Synthetic operands, reused cache residency, the standalone command composition
and device state can differ from ordinary MLA. This measurement supplies a
useful investment discriminator, not a formal saving bound or roofline estimate.
Closed K64, RowOwned and fixed GROUP32 INT8 families remain closed.

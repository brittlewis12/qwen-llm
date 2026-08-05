# DeepSeek V4 Packed-Prefill Attribution

Status: attribution `GO`; direct all-slot packed expert execution `KILL` as a
performance ceiling. Exact GPU route-and-schedule and grouped expert compute
remain separate bounded falsifiers.

## Question

The packed path submits and waits twice per layer: once after attention through
the router projection, then again after CPU routing and expert-major execution.
The old aggregate labels made the first phase look like a 58% router bottleneck.
This packet asks how much time is actually removable host routing or command
boundary overhead, and how much is useful GPU work on either side.

## Instrument

`QWEN_DSV4_PREFILL_TRACE=1` records, per layer and after the complete chunk:

- pre-expert wall, encode, command-GPU, wait, signed wait residual, and
  post-completion validation;
- host route, stable expert-major schedule construction, and upload wall;
- post-route wall, encode, command-GPU, wait, and signed wait residual;
- active expert-bucket count.

Apart from one environment lookup, Metal timestamp queries, clocks, records,
and output are absent from the ordinary path. Records are buffered until the
chunk completes, so output does not perturb layer-to-layer execution. Signed
residuals are not clamped; totals report negative samples explicitly.

The captured executable predates the final terminology cleanup. Its raw
`router` fields mean `pre_expert`, and `experts` means `post_route`, which also
contains the shared expert, residual composition, and final head work.

## Fixture

- Engine revision: `3d6e4fc3a071a6f3c56c9e33fc3d83d723cc8e75`
- Device: Apple M4 Max
- Rust: 1.97.1 (`8bab26f4f68e0e26f0bb7960be334d5b520ea452`)
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Input: `probe_code.json`, official 0731 chat renderer, reasoning `high`
- Prompt: 140 tokens, packed as 128 plus 12
- Output: one token; timing authority ends at packed prefill

Command shape:

```text
QWEN_DSV4_PREFILL_TRACE=1 target/release/qwen \
  -m .../DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
  --messages .../probe_code.json --reasoning high -n 1
```

The attributed and ordinary runs produce generated ID 2581 and measure
6,065.5/6,064.5 ms prefill. Raw local log hashes are:

- ordinary: `e6f4c6cb8e76236c18d43043ba918dd352f8caceddc0f7c1e46b6834ff47605f`
- attributed: `66e38a71a76a570d160759dd465bd944791c4401391d23cd5bc666b35e4a4b9c`

## Results

| Chunk | Pre-expert wall | Pre-expert GPU | Host route/schedule | Post-route wall | Post-route GPU |
|---|---:|---:|---:|---:|---:|
| 128 tokens | 1.920 s | 1.077 s | 0.026 s | 2.836 s | 2.810 s |
| 12 tokens | 0.635 s | 0.225 s | 0.004 s | 0.637 s | 0.624 s |
| Total | 2.555 s | 1.302 s | 0.030 s | 3.473 s | 3.434 s |

Ordinary per-command wait residual is about 0.2 ms. Across 86 waits, that puts
the directly defensible command-boundary ceiling near 17 ms for this request.
Adding all CPU route/schedule work gives only about 47 ms, or 0.8% of the
6.065-second packed-prefill wall.

Large pre-expert residuals concentrate at a few first-touch layers: 0, 21, 36,
and 41 in the 128-token chunk; 0, 39, and 41 in the 12-token tail. They are
consistent with residency, queue, driver, or wakeup effects, but this packet
does not identify their cause and does not price them as removable routing
overhead.

## Decision

The old 58% router label is retired. It combined attention, compressors, mHC,
router projection GPU work, the command wait, and host routing. CPU route and
schedule construction are too small to justify a broad packed rewrite alone.

Proceed with an exact GPU route-and-schedule microproof because it is the
dependency that permits command merging and future GPU consumers. Do not use a
token-slot all-slot expert backend as the performance ceiling: the current CPU
path already groups by expert and reuses each selected bank across rows.

After routing is proven, falsify only the dominant current-asset grouped pair:
IQ2_XS gate/up with IQ3_XXS down. Require a meaningful aggregate N=128 win
before widening to the remaining dtype matrix. MXFP4 down may retain its
two-layer row fallback initially.

CX review session: `019fcf7d-e9d4-7150-b496-e70a31958e80`.

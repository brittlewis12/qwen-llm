# Muse Full-R Block-Major R256 KEEP

Decision: **KEEP** the block-major full-transport engine. It reuses one
prompt/layer replay across eight B32 reverse banks, clears the predeclared 15%
wall gate by a wide margin, and remains bitwise equal to chunk-major execution
over the complete released-geometry R256 payload.

## Contract

- released `Muse-Glimmer-30B-Q8_0.gguf` on the internal SSD;
- release test binary built from the candidate diff on parent `e3acfc38`;
- explicit token IDs 1 through 16, R rule, target 51, sources 0 through 50;
- output rows `[0,256)`, inner query batch 32;
- candidate first, then chunk-major control in one process;
- no artifact or checkpoint publication;
- one invocation, hard-stopped at 180 seconds.

The complete test finished in `79.37 s`, including a `1.180145 s` capture.
The mechanism gate compares only candidate and control engine wall from the
same captured prompt.

## Result

| Measure | Block-major | Chunk-major | Change |
| --- | ---: | ---: | ---: |
| external engine wall | `32.817480 s` | `45.051726 s` | `1.372797x` |
| replay | `1.178145 s` | `9.861465 s` | `8.683320 s` saved |
| full-attention bank | `6.043182 s` | `6.191810 s` | `0.148628 s` saved |
| feed-forward reverse | `14.231502 s` | `14.978422 s` | `0.746920 s` saved |
| attention-output reverse | `1.187842 s` | `1.234069 s` | `0.046227 s` saved |
| CPU attention reverse | `3.757799 s` | `3.958215 s` | `0.200417 s` saved |
| attention-input reverse | `4.932580 s` | `5.355285 s` | `0.422705 s` saved |

Candidate wall falls by `27.156%`, versus the predeclared `15%` KEEP floor.
Replay collapse explains `70.98%` of the `12.234246 s` external saving. The
candidate components account for `31.331050 s`, or `95.47%` of its outer wall.

## Correctness And Bounds

Candidate and control agree bitwise over all `86,900,736` F32 values in the
`[51,256,6656]` result: max absolute error and RMS error are both zero. Replay
diagnostics also match exactly. The model-free Q32+Q1 mixed-full/sliding gate
independently passes under both J and R.

The engine retains at most 256 logical rows and executes every reverse in a
physical batch of at most 32. It removes the transient `[51,32,16,6656]`
composed source bank and reduces each reached source directly into `[S,R,H]`.
The run neither invoked checkpoint identity nor hashed model-weight bytes.

## Projection And Disposition

At 26 R256 shards and 25 prompts, measured engine wall projects to `5.925 h`,
down from the in-process chunk-major control projection of `8.134 h`. Capture,
checkpoint I/O, and final assembly remain outside this projection; it is not an
end-to-end fit forecast.

The next leverage is the 38-block sliding path. Its feed-forward,
attention-output, CPU-attention, and attention-input phases total `24.109723 s`
per shard-prompt. Before changing the full fitter again, qualify one private
RoPE-correct all-Metal block-50 B32 seam against the prepared CPU fallback with
the existing 15% wall and numerical gates.

Adversarial promotion and updated leverage review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.

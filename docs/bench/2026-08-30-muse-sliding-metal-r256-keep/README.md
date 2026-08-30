# Muse Sliding Metal R256 KEEP

Decision: **KEEP** the RoPE-correct Metal bank for all 38 sliding blocks. The
private block-50 falsifier clears its 15% gate, and the promoted full R256
engine reduces wall by a further `16.734%` without reintroducing CPU attention.

## Representation Contract

The released GGUF projection weights use adjacent rotary pairs
`(0,1),(2,3),...`; HF/vLLM weights use a converter-permuted NeoX layout. The
native path therefore follows the existing Muse GGUF forward convention, not
the model-name-level convention in vLLM. Local MLX and llama.cpp implementations
confirm that the VJP is the same rotation with the sine sign inverted.

The new periodic kernel accepts compact F32 banks `[heads*D,B*T]`, derives
`position = row % T`, and applies inverse adjacent-pair RoPE in place. Sliding
replay publishes its rotated Q/K primals to resident tensors; causal GQA writes
rotated-space Q/K cotangents, which are inverse-rotated before Q/K norm VJPs.

## Block-50 Falsifier

- released Q8, block 50, T16/B32/R;
- one shared replay and cotangent bank;
- candidate/control warmup, then five alternating pairs;
- complete `3,407,872`-value comparison;
- no artifacts, checkpoints, identity resolution, or model hashing.

| Path | Median reverse wall |
| --- | ---: |
| periodic-RoPE Metal bank | `57.232417 ms` |
| prepared CPU sliding reverse | `77.519875 ms` |

The Metal path is `1.354475x` faster, a `26.17%` wall reduction. Numerical
agreement is relative L2 `9.10e-8`, scaled max `1.788e-7`, and cosine
`0.999999999999824`. The complete process finishes in `12.26 s`.

Model-free gates independently cover periodic forward/inverse round trips and
B1/B3/B32 sliding blocks under both J and R. The mixed full/sliding Q32+Q1
full-transport test remains within its existing differential gates.

## Promoted R256 Result

One candidate-only, no-publication R256 run reports:

| Measure | Seconds |
| --- | ---: |
| capture | `1.160776` |
| engine wall | `27.325800` |
| replay | `1.165753` |
| 13 full-block banks | `6.063598` |
| 38 sliding-block banks | `18.005166` |
| legacy CPU sliding phases | `0.000000` |

Sliding and full commands reach parity at `59.228` and `58.304 ms` per
block-bank. Against the prior block-major engine, R256 wall improves by
`1.200971x`; against the original chunk-major control it improves by
`1.648688x`. At 26 shards and 25 prompts, measured engine wall projects to
`4.934 h`, excluding capture, checkpoint I/O, and assembly.

Row/checkpoint and assembled full-transport artifacts move together to schema
v3 so full- and sliding-bank command timings remain distinct and factual.

## Disposition

The common Metal command train now owns `24.068764 s`, or `88.08%`, of R256
engine wall. The next source-first gate is one private released-Q8 B64 versus
two B32 falsifier on blocks 50 and 51. It must compare complete outputs, use
three alternating samples after warmup, clear 15% on both blocks, write nothing,
and hard-stop at 180 seconds.

Adversarial design, promotion, and updated leverage review:
`01a0516b-475c-7291-92c4-7017a79fa3d8`.

# Decode Unsupported Notes

This artifact records why the v0.352 Q6/Q8 MoE guardrail is prompt-only. Both
models win `pp512`, but `tg128` cannot run because the single-token routed gate/up
path has no native Q6_K or Q8_0 expert-bank kernels.

Commands:

```sh
target/release/qwen-bench suite \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q6_K.gguf \
  --tg 1 --runs 1 --no-warmup -o json

target/release/qwen-bench suite \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q8_0.gguf \
  --tg 1 --runs 1 --no-warmup -o json
```

Errors:

```text
v1 driver requires F32 weights; tensor MoE routed gate/up expert banks is Q6_K
v1 driver requires F32 weights; tensor MoE routed gate/up expert banks is Q8_0
```

Read: this is a decode coverage hole, not a prompt-path performance result.

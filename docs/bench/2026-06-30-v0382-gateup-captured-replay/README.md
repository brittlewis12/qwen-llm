# v0.382 Gate/Up Captured Replay

Adds `--route-capture-ctx` to `qwen-bench moe-gateup-micro`. The captured path
ramps a real decode session to the requested context, runs one real MoE token,
captures each Q4 routed gate/up layer's post-norm hidden vector plus top-k expert
ids, and replays those tensors in the isolated gate/up microbench.

## Commands

```bash
cargo fmt
cargo check -p qwen-cli --bin qwen-bench
cargo build --release -p qwen-cli --bin qwen-bench

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --warmup 3 --iters 10 \
  > target/profiles/v0382-a3b-q4-moe-gateup-micro-synthetic.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --warmup 3 --iters 10 --route-capture-ctx 8192 \
  > target/profiles/v0382-a3b-q4-moe-gateup-micro-capture-hidden-ctx8192.out

QWEN_PHASE_MOE_FFN_SPLIT=deep target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 8192 \
  > target/profiles/v0382-a3b-q4-phase-ctx8192-ffn-deep.out

target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --warmup 3 --iters 10 --route-capture-ctx 8192 \
  > target/profiles/v0382-a10b-q4xl-moe-gateup-micro-capture-hidden-ctx8192.out

QWEN_PHASE_MOE_FFN_SPLIT=deep target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0382-a10b-q4xl-phase-ctx8192-ffn-deep.out
```

Dirty NR4 was also rebuilt temporarily by changing `NR0_Q4K` from `2` to `4`,
then reverted before this checkpoint:

```bash
target/release/qwen-bench moe-gateup-micro \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --warmup 3 --iters 10 --route-capture-ctx 8192 \
  > target/profiles/v0382-dirty-nr4-a3b-q4-moe-gateup-micro-capture-hidden-ctx8192.out
```

## Results

| Row | Gate/up GPU | Comparator | Read |
| --- | ---: | ---: | --- |
| A3B synthetic micro | `1.3617 ms` | phase `1.08 ms` | still misleading |
| A3B captured replay ctx8192 | `1.0836 ms` | phase `1.08 ms` | phase-faithful |
| A10B captured replay ctx8192 | `3.1147 ms` | phase `3.19 ms` | within gate |
| A3B dirty NR4 captured replay | `1.1014 ms` | default `1.0836 ms` | false positive rejected |

The important finding is that captured expert ids alone are not enough. The
isolated Q4 gate/up kernel is sensitive enough to the input activation
distribution that replay must capture both the post-norm hidden vector and the
top-k expert ids. With both captured, the microbench tracks the deep phase split
on A3B and A10B and rejects the previously misleading NR4 synthetic win.

## Decision

Use `moe-gateup-micro --route-capture-ctx <ctx>` as the admission gate for any
future Q4 routed gate/up shape variant. Do not trust synthetic gate/up wins on
A3B. Since the captured default is already close to the phase budget, the next
actual speed work should be structural byte/dataflow reduction rather than more
row-width retuning.

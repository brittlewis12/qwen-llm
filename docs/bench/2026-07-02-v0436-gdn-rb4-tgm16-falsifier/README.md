# v0.436 GDN RB4 TGM16 Staging Falsifier

Goal: answer the narrow GDN recurrence question from the hardware-headroom
review: can the packed prefill recurrence get a win by sharing Q/K loads across
the existing four resident `dv` rows in one threadgroup?

Prototype shape, tested dirty and then removed:

- `kernel_gdn_step_decay_packed_rb4_tgm16_f32`
- One threadgroup owns four `dv` rows for one `hi`, matching the active NSG4
  production shape.
- State rows stay resident in registers across the token loop, like the active
  packed kernel.
- A `T=16` tile stages `q_tile[16][128]` and `k_tile[16][128]` in threadgroup
  memory (`16 KiB`) so the four rows share Q/K loads.
- Env gate used for the dirty run: `QWEN_GDN_STEP_PACKED_RB4_TGM16=1`.

Commands:

```sh
QWEN_GDN_STEP_PACKED_RB4_TGM16=1 cargo test --release -p qwen-llm \
  prefill_tokens_matches_single_token_loop_0_8b -- --nocapture

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --n-prompt 512 --runs 3 --cooldown-seconds 2 \
  --repeat-blocks 2 --shuffle-seed 4351 \
  --variant base --variant rb4:QWEN_GDN_STEP_PACKED_RB4_TGM16=1 \
  --output target/profiles/v0436-0p8b-pp512-gdn-rb4-sweep.json

uv run scripts/profile/prefill_sweep.py \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --n-prompt 512 --runs 2 --cooldown-seconds 5 \
  --repeat-blocks 1 --shuffle-seed 4352 \
  --variant base:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1 \
  --variant rb4:QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1,QWEN_GDN_STEP_PACKED_RB4_TGM16=1 \
  --output target/profiles/v0436-27b-pp512-gdn-rb4-sweep.json

QWEN_PREFILL_TRACE_LAYER_PHASES=1 target/release/qwen-bench pp \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -p 512 --runs 1 -o json \
  > target/profiles/v0436-0p8b-pp512-gdn-rb4-base-phase.json \
  2> target/profiles/v0436-0p8b-pp512-gdn-rb4-base-phase.log

QWEN_PREFILL_TRACE_LAYER_PHASES=1 QWEN_GDN_STEP_PACKED_RB4_TGM16=1 \
  target/release/qwen-bench pp \
  -m /Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -p 512 --runs 1 -o json \
  > target/profiles/v0436-0p8b-pp512-gdn-rb4-on-phase.json \
  2> target/profiles/v0436-0p8b-pp512-gdn-rb4-on-phase.log
```

Validation:

- `cargo check -p qwen-llm -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- 0.8B prefill-vs-single with the RB4 TGM16 env: green across chunk edge cases
- cx design review session `019f241f-6c1a-73b3-ba6a-c24f049fbe7a`

## Results

0.8B `pp512` sweep:

| Block | Base t/s | RB4 TGM16 t/s | Read |
| ---: | ---: | ---: | --- |
| `0` | `8181.82` | `7286.46` | `-10.9%` |
| `1` | `8205.72` | `7326.54` | `-10.7%` |

27B `pp512` G6 matrix spot:

| Base t/s | RB4 TGM16 t/s | Read |
| ---: | ---: | --- |
| `244.24` | `237.86` | `-2.6%` |

0.8B one-run phase trace shows the mechanism clearly enough despite other phase
noise: `gdn_step` worsens from `21.93 ms` to `32.59 ms` (`+48.6%`). The TGM
load/barrier path costs more than the redundant Q/K loads it removes.

## Decision

Kill the branch and keep no env flag or dead kernel. The active NSG4 packed GDN
kernel already captured the large row-residency win; Q/K duplicate loads appear
cache-resident or too small to justify TGM staging. Do not pursue local row/block
GDN staging variants unless a future floor/no-op ladder proves Q/K loads dominate
the active recurrence. A broader chunked delta-rule formulation remains a
separate algorithmic question, not supported by this local staging result.

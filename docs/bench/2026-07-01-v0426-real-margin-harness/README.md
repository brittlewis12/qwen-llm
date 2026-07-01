# v0.426 Real-Prompt Margin Harness

Goal: move the replay margin decision off synthetic hidden states and onto real
prompt contexts.

`decode-block-slice-real-margin` tokenizes real text prompts, runs exact prefix
decode to the requested context, clones recurrent/KV state, prepares each slot to
the requested block, and then compares exact block-slice execution against GDN
replay. Multiple `--file` entries represent same-context slots from independent
prompts; a single file currently supports `--tokens 1`.

Commands:

```sh
target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --tokens 1 --context 8 --blocks 1 --start-block 0 \
  > target/profiles/v0426-a3b-real-margin-smoke.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --tokens 1 --context 512 --blocks 4 --start-block 20,28,32,36 \
  > target/profiles/v0426-a3b-real-margin-the-current-c512-s1.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/witness_v0_clinical.md \
  --tokens 4 --context 512 --blocks 4 --start-block 20,28,32,36 \
  > target/profiles/v0426-a3b-real-margin-md4-c512-s4.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --tokens 1 --context 2048 --blocks 4 --start-block 20,28,32,36 \
  > target/profiles/v0426-a3b-real-margin-the-current-c2048-s1.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --file /Users/tito/code/llm/game/witness_v0_clinical.md \
  --tokens 4 --context 128,512 --blocks 4 --start-block 0,20,28,32,36 \
  > target/profiles/v0426-a3b-real-margin-md4-c128-512-s4.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --file /Users/tito/code/llm/game/the_current_ring0.md \
  --file /Users/tito/code/llm/game/the_current_ring0_v0.2.md \
  --tokens 3 --context 2048 --blocks 4 --start-block 0,20,28,32,36 \
  > target/profiles/v0426-a3b-real-margin-md3-c2048-s3.out

target/release/qwen-bench decode-block-slice-real-margin \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --file /Users/tito/code/llm/game/the_current.md \
  --tokens 1 --context 3072 --blocks 4 --start-block 0,20,28,32,36 \
  > target/profiles/v0426-a3b-real-margin-the-current-c3072-s1.out

uv run scripts/profile/block_slice_margin_summary.py \
  target/profiles/v0426-a3b-real-margin-the-current-c512-s1.out \
  target/profiles/v0426-a3b-real-margin-md4-c512-s4.out \
  target/profiles/v0426-a3b-real-margin-the-current-c2048-s1.out \
  target/profiles/v0426-a3b-real-margin-md4-c128-512-s4.out \
  target/profiles/v0426-a3b-real-margin-md3-c2048-s3.out \
  target/profiles/v0426-a3b-real-margin-the-current-c3072-s1.out \
  > target/profiles/v0426-a3b-real-margin-summary.tsv
```

Validation:

- `cargo fmt`
- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- A3B smoke, single-prompt c512/c2048/c3072, four-file c128/c512, and
  three-file c2048 real-margin probes
- cx adversarial review session `019f1ffe-1f3a-7120-a872-f068a90fd92d`

## Results

All focused real-prompt probes had zero route-set mismatches:

| Probe | Context | Slots | Windows | Route-set mismatch windows | Worst min replay margin | Min `x` cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `the_current.md` | `512` | `1` | `4` | `0` | `0.002045` | `0.999999984` |
| four markdown prompts | `512` | `4` | `4` | `0` | `0.000288` | `0.999999980` |
| `the_current.md` | `2048` | `1` | `4` | `0` | `0.000929` | `0.999999983` |
| four markdown prompts | `128,512` | `4` | `10` | `0` | `0.000288` | `0.999999965` |
| three markdown prompts | `2048` | `3` | `5` | `0` | `0.000047` | `0.999999980` |
| `the_current.md` | `3072` | `1` | `5` | `0` | `0.004754` | `0.999999984` |

Across the 32 non-smoke rows, route sets stayed stable, route order differed in
2 rows, worst `min_replay_margin` was `0.000047`, and min `x` cosine was
`0.999999965`. A window-level margin guard would fallback at these rates:

| Threshold | Fallback windows | Fallback rate |
| ---: | ---: | ---: |
| `1e-4` | `1/32` | `3.12%` |
| `3e-4` | `4/32` | `12.50%` |
| `1e-3` | `8/32` | `25.00%` |
| `3e-3` | `14/32` | `43.75%` |
| `5e-3` | `17/32` | `53.12%` |

## Decision

The real-prompt harness is now the right replay decision surface. These rows are
encouraging and suggest synthetic constant-hidden windows are an adversarial
stressor, but cx correctly warned that this still is not enough to promote a
scheduler path. Next evidence should expand prompt classes beyond these markdown
rollouts and convert the margin fallback table into net replay savings.

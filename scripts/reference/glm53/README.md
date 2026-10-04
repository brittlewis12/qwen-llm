# GLM-5.3-Flash Same-Artifact Oracle

`glm53_oracle` runs llama.cpp serially (one token per decode, positions from 0)
on exact token IDs, against the same GGUF the native family executes. It writes
full-vocabulary logits for every position and, optionally, named graph tensors.
It is the integration oracle in `docs/GLM53-FLASH-PLAN.md`; operation-level
fixtures for new contracts are separate.

## Build

The harness builds llama.cpp from the pinned, clean checkout (a build directory,
not a source copy) and refuses any other revision:

```sh
cmake -S scripts/reference/glm53 -B target/glm53-oracle \
  -DLLAMA_CPP_DIR=$HOME/code/llama.cpp -DCMAKE_BUILD_TYPE=Release
nice -n 10 cmake --build target/glm53-oracle --target glm53_oracle -j 4
target/glm53-oracle/bin/glm53_oracle --identity
```

The pinned revision `e845373ff` is upstream `11fe02151` plus a LoRA-only
glm5-next attention wiring fix that does not execute without adapters;
`src/models/glm5-next.cpp` semantics match `42d958167`.

Context settings: `n_batch = n_ubatch = 1`, F16 cache, flash attention off,
library defaults otherwise. At this revision the fused gated delta net, fused
lightning indexer and fused mHC ops default on (`llama-context.cpp:233`), and
with `n_rs_seq = 0` glm5-next takes the fused delta-net path. The unfused
autoregressive path (`delta-net-base.cpp:335-340`) decays the wrong axis and
must not be the oracle.

## Run

```sh
MODEL=/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/glm5-next/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
target/glm53-oracle/bin/glm53_oracle --capture-default "$MODEL" target/profiles/glm53/ref.bin ID...
```

`--capture NAME,...` selects base names (llama.cpp appends `-<layer>`);
`--capture-default` selects mHC controls/residuals, KDA and MLA internals,
FFN outputs and `result_norm`; `--capture-checkpoint` adds KDA `new_state`
(4 MiB per KDA block per step) and the new and cached pooled indexer keys,
which short-context logits cannot validate. `--steps I,...` limits captures to
those positions. A requested name that produces no record fails the run.
Captures force graph splits, which can change backend fusion; compare one
capture-enabled run against a logits-only run before relying on captures.

Every run writes `OUTPUT.manifest.json`: identity, model path, effective
context settings, token IDs and record counts per name.

Memory: the model needs about 110 GiB resident, so llama.cpp and the native
family cannot run together. Check that wired memory is low before a capture,
capture in one window, then run native under the production lease
(`QWEN_METAL_LEASE_WAIT=1`). Never stop another session's process to make room.

## Formats (little-endian)

Logits, `OUTPUT`: `GLMREF01`, u32 vocab, u32 count, then per position u32
position, u32 token, `vocab` f32 logits.

Captures, `OUTPUT.captures`: `GLMCAP01`, then records until EOF: u32 step
(position), u32 layer (`0xffffffff` when the name has no layer suffix), u32
occurrence of (step, layer, name) in evaluation order (mHC controls run once
per sub-block: 0 attention, 1 FFN), u32 name length, name bytes, u32 ggml type
(F32 0, F16 1, I32 26), 4 x i64 `ne`, i64 byte length, raw contiguous bytes. `l_out` and `hc_*_post` hold the four
mHC streams (`[4096, 4]` per token), not a single 4096-wide row.

`inspect_capture.py OUTPUT.captures` lists records with shapes and norms.

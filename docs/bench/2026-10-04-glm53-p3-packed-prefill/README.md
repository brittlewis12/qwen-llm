# GLM-5.3-Flash P3 Packed Prefill (2026-10-04)

Native packed prefill (fast lineage, 512-row chunks) and serial decode, commit
`c1933762` (clean), against llama.cpp on the same artifact (unsloth
`UD-IQ3_XXS`, `glm5-next/` view, pages warm). M4 Max 128 GB, AC power, idle
box, blocks in A-B-B-A order, 3 timed reps each.

| Block | Engine | pp512 (t/s) | pp1024 (t/s) | tg128 (t/s) |
|---|---|---|---|---|
| A1 | qwen-llm | 209.65 +/- 4.04 | 190.06 +/- 6.43 | 27.65 +/- 0.16 |
| B1 | llama.cpp | 212.75 +/- 0.85 | 199.93 +/- 0.74 | 22.93 +/- 0.45 |
| B2 | llama.cpp | 222.71 +/- 2.60 | 202.72 +/- 0.93 | 22.17 +/- 1.11 |
| A2 | qwen-llm | 207.52 +/- 1.96 | 192.92 +/- 10.33 | 27.72 +/- 0.31 |

- Prefill: 7.5x the P2 serial baseline (27.7-28.1 t/s); 93-98% of llama.cpp
  at pp512 and 94-96% at pp1024.
- Decode: +21% to +25% over llama.cpp tg128.
- Fast lineage stages activations in half like llama.cpp's batched prefill;
  on ckpt-v1 it stays tighter to serial than llama.cpp's own batched path
  (KL 2.2e-4 vs up to 1.7e-2), and exact lineage reproduces serial bitwise
  (`glm5_next_metal::tests`).

Commands:

```sh
QWEN_METAL_LEASE_WAIT=1 target/release/qwen-bench suite -m "$MODEL" --pp 512,1024 --tg 128 --runs 3
~/code/llama.cpp/build-glm5/bin/llama-bench -m "$MODEL" -p 512,1024 -n 128 -r 3 -ngl 99 -o json
```

llama.cpp binary as in `2026-10-04-glm53-p2-baseline`. Raw rows: `qwen-A*.json`,
`lcpp-B*.json`.

# GLM-5.3-Flash P2 Baseline (2026-10-04)

Native serial decode (`glm5_next_metal`, commit `89d23beb6`, clean) against
llama.cpp on the same artifact: unsloth `GLM-5.3-Flash-GGUF` revision
`621d456e`, `UD-IQ3_XXS`, `glm5-next/` view on the external WD_BLACK drive,
pages warm. M4 Max 128 GB, AC power, idle box, blocks in A-B-B-A order.

| Block | Engine | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| A1 | qwen-llm | 27.69 +/- 0.22 | 27.34 +/- 0.03 |
| B1 | llama.cpp | 222.67 +/- 1.20 | 23.51 +/- 0.07 |
| B2 | llama.cpp | 223.46 +/- 0.93 | 23.71 +/- 0.28 |
| A2 | qwen-llm | 28.12 +/- 0.01 | 27.90 +/- 0.03 |

- Decode: native +16% to +19% over llama.cpp tg128 (27.3-27.9 vs 23.5-23.7).
- Prefill: native is serial (`prefill_mode=serial_advance+last_logits`, one
  token per command buffer, head skipped); 8x slower than llama.cpp's batched
  pp512. Packed prefill is P3.

Commands (3 timed reps after warm-up, fresh session per rep):

```sh
QWEN_METAL_LEASE_WAIT=1 target/release/qwen-bench suite -m "$MODEL" --pp 512 --tg 128 --runs 3
~/code/llama.cpp/build-glm5/bin/llama-bench -m "$MODEL" -p 512 -n 128 -r 3 -ngl 99 -o json
```

llama.cpp binary: `build-glm5`, built at `e1425c0be` (upstream `42d958167` plus
a LoRA-only wiring fix; glm5-next graph identical to the oracle pin
`e845373ff`). Not the `scripts/bench/llama-cpp.lock.json` comparator, which
predates `glm5-next`. Raw rows: `qwen-A*.json`, `lcpp-B*.json`.

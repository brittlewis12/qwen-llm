# feat/glm53-p6 integration commit map (2026-10-05)

`feat/glm53-p6` was rebased onto main `bffe76f9` and fast-forwarded into main.
The packets and PERF-LOG entries dated 2026-10-05 cite pre-rebase commit
ids. Their suite JSONs and request stats record those same ids as build
stamps, so the records are left unchanged. This table maps each pre-rebase
id to the commit that main carries.

Only the first commit conflicted (`docs/PERF-LOG.md`, resolved by keeping
both sides). Every other commit applied unchanged.

| pre-rebase | on main | subject |
|---|---|---|
| `a0021be3` | `aba88c5d` | docs(perf): leverage map 2026-10-05 after the GLM-5.3-Flash bring-up |
| `1c221312` | `4d0528e5` | feat(bench): CPU sampler replay on captured logits, and a GLM acquisition test |
| `e06ba84d` | `78788d0a` | style(bench): read replay rows with as_chunks |
| `b7ea1622` | `da202a13` | perf(sampling): radix-order the full vocabulary when top-k is off |
| `d4e5a301` | `5af256f7` | docs(perf): record the radix sampler order and its GLM decode gain |
| `b38827d3` | `cf9e08e1` | docs(perf): placement screen — wiring follows GPU activity, ~1 s re-wire after idle |
| `63085f6f` | `e55b8460` | feat(serve): opt-in idle keep-alive that keeps GLM weights wired between requests |
| `e68053fe` | `91d0d3fa` | feat(serve): --idle-residency-secs, with a hardened keep-alive lifecycle |
| `684bec62` | `8da4e530` | docs(perf): record the opt-in idle residency keep-alive |
| `f400e704` | `c18960e7` | test(metal): child and detached observer for the bounded kill check of command-buffer wiring |
| `02a41b39` | `a3f7c8cb` | fix(bench): wait for the kill-check child's commands through wait_completed |
| `bee3b6ac` | `bcde17e0` | feat(glm): decode stage profiler and the attribution packet |
| `25a8fbc4` | `9bd87b0d` | docs(perf): GLM decode attribution and the command-buffer wiring kill check |
| `422a25a1` | `b6e4acbb` | docs(perf): selected-attention split is a numerical change, not bitwise |
| `479fa9c1` | `93514f38` | feat(serve): keep GLM weights wired for 60 s after activity by default |
| `ea49595f` | `e2860321` | perf(glm): split selected attention across 128-row partials |
| `02d3ccdf` | `694ac7ec` | fix(glm): finite empty state for split attention partials, directed edge tests |
| `ca364455` | `481333ac` | test(metal): mHC pre dispatch-cost screen at GLM width |
| `a99792ba` | `2b24bca8` | style(metal): name the mHC screen's encoder type (clippy type_complexity) |
| `a23a9202` | `19b30620` | docs(perf): GLM split selected attention A/B and decode attribution v2 |
| `b1ff3b8e` | `2de166a5` | perf(glm): fused mHC pre (split-K Q8_0 mix partials + one finish threadgroup per row) |
| `9309670d` | `83318001` | docs(perf): GLM fused mHC pre A/B and decode attribution v3 |
| `344da7e7` | `e019b22b` | feat(serve): idle residency for every no-copy family, window closed until first activity |
| `b7655f61` | `ec970467` | perf(glm): short-K Q8_0 mat-vec for the KDA low-rank expansions |
| `dc2d5ada` | `3e596b10` | test(serve): idle re-wire pause screen for any serve family |
| `5b3673b4` | `dc61bbef` | docs(perf): short-K KDA A/B, DS4 idle residency screen, roadmap rows #1 and #2 |
| `a9fafb50` | `7e26c003` | test(metal): routed-expert block dispatch-cost screen at GLM shapes |
| `d786a9f5` | `99c343ee` | perf(glm): one dispatch for KDA q/k/v and the fused shared-expert SwiGLU (bitwise) |
| `4c88ab1c` | `744026c7` | docs(perf): GLM bitwise dispatch fusion A/B, routed-expert screen, roadmap row #2 |

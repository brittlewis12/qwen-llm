# Restored suffix crossover: current layout fails memory screen

## Outcome

No production threshold or allocation change. Test-only candidate `5e58fee4`
is removed at `daa4ff5b`; preserve the result, not an unqualified prototype.
The existing 48-token serve serial threshold and scratch omission remain.

One isolated Qwen3.6-27B Q4 release pilot shows a large phase ceiling but
rejects the current packed workspace at the frozen 128 MiB limit:

| Suffix path | Scratch allocation | Construction | Prefill |
| --- | ---: | ---: | ---: |
| Existing no-tail serial | 0 bytes | 0.000 ms | 1262.563 ms |
| Existing packed, query 32 | 323,256,320 bytes | 0.204 ms | 252.137 ms |

Construction plus suffix execution is approximately 5.003x faster in this
single isolated pair. **This is not an endpoint or promotion-grade speedup.**
The memory failure stops the test before numerical state and 64-token greedy
continuation gates; neither correctness authority is established. Do not infer
them from packed prefill returning successfully.

## Workload and gates

- Base `b62a79cb`, release test build at `5e58fee4`, normal exclusive Metal
  runtime, no wiring/residency changes. Qwen3.6 avoids the installed Qwen3.8
  artifact's template-admission blocker.
- Reuse the natural 8,840-token raw fixture from the CLI lifetime packet. Build
  one common packed prefix of 8,808 consumed tokens and cache its completed
  boundary with the next prompt token pending. Each arm restores that same
  checkpoint: 8,809 matches, 8,808 consumed tokens, 32 required forwards.
- Candidate query/block width is 32, full matrix key extent 8,840. The test
  verifies those plan dimensions. It does not allocate a 1,024-query scratch
  just because the historical prompt is long.
- Model-free boundaries cover a hypothetical restored dense/no-drafter,
  non-exact suffix 7-32; fresh, MoE, drafter, <=6, >32, and invalid boundaries
  remain excluded. That model-free test passes; no policy is installed.
- Frozen screen: scratch <=128 MiB, >=30% construction+prefill phase saving,
  >=0.999 final-logit/per-layer GDN/new-KV cosine, unchanged restored KV bytes,
  and equal next 64 greedy tokens including EOS. Packed-versus-serial is a
  numerical/greedy contract, not bitwise or distributional equivalence.
- First observer launch cannot find its relative prompt path because cargo
  runs tests from the package directory. It fails before Metal initialization.
  The sole GPU execution uses the absolute path to the same bytes. Both logs
  and the unchanged gates are retained; there is no prompt/threshold search.

## Attribution that changes the next choice

`MetalDFlashLayerMajorScratch::attn_matrix_vt_pack` retains an F16 transposed
V cache for **every** attention layer, independent of small query width:

`16 layers * 4 KV heads * 256 dimensions * 8840 positions * 2 bytes`

That is 289,669,120 logical bytes, 89.61% of the observed 323,256,320-byte
workspace. Query-width reduction alone cannot meet the frozen cap. The
existing packed API starts per-layer VT-validity counters at zero for each
invocation and rebuilds restored V prefixes before matrix attention.

For a single packed block, each layer's transposed view has no later-chunk
consumer in that invocation. Arithmetic substitution of one layer slot gives
`323256320 - 289669120 + 18104320 = 51691520` bytes. This is an optimistic
allocation estimate, not a constructed/priced plan or measured result.
Multi-chunk prefill deliberately retains per-layer VT across chunks; do not
transfer single-use lifetime reasoning there.

Independent Luna review (cx session `01a07893-0d8c-7c12-a567-07f5c8c1331e`)
supports the bounded initial screen and the memory-only rejection. Its initial
requirement for a CPU wait between layers was challenged and corrected: serial
tracked compute encoders can order shared-scratch reuse within a command buffer.
The unresolved proof is that **all** VT producers/consumers remain in that
ordering, with no concurrent or cross-buffer escape. Any single-slot follow-up
must establish that proof, full-prefix rebuild, slot offsets, one-chunk
fail-closed validation, and actual plan pricing before GPU work. Full-VT versus
single-slot storage must be bitwise; packed versus serial still needs its
separate numerical and greedy checks. No slot-sharing implementation is retained.

Raw protocol and both acquisition logs remain at
`target/profiles/restored-tail-crossover/` in the dedicated worktree. The
test instrument remains recoverable from its committed experiment identity.

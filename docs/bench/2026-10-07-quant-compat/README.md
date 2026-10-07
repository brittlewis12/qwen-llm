# GGML quant compatibility: format and CPU packet

Base: qwen-llm `05fca4c4`. Work branch: `feat/ggml-q2-compat`.

## Scope

Q2_0 is mainline GGML type 42, not a private GSQ wire format. Verified
against ggml-org/llama.cpp `4d756bc72bf00a4aacf410ae15a2d315f3db400d`
(`ggml/include/ggml.h`, `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`).
It packs 64 weights into 18 bytes: a little-endian FP16 scale followed by
16 bytes of consecutive low-to-high 2-bit codes. Value j is
`(code - 1) * scale`, with codes -1, 0, 1, 2 after centering.

This packet adds storage recognition and CPU codec coverage, not Metal
kernels or a claim that GSQ-RCO Flash-Next can generate:

| Type | Wire ID | Elements/block | Bytes/block |
| --- | ---: | ---: | ---: |
| TQ1_0 | 34 | 256 | 54 |
| TQ2_0 | 35 | 256 | 66 |
| NVFP4 | 40 | 64 | 36 |
| Q1_0 | 41 | 128 | 18 |
| Q2_0 | 42 | 64 | 18 |

The ternary raw layouts already existed; their enum names were missing.
The pinned llama-cpp-sys dependency already implements all five codecs.
The GGUF header's obsolete `kind >= 40` rejection is replaced by explicit
storage-layout validation. Removed and unknown types still fail closed.
FFI codec lookup now checks its enum bound and storage geometry before use.
Individual C calls decode at most 65,536 elements in whole blocks, avoiding
the signed-int indexing inside some GGML codecs without limiting tensor size.
Unaligned byte slices use bounded, explicitly aligned staging before decoding.

The existing parser commit `3a92c518bce43959686bef1093b31b4067502d2b`
was published to `brittlewis12/gguf`'s `bump-deps-add-quants` branch with user
authorization and is now the qwen-llm dependency pin. Additional parser
regression tests live separately in `gguf-quant-compat`, based on that commit.

## Evidence

- CPU codec tests: directed Q1 bit order/signed scale; NVFP4 four-scale and
  nibble order; independent Q2 packing with signed and zero scales, unaligned
  input, multiple rows and K=640; malformed lengths and partial rows refused.
- Loader tests: all five formats through the dependency parser; Q2 split
  shards and nonzero offsets; truncated payloads, partial rows, removed and
  unknown tags refused.
- `gsq-iq3_s-inventory.json`: header inventory plus first/middle/last row
  from every Q2 bank. Each row is 640 elements / 180 bytes. All 27 samples
  match an independent decoder bit-for-bit and contain finite values.
  Sample hashes bind these tiny payloads, not complete checkpoint content.
- No Metal initialization, prefetch, whole-tensor decode, or model execution.
- Adversarial design/review session: `cx` `01a11507-83c0-74e1-8bc1-0e1c83c1235e`.
- The saved report includes compiled source/lockfile hashes and per-shard
  device/inode/size/time stamps, revalidated before and after sampling.

Validation: workspace check passes with the published parser pin; selected
CPU tests pass (77 passed, one unrelated ignored no-copy probe). The published
gguf commit's 60 tests pass; the separate parser regression worktree passes
63. Strict workspace Clippy is blocked by existing `ptr_arg` warnings in
`crates/qwen-llm/build.rs` under this toolchain; that unrelated file is unchanged.

Reproduce the CPU-only inventory:

```sh
cargo run -p qwen-llm --example quant_inventory -- FIRST_SHARD.gguf
```

## Remaining execution work

The archive has 1,224 tensors across two shards. Nine Q2 banks are only the
format-level gap. Flash-Next currently admits an exact UD-Q3_K_XL role
allocation (`qwen4exp_residency.rs`) and narrower kernel dtypes
(`qwen4exp_moe.rs`). GSQ also changes:

- routed gate/up banks: IQ2_S and IQ3_S in addition to existing formats;
- routers and hyperconnection weights: BF16;
- shared gate/up: mixed Q4_K/Q5_K/Q6_K/IQ4_XS;
- shared down: 47 IQ4_NL banks, rather than all Q8_0;
- ordinary projections, embedding and output head storage.

Next: role-owned CPU execution coverage, Q2 native projections and routed
decode/prefill at K=640 (ten blocks / 180-byte rows), reuse existing qualified
kernels for the other roles, then independent model-reference checks under
the normal Metal lease. Do not relax dtype admission merely because the
file parses. Coherent generated text alone is not a codec/kernel oracle.

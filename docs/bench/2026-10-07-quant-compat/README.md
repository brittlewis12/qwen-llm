# GGML quant compatibility and GSQ-RCO execution

Base: qwen-llm `05fca4c4`. Work branch: `feat/ggml-q2-compat`.

## Scope

Q2_0 is mainline GGML type 42, not a private GSQ wire format. Verified
against ggml-org/llama.cpp `4d756bc72bf00a4aacf410ae15a2d315f3db400d`
(`ggml/include/ggml.h`, `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`).
It packs 64 weights into 18 bytes: a little-endian FP16 scale followed by
16 bytes of consecutive low-to-high 2-bit codes. Value j is
`(code - 1) * scale`, with codes -1, 0, 1, 2 after centering.

The first packet (`7c75b21f`) adds storage recognition and CPU codec coverage:

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
was published, as authorized, to `brittlewis12/gguf`'s `bump-deps-add-quants`
branch and is now the qwen-llm dependency pin. Additional parser
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
- That inventory performs no Metal initialization, prefetch, whole-tensor
  decode, or model execution.
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

## Native execution packet

The archive has 1,224 tensors across two shards. Nine Q2 banks are only the
format-level gap. Flash-Next previously admitted an exact UD-Q3_K_XL role
allocation. GSQ also changes:

- routed gate/up banks: IQ2_S and IQ3_S in addition to existing formats;
- routers and hyperconnection weights: BF16;
- shared gate/up: mixed Q4_K/Q5_K/Q6_K/IQ4_XS;
- shared down: 47 IQ4_NL banks, rather than all Q8_0;
- ordinary projections, embedding and output head storage.

Production now admits supported native dtypes per role, preserving the exact
released geometry and the strict UD qualification API. Q2 has compressed
SIMD matvec/all-slot expert down and tiled grouped kernels, including K=640
(ten blocks / 180-byte rows). IQ2_S has unclamped singleton SwiGLU; packed
IQ2_S/IQ3_S reuse generic grouped matrix kernels. Mixed shared projections
reuse existing kernels and already-priced scratch. IQ4_XS embedding lookup
dequantizes only requested rows on GPU.

Only F32-only auxiliary roles normalize BF16/F16 into owned F32 storage:
routers, HC injection, GDN alpha/beta and PLE convolution. Initialization
writes directly into Metal storage. Original retained windows/fallbacks and
new owned buffers are both charged. The 28.8 GB PLE table stays CPU
row-addressed, not a wired Metal tensor or full-table dequantization.

Adversarial integration review: `cx` `01a11528-973c-7e40-ba8c-d4bece18c93b`.
Its shared-projection admission mismatch was fixed by using the adapter's
role-specific supported sets, with negative tests. First real execution
exposed a Metal API validation boundary at 512 experts in generic grouped
kernels. Their builtin thread index is now wide, matching the existing
release-specialized fix; local TG128 arithmetic is unchanged. Independent
512-expert regression coverage exercises the boundary and expert 511.

Validation on Apple M4 Max, under the normal Metal lease with
`MTL_DEBUG_LAYER=1`:

- Q2/embedding: 9 tests pass, covering independent f64 references, offsets,
  tails, K=64/128/640, invalid/repeated routes, grouped down/gate/up, and
  IQ4_XS row lookup against the pinned CPU codec.
- IQ2_S: 4 tests pass, including independent f64/codec references,
  unclamped semantics, K=256/1280/2560 and the 512-expert grouped boundary.
- CPU role/projection/scratch/normalization tests pass; memory tests price
  simultaneous retained and converted storage. Existing UD role tests pass.

## Downloaded model smoke evidence

Artifact: the two-shard IQ3_S inventory recorded here. Release build from
this worktree; JSONL records report parent `7c75b21f` with `dirty=true`, i.e.
the implementation packet accompanying these records, not vanilla 7c75.
All three requests ran with temperature zero, `--no-thinking`, normal lease
ownership, and Metal API validation. No full-shard prefetch was enabled.

| Record | Prompt / budget / capacity | Outcome |
| --- | --- | --- |
| `gsq-packed-request.jsonl` | 21 / 16 / 256 | packed prefill; `Hello from GSQ.`; EOS |
| `gsq-scalar-request.jsonl` | 21 / 16 / 256 | scalar profiled prefill; same token fingerprint; EOS |
| `gsq-code-request.jsonl` | 36 / 192 / 1024 | 115 output tokens; Fibonacci function with n=0 handled; EOS |

First two use `--user 'Reply with exactly: Hello from GSQ.'`; scalar uses
`QWEN4EXP_LAYER_PROFILE=1`. Coding uses `--user 'Write a short Python
function that returns the first n Fibonacci numbers. Handle n=0 and briefly
state its time complexity.'` (one line).

Observed weight allocation: 55,110,090,752 bytes. The 21-token packed
request's aggregate admission bound was 55,836,770,304 bytes. Initial cold
prefill took 38.0 s wall versus 278 ms reported GPU time; the subsequent
packed smoke took 888 ms wall / 275 ms GPU. The longer coding request
decoded at 27.08 output tokens/s over 4.25 s. These are smoke timings, not a
controlled benchmark or a promise about cold external-drive latency.

These smoke requests used capacities of 256 or 1024, below the selected-range
threshold of 2051 tokens (`packed_selected_capable` compares the prompt extent
or session capacity with the selected output width). They exercised ordinary
packed execution, not selected packed attention. Native GSQ admission does not
disable the selected path; its execution and numerical qualification above
the frontier remain outstanding (PERF-ROADMAP #18; corrected 2026-10-07).
Coherent generation and matching
short continuations do not establish full-model numerical parity with an
independent engine. That broader quality gate, other GSQ variants and a
controlled performance comparison remain unclaimed.

## Main integration

Integrated with main `01acecb4`. The merge preserves `d688e049`'s
Flash-Next Q8 few-row dispatch exclusion; projection preflight now excludes
those unreachable kernels and tests that policy explicitly. Adversarial
review of both parents found no remaining integration blockers.

Combined-tree validation: workspace check, 19 focused CPU contract tests,
and 15 leased kernel-packet tests pass. A fresh release-build GSQ request
with Metal API validation again returns `Hello from GSQ.` through EOS.
Its 7.48 s first-use prefill versus 279 ms reported GPU time remains storage
warmth evidence, not a kernel-performance comparison.

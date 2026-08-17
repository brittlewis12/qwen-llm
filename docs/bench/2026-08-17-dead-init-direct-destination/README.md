# Dead Initialization And Direct Destination Falsifiers

Date: 2026-08-17

Status:

- **GO** for direct Q4_K-to-F32 production into final Shared Metal storage.
- **KEEP-HYGIENE** for fallible uninitialized host dequantization, checked
  quantized row geometry, initialized-prefix CPU KV, and fallible tokenizer FFI
  outputs. Host dequantization alone has no speed authority.
- **KILL** for a broad anonymous no-copy Metal output allocator from this
  premise. Its first-write saving is small, noisy, and below the gate.

This is an exploratory dirty-worktree packet, not release or default-promotion
authority. It records complete final timing vectors in `results.json`; the
earlier in-process reconnaissance is excluded because allocator reuse was
bimodal and order-confounded.

## Question

When a producer overwrites every destination element, can we delete either:

1. the destination's prior zero value; or
2. the entire temporary destination and its later copy?

The first question mirrors Bun PR 39417. The second is the stronger formulation
available on unified memory: llama.cpp dequantizes directly into the final
Shared `MTLBuffer`.

## Protocol

- Host: Apple M4 Max, 128 GiB, 16 KiB pages.
- Source shape: Q4_K `[4096, 16384]`, 67,108,864 output elements.
- Source bytes: 37,748,736; F32 output bytes: 268,435,456.
- Payload: deterministic valid Q4_K blocks, identical checksum in every arm.
- Timing: six fresh-process AB/BA pairs after one neutral CPU warm child for the
  host-only floor; six fresh-process AB/BA pairs for direct Metal destination.
- Metal initialization: six AB/BA pairs at 512 MiB and six at 2 GiB. Each arm
  performs one first full overwrite and one warm overwrite with the same kernel.
- Untimed correctness checks hash every byte of both 256 MiB dequant destinations
  and every byte after both fills of a 64 MiB Metal output buffer. Timed large
  Metal arms retain sampled assertions to avoid charging verification reads.
- Allocation, first producer/copy, and GPU completion are inside the reported
  primitive boundary. Metal context and pipeline construction are outside it.
- No model is loaded for the primitive floors. One separate 0.8B smoke checks
  the production converted-F32 loader.

All model/Metal commands were serialized by the process lease. No residency
set, `requestResidency`, pre-wire, `mlock`, cache-bypass read, uncached read, or
residency-coupled A10B pread path ran. The anonymous arm uses ordinary
demand-zero `mmap` plus `newBufferWithBytesNoCopy`; it performs no prefault or
wiring. PID 8770 was absent before and after acquisition, and no process was
killed.

## Result 1: Host Zero Deletion

| Arm | Median total | Paired saving | Wins | AB median | BA median |
|---|---:|---:|---:|---:|---:|
| Zeroed host output | 24.529 ms | - | - | - | - |
| Uninitialized host output | 24.618 ms | -0.044 ms | 2/6 | +0.006 ms | -0.067 ms |

`vec![0.0; n]` is effectively demand-zero in the fresh process. Removing the
logical initialization moves first-touch work into `to_float`; it does not
remove a measurable physical pass in this cell. This kills a speedup claim for
host zero deletion alone.

The implementation remains useful hygiene: allocation is fallible, vector
length is committed only after the FFI producer returns, element count must fit
`i64`, source bytes must match checked GGML storage geometry, and each quantized
row must be block aligned. The measured difference is below 0.4% and has no
consistent candidate advantage or regression.

## Result 2: Direct Final Metal Destination

| Arm | Median total | Median paired saving | Wins | AB median | BA median |
|---|---:|---:|---:|---:|---:|
| Host dequant + `from_bytes` | 37.855 ms | - | - | - | - |
| Direct final Metal dequant | 25.714 ms | 12.261 ms | 6/6 | 12.273 ms | 12.250 ms |

Median speedup is **1.472x**. Staged host dequant is 25.495 ms and the subsequent
Metal population copy is 11.413 ms. Direct Metal allocation is 0.043 ms and
direct dequantization is 25.665 ms. The arithmetic identifies the deleted copy;
there is no claimed dequant-kernel improvement.

`/usr/bin/time -l` confirms the representation deletion:

| Metric | Staged | Direct | Deleted |
|---|---:|---:|---:|
| Maximum RSS | 587,284,480 | 318,832,640 | 268,451,840 bytes |
| Peak footprint | 579,339,080 | 310,755,976 | 268,583,104 bytes |

All 12 final timed arms have checksum `11087277778870641445`. Separate staged
and direct arms hash every output byte to the identical digest
`fa6673d6...58d3`. The direct timing arm invokes the same llama.cpp producer
against the final Metal pointer; production validation is covered separately by
the destination-length unit test and model smoke. A forced converted-embedding
smoke on the local 0.8B Q4_K_M model succeeds with ledger
`208,588,800 -> 1,017,118,720` converted bytes, one prompt token, one generated
token, and `load_ms=227.6`. That smoke is correctness/realization evidence, not
a staged-versus-direct model-load timing comparison.

## Result 3: Physical Metal Zero-Clear Avoidance

| Size | Device-created median | Wrapped-anon median | Paired saving | Wins |
|---|---:|---:|---:|---:|
| 512 MiB | 13.650 ms | 13.372 ms | 0.238 ms | 4/6 |
| 2 GiB | 38.827 ms | 38.351 ms | 1.367 ms | 4/6 |

Both order strata are directionally positive, but the result is noisy, misses
the win-count gate, and projects below 5 ms even at the surviving 4 GiB DSV4
matrix geometry. GPU time does not show a stable second-write advantage. Do not
add a broad anonymous no-copy scratch allocator, change session topology, or
reopen the killed giant-arena/overlay work from this result.

A separate 64 MiB conformance hashes every byte after both values. Device-created
and wrapped-anon buffers match the same first digest `5ace0bc6...1ae1` and second
digest `96108b26...626`; the KILL is performance-only, not a correctness failure.

## Retained Changes

- `codec::dequant_to_f32` reserves fallibly and commits only initialized output.
- `MetalWeightLoader::load_f32` dequantizes directly into its final Shared
  `MetalTensor`, deleting one full host temporary and copy.
- CPU `KvCache` retains exact capacity but materializes only its strictly
  appended prefix.
- llama.cpp tokenizer output buffers reserve fallibly and expose only the
  producer-reported prefix.
- Model-free examples retain the two falsifiers and exact checksums.

Full-logit readback zero deletion remains closed by v0.659's 0.223% generation
ceiling. Runtime MTP repacks and typed snapshot decode remain separate cold-path
candidates; neither inherits authority from the direct dequant result.

## Validation

- `cargo fmt --all -- --check`
- `cargo check -p qwen-llm --all-targets`
- `cargo test -p qwen-llm codec::tests --lib -- --test-threads=1`: 20 passed
- tokenizer suite: 25 passed, 4 fixture-dependent ignored
- exact CPU forward primitive, 0.8B one-token, and 0.8B MTP tests: passed
- `cargo test -p qwen-cli --bin qwen -- --test-threads=1`: 180 passed,
  2 fixture-dependent ignored
- release `qwen`, `codec_init_floor`, and `metal_output_init_floor`: built
- adversarial read-only review: no blocking issue; full-digest and raw-vector
  evidence repairs accepted on follow-up

Strict all-target/all-feature Clippy first stopped on unrelated existing WIP in
DSV4 diagnostics, the K160 floor, and `ffn_inner_census_capture`. Re-running with
only those four pre-existing lint classes allowed passed with `-D warnings` for
the remaining code. They were not edited here. One early substring-filtered test
command entered unrelated active Metal tests and hit the 120-second supervisor
timeout; its preceding CPU tests passed, all child processes exited, and none of
that attempt contributes timing evidence.

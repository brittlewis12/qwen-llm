# M5 Ultra First Light

## Status

Native qwen-llm runs successfully on an Apple M5 Ultra with 256 GiB
unified memory and 80 GPU cores. Qwen3.5-0.8B-Q8_0 and Qwen3.6-35B-A3B smoke and
warmed suite results are complete. GLM Maya is transferring, not yet measured.
This is native engine bring-up, not the separate upstream MTP experiment.

## Installation and identity

- Execution OS: macOS 27.0.1 (26A434).
- Engine commit: `d7b2a05b5ad1b9706486c69718a1e11dbbc58937`.
- Tree: `11883f782fc7da4060bcb0b25cfae11f3a63346f`.
- Build: `CARGO_BUILD_JOBS=4 cargo build --release --locked --offline -p qwen-cli
  --bin qwen --bin qwen-bench`, on the M4 Max source host.
- The worker has Rust, uv, mise, CMake, and Command Line Tools, but no Xcode Metal
  compiler. The copied native binaries embed their metallibs and link only system
  libraries; a remote source build remains dependent on installing Metal tools.
- Source was cloned from a local Git bundle into `~/code/qwen-llm`, then its
  origin was set to the existing GitHub repository. No push occurred.
- Executables: `~/qwen-runs/2026-10-16-first-light/bin/`, with non-overwriting
  symlinks at `~/.local/bin/qwen` and `~/.local/bin/qwen-bench`.
- `binaries.sha256` pins both executables. Benchmark JSON reports clean matching
  compiled/runtime source identities, with no identity overrides.
- `iogpu.wired_limit_mb` stays at `0` (automatic). The loader reports a
  239143780352-byte recommended working set. No sysctl tuning was performed.

Dedicated pilotty sessions keep these runs separate from existing user sessions.
The queue completed Qwen35B before resuming the owned GLM shard prefix using
`rsync --append`, with full SHA-256 verification before inference. No unrelated
files are append targets. Hostnames and connection identifiers are omitted from
this packet; the captured hostname line in the memory log is redacted.

## Completed small-model screen

Artifact: `Qwen3.5-0.8B-Q8_0.gguf`, SHA-256
`0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`.

The short greedy, non-thinking prompt asked for 17 times 23. The M5 Ultra answered
`17 times 23 equals 391.` and stopped at EOS. This checks basic execution and
obvious answer correctness, not numerical agreement with a reference backend.

```sh
qwen run -m ~/models/Qwen3.5-0.8B-Q8_0.gguf \
  --no-thinking --temp 0 -n 128 \
  --user 'What is 17 times 23? Answer in one sentence.'

env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=0 qwen-bench suite \
  -m ~/models/Qwen3.5-0.8B-Q8_0.gguf \
  --pp 128,512,4096 --tg 128,256 -d 0 --runs 3
```

The same freshly built binary and exact model ran on both machines. Transfers
were paused during the suite measurements. Each row has its normal warmup and
three timed repetitions, with fresh sessions; loading is excluded. Decode rows
are synthetic compute throughput, not sampled user-visible streaming rates.
The M4 is the user's active desktop, not a controlled laboratory host.

| Shape, depth 0 | M4 Max tok/s | M5 Ultra tok/s | Ratio |
| --- | ---: | ---: | ---: |
| pp128 | 6124.1 | 8854.0 | 1.446x |
| pp512 | 8562.0 | 14938.2 | 1.745x |
| pp4096 | 8700.5 | 16174.0 | 1.859x |
| tg128 | 263.0 | 439.4 | 1.671x |
| tg256 | 261.1 | 435.2 | 1.667x |

Raw rows, sample dispersion, and build identities are in `m4-qwen08-suite.json`
and `m5-qwen08-suite.json`. These ratios apply only to this small-model screen;
they do not establish GLM, MoE, long-context, or general M5 speedups. M4-specific
qualification checks decline on M5 as intended; no device-name override was used.

## Completed Qwen35B screen

Artifact: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22134528992 bytes. Full SHA-256
verification passed against `qwen35b.sha256`. The pinned binary and runtime
source identities match, and every run used the default device qualification.

The smoke produced a correct `sum_even(xs: &[i64]) -> i64` using an iterator
filter and sum, with `assert_eq!(sum_even(&[1, 2, 3, 4]), 6)` inside a test.
Manual review confirms this requested behavior; it is not a reference-logit
comparison or broad quality qualification. Completion reached EOS after 79
tokens. CLI telemetry reports 133.47 decode tokens/s, 339.1 ms TTFT, and a
separate 2708.0 ms model load; do not substitute synthetic rates for this result.

| Shape | M4 Max tok/s | M5 Ultra tok/s | Ratio |
| --- | ---: | ---: | ---: |
| pp128, depth 0 | 950.0 | 1613.3 | 1.698x |
| pp512, depth 0 | 1538.1 | 2914.1 | 1.895x |
| pp2048, depth 0 | 1829.4 | 3695.2 | 2.020x |
| tg128, depth 0 | 112.6 | 169.9 | 1.509x |
| tg256, depth 0 | 112.0 | 168.6 | 1.506x |
| tg128, depth 8192 | 99.9 | 139.0 | 1.391x |

Three repetitions plus warmup per row; the same binary, artifact, settings,
and clean source identity on both hosts. The retained M4 comparison is a repeat
after source-checksum work finished. Its desktop and outgoing network transfer
remained active, while M5 Ultra inference ran between transfers. These are
provisional operational comparisons, not fully controlled hardware attribution.
An earlier M4 pass overlapping checksum work is excluded from this packet.

M5 Ultra swap and compressor usage both remained zero before/after the run.
The runner exited zero. Raw smoke, hashes, before/after memory, and suite files
are under `m5-qwen35b/`; M4 rows are `m4-qwen35b-suite.json` and
`m4-qwen35b-depth8192.json` with their stderr logs.

## Larger-model protocol

Transfers use resumable rsync over independent SSH connections. At roughly
14 MB/s initially, transferring weights dominates bring-up time. The user added
Ethernet to a wirelessly connected eero during Qwen's copy; instantaneous rate
rose to about 17 MB/s. Qwen35B ran before resuming all three GLM Maya shards.
Transfers and remote inference are
serialized; no benchmark runs concurrently with this queue's file transfer.

Copy `run-model.sh`, `deadline.py`, and the three checksum manifests to
`~/qwen-runs/2026-10-16-first-light/`. The runner requires the pinned checkout
and binaries above and creates a fresh output directory, refusing overwrite.
For example, after all GLM shards finish transferring:

```sh
sh ~/qwen-runs/2026-10-16-first-light/run-model.sh glm \
  ~/models/glm-5.3-flash-maya-m/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf
```

The runner verifies full binary and model SHA-256 checksums (all GLM shards),
records memory/swap before and after even on failure, and gives each owned
inference child a 900-second deadline. Native commands acquire the Metal lease;
they are not nested inside the external-engine lease wrapper.

The smoke asks for a Rust even-element sum and an assertion. Qwen uses
non-thinking/256 tokens; GLM uses low reasoning/1024 tokens. A non-EOS completion
stops the queue for inspection. EOS alone is not a semantic correctness check;
the generated code still needs review before a successful-smoke claim.

Each larger-model suite measures pp128/512/2048 and tg128/256 at depth zero,
then tg128 at depth 8192, with three repetitions per row plus warmup. The latter
performs four untimed 8192-token fills; reported decode throughput excludes that
prefill cost. These are separate synthetic pp/tg rows, not end-to-end request
latency. Full checksum reads warm the filesystem cache before model loading.

Validation:

```sh
shellcheck docs/bench/2026-10-16-m5-ultra-first-light/run-model.sh
uv run --offline --no-project python -B \
  docs/bench/2026-10-16-m5-ultra-first-light/test_deadline.py
```

Five deadline tests pass on both machines. Structural contracts passed for the
initial packet using a clean temporary worktree to preserve unrelated untracked
documents on main. Independent review confirmed the CLI protocol
and narrow small-model ratios; safeguards above address timeout, failure-memory,
and transfer-integrity concerns. No kernel optimization is promoted by this packet.

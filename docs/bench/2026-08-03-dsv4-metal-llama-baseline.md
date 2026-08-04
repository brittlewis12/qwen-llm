# DeepSeek V4 Metal llama.cpp Baseline

Status: promoted comparison packet, 2026-08-03.

This packet brackets qwen-llm's first optimized DeepSeek V4 decode against a
clean current llama.cpp Metal build. It answers one narrow question: after
cooperative dense attention removed the context-length cliff, how much
short-context throughput remains between qwen-llm and the maintained upstream
implementation on the same 95.93 GiB GGUF?

## Verdict

- qwen-llm sustains 16.81 and 16.70 tokens/s near contexts 128 and 512.
- current llama.cpp sustains 27.67 and 27.58 tokens/s at cached depths 128 and
  512. qwen-llm takes 1.646x/1.651x as long per token and delivers
  60.7%/60.6% of upstream throughput.
- Both curves are nearly flat. Attention history growth is no longer the
  short-context limiter; the remaining roughly 23.5 ms/token is a
  context-independent execution and synchronization floor.
- The exact packed-prefix packet records one cold-first qwen prefill and warmed
  llama_core medians. It exposes a large prompt-path diagnostic but does not
  promote a cross-engine ratio until warm treatment is matched.
- The S6 throughput gate is not met. Per-layer host routing completion is the
  first short-context candidate to attribute and falsify; full-history index
  scoring remains the separate far-context target.

## Pinned Systems

qwen-llm:

- commit `1e63292c20c0b2a5883001eef7174ececf5fdbae`
- release profile, native Metal session
- retained packed prompt chunks of at most 128 tokens
- F16 raw and compressed causal state

Current upstream comparator:

- llama.cpp build `b10254`
- commit `0ef6e55edb306fcbcf73e6f1f41923cccb9cf7f8`
- clean detached worktree `/Users/tito/code/llama.cpp-dsv4-bench`
- `GGML_METAL=ON`, release, native CPU support, embedded Metal library
- all 43 layers offloaded, F16 K/V, mmap load, batch/ubatch 128

Exact-request comparator:

- llm `e07acac20fcd2ee0faca90aa91078ff142724d63`
- llama_core `b1cd3a914175adedcc976388c7c638d9a2f9a189`
- llama-cpp-rs `1efd8b59c236f139fc22206eb9a5f1f702e1d706`
- effective llama.cpp build `b10235`, commit
  `7cfc994352e2abc2117b0c29d5440b5d7387156e`
- release binary built after those three revisions, Metal backend

Hardware and asset:

- Apple M4 Max, 128 GiB unified memory
- model size 102,994,542,940 bytes, 284,334,567,511 parameters
- shard SHA-256:
  - `9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a`
  - `afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd`
  - `64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca`
  - `5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2`

The pre-existing llama.cpp build was rejected before measurement because its
executable and `libllama-common` had incompatible ABIs. The comparator above was
built from a clean worktree rather than repairing or reusing that mixed tree.

## Sequential Decode

qwen-llm command:

```bash
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live \
  profile_native_deepseek_v4_singleton_decode_at_128_and_512 -- \
  --ignored --exact --nocapture --test-threads=1
```

llama.cpp command:

```bash
build-bench/bin/llama-bench --offline -m MODEL \
  -p 0 -n 5 -d 128,512 -r 5 -b 128 -ub 128 -t 12 -ngl 999 \
  -ctk f16 -ctv f16 -fa auto -lm mmap --progress -o jsonl
```

`llama-bench --n-depth` prepares and restores the prefix outside the timed
region. Each sample times five successive generation tokens; the table divides
the reported sample duration by five.

| Engine | Depth | Per-token samples (ms) | Median (ms) | tokens/s |
|---|---:|---|---:|---:|
| qwen-llm | about 128 | 59.974, 59.498, 60.166, 59.022, 58.860 | 59.498 | 16.807 |
| llama.cpp b10254 | 128 | 37.896, 36.137, 36.130, 36.173, 36.036 | 36.137 | 27.673 |
| qwen-llm | about 512 | 60.107, 59.865, 60.703, 59.439, 59.369 | 59.865 | 16.704 |
| llama.cpp b10254 | 512 | 36.763, 36.211, 36.255, 36.285, 36.189 | 36.255 | 27.582 |

The qwen-llm test warms one singleton token at each depth and then measures five
successive tokens. llama-bench uses deterministic synthetic token IDs. This is
the product-warm throughput comparison; the exact-token packet below isolates
request identity separately.

## Exact Token Packet

The exact request uses the native-tokenizer-verified pattern
`A\n\t@ -> [35, 201, 200, 34]`. Position 129 has the 129-token prefix
`pattern * 32 + [35]`; position 513 has `pattern * 128 + [35]`. Both inject
token ID 201 and capture the resulting full-vocabulary vector.

llm commands:

```bash
uv run --no-project python -c 'import sys; sys.stdout.write("A\n\t@" * 32 + "A")' | \
  llm --model MODEL --raw --context 136 --snapshot OUT \
  --snapshot-id exact_position129_token201 --inject-token-id 201 \
  --seed 42 --temp 0 --repeat 5 --snapshot-top-k 10

uv run --no-project python -c 'import sys; sys.stdout.write("A\n\t@" * 128 + "A")' | \
  llm --model MODEL --raw --context 520 --snapshot OUT \
  --snapshot-id exact_position513_token201 --inject-token-id 201 \
  --seed 42 --temp 0 --repeat 5 --snapshot-top-k 10
```

qwen-llm command:

```bash
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live \
  profile_native_deepseek_v4_exact_packed_prefix_decode_at_129_and_513 -- \
  --ignored --exact --nocapture --test-threads=1
```

llm exact injection results:

| Position | Injection samples (ms) | Median (ms) | tokens/s | Repeated vector SHA-256 |
|---:|---|---:|---:|---|
| 129 | 36.457, 36.255, 36.414, 36.413, 36.231 | 36.413 | 27.463 | `a6d8be85ce24f9ab9e52d4b041343cc98cd6e83b7e2f2ff550d8736348776605` |
| 513 | 36.819, 36.689, 36.869, 36.968, 36.954 | 36.869 | 27.123 | `a834cc89a9f512f1eef1cd20a2711ad591b591b6c2b842318140c488a4cf0a78` |

qwen-llm exact restored-state results:

| Position | Warm restored samples (ms) | Median (ms) | tokens/s | Repeated vector SHA-256 |
|---:|---|---:|---:|---|
| 129 | 73.012, 68.387, 72.334, 75.147, 73.198 | 73.012 | 13.696 | `4490618b733ff6e4841b3beba176220f939f43366584f58c513bf3b8c38ec2fc` |
| 513 | 73.466, 73.971, 72.106, 74.350, 74.946 | 73.971 | 13.519 | `7f358590cd7d483f6dbe30a9cf7f8acb747845fefd44a84c0c940f7458e2d179` |

Every vector repeats byte-for-byte. Vector equality across engines is not
expected: b10235 uses its native batched prompt schedule, while qwen-llm uses
fixed 128-token product chunks and its promoted reduction contracts.

The qwen timings restore a canonical snapshot before each token. That host write
changes cache residency: the first restored sample was 316.646 ms at position
129 in the retained packet, while its five warm restored repeats were stable.
These numbers diagnose restore-to-decode behavior and must not replace the
sequential product-warm gate above.

## Exact Prefix Prefill

| Engine | Prefix tokens | Time (ms) | tokens/s |
|---|---:|---:|---:|
| qwen-llm packed chunks | 129 | 3339.109 | 38.633 |
| llama_core b10235 median | 129 | 660.698 | 195.248 |
| qwen-llm packed chunks | 513 | 10361.757 | 49.509 |
| llama_core b10235 median | 513 | 2554.937 | 200.788 |

qwen-llm's timer excludes model/session construction and snapshot capture. The
513-token result is the sum of the independently timed 129-token prefix and its
three 128-token continuation chunks. llama_core reuses the loaded model but
creates a fresh context for each repeat. The qwen row is one cold-first-use
observation and can include lazy Metal pipeline creation; the llama_core row is
a five-repeat median over a warmed backend. These values are retained as a
prompt-path diagnostic, not a comparable ratio or promotion gate. The packet
does not retain all five llama_core prefill samples.

## Roofline Prior

The existing Qwen MoE scoreboard provides a planning prior, not attribution for
the measured 23.5 ms gap:

- Qwen3.6-35B-A3B reaches about 82 tokens/s in qwen-llm and 76 in llama.cpp.
  A census estimate of 2.0-2.1 GB active bytes/token implies about 168 GB/s, or
  roughly 35% of the maintained 474 GB/s stream anchor. Small gathered experts
  realize much less bandwidth than dense 27B.
- Qwen3.5-122B-A10B reaches about 45 tokens/s in qwen-llm and 36 in llama.cpp.
  Its estimated 6.3-7.2 GB active bytes/token realizes about 285-325 GB/s, or
  60-69% of the anchor. Larger experts and a larger active set amortize the
  fixed per-token floor materially better than A3B.
- DeepSeek V4 has an estimated 9.19 GB active bytes/token, with about 76% in
  dense/batched-friendly projections and 24% in gathered expert projections.
  Applying the measured A10B realization band gives a planning range near
  31-36 tokens/s; the higher dense share supports a working center near
  35 tokens/s and an honest 30-40 range.

That range is deliberately not an S6 gate. It says the current 16.7 tokens/s is
not a demonstrated engine or bandwidth ceiling: this binary already executes a
roughly 72 GiB sharded A10B MoE at about 22 ms/token. It does not prove that
routing owns the current fixed gap. The next profile must separate router GPU
completion, CPU route/copy, router-to-expert GPU idle, and expert GPU work.

## Consequence

The dense-attention checkpoint solved the wrong-scaling component: both decode
curves are flat over this range. It did not solve the constant floor. The next
short-context experiment must attribute the route seam before replacing it.
Proceed to a GPU route-record ABI only if the aggregate interval from router GPU
completion to expert GPU start exposes at least 5 ms/token of removable wall
time. CPU route/copy explains that interval and must not be added to it again.
Otherwise target command submission or expert projections. Any route design must
keep selected expert IDs asynchronously observable for future SSD expert
streaming. Far-context work remains independently led by the measured 8.232 ms
Lightning Indexer score phase.

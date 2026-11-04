# How qwen-llm uses llama.cpp

Models never run through llama.cpp. It plays four separate roles here. Each
has its own pin and its own meaning, and evidence from one role does not carry
over to another.

| Role | What it answers | Where | Pin |
|---|---|---|---|
| Correctness oracle | Does our output match an independent implementation of the same artifact or operation? | `scripts/reference/glm53/` (GLM-5.3-Flash same-artifact logits), `scripts/reference/dsv4_llama_cpp_cpu_oracle.cpp` (DeepSeek V4 operation fixtures), `LlamaCppTokenizer` as a check on the native tokenizers | Per harness: `GLM53_REFERENCE_REVISION` in `scripts/reference/glm53/CMakeLists.txt`; the linked library for the tokenizer |
| CPU library | How does a GGML type decode or encode on the CPU, and how do families without a native tokenizer tokenize? | `llama-cpp-sys-2` in the workspace `Cargo.toml`; `crates/qwen-llm/src/codec.rs`; `LlamaCppTokenizer` in `crates/qwen-llm/src/tokenizer.rs` (Muse Glimmer serving and some lens tools) | The `llama-cpp-sys-2` git revision |
| Family profiling baseline | How does a supported family compare with `llama-bench` on the same file and machine? | `scripts/bench/llama-cpp.lock.json`, `scripts/bench/models-families.toml`, [BENCH](BENCH.md) | The lock file |
| Performance reference | Which kernel shapes, data flows, fusions and Metal techniques are worth studying? | Upstream source and pull requests; adapted code is listed in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES.md) | None: read current upstream |

## Intent and limits

**Correctness oracle.** llama.cpp is one reference among several, used where
it runs the exact artifact under test (GLM-5.3-Flash), where its ggml CPU ops
give an independent operation fixture (DeepSeek V4), or as a byte-exact
tokenizer check. Other families and operations are checked against
independent references under `scripts/reference/`. Agreement shows that we
match llama.cpp's implementation, which can itself be wrong: its qwen4exp
sparse-attention selection was incorrect beyond about 2K visible tokens until
upstream #29751. The GLM harness pins its revision and settings and refuses
others; re-pinning an oracle means re-capturing its references, not diffing
against old captures.

**CPU library.** The codec covers every GGML type through one interface for
tooling, verification and re-pack-on-load; it is never the GPU hot path. The
tokenizer backend serves families that have no native tokenizer (Muse
Glimmer); the others tokenize natively. Updating the library is maintenance
(new types, API changes) and moves both the codec and that tokenizer. It is
not a reason for new kernels: a new type gains native GPU support only when an
artifact we run needs it.

**Family profiling baseline.** The lock exists to produce like-for-like
`llama-bench` numbers. A ratio against it describes one file, one machine and
one llama.cpp revision. Record the revision and hardware with every
comparison, and re-pin deliberately: a baseline that is stale or wrong for a
family makes its ratios meaningless. Families the pinned revision cannot run
are reported, not measured (`lcpp = false`).

**Performance reference.** llama.cpp is a source of ideas, not a performance
oracle or a roofline. Our targets come from hardware limits (bytes moved,
arithmetic, measured kernel ceilings) recorded in [PERF-LOG](PERF-LOG.md).
An upstream technique becomes ours only after our own measurement and
qualification; upstream numbers on other chips are not evidence for ours.

Because llama.cpp defines GGUF, changes to its converter and quantizer reach
us as artifact admission work. That is a consequence of the format rather than
a fifth role.

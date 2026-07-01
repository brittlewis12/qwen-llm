# v0.405 Group8 S2 Read-Once Kill

Goal: test the strongest bounded A3B attention hypothesis left after v0.404:
whether reading each K/V tile once across all eight group-8 Q heads can beat the
current tile4 path that reads K/V once per four Q heads.

Implementation shape tested, then reverted:

- `GROUP=8`, F16 KV, `head_dim=256`
- one threadgroup per `(kv_head, partition)`
- two simdgroups per threadgroup, each handling four Q heads
- K/V tile staged once in threadgroup memory and consumed by both simdgroups
- existing partial layout and h2 reduce path reused unchanged
- gated by `QWEN_ATTN_V4_G8_S2_ORACLE=1`

Validation:

- `cargo check -p qwen-cli --bin qwen-bench`
- `cargo build --release -p qwen-cli --bin qwen-bench`
- `QWEN_ATTN_V4_G8_S2_ORACLE=1 cargo test -p qwen-llm \
  attn_v4_group8_subgroup_matches_naive_f16kv --release -- --ignored --nocapture`
- A3B `attn-intra` oracle at `ctx16384` and `ctx32768`
- `cx ask` review, session `019f1e10-a733-73b1-9f89-dbd01eeb55bd`

Artifacts:

- `target/profiles/v0405-a3b-ctx16384-attn-intra-g8-s2-oracle-nwg256.out`
- `target/profiles/v0405-a3b-ctx32768-attn-intra-g8-s2-oracle-nwg256.out`
- `target/profiles/v0405-a3b-ctx16384-attn-intra-g8-s2-oracle-c32-nwg256.out`

## Results

Correctness passed against the naive F16-KV attention oracle:

| Shape | max abs | Cosine |
| --- | ---: | ---: |
| `n_pos=4096`, `nwg=64`, `C=64` | `7.39e-10` | `1.000000` |
| `n_pos=6144`, `nwg=64`, `C=64` | `8.26e-10` | `1.000000` |

Performance failed the gate by a large margin:

| Ctx | Variant | Layer ms | Main ms | Reduce ms | Read |
| ---: | --- | ---: | ---: | ---: | --- |
| `16384` | default C64 | `0.2686` | `0.0987` | `0.0675` | baseline |
| `16384` | S2 C64 | `0.7116` | `0.5410` | `0.0680` | `2.65x` slower layer |
| `16384` | S2 C32 | `0.5313` | `0.3621` | `0.0680` | still `1.98x` slower |
| `32768` | default C64 | `0.3466` | `0.1758` | `0.0676` | baseline |
| `32768` | S2 C64 | `1.1475` | `0.9768` | `0.0678` | `3.31x` slower layer |

## Decision

Kill the cooperative threadgroup-memory read-once branch. It was exact, but the
main body regressed `3.7-5.6x` while reduce stayed flat. The byte model was too
optimistic: the default tile4 path appears to preserve enough grid parallelism,
cache behavior, and occupancy that duplicating K/V reads is cheaper than staging
large K/V tiles through threadgroup memory and synchronizing two simdgroups.

This does not kill every future attention idea, but it demotes read-once attention
unless the next design avoids this failure mode. Do not trade away Q-head/grid
parallelism or add large TGM staging only to reduce theoretical K/V bytes. The
next hardware-saturation branch should move to A3B multi-slot batching or a broad
decode byte/fusion audit before reopening attention.

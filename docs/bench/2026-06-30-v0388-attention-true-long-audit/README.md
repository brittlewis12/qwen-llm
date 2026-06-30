# v0.388 Attention True-Long Audit

Refreshed the A3B `ctx32768` attention intra-layer harness after the route and
gate/up falsifiers. The goal was to check whether any existing `attn_v4` group8
tile or `NWG` knob was still a plausible true-long recovery path before starting
a larger attention body rewrite.

## Commands

```bash
target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  --runs 3 \
  > target/profiles/v0388-a3b-q4-attn-intra-ctx32768-default.out

QWEN_ATTN_V4_G8_TILE=2 QWEN_ATTN_V4_NWG=64 \
  target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  --runs 3 \
  > target/profiles/v0388-a3b-q4-attn-intra-ctx32768-tile2-nwg64.out

QWEN_ATTN_V4_G8_TILE=8 QWEN_ATTN_V4_NWG=256 \
  target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  --runs 3 \
  > target/profiles/v0388-a3b-q4-attn-intra-ctx32768-tile8-nwg256.out

QWEN_ATTN_V4_G8_TILE=4 QWEN_ATTN_V4_NWG=128 \
  target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  --runs 3 \
  > target/profiles/v0388-a3b-q4-attn-intra-ctx32768-tile4-nwg128.out
```

## Results

| Variant | Tile | NWG | One layer | Extrapolated | Main | Reduce | Main est BW | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default | 4 | 256 | `0.3469 ms` | `3.4686 ms` | `0.1752 ms` | `0.0676 ms` | `790.4 GB/s` | best current shape |
| tile2/nwg64 | 2 | 64 | `0.5265 ms` | `5.2647 ms` | `0.3887 ms` | `0.0344 ms` | `693.3 GB/s` | worse main body |
| tile8/nwg256 | 8 | 256 | `0.3971 ms` | `3.9710 ms` | `0.2243 ms` | `0.0675 ms` | `318.1 GB/s` | fewer logical bytes, worse occupancy |
| tile4/nwg128 | 4 | 128 | `0.3810 ms` | `3.8100 ms` | `0.2411 ms` | `0.0353 ms` | `565.5 GB/s` | reduce halves, main loses |

The current default extrapolates to `3.47 ms` across A3B's 10 attention layers,
close to the latest full phase row's `attn mixer = 3.31 ms` at `ctx32768`. That
confirms the harness is still phase-faithful enough to reject knob-level changes.

## Decision

Do not promote any existing `group_tile` or `NWG` override. `NWG=128` proves the
reduce row is movable, but it slows the main pass more than it saves. Tile8
reduces the logical K/V reread but loses too much main-body throughput.

The attention branch remains live only for a materially new mechanism: hidden
traffic counters, a true read-once/body rewrite that preserves occupancy, or an
end-to-end `ctx32768` prototype that moves throughput despite the intra-layer
body already estimating near stream bandwidth.

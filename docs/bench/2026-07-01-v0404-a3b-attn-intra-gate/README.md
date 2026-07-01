# v0.404 A3B Attention Intra Gate

Goal: decompose the current A3B group-8 decode attention cost at long context,
verify that `attn-intra` remains phase-faithful, and decide whether another
local NWG/reduce selector retune is worth defaulting.

Command shape:

```sh
target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx 16384 --runs 5

QWEN_ATTN_V4_NWG=192 target/release/qwen-bench attn-intra \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --ctx 32768 --runs 5
```

Validation:

- AC power confirmed with `pmset -g batt`
- A3B `attn-intra` default at `ctx8192`, `ctx16384`, and `ctx32768`
- A3B `attn-intra` NWG override probes at `ctx16384`
- A3B `attn-intra` NWG192 at `ctx32768`
- A3B deep phase NWG192 at `ctx16384` and `ctx32768`
- `cx ask` review, session `019f1dbf-a094-7e52-841b-7b0b9c49a80a`
- Code inspection confirms h2 reduce is already selected for `nwg >= 128`

Artifacts:

- `target/profiles/v0404-a3b-ctx8192-attn-intra-default.out`
- `target/profiles/v0404-a3b-ctx16384-attn-intra-default.out`
- `target/profiles/v0404-a3b-ctx32768-attn-intra-default.out`
- `target/profiles/v0404-a3b-ctx16384-attn-intra-nwg128.out`
- `target/profiles/v0404-a3b-ctx16384-attn-intra-nwg192.out`
- `target/profiles/v0404-a3b-ctx16384-attn-intra-nwg224.out`
- `target/profiles/v0404-a3b-ctx32768-attn-intra-nwg192.out`
- `target/profiles/v0404-a3b-ctx32768-deep-phase.out`
- `target/profiles/v0404-a3b-ctx16384-deep-phase-nwg192.out`
- `target/profiles/v0404-a3b-ctx32768-deep-phase-nwg192.out`

## Results

Default attention decomposition:

| Ctx | NWG | Layer ms | Main ms | Reduce ms | Extrapolated ms | Phase check |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `8192` | `64` | `0.2392` | `0.1051` | `0.0341` | `2.3923` | vs v0.403 phase `2.30` |
| `16384` | `256` | `0.2686` | `0.0987` | `0.0675` | `2.6861` | vs v0.403 phase `2.53` |
| `32768` | `256` | `0.3466` | `0.1758` | `0.0676` | `3.4662` | vs phase `3.30` |

NWG interpolation at `ctx16384`:

| NWG | Layer ms | Main ms | Reduce ms | Extrapolated ms | Read |
| ---: | ---: | ---: | ---: | ---: | --- |
| `128` | `0.2650` | `0.1281` | `0.0353` | `2.6498` | reduce lower, main worse |
| `192` | `0.2515` | `0.0975` | `0.0516` | `2.5152` | best local row |
| `224` | `0.2587` | `0.0968` | `0.0593` | `2.5872` | between 192 and 256 |
| `256` | `0.2686` | `0.0987` | `0.0675` | `2.6861` | current default |

NWG192 production-phase check:

| Ctx | Default phase sum | NWG192 phase sum | Default attn | NWG192 attn | Read |
| ---: | ---: | ---: | ---: | ---: | --- |
| `16384` | `11.09 ms` | `10.94 ms` | `2.53 ms` | `2.39 ms` | small positive |
| `32768` | `11.78 ms` | `11.65 ms` | `3.30 ms` | `3.14 ms` | small positive |

## Decision

`attn-intra` is phase-faithful enough to drive A3B attention decisions: its
extrapolated attention cost is within roughly `5-7%` of the phase bucket through
`ctx32768`.

The local selector shelf is still not the main branch. NWG192 is the best tested
local override and saves about `0.13-0.15 ms` in the deep phase at long context,
but it does not clear the `>=0.30-0.40 ms` phase-equivalent gate for a production
attention rewrite. Treat it as a known small A3B long-context knob, not as the
highest-leverage default change.

The next attention branch should be a main-body traffic oracle: read K/V fewer
times or reorganize partial traffic without reproducing the tile8 occupancy
collapse. If that oracle cannot clear `>=1.15x` one-layer speedup or
`>=0.30-0.40 ms` full-decode-equivalent savings at `ctx16384/32768`, pivot back
to the next broad decode byte-reduction branch.

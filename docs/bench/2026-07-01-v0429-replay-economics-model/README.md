# v0.429 Replay Economics Model

Goal: make the replay scheduler gate reproducible by combining measured gross
block-slice savings with measured real-prompt margin fallback rates.

Command:

```sh
uv run scripts/profile/replay_economics.py \
  --margin-summary target/profiles/v0427-a3b-real-margin-summary.tsv \
  --replay target/profiles/v0428-a3b-block-slice-occupancy-b0-pos4096.out \
  --replay target/profiles/v0428-a3b-block-slice-occupancy-b20-pos4096.out \
  --replay target/profiles/v0428-a3b-block-slice-occupancy-b0-blocks2-pos4096.out \
  --replay target/profiles/v0428-a3b-block-slice-occupancy-b20-blocks2-pos4096.out \
  > target/profiles/v0429-a3b-replay-economics.tsv
```

Validation:

- `uv run scripts/profile/replay_economics.py` over v0.427/v0.428 artifacts

## Model

This intentionally uses cx's conservative post-replay fallback model:

```text
net_save_pct ~= gross_replay_save_pct - fallback_rate_pct
```

That assumes a low-margin row pays both replay and exact fallback, with no early
abort and no extra validation overhead. It is not a final scheduler model; it is
the first kill/keep gate.

## Results

At the first plausible `3e-4` margin point:

| Window | Slots | Gross save | Net save after fallback |
| --- | ---: | ---: | ---: |
| `block0..2` | `6` | `19.60%` | `11.09%` |
| `block20..22` | `6` | `18.90%` | `10.39%` |
| `block0..2` | `8` | `23.30%` | `14.79%` |
| `block20..22` | `8` | `23.10%` | `14.59%` |
| `block0..4` | `6` | `11.00%` | `2.49%` |
| `block20..24` | `6` | `11.60%` | `3.09%` |
| `block0..4` | `8` | `16.20%` | `7.69%` |
| `block20..24` | `8` | `16.80%` | `8.29%` |

At `1e-3`, all S=6 rows are negative and S=8 `blocks=2` is roughly parity
(`-0.10%` / `-0.30%`). At S<=4, every `3e-4` row is negative.

## Decision

Replay is still viable only in a narrow policy envelope: high occupancy (`S>=6`),
guard threshold near `3e-4`, and preferably `blocks=2` GDN-only pairs. `blocks=4`
S=8 still clears a modest `~8%` net model; S=6 is too thin unless exact fallback
can abort before paying most replay work.

The next unknown is not arithmetic anymore. It is mechanism: validation overhead,
exact fallback restore cost, and realistic ragged occupancy. If those cannot keep
S>=6 `blocks=2` above a durable `>5-8%` end-to-end net gate, park replay.

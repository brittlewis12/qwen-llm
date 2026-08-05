# DeepSeek V4 Multi-Group Selector Crossover

Date: 2026-08-05

Status: crossover map `GO`; integrated experimental route `HOLD`.

## Question

The terminal Phase-B selector is substantially faster than radix4, but its 18
dispatches and private/public mask passes impose a fixed cost. Production must
not infer eligibility from terminal evidence alone, especially when a large
request-sized physical cache is only partly visible.

This campaign maps both physical capacity and visible rows before any session
scratch or production dispatch is added.

## Protocol

Thirteen capacity/visibility cells cover equal-capacity crossover, half-full
large sessions, the proposed three-quarter boundary, the exact 1,048,576-forward
capacity, and the decimal-million capacity. Mixed and all-tied scores run
separately.

Each cell performs one exact current/candidate comparison, four additional warm
pairs, then 16 current/candidate/current samples. Every candidate invocation has
a fresh nonzero generation. Full mask, cache-order IDs, count, and status match
outside timing. Tensor allocation and readback remain outside timed commands;
wall timing includes command creation and encoding.

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_multigroup_selector_crossover \
  -- --ignored --exact --nocapture
```

## Equal Capacity

Savings are candidate improvement against the faster control in milliseconds.

| Capacity = visible | Mixed GPU | Mixed wall | Tied GPU | Tied wall | Stable |
|---:|---:|---:|---:|---:|---|
| 16,384 | -1.309 | -1.361 | -1.297 | -1.308 | no |
| 32,768 | -0.279 | -0.299 | -0.516 | -0.544 | no |
| 49,152 | -0.047 | -0.070 | -0.257 | -0.314 | mixed no |
| 65,536 | 0.048 | 0.043 | -0.153 | -0.184 | yes |
| 98,304 | 0.205 | 0.232 | 0.059 | 0.059 | yes |
| 131,072 | 0.427 | 0.461 | 0.264 | 0.268 | yes |
| 196,608 | 0.848 | 0.876 | 0.691 | 0.677 | yes |
| 262,144 | 1.197 | 1.182 | 1.100 | 1.103 | yes |

The first both-case win at 98,304 rows is real but far below the 0.50 ms
production headroom. The first measured equal-capacity cell clearing that bound
in all four columns is 196,608.

The 16K/32K cells and mixed 49K cell are discovery-only: control clocks change
materially inside their brackets. They agree directionally with the stable
larger cells but do not authorize a boundary.

## Split Capacity

| Capacity | Visible | Mixed GPU | Mixed wall | Tied GPU | Tied wall | Stable |
|---:|---:|---:|---:|---:|---:|---|
| 131,072 | 65,536 | -0.021 | -0.024 | -0.224 | -0.232 | yes |
| 262,144 | 65,536 | -0.165 | -0.175 | -0.372 | -0.399 | yes |
| 262,144 | 131,072 | 0.297 | 0.310 | 0.122 | 0.103 | tied wall no |
| 262,144 | 196,608 | 0.787 | 0.794 | 0.627 | 0.638 | yes |
| 250,112 | 196,608 | 0.800 | 0.796 | 0.641 | 0.635 | yes |

Visible rows cannot stand alone: private validation and publication still scan
physical capacity. The max-capacity half-full cell wins, but lacks the frozen
headroom and one tied wall bracket exceeds 5% drift. Both the binary 1M and
decimal-million three-quarter cells clear the bound cleanly.

`run.log` retains all 2,496 GPU/wall samples and exact output assertions. Its
53,417 bytes have SHA-256
`bd2d38ed740db2cc33a8d08a844d611504fedc3a57f854381a3b625e8f9b89eb`.

## Decision

Freeze this experimental predicate:

```text
explicit opt-in
query_count = 1
top_k = 512
visible_rows <= capacity_rows <= 262,144
visible_rows >= 196,608
visible_rows >= capacity_rows - capacity_rows / 4
```

The explicit capacity ceiling prevents these measurements from authorizing a
future larger model geometry. Division-first ratio arithmetic is overflow-safe;
current capacities are 256-row aligned, so the three-quarter boundary is exact.

This packet does not enable routing. Next add session-owned scratch, memory
accounting, and a wrap-safe generation owner behind an off-by-default seam. The
final gate reruns 24-sample exact current/candidate/current cells at
196,608/196,608, 250,112/196,608, and 262,144/196,608 through that integrated
seam, retaining the Phase-B 1.35/1.40 ms candidate ceilings and requiring at
least 0.50 ms GPU and wall saving in every mixed/tied cell.

CX review: `019fcf7d-e9d4-7150-b496-e70a31958e80`.

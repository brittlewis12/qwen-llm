# Performance instrumentation guide

This is the agent-first profiling guide for `qwen-llm` on macOS. It documents
how to measure throughput, attribute bottlenecks, and choose the right profiling
tool without sudo, GUI automation, or optimization guesswork.

The working model is:

1. Use `qwen-bench` for model-aware throughput and phase numbers.
2. Use `hyperfine` for stable command-level before/after comparisons.
3. Use `xcrun xctrace` for headless Instruments recordings.
4. Use compact parsers (`ztrace`, repo-local scripts, small table extractors)
   instead of raw XML.
5. Use source-built `samply` when off-CPU profiles are useful.
6. Add repo-native summary scripts where vendor tools are not enough.

## Scope and rules

- This guide is about measurement workflow, not tool provisioning. If tools are
  missing, report the missing tool or use a fallback; do not install or update
  tools as part of a profiling session. See `docs/PERF-TOOLS-SETUP.md` for
  intentional setup work.
- Keep throughput measurement separate from attribution. Profiler overhead is
  expected; use profiler traces to explain, not to set regression thresholds.
- Label the measurement regime every time: cold/tooling, throughput, steady
  decode attribution, GPU phase attribution, Metal timeline, allocation, or
  leak check.
- Keep raw trace exports out of agent context. Export only the table you need,
  write large exports to files, and summarize them.
- Prefer unique output names under `target/profiles`; `xctrace` will not
  overwrite an existing trace bundle unless `--append-run` is used.

## Why this is a document first

Keep this as a project document before turning it into a skill. The commands are
repo-specific, depend on model paths, and need to evolve with `qwen-bench`.
A skill can be extracted later as a thin wrapper once the command shapes and
summary formats are stable.

Good candidates for a later skill:

- `profile-cpu`: record `Time Profiler`, summarize with `ztrace`.
- `profile-gpu`: record `Metal System Trace`, summarize command buffers and GPU
  intervals.
- `profile-memory`: run `Allocations` or `leaks`, summarize allocation churn.
- `profile-compare`: run `hyperfine` and diff JSON results.

## Preflight

Use non-mutating checks before a profiling session:

```sh
mkdir -p target/profiles
: "${MODEL:?set MODEL to a GGUF}"
test -f "$MODEL"
xcrun xctrace list templates >/dev/null
ztrace --help >/dev/null
hyperfine --version >/dev/null
```

Optional tools can be checked only when needed:

```sh
samply record --help >/dev/null
uniprof --version >/dev/null
```

If an optional tool is missing, use another path from the decision table instead
of changing the environment during measurement.

## Quickstart

This path gives a baseline, a model-aware phase view, and a CPU attribution trace:

```sh
mkdir -p target/profiles
: "${MODEL:?set MODEL to a GGUF}"
test -f "$MODEL"

cargo build --release -p qwen-cli --bin qwen-bench

./target/release/qwen-bench decode -m "$MODEL" --tokens 128
./target/release/qwen-bench phase -m "$MODEL" --ctx 256

TRACE="target/profiles/time-decode-$(date +%Y%m%d-%H%M%S).trace"
xcrun xctrace record --no-prompt \
  --template "Time Profiler" \
  --time-limit 30s \
  --output "$TRACE" \
  --target-stdout - \
  --launch -- ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 256 --no-warmup

ztrace summary "$TRACE" --threshold 0.5 --depth 8
```

Use this as the starting point, then switch tools based on the question below.

## Decision table

| Question | First tool | Why |
| --- | --- | --- |
| Did throughput regress? | `qwen-bench`, `hyperfine` | Stable wall-clock and model-aware numbers. |
| Which model phase dominates? | `qwen-bench phase` | Knows GDN, attention, LM head, DFlash phases. |
| Is host/FFI/argmax/load CPU expensive? | `xctrace` + `ztrace` | Best headless symbolicated CPU stack summaries. |
| Is the process blocked on GPU completion? | source-built `samply --presymbolicate` | Shows off-CPU waits and CPU deltas. |
| Are command buffers/gaps/competing GPU work visible? | `Metal System Trace` | Exposes Metal app and GPU interval tables. |
| Are hardware GPU counters needed? | `Metal GPU Counters` or `GPU`, validated by export | Useful only if counter tables contain rows. |
| Are there excess allocations? | `xctrace Allocations`, then repo allocator counters | Instruments finds churn; code counters can enforce steady-state budgets. |
| Is there a leak? | `leaks --atExit` | Fast no-GUI leak sanity check. |
| Is a kernel microbench better/worse? | Criterion JSON | Existing bench framework already emits estimates. |
| Need a general CPU profiler for another runtime? | `uniprof` | Unified agent/MCP-friendly interface. |

## Measurement regimes

Do not compare numbers across regimes without labeling them.

| Regime | Use when | Command shape | Compare? |
| --- | --- | --- | --- |
| Cold/tooling trial | Characterizing profiler overhead, load, or first-use costs | `qwen-bench ... --no-warmup` under the profiler | No, use for attribution only. |
| Throughput comparison | Confirming before/after speed | `qwen-bench` without `--no-warmup`; optionally wrap in `hyperfine --warmup` | Yes, if command, model, prompt, and build are fixed. |
| Steady decode attribution | Finding CPU work during decode | Longer token counts under `Time Profiler` or `samply` | No, explain a separate throughput result. |
| GPU phase attribution | Understanding model phase share | `qwen-bench phase` or `qwen-bench dflash --profile` | Compare phase shares within the same harness. |
| Metal timeline | Finding queue gaps, command-buffer cadence, or competing GPU ownership | `Metal System Trace` | Compare timeline summaries, not raw wall time. |
| DFlash correctness/perf | Measuring speculative decode safely | `qwen-bench dflash` with equivalence check enabled | Yes, if correctness passes. |
| Allocation churn | Finding heap/VM categories | `xctrace` `Allocations`; code counters for budgets | Compare normalized counters, not trace size. |

## Trace hygiene

- Create `target/profiles` before recording.
- Use a unique trace path for each run, or intentionally pass `--append-run`.
- Keep the profiled command simple and explicit after `--launch --`; do not rely
  on app lookup.
- Use `xcrun xctrace export --toc` to discover table names before writing a
  parser.
- Export only the table you need with `--xpath`; avoid dumping full XML into an
  agent session.
- Run `qwen-bench` or `hyperfine` separately for throughput. Traced runs are for
  attribution.

## Baseline commands

Build the release binary before profiling:

```sh
cargo build --release -p qwen-cli --bin qwen-bench
```

Set `MODEL` explicitly in scripts and recorded commands:

```sh
: "${MODEL:?set MODEL to a GGUF}"
test -f "$MODEL"

# Plain decode throughput. Omit --no-warmup for throughput comparisons.
./target/release/qwen-bench decode -m "$MODEL" --tokens 128

# Cold/tooling smoke run.
./target/release/qwen-bench decode -m "$MODEL" --tokens 64 --no-warmup

# Context sensitivity.
./target/release/qwen-bench ctx-sweep -m "$MODEL" --checkpoints 1,16,64 --window 2

# Phase attribution at one context length.
./target/release/qwen-bench phase -m "$MODEL" --ctx 256
```

## Speculative and DFlash targets

Plain `decode` is not always the right target. For speculative work, set model
variables explicitly and keep the prompt and EOS policy consistent across
comparisons:

```sh
: "${TARGET_MODEL:?set TARGET_MODEL to target GGUF}"
: "${DRAFTER_MODEL:?set DRAFTER_MODEL to DFlash drafter GGUF}"
: "${MTP_MODEL:?set MTP_MODEL to MTP-aware GGUF}"
test -f "$TARGET_MODEL"
test -f "$DRAFTER_MODEL"
test -f "$MTP_MODEL"

# MTP acceptance/speedup surface.
./target/release/qwen-bench mtp \
  -m "$MTP_MODEL" --tokens 64 --spec-tokens 1

# DFlash acceptance signal; use --effective-n to see where alpha decays.
./target/release/qwen-bench dflash-lazy \
  -m "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" \
  --tokens 32 --effective-n 0

# Production DFlash path with drafter phase timers and correctness gate.
./target/release/qwen-bench dflash \
  -m "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" \
  --tokens 64 --profile --n-policy adaptive
```

DFlash rules:

- `dflash` defaults to an equivalence check against DFlash-off. Keep that on for
  correctness-gated measurements; use `--skip-equivalence-check` only for narrow
  profiling runs where token equivalence has already been established.
- Compare `--n-policy adaptive`, `static-16`, `static-8`, `static-4`, and `off`
  when tuning verify-chain policy.
- `--profile` reports drafter phase timers; it is not a substitute for a full
  Metal System Trace when queue gaps, command-buffer cadence, or GPU ownership
  are the question.

## Headless CPU profiling

Record with Instruments, then summarize with `ztrace`:

```sh
: "${MODEL:?set MODEL to a GGUF}"
TRACE="target/profiles/time-decode-$(date +%Y%m%d-%H%M%S).trace"

xcrun xctrace record --no-prompt \
  --template "Time Profiler" \
  --time-limit 30s \
  --output "$TRACE" \
  --target-stdout - \
  --launch -- ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 256 --no-warmup

ztrace summary "$TRACE" --threshold 0.5 --depth 8
```

Use this path to inspect host-side costs such as model load, tokenizer/vocab
work, GGUF reading, Metal host dispatch, CPU sampling/argmax, FFI boundaries,
and per-token control flow.

Rules:

- Prefer a direct binary path after `--launch --`.
- `qwen-bench` prints result lines to stderr. `xctrace` supports
  `--target-stdout -`, but not `--target-stderr`; do not build automation that
  depends on traced bench output being captured.
- Export `time-profile` only to a file or parser; do not dump raw XML into agent
  context.
- Verify the trace target path in `xcrun xctrace export --toc` if results look
  strange.

## CPU/off-CPU profiling with samply

Use `samply` when blocked time is the question, especially GPU waits, locks, or
host-side synchronization:

```sh
: "${MODEL:?set MODEL to a GGUF}"
samply record \
  --save-only --presymbolicate \
  -o target/profiles/samply-decode.json.gz \
  ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 256 --no-warmup
```

Notes:

- On macOS, `samply` can profile locally built or unsigned binaries. Attaching
  to running processes may require setup outside the profiling session.
- Saved `samply` JSON is less directly readable than `ztrace` output unless a
  summarizer is used. A future `mcp-samply` or repo-local summarizer would make
  it more agent-friendly.

## GPU and Metal timeline profiling

Record Metal System Trace when the question involves command buffers, queue
gaps, GPU interval ownership, resource events, or competing GPU work:

```sh
: "${MODEL:?set MODEL to a GGUF}"
TRACE="target/profiles/metal-decode-$(date +%Y%m%d-%H%M%S).trace"

xcrun xctrace record --no-prompt \
  --template "Metal System Trace" \
  --time-limit 30s \
  --output "$TRACE" \
  --target-stdout - \
  --launch -- ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 64 --no-warmup
```

For big models where load time would consume most of the trace window, use the
attach helper instead so recording starts after the bench prints its ready line:

```sh
uv run scripts/profile/trace_attach.py \
  --trace target/profiles/metal-attach.trace \
  --stdout /tmp/qwen-attach.out \
  --stderr /tmp/qwen-attach.err \
  --time-limit 45s \
  --env QWEN_PREFILL_TRACE_LABELS=1 \
  -- target/release/qwen-bench pp -m "$MODEL" -p 4100 --runs 1 --no-warmup
```

Inspect available tables before parsing:

```sh
xcrun xctrace export --input "$TRACE" --toc
```

Tables that are usually useful for this workload:

- `metal-application-command-buffer-submissions`
- `metal-application-encoders-list`
- `metal-gpu-intervals`
- `metal-driver-event-intervals`
- `metal-current-allocated-size`
- `metal-resource-allocations`
- `time-profile`

Until a repo-local parser exists, this compact extractor gives a first-pass
summary of target-process compute intervals and gaps. Treat schema names,
process names, and timestamp fields as trace-version dependent:

```sh
uv run python - <<'PY'
import subprocess, xml.etree.ElementTree as ET, statistics as st, os
trace = os.environ.get('TRACE', 'target/profiles/metal-decode.trace')
schema = 'metal-gpu-intervals'
xml = subprocess.check_output([
    'xcrun', 'xctrace', 'export', '--input', trace,
    '--xpath', f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
], stderr=subprocess.DEVNULL)
root = ET.fromstring(xml)
refs = {}
for el in root.iter():
    if 'id' in el.attrib:
        refs[el.attrib['id']] = el.attrib.get('fmt') or (el.text.strip() if el.text else '')
intervals = []
for row in root.iter('row'):
    d, raw = {}, {}
    for child in row:
        val = refs.get(child.attrib.get('ref')) if 'ref' in child.attrib else None
        if val is None:
            val = child.attrib.get('fmt') or (child.text.strip() if child.text else '')
        d.setdefault(child.tag, val)
        raw.setdefault(child.tag, child.text.strip() if child.text else '')
    if d.get('process') == 'qwen-bench' and d.get('gpu-channel-name') == 'Compute':
        start_ms = int(raw.get('start-time', '0')) / 1e6
        dur_ms = int(raw.get('duration', '0')) / 1e6
        intervals.append((start_ms, dur_ms))
intervals.sort()
gaps = [max(0, intervals[i][0] - (intervals[i-1][0] + intervals[i-1][1]))
        for i in range(1, len(intervals))]
print('target_compute_interval_count', len(intervals))
if intervals:
    durs = [d for _, d in intervals]
    print('compute_total_ms', round(sum(durs), 3), 'compute_median_ms', round(st.median(durs), 3))
if gaps:
    print('gap_total_ms', round(sum(gaps), 3), 'gap_median_us', round(st.median(gaps) * 1000, 1))
if not intervals:
    print('no qwen-bench Compute intervals matched; inspect --toc and process names')
PY
```

### GPU counters and the GPU flag

`xctrace` exposes GPU-related instruments, including `GPU` and
`Metal GPU Counters`. Use them only for hardware-counter questions such as
utilization, stalls, or bandwidth, and validate that the exported counter tables
contain rows before drawing conclusions.

Counter guidance:

- Keep `Metal System Trace` as the primary timeline tool for queue gaps,
  command-buffer cadence, and GPU ownership.
- Use `qwen-bench phase` or `qwen-bench dflash --profile` for model-aware phase
  attribution; raw GPU counters do not know model phases.
- If adding `--instrument "Metal GPU Counters"` or `--instrument "GPU"`, inspect
  `--toc` and verify non-empty counter tables such as `gpu-counter-value` or
  `metal-gpu-counter-intervals` before using the result.
- Treat empty counter tables as "unsupported or not captured for this run", not
  as evidence that the GPU did no work.
- On this M4 Max, default `Metal System Trace` has produced timeline tables but
  only the unhelpful `RT Unit Active` counter; adding `--instrument "Metal GPU
  Counters"` can warn `Selected counter profile is not supported on target
  device` and produce empty counter tables. Treat that as a tooling miss, not a
  kernel conclusion.
- For autonomous GPU efficiency and memory bandwidth, prefer in-process
  `MTLCounterSampleBuffer` support behind a feature or bench flag once the
  project needs stable counter data.

`.trace` versus `.gputrace`:

- `xctrace` and Instruments CLI produce `.trace` bundles.
- `.gputrace` bundles come from Xcode Metal capture or in-process
  `MTLCaptureManager` code.
- If an agent needs `.gputrace`, the host project must include capture code; a
  CLI trace cannot synthesize it after the fact.

## Command-level benchmarking

Use `hyperfine` when comparing builds, feature flags, or external baselines:

```sh
: "${MODEL:?set MODEL to a GGUF}"
hyperfine --warmup 1 --runs 5 \
  --export-json target/profiles/hyperfine-decode.json \
  "./target/release/qwen-bench decode -m \"$MODEL\" --tokens 128"
```

Use `hyperfine` for before/after comparisons, not for attribution. Keep the
command, model, prompt, build profile, and environment fixed across variants.

### qwen-bench pp prompt sweeps

Use `qwen-bench pp` for repo-native prompt-only measurements that line up with
`llama-bench pp<N>` semantics. It times the prompt prefill path only, keeps
session/scratch allocation outside the timed interval, and can skip the final
norm / `lm_head` / logits tail.

```sh
: "${MODEL:?set MODEL to a GGUF}"
./target/release/qwen-bench pp \
  -m "$MODEL" \
  -p 320 \
  --prefill-chunk 320 \
  --runs 5
```

Rules for pp sweeps:

- Do not run pp benchmarks in parallel with any other repo build/bench workload.
- Run promotion/kill sweeps on AC power. `qwen-bench` records a `pmset` power
  snapshot in JSON rows and prints it in text mode; treat battery, battery
  warnings, or thermal/performance warnings as benchmark identity/confounds.
- Treat `pp320` as the llama-bench scoreboard anchor, not as the sole
  representative prompt. For keeper/regression decisions, sweep at least
  `64,128,320,512,1024` when model size allows; for 122B-class runs, `128,320,512`
  is the minimum useful range.
- Treat `--prefill-chunk 1024` as a safe historical cap, not a principled
  long-context optimum. For true-long keeper decisions, sweep `512/1024/2048/4096`
  when scratch and wall-clock budget allow.
- Prefer synthetic token ids for parity with `llama-bench`; use `--prompt` only
  when the question is tokenizer/template dependent.
- Keep `--with-tail` off for pure `llama-bench pp<N>` comparison; use it only to
  price final-logits overhead.
- Record the lowering summary (`gdn_batched`, `attn_batched`, `dense_ffn_batched`,
  `moe_gpu_token_loop`) with any result.
- Treat experimental MoE flags such as `QWEN_PREFILL_MOE_PACKED_ROUTED` as part of
  the benchmark identity and report them explicitly.
- Treat experimental attention flags such as `QWEN_PREFILL_ATTN_MATRIX_G8=1` as
  diagnostic-only unless the roadmap says otherwise. `qwen-bench pp`, `pp-wait`,
  and packed `decode` prefill now size matrix score/V_T scratch from the prompt
  length automatically; only lower-level/custom callers should need
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS=<tokens>`, and they must report it.
- When comparing against `llama-bench`, remember that bench-tool `-fa 0` disables
  flash attention; it is not the library's auto flash-attention setting.
- For cold A10B `pp128` methodology, `QWEN_PP_WARM_MOE_BANKS=1` or
  `QWEN_PP_RESIDENCY_SET=1` removes expert-bank first-touch outliers. Report those
  knobs explicitly and do not claim them as steady-state throughput wins.

### MoE fast-path coverage lane

Correct logits do not prove the intended fast path ran. The A3B Q6-down fix came
from noticing that A3B has `40` MoE layers but a trace showed only `37` grouped
routed labels; three late `Q6_K` down-expert layers had silently fallen to the
per-token fallback. When changing MoE dtype gates, router paths, packed/grouped
thresholds, or loader tensor formats, run a coverage check before trusting the
throughput row.

Minimal trace-label coverage check:

```sh
TRACE="target/profiles/qwen-a3b-pp512-coverage-$(date +%Y%m%d-%H%M%S).trace"
QWEN_PREFILL_TRACE_LABELS=1 xcrun xctrace record --no-prompt \
  --template "Metal System Trace" \
  --output "$TRACE" \
  --launch -- ./target/release/qwen-bench pp \
    -m "$MODEL" -p 512 --prefill-chunk 512 --runs 1 --no-warmup -o json

uv run scripts/profile/trace-metal.py "$TRACE" --process-prefix qwen-bench
```

Rules:

- Compare label counts with model metadata, not vibes. For A3B, expect `40`
  `moe-route-fused`, `40` `moe-routed-grouped`, and `40` `moe-shared-packed`
  labels on the default grouped path.
- Pair trace coverage with `~/code/gguf/target/debug/gguf <model> --tensors` when
  auditing dtype support. Expert `gate/up/down` dtypes can vary by layer.
- Treat any unexpected `moe-routed-token-loop`, `moe-route-token-loop`, or missing
  layer label as a higher-EV bug candidate than local kernel retuning.
- Keep this separate from throughput timing; traced runs are attribution and
  coverage evidence, not promotion numbers.

When long-prompt variants are close enough that run-order drift or thermal sag
can flip the ranking, use the cooled sweep harness instead of ad hoc shell
loops:

```sh
uv run scripts/profile/prefill_sweep.py \
  --model "$MODEL" \
  --n-prompt 19591 \
  --runs 1 \
  --no-warmup \
  --cooldown-seconds 15 \
  --repeat-blocks 2 \
  --shuffle-seed 7 \
  --variant baseline-a \
  --variant packed-r4:QWEN_PREFILL_ATTN_PACKED_G16=1,QWEN_PREFILL_ATTN_PACKED_G16_ROWS=4 \
  --variant baseline-b \
  --output target/profiles/prefill-sweep.json
```

Rules for the cooled harness:

- Keep one or more repeated baseline anchors in the same batch.
- Use `--repeat-blocks` plus `--shuffle-seed` when run-order drift could be as
  large as the claimed effect; the output records block/order for every run.
- Use fresh processes per variant; that keeps env identity simple and captures
  model-load / residency side effects in the outer wall.
- Treat `pmset -g therm` and `memory_pressure -Q` as supporting probes only;
  on this box they stayed flat even when long A10B rankings drifted.
- Split long sweeps into smaller batches instead of relying on one giant timeout.
- When changing packed-attention activation thresholds, require exactness at the
  first newly activated chunk size (for example `pp512` before promoting
  `min_pos=512`). Do not infer safety from later-context oracles alone.

### Prefill phase trace lane

Use phase traces when total-throughput drift is as large as the candidate win.
The trace path commits/waits at phase boundaries, so it is attribution evidence,
not a promotion throughput number.

```sh
QWEN_PREFILL_TRACE_LAYER_PHASES=1 \
QWEN_PREFILL_TRACE_ATTN_PHASES=1 \
target/release/qwen-bench pp \
  -m "$MODEL" \
  -p 4096 \
  --runs 1 \
  --no-warmup \
  2> target/profiles/prefill-phases.log

uv run scripts/profile/prefill_phase_summary.py \
  target/profiles/prefill-phases.log
```

Rules:

- `QWEN_PREFILL_TRACE_LAYER_PHASES=1` emits `prefill-layer-phase` rows for dense
  pre-norm, GDN, mixer residual, and dense FFN phases.
- Dense GDN front traces split the projection bucket into `gdn_qkv`, `gdn_z`, and
  `gdn_beta_alpha`; use that split to catch dispatcher-class mismatches before
  returning to broad FFN or attention hypotheses.
- Add `QWEN_PREFILL_TRACE_ATTN_PHASES=1` when attention attribution matters; by
  itself, layer tracing collapses attention work into one `attn` phase.
- Compare phase-local sums across paired runs before defaulting a sub-noise
  total-throughput candidate such as dense fused-SwiGLU or reduced mat-mat smem.

For dense fused-SwiGLU specifically, use the hidden same-process A/B harness to
avoid model-load and first-process drift while alternating the runtime fused flag:

```sh
QWEN_PREFILL_ATTN_MATRIX_G6=1 \
target/release/qwen-bench pp-ffn-ab \
  -m "$MODEL" \
  -p 4096 \
  --pairs 2
```

Treat this as a candidate-local gate. It overrides only the dense fused-SwiGLU
flag; other env-gated paths such as matrix attention or reduced smem still come
from the process environment.

For llama.cpp Metal node profiles, enable its serialized profile mode and digest
the log with the local summary helper:

```sh
GGML_METAL_PROFILE_OPS=1 \
~/code/llama.cpp/build/bin/llama-bench \
  -m "$MODEL" \
  -p 4096 \
  -n 0 \
  -r 1 \
  --no-warmup \
  -o json \
  -v \
  > target/profiles/lcpp-profile.json \
  2> target/profiles/lcpp-profile.log

uv run scripts/profile/lcpp_metal_profile_summary.py \
  target/profiles/lcpp-profile.log
```

`GGML_METAL_PROFILE_OPS=1` serializes llama.cpp graph nodes, so compare it only
against qwen phase traces, not normal throughput rows.

### Real rollout prompt lane

Synthetic `pp<N>` remains the fast scoreboard harness, but it is not the only
prompt regime we care about. Keep a small sparse lane of real rendered prompts
derived from narrative / interactive rollouts so prompt-shape and chat-template
effects are not forgotten.

Canonical corpus and usage rules live in:

- `docs/bench/real-rollouts/README.md`

Design rule:

- Use synthetic `pp<N>` for short-feedback kernel work.
- Use the real-rollout lane when the hypothesis might depend on prompt shape,
  long preserved assistant history, `<think>` preservation/stripping, or any
  effect that may scale differently from synthetic prompts.
- Do not run a huge ladder by default. Start with one short and one longer real
  prompt; only add more points if the endpoints imply a crossover story.

### qwen-bench tg decode sweeps

Use `qwen-bench tg` for apples-to-apples decode numbers: empty KV per rep,
random tokens, no logits readback.

```sh
: "${MODEL:?set MODEL to a GGUF}"
./target/release/qwen-bench tg \
  -m "$MODEL" \
  -n 128 \
  --runs 3
```

Rules for tg sweeps:

- Keep `tg32` and `tg128` together for MoE decode rollout decisions.
- For the current repo default, MoE decode already includes concurrent GDN front
  projections; use `QWEN_DECODE_MOE_CONCURRENT_GDN=0` for the serial fallback A/B.
- Pair `tg` with `decode-window --target-ctx 4096 --window 32` when the question
  is whether a decode win survives longer context, not just empty-KV decode.

### llama.cpp baselines

For apples-to-apples external baselines, keep two rules straight:

- Use `llama-bench` for pure prompt/decode phase numbers.
- Use `llama-cli` for user-facing prompt+generate checks, but force single-turn
  exit with `-st` / `--single-turn`.

Important:

- `llama-cli` is interactive by default even when `-p` is provided.
- Do not rely on `-no-cnv` / `--no-conversation` with `llama-cli`; current builds
  reject it and tell you to use `llama-completion` instead.
- For MoE Metal attribution, record llama.cpp's startup line `has tensor = ...`.
  On the current M4 Max, `has tensor = false`; forcing
  `GGML_METAL_TENSOR_ENABLE=1` does not make the Metal4 tensor path live.
- Use `~/code/gguf/target/debug/gguf <model> --tensors` when you need quick GGUF
  tensor-name/type confirmation, for example whether a model has separate
  `ffn_gate_exps` / `ffn_up_exps` or a fused `ffn_gate_up_exps` tensor.
- The bounded `llama-cli` shape for this repo is:

```sh
PROMPT="$(python - <<'PY'
print(('The quick brown fox jumps over the lazy dog. ' * 32).strip())
PY
)"

~/code/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" \
  -st \
  --temp 0 \
  --no-warmup \
  --no-display-prompt \
  -n 64 \
  -p "$PROMPT"
```

This prints the prompt / generation throughput summary and then exits.

### Hidden packed-attention micros

`qwen-bench attn-prefill-micro` is useful for packed-attention shape triage, but
it is **not** a ship gate by itself.

Rules:

- Warm both baseline and packed variants before timing; first-use pipeline
  compilation can swamp the real kernel signal.
- Treat body-only wins as hypothesis generators only. We already have a concrete
  counterexample where packed body `NWG=32` beat `64` at multiple contexts in the
  microbench, then regressed badly in full long-prompt prefill.
- Prefer end-to-end `pp` confirmation or a more faithful one-layer stack micro
  before promoting any packed main-pass knob.

## Allocation and leak profiling

Record Allocations when the question is heap/VM churn or unexpected allocation
categories:

```sh
: "${MODEL:?set MODEL to a GGUF}"
TRACE="target/profiles/alloc-decode-$(date +%Y%m%d-%H%M%S).trace"

xcrun xctrace record --no-prompt \
  --template "Allocations" \
  --time-limit 30s \
  --output "$TRACE" \
  --target-stdout - \
  --launch -- ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 64 --no-warmup
```

Summarize aggregate allocation categories:

```sh
uv run python - <<'PY'
import os, subprocess, xml.etree.ElementTree as ET
trace = os.environ.get('TRACE', 'target/profiles/alloc-decode.trace')
xp = '/trace-toc/run[@number="1"]/tracks/track[@name="Allocations"]/details/detail[@name="Statistics"]'
xml = subprocess.check_output(['xcrun', 'xctrace', 'export', '--input', trace, '--xpath', xp])
root = ET.fromstring(xml)
rows = []
for r in root.iter('row'):
    rows.append({k: int(v) if v.isdigit() else v for k, v in r.attrib.items()})
for r in sorted(rows, key=lambda x: x.get('total-bytes', 0), reverse=True)[:12]:
    print(f"{r.get('total-bytes', 0)/1024/1024:9.1f} MiB total | "
          f"{r.get('transient-bytes', 0)/1024/1024:9.1f} transient | "
          f"{r.get('persistent-bytes', 0)/1024/1024:7.2f} persistent | "
          f"events {r.get('count-events', 0):8} | {r.get('category')}")
PY
```

Run a leak sanity check:

```sh
: "${MODEL:?set MODEL to a GGUF}"
leaks --atExit -- ./target/release/qwen-bench decode \
  -m "$MODEL" --tokens 4 --no-warmup
```

Recommended repo-native addition:

- Add a feature-gated global allocation counter or `dhat-rs` mode for
  steady-state decode. Instruments is good at broad churn, but a code-level
  counter can enforce explicit per-token allocation budgets in tests/benches.

## Criterion kernel benches

Criterion output is already machine-readable. Parse estimates with:

```sh
uv run python - <<'PY'
import json, pathlib
root = pathlib.Path('target/criterion')
for p in sorted(root.glob('**/new/estimates.json')):
    data = json.load(open(p))
    mean_ns = data['mean']['point_estimate']
    parts = p.relative_to(root).parts
    if len(parts) >= 4:
        print(f"{parts[0]:16} {parts[1]:24} {parts[2]:14} {mean_ns/1e6:9.3f} ms")
PY
```

Use this for kernel-level regressions. Do not use it as the only guide for
end-to-end decode headroom; command-buffer cadence, queue gaps, readback, and
speculative decode behavior matter.

## Cargo flamegraph

`cargo flamegraph` is useful for human visual inspection, but less
agent-friendly than `xctrace` plus `ztrace` because the SVG still needs
secondary parsing.

```sh
: "${MODEL:?set MODEL to a GGUF}"
cargo flamegraph --profile release \
  -p qwen-cli --bin qwen-bench \
  -o target/profiles/flamegraph-decode.svg -- \
  decode -m "$MODEL" --tokens 16 --no-warmup
```

## Symbols and debug info

The release profile uses line tables, which is usually enough for `ztrace` and
Instruments to show useful Rust frames. If profiles show mostly addresses,
`<unknown>`, or misleading system-library dominance, fix symbol quality before
inferring a bottleneck.

Options:

- Build with the bench profile, which is release-like but has full debug info:
  `cargo build --profile bench -p qwen-cli --bin qwen-bench`.
- Add frame pointers for deeper native stacks when needed:
  `RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile bench -p qwen-cli --bin qwen-bench`.
- The built-in `bench` profile still writes the binary under `target/release`;
  rebuild with the desired profile before launching `target/release/qwen-bench`.

## uniprof

`uniprof` is an optional general-purpose agent interface, especially for
non-Rust or mixed-runtime tools. It uses Instruments for native macOS binaries.

```sh
: "${MODEL:?set MODEL to a GGUF}"
uniprof record --mode host \
  -o target/profiles/uniprof-decode.json -- \
  ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 64 --no-warmup

uniprof analyze target/profiles/uniprof-decode.json --threshold 0.5
```

Keep it as a fallback or cross-runtime profiler, not the primary path for this
repo's Rust/Metal decode workload.

## xctrace troubleshooting

- Create output directories first: `mkdir -p target/profiles`.
- Use unique trace names or delete old traces; `xctrace` requires
  `--append-run` for an existing `.trace` bundle.
- Verify templates with `xcrun xctrace list templates` before recording.
- If recording/export fails, check Developer Tools permissions, Xcode first-run
  state, writable temp/cache directories, and available disk space.
- If a trace has no useful samples, increase token count/duration or ensure the
  workload runs during the recording window.
- If a Metal parser returns zero rows, inspect `xcrun xctrace export --toc` and
  confirm process names; WindowServer and prior runs can also own GPU intervals.

## What to add to the repo next

The strongest autonomous profiling setup would be repo-native:

1. `scripts/profile/trace-cpu` wrapping `xctrace` + `ztrace`.
2. `scripts/profile/trace-metal.py` for Metal table summaries.
3. `scripts/profile/trace-alloc.py` for Allocations summaries.
4. `scripts/profile/bench-compare.py` wrapping `hyperfine` output.
5. Optional `--profile-metal-counters` using `MTLCounterSampleBuffer`.
6. Optional feature-gated allocation counter or `dhat-rs` profile.
7. Optional `MTLCaptureManager` capture flag for `.gputrace` snapshots.

Once (1)-(4) exist and have stable machine-readable summaries, extract a skill
that simply invokes those scripts and reports concise results.

Suggested script output contracts:

- Use JSON by default for automation and a compact text summary for humans.
- Include units in field names (`*_ms`, `*_bytes`, `*_count`).
- Include trace metadata: command, model path, process name, template, Xcode
  version, OS version, git SHA, and wall-clock duration.
- Exit nonzero when the trace cannot be parsed or expected tables are absent;
  warn, but do not fail, when optional tables are missing.
- Keep thresholds explicit and externally configurable; do not bake local trial
  numbers into pass/fail gates.

## References

- `agent-scripts` Instruments skill:
  https://github.com/steipete/agent-scripts/blob/main/skills/instruments-profiling/SKILL.md
- `xtrace-skill`:
  https://github.com/Kr1sso/xtrace-skill/blob/main/SKILL.md
- `uniprof`:
  https://www.uniprof.sh/
- `ztrace`:
  https://github.com/frr149/ztrace
- `samply`:
  https://github.com/mstange/samply

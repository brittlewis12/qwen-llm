# Documentation index

## Using the engine

- [SERVE](SERVE.md): `qwen serve` contract, endpoints, limits, and gate records.
- [serve-opencode](serve-opencode.md): connect OpenCode to a local serve process.
- [CLI-UX](CLI-UX.md): command-line design evidence and decisions.
- [ENV](ENV.md): generated snapshot of the environment-knob reference.
- [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES.md): licenses and notices for third-party components.

## Lens

- [LENS-RUN](LENS-RUN.md): run inference with live workspace-lens readouts.
- [LENS-WEB](LENS-WEB.md): Lens Workbench capabilities and interfaces.
- [LENS-DATA-CONTRACT](LENS-DATA-CONTRACT.md): producer and consumer contract for imported linear-lens data.
- [LENS-OBSERVER-CONTROLS](LENS-OBSERVER-CONTROLS.md): native observer readouts and controls.

## Performance

- [PERF-ROADMAP](PERF-ROADMAP.md): current shipped state, evidence index, and closed questions.
- [PERF-LOG](PERF-LOG.md): append-only record of performance measurements and decisions.
- [BENCH](BENCH.md): benchmark commands, evidence, and known measurement gaps.
- [REQUEST-TIMING](REQUEST-TIMING.md): what each `qwen run` timing field measures in each family.
- [PERF-TOOLS](PERF-TOOLS.md): profiling and performance instrumentation guide.
- [PERF-TOOLS-SETUP](PERF-TOOLS-SETUP.md): setup instructions for profiling tools.
- [APPLE-GPU-OPTIMIZATION](APPLE-GPU-OPTIMIZATION.md): Metal optimization guidance for Apple GPUs.
- [bench/](bench/): dated measurement packets use `YYYY-MM-DD[-HHMM]-slug`; packet directories contain measurement artifacts and README files where provided.

## Current reference

- [DEEPSEEK-V41-BRINGUP](DEEPSEEK-V41-BRINGUP.md): current V4.1 bring-up requirements and evidence.
- [INFERENCE-GRAPH](INFERENCE-GRAPH.md): current semantic map of Qwen hybrid inference paths.

## Design records for shipped features

- [DEEPSEEK-V41-STRATEGY](DEEPSEEK-V41-STRATEGY.md): concise V4.1 requirement and design reference.

## Lens evidence packets

- [Crossed-task study](bench/2026-09-08-lens-crossed-task/README.md)
- [Depth and order study](bench/2026-09-08-lens-depth-and-order/README.md)
- [Comparative physiology](bench/2026-10-05-lens-comparative-physiology/README.md)
- [Construal corpus](bench/2026-10-12-lens-construal-corpus/README.md)
- [RepoRef study](bench/2026-09-08-lens-reporef/README.md)

## Archive and source references

- [PLAN](archive/PLAN-2026-09-03.md): historical architecture plan.
- [K2-HORIZON-PLAN](archive/K2-HORIZON-PLAN-2026-10-08.md): completed implementation map.
- [K2-HORIZON-REVIEW](archive/K2-HORIZON-REVIEW-2026-10-05.md) and [review request](archive/K2-HORIZON-REVIEW-REQUEST-2026-10-08.md).
- [LENS-MVP](archive/LENS-MVP-2026-09-08.md): completed capability and qualification record.
- [H4-MTP](archive/H4-MTP-2026-05-03.md) and [H5-DFLASH](archive/H5-DFLASH-2026-10-05.md): shipped decode implementation records; benchmarks run through `qwen-bench`.
- [DEEPSEEK-V4-STRATEGY](archive/DEEPSEEK-V4-STRATEGY-2026-10-19.md): completed implementation history.
- [IQ-QUANT-NATIVE-CAPACITY](archive/IQ-QUANT-NATIVE-CAPACITY-2026-10-12.md): completed coverage and promotion record.
- [GLM53-FLASH-PLAN](archive/GLM53-FLASH-PLAN-2026-10-12.md): completed bring-up plan.
- [dsv4-paper](dsv4-paper.md): paper citation and repository implementation notes.
- [archive/](archive/): retained historical design and experiment records.
- [experiments/](experiments/): retained Lens experiment inputs and rubrics.

# Documentation index

## Using the engine

- [SERVE](SERVE.md): `qwen serve` contract, endpoints, limits, and gate records.
- [serve-opencode](serve-opencode.md): connect OpenCode to a local serve process.
- [CLI-UX](CLI-UX.md): command-line design evidence and decisions.
- [ENV](ENV.md): generated snapshot of the environment-knob reference.
- [K2-HORIZON-PLAN](K2-HORIZON-PLAN.md): current K2 Horizon implementation map and evidence.
- [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES.md): licenses and notices for third-party components.

## Lens

- [LENS-RUN](LENS-RUN.md): run inference with live workspace-lens readouts.
- [LENS-WEB](LENS-WEB.md): Lens Workbench capabilities and interfaces.
- [LENS-DATA-CONTRACT](LENS-DATA-CONTRACT.md): producer and consumer contract for imported linear-lens data.
- [LENS-OBSERVER-CONTROLS](LENS-OBSERVER-CONTROLS.md): native observer readouts and controls.

## Performance

- [PERF-ROADMAP](PERF-ROADMAP.md): ranked performance work across model families.
- [PERF-LOG](PERF-LOG.md): append-only record of performance measurements and decisions.
- [BENCH](BENCH.md): benchmark commands, evidence, and known measurement gaps.
- [REQUEST-TIMING](REQUEST-TIMING.md): what each `qwen run` timing field measures in each family.
- [PERF-TOOLS](PERF-TOOLS.md): profiling and performance instrumentation guide.
- [PERF-TOOLS-SETUP](PERF-TOOLS-SETUP.md): setup instructions for profiling tools.
- [APPLE-GPU-OPTIMIZATION](APPLE-GPU-OPTIMIZATION.md): Metal optimization guidance for Apple GPUs.
- [bench/](bench/): dated measurement packets use `YYYY-MM-DD[-HHMM]-slug`; packet directories contain measurement artifacts and README files where provided.

## Active plans

- [GLM53-FLASH-PLAN](GLM53-FLASH-PLAN.md): implementation map for the GLM-5.3-Flash family.
- [IQ-QUANT-NATIVE-CAPACITY](IQ-QUANT-NATIVE-CAPACITY.md): capacity and format-coverage work queue for native IQ quant support.
- [LENS-TRAJECTORY-OBSERVATORY](LENS-TRAJECTORY-OBSERVATORY.md): living tracker for trajectory-observatory work.

## Design records for shipped features

- [H4-MTP](H4-MTP.md): design and implementation record for Qwen3.6 MTP speculative decoding.
- [H5-DFLASH](H5-DFLASH.md): design and implementation record for the DFlash speculative-decoding path.
- [DEEPSEEK-V4-STRATEGY](DEEPSEEK-V4-STRATEGY.md): architecture and implementation record for DeepSeek V4 support.

## Research notes

- [DEEPSEEK-V41-STRATEGY](DEEPSEEK-V41-STRATEGY.md): source-verified assessment of DeepSeek V4.1 architecture.
- [H6-VISION](H6-VISION.md): unstarted design for native Qwen vision input.
- [LENS-CROSSED-TASK](LENS-CROSSED-TASK.md): exploratory study of transfer across crossed tasks.
- [LENS-DEPTH-AND-ORDER](LENS-DEPTH-AND-ORDER.md): exploratory findings on layer depth and sequence order.
- [LENS-COMPARATIVE-PHYSIOLOGY](LENS-COMPARATIVE-PHYSIOLOGY.md): observations about context, reader, and response boundaries.
- [LENS-CONSTRUAL-CORPUS](LENS-CONSTRUAL-CORPUS.md): research audit and corpus design for construal plurality.
- [LENS-REPOREF](LENS-REPOREF.md): research audit of evidence-seeking action and RepoRef.

## History

- [PLAN](PLAN.md): historical May 2026 architecture plan.
- [INFERENCE-GRAPH](INFERENCE-GRAPH.md): semantic map of Qwen hybrid inference paths.
- [K2-HORIZON-REVIEW](K2-HORIZON-REVIEW.md): archived adversarial review and findings for K2 Horizon.
- [K2-HORIZON-REVIEW-REQUEST](K2-HORIZON-REVIEW-REQUEST.md): archived brief for the K2 Horizon review.
- [LENS-MVP](LENS-MVP.md): archived capability and qualification record with a frozen gate.
- [dsv4-paper](dsv4-paper.md): third-party DeepSeek V4 paper text.
- [archive/](archive/): retained historical design and experiment records.
- [experiments/](experiments/): retained Lens experiment inputs and rubrics.

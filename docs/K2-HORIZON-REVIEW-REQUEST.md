# Adversarial review request: K2 Horizon milestones

You are reviewing a proposed implementation, not implementing it. Read
`docs/K2-HORIZON-PLAN.md` and inspect relevant local source before judging reuse
claims. Worktree is a fresh branch from main at `4d8716ab`.

## Hard boundaries

Read-only review. Do not edit files, build/run inference, execute GPU tests,
load models, download weights, install tools, restart/stop servers, commit, or
push. The shared machine is in use. An idle device or lock is not permission.
No Task/Todo tools. Source reads/searches and read-only git operations are fine.
Use gh if checking GitHub sources. No need to repeat broad web research.

## User decisions (not open questions)

- Dense K2 Horizon 7B first, directly; compatible intermediate checkpoints are
  first-class in run/serve/bench/forward-only lens. MoVA later.
- Native Rust tokenizer. HF is an independent conformance oracle; do not make
  updating llama.cpp bindings a production prerequisite.
- Lens capture/readout/imported CUDA-fitted assets/forward interventions, but
  no local fitting or transformer backward/VJP implementation.
- Checkpoint conversion is not this project's task. Do not treat lack of a fleet
  of converted files as reason to block model-free engineering; identify which
  later claims require real artifacts and live validation.
- Interested in sophisticated KV representation savings without naive history
  deletion. F16 control, Q8 first adaptation, further research independently.
- No existing unrelated paths should regress, and no generic framework rewrite
  should be required merely to add this family.

## Questions to attack

1. Are M0-M4 a sensible minimum complete vertical slice? Is any milestone hiding
   a large prerequisite or postponing a checkpoint/lens requirement too late?
2. Where does this repo's existing loader/tokenizer/runtime/lens coupling make
   the proposed reuse misleading? Give exact local file references and bounded
   fixes, not generic advice. Especially inspect FFI-bound frontend interfaces,
   lens fitting assumptions, and whether completion-style serving exists.
3. What normalization, RoPE, cache-rounding, packed/serial, or stage-dependent
   tokenizer assumptions could produce plausible but wrong inference/research?
4. Is artifact/profile/request-capability separation sufficient to admit
   intermediate checkpoints safely without a final-only hash whitelist?
5. What minimal independent fixture/oracle ladder should M0/M1/M2 use? Avoid
   calling self-generated outputs independent or requiring bitwise agreement
   across legitimately different BF16/F32 reduction topologies.
6. Which existing GQA kernels actually fit head-dim128/group4? Is Q8 adaptation
   bounded? Check compact cache lifetime, admission, and snapshot implications.
   Arithmetic: 36 layers * 2(K,V) * 8 KV heads * 128 * 2 bytes is144KiB/token;
   this is36layers, not32. No measured speed or full-model memory claims yet.
7. What is the smallest useful forward-only lens contract for CUDA-fitted
   assets, including matrix orientation/site semantics and explicit transfer?
8. Which tasks should move earlier, run in parallel, or be deferred? Identify
   likely scope creep and concrete stop/go conditions for MoVA and advanced KV.

## Required response

Return:
- Verdict: proceed / revise before implementation, with reasons.
- Ranked P0/P1/P2 findings. Label verified source facts versus hypotheses.
- Revised milestone ordering only if needed, with observable exit criteria.
- First bounded implementation packet, executable without weights/GPU, with
  proposed touched files and targeted CPU-only tests.
- Remaining decisions that truly require user input (do not invent blockers).

Prefer concrete objections and corrections over agreement or restating the plan.

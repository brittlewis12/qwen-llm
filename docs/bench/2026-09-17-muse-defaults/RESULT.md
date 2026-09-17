# Muse qualified defaults: source delivered, GPU validation incomplete

Source `d4bb5703` makes already-qualified optimized generation math automatic for
Muse Q8_0 on unified Apple M4 Max. No shader, numerical gate, context limit, extra
scratch, or new switch is introduced. Both run and serve preserve independent
strict0 rollbacks; unset/1 allows qualified execution, not force-on. Unsupported
profiles/devices resolve to reference math. All diagnostics use effective options.

Lens/fit consumers and the existing exact-labelled muse-request benchmark explicitly
use load_reference(), preserving their arithmetic/artifact contracts. Ordinary
library load() follows generation defaults. Historical reference backend tests
retain their original arithmetic rather than silently changing the baseline.

## Initial validation at default integration

- CPU128-case artifact/device/unified/independent-rollback resolver PASS.
- Three focused backend CPU tests PASS, including strict shared run/serve parsing.
- All-target qwen-cli check and release qwen build PASS without warnings before
  rebase; resolver/backend tests and all-target check also PASS after rebasing onto
  concurrent main CLI admission work. First combined build command hit its240s
  tool timeout during CLI compilation; rerun completed, no GPU had started.
- Frozen native test compiles. Attempt01 FAIL at the production lease in0.00s,
  before GGUF/tokenizer/Metal initialization, allocation or forward. Owner is the
  existing Muse server PID92618 on8737, opencode session
  `ses_078e40a1affeVT8UiBF3yb1AFr`. No process stopped, no lock or wired-gate bypass.
  This is **BLOCKED**, not a numerical/performance failure or a passing GPU check.
- New default/explicit native and CLI identity packets remain pending GPU ownership.
  No new HTTP packet or speed measurement is claimed. Existing numerical, CLI and
  serving lifecycle evidence remains the promotion basis, not this blocked packet.

Raw logs: `target/profiles/2026-09-17-muse-defaults-{cpu,cli-cpu-02,check,build,
cpu-rebased,cli-cpu-rebased,check-rebased,delivery-01}.log` in the optimization worktree.

## Attempt02 FAIL and source-only alignment repair

After the old Muse PID exited, its exact process/listener checks and live lock-holder
check were empty. A narrow direct-executable process filter also found none. The
same frozen test acquired the production lease and passed the real wired gate,
then aborted with Metal API validation enabled:

`length(24900) must be a multiple of 16 bytes` (SIGABRT).

This is a real validation FAIL, not a pass, timing result or numerical comparison.
No lane completed. The unchanged scalar-prefill tail at6225 positions delegates to
shared materialized F16 attention, which requested6225*4 bytes without aligning
each dynamic threadgroup allocation. This affects original scalar math as well as
the scalar tail of optimized prefill; it is not new split-attention arithmetic.

User correction exposed a Flash-Next server launch/readiness shell via the requested
`ps aux | rg qwen`. The narrow direct-process filter missed that activity. Absence
of an old PID/listener/live lock holder was insufficient operational coordination
after the user said the server was in use. No server process was stopped/restarted,
but the attempted GPU packet should not have been launched in that window. All
further GPU work is deferred; do not infer permission from an apparently idle lock.

Source `e8768272` shares checked16-byte rounding across scalar F16 attention and
Muse materialized packed attention. The failing24900-byte score allocation becomes
24912; packed admission prices the aligned sizes. Kernel arguments, logical visible
positions, loops, shaders and7168-position materialized limit do not change. This
is an API-contract repair, not a performance optimization or gate relaxation.

CPU test PASS over all positions0..7168 and SIMDgroup counts1..32, including explicit
6225/7168 boundaries and multiplication/alignment overflow. All-target CLI check
PASS without warnings. cx source review found no blocker; its reply accidentally
called the native failure attempt01, but the preserved failure is **attempt02**.
No GPU repair rerun or CLI replay has happened, so default-delivery remains open.

Raw: `target/profiles/2026-09-17-muse-defaults-delivery-02.log`,
`2026-09-17-muse-alignment-cpu.log`, `2026-09-17-muse-alignment-check.log`.
Main release binary was rebuilt at73d38909 before this repair; subsequent repair
delivery must not be confused with a running server adopting a rebuilt executable.

## Evidence and leverage

Prior controlled split decode:6229 tokens8.663->15.938 consumed forwards/s;
32K3.955->14.722. These are warm phase comparisons, not promises for arbitrary
sampled generation or complete-request speed. Prefill also becomes automatic;
the scalar_tail label still describes prompt packing, not a decode fallback.

After selection is fixed, use the **optimized** profile:32K generated FFN44.200ms
and64.96% versus attention12.43%. Decode FFN removable work is next; neither small
gate values nor post-up product observations establish avoidable work. Payload
rate is not measured DRAM bandwidth. Late32K packed attention49.91% leads FFN38.43%;
at8K **packed** FFN is57.0%. Different phases require different next candidates.
Cold-load variance and prefix reuse remain separate. No upper benchmark cliff is
reintroduced; whole-model131K or sampled-exact equivalence is not established.

cx Luna01a0b18d-c63c reviewed policy, implementation and frozen checks. Its lens,
effective-telemetry and stale-documentation findings were addressed. Its later map
mistakenly reverted to incumbent attention/GEMV bottlenecks; source-specific phase
numbers above take precedence. Final review found no remaining source blocker,
while retaining the pending GPU delivery gate.

Authority: `../2026-09-08-muse-math/RESULT.md`,
`../2026-09-08-muse-long-context/RESULT.md`,
`../2026-09-09-muse-serve-math/RESULT.md`. All prior FAIL/HOLD/KILL results retain scope.

# Muse qualified defaults: source delivered, GPU check blocked

Source `d4bb5703` makes already-qualified optimized generation math automatic for
Muse Q8_0 on unified Apple M4 Max. No shader, numerical gate, context limit, extra
scratch, or new switch is introduced. Both run and serve preserve independent
strict0 rollbacks; unset/1 allows qualified execution, not force-on. Unsupported
profiles/devices resolve to reference math. All diagnostics use effective options.

Lens/fit consumers and the existing exact-labelled muse-request benchmark explicitly
use load_reference(), preserving their arithmetic/artifact contracts. Ordinary
library load() follows generation defaults. Historical reference backend tests
retain their original arithmetic rather than silently changing the baseline.

## Validation

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

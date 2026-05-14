# Performance tool setup

This file is the provisioning companion to `docs/PERF-TOOLS.md`. Use it only
when intentionally preparing a local profiling environment. A normal profiling
session should not install or update global tools.

## Policy

- Prefer the tool manager already used by the project or shell environment.
- Keep profiling commands and setup commands separate. If a tool is missing
  during profiling, report that fact and use an available fallback.
- Do not assume a specific shell integration or PATH layout.
- Avoid `brew` unless there is no better project-compatible option.
- Use unpinned install/update commands for ordinary local setup unless a
  project-local manifest or reproducibility task explicitly calls for pins.

## Tool roles

| Tool | Role | Setup owner |
| --- | --- | --- |
| `xcrun xctrace` | Headless Instruments recordings | Xcode / Command Line Tools |
| `hyperfine` | Command-level before/after timing | Project/user tool manager |
| `ztrace` | Compact summaries for `xctrace` CPU traces | Project/user tool manager |
| `samply` | CPU/off-CPU profiles on macOS | Project/user tool manager |
| `uniprof` | Optional unified profiler interface | Project/user tool manager |
| `cargo-flamegraph` | Optional human-readable flamegraphs | Project/user tool manager |

## Preflight

These checks are safe in a profiling session because they do not mutate global
state:

```sh
xcrun xctrace list templates >/dev/null
hyperfine --version >/dev/null
ztrace --help >/dev/null
samply record --help >/dev/null
uniprof --version >/dev/null
```

If one of the optional tools is unavailable, choose a fallback from
`docs/PERF-TOOLS.md` instead of installing it mid-profile.

## Intentional setup

Run setup commands only when the task is explicitly to prepare or refresh the
profiling toolbox. Do not run them as part of normal profiling.

```sh
bun i -g uniprof
uv tool install git+https://github.com/frr149/ztrace --force
cargo install --git https://github.com/mstange/samply --locked --force samply
```

Use pins only when reproducing a known environment or building a project-local
tool manifest/wrapper. If you record pins, label them as last-known-working
observations, not mandatory update policy.

When doing setup work deliberately:

- Inspect the project config and active shell environment first.
- Prefer project-local manifests over global installs when practical.

# Bounded kill check of ordinary command-buffer wiring (2026-10-05)

PERF-ROADMAP 2026-10-05 #1 protocol, cx-reviewed. Commits `f400e704` and
`02a41b39`; observer `scripts/reference/residency_kill_check.py`; child
`qwen-bench residency-kill-child`.

Setup:

- Detached observer: launched with `setsid`; a deliberately timed-out tool
  call left a detached probe process alive (reparented to launchd).
- Quiet host: no engine processes, pressure level 1, baseline wired 5.00
  GiB with 0.25 GB spread.
- Disposable buffer: a 1 GiB random file, mapped read-only and wrapped in
  a no-copy Metal buffer. No residency set, no mlock, no sysctl changes,
  no gate bypass.
- The child holds the production lease. The 384 MiB tolerance is well below
  the ~1.2 GiB increase.

| Case | Wired increase | Kill | Excess after kill (MiB) at 1 / 5 / 15 / 30 / 60 s |
|---|---:|---|---|
| normal (pulses, exits) | 1.17 GiB | none (exit 0) | -50 / 109 / 23 / -50 / -46 |
| stopped (pulses, idles 4 s: -0.09 GiB) | 0.70 GiB | SIGKILL | 110 / 22 / 28 / 22 / 29 |
| pulse (killed between 500 ms pulses) | 1.17 GiB | SIGKILL | -59 / 23 / 23 / 25 / 25 |
| execute (killed 0.3 s into a ~1.9 s command, in flight) | 1.20 GiB | SIGKILL | 765 / 196 / 107 / 114 / 189 |

Verdict: recovered in every case. Wired memory read 4.94 GiB (below the
5.00 GiB baseline) a few minutes later; the execute case's 107-189 MiB
residual sat within the baseline's 0.25 GB spread and was gone on re-check.

Ordinary command-buffer wiring, both between keep-alive pulses and with a
command in flight, was reclaimed after SIGKILL at 1 GiB. This is lifecycle
evidence at 1 GiB, not proof at 115 GiB, and it says nothing about
residency sets, which stay closed.

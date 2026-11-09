# Historical profile scripts

This archive contains 50 profile scripts from the v0.593–v0.660 sequence. They
are retained as evidence and are not runnable from this new location as-is:
they import one another by module name and resolve working files relative to
their original repository paths. The script contents have not been modified.

To replay one, check out commit
`0f75a4bdfbadfe6d177397e0375742a44e0ff330` in a detached checkout or worktree.
That is the last commit containing these scripts at `scripts/profile/`. Run the
chosen `scripts/profile/<name>.py` from the repository root in that checkout.

The v0.636/v0.637 checkpoint-floor runners are historical packets. Their ignored
entry point was retired in the 2026-11-09 ignored-test cleanup. Reproduce with
the packet’s frozen source commit (also present at pre-cleanup `d3087f8d3`),
rather than the current tree.

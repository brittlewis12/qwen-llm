# V4.1 Mapping Ownership And Combined Host Admission

## Failure And Scope

Repeated cold 8K qualification of stock V4.1 Flash 1017 UD-IQ3_XXS on
256 GiB ended in a watchdog timeout. Its 217.414 GiB resident-window plan
included five private EOF windows totaling 156.059 GiB, solely to expose
30,496 padding bytes. Engram was separate, not inside those windows.

`map_copy(...).make_read_only()` retains private/COW ownership. Read-only
protection does not turn it into a shared file mapping. The recorded raw
process-limit signal was zero and the old admission omitted that constraint,
so CPU estimates did not constrain the accepted plan.

The crash process's approximately 139 GiB-equivalent page-in and COW counters
are cumulative events, not distinct resident totals or additive memory charges.
Neither those counters nor the bounded experiments establish the exact M5
watchdog mechanism. Likewise, weight bytes and global wired memory cannot
simply be added: their physical pages can overlap.

## Intrinsic Repair

- Pure partial-EOF windows use read-only shared mappings. The final partial
  page supplies its normal zero tail; files are not padded, mutated or truncated.
  Tensor payloads still must fit inside raw EOF.
- A genuine Engram boundary clear remains private. Its ownership carries the
  entire potentially shadowed span, not merely the few CPU-cleared bytes.
- The loader and public `resident_storage_plan` share segment, window and
  ownership planning. Planning reads metadata/descriptors, not tensor payloads,
  creates no additional mappings and does not probe hardware or import buffers.
  Opening GGUF itself still retains ordinary shared shard mappings.
- Whole-load admission happens before window realization/import. Each import
  also refreshes admission under the allocation transaction and reserves its
  accounting debt before calling Metal.
- External accounting follows the actual Metal resource's deallocator, not a
  Rust tensor wrapper. Buffer retains preserve both mapping and debt. Failure
  rollback and final release retire each charge exactly once.

## One Combined Budget

CPU demand, Metal demand and reserve must fit a validated host budget. A raw
zero Mach process limit can omit only the process constraint, never the host
constraint. Missing or invalid telemetry refuses admission.

The current process's device/import ownership plus private-shadow exposure is
capped at 80% of physical RAM. A second bound uses fresh conservative available
memory, retaining the other 20% for the host. Availability credits free pages
and a lower bound on inactive non-anonymous pages; it does not guess that active
file cache, speculative and purgeable counters are independent reclaimable pools.

Raw Metal counters remain raw. The ledger adds uncovered external import lengths
and full private shadows. It credits `allocatedSize` only when the observed
device allocation increase corroborates that coverage; otherwise the external
debt remains. This corrects API coverage, not physical-residency measurement.

Post-import validation accepts exact exhaustion but checks the underlying
inequalities to reject overshoot; a saturated zero remaining-byte counter alone
cannot distinguish those cases. CPU snapshot relief retains typed causes:
availability/process shortages may evict unpinned snapshots and recheck once;
an immutable device/host-ceiling shortage cannot be repaired that way. GLM
capacity advice uses the same host bound.

These are conservative admission bounds, not OS reservations or a guarantee
against driver, kernel or hardware failure. Other engines and old binaries are
not retroactively constrained by this policy.

## Bounded Hardware Evidence

Source checkpoints: shared EOF `117cb211`, combined host/import policy
`fb8882d3`, integrated storage planning `007bb136` / `482298d1`.

The release-mode import fixture plans 26,411,008 bytes of explicit resources,
below 32 MiB. Context, pipeline and driver overhead are separate. Cases cover
shared aligned, shared partial EOF and the old private EOF control, sequentially
dropping and draining each case. Phases separate import, obtaining `contents`,
explicit CPU reads, blit, compute and final release.

| Observation | M4 Max | M5 Ultra |
| --- | ---: | ---: |
| Partial-EOF imported span | 8,404,992 bytes | 8,404,992 bytes |
| Source `allocatedSize` and import allocation delta | span length | span length |
| Shared EOF COW events across the case | 0 | 0 |
| Old private EOF COW events at blit | 513 x 16 KiB | 0 |
| Old private EOF footprint increase at explicit CPU read | 0 | 8,421,400 bytes |
| Shared EOF footprint increase at explicit CPU read | 0 | 16,408 bytes |

The first shared-aligned case records two COW events at first GPU use on both
devices; those process-wide events are not attributed to mapped source pages.
The M5 footprint observation is not proof that distinct physical pages were
copied. It demonstrates why a zero COW-event delta is not a zero memory charge.
No larger probe is needed to justify eliminating unnecessary private ownership.

Both hardware classes pass byte-exact payload/padding/readback checks, guarded
copy bounds, unchanged source identity/content, retained mapping lifetime and
one final deallocator callback. A separate two-page private import proves that
the full shadow debt survives dropping the backing wrapper while a Metal buffer
retain is live, then retires on final resource release.

Verification also includes 34 pure admission/planning tests on each hardware
class, 1,493 CLI library/binary tests, 21 pilot-controller tests, and release
checks of all affected library, CLI, example and integration targets.

## Metadata-Only Artifact Selection

The shared loader planner, with the recorded 16 KiB page geometry, reports:

| Artifact | Resident views | Private shadow | Windows | Shared EOF windows |
| --- | ---: | ---: | ---: | ---: |
| Maya-S | 165.950 GiB | 0 | 44 | 4 |
| UD-Q2_K_XL | 185.907 GiB | 0 | 45 | 5 |
| UD-IQ3_XXS | 217.414 GiB | 0 | 45 | 5 |

These audits do not import weights. The 256 GiB ownership ceiling is 204.8 GiB
before runtime demand, so full IQ3 is refused even after its private EOF exposure
is removed. Shortening a prompt to 2K does not shrink resident weights. Maya-S
offers more room for a fitting CED pilot, not a runtime-fit or quality-substitute
claim; runtime CPU/Metal bounds and fresh host availability must still admit it.

Execution-policy comparisons should reuse one resident model with fresh KV
states, not repeatedly cold-import the entire model. Start with a fitting pilot;
authorize timing repetitions only after stable native samples and healthy host
telemetry. The secondary pilot controller stops scheduling on unstable or bad
evidence and signals only its owned child. Cancellation is not a recovery plan
for an already wedged VM/driver. No CED speedup is established by this repair.

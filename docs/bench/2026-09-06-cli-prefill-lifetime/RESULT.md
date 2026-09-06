# Serial CLI prefill lifetime: resource KEEP, broader scope HOLD

## Surviving change

`c3ba57e4` retains only early destruction of prefill scratch in the ordinary
serial CLI decode branch. DFlash (including shadow prefill), prompt lookup,
and serve retain their original lifetimes. Allocation planning, arithmetic,
sequence capacity, sampler, and checkpoint rules do not change. Existing
`after_prefill` telemetry remains pre-drop; the final post-state-drop sample
keeps its original meaning.

The earlier broad candidate `4d5f8834` released scratch before CLI DFlash/lookup
and served snapshots/decode too. It was narrowed, not promoted wholesale.
Independent read-only review challenged aliasing, command completion, shadow
ordering, admission, sampled peaks, and the final disposition. Review session:
`01a07893-0d8c-7c12-a567-07f5c8c1331e` (Luna via cx), with initial audit,
committed-patch review, and disposition jam. It supports the serial-only scope
and explicitly rejects interpreting allocation samples as true peak or latency.

## Measured resource result

Release baseline `f37b53a9` versus broad candidate `4d5f8834`, A-B-B-A, two
disposable processes per arm per lane. Qwen3.6-27B Q4_K_M, 8,840 input / 64
output tokens, natural technical prefix plus code question. The CLI's supported
raw-file surface consumes identical frozen ChatML bytes with a preclosed think
suffix. This measures the execution path, not the modern templating frontend.

At first stdout flush, ordinary no-spec CLI allocation falls by exactly
**1,364,262,912 bytes** in both orders:

| Phase | Baseline bytes | Candidate bytes |
| --- | ---: | ---: |
| Model ready | 16,808,574,976 | 16,808,574,976 |
| After prefill | 18,942,214,144 | 18,942,214,144 |
| First stdout flush | 18,942,214,144 | 17,577,951,232 |
| Request end before state drop | 18,942,214,144 | 17,577,951,232 |
| After state drop | 16,808,837,120 | 16,808,837,120 |
| Sampled maximum | 18,942,214,144 | 18,942,214,144 |

This exceeds the frozen 256 MiB resource gate. It is a reduction in actual
Metal allocated bytes during generation, **not** RSS, resident memory, a lower
no-spec peak, or improved admission. The initial admission plan is unchanged.
Final-candidate short and long executions reproduce all allocation samples and
token/stop/consumption metadata of the measured serial candidate exactly.

## Wall guard and rejected widening

Median milliseconds across two processes per arm. Positive change is slower.

| Lane / endpoint | Baseline | Broad candidate | Change | Control spread |
| --- | ---: | ---: | ---: | ---: |
| No-spec request TTFT | 38789.038 | 39295.507 | +1.306% | 0.006% |
| No-spec complete request | 41481.897 | 41977.288 | +1.194% | 0.173% |
| No-spec full process | 44905.147 | 44148.528 | -1.685% | 5.566% |
| Spec request TTFT | 37986.605 | 39243.711 | +3.309% | 1.758% |
| Spec complete request | 40724.930 | 41983.767 | +3.091% | 1.718% |
| Spec full process | 43203.078 | 44386.408 | +2.739% | 1.922% |

Serial request wall stays within the frozen 3% resource guard, but is slower:
this is not a latency promotion or strict Pareto improvement. Process-cold
invocation is real; filesystem-cache conditions are not forced cold or symmetric.
No-spec A1 warms the shard for 2470 ms. Its apparent process-wall saving is
therefore confounded as well as above the control-spread gate. No cold-load win.

The matching legacy Qwen3.6 Q8 drafter is active in the spec packet. Broad early
release also removes 1,364,262,912 bytes from first-delivery allocation there,
but complete request wall regresses 3.091%, outside the 3% guard. Do not choose
the favorable process-clock endpoint to override that failure. DFlash early
release is HOLD. Its lower sampled maximum is not a true-peak measurement.

Serve has only an eight-request-per-arm correctness pair, not balanced timing
authority. One code continuation is slower despite using no prefill scratch in
either arm; the pair must not support a latency claim. Serve release is HOLD.
Prompt lookup rejects both local small/Qwen3.6 assets at its historical exact
architecture gate, so that branch is not performance-validated and stays unchanged.

## Correctness and acquisition accounting

- All eight long CLI outputs have identical generated-token SHA-256,
  `60569e8379f5d3a974e7003256586d746fd4c8dd95b0ed6e8ac1ad3a88a8dbc9`,
  64 tokens, token-limit stop, and an unconsumed final token. All stdout hashes
  agree too. Small-model and shadow A/B token/stop gates pass.
- The new ignored keep/drop Metal test compares packed final logits, KV/GDN
  snapshots, next-token logits and persistent state bit-for-bit, and observes
  allocated bytes fall at the drop. It passes separately on 0.8B Q4 and 27B Q4.
  It is not a failure-injection or DFlash-capture-tail proof.
- All 16 served response texts, usage/cache counts, completion statuses and
  reasons agree, including subsequent EOS/token-limit continuations. Those
  checks do not authorize the removed serve optimization.
- The final narrowed candidate passes 372 CLI tests (eight ignored). Final
  small/long serial checks match the measured serial allocation envelopes;
  final shadow matches the original baseline envelope, confirming retention.
  Final validation timings are not pooled into the original balanced packet.
- Failed observer attempts remain: legacy CLI rejects `--no-thinking` before
  acquisition; the repaired observer uses explicit raw prompt files. A helper
  named `inspect.py` initially shadows Python's stdlib and is renamed before
  any spec child starts. Prompt-lookup admission failures remain recorded.
- GPU execution is serial; no residency/wiring or admission bypass is used.
  CLI processes exit normally; owned servers unwind via SIGINT. The first
  independent review call times out and is resumed to obtain its conclusion.

Raw fixtures, manifests, timing JSONL, stdout/stderr, protocols, metric-only
scorers, and final checks remain at `target/profiles/prefill-lifetime/` in the
dedicated worktree. Existing request-timing telemetry and the tracked
`scripts/serve/request_probe.py` provide the engine/HTTP measurements. No new
public benchmark framework or performance switch is added.

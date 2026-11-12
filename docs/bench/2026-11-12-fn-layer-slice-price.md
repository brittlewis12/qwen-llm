# Flash-Next layer-slice price

Fresh three-agent release baseline at `abc3ba1a3`, GSQ-RCO IQ3_S,49152 warm
tokens per agent, real sampling,180 seconds: **127.111 tok/s**, ITL p50/p99/max
**19.880/25.788/1885.943ms**. All180 measured seconds were at pressure1.
Both pre-load and post-warm settling required continuous60 seconds<=1;
measurement/warm-up permitted2 and aborted3. Raw baseline is under
`2026-11-12-v41-layer-slice/protocol-v2/fn-price` on the M5.

The maximum is common to the two decoding peers while agent2 starts turn2:
5479 new tokens after position50486, TTFT4586.9ms. The incumbent uses packed
2048-row commands beyond the QSA dense-to-selected frontier. This episode
therefore has two full commands and a1383-row tail. The stall clears the size
threshold for further pricing. No Flash-Next production slicing code has been
added yet.

`fn_append_price` (perf-ablations) measures the unchanged native forward's
encode CPU, completed GPU, wait and wall time at a48K root. It records first
and repeated appends at32/342/1383/1798/2048/5479 rows, with process disk-read
bytes, so preparation or I/O is not silently attributed to layer GPU work.
This follows the V4.1 finding that its worst live gap can occur before the
first peer round despite exact layer-sliced GPU execution.

Flash-Next has a fused first-two-layer PLE stage followed by46 post-PLE layers.
HC state remains in the packed hyper-residual/bridge scratch. GDN recurrence
and QSA cache writes are per-layer/private to a runner. Native pending-command
validation gates committed positions and snapshots. A port would need to keep
that gate across slices, preserve the dense-to-selected row plan, and retain
the first-two-layer dependency semantics. Decode's native feedback batch loop
uses the shared Driver and needs the same slice/peer-round handoff as GLM.

The baseline's reported partial_prefill_s is0; the incumbent Flash-Next path
lacks the newer complete/partial prefill accounting records, so this is not
proof that no unfinished request paid prefill. A production port must address
that accounting. Component pricing and the build/no-build decision follow.

## Component price and native checkpoint probe

The unchanged native append at pressure0 is GPU-dominated.2048 rows takes
1063–1068ms wall,1054–1055ms GPU and8ms CPU encode;5479 rows takes2941–2954ms
wall over three commands,2909–2912ms GPU and26–41ms CPU encode.1383 rows is
805–810ms wall /797–798ms GPU;1798 is1018–1020ms /1010–1012ms.32/342 rows
are only230/286ms and should remain unsliced. First/repeated appends had zero
recorded process disk reads. This clears a layer-scope experiment's price gate.

Native checkpoint work is isolated on `perf/fn-layer-sliced-prefill`, leaving
the V4.1 branch's default-off decision intact. A packed command stages inputs
and records its checkpoint lineage once. It parks the same hyper-residual and
bridge, keeps pending length/snapshot/decode gating across completed commands,
and publishes through the original child validation only after the last layer.
The fused PLE0/1 stage is indivisible. Submitted commands drain on every yield
and unwind path. No serve scheduling/default has been changed yet.

`fn_layer_slice_probe` compares1/2/4/8 cuts, complete-logit bits, snapshot payload
and PLE-history digests, post-append decode, and two peers' logits/state. Peers
run once between cuts; the control runs equal rounds afterward. Cancellation
rejects provisional snapshots and replays a prior immutable checkpoint in a
fresh runner. Native results are pending; CLI integration waits for this gate.

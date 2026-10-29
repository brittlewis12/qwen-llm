import type { Allocation, Evidence, FlowEdge, FlowNode, FlowVariant, Investigation, SourceRef } from "./model";

const CURRENT = "ad7e3b9acbffb704a233b97e48e52b56c2c617a9";
const Q_BEFORE = "d5954790887fad39b44f30b3ff3ef5ecbb6168db";
const Q_AFTER = "1544c9bc94d5dfae79366f2c0cf594f2924308e3";
const SCRATCH = "87ce11b6cfb55431df51e4bc139886bedded6f27";
const BRIDGE = "cd6f4f068606cf649a280cf1628795d0e5c65bae";
const FORWARD = "crates/qwen-llm/src/deepseek_v41/forward/";
const METAL = "kernels/tensor/deepseek_v41_attention_tensor.metal";
const SCHEMATIC = "Source-linked reconstruction, not a captured live trace. Steps and lanes are curated ordinal source order, not elapsed time. Allocation bars are schematic source-order use windows, not measured lifetimes or safe-reuse proofs.";
const WINDOW = "Schematic source-order use window only; not a measured lifetime, allocation trace, or safe-reuse proof. Extent is not priced unless explicitly stated.";
const KiB = 1024;
const MiB = KiB * KiB;
const Q32 = 64 * 512 * 4;
const Q16 = 64 * 512 * 2;

function ref(id: string, path: string, revision: string, startLine: number, endLine: number, label: string): SourceRef {
  return { id, path, revision, startLine, endLine, label };
}
function proof(id: string, kind: Evidence["kind"], label: string, detail: string, sourceIds: string[]): Evidence {
  return { id, kind, label, detail, sourceIds };
}
function node(id: string, label: string, tensor: string, dtype: string, bytes: number | null, step: number, lane: number, role: FlowNode["role"], evidenceIds: string[], detail: string): FlowNode {
  return { id, label, tensor, dtype, bytes, step, lane, role, evidenceIds, detail };
}
function edge(id: string, from: string, to: string, label: string, kind: FlowEdge["kind"], logicalBytes: number | null, repeat: number, evidenceIds: string[]): FlowEdge {
  return { id, from, to, label, kind, logicalBytes, repeat, evidenceIds };
}
function allocation(id: string, label: string, bytes: number | null, evidenceIds: string[], segments: Allocation["segments"], detail = WINDOW): Allocation {
  return { id, label, bytes, evidenceIds, segments, detail };
}

function queryVariant(after: boolean): FlowVariant {
  const p = after ? "qa" : "qb";
  const code = after ? "q-half-code" : "q-f32-code";
  return {
    id: after ? "after" : "before", label: after ? "Q half mirror" : "F32 Q per panel",
    summary: after ? "Convert post-RoPE Q once; still read the half query for every 32-key panel. K/V remain F32." : "Read and convert the F32 query again for each 32-key panel; the tensor matmul already uses half staging.",
    steps: [
      { label: "Post-RoPE Q", detail: "One query: 64 heads x 512 dimensions. Not a 512-query dispatch diagram." },
      { label: after ? "Convert once" : "Convert per panel", detail: after ? "encode_rows_prepare stages all rows; the diagram prices just one query." : "The k0 loop encloses all four dimension panels and the F32-to-half conversion." },
      { label: after ? "Half mirror" : "Threadgroup Q", detail: after ? "Device half scratch, read again for every key panel." : "Transient threadgroup half staging; no new device Q mirror." },
      { label: "Score panels", detail: "640 keys / 32 keys per panel = 20 panels; K/V loads are not priced here." },
      { label: "Scores", detail: "Scores are written in F32. Softmax, PV and finish_rows are outside this bounded view." },
    ],
    nodes: [
      node(`${p}-q`, "Q after RoPE", "Query Q", "F32", Q32, 0, 0, "value", ["q-geometry", code], "Per-query logical tensor extent: 64 x 512 x 4 = 128 KiB."),
      node(`${p}-stage`, after ? "Stage Q once" : "Convert each panel", "Query Q", "F32 -> F16", null, 1, 0, "operation", [code], after ? "One conversion into the existing HalfScratch for the row batch." : "F32 read and conversion are inside the 32-key-panel loop."),
      node(`${p}-half`, after ? "Q half mirror" : "Q threadgroup tile", "Query Q", "F16", Q16, 2, 0, "value", [code, "q-geometry"], after ? "64 KiB per query in device half scratch; this is not a new allocation." : "64 KiB logical query across four 128-dimension slices; these slices do not all coexist as one threadgroup allocation."),
      node(`${p}-kv`, "Keys / values", "K/V history", "F32", null, 2, 2, "value", [code], "History and selected compressed rows remain F32. Their traffic is deliberately unpriced."),
      node(`${p}-score`, "Tensor64 score", "Attention scores", "F32 accumulation", null, 3, 1, "operation", [code], "Q and keys are staged in threadgroup half tensors for the score matmul."),
      node(`${p}-scores`, "Score rows", "Attention scores", "F32", null, 4, 1, "value", [code], "The score output exists in source, but its full padded allocation is not priced by this per-query sketch."),
    ],
    edges: [
      edge(`${p}-input`, `${p}-q`, `${p}-stage`, after ? "F32 read once for conversion" : "F32 Q read per panel", "read", Q32, after ? 1 : 20, [code, "q-geometry"]),
      edge(`${p}-convert`, `${p}-stage`, `${p}-half`, after ? "One half conversion" : "Half staging per panel", "convert", Q16, after ? 1 : 20, [code, "q-geometry"]),
      edge(`${p}-panel`, `${p}-half`, `${p}-score`, after ? "Half Q read per 32-key panel" : "Threadgroup Q feeds matmul", "read", after ? Q16 : null, 20, [code, "q-geometry"]),
      edge(`${p}-keys`, `${p}-kv`, `${p}-score`, "F32 key gather; bytes unknown", "read", null, 20, [code]),
      edge(`${p}-write`, `${p}-score`, `${p}-scores`, "Write scaled score rows", "write", null, 1, [code]),
    ],
    allocations: after ? [allocation("q-shared-half", "Existing half scratch", null, ["q-half-code", "q-layout"], [
      { start: 1, end: 5, label: "Query half use", tensor: "Query Q", kind: "live" },
    ], `${WINDOW} The 64 KiB node is a per-query view, not the whole allocation. finish_rows later overwrites this shared scratch for projection; that later use is outside this diagram.`)] : [],
  };
}

const query: Investigation = {
  schemaVersion: 1, id: "query-half", title: "Convert once. Read every panel.", subtitle: "V4.1 Q-half mirror / a reported win",
  question: "Can an existing half scratch remove repeated F32-to-half query conversion without changing the score arithmetic?",
  reconstruction: SCHEMATIC,
  scope: { family: "DeepSeek V4.1", phase: "Tensor-tier prefill attention", artifact: "GSQ-RCO IQ2_XS, 40 layers", device: "Apple M5 Ultra", workload: "8K CED report; separate 512-query probe at K=640, window=128, topk=512", regime: "Historical same-binary O S S O CED A/B; per-query source diagram", revision: CURRENT },
  sources: [
    ref("q-report", "docs/bench/2026-10-29-v41-attention-qhalf.md", CURRENT, 1, 84, "2026-10-29 Q-half report"),
    ref("q-source-before", METAL, Q_BEFORE, 251, 325, "Historical F32 score kernel"),
    ref("q-source-after", METAL, Q_AFTER, 611, 685, "Historical Q-half score kernel"),
    ref("q-source-stage", `${FORWARD}attention_gpu/query_batch.rs`, CURRENT, 755, 773, "Convert post-RoPE Q into shared half scratch"),
    ref("q-source-bind", `${FORWARD}attention_gpu/query_batch.rs`, CURRENT, 1514, 1543, "Bind half query view for tensor64"),
    ref("q-source-finish", `${FORWARD}attention_gpu.rs`, CURRENT, 3052, 3102, "Later projection use of half scratch"),
  ],
  evidence: [
    proof("q-f32-code", "curated", "F32 source path", "The historical score kernel reloads and converts Q inside the k0 += 32 loop. The diagram collapses the four dimension slices and does not claim their half staging coexists.", ["q-source-before"]),
    proof("q-half-code", "curated", "Half source path", "The preparation path converts post-RoPE Q once, sink_tile binds its view, and the half score kernel still reads Q for each 32-key panel. K/V stay F32.", ["q-source-stage", "q-source-bind", "q-source-after", "q-source-finish"]),
    proof("q-geometry", "derived", "Per-query arithmetic", "64 x 512 elements: F32 = 131072 bytes (128 KiB), F16 = 65536 bytes (64 KiB). At 640 keys, 640/32 = 20 panels. Repeated score Q reads: 2.5 MiB F32 versus 1.25 MiB F16, plus a separate 128 KiB F32 staging read in the half path. These are logical source reads, not measured DRAM transactions.", ["q-source-before", "q-source-after", "q-report"]),
    proof("q-warm", "measured", "Reported CED A/B", "The dated report records off 1863.2 versus on 1901.9 tok/s, +2.08% and 89.4 ms, same-binary O S S O. This tool has not replayed the raw runs.", ["q-report"]),
    proof("q-probe", "measured", "Reported 512-query probe", "A separate production-shaped 512-query probe reports Q half 1.113x non-ragged / 1.115x ragged and bitwise equality on finite lanes. These ratios are not the per-query diagram or the model-level CED result.", ["q-report"]),
    proof("q-layout", "curated", "Bounded layout", "Five ordinal steps and source-order half-scratch use only. No timestamps, allocation capture or independently established reuse proof.", ["q-source-stage", "q-source-finish", "q-report"]),
  ],
  metrics: [
    { label: "CED throughput", value: "+2.08%", unit: "reported", detail: "1863.2 -> 1901.9 tok/s; 89.4 ms less wall time in the same-binary A/B.", evidenceId: "q-warm" },
    { label: "Q extent / query", value: "128 -> 64", unit: "KiB", detail: "F32 versus F16 logical extent; 64 heads x 512 dimensions.", evidenceId: "q-geometry" },
    { label: "Score Q reads", value: "2.5 -> 1.25", unit: "MiB/query", detail: "20 panels at K=640. Half path additionally reads F32 once for staging. Not DRAM bytes.", evidenceId: "q-geometry" },
    { label: "512-query probe", value: "1.113x / 1.115x", unit: "reported", detail: "Non-ragged / ragged Q-half probe; separate from the per-query view and CED A/B.", evidenceId: "q-probe" },
  ],
  variants: [queryVariant(false), queryVariant(true)],
  finding: "The win removes repeated conversion and halves score-kernel Q source-read width; it does not make Q a one-read tensor. Existing half scratch avoids a new allocation.",
  nextMeasurement: "A future matched run could attribute conversion and source traffic separately; no new measurement is performed by this offline explorer.",
  cautions: ["One artifact, prompt and host session; CED and probe observations have different scopes.", "Conversion occurs once, but Q is still read per 32-key panel. Per-query geometry is separate from 512-query tile evidence.", "Edge byte counts describe distinct operations and must not be summed as unique DRAM traffic or a speedup model.", WINDOW],
};

function scratchVariant(after: boolean): FlowVariant {
  const p = after ? "sa" : "sb";
  const e = ["s-code", "s-layout"];
  const steps = [
    { label: "Prepare full A", detail: "Allocate full-size set and reset incoming before the first chunk, even though construction initialized it." },
    { label: "Run full A", detail: "Use x, half, outputs, incoming, HcRows, FFN, attention and Engram scratch." },
    { label: "Prepare full B", detail: after ? "Retain the same full-size set and reset incoming to [1,0,0,0] for every row." : "Allocate a fresh full-size set; reset incoming before execution." },
    { label: "Run full B", detail: "Continue persistent causal state; scratch is transient workspace, not that state." },
    { label: "Prepare small tail", detail: after ? "Drop the max-row set before allocating a differently sized multi-row tail set." : "The previous per-chunk set has left its scope before the tail allocation." },
    { label: "Run tail", detail: "A schematic multi-row tail; a one-token tail takes the scalar route instead." },
  ];
  return {
    id: after ? "after" : "before", label: after ? "Retain full scratch" : "Allocate per chunk",
    summary: "Two full chunks plus a smaller multi-row tail illustrate source policy, not captured production route counts or exact memory totals.", steps,
    nodes: [
      node(`${p}-alloc-a`, "Allocate + reset", "Chunk scratch", "Mixed", null, 0, 0, "operation", e, "The first run_chunk resets incoming too; construction's initialization is not a replacement for this reset."),
      node(`${p}-set-a`, "Full set A", "Chunk scratch", "Mixed", null, 0, 1, "value", e, "Model-dependent aggregate scratch size is intentionally unknown."),
      node(`${p}-run-a`, "Full chunk A", "Causal state", "Mixed", null, 1, 1, "operation", e, "Uses scratch and advances persistent causal state."),
      node(`${p}-prepare-b`, after ? "Reset incoming" : "Allocate + reset", "Chunk scratch", "Mixed", null, 2, 0, "operation", e, "incoming is reset to [1,0,0,0] per row at every chunk start."),
      node(`${p}-set-b`, after ? "Same full set" : "Fresh full set B", "Chunk scratch", "Mixed", null, 2, 1, "value", e, after ? "A view of the same ChunkScratch object, not a tensor copy." : "Fresh zeroed buffers; no claimed exact GiB total."),
      node(`${p}-run-b`, "Full chunk B", "Causal state", "Mixed", null, 3, 1, "operation", e, "The causal state comes from the previous chunk, independent of allocation policy."),
      node(`${p}-alloc-tail`, "Allocate tail set", "Tail scratch", "Mixed", null, 4, 0, "operation", e, "After releasing the full set; scalar one-token tails are outside this view."),
      node(`${p}-tail-set`, "Smaller tail set", "Tail scratch", "Mixed", null, 4, 1, "value", e, "A new, smaller allocation, not an in-place prefix reuse claim."),
      node(`${p}-tail`, "Multi-row tail", "Causal state", "Mixed", null, 5, 1, "operation", e, "Tail execution continues persistent state and resets its incoming rows too."),
    ],
    edges: [
      edge(`${p}-create-a`, `${p}-alloc-a`, `${p}-set-a`, "Construct scratch; initialize/reset", "write", null, 1, e),
      edge(`${p}-use-a`, `${p}-set-a`, `${p}-run-a`, "Use chunk workspace", "read", null, 1, e),
      ...(after ? [edge(`${p}-retain`, `${p}-set-a`, `${p}-prepare-b`, "Retain full-size scratch", "reuse", null, 1, e)] : []),
      edge(`${p}-create-b`, `${p}-prepare-b`, `${p}-set-b`, after ? "Same backing; incoming reset" : "Fresh backing; incoming reset", after ? "view" : "write", null, 1, e),
      edge(`${p}-use-b`, `${p}-set-b`, `${p}-run-b`, "Use chunk workspace", "read", null, 1, e),
      edge(`${p}-state-b`, `${p}-run-a`, `${p}-run-b`, "Continue persistent causal state", "read", null, 1, e),
      edge(`${p}-create-tail`, `${p}-alloc-tail`, `${p}-tail-set`, "Construct smaller tail scratch", "write", null, 1, e),
      edge(`${p}-use-tail`, `${p}-tail-set`, `${p}-tail`, "Use tail workspace", "read", null, 1, e),
      edge(`${p}-state-tail`, `${p}-run-b`, `${p}-tail`, "Continue persistent causal state", "read", null, 1, e),
    ],
    allocations: [
      ...(after ? [allocation("s-full", "Retained full set", null, e, [
        { start: 0, end: 2, label: "First full chunk", tensor: "Chunk scratch", kind: "live" },
        { start: 2, end: 3, label: "Retain + reset", tensor: "Chunk scratch", kind: "retained" },
        { start: 3, end: 4, label: "Second full chunk", tensor: "Chunk scratch", kind: "live" },
      ])] : [
        allocation("s-full-a", "Full set A", null, e, [{ start: 0, end: 2, label: "First full chunk", tensor: "Chunk scratch", kind: "live" }]),
        allocation("s-full-b", "Full set B", null, e, [{ start: 2, end: 4, label: "Second full chunk", tensor: "Chunk scratch", kind: "live" }]),
      ]),
      allocation("s-tail", "New smaller tail set", null, e, [{ start: 4, end: 6, label: "After full-set drop", tensor: "Tail scratch", kind: "live" }]),
    ],
  };
}

const scratch: Investigation = {
  schemaVersion: 1, id: "scratch-reuse", title: "Keep the set. Reset the state.", subtitle: "V4.1 chunk scratch / a reported warm win",
  question: "Can full chunks retain their scratch set while preserving initialization and dropping the max set before a smaller tail?",
  reconstruction: SCHEMATIC,
  scope: { family: "DeepSeek V4.1", phase: "Chunked prefill", artifact: "GSQ-RCO IQ2_XS", device: "M5 lifetime witness; GSQ campaign host", workload: "Warm pp8192 in-binary campaign; two-full-chunk plus tail source sketch", regime: "Historical N C C N, discarded warm-up, same binary; no production route-count capture", revision: SCRATCH },
  sources: [
    ref("s-report", "docs/bench/2026-10-26-v41-scratch-reuse.md", CURRENT, 1, 92, "2026-10-26 scratch-reuse report"),
    ref("s-policy", `${FORWARD}prefill.rs`, SCRATCH, 1004, 1070, "Full-size reuse and tail drop-before-allocate"),
    ref("s-reset", `${FORWARD}gpu_step.rs`, SCRATCH, 302, 390, "ChunkScratch construction and per-chunk incoming reset"),
    ref("s-allocate", `${FORWARD}prefill.rs`, SCRATCH, 739, 755, "Fresh unstaged zero allocation helper"),
    ref("s-state", `${FORWARD}prefill.rs`, CURRENT, 1193, 1212, "Chunk execution updates persistent request state"),
  ],
  evidence: [
    proof("s-code", "curated", "Source allocation policy", "Full-size scratch can survive full chunks. run_chunk resets incoming even on the first chunk. A differently sized multi-row tail drops the full set before allocation. Per-chunk mode constructs fresh sets; a scalar tail uses execute_token.", ["s-policy", "s-reset", "s-allocate", "s-state"]),
    proof("s-warm", "measured", "Reported in-binary warm A/B", "The dated report gives reuse 863.725 versus per-chunk 834.165 tok/s, +3.54%, reversed N C C N in one binary after discarded warm-up. No raw run is executed here.", ["s-report"]),
    proof("s-layout", "curated", "Schematic use windows", "Two full chunks and a multi-row tail are an explanatory layout. Bytes are null: neither exact production route counts nor an exact aggregate scratch GiB total is established by this view. Bars are not measured lifetimes.", ["s-policy", "s-reset", "s-report"]),
    proof("s-size-unknown", "unknown", "Aggregate scratch not priced", "Artifact-dependent aggregate scratch bytes and measured lifetime endpoints are not supplied by this reconstruction. Small-fixture device totals in the report must not be treated as full-model totals.", []),
  ],
  metrics: [
    { label: "Warm throughput", value: "+3.54%", unit: "reported", detail: "Robust in-binary warm estimate, not the exploratory cross-build +5.99%.", evidenceId: "s-warm" },
    { label: "Warm means", value: "834.165 -> 863.725", unit: "tok/s", detail: "Per-chunk -> reuse, same binary and reversed campaign order.", evidenceId: "s-warm" },
    { label: "Scratch total", value: "Unknown", unit: "not priced", detail: "No invented exact GiB total; bars show source-order use only.", evidenceId: "s-size-unknown" },
  ],
  variants: [scratchVariant(false), scratchVariant(true)],
  finding: "Retaining the full-size set removes repeated allocation/zeroing, but every chunk still requires an incoming reset. Tail replacement is drop-before-allocate, not overlapping sets or an in-place resize.",
  nextMeasurement: "A future CPU-side allocation/route census could price actual full-model sets and verify production chunk routing without reinterpreting this layout as a capture.",
  cautions: ["The +5.99% cross-build comparison is exploratory; +3.54% is the reported warm in-binary result.", "Reset incoming at every chunk start, including the first reuse chunk; earlier lifetime testing caught a capture-bit difference.", "The report's tiny lifetime-witness byte totals are not a full-model scratch budget. The 8192 chunk remained a model pending production route counts.", "One artifact and host session; a shared page cache and separate causal-state/continuation scope limit generalization.", WINDOW],
};

function bridgeVariant(split: boolean): FlowVariant {
  const p = split ? "bc" : "br";
  const e = ["b-code", "b-layout"];
  return {
    id: split ? "copy-split" : "control", label: split ? "Reported copy split" : "Reported control",
    summary: split ? "Reported profiling-only eight-stage split isolates copies. Tensor ownership and kernels are unchanged; the temporary patch was removed." : "Six-stage sampled control combines each copy with its following combine span. The six diagram steps are semantic order, not sampled encoders.",
    steps: [
      { label: "Mixer output", detail: "GDN or QSA produces [2560,512] F32. The sketch groups 46 post-PLE blocks by operation." },
      { label: "Mixer bridge", detail: split ? "Copy sampled separately in the reported temporary split." : "Copy shares a sampled stage with attention combine." },
      { label: "Attention combine", detail: "Combine reads the shared bridge and updates hyper-residual state." },
      { label: "FFN + MoE", detail: "FFN residual mix feeds MoE; its result has the same logical [2560,512] F32 extent." },
      { label: "MoE bridge", detail: split ? "Second copy sampled separately in the report." : "Copy shares a sampled stage with FFN combine." },
      { label: "FFN combine", detail: "Combine reads the same bridge backing after the second copy." },
    ],
    nodes: [
      node(`${p}-mixer`, "Mixer output", "Mixer output", "F32", 5 * MiB, 0, 0, "value", ["b-code", "b-arithmetic"], "2560 x 512 x 4 bytes = 5 MiB payload per block."),
      node(`${p}-initial`, "Hyper residual", "Hyper residual", "F32", null, 0, 2, "value", e, "Persistent residual input to the attention combine; its traffic is unpriced."),
      node(`${p}-bridge-a`, "Bridge / mixer", "Shared bridge", "F32", 5 * MiB, 1, 0, "value", ["b-code", "b-arithmetic"], "Copy destination, shared with the later MoE copy."),
      node(`${p}-att`, "Attention combine", "Hyper residual", "F32", null, 2, 1, "operation", e, "Consumes bridge plus residual. Not a measured copy saving."),
      node(`${p}-residual`, "Updated residual", "Hyper residual", "F32", null, 2, 2, "value", e, "Attention-combine output; FFN residual mix and combine consume it."),
      node(`${p}-moe`, "FFN mix + MoE", "MoE output", "F32", null, 3, 0, "operation", e, "Bounded composite of residual mixing and MoE execution, not a fused-kernel claim."),
      node(`${p}-moe-out`, "MoE output", "MoE output", "F32", 5 * MiB, 3, 1, "value", ["b-code", "b-arithmetic"], "Second 5 MiB source payload per block."),
      node(`${p}-bridge-b`, "Bridge / MoE", "Shared bridge", "F32", 5 * MiB, 4, 0, "value", ["b-code", "b-arithmetic"], "The same bridge buffer is overwritten after attention combine."),
      node(`${p}-ffn`, "FFN combine", "Hyper residual", "F32", null, 5, 1, "operation", e, "Consumes the MoE bridge contents and residual."),
    ],
    edges: [
      edge(`${p}-copy-a`, `${p}-mixer`, `${p}-bridge-a`, "5 MiB payload x 46 blocks", "copy", 5 * MiB, 46, ["b-code", "b-arithmetic"]),
      edge(`${p}-read-a`, `${p}-bridge-a`, `${p}-att`, "Read bridge for attention combine", "read", null, 46, e),
      edge(`${p}-residual-a`, `${p}-initial`, `${p}-att`, "Read attention residual", "read", null, 46, e),
      edge(`${p}-write-state`, `${p}-att`, `${p}-residual`, "Update hyper residual", "write", null, 46, e),
      edge(`${p}-mix`, `${p}-residual`, `${p}-moe`, "FFN residual mix feeds MoE", "read", null, 46, e),
      edge(`${p}-write-moe`, `${p}-moe`, `${p}-moe-out`, "Produce MoE output", "write", null, 46, e),
      edge(`${p}-copy-b`, `${p}-moe-out`, `${p}-bridge-b`, "5 MiB payload x 46 blocks", "copy", 5 * MiB, 46, ["b-code", "b-arithmetic"]),
      edge(`${p}-read-b`, `${p}-bridge-b`, `${p}-ffn`, "Read bridge for FFN combine", "read", null, 46, e),
      edge(`${p}-residual-b`, `${p}-residual`, `${p}-ffn`, "Read FFN residual", "read", null, 46, e),
    ],
    allocations: [allocation("b-shared", "One shared bridge", 5 * MiB, ["b-code", "b-arithmetic", "b-layout"], [
      { start: 1, end: 3, label: "Mixer copy + combine use", tensor: "Shared bridge", kind: "live" },
      { start: 3, end: 4, label: "Retained backing", tensor: "Shared bridge", kind: "retained" },
      { start: 4, end: 6, label: "MoE copy + combine use", tensor: "Shared bridge", kind: "live" },
    ], `${WINDOW} One logical 5 MiB bridge at N=512; 46 repetitions do not imply 46 simultaneously allocated buffers.`)],
  };
}

const bridge: Investigation = {
  schemaVersion: 1, id: "bridge-copy", title: "A visible copy. A small ceiling.", subtitle: "Flash-Next bridge / a reported negative",
  question: "Is eliminating the packed post-PLE bridge ownership copy large enough to justify implementation?",
  reconstruction: SCHEMATIC,
  scope: { family: "Qwen Flash-Next", phase: "Packed post-PLE prefill", artifact: "Internal-SSD UD-Q3_K_XL; unsloth/Qwen3.8-Flash-Next-GGUF", device: "Apple M4 Max", workload: "Natural N=512, one generated token, one packed command, no scalar tail", regime: "Historical separate-process A1 B1 B2 A2; six-stage control vs profiling-only eight-stage split", revision: BRIDGE },
  sources: [
    ref("b-report", "docs/bench/2026-08-29-qwen4exp-bridge-copy-ceiling-kill/README.md", CURRENT, 1, 73, "2026-08-29 bridge-copy ceiling KILL report"),
    ref("b-source-mixer", "crates/qwen-llm/src/qwen4exp_post_ple_block.rs", BRIDGE, 1190, 1248, "Historical mixer variants, bridge copy, attention combine"),
    ref("b-source-moe", "crates/qwen-llm/src/qwen4exp_post_ple_block.rs", BRIDGE, 1250, 1311, "Historical MoE, bridge copy, FFN combine"),
    ref("b-source-serial", "crates/qwen-llm/src/qwen4exp_packed_prefill.rs", BRIDGE, 835, 902, "Serial GDN bridge source semantics"),
  ],
  evidence: [
    proof("b-code", "curated", "Source bridge ownership", "Mixer and MoE each copy F32 output into the same shared bridge before their respective combines. The historical sampled control places each copy with combine. The temporary split is documented, not reconstructed as surviving executable code.", ["b-source-mixer", "b-source-moe", "b-source-serial", "b-report"]),
    proof("b-arithmetic", "derived", "Logical payload, not DRAM", "2560 x 512 x 4 = 5242880 bytes = 5 MiB per copy. 2 x (34 GDN + 12 QSA) = 92 copies: 460 MiB payload; counting each payload read plus write gives 920 MiB logical R/W. This is neither measured DRAM traffic nor 920 MiB of payload.", ["b-report", "b-source-mixer", "b-source-moe"]),
    proof("b-timing", "measured", "Reported sampled copy spans", "The dated report gives individual sampled copy spans of 0.018250-0.018625 ms, with stage-boundary overhead charged to the copy. No new GPU timing or raw-run replay is performed here.", ["b-report"]),
    proof("b-projection", "derived", "Reported adjusted GPU projection", "Two accepted split captures project representative layers across mixer families: raw 1.703394 / 1.697572 ms, topology debit 0.009584 / 0 ms, adjusted 1.693810 / 1.697572 ms. The report gives 0.175% of profiled command GPU, not model wall time. These are reported projections, not direct timing of all 92 copies.", ["b-report"]),
    proof("b-gate", "derived", "Reported decision gate", "The adjusted 1.69-1.70 ms ceiling is 17.5% of the preregistered 9.671346 ms implementation gate; decision KILL. A stage boundary charges extra overhead to copy spans, so the screen is optimistic, not realizable savings.", ["b-report"]),
    proof("b-layout", "curated", "Same flow, different observation", "Both variants intentionally have the same tensor graph. Six ordinal source steps do not denote the six or eight sampled encoders. Allocation windows are schematic, and the temporary split was removed.", ["b-report", "b-source-mixer", "b-source-moe"]),
  ],
  metrics: [
    { label: "Copy payload", value: "460", unit: "MiB", detail: "92 copies x 5 MiB; grouped edges each repeat 46 times.", evidenceId: "b-arithmetic" },
    { label: "Logical read + write", value: "920", unit: "MiB", detail: "Twice the payload, not measured DRAM traffic.", evidenceId: "b-arithmetic" },
    { label: "Adjusted projection", value: "1.693810 / 1.697572", unit: "ms", detail: "B1 / B2 family projections after topology debit; not independently timing every copy.", evidenceId: "b-projection" },
    { label: "Profiled GPU share", value: "0.175%", unit: "reported", detail: "Adjusted projection / profiled command GPU; not a model wall-time fraction.", evidenceId: "b-projection" },
    { label: "Implementation gate", value: "9.671346", unit: "ms", detail: "Only 17.5% of this preregistered threshold; ownership work was killed.", evidenceId: "b-gate" },
  ],
  variants: [bridgeVariant(false), bridgeVariant(true)],
  finding: "The copy is real, but its optimistic reported ownership ceiling is too small. The split changes observation boundaries, not dataflow, and provides no demonstrated model speedup.",
  nextMeasurement: "Revisit only if workload, bridge geometry or ownership costs materially change; this negative does not warrant an implementation from the historical evidence.",
  cautions: ["920 MiB is logical read plus write for 460 MiB payload, not measured DRAM movement.", "0.175% is a reported adjusted projection of profiled command GPU, not measured model wall-time share.", "Representative layers 5 and 7 are projected across families; new encoder-boundary costs make this an optimistic ownership screen.", "The temporary copy split and binaries were removed; only the dated report and historical control source are linked. Replay checks were not an external semantic oracle.", WINDOW],
};

export const investigations: Investigation[] = [query, scratch, bridge];

import { edgeBytes, formatBytes, neighborhood, tensorPath } from "./model";
import type { Evidence, EvidenceBundle, FlowNode, FlowVariant, Investigation, SourceRef } from "./model";
import { encodeBundle, MAX_PACKET_BYTES, validateBundle, validateInvestigation } from "./validate";

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, string | number | boolean | undefined>;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === false) continue;
    node.setAttribute(key === "className" ? "class" : key, value === true ? "" : String(value));
  }
  for (const child of children) if (child !== null && child !== undefined && child !== false) node.append(child instanceof Node ? child : String(child));
  return node;
}
function button(label: string, action: () => void, attrs: Attrs = {}): HTMLButtonElement {
  const node = el("button", { type: "button", ...attrs }, label);
  node.addEventListener("click", action);
  return node;
}
function svg(tag: string, attrs: Attrs, ...children: SVGElement[]): SVGElement {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== undefined) node.setAttribute(key, String(value));
  node.append(...children);
  return node;
}
const root = document.querySelector<HTMLDivElement>("#app")!;
const notes = new Map<Investigation, string>();
let original: EvidenceBundle;
let bundle: EvidenceBundle;
let imported = false;
let caseId = "query-half";
let variantId = "";
let nodeId = "";
let step = Number.MAX_SAFE_INTEGER;
let follow = true;
let compare = false;
let notice = "";
let error = "";

const investigation = (): Investigation => bundle.investigations.find(item => item.id === caseId) ?? bundle.investigations[0];
const variant = (): FlowVariant => investigation().variants.find(item => item.id === variantId) ?? investigation().variants[0];
const selected = (): FlowNode => variant().nodes.find(item => item.id === nodeId) ?? variant().nodes[0];
const evidenceFor = (id: string): Evidence => investigation().evidence.find(item => item.id === id)!;
const kindBadge = (kind: Evidence["kind"]): HTMLElement => el("span", { className: `badge ${kind}` }, kind);

function rememberLocation(): void {
  const query = new URLSearchParams({ case: investigation().id, variant: variant().id, node: selected().id, step: String(step), follow: follow ? "1" : "0", compare: compare ? "1" : "0" });
  history.replaceState(null, "", `#${query}`);
}
function readLocation(): void {
  const query = new URLSearchParams(location.hash.slice(1));
  caseId = query.get("case") ?? caseId;
  variantId = query.get("variant") ?? "";
  nodeId = query.get("node") ?? "";
  const candidateStep = Number(query.get("step") ?? Number.MAX_SAFE_INTEGER);
  step = Number.isSafeInteger(candidateStep) && candidateStep >= 0 ? candidateStep : Number.MAX_SAFE_INTEGER;
  follow = query.get("follow") !== "0";
  compare = query.get("compare") === "1";
}
function chooseCase(id: string): void {
  caseId = id;
  variantId = "";
  nodeId = "";
  step = Number.MAX_SAFE_INTEGER;
  error = "";
  render();
}
function chooseVariant(id: string): void {
  const tensor = selected().tensor;
  variantId = id;
  nodeId = variant().nodes.find(node => node.tensor === tensor)?.id ?? variant().nodes[0].id;
  render();
}
function chooseNode(flow: FlowVariant, node: FlowNode): void {
  variantId = flow.id;
  nodeId = node.id;
  render();
}

function sourceButton(source: SourceRef): HTMLButtonElement {
  return button(source.label, () => openSource(source), { className: "source-link", "data-source-id": source.id, "data-focus-key": `source-${source.id}` });
}
function evidenceRow(evidence: Evidence): HTMLElement {
  const sources = evidence.sourceIds.map(id => investigation().sources.find(source => source.id === id)!);
  return el("div", { className: "evidence-row" },
    el("div", { className: "evidence-heading" }, kindBadge(evidence.kind), el("strong", {}, evidence.label)),
    el("p", {}, evidence.detail),
    sources.length ? el("div", { className: "source-links" }, ...sources.map(sourceButton)) : el("p", { className: "unavailable" }, "No supporting observation supplied."),
  );
}

function openSource(source: SourceRef): void {
  const dialog = document.querySelector<HTMLDialogElement>("#source-dialog")!;
  const snapshot = bundle.snapshots.find(item => item.id === source.id);
  const close = button("Close source", () => dialog.close(), { "data-close-source": true, className: "quiet" });
  const lines = snapshot?.excerpt.split("\n").map((line, index) => el("div", { className: "source-line" }, el("span", { className: "line-number", "aria-hidden": "true" }, source.startLine + index), el("code", {}, line)));
  dialog.replaceChildren(
    el("div", { className: "dialog-heading" }, el("div", {}, el("span", { className: "eyebrow" }, "Pinned source / read only"), el("h2", {}, source.label)), close),
    el("p", { className: "source-path" }, `${source.path}:${source.startLine} at ${source.revision}`),
    el("p", { className: "notice" }, imported ? "Imported excerpt. Hashes and citations are claims in this packet; the browser has not authenticated them against Git." : "Bundled from the pinned Git revision at build time. This is historical evidence, not a reading of today's working tree."),
    snapshot ? el("div", { className: "source-code", tabindex: "0", "aria-label": "Source excerpt" }, ...(lines ?? [])) : el("p", { className: "unavailable" }, "Source excerpt not included. The citation remains available, but this browser cannot recover missing source from a local path."),
    ...snapshot ? [el("p", { className: "hash" }, `Full source file SHA-256: ${snapshot.sha256}`)] : [],
  );
  dialog.showModal();
}

function renderGraph(flow: FlowVariant): HTMLElement {
  const cell = 154;
  const nodeWidth = 130;
  const width = flow.steps.length * cell + 28;
  const height = (Math.max(...flow.nodes.map(node => node.lane)) + 1) * 96 + 76;
  const x = (node: FlowNode) => 14 + node.step * cell + cell / 2;
  const y = (node: FlowNode) => 87 + node.lane * 96;
  const focused = new Set([...tensorPath(flow, selected().tensor), ...neighborhood(flow, flow.id === variant().id ? selected().id : flow.nodes.find(node => node.tensor === selected().tensor)?.id ?? "")]);
  const markerId = `arrow-${flow.id}`;
  const connectors = svg("svg", { class: "connectors", viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true" },
    svg("defs", {}, svg("marker", { id: markerId, viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, svg("path", { d: "M0,0 L8,4 L0,8 z", fill: "currentColor" }))),
  );
  for (const edge of flow.edges) {
    const from = flow.nodes.find(node => node.id === edge.from)!;
    const to = flow.nodes.find(node => node.id === edge.to)!;
    const horizontal = from.step !== to.step;
    const direction = to.step >= from.step ? 1 : -1;
    const startX = x(from) + (horizontal ? direction * (nodeWidth / 2 + 1) : 0);
    const endX = x(to) - (horizontal ? direction * (nodeWidth / 2 + 5) : 0);
    const startY = y(from) + (horizontal ? 0 : 31);
    const endY = y(to) - (horizontal ? 0 : 36);
    const middle = (startX + endX) / 2;
    const d = horizontal ? `M${startX},${startY} C${middle},${startY} ${middle},${endY} ${endX},${endY}` : `M${startX},${startY} L${endX},${endY}`;
    const dimmed = follow && !focused.has(edge.from) && !focused.has(edge.to);
    const path = svg("path", { d, class: `edge ${edge.kind}${dimmed ? " dimmed" : ""}`, "marker-end": `url(#${markerId})` });
    const title = svg("title", {});
    title.textContent = `${edge.label}; ${edge.repeat} repetition(s); ${edgeBytes(edge) === null ? "logical bytes not priced" : `${formatBytes(edgeBytes(edge))} logical bytes, not measured DRAM traffic`}`;
    path.append(title);
    connectors.append(path);
  }
  connectors.append(svg("line", { x1: 14 + (Math.min(step, flow.steps.length - 1) + 1) * cell - 3, y1: 43, x2: 14 + (Math.min(step, flow.steps.length - 1) + 1) * cell - 3, y2: height - 10, class: "order-cursor", "data-order-cursor": flow.id }));
  const canvas = el("div", { className: "flow-canvas", style: `width:${width}px;height:${height}px` }, connectors);
  for (const [index, phase] of flow.steps.entries()) {
    canvas.append(el("div", { className: `step-label${index === step ? " current" : ""}`, "data-source-step": index, style: `left:${14 + index * cell}px;width:${cell}px` }, el("span", {}, String(index + 1).padStart(2, "0")), el("strong", {}, phase.label)));
  }
  for (const node of flow.nodes) {
    const isSelected = flow.id === variant().id && node.id === selected().id;
    const nodeButton = button("", () => chooseNode(flow, node), {
      className: `flow-node ${node.role}${isSelected ? " selected" : ""}${follow && !focused.has(node.id) ? " dimmed" : ""}${node.step > step ? " future" : ""}`,
      style: `left:${x(node) - nodeWidth / 2}px;top:${y(node) - 32}px;width:${nodeWidth}px`,
      "aria-pressed": String(isSelected),
      "aria-label": `${node.label}, ${node.dtype || node.role}, ${node.tensor}`,
      "data-node-id": node.id,
      "data-source-step": node.step,
      "data-focus-key": `node-${flow.id}-${node.id}`,
    });
    nodeButton.append(el("span", { className: "node-role" }, node.role === "operation" ? "operation" : node.dtype || "value"), el("strong", {}, node.label), el("span", { className: "node-size" }, node.role === "operation" ? "inspect access" : formatBytes(node.bytes)));
    canvas.append(nodeButton);
  }
  return el("section", { className: `flow-panel${flow.id === variant().id ? " active-flow" : ""}`, "aria-label": `${flow.label} dataflow` },
    el("div", { className: "flow-heading" }, el("h3", {}, flow.label), el("span", { className: "eyebrow" }, "Source-order reconstruction")),
    el("p", { className: "flow-summary" }, flow.summary),
    el("div", { className: "flow-scroll", "data-graph-scroll": flow.id, tabindex: "0", "aria-label": "Scrollable dataflow. Tab to select nodes." }, canvas),
  );
}

function renderInspector(): HTMLElement {
  const flow = variant();
  const node = selected();
  const accesses = flow.edges.filter(edge => edge.from === node.id || edge.to === node.id);
  return el("aside", { className: "inspector", "data-inspector": true, "aria-label": "Selected tensor and access evidence" },
    el("span", { className: "eyebrow" }, "Under the lens"),
    el("h2", {}, node.label),
    el("p", { className: "tensor-name" }, node.tensor),
    el("div", { className: "inspector-facts" }, el("span", {}, node.dtype || node.role), el("span", {}, formatBytes(node.bytes)), el("span", {}, flow.label)),
    el("p", {}, node.detail),
    el("h3", {}, "Access neighborhood"),
    ...accesses.map(edge => {
      const neighbor = flow.nodes.find(item => item.id === (edge.from === node.id ? edge.to : edge.from))!;
      const jump = button(`${edge.from === node.id ? "To" : "From"} ${neighbor.label}`, () => chooseNode(flow, neighbor), { className: "neighbor", "data-focus-key": `edge-${edge.id}` });
      const units = edge.kind === "copy" ? "logical payload bytes" : edge.kind === "convert" ? "logical result bytes" : "logical access bytes";
      return el("div", { className: "access" }, el("div", { className: "access-title" }, el("span", { className: `access-kind ${edge.kind}` }, edge.kind), el("strong", {}, edge.repeat > 1 ? `x${edge.repeat}` : "x1")), el("p", {}, edge.label), el("p", { className: "access-bytes" }, `${formatBytes(edgeBytes(edge))}${edge.logicalBytes === null ? "" : ` ${units}`}`), jump,
        el("details", { className: "access-evidence", "data-edge-evidence": edge.id }, el("summary", {}, "Basis for this access"), ...edge.evidenceIds.map(id => evidenceRow(evidenceFor(id)))),
      );
    }),
    el("p", { className: "small-note" }, "Logical access bytes are not measured DRAM traffic. A view is not a transfer. Edge widths do not encode cost."),
    el("h3", {}, "Basis for this node"),
    ...node.evidenceIds.map(id => evidenceRow(evidenceFor(id))),
  );
}

function renderLifetimes(): HTMLElement {
  const flow = variant();
  const count = flow.steps.length;
  return el("section", { className: "lifetime-panel", "data-lifetimes": true },
    el("div", { className: "section-heading" }, el("div", {}, el("span", { className: "eyebrow" }, "Storage / across source order"), el("h2", {}, "What is still needed?")), el("span", { className: "badge curated" }, "schematic use windows")),
    el("p", { className: "small-note" }, "These windows explain source dependencies, not observed allocation lifetimes, physical residency, GPU completion, or permission to alias storage. Bar widths are ordinal, never milliseconds."),
    flow.allocations.length === 0 && el("p", { className: "unavailable" }, "No backing-allocation windows are supplied for this variant."),
    ...flow.allocations.map(allocation => el("div", { className: "allocation" },
      el("div", { className: "allocation-label" }, el("strong", {}, allocation.label), el("span", {}, formatBytes(allocation.bytes))),
      el("div", { className: "allocation-track", style: `--steps:${count}` },
        ...allocation.segments.map(segment => {
          const segmentButton = button(segment.label, () => {
            const node = flow.nodes.find(item => item.tensor === segment.tensor);
            if (node) chooseNode(flow, node);
          }, { className: `allocation-segment ${segment.kind}${step >= segment.start && step < segment.end ? " at-cursor" : ""}`, "data-start": segment.start, "data-end": segment.end, "data-focus-key": `allocation-${flow.id}-${allocation.id}-${segment.start}`, style: `left:${segment.start / count * 100}%;width:${(segment.end - segment.start) / count * 100}%`, title: `${segment.label}: boundaries ${segment.start + 1} through ${segment.end}; ${segment.kind}` });
          return segmentButton;
        }),
      ),
      el("p", { className: "allocation-detail" }, allocation.detail),
    )),
    el("div", { className: "lifetime-legend" }, el("span", { className: "live" }, "logical use"), el("span", { className: "retained" }, "retained backing"), el("span", { className: "inflight" }, "in-flight obligation"), el("span", { className: "unknown" }, "unknown")),
  );
}

function download(name: string, contents: string, type = "application/json"): void {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = el("a", { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportPacket(): void {
  const packet = investigation();
  const ids = new Set(packet.sources.map(source => source.id));
  download(`work-microscope-${packet.id}.json`, encodeBundle({ schemaVersion: 1, investigations: [packet], snapshots: bundle.snapshots.filter(source => ids.has(source.id)) }));
}
function exportPlan(): void {
  const packet = investigation();
  const text = [
    `# ${packet.title} / investigation note`, "", `Scope: ${Object.values(packet.scope).join("; ")}`, "",
    `Selected variant: ${variant().label}`, `Selected value: ${selected().tensor} / ${selected().label}`, "",
    "## Hypothesis", notes.get(packet) || "No hypothesis entered.", "",
    "## Suggested discriminating measurement", packet.nextMeasurement, "",
    "## Evidence limits", ...packet.cautions.map(caution => `- ${caution}`), "",
    "## Pinned sources", ...packet.sources.map(source => `- ${source.path}:${source.startLine} at ${source.revision}`), "",
    "This note is not a benchmark result or implementation approval.",
  ].join("\n");
  download(`work-microscope-${packet.id}-hypothesis.md`, text, "text/markdown");
}

async function importPacket(file: File): Promise<void> {
  try {
    if (file.size > MAX_PACKET_BYTES) throw new Error("Packet exceeds the 2 MiB import limit.");
    const value: unknown = JSON.parse(await file.text());
    const next = value && typeof value === "object" && "investigations" in value
      ? validateBundle(value)
      : { schemaVersion: 1 as const, investigations: [validateInvestigation(value)], snapshots: [] };
    bundle = next;
    imported = true;
    notice = "Imported packet. Annotations and excerpts are untrusted supplied evidence, not authenticated measurements.";
    chooseCase(next.investigations[0].id);
  } catch (reason) {
    error = `Import rejected: ${reason instanceof Error ? reason.message : String(reason)}`;
    render();
  }
}

function updateSourceOrder(): void {
  const flow = variant();
  const range = document.querySelector<HTMLInputElement>("#step-range")!;
  step = Number(range.value);
  range.setAttribute("aria-valuetext", `${step + 1}: ${flow.steps[step].label}; source order, not time`);
  document.querySelector("#step-counter")!.textContent = `${step + 1} / ${flow.steps.length} boundaries; not clock time`;
  document.querySelector("#step-description")!.replaceChildren(el("strong", {}, flow.steps[step].label), " / ", flow.steps[step].detail);
  for (const node of document.querySelectorAll<HTMLElement>(".flow-node[data-source-step]")) node.classList.toggle("future", Number(node.dataset.sourceStep) > step);
  for (const label of document.querySelectorAll<HTMLElement>(".step-label[data-source-step]")) label.classList.toggle("current", Number(label.dataset.sourceStep) === step);
  for (const cursor of document.querySelectorAll<SVGElement>("[data-order-cursor]")) {
    const length = investigation().variants.find(item => item.id === cursor.dataset.orderCursor)!.steps.length;
    const x = String(14 + (Math.min(step, length - 1) + 1) * 154 - 3);
    cursor.setAttribute("x1", x);
    cursor.setAttribute("x2", x);
  }
  for (const segment of document.querySelectorAll<HTMLElement>(".allocation-segment")) segment.classList.toggle("at-cursor", step >= Number(segment.dataset.start) && step < Number(segment.dataset.end));
  rememberLocation();
}

function render(): void {
  const focusedKey = document.activeElement?.getAttribute("data-focus-key");
  const scrollPositions = new Map([...document.querySelectorAll<HTMLElement>("[data-graph-scroll]")].map(node => [node.dataset.graphScroll, node.scrollLeft]));
  const packet = investigation();
  caseId = packet.id;
  const flow = variant();
  variantId = flow.id;
  nodeId = selected().id;
  step = Math.min(step, flow.steps.length - 1);
  rememberLocation();

  const fileInput = el("input", { type: "file", accept: ".json,application/json", hidden: true, "data-import": true, "aria-label": "Import annotated investigation JSON" });
  fileInput.addEventListener("change", () => { const file = fileInput.files?.[0]; if (file) void importPacket(file); });
  const reset = button("Bundled examples", () => { bundle = original; imported = false; notice = "Bundled examples restored."; chooseCase(original.investigations[0].id); }, { className: "quiet" });
  const header = el("header", { className: "masthead" },
    el("a", { className: "brand", href: "#workspace" }, el("span", { className: "reticle", "aria-hidden": "true" }, "+"), el("span", {}, "qwen-llm", el("strong", {}, "Work Microscope"))),
    el("div", { className: "header-actions" }, el("span", { className: "local-status" }, "OFFLINE / NO INFERENCE"), imported && reset, button("Import packet", () => fileInput.click(), { className: "quiet", "data-focus-key": "import" }), button("Export packet", exportPacket, { className: "quiet", "data-export": true }), fileInput),
  );
  const navigation = el("nav", { className: "case-nav", "aria-label": "Investigations" },
    el("span", { className: "eyebrow" }, "Investigations / select a pattern"),
    ...bundle.investigations.map((item, index) => {
      const choice = button("", () => chooseCase(item.id), { className: "case-choice", "data-case-id": item.id, "data-focus-key": `case-${item.id}`, "aria-current": item.id === packet.id ? "page" : undefined });
      choice.append(el("span", { className: "case-number" }, String(index + 1).padStart(2, "0")), el("strong", {}, item.title), el("span", { className: "case-subtitle" }, item.subtitle));
      return choice;
    }),
    el("div", { className: "nav-note" }, el("span", { className: "reticle", "aria-hidden": "true" }, "+"), el("p", {}, "Large traffic is a question. Exposed cost makes it an opportunity.")),
    el("details", { className: "guide" }, el("summary", {}, "How to use the instrument"), el("ol", {}, el("li", {}, "Select a value or an operation to inspect its immediate access neighborhood."), el("li", {}, "Follow its tensor identity across representations, then compare variants."), el("li", {}, "Scrub source order to inspect schematic storage use. This is not a runtime replay."), el("li", {}, "Read the cost evidence and its limits before writing a hypothesis."))),
  );
  const scope = el("details", { className: "scope" }, el("summary", {}, `${packet.scope.family} / ${packet.scope.phase} / ${packet.scope.device}`), el("dl", {}, ...Object.entries(packet.scope).flatMap(([key, value]) => [el("dt", {}, key), el("dd", {}, value)])));
  const metrics = el("section", { className: "metrics", "aria-label": "Cost and work evidence" }, ...packet.metrics.map(metric => el("div", { className: "metric" }, el("div", { className: "metric-label" }, metric.label, kindBadge(evidenceFor(metric.evidenceId).kind)), el("p", { className: "metric-value" }, metric.value, el("span", {}, metric.unit)), el("p", { className: "metric-detail" }, metric.detail))));
  const comparison = el("input", { type: "checkbox", checked: compare, "data-compare": true, "data-focus-key": "compare" });
  comparison.addEventListener("change", () => { compare = comparison.checked; render(); });
  const following = el("input", { type: "checkbox", checked: follow, "data-follow": true, "data-focus-key": "follow" });
  following.addEventListener("change", () => { follow = following.checked; render(); });
  const range = el("input", { id: "step-range", type: "range", min: 0, max: flow.steps.length - 1, step: 1, value: step, "aria-label": "Source-order boundary, not elapsed time", "aria-valuetext": `${step + 1}: ${flow.steps[step].label}; source order, not time`, "data-focus-key": "step" });
  range.addEventListener("input", updateSourceOrder);
  const controls = el("div", { className: "flow-controls" },
    el("div", { className: "variant-picker", "aria-label": "Execution variant" }, ...packet.variants.map(item => button(item.label, () => chooseVariant(item.id), { "data-variant-id": item.id, "aria-pressed": String(item.id === flow.id), "data-focus-key": `variant-${item.id}` }))),
    el("div", { className: "checkboxes" }, el("label", {}, following, "Follow tensor"), el("label", {}, comparison, "Compare")),
  );
  const scrubber = el("div", { className: "scrubber" }, el("div", { className: "scrubber-label" }, el("label", { for: "step-range" }, "Source order"), el("span", { id: "step-counter" }, `${step + 1} / ${flow.steps.length} boundaries; not clock time`)), range, el("p", { id: "step-description" }, el("strong", {}, flow.steps[step].label), " / ", flow.steps[step].detail));
  const graphs = el("div", { className: "graphs" }, controls, scrubber, ...(compare ? packet.variants : [flow]).map(renderGraph), el("p", { className: "graph-legend" }, "Solid values / dashed operations / dashed view and reuse edges. Select a node to inspect its evidence."));
  const noteInput = el("textarea", { id: "hypothesis", rows: 4, placeholder: "I suspect this representation or lifetime is avoidable because... A discriminating experiment would be..." }, notes.get(packet) ?? "");
  noteInput.addEventListener("input", () => { notes.set(packet, noteInput.value); });
  const findings = el("section", { className: "finding-panel" },
    el("span", { className: "eyebrow" }, "From pattern to experiment"), el("h2", {}, "Would removing it matter?"), el("p", { className: "finding" }, packet.finding),
    el("div", { className: "measurement" }, el("h3", {}, "The next discriminating measurement"), el("p", {}, packet.nextMeasurement)),
    el("details", { className: "limits", open: true }, el("summary", {}, "Limits on this conclusion"), el("ul", {}, ...packet.cautions.map(caution => el("li", {}, caution)))),
    el("label", { for: "hypothesis", className: "note-label" }, "Your hypothesis"), noteInput,
    el("div", { className: "note-footer" }, el("p", { className: "small-note" }, "Notes stay in this tab. Download to retain them; no experiment is launched."), button("Download investigation note", exportPlan, { className: "primary" })),
  );
  const ledger = el("details", { className: "evidence-ledger" }, el("summary", {}, "Evidence ledger", el("span", {}, `${packet.evidence.length} annotations / ${packet.sources.length} sources`)), ...packet.evidence.map(evidenceRow));
  const main = el("main", { id: "workspace", className: "workspace", tabindex: "-1" },
    el("div", { className: "page-heading" }, el("span", { className: "eyebrow" }, "An instrument for discovering optimization opportunities"), el("h1", {}, "Follow the ", el("em", {}, "work."))),
    el("div", { className: "investigation-heading" }, el("h2", {}, packet.title), el("p", {}, packet.question)), scope,
    el("p", { className: "reconstruction" }, el("strong", {}, "Source reconstruction, not a live trace. "), packet.reconstruction),
    metrics,
    el("section", { className: "microscope", "aria-label": "Linked dataflow and evidence inspector" }, graphs, renderInspector()),
    renderLifetimes(), findings, ledger,
  );
  const dialog = el("dialog", { id: "source-dialog", "aria-label": "Pinned source excerpt" });
  root.replaceChildren(header,
    ...error ? [el("div", { role: "alert", className: "error" }, error)] : [],
    ...notice ? [el("div", { role: "status", className: "import-notice" }, notice)] : [],
    el("div", { className: "shell" }, navigation, main),
    el("footer", {}, el("span", {}, "WORK MICROSCOPE / EXPERIMENT 01"), el("span", {}, "Measured != derived != unknown. No models, telemetry uploads, or external assets.")), dialog,
  );
  for (const container of document.querySelectorAll<HTMLElement>("[data-graph-scroll]")) container.scrollLeft = scrollPositions.get(container.dataset.graphScroll) ?? 0;
  if (focusedKey) document.querySelector<HTMLElement>(`[data-focus-key="${CSS.escape(focusedKey)}"]`)?.focus({ preventScroll: true });
}

async function start(): Promise<void> {
  try {
    const response = await fetch("./evidence.json");
    if (!response.ok) throw new Error(`Evidence bundle returned HTTP ${response.status}`);
    original = validateBundle(await response.json());
    bundle = original;
    readLocation();
    render();
    window.addEventListener("hashchange", () => { if (location.hash !== "#workspace") { readLocation(); render(); } });
    document.addEventListener("click", event => {
      if (event.target instanceof Element && event.target.closest('a[href="#workspace"]')) {
        event.preventDefault();
        document.querySelector<HTMLElement>("#workspace")?.focus();
      }
    });
  } catch (reason) {
    root.replaceChildren(el("main", { className: "fatal", role: "alert" }, el("h1", {}, "The instrument could not open."), el("p", {}, reason instanceof Error ? reason.message : String(reason)), el("p", {}, "Run bun run build, then bun run preview from tools/work-microscope.")));
  }
}
void start();

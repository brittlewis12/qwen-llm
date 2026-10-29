export type EvidenceKind = "measured" | "derived" | "curated" | "unknown";
export type SourceRef = {
  id: string;
  path: string;
  revision: string;
  startLine: number;
  endLine: number;
  label: string;
};
export type Evidence = {
  id: string;
  kind: EvidenceKind;
  label: string;
  detail: string;
  sourceIds: string[];
};
export type Metric = {
  label: string;
  value: string;
  unit: string;
  detail: string;
  evidenceId: string;
};
export type FlowNode = {
  id: string;
  label: string;
  detail: string;
  tensor: string;
  dtype: string;
  bytes: number | null;
  step: number;
  lane: number;
  role: "value" | "operation";
  evidenceIds: string[];
};
export type FlowEdge = {
  id: string;
  from: string;
  to: string;
  label: string;
  kind: "view" | "copy" | "convert" | "read" | "write" | "reuse";
  logicalBytes: number | null;
  repeat: number;
  evidenceIds: string[];
};
export type Allocation = {
  id: string;
  label: string;
  bytes: number | null;
  detail: string;
  evidenceIds: string[];
  segments: {
    start: number;
    end: number;
    label: string;
    tensor: string;
    kind: "live" | "retained" | "inflight" | "unknown";
  }[];
};
export type FlowVariant = {
  id: string;
  label: string;
  summary: string;
  steps: { label: string; detail: string }[];
  nodes: FlowNode[];
  edges: FlowEdge[];
  allocations: Allocation[];
};
export type Investigation = {
  schemaVersion: 1;
  id: string;
  title: string;
  subtitle: string;
  question: string;
  reconstruction: string;
  scope: {
    family: string;
    phase: string;
    artifact: string;
    device: string;
    workload: string;
    regime: string;
    revision: string;
  };
  sources: SourceRef[];
  evidence: Evidence[];
  metrics: Metric[];
  variants: FlowVariant[];
  finding: string;
  nextMeasurement: string;
  cautions: string[];
};
export type SourceSnapshot = SourceRef & { sha256: string; excerpt: string };
export type EvidenceBundle = {
  schemaVersion: 1;
  investigations: Investigation[];
  snapshots: SourceSnapshot[];
};

export function formatBytes(bytes: number | null): string {
  if (bytes === null) return "Not priced";
  if (bytes === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${Number((bytes / 1024 ** index).toFixed(2))} ${units[index]}`;
}

export function neighborhood(variant: FlowVariant, nodeId: string): Set<string> {
  const ids = new Set([nodeId]);
  for (const edge of variant.edges) {
    if (edge.from === nodeId || edge.to === nodeId) {
      ids.add(edge.from);
      ids.add(edge.to);
    }
  }
  return ids;
}

export function tensorPath(variant: FlowVariant, tensor: string): Set<string> {
  return new Set(variant.nodes.filter(node => node.tensor === tensor).map(node => node.id));
}

export function edgeBytes(edge: FlowEdge): number | null {
  return edge.logicalBytes === null ? null : edge.logicalBytes * edge.repeat;
}

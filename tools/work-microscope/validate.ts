import type { EvidenceBundle, Investigation, SourceRef } from "./model";

export const MAX_PACKET_BYTES = 2 * 1024 * 1024;
type RecordValue = Record<string, unknown>;

function fail(at: string, message: string): never {
  throw new Error(`${at}: ${message}`);
}

function packet(value: unknown): void {
  let text: string | undefined;
  try {
    text = JSON.stringify(value);
  } catch {
    fail("packet", "must be JSON-serializable (no cycles or unsupported numbers)");
  }
  if (text === undefined || new TextEncoder().encode(text).byteLength > MAX_PACKET_BYTES) {
    fail("packet", "must be JSON and at most 2 MiB");
  }
}

function object(value: unknown, at: string, keys: string[]): RecordValue {
  if (value === null || typeof value !== "object" || Array.isArray(value)
    || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) {
    fail(at, "expected a plain object");
  }
  const result = value as RecordValue;
  for (const key of Object.keys(result)) {
    if (!keys.includes(key)) fail(`${at}.${key}`, "unknown field");
  }
  for (const key of keys) {
    if (!Object.hasOwn(result, key)) fail(`${at}.${key}`, "required field missing");
  }
  return result;
}

function text(value: unknown, at: string, max = 4096): string {
  if (typeof value !== "string" || !value.trim() || value.length > max
    || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(value)) {
    fail(at, `expected nonempty text, at most ${max} characters, without control characters`);
  }
  return value;
}

function id(value: unknown, at: string): string {
  const result = text(value, at, 96);
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(result)) fail(at, "invalid id");
  return result;
}

function integer(value: unknown, at: string, min: number, max = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min || value > max) {
    fail(at, `expected safe integer in ${min}..${max}`);
  }
  return value;
}

function bytes(value: unknown, at: string): number | null {
  return value === null ? null : integer(value, at, 0);
}

function choice<T extends string>(value: unknown, at: string, choices: readonly T[]): T {
  if (typeof value !== "string" || !choices.includes(value as T)) fail(at, `expected ${choices.join(" | ")}`);
  return value as T;
}

function array<T>(value: unknown, at: string, min: number, max: number, parse: (v: unknown, at: string) => T): T[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) fail(at, `expected array length ${min}..${max}`);
  for (let i = 0; i < value.length; i++) if (!Object.hasOwn(value, i)) fail(at, "sparse arrays are not supported");
  return value.map((v, i) => parse(v, `${at}[${i}]`));
}

function unique<T extends { id: string }>(values: T[], at: string): Set<string> {
  const ids = new Set<string>();
  for (const value of values) {
    if (ids.has(value.id)) fail(at, `duplicate id ${value.id}`);
    ids.add(value.id);
  }
  return ids;
}

function references(value: unknown, at: string, known: Set<string>, min = 1): string[] {
  const result = array(value, at, min, 32, id);
  if (new Set(result).size !== result.length) fail(at, "duplicate reference");
  for (const ref of result) if (!known.has(ref)) fail(at, `unknown reference ${ref}`);
  return result;
}

function revision(value: unknown, at: string): string {
  const result = text(value, at, 40);
  if (!/^[0-9a-f]{40}$/.test(result)) fail(at, "expected full 40-character lowercase Git SHA");
  return result;
}

function source(value: unknown, at: string): SourceRef {
  const v = object(value, at, ["id", "path", "revision", "startLine", "endLine", "label"]);
  const path = text(v.path, `${at}.path`, 512);
  if (path.split("/").some(part => !/^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/.test(part))) {
    fail(`${at}.path`, "expected restricted repo-relative path (no traversal, hidden segments, URL or absolute path)");
  }
  const startLine = integer(v.startLine, `${at}.startLine`, 1, 10_000_000);
  const endLine = integer(v.endLine, `${at}.endLine`, startLine, Math.min(startLine + 99, 10_000_000));
  return { id: id(v.id, `${at}.id`), path, revision: revision(v.revision, `${at}.revision`), startLine, endLine, label: text(v.label, `${at}.label`, 240) };
}

function investigation(value: unknown, at: string): Investigation {
  const v = object(value, at, ["schemaVersion", "id", "title", "subtitle", "question", "reconstruction", "scope", "sources", "evidence", "metrics", "variants", "finding", "nextMeasurement", "cautions"]);
  if (v.schemaVersion !== 1) fail(`${at}.schemaVersion`, "unsupported version; expected 1");
  const scopeKeys = ["family", "phase", "artifact", "device", "workload", "regime", "revision"];
  const s = object(v.scope, `${at}.scope`, scopeKeys);
  const scope = {
    family: text(s.family, `${at}.scope.family`, 240), phase: text(s.phase, `${at}.scope.phase`, 240),
    artifact: text(s.artifact, `${at}.scope.artifact`, 512), device: text(s.device, `${at}.scope.device`, 240),
    workload: text(s.workload, `${at}.scope.workload`, 1024), regime: text(s.regime, `${at}.scope.regime`, 1024),
    revision: revision(s.revision, `${at}.scope.revision`),
  };
  const sources = array(v.sources, `${at}.sources`, 0, 48, source);
  const sourceIds = unique(sources, `${at}.sources`);
  const evidence = array(v.evidence, `${at}.evidence`, 1, 64, (value, at) => {
    const e = object(value, at, ["id", "kind", "label", "detail", "sourceIds"]);
    const kind = choice(e.kind, `${at}.kind`, ["measured", "derived", "curated", "unknown"] as const);
    return { id: id(e.id, `${at}.id`), kind, label: text(e.label, `${at}.label`, 240), detail: text(e.detail, `${at}.detail`), sourceIds: references(e.sourceIds, `${at}.sourceIds`, sourceIds, kind === "unknown" ? 0 : 1) };
  });
  const evidenceIds = unique(evidence, `${at}.evidence`);
  const metrics = array(v.metrics, `${at}.metrics`, 0, 16, (value, at) => {
    const m = object(value, at, ["label", "value", "unit", "detail", "evidenceId"]);
    const evidenceId = id(m.evidenceId, `${at}.evidenceId`);
    if (!evidenceIds.has(evidenceId)) fail(`${at}.evidenceId`, `unknown reference ${evidenceId}`);
    return { label: text(m.label, `${at}.label`, 240), value: text(m.value, `${at}.value`, 160), unit: text(m.unit, `${at}.unit`, 80), detail: text(m.detail, `${at}.detail`), evidenceId };
  });
  const variants = array(v.variants, `${at}.variants`, 1, 4, (value, at) => {
    const f = object(value, at, ["id", "label", "summary", "steps", "nodes", "edges", "allocations"]);
    const steps = array(f.steps, `${at}.steps`, 1, 6, (value, at) => {
      const s = object(value, at, ["label", "detail"]);
      return { label: text(s.label, `${at}.label`, 120), detail: text(s.detail, `${at}.detail`) };
    });
    const nodes = array(f.nodes, `${at}.nodes`, 1, 10, (value, at) => {
      const n = object(value, at, ["id", "label", "detail", "tensor", "dtype", "bytes", "step", "lane", "role", "evidenceIds"]);
      return {
        id: id(n.id, `${at}.id`), label: text(n.label, `${at}.label`, 120), detail: text(n.detail, `${at}.detail`),
        tensor: text(n.tensor, `${at}.tensor`, 160), dtype: text(n.dtype, `${at}.dtype`, 80), bytes: bytes(n.bytes, `${at}.bytes`),
        step: integer(n.step, `${at}.step`, 0, steps.length - 1), lane: integer(n.lane, `${at}.lane`, 0, 2),
        role: choice(n.role, `${at}.role`, ["value", "operation"] as const), evidenceIds: references(n.evidenceIds, `${at}.evidenceIds`, evidenceIds),
      };
    });
    const nodeIds = unique(nodes, `${at}.nodes`);
    const cells = new Set<string>();
    for (const n of nodes) {
      const cell = `${n.step}:${n.lane}`;
      if (cells.has(cell)) fail(`${at}.nodes`, `overlapping step/lane ${cell}`);
      cells.add(cell);
    }
    const edges = array(f.edges, `${at}.edges`, 0, 24, (value, at) => {
      const e = object(value, at, ["id", "from", "to", "label", "kind", "logicalBytes", "repeat", "evidenceIds"]);
      const from = id(e.from, `${at}.from`), to = id(e.to, `${at}.to`);
      if (!nodeIds.has(from) || !nodeIds.has(to)) fail(at, "edge endpoint references unknown node");
      if (from === to) fail(at, "self edges are not supported");
      if (nodes.find(n => n.id === from)!.step > nodes.find(n => n.id === to)!.step) fail(at, "edge runs backward in source order");
      const logicalBytes = bytes(e.logicalBytes, `${at}.logicalBytes`);
      const repeat = integer(e.repeat, `${at}.repeat`, 1);
      if (logicalBytes !== null && !Number.isSafeInteger(logicalBytes * repeat)) fail(at, "logicalBytes * repeat exceeds safe integer range");
      return { id: id(e.id, `${at}.id`), from, to, label: text(e.label, `${at}.label`, 240), kind: choice(e.kind, `${at}.kind`, ["view", "copy", "convert", "read", "write", "reuse"] as const), logicalBytes, repeat, evidenceIds: references(e.evidenceIds, `${at}.evidenceIds`, evidenceIds) };
    });
    unique(edges, `${at}.edges`);
    const reachable = new Set([nodes[0]!.id]);
    for (let i = 0; i < nodes.length; i++) for (const e of edges) {
      if (reachable.has(e.from) || reachable.has(e.to)) { reachable.add(e.from); reachable.add(e.to); }
    }
    if (reachable.size !== nodes.length) fail(`${at}.edges`, "graph must be connected; isolated nodes are not supported");
    const visiting = new Set<string>(), visited = new Set<string>();
    const visit = (node: string) => {
      if (visiting.has(node)) fail(`${at}.edges`, "cycle in source-order graph");
      if (visited.has(node)) return;
      visiting.add(node);
      for (const edge of edges.filter(e => e.from === node)) visit(edge.to);
      visiting.delete(node); visited.add(node);
    };
    for (const n of nodes) visit(n.id);
    const allocations = array(f.allocations, `${at}.allocations`, 0, 12, (value, at) => {
      const a = object(value, at, ["id", "label", "bytes", "detail", "evidenceIds", "segments"]);
      const segments = array(a.segments, `${at}.segments`, 1, 12, (value, at) => {
        const s = object(value, at, ["start", "end", "label", "tensor", "kind"]);
        const start = integer(s.start, `${at}.start`, 0, steps.length - 1);
        return { start, end: integer(s.end, `${at}.end`, start + 1, steps.length), label: text(s.label, `${at}.label`, 160), tensor: text(s.tensor, `${at}.tensor`, 160), kind: choice(s.kind, `${at}.kind`, ["live", "retained", "inflight", "unknown"] as const) };
      });
      for (const segment of segments) if (!nodes.some(n => n.tensor === segment.tensor)) fail(`${at}.segments`, `unknown tensor ${segment.tensor}`);
      for (let i = 1; i < segments.length; i++) if (segments[i]!.start < segments[i - 1]!.end) fail(`${at}.segments`, "allocation segments must be ordered and non-overlapping");
      return { id: id(a.id, `${at}.id`), label: text(a.label, `${at}.label`, 160), bytes: bytes(a.bytes, `${at}.bytes`), detail: text(a.detail, `${at}.detail`), evidenceIds: references(a.evidenceIds, `${at}.evidenceIds`, evidenceIds), segments };
    });
    unique(allocations, `${at}.allocations`);
    return { id: id(f.id, `${at}.id`), label: text(f.label, `${at}.label`, 160), summary: text(f.summary, `${at}.summary`), steps, nodes, edges, allocations };
  });
  unique(variants, `${at}.variants`);
  return {
    schemaVersion: 1, id: id(v.id, `${at}.id`), title: text(v.title, `${at}.title`, 240), subtitle: text(v.subtitle, `${at}.subtitle`, 512),
    question: text(v.question, `${at}.question`), reconstruction: text(v.reconstruction, `${at}.reconstruction`), scope, sources, evidence, metrics, variants,
    finding: text(v.finding, `${at}.finding`), nextMeasurement: text(v.nextMeasurement, `${at}.nextMeasurement`), cautions: array(v.cautions, `${at}.cautions`, 1, 16, text),
  };
}

export function validateInvestigation(value: unknown): Investigation {
  packet(value);
  return investigation(value, "investigation");
}

export function validateBundle(value: unknown): EvidenceBundle {
  packet(value);
  const b = object(value, "bundle", ["schemaVersion", "investigations", "snapshots"]);
  if (b.schemaVersion !== 1) fail("bundle.schemaVersion", "unsupported version; expected 1");
  const investigations = array(b.investigations, "bundle.investigations", 1, 8, investigation);
  unique(investigations, "bundle.investigations");
  const sources = investigations.flatMap(i => i.sources);
  unique(sources, "bundle.sources (globally unique)");
  const snapshots = array(b.snapshots, "bundle.snapshots", 0, 384, (value, at) => {
    const s = object(value, at, ["id", "path", "revision", "startLine", "endLine", "label", "sha256", "excerpt"]);
    const ref = source(Object.fromEntries(["id", "path", "revision", "startLine", "endLine", "label"].map(k => [k, s[k]])), at);
    const expected = sources.find(source => source.id === ref.id);
    if (!expected || Object.entries(ref).some(([k, v]) => expected[k as keyof SourceRef] !== v)) fail(at, "snapshot does not exactly match a referenced source");
    const sha256 = text(s.sha256, `${at}.sha256`, 64);
    if (!/^[0-9a-f]{64}$/.test(sha256)) fail(`${at}.sha256`, "expected 64-character lowercase SHA-256 (format only, not cryptographic verification)");
    const excerpt = text(s.excerpt, `${at}.excerpt`, 65_536);
    if (excerpt.split("\n").length !== ref.endLine - ref.startLine + 1) fail(`${at}.excerpt`, "line count does not match source range");
    return { ...ref, sha256, excerpt };
  });
  unique(snapshots, "bundle.snapshots");
  return { schemaVersion: 1, investigations, snapshots };
}

export function encodeBundle(value: EvidenceBundle): string {
  return JSON.stringify(validateBundle(value));
}

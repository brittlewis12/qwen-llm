import { beforeAll, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { investigations } from "./cases";
import { buildEvidenceBundle } from "./evidence";
import { edgeBytes, formatBytes, neighborhood, tensorPath } from "./model";
import type { EvidenceBundle, Investigation } from "./model";
import { encodeBundle, MAX_PACKET_BYTES, validateBundle, validateInvestigation } from "./validate";

const repoRoot = resolve(import.meta.dir, "../..");
const query = investigations[0]!;
const scratch = investigations[1]!;
const bridge = investigations[2]!;
const clone = <T>(value: T): T => structuredClone(value);
const bare = (): EvidenceBundle => ({ schemaVersion: 1, investigations: clone(investigations), snapshots: [] });

function changed(path: string, value: unknown): unknown {
  const packet = clone(query);
  const keys = path.split(".");
  let parent = packet as unknown as Record<string, unknown>;
  for (const key of keys.slice(0, -1)) parent = parent[key] as Record<string, unknown>;
  parent[keys.at(-1)!] = value;
  return packet;
}

describe("curated geometry and evidence contract", () => {
  test("all three packets and a snapshot-free bundle validate and normalize", () => {
    expect(investigations.map(i => i.id)).toEqual(["query-half", "scratch-reuse", "bridge-copy"]);
    for (const packet of investigations) expect(validateInvestigation(packet)).toEqual(packet);
    expect(validateBundle(bare()).snapshots).toEqual([]);
    expect(validateInvestigation(clone(query))).not.toBe(query);
  });

  test("bounded two-variant graphs use synchronized tensor keys", () => {
    for (const packet of investigations) {
      expect(packet.variants).toHaveLength(2);
      const [before, after] = packet.variants;
      expect(before!.steps.length).toBe(after!.steps.length);
      expect(before!.steps.length).toBeGreaterThanOrEqual(4);
      for (const variant of packet.variants) {
        expect(variant.steps.length).toBeLessThanOrEqual(6);
        expect(variant.nodes.length).toBeLessThanOrEqual(10);
        const cells = variant.nodes.map(n => `${n.step}/${n.lane}`);
        expect(new Set(cells).size).toBe(cells.length);
        for (const n of variant.nodes) {
          expect(n.step).toBeLessThan(variant.steps.length);
          expect(n.lane).toBeGreaterThanOrEqual(0);
          expect(n.lane).toBeLessThanOrEqual(2);
          expect(tensorPath(after!, n.tensor).size).toBeGreaterThan(0);
        }
      }
      expect(packet.reconstruction).toContain("not a captured live trace");
      for (const allocation of packet.variants.flatMap(v => v.allocations)) {
        expect(allocation.detail).toContain("not a measured lifetime");
      }
    }
  });

  test("every metric has evidence; measured, derived and curated evidence is sourced", () => {
    const sourceIds = investigations.flatMap(i => i.sources.map(s => s.id));
    expect(new Set(sourceIds).size).toBe(sourceIds.length);
    for (const packet of investigations) {
      for (const metric of packet.metrics) expect(packet.evidence.some(e => e.id === metric.evidenceId)).toBe(true);
      for (const evidence of packet.evidence) {
        if (evidence.kind !== "unknown") expect(evidence.sourceIds.length).toBeGreaterThan(0);
        for (const id of evidence.sourceIds) expect(packet.sources.some(s => s.id === id)).toBe(true);
      }
      for (const ref of packet.sources) {
        expect(ref.revision).toMatch(/^[0-9a-f]{40}$/);
        expect(ref.endLine - ref.startLine + 1).toBeLessThanOrEqual(100);
        if (ref.path.startsWith("docs/")) expect(ref.revision).toBe("ad7e3b9acbffb704a233b97e48e52b56c2c617a9");
      }
    }
  });

  test("query arithmetic distinguishes one conversion from twenty score reads", () => {
    const [before, after] = query.variants;
    const input = before!.edges.find(e => e.id === "qb-input")!;
    const halfRead = after!.edges.find(e => e.id === "qa-panel")!;
    expect(before!.nodes.find(n => n.id === "qb-q")!.bytes).toBe(64 * 512 * 4);
    expect(after!.nodes.find(n => n.id === "qa-half")!.bytes).toBe(64 * 512 * 2);
    expect(input.repeat).toBe(640 / 32);
    expect(edgeBytes(input)).toBe(2.5 * 1024 ** 2);
    expect(halfRead.repeat).toBe(20);
    expect(edgeBytes(halfRead)).toBe(1.25 * 1024 ** 2);
    expect(edgeBytes(after!.edges.find(e => e.id === "qa-input")!)).toBe(128 * 1024);
    expect(after!.edges.find(e => e.kind === "convert")!.repeat).toBe(1);
    expect(before!.edges.find(e => e.kind === "convert")!.repeat).toBe(20);
    expect(query.metrics.find(m => m.label === "CED throughput")!.value).toBe("+2.08%");
    expect(query.metrics.find(m => m.label === "512-query probe")!.evidenceId).toBe("q-probe");
    expect(query.evidence.find(e => e.id === "q-geometry")!.kind).toBe("derived");
    expect(before!.allocations).toEqual([]);
    expect(after!.allocations[0]!.bytes).toBeNull();
  });

  test("scratch scope remains warm, unknown-sized, reset-first and drop-first", () => {
    expect(scratch.metrics.find(m => m.label === "Warm throughput")!.value).toBe("+3.54%");
    expect(scratch.metrics.find(m => m.label === "Warm means")!.value).toBe("834.165 -> 863.725");
    expect(scratch.metrics.find(m => m.label === "Scratch total")!.evidenceId).toBe("s-size-unknown");
    const unknown = scratch.evidence.find(e => e.id === "s-size-unknown")!;
    expect(unknown.kind).toBe("unknown");
    expect(unknown.sourceIds).toEqual([]);
    for (const v of scratch.variants) {
      expect(v.nodes.every(n => n.bytes === null)).toBe(true);
      expect(v.allocations.every(a => a.bytes === null)).toBe(true);
      expect(v.steps[0]!.detail).toContain("before the first chunk");
      expect(v.nodes.find(n => n.id.endsWith("prepare-b"))!.detail).toContain("every chunk start");
    }
    expect(scratch.variants[1]!.steps[4]!.detail).toContain("Drop the max-row set before allocating");
    expect(scratch.variants[1]!.edges.filter(e => e.kind === "reuse")).toHaveLength(1);
    expect(scratch.variants[0]!.edges.filter(e => e.kind === "reuse")).toHaveLength(0);
    const full = scratch.variants[1]!.allocations.find(a => a.id === "s-full")!;
    const tail = scratch.variants[1]!.allocations.find(a => a.id === "s-tail")!;
    expect(full.segments.at(-1)!.end).toBe(tail.segments[0]!.start);
  });

  test("bridge payload and logical R/W are distinct; observation-only split has unchanged flow", () => {
    const [control, split] = bridge.variants;
    for (const variant of bridge.variants) {
      const copies = variant.edges.filter(e => e.kind === "copy");
      expect(copies).toHaveLength(2);
      expect(copies.reduce((n, e) => n + e.repeat, 0)).toBe(92);
      expect(copies.every(e => e.logicalBytes === 2560 * 512 * 4)).toBe(true);
      const payload = copies.reduce((n, e) => n + edgeBytes(e)!, 0);
      expect(payload).toBe(460 * 1024 ** 2);
      expect(payload * 2).toBe(920 * 1024 ** 2);
      expect(variant.allocations[0]!.bytes).toBe(5 * 1024 ** 2);
    }
    const shape = (v: typeof control) => v!.edges.map(e => [e.kind, e.logicalBytes, e.repeat]);
    expect(shape(control)).toEqual(shape(split));
    expect(bridge.metrics.find(m => m.label === "Profiled GPU share")!.value).toBe("0.175%");
    expect(bridge.metrics.find(m => m.label === "Adjusted projection")!.value).toBe("1.693810 / 1.697572");
    expect(bridge.evidence.find(e => e.id === "b-projection")!.kind).toBe("derived");
    expect(bridge.evidence.find(e => e.id === "b-timing")!.kind).toBe("measured");
    expect(1.703394 - 0.009584).toBeCloseTo(1.693810, 6);
    expect(1.697572 - 0).toBeCloseTo(1.697572, 6);
    expect(bridge.finding).toContain("no demonstrated model speedup");
  });

  test("tensor selection and one-hop neighborhood do not silently expand to whole graph", () => {
    const v = query.variants[1]!;
    expect([...tensorPath(v, "Query Q")]).toEqual(["qa-q", "qa-stage", "qa-half"]);
    expect([...neighborhood(v, "qa-score")].sort()).toEqual(["qa-half", "qa-kv", "qa-score", "qa-scores"].sort());
    expect(tensorPath(v, "not-a-tensor").size).toBe(0);
    expect(formatBytes(null)).toBe("Not priced");
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(131072)).toBe("128 KiB");
    expect(edgeBytes({ ...v.edges[0]!, logicalBytes: null })).toBeNull();
  });
});

describe("untrusted investigation validation", () => {
  const bad: [string, string, unknown, RegExp][] = [
    ["version", "schemaVersion", 2, /unsupported version/],
    ["short revision", "scope.revision", "ad7e3b9a", /40-character/],
    ["title bound", "title", "x".repeat(241), /240/],
    ["empty title", "title", " ", /nonempty/],
    ["control text", "title", "bad\u0000text", /control characters/],
    ["unsafe id", "id", "x<script>", /invalid id/],
    ["unknown field", "onload", "alert(1)", /unknown field/],
    ["negative bytes", "variants.0.nodes.0.bytes", -1, /safe integer/],
    ["infinite bytes", "variants.0.nodes.0.bytes", Infinity, /safe integer/],
    ["NaN bytes", "variants.0.nodes.0.bytes", NaN, /safe integer/],
    ["unsafe bytes", "variants.0.nodes.0.bytes", Number.MAX_SAFE_INTEGER + 1, /safe integer/],
    ["fractional bytes", "variants.0.nodes.0.bytes", 0.5, /safe integer/],
    ["negative edge bytes", "variants.0.edges.0.logicalBytes", -1, /safe integer/],
    ["zero repeat", "variants.0.edges.0.repeat", 0, /safe integer/],
    ["negative repeat", "variants.0.edges.0.repeat", -1, /safe integer/],
    ["fractional repeat", "variants.0.edges.0.repeat", 1.1, /safe integer/],
    ["overflow product", "variants.0.edges.0.repeat", Number.MAX_SAFE_INTEGER, /product|exceeds safe/],
    ["step bound", "variants.0.nodes.0.step", 5, /safe integer/],
    ["negative step", "variants.0.nodes.0.step", -1, /safe integer/],
    ["lane bound", "variants.0.nodes.0.lane", 3, /safe integer/],
    ["unknown role", "variants.0.nodes.0.role", "html", /value \| operation/],
    ["unknown edge kind", "variants.0.edges.0.kind", "fetch", /expected/],
    ["missing node", "variants.0.edges.0.from", "absent", /unknown node/],
    ["self edge", "variants.0.edges.0.to", "qb-q", /self edges/],
    ["unknown node evidence", "variants.0.nodes.0.evidenceIds", ["absent"], /unknown reference/],
    ["empty node evidence", "variants.0.nodes.0.evidenceIds", [], /array length/],
    ["unknown metric evidence", "metrics.0.evidenceId", "absent", /unknown reference/],
    ["unknown evidence source", "evidence.0.sourceIds", ["absent"], /unknown reference/],
    ["empty curated source", "evidence.0.sourceIds", [], /array length/],
    ["duplicate source reference", "evidence.0.sourceIds", ["q-source-before", "q-source-before"], /duplicate reference/],
    ["empty measured source", "evidence.3.sourceIds", [], /array length/],
    ["empty derived source", "evidence.2.sourceIds", [], /array length/],
    ["unknown evidence kind", "evidence.0.kind", "verified", /expected/],
    ["zero start", "sources.0.startLine", 0, /safe integer/],
    ["range over 100", "sources.0.endLine", 101, /safe integer/],
    ["negative range", "sources.0.endLine", -1, /safe integer/],
    ["short SHA", "sources.0.revision", "ad7e3b9a", /40-character/],
    ["empty variants", "variants", [], /array length/],
    ["too many steps", "variants.0.steps", Array(7).fill({ label: "Step", detail: "Detail" }), /array length/],
    ["too many nodes", "variants.0.nodes", Array(11).fill(query.variants[0]!.nodes[0]), /array length/],
    ["too much evidence", "evidence", Array(65).fill(query.evidence[0]), /array length/],
    ["too many sources", "sources", Array(49).fill(query.sources[0]), /array length/],
    ["too many metrics", "metrics", Array(17).fill(query.metrics[0]), /array length/],
  ];
  test.each(bad)("rejects %s", (_label, path, value, error) => {
    expect(() => validateInvestigation(changed(path, value))).toThrow(error);
  });

  test.each(["/etc/passwd", "../secret", "docs/../secret", "docs//x", "./docs/x", ".git/config", "C:/secret", "docs\\x", "https://evil/x", "docs/x:y", "docs/%2e%2e/x", "-option", "docs/x\nfile"])("rejects dangerous source path %s", path => {
    expect(() => validateInvestigation(changed("sources.0.path", path))).toThrow(/repo-relative/);
  });

  test("requires fields; null bytes are explicit unknowns, not absent or fabricated zeroes", () => {
    const value = clone(query);
    delete (value.variants[0]!.nodes[0] as Partial<typeof value.variants[0]["nodes"][0]>).bytes;
    expect(() => validateInvestigation(value)).toThrow(/bytes.*required field/);
    const nullable = changed("variants.0.nodes.0.bytes", null);
    expect(validateInvestigation(nullable).variants[0]!.nodes[0]!.bytes).toBeNull();
    expect(validateInvestigation(changed("variants.0.nodes.0.bytes", 0)).variants[0]!.nodes[0]!.bytes).toBe(0);
    expect(() => validateInvestigation(null)).toThrow(/plain object/);
    expect(() => validateInvestigation([])).toThrow(/plain object/);
  });

  test("rejects duplicate ids within each scope", () => {
    const duplicate = (mutate: (v: Investigation) => void) => {
      const v = clone(query); mutate(v);
      expect(() => validateInvestigation(v)).toThrow(/duplicate id/);
    };
    duplicate(v => { v.sources.push(v.sources[0]!); });
    duplicate(v => { v.evidence.push(v.evidence[0]!); });
    duplicate(v => { v.variants[0]!.nodes[1]!.id = v.variants[0]!.nodes[0]!.id; });
    duplicate(v => { v.variants[0]!.edges.push(v.variants[0]!.edges[0]!); });
    duplicate(v => { v.variants.push(v.variants[0]!); });
    duplicate(v => { v.variants[1]!.allocations.push(v.variants[1]!.allocations[0]!); });
  });

  test("rejects overlapping cells, backward edges, cycles and disconnected graphs", () => {
    const v = clone(query);
    v.variants[0]!.nodes[1]!.step = 0;
    expect(() => validateInvestigation(v)).toThrow(/overlapping step\/lane/);
    const backward = clone(query);
    [backward.variants[0]!.edges[0]!.from, backward.variants[0]!.edges[0]!.to] = ["qb-stage", "qb-q"];
    expect(() => validateInvestigation(backward)).toThrow(/backward/);
    const cyclic = clone(query);
    cyclic.variants[0]!.edges.push({ ...cyclic.variants[0]!.edges[0]!, id: "cycle", from: "qb-scores", to: "qb-score" });
    cyclic.variants[0]!.nodes.find(n => n.id === "qb-scores")!.step = 3;
    cyclic.variants[0]!.nodes.find(n => n.id === "qb-scores")!.lane = 2;
    expect(() => validateInvestigation(cyclic)).toThrow(/cycle/);
    const disconnected = clone(query);
    disconnected.variants[0]!.edges = disconnected.variants[0]!.edges.filter(e => ![e.from, e.to].includes("qb-kv"));
    expect(() => validateInvestigation(disconnected)).toThrow(/connected/);
  });

  test("allocation windows require bounded, ordered ranges and known tensors", () => {
    const change = (mutate: (v: Investigation) => void, error: RegExp) => {
      const v = clone(scratch); mutate(v);
      expect(() => validateInvestigation(v)).toThrow(error);
    };
    change(v => { v.variants[0]!.allocations[0]!.segments[0]!.end = 7; }, /safe integer/);
    change(v => { v.variants[0]!.allocations[0]!.segments[0]!.end = 0; }, /safe integer/);
    change(v => { v.variants[0]!.allocations[0]!.segments[0]!.tensor = "absent"; }, /unknown tensor/);
    change(v => { v.variants[1]!.allocations[0]!.segments[1]!.start = 1; }, /non-overlapping/);
  });

  test("unknown evidence alone may have no sources; missing allocations are allowed", () => {
    const v = clone(query);
    v.evidence = v.evidence.map(e => ({ ...e, kind: "unknown", sourceIds: [] }));
    v.sources = [];
    for (const variant of v.variants) variant.allocations = [];
    expect(validateInvestigation(v).sources).toEqual([]);
    expect(() => validateInvestigation({ ...v, evidence: v.evidence.map(e => ({ ...e, kind: "measured" })) })).toThrow(/array length/);
    expect(() => validateInvestigation(changed("variants.0.nodes", Array(2)))).toThrow(/sparse/);
  });

  test("packet bound is 2 MiB; oversized and cyclic objects fail descriptively", () => {
    expect(MAX_PACKET_BYTES).toBe(2 * 1024 * 1024);
    expect(() => validateInvestigation({ ...query, reconstruction: "x".repeat(MAX_PACKET_BYTES) })).toThrow(/at most 2 MiB/);
    expect(() => validateBundle({ ...bare(), padding: "x".repeat(MAX_PACKET_BYTES) })).toThrow(/at most 2 MiB/);
    const cycle: Record<string, unknown> = {}; cycle.self = cycle;
    expect(() => validateInvestigation(cycle)).toThrow(/JSON-serializable/);
  });

  test("a near-limit export stays reimportable rather than growing with indentation", () => {
    const packet = clone(query);
    while (packet.sources.length < 32) packet.sources.push({ ...packet.sources[0]!, id: `padding-${packet.sources.length}` });
    const value: EvidenceBundle = { schemaVersion: 1, investigations: [packet], snapshots: packet.sources.map(source => ({ ...source, sha256: "0".repeat(64), excerpt: `x${"\n".repeat(source.endLine - source.startLine)}` })) };
    let remaining = MAX_PACKET_BYTES - 16 - new TextEncoder().encode(JSON.stringify(value)).byteLength;
    for (const snapshot of value.snapshots) {
      const count = Math.min(65_536 - snapshot.excerpt.length, remaining);
      snapshot.excerpt += "x".repeat(count);
      remaining -= count;
    }
    expect(remaining).toBe(0);
    const encoded = encodeBundle(value);
    expect(new TextEncoder().encode(encoded).byteLength).toBeLessThanOrEqual(MAX_PACKET_BYTES);
    expect(new TextEncoder().encode(JSON.stringify(value, null, 2)).byteLength).toBeGreaterThan(MAX_PACKET_BYTES);
    expect(validateBundle(JSON.parse(encoded))).toEqual(value);
  });
});

describe("bundle import and immutable CPU-only source snapshots", () => {
  let bundle: EvidenceBundle;
  beforeAll(async () => { bundle = await buildEvidenceBundle(repoRoot); });

  test("all refs get bounded matching snapshots and fixed full-source SHA-256 hashes", () => {
    const hashes: Record<string, string> = {
      "q-report": "0bc8b6adb0dc119f2b168af7c8edf7de9038154a6f4485d0abf034b137251c7f",
      "q-source-before": "54493a89e556149aa1ee9746bee9be6fa60db9203fb76f30a0f45d0b123b2c21",
      "q-source-after": "54493a89e556149aa1ee9746bee9be6fa60db9203fb76f30a0f45d0b123b2c21",
      "q-source-stage": "93b6dfe1323d21a0bfd1db3bd3620ab5e8a390f836077476a6e279a912d73a8d",
      "q-source-bind": "93b6dfe1323d21a0bfd1db3bd3620ab5e8a390f836077476a6e279a912d73a8d",
      "q-source-finish": "bd1289dfe8dc7fc01b3b244ede45546566912d22e3873d05d43e0c25b1b3c705",
      "s-report": "4a3b0fc6e8077cc9aee26e71b37482ddd45f5600ec28d29255a1770574f57519",
      "s-policy": "9b7d6621468dc418daa29fed482bfa490086119b8da6e33f6934f3158d3b53a7",
      "s-reset": "a82b6ed35de695b2bae56d7757cd859d0338b6a3ad3c0d58612bee6efcfe216c",
      "s-allocate": "9b7d6621468dc418daa29fed482bfa490086119b8da6e33f6934f3158d3b53a7",
      "s-state": "14ca0d35dc772fe96a9efba4059d8ee2f26cd2ffb589c835a4f9a42be1900dfb",
      "b-report": "fd0c32232902652e0ee380d974753a551b2abe851a1b766ee4760f6c1c9d0101",
      "b-source-mixer": "78024f14c992836e8e83016157339041beb017579138f74959cd015d54cfc26f",
      "b-source-moe": "78024f14c992836e8e83016157339041beb017579138f74959cd015d54cfc26f",
      "b-source-serial": "943d5288beba8edc4498f4a6c063c7071d2a50c71322eb5bbc6653496f794caa",
    };
    expect(bundle.snapshots).toHaveLength(investigations.flatMap(i => i.sources).length);
    for (const snapshot of bundle.snapshots) {
      expect(snapshot.sha256).toBe(hashes[snapshot.id]);
      expect(snapshot.excerpt.split("\n")).toHaveLength(snapshot.endLine - snapshot.startLine + 1);
      expect(snapshot.endLine - snapshot.startLine + 1).toBeLessThanOrEqual(100);
    }
    expect(validateBundle(bundle)).toEqual(bundle);
  });

  test("snapshots equal independent git blob hashing and exact source line slicing", async () => {
    for (const snapshot of bundle.snapshots) {
      const child = Bun.spawn(["git", "-C", repoRoot, "show", `${snapshot.revision}:${snapshot.path}`], { stdout: "pipe", stderr: "pipe" });
      const [buffer, exitCode] = await Promise.all([new Response(child.stdout).arrayBuffer(), child.exited]);
      expect(exitCode).toBe(0);
      expect(createHash("sha256").update(new Uint8Array(buffer)).digest("hex")).toBe(snapshot.sha256);
      const excerpt = new TextDecoder().decode(buffer).split("\n").slice(snapshot.startLine - 1, snapshot.endLine).join("\n");
      expect(snapshot.excerpt).toBe(excerpt);
    }
  });

  test("source range anchors preserve semantics rather than merely valid line numbers", () => {
    const excerpt = (id: string) => bundle.snapshots.find(s => s.id === id)!.excerpt;
    expect(excerpt("q-source-before")).toContain("device const float *q");
    expect(excerpt("q-source-before")).toContain("for (uint k0 = 0; k0 < count; k0 += V41_TT64_KEYS)");
    expect(excerpt("q-source-before")).toContain("half4(src[c4])");
    expect(excerpt("q-source-after")).toContain("device const half *q");
    expect(excerpt("q-source-after")).toContain("((threadgroup half4 *)qblk)[e] = src[c4]");
    expect(excerpt("q-source-stage")).toContain(".convert(ctx, enc, &rows.query, elements)");
    expect(excerpt("q-source-bind")).toContain('"score_tensor64_qhalf"');
    expect(excerpt("q-source-finish")).toContain(".half_input(");
    const policy = excerpt("s-policy");
    expect(policy.indexOf("scratch = None;")).toBeGreaterThan(0);
    expect(policy.indexOf("scratch = None;")).toBeLessThan(policy.indexOf("let tail = state"));
    expect(excerpt("s-reset")).toContain("write(&scratch.incoming, &[1.0, 0.0, 0.0, 0.0].repeat(n))?");
    expect(excerpt("s-allocate")).toContain("zeros_dtype_unstaged");
    expect(excerpt("b-source-mixer")).toContain("&mixer_output,");
    expect(excerpt("b-source-mixer")).toContain("attention.encode_combine(&combine_encoder)");
    expect(excerpt("b-source-moe")).toContain("&moe_output,");
    expect(excerpt("b-source-moe")).toContain("ffn.encode_combine(&final_encoder)");
    expect(excerpt("q-report")).toContain("+2.08% / 89.4 ms");
    expect(excerpt("s-report")).toContain("**+3.54%**");
    expect(excerpt("b-report")).toContain("0.175% of profiled command GPU");
  });

  test.each(["id", "path", "revision", "startLine", "endLine", "label"])("snapshot %s must match its exact source ref", key => {
    const value = clone(bundle);
    const snapshot = value.snapshots[0]!;
    const replacements: Record<string, unknown> = { id: "absent", path: "docs/other.md", revision: "a".repeat(40), startLine: 2, endLine: 83, label: "Other label" };
    (snapshot as unknown as Record<string, unknown>)[key] = replacements[key];
    expect(() => validateBundle(value)).toThrow(/exactly match/);
  });

  test("bundle metadata, references, hash format and excerpts are bounded", () => {
    expect(() => validateBundle({ ...bare(), schemaVersion: 2 })).toThrow(/unsupported version/);
    expect(() => validateBundle({ schemaVersion: 1, investigations })).toThrow(/snapshots.*required/);
    expect(() => validateBundle({ ...bare(), investigations: [] })).toThrow(/array length/);
    expect(() => validateBundle({ ...bare(), snapshots: [bundle.snapshots[0], bundle.snapshots[0]] })).toThrow(/duplicate id/);
    const duplicateCase = bare(); duplicateCase.investigations.push(clone(duplicateCase.investigations[0]!));
    expect(() => validateBundle(duplicateCase)).toThrow(/duplicate id/);
    const duplicateSource = bare(); duplicateSource.investigations[1]!.sources.push(clone(duplicateSource.investigations[0]!.sources[0]!));
    expect(() => validateBundle(duplicateSource)).toThrow(/globally unique/);
    const badHash = clone(bundle); badHash.snapshots[0]!.sha256 = "1234";
    expect(() => validateBundle(badHash)).toThrow(/64-character/);
    const longExcerpt = clone(bundle); longExcerpt.snapshots[0]!.excerpt = "x".repeat(65_537);
    expect(() => validateBundle(longExcerpt)).toThrow(/65536/);
    const missingLine = clone(bundle); missingLine.snapshots[0]!.excerpt = "just one line";
    expect(() => validateBundle(missingLine)).toThrow(/line count/);
  });

  test("import validates structure, not cryptographic authenticity", () => {
    const untrusted = clone(bundle);
    untrusted.snapshots[0]!.sha256 = "0".repeat(64);
    const imported = validateBundle(untrusted);
    expect(imported.snapshots[0]!.sha256).toBe("0".repeat(64));
    expect(Object.hasOwn(imported.snapshots[0]!, "verified")).toBe(false);
    expect(validateBundle({ ...untrusted, snapshots: [] }).snapshots).toEqual([]);
  });
});

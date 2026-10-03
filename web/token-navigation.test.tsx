import { expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { decodeResult, type Job, type Prepared } from "./contract";
import { TokenNavigator } from "./token-navigator";
import { savedSpanLabels, savedSpanContext, pieceLabel, tokenAt, tokenCoverage, adjacentToken, TOKEN_WINDOW } from "./token-navigation";

const prepared: Prepared = { seq: 0, kind: "prepared_input", token_ids: [10, 11], rendering: { renderer: "saved", spans: [
  { kind: "message_content", role: "user", message_index: 0, token_start: 0, token_end: 2 },
  { kind: "message_end_marker", role: "user", token_start: 1, token_end: 2 },
  { kind: "invalid", token_start: -1, token_end: 2 },
] }, resolved_scopes: [{ id: "r", kind: "readout", scope: { layers: { kind: "values", values: [2] }, prefill: { kind: "values", values: [1] } } }] };
const rows = [prepared,
  { seq: 1, kind: "sampled_token", index: 0, token_id: 20, text: "not authoritative", piece_bytes: [0xe2], consumed: true },
  { seq: 2, kind: "sampled_token", index: 1, token_id: 21, text: "", piece_bytes: [0x82, 0xac], consumed: false },
];

test("saved coordinates and split UTF8 bytes need no tokenizer or text guess", () => {
  expect(tokenAt(rows, { phase: "prefill", index: 1 })).toEqual({ id: 11, position: 1, consumed: null, bytes: null });
  expect(tokenAt(rows, { phase: "decode", index: 0 })).toEqual({ id: 20, position: 2, consumed: true, bytes: [0xe2] });
  expect(tokenAt(rows.slice(1), { phase: "decode", index: 1 })?.position).toBeNull();
  expect(tokenAt(rows, { phase: "decode", index: 2 })).toBeNull();
  expect(savedSpanLabels(prepared, 1)).toEqual(["Message 1 / user / message_content", "user / message_end_marker"]);
  expect(savedSpanLabels(prepared, 2)).toEqual([]);
});

test("adjacent exploration includes unobserved and terminal tokens without skipping", () => {
  expect(adjacentToken(rows, { phase: "prefill", index: 0 }, -1)).toBeNull();
  expect(adjacentToken(rows, { phase: "prefill", index: 0 }, 1)).toEqual({ phase: "prefill", index: 1 });
  expect(adjacentToken(rows, { phase: "prefill", index: 1 }, 1)).toEqual({ phase: "decode", index: 0 });
  expect(adjacentToken(rows, { phase: "decode", index: 0 }, -1)).toEqual({ phase: "prefill", index: 1 });
  expect(adjacentToken(rows, { phase: "decode", index: 0 }, 1)).toEqual({ phase: "decode", index: 1 });
  expect(adjacentToken(rows, { phase: "decode", index: 1 }, 1)).toBeNull();
});

test("coverage distinguishes omitted scope, incomplete publication, failure and unconsumed tokens", () => {
  expect(tokenCoverage(rows, { phase: "decode", index: 1 }, true)).toContain("Unconsumed sample");
  expect(tokenCoverage(rows, { phase: "prefill", index: 0 }, true)).toContain("Outside saved");
  expect(tokenCoverage(rows, { phase: "prefill", index: 1 }, false)).toContain("pending or not loaded");
  expect(tokenCoverage(rows, { phase: "prefill", index: 1 }, true)).toContain("no published record");
  expect(tokenCoverage(rows, { phase: "prefill", index: 1 }, true, { result: { error: "failure" }, observations: { error: null } } as Job)).toContain("publication failed");
  const pairOnly = { ...prepared, resolved_scopes: [], residual_pair_scopes: [{ id: "pair", scope: { layers: { kind: "all" as const }, decode: { kind: "all" as const } } }] };
  expect(tokenCoverage([pairOnly, ...rows.slice(1)], { phase: "decode", index: 0 }, false)).toContain("pending or not loaded");
  const packet = { schema_version: 1, job_id: "test", records: [pairOnly], next_cursor: null, complete: true };
  expect(decodeResult(packet).records).toHaveLength(1);
  expect(() => decodeResult({ ...packet, records: [{ ...pairOnly, residual_pair_scopes: [{ id: "p", scope: {} }] }] })).toThrow();
});

test("phone token strip bounds DOM and escapes saved annotations", () => {
  const p = { ...prepared, token_ids: Array.from({ length: 10_000 }, (_, i) => i), rendering: { renderer: "saved", spans: [{ kind: "<script>bad</script>", token_start: 0, token_end: 1 }] } };
  const html = renderToStaticMarkup(<TokenNavigator records={[p]} site={{ phase: "prefill", index: 0 }} select={() => { throw new Error("render mutates selection"); }} complete={true} />);
  expect(html.match(/aria-label="Inspect prefill token/g)).toHaveLength(TOKEN_WINDOW);
  expect(html).toContain("&lt;script&gt;"); expect(html).not.toContain("<script>bad");
});

test("readable pieces and span context never invent UTF8 replacement characters", () => {
  expect(pieceLabel([0xe2])).toBe("UTF-8 byte fragment");
  expect(pieceLabel([0xe2, 0x82, 0xac])).toBe('"\u20ac"');
  expect(pieceLabel([10, 32])).toBe('"\\n "');
  expect(pieceLabel(Array(100).fill(65))).toBe(`"${"A".repeat(48)}"...`);
  const span = { kind: "content", token_start: 0, token_end: 1, byte_start: 0, byte_end: 3 };
  expect(savedSpanContext({ ...prepared, prompt_bytes: [0xe2, 0x82, 0xac], rendering: { spans: [span] } }, 0)).toEqual(['"\u20ac"']);
  expect(savedSpanContext({ ...prepared, prompt_bytes: [0xe2, 0x82, 0xac], rendering: { spans: [{ ...span, byte_start: 1 }] } }, 0)[0]).toContain("not independently valid");
  expect(savedSpanContext({ ...prepared, prompt_bytes: [65], rendering: { spans: [span] } }, 0)).toEqual([]);
});

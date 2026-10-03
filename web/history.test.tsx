import { expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { decodeHistory, type RequestPreview } from "./contract";
import { HistoryPreview, mergeRequestPreviews } from "./history";

const root = `${import.meta.dir}/../crates/qwen-cli/tests/fixtures/lens_http_v1`;
const status = await Bun.file(`${root}/status.json`).json();
const preview: RequestPreview = await Bun.file(`${root}/request_preview.json`).json();
const page = (value: unknown) => ({ schema_version: 1, jobs: [status], next_cursor: null, request_previews: { [status.id]: value } });

test("history previews share the server fixture with exact page coverage and optional legacy fallback", () => {
  expect(decodeHistory(page(preview)).request_previews?.[status.id]).toEqual(preview);
  expect(decodeHistory(page(null)).request_previews?.[status.id]).toBeNull();
  expect(decodeHistory({ schema_version: 1, jobs: [status], next_cursor: null }).request_previews).toBeUndefined();
  for (const bad of [null, {}, { wrong: preview }, { [status.id]: preview, extra: null }]) {
    expect(() => decodeHistory({ ...page(preview), request_previews: bad })).toThrow();
  }
  expect(() => decodeHistory({ ...page(preview), jobs: [status, status] })).toThrow("duplicate");
});

test("history preview decoder validates scalar, shape, index and truncation bounds", () => {
  const astral = "\u{1f680}".repeat(240);
  expect(decodeHistory(page({ ...preview, text: astral, truncated: true })).request_previews?.[status.id]?.text).toBe(astral);
  for (const bad of [false, { ...preview, message_index: 1 }, { ...preview, message_count: -1 },
    { ...preview, message_count: 1.5 }, { ...preview, message_count: Number.MAX_SAFE_INTEGER + 1 },
    { ...preview, truncated: "yes" }, { ...preview, truncated: true }, { ...preview, text: null },
    { ...preview, text: astral + "x" }, { ...preview, text: "\ud800" }]) {
    expect(() => decodeHistory(page(bad))).toThrow("request preview");
  }
});

test("older previews survive head refresh and status-only updates; mutation fails atomically", () => {
  const oldest = mergeRequestPreviews(new Map(), { old: preview });
  const newer = mergeRequestPreviews(oldest, { latest: null });
  expect(mergeRequestPreviews(newer, { latest: null }).get("old")).toEqual(preview);
  expect(mergeRequestPreviews(newer)).toEqual(newer);
  expect(mergeRequestPreviews(newer, { old: { truncated: preview.truncated, text: preview.text, message_count: preview.message_count, message_index: preview.message_index } })).toEqual(newer);
  for (const changed of [null, { ...preview, text: "rewritten" }, { ...preview, message_index: 1 }]) {
    expect(() => mergeRequestPreviews(newer, { added: preview, old: changed })).toThrow("preview changed");
    expect(newer.has("added")).toBe(false);
    expect(newer.get("old")).toEqual(preview);
  }
  expect(() => mergeRequestPreviews(newer, { latest: preview })).toThrow();
  const ownKey = JSON.parse('{"__proto__":null}');
  expect(mergeRequestPreviews(new Map(), ownKey).has("__proto__")).toBe(true);
});

test("history renders authored text as isolated escaped text, distinct from missing or empty", () => {
  const html = renderToStaticMarkup(<HistoryPreview preview={{ ...preview, text: '<img src=x onerror="bad()">\n  & <script>' }} />);
  expect(html).toContain("Last user message / 1 of 1");
  expect(html).toContain('dir="auto"');
  expect(html).toContain("&lt;img");
  expect(html).not.toContain("<img");
  expect(html).not.toContain("<script>");
  expect(renderToStaticMarkup(<HistoryPreview preview={null} />)).toContain("unavailable");
  expect(renderToStaticMarkup(<HistoryPreview preview={undefined} />)).toContain("not supplied");
  expect(renderToStaticMarkup(<HistoryPreview preview={{ ...preview, text: "" }} />)).toContain("Empty user message");
  expect(renderToStaticMarkup(<HistoryPreview preview={{ ...preview, text: " \n\t " }} />)).not.toContain("Empty user message");
  expect(renderToStaticMarkup(<HistoryPreview preview={{ ...preview, text: "x".repeat(240), truncated: true }} />)).toContain("Excerpt: first 240");
});

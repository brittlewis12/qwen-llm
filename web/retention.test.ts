import { expect, test } from "bun:test";
import { decodeResult, type RetainedArray } from "./contract";
import { decodeFloats, distributionSummary, fetchArray, tokenMeasurement, residualMetrics } from "./retention";
import { RecordAccumulator, type Sequenced } from "./records";

const encode = (values: number[]) => { const bytes = new Uint8Array(values.length * 4); const view = new DataView(bytes.buffer); values.forEach((v, i) => view.setFloat32(i * 4, v, true)); return bytes; };
const scores = [1000, 999, 1000, -1000];
const bytes = encode(scores);
const digest = async (value: Uint8Array<ArrayBuffer>) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", value))].map(b => b.toString(16).padStart(2, "0")).join("");
const site = { position: 0, source_layer: 1, phase: "prefill", index: 0, input_token_id: 7, capture_stage: "post_block_after_operations", applied_operation_ids: ["op1", "op2"], provenance: "original_forward" };
const source = { seq: 1, kind: "retained_array", key: "source-0-1", quantity: "source_residual", ...site,
  array: { dtype: "f32le", length: 3, offset: 0, byte_length: 12, sha256: await digest(encode([1, 2, 3])), url: "/v1/lens/jobs/test/arrays/100" } };
const logits = { seq: 2, kind: "retained_array", key: "logits-0-1-j", quantity: "readout_logits", source_key: source.key, lens: "j", target_layer: 2, asset_identity: "fit", score_kind: "logit", candidate_universe: "full_vocabulary", ...site,
  array: { dtype: "f32le", length: 4, offset: 12, byte_length: 16, sha256: await digest(bytes), url: "/v1/lens/jobs/test/arrays/200" } } as RetainedArray;
const prepared = { seq: 0, kind: "prepared_input", token_ids: [7], rendering: { renderer: "test", spans: [] }, resolved_scopes: [], retention_admission: { hidden_size: 3, vocabulary_size: 4, source_arrays_upper: 1, score_arrays_upper: 1, raw_bytes_upper: 28 } };
const readout = { seq: 3, kind: "readout", ...site, readout_id: "r", lens: "j", target_layer: 2, asset_identity: "fit", score_kind: "logit", candidate_universe: "full_vocabulary", predicts_position: 1,
  scores: [{ token_id: 0, row_id: 0, label: null, score: 1000 }], cost: { readout_ms: null }, retained: { source_key: source.key, logits_key: logits.key } };
const page = (records: unknown[]) => ({ schema_version: 1, job_id: "test", records, complete: true, next_cursor: null });

test("whole-site pairs join saved before/after arrays atomically across pages", () => {
  const before = { ...source, seq: 2, key: "before-0-1", quantity: "source_residual_before", capture_stage: "post_block_before_operations", applied_operation_ids: [], site_operation_ids: site.applied_operation_ids };
  const pair = { ...site, seq: 3, kind: "residual_pair", id: "pair", before_key: before.key, after_key: source.key, capture_stage: "whole_post_block_program", metrics: residualMetrics(new Float32Array([1, 2, 3]), new Float32Array([1, 2, 3])) };
  const acc = new RecordAccumulator();
  acc.append(decodeResult(page([prepared, source])).records, undefined, "next");
  expect(() => acc.append(decodeResult(page([pair])).records, "next", undefined)).toThrow();
  acc.append(decodeResult(page([before])).records, "next", "pair");
  const saved = acc.values();
  for (const patch of [{ input_token_id: 8 }, { applied_operation_ids: ["op2", "op1"] }, { before_key: source.key }]) {
    expect(() => acc.append(decodeResult(page([{ ...pair, ...patch }])).records, "pair", undefined)).toThrow();
    expect(acc.values()).toEqual(saved);
  }
  acc.append(decodeResult(page([pair])).records, "pair", undefined);
  expect(acc.values()).toHaveLength(4);
  expect(() => decodeResult(page([{ ...pair, metrics: { ...pair.metrics, relative_delta: null } }]))).toThrow();
  expect(() => decodeResult(page([{ ...before, applied_operation_ids: ["op1"] }]))).toThrow();
});

test("residual metrics distinguish zero baseline and zero change", () => {
  expect(residualMetrics(new Float32Array([3, 4]), new Float32Array([6, 8]))).toEqual({ norm_before: 5, norm_after: 10, delta_norm: 5, relative_delta: 1 });
  expect(residualMetrics(new Float32Array([0, 0]), new Float32Array([3, 4])).relative_delta).toBeNull();
  expect(residualMetrics(new Float32Array([3, 4]), new Float32Array([3, 4])).relative_delta).toBe(0);
  expect(() => residualMetrics(new Float32Array([NaN]), new Float32Array([1]))).toThrow();
});

test("retained measurements preserve saved widths and site/head joins across pages", () => {
  const acc = new RecordAccumulator();
  acc.append(decodeResult(page([prepared, source])).records, undefined, "next");
  const original = acc.values();
  for (const patch of [{ input_token_id: 8 }, { phase: "decode" }, { index: 1 }, { applied_operation_ids: ["op2", "op1"] }, { array: { ...logits.array, length: 3, byte_length: 12 } }]) {
    expect(() => acc.append(decodeResult(page([{ ...logits, ...patch }])).records, "next", "after")).toThrow();
    expect(acc.values()).toEqual(original);
  }
  acc.append(decodeResult(page([logits])).records, "next", "after");
  expect(() => acc.append(decodeResult(page([{ ...readout, asset_identity: "other" }])).records, "after", undefined)).toThrow("asset_identity");
  acc.append(decodeResult(page([readout])).records, "after", undefined);
  expect(acc.values()).toHaveLength(4);
  expect(() => new RecordAccumulator().append([logits], undefined, undefined)).toThrow("saved model widths");
  const duplicate = new RecordAccumulator();
  expect(() => duplicate.append([prepared, source, { ...source, seq: 2 }], undefined, undefined)).toThrow("Duplicate retained");
  expect(duplicate.values()).toEqual([]);
});

test("retained array wire rejects foreign locators, bad sizes, invalid keys and widths", () => {
  expect(decodeResult(page([prepared, source, logits, readout])).records).toHaveLength(4);
  for (const patch of [{ dtype: "f64le" }, { length: 0 }, { byte_length: 15 }, { byte_length: 8 * 1024 * 1024, length: 2 * 1024 * 1024 }, { sha256: "no" }, { url: "/v1/lens/jobs/other/arrays/200" }]) {
    expect(() => decodeResult(page([{ ...logits, array: { ...logits.array, ...patch } }]))).toThrow();
  }
  expect(() => decodeResult(page([{ ...logits, key: "wrong" }]))).toThrow("site key");
});

test("bounded binary reads verify SHA256, shape, finite F32LE and reject corrupt responses", async () => {
  const response = (body: Uint8Array<ArrayBuffer> = bytes) => new Response(body, { headers: { "content-type": "application/octet-stream" } });
  const signal = new AbortController().signal;
  expect(decodeFloats(await fetchArray(logits, signal, async () => response()), 4)).toEqual(new Float32Array(scores));
  for (const body of [bytes.slice(0, 12), new Uint8Array(20), encode([999, 999, 1000, -1000])]) {
    await expect(fetchArray(logits, signal, async () => response(body))).rejects.toThrow();
  }
  const nonfinite = encode([1, NaN, 2, 3]);
  await expect(fetchArray({ ...logits, array: { ...logits.array, sha256: await digest(nonfinite) } }, signal, async () => response(nonfinite))).rejects.toThrow("Nonfinite");
  await expect(fetchArray(logits, signal, async () => Response.json({}))).rejects.toThrow("binary");
  await expect(fetchArray(logits, signal, async () => new Response(bytes, { headers: { "content-type": "application/octet-stream", "content-length": "12" } }))).rejects.toThrow("Content-Length");
  const wrongWidth = { ...logits, array: { ...logits.array, length: 3, byte_length: 12, sha256: await digest(bytes.slice(0, 12)) } };
  expect(() => new RecordAccumulator().append([prepared, source, wrongWidth] as Sequenced[], undefined, undefined)).toThrow("width");
});

test("full-vocabulary queries are stable for large logits and deterministic ties", () => {
  const full = new Float32Array(scores);
  const summary = distributionSummary(full);
  const denominator = 2 + Math.exp(-1);
  expect(tokenMeasurement(full, 0, summary).rank).toBe(1);
  expect(tokenMeasurement(full, 2, summary).rank).toBe(2);
  expect(tokenMeasurement(full, 1, summary).rank).toBe(3);
  expect(tokenMeasurement(full, 1, summary).probability).toBeCloseTo(Math.exp(-1) / denominator, 12);
  expect(summary.entropy_nats).toBeCloseTo(Math.log(denominator) + Math.exp(-1) / denominator, 12);
  expect(summary.log_partition).toBeCloseTo(1000 + Math.log(denominator), 12);
  expect(() => tokenMeasurement(full, 4)).toThrow();
  expect(() => distributionSummary(new Float32Array([Infinity]))).toThrow();
  expect(tokenMeasurement(new Float32Array([-0, 0]), 1).rank).toBe(2);
});

import { ApiHttpError, type Transport } from "./api";
import type { PairMetrics, RetainedArray } from "./contract";

export async function fetchArray(record: RetainedArray, signal: AbortSignal, transport: Transport = fetch): Promise<Uint8Array<ArrayBuffer>> {
  const { array } = record;
  if (!/^\/v1\/lens\/jobs\/[^/\\\s]+\/arrays\/[0-9]+$/.test(array.url) || !Number.isSafeInteger(array.byte_length) || array.byte_length < 4 || array.byte_length > 4 * 1024 * 1024) throw new Error("Invalid retained array locator or byte bound.");
  const response = await transport(array.url, { signal, cache: "no-store" });
  if (!response.ok) throw new ApiHttpError(response, await response.text());
  if (response.headers.get("content-type")?.split(";")[0] !== "application/octet-stream" || !response.body) throw new Error("Expected a retained binary array.");
  const length = response.headers.get("content-length");
  if (length !== null && Number(length) !== array.byte_length) throw new Error("Retained array Content-Length mismatch.");
  const bytes = new Uint8Array(array.byte_length);
  const reader = response.body.getReader();
  let offset = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value.length > bytes.length - offset) throw new Error("Retained array exceeds declared bytes.");
      bytes.set(value, offset); offset += value.length;
    }
  } finally { await reader.cancel(); reader.releaseLock(); }
  if (offset !== bytes.length) throw new Error("Retained array is truncated.");
  const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(byte => byte.toString(16).padStart(2, "0")).join("");
  if (digest !== array.sha256) throw new Error("Retained array SHA-256 mismatch.");
  if (bytes.length !== array.length * 4) throw new Error("Retained array shape mismatch.");
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < array.length; i++) if (!Number.isFinite(view.getFloat32(i * 4, true))) throw new Error("Nonfinite retained value.");
  return bytes;
}

export function decodeFloats(bytes: Uint8Array, length: number): Float32Array {
  if (bytes.byteLength !== length * 4) throw new Error("Retained array shape mismatch.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value)) throw new Error("Nonfinite retained value.");
    values[i] = value;
  }
  return values;
}

export function distributionSummary(scores: Float32Array) {
  if (!scores.length) throw new Error("Empty score distribution.");
  let max = -Infinity;
  for (const score of scores) { if (!Number.isFinite(score)) throw new Error("Nonfinite score."); max = Math.max(max, score); }
  let sum = 0, weighted = 0;
  for (const score of scores) { const delta = score - max; const weight = Math.exp(delta); sum += weight; weighted += weight * delta; }
  return { max, sum, log_partition: max + Math.log(sum), entropy_nats: Math.log(sum) - weighted / sum };
}

export function tokenMeasurement(scores: Float32Array, token: number, summary = distributionSummary(scores)) {
  if (!Number.isSafeInteger(token) || token < 0 || token >= scores.length) throw new Error("Token ID outside retained vocabulary.");
  const score = scores[token]!;
  let rank = 1;
  for (let id = 0; id < scores.length; id++) if (scores[id]! > score || (scores[id] === score && id < token)) rank++;
  return { score, rank, probability: Math.exp(score - summary.max) / summary.sum };
}

export function residualMetrics(before: Float32Array, after: Float32Array): PairMetrics {
  if (!before.length || before.length !== after.length) throw new Error("Residual pair dimensions differ or are empty.");
  let b = 0, a = 0, d = 0;
  for (let i = 0; i < before.length; i++) {
    const x = before[i]!, y = after[i]!;
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("Nonfinite residual pair.");
    b += x * x; a += y * y; d += (y - x) * (y - x);
  }
  return { norm_before: Math.sqrt(b), norm_after: Math.sqrt(a), delta_norm: Math.sqrt(d), relative_delta: b > 0 ? Math.sqrt(d) / Math.sqrt(b) : null };
}

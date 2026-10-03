import { isPrepared, isReadout, isResidualPair, isSample, selectorContains, type Job, type Prepared } from "./contract";
import { isObject } from "./draft";
import type { Sequenced } from "./records";

export type TokenSite = { phase: "prefill" | "decode"; index: number };
export const sameSite = (a: TokenSite | null | undefined, b: TokenSite | null | undefined) => !!a && !!b && a.phase === b.phase && a.index === b.index;
export const TOKEN_WINDOW = 32;

export function adjacentToken(records: Sequenced[], site: TokenSite, direction: -1 | 1): TokenSite | null {
  const prepared = records.find(isPrepared);
  let next: TokenSite = { phase: site.phase, index: site.index + direction };
  if (site.phase === "prefill" && direction === 1 && next.index === prepared?.token_ids.length) next = { phase: "decode", index: 0 };
  if (site.phase === "decode" && direction === -1 && site.index === 0 && prepared) next = { phase: "prefill", index: prepared.token_ids.length - 1 };
  return next.index >= 0 && tokenAt(records, next) ? next : null;
}

export function pieceLabel(bytes: number[]) {
  try {
    const text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(new Uint8Array(bytes));
    const chars = Array.from(text);
    return JSON.stringify(chars.slice(0, 48).join("")) + (chars.length > 48 ? "..." : "");
  } catch { return "UTF-8 byte fragment"; }
}

export function savedSpanContext(prepared: Prepared | undefined, index: number): string[] {
  if (!prepared?.prompt_bytes || !Array.isArray(prepared.rendering.spans)) return [];
  const bytes = prepared.prompt_bytes;
  return prepared.rendering.spans.filter(isObject).filter(span =>
    Number.isSafeInteger(span.token_start) && Number.isSafeInteger(span.token_end) && Number(span.token_start) >= 0 && Number(span.token_start) <= index && index < Number(span.token_end) && Number(span.token_end) <= prepared.token_ids.length &&
    Number.isSafeInteger(span.byte_start) && Number.isSafeInteger(span.byte_end) && Number(span.byte_start) >= 0 && Number(span.byte_end) >= Number(span.byte_start) && Number(span.byte_end) <= bytes.length,
  ).slice(0, 8).map(span => {
    const start = Number(span.byte_start), end = Number(span.byte_end);
    // Bound decoding too; a truncated prefix may legitimately end mid-character.
    const capped = Math.min(end, start + 1024);
    try {
      const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
      const text = decoder.decode(new Uint8Array(bytes.slice(start, capped)), { stream: capped < end });
      const chars = Array.from(text);
      return JSON.stringify(chars.slice(0, 160).join("")) + (capped < end || chars.length > 160 ? "..." : "");
    } catch { return "Span byte range is not independently valid UTF-8; inspect exact prompt bytes."; }
  });
}

export function tokenAt(records: Sequenced[], site: TokenSite) {
  const prepared = records.find(isPrepared);
  if (site.phase === "prefill") {
    const id = prepared?.token_ids[site.index];
    return id === undefined ? null : { id, position: site.index, consumed: null, bytes: null };
  }
  const sample = records.filter(isSample).find(row => row.index === site.index);
  return sample ? { id: sample.token_id, position: prepared ? prepared.token_ids.length + site.index : null, consumed: sample.consumed, bytes: sample.piece_bytes } : null;
}

export function savedSpanLabels(prepared: Prepared | undefined, index: number): string[] {
  if (!prepared || !Array.isArray(prepared.rendering.spans)) return [];
  return prepared.rendering.spans.filter(isObject).filter(span => Number.isSafeInteger(span.token_start) && Number.isSafeInteger(span.token_end) && Number(span.token_start) >= 0 && Number(span.token_start) <= index && index < Number(span.token_end) && Number(span.token_end) <= prepared.token_ids.length).map(span => [
    typeof span.message_index === "number" ? `Message ${span.message_index + 1}` : null,
    typeof span.role === "string" ? span.role : null,
    typeof span.kind === "string" ? span.kind : "saved span",
    typeof span.channel === "string" ? span.channel : null,
  ].filter(Boolean).join(" / "));
}

export function tokenCoverage(records: Sequenced[], site: TokenSite, complete: boolean, job?: Job | null) {
  const token = tokenAt(records, site);
  if (!token) return "Token record not loaded.";
  if (token.consumed === false) return "Unconsumed sample: no source forward or readout exists for this token.";
  const readouts = records.filter(isReadout).filter(r => sameSite(r, site)).length;
  const pairs = records.filter(isResidualPair).filter(r => sameSite(r, site)).length;
  if (readouts || pairs) return `${readouts} readouts and ${pairs} before/after pairs loaded at this token. ${complete ? "Publication ended; loaded records do not guarantee every requested measurement succeeded." : "More requested measurements may still be pending or not loaded."}`;
  const prepared = records.find(isPrepared);
  if (!prepared) return "Prepared scopes not loaded; measurement coverage is unknown.";
  const scopes = [...prepared.resolved_scopes.filter(row => row.kind === "readout"), ...(prepared.residual_pair_scopes ?? [])];
  if (!scopes.some(row => selectorContains(row.scope[site.phase], site.index))) return "Outside saved readout/pair scopes. No diagnostic capture was requested here.";
  if (job?.result.error !== null && job?.result.error !== undefined || job?.observations.error !== null && job?.observations.error !== undefined) return "Requested measurement has no loaded record; publication failed or is partial. This is not evidence of absent model activity.";
  if (!complete) return "Requested measurement pending or not loaded yet; no new inference is triggered by selection.";
  return "Requested measurement has no published record. Publication ended; early stop, cancellation or failure may have left this site uncaptured.";
}

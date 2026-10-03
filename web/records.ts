export type Sequenced = { seq: number; kind: string; [key: string]: unknown };

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  return JSON.stringify(value);
}

// This is an internal accumulator, not an assumed HTTP result-page envelope.
export class RecordAccumulator<T extends Sequenced> {
  private records = new Map<number, T>();
  private consumedCursors = new Set<string>();
  cursor: string | undefined;

  append(records: readonly T[], requestedCursor: string | undefined, nextCursor: string | undefined) {
    if (requestedCursor !== this.cursor) throw new Error("Stale result page; the cursor has already changed.");
    if (nextCursor === requestedCursor && records.length === 0) return;
    if (nextCursor !== undefined && (nextCursor === requestedCursor || this.consumedCursors.has(nextCursor))) throw new Error("Result cursor did not advance; pagination stopped.");
    const staged = new Map(this.records);
    let previous = -1;
    for (const record of records) {
      if (!Number.isSafeInteger(record.seq) || record.seq < 0 || record.seq <= previous) throw new Error("Result page sequences must be strictly increasing exact nonnegative integers.");
      previous = record.seq;
      const existing = staged.get(record.seq);
      if (existing && canonical(existing) !== canonical(record)) throw new Error(`Immutable record ${record.seq} changed. Original: ${JSON.stringify(existing)}; incoming: ${JSON.stringify(record)}`);
      staged.set(record.seq, structuredClone(record));
    }
    const arrayKeys = new Set<unknown>();
    for (const record of staged.values()) if (record.kind === "retained_array") {
      if (arrayKeys.has(record.key)) throw new Error("Duplicate retained array identity; original records were preserved.");
      arrayKeys.add(record.key);
    }
    validateRetainedJoins([...staged.values()]);
    this.records = staged;
    if (requestedCursor !== undefined) this.consumedCursors.add(requestedCursor);
    this.cursor = nextCursor;
  }

  values(): T[] { return [...this.records.values()].sort((a, b) => a.seq - b.seq).map(record => structuredClone(record)); }
}

function validateRetainedJoins(records: Sequenced[]) {
  const arrays = records.filter(record => record.kind === "retained_array");
  const pairs = records.filter(record => record.kind === "residual_pair");
  if (!arrays.length && !pairs.length) return;
  const prepared = records.find(record => record.seq === 0 && record.kind === "prepared_input");
  const dims = prepared?.retention_admission as Record<string, unknown> | undefined;
  if (!dims || !Number.isSafeInteger(dims.hidden_size) || !Number.isSafeInteger(dims.vocabulary_size)) throw new Error("Retained arrays lack saved model widths.");
  const byKey = new Map(arrays.map(record => [record.key, record]));
  const compare = (a: Sequenced, b: Sequenced, head: boolean) => {
    const fields = ["position", "source_layer", "phase", "index", "input_token_id", "capture_stage", "applied_operation_ids", "provenance",
      ...(head ? ["lens", "target_layer", "asset_identity", "score_kind", "candidate_universe"] : [])];
    for (const field of fields) if (canonical(a[field]) !== canonical(b[field])) throw new Error(`Retained measurement join disagrees on ${field}; original records were preserved.`);
  };
  for (const array of arrays) {
    const descriptor = array.array as Record<string, unknown>;
    if (descriptor.length !== dims[array.quantity === "readout_logits" ? "vocabulary_size" : "hidden_size"]) throw new Error("Retained array width differs from saved model dimensions.");
    if (array.quantity === "readout_logits") {
      const source = byKey.get(array.source_key);
      if (source) compare(array, source, false);
    }
  }
  const pairKeys = new Set<string>();
  for (const pair of pairs) {
    const before = byKey.get(pair.before_key), after = byKey.get(pair.after_key);
    if (!before || !after || before.quantity !== "source_residual_before" || after.quantity !== "source_residual") throw new Error("Residual pair is missing its before/after arrays.");
    const key = JSON.stringify([pair.id, pair.position, pair.source_layer]);
    if (pairKeys.has(key)) throw new Error("Duplicate residual pair identity.");
    pairKeys.add(key);
    for (const field of ["position", "source_layer", "phase", "index", "input_token_id", "provenance"]) {
      if (canonical(pair[field]) !== canonical(before[field]) || canonical(pair[field]) !== canonical(after[field])) throw new Error(`Residual pair disagrees on ${field}.`);
    }
    if (canonical(pair.applied_operation_ids) !== canonical(before.site_operation_ids) || canonical(pair.applied_operation_ids) !== canonical(after.applied_operation_ids)) throw new Error("Residual pair program order differs.");
  }
  for (const record of records) if (record.kind === "readout" && record.retained) {
    const reference = record.retained as Record<string, unknown>;
    const source = byKey.get(reference.source_key), logits = byKey.get(reference.logits_key);
    if (source) compare(record, source, false);
    if (logits) compare(record, logits, true);
  }
}

export function startPolling(
  poll: () => Promise<void>,
  onError: (error: unknown) => void,
  delay = 1500,
) {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  async function tick() {
    if (stopped) return;
    try { await poll(); }
    catch (error) { if (!stopped) onError(error); }
    if (!stopped) timer = setTimeout(tick, delay);
  }
  void tick();
  // Viewer lifetime only. Never issues a server cancellation or aborts job creation.
  return () => { stopped = true; clearTimeout(timer); };
}

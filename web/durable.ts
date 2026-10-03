import { isObject } from "./draft";
import { ApiHttpError } from "./api";

export interface StoragePort {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}
export const INTENT_KEY = "qwen-lens.submission.v1";
type Attempt = { at: string; outcome: "started" | "error" | "response"; detail?: string };
export type Intent = {
  version: 1;
  key: string;
  body: string;
  createdAt: string;
  jobId: string | null;
  attempts: Attempt[];
  response?: unknown;
  rejected?: boolean;
};

export function readIntent(storage: StoragePort): Intent | null {
  const raw = storage.getItem(INTENT_KEY);
  if (raw === null) return null;
  const value: unknown = JSON.parse(raw);
  if (!isObject(value) || value.version !== 1 || typeof value.key !== "string" || !value.key
    || typeof value.body !== "string" || typeof value.createdAt !== "string"
    || !(value.jobId === null || typeof value.jobId === "string") || !Array.isArray(value.attempts)
    || (value.rejected !== undefined && typeof value.rejected !== "boolean")
    || !value.attempts.every(attempt => isObject(attempt) && typeof attempt.at === "string"
      && ["started", "error", "response"].includes(String(attempt.outcome))
      && (attempt.detail === undefined || typeof attempt.detail === "string"))) {
    throw new Error("Submission recovery record is invalid. It was not overwritten; do not rerun blindly.");
  }
  const body: unknown = JSON.parse(value.body);
  if (!isObject(body) || body.schema_version !== 1 || body.idempotency_key !== value.key) throw new Error("Submission key/body mismatch; recovery blocked.");
  return value as Intent;
}

export class DurableSubmission {
  private busy = false;
  constructor(private storage: StoragePort, private submit: (body: string) => Promise<unknown>) {}

  prepare(config: { schema_version: 1; [key: string]: unknown }, key: string = crypto.randomUUID()): Intent {
    if (!key) throw new Error("Idempotency key must be nonempty.");
    if (this.busy) throw new Error("A submission is already in progress.");
    const previous = readIntent(this.storage);
    if (previous && !previous.jobId && !previous.rejected) throw new Error("Resolve the existing submission before creating a new run; retry uses its original key and body.");
    if (previous?.key === key) throw new Error("Run again requires a new idempotency key.");
    const intent: Intent = {
      version: 1, key, body: JSON.stringify({ ...config, idempotency_key: key }),
      createdAt: new Date().toISOString(), jobId: null, attempts: [],
    };
    if (previous) this.storage.setItem(`${INTENT_KEY}.archive.${previous.key}`, JSON.stringify(previous));
    this.save(intent);
    return intent;
  }

  private save(intent: Intent) { this.storage.setItem(INTENT_KEY, JSON.stringify(intent)); }

  async retry(): Promise<unknown> {
    if (this.busy) throw new Error("A submission is already in progress.");
    this.busy = true;
    try {
      const intent = readIntent(this.storage);
      if (!intent) throw new Error("No durable submission exists.");
      if (intent.jobId) throw new Error("This submission already has a job; open that job instead.");
      if ("response" in intent) return intent.response;
      intent.rejected = false;
      intent.attempts.push({ at: new Date().toISOString(), outcome: "started" });
      this.save(intent);
      let response: unknown;
      try { response = await this.submit(intent.body); }
      catch (error) {
        intent.rejected = rejectsThisKey(error, intent.key);
        intent.attempts.push({ at: new Date().toISOString(), outcome: "error", detail: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
        try { this.save(intent); }
        catch (storageError) { throw new AggregateError([error, storageError], "Submission failed and error persistence failed; retry only the saved original body/key."); }
        throw error;
      }
      intent.response = response;
      intent.attempts.push({ at: new Date().toISOString(), outcome: "response" });
      try { this.save(intent); }
      catch (cause) { throw new Error("Server responded but local confirmation could not be saved. Outcome is uncertain; never create a fresh key for this attempt.", { cause }); }
      return response;
    } finally { this.busy = false; }
  }

  confirmJob(key: string, jobId: string) {
    const intent = readIntent(this.storage);
    if (!intent || intent.key !== key || !jobId || !("response" in intent)) throw new Error("Cannot associate an unconfirmed submission with a job.");
    if (intent.jobId && intent.jobId !== jobId) throw new Error("A submission cannot change job IDs.");
    this.save({ ...intent, jobId });
  }
}

function rejectsThisKey(error: unknown, key: string): boolean {
  if (!(error instanceof ApiHttpError) || ![400, 412, 413, 429, 503].includes(error.response.status)) return false;
  try {
    const body: unknown = JSON.parse(error.body);
    return isObject(body) && isObject(body.admission) && body.admission.schema_version === 1
      && body.admission.idempotency_key === key && body.admission.state === "not_accepted";
  } catch { return false; }
}

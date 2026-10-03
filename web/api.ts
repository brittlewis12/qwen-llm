export const API_BASE = "/v1/lens" as const;
export const apiContractStatus = "v1-fixtures-integrated" as const;
export type Transport = (path: string, init?: RequestInit) => Promise<Response>;

export class ApiHttpError extends Error {
  constructor(readonly response: Response, readonly body: string) {
    super(`HTTP ${response.status} ${response.statusText}: ${body}`);
    this.name = "ApiHttpError";
  }
}

export async function requestJson(
  path: `/${string}`,
  init?: RequestInit,
  transport: Transport = fetch,
): Promise<unknown> {
  if (path.startsWith("//") || path.includes("\\")) throw new Error("API path must be same-origin");
  if (/[\u0000-\u0020]/.test(path)) throw new Error("API path contains whitespace or control characters");
  const response = await transport(path, init);
  const body = await response.text();
  if (!response.ok) throw new ApiHttpError(response, body);
  try { return JSON.parse(body) as unknown; }
  catch (cause) { throw new Error(`Invalid API JSON: ${body}`, { cause }); }
}

// Wire values remain unknown until the client decoders validate them.
export function createLensApi(transport: Transport = fetch) {
  const get = async (path: `/${string}`) => {
    for (let attempt = 0; ; attempt++) {
      try { return await requestJson(path, { cache: "no-store" }, transport); }
      catch (error) {
        // Control saturation and reverse-proxy failures can interrupt reads.
        // Only retry reads; submissions retain their explicit durable-key flow.
        if (!(error instanceof ApiHttpError) || ![502, 503, 504].includes(error.response.status) || attempt >= 2) throw error;
        await new Promise(resolve => setTimeout(resolve, 100 * (attempt + 1)));
      }
    }
  };
  const jobPath = (id: string): `${typeof API_BASE}/jobs/${string}` => {
    if (!id || id === "." || id === "..") throw new Error("A job ID is required");
    return `${API_BASE}/jobs/${encodeURIComponent(id)}`;
  };
  const pageQuery = (cursor?: string, limit?: number) => {
    const query = new URLSearchParams();
    if (cursor !== undefined) query.set("cursor", cursor);
    if (limit !== undefined) {
      if (!Number.isSafeInteger(limit) || limit < 1) throw new Error("Page limit must be a positive integer");
      query.set("limit", String(limit));
    }
    return query.size ? `?${query}` : "";
  };
  return {
    capabilities: () => get(`${API_BASE}/capabilities`),
    assets: () => get(`${API_BASE}/assets`),
    jobs: (cursor?: string, limit?: number) => get(`${API_BASE}/jobs${pageQuery(cursor, limit)}`),
    job: (id: string) => get(jobPath(id)),
    request: (id: string) => get(`${jobPath(id)}/request`),
    result: (id: string, cursor?: string, limit?: number) => get(`${jobPath(id)}/result${pageQuery(cursor, limit)}`),
    submit: (body: string) => requestJson(`${API_BASE}/jobs`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body,
    }, transport),
    cancel: (id: string) => requestJson(`${jobPath(id)}/cancel`, { method: "POST" }, transport),
    delete: (id: string) => requestJson(`${jobPath(id)}/delete`, { method: "POST" }, transport),
  };
}

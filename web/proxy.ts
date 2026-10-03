export function proxyConfig(env: Record<string, string | undefined>) {
  const upstream = new URL(env.QWEN_SERVE_ORIGIN ?? "http://127.0.0.1:8737");
  if (upstream.protocol !== "http:" || !["localhost", "127.0.0.1", "[::1]"].includes(upstream.hostname)
    || upstream.pathname !== "/" || upstream.search || upstream.hash || upstream.username || upstream.password) {
    throw new Error("QWEN_SERVE_ORIGIN must be a local HTTP origin without a path or credentials");
  }
  const prefix = env.QWEN_API_PREFIX ?? "/v1";
  if (!/^\/[A-Za-z0-9_/-]+$/.test(prefix) || prefix.includes("//") || prefix.endsWith("/")) {
    throw new Error("QWEN_API_PREFIX must be a non-root path without a trailing slash");
  }
  return { upstream, prefix };
}

export function proxyTarget(requestUrl: string, config: ReturnType<typeof proxyConfig>) {
  const url = new URL(requestUrl);
  if (url.pathname !== config.prefix && !url.pathname.startsWith(`${config.prefix}/`)) return null;
  return new URL(url.pathname + url.search, config.upstream);
}

export async function forwardRequest(request: Request, target: URL): Promise<Response> {
  const incoming = new URL(request.url);
  const origin = request.headers.get("origin");
  if (!["localhost", "127.0.0.1", "[::1]"].includes(incoming.hostname) || (origin !== null && origin !== incoming.origin)) {
    return Response.json({ error: { type: "invalid_request_error", code: "untrusted_origin", message: "Dev proxy only accepts its own local browser origin" } }, { status: 403 });
  }
  const headers = new Headers(request.headers);
  headers.delete("host");
  // Preserve the same-origin decision across the local development hop only
  // after validating the browser's origin; never launder a foreign Origin.
  if (origin !== null) headers.set("origin", target.origin);
  return fetch(target, {
    method: request.method, headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "manual",
  });
}

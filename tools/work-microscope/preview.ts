import { resolve, sep } from "node:path";
import { realpath, stat } from "node:fs/promises";

export async function previewResponse(request: Request, root: string): Promise<Response> {
  if (request.method !== "GET" && request.method !== "HEAD") return new Response("Read-only preview", { status: 405, headers: { Allow: "GET, HEAD" } });
  let pathname: string;
  try { pathname = decodeURIComponent(new URL(request.url).pathname); }
  catch { return new Response("Bad path", { status: 400 }); }
  if (/[\x00-\x1f\x7f]/.test(pathname)) return new Response("Bad path", { status: 400 });
  const path = resolve(root, pathname === "/" ? "index.html" : `.${pathname}`);
  if (!path.startsWith(`${root}${sep}`)) return new Response("Not found", { status: 404 });
  try {
    const [actualRoot, actualPath, info] = await Promise.all([realpath(root), realpath(path), stat(path)]);
    if (!actualPath.startsWith(`${actualRoot}${sep}`) || !info.isFile()) return new Response("Not found", { status: 404 });
  } catch { return new Response("Not found", { status: 404 }); }
  const file = Bun.file(path);
  if (!await file.exists()) return new Response("Not found", { status: 404 });
  const headers = {
    "Content-Type": file.type,
    "Content-Length": String(file.size),
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
  };
  return new Response(request.method === "HEAD" ? null : file, { headers });
}

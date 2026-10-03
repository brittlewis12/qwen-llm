import index from "./index.html";
import { forwardRequest, proxyConfig, proxyTarget } from "./proxy";

if (Bun.env.NODE_ENV === "production") throw new Error("Development only. Rust serves web/dist in production.");
const config = proxyConfig(Bun.env);
const server = Bun.serve({
  hostname: "127.0.0.1",
  port: Number(Bun.env.PORT ?? 3000),
  development: { hmr: true },
  routes: { "/": index },
  async fetch(request) {
    const target = proxyTarget(request.url, config);
    if (!target) return new Response("Not found", { status: 404 });
    try {
      return await forwardRequest(request, target);
    } catch (error) {
      console.error("qwen-serve proxy failure", error);
      return new Response(`qwen-serve proxy failure: ${String(error)}`, { status: 502 });
    }
  },
});
console.log(`Lens dev: ${server.url}; ${config.prefix} -> ${config.upstream.origin} (path preserved)`);

import { resolve } from "node:path";
import { previewResponse } from "./preview";

const root = resolve(import.meta.dir, "dist");
if (!await Bun.file(resolve(root, "index.html")).exists()) throw new Error("Run bun run build first.");
const port = Number(Bun.env.PORT ?? 4187);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error("PORT must be an integer in 0..65535.");
const server = Bun.serve({
  hostname: "127.0.0.1",
  port,
  fetch: request => previewResponse(request, root),
});
console.log(`Work Microscope: ${server.url}`);
for (const signal of ["SIGINT", "SIGTERM"] as const) process.on(signal, () => { server.stop(true); process.exit(0); });

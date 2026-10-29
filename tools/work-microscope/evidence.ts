import { investigations } from "./cases";
import type { EvidenceBundle } from "./model";
import { validateBundle } from "./validate";

async function git(repoRoot: string, args: string[]): Promise<Uint8Array> {
  const child = Bun.spawn(["git", "--no-pager", "-C", repoRoot, ...args], {
    stdin: "ignore", stdout: "pipe", stderr: "pipe",
  });
  const [output, error, exitCode] = await Promise.all([
    new Response(child.stdout).arrayBuffer(), new Response(child.stderr).text(), child.exited,
  ]);
  if (exitCode !== 0) throw new Error(`Evidence source Git read failed (${args[0]}): ${error.slice(0, 1024).trim()}`);
  return new Uint8Array(output);
}

export async function buildEvidenceBundle(repoRoot: string): Promise<EvidenceBundle> {
  // Validate curated refs with exactly the same restrictions as imported refs.
  const bundle = validateBundle({ schemaVersion: 1, investigations, snapshots: [] });
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const files = new Map<string, { bytes: Uint8Array; lines: string[]; sha256: string }>();
  const revisions = new Set<string>();
  for (const ref of bundle.investigations.flatMap(i => i.sources)) {
    if (!revisions.has(ref.revision)) {
      const kind = decoder.decode(await git(repoRoot, ["cat-file", "-t", ref.revision])).trim();
      if (kind !== "commit") throw new Error(`Evidence source ${ref.id}: revision must identify a Git commit`);
      revisions.add(ref.revision);
    }
    const key = `${ref.revision}:${ref.path}`;
    let file = files.get(key);
    if (!file) {
      const entry = decoder.decode(await git(repoRoot, ["ls-tree", "-z", ref.revision, "--", ref.path]));
      const match = /^(100644|100755) blob [0-9a-f]{40}\t([^\0]+)\0$/.exec(entry);
      if (!match || match[2] !== ref.path) throw new Error(`Evidence source ${ref.id}: path must be a tracked regular file at the pinned revision`);
      const bytes = await git(repoRoot, ["show", "--no-ext-diff", "--no-textconv", key]);
      if (bytes.byteLength > 8 * 1024 * 1024) throw new Error(`Evidence source ${ref.id}: source file exceeds 8 MiB`);
      const lines = decoder.decode(bytes).split("\n");
      if (lines.at(-1) === "") lines.pop();
      file = { bytes, lines, sha256: new Bun.CryptoHasher("sha256").update(bytes).digest("hex") };
      files.set(key, file);
    }
    if (ref.endLine > file.lines.length) throw new Error(`Evidence source ${ref.id}: requested lines ${ref.startLine}-${ref.endLine} exceed ${file.lines.length} source lines`);
    const excerpt = file.lines.slice(ref.startLine - 1, ref.endLine).join("\n");
    if (excerpt.length > 65_536) throw new Error(`Evidence source ${ref.id}: excerpt exceeds 65536 characters`);
    bundle.snapshots.push({ ...ref, sha256: file.sha256, excerpt });
  }
  // Hashes attest what this local build read, not the authenticity of imported packets.
  return validateBundle(bundle);
}

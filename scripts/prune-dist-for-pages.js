/**
 * After `astro build`, remove routes that must not ship on GitHub Pages.
 * Keeps `dist/data/**` etc.; only deletes configured top-level folders under dist/.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, "..");
const distPath = path.join(rootDir, "dist");
const configPath = path.join(__dirname, "prune-dist-config.json");

function loadRemoveList() {
  try {
    const raw = fs.readFileSync(configPath, "utf-8");
    const parsed = JSON.parse(raw);
    const dirs = parsed.removeTopLevelDirs;
    if (!Array.isArray(dirs)) return [];
    return dirs.map(String);
  } catch {
    console.warn("[prune-dist] Missing or invalid prune-dist-config.json");
    return [];
  }
}

const toRemove = loadRemoveList();
if (!fs.existsSync(distPath)) {
  console.warn("[prune-dist] dist/ not found, skip");
  process.exit(0);
}

for (const dir of toRemove) {
  const fullPath = path.join(distPath, dir);
  if (fs.existsSync(fullPath)) {
    fs.rmSync(fullPath, { recursive: true, force: true });
    console.log("[prune-dist] Removed:", fullPath);
  }
}

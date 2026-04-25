import fs from "node:fs";
import path from "node:path";

type ContentVisibilityConfig = {
  hidden?: string[];
};

const visibilityPath = path.join(process.cwd(), "src", "content-visibility.json");

let cachedMtimeMs = -1;
let cachedHidden = new Set<string>();

function postKey(collection: string, id: string): string {
  return `${collection}/${id}`.replace(/\\/g, "/").replace(/\/+/g, "/");
}

function readHiddenPostKeys(): Set<string> {
  let stat: fs.Stats | null = null;
  try {
    stat = fs.statSync(visibilityPath);
  } catch {
    cachedMtimeMs = -1;
    cachedHidden = new Set();
    return cachedHidden;
  }

  if (stat.mtimeMs === cachedMtimeMs) return cachedHidden;

  try {
    const raw = fs.readFileSync(visibilityPath, "utf-8");
    const parsed = JSON.parse(raw) as ContentVisibilityConfig;
    cachedHidden = new Set(
      Array.isArray(parsed.hidden)
        ? parsed.hidden.map((key) => String(key).replace(/\\/g, "/"))
        : []
    );
    cachedMtimeMs = stat.mtimeMs;
  } catch {
    cachedHidden = new Set();
    cachedMtimeMs = stat.mtimeMs;
  }

  return cachedHidden;
}

export function isPostVisible(collection: string, id: string): boolean {
  return !readHiddenPostKeys().has(postKey(collection, id));
}

export function isPostHidden(collection: string, id: string): boolean {
  return !isPostVisible(collection, id);
}

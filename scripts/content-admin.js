import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, "..");
const contentRoot = path.join(rootDir, "src", "content");
const visibilityPath = path.join(rootDir, "src", "content-visibility.json");
const htmlPath = path.join(__dirname, "content-admin.html");
const host = "127.0.0.1";
const port = Number(process.env.CONTENT_ADMIN_PORT || 4174);

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return;
  for (const line of fs.readFileSync(filePath, "utf-8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const idx = trimmed.indexOf("=");
    if (idx < 0) continue;
    const key = trimmed.slice(0, idx).trim();
    if (!key || Object.prototype.hasOwnProperty.call(process.env, key)) continue;
    let value = trimmed.slice(idx + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

loadEnvFile(path.join(rootDir, ".env.local"));
loadEnvFile(path.join(rootDir, ".env"));

function inferRepoFromGitConfig() {
  try {
    const raw = fs.readFileSync(path.join(rootDir, ".git", "config"), "utf-8");
    const match = raw.match(/url = (.+)/);
    if (!match) return "";
    const url = match[1].trim();
    const ssh = url.match(/git@github\.com:([^/]+\/[^/]+?)(?:\.git)?$/i);
    if (ssh) return ssh[1];
    const https = url.match(/github\.com[:/](?:[^/]+\/)?([^/]+\/[^/]+?)(?:\.git)?$/i);
    if (https) return https[1];
    return "";
  } catch {
    return "";
  }
}

const githubToken = process.env.GITHUB_TOKEN || "";
const githubRepo = process.env.GITHUB_REPO || inferRepoFromGitConfig();

function readJson(filePath, fallback) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return fallback;
  }
}

function writeVisibility(hidden) {
  const unique = [...new Set(hidden.map(String))].sort((a, b) =>
    a.localeCompare(b, "zh-CN")
  );
  fs.writeFileSync(
    visibilityPath,
    `${JSON.stringify({ hidden: unique }, null, 2)}\n`,
    "utf-8"
  );
}

function parseFrontmatter(source) {
  if (!source.startsWith("---")) return {};
  const end = source.indexOf("\n---", 3);
  if (end === -1) return {};
  const block = source.slice(3, end).trim();
  const data = {};
  for (const line of block.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    data[match[1]] = value;
  }
  return data;
}

function walkPosts() {
  if (!fs.existsSync(contentRoot)) return [];
  const collections = fs
    .readdirSync(contentRoot, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith("."))
    .map((d) => d.name);
  const posts = [];
  for (const collection of collections) {
    const base = path.join(contentRoot, collection);
    const walk = (dir) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.name.startsWith(".")) continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
          continue;
        }
        if (!/\.(md|mdx)$/i.test(entry.name)) continue;
        const rel = path.relative(base, full).replace(/\\/g, "/");
        const id = rel.replace(/\.(md|mdx)$/i, "");
        const fm = parseFrontmatter(fs.readFileSync(full, "utf-8"));
        posts.push({
          key: `${collection}/${id}`,
          collection,
          id,
          title: fm.title || path.basename(id),
          date: fm.date || "",
          file: path.relative(rootDir, full).replace(/\\/g, "/"),
        });
      }
    };
    walk(base);
  }
  return posts.sort((a, b) => String(b.date).localeCompare(String(a.date)));
}

async function syncVisibilityToGitHub() {
  if (!githubToken || !githubRepo) {
    return { synced: false, reason: "Missing GITHUB_TOKEN or GITHUB_REPO" };
  }

  const url = `https://api.github.com/repos/${githubRepo}/contents/src/content-visibility.json`;
  const headers = {
    Authorization: `Bearer ${githubToken}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "content-visibility-admin",
  };

  const current = await fetch(url, { headers });
  if (!current.ok && current.status !== 404) {
    throw new Error(`GitHub content fetch failed: ${current.status} ${await current.text()}`);
  }

  const payload = {
    message: "chore(content): update visibility settings",
    content: Buffer.from(fs.readFileSync(visibilityPath, "utf-8"), "utf-8").toString(
      "base64"
    ),
    branch: "main",
  };
  if (current.ok) {
    const data = await current.json();
    payload.sha = data.sha;
  }

  const update = await fetch(url, {
    method: "PUT",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!update.ok) {
    throw new Error(`GitHub content update failed: ${update.status} ${await update.text()}`);
  }

  return { synced: true, repo: githubRepo };
}

function sendJson(res, data, status = 200) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(data));
}

function sendHtml(res) {
  res.writeHead(200, {
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(fs.readFileSync(htmlPath, "utf-8"));
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf-8");
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        req.destroy();
        reject(new Error("Request body too large"));
      }
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${host}:${port}`);
    if (req.method === "GET" && url.pathname === "/") {
      sendHtml(res);
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/posts") {
      const visibility = readJson(visibilityPath, { hidden: [] });
      sendJson(res, {
        posts: walkPosts(),
        hidden: Array.isArray(visibility.hidden) ? visibility.hidden : [],
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/api/visibility") {
      const payload = JSON.parse((await readRequestBody(req)) || "{}");
      if (!Array.isArray(payload.hidden)) {
        sendJson(res, { error: "hidden must be an array" }, 400);
        return;
      }
      writeVisibility(payload.hidden);
      try {
        sendJson(res, { ok: true, ...(await syncVisibilityToGitHub()) });
      } catch (error) {
        sendJson(
          res,
          {
            ok: true,
            synced: false,
            reason: error instanceof Error ? error.message : String(error),
          },
          200
        );
      }
      return;
    }
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  } catch (error) {
    res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(error instanceof Error ? error.message : String(error));
  }
});

server.listen(port, host, () => {
  console.log(`Content visibility admin: http://${host}:${port}`);
});

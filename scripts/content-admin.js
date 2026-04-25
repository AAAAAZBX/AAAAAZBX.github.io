import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, "..");
const contentRoot = path.join(rootDir, "src", "content");
const visibilityPath = path.join(rootDir, "src", "content-visibility.json");
const host = "127.0.0.1";
const port = Number(process.env.CONTENT_ADMIN_PORT || 4174);

function toPosix(input) {
  return input.split(path.sep).join("/");
}

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
    const key = match[1];
    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    data[key] = value;
  }
  return data;
}

function walkPosts() {
  if (!fs.existsSync(contentRoot)) return [];
  const collections = fs
    .readdirSync(contentRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
    .map((entry) => entry.name);
  const posts = [];

  for (const collection of collections) {
    const base = path.join(contentRoot, collection);
    const walk = (dir) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.name.startsWith(".")) continue;
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(fullPath);
          continue;
        }
        if (!/\.(md|mdx)$/i.test(entry.name)) continue;
        const rel = path.relative(base, fullPath);
        const id = toPosix(rel).replace(/\.(md|mdx)$/i, "");
        const key = `${collection}/${id}`;
        const source = fs.readFileSync(fullPath, "utf-8");
        const fm = parseFrontmatter(source);
        posts.push({
          key,
          collection,
          id,
          title: fm.title || path.basename(id),
          date: fm.date || "",
          file: toPosix(path.relative(rootDir, fullPath)),
        });
      }
    };
    walk(base);
  }

  return posts.sort((a, b) => {
    const byDate = String(b.date).localeCompare(String(a.date));
    if (byDate !== 0) return byDate;
    return a.key.localeCompare(b.key, "zh-CN");
  });
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
  res.end(`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Content Visibility Admin</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #172033; }
    header { position: sticky; top: 0; z-index: 2; background: rgba(246, 248, 251, 0.92); border-bottom: 1px solid #d9e0ea; backdrop-filter: blur(12px); }
    .bar { max-width: 1180px; margin: 0 auto; padding: 18px 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    h1 { margin: 0; font-size: 20px; }
    main { max-width: 1180px; margin: 0 auto; padding: 22px 20px 44px; }
    .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
    button { border: 1px solid #0f766e; background: #0f766e; color: white; border-radius: 7px; padding: 9px 13px; font: inherit; font-weight: 700; cursor: pointer; }
    button.secondary { border-color: #cbd5e1; background: white; color: #172033; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    input[type="search"] { width: min(520px, 100%); border: 1px solid #cbd5e1; border-radius: 7px; padding: 10px 12px; font: inherit; }
    .summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .metric { background: white; border: 1px solid #d9e0ea; border-radius: 8px; padding: 14px; }
    .metric span { display: block; color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric strong { display: block; margin-top: 4px; font-size: 24px; }
    .panel { background: white; border: 1px solid #d9e0ea; border-radius: 8px; overflow: hidden; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { padding: 11px 12px; border-bottom: 1px solid #eef2f6; text-align: left; vertical-align: top; }
    th { background: #f8fafc; color: #475569; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    tr:last-child td { border-bottom: 0; }
    .title { font-weight: 700; overflow-wrap: anywhere; }
    .meta { margin-top: 3px; color: #64748b; font-size: 12px; overflow-wrap: anywhere; }
    .status { display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 700; }
    .status.public { background: #dcfce7; color: #166534; }
    .status.hidden { background: #fee2e2; color: #991b1b; }
    .switch { display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }
    .notice { color: #64748b; font-size: 13px; }
    .toast { min-height: 20px; color: #0f766e; font-size: 13px; font-weight: 700; }
    @media (max-width: 760px) {
      .bar { align-items: stretch; flex-direction: column; }
      .summary { grid-template-columns: 1fr; }
      th:nth-child(3), td:nth-child(3) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>Content Visibility Admin</h1>
        <div class="notice">只管理本地配置。隐藏文章不会进入下一次公开 build。</div>
      </div>
      <div class="actions">
        <button id="save">Save visibility</button>
        <button class="secondary" id="reload">Reload</button>
      </div>
    </div>
  </header>
  <main>
    <div class="summary">
      <div class="metric"><span>Total</span><strong id="total">0</strong></div>
      <div class="metric"><span>Public</span><strong id="public">0</strong></div>
      <div class="metric"><span>Hidden</span><strong id="hidden">0</strong></div>
    </div>
    <div class="actions" style="margin-bottom: 14px;">
      <input id="filter" type="search" placeholder="Search title, path, collection..." />
      <button class="secondary" id="show-all">Show all</button>
      <button class="secondary" id="hide-all-filtered">Hide filtered</button>
      <button class="secondary" id="show-all-filtered">Show filtered</button>
      <span class="toast" id="toast"></span>
    </div>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th style="width: 44%;">Article</th>
            <th style="width: 15%;">Visibility</th>
            <th style="width: 18%;">Collection</th>
            <th style="width: 23%;">File</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </main>
  <script>
    let posts = [];
    let hidden = new Set();
    let query = "";

    const rows = document.getElementById("rows");
    const toast = document.getElementById("toast");

    function matches(post) {
      if (!query) return true;
      const haystack = [post.title, post.key, post.collection, post.file].join(" ").toLowerCase();
      return haystack.includes(query.toLowerCase());
    }

    function visiblePosts() {
      return posts.filter(matches);
    }

    function updateSummary() {
      document.getElementById("total").textContent = posts.length;
      document.getElementById("hidden").textContent = hidden.size;
      document.getElementById("public").textContent = posts.length - hidden.size;
    }

    function render() {
      updateSummary();
      const html = visiblePosts().map((post) => {
        const isHidden = hidden.has(post.key);
        return \`
          <tr>
            <td>
              <div class="title">\${escapeHtml(post.title)}</div>
              <div class="meta">\${escapeHtml(post.key)}\${post.date ? " · " + escapeHtml(post.date) : ""}</div>
            </td>
            <td>
              <label class="switch">
                <input type="checkbox" data-key="\${escapeHtml(post.key)}" \${isHidden ? "" : "checked"} />
                <span class="status \${isHidden ? "hidden" : "public"}">\${isHidden ? "Hidden" : "Public"}</span>
              </label>
            </td>
            <td>\${escapeHtml(post.collection)}</td>
            <td><div class="meta">\${escapeHtml(post.file)}</div></td>
          </tr>
        \`;
      }).join("");
      rows.innerHTML = html || '<tr><td colspan="4" class="notice">No matching posts.</td></tr>';
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      }[ch]));
    }

    async function load() {
      const res = await fetch("/api/posts", { cache: "no-store" });
      const data = await res.json();
      posts = data.posts;
      hidden = new Set(data.hidden);
      render();
      toast.textContent = "Loaded";
      setTimeout(() => (toast.textContent = ""), 1200);
    }

    async function save() {
      const button = document.getElementById("save");
      button.disabled = true;
      try {
        const res = await fetch("/api/visibility", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ hidden: [...hidden] }),
        });
        if (!res.ok) throw new Error(await res.text());
        toast.textContent = "Saved. Run npm run build to update dist.";
      } finally {
        button.disabled = false;
      }
    }

    rows.addEventListener("change", (event) => {
      const input = event.target.closest("input[data-key]");
      if (!input) return;
      if (input.checked) hidden.delete(input.dataset.key);
      else hidden.add(input.dataset.key);
      render();
    });

    document.getElementById("filter").addEventListener("input", (event) => {
      query = event.target.value.trim();
      render();
    });
    document.getElementById("reload").addEventListener("click", load);
    document.getElementById("save").addEventListener("click", save);
    document.getElementById("show-all").addEventListener("click", () => {
      document.getElementById("filter").value = "";
      query = "";
      render();
    });
    document.getElementById("hide-all-filtered").addEventListener("click", () => {
      for (const post of visiblePosts()) hidden.add(post.key);
      render();
    });
    document.getElementById("show-all-filtered").addEventListener("click", () => {
      for (const post of visiblePosts()) hidden.delete(post.key);
      render();
    });

    load().catch((error) => {
      toast.textContent = "Failed to load: " + error.message;
    });
  </script>
</body>
</html>`);
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
      const raw = await readRequestBody(req);
      const payload = JSON.parse(raw || "{}");
      if (!Array.isArray(payload.hidden)) {
        sendJson(res, { error: "hidden must be an array" }, 400);
        return;
      }
      writeVisibility(payload.hidden);
      sendJson(res, { ok: true });
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

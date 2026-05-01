/**
 * 构建前生成文章 AI 摘要，写入 src/generated/article-summaries.json。
 * 需要环境变量 ANTHROPIC_AUTH_TOKEN（或兼容别名 ANTHROPIC_API_KEY）。
 * 兼容 DeepSeek Anthropic 网关：ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
 *
 * 未设置密钥时静默跳过，不报错。
 */
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const outPath = path.join(root, "src/generated/article-summaries.json");
const contentRoot = path.join(root, "src/content");

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function sha256Utf8(s) {
  return crypto.createHash("sha256").update(s, "utf8").digest("hex");
}

async function walkMarkdown(dir) {
  const out = [];
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...(await walkMarkdown(p)));
    else if (/\.(md|mdx)$/i.test(e.name)) out.push(p);
  }
  return out;
}

function splitFrontmatter(raw) {
  const text = raw.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
  if (!text.startsWith("---")) return { title: "", body: text };
  const end = text.indexOf("\n---\n", 3);
  if (end === -1) return { title: "", body: text };
  const fm = text.slice(3, end).trim();
  const body = text.slice(end + 5).trim();
  const titleMatch = fm.match(/^title:\s*(.+)$/m);
  let title = titleMatch ? titleMatch[1].trim() : "";
  if (
    (title.startsWith('"') && title.endsWith('"')) ||
    (title.startsWith("'") && title.endsWith("'"))
  ) {
    title = title.slice(1, -1);
  }
  return { title, body };
}

async function readJsonSafe(p, fallback) {
  try {
    return JSON.parse(await fs.readFile(p, "utf8"));
  } catch {
    return fallback;
  }
}

async function writeSummaries(data) {
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

async function callAnthropicMessages({ userText }) {
  const token = (
    process.env.ANTHROPIC_AUTH_TOKEN ||
    process.env.ANTHROPIC_API_KEY ||
    ""
  ).trim();
  const base = (process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com").replace(/\/$/, "");
  const model = (process.env.ANTHROPIC_MODEL || "deepseek-v4-pro").trim();
  const url = `${base}/v1/messages`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": token,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: 2400,
      messages: [{ role: "user", content: userText }],
    }),
  });

  const rawText = await res.text();
  if (!res.ok) {
    throw new Error(`Anthropic-compatible API ${res.status}: ${rawText.slice(0, 800)}`);
  }
  let data;
  try {
    data = JSON.parse(rawText);
  } catch {
    throw new Error(`Non-JSON response: ${rawText.slice(0, 400)}`);
  }
  const block = Array.isArray(data?.content) ? data.content.find((c) => c.type === "text") : null;
  const text = block?.text?.trim();
  if (!text) throw new Error("Empty model response");
  return text;
}

async function main() {
  const token = (
    process.env.ANTHROPIC_AUTH_TOKEN ||
    process.env.ANTHROPIC_API_KEY ||
    ""
  ).trim();
  if (!token) {
    console.log("[article-summaries] No ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY; skip.");
    return;
  }

  let store = await readJsonSafe(outPath, { version: 1, generatedAt: "", items: {} });
  if (!store.items || typeof store.items !== "object") store.items = {};

  const dirents = await fs.readdir(contentRoot, { withFileTypes: true }).catch(() => []);
  const collections = dirents.filter((d) => d.isDirectory() && !d.name.startsWith(".")).map((d) => d.name);

  const maxBodyChars = Number(process.env.ARTICLE_SUMMARY_INPUT_CHARS || 14000) || 14000;
  let generated = 0;

  for (const col of collections.sort((a, b) => a.localeCompare(b, "zh-CN"))) {
    const colDir = path.join(contentRoot, col);
    const files = await walkMarkdown(colDir);
    for (const filePath of files.sort((a, b) => a.localeCompare(b))) {
      const raw = await fs.readFile(filePath, "utf8");
      const hash = sha256Utf8(raw);
      // 与 Astro glob loader 的 ID 规范化保持一致：小写 + 空格转连字符
      const relId = path.relative(colDir, filePath)
        .replace(/\\/g, "/")
        .replace(/\.(md|mdx)$/i, "")
        .toLowerCase()
        .replace(/ /g, "-");
      const key = `${col}:${relId}`;
      const prev = store.items[key];
      if (prev?.contentSha256 === hash && typeof prev.summaryZh === "string" && prev.summaryZh.trim()) {
        continue;
      }

      const { title, body } = splitFrontmatter(raw);
      const excerpt = body.slice(0, maxBodyChars);
      const prompt =
        `请用简要概括这篇文章的主要内容与要点。\n` +
        `要求：用简体中文；3～6 句话，总共不超过 320 字；不要标题、不要使用 Markdown、不要列表符号；只依据下文节选总结，不要编造。\n\n` +
        `文章标题：${title || "（无）"}\n\n` +
        `正文节选：\n${excerpt}`;

      console.log("[article-summaries] generate:", key);
      const summaryZh = await callAnthropicMessages({ userText: prompt });
      store.items[key] = {
        title: title || "",
        summaryZh,
        model: (process.env.ANTHROPIC_MODEL || "deepseek-v4-pro").trim(),
        contentSha256: hash,
        updatedAt: new Date().toISOString(),
      };
      store.generatedAt = new Date().toISOString();
      await writeSummaries(store);
      generated += 1;
      await sleep(Number(process.env.ARTICLE_SUMMARY_DELAY_MS || 600));
    }
  }

  if (generated === 0) {
    console.log("[article-summaries] Up to date.");
  } else {
    console.log(`[article-summaries] Done. Generated/updated: ${generated}`);
  }
}

main().catch((e) => {
  console.error("[article-summaries]", e);
  process.exit(1);
});

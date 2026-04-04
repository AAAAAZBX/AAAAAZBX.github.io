export const DEFAULT_SITE_AUTHOR = "Boxuan Zhang";

/** 规范化 Astro base，保证 public 资源为根路径，避免嵌套 URL（如 /ai/xxx、/tools/xxx）下相对路径解析错误 */
function joinPublicPath(baseUrl: string, path: string): string {
  const base =
    !baseUrl || baseUrl === "/"
      ? "/"
      : baseUrl.endsWith("/")
        ? baseUrl
        : `${baseUrl}/`;
  const rel = path.startsWith("/") ? path.slice(1) : path;
  return `${base}${rel}`.replace(/\/{2,}/g, "/");
}

/** 顶图 URL：`heroImage` 可为绝对链接或站內相对 `public/` 的路径 */
export function resolveHeroImageUrl(
  heroImage: string | undefined,
  baseUrl: string,
): string {
  const fallback = joinPublicPath(baseUrl, "site-profile.png");
  const h = heroImage?.trim();
  if (!h) return fallback;
  if (h.startsWith("http://") || h.startsWith("https://")) return h;
  const path = h.startsWith("/") ? h.slice(1) : h;
  return joinPublicPath(baseUrl, path);
}

/** 为字数 / 阅读时长统计剥离 Markdown（与首页摘要逻辑一致） */
export function stripMarkdownForStats(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]+`/g, " ")
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
    .replace(/!\[([^\]]*)\]\([^\)]+\)/g, "")
    .replace(/^#+\s+/gm, "")
    .replace(/\*\*([^\*]+)\*\*/g, "$1")
    .replace(/\*([^\*]+)\*/g, "$1")
    .replace(/~~([^~]+)~~/g, "$1")
    .replace(/^>\s+/gm, "")
    .replace(/^[\*\-\+]\s+/gm, "")
    .replace(/^\d+\.\s+/gm, "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * 阅读单位：拉丁词数 + CJK 字符数（与常见中文技术博客统计方式接近）
 */
export function estimateReadingUnits(body: string): number {
  const plain = stripMarkdownForStats(body);
  const cjk = (plain.match(/[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]/g) || []).length;
  const latin = plain
    .replace(/[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
  return cjk + latin;
}

export function formatWordCountLabel(units: number): string {
  if (units >= 1000) {
    const k = units / 1000;
    const s = k >= 10 ? k.toFixed(0) : k.toFixed(1).replace(/\.0$/, "");
    return `${s}k words`;
  }
  return `${units} words`;
}

/** 混合中英文：约 200–250 词/分钟；此处取 220 */
export function estimateReadMinutes(units: number): number {
  return Math.max(1, Math.round(units / 220));
}

/** `<time datetime="...">`：与 Hexo Fluid 导出格式一致，如 `2025-05-08 11:45` */
export function formatPostDatetimeAttr(dateInput: string | Date | undefined): string {
  if (!dateInput) return "";
  const raw = typeof dateInput === "string" ? dateInput.trim() : "";
  let d: Date;
  if (typeof dateInput === "string" && /^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    const [y, m, day] = raw.split("-").map(Number);
    d = new Date(y, m - 1, day, 0, 0, 0);
  } else {
    d = typeof dateInput === "string" ? new Date(dateInput) : dateInput;
  }
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Fluid / Lynx 风格日期行（见 zhenglinblog 文首 meta） */
export function formatPostMetaDateLine(dateInput: string | Date | undefined): string {
  if (!dateInput) return "";
  const raw = typeof dateInput === "string" ? dateInput.trim() : "";
  let d: Date;
  if (typeof dateInput === "string" && /^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    const [y, m, day] = raw.split("-").map(Number);
    d = new Date(y, m - 1, day);
  } else {
    d = typeof dateInput === "string" ? new Date(dateInput) : dateInput;
  }
  if (Number.isNaN(d.getTime())) return "";
  const datePart = d.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  const hasTime =
    typeof dateInput === "string" &&
    (raw.includes("T") || /^\d{4}-\d{2}-\d{2}\s+\d/.test(raw));
  if (!hasTime) {
    return `${datePart} am`;
  }
  const timePart = d
    .toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    })
    .toLowerCase();
  return `${datePart} ${timePart}`;
}

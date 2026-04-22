/**
 * 站点公开访问统计（CountAPI，可浏览器直连，适合 GitHub Pages 纯静态站）
 * @see https://countapi.xyz
 * 可在仓库环境变量中设置 `PUBLIC_COUNTAPI_NAMESPACE`（仅英数字，用于区分不同站点/仓库）
 */
import { createHash } from "node:crypto";

export const COUNTAPI_BASE = "https://api.countapi.xyz";

const DEFAULT_NAMESPACE = "AAAAAZBXio";

function sanitizeSegment(s: string): string {
  return s.replace(/[^a-zA-Z0-9]/g, "");
}

export function getCountapiNamespace(): string {
  const raw = import.meta.env.PUBLIC_COUNTAPI_NAMESPACE;
  if (typeof raw === "string" && raw.trim().length > 0) {
    const t = sanitizeSegment(raw);
    return t || DEFAULT_NAMESPACE;
  }
  return DEFAULT_NAMESPACE;
}

/** 与 /hit /get 路径段一致，仅 a-zA-Z0-9 */
export const COUNTAPI_KEYS = {
  sitePageViews: "sitewide",
} as const;

/**
 * 每篇文章独立计数器 key（仅小写十六进制，满足 CountAPI 段要求）
 * 与 `getEntry(collection, id)` 的 id 即 merged post 的 `slug` 一致
 */
export function countapiArticleKey(collection: string, postId: string): string {
  return createHash("sha256")
    .update(`${collection}::${postId}`, "utf8")
    .digest("hex")
    .slice(0, 20);
}

export function countapiGetUrl(
  namespace: string,
  key: string
): string {
  return `${COUNTAPI_BASE}/get/${encodeURIComponent(sanitizeSegment(namespace))}/${encodeURIComponent(
    sanitizeSegment(key)
  )}`;
}

export function countapiHitUrl(
  namespace: string,
  key: string
): string {
  return `${COUNTAPI_BASE}/hit/${encodeURIComponent(sanitizeSegment(namespace))}/${encodeURIComponent(
    sanitizeSegment(key)
  )}`;
}

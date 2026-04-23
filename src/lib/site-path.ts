/**
 * 浏览器 pathname 去掉 Astro `base` 前缀，与 `MergedPost.href` 对齐（无尾斜杠）。
 */
export function siteRelativePathname(pathname: string, siteBase: string): string {
  const p0 = (pathname || "/").split("?")[0]!.split("#")[0]!;
  let b = (siteBase || "/").trim();
  if (b === "" || b === "./") b = "/";
  if (!b.startsWith("/")) b = `/${b}`;
  const baseNoTrail = b === "/" ? "/" : b.replace(/\/+$/, "");
  let p = p0;
  if (baseNoTrail !== "/") {
    if (p === baseNoTrail) p = "/";
    else if (p.startsWith(`${baseNoTrail}/`)) p = p.slice(baseNoTrail.length) || "/";
  }
  if (!p.startsWith("/")) p = `/${p}`;
  return p.replace(/\/+$/, "") || "/";
}

/** 与 `post.href` 比较用，与入库的 `page_path` 规则一致 */
export function normalizeArticlePathForStats(href: string): string {
  if (!href) return "/";
  const raw = String(href).split("?")[0]!.split("#")[0]!;
  const withSlash = raw.startsWith("/") ? raw : `/${raw}`;
  // 统一“已编码路径(%E4...)”与“未编码路径(中文/空格)”为同一键，避免聚合对不上。
  let decoded = withSlash;
  try {
    decoded = decodeURI(withSlash);
  } catch {
    decoded = withSlash;
  }
  return decoded.replace(/\/+$/, "") || "/";
}

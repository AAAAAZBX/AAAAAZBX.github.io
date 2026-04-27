import { isOperatorConsolePath, siteRelativePathname } from "../lib/site-path";

/** 全站 CountAPI 命中；控制台路由与访客日志跳过规则一致。 */
export function runSitewideCountHit(options: { siteBasePath?: string; hitUrl: string | undefined }): void {
  const hitUrl = options.hitUrl;
  if (!hitUrl) return;
  if (typeof window === "undefined" || !window.location?.pathname) return;
  const p = siteRelativePathname(window.location.pathname, options.siteBasePath ?? "/");
  if (isOperatorConsolePath(p)) return;
  void fetch(hitUrl, { mode: "cors", cache: "no-store" }).catch(() => {});
}

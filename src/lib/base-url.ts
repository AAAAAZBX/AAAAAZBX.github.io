/** Resolve a site path against Astro `import.meta.env.BASE_URL` (supports GitHub Pages project sites). */
export function withBase(baseUrl: string, path: string): string {
  const base = baseUrl || "/";
  const normalizedPath = path.replace(/^\/+/, "");
  if (!normalizedPath) {
    return base.endsWith("/") ? base : `${base}/`;
  }
  return base.endsWith("/") ? `${base}${normalizedPath}` : `${base}/${normalizedPath}`;
}

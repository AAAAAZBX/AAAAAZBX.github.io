import { getEntry } from "astro:content";
import { LEARNING_TECHNOLOGY_INDEX_SLUG } from "./learning-constants";

export type TechNavLink = { label: string; href: string };
export type TechNavSubgroup = { name: string; links: TechNavLink[] };
export type TechNavGroup = { title: string; subgroups: TechNavSubgroup[] };

function extractLinks(block: string): TechNavLink[] {
  const links: TechNavLink[] = [];
  const re = /-\s*\[([^\]]*)\]\(([^)]+)\)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(block)) !== null) {
    links.push({ label: m[1].trim(), href: m[2].trim() });
  }
  return links;
}

function parseGroupChunk(chunk: string): TechNavGroup | null {
  const trimmed = chunk.trim();
  if (!trimmed) return null;
  const lines = trimmed.split("\n");
  const title = lines[0].trim();
  if (!title) return null;
  const rest = lines.slice(1).join("\n");
  const subgroups: TechNavSubgroup[] = [];
  const parts = rest.split(/^###\s+/m);

  const intro = (parts[0] ?? "").trim();
  const introLinks = extractLinks(intro);
  if (introLinks.length) subgroups.push({ name: "", links: introLinks });

  for (let i = 1; i < parts.length; i++) {
    const sub = parts[i].trim();
    if (!sub) continue;
    const subLines = sub.split("\n");
    const name = subLines[0].trim();
    const body = subLines.slice(1).join("\n");
    const links = extractLinks(body);
    if (name || links.length) subgroups.push({ name, links });
  }

  return { title, subgroups };
}

/** 从 technology_content.md 正文解析 ## / ### 结构（与林正式 Technology 目录一致） */
export async function getTechnologyNavFromIndex(): Promise<TechNavGroup[]> {
  const entry = await getEntry("learning", LEARNING_TECHNOLOGY_INDEX_SLUG);
  if (!entry) return [];
  const text = entry.body;
  const h2Parts = text.split(/^##\s+/m);
  const groups: TechNavGroup[] = [];
  for (let i = 1; i < h2Parts.length; i++) {
    const g = parseGroupChunk(h2Parts[i]);
    if (g) groups.push(g);
  }
  return groups;
}

/** 站内链接统一为 /learning/.../ 便于本地与部署 */
export function normalizeLearningTocHref(href: string): string {
  if (href.startsWith("/")) {
    return href.endsWith("/") || href.includes("?") ? href : `${href}/`;
  }
  try {
    const u = new URL(href);
    if (u.pathname.startsWith("/learning/")) {
      const p = u.pathname;
      return p.endsWith("/") ? p : `${p}/`;
    }
  } catch {
    /* ignore */
  }
  return href;
}

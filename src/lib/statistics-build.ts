import type { MergedPost } from "./content-posts";
import { countapiArticleKey, countapiGetUrl } from "./site-stats";
import { normalizeArticlePathForStats } from "./site-path";

export function calculateWordCount(body: string | undefined): number {
  const safeBody = typeof body === "string" ? body : "";
  let text = safeBody
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`[^`]+`/g, "")
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

  const chineseChars = (text.match(/[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]/g) || [])
    .length;
  const englishWords = text
    .replace(/[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 0).length;

  return chineseChars + englishWords;
}

/**
 * Aligned with the nav: --nav-bg ~210° blue-gray, --fluid-link-hover-color cyan.
 * Lighter mid-tones for contrast on light and dark card backgrounds.
 */
const PIE_COLORS = [
  "hsl(210 32% 34%)",
  "hsl(210 30% 42%)",
  "hsl(210 32% 50%)",
  "hsl(208 35% 54%)",
  "hsl(204 40% 52%)",
  "hsl(200 48% 50%)",
  "hsl(198 60% 52%)",
  "hsl(197 58% 58%)",
];

export type CollectionPieItem = {
  label: string;
  count: number;
  color: string;
  /** `d` for an annular sector in 200×200 viewBox; clockwise from 12 o’clock */
  slicePath: string;
};

const PIE_SVG_CX = 100;
const PIE_SVG_CY = 100;
const PIE_SVG_R_OUT = 100;
const PIE_SVG_R_IN = 56; /* Matches .stat-pie__hole ~22% inset (donut width) */

/**
 * Donut sector path. 0–100% is a full turn; 0% at 12 o’clock, same as conic-gradient(from -90deg).
 */
function annularSectorPath(
  p0: number,
  p1: number
): string {
  const { cx, cy, ro, ri } = {
    cx: PIE_SVG_CX,
    cy: PIE_SVG_CY,
    ro: PIE_SVG_R_OUT,
    ri: PIE_SVG_R_IN,
  };
  if (p1 - p0 >= 99.99) {
    const oT = { x: cx, y: cy - ro };
    const oB = { x: cx, y: cy + ro };
    const iT = { x: cx, y: cy - ri };
    const iB = { x: cx, y: cy + ri };
    return `M ${iT.x} ${iT.y} L ${oT.x} ${oT.y} A ${ro} ${ro} 0 1 1 ${oB.x} ${oB.y} A ${ro} ${ro} 0 1 1 ${oT.x} ${oT.y} L ${iT.x} ${iT.y} A ${ri} ${ri} 0 1 0 ${iB.x} ${iB.y} A ${ri} ${ri} 0 1 0 ${iT.x} ${iT.y} Z`;
  }
  const t0 = (2 * Math.PI * p0) / 100;
  const t1 = (2 * Math.PI * p1) / 100;
  const o0 = { x: cx + ro * Math.sin(t0), y: cy - ro * Math.cos(t0) };
  const o1 = { x: cx + ro * Math.sin(t1), y: cy - ro * Math.cos(t1) };
  const i0 = { x: cx + ri * Math.sin(t0), y: cy - ri * Math.cos(t0) };
  const i1 = { x: cx + ri * Math.sin(t1), y: cy - ri * Math.cos(t1) };
  const large = p1 - p0 > 50 ? 1 : 0;
  return `M ${i0.x} ${i0.y} L ${o0.x} ${o0.y} A ${ro} ${ro} 0 ${large} 1 ${o1.x} ${o1.y} L ${i1.x} ${i1.y} A ${ri} ${ri} 0 ${large} 0 ${i0.x} ${i0.y} Z`;
}

/** Placeholder full ring when there are zero posts; same 200×200 path space */
export const PIE_EMPTY_DONUT_PATH = annularSectorPath(0, 100);

export function buildCollectionPieStats(allPosts: MergedPost[]): {
  totalPosts: number;
  pieItems: CollectionPieItem[];
} {
  const totalPosts = allPosts.length;
  const byCollection = new Map<string, number>();
  for (const p of allPosts) {
    byCollection.set(p.collection, (byCollection.get(p.collection) || 0) + 1);
  }
  const pieItems: CollectionPieItem[] = [];
  const sorted = [...byCollection.entries()].sort((a, b) => b[1] - a[1]);
  let accPct = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (totalPosts <= 0) break;
    const [label, count] = sorted[i]!;
    const start = accPct * 100;
    accPct += count / totalPosts;
    const end = accPct * 100;
    const slicePath = annularSectorPath(
      start,
      Math.min(100, end)
    );
    pieItems.push({
      label,
      count,
      color: PIE_COLORS[i % PIE_COLORS.length]!,
      slicePath,
    });
  }
  return { totalPosts, pieItems };
}

export function buildPerArticleViewRows(
  allPosts: MergedPost[],
  namespace: string
): { title: string; href: string; getUrl: string; pathKey: string }[] {
  return allPosts.map((post) => ({
    title: post.title,
    href: post.href,
    getUrl: countapiGetUrl(
      namespace,
      countapiArticleKey(post.collection, post.slug)
    ),
    pathKey: normalizeArticlePathForStats(post.href),
  }));
}

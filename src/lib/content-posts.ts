import { getCollection } from "astro:content";
import { LEARNING_TECHNOLOGY_INDEX_SLUG } from "./learning-constants";

export type ColKey = "algorithms" | "learning" | "research" | "travel";

export const collectionKeys: ColKey[] = ["algorithms", "learning", "research", "travel"];

export const collectionLabels: Record<ColKey, { en: string; zh: string }> = {
  algorithms: { en: "Competitive Programming", zh: "算法竞赛" },
  learning: { en: "Technology", zh: "技术笔记" },
  research: { en: "Research", zh: "研究" },
  travel: { en: "Daily Life", zh: "日常" },
};

export type MergedPost = {
  collection: ColKey;
  slug: string;
  title: string;
  date: string | undefined;
  href: string;
  description: string | undefined;
  body: string;
};

export async function getMergedPosts(): Promise<MergedPost[]> {
  const buckets = await Promise.all(
    collectionKeys.map(async (key) => {
      const posts = await getCollection(key);
      return posts
        .filter(
          (post) =>
            key !== "learning" || post.slug !== LEARNING_TECHNOLOGY_INDEX_SLUG
        )
        .map((post) => ({
          collection: key,
          slug: post.slug,
          title: post.data.title,
          date: post.data.date,
          href: `/${key}/${post.slug}`,
          description: post.data.description,
          body: post.body,
        }));
    })
  );
  return buckets.flat().sort((a, b) => {
    const ta = a.date ? new Date(a.date).getTime() : 0;
    const tb = b.date ? new Date(b.date).getTime() : 0;
    return tb - ta;
  });
}

export function formatDateIso(dateString: string | undefined): string {
  if (!dateString) return "";
  const d = new Date(dateString);
  if (Number.isNaN(d.getTime())) return "";
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

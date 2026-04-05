import { getCollection } from "astro:content";

export type ColKey = "algorithms" | "ai" | "tools" | "travel";

export const collectionKeys: ColKey[] = ["algorithms", "ai", "tools", "travel"];

export const collectionLabels: Record<ColKey, { en: string; zh: string }> = {
  algorithms: { en: "Competitive Programming", zh: "算法竞赛" },
  ai: { en: "AI", zh: "人工智能" },
  tools: { en: "Tools", zh: "工具与环境" },
  travel: { en: "Daily Life", zh: "日常" },
};

/** src/content 下实际文件夹名（可与集合键不同，例如 ai → `AI`） */
export const contentFolderByColKey: Record<ColKey, string> = {
  algorithms: "algorithms",
  ai: "AI",
  tools: "Tools",
  travel: "travel",
};

export type MergedPost = {
  collection: ColKey;
  slug: string;
  title: string;
  date: string | undefined;
  href: string;
  description: string | undefined;
  tags: string[];
  body: string;
};

export async function getMergedPosts(): Promise<MergedPost[]> {
  const buckets = await Promise.all(
    collectionKeys.map(async (key) => {
      const posts = await getCollection(key);
      return posts.map((post) => ({
        collection: key,
        /* glob loader 集合无 slug，路径键与 getEntry(collection, id) 一致 */
        slug: post.id,
        title: post.data.title,
        date: post.data.date,
        href: `/categories/${key}/${post.id}`,
        description: post.data.description,
        tags: post.data.tags ?? [],
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

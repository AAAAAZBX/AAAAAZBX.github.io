import { getCollection } from "astro:content";
import { collections } from "../content.config";
import { isPostVisible, fetchHiddenPostKeys } from "./content-visibility";
import { resolvePostId } from "./post-id";

export type ColKey = string;

export function getCollectionKeys(): ColKey[] {
  return Object.keys(collections).sort((a, b) => a.localeCompare(b, "zh-CN"));
}

export function getCollectionLabel(key: ColKey): { en: string; zh: string } {
  return { en: key, zh: key };
}

export type MergedPost = {
  collection: ColKey;
  slug: string;
  sortId: string;
  title: string;
  date: string | undefined;
  href: string;
  description: string | undefined;
  tags: string[];
  body: string;
};

export async function getMergedPosts(): Promise<MergedPost[]> {
  const hiddenKeys = await fetchHiddenPostKeys();
  const collectionKeys = getCollectionKeys();
  const buckets = await Promise.all(
    collectionKeys.map(async (key) => {
      const posts = await getCollection(key as never);
      return posts.filter((post) => isPostVisible(resolvePostId({ id: post.data.id, slug: post.id }), hiddenKeys)).map((post) => ({
        collection: key,
        slug: post.id,
        sortId: resolvePostId({ id: post.data.id, slug: post.id }),
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

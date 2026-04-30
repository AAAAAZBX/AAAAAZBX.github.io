import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { getCollectionKeys } from '../../lib/content-posts';
import { readHiddenPostIds } from '../../lib/content-visibility';
import { resolvePostId } from '../../lib/post-id';

export const prerender = false;

export const GET: APIRoute = async () => {
  try {
    const hidden = await readHiddenPostIds();
    const collectionKeys = getCollectionKeys();

    const allPosts: Array<{ id: string; title: string; date: string; collection: string }> = [];
    for (const col of collectionKeys) {
      const posts = await getCollection(col as never);
      for (const p of posts) {
        allPosts.push({
          id: resolvePostId({ id: p.data.id, slug: p.id }),
          title: p.data.title,
          date: p.data.date || '',
          collection: col,
        });
      }
    }

    allPosts.sort((a, b) => {
      const colOrderA = collectionKeys.indexOf(a.collection);
      const colOrderB = collectionKeys.indexOf(b.collection);
      if (colOrderA !== colOrderB) return colOrderA - colOrderB;
      const ta = a.date ? new Date(a.date).getTime() : 0;
      const tb = b.date ? new Date(b.date).getTime() : 0;
      return tb - ta;
    });

    return new Response(
      JSON.stringify({ posts: allPosts, hidden }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' },
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Internal server error' }),
      { status: 500, headers: { 'Content-Type': 'application/json; charset=utf-8' } }
    );
  }
};

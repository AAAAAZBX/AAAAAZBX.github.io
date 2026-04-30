import type { APIRoute } from 'astro';
import { getMergedPosts, buildPostExcerpt, formatDateIso } from '../lib/content-posts';
import {
  buildContentFolderDisplayMap,
  categoryIndexHref,
  normSeg,
} from '../lib/category-tree';

export const prerender = true;

type RuntimeHomePost = {
  sortId: string;
  title: string;
  date: string;
  dateFormatted: string;
  href: string;
  excerpt: string;
  crumbs: Array<{ label: string; href: string }>;
  tags: Array<{ label: string; href: string }>;
};

function tagSlug(tag: string): string {
  return tag.trim().toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9一-鿿_-]/g, '');
}

export const GET: APIRoute = async () => {
  try {
    const baseUrl = import.meta.env.BASE_URL;
    const folderDisplayMap = buildContentFolderDisplayMap();
    const posts = await getMergedPosts();

    const payload: RuntimeHomePost[] = posts.map((post) => {
      const dirs = post.slug.split('/').filter(Boolean).slice(0, -1);
      const crumbs = [
        {
          label: post.collection,
          href: categoryIndexHref(baseUrl, post.collection),
        },
      ];

      dirs.forEach((seg, idx) => {
        const pathKey = `${post.collection}/${dirs.slice(0, idx + 1).map(normSeg).join('/')}`;
        crumbs.push({
          label: folderDisplayMap.get(pathKey) ?? seg,
          href: categoryIndexHref(baseUrl, pathKey),
        });
      });

      return {
        sortId: post.sortId,
        title: post.title,
        date: post.date ?? '',
        dateFormatted: formatDateIso(post.date),
        href: post.href,
        excerpt: buildPostExcerpt(post.body),
        crumbs,
        tags: (post.tags ?? []).map((tag) => ({
          label: tag,
          href: `${baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`}tags/${encodeURIComponent(tagSlug(tag))}`,
        })),
      };
    });

    return new Response(JSON.stringify({ posts: payload }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'public, max-age=3600, stale-while-revalidate=86400',
      },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Internal server error' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
      }
    );
  }
};

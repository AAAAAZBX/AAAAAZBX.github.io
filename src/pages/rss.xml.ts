import rss from '@astrojs/rss';
import type { APIRoute } from 'astro';
import { getMergedPosts, buildPostExcerpt } from '../lib/content-posts';

export const prerender = true;

export const GET: APIRoute = async (context) => {
  const site = context.site;
  if (!site) {
    throw new Error('Missing site URL (set `site` in astro.config)');
  }

  const posts = await getMergedPosts();

  return rss({
    title: 'Boxuan Zhang',
    description: 'Algorithms, systems, and machine learning notes.',
    site,
    items: posts.map((post) => ({
      title: post.title,
      pubDate: post.date ? new Date(post.date) : undefined,
      description: buildPostExcerpt(post.body, 400),
      link: new URL(post.href, site).href,
      categories: post.tags,
    })),
  });
};

import type { APIRoute } from 'astro';
import fs from 'node:fs';
import path from 'node:path';
import { getCollection } from 'astro:content';
import { createClient } from '@supabase/supabase-js';
import { getCollectionKeys } from '../../lib/content-posts';

function readJson<T>(filePath: string, fallback: T): T {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return fallback;
  }
}

async function readHiddenList(): Promise<string[]> {
  const supabaseUrl = String(import.meta.env.PUBLIC_SUPABASE_URL ?? '').trim();
  const anonKey = String(import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? '').trim();

  if (supabaseUrl && anonKey && supabaseUrl.startsWith('http')) {
    try {
      const supabase = createClient(supabaseUrl, anonKey);
      const { data, error } = await supabase
        .from('content_visibility')
        .select('hidden_posts')
        .eq('id', 1)
        .single();
      if (!error && data && Array.isArray(data.hidden_posts)) {
        return data.hidden_posts.map((k: unknown) => String(k).trim().toLowerCase());
      }
    } catch {
      // fall through to local file
    }
  }

  const visibilityPath = path.join(process.cwd(), 'src', 'content-visibility.json');
  const fallback = readJson<{ hidden?: string[] }>(visibilityPath, { hidden: [] });
  return (Array.isArray(fallback.hidden) ? fallback.hidden : []).map(k => String(k).trim().toLowerCase());
}

export const GET: APIRoute = async () => {
  try {
    const hidden = await readHiddenList();
    const collectionKeys = getCollectionKeys();

    const allPosts: Array<{ id: string; title: string; date: string; collection: string }> = [];
    for (const col of collectionKeys) {
      const posts = await getCollection(col as never);
      for (const p of posts) {
        allPosts.push({
          id: p.data.id?.trim() || '',
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

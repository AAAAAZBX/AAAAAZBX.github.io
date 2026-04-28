import type { APIRoute } from 'astro';
import fs from 'node:fs';
import path from 'node:path';
import { createClient } from '@supabase/supabase-js';

function parseFrontmatter(source: string): Record<string, string> {
  if (!source.startsWith('---')) return {};
  const end = source.indexOf('\n---', 3);
  if (end === -1) return {};
  const block = source.slice(3, end).trim();
  const data: Record<string, string> = {};
  for (const line of block.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    data[match[1]] = value;
  }
  return data;
}

function walkPosts(): Array<{
  key: string;
  collection: string;
  id: string;
  title: string;
  date: string;
  file: string;
}> {
  const contentRoot = path.join(process.cwd(), 'src', 'content');
  if (!fs.existsSync(contentRoot)) return [];

  const collections = fs
    .readdirSync(contentRoot, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith('.'))
    .map((d) => d.name);

  const posts: ReturnType<typeof walkPosts> = [];

  for (const collection of collections) {
    const base = path.join(contentRoot, collection);
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.name.startsWith('.')) continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
          continue;
        }
        if (!/\.(md|mdx)$/i.test(entry.name)) continue;
        const rel = path.relative(base, full).replace(/\\/g, '/');
        const id = rel.replace(/\.(md|mdx)$/i, '');
        const fm = parseFrontmatter(fs.readFileSync(full, 'utf-8'));
        posts.push({
          key: `${collection}/${id}`,
          collection,
          id,
          title: fm.title || path.basename(id),
          date: fm.date || '',
          file: path.relative(process.cwd(), full).replace(/\\/g, '/'),
        });
      }
    };
    walk(base);
  }

  return posts.sort((a, b) => String(b.date).localeCompare(String(a.date)));
}

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
        return data.hidden_posts;
      }
    } catch {
      // fall through to local file
    }
  }

  const visibilityPath = path.join(process.cwd(), 'src', 'content-visibility.json');
  const fallback = readJson<{ hidden?: string[] }>(visibilityPath, { hidden: [] });
  return Array.isArray(fallback.hidden) ? fallback.hidden : [];
}

export const GET: APIRoute = async () => {
  try {
    const hidden = await readHiddenList();

    return new Response(
      JSON.stringify({
        posts: walkPosts(),
        hidden,
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json; charset=utf-8',
          'Cache-Control': 'no-store',
        },
      }
    );
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

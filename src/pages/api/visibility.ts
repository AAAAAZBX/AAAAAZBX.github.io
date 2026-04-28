import type { APIRoute } from 'astro';
import fs from 'node:fs';
import path from 'node:path';
import { createClient } from '@supabase/supabase-js';

export const POST: APIRoute = async ({ request }) => {
  try {
    const payload = (await request.json()) as { hidden?: unknown };
    if (!Array.isArray(payload.hidden)) {
      return new Response(JSON.stringify({ error: 'hidden must be an array' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
      });
    }

    const unique = [...new Set(payload.hidden.map(String))].sort((a, b) =>
      a.localeCompare(b, 'zh-CN')
    );

    // Always write local file as fallback cache
    const visibilityPath = path.join(process.cwd(), 'src', 'content-visibility.json');
    fs.writeFileSync(visibilityPath, `${JSON.stringify({ hidden: unique }, null, 2)}\n`, 'utf-8');

    // Sync to Supabase for production reads
    let synced = false;
    const supabaseUrl = String(import.meta.env.PUBLIC_SUPABASE_URL ?? '').trim();
    const serviceRoleKey = String(import.meta.env.SUPABASE_SERVICE_ROLE_KEY ?? '').trim();

    if (supabaseUrl && serviceRoleKey) {
      try {
        const supabase = createClient(supabaseUrl, serviceRoleKey);
        const { error } = await supabase
          .from('content_visibility')
          .upsert({ id: 1, hidden_posts: unique, updated_at: new Date().toISOString() });

        if (!error) synced = true;
      } catch {
        // Supabase unreachable, local file is enough
      }
    }

    // Trigger GitHub Pages rebuild so public site reflects new visibility
    let rebuilt = false;
    const githubToken = String(import.meta.env.GITHUB_TOKEN ?? '').trim();
    const githubRepo = String(import.meta.env.GITHUB_REPO ?? '').trim();

    if (githubToken && githubRepo) {
      try {
        const dispatchUrl = `https://api.github.com/repos/${githubRepo}/dispatches`;
        const res = await fetch(dispatchUrl, {
          method: 'POST',
          headers: {
            Authorization: `token ${githubToken}`,
            Accept: 'application/vnd.github+json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ event_type: 'rebuild' }),
        });
        if (res.ok) rebuilt = true;
      } catch {
        // GitHub unreachable, skip rebuild trigger
      }
    }

    return new Response(JSON.stringify({ ok: true, synced, rebuilt }), {
      status: 200,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' },
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

import type { APIRoute } from 'astro';
import { createClient } from '@supabase/supabase-js';

// The Vite plugin in astro.config.mjs parses hidden IDs from req.url
// (bypassing Node.js 24 bug) and stores them in globalThis.__hiddenFromUrl.
// This API route reads them and syncs to Supabase.

export const GET: APIRoute = async () => {
  try {
    const hidden: string[] = (globalThis as any).__hiddenFromUrl ?? [];
    const unique = [...new Set(hidden.map(s => s.toLowerCase()))].sort((a, b) =>
      a.localeCompare(b, 'zh-CN')
    );

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
      } catch { /* Supabase unreachable */ }
    }

    return new Response(JSON.stringify({ ok: true, synced }), {
      status: 200,
      headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Internal server error' }),
      { status: 500, headers: { 'Content-Type': 'application/json; charset=utf-8' } }
    );
  }
};

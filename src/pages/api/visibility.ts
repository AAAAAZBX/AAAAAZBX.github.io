import type { APIRoute } from 'astro';
import { createClient } from '@supabase/supabase-js';
import { normalizeHiddenPostIds } from '../../lib/post-id';

// The Vite plugin in astro.config.mjs parses hidden IDs from req.url
// (bypassing Node.js 24 bug) and stores them in globalThis.__hiddenFromUrl.
// This API route reads them and syncs to Supabase.

function allowServiceRoleSync(request: Request): boolean {
  if (import.meta.env.DEV) return true;
  const secret = String(import.meta.env.VISIBILITY_SYNC_SECRET ?? '').trim();
  if (!secret) return false;
  const auth = request.headers.get('authorization')?.trim();
  return auth === `Bearer ${secret}`;
}

export const GET: APIRoute = async ({ request }) => {
  try {
    const hidden: string[] = (globalThis as any).__hiddenFromUrl ?? [];
    const unique = normalizeHiddenPostIds(hidden);

    let synced = false;
    const supabaseUrl = String(import.meta.env.PUBLIC_SUPABASE_URL ?? '').trim();
    const serviceRoleKey = String(import.meta.env.SUPABASE_SERVICE_ROLE_KEY ?? '').trim();

    if (supabaseUrl && serviceRoleKey && allowServiceRoleSync(request)) {
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

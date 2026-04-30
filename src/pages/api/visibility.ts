import type { APIRoute } from 'astro';
import { normalizeHiddenPostIds } from '../../lib/post-id';
import { persistHiddenPostIds, readHiddenPostIds } from '../../lib/content-visibility';

export const prerender = false;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' },
  });
}

async function readHiddenInput(request: Request, url: URL): Promise<{ hidden: string[]; malformedJson: boolean }> {
  const queryHidden = url.searchParams.getAll('hidden');
  const contentType = String(request.headers.get('content-type') || '').toLowerCase();
  const rawText = await request.text().catch(() => '');

  let bodyHidden: unknown[] = [];
  let malformedJson = false;

  if (contentType.includes('application/json') && rawText.trim()) {
    try {
      const payload = JSON.parse(rawText) as { hidden?: unknown[] } | unknown[];
      if (Array.isArray(payload)) bodyHidden = payload;
      else if (Array.isArray(payload?.hidden)) bodyHidden = payload.hidden;
    } catch {
      malformedJson = true;
    }
  }

  return {
    hidden: normalizeHiddenPostIds([...queryHidden, ...bodyHidden]),
    malformedJson,
  };
}

export const GET: APIRoute = async () => {
  try {
    const hidden = await readHiddenPostIds();
    return json({ ok: true, hidden });
  } catch (error) {
    return json({ error: error instanceof Error ? error.message : 'Internal server error' }, 500);
  }
};

export const POST: APIRoute = async ({ request, url }) => {
  try {
    const { hidden, malformedJson } = await readHiddenInput(request, url);
    if (malformedJson) {
      return json({ error: 'Malformed JSON body' }, 400);
    }

    const { hidden: savedHidden, synced } = await persistHiddenPostIds(hidden);
    return json({ ok: true, hidden: savedHidden, synced });
  } catch (error) {
    return json({ error: error instanceof Error ? error.message : 'Internal server error' }, 500);
  }
};

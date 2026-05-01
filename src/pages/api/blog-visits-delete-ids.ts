import type { APIRoute } from "astro";
import { createClient } from "@supabase/supabase-js";

function allowPurge(request: Request): boolean {
  if (import.meta.env.DEV) return true;
  const secret = String(import.meta.env.VISIBILITY_SYNC_SECRET ?? "").trim();
  if (!secret) return false;
  const auth = request.headers.get("authorization")?.trim();
  return auth === `Bearer ${secret}`;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export const POST: APIRoute = async ({ request }) => {
  try {
    if (!allowPurge(request)) {
      return new Response(JSON.stringify({ ok: false, reason: "forbidden" }), {
        status: 403,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }

    const supabaseUrl = String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
    const serviceRoleKey = String(import.meta.env.SUPABASE_SERVICE_ROLE_KEY ?? "").trim();
    if (!supabaseUrl || !serviceRoleKey) {
      return new Response(
        JSON.stringify({ ok: false, reason: "missing_supabase_service_role" }),
        {
          status: 503,
          headers: { "Content-Type": "application/json; charset=utf-8" },
        },
      );
    }

    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return new Response(JSON.stringify({ ok: false, reason: "invalid_json" }), {
        status: 400,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }

    const idsRaw = (body as { ids?: unknown }).ids;
    const ids = Array.isArray(idsRaw)
      ? [...new Set(idsRaw.map((x) => String(x).trim()).filter((id) => UUID_RE.test(id)))].slice(
          0,
          500,
        )
      : [];

    if (!ids.length) {
      return new Response(JSON.stringify({ ok: true, deleted: 0 }), {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
      });
    }

    const supabase = createClient(supabaseUrl, serviceRoleKey);
    const { error } = await supabase.from("blog_visits").delete().in("id", ids);

    if (error) {
      return new Response(JSON.stringify({ ok: false, reason: error.message }), {
        status: 500,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }

    return new Response(JSON.stringify({ ok: true, deleted: ids.length }), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({
        ok: false,
        reason: error instanceof Error ? error.message : "Internal server error",
      }),
      { status: 500, headers: { "Content-Type": "application/json; charset=utf-8" } },
    );
  }
};

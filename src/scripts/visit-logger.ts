import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const TABLE = "blog_visits";

function isBot(ua: string): boolean {
  return /bot|crawl|spider|slurp|bingpreview|facebookexternal|embedly|quora|whatsapp/i.test(ua);
}

type IpApi = {
  ip?: string;
  error?: boolean;
  reason?: string;
  city?: string | null;
  region?: string | null;
  country_name?: string;
  country?: string;
  country_code?: string;
  latitude?: number | null;
  longitude?: number | null;
};

async function fetchGeo(): Promise<IpApi | null> {
  const r = await fetch("https://ipapi.co/json/", {
    mode: "cors",
    credentials: "omit",
  });
  if (!r.ok) return null;
  const j = (await r.json()) as IpApi;
  if (j.error) return null;
  return j;
}

export async function runVisitLog(): Promise<void> {
  const url = import.meta.env.PUBLIC_SUPABASE_URL;
  const key = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return;

  if (typeof navigator !== "undefined" && isBot(navigator.userAgent || "")) {
    return;
  }

  let supabase: SupabaseClient;
  try {
    supabase = createClient(String(url), String(key));
  } catch {
    return;
  }

  const data = await fetchGeo();
  if (!data || !data.ip) return;

  const row = {
    ip: data.ip,
    city: data.city ?? null,
    region: data.region ?? null,
    country: data.country_name || data.country || null,
    lat: typeof data.latitude === "number" ? data.latitude : null,
    lon: typeof data.longitude === "number" ? data.longitude : null,
    user_agent: (navigator.userAgent || "").slice(0, 1024),
  };

  const { error } = await supabase.from(TABLE).insert(row);
  if (error) {
    console.warn("[visit-log]", error.message);
  }
}

// 由 Layout 在配置好 Supabase 时 `import { runVisitLog }` 后调用

import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import {
  amapDisplayCity,
  amapRectangleCenter,
  fetchAmapV3IpByJsonp,
  isAmapIpSuccess,
} from "../lib/amap-ip";

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

function getAmapKey(): string {
  return String(import.meta.env.PUBLIC_AMAP_WEB_KEY ?? "").trim();
}

async function fetchIpapi(): Promise<IpApi | null> {
  const r = await fetch("https://ipapi.co/json/", {
    mode: "cors",
    credentials: "omit",
  });
  if (!r.ok) {
    return null;
  }
  const j = (await r.json()) as IpApi;
  if (j.error) {
    return null;
  }
  return j;
}

type VisitRow = {
  ip: string;
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
  user_agent: string;
};

function toRowFromIpapi(data: IpApi): VisitRow {
  return {
    ip: data.ip!,
    city: data.city ?? null,
    region: data.region ?? null,
    country: data.country_name || data.country || null,
    lat: typeof data.latitude === "number" ? data.latitude : null,
    lon: typeof data.longitude === "number" ? data.longitude : null,
    user_agent: (typeof navigator !== "undefined" && navigator.userAgent) ? navigator.userAgent.slice(0, 1024) : "",
  };
}

/**
 * 国内 IP 优先用高德 v3（省/市/矩形经国内库），与 ipapi 的 IP 串对齐；非国内或未配置/失败时仍用 ipapi 全文。
 */
async function buildVisitRow(): Promise<VisitRow | null> {
  const data = await fetchIpapi();
  if (!data || !data.ip) {
    return null;
  }
  const base = toRowFromIpapi(data);
  const isCn = (data.country_code || "").toUpperCase() === "CN" || /中国/i.test(String(data.country_name || ""));
  const amapKey = getAmapKey();

  if (!amapKey || !isCn) {
    return base;
  }

  try {
    const a = await fetchAmapV3IpByJsonp(amapKey, data.ip);
    if (!isAmapIpSuccess(a)) {
      return base;
    }
    const city = amapDisplayCity(a) || (data.city as string) || "—";
    const region = (a.province ?? "").trim() || data.region || null;
    const center = amapRectangleCenter(a.rectangle);
    const lat =
      center != null
        ? center.lat
        : typeof data.latitude === "number"
          ? data.latitude
          : null;
    const lon =
      center != null
        ? center.lon
        : typeof data.longitude === "number"
          ? data.longitude
          : null;

    return {
      ...base,
      city: city === "—" ? (data.city ?? null) : city,
      region,
      country: "China",
      lat,
      lon,
    };
  } catch (e) {
    console.warn("[visit-log] amap fallback to ipapi", e);
    return base;
  }
}

export async function runVisitLog(): Promise<void> {
  const url = import.meta.env.PUBLIC_SUPABASE_URL;
  const key = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    return;
  }

  if (typeof navigator !== "undefined" && isBot(navigator.userAgent || "")) {
    return;
  }

  let supabase: SupabaseClient;
  try {
    supabase = createClient(String(url), String(key));
  } catch {
    return;
  }

  const row = await buildVisitRow();
  if (!row) {
    return;
  }

  const { error } = await supabase.from(TABLE).insert({
    ip: row.ip,
    city: row.city,
    region: row.region,
    country: row.country,
    lat: row.lat,
    lon: row.lon,
    user_agent: row.user_agent,
  });
  if (error) {
    console.warn("[visit-log]", error.message);
  }
}

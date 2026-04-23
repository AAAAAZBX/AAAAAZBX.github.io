import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import {
  amapDisplayCity,
  amapRectangleCenter,
  fetchAmapV3IpByJsonp,
  isAmapIpSuccess,
} from "../lib/amap-ip";
import { siteRelativePathname } from "../lib/site-path";

const TABLE = "blog_visits";
const IP_CACHE_TABLE = "blog_visit_ip_cache";

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

async function fetchJsonWithTimeout(url: string, timeoutMs: number): Promise<unknown | null> {
  const ctrl = new AbortController();
  const tid = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, {
      mode: "cors",
      credentials: "omit",
      signal: ctrl.signal,
    });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  } finally {
    clearTimeout(tid);
  }
}

function getAmapKey(): string {
  return String(import.meta.env.PUBLIC_AMAP_WEB_KEY ?? "").trim();
}

async function fetchIpapi(): Promise<IpApi | null> {
  const j = (await fetchJsonWithTimeout("https://ipapi.co/json/", 2500)) as IpApi | null;
  if (!j) return null;
  if (j.error) {
    return null;
  }
  return j;
}

async function fetchIpOnly(): Promise<string | null> {
  const providers = [
    "https://api64.ipify.org?format=json",
    "https://api.ip.sb/geoip",
  ];
  for (const u of providers) {
    const j = (await fetchJsonWithTimeout(u, 1800)) as { ip?: string } | null;
    const ip = String(j?.ip || "").trim();
    if (ip) return ip;
  }
  return null;
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

type CachedGeo = {
  ip: string;
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
};

function mergeGeoPreferCurrent<T extends {
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
}>(current: T, cached: CachedGeo | null): T {
  if (!cached) return current;
  return {
    ...current,
    city: current.city ?? cached.city ?? null,
    region: current.region ?? cached.region ?? null,
    country: current.country ?? cached.country ?? null,
    lat: typeof current.lat === "number" ? current.lat : (cached.lat ?? null),
    lon: typeof current.lon === "number" ? current.lon : (cached.lon ?? null),
  };
}

function fallbackClientIpLikeId(): string {
  const key = "__visit_fallback_client_id__";
  try {
    if (typeof localStorage !== "undefined") {
      const cached = String(localStorage.getItem(key) || "").trim();
      if (cached) return `local-${cached}`;
      const next = Math.random().toString(36).slice(2, 10);
      localStorage.setItem(key, next);
      return `local-${next}`;
    }
  } catch {
    // ignore localStorage access errors
  }
  return `local-${Math.random().toString(36).slice(2, 10)}`;
}

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

function hasGeoInfo(row: {
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
}): boolean {
  return Boolean(
    row.city || row.region || row.country || typeof row.lat === "number" || typeof row.lon === "number"
  );
}

function hasMissingGeoField(row: {
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
}): boolean {
  return !row.city || !row.region || !row.country || typeof row.lat !== "number" || typeof row.lon !== "number";
}

async function readGeoFromCache(supabase: SupabaseClient, ip: string): Promise<CachedGeo | null> {
  if (!ip || /^local-/i.test(ip)) return null;
  try {
    const r = await supabase
      .from(IP_CACHE_TABLE)
      .select("ip, city, region, country, lat, lon")
      .eq("ip", ip)
      .limit(1)
      .maybeSingle();
    if (!r.error && r.data && hasGeoInfo(r.data)) {
      return r.data as CachedGeo;
    }
  } catch {
    // ignore missing table / policy / network errors
  }

  // 回退：即使缓存表未建，也尝试从历史访问中复用该 IP 的地理信息
  try {
    const r2 = await supabase
      .from(TABLE)
      .select("ip, city, region, country, lat, lon")
      .eq("ip", ip)
      .order("created_at", { ascending: false })
      .limit(12);
    if (!r2.error && Array.isArray(r2.data)) {
      const found = r2.data.find((x) => hasGeoInfo(x));
      if (found) {
        return found as CachedGeo;
      }
    }
  } catch {
    // ignore
  }
  return null;
}

async function writeGeoCache(supabase: SupabaseClient, row: VisitRow): Promise<void> {
  if (!row.ip || /^local-/i.test(row.ip) || !hasGeoInfo(row)) return;
  try {
    await supabase.from(IP_CACHE_TABLE).upsert(
      {
        ip: row.ip,
        city: row.city,
        region: row.region,
        country: row.country,
        lat: row.lat,
        lon: row.lon,
      },
      { onConflict: "ip" }
    );
  } catch {
    // ignore when cache table is not ready
  }
}

/**
 * 国内 IP 优先用高德 v3（省/市/矩形经国内库），与 ipapi 的 IP 串对齐；非国内或未配置/失败时仍用 ipapi 全文。
 */
async function buildVisitRow(): Promise<VisitRow | null> {
  const data = await fetchIpapi();
  if (!data || !data.ip) {
    // 某些设备/网络会拦截 ipapi；退化为仅拿 IP，也拿不到就用本机稳定标识，确保仍能记访问。
    const ip = (await fetchIpOnly()) || fallbackClientIpLikeId();
    return {
      ip,
      city: null,
      region: null,
      country: null,
      lat: null,
      lon: null,
      user_agent:
        typeof navigator !== "undefined" && navigator.userAgent
          ? navigator.userAgent.slice(0, 1024)
          : "",
    };
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

export async function runVisitLog(options?: { siteBasePath?: string }): Promise<void> {
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

  // 快速拿 IP 并命中缓存：同 IP 曾解析过地理信息时可立即复用。
  const ipHint = await fetchIpOnly();
  if (ipHint) {
    const cached = await readGeoFromCache(supabase, ipHint);
    if (cached) {
      const siteBase = options?.siteBasePath ?? "/";
      const pagePath =
        typeof window !== "undefined" && window.location?.pathname
          ? siteRelativePathname(window.location.pathname, siteBase)
          : null;
      const fromCachePayload = {
        ip: cached.ip,
        city: cached.city,
        region: cached.region,
        country: cached.country,
        lat: cached.lat,
        lon: cached.lon,
        user_agent:
          typeof navigator !== "undefined" && navigator.userAgent
            ? navigator.userAgent.slice(0, 1024)
            : "",
      };
      let ins = await supabase.from(TABLE).insert({
        ...fromCachePayload,
        page_path: pagePath,
      });
      if (!ins.error) return;
      const insMsg = String(ins.error?.message || "");
      if (/page_path/i.test(insMsg) || /PGRST204/i.test(insMsg)) {
        ins = await supabase.from(TABLE).insert(fromCachePayload);
        if (!ins.error) return;
      }
    }
  }

  const rowRaw = await Promise.race<VisitRow | null>([
    buildVisitRow(),
    new Promise<VisitRow>((resolve) =>
      setTimeout(
        () =>
          resolve({
            ip: fallbackClientIpLikeId(),
            city: null,
            region: null,
            country: null,
            lat: null,
            lon: null,
            user_agent:
              typeof navigator !== "undefined" && navigator.userAgent
                ? navigator.userAgent.slice(0, 1024)
                : "",
          }),
        3000,
      ),
    ),
  ]);
  if (!rowRaw) {
    return;
  }

  // 第二道兜底：即便前面的 ipHint 失败，也用最终解析出的 row.ip 再查一次缓存补齐空地理字段。
  let row: VisitRow = rowRaw;
  if (row.ip && !/^local-/i.test(row.ip) && hasMissingGeoField(row)) {
    const cachedByRowIp = await readGeoFromCache(supabase, row.ip);
    row = mergeGeoPreferCurrent(row, cachedByRowIp);
  }

  const siteBase = options?.siteBasePath ?? "/";
  const pagePath =
    typeof window !== "undefined" && window.location?.pathname
      ? siteRelativePathname(window.location.pathname, siteBase)
      : null;

  const basePayload = {
    ip: row.ip,
    city: row.city,
    region: row.region,
    country: row.country,
    lat: row.lat,
    lon: row.lon,
    user_agent: row.user_agent,
  };

  let { error } = await supabase.from(TABLE).insert({
    ...basePayload,
    page_path: pagePath,
  });
  if (error) {
    const msg = String(error.message || "");
    // 兼容旧表：尚未执行 page_path 迁移时，退回旧字段写入，避免统计全停。
    if (/page_path/i.test(msg) || /PGRST204/i.test(msg)) {
      const retry = await supabase.from(TABLE).insert(basePayload);
      error = retry.error;
      if (!error) {
        console.warn("[visit-log] page_path column missing, inserted without page_path");
        return;
      }
    }
    console.warn("[visit-log]", msg);
    return;
  }
  await writeGeoCache(supabase, row);
}

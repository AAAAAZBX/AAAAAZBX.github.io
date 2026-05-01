import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import {
  amapDisplayCity,
  amapRectangleCenter,
  fetchAmapV3IpByJsonp,
  isAmapIpSuccess,
} from "../lib/amap-ip";
import { getVisitBehaviorPayload } from "../lib/visit-behavior";
import { isOwnerIp } from "../lib/visitor-env";
import { isOperatorConsolePath, siteRelativePathname } from "../lib/site-path";

const TABLE = "blog_visits";
const IP_CACHE_TABLE = "blog_visit_ip_cache";

/** 不再提前 return：爬虫也入库，由 visitor_kind / risk_score 标记。 */

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
  org?: string | null;
};

function normalizeCountryCode(cc: string | null | undefined): string | null {
  const c = String(cc ?? "")
    .trim()
    .toUpperCase();
  return c.length === 2 && /^[A-Z]{2}$/.test(c) ? c : null;
}

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

function sliceUa(): string {
  return typeof navigator !== "undefined" && navigator.userAgent ? navigator.userAgent.slice(0, 1024) : "";
}

/** ipapi 失败时：用 ip.sb 按公网 IP 补全国家/城市/坐标（浏览器可 CORS）。 */
function visitRowFromIpsbJson(j: unknown, ip: string): VisitRow | null {
  if (!j || typeof j !== "object") return null;
  const o = j as Record<string, unknown>;
  const country = o.country != null ? String(o.country).trim() : "";
  const city = o.city != null ? String(o.city).trim() : "";
  const region = o.region != null ? String(o.region).trim() : "";
  const latRaw = o.latitude;
  const lonRaw = o.longitude;
  const latN =
    typeof latRaw === "number"
      ? latRaw
      : latRaw != null && String(latRaw).trim() !== ""
        ? Number(latRaw)
        : NaN;
  const lonN =
    typeof lonRaw === "number"
      ? lonRaw
      : lonRaw != null && String(lonRaw).trim() !== ""
        ? Number(lonRaw)
        : NaN;
  const lat = Number.isFinite(latN) ? latN : null;
  const lon = Number.isFinite(lonN) ? lonN : null;
  if (!country && !city && !region && lat == null && lon == null) return null;
  const isp =
    String(o.isp ?? o.org ?? o.organization ?? "").trim() || null;
  const country_code = normalizeCountryCode(o.country_code != null ? String(o.country_code) : null);
  return {
    ip,
    city: city || null,
    region: region || null,
    country: country || null,
    lat,
    lon,
    user_agent: sliceUa(),
    net_isp: isp,
    country_code,
  };
}

async function fetchIpsbGeoByIp(ip: string): Promise<VisitRow | null> {
  if (!ip || /^local-/i.test(ip)) return null;
  const j = await fetchJsonWithTimeout(`https://api.ip.sb/geoip/${encodeURIComponent(ip)}`, 5000);
  const row = visitRowFromIpsbJson(j, ip);
  if (!row) return null;
  const o = (j && typeof j === "object" ? j : {}) as Record<string, unknown>;
  const cc = normalizeCountryCode(o.country_code != null ? String(o.country_code) : null);
  return { ...row, country_code: cc ?? row.country_code ?? null };
}

type VisitRow = {
  ip: string;
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
  user_agent: string;
  net_isp?: string | null;
  country_code?: string | null;
};

type CachedGeo = {
  ip: string;
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
  country_code?: string | null;
};

function mergeGeoPreferCurrent<T extends {
  city: string | null;
  region: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
  country_code?: string | null;
}>(current: T, cached: CachedGeo | null): T {
  if (!cached) return current;
  return {
    ...current,
    city: current.city ?? cached.city ?? null,
    region: current.region ?? cached.region ?? null,
    country: current.country ?? cached.country ?? null,
    lat: typeof current.lat === "number" ? current.lat : (cached.lat ?? null),
    lon: typeof current.lon === "number" ? current.lon : (cached.lon ?? null),
    country_code: current.country_code ?? cached.country_code ?? null,
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
  const org = String(data.org ?? "").trim() || null;
  return {
    ip: data.ip!,
    city: data.city ?? null,
    region: data.region ?? null,
    country: data.country_name || data.country || null,
    lat: typeof data.latitude === "number" ? data.latitude : null,
    lon: typeof data.longitude === "number" ? data.longitude : null,
    user_agent: (typeof navigator !== "undefined" && navigator.userAgent) ? navigator.userAgent.slice(0, 1024) : "",
    net_isp: org,
    country_code: normalizeCountryCode(data.country_code),
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
    let r = await supabase
      .from(IP_CACHE_TABLE)
      .select("ip, city, region, country, country_code, lat, lon")
      .eq("ip", ip)
      .limit(1)
      .maybeSingle();
    if (
      r.error &&
      /country_code|column|schema cache|PGRST204/i.test(String(r.error.message || ""))
    ) {
      r = await supabase
        .from(IP_CACHE_TABLE)
        .select("ip, city, region, country, lat, lon")
        .eq("ip", ip)
        .limit(1)
        .maybeSingle();
    }
    if (!r.error && r.data && hasGeoInfo(r.data)) {
      return r.data as CachedGeo;
    }
  } catch {
    // ignore missing table / policy / network errors
  }

  // 回退：即使缓存表未建，也尝试从历史访问中复用该 IP 的地理信息
  try {
    let r2 = await supabase
      .from(TABLE)
      .select("ip, city, region, country, country_code, lat, lon")
      .eq("ip", ip)
      .order("created_at", { ascending: false })
      .limit(12);
    if (
      r2.error &&
      /country_code|column|schema cache|PGRST204/i.test(String(r2.error.message || ""))
    ) {
      r2 = await supabase
        .from(TABLE)
        .select("ip, city, region, country, lat, lon")
        .eq("ip", ip)
        .order("created_at", { ascending: false })
        .limit(12);
    }
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
    const payload: Record<string, unknown> = {
      ip: row.ip,
      city: row.city,
      region: row.region,
      country: row.country,
      lat: row.lat,
      lon: row.lon,
      country_code: row.country_code ?? null,
    };
    let u = await supabase.from(IP_CACHE_TABLE).upsert(payload, { onConflict: "ip" });
    if (
      u.error &&
      /country_code|column|schema cache|PGRST204/i.test(String(u.error.message || ""))
    ) {
      delete payload.country_code;
      u = await supabase.from(IP_CACHE_TABLE).upsert(payload, { onConflict: "ip" });
    }
  } catch {
    // ignore when cache table is not ready
  }
}

/**
 * 国内 IP 优先高德 v3；ipapi 可用时走 ipapi；不可用（超时/拦截）时用 ip.sb 按公网 IP 补地理，避免只有 IP 没有国家城市。
 */
async function buildVisitRow(): Promise<VisitRow> {
  const data = await fetchIpapi();
  if (!data?.ip) {
    const ip = (await fetchIpOnly()) || fallbackClientIpLikeId();
    if (/^local-/i.test(ip)) {
      return {
        ip,
        city: null,
        region: null,
        country: null,
        country_code: null,
        lat: null,
        lon: null,
        user_agent: sliceUa(),
        net_isp: null,
      };
    }
    const fromSb = await fetchIpsbGeoByIp(ip);
    if (fromSb) {
      const isCn = fromSb.country_code === "CN" || /中国/.test(String(fromSb.country || ""));
      const amapKey = getAmapKey();
      if (amapKey && isCn) {
        try {
          const a = await fetchAmapV3IpByJsonp(amapKey, ip);
          if (isAmapIpSuccess(a)) {
            const city = amapDisplayCity(a) || (fromSb.city as string) || "—";
            const region = (a.province ?? "").trim() || fromSb.region || null;
            const center = amapRectangleCenter(a.rectangle);
            const lat = center != null ? center.lat : fromSb.lat;
            const lon = center != null ? center.lon : fromSb.lon;
            return {
              ...fromSb,
              city: city === "—" ? fromSb.city : city,
              region,
              country: "China",
              country_code: "CN",
              lat,
              lon,
            };
          }
        } catch (e) {
          console.warn("[visit-log] amap after ip.sb", e);
        }
      }
      return fromSb;
    }
    return {
      ip,
      city: null,
      region: null,
      country: null,
      country_code: null,
      lat: null,
      lon: null,
      user_agent: sliceUa(),
      net_isp: null,
    };
  }
  let base = toRowFromIpapi(data);
  if (!hasGeoInfo(base)) {
    const r = await fetchIpsbGeoByIp(data.ip);
    if (r) {
      base = {
        ...base,
        city: r.city ?? base.city,
        region: r.region ?? base.region,
        country: r.country ?? base.country,
        lat: r.lat ?? base.lat,
        lon: r.lon ?? base.lon,
        net_isp: r.net_isp ?? base.net_isp ?? null,
        country_code: r.country_code ?? base.country_code ?? null,
      };
    }
  }
  const isCn =
    (data.country_code || "").toUpperCase() === "CN" ||
    /中国/i.test(String(data.country_name || "")) ||
    (base.country && /china|中国/i.test(base.country));
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
      country_code: "CN",
      lat,
      lon,
    };
  } catch (e) {
    console.warn("[visit-log] amap fallback to ipapi", e);
    return base;
  }
}

const OPTIONAL_VISIT_COLS = [
  "page_path",
  "session_id",
  "referrer",
  "path_sequence",
  "path_chain",
  "visitor_kind",
  "risk_score",
  "net_isp",
  "dwell_ms",
  "country_code",
] as const;

async function insertBlogVisit(supabase: SupabaseClient, payload: Record<string, unknown>): Promise<boolean> {
  let p: Record<string, unknown> = { ...payload };
  let lastMsg = "";
  for (let attempt = 0; attempt <= OPTIONAL_VISIT_COLS.length + 2; attempt += 1) {
    const ins = await supabase.from(TABLE).insert(p);
    if (!ins.error) {
      return true;
    }
    lastMsg = String(ins.error.message || "");
    if (!/PGRST204|column|schema cache|Could not find the/i.test(lastMsg)) {
      break;
    }
    let dropped = false;
    const low = lastMsg.toLowerCase();
    for (const key of OPTIONAL_VISIT_COLS) {
      if (low.includes(key.toLowerCase()) && key in p) {
        delete p[key];
        dropped = true;
        break;
      }
    }
    if (!dropped) {
      const next = OPTIONAL_VISIT_COLS.find((k) => k in p);
      if (!next) break;
      delete p[next];
    }
  }
  console.warn("[visit-log]", lastMsg || "insert failed");
  return false;
}

export async function runVisitLog(options?: { siteBasePath?: string }): Promise<void> {
  const url = import.meta.env.PUBLIC_SUPABASE_URL;
  const key = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    return;
  }

  /* 本机标记：以下任一条件满足即跳过入库
     - localStorage bx_owner_skip_visit_log=1 (admin 登录时写入)
     - sessionStorage bx_admin_unlocked_v1=1 (AdminGate 当前标签页已解锁)
  */
  try {
    var _ls = typeof localStorage !== "undefined" ? localStorage.getItem("bx_owner_skip_visit_log") : null;
    var _ss = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("bx_admin_unlocked_v1") : null;
    console.log("[visit-log] internal check: localStorage=" + _ls + " sessionStorage=" + _ss);
    if (_ls === "1") {
      console.log("[visit-log] skipped by localStorage flag");
      return;
    }
    if (_ss === "1") {
      console.log("[visit-log] skipped by sessionStorage flag");
      return;
    }
  } catch (_) { /* ignore */ }

  const siteBase = options?.siteBasePath ?? "/";
  const pagePath =
    typeof window !== "undefined" && window.location?.pathname
      ? siteRelativePathname(window.location.pathname, siteBase)
      : null;
  if (pagePath != null && isOperatorConsolePath(pagePath)) {
    return;
  }

  let supabase: SupabaseClient;
  try {
    supabase = createClient(String(url), String(key));
  } catch {
    return;
  }
  const uaNav =
    typeof navigator !== "undefined" && navigator.userAgent
      ? navigator.userAgent.slice(0, 1024)
      : "";

  let row: VisitRow | null = null;
  let usedCacheGeo = false;

  const ipHint = await fetchIpOnly();
  if (ipHint) {
    const cached = await readGeoFromCache(supabase, ipHint);
    if (cached) {
      row = {
        ip: cached.ip,
        city: cached.city,
        region: cached.region,
        country: cached.country,
        country_code: cached.country_code ?? null,
        lat: cached.lat,
        lon: cached.lon,
        user_agent: uaNav,
        net_isp: null,
      };
      usedCacheGeo = true;
    }
  }

  if (!row) {
    row = await buildVisitRow();
    if (row.ip && !/^local-/i.test(row.ip) && hasMissingGeoField(row)) {
      const cachedByRowIp = await readGeoFromCache(supabase, row.ip);
      row = mergeGeoPreferCurrent(row, cachedByRowIp);
    }
  }

  if (isOwnerIp(row.ip)) {
    console.log("[visit-log] skipped by PUBLIC_OWNER_IP match: " + row.ip);
    return;
  }

  console.log("[visit-log] proceeding to insert visit record, ip=" + row.ip);

  const behavior = getVisitBehaviorPayload({ pagePath, userAgent: row.user_agent });

  const ok = await insertBlogVisit(supabase, {
    ip: row.ip,
    city: row.city,
    region: row.region,
    country: row.country,
    country_code: row.country_code ?? null,
    lat: row.lat,
    lon: row.lon,
    user_agent: row.user_agent,
    net_isp: row.net_isp ?? null,
    dwell_ms: null,
    page_path: pagePath,
    ...behavior,
  });

  if (!ok) {
    return;
  }
  if (!usedCacheGeo) {
    await writeGeoCache(supabase, row);
  }
}

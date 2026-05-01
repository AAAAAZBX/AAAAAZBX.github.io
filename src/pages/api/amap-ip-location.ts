import type { APIRoute } from "astro";
import {
  fetchAmapV5IpLocation,
  isAmapV5IpSuccess,
} from "../../lib/amap-ip";

/**
 * 浏览器无法直连高德 v5（易被 CORS 拦）；本地 dev / SSR 宿主通过此路由代查。
 * v5 支持 IPv6；需在控制台开通「高级 IP 定位」配额，与 PUBLIC_AMAP_WEB_KEY 同一 Web 服务 Key。
 */
export const GET: APIRoute = async ({ url }) => {
  const key = String(import.meta.env.PUBLIC_AMAP_WEB_KEY ?? "").trim();
  if (!key) {
    return new Response(JSON.stringify({ ok: false, reason: "no_amap_key" }), {
      status: 503,
      headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
    });
  }

  const ip = new URL(url).searchParams.get("ip")?.trim();
  if (!ip) {
    return new Response(JSON.stringify({ ok: false, reason: "missing_ip" }), {
      status: 400,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }

  try {
    const a = await fetchAmapV5IpLocation(key, ip);
    if (!isAmapV5IpSuccess(a)) {
      return new Response(
        JSON.stringify({
          ok: false,
          reason: "amap_v5_not_ok",
          info: a.info ?? null,
          infocode: a.infocode ?? null,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
        },
      );
    }

    const parts = String(a.location ?? "")
      .split(",")
      .map((s) => s.trim());
    const lon = parts[0] != null ? Number(parts[0]) : NaN;
    const lat = parts[1] != null ? Number(parts[1]) : NaN;

    return new Response(
      JSON.stringify({
        ok: true,
        province: a.province ?? null,
        city: a.city ?? null,
        district: a.district ?? null,
        lat: Number.isFinite(lat) ? lat : null,
        lon: Number.isFinite(lon) ? lon : null,
        isp: a.isp ?? null,
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
      },
    );
  } catch (e) {
    return new Response(
      JSON.stringify({
        ok: false,
        reason: "exception",
        message: e instanceof Error ? e.message : String(e),
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      },
    );
  }
};

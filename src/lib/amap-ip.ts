/**
 * 高德开放 Web 服务 v3 /ip：国内 IP 城市/省/矩形，用于替代境外 ipapi 的偏差。
 * 浏览器内用 JSONP 调用，避免 restapi 对浏览器的 CORS 限制。
 * 需在 高德控制台 申请 **Web 服务** key，并设置 Referer 白名单（如 https://你的域名/* 与本地开发端口）。
 * @see https://lbs.amap.com/api/webservice/guide/api/ipconfig
 */

export type AmapV3IpResult = {
  status: string;
  info: string;
  infocode: string;
  province?: string;
  city?: string | string[];
  adcode?: string;
  rectangle?: string;
};

/**
 * 使用 JSONP 请求 v3/ip；建议传入公网 IP（如来自 ipapi），与入库 IP 一致。
 * 不填 `ip` 时高德以「请求方」为定位对象，在部分环境下可能非用户真实出口 IP。
 */
export function fetchAmapV3IpByJsonp(key: string, ip: string | undefined): Promise<AmapV3IpResult> {
  const k = String(key).trim();
  if (!k) {
    return Promise.reject(new Error("amap key empty"));
  }
  return new Promise((resolve, reject) => {
    const fn = `__amap_ip_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const t = window.setTimeout(() => {
      try {
        cleanup();
      } catch {
        /* empty */
      }
      reject(new Error("amap jsonp timeout"));
    }, 12_000);

    function cleanup() {
      clearTimeout(t);
      delete (window as unknown as Record<string, unknown>)[fn];
    }

    (window as unknown as Record<string, (data: AmapV3IpResult) => void>)[fn] = (data) => {
      cleanup();
      resolve(data);
    };

    const s = document.createElement("script");
    const p = new URLSearchParams();
    p.set("key", k);
    p.set("output", "json");
    p.set("callback", fn);
    if (ip) {
      p.set("ip", ip);
    }
    s.src = `https://restapi.amap.com/v3/ip?${p.toString()}`;
    s.onerror = () => {
      cleanup();
      reject(new Error("amap jsonp load error"));
    };
    document.head.appendChild(s);
  });
}

/** 矩形左右下角与右上角，中心点作地图用坐标 */
export function amapRectangleCenter(rectangle: string | undefined): { lat: number; lon: number } | null {
  if (!rectangle || !rectangle.includes(";")) {
    return null;
  }
  const [lb, ru] = rectangle.split(";");
  const p1 = lb.split(",");
  const p2 = ru.split(",");
  if (p1.length < 2 || p2.length < 2) {
    return null;
  }
  const lo1 = Number(p1[0]);
  const la1 = Number(p1[1]);
  const lo2 = Number(p2[0]);
  const la2 = Number(p2[1]);
  if (![lo1, la1, lo2, la2].every((n) => Number.isFinite(n))) {
    return null;
  }
  return { lon: (lo1 + lo2) / 2, lat: (la1 + la2) / 2 };
}

/** 直辖市名或城市名、省名，供入库 */
export function amapDisplayCity(a: AmapV3IpResult): string {
  const prov = (a.province ?? "").trim();
  let c = a.city;
  const cityStr = Array.isArray(c) ? (c[0] != null ? String(c[0]) : "") : String(c ?? "").trim();
  if (cityStr && cityStr !== "[]") {
    return cityStr;
  }
  const 直辖 = new Set(["北京市", "上海市", "天津市", "重庆市"]);
  if (prov && 直辖.has(prov)) {
    return prov;
  }
  return prov || "—";
}

export function isAmapIpSuccess(a: AmapV3IpResult): boolean {
  const code = String(a.infocode ?? "");
  if (a.status !== "1" || code !== "10000") {
    return false;
  }
  const prov = (a.province ?? "").trim();
  if (!prov || prov === "局域网") {
    return false;
  }
  return true;
}

/**
 * 高德 Web 服务 **高级** IP 定位 v5（支持 IPv4/IPv6、国内外）。
 * 与 v3 不同：需在控制台通过工单开通「高级 IP 定位」，否则接口会返回权限/服务类错误。
 * @see https://lbs.amap.com/api/webservice/guide/api-advanced/ip
 */
export type AmapV5IpLocationResult = {
  status: string;
  info: string;
  infocode: string;
  country?: string;
  province?: string;
  city?: string;
  district?: string;
  adcode?: string;
  /** 经度在前、纬度在后，逗号分隔 */
  location?: string;
  isp?: string;
  ip?: string;
};

/** 根据地址形态推断高德 `type`：4=IPv4，6=IPv6 */
export function amapIpQueryType(ip: string): 4 | 6 {
  return String(ip).includes(":") ? 6 : 4;
}

/**
 * 服务端或任意支持 `fetch` 的环境调用 v5/ip/location（非 JSONP，无浏览器 CORS 问题）。
 * IPv6 示例：`2001:250:5006:2311:fd07:6587:b8d2:4aec` → `type=6`
 */
export async function fetchAmapV5IpLocation(
  key: string,
  ip: string,
  init?: RequestInit,
): Promise<AmapV5IpLocationResult> {
  const k = String(key).trim();
  if (!k) {
    throw new Error("amap key empty");
  }
  const addr = String(ip).trim();
  if (!addr) {
    throw new Error("ip empty");
  }
  const type = amapIpQueryType(addr);
  const u = new URL("https://restapi.amap.com/v5/ip/location");
  u.searchParams.set("key", k);
  u.searchParams.set("ip", addr);
  u.searchParams.set("type", String(type));
  const res = await fetch(u.toString(), init);
  if (!res.ok) {
    throw new Error(`amap v5 http ${res.status}`);
  }
  return (await res.json()) as AmapV5IpLocationResult;
}

export function isAmapV5IpSuccess(a: AmapV5IpLocationResult): boolean {
  return a.status === "1" && String(a.infocode ?? "") === "10000";
}

/** 国家/省/市/区拼接展示（与 v3 直辖市展示习惯接近） */
export function amapV5DisplayRegion(a: AmapV5IpLocationResult): string {
  const parts = [a.country, a.province, a.city, a.district]
    .map((x) => (x != null ? String(x).trim() : ""))
    .filter(Boolean);
  return parts.join(" ") || "—";
}

import { mergeIpToFrontOfStorage } from "./local-admin-ip";

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

/** 与 visit-logger 一致：仅取公网 IP 字符串 */
async function fetchPublicIpQuick(): Promise<string | null> {
  const providers = ["https://api64.ipify.org?format=json", "https://api.ip.sb/geoip"];
  for (const u of providers) {
    const j = (await fetchJsonWithTimeout(u, 2200)) as { ip?: string } | null;
    const ip = String(j?.ip || "").trim();
    if (ip) return ip;
  }
  return null;
}

function isLoopbackHostname(hostname: string): boolean {
  const h = hostname.trim().toLowerCase();
  return h === "localhost" || h === "127.0.0.1" || h === "[::1]" || h === "::1";
}

/**
 * 本地 dev（localhost）尽早把当前公网 IP 写入 bx_admin_ips，
 * 避免 visit-logger 在 admin 脚本跑完之前就入库。
 */
export async function ensurePrimedOnDevLocalhost(): Promise<void> {
  if (!import.meta.env.DEV) return;
  if (typeof window === "undefined") return;
  if (!isLoopbackHostname(window.location.hostname)) return;

  const ip = await fetchPublicIpQuick();
  if (!ip || /^local-/i.test(ip)) return;
  mergeIpToFrontOfStorage(ip);
}

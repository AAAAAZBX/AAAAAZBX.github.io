/**
 * 浏览器端「本机公网 IP」白名单（localStorage），用于开发时跳过访客入库 / 从画像列表剔除。
 */

export const ADMIN_IP_STORAGE_KEY = "bx_admin_ips";

/** 规范化 IPv4 / IPv6 文本便于比较（忽略大小写、zone id、IPv6 压缩写法差异） */
export function canonicalIpForCompare(ip: string): string {
  let s = ip.trim().toLowerCase();
  if (s.startsWith("[") && s.endsWith("]")) s = s.slice(1, -1);
  const zi = s.indexOf("%");
  if (zi !== -1) s = s.slice(0, zi);
  if (!s.includes(":")) return s;

  const sep = "::";
  const di = s.indexOf(sep);
  if (di === -1) {
    return s
      .split(":")
      .map((p) => (p === "" ? "0" : p).padStart(4, "0"))
      .join(":");
  }
  const head = s.slice(0, di);
  const tail = s.slice(di + sep.length);
  const left = head ? head.split(":").filter(Boolean) : [];
  const right = tail ? tail.split(":").filter(Boolean) : [];
  const missing = 8 - left.length - right.length;
  const mid = Array(Math.max(0, missing)).fill("0");
  return [...left, ...mid, ...right].map((p) => p.padStart(4, "0")).join(":");
}

export function ipsEquivalentCanonically(a: string, b: string): boolean {
  return canonicalIpForCompare(a) === canonicalIpForCompare(b);
}

export function parseStoredAdminIps(raw: string | null | undefined): string[] {
  if (raw == null || raw === "") return [];
  try {
    const j = JSON.parse(raw) as unknown;
    if (Array.isArray(j)) {
      return j.map((x) => String(x).trim()).filter(Boolean);
    }
  } catch {
    /* fallthrough */
  }
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function ipMatchesAnyStored(ip: string | null | undefined, storedList: string[]): boolean {
  if (!ip || storedList.length === 0) return false;
  const c = canonicalIpForCompare(ip);
  return storedList.some((s) => canonicalIpForCompare(s) === c);
}

/** 与 admin 页 pill 逻辑一致：去重后前置，最多 max 条 */
export function mergeIpToFrontOfStorage(ip: string, max = 10): void {
  try {
    if (typeof localStorage === "undefined") return;
    const t = ip.trim();
    if (!t || /^local-/i.test(t)) return;
    const raw = localStorage.getItem(ADMIN_IP_STORAGE_KEY);
    let ips = parseStoredAdminIps(raw);
    ips = ips.filter((x) => !ipsEquivalentCanonically(x, t));
    ips.unshift(t);
    if (ips.length > max) ips.length = max;
    localStorage.setItem(ADMIN_IP_STORAGE_KEY, JSON.stringify(ips));
  } catch {
    /* ignore */
  }
}

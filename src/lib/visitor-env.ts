/** 仅当存在 URL 与 anon key 时启用前端记访问（写入 blog_visits） */
export function isVisitorLogConfigured(): boolean {
  const u = String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
  const k = String(import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
  return Boolean(u && k && u.startsWith("http"));
}

/** 从 PUBLIC_OWNER_IP 读取本机 IP 列表（逗号/空格分隔），匹配时跳过记录 */
export function getOwnerIps(): string[] {
  const raw = String(import.meta.env.PUBLIC_OWNER_IP ?? "").trim();
  if (!raw) return [];
  return raw
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** 检查 ip 是否匹配本机 IP 列表中的任意一个 */
export function isOwnerIp(ip: string): boolean {
  if (!ip) return false;
  const ownerIps = getOwnerIps();
  return ownerIps.length > 0 && ownerIps.includes(ip);
}

export function getSupabaseClientConfig(): { url: string; anonKey: string } | null {
  const url = String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
  const anonKey = String(import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
  if (!url || !anonKey) return null;
  return { url, anonKey };
}

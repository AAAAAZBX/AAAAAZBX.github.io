/** 仅当存在 URL 与 anon key 时启用前端记访问（写入 blog_visits） */
export function isVisitorLogConfigured(): boolean {
  const u = String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
  const k = String(import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
  return Boolean(u && k && u.startsWith("http"));
}

export function getSupabaseClientConfig(): { url: string; anonKey: string } | null {
  const url = String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
  const anonKey = String(import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
  if (!url || !anonKey) return null;
  return { url, anonKey };
}

/** 仅当存在 URL 与 anon key 时启用前端记访问（写入 blog_visits） */
export function isVisitorLogConfigured(): boolean {
  const u = import.meta.env.PUBLIC_SUPABASE_URL;
  const k = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;
  return Boolean(u && k && String(u).startsWith("http"));
}

export function getSupabaseClientConfig(): { url: string; anonKey: string } | null {
  const url = import.meta.env.PUBLIC_SUPABASE_URL;
  const anonKey = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anonKey) return null;
  return { url: String(url), anonKey: String(anonKey) };
}

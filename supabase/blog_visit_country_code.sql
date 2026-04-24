-- 可选：为 blog_visits / IP 缓存增加 ISO 3166-1 alpha-2，便于前端展示国旗等。
-- 在 Supabase SQL Editor 执行一次；未执行时 visit-logger / 读表会自动降级（不写、不选该列）。

alter table public.blog_visits add column if not exists country_code text;
alter table public.blog_visit_ip_cache add column if not exists country_code text;

create index if not exists blog_visits_country_code_idx on public.blog_visits (country_code);

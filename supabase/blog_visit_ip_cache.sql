-- 可选增强：IP 地理缓存表。用于“同一 IP 二次访问”直接复用城市/经纬度，减少外部 IP 接口抖动导致的空值。

create table if not exists public.blog_visit_ip_cache (
  ip text primary key,
  city text,
  region text,
  country text,
  lat double precision,
  lon double precision,
  updated_at timestamptz not null default now()
);

create index if not exists blog_visit_ip_cache_updated_at_idx
  on public.blog_visit_ip_cache (updated_at desc);

alter table public.blog_visit_ip_cache enable row level security;

drop policy if exists "blog_visit_ip_cache_select_anon" on public.blog_visit_ip_cache;
create policy "blog_visit_ip_cache_select_anon" on public.blog_visit_ip_cache
  for select to anon
  using (true);

drop policy if exists "blog_visit_ip_cache_insert_anon" on public.blog_visit_ip_cache;
create policy "blog_visit_ip_cache_insert_anon" on public.blog_visit_ip_cache
  for insert to anon
  with check (true);

drop policy if exists "blog_visit_ip_cache_update_anon" on public.blog_visit_ip_cache;
create policy "blog_visit_ip_cache_update_anon" on public.blog_visit_ip_cache
  for update to anon
  using (true)
  with check (true);

grant select, insert, update on public.blog_visit_ip_cache to anon, authenticated, service_role;

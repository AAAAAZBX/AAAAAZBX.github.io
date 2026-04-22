-- 在 Supabase SQL Editor 中执行一次。用于访客 IP/城市 记录与 Tags 页访客区块。
-- 公网 anon 可插入与读取： anyone with 你的网站里的 anon key 能读表；若只希望自己可见，应改用服务端或关闭 select 在 Dashboard 看数据。

create table if not exists public.blog_visits (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  ip text not null,
  city text,
  region text,
  country text,
  lat double precision,
  lon double precision,
  user_agent text
);

create index if not exists blog_visits_created_at_idx on public.blog_visits (created_at desc);
create index if not exists blog_visits_ip_idx on public.blog_visits (ip);

alter table public.blog_visits enable row level security;

-- 与 Astro 的 PUBLIC_SUPABASE_ANON_KEY 对应（anon 角色）
drop policy if exists "blog_visits_insert_anon" on public.blog_visits;
create policy "blog_visits_insert_anon" on public.blog_visits
  for insert to anon
  with check (true);

drop policy if exists "blog_visits_select_anon" on public.blog_visits;
create policy "blog_visits_select_anon" on public.blog_visits
  for select to anon
  using (true);

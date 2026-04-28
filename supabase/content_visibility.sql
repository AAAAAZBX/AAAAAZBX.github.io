-- 内容可见性管理表。用于替代本地 JSON 文件，通过 Supabase 在本地和公网之间同步隐藏文章列表。
-- 在 Supabase SQL Editor 中执行即可。

drop table if exists public.content_visibility;

create table public.content_visibility (
  id int primary key default 1,
  hidden_posts jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now(),
  constraint content_visibility_single_row check (id = 1)
);

alter table public.content_visibility enable row level security;

-- 任何人可读（anon key 即可 SELECT）
create policy "anyone can select content_visibility"
  on public.content_visibility for select
  using (true);

-- 插入初始行
insert into public.content_visibility (id, hidden_posts)
values (1, '[]'::jsonb)
on conflict (id) do nothing;

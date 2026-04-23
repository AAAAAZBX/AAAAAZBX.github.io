-- 在 Supabase SQL Editor 中执行一次：为已有表增加文章路径，并创建按 path 聚合计数的视图（Tags 页「每篇文章阅读量」用）。
-- 新仓库若已用 `blog_visits.sql` 且含 `page_path` 可跳过本文件。

alter table if exists public.blog_visits
  add column if not exists page_path text;

create index if not exists blog_visits_page_path_idx on public.blog_visits (page_path)
  where page_path is not null and btrim(page_path) <> '';

-- 与 blog_visits 的 RLS/权限一致，anon 可 select 时即可读本视图
drop view if exists public.blog_visit_path_stats;
create or replace view public.blog_visit_path_stats as
select
  b.page_path,
  count(*)::bigint as views
from public.blog_visits b
where b.page_path is not null
  and btrim(b.page_path) <> ''
group by b.page_path;

grant select on public.blog_visit_path_stats to anon, authenticated, service_role;

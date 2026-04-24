-- 访客画像扩展（与现有静态博客 + Supabase anon 一致；生产环境用 Postgres，无需 SQLite）。
-- 在 Supabase SQL Editor 中执行一次。

alter table public.blog_visits add column if not exists session_id text;
alter table public.blog_visits add column if not exists referrer text;
alter table public.blog_visits add column if not exists path_sequence int;
alter table public.blog_visits add column if not exists path_chain text;
alter table public.blog_visits add column if not exists visitor_kind text;
alter table public.blog_visits add column if not exists risk_score int;
alter table public.blog_visits add column if not exists net_isp text;
alter table public.blog_visits add column if not exists dwell_ms int;

create index if not exists blog_visits_session_idx on public.blog_visits (session_id, created_at);
create index if not exists blog_visits_kind_idx on public.blog_visits (visitor_kind);
create index if not exists blog_visits_risk_idx on public.blog_visits (risk_score desc nulls last);

-- 热门路径（供 ECharts / 后台）
drop view if exists public.blog_visit_hot_paths;
create view public.blog_visit_hot_paths as
select
  b.page_path,
  count(*)::bigint as views,
  count(distinct b.ip)::bigint as visitors
from public.blog_visits b
where b.page_path is not null
  and btrim(b.page_path) <> ''
group by b.page_path;

-- 按 IP 聚合风险（告警列表）
drop view if exists public.blog_visit_risk_summary;
create view public.blog_visit_risk_summary as
select
  b.ip,
  max(coalesce(b.risk_score, 0))::int as max_risk,
  count(*)::bigint as total_events,
  count(*) filter (where b.visitor_kind = 'crawler')::bigint as crawler_events,
  count(*) filter (where b.visitor_kind = 'scan')::bigint as scan_events,
  count(*) filter (where b.visitor_kind = 'suspect')::bigint as suspect_events,
  max(b.created_at) as last_seen
from public.blog_visits b
where coalesce(b.ip, '') not like 'local-%'
group by b.ip;

-- 同会话内推断停留（毫秒）：下一页打开时间与当前行时间差（最后一条为 null）
drop view if exists public.blog_visit_session_flow;
create view public.blog_visit_session_flow as
select
  b.id,
  b.session_id,
  b.page_path,
  b.created_at,
  lead(b.created_at) over (partition by b.session_id order by b.created_at) as next_at,
  extract(epoch from (
    lead(b.created_at) over (partition by b.session_id order by b.created_at) - b.created_at
  ))::double precision * 1000.0 as inferred_dwell_ms
from public.blog_visits b
where b.session_id is not null
  and btrim(b.session_id) <> '';

grant select on public.blog_visit_hot_paths to anon, authenticated, service_role;
grant select on public.blog_visit_risk_summary to anon, authenticated, service_role;
grant select on public.blog_visit_session_flow to anon, authenticated, service_role;

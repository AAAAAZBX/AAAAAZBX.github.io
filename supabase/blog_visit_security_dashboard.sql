-- Security/quant analytics layer for blog_visits.
-- Run in Supabase SQL Editor after blog_visits.sql and blog_visit_intel.sql.
--
-- Privacy model:
-- 1. Public or browser-facing pages should read aggregate views only.
-- 2. Raw IP/UA/path-chain details should be read through authenticated/admin
--    tooling or a service-role Edge Function, not by anon SELECT.

alter table public.blog_visits add column if not exists country_code text;
alter table public.blog_visits add column if not exists session_id text;
alter table public.blog_visits add column if not exists referrer text;
alter table public.blog_visits add column if not exists path_sequence int;
alter table public.blog_visits add column if not exists path_chain text;
alter table public.blog_visits add column if not exists visitor_kind text;
alter table public.blog_visits add column if not exists risk_score int;
alter table public.blog_visits add column if not exists net_isp text;
alter table public.blog_visits add column if not exists dwell_ms int;

create index if not exists blog_visits_security_created_idx
  on public.blog_visits (created_at desc);
create index if not exists blog_visits_page_path_idx
  on public.blog_visits (page_path);
create index if not exists blog_visits_country_code_idx
  on public.blog_visits (country_code);
create index if not exists blog_visits_net_isp_idx
  on public.blog_visits (net_isp);

create table if not exists public.blog_visit_alert_events (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  ip text not null,
  alert_type text not null,
  severity text not null default 'medium',
  page_path text,
  country text,
  city text,
  region text,
  visitor_kind text,
  risk_score int,
  message text,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists blog_visit_alert_events_created_idx
  on public.blog_visit_alert_events (created_at desc);
create index if not exists blog_visit_alert_events_ip_idx
  on public.blog_visit_alert_events (ip, created_at desc);
create index if not exists blog_visit_alert_events_severity_idx
  on public.blog_visit_alert_events (severity, created_at desc);

alter table public.blog_visit_alert_events enable row level security;

drop view if exists public.blog_visit_public_summary;
create view public.blog_visit_public_summary as
select
  date_trunc('hour', b.created_at) as bucket_hour,
  count(*)::bigint as events,
  count(distinct b.ip)::bigint as unique_ips,
  coalesce(b.country, 'unknown') as country,
  coalesce(b.country_code, 'unknown') as country_code,
  coalesce(b.city, b.region, 'unknown') as city,
  coalesce(b.visitor_kind, 'unknown') as visitor_kind,
  percentile_cont(0.90) within group (order by coalesce(b.risk_score, 0))::int as p90_risk,
  avg(coalesce(b.risk_score, 0))::numeric(8, 2) as avg_risk
from public.blog_visits b
group by
  date_trunc('hour', b.created_at),
  coalesce(b.country, 'unknown'),
  coalesce(b.country_code, 'unknown'),
  coalesce(b.city, b.region, 'unknown'),
  coalesce(b.visitor_kind, 'unknown');

drop view if exists public.blog_visit_risk_timeseries;
create view public.blog_visit_risk_timeseries as
select
  date_trunc('hour', b.created_at) as bucket_hour,
  count(*)::bigint as events,
  count(distinct b.ip)::bigint as unique_ips,
  count(*) filter (where coalesce(b.risk_score, 0) >= 60)::bigint as high_risk_events,
  count(*) filter (where b.visitor_kind = 'scan')::bigint as scan_events,
  count(*) filter (where b.visitor_kind = 'crawler')::bigint as crawler_events,
  percentile_cont(0.90) within group (order by coalesce(b.risk_score, 0))::int as p90_risk,
  least(
    100,
    (
      percentile_cont(0.90) within group (order by coalesce(b.risk_score, 0)) * 0.62
      + least(22, count(*) filter (where coalesce(b.risk_score, 0) >= 60) * 2)
      + least(16, count(*) filter (where b.visitor_kind = 'scan') * 2)
      + least(10, count(distinct b.ip) / 5)
    )::int
  ) as risk_index
from public.blog_visits b
group by date_trunc('hour', b.created_at);

drop view if exists public.blog_visit_ip_profile_summary;
create view public.blog_visit_ip_profile_summary as
select
  b.ip,
  min(b.created_at) as first_seen,
  max(b.created_at) as last_seen,
  count(*)::bigint as total_events,
  count(distinct b.session_id)::bigint as sessions,
  count(distinct b.page_path)::bigint as unique_paths,
  max(coalesce(b.risk_score, 0))::int as max_risk,
  avg(coalesce(b.risk_score, 0))::numeric(8, 2) as avg_risk,
  count(*) filter (where b.visitor_kind = 'crawler')::bigint as crawler_events,
  count(*) filter (where b.visitor_kind = 'scan')::bigint as scan_events,
  count(*) filter (
    where b.page_path ~* '(\.env|\.git|wp-admin|wp-login|phpmyadmin|actuator|swagger|graphql)'
  )::bigint as sensitive_path_events,
  max(b.net_isp) filter (where b.net_isp is not null and btrim(b.net_isp) <> '') as net_isp,
  max(b.country) filter (where b.country is not null and btrim(b.country) <> '') as country,
  max(b.city) filter (where b.city is not null and btrim(b.city) <> '') as city,
  max(b.region) filter (where b.region is not null and btrim(b.region) <> '') as region
from public.blog_visits b
where coalesce(b.ip, '') <> ''
  and b.ip not like 'local-%'
group by b.ip;

drop view if exists public.blog_visit_attack_surface;
create view public.blog_visit_attack_surface as
select
  coalesce(b.page_path, 'unknown') as page_path,
  count(*)::bigint as events,
  count(distinct b.ip)::bigint as unique_ips,
  max(coalesce(b.risk_score, 0))::int as max_risk,
  count(*) filter (where b.visitor_kind in ('scan', 'crawler'))::bigint as automated_events,
  count(*) filter (
    where coalesce(b.page_path, '') ~* '(\.env|\.git|wp-admin|wp-login|phpmyadmin|actuator|swagger|graphql)'
  )::bigint as sensitive_hits
from public.blog_visits b
where b.page_path is not null
  and btrim(b.page_path) <> ''
group by coalesce(b.page_path, 'unknown');

drop view if exists public.blog_visit_alert_rollup;
create view public.blog_visit_alert_rollup as
select
  date_trunc('day', a.created_at) as day,
  a.alert_type,
  a.severity,
  count(*)::bigint as alerts,
  count(distinct a.ip)::bigint as unique_ips,
  max(a.created_at) as last_alert_at
from public.blog_visit_alert_events a
group by date_trunc('day', a.created_at), a.alert_type, a.severity;

grant select on public.blog_visit_public_summary to anon, authenticated, service_role;
grant select on public.blog_visit_risk_timeseries to anon, authenticated, service_role;
grant select on public.blog_visit_attack_surface to anon, authenticated, service_role;
grant select on public.blog_visit_alert_rollup to anon, authenticated, service_role;

-- Keep IP profiles private by default because they contain identifiable IP-level history.
grant select on public.blog_visit_ip_profile_summary to authenticated, service_role;
grant select, insert, update on public.blog_visit_alert_events to service_role;

-- Optional hardening for production:
-- Revoke raw event reads from anon after your public pages use aggregate views only.
-- revoke select on public.blog_visits from anon;
-- drop policy if exists "blog_visits_select_anon" on public.blog_visits;
--
-- Existing browser logging still needs anon INSERT:
-- create policy "blog_visits_insert_anon" on public.blog_visits
--   for insert to anon
--   with check (true);

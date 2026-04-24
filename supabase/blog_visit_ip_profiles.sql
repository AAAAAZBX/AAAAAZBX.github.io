-- Persistent editable profile names for the private analytics IP portrait table.
-- Run this once in Supabase SQL Editor.

create table if not exists public.blog_visit_ip_profiles (
  ip text primary key,
  profile_name text not null default '',
  updated_at timestamptz not null default now()
);

create index if not exists blog_visit_ip_profiles_updated_idx
  on public.blog_visit_ip_profiles (updated_at desc);

alter table public.blog_visit_ip_profiles enable row level security;

drop policy if exists "blog_visit_ip_profiles_select_anon" on public.blog_visit_ip_profiles;
create policy "blog_visit_ip_profiles_select_anon"
  on public.blog_visit_ip_profiles
  for select
  to anon
  using (true);

drop policy if exists "blog_visit_ip_profiles_insert_anon" on public.blog_visit_ip_profiles;
create policy "blog_visit_ip_profiles_insert_anon"
  on public.blog_visit_ip_profiles
  for insert
  to anon
  with check (true);

drop policy if exists "blog_visit_ip_profiles_update_anon" on public.blog_visit_ip_profiles;
create policy "blog_visit_ip_profiles_update_anon"
  on public.blog_visit_ip_profiles
  for update
  to anon
  using (true)
  with check (true);

drop policy if exists "blog_visit_ip_profiles_delete_anon" on public.blog_visit_ip_profiles;
create policy "blog_visit_ip_profiles_delete_anon"
  on public.blog_visit_ip_profiles
  for delete
  to anon
  using (true);

grant select, insert, update, delete on public.blog_visit_ip_profiles to anon;
grant select, insert, update, delete on public.blog_visit_ip_profiles to authenticated, service_role;

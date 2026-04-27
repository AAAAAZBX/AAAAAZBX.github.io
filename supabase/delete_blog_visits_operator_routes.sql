-- 删除「控制台」路由下的访客行（与 isOperatorConsolePath 首段一致）。
-- blog_visit_path_stats 为视图，会随 blog_visits 自动变化。
-- 执行前可先预览：将 DELETE 改为 SELECT count(*) 或 SELECT id, page_path, created_at ...

begin;

delete from public.blog_visits
where nullif(trim(page_path), '') is not null
  and split_part(trim(both '/' from page_path), '/', 1) in (
    'statistics',
    'admin',
    'analytics',
    'quant',
    'ai',
    'fintech'
  );

commit;

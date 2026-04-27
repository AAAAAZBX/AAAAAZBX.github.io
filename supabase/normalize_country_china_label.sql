-- 将国家名「中国」统一为 English 标签 China（访客表 + IP 地理缓存）。
-- 执行前可预览：
--   select count(*) from public.blog_visits where btrim(coalesce(country, '')) = '中国';
--   select count(*) from public.blog_visit_ip_cache where btrim(coalesce(country, '')) = '中国';

begin;

update public.blog_visits
set country = 'China'
where btrim(coalesce(country, '')) = '中国';

update public.blog_visit_ip_cache
set country = 'China'
where btrim(coalesce(country, '')) = '中国';

commit;

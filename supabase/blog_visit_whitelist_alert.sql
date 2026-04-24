-- 非白名单访问 → 邮件告警（配合 Edge Function visit-ip-whitelist-alert）
-- 浏览器无法安全发信；由 Supabase Database Webhook 在 blog_visits 插入后 POST 到函数 URL。
--
-- 【1】本文件：在 SQL Editor 执行一次
-- 【2】Dashboard → Edge Functions → 创建 visit-ip-whitelist-alert（见仓库 supabase/functions/visit-ip-whitelist-alert）
-- 【3】Functions → Secrets 设置（示例名称）：
--     WHITELIST_ALERT_WEBHOOK_SECRET   随机长串，与 Webhook 请求头 Authorization: Bearer <同值> 一致
--     VISIT_IP_WHITELIST                 白名单 IP，逗号/空格/分号分隔；留空则不发任何告警
--     RESEND_API_KEY                     https://resend.com API Key
--     ALERT_EMAIL_TO                     你的收件邮箱
--     ALERT_FROM_EMAIL                   发件人（需在 Resend 验证的域名；测试可用 onboarding@resend.dev）
--     SUPABASE_SERVICE_ROLE_KEY          项目 Settings → API → service_role（仅服务端，勿进前端）
--     VISIT_ALERT_COOLDOWN_SEC           同一 IP 告警最小间隔秒，默认 3600
--     （SUPABASE_URL 通常由平台注入；若缺失请在 Secrets 补上项目 URL）
--
-- 【4】Database → Webhooks → Create hook
--     Table: public.blog_visits  Events: INSERT
--     HTTP Request URL: https://<project-ref>.supabase.co/functions/v1/visit-ip-whitelist-alert
--     HTTP Headers: Authorization = Bearer <WHITELIST_ALERT_WEBHOOK_SECRET>
--
-- 说明：仅当 VISIT_IP_WHITELIST 非空时启用；在白名单内不告警；local-* 占位 IP 不告警。

create table if not exists public.visit_ip_alert_sent (
  ip text primary key,
  last_sent_at timestamptz not null default now()
);

create index if not exists visit_ip_alert_sent_last_idx on public.visit_ip_alert_sent (last_sent_at desc);

alter table public.visit_ip_alert_sent enable row level security;

-- 不向 anon/authenticated 开放；Edge Function 使用 service_role 可绕过 RLS 读写。

comment on table public.visit_ip_alert_sent is '非白名单访问邮件告警冷却（按 IP），仅服务端使用';

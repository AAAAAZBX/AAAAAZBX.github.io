/**
 * Sends an email when a newly inserted blog_visits row comes from an IP that
 * is not in VISIT_IP_WHITELIST. Intended to be called by a Supabase Database
 * Webhook on public.blog_visits INSERT.
 */
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.49.1";

const corsHeaders: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

type VisitRecord = {
  ip?: string | null;
  page_path?: string | null;
  country?: string | null;
  city?: string | null;
  region?: string | null;
  created_at?: string | null;
  user_agent?: string | null;
  visitor_kind?: string | null;
  risk_score?: number | null;
  net_isp?: string | null;
};

type WebhookBody = {
  type?: string;
  table?: string;
  record?: VisitRecord | null;
};

function parseWhitelist(raw: string | undefined): string[] {
  if (!raw?.trim()) return [];
  return raw
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function ipAllowed(ip: string, list: string[]): boolean {
  const x = ip.trim().toLowerCase();
  for (const w of list) {
    const rule = w.trim().toLowerCase();
    if (!rule) continue;
    if (rule === x) return true;
    if (rule.endsWith("*") && rule.length > 1) {
      const prefix = rule.slice(0, -1);
      if (prefix && x.startsWith(prefix)) return true;
    }
  }
  return false;
}

function escapeHtml(s: string): string {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  const expectSecret = Deno.env.get("WHITELIST_ALERT_WEBHOOK_SECRET")?.trim();
  const auth = (req.headers.get("authorization") || "").trim();
  if (!expectSecret || auth !== `Bearer ${expectSecret}`) {
    return new Response(JSON.stringify({ error: "unauthorized" }), {
      status: 401,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  let body: WebhookBody;
  try {
    body = (await req.json()) as WebhookBody;
  } catch {
    return new Response(JSON.stringify({ error: "invalid_json" }), {
      status: 400,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  const record = body.record;
  const ip = String(record?.ip ?? "").trim();
  if (!ip || ip.toLowerCase().startsWith("local-")) {
    return new Response(JSON.stringify({ skipped: "local_or_empty_ip" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  const whitelist = parseWhitelist(Deno.env.get("VISIT_IP_WHITELIST"));
  if (whitelist.length === 0) {
    return new Response(JSON.stringify({ skipped: "whitelist_empty_alerts_off" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
  if (ipAllowed(ip, whitelist)) {
    return new Response(JSON.stringify({ skipped: "whitelisted" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL")?.trim();
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")?.trim();
  if (!supabaseUrl || !serviceKey) {
    return new Response(JSON.stringify({ error: "missing_supabase_service_env" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  const supabase = createClient(supabaseUrl, serviceKey);
  const cooldownSec = Math.max(
    60,
    parseInt(Deno.env.get("VISIT_ALERT_COOLDOWN_SEC") || "3600", 10) || 3600,
  );
  const now = Date.now();

  const { data: prev } = await supabase
    .from("visit_ip_alert_sent")
    .select("last_sent_at")
    .eq("ip", ip)
    .maybeSingle();

  if (prev?.last_sent_at) {
    const lastSentMs = new Date(prev.last_sent_at as string).getTime();
    if (Number.isFinite(lastSentMs) && now - lastSentMs < cooldownSec * 1000) {
      return new Response(JSON.stringify({ skipped: "cooldown" }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  }

  const resendKey = Deno.env.get("RESEND_API_KEY")?.trim();
  const to = Deno.env.get("ALERT_EMAIL_TO")?.trim();
  const from = Deno.env.get("ALERT_FROM_EMAIL")?.trim() || "onboarding@resend.dev";
  if (!resendKey || !to) {
    return new Response(JSON.stringify({ error: "missing_resend_or_alert_email_to" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  const loc = [record?.city, record?.region, record?.country].filter(Boolean).join(", ");
  const subject = `[Visitor Alert] Non-whitelisted IP: ${ip}`;
  const html = `<p>A non-whitelisted visitor IP accessed your site.</p>
<ul>
  <li><strong>IP:</strong> <code>${escapeHtml(ip)}</code></li>
  <li><strong>Path:</strong> <code>${escapeHtml(record?.page_path || "unknown")}</code></li>
  <li><strong>Location:</strong> ${escapeHtml(loc || "unknown")}</li>
  <li><strong>Kind:</strong> ${escapeHtml(record?.visitor_kind || "unknown")}</li>
  <li><strong>Risk score:</strong> ${escapeHtml(String(record?.risk_score ?? "unknown"))}</li>
  <li><strong>Time:</strong> ${escapeHtml(record?.created_at || new Date().toISOString())}</li>
</ul>
<p><strong>User-Agent:</strong><br/><code>${escapeHtml(String(record?.user_agent || "").slice(0, 500))}</code></p>`;

  const sendRes = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${resendKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from,
      to: [to],
      subject,
      html,
    }),
  });

  if (!sendRes.ok) {
    const detail = await sendRes.text();
    return new Response(JSON.stringify({ error: "resend_failed", detail: detail.slice(0, 500) }), {
      status: 502,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  await supabase.from("visit_ip_alert_sent").upsert({
    ip,
    last_sent_at: new Date().toISOString(),
  });

  try {
    await supabase.from("blog_visit_alert_events").insert({
      ip,
      alert_type: "non_whitelist_ip",
      severity: Number(record?.risk_score ?? 0) >= 70 ? "high" : "medium",
      page_path: record?.page_path ?? null,
      country: record?.country ?? null,
      city: record?.city ?? null,
      region: record?.region ?? null,
      visitor_kind: record?.visitor_kind ?? null,
      risk_score: record?.risk_score ?? null,
      message: "Non-whitelisted visitor IP triggered an email notification.",
      metadata: {
        net_isp: record?.net_isp ?? null,
        user_agent: String(record?.user_agent || "").slice(0, 500),
        source: "visit-ip-whitelist-alert",
      },
    });
  } catch {
    // Keep email alerting working even before blog_visit_security_dashboard.sql is applied.
  }

  return new Response(JSON.stringify({ ok: true }), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
});

import type { APIRoute } from "astro";

export const POST: APIRoute = async ({ request }) => {
  try {
    const payload = await request.json();
    const body = payload?.body;
    if (!body || typeof body !== "string" || body.trim().length < 100) {
      return new Response(JSON.stringify({ error: "正文内容过短" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const token = (import.meta.env.ANTHROPIC_AUTH_TOKEN || "").trim();
    const baseUrl = (import.meta.env.ANTHROPIC_BASE_URL || "https://api.deepseek.com/anthropic").trim();
    const model = (import.meta.env.ANTHROPIC_MODEL || "deepseek-v4-flash").trim();

    if (!token) {
      return new Response(JSON.stringify({ error: "未配置 ANTHROPIC_AUTH_TOKEN" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }

    const maxBodyChars = 10000;
    const truncatedBody = body.slice(0, maxBodyChars);
    const userText = `请用简要概括这篇文章的主要内容与要点。要求：3-6句话，不超过320字，使用简体中文，不要使用markdown格式，不要使用列表或标题。\n\n${truncatedBody}`;

    const res = await fetch(`${baseUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": token,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model,
        max_tokens: 2400,
        messages: [{ role: "user", content: userText }],
      }),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      return new Response(
        JSON.stringify({ error: `DeepSeek API ${res.status}: ${errText.slice(0, 300)}` }),
        { status: 502, headers: { "Content-Type": "application/json" } }
      );
    }

    const data = await res.json();
    const textBlock = data?.content?.find((c: any) => c.type === "text");
    const summary = textBlock?.text?.trim() || "";

    if (!summary) {
      return new Response(JSON.stringify({ error: "模型返回空内容" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({ summary, model: data.model || model }),
      { headers: { "Content-Type": "application/json" } }
    );
  } catch (e: any) {
    return new Response(JSON.stringify({ error: e.message || "未知错误" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};

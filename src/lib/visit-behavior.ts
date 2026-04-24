/**
 * 浏览器端会话与简单风险规则（与 visit-logger 配合）。
 * 规则仅为启发式：MPA 整页跳转才算一次 path，锚点滚动不计入。
 */

const K_SESSION = "__bx_vsid__";
const K_PATHS = "__bx_vpaths__";
const K_SEQ = "__bx_vseq__";

export type VisitBehaviorPayload = {
  session_id: string;
  referrer: string | null;
  path_sequence: number;
  path_chain: string;
  visitor_kind: string;
  risk_score: number;
};

export function classifyVisitor(
  ua: string,
  paths: string[],
  sequence: number,
): { visitor_kind: string; risk_score: number } {
  const u = ua.toLowerCase();
  if (!ua || ua.length < 10) {
    return { visitor_kind: "suspect", risk_score: 22 };
  }
  if (
    /\bbot\b|crawl|spider|slurp|bingpreview|facebookexternal|embedly|scrapy|python-requests|aiohttp|curl\/|wget|^java\/|\bhttpx\b|go-http|libwww|okhttp|axios\/|postman|headless|phantom|puppeteer|playwright|selenium|preview/i.test(
      u,
    )
  ) {
    return { visitor_kind: "crawler", risk_score: 92 };
  }

  const unique = new Set(paths).size;
  let risk = 0;
  if (unique >= 22) risk += 48;
  else if (unique >= 14) risk += 34;
  else if (unique >= 9) risk += 20;
  if (sequence >= 30) risk += 28;
  else if (sequence >= 18) risk += 16;

  const pathStr = paths.join("\n");
  if (
    /\.(env|git|sql|bak|zip)(\?|$)|\/wp-admin|\/wp-login|phpmyadmin|\/\.git|actuator\/|\/swagger|\/v1\/api-docs|\/graphql|\/\.well-known\/security/i.test(
      pathStr,
    )
  ) {
    risk += 55;
  }

  if (risk >= 70) return { visitor_kind: "scan", risk_score: Math.min(100, risk) };
  if (risk >= 42) return { visitor_kind: "suspect", risk_score: risk };
  return { visitor_kind: "normal", risk_score: Math.min(25, risk) };
}

export function getVisitBehaviorPayload(opts: {
  pagePath: string | null;
  userAgent: string;
}): VisitBehaviorPayload {
  if (typeof sessionStorage === "undefined") {
    const { visitor_kind, risk_score } = classifyVisitor(opts.userAgent, [String(opts.pagePath || "/")], 1);
    return {
      session_id: "ssr-unknown",
      referrer: null,
      path_sequence: 1,
      path_chain: String(opts.pagePath || "/"),
      visitor_kind,
      risk_score,
    };
  }

  let sid = sessionStorage.getItem(K_SESSION);
  if (!sid) {
    sid =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `s-${Math.random().toString(36).slice(2, 12)}`;
    sessionStorage.setItem(K_SESSION, sid);
  }

  const path = String(opts.pagePath || "/").trim() || "/";
  let paths: string[] = [];
  try {
    const raw = sessionStorage.getItem(K_PATHS);
    paths = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(paths)) paths = [];
  } catch {
    paths = [];
  }
  paths.push(path);
  const trimmed = paths.slice(-36);
  sessionStorage.setItem(K_PATHS, JSON.stringify(trimmed));

  const seq = Number(sessionStorage.getItem(K_SEQ) || "0") + 1;
  sessionStorage.setItem(K_SEQ, String(seq));

  const chain = trimmed.slice(-12).join(" → ");
  let ref = "";
  if (typeof document !== "undefined") {
    ref = String(document.referrer || "").trim().slice(0, 512);
  }
  const { visitor_kind, risk_score } = classifyVisitor(opts.userAgent, trimmed, seq);

  return {
    session_id: sid,
    referrer: ref || null,
    path_sequence: seq,
    path_chain: chain.slice(0, 2000),
    visitor_kind,
    risk_score,
  };
}

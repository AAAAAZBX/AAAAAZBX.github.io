import type { MergedPost } from "./content-posts";

export type GraphEdgeKind = "markdown" | "wikilink" | "tag";

export type KnowledgeGraphNode = {
  id: string;
  title: string;
  collection: string;
  href: string;
  tags: string[];
  hidden: boolean;
};

export type KnowledgeGraphEdge = {
  source: string;
  target: string;
  kinds: GraphEdgeKind[];
};

export type KnowledgeGraphPayload = {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
};

function stripFencedCode(body: string | undefined): string {
  return String(body ?? "").replace(/```[\s\S]*?```/g, " ");
}

function normalizeHiddenSet(hiddenIds: Iterable<string>): Set<string> {
  return new Set(
    [...hiddenIds].map((x) => String(x).trim().toLowerCase()).filter(Boolean)
  );
}

function dedupeTags(tags: string[] | undefined): string[] {
  const input = Array.isArray(tags) ? tags : [];
  const seen = new Map<string, string>();
  for (const raw of input) {
    const t = String(raw ?? "").trim();
    if (!t) continue;
    const key = t.toLowerCase();
    if (!seen.has(key)) seen.set(key, t);
  }
  return [...seen.values()];
}

function mergeEdgeKinds(
  map: Map<string, { source: string; target: string; kinds: Set<GraphEdgeKind> }>,
  sourceCanon: string,
  targetCanon: string,
  kind: GraphEdgeKind
) {
  const la = sourceCanon.trim().toLowerCase();
  const lb = targetCanon.trim().toLowerCase();
  if (!la || !lb || la === lb) return;
  const loKey = la < lb ? la : lb;
  const hiKey = la < lb ? lb : la;
  const key = `${loKey}\0${hiKey}`;
  const src = la < lb ? sourceCanon : targetCanon;
  const tgt = la < lb ? targetCanon : sourceCanon;
  let row = map.get(key);
  if (!row) {
    row = { source: src, target: tgt, kinds: new Set() };
    map.set(key, row);
  }
  row.kinds.add(kind);
}

export function buildKnowledgeGraph(
  posts: MergedPost[],
  hiddenIds: Iterable<string>,
  baseUrl: string
): KnowledgeGraphPayload {
  const hidden = normalizeHiddenSet(hiddenIds);

  const nodes: KnowledgeGraphNode[] = posts.map((p) => ({
    id: p.sortId,
    title: p.title,
    collection: p.collection,
    href: p.href,
    tags: dedupeTags(p.tags),
    hidden: hidden.has(p.sortId.trim().toLowerCase()),
  }));

  const canonLower = new Map<string, string>();
  for (const n of nodes) {
    canonLower.set(n.id.trim().toLowerCase(), n.id);
  }

  const pathToId = new Map<string, string>();
  const slugTailToIds = new Map<string, string[]>();
  const titleToId = new Map<string, string>();

  for (const p of posts) {
    const pathKey = `${p.collection}/${p.slug}`.replace(/\\/g, "/").trim().toLowerCase();
    pathToId.set(pathKey, p.sortId);

    const tail = p.slug.split("/").filter(Boolean).slice(-1)[0]?.trim().toLowerCase() ?? "";
    if (tail) {
      if (!slugTailToIds.has(tail)) slugTailToIds.set(tail, []);
      slugTailToIds.get(tail)!.push(p.sortId);
    }
    titleToId.set(p.title.trim().toLowerCase(), p.sortId);
  }

  const base = String(baseUrl ?? "/").replace(/\/$/, "");

  function resolveCategoriesPath(collection: string, slugRest: string): string | null {
    const rest = slugRest.replace(/\\/g, "/").trim().replace(/\.md$/i, "");
    const key = `${collection}/${rest}`.toLowerCase();
    if (pathToId.has(key)) return pathToId.get(key)!;

    const tail = rest.split("/").filter(Boolean).slice(-1)[0]?.toLowerCase() ?? "";
    const candidates = slugTailToIds.get(tail);
    if (candidates?.length === 1) return candidates[0];
    return null;
  }

  function resolveLinkHref(href: string): string | null {
    let h = href.trim();
    if (!h || h.startsWith("#") || h.toLowerCase().startsWith("mailto:")) return null;

    const hash = h.indexOf("#");
    if (hash >= 0) h = h.slice(0, hash);

    try {
      if (/^https?:\/\//i.test(h)) {
        const u = new URL(h);
        h = u.pathname || "";
      }
    } catch {
      // keep relative
    }

    if (base && base !== "/" && h.startsWith(base)) {
      h = h.slice(base.length);
    }
    h = h.replace(/^\/+/, "");

    try {
      h = decodeURIComponent(h);
    } catch {
      // ignore
    }

    const m = h.match(/^categories\/([^/]+)\/(.+)$/i);
    if (!m) return null;
    return resolveCategoriesPath(m[1], m[2]);
  }

  function resolveWikilink(label: string): string | null {
    const t = label.trim().replace(/\.md$/i, "");
    if (!t) return null;

    if (t.includes("/")) {
      const parts = t.split("/").filter(Boolean);
      if (parts.length >= 2) {
        const coll = parts[0];
        const rest = parts.slice(1).join("/");
        const resolved = resolveCategoriesPath(coll, rest);
        if (resolved) return resolved;
      }
    }

    const lower = t.toLowerCase();
    if (titleToId.has(lower)) return titleToId.get(lower)!;

    const tail = t.split("/").filter(Boolean).slice(-1)[0]?.toLowerCase() ?? "";
    const cands = slugTailToIds.get(tail);
    if (cands?.length === 1) return cands[0];

    let match: string | null = null;
    for (const p of posts) {
      if (p.title.toLowerCase().includes(lower)) {
        if (match && match !== p.sortId) return null;
        match = p.sortId;
      }
    }
    return match;
  }

  const edgeMap = new Map<
    string,
    { source: string; target: string; kinds: Set<GraphEdgeKind> }
  >();

  function addEdge(fromId: string, toId: string, kind: GraphEdgeKind) {
    const ca = canonLower.get(fromId.trim().toLowerCase());
    const cb = canonLower.get(toId.trim().toLowerCase());
    if (!ca || !cb || ca === cb) return;
    mergeEdgeKinds(edgeMap, ca, cb, kind);
  }

  for (const p of posts) {
    const text = stripFencedCode(p.body);

    const mdLink = /\]\(\s*([^)\s]+)\s*\)/g;
    let mm: RegExpExecArray | null;
    while ((mm = mdLink.exec(text)) !== null) {
      const target = resolveLinkHref(mm[1]);
      if (target) addEdge(p.sortId, target, "markdown");
    }

    const htmlLink = /<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']/gi;
    while ((mm = htmlLink.exec(text)) !== null) {
      const target = resolveLinkHref(mm[1]);
      if (target) addEdge(p.sortId, target, "markdown");
    }

    const wiki = /\[\[\s*([^\]|#]+)\s*(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g;
    while ((mm = wiki.exec(text)) !== null) {
      const target = resolveWikilink(mm[1]);
      if (target) addEdge(p.sortId, target, "wikilink");
    }
  }

  const tagCount = new Map<string, number>();
  for (const p of posts) {
    for (const tag of p.tags ?? []) {
      tagCount.set(tag, (tagCount.get(tag) ?? 0) + 1);
    }
  }
  const maxTagOcc = Math.max(15, Math.floor(posts.length * 0.25));
  const allowedTags = new Set<string>();
  for (const [tag, c] of tagCount) {
    if (c >= 2 && c <= maxTagOcc) allowedTags.add(tag);
  }

  for (let i = 0; i < posts.length; i++) {
    for (let j = i + 1; j < posts.length; j++) {
      const a = posts[i];
      const b = posts[j];
      const shared = (a.tags ?? []).filter((t) => (b.tags ?? []).includes(t) && allowedTags.has(t));
      if (shared.length) addEdge(a.sortId, b.sortId, "tag");
    }
  }

  const edges: KnowledgeGraphEdge[] = [...edgeMap.values()].map((row) => ({
    source: row.source,
    target: row.target,
    kinds: [...row.kinds],
  }));

  return { nodes, edges };
}

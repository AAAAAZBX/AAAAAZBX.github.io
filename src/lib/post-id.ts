export type PostIdentityInput = {
  id?: string | null;
  slug?: string | null;
};

function normalizePostId(value: string | null | undefined): string {
  return String(value ?? "").trim().toLowerCase();
}

export function resolvePostId(input: PostIdentityInput): string {
  const explicitId = normalizePostId(input.id);
  if (explicitId) return explicitId;

  const slug = String(input.slug ?? "").trim();
  if (slug) {
    const slugTail = slug.split("/").filter(Boolean).slice(-1)[0] ?? "";
    const normalizedSlugTail = normalizePostId(slugTail);
    if (normalizedSlugTail) return normalizedSlugTail;
    return normalizePostId(slug);
  }

  return "";
}

export function normalizeHiddenPostIds(values: Iterable<unknown>): string[] {
  return [...new Set(Array.from(values, (value) => normalizePostId(String(value ?? ""))).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b, "zh-CN")
  );
}

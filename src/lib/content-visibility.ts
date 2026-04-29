import fs from "node:fs";
import path from "node:path";
import { createClient } from "@supabase/supabase-js";
import { normalizeHiddenPostIds } from "./post-id";

const visibilityLocalPath = path.join(process.cwd(), "src", "content-visibility.json");

function getSupabase() {
  const url = String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
  const key = String(import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? "").trim();
  if (!url || !key || !url.startsWith("http")) return null;
  return createClient(url, key);
}

function readLocalHidden(): string[] {
  try {
    const raw = fs.readFileSync(visibilityLocalPath, "utf-8");
    const parsed = JSON.parse(raw) as { hidden?: string[] };
    return normalizeHiddenPostIds(Array.isArray(parsed.hidden) ? parsed.hidden : []);
  } catch {
    return [];
  }
}

export async function readHiddenPostIds(): Promise<string[]> {
  const supabase = getSupabase();
  if (supabase) {
    try {
      const { data, error } = await supabase
        .from("content_visibility")
        .select("hidden_posts")
        .eq("id", 1)
        .single();
      if (!error && data && Array.isArray(data.hidden_posts)) {
        return normalizeHiddenPostIds(data.hidden_posts);
      }
    } catch {
      // Supabase unreachable, use local file
    }
  }
  return readLocalHidden();
}

export async function fetchHiddenPostKeys(): Promise<Set<string>> {
  return new Set(await readHiddenPostIds());
}

export function countHiddenPosts(postIds: Iterable<string>, hiddenKeys: Set<string>): number {
  let total = 0;
  for (const postId of postIds) {
    if (hiddenKeys.has(String(postId).trim().toLowerCase())) total += 1;
  }
  return total;
}

export function isPostVisible(postId: string | undefined, hiddenKeys: Set<string>): boolean {
  if (!postId) return true;
  return !hiddenKeys.has(postId.trim().toLowerCase());
}

export function isPostHidden(postId: string | undefined, hiddenKeys: Set<string>): boolean {
  return !isPostVisible(postId, hiddenKeys);
}

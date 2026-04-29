import fs from "node:fs";
import path from "node:path";
import { createClient } from "@supabase/supabase-js";

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
    return (Array.isArray(parsed.hidden) ? parsed.hidden : []).map(k => String(k).trim().toLowerCase());
  } catch {
    return [];
  }
}

export async function fetchHiddenPostKeys(): Promise<Set<string>> {
  const local = new Set(readLocalHidden());
  const supabase = getSupabase();
  if (supabase) {
    try {
      const { data, error } = await supabase
        .from("content_visibility")
        .select("hidden_posts")
        .eq("id", 1)
        .single();
      if (!error && data && Array.isArray(data.hidden_posts)) {
        for (const k of data.hidden_posts) {
          local.add(String(k).trim().toLowerCase());
        }
      }
    } catch {
      // Supabase unreachable, keep local only
    }
  }
  return local;
}

export function isPostVisible(postId: string | undefined, hiddenKeys: Set<string>): boolean {
  if (!postId) return true;
  return !hiddenKeys.has(postId.trim().toLowerCase());
}

export function isPostHidden(postId: string | undefined, hiddenKeys: Set<string>): boolean {
  return !isPostVisible(postId, hiddenKeys);
}

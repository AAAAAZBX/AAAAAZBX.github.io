import fs from "node:fs";
import path from "node:path";
import { createClient } from "@supabase/supabase-js";
import { normalizeHiddenPostIds } from "./post-id";

const visibilityLocalPath = path.join(process.cwd(), "content-visibility.local.json");
const legacyVisibilityLocalPath = path.join(process.cwd(), "src", "content-visibility.json");

type LocalVisibilityState = {
  hidden: string[];
  synced: boolean | null;
};

function getSupabaseUrl(): string {
  return String(import.meta.env.PUBLIC_SUPABASE_URL ?? "").trim();
}

function getSupabaseReadKey(): string {
  return String(
    import.meta.env.SUPABASE_SERVICE_ROLE_KEY ?? import.meta.env.PUBLIC_SUPABASE_ANON_KEY ?? ""
  ).trim();
}

function getSupabaseWriteKey(): string {
  return String(import.meta.env.SUPABASE_SERVICE_ROLE_KEY ?? "").trim();
}

function getSupabaseClient(key: string) {
  const url = getSupabaseUrl();
  if (!url || !key || !url.startsWith("http")) return null;
  return createClient(url, key);
}

function normalizeLocalState(input: { hidden?: unknown; synced?: unknown } | null | undefined): LocalVisibilityState {
  return {
    hidden: normalizeHiddenPostIds(Array.isArray(input?.hidden) ? input.hidden : []),
    synced: typeof input?.synced === "boolean" ? input.synced : null,
  };
}

function readLocalStateFromFile(filePath: string): LocalVisibilityState | null {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    return normalizeLocalState(JSON.parse(raw) as { hidden?: unknown; synced?: unknown });
  } catch {
    return null;
  }
}

function readLocalVisibilityState(): LocalVisibilityState {
  const primary = readLocalStateFromFile(visibilityLocalPath);
  if (primary) return primary;

  const legacy = readLocalStateFromFile(legacyVisibilityLocalPath);
  if (legacy) return legacy;

  return { hidden: [], synced: null };
}

function writeLocalVisibilityState(hiddenIds: Iterable<unknown>, synced: boolean | null): LocalVisibilityState {
  const state: LocalVisibilityState = {
    hidden: normalizeHiddenPostIds(hiddenIds),
    synced,
  };
  fs.writeFileSync(visibilityLocalPath, `${JSON.stringify(state, null, 2)}\n`, "utf-8");
  return state;
}

export function writeLocalHidden(hiddenIds: Iterable<unknown>): string[] {
  return writeLocalVisibilityState(hiddenIds, false).hidden;
}

export async function persistHiddenPostIds(hiddenIds: Iterable<unknown>): Promise<{ hidden: string[]; synced: boolean }> {
  const hidden = normalizeHiddenPostIds(hiddenIds);
  const supabase = getSupabaseClient(getSupabaseWriteKey());
  if (!supabase) {
    writeLocalVisibilityState(hidden, false);
    return { hidden, synced: false };
  }

  try {
    const { error } = await supabase
      .from("content_visibility")
      .upsert({ id: 1, hidden_posts: hidden, updated_at: new Date().toISOString() });

    const synced = !error;
    writeLocalVisibilityState(hidden, synced);
    return { hidden, synced };
  } catch {
    writeLocalVisibilityState(hidden, false);
    return { hidden, synced: false };
  }
}

export async function readHiddenPostIds(): Promise<string[]> {
  const localState = readLocalVisibilityState();
  if (localState.synced === false) {
    return localState.hidden;
  }

  const supabase = getSupabaseClient(getSupabaseReadKey());
  if (supabase) {
    try {
      const { data, error } = await supabase
        .from("content_visibility")
        .select("hidden_posts")
        .eq("id", 1)
        .single();
      if (!error && data && Array.isArray(data.hidden_posts)) {
        return writeLocalVisibilityState(data.hidden_posts, true).hidden;
      }
    } catch {
      // Supabase unreachable, use local file
    }
  }
  return localState.hidden;
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

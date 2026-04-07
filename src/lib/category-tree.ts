import fs from "node:fs";
import path from "node:path";
import type { MergedPost } from "./content-posts";
import { getCollectionKeys } from "./content-posts";

export type CatPathNode = {
  name: string;
  /** 稳定路径键，如 `ai` 或 `algorithms/比赛/xxx`，用于 collapse 的 DOM id */
  pathKey: string;
  posts: MergedPost[];
  children: CatPathNode[];
};

type MutableNode = {
  name: string;
  pathKey: string;
  posts: MergedPost[];
  childMap: Map<string, MutableNode>;
};

function postIdLexKey(post: MergedPost): string {
  return post.sortId;
}

function byIdLexDesc(a: MergedPost, b: MergedPost): number {
  return postIdLexKey(b).localeCompare(postIdLexKey(a), "zh-CN");
}

function byDateDesc(a: MergedPost, b: MergedPost): number {
  const ta = a.date ? new Date(a.date).getTime() : 0;
  const tb = b.date ? new Date(b.date).getTime() : 0;
  return tb - ta;
}

/**
 * 统一路径段：大小写不敏感，并把空白按 Astro slug 习惯转成 `-`。
 * 例如：`Neural Networks Zero to Hero - Andrej Karpathy`
 * -> `neural-networks-zero-to-hero---andrej-karpathy`
 */
function normSeg(seg: string): string {
  return seg.trim().replace(/\s/g, "-").replace(/[A-Z]/g, (c) => c.toLowerCase());
}

/**
 * 扫描 src/content/&lt;collection&gt;/ 下的真实文件夹名。
 * 键：`collection/a/b`（每段经 normSeg）；值：该层文件夹在磁盘上的原始名称（大小写与之一致）。
 */
export function buildContentFolderDisplayMap(): Map<string, string> {
  const map = new Map<string, string>();
  const contentRoot = path.join(process.cwd(), "src", "content");
  const collectionKeys = getCollectionKeys();

  for (const col of collectionKeys) {
    const root = path.join(contentRoot, col);
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) continue;

    const walk = (dir: string, rel: string[]) => {
      let dirents: fs.Dirent[];
      try {
        dirents = fs.readdirSync(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const e of dirents) {
        if (!e.isDirectory() || e.name.startsWith(".")) continue;
        const relNext = [...rel, e.name];
        const mapKey = `${col}/${relNext.map(normSeg).join("/")}`;
        map.set(mapKey, e.name);
        walk(path.join(dir, e.name), relNext);
      }
    };

    walk(root, []);
  }

  return map;
}

function maxIdLexInNode(node: CatPathNode): string {
  let maxKey = "";
  for (const p of node.posts) {
    const k = postIdLexKey(p);
    if (k.localeCompare(maxKey, "zh-CN") > 0) maxKey = k;
  }
  for (const c of node.children) {
    const k = maxIdLexInNode(c);
    if (k.localeCompare(maxKey, "zh-CN") > 0) maxKey = k;
  }
  return maxKey;
}

function finalize(m: MutableNode): CatPathNode {
  const children = [...m.childMap.values()].map(finalize);
  children.sort((a, b) => {
    const ka = maxIdLexInNode(a);
    const kb = maxIdLexInNode(b);
    const cmp = kb.localeCompare(ka, "zh-CN");
    if (cmp !== 0) return cmp;
    return a.name.localeCompare(b.name, "zh-CN");
  });
  return {
    name: m.name,
    pathKey: m.pathKey,
    posts: [...m.posts].sort((a, b) => {
      const byId = byIdLexDesc(a, b);
      if (byId !== 0) return byId;
      return byDateDesc(a, b);
    }),
    children,
  };
}

export type BuildCategoryPathTreeOptions = {
  collection: string;
  folderDisplayMap: Map<string, string>;
};

/** 按文章 slug 建树；文件夹展示名与 src/content 目录一致，不被 slug 小写改写 */
export function buildCategoryPathTree(
  posts: MergedPost[],
  opts: BuildCategoryPathTreeOptions
): CatPathNode {
  const { collection, folderDisplayMap } = opts;
  const root: MutableNode = {
    name: "",
    pathKey: collection,
    posts: [],
    childMap: new Map(),
  };

  for (const p of posts) {
    const parts = p.slug.split("/").filter(Boolean);
    if (parts.length === 0) continue;

    let cur = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const seg = parts[i];
      const pathKey = `${collection}/${parts.slice(0, i + 1).map(normSeg).join("/")}`;
      const displayName = folderDisplayMap.get(pathKey) ?? seg;

      let next = cur.childMap.get(seg);
      if (!next) {
        next = {
          name: displayName,
          pathKey,
          posts: [],
          childMap: new Map(),
        };
        cur.childMap.set(seg, next);
      }
      cur = next;
    }
    cur.posts.push(p);
  }

  return finalize(root);
}

export function totalInSubtree(node: CatPathNode): number {
  return (
    node.posts.length +
    node.children.reduce((sum, c) => sum + totalInSubtree(c), 0)
  );
}

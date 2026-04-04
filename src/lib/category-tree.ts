import fs from "node:fs";
import path from "node:path";
import type { ColKey, MergedPost } from "./content-posts";
import { collectionKeys, contentFolderByColKey } from "./content-posts";

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

function byDateDesc(a: MergedPost, b: MergedPost): number {
  const ta = a.date ? new Date(a.date).getTime() : 0;
  const tb = b.date ? new Date(b.date).getTime() : 0;
  return tb - ta;
}

/** 仅将拉丁大写字母转为小写，便于 slug 段与磁盘路径对齐；中文等原样保留 */
function normSeg(seg: string): string {
  return seg.replace(/[A-Z]/g, (c) => c.toLowerCase());
}

/**
 * 扫描 src/content/&lt;collection&gt;/ 下的真实文件夹名。
 * 键：`collection/a/b`（每段经 normSeg）；值：该层文件夹在磁盘上的原始名称（大小写与之一致）。
 */
export function buildContentFolderDisplayMap(): Map<string, string> {
  const map = new Map<string, string>();
  const contentRoot = path.join(process.cwd(), "src", "content");

  for (const col of collectionKeys) {
    const root = path.join(contentRoot, contentFolderByColKey[col]);
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

function finalize(m: MutableNode): CatPathNode {
  const children = [...m.childMap.values()]
    .map(finalize)
    .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
  return {
    name: m.name,
    pathKey: m.pathKey,
    posts: [...m.posts].sort(byDateDesc),
    children,
  };
}

export type BuildCategoryPathTreeOptions = {
  collection: ColKey;
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

import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import fs from "node:fs";
import path from "node:path";

const postSchema = z.object({
  title: z.string(),
  id: z.string().optional(),
  date: z.string().optional(),
  description: z.string().optional(),
  tags: z.array(z.string()).optional(),
  author: z.string().optional(),
  /** 文章顶图：完整 URL，或站內路径如 `site-profile.png`、`/images/foo.jpg` */
  heroImage: z.string().optional(),
});

function mdCollection(base: string) {
  return defineCollection({
    loader: glob({ pattern: "**/*.{md,mdx}", base }),
    schema: postSchema,
  });
}

function discoverContentDirectories(): string[] {
  const contentRoot = path.join(process.cwd(), "src", "content");
  let dirents: fs.Dirent[] = [];
  try {
    dirents = fs.readdirSync(contentRoot, { withFileTypes: true });
  } catch {
    return [];
  }
  return dirents
    .filter((d) => d.isDirectory() && !d.name.startsWith("."))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, "zh-CN"));
}

/** 按 src/content 下真实目录自动建集合；路由统一由 pages/categories/[...slug].astro 生成 */
export const collections: Record<string, ReturnType<typeof mdCollection>> = Object.fromEntries(
  discoverContentDirectories().map((dirname) => [
    dirname,
    mdCollection(`./src/content/${dirname}`),
  ])
);

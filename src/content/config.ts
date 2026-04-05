import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

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

/** glob 的 base 为磁盘实际目录；文章路由统一由 pages/categories/[...slug].astro 生成 */
export const collections = {
  algorithms: mdCollection("./src/content/algorithms"),
  ai: mdCollection("./src/content/AI"),
  tools: mdCollection("./src/content/Tools"),
  travel: mdCollection("./src/content/travel"),
};

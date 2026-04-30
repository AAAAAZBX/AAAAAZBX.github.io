import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  redirects: {
    "/learning": "/categories",
    "/research": "/categories",
  },
  markdown: {
    syntaxHighlight: 'shiki',
    gfm: true,
    remarkPlugins: [
      remarkMath,
      [
        remarkGfm,
        {
          autolinkLiteral: false,
        },
      ],
    ],
    rehypePlugins: [rehypeKatex],
  },
  site: 'https://aaaaazbx.github.io',
});

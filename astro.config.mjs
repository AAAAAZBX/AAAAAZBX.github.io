import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  markdown: {
    syntaxHighlight: 'shiki',
    // 启用 GFM 支持，确保基本的 Markdown 链接 []() 格式能正确解析
    gfm: true,
    remarkPlugins: [
      remarkMath,
      // 配置 remark-gfm，禁用自动链接功能
      // 这样 []() 格式的链接可以正常工作，但 https:// 不会被自动转换为链接
      [
        remarkGfm,
        {
          autolinkLiteral: false, // 禁用自动链接，让 []() 格式优先
        },
      ],
    ],
    rehypePlugins: [rehypeKatex],
  },
  site: 'https://AAAAAZBX.github.io',
});


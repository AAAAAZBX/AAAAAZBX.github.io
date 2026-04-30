import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import fs from 'node:fs';
import path from 'node:path';

// Vite plugin: parse hidden IDs from raw Node.js req.url (bypassing Node.js 24
// Request bug that strips query string). Writes local JSON, then calls next()
// so the Astro API route can sync to Supabase via import.meta.env.
function visibilityApiPlugin() {
  const visibilityPath = path.join(process.cwd(), 'src', 'content-visibility.json');

  return {
    name: 'visibility-api',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        if (!req.url?.startsWith('/api/visibility')) { next(); return; }
        const qs = req.url.includes('?') ? req.url.slice(req.url.indexOf('?') + 1) : '';
        const params = new URLSearchParams(qs);
        const hidden = [...new Set(params.getAll('hidden').map(s => String(s).trim().toLowerCase()).filter(Boolean))].sort((a, b) =>
          a.localeCompare(b, 'zh-CN')
        );

        fs.writeFileSync(visibilityPath, `${JSON.stringify({ hidden }, null, 2)}\n`, 'utf-8');
        // Store parsed IDs so the API route can use them for Supabase sync
        globalThis.__hiddenFromUrl = hidden;
        next();
      });
    },
  };
}

// https://astro.build/config
export default defineConfig({
  integrations: [
    mdx(),
    sitemap({
      filter: (page) =>
        !page.includes('/search') &&
        !page.includes('/404'),
    }),
  ],
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
  vite: {
    plugins: [visibilityApiPlugin()],
  },
});

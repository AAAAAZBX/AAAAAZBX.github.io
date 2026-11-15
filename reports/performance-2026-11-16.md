# Performance audit
Scanned content loading, shared layouts, home/about pages, navigation, visitor scripts, service worker, public asset sizes and build configuration.

Changes: local Bootstrap/Markdown/KaTeX/icon CSS and fonts; 320px WebP avatar (1,283,236 -> 6,492 bytes); concurrent content/visibility reads; shared 30-second development visibility snapshot with immediate local-file invalidation; development remote timeout reduced from 5s to 1.2s; hover prefetch; visitor logging starts after load.

Verification: 55-page Astro static build succeeded. Warm localhost response samples: home 57ms, categories 68ms, archives 29ms, about 90ms. Local resources and final icon CSS returned 200. These are HTTP response timings, not browser paint or before/after benchmarks. Remote visibility changes can take up to 30 seconds to appear in dev. Production build retains existing visibility semantics.

Large PDFs, source photos and legacy search XML are not downloaded on every navigation, so were preserved. No blanket cache of HTML/admin/API responses added. Browser visual/performance trace and deployed-network measurements remain unverified. Existing empty Math collection and Markdown math warnings remain.
